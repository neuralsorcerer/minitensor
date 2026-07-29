// Copyright (c) Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

use crate::{
    autograd::{NormBackward, add_to_graph},
    error::{MinitensorError, Result},
    ops::{activation, arithmetic, map::unary_map, reduction},
    tensor::{DataType, Shape, Tensor, TensorData},
};
use std::sync::Arc;

/// Normalise `dim` into a sorted, deduplicated list of axes. `None` means every
/// axis, which is what makes `norm(t)` the norm of the flattened tensor.
pub(crate) fn normalize_norm_dims(dim: Option<Vec<isize>>, ndim: usize) -> Result<Vec<usize>> {
    match dim {
        None => Ok((0..ndim).collect()),
        Some(dims) => {
            let n = ndim as isize;
            let mut out = Vec::with_capacity(dims.len());
            for d in dims {
                let normalized = if d < 0 { d + n } else { d };
                if normalized < 0 || normalized >= n {
                    return Err(MinitensorError::index_error(d, 0, ndim));
                }
                out.push(normalized as usize);
            }
            out.sort_unstable();
            out.dedup();
            Ok(out)
        }
    }
}

/// The shape a reduction over `dims` produces with `keepdim = true`.
fn keepdim_shape(shape: &Shape, dims: &[usize]) -> Shape {
    let mut out = shape.dims().to_vec();
    for &d in dims {
        out[d] = 1;
    }
    Shape::new(out)
}

/// Reduce `tensor` over every axis in `dims` with `max` (or `min`), keeping the
/// rank. Applying one axis at a time keeps the remaining axis indices valid,
/// which a rank-shrinking reduction would not.
fn reduce_extremum(tensor: &Tensor, dims: &[usize], is_max: bool) -> Result<Tensor> {
    let mut acc = tensor.clone();
    for &d in dims {
        acc = if is_max {
            reduction::max(&acc, Some(d as isize), true)?
        } else {
            reduction::min(&acc, Some(d as isize), true)?
        };
    }
    Ok(acc)
}

/// Replace exact zeros with ones.
///
/// Used on the scale factor: a slice whose maximum magnitude is zero is a slice
/// of all zeros, so dividing it by one instead leaves `0 / 1 = 0` and the norm
/// still comes out zero — without the `0 / 0` the raw scale would produce.
fn zeros_to_ones(tensor: &Tensor) -> Result<Tensor> {
    let data = match tensor.dtype() {
        DataType::Float32 => {
            let src = tensor.data().as_f32_slice().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get f32 slice from scale")
            })?;
            TensorData::from_vec::<f32>(
                unary_map(src, |v: f32| if v == 0.0 { 1.0 } else { v }),
                DataType::Float32,
                tensor.device(),
            )
        }
        DataType::Float64 => {
            let src = tensor.data().as_f64_slice().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get f64 slice from scale")
            })?;
            TensorData::from_vec::<f64>(
                unary_map(src, |v: f64| if v == 0.0 { 1.0 } else { v }),
                DataType::Float64,
                tensor.device(),
            )
        }
        _ => {
            return Err(MinitensorError::invalid_operation(
                "norm requires floating point tensors",
            ));
        }
    };

    Ok(Tensor::new(
        Arc::new(data),
        tensor.shape().clone(),
        tensor.dtype(),
        tensor.device(),
        false,
    ))
}

/// Count the non-zero entries along `dims`, as a float tensor of the same dtype.
fn count_nonzero_over(tensor: &Tensor, dims: &[usize]) -> Result<Tensor> {
    let data = match tensor.dtype() {
        DataType::Float32 => {
            let src = tensor.data().as_f32_slice().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get f32 slice from tensor")
            })?;
            TensorData::from_vec::<f32>(
                unary_map(src, |v: f32| if v != 0.0 { 1.0 } else { 0.0 }),
                DataType::Float32,
                tensor.device(),
            )
        }
        DataType::Float64 => {
            let src = tensor.data().as_f64_slice().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get f64 slice from tensor")
            })?;
            TensorData::from_vec::<f64>(
                unary_map(src, |v: f64| if v != 0.0 { 1.0 } else { 0.0 }),
                DataType::Float64,
                tensor.device(),
            )
        }
        _ => {
            return Err(MinitensorError::invalid_operation(
                "norm requires floating point tensors",
            ));
        }
    };

    let indicator = Tensor::new(
        Arc::new(data),
        tensor.shape().clone(),
        tensor.dtype(),
        tensor.device(),
        false,
    );
    let dims_isize: Vec<isize> = dims.iter().map(|&d| d as isize).collect();
    reduction::sum(&indicator, Some(dims_isize), true)
}

/// Vector p-norm over `dim`, or over the flattened tensor when `dim` is `None`.
///
/// Supported orders: any finite `p > 0`, plus `inf` (maximum magnitude), `-inf`
/// (minimum magnitude) and `0` (count of non-zeros, which is not a norm but is
/// the conventional reading of `p = 0`). Finite negative orders are rejected —
/// they are defined by a limit but have no useful derivative here, and silently
/// returning something differentiable-looking would be worse than an error.
///
/// The 2-norm is computed as `m * sqrt(sum((x / m)^2))` with `m = max|x|`
/// rather than as `sqrt(sum(x^2))`. Squaring first overflows f32 once `|x|`
/// passes about 1.8e19, so the direct form reports `inf` for a norm that is
/// perfectly representable — and `norm` is exactly what you call to *detect*
/// a blow-up, so saturating right at that point defeats the purpose. Scaling by
/// the largest magnitude keeps every intermediate inside the range.
pub fn norm(tensor: &Tensor, p: f64, dim: Option<Vec<isize>>, keepdim: bool) -> Result<Tensor> {
    if !tensor.dtype().is_float() {
        return Err(MinitensorError::invalid_operation(
            "norm requires floating point tensors",
        ));
    }
    if p.is_nan() {
        return Err(MinitensorError::invalid_operation(
            "norm order p must not be NaN",
        ));
    }
    if p.is_finite() && p < 0.0 {
        return Err(MinitensorError::invalid_operation(format!(
            "norm order p must be positive, 0, or +/-inf, got {p}"
        )));
    }

    let dims = normalize_norm_dims(dim, tensor.ndim())?;
    let reduced_shape = keepdim_shape(tensor.shape(), &dims);

    let input = tensor.detach();

    // An empty reduction has nothing to accumulate, so every order gives 0.
    let value_kd = if tensor.numel() == 0 {
        Tensor::zeros(
            reduced_shape.clone(),
            tensor.dtype(),
            tensor.device(),
            false,
        )
    } else {
        let abs_x = activation::abs(&input)?;
        if p == f64::INFINITY {
            reduce_extremum(&abs_x, &dims, true)?
        } else if p == f64::NEG_INFINITY {
            reduce_extremum(&abs_x, &dims, false)?
        } else if p == 0.0 {
            count_nonzero_over(&input, &dims)?
        } else {
            let scale = zeros_to_ones(&reduce_extremum(&abs_x, &dims, true)?)?;
            let scaled = arithmetic::div(&abs_x, &scale)?;
            let powered = activation::powf(&scaled, p)?;
            let dims_isize: Vec<isize> = dims.iter().map(|&d| d as isize).collect();
            let summed = reduction::sum(&powered, Some(dims_isize), true)?;
            let rooted = activation::powf(&summed, 1.0 / p)?;
            arithmetic::mul(&scale, &rooted)?
        }
    };

    let output = if keepdim {
        value_kd.clone()
    } else {
        let kept: Vec<usize> = tensor
            .shape()
            .dims()
            .iter()
            .enumerate()
            .filter(|(i, _)| !dims.contains(i))
            .map(|(_, &d)| d)
            .collect();
        value_kd.reshape(Shape::new(kept))?
    };

    // p = 0 counts non-zeros: it is a step function, so its derivative is zero
    // wherever it exists and there is no gradient worth recording.
    if tensor.requires_grad() && p != 0.0 {
        let grad_fn = Arc::new(NormBackward {
            input_id: tensor.id(),
            input,
            norm: value_kd,
            p,
            dims,
            keepdim_shape: reduced_shape,
        });
        let mut output = output.requires_grad_(true);
        output.set_grad_fn(Some(grad_fn.clone()));
        add_to_graph(&output, Some(grad_fn))?;
        Ok(output)
    } else {
        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::Device;

    fn tensor_f64(data: Vec<f64>, shape: Vec<usize>, requires_grad: bool) -> Tensor {
        Tensor::new(
            Arc::new(TensorData::from_vec_f64(data, Device::cpu())),
            Shape::new(shape),
            DataType::Float64,
            Device::cpu(),
            requires_grad,
        )
    }

    /// Independent p-norm over a flat slice.
    fn reference(values: &[f64], p: f64) -> f64 {
        if p == f64::INFINITY {
            return values.iter().fold(0.0f64, |a, v| a.max(v.abs()));
        }
        if p == f64::NEG_INFINITY {
            return values
                .iter()
                .fold(f64::INFINITY, |a: f64, v| a.min(v.abs()));
        }
        if p == 0.0 {
            return values.iter().filter(|v| **v != 0.0).count() as f64;
        }
        values
            .iter()
            .map(|v| v.abs().powf(p))
            .sum::<f64>()
            .powf(1.0 / p)
    }

    #[test]
    fn test_norm_matches_reference_for_every_supported_order() {
        let values = vec![1.5, -2.0, 0.25, 3.0, -0.5, 4.0];
        for &p in &[1.0, 2.0, 3.0, 0.5, 0.0, f64::INFINITY, f64::NEG_INFINITY] {
            let t = tensor_f64(values.clone(), vec![6], false);
            let got = norm(&t, p, None, false).unwrap();
            let want = reference(&values, p);
            let got = got.data().as_f64_slice().unwrap()[0];
            assert!(
                (got - want).abs() <= 1e-9 * want.abs().max(1.0),
                "p={p}: {got} != {want}"
            );
        }
    }

    #[test]
    fn test_norm_over_dim_keepdim_shapes_and_values() {
        // [2, 3], reduced over the last axis.
        let t = tensor_f64(vec![3.0, 4.0, 0.0, 1.0, 2.0, 2.0], vec![2, 3], false);

        let squeezed = norm(&t, 2.0, Some(vec![1]), false).unwrap();
        assert_eq!(squeezed.shape().dims(), &[2]);
        let got = squeezed.data().as_f64_slice().unwrap();
        assert!((got[0] - 5.0).abs() < 1e-12);
        assert!((got[1] - 3.0).abs() < 1e-12);

        let kept = norm(&t, 2.0, Some(vec![1]), true).unwrap();
        assert_eq!(kept.shape().dims(), &[2, 1]);

        // A negative axis means the same axis.
        let negative = norm(&t, 2.0, Some(vec![-1]), false).unwrap();
        assert_eq!(
            negative.data().as_f64_slice().unwrap(),
            squeezed.data().as_f64_slice().unwrap()
        );
    }

    #[test]
    fn test_norm_two_scales_instead_of_squaring() {
        // 1e200 squared overflows f64, so an unscaled sum-of-squares would give
        // inf for a norm that is perfectly representable.
        let t = tensor_f64(vec![1e200, 1e200], vec![2], false);
        let got = norm(&t, 2.0, None, false)
            .unwrap()
            .data()
            .as_f64_slice()
            .unwrap()[0];
        let want = 1e200 * std::f64::consts::SQRT_2;
        assert!(got.is_finite(), "norm overflowed: {got}");
        assert!((got - want).abs() <= 1e-6 * want, "{got} != {want}");
    }

    #[test]
    fn test_norm_gradient_is_zero_at_origin() {
        // The p-norm has a corner at the origin; report the zero subgradient
        // rather than the 0/0 a composed implementation would produce.
        for &p in &[1.0, 2.0, 3.0] {
            let t = tensor_f64(vec![0.0, 0.0, 0.0], vec![3], true);
            let out = norm(&t, p, None, false).unwrap();
            let grads = crate::autograd::backward_collect(&out, None).unwrap();
            for g in grads[&t.id()].data().as_f64_slice().unwrap() {
                assert!(g.is_finite() && *g == 0.0, "p={p}: grad {g}");
            }
        }
    }

    #[test]
    fn test_norm_gradient_matches_central_differences() {
        let base = vec![1.5, -2.0, 0.25, 3.0, -0.5, 4.0];
        for &p in &[1.0, 2.0, 3.0, 0.5] {
            let t = tensor_f64(base.clone(), vec![6], true);
            let out = norm(&t, p, None, false).unwrap();
            let grads = crate::autograd::backward_collect(&out, None).unwrap();
            let analytic = grads[&t.id()].data().as_f64_slice().unwrap().to_vec();

            let h = 1e-6;
            for i in 0..base.len() {
                let mut plus = base.clone();
                plus[i] += h;
                let mut minus = base.clone();
                minus[i] -= h;
                let central = (reference(&plus, p) - reference(&minus, p)) / (2.0 * h);
                assert!(
                    (analytic[i] - central).abs() < 1e-6,
                    "p={p} i={i}: {} != {central}",
                    analytic[i]
                );
            }
        }
    }

    #[test]
    fn test_norm_inf_splits_gradient_among_ties() {
        // |x| peaks at 3 three times, so each tied entry takes a third, signed.
        let t = tensor_f64(vec![3.0, -3.0, 1.0, 3.0], vec![4], true);
        let out = norm(&t, f64::INFINITY, None, false).unwrap();
        let grads = crate::autograd::backward_collect(&out, None).unwrap();
        let g = grads[&t.id()].data().as_f64_slice().unwrap();
        let third = 1.0 / 3.0;
        for (got, want) in g.iter().zip([third, -third, 0.0, third]) {
            assert!((got - want).abs() < 1e-12, "{got} != {want}");
        }
    }

    #[test]
    fn test_norm_zero_order_counts_nonzeros_without_gradient() {
        let t = tensor_f64(vec![0.0, 1.0, 0.0, 2.0], vec![4], true);
        let out = norm(&t, 0.0, None, false).unwrap();
        assert_eq!(out.data().as_f64_slice().unwrap()[0], 2.0);
        // Piecewise constant, so there is no gradient edge to record.
        assert!(!out.requires_grad());
    }

    #[test]
    fn test_norm_of_empty_tensor_is_zero() {
        let t = tensor_f64(vec![], vec![0, 3], false);
        let out = norm(&t, 2.0, Some(vec![0]), false).unwrap();
        assert_eq!(out.shape().dims(), &[3]);
        assert!(out.data().as_f64_slice().unwrap().iter().all(|v| *v == 0.0));
    }

    #[test]
    fn test_norm_rejects_unsupported_orders() {
        let t = tensor_f64(vec![1.0], vec![1], false);
        assert!(norm(&t, -2.0, None, false).is_err());
        assert!(norm(&t, f64::NAN, None, false).is_err());
        assert!(norm(&t, 2.0, Some(vec![5]), false).is_err());
    }
}
