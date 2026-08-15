// Copyright (c) Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

use crate::autograd::with_grad_fn;
use crate::{
    autograd::NormBackward,
    error::{MinitensorError, Result},
    ops::{
        activation, arithmetic,
        map::{EXPENSIVE_PAR_THRESHOLD, PAR_CHUNK, PAR_THRESHOLD, unary_map},
        reduction,
    },
    tensor::{DataType, Shape, Tensor, TensorData},
};
use rayon::prelude::*;
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
                    return Err(MinitensorError::dim_out_of_range(d, ndim));
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
///
/// Covering every axis is the exception: that is the whole-tensor reduction,
/// which reads the buffer once instead of once per axis and has its own
/// lane-folded kernel. Going axis by axis meant the first step was a reduction
/// over dimension 0, the most expensive layout there is -- 1.56 ms against
/// 0.28 ms for the global fold on a 4096x1024 f32 tensor.
fn reduce_extremum(tensor: &Tensor, dims: &[usize], is_max: bool) -> Result<Tensor> {
    if dims.len() == tensor.ndim() {
        return if is_max {
            reduction::max(tensor, None, true)
        } else {
            reduction::min(tensor, None, true)
        };
    }

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

/// Replace scale factors that cannot be divided by with ones.
///
/// The scale is the largest magnitude in a slice, and it is used both to divide
/// the slice and to multiply the result back up. Two values leave it unusable:
///
/// * Zero, meaning a slice of all zeros. Dividing by one instead leaves
///   `0 / 1 = 0` and the norm still comes out zero, without the `0 / 0` the raw
///   scale would produce.
/// * Infinity, meaning the slice contains one. Dividing by it gave `inf / inf`
///   for that element and `finite / inf = 0` for the rest, so the norm of a
///   slice holding an infinity came back NaN where `inf` is the right answer.
///   With a scale of one the accumulation runs unscaled, `inf` survives the
///   sum and the root, and the final multiply by one keeps it -- while a NaN
///   elsewhere in the slice still poisons the sum and wins.
///
/// Neither substitution weakens the overflow guard, which only ever mattered
/// for a finite scale: every `|x| <= m` gives `|x / m| <= 1`.
fn unusable_scales_to_ones(tensor: &Tensor) -> Result<Tensor> {
    let data = match tensor.dtype() {
        DataType::Float32 => {
            let src = tensor.data().as_f32_slice().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get f32 slice from scale")
            })?;
            TensorData::from_vec::<f32>(
                unary_map(
                    src,
                    |v: f32| if v == 0.0 || v.is_infinite() { 1.0 } else { v },
                ),
                DataType::Float32,
                tensor.device(),
            )
        }
        DataType::Float64 => {
            let src = tensor.data().as_f64_slice().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get f64 slice from scale")
            })?;
            TensorData::from_vec::<f64>(
                unary_map(
                    src,
                    |v: f64| if v == 0.0 || v.is_infinite() { 1.0 } else { v },
                ),
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

/// The largest magnitude along `dims`, without materialising `|x|`.
///
/// `max|x|` is `max(max(x), -min(x))`, and both reductions read the input
/// directly where taking the maximum of `abs(x)` needed a full-size copy of it
/// first. The two agree on every input: a NaN anywhere makes both reductions
/// NaN, an infinity of either sign reaches the maximum through one side or the
/// other, and negating an already-reduced tensor is exact.
fn abs_max_over(tensor: &Tensor, dims: &[usize]) -> Result<Tensor> {
    let highest = reduce_extremum(tensor, dims, true)?;
    let lowest = reduce_extremum(tensor, dims, false)?;
    crate::ops::minmax::maximum(&highest, &arithmetic::neg(&lowest)?)
}

/// How `scale` broadcasts back over the input it was reduced from.
enum ScaleLayout {
    /// Every axis reduced, so the scale is one value for the whole tensor.
    Scalar,
    /// One axis reduced: input position `(o, i, r)` divides by `scale[o, r]`.
    Axis { dim_size: usize, inner: usize },
}

/// Build `|x / scale|^p` in one pass and one allocation.
///
/// This is `abs` -> `div` -> `powf` collapsed into a single kernel. Those were
/// three full-size temporaries, and on a 4096x1024 f32 tensor the allocations
/// cost more than the arithmetic they carried: `norm(2, dim=1)` took 8.88 ms,
/// and the identical code took 3.63 ms with glibc told to keep large blocks
/// resident instead of returning their pages. One buffer per call is recycled
/// between calls; three overlapping ones are not.
///
/// Only the two layouts whose index arithmetic is cheap are handled here, which
/// between them cover `norm(x)` and `norm(x, dim)`; [`norm`] keeps the general
/// composition for a reduction over some other subset of the axes.
///
/// The small-integer exponents match the fast paths in
/// [`crate::ops::activation::pow`] exactly -- repeated multiplication, not
/// `powf` -- so which route a norm takes cannot change the value it returns.
fn scaled_powers(input: &Tensor, scale: &Tensor, p: f64, dims: &[usize]) -> Result<Option<Tensor>> {
    let shape = input.shape().dims();
    let layout = if dims.len() == shape.len() {
        ScaleLayout::Scalar
    } else if dims.len() == 1 {
        let d = dims[0];
        ScaleLayout::Axis {
            dim_size: shape[d],
            inner: shape[d + 1..].iter().product::<usize>(),
        }
    } else {
        return Ok(None);
    };

    macro_rules! build {
        ($ty:ty, $slice:ident, $from_vec:ident) => {{
            let src = input.data().$slice().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get input slice for norm")
            })?;
            let scales = scale.data().$slice().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get scale slice for norm")
            })?;
            let mut out = vec![<$ty>::default(); src.len()];

            // `$power` is chosen once, outside the loops, so the exponent never
            // costs a branch per element.
            macro_rules! fill {
                ($power:expr, $threshold:expr) => {{
                    let power = $power;
                    match layout {
                        ScaleLayout::Scalar => {
                            let s = scales[0];
                            let run = |base: usize, chunk: &mut [$ty]| {
                                for (k, slot) in chunk.iter_mut().enumerate() {
                                    *slot = power(src[base + k].abs() / s);
                                }
                            };
                            if src.len() < $threshold {
                                run(0, &mut out);
                            } else {
                                let grain = PAR_CHUNK.max(1);
                                out.par_chunks_mut(grain)
                                    .enumerate()
                                    .for_each(|(c, chunk)| run(c * grain, chunk));
                            }
                        }
                        ScaleLayout::Axis { dim_size, inner } => {
                            let block = dim_size * inner;
                            let run = |o: usize, chunk: &mut [$ty]| {
                                let base = o * block;
                                for i in 0..dim_size {
                                    let row = i * inner;
                                    for r in 0..inner {
                                        let s = scales[o * inner + r];
                                        chunk[row + r] = power(src[base + row + r].abs() / s);
                                    }
                                }
                            };
                            if src.len() < $threshold {
                                out.chunks_mut(block)
                                    .enumerate()
                                    .for_each(|(o, chunk)| run(o, chunk));
                            } else {
                                out.par_chunks_mut(block)
                                    .enumerate()
                                    .for_each(|(o, chunk)| run(o, chunk));
                            }
                        }
                    }
                }};
            }

            let exponent = p as $ty;
            if exponent == 2.0 {
                fill!(|v: $ty| v * v, PAR_THRESHOLD)
            } else if exponent == 1.0 {
                fill!(|v: $ty| v, PAR_THRESHOLD)
            } else if exponent == 3.0 {
                fill!(|v: $ty| v * v * v, PAR_THRESHOLD)
            } else {
                // `powf` is a transcendental, so it repays parallelism far
                // sooner than the multiply-only exponents do.
                fill!(move |v: $ty| v.powf(exponent), EXPENSIVE_PAR_THRESHOLD)
            }
            TensorData::$from_vec(out, input.device())
        }};
    }

    let data = match input.dtype() {
        DataType::Float32 => build!(f32, as_f32_slice, from_vec_f32),
        DataType::Float64 => build!(f64, as_f64_slice, from_vec_f64),
        _ => {
            return Err(MinitensorError::invalid_operation(
                "norm requires floating point tensors",
            ));
        }
    };

    Ok(Some(Tensor::new(
        Arc::new(data),
        input.shape().clone(),
        input.dtype(),
        input.device(),
        false,
    )))
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
/// the largest magnitude keeps every intermediate inside the range. When that
/// magnitude is itself zero or infinite there is nothing to scale by, and
/// [`unusable_scales_to_ones`] stands one in for it.
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
        if p == f64::INFINITY {
            abs_max_over(&input, &dims)?
        } else if p == f64::NEG_INFINITY {
            // The *smallest* magnitude is not recoverable from the reductions of
            // `x` alone -- a slice straddling zero has neither endpoint nearest
            // it -- so this order still materialises `|x|`.
            reduce_extremum(&activation::abs(&input)?, &dims, false)?
        } else if p == 0.0 {
            count_nonzero_over(&input, &dims)?
        } else {
            let scale = unusable_scales_to_ones(&abs_max_over(&input, &dims)?)?;
            let powered = match scaled_powers(&input, &scale, p, &dims)? {
                Some(fused) => fused,
                None => {
                    let abs_x = activation::abs(&input)?;
                    let scaled = arithmetic::div(&abs_x, &scale)?;
                    activation::powf(&scaled, p)?
                }
            };
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
    if tensor.requires_grad() && p != 0.0 && crate::autograd::is_grad_enabled() {
        let grad_fn = Arc::new(NormBackward {
            input_id: tensor.id(),
            input,
            norm: value_kd,
            p,
            dims,
            keepdim_shape: reduced_shape,
        });
        let output = output.requires_grad_(true);
        with_grad_fn(output, grad_fn)
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
