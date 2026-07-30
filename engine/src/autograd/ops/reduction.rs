// Copyright (c) Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

use super::*;
use crate::{
    error::{MinitensorError, Result},
    ops::util::create_scalar_tensor,
    ops::{arithmetic, reduction},
    tensor::{DataType, Shape, Tensor, TensorData},
};
use rustc_hash::FxHashMap;
use smallvec::SmallVec;
use std::sync::Arc;

/// Gradient function for layer normalization
pub struct LayerNormBackward {
    pub input_ids: SmallVec<[TensorId; 3]>,
    pub input_id: TensorId,
    pub weight_id: Option<TensorId>,
    pub bias_id: Option<TensorId>,
    pub normalized: Tensor,
    pub inv_std: Tensor,
    pub weight_broadcast: Option<Tensor>,
    pub normalized_shape: Vec<usize>,
    pub axis_start: usize,
    pub element_count: usize,
    pub input_requires_grad: bool,
    pub weight_requires_grad: bool,
    pub bias_requires_grad: bool,
}

impl GradientFunction for LayerNormBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<FxHashMap<TensorId, Tensor>> {
        let mut gradients = FxHashMap::default();

        let grad_output_detached = grad_output.detach();
        let normalized = self.normalized.detach();

        if self.element_count == 0 {
            if self.input_requires_grad {
                let zero = Tensor::zeros(
                    grad_output.shape().clone(),
                    grad_output.dtype(),
                    grad_output.device(),
                    false,
                );
                accumulate_grad(&mut gradients, self.input_id, zero)?;
            }
            if self.weight_requires_grad
                && let Some(weight_id) = self.weight_id
            {
                let zero = Tensor::zeros(
                    Shape::new(self.normalized_shape.clone()),
                    grad_output.dtype(),
                    grad_output.device(),
                    false,
                );
                accumulate_grad(&mut gradients, weight_id, zero)?;
            }
            if self.bias_requires_grad
                && let Some(bias_id) = self.bias_id
            {
                let zero = Tensor::zeros(
                    Shape::new(self.normalized_shape.clone()),
                    grad_output.dtype(),
                    grad_output.device(),
                    false,
                );
                accumulate_grad(&mut gradients, bias_id, zero)?;
            }

            return Ok(gradients);
        }

        if self.input_requires_grad {
            let mut grad_output_hat = if let Some(weight) = &self.weight_broadcast {
                arithmetic::mul(&grad_output_detached, weight)?
            } else {
                grad_output_detached.clone()
            };

            let axes: Vec<isize> = (self.axis_start..grad_output_hat.ndim())
                .map(|d| d as isize)
                .collect();
            let sum_grad = reduction::sum(&grad_output_hat, Some(axes.clone()), true)?;
            let grad_norm_mul = arithmetic::mul(&grad_output_hat, &normalized)?;
            let sum_grad_norm = reduction::sum(&grad_norm_mul, Some(axes), true)?;

            let count = self.element_count as f64;
            let m_tensor = create_scalar_tensor(count, grad_output.dtype(), grad_output.device())?;
            let inv_m_tensor =
                create_scalar_tensor(1.0 / count, grad_output.dtype(), grad_output.device())?;
            grad_output_hat = arithmetic::mul(&grad_output_hat, &m_tensor)?;
            let tmp = arithmetic::sub(&grad_output_hat, &sum_grad)?;
            let norm_term = arithmetic::mul(&normalized, &sum_grad_norm)?;
            let numerator = arithmetic::sub(&tmp, &norm_term)?;
            let grad_input = arithmetic::mul(&numerator, &self.inv_std)?;
            let grad_input = arithmetic::mul(&grad_input, &inv_m_tensor)?;
            accumulate_grad(&mut gradients, self.input_id, grad_input)?;
        }

        if self.weight_requires_grad
            && let Some(weight_id) = self.weight_id
        {
            let mut grad_weight = arithmetic::mul(&grad_output_detached, &normalized)?;
            if self.axis_start > 0 {
                let axes: Vec<isize> = (0..self.axis_start).map(|d| d as isize).collect();
                grad_weight = reduction::sum(&grad_weight, Some(axes), false)?;
            }
            if grad_weight.shape().dims() != self.normalized_shape.as_slice() {
                grad_weight = grad_weight.view(Shape::new(self.normalized_shape.clone()))?;
            }
            accumulate_grad(&mut gradients, weight_id, grad_weight)?;
        }

        if self.bias_requires_grad
            && let Some(bias_id) = self.bias_id
        {
            let mut grad_bias = grad_output_detached.clone();
            if self.axis_start > 0 {
                let axes: Vec<isize> = (0..self.axis_start).map(|d| d as isize).collect();
                grad_bias = reduction::sum(&grad_bias, Some(axes), false)?;
            }
            if grad_bias.shape().dims() != self.normalized_shape.as_slice() {
                grad_bias = grad_bias.view(Shape::new(self.normalized_shape.clone()))?;
            }
            accumulate_grad(&mut gradients, bias_id, grad_bias)?;
        }

        Ok(gradients)
    }

    fn input_ids(&self) -> &[TensorId] {
        &self.input_ids
    }
}

/// Gradient function for `min`/`max` reductions (global with `dim == None`, or
/// along a single `dim`).
///
/// The gradient flows to every input element equal to the reduced extremum,
/// split equally among ties so the contributions sum to the upstream gradient. The
/// extremum, its selection mask and the tie count are recomputed from the stored
/// (detached) input, so nothing beyond the input needs to be retained.
pub struct MinMaxBackward {
    pub input_id: TensorId,
    pub input: Tensor,
    pub dim: Option<usize>,
    pub keepdim: bool,
    pub is_max: bool,
    pub nan_aware: bool,
}

/// Route `grad_output` to every input element equal to the selected reduction
/// value (`reduced`, recomputed with keepdim so it broadcasts), splitting equally
/// among ties. Shared by min/max and median value reductions.
fn distribute_selection_grad(
    input: &Tensor,
    reduced: &Tensor,
    grad_output: &Tensor,
    dim: Option<usize>,
    keepdim: bool,
) -> Result<Tensor> {
    let input_shape = input.shape().dims().to_vec();
    let mask = crate::ops::comparison::eq(input, reduced)?;
    let mask_f = mask.astype(input.dtype())?;

    let sum_dims = dim.map(|d| vec![d as isize]);
    let count = reduction::sum(&mask_f, sum_dims, true)?;

    let dims_vec = dim.map(|d| vec![d]);
    let grad_kd = expand_reduction_grad(grad_output, &input_shape, &dims_vec, keepdim)?;
    let scaled = arithmetic::div(&grad_kd, &count)?;
    arithmetic::mul(&mask_f, &scaled)
}

impl GradientFunction for MinMaxBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<FxHashMap<TensorId, Tensor>> {
        let mut gradients = FxHashMap::default();
        let input = &self.input;

        if input.numel() == 0 {
            let zero = Tensor::zeros(input.shape().clone(), input.dtype(), input.device(), false);
            accumulate_grad(&mut gradients, self.input_id, zero)?;
            return Ok(gradients);
        }

        let dim_isize = self.dim.map(|d| d as isize);
        // Recompute the extremum with keepdim so it broadcasts against the input.
        // NaN-aware reductions must recompute with the matching op, otherwise the
        // propagated NaN would fail the equality mask and zero every gradient.
        let reduced = match (self.is_max, self.nan_aware) {
            (true, false) => reduction::max(input, dim_isize, true)?,
            (false, false) => reduction::min(input, dim_isize, true)?,
            (true, true) => reduction::nanmax(input, dim_isize, true)?,
            (false, true) => reduction::nanmin(input, dim_isize, true)?,
        };
        let grad_input =
            distribute_selection_grad(input, &reduced, grad_output, self.dim, self.keepdim)?;
        accumulate_grad(&mut gradients, self.input_id, grad_input)?;
        Ok(gradients)
    }

    fn input_ids(&self) -> &[TensorId] {
        std::slice::from_ref(&self.input_id)
    }
}

/// Gradient function for `median`/`nanmedian` value reductions. The median is one
/// of the input elements, so the gradient flows to every element equal to it,
/// split over ties (a valid subgradient, matching the min/max convention).
pub struct MedianBackward {
    pub input_id: TensorId,
    pub input: Tensor,
    pub dim: Option<usize>,
    pub keepdim: bool,
    pub nan_aware: bool,
}

impl GradientFunction for MedianBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<FxHashMap<TensorId, Tensor>> {
        let mut gradients = FxHashMap::default();
        let input = &self.input;

        if input.numel() == 0 {
            let zero = Tensor::zeros(input.shape().clone(), input.dtype(), input.device(), false);
            accumulate_grad(&mut gradients, self.input_id, zero)?;
            return Ok(gradients);
        }

        let dim_isize = self.dim.map(|d| d as isize);
        let reduced = if self.nan_aware {
            reduction::nanmedian(input, dim_isize, true)?
        } else {
            reduction::median(input, dim_isize, true)?.0
        };
        let grad_input =
            distribute_selection_grad(input, &reduced, grad_output, self.dim, self.keepdim)?;
        accumulate_grad(&mut gradients, self.input_id, grad_input)?;
        Ok(gradients)
    }

    fn input_ids(&self) -> &[TensorId] {
        std::slice::from_ref(&self.input_id)
    }
}

/// Gradient function for the `quantile` reduction (global with `dim == None`, or
/// along a single `dim`).
///
/// A quantile is a fixed linear combination of the two order statistics that
/// bracket the requested position, so the gradient routes back to the two
/// original elements occupying those sorted ranks with the interpolation weights
/// (`Lower`/`Higher`/`Nearest` collapse to a single element; `Midpoint` splits
/// evenly). Groups containing NaN produced NaN and receive no gradient.
pub struct QuantileBackward {
    pub input_id: TensorId,
    pub input: Tensor,
    pub dim: Option<usize>,
    pub q: f64,
    pub interpolation: crate::ops::reduction::QuantileInterpolation,
    pub nan_aware: bool,
}

/// Sorted-rank indices and their gradient weights for a group of length `len`.
fn quantile_grad_coeffs(
    len: usize,
    q: f64,
    interp: crate::ops::reduction::QuantileInterpolation,
) -> (usize, usize, f64, f64) {
    use crate::ops::reduction::QuantileInterpolation as Qi;
    if len <= 1 {
        return (0, 0, 1.0, 0.0);
    }
    let pos = q * (len - 1) as f64;
    let lower = pos.floor() as usize;
    let upper = pos.ceil() as usize;
    let weight = (pos - lower as f64).clamp(0.0, 1.0);
    match interp {
        Qi::Linear => (lower, upper, 1.0 - weight, weight),
        Qi::Lower => (lower, upper, 1.0, 0.0),
        Qi::Higher => (lower, upper, 0.0, 1.0),
        Qi::Midpoint => (lower, upper, 0.5, 0.5),
        Qi::Nearest => {
            // Ties at weight == 0.5 round to the even index.
            let nearest = if weight < 0.5 {
                lower
            } else if weight > 0.5 {
                upper
            } else {
                lower + (lower & 1)
            };
            if nearest == lower {
                (lower, upper, 1.0, 0.0)
            } else {
                (lower, upper, 0.0, 1.0)
            }
        }
    }
}

impl GradientFunction for QuantileBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<FxHashMap<TensorId, Tensor>> {
        let mut gradients = FxHashMap::default();
        let input = &self.input;
        let numel = input.numel();
        let mut grad_data = TensorData::zeros_on_device(numel, input.dtype(), input.device());

        if numel != 0 {
            // Treat the global reduction as one group over the flattened tensor.
            let dims = input.shape().dims();
            let (outer, inner, dim_size) = match self.dim {
                None => (1usize, 1usize, numel),
                Some(d) => {
                    let outer: usize = dims[..d].iter().product();
                    let inner: usize = dims[d + 1..].iter().product();
                    (outer, inner, dims[d])
                }
            };
            let outer_stride = dim_size * inner;

            macro_rules! scatter {
                ($slice:ident, $mut_slice:ident, $ty:ty) => {{
                    let x = input.data().$slice().ok_or_else(|| {
                        MinitensorError::internal_error(
                            "Failed to read input for quantile backward",
                        )
                    })?;
                    let go = grad_output.data().$slice().ok_or_else(|| {
                        MinitensorError::internal_error(
                            "Failed to read grad_output for quantile backward",
                        )
                    })?;
                    let gi = grad_data.$mut_slice().ok_or_else(|| {
                        MinitensorError::internal_error(
                            "Failed to write grad for quantile backward",
                        )
                    })?;
                    let mut buffer: Vec<(usize, $ty)> = Vec::with_capacity(dim_size);
                    for o in 0..outer {
                        for r in 0..inner {
                            buffer.clear();
                            let mut skip_group = false;
                            for d in 0..dim_size {
                                let v = x[o * outer_stride + d * inner + r];
                                if v.is_nan() {
                                    if self.nan_aware {
                                        // nanquantile ignores NaN entries.
                                        continue;
                                    }
                                    // quantile propagates NaN, so the whole group's
                                    // output is NaN and gets no gradient.
                                    skip_group = true;
                                    break;
                                }
                                buffer.push((d, v));
                            }
                            if skip_group || buffer.is_empty() {
                                continue;
                            }
                            let (lo, up, c_lo, c_up) =
                                quantile_grad_coeffs(buffer.len(), self.q, self.interpolation);
                            // Only the elements at sorted ranks `lo` and `up`
                            // (adjacent) are needed, so select them in O(n) rather
                            // than fully sorting. NaN is already filtered, so the
                            // comparator never sees an incomparable value.
                            let cmp =
                                |a: &(usize, $ty), b: &(usize, $ty)| a.1.partial_cmp(&b.1).unwrap();
                            buffer.select_nth_unstable_by(up, cmp);
                            let d_up = buffer[up].0;
                            let d_lo = if lo == up {
                                d_up
                            } else {
                                // `lo == up - 1`: the lo-th order statistic is the
                                // largest of the elements left of `up`.
                                buffer[..up].select_nth_unstable_by(lo, cmp);
                                buffer[lo].0
                            };
                            let g = go[o * inner + r];
                            gi[o * outer_stride + d_lo * inner + r] += g * c_lo as $ty;
                            gi[o * outer_stride + d_up * inner + r] += g * c_up as $ty;
                        }
                    }
                }};
            }

            match input.dtype() {
                DataType::Float32 => scatter!(as_f32_slice, as_f32_slice_mut, f32),
                DataType::Float64 => scatter!(as_f64_slice, as_f64_slice_mut, f64),
                _ => {
                    return Err(MinitensorError::invalid_operation(
                        "quantile backward only supported for floating point tensors",
                    ));
                }
            }
        }

        let grad_input = Tensor::new(
            Arc::new(grad_data),
            input.shape().clone(),
            input.dtype(),
            input.device(),
            false,
        );
        accumulate_grad(&mut gradients, self.input_id, grad_input)?;
        Ok(gradients)
    }

    fn input_ids(&self) -> &[TensorId] {
        std::slice::from_ref(&self.input_id)
    }
}

/// Gradient function for [`crate::ops::reduction::norm`].
///
/// For a finite order `p > 0` the derivative is
///
/// ```text
/// d||x||_p / dx_i = sign(x_i) * (|x_i| / ||x||_p)^(p - 1)
/// ```
///
/// written with the ratio inside the power rather than as
/// `|x_i|^(p-1) / ||x||^(p-1)`: the ratio is bounded by 1, so neither half can
/// overflow on its own for large `p` or large inputs.
///
/// Two points are genuinely undefined rather than merely awkward, and both are
/// resolved the way PyTorch resolves them:
///
/// * At `x = 0` the p-norm has a corner and no derivative. The gradient is
///   reported as 0 (the subgradient of least magnitude). Composing this out of
///   `sqrt(sum(x*x))` instead yields `0/0` = NaN, which then poisons every
///   parameter it touches — a weight-decay term on a freshly zeroed parameter
///   is enough to trigger it.
/// * For `p = ±inf` the norm depends only on the extreme entries, so the
///   gradient is zero elsewhere. When several entries tie, the gradient is
///   split equally among them, matching the min/max reductions in this crate.
pub struct NormBackward {
    pub input_id: TensorId,
    pub input: Tensor,
    /// The norm, kept at reduced rank so it broadcasts back over the input.
    pub norm: Tensor,
    pub p: f64,
    pub dims: Vec<usize>,
    pub keepdim_shape: Shape,
}

impl GradientFunction for NormBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<FxHashMap<TensorId, Tensor>> {
        let mut gradients = FxHashMap::default();
        gradients.reserve(1);

        if self.input.numel() == 0 {
            gradients.insert(
                self.input_id,
                Tensor::zeros(
                    self.input.shape().clone(),
                    self.input.dtype(),
                    self.input.device(),
                    false,
                ),
            );
            return Ok(gradients);
        }

        let full_dims: Vec<isize> = self
            .input
            .shape()
            .dims()
            .iter()
            .map(|&d| d as isize)
            .collect();

        // grad_output arrives at the output's rank; restore the reduced rank so
        // it lines up with the input before broadcasting.
        let grad_kd = grad_output.reshape(self.keepdim_shape.clone())?;
        let grad_full = grad_kd.expand(full_dims.clone())?.contiguous()?;
        let norm_full = self.norm.expand(full_dims.clone())?.contiguous()?;

        let grad_input = if self.p.is_infinite() {
            // Only the extreme entries move the norm. Build the selection mask,
            // count the ties per reduced slice, and share the gradient equally.
            let mask = extremum_mask(&self.input, &norm_full)?;
            let dims_isize: Vec<isize> = self.dims.iter().map(|&d| d as isize).collect();
            let counts = reduction::sum(&mask, Some(dims_isize), true)?;
            let counts_full = counts.expand(full_dims)?.contiguous()?;
            norm_grad_kernel(&self.input, &counts_full, &grad_full, self.p, Some(&mask))?
        } else {
            norm_grad_kernel(&self.input, &norm_full, &grad_full, self.p, None)?
        };

        gradients.insert(self.input_id, grad_input);
        Ok(gradients)
    }

    fn input_ids(&self) -> &[TensorId] {
        std::slice::from_ref(&self.input_id)
    }
}

/// 1.0 where `|x|` equals the (already broadcast) extremum, 0.0 elsewhere.
fn extremum_mask(input: &Tensor, extremum: &Tensor) -> Result<Tensor> {
    macro_rules! mask_for {
        ($ty:ty, $slice:ident, $dtype:expr) => {{
            let x = input.data().$slice().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get slice from norm input")
            })?;
            let m = extremum.data().$slice().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get slice from norm extremum")
            })?;
            TensorData::from_vec::<$ty>(
                crate::ops::map::binary_map(
                    x,
                    m,
                    |x: $ty, m: $ty| {
                        if x.abs() == m { 1.0 } else { 0.0 }
                    },
                ),
                $dtype,
                input.device(),
            )
        }};
    }

    let data = match input.dtype() {
        DataType::Float32 => mask_for!(f32, as_f32_slice, DataType::Float32),
        DataType::Float64 => mask_for!(f64, as_f64_slice, DataType::Float64),
        _ => {
            return Err(MinitensorError::invalid_operation(
                "norm requires floating point tensors",
            ));
        }
    };

    Ok(Tensor::new(
        Arc::new(data),
        input.shape().clone(),
        input.dtype(),
        input.device(),
        false,
    ))
}

/// The elementwise gradient itself. `divisor` is the broadcast norm for a
/// finite order, or the broadcast tie count when `mask` is supplied for
/// `p = ±inf`.
fn norm_grad_kernel(
    input: &Tensor,
    divisor: &Tensor,
    grad: &Tensor,
    p: f64,
    mask: Option<&Tensor>,
) -> Result<Tensor> {
    macro_rules! grad_for {
        ($ty:ty, $slice:ident, $dtype:expr) => {{
            let x = input.data().$slice().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get slice from norm input")
            })?;
            let d = divisor.data().$slice().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get slice from norm divisor")
            })?;
            let g = grad.data().$slice().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get slice from norm grad")
            })?;
            let p_ty = p as $ty;

            let values: Vec<$ty> = match mask {
                Some(mask) => {
                    let m = mask.data().$slice().ok_or_else(|| {
                        MinitensorError::internal_error("Failed to get slice from norm mask")
                    })?;
                    // d is the tie count here, never zero where m is 1.
                    (0..x.len())
                        .map(|i| {
                            if m[i] == 0.0 {
                                0.0
                            } else {
                                g[i] * sign_of(x[i]) / d[i]
                            }
                        })
                        .collect()
                }
                None => (0..x.len())
                    .map(|i| {
                        let n = d[i];
                        if n == 0.0 {
                            // Every entry of this slice is zero: the norm has a
                            // corner here, so take the zero subgradient rather
                            // than dividing 0 by 0.
                            0.0
                        } else if p == 1.0 {
                            g[i] * sign_of(x[i])
                        } else if p == 2.0 {
                            g[i] * x[i] / n
                        } else {
                            let ratio = x[i].abs() / n;
                            g[i] * sign_of(x[i]) * ratio.powf(p_ty - 1.0)
                        }
                    })
                    .collect(),
            };
            TensorData::from_vec::<$ty>(values, $dtype, input.device())
        }};
    }

    let data = match input.dtype() {
        DataType::Float32 => grad_for!(f32, as_f32_slice, DataType::Float32),
        DataType::Float64 => grad_for!(f64, as_f64_slice, DataType::Float64),
        _ => {
            return Err(MinitensorError::invalid_operation(
                "norm requires floating point tensors",
            ));
        }
    };

    Ok(Tensor::new(
        Arc::new(data),
        input.shape().clone(),
        input.dtype(),
        input.device(),
        false,
    ))
}

fn sign_of<T>(v: T) -> T
where
    T: PartialOrd + Default + Copy + std::ops::Neg<Output = T> + From<f32>,
{
    let zero = T::default();
    if v > zero {
        T::from(1.0f32)
    } else if v < zero {
        T::from(-1.0f32)
    } else {
        zero
    }
}
