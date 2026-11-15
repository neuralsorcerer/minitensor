// Copyright (c) Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

use super::*;
use crate::ops::map::{
    PAR_THRESHOLD, outputs_per_task, par_map_indexed, par_out_chunks, par_out_chunks_gated,
};
use crate::ops::normalization::lane_sum_zip;
use crate::ops::shape_ops;
use crate::{
    error::{MinitensorError, Result},
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
    /// The weight as given, `normalized_shape` wide.
    pub weight: Option<Tensor>,
    pub normalized_shape: Vec<usize>,
    pub axis_start: usize,
    pub element_count: usize,
    pub input_requires_grad: bool,
    pub weight_requires_grad: bool,
    pub bias_requires_grad: bool,
}

/// `layer_norm`'s input gradient, a row at a time:
/// `inv_std * (gh - mean(gh) - xhat * mean(gh * xhat))`, with `gh = g * w`.
///
/// This was twelve tensor operations -- two broadcasting multiplies, two
/// reductions, five element-wise passes and the scalars between them -- each
/// allocating and walking a tensor the size of the input. Here each row is
/// read and written while it is in L1, and its two sums run in eight lanes.
/// `gh` is formed in the input's dtype, as the product it replaces was.
macro_rules! layer_norm_input_grad {
    ($name:ident, $ty:ty) => {
        fn $name(
            grad: &[$ty],
            normalized: &[$ty],
            inv_std: &[$ty],
            weight: Option<&[$ty]>,
            norm: usize,
        ) -> Vec<$ty> {
            let mut grad_input = vec![0 as $ty; grad.len()];
            if norm == 0 {
                return grad_input;
            }
            let recip = 1.0 / norm as f64;
            par_out_chunks_gated(&mut grad_input, norm, PAR_THRESHOLD, &|start, gi| {
                let g = &grad[start..start + norm];
                let xhat = &normalized[start..start + norm];
                match weight {
                    Some(w) => {
                        for ((slot, &gv), &wv) in gi.iter_mut().zip(g).zip(w) {
                            *slot = gv * wv;
                        }
                    }
                    None => gi.copy_from_slice(g),
                }
                let mean_g = lane_sum_zip(gi, gi, |v, _| v as f64) * recip;
                let mean_gx = lane_sum_zip(gi, xhat, |v, x| v as f64 * x as f64) * recip;
                let scale = inv_std[start / norm] as f64;
                for (slot, &x) in gi.iter_mut().zip(xhat) {
                    *slot = (scale * (*slot as f64 - mean_g - x as f64 * mean_gx)) as $ty;
                }
            });
            grad_input
        }
    };
}

layer_norm_input_grad!(layer_norm_input_grad_f32, f32);
layer_norm_input_grad!(layer_norm_input_grad_f64, f64);

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
            let grad = grad_output_detached.contiguous()?;
            let weight = self.weight.as_ref().map(|w| w.contiguous()).transpose()?;
            let norm = self.element_count;
            let device = grad.device();
            macro_rules! arm {
                ($accessor:ident, $ty:ty, $variant:ident, $kernel:ident) => {{
                    let bad = |what: &str| {
                        MinitensorError::internal_error(format!(
                            "layer_norm backward: bad {what} slice"
                        ))
                    };
                    let g = grad.data().$accessor().ok_or_else(|| bad("gradient"))?;
                    let xhat = normalized
                        .data()
                        .$accessor()
                        .ok_or_else(|| bad("normalized"))?;
                    let inv = self
                        .inv_std
                        .data()
                        .$accessor()
                        .ok_or_else(|| bad("inv_std"))?;
                    let w = match &weight {
                        Some(w) => Some(w.data().$accessor().ok_or_else(|| bad("weight"))?),
                        None => None,
                    };
                    let values = $kernel(g, xhat, inv, w, norm);
                    TensorData::from_vec::<$ty>(values, DataType::$variant, device)
                }};
            }
            let data = match grad.dtype() {
                DataType::Float32 => arm!(as_f32_slice, f32, Float32, layer_norm_input_grad_f32),
                DataType::Float64 => arm!(as_f64_slice, f64, Float64, layer_norm_input_grad_f64),
                _ => {
                    return Err(MinitensorError::invalid_operation(
                        "layer_norm backward only supports floating point tensors",
                    ));
                }
            };
            let grad_input = Tensor::new(
                Arc::new(data),
                grad.shape().clone(),
                grad.dtype(),
                device,
                false,
            );
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
/// Two conventions, chosen by whether the forward reported an index.
///
/// `max(dim)` and `min(dim)` return the position they selected, so the gradient
/// goes there and nowhere else: the value, the index and the gradient then all
/// name the same element. `amax`/`amin` report no index and have no such
/// element to name, so their gradient is split equally among everything equal
/// to the extremum -- the mean subgradient.
///
/// Both are valid subgradients of a function that has no derivative at a tie.
/// What is not valid is the pair disagreeing, which is what happened while
/// every reduction took the split: `max(dim)` said "position 1" and then fed
/// positions 1 and 2.
///
/// The split path recomputes the extremum, its selection mask and the tie count
/// from the stored (detached) input, so it retains nothing beyond that input.
pub struct MinMaxBackward {
    pub input_id: TensorId,
    pub input: Tensor,
    pub dim: Option<usize>,
    pub keepdim: bool,
    pub is_max: bool,
    pub nan_aware: bool,
    /// The positions the forward reported, for the forms that report any.
    pub selected: Option<Tensor>,
}

/// Put the upstream gradient exactly where the forward said it came from.
///
/// The reductions that report an index -- `max(dim)`, `min(dim)`, `median(dim)`
/// and their NaN-aware forms -- have already chosen one element per reduced
/// slice, so their gradient has one place to go. Splitting it among everything
/// equal to the extremum instead would contradict the index they returned.
///
/// It also sidesteps a hole in the equality-mask path: an all-NaN slice reduces
/// to NaN, `NaN == NaN` is false, so that mask is empty, the tie count is zero
/// and the division is 0/0.
fn route_grad_to_selected(
    input: &Tensor,
    indices: &Tensor,
    grad_output: &Tensor,
    dim: usize,
    keepdim: bool,
) -> Result<Tensor> {
    let input_shape = input.shape().dims().to_vec();
    let dims_vec = Some(vec![dim]);
    let grad_kd = expand_reduction_grad(grad_output, &input_shape, &dims_vec, keepdim)?;
    let index_kd = if keepdim {
        indices.clone()
    } else {
        shape_ops::reshape(indices, grad_kd.shape().clone())?
    };
    let zeros = Tensor::zeros(input.shape().clone(), input.dtype(), input.device(), false);
    shape_ops::scatter(&zeros, dim as isize, &index_kd, &grad_kd)
}

/// Route `grad_output` to every input element equal to the selected reduction
/// value (`reduced`, recomputed with keepdim so it broadcasts), splitting equally
/// among ties. Shared by min/max and median value reductions.
///
/// Each element a slice's value was read from takes the float nearest its
/// share, the gradient over the number of them, and the others take
/// nothing, an infinite or NaN gradient included: multiplying a mask of the
/// ties by the share made every other element `0 · inf`, NaN. A slice whose
/// value is NaN was read from its NaNs, which share its gradient -- equal to
/// nothing, it gave every element of the slice NaN -- unless the reduction
/// leaves NaNs out (`nan_aware`), when a NaN says there was nothing to read
/// and no element takes anything.
fn distribute_selection_grad(
    input: &Tensor,
    reduced: &Tensor,
    grad_output: &Tensor,
    dim: Option<usize>,
    nan_aware: bool,
) -> Result<Tensor> {
    let dims = input.shape().dims();
    let (len, inner) = match dim {
        None => (input.numel(), 1),
        Some(d) => (dims[d], dims[d + 1..].iter().product()),
    };
    let x = input.contiguous()?;
    let value = reduced.contiguous()?;
    let grad = grad_output.contiguous()?;
    let missing = || MinitensorError::internal_error("selection backward: mismatched dtype");
    let data = match input.dtype() {
        DataType::Float32 => {
            let (x, value, grad) = (
                x.data().as_f32_slice().ok_or_else(missing)?,
                value.data().as_f32_slice().ok_or_else(missing)?,
                grad.data().as_f32_slice().ok_or_else(missing)?,
            );
            TensorData::from_vec(
                spread_over_ties(x, value, grad, (len, inner), !nan_aware),
                DataType::Float32,
                input.device(),
            )
        }
        DataType::Float64 => {
            let (x, value, grad) = (
                x.data().as_f64_slice().ok_or_else(missing)?,
                value.data().as_f64_slice().ok_or_else(missing)?,
                grad.data().as_f64_slice().ok_or_else(missing)?,
            );
            TensorData::from_vec(
                spread_over_ties(x, value, grad, (len, inner), !nan_aware),
                DataType::Float64,
                input.device(),
            )
        }
        _ => {
            return Err(MinitensorError::invalid_operation(
                "selection backward only supported for floating point tensors",
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

/// Whether `element` is one a slice whose value is `value` was read from:
/// equal to it, or a NaN where it is NaN and `nans_read`. `x != x` holds
/// only for NaN.
///
/// The tests are combined without branching, so that a row of them runs in
/// vector lanes.
#[allow(clippy::eq_op)]
#[inline(always)]
fn read_from<T: Copy + PartialOrd>(element: T, value: T, nans_read: bool) -> bool {
    (element == value) | (nans_read & (value != value) & (element != element))
}

/// [`distribute_selection_grad`] over slices of `len` elements `inner`
/// apart, `value` and `grad` holding one entry for each.
///
/// The input is taken a band of whole rows at a time, the elements of a row
/// side by side: once to count each slice's elements read, and once more
/// to give them their shares.
fn spread_over_ties<T>(
    x: &[T],
    value: &[T],
    grad: &[T],
    (len, inner): (usize, usize),
    nans_read: bool,
) -> Vec<T>
where
    T: crate::ops::arithmetic::exact_sum::Element
        + crate::ops::arithmetic::exact_sum::Output
        + Copy
        + Default
        + PartialOrd
        + Send
        + Sync,
{
    let mut out = vec![T::default(); x.len()];
    if len == 0 || inner == 0 {
        return out;
    }
    let rows = x.len() / inner;
    let band = ((1usize << 16) / inner).max(1);
    // Each run of a band's rows that lies in one slice, as the slice, its
    // first row and the row past its last.
    let runs = |first: usize, last: usize| {
        let (mut row, mut slice) = (first, first / len);
        std::iter::from_fn(move || {
            (row < last).then(|| {
                let end = ((slice + 1) * len).min(last);
                let run = (slice, row, end);
                (row, slice) = (end, slice + 1);
                run
            })
        })
    };
    let counted = par_map_indexed(rows.div_ceil(band), &|task| {
        let (first, last) = (task * band, ((task + 1) * band).min(rows));
        // A band holds at most 2^16 rows, so its counts fit in 32 bits,
        // which take twice the lanes 64 would.
        let start = first / len;
        let mut counts = vec![0u32; ((last - 1) / len + 1 - start) * inner];
        for (slice, row, end) in runs(first, last) {
            let elements = &x[row * inner..end * inner];
            let tally = &mut counts[(slice - start) * inner..][..inner];
            if inner == 1 {
                let value = value[slice];
                tally[0] += elements
                    .iter()
                    .map(|&element| u32::from(read_from(element, value, nans_read)))
                    .sum::<u32>();
                continue;
            }
            let values = &value[slice * inner..][..inner];
            for row in elements.chunks_exact(inner) {
                for ((count, &element), &value) in tally.iter_mut().zip(row).zip(values) {
                    *count += u32::from(read_from(element, value, nans_read));
                }
            }
        }
        (start, counts)
    });
    let mut counts = vec![0u64; value.len()];
    for (start, part) in counted {
        for (total, count) in counts[start * inner..].iter_mut().zip(part) {
            *total += u64::from(count);
        }
    }
    let shares: Vec<T> = counts
        .iter()
        .zip(grad)
        .map(|(&count, &g)| match count {
            0 | 1 => g,
            _ => crate::ops::arithmetic::exact_sum::weighted_quotient(
                std::iter::once((1u64, g)),
                count,
            ),
        })
        .collect();
    par_out_chunks(&mut out, band * inner, &|offset, chunk| {
        let first = offset / inner;
        for (slice, row, end) in runs(first, first + chunk.len() / inner) {
            let cells = &mut chunk[(row - first) * inner..(end - first) * inner];
            let elements = &x[row * inner..end * inner];
            // Every cell is written, the share or zero, so that the row is a
            // select in vector lanes rather than a branch at each cell.
            let zero = T::default();
            if inner == 1 {
                let (value, share) = (value[slice], shares[slice]);
                for (cell, &element) in cells.iter_mut().zip(elements) {
                    *cell = if read_from(element, value, nans_read) {
                        share
                    } else {
                        zero
                    };
                }
                continue;
            }
            let values = &value[slice * inner..][..inner];
            let given = &shares[slice * inner..][..inner];
            for (cells, row) in cells
                .chunks_exact_mut(inner)
                .zip(elements.chunks_exact(inner))
            {
                for (((cell, &element), &value), &share) in
                    cells.iter_mut().zip(row).zip(values).zip(given)
                {
                    *cell = if read_from(element, value, nans_read) {
                        share
                    } else {
                        zero
                    };
                }
            }
        }
    });
    out
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

        // When the forward named a position, that is the whole answer.
        if let (Some(indices), Some(dim)) = (&self.selected, self.dim) {
            let grad_input =
                route_grad_to_selected(input, indices, grad_output, dim, self.keepdim)?;
            accumulate_grad(&mut gradients, self.input_id, grad_input)?;
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
            distribute_selection_grad(input, &reduced, grad_output, self.dim, self.nan_aware)?;
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
    /// The positions `median(dim)` reported. `nanmedian` reports none.
    pub selected: Option<Tensor>,
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

        if let (Some(indices), Some(dim)) = (&self.selected, self.dim) {
            let grad_input =
                route_grad_to_selected(input, indices, grad_output, dim, self.keepdim)?;
            accumulate_grad(&mut gradients, self.input_id, grad_input)?;
            return Ok(gradients);
        }

        let dim_isize = self.dim.map(|d| d as isize);
        let reduced = if self.nan_aware {
            reduction::nanmedian(input, dim_isize.map(|d| vec![d]), true)?
        } else {
            reduction::median(input, dim_isize, true)?.0
        };
        let grad_input =
            distribute_selection_grad(input, &reduced, grad_output, self.dim, self.nan_aware)?;
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
/// evenly). Each element's share is the float nearest its exact value, and an
/// element the quantile does not read gets nothing, an infinite or NaN
/// gradient included. Groups containing NaN produced NaN and receive no
/// gradient.
pub struct QuantileBackward {
    pub input_id: TensorId,
    pub input: Tensor,
    pub dim: Option<usize>,
    pub q: f64,
    pub interpolation: crate::ops::reduction::QuantileInterpolation,
    pub nan_aware: bool,
}

/// The sorted ranks a quantile of `len` values reads, and how its gradient
/// divides between them: the forward's own position, so that the gradient
/// reaches the ranks the value was read from, ties at a half included.
fn quantile_reading(
    len: usize,
    q: f64,
    interp: crate::ops::reduction::QuantileInterpolation,
) -> Reading {
    use crate::ops::reduction::QuantileInterpolation as Qi;
    let position = crate::ops::reduction::quantile_position_for_len_q(len, q);
    let (lower, upper) = (position.lower_idx, position.upper_idx);
    if lower == upper {
        // A whole position reads its rank alone, whatever the mode.
        return Reading::One(lower);
    }
    match interp {
        Qi::Linear => Reading::Linear(lower, position.weight),
        Qi::Midpoint => Reading::Midpoint(lower),
        Qi::Lower => Reading::One(lower),
        Qi::Higher => Reading::One(upper),
        Qi::Nearest => Reading::One(position.nearest_idx),
    }
}

/// The ranks a quantile reads, from [`quantile_reading`].
#[derive(Clone, Copy)]
enum Reading {
    /// One rank, whose element the quantile is: it takes the whole gradient.
    One(usize),
    /// A rank and the next, the next taking `weight` of the gradient and the
    /// first the rest. The weight lies strictly between 0 and 1.
    Linear(usize, f64),
    /// A rank and the next, half of the gradient each.
    Midpoint(usize),
}

/// A gradient `g` divided between the lower and the upper of two ranks the
/// upper of which takes `weight` of it: `g - g · weight` and `g · weight`,
/// each the float nearest its exact value. Narrowing `1 - weight` and
/// `weight` to the element type and multiplying rounded each share twice.
/// An infinite or NaN `g` is itself in both, as is a zero, the weight lying
/// strictly between 0 and 1.
fn linear_shares<T: crate::ops::arithmetic::exact_sum::Output>(g: f64, weight: f64) -> (T, T) {
    use crate::ops::arithmetic::exact_sum::lerp_of;
    if !g.is_finite() || g == 0.0 {
        return (T::narrow(g), T::narrow(g));
    }
    (
        lerp_of::<T, false>(g, 0.0, weight),
        lerp_of::<T, false>(0.0, g, weight),
    )
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
                            // Only the elements at the ranks read are needed,
                            // so they are selected in O(n) rather than sorted
                            // for. NaN is already filtered, so the comparator
                            // never sees an incomparable value.
                            let cmp =
                                |a: &(usize, $ty), b: &(usize, $ty)| a.1.partial_cmp(&b.1).unwrap();
                            let g = go[o * inner + r];
                            let cell = |d: usize| o * outer_stride + d * inner + r;
                            let reading =
                                quantile_reading(buffer.len(), self.q, self.interpolation);
                            let lower = match reading {
                                Reading::One(rank) => {
                                    buffer.select_nth_unstable_by(rank, cmp);
                                    gi[cell(buffer[rank].0)] = g;
                                    continue;
                                }
                                Reading::Linear(lower, _) | Reading::Midpoint(lower) => lower,
                            };
                            // The upper of the two is selected first, and the
                            // lower is then the largest of the elements left
                            // of it.
                            buffer.select_nth_unstable_by(lower + 1, cmp);
                            let d_up = buffer[lower + 1].0;
                            buffer[..lower + 1].select_nth_unstable_by(lower, cmp);
                            let d_lo = buffer[lower].0;
                            let (low, high) = match reading {
                                Reading::Linear(_, weight) => {
                                    linear_shares::<$ty>(g as f64, weight)
                                }
                                // Halving is exact but where it underflows,
                                // and rounds once there.
                                _ => (g * 0.5, g * 0.5),
                            };
                            gi[cell(d_lo)] = low;
                            gi[cell(d_up)] = high;
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
/// resolved deliberately:
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

/// Gradient of RMSNorm.
///
/// With `r = 1/sqrt(mean(x^2) + eps)` and `y_i = x_i * r * w_i`, the input
/// couples to every element of its row through `r`:
///
/// ```text
///     dL/dx_j = r * (g_j w_j  -  x_j * r^2 * dot / N),   dot = sum_i g_i w_i x_i
///     dL/dw_i = sum over rows of  g_i x_i r
/// ```
///
/// `r` is saved from the forward, so this is one reduction pass per row for
/// `dot` and one to write -- the same shape as the fused forward. The forward
/// used to be a chain of autograd-tracked primitives, which is what let RMSNorm
/// go without an explicit gradient at all; fusing it means writing this out.
pub struct RmsNormBackward {
    pub input_ids: SmallVec<[TensorId; 2]>,
    pub input_id: TensorId,
    pub weight_id: Option<TensorId>,
    pub input: Tensor,
    pub inv_rms: Tensor,
    pub weight: Option<Tensor>,
    pub normalized_shape: Vec<usize>,
    pub element_count: usize,
    pub input_requires_grad: bool,
    pub weight_requires_grad: bool,
}

macro_rules! rms_norm_grad {
    ($name:ident, $ty:ty) => {
        #[allow(clippy::too_many_arguments)]
        fn $name(
            input: &[$ty],
            grad: &[$ty],
            inv_rms: &[$ty],
            weight: Option<&[$ty]>,
            norm: usize,
            want_input: bool,
            want_weight: bool,
        ) -> (Vec<$ty>, Vec<$ty>) {
            let mut grad_input = vec![0 as $ty; if want_input { input.len() } else { 0 }];
            if norm == 0 {
                return (
                    grad_input,
                    vec![0 as $ty; if want_weight { norm } else { 0 }],
                );
            }
            let recip = 1.0 / norm as f64;

            if want_input {
                par_out_chunks_gated(&mut grad_input, norm, PAR_THRESHOLD, &|start, gi| {
                    let r = inv_rms[start / norm] as f64;
                    let row = &input[start..start + norm];
                    let g = &grad[start..start + norm];
                    let mut dot = 0.0f64;
                    for i in 0..norm {
                        let w = weight.map_or(1.0, |w| w[i] as f64);
                        dot += g[i] as f64 * w * row[i] as f64;
                    }
                    let coeff = r * r * dot * recip;
                    for i in 0..norm {
                        let w = weight.map_or(1.0, |w| w[i] as f64);
                        gi[i] = (r * (g[i] as f64 * w - row[i] as f64 * coeff)) as $ty;
                    }
                });
            }

            // The weight's gradient sums over every row, in bands of rows
            // whose size follows from the shape alone: the grouping of a
            // float sum decides its rounding, and a fold whose grouping the
            // pool chose answered by the thread count. Below
            // `PAR_THRESHOLD` elements the whole of it is one band on the
            // calling thread.
            let grad_weight = if want_weight {
                let rows = inv_rms.len();
                let band_rows = if rows * norm < PAR_THRESHOLD {
                    rows.max(1)
                } else {
                    outputs_per_task(norm)
                };
                let partial = |band: usize| {
                    let mut acc = vec![0.0f64; norm];
                    for r in band * band_rows..((band + 1) * band_rows).min(rows) {
                        let scale = inv_rms[r] as f64;
                        let row = &input[r * norm..][..norm];
                        let g = &grad[r * norm..][..norm];
                        for i in 0..norm {
                            acc[i] += g[i] as f64 * row[i] as f64 * scale;
                        }
                    }
                    acc
                };
                let bands = rows.div_ceil(band_rows);
                let mut partials = if bands > 1 {
                    par_map_indexed(bands, &partial).into_iter()
                } else {
                    vec![partial(0)].into_iter()
                };
                let mut total = partials.next().unwrap_or_else(|| vec![0.0f64; norm]);
                for part in partials {
                    for (sum, value) in total.iter_mut().zip(part) {
                        *sum += value;
                    }
                }
                total.into_iter().map(|v| v as $ty).collect()
            } else {
                Vec::new()
            };
            (grad_input, grad_weight)
        }
    };
}

rms_norm_grad!(rms_norm_grad_f32, f32);
rms_norm_grad!(rms_norm_grad_f64, f64);

impl GradientFunction for RmsNormBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<FxHashMap<TensorId, Tensor>> {
        let mut gradients = FxHashMap::default();
        let grad_output = grad_output.detach();
        let norm = self.element_count;
        let device = self.input.device();
        let dtype = self.input.dtype();

        macro_rules! arm {
            ($accessor:ident, $ty:ty, $variant:ident, $kernel:ident) => {{
                let inp = self.input.data().$accessor().ok_or_else(|| {
                    MinitensorError::internal_error("rms_norm backward: bad input slice")
                })?;
                let g = grad_output.data().$accessor().ok_or_else(|| {
                    MinitensorError::internal_error("rms_norm backward: bad gradient slice")
                })?;
                let r = self.inv_rms.data().$accessor().ok_or_else(|| {
                    MinitensorError::internal_error("rms_norm backward: bad inv_rms slice")
                })?;
                let w = self.weight.as_ref().and_then(|t| t.data().$accessor());
                let (gi, gw) = $kernel(
                    inp,
                    g,
                    r,
                    w,
                    norm,
                    self.input_requires_grad,
                    self.weight_requires_grad,
                );
                if self.input_requires_grad {
                    let t = Tensor::new(
                        Arc::new(TensorData::from_vec::<$ty>(gi, DataType::$variant, device)),
                        self.input.shape().clone(),
                        DataType::$variant,
                        device,
                        false,
                    );
                    accumulate_grad(&mut gradients, self.input_id, t)?;
                }
                if self.weight_requires_grad
                    && let Some(weight_id) = self.weight_id
                {
                    let t = Tensor::new(
                        Arc::new(TensorData::from_vec::<$ty>(gw, DataType::$variant, device)),
                        Shape::new(self.normalized_shape.clone()),
                        DataType::$variant,
                        device,
                        false,
                    );
                    accumulate_grad(&mut gradients, weight_id, t)?;
                }
            }};
        }

        match dtype {
            DataType::Float32 => arm!(as_f32_slice, f32, Float32, rms_norm_grad_f32),
            DataType::Float64 => arm!(as_f64_slice, f64, Float64, rms_norm_grad_f64),
            _ => {
                return Err(MinitensorError::invalid_operation(
                    "rms_norm backward only supports floating point tensors",
                ));
            }
        }
        Ok(gradients)
    }

    fn input_ids(&self) -> &[TensorId] {
        &self.input_ids
    }
}

/// Gradient function for a mean: the output's gradient over the count,
/// repeated over the axes averaged away.
///
/// The quotient is formed as the forward one is, by the exact count; see
/// `divide_by_count`.
pub struct MeanBackward {
    pub input_id: TensorId,
    pub input_shape: Vec<usize>,
    pub dims: Option<Vec<usize>>,
    pub keepdim: bool,
    pub count: f64,
}

impl GradientFunction for MeanBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<FxHashMap<TensorId, Tensor>> {
        let mut gradients = FxHashMap::default();
        let share = reduction::divide_by_count(grad_output, self.count)?;
        let grad = expand_reduction_grad(&share, &self.input_shape, &self.dims, self.keepdim)?;
        // An empty input has nothing to repeat over; see `SumBackward`.
        let grad_input = if self.input_shape.contains(&0) {
            Tensor::zeros(
                Shape::new(self.input_shape.clone()),
                grad_output.dtype(),
                grad_output.device(),
                false,
            )
        } else {
            let dims: Vec<isize> = self.input_shape.iter().map(|&d| d as isize).collect();
            grad.expand(dims)?.contiguous()?
        };
        gradients.insert(self.input_id, grad_input);
        Ok(gradients)
    }

    fn input_ids(&self) -> &[TensorId] {
        std::slice::from_ref(&self.input_id)
    }
}

/// Gradient of the square root that turns a variance into a standard
/// deviation.
///
/// It is [`SqrtBackward`] except at a deviation of zero, where the square root
/// has an infinite slope and the variance a zero one. Composed, that is
/// `inf * 0` = NaN for every element of a constant slice; reported here as 0,
/// the subgradient `norm` reports at the origin, which the variance's own
/// gradient (proportional to each deviation) then carries back unchanged.
pub struct StdBackward {
    pub input_id: TensorId,
    /// The standard deviation, same shape as the variance it was taken from.
    pub output: Tensor,
}

impl GradientFunction for StdBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<FxHashMap<TensorId, Tensor>> {
        let mut gradients = FxHashMap::default();
        gradients.reserve(1);
        if grad_output.shape() != self.output.shape() || grad_output.dtype() != self.output.dtype()
        {
            return Err(MinitensorError::internal_error(
                "standard deviation gradient does not match its output",
            ));
        }

        macro_rules! scaled {
            ($slice:ident, $from_vec:ident, $t:ty) => {{
                let s = self.output.data().$slice().ok_or_else(|| {
                    MinitensorError::internal_error("standard deviation has an unexpected dtype")
                })?;
                let g = grad_output.data().$slice().ok_or_else(|| {
                    MinitensorError::internal_error("gradient has an unexpected dtype")
                })?;
                let values: Vec<$t> = s
                    .iter()
                    .zip(g)
                    .map(|(&s, &g)| if s == 0.0 { 0.0 } else { g / (s + s) })
                    .collect();
                TensorData::$from_vec(values, self.output.device())
            }};
        }
        let data = match self.output.dtype() {
            DataType::Float32 => scaled!(as_f32_slice, from_vec_f32, f32),
            DataType::Float64 => scaled!(as_f64_slice, from_vec_f64, f64),
            _ => {
                return Err(MinitensorError::internal_error(
                    "standard deviation of a non-float tensor",
                ));
            }
        };
        gradients.insert(
            self.input_id,
            Tensor::new(
                Arc::new(data),
                self.output.shape().clone(),
                self.output.dtype(),
                self.output.device(),
                false,
            ),
        );
        Ok(gradients)
    }

    fn input_ids(&self) -> &[TensorId] {
        std::slice::from_ref(&self.input_id)
    }
}
