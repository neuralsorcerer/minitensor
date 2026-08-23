// Copyright (c) Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

use super::*;
use crate::{
    error::{MinitensorError, Result},
    ops::map::{PAR_CHUNK, par_out_chunks},
    ops::reduction,
    tensor::{DataType, Shape, Strides, Tensor, TensorData},
};
use rustc_hash::FxHashMap;
use smallvec::SmallVec;
use std::sync::Arc;

/// Write `grad_out[i]` where `mask[i]` is set and `0` elsewhere.
///
/// This is the *select* form used by hardshrink: a cleared mask yields an
/// exact zero, discarding any NaN in the incoming gradient. Kernels that must
/// propagate NaN (ReLU) multiply by the mask via [`zip_mask_into`] instead.
///
/// Chunked so it stays on the calling thread for small gradients and
/// vectorises inside each chunk; replaces the raw-pointer parallel loops these
/// kernels used to hand-roll (which laundered pointers through `usize` to
/// cross the rayon closure boundary).
pub(super) fn mask_select_into<T: Copy + Default + Send + Sync>(
    out: &mut [T],
    grad_out: &[T],
    mask: &[bool],
) {
    zip_mask_into(
        out,
        grad_out,
        mask,
        |g, keep| if keep { g } else { T::default() },
    );
}

/// Generalisation of [`mask_select_into`] for masks that scale rather than
/// replace the gradient (ReLU, leaky ReLU).
pub(super) fn zip_mask_into<T, F>(out: &mut [T], grad_out: &[T], mask: &[bool], op: F)
where
    T: Copy + Send + Sync,
    F: Fn(T, bool) -> T + Send + Sync,
{
    debug_assert_eq!(out.len(), grad_out.len());
    debug_assert_eq!(out.len(), mask.len());
    let apply = |out: &mut [T], grad_out: &[T], mask: &[bool]| {
        for ((o, &g), &m) in out.iter_mut().zip(grad_out.iter()).zip(mask.iter()) {
            *o = op(g, m);
        }
    };
    if out.len() < PAR_THRESHOLD {
        apply(out, grad_out, mask);
    } else {
        par_out_chunks(out, PAR_CHUNK, &|start, o| {
            let span = start..start + o.len();
            apply(o, &grad_out[span.clone()], &mask[span]);
        });
    }
}

/// Scatter-add `grad_output` back to the source positions selected along `dim`.
///
/// `indices[i]` is the source position (along `dim`) that produced output row `i`
/// for every outer/inner coordinate. This is the shared backward for
/// `index_select` and `slice` (and, transitively, `narrow`/`flip`/`roll`).
/// Duplicated source indices accumulate, matching the forward gather semantics.
fn index_select_backward_grad(
    grad_output: &Tensor,
    input_shape: &[usize],
    dim: usize,
    indices: &[usize],
) -> Result<Tensor> {
    let numel: usize = input_shape.iter().product();
    let mut grad_data =
        TensorData::zeros_on_device(numel, grad_output.dtype(), grad_output.device());

    let dim_size = input_shape[dim];
    let inner: usize = input_shape[dim + 1..].iter().product();
    let out_dim = indices.len();

    if numel != 0 && out_dim != 0 && inner != 0 {
        let in_chunk = dim_size * inner;
        let out_chunk = out_dim * inner;

        macro_rules! fill {
            ($slice:ident, $mut_slice:ident) => {{
                let go = grad_output.data().$slice().ok_or_else(|| {
                    MinitensorError::internal_error("Failed to read grad_output for index backward")
                })?;
                let gi = grad_data.$mut_slice().ok_or_else(|| {
                    MinitensorError::internal_error("Failed to write grad for index backward")
                })?;
                par_out_chunks(gi, in_chunk, &|start, gi_chunk| {
                    let o = start / in_chunk;
                    let go_chunk = &go[o * out_chunk..(o + 1) * out_chunk];
                    for (i, &idx) in indices.iter().enumerate() {
                        let dst = idx * inner;
                        let src = i * inner;
                        for j in 0..inner {
                            gi_chunk[dst + j] += go_chunk[src + j];
                        }
                    }
                });
            }};
        }

        match grad_output.dtype() {
            DataType::Float32 => fill!(as_f32_slice, as_f32_slice_mut),
            DataType::Float64 => fill!(as_f64_slice, as_f64_slice_mut),
            _ => {
                return Err(MinitensorError::invalid_operation(
                    "index/slice backward only supported for floating point tensors",
                ));
            }
        }
    }

    Ok(Tensor::new(
        Arc::new(grad_data),
        Shape::new(input_shape.to_vec()),
        grad_output.dtype(),
        grad_output.device(),
        false,
    ))
}

/// Scatter-add `grad_output` back to the input positions named by a full `index`
/// tensor (`gather` backward, also reused by min/max/sort/topk along a dim). The
/// `index` slice is laid out identically to `grad_output`; entry `index[..]` is
/// the source coordinate along `dim`. Colliding indices accumulate.
fn gather_backward_grad(
    grad_output: &Tensor,
    input_shape: &[usize],
    dim: usize,
    index: &[i64],
) -> Result<Tensor> {
    let numel: usize = input_shape.iter().product();
    let mut grad_data =
        TensorData::zeros_on_device(numel, grad_output.dtype(), grad_output.device());

    let dim_size = input_shape[dim];
    let inner: usize = input_shape[dim + 1..].iter().product();
    // The index tensor shares `grad_output`'s shape, so the output extent along
    // `dim` is read directly from it.
    let out_dim = grad_output.shape().dims()[dim];

    if numel != 0 && !index.is_empty() && inner != 0 {
        let in_chunk = dim_size * inner;
        let out_chunk = out_dim * inner;

        macro_rules! fill {
            ($slice:ident, $mut_slice:ident) => {{
                let go = grad_output.data().$slice().ok_or_else(|| {
                    MinitensorError::internal_error(
                        "Failed to read grad_output for gather backward",
                    )
                })?;
                let gi = grad_data.$mut_slice().ok_or_else(|| {
                    MinitensorError::internal_error("Failed to write grad for gather backward")
                })?;
                par_out_chunks(gi, in_chunk, &|start, gi_chunk| {
                    let o = start / in_chunk;
                    let go_chunk = &go[o * out_chunk..(o + 1) * out_chunk];
                    let idx_chunk = &index[o * out_chunk..(o + 1) * out_chunk];
                    for i in 0..out_dim {
                        for j in 0..inner {
                            let pos = i * inner + j;
                            let src_idx = idx_chunk[pos] as usize;
                            gi_chunk[src_idx * inner + j] += go_chunk[pos];
                        }
                    }
                });
            }};
        }

        match grad_output.dtype() {
            DataType::Float32 => fill!(as_f32_slice, as_f32_slice_mut),
            DataType::Float64 => fill!(as_f64_slice, as_f64_slice_mut),
            _ => {
                return Err(MinitensorError::invalid_operation(
                    "gather backward only supported for floating point tensors",
                ));
            }
        }
    }

    Ok(Tensor::new(
        Arc::new(grad_data),
        Shape::new(input_shape.to_vec()),
        grad_output.dtype(),
        grad_output.device(),
        false,
    ))
}

/// Gradient function for `index_select` and `slice` (source indices along `dim`).
pub struct IndexSelectBackward {
    pub input_id: TensorId,
    pub input_shape: Vec<usize>,
    pub dim: usize,
    pub indices: Vec<usize>,
}

impl GradientFunction for IndexSelectBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<FxHashMap<TensorId, Tensor>> {
        let grad_input =
            index_select_backward_grad(grad_output, &self.input_shape, self.dim, &self.indices)?;
        let mut gradients = FxHashMap::default();
        accumulate_grad(&mut gradients, self.input_id, grad_input)?;
        Ok(gradients)
    }

    fn input_ids(&self) -> &[TensorId] {
        std::slice::from_ref(&self.input_id)
    }
}

/// Gradient function for `gather` (and, reused, min/max/sort/topk along a dim).
pub struct GatherBackward {
    pub input_id: TensorId,
    pub input_shape: Vec<usize>,
    pub dim: usize,
    pub index: Vec<i64>,
}

impl GradientFunction for GatherBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<FxHashMap<TensorId, Tensor>> {
        let grad_input =
            gather_backward_grad(grad_output, &self.input_shape, self.dim, &self.index)?;
        let mut gradients = FxHashMap::default();
        accumulate_grad(&mut gradients, self.input_id, grad_input)?;
        Ok(gradients)
    }

    fn input_ids(&self) -> &[TensorId] {
        std::slice::from_ref(&self.input_id)
    }
}

/// Gradient function for `concatenate` (and, transitively, `cat`/`stack`/`roll`).
pub struct ConcatBackward {
    pub input_ids: SmallVec<[TensorId; 4]>,
    pub sizes: SmallVec<[usize; 4]>,
    pub dim: usize,
    /// Which inputs actually need a gradient; frozen inputs skip their
    /// slice extraction.
    pub input_requires_grad: SmallVec<[bool; 4]>,
}

impl GradientFunction for ConcatBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<FxHashMap<TensorId, Tensor>> {
        let mut gradients = FxHashMap::default();
        let mut offset = 0usize;
        for ((&id, &size), &needs_grad) in self
            .input_ids
            .iter()
            .zip(self.sizes.iter())
            .zip(self.input_requires_grad.iter())
        {
            if needs_grad {
                let grad_slice =
                    crate::ops::shape_ops::narrow(grad_output, self.dim as isize, offset, size)?;
                accumulate_grad(&mut gradients, id, grad_slice)?;
            }
            offset += size;
        }
        Ok(gradients)
    }

    fn input_ids(&self) -> &[TensorId] {
        &self.input_ids
    }
}

/// Gradient function for `roll`: rolling is a bijection, so the gradient is the
/// input rolled back by the negated shifts. Computed with a dedicated node rather
/// than by composing `slice`/`concatenate`, because `roll`'s flatten path builds
/// a storage-sharing view whose gradient edges cannot be composed safely.
pub struct RollBackward {
    pub input_id: TensorId,
    pub shifts: Vec<isize>,
    pub dims: Option<Vec<isize>>,
}

impl GradientFunction for RollBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<FxHashMap<TensorId, Tensor>> {
        let neg: Vec<isize> = self.shifts.iter().map(|s| -s).collect();
        let grad_input = crate::ops::shape_ops::roll(grad_output, &neg, self.dims.as_deref())?;
        let mut gradients = FxHashMap::default();
        accumulate_grad(&mut gradients, self.input_id, grad_input)?;
        Ok(gradients)
    }

    fn input_ids(&self) -> &[TensorId] {
        std::slice::from_ref(&self.input_id)
    }
}

/// Gradient function for `flip`.
///
/// Reversing an axis is its own inverse, so the gradient is the same flip
/// applied to the incoming one. Carried as a single node because `flip` fills
/// its output directly; going through one `index_select` per dimension, as it
/// used to, left a gradient edge per dimension as well.
pub struct FlipBackward {
    pub input_id: TensorId,
    pub dims: Vec<isize>,
}

impl GradientFunction for FlipBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<FxHashMap<TensorId, Tensor>> {
        let grad_input = crate::ops::shape_ops::flip(grad_output, &self.dims)?;
        let mut gradients = FxHashMap::default();
        accumulate_grad(&mut gradients, self.input_id, grad_input)?;
        Ok(gradients)
    }

    fn input_ids(&self) -> &[TensorId] {
        std::slice::from_ref(&self.input_id)
    }
}

/// Gradient function for `repeat` (tiling): sum the gradient over the tiled copies.
pub struct RepeatBackward {
    pub input_id: TensorId,
    pub input_shape: Vec<usize>,
    pub repeats: Vec<usize>,
}

impl GradientFunction for RepeatBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<FxHashMap<TensorId, Tensor>> {
        // `repeat` may prepend leading singleton axes; align the input rank to the
        // repeat/output rank, tile every axis, then sum the tiled copies back down.
        let out_ndim = self.repeats.len();
        let pad = out_ndim - self.input_shape.len();
        let mut aligned = vec![1usize; pad];
        aligned.extend_from_slice(&self.input_shape);

        // View grad_output as (rep_0, in_0, rep_1, in_1, ...) then sum the rep axes.
        let mut split_shape = Vec::with_capacity(2 * out_ndim);
        for axis in 0..out_ndim {
            split_shape.push(self.repeats[axis]);
            split_shape.push(aligned[axis]);
        }
        let reshaped = crate::ops::shape_ops::reshape(grad_output, Shape::new(split_shape))?;
        let rep_axes: Vec<isize> = (0..out_ndim).map(|axis| (2 * axis) as isize).collect();
        let summed = reduction::sum(&reshaped, Some(rep_axes), false)?;
        let grad_input =
            crate::ops::shape_ops::reshape(&summed, Shape::new(self.input_shape.clone()))?;

        let mut gradients = FxHashMap::default();
        accumulate_grad(&mut gradients, self.input_id, grad_input)?;
        Ok(gradients)
    }

    fn input_ids(&self) -> &[TensorId] {
        std::slice::from_ref(&self.input_id)
    }
}

/// Gradient function for basic indexing (`tensor[...]` via [`Tensor::index`]).
///
/// The forward gathers input element `offset + Σ_j (start_j + coord_j·step_j)·
/// input_stride_{dim_j}` for each output coordinate; the backward scatters the
/// gradient straight back to those positions. Assumes contiguous input storage,
/// which always holds at the Python boundary where indexing is applied.
pub struct IndexBackward {
    pub input_id: TensorId,
    pub input_shape: Vec<usize>,
    pub input_strides: Vec<usize>,
    pub offset: usize,
    pub out_dims: Vec<usize>,
    pub orig_dim_map: Vec<usize>,
    pub starts: Vec<usize>,
    pub steps: Vec<usize>,
}

impl GradientFunction for IndexBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<FxHashMap<TensorId, Tensor>> {
        let numel: usize = self.input_shape.iter().product();
        let mut grad_data =
            TensorData::zeros_on_device(numel, grad_output.dtype(), grad_output.device());
        let out_strides = Strides::from_shape(&Shape::new(self.out_dims.clone()));
        let out_strides = out_strides.as_slice();

        macro_rules! scatter {
            ($slice:ident, $mut_slice:ident) => {{
                let go = grad_output.data().$slice().ok_or_else(|| {
                    MinitensorError::internal_error("Failed to read grad_output for index backward")
                })?;
                let gi = grad_data.$mut_slice().ok_or_else(|| {
                    MinitensorError::internal_error("Failed to write grad for index backward")
                })?;
                if self.out_dims.is_empty() {
                    // Scalar result: a single collapsed element.
                    gi[self.offset] += go[0];
                } else {
                    for (idx, &g) in go.iter().enumerate() {
                        let mut rem = idx;
                        let mut src = self.offset;
                        for (j, &ostride) in out_strides.iter().enumerate() {
                            let coord = rem / ostride;
                            rem %= ostride;
                            src += (self.starts[j] + coord * self.steps[j])
                                * self.input_strides[self.orig_dim_map[j]];
                        }
                        gi[src] += g;
                    }
                }
            }};
        }

        match grad_output.dtype() {
            DataType::Float32 => scatter!(as_f32_slice, as_f32_slice_mut),
            DataType::Float64 => scatter!(as_f64_slice, as_f64_slice_mut),
            _ => {
                return Err(MinitensorError::invalid_operation(
                    "index backward only supported for floating point tensors",
                ));
            }
        }

        let grad_input = Tensor::new(
            Arc::new(grad_data),
            Shape::new(self.input_shape.clone()),
            grad_output.dtype(),
            grad_output.device(),
            false,
        );
        let mut gradients = FxHashMap::default();
        accumulate_grad(&mut gradients, self.input_id, grad_input)?;
        Ok(gradients)
    }

    fn input_ids(&self) -> &[TensorId] {
        std::slice::from_ref(&self.input_id)
    }
}

/// Gradient function for `scatter_add`.
///
/// `out = input.clone(); out[index] += src`, so the input passes its gradient
/// through untouched — every original value still contributes exactly once —
/// while each source element collects the gradient at the slot it was added to.
/// Duplicate indices need no special handling: addition is linear, so several
/// sources landing on one slot all see that slot's gradient.
pub struct ScatterAddBackward {
    /// [input, src]. Both must be listed: the engine walks this to reach each
    /// operand's own grad_fn, so omitting `src` silently truncates the graph
    /// whenever `src` is computed rather than a leaf.
    pub input_ids: [TensorId; 2],
    /// Which of [input, src] need a gradient; each half is skipped when frozen.
    pub input_requires_grad: [bool; 2],
    pub src_shape: Shape,
    pub dim: usize,
    pub indices: Vec<i64>,
}

impl GradientFunction for ScatterAddBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<FxHashMap<TensorId, Tensor>> {
        let mut gradients = FxHashMap::default();

        if self.input_requires_grad[0] {
            accumulate_grad(&mut gradients, self.input_ids[0], grad_output.clone())?;
        }

        if self.input_requires_grad[1] {
            let dims = grad_output.shape().dims();
            let inner: usize = dims[self.dim + 1..].iter().product();
            let src_grad = crate::ops::shape_ops::gather_grad_for_src(
                grad_output,
                &self.src_shape,
                inner,
                dims[self.dim],
                self.src_shape.dims()[self.dim],
                &self.indices,
                None,
            )?;
            accumulate_grad(&mut gradients, self.input_ids[1], src_grad)?;
        }

        Ok(gradients)
    }

    fn input_ids(&self) -> &[TensorId] {
        &self.input_ids
    }
}

/// Gradient function for `scatter`.
///
/// Overwriting, unlike accumulating, severs the dependency: a slot that was
/// written no longer depends on the input's original value there, and when two
/// indices name the same slot only the surviving writer affected the output.
/// `winners` records which one that was, so both halves of the gradient stay
/// exact even with duplicate indices.
pub struct ScatterBackward {
    /// [input, src]; see [`ScatterAddBackward`] on why both belong here.
    pub input_ids: [TensorId; 2],
    pub input_requires_grad: [bool; 2],
    pub src_shape: Shape,
    pub dim: usize,
    pub inner: usize,
    pub input_dim: usize,
    pub index_dim: usize,
    pub indices: Vec<i64>,
    /// Per destination slot, the index-axis position that wrote it last, or
    /// `usize::MAX` where nothing did.
    pub winners: Vec<usize>,
}

impl GradientFunction for ScatterBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<FxHashMap<TensorId, Tensor>> {
        let mut gradients = FxHashMap::default();

        if self.input_requires_grad[0] {
            let masked = crate::ops::shape_ops::mask_overwritten(grad_output, &self.winners)?;
            accumulate_grad(&mut gradients, self.input_ids[0], masked)?;
        }

        if self.input_requires_grad[1] {
            let src_grad = crate::ops::shape_ops::gather_grad_for_src(
                grad_output,
                &self.src_shape,
                self.inner,
                self.input_dim,
                self.index_dim,
                &self.indices,
                Some(&self.winners),
            )?;
            accumulate_grad(&mut gradients, self.input_ids[1], src_grad)?;
        }

        Ok(gradients)
    }

    fn input_ids(&self) -> &[TensorId] {
        &self.input_ids
    }
}

pub(crate) fn repeat_interleave_backward_impl(
    grad_output: &Tensor,
    input_shape: &[usize],
    repeats: &[usize],
    dim: usize,
) -> Result<Tensor> {
    if dim >= input_shape.len() {
        return Err(MinitensorError::index_error(
            dim as isize,
            0,
            input_shape.len(),
        ));
    }

    let dim_size = input_shape[dim];
    if repeats.len() != dim_size {
        return Err(MinitensorError::invalid_operation(
            "repeat_interleave backward: repeats must match input dimension size".to_string(),
        ));
    }

    let grad_shape_vec = input_shape.to_vec();
    let grad_shape = Shape::new(grad_shape_vec.clone());
    let numel = grad_shape.numel();
    let dtype = grad_output.dtype();
    let device = grad_output.device();
    let total_repeats: usize = repeats.iter().sum();

    let inner: usize = if dim + 1 >= input_shape.len() {
        1
    } else {
        input_shape[dim + 1..].iter().product()
    };
    let outer: usize = if dim == 0 {
        1
    } else {
        input_shape[..dim].iter().product()
    };

    if numel == 0 || total_repeats == 0 || inner == 0 || outer == 0 {
        return Ok(Tensor::zeros(
            Shape::new(grad_shape_vec),
            dtype,
            device,
            false,
        ));
    }

    let output_dims = grad_output.shape().dims();
    if output_dims.len() != input_shape.len() || output_dims[dim] != total_repeats {
        return Err(MinitensorError::shape_mismatch(
            input_shape.to_vec(),
            output_dims.to_vec(),
        ));
    }

    macro_rules! repeat_interleave_backward_impl_inner {
        ($ty:ty, $slice:ident, $from_vec:ident) => {{
            let src = grad_output.data().$slice().ok_or_else(|| {
                MinitensorError::invalid_operation(
                    "repeat_interleave backward: gradient tensor must be contiguous".to_string(),
                )
            })?;
            let mut dst = vec![<$ty>::default(); numel];
            let chunk = total_repeats * inner;
            let span = dim_size * inner;
            par_out_chunks(&mut dst, span, &|start, dst_chunk| {
                let mut src_offset = (start / span) * chunk;
                for (i, &rep) in repeats.iter().enumerate() {
                    if rep == 0 {
                        continue;
                    }
                    let dst_start = i * inner;
                    let dst_slice = &mut dst_chunk[dst_start..dst_start + inner];
                    for _ in 0..rep {
                        let src_slice = &src[src_offset..src_offset + inner];
                        dst_slice.iter_mut().zip(src_slice.iter()).for_each(
                            |(dst_val, &src_val)| {
                                *dst_val += src_val;
                            },
                        );
                        src_offset += inner;
                    }
                }
            });
            TensorData::$from_vec(dst, device)
        }};
    }

    let data = match dtype {
        DataType::Float32 => {
            repeat_interleave_backward_impl_inner!(f32, as_f32_slice, from_vec_f32)
        }
        DataType::Float64 => {
            repeat_interleave_backward_impl_inner!(f64, as_f64_slice, from_vec_f64)
        }
        DataType::Int32 => repeat_interleave_backward_impl_inner!(i32, as_i32_slice, from_vec_i32),
        DataType::Int64 => repeat_interleave_backward_impl_inner!(i64, as_i64_slice, from_vec_i64),
        DataType::Bool => {
            return Ok(Tensor::zeros(grad_shape, dtype, device, false));
        }
    };

    Ok(Tensor::new(
        Arc::new(data),
        grad_shape,
        dtype,
        device,
        false,
    ))
}
/// Gradient function for expand operation which reduces broadcasted gradients
pub struct ExpandBackward {
    pub input_shape: Vec<usize>,
    pub input_id: TensorId,
}
impl GradientFunction for ExpandBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<FxHashMap<TensorId, Tensor>> {
        let mut gradients = FxHashMap::default();
        gradients.reserve(1);

        let shape = Shape::new(self.input_shape.clone());
        let grad_input = reduce_gradient_for_broadcasting(grad_output, &shape)?;
        gradients.insert(self.input_id, grad_input);
        Ok(gradients)
    }

    fn input_ids(&self) -> &[TensorId] {
        std::slice::from_ref(&self.input_id)
    }
}

/// Gradient function for reshape operation
/// Gradient of [`crate::ops::shape_ops::pad`].
///
/// Padding sends each output position to an input position or to nothing, so
/// the gradient sends each output gradient back the same way -- and *adds*,
/// because reflect and replicate send many output positions to one input. A
/// replicated edge that was copied five times has five gradients arriving at
/// it, and dropping four of them would be a silent under-count of exactly the
/// elements padding touched. Constant padding is the case where nothing
/// accumulates: its map is injective and the positions with no source
/// contribute nothing at all.
///
/// The map is the one the forward built, carried here rather than rebuilt, so
/// the two cannot disagree about where anything came from.
pub struct PadBackward {
    pub input_shape: Vec<usize>,
    pub map: Vec<Option<usize>>,
    pub input_id: TensorId,
    pub ids: [TensorId; 1],
}

impl GradientFunction for PadBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<FxHashMap<TensorId, Tensor>> {
        let mut gradients = FxHashMap::default();
        gradients.reserve(1);

        let shape = Shape::new(self.input_shape.clone());
        let numel = shape.numel();
        let grad = grad_output.contiguous()?;
        let mut data =
            crate::tensor::TensorData::zeros_on_device(numel, grad.dtype(), grad.device());

        // Serial: the scatter is many-to-one, so parallel writers would race on
        // the shared edge elements. It is one pass over the padded tensor.
        macro_rules! scatter {
            ($accessor:ident, $accessor_mut:ident) => {{
                let src = grad.data().$accessor().ok_or_else(|| {
                    MinitensorError::internal_error("pad backward: unexpected gradient dtype")
                })?;
                let dst = data.$accessor_mut().ok_or_else(|| {
                    MinitensorError::internal_error("pad backward: unexpected output dtype")
                })?;
                for (out, source) in self.map.iter().enumerate() {
                    if let Some(index) = source {
                        dst[*index] += src[out];
                    }
                }
            }};
        }

        match grad.dtype() {
            crate::tensor::DataType::Float32 => scatter!(as_f32_slice, as_f32_slice_mut),
            crate::tensor::DataType::Float64 => scatter!(as_f64_slice, as_f64_slice_mut),
            other => {
                return Err(MinitensorError::internal_error(format!(
                    "pad backward: {other:?} tensors carry no gradient"
                )));
            }
        }

        let grad_input = Tensor::new(
            std::sync::Arc::new(data),
            shape,
            grad.dtype(),
            grad.device(),
            false,
        );
        accumulate_grad(&mut gradients, self.input_id, grad_input)?;
        Ok(gradients)
    }

    fn input_ids(&self) -> &[TensorId] {
        &self.ids
    }
}

pub struct ReshapeBackward {
    pub input_shape: Vec<usize>,
    pub input_id: TensorId,
}
impl GradientFunction for ReshapeBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<FxHashMap<TensorId, Tensor>> {
        let mut gradients = FxHashMap::default();
        gradients.reserve(1);

        // Reshape gradient: reshape back to original shape
        let original_shape = Shape::new(self.input_shape.clone());
        let grad_input = crate::ops::shape_ops::reshape(grad_output, original_shape)?;
        accumulate_grad(&mut gradients, self.input_id, grad_input)?;

        Ok(gradients)
    }

    fn input_ids(&self) -> &[TensorId] {
        std::slice::from_ref(&self.input_id)
    }
}
/// Gradient function for a dtype conversion between two float types.
///
/// A cast is the identity on values, so its gradient is the identity too --
/// carried back to whatever precision the input was held in. Only float-to-float
/// conversions get one: an integer or bool result cannot carry a gradient at
/// all, and [`Tensor::astype`] marks those as not requiring one rather than
/// producing a tensor that claims to and then delivers nothing.
pub struct AstypeBackward {
    pub input_id: TensorId,
    pub input_dtype: DataType,
}
impl GradientFunction for AstypeBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<FxHashMap<TensorId, Tensor>> {
        let mut gradients = FxHashMap::default();
        gradients.reserve(1);
        accumulate_grad(
            &mut gradients,
            self.input_id,
            grad_output.astype(self.input_dtype)?,
        )?;
        Ok(gradients)
    }

    fn input_ids(&self) -> &[TensorId] {
        std::slice::from_ref(&self.input_id)
    }
}

/// Gradient function for repeat_interleave operation
pub struct RepeatInterleaveBackward {
    pub input_shape: Vec<usize>,
    pub repeats: Vec<usize>,
    pub input_id: TensorId,
    pub dim: usize,
}
impl GradientFunction for RepeatInterleaveBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<FxHashMap<TensorId, Tensor>> {
        let grad_input = repeat_interleave_backward_impl(
            grad_output,
            &self.input_shape,
            &self.repeats,
            self.dim,
        )?;

        let mut gradients = FxHashMap::default();
        accumulate_grad(&mut gradients, self.input_id, grad_input)?;
        Ok(gradients)
    }

    fn input_ids(&self) -> &[TensorId] {
        std::slice::from_ref(&self.input_id)
    }
}
