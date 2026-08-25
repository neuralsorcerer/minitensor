// Copyright (c) Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

use crate::autograd::with_grad_fn;
use crate::ops::map::par_out_chunks;
use crate::{
    autograd::{ScatterAddBackward, ScatterBackward, ScatterReduceBackward},
    error::{MinitensorError, Result},
    ops::util::normalize_dim,
    tensor::{DataType, Shape, Tensor, TensorData},
};
use std::sync::Arc;

/// How a value already at a destination and one arriving there are combined.
///
/// `scatter` and `scatter_add` are two of these, and were a `bool` until the
/// others arrived. The kernel already took the combination as a function, so
/// this names what that function is rather than adding machinery.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Reduction {
    /// The arriving value wins. Duplicate indices resolve in a fixed order --
    /// see [`scatter`] -- and only the survivor carries gradient.
    Replace,
    /// Every arriving value is added. The adjoint of `gather`.
    Sum,
    /// Every arriving value is multiplied in.
    Prod,
    /// The largest of what is there and what arrives.
    Amax,
    /// The smallest of the same.
    Amin,
    /// The mean of everything that arrived, and of what was there when
    /// `include_self`.
    Mean,
}

impl Reduction {
    pub fn from_name(name: &str) -> Result<Self> {
        match name {
            "sum" => Ok(Reduction::Sum),
            "prod" => Ok(Reduction::Prod),
            "amax" => Ok(Reduction::Amax),
            "amin" => Ok(Reduction::Amin),
            "mean" => Ok(Reduction::Mean),
            other => Err(MinitensorError::invalid_argument(format!(
                "unknown scatter_reduce reduction {other:?}; expected \"sum\", \"prod\", \"amax\", \"amin\" or \"mean\""
            ))),
        }
    }

    /// What a destination starts from when it is not to count its own value.
    ///
    /// `Replace` has no identity and needs none -- it overwrites. `Mean`
    /// accumulates as a sum and divides afterwards, so it starts where `Sum`
    /// does.
    fn identity<T: Seedable>(self) -> Option<T> {
        match self {
            Reduction::Replace => None,
            Reduction::Sum | Reduction::Mean => Some(T::ZERO),
            Reduction::Prod => Some(T::ONE),
            Reduction::Amax => Some(T::LOWEST),
            Reduction::Amin => Some(T::HIGHEST),
        }
    }
}

/// The four constants a reduction can start from, for every dtype that can be
/// reduced.
///
/// Floats go to infinity at the ends and integers to their extremes, which is
/// the same statement -- "nothing here yet, and anything beats it" -- in the
/// arithmetic each type actually has. `bool` implements none of it: the only
/// reduction it accepts is replacement, which has no identity.
pub(crate) trait Seedable: Copy {
    const ZERO: Self;
    const ONE: Self;
    const LOWEST: Self;
    const HIGHEST: Self;
}

macro_rules! seedable {
    ($ty:ty, $zero:expr, $one:expr, $low:expr, $high:expr) => {
        impl Seedable for $ty {
            const ZERO: Self = $zero;
            const ONE: Self = $one;
            const LOWEST: Self = $low;
            const HIGHEST: Self = $high;
        }
    };
}

seedable!(f32, 0.0, 1.0, f32::NEG_INFINITY, f32::INFINITY);
seedable!(f64, 0.0, 1.0, f64::NEG_INFINITY, f64::INFINITY);
seedable!(i32, 0, 1, i32::MIN, i32::MAX);
seedable!(i64, 0, 1, i64::MIN, i64::MAX);

/// Geometry shared by both scatter kernels, in the same terms `gather` uses:
/// the tensor is a stack of `outer` chunks, each holding `dim_size` rows of
/// `inner` contiguous elements.
pub struct ScatterLayout {
    pub dim: usize,
    pub inner: usize,
    pub input_dim: usize,
    pub index_dim: usize,
    pub indices: Vec<i64>,
}

/// Validate a scatter triple and extract its layout.
///
/// The rules mirror `gather` so the two stay adjoint: `index` and `src` line up
/// exactly, and `index` matches `input` on every axis except the scattered one.
/// The scattered axis is deliberately unconstrained — scattering more values
/// than the axis is long is exactly what duplicate indices are for.
pub(crate) fn scatter_layout(
    tensor: &Tensor,
    dim: isize,
    index: &Tensor,
    src: &Tensor,
) -> Result<ScatterLayout> {
    let dim = normalize_dim(dim, tensor.ndim())?;

    if index.ndim() != tensor.ndim() || src.ndim() != tensor.ndim() {
        return Err(MinitensorError::invalid_operation(
            "scatter index and source must have the same number of dimensions as input",
        ));
    }
    if index.dtype() != DataType::Int64 {
        return Err(MinitensorError::invalid_operation(
            "scatter indices must be int64",
        ));
    }
    if src.dtype() != tensor.dtype() {
        return Err(MinitensorError::type_mismatch(
            format!("{:?}", src.dtype()),
            format!("{:?}", tensor.dtype()),
        ));
    }
    if src.device() != tensor.device() || index.device() != tensor.device() {
        return Err(MinitensorError::device_mismatch(
            format!("{:?}", src.device()),
            format!("{:?}", tensor.device()),
        ));
    }
    if !tensor.device().is_cpu() {
        return Err(MinitensorError::invalid_operation(
            "scatter currently supports only CPU tensors",
        ));
    }
    if index.shape() != src.shape() {
        return Err(MinitensorError::shape_mismatch(
            index.shape().dims().to_vec(),
            src.shape().dims().to_vec(),
        ));
    }

    let input_dims = tensor.shape().dims();
    let index_dims = index.shape().dims();
    for (i, (&idx_d, &in_d)) in index_dims.iter().zip(input_dims.iter()).enumerate() {
        if i != dim && idx_d != in_d {
            return Err(MinitensorError::shape_mismatch(
                input_dims.to_vec(),
                index_dims.to_vec(),
            ));
        }
    }

    let dim_size = input_dims[dim];
    let indices = index
        .data()
        .as_i64_slice()
        .ok_or_else(|| MinitensorError::invalid_operation("scatter indices must be int64"))?;
    for &v in indices {
        if v < 0 || v as usize >= dim_size {
            return Err(MinitensorError::index_error(v as isize, 0, dim_size));
        }
    }

    Ok(ScatterLayout {
        dim,
        inner: input_dims[dim + 1..].iter().product(),
        input_dim: dim_size,
        index_dim: index_dims[dim],
        indices: indices.to_vec(),
    })
}

/// For each destination slot, the position along the index axis that writes it
/// last — the writer whose value survives, and therefore the only one that
/// carries gradient. `usize::MAX` marks a slot nothing writes.
///
/// Only `scatter` needs this. `scatter_add` accumulates, so every writer counts.
pub(crate) fn surviving_writers(layout: &ScatterLayout, outer: usize) -> Vec<usize> {
    let ScatterLayout {
        inner,
        input_dim,
        index_dim,
        indices,
        ..
    } = layout;
    let mut winners = vec![usize::MAX; outer * input_dim * inner];
    for o in 0..outer {
        let idx_chunk = &indices[o * index_dim * inner..(o + 1) * index_dim * inner];
        let win_chunk = &mut winners[o * input_dim * inner..(o + 1) * input_dim * inner];
        for i in 0..*index_dim {
            for j in 0..*inner {
                let target = idx_chunk[i * inner + j] as usize;
                win_chunk[target * inner + j] = i;
            }
        }
    }
    winners
}

/// Write `src` into a copy of `tensor` at the positions named by `index`,
/// combining with whatever is already there according to `reduce`.
///
/// `include_self` decides whether the destination's existing value takes part.
/// It changes nothing for `Replace`, which overwrites regardless, and nothing
/// for `Sum`, where starting from zero and adding to zero agree -- so it is only
/// ever consulted for the other three.
fn scatter_impl(
    tensor: &Tensor,
    dim: isize,
    index: &Tensor,
    src: &Tensor,
    reduce: Reduction,
    include_self: bool,
) -> Result<Tensor> {
    let layout = scatter_layout(tensor, dim, index, src)?;
    let dtype = tensor.dtype();
    let device = tensor.device();
    let requires_grad = tensor.requires_grad() || src.requires_grad();

    let inner = layout.inner;
    let in_chunk = layout.input_dim * inner;
    let idx_chunk = layout.index_dim * inner;

    // `combine` is a macro parameter rather than a runtime branch because bool
    // has no addition: the accumulating form must not even be generated for it.
    macro_rules! scatter_kernel {
        ($ty:ty, $slice:ident, $mut_slice:ident, $combine:expr, $seed:expr) => {{
            let base = tensor.data().$slice().ok_or_else(|| {
                MinitensorError::invalid_operation("Tensor data access failed for scatter")
            })?;
            let updates = src.data().$slice().ok_or_else(|| {
                MinitensorError::invalid_operation("Source data access failed for scatter")
            })?;
            let mut out = TensorData::from_vec::<$ty>(base.to_vec(), dtype, device);
            let combine: fn($ty, $ty) -> $ty = $combine;
            let seed: Option<$ty> = $seed;
            {
                let dst = out.$mut_slice().ok_or_else(|| {
                    MinitensorError::internal_error("Failed to write scatter output")
                })?;
                if in_chunk != 0 && idx_chunk != 0 {
                    // Each task owns one destination chunk, so writes never
                    // race and duplicate indices inside a chunk resolve in a
                    // fixed order. That makes both kernels deterministic --
                    // including `scatter_add`, whose float accumulation would
                    // otherwise depend on how the work was scheduled.
                    par_out_chunks(dst, in_chunk, &|start, dst_chunk| {
                        let o = start / in_chunk;
                        let idx = &layout.indices[o * idx_chunk..(o + 1) * idx_chunk];
                        let upd = &updates[o * idx_chunk..(o + 1) * idx_chunk];
                        // Only the destinations something actually writes are
                        // reset: one nothing addresses keeps its own value, for
                        // every reduction and either way round on
                        // `include_self`.
                        if let Some(start_from) = seed {
                            for i in 0..layout.index_dim {
                                for j in 0..inner {
                                    dst_chunk[idx[i * inner + j] as usize * inner + j] = start_from;
                                }
                            }
                        }
                        for i in 0..layout.index_dim {
                            for j in 0..inner {
                                let pos = i * inner + j;
                                let target = idx[pos] as usize * inner + j;
                                dst_chunk[target] = combine(dst_chunk[target], upd[pos]);
                            }
                        }
                    });
                }
            }
            out
        }};
    }

    // Where each written destination starts when it is not to count itself.
    fn seed_of<T: Seedable>(reduce: Reduction, include_self: bool) -> Option<T> {
        if include_self {
            None
        } else {
            reduce.identity::<T>()
        }
    }
    // `combine` stays a macro parameter rather than a runtime branch: `bool` has
    // no addition, so the accumulating forms must not even be generated for it.
    macro_rules! dispatch {
        ($ty:ty, $slice:ident, $mut_slice:ident) => {
            match reduce {
                Reduction::Replace => {
                    scatter_kernel!($ty, $slice, $mut_slice, |_old, new| new, None)
                }
                Reduction::Sum | Reduction::Mean => {
                    scatter_kernel!(
                        $ty,
                        $slice,
                        $mut_slice,
                        |old, new| old + new,
                        seed_of::<$ty>(reduce, include_self)
                    )
                }
                Reduction::Prod => scatter_kernel!(
                    $ty,
                    $slice,
                    $mut_slice,
                    |old, new| old * new,
                    seed_of::<$ty>(reduce, include_self)
                ),
                Reduction::Amax => {
                    scatter_kernel!(
                        $ty,
                        $slice,
                        $mut_slice,
                        |old: $ty, new: $ty| if new > old { new } else { old },
                        seed_of::<$ty>(reduce, include_self)
                    )
                }
                Reduction::Amin => {
                    scatter_kernel!(
                        $ty,
                        $slice,
                        $mut_slice,
                        |old: $ty, new: $ty| if new < old { new } else { old },
                        seed_of::<$ty>(reduce, include_self)
                    )
                }
            }
        };
    }

    let data = match dtype {
        DataType::Float32 => dispatch!(f32, as_f32_slice, as_f32_slice_mut),
        DataType::Float64 => dispatch!(f64, as_f64_slice, as_f64_slice_mut),
        DataType::Int32 => dispatch!(i32, as_i32_slice, as_i32_slice_mut),
        DataType::Int64 => dispatch!(i64, as_i64_slice, as_i64_slice_mut),
        DataType::Bool => {
            if reduce != Reduction::Replace {
                return Err(MinitensorError::invalid_operation(
                    "scatter reductions other than replacement are not supported for boolean tensors",
                ));
            }
            scatter_kernel!(
                bool,
                as_bool_slice,
                as_bool_slice_mut,
                |_old, new| new,
                None
            )
        }
    };

    let outer: usize = tensor.shape().dims()[..layout.dim].iter().product();

    // `Mean` accumulates as a sum and divides here, because the divisor is not
    // known until every contribution has arrived. A destination nothing wrote
    // keeps its own value and is not divided -- its count is zero, and averaging
    // an untouched entry would change a value nobody scattered to.
    let mut data = data;
    if reduce == Reduction::Mean {
        let counts = contribution_counts(&layout, outer, include_self);
        macro_rules! average {
            ($accessor_mut:ident, $ty:ty) => {{
                let out = data.$accessor_mut().ok_or_else(|| {
                    MinitensorError::internal_error("scatter_reduce: dtype does not match")
                })?;
                for (slot, &count) in out.iter_mut().zip(&counts) {
                    if count > 0 {
                        *slot /= count as $ty;
                    }
                }
            }};
        }
        match dtype {
            DataType::Float32 => average!(as_f32_slice_mut, f32),
            DataType::Float64 => average!(as_f64_slice_mut, f64),
            _ => {
                return Err(MinitensorError::invalid_operation(
                    "scatter_reduce with \"mean\" requires a floating point tensor",
                ));
            }
        }
    }

    let output = Tensor::new(
        Arc::new(data),
        tensor.shape().clone(),
        dtype,
        device,
        requires_grad,
    );

    if requires_grad && dtype.is_float() {
        let mut output = output;
        if !matches!(reduce, Reduction::Replace | Reduction::Sum) {
            let grad_fn = Arc::new(ScatterReduceBackward {
                input_ids: [tensor.id(), src.id()],
                input_requires_grad: [tensor.requires_grad(), src.requires_grad()],
                input: tensor.detach(),
                src: src.detach(),
                output: output.detach(),
                layout,
                outer,
                reduce,
                include_self,
            });
            return with_grad_fn(output, grad_fn);
        }
        if reduce == Reduction::Sum {
            let grad_fn = Arc::new(ScatterAddBackward {
                input_ids: [tensor.id(), src.id()],
                input_requires_grad: [tensor.requires_grad(), src.requires_grad()],
                src_shape: src.shape().clone(),
                dim: layout.dim,
                indices: layout.indices,
            });
            output = with_grad_fn(output, grad_fn)?;
        } else {
            let winners = surviving_writers(&layout, outer);
            let grad_fn = Arc::new(ScatterBackward {
                input_ids: [tensor.id(), src.id()],
                input_requires_grad: [tensor.requires_grad(), src.requires_grad()],
                src_shape: src.shape().clone(),
                dim: layout.dim,
                inner,
                input_dim: layout.input_dim,
                index_dim: layout.index_dim,
                indices: layout.indices,
                winners,
            });
            output = with_grad_fn(output, grad_fn)?;
        }
        return Ok(output);
    }

    Ok(output)
}

/// Write `src` into a copy of `tensor` at the positions named by `index`.
///
/// `index` and `src` must have the same shape, and must match `tensor` on every
/// axis except `dim` — the same rule `gather` uses, which is what makes the two
/// operations adjoint.
///
/// When two entries of `index` name the same destination the later one wins.
/// Rather than leave that case non-deterministic, the destination is
/// partitioned across tasks so the order is fixed, and the gradient follows
/// it — only the writer whose value survived receives any.
pub fn scatter(tensor: &Tensor, dim: isize, index: &Tensor, src: &Tensor) -> Result<Tensor> {
    scatter_impl(tensor, dim, index, src, Reduction::Replace, true)
}

/// Add `src` into a copy of `tensor` at the positions named by `index`.
///
/// This is the adjoint of `gather`, and unlike [`scatter`] duplicate indices are
/// meaningful rather than merely tolerated: every value addressed at a
/// destination is accumulated there, which is what makes it the natural way to
/// express segment sums and embedding-style gradient accumulation.
///
/// Float addition is not associative, so an accumulation order that varied with
/// thread scheduling would make results irreproducible run to run. Tasks here
/// own disjoint destination chunks and walk each chunk in index order, so the
/// result is bit-for-bit deterministic.
pub fn scatter_add(tensor: &Tensor, dim: isize, index: &Tensor, src: &Tensor) -> Result<Tensor> {
    scatter_impl(tensor, dim, index, src, Reduction::Sum, true)
}

/// How many values arrived at each destination, plus one for the destination
/// itself when it counted.
///
/// `Mean` needs this to divide by, and its gradient needs the same numbers, so
/// they are computed once and handed to both.
pub(crate) fn contribution_counts(
    layout: &ScatterLayout,
    outer: usize,
    include_self: bool,
) -> Vec<i64> {
    let ScatterLayout {
        inner,
        input_dim,
        index_dim,
        indices,
        ..
    } = layout;
    let mut counts = vec![0i64; outer * input_dim * inner];
    for o in 0..outer {
        let idx = &indices[o * index_dim * inner..(o + 1) * index_dim * inner];
        let block = &mut counts[o * input_dim * inner..(o + 1) * input_dim * inner];
        for i in 0..*index_dim {
            for j in 0..*inner {
                block[idx[i * inner + j] as usize * inner + j] += 1;
            }
        }
        if include_self {
            // A destination nothing wrote keeps its own value and is not
            // averaged, so it stays at zero rather than becoming one -- the
            // division below skips it entirely.
            for slot in block.iter_mut() {
                if *slot > 0 {
                    *slot += 1;
                }
            }
        }
    }
    counts
}

/// Read `input` at the positions named by `index`, combining what arrives at
/// each destination according to `reduce`.
///
/// `scatter` and `scatter_add` are `"replace"` and `"sum"` -- they keep their
/// own names because those two are what most callers want, and because their
/// gradients are the two the library already had.
///
/// `include_self` decides whether a destination's existing value takes part in
/// the reduction. It changes nothing for replacement, which overwrites anyway,
/// nor for summation, where starting from zero and adding to a zero agree.
pub fn scatter_reduce(
    tensor: &Tensor,
    dim: isize,
    index: &Tensor,
    src: &Tensor,
    reduce: Reduction,
    include_self: bool,
) -> Result<Tensor> {
    if reduce == Reduction::Replace {
        return scatter_impl(tensor, dim, index, src, reduce, include_self);
    }
    if reduce == Reduction::Mean && !tensor.dtype().is_float() {
        return Err(MinitensorError::invalid_operation(
            "scatter_reduce with \"mean\" over an integer tensor would truncate every average; cast to a float first",
        ));
    }
    scatter_impl(tensor, dim, index, src, reduce, include_self)
}

/// Gather `grad` at the scatter positions — the source-side gradient shared by
/// both scatter flavours, with `keep` optionally masking out writers whose value
/// did not survive.
pub(crate) fn gather_grad_for_src(
    grad: &Tensor,
    src_shape: &Shape,
    inner: usize,
    input_dim: usize,
    index_dim: usize,
    indices: &[i64],
    keep: Option<&[usize]>,
) -> Result<Tensor> {
    let numel = src_shape.numel();
    let mut out = TensorData::zeros_on_device(numel, grad.dtype(), grad.device());
    let in_chunk = input_dim * inner;
    let idx_chunk = index_dim * inner;

    if numel != 0 && in_chunk != 0 {
        macro_rules! fill {
            ($slice:ident, $mut_slice:ident) => {{
                let g = grad.data().$slice().ok_or_else(|| {
                    MinitensorError::internal_error("Failed to read grad for scatter backward")
                })?;
                let dst = out.$mut_slice().ok_or_else(|| {
                    MinitensorError::internal_error("Failed to write scatter source gradient")
                })?;
                par_out_chunks(dst, idx_chunk, &|start, dst_chunk| {
                    let o = start / idx_chunk;
                    let g_chunk = &g[o * in_chunk..(o + 1) * in_chunk];
                    let idx = &indices[o * idx_chunk..(o + 1) * idx_chunk];
                    let win = keep.map(|w| &w[o * in_chunk..(o + 1) * in_chunk]);
                    for i in 0..index_dim {
                        for j in 0..inner {
                            let pos = i * inner + j;
                            let target = idx[pos] as usize * inner + j;
                            // An overwritten writer contributed nothing to
                            // the output, so it earns no gradient.
                            if win.is_none_or(|w| w[target] == i) {
                                dst_chunk[pos] = g_chunk[target];
                            }
                        }
                    }
                });
            }};
        }

        match grad.dtype() {
            DataType::Float32 => fill!(as_f32_slice, as_f32_slice_mut),
            DataType::Float64 => fill!(as_f64_slice, as_f64_slice_mut),
            _ => {
                return Err(MinitensorError::invalid_operation(
                    "scatter backward only supported for floating point tensors",
                ));
            }
        }
    }

    Ok(Tensor::new(
        Arc::new(out),
        src_shape.clone(),
        grad.dtype(),
        grad.device(),
        false,
    ))
}

/// Zero every position that `scatter` overwrote: the output no longer depends on
/// the original value there, so no gradient flows back through it.
pub(crate) fn mask_overwritten(grad: &Tensor, winners: &[usize]) -> Result<Tensor> {
    macro_rules! mask {
        ($ty:ty, $slice:ident, $from_vec:ident) => {{
            let g = grad.data().$slice().ok_or_else(|| {
                MinitensorError::internal_error("Failed to read grad for scatter backward")
            })?;
            let values: Vec<$ty> = g
                .iter()
                .zip(winners.iter())
                .map(|(&v, &w)| if w == usize::MAX { v } else { 0.0 })
                .collect();
            TensorData::$from_vec(values, grad.device())
        }};
    }

    let data = match grad.dtype() {
        DataType::Float32 => mask!(f32, as_f32_slice, from_vec_f32),
        DataType::Float64 => mask!(f64, as_f64_slice, from_vec_f64),
        _ => {
            return Err(MinitensorError::invalid_operation(
                "scatter backward only supported for floating point tensors",
            ));
        }
    };

    Ok(Tensor::new(
        Arc::new(data),
        grad.shape().clone(),
        grad.dtype(),
        grad.device(),
        false,
    ))
}

/// The gradient of the four reductions `scatter` and `scatter_add` do not
/// cover.
///
/// Two passes, because the index maps source to destination and three of the
/// four need to know something about *every* contributor to a destination
/// before any one of them can be answered.
///
/// Returns the gradient for the destination tensor and for the source, either
/// of which the caller may not have asked for.
#[allow(clippy::too_many_arguments)]
pub(crate) fn scatter_reduce_backward(
    input: &Tensor,
    src: &Tensor,
    output: &Tensor,
    grad_output: &Tensor,
    layout: &ScatterLayout,
    outer: usize,
    reduce: Reduction,
    include_self: bool,
    wanted: [bool; 2],
) -> Result<(Option<Tensor>, Option<Tensor>)> {
    let inner = layout.inner;
    let in_chunk = layout.input_dim * inner;
    let idx_chunk = layout.index_dim * inner;

    let base_t = input.contiguous()?;
    let src_t = src.contiguous()?;
    let out_t = output.contiguous()?;
    let grad_t = grad_output.contiguous()?;

    let dtype = grad_output.dtype();
    let device = grad_output.device();
    let mut into_input = TensorData::zeros_on_device(input.numel(), dtype, device);
    let mut into_src = TensorData::zeros_on_device(src.numel(), dtype, device);

    macro_rules! route {
        ($accessor:ident, $accessor_mut:ident, $ty:ty) => {{
            let base = base_t.data().$accessor().ok_or_else(|| {
                MinitensorError::internal_error("scatter_reduce backward: dtype mismatch")
            })?;
            let values = src_t.data().$accessor().ok_or_else(|| {
                MinitensorError::internal_error("scatter_reduce backward: dtype mismatch")
            })?;
            let result = out_t.data().$accessor().ok_or_else(|| {
                MinitensorError::internal_error("scatter_reduce backward: dtype mismatch")
            })?;
            let seeds = grad_t.data().$accessor().ok_or_else(|| {
                MinitensorError::internal_error("scatter_reduce backward: dtype mismatch")
            })?;

            let zero = <$ty>::default();
            let mut for_input = vec![zero; input.numel()];
            let mut for_src = vec![zero; src.numel()];

            for o in 0..outer {
                let idx = &layout.indices[o * idx_chunk..(o + 1) * idx_chunk];
                let dst_base = o * in_chunk;
                let src_base = o * idx_chunk;

                // How many contributions reached each destination, and -- for
                // `Prod` -- how many were zero and what the rest multiply to.
                let mut arrivals = vec![0usize; in_chunk];
                let mut zeros = vec![0usize; in_chunk];
                let mut nonzero = vec![1 as $ty; in_chunk];
                for pos in 0..idx_chunk {
                    let d = idx[pos] as usize * inner + pos % inner;
                    arrivals[d] += 1;
                    let v = values[src_base + pos];
                    if v == zero {
                        zeros[d] += 1;
                    } else {
                        nonzero[d] *= v;
                    }
                }
                if include_self {
                    for d in 0..in_chunk {
                        if arrivals[d] > 0 {
                            let v = base[dst_base + d];
                            if v == zero {
                                zeros[d] += 1;
                            } else {
                                nonzero[d] *= v;
                            }
                        }
                    }
                }

                // The destination's own gradient. A destination nothing wrote
                // to is the input untouched, so its gradient passes straight
                // through -- whatever the reduction, and either way round on
                // `include_self`.
                let mut claimed = vec![false; in_chunk];
                for d in 0..in_chunk {
                    let seed = seeds[dst_base + d];
                    if arrivals[d] == 0 {
                        for_input[dst_base + d] = seed;
                        continue;
                    }
                    if !include_self {
                        continue;
                    }
                    let own = base[dst_base + d];
                    match reduce {
                        Reduction::Mean => {
                            let n = (arrivals[d] + 1) as $ty;
                            for_input[dst_base + d] = seed / n;
                        }
                        Reduction::Prod => {
                            for_input[dst_base + d] =
                                seed * excluding(zeros[d], nonzero[d], own, zero);
                        }
                        Reduction::Amax | Reduction::Amin => {
                            // It was there before anything arrived, so it is the
                            // earliest claimant of a tie.
                            if own == result[dst_base + d] {
                                for_input[dst_base + d] = seed;
                                claimed[d] = true;
                            }
                        }
                        _ => {}
                    }
                }

                for pos in 0..idx_chunk {
                    let d = idx[pos] as usize * inner + pos % inner;
                    let seed = seeds[dst_base + d];
                    let v = values[src_base + pos];
                    for_src[src_base + pos] = match reduce {
                        Reduction::Mean => {
                            let n = (arrivals[d] + usize::from(include_self)) as $ty;
                            seed / n
                        }
                        Reduction::Prod => seed * excluding(zeros[d], nonzero[d], v, zero),
                        Reduction::Amax | Reduction::Amin => {
                            if !claimed[d] && v == result[dst_base + d] {
                                claimed[d] = true;
                                seed
                            } else {
                                zero
                            }
                        }
                        _ => zero,
                    };
                }
            }

            if wanted[0] {
                let out = into_input.$accessor_mut().ok_or_else(|| {
                    MinitensorError::internal_error("scatter_reduce backward: dtype mismatch")
                })?;
                out.copy_from_slice(&for_input);
            }
            if wanted[1] {
                let out = into_src.$accessor_mut().ok_or_else(|| {
                    MinitensorError::internal_error("scatter_reduce backward: dtype mismatch")
                })?;
                out.copy_from_slice(&for_src);
            }
        }};
    }

    match dtype {
        DataType::Float32 => route!(as_f32_slice, as_f32_slice_mut, f32),
        _ => route!(as_f64_slice, as_f64_slice_mut, f64),
    }

    Ok((
        wanted[0].then(|| {
            Tensor::new(
                Arc::new(into_input),
                input.shape().clone(),
                dtype,
                device,
                false,
            )
        }),
        wanted[1].then(|| {
            Tensor::new(
                Arc::new(into_src),
                src.shape().clone(),
                dtype,
                device,
                false,
            )
        }),
    ))
}

/// The product of every contribution to a destination except `mine`.
///
/// Counted rather than divided. `total / mine` is the obvious form and it is
/// wrong exactly when `mine` is zero -- which is the case that makes the
/// question interesting, since that is when the product collapses and the other
/// factors still have gradients.
#[inline]
fn excluding<T: Copy + PartialEq + std::ops::Div<Output = T>>(
    zeros: usize,
    nonzero: T,
    mine: T,
    zero: T,
) -> T {
    match zeros {
        // Nothing was zero, so dividing the product of everything by my own
        // factor is safe and exact.
        0 => nonzero / mine,
        // Exactly one zero: every other factor's product includes it and is
        // therefore zero, and the zero's own excluded product is the rest.
        1 if mine == zero => nonzero,
        1 => zero,
        // Two or more zeros: excluding any single one still leaves a zero.
        _ => zero,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::Device;
    use crate::ops::shape_ops::gather;

    fn f32_tensor(data: Vec<f32>, shape: Vec<usize>, requires_grad: bool) -> Tensor {
        Tensor::new(
            Arc::new(TensorData::from_vec_f32(data, Device::cpu())),
            Shape::new(shape),
            DataType::Float32,
            Device::cpu(),
            requires_grad,
        )
    }

    fn i64_tensor(data: Vec<i64>, shape: Vec<usize>) -> Tensor {
        Tensor::new(
            Arc::new(TensorData::from_vec_i64(data, Device::cpu())),
            Shape::new(shape),
            DataType::Int64,
            Device::cpu(),
            false,
        )
    }

    #[test]
    fn test_scatter_writes_and_scatter_add_accumulates() {
        let base = f32_tensor(vec![0.0; 6], vec![2, 3], false);
        let index = i64_tensor(vec![0, 0, 2, 1, 1, 1], vec![2, 3]);
        let src = f32_tensor(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3], false);

        // Row 0 writes columns [0, 0, 2] with [1, 2, 3]; row 1 writes column 1
        // three times with [4, 5, 6].
        let written = scatter(&base, 1, &index, &src).unwrap();
        assert_eq!(
            written.data().as_f32_slice().unwrap(),
            &[2.0, 0.0, 3.0, 0.0, 6.0, 0.0]
        );

        let added = scatter_add(&base, 1, &index, &src).unwrap();
        assert_eq!(
            added.data().as_f32_slice().unwrap(),
            &[3.0, 0.0, 3.0, 0.0, 15.0, 0.0]
        );
    }

    #[test]
    fn test_scatter_add_is_adjoint_of_gather() {
        // <gather(x), v> == <x, scatter_add(0, v)>, the identity that makes
        // scatter_add the operation gather's own backward pass performs.
        let x = f32_tensor(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3], false);
        let index = i64_tensor(vec![0, 2, 1, 1], vec![2, 2]);
        let v = f32_tensor(vec![0.5, -1.5, 2.0, 3.0], vec![2, 2], false);

        let gathered = gather(&x, 1, &index).unwrap();
        let lhs: f32 = gathered
            .data()
            .as_f32_slice()
            .unwrap()
            .iter()
            .zip(v.data().as_f32_slice().unwrap())
            .map(|(a, b)| a * b)
            .sum();

        let zeros = f32_tensor(vec![0.0; 6], vec![2, 3], false);
        let scattered = scatter_add(&zeros, 1, &index, &v).unwrap();
        let rhs: f32 = x
            .data()
            .as_f32_slice()
            .unwrap()
            .iter()
            .zip(scattered.data().as_f32_slice().unwrap())
            .map(|(a, b)| a * b)
            .sum();

        assert!((lhs - rhs).abs() < 1e-6, "{lhs} != {rhs}");
    }

    #[test]
    fn test_scatter_add_gradient_reaches_every_duplicate_writer() {
        let base = f32_tensor(vec![0.0; 6], vec![2, 3], true);
        let index = i64_tensor(vec![0, 0, 2, 1, 1, 1], vec![2, 3]);
        let src = f32_tensor(vec![1.0; 6], vec![2, 3], true);

        let out = scatter_add(&base, 1, &index, &src).unwrap();
        let seed = f32_tensor(vec![1.0; 6], vec![2, 3], false);
        let grads = crate::autograd::backward_collect(&out, Some(seed)).unwrap();

        // Accumulating keeps the original value, so the input passes gradient
        // through untouched.
        assert_eq!(grads[&base.id()].data().as_f32_slice().unwrap(), &[1.0; 6]);
        // Addition is linear: every writer aimed at a slot sees its gradient.
        assert_eq!(grads[&src.id()].data().as_f32_slice().unwrap(), &[1.0; 6]);
    }

    #[test]
    fn test_scatter_gradient_goes_only_to_the_surviving_writer() {
        let base = f32_tensor(vec![0.0; 6], vec![2, 3], true);
        let index = i64_tensor(vec![0, 0, 2, 1, 1, 1], vec![2, 3]);
        let src = f32_tensor(vec![1.0; 6], vec![2, 3], true);

        let out = scatter(&base, 1, &index, &src).unwrap();
        let seed = f32_tensor(vec![1.0; 6], vec![2, 3], false);
        let grads = crate::autograd::backward_collect(&out, Some(seed)).unwrap();

        // Row 0 writes [0, 0, 2]: the first writer on column 0 is overwritten
        // and contributed nothing. Row 1 writes column 1 three times; only the
        // last survives.
        assert_eq!(
            grads[&src.id()].data().as_f32_slice().unwrap(),
            &[0.0, 1.0, 1.0, 0.0, 0.0, 1.0]
        );
        // A slot that was written no longer depends on the input's old value.
        assert_eq!(
            grads[&base.id()].data().as_f32_slice().unwrap(),
            &[0.0, 1.0, 0.0, 1.0, 0.0, 1.0]
        );
    }

    #[test]
    fn test_scatter_add_is_deterministic_under_collision() {
        // Every update lands on one of four slots. Float addition is not
        // associative, so a scheduling-dependent order would change the low
        // bits between runs.
        let n = 50_000;
        let base = f32_tensor(vec![0.0; 4], vec![1, 4], false);
        let index = i64_tensor((0..n).map(|i| (i % 4) as i64).collect(), vec![1, n]);
        let src = f32_tensor(
            (0..n).map(|i| (i as f32).sin() * 1e-3).collect(),
            vec![1, n],
            false,
        );

        let first = scatter_add(&base, 1, &index, &src)
            .unwrap()
            .data()
            .as_f32_slice()
            .unwrap()
            .to_vec();
        for _ in 0..8 {
            let again = scatter_add(&base, 1, &index, &src).unwrap();
            assert_eq!(
                again.data().as_f32_slice().unwrap(),
                first.as_slice(),
                "scatter_add is not reproducible"
            );
        }
    }

    #[test]
    fn test_scatter_rejects_malformed_arguments() {
        let base = f32_tensor(vec![0.0; 6], vec![2, 3], false);
        let src = f32_tensor(vec![1.0; 6], vec![2, 3], false);
        let good = i64_tensor(vec![0, 1, 2, 0, 1, 2], vec![2, 3]);

        // Index outside the scattered axis, and a negative index.
        let oob = i64_tensor(vec![0, 1, 9, 0, 1, 2], vec![2, 3]);
        assert!(scatter(&base, 1, &oob, &src).is_err());
        let negative = i64_tensor(vec![0, 1, -1, 0, 1, 2], vec![2, 3]);
        assert!(scatter(&base, 1, &negative, &src).is_err());

        // Index dtype must be int64.
        let float_index = f32_tensor(vec![0.0; 6], vec![2, 3], false);
        assert!(scatter(&base, 1, &float_index, &src).is_err());

        // Index and source must line up, and dim must exist.
        let short_src = f32_tensor(vec![1.0; 4], vec![2, 2], false);
        assert!(scatter(&base, 1, &good, &short_src).is_err());
        assert!(scatter(&base, 5, &good, &src).is_err());

        // scatter_add has no meaning for bool.
        let flags = Tensor::new(
            Arc::new(TensorData::from_vec_bool(vec![false; 6], Device::cpu())),
            Shape::new(vec![2, 3]),
            DataType::Bool,
            Device::cpu(),
            false,
        );
        let ones = Tensor::new(
            Arc::new(TensorData::from_vec_bool(vec![true; 6], Device::cpu())),
            Shape::new(vec![2, 3]),
            DataType::Bool,
            Device::cpu(),
            false,
        );
        assert!(scatter(&flags, 1, &good, &ones).is_ok());
        assert!(scatter_add(&flags, 1, &good, &ones).is_err());
    }
}
