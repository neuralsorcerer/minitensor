// Copyright (c) Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

use crate::ops::map::{outputs_per_task, par_fold_chunks, par_map_indexed, par_out_chunks};
use crate::ops::simd::*;
use crate::ops::util::Accumulate;
use crate::{
    error::{MinitensorError, Result},
    tensor::{DataType, Shape, Tensor, TensorData},
};
use rayon::prelude::*;
use std::sync::Arc;

/// Floor on the width of a `dim == 0` column block: at least a cache line
/// (64 f32 = 256 B), so neighbouring blocks never write into the same line.
const DIM0_MIN_BLOCK: usize = 64;

/// Row-band shape for the `dim == 0` reductions. The target caps how many
/// partial buffers are allocated, the floor keeps each band big enough to be
/// worth a task, and below `DIM0_MIN_BANDS` there is not enough row
/// parallelism to bother and the column path takes over. All three are
/// constants so the band layout — which does affect the result — depends only
/// on the row count.
const DIM0_TARGET_BANDS: usize = 64;
const DIM0_MIN_ROW_BAND: usize = 256;
const DIM0_MIN_BANDS: usize = 4;

/// Reduce a row-major `(rows, cols)` slice along dimension 0, writing one value
/// per column.
///
/// Parallelism runs across the *output* columns, never across rows, so every
/// output element is accumulated by a single thread walking the rows in index
/// order. The natural shape for this loop -- fold a per-worker accumulator over
/// `par_chunks_exact(cols)` and reduce the partials -- instead lets rayon decide
/// how rows are grouped, and that grouping changes with the thread count. For
/// floating point that changes the rounding, so the same program produced
/// different sums on machines with different core counts.
///
/// Note what the block width does *not* affect: every output element still
/// accumulates rows `0..rows` in index order whatever the partition, so the
/// result is identical for any block size. That leaves the width free to be
/// chosen purely for locality -- one wide contiguous run per thread rather than
/// many narrow interleaved ones -- including from the thread count, without
/// costing reproducibility.
fn reduce_along_dim0<T, F>(input: &[T], out: &mut [T], cols: usize, init: T, combine: F)
where
    T: Copy + Send + Sync,
    F: Fn(T, T) -> T + Send + Sync + Copy,
{
    if cols == 0 || out.is_empty() {
        return;
    }
    let rows = input.len() / cols;

    // Contiguous bands of rows, when there are enough of them to go around.
    // Each band streams the input in memory order, which the prefetcher likes
    // far more than walking a column band down the matrix, and the partial
    // buffers it needs cost only `bands * cols`. The band boundaries come from
    // the row count alone -- never from the thread count -- because here the
    // partition *does* decide how the partial sums are grouped.
    let band = rows.div_ceil(DIM0_TARGET_BANDS).max(DIM0_MIN_ROW_BAND);
    let bands = rows.div_ceil(band);
    if bands >= DIM0_MIN_BANDS {
        // Only the split is erased; `combine` stays a concrete closure type
        // inside the band body, so the accumulate loop still inlines and
        // vectorizes. Erasing it here instead would put an indirect call on
        // every element.
        let partials: Vec<Vec<T>> = par_map_indexed(bands, &|index| {
            let start = index * band;
            let end = ((index + 1) * band).min(rows);
            let mut acc = vec![init; cols];
            for row in input[start * cols..end * cols].chunks_exact(cols) {
                for (slot, &value) in acc.iter_mut().zip(row) {
                    *slot = combine(*slot, value);
                }
            }
            acc
        });

        out.copy_from_slice(&partials[0]);
        for partial in &partials[1..] {
            for (slot, &value) in out.iter_mut().zip(partial) {
                *slot = combine(*slot, value);
            }
        }
        return;
    }

    // Too few rows to split: give each thread its own band of output columns
    // instead. Unlike the row split above, the column width cannot change the
    // result -- each output still accumulates rows in index order -- so it is
    // free to follow the thread count.
    let block = cols
        .div_ceil(rayon::current_num_threads().max(1))
        .max(DIM0_MIN_BLOCK);
    par_out_chunks(out, block, &|start, out_block| {
        let width = out_block.len();
        for slot in out_block.iter_mut() {
            *slot = init;
        }
        for row in input.chunks_exact(cols) {
            let segment = &row[start..start + width];
            for (slot, &value) in out_block.iter_mut().zip(segment) {
                *slot = combine(*slot, value);
            }
        }
    });
}

/// Generates a sum-along-dim reduction kernel. The body is identical across
/// numeric dtypes; only the element type, the additive identity, and the SIMD
/// row-sum helper differ.
macro_rules! sum_along_dim_kernel {
    ($name:ident, $accessor:ident, $accessor_mut:ident, $tyname:literal, $zero:expr, $simd_sum:ident) => {
        pub(crate) fn $name(
            tensor: &Tensor,
            result_data: &mut TensorData,
            dim: usize,
        ) -> Result<()> {
            let input_data = tensor.data().$accessor().ok_or_else(|| {
                MinitensorError::internal_error(concat!("Failed to get ", $tyname, " slice"))
            })?;
            let result_slice = result_data.$accessor_mut().ok_or_else(|| {
                MinitensorError::internal_error(concat!(
                    "Failed to get mutable ",
                    $tyname,
                    " slice"
                ))
            })?;
            let input_shape = tensor.shape().dims();
            // A zero-length reduced axis contributes no terms, so every output
            // slot is the additive identity. Handled up front because the 2-D
            // `dim == 1` branch below chunks the input by `cols`, and
            // `chunks_exact(0)` panics rather than yielding no chunks. `.get`
            // rather than `[dim]` so an out-of-range `dim` still reaches the
            // index_error paths below.
            if input_shape.get(dim) == Some(&0) {
                result_slice.fill($zero);
                return Ok(());
            }
            if tensor.ndim() == 1 {
                if dim != 0 {
                    return Err(MinitensorError::dim_out_of_range(
                        dim as isize,
                        tensor.ndim(),
                    ));
                }
                result_slice[0] = $simd_sum(input_data);
            } else if tensor.ndim() == 2 {
                let cols = input_shape[1];
                match dim {
                    0 => {
                        reduce_along_dim0(input_data, result_slice, cols, $zero, |a, v| {
                            a.acc_add(v)
                        });
                    }
                    1 => {
                        par_out_chunks(result_slice, outputs_per_task(cols), &|start, chunk| {
                            for (offset, out) in chunk.iter_mut().enumerate() {
                                let base = (start + offset) * cols;
                                *out = $simd_sum(&input_data[base..base + cols]);
                            }
                        });
                    }
                    _ => {
                        return Err(MinitensorError::dim_out_of_range(
                            dim as isize,
                            tensor.ndim(),
                        ));
                    }
                }
            } else {
                let dim_size = input_shape[dim];
                let inner = input_shape[dim + 1..].iter().product::<usize>();
                let outer_stride = dim_size * inner;
                par_out_chunks(result_slice, outputs_per_task(dim_size), &|start, chunk| {
                    for (offset, out) in chunk.iter_mut().enumerate() {
                        let idx = start + offset;
                        let o = idx / inner;
                        let r = idx % inner;
                        let mut sum_val = $zero;
                        let mut base = o * outer_stride + r;
                        for _ in 0..dim_size {
                            sum_val = sum_val.acc_add(input_data[base]);
                            base += inner;
                        }
                        *out = sum_val;
                    }
                });
            }
            Ok(())
        }
    };
}

/// Generates a NaN-ignoring sum-along-dim reduction kernel. Float dtypes only
/// (integer dtypes have no NaN, so they route through the plain sum kernel).
macro_rules! nansum_along_dim_kernel {
    ($name:ident, $ty:ty, $accessor:ident, $accessor_mut:ident, $tyname:literal, $zero:expr) => {
        pub(crate) fn $name(
            tensor: &Tensor,
            result_data: &mut TensorData,
            dim: usize,
        ) -> Result<()> {
            let input_data = tensor.data().$accessor().ok_or_else(|| {
                MinitensorError::internal_error(concat!("Failed to get ", $tyname, " slice"))
            })?;
            let result_slice = result_data.$accessor_mut().ok_or_else(|| {
                MinitensorError::internal_error(concat!(
                    "Failed to get mutable ",
                    $tyname,
                    " slice"
                ))
            })?;
            let input_shape = tensor.shape().dims();
            // See the note in `sum_along_dim_kernel!`: an empty reduced axis
            // yields the identity everywhere, and short-circuiting here keeps
            // the 2-D `dim == 1` branch from calling `chunks_exact(0)`.
            if input_shape.get(dim) == Some(&0) {
                result_slice.fill($zero);
                return Ok(());
            }
            if tensor.ndim() == 1 {
                if dim != 0 {
                    return Err(MinitensorError::dim_out_of_range(
                        dim as isize,
                        tensor.ndim(),
                    ));
                }
                result_slice[0] = input_data.iter().filter(|v| !v.is_nan()).sum::<$ty>();
            } else if tensor.ndim() == 2 {
                let cols = input_shape[1];
                match dim {
                    0 => {
                        reduce_along_dim0(input_data, result_slice, cols, $zero, |a, v| {
                            if v.is_nan() { a } else { a + v }
                        });
                    }
                    1 => {
                        par_out_chunks(result_slice, outputs_per_task(cols), &|start, chunk| {
                            for (offset, out) in chunk.iter_mut().enumerate() {
                                let base = (start + offset) * cols;
                                *out = input_data[base..base + cols]
                                    .iter()
                                    .filter(|v| !v.is_nan())
                                    .sum::<$ty>();
                            }
                        });
                    }
                    _ => {
                        return Err(MinitensorError::dim_out_of_range(
                            dim as isize,
                            tensor.ndim(),
                        ));
                    }
                }
            } else {
                let dim_size = input_shape[dim];
                let inner = input_shape[dim + 1..].iter().product::<usize>();
                let outer_stride = dim_size * inner;
                par_out_chunks(result_slice, outputs_per_task(dim_size), &|start, chunk| {
                    for (offset, out) in chunk.iter_mut().enumerate() {
                        let idx = start + offset;
                        let o = idx / inner;
                        let r = idx % inner;
                        let mut sum_val = $zero;
                        let mut base = o * outer_stride + r;
                        for _ in 0..dim_size {
                            let value = input_data[base];
                            if !value.is_nan() {
                                sum_val += value;
                            }
                            base += inner;
                        }
                        *out = sum_val;
                    }
                });
            }
            Ok(())
        }
    };
}

sum_along_dim_kernel!(
    sum_along_dim_f32,
    as_f32_slice,
    as_f32_slice_mut,
    "f32",
    0f32,
    simd_sum_f32
);

nansum_along_dim_kernel!(
    nansum_along_dim_f32,
    f32,
    as_f32_slice,
    as_f32_slice_mut,
    "f32",
    0f32
);

sum_along_dim_kernel!(
    sum_along_dim_f64,
    as_f64_slice,
    as_f64_slice_mut,
    "f64",
    0f64,
    simd_sum_f64
);

nansum_along_dim_kernel!(
    nansum_along_dim_f64,
    f64,
    as_f64_slice,
    as_f64_slice_mut,
    "f64",
    0f64
);

sum_along_dim_kernel!(
    sum_along_dim_i32,
    as_i32_slice,
    as_i32_slice_mut,
    "i32",
    0i32,
    simd_sum_i32
);

sum_along_dim_kernel!(
    sum_along_dim_i64,
    as_i64_slice,
    as_i64_slice_mut,
    "i64",
    0i64,
    simd_sum_i64
);

#[inline]
pub fn prod_along_dim(tensor: &Tensor, dim: usize, keepdim: bool) -> Result<Tensor> {
    if dim >= tensor.ndim() {
        return Err(MinitensorError::dim_out_of_range(
            dim as isize,
            tensor.ndim(),
        ));
    }

    let input_shape = tensor.shape().dims();
    let mut output_shape = input_shape.to_vec();
    if keepdim {
        output_shape[dim] = 1;
    } else {
        output_shape.remove(dim);
    }
    let output_shape_obj = Shape::new(output_shape);
    let mut result_data =
        TensorData::zeros_on_device(output_shape_obj.numel(), tensor.dtype(), tensor.device());

    match tensor.dtype() {
        DataType::Float32 => prod_along_dim_f32(tensor, &mut result_data, dim)?,
        DataType::Float64 => prod_along_dim_f64(tensor, &mut result_data, dim)?,
        DataType::Int32 => prod_along_dim_i32(tensor, &mut result_data, dim)?,
        DataType::Int64 => prod_along_dim_i64(tensor, &mut result_data, dim)?,
        DataType::Bool => prod_along_dim_bool(tensor, &mut result_data, dim)?,
    }

    let requires_grad = tensor.requires_grad() && tensor.dtype() != DataType::Bool;
    Ok(Tensor::new(
        Arc::new(result_data),
        output_shape_obj,
        tensor.dtype(),
        tensor.device(),
        requires_grad,
    ))
}

/// Generates a product-along-dim reduction kernel. Body is identical across
/// numeric dtypes; only the element type and multiplicative identity differ.
macro_rules! prod_along_dim_kernel {
    ($name:ident, $accessor:ident, $accessor_mut:ident, $tyname:literal, $one:expr) => {
        fn $name(tensor: &Tensor, result_data: &mut TensorData, dim: usize) -> Result<()> {
            let input_data = tensor.data().$accessor().ok_or_else(|| {
                MinitensorError::internal_error(concat!("Failed to get ", $tyname, " slice"))
            })?;
            let result_slice = result_data.$accessor_mut().ok_or_else(|| {
                MinitensorError::internal_error(concat!(
                    "Failed to get mutable ",
                    $tyname,
                    " slice"
                ))
            })?;
            let input_shape = tensor.shape().dims();
            let dim_size = input_shape[dim];
            let inner = input_shape[dim + 1..].iter().product::<usize>();
            let outer_stride = dim_size * inner;
            if inner == 0 {
                return Ok(());
            }
            // Accumulate the reduced dimension by multiplying contiguous slabs
            // (`input[.. k*inner ..]`) into a per-`outer` product buffer, so
            // every read and write is sequential (cache-friendly) rather than
            // striding by `inner` per output element. Parallel over the outer
            // index.
            par_out_chunks(result_slice, inner, &|start, out_chunk| {
                out_chunk.fill($one);
                let block_base = (start / inner) * outer_stride;
                for k in 0..dim_size {
                    let slab_base = block_base + k * inner;
                    let slab = &input_data[slab_base..slab_base + inner];
                    for (acc, &v) in out_chunk.iter_mut().zip(slab) {
                        *acc = acc.acc_mul(v);
                    }
                }
            });
            Ok(())
        }
    };
}

prod_along_dim_kernel!(
    prod_along_dim_f32,
    as_f32_slice,
    as_f32_slice_mut,
    "f32",
    1f32
);

prod_along_dim_kernel!(
    prod_along_dim_f64,
    as_f64_slice,
    as_f64_slice_mut,
    "f64",
    1f64
);

prod_along_dim_kernel!(
    prod_along_dim_i32,
    as_i32_slice,
    as_i32_slice_mut,
    "i32",
    1i32
);

prod_along_dim_kernel!(
    prod_along_dim_i64,
    as_i64_slice,
    as_i64_slice_mut,
    "i64",
    1i64
);

fn prod_along_dim_bool(tensor: &Tensor, result_data: &mut TensorData, dim: usize) -> Result<()> {
    let input_data = tensor
        .data()
        .as_bool_slice()
        .ok_or_else(|| MinitensorError::internal_error("Failed to get bool slice"))?;
    let result_slice = result_data
        .as_bool_slice_mut()
        .ok_or_else(|| MinitensorError::internal_error("Failed to get mutable bool slice"))?;
    let input_shape = tensor.shape().dims();
    let dim_size = input_shape[dim];
    let inner = input_shape[dim + 1..].iter().product::<usize>();
    let outer_stride = dim_size * inner;
    par_out_chunks(result_slice, outputs_per_task(dim_size), &|start, chunk| {
        for (offset, out) in chunk.iter_mut().enumerate() {
            let idx = start + offset;
            let o = idx / inner;
            let r = idx % inner;
            let mut val = true;
            let mut base = o * outer_stride + r;
            for _ in 0..dim_size {
                val &= input_data[base];
                if !val {
                    break;
                }
                base += inner;
            }
            *out = val;
        }
    });

    Ok(())
}

// Helper implementations for max/min operations
//
// These fold over contiguous chunks rather than reducing element by element.
// A per-element `par_iter().reduce(..)` hands rayon one work item per value and
// leaves the comparison behind an opaque closure, so nothing vectorizes; over
// a few million elements that ran an order of magnitude slower than `sum` on
// identical data. Splitting into chunks lets the inner loop become plain
// min/max instructions and keeps the parallel split coarse.

/// Chunk length for the parallel min/max folds. Large enough that the per-chunk
/// overhead disappears, small enough to keep every core fed.
const MINMAX_CHUNK: usize = 8 * 1024;

/// Float min/max over a chunked parallel fold.
///
/// NaN propagates, matching the previous element-wise behaviour: it is tracked
/// as a separate flag so the value loop stays a bare comparison. `v > best`
/// (rather than `f32::max`) is deliberate — comparisons against NaN are false,
/// so NaN never displaces a real value, and the flag decides the result.
///
/// The fold runs over `$lanes` independent accumulators rather than one. A
/// single `best` makes the compare-and-select a serial dependency chain across
/// the whole slice, which cannot vectorize; splitting it the way `simd_sum_f32`
/// splits its addition measured 6.2x faster per chunk on f32 (2.67ms -> 0.43ms
/// over 2M elements, single-threaded), with identical results including the NaN
/// flag. That gap was visible from Python: `max` was the one f32 reduction
/// lagging the others, while `sum` was already four times quicker.
macro_rules! float_extremum_all {
    ($name:ident, $accessor:ident, $accessor_mut:ident, $ty:ty, $tyname:literal, $identity:expr, $better:tt, $lanes:expr) => {
        pub(crate) fn $name(tensor: &Tensor, result_data: &mut TensorData) -> Result<()> {
            let data = tensor.data().$accessor().ok_or_else(|| {
                MinitensorError::internal_error(concat!("Failed to get ", $tyname, " slice"))
            })?;

            let (value, has_nan) = par_fold_chunks(
                data,
                MINMAX_CHUNK,
                ($identity, false),
                &|chunk| {
                    const LANES: usize = $lanes;
                    let mut bests = [$identity; LANES];
                    let mut nans = [0u32; LANES];
                    let mut blocks = chunk.chunks_exact(LANES);
                    for block in &mut blocks {
                        for lane in 0..LANES {
                            let v = block[lane];
                            if v $better bests[lane] {
                                bests[lane] = v;
                            }
                            // `as u32` rather than a bool `|=`: keeps the lane
                            // update branch-free so it vectorizes with the
                            // comparison above.
                            nans[lane] |= (v != v) as u32;
                        }
                    }
                    let mut best: $ty = $identity;
                    let mut nan = 0u32;
                    for lane in 0..LANES {
                        if bests[lane] $better best {
                            best = bests[lane];
                        }
                        nan |= nans[lane];
                    }
                    for &v in blocks.remainder() {
                        if v $better best {
                            best = v;
                        }
                        nan |= (v != v) as u32;
                    }
                    (best, nan != 0)
                },
                &|a, b| (if b.0 $better a.0 { b.0 } else { a.0 }, a.1 | b.1),
            );

            let result_slice = result_data.$accessor_mut().ok_or_else(|| {
                MinitensorError::internal_error(concat!(
                    "Failed to get mutable ",
                    $tyname,
                    " slice"
                ))
            })?;

            result_slice[0] = if has_nan { <$ty>::NAN } else { value };
            Ok(())
        }
    };
}

/// Integer min/max over the same chunked fold; no NaN to consider.
/// Integer min/max, split across `$lanes` accumulators for the same reason as
/// the float version above: one `best` serializes the compare-and-select.
macro_rules! int_extremum_all {
    ($name:ident, $accessor:ident, $accessor_mut:ident, $ty:ty, $tyname:literal, $identity:expr, $better:tt, $lanes:expr) => {
        pub(crate) fn $name(tensor: &Tensor, result_data: &mut TensorData) -> Result<()> {
            let data = tensor.data().$accessor().ok_or_else(|| {
                MinitensorError::internal_error(concat!("Failed to get ", $tyname, " slice"))
            })?;

            let value = par_fold_chunks(
                data,
                MINMAX_CHUNK,
                $identity,
                &|chunk| {
                    const LANES: usize = $lanes;
                    let mut bests = [$identity; LANES];
                    let mut blocks = chunk.chunks_exact(LANES);
                    for block in &mut blocks {
                        for lane in 0..LANES {
                            if block[lane] $better bests[lane] {
                                bests[lane] = block[lane];
                            }
                        }
                    }
                    let mut best: $ty = $identity;
                    for lane in 0..LANES {
                        if bests[lane] $better best {
                            best = bests[lane];
                        }
                    }
                    for &v in blocks.remainder() {
                        if v $better best {
                            best = v;
                        }
                    }
                    best
                },
                &|a, b| if b $better a { b } else { a },
            );

            let result_slice = result_data.$accessor_mut().ok_or_else(|| {
                MinitensorError::internal_error(concat!(
                    "Failed to get mutable ",
                    $tyname,
                    " slice"
                ))
            })?;

            result_slice[0] = value;
            Ok(())
        }
    };
}

float_extremum_all!(
    max_all_f32,
    as_f32_slice,
    as_f32_slice_mut,
    f32,
    "f32",
    f32::NEG_INFINITY,
    >,
    8
);
float_extremum_all!(
    max_all_f64,
    as_f64_slice,
    as_f64_slice_mut,
    f64,
    "f64",
    f64::NEG_INFINITY,
    >,
    4
);
int_extremum_all!(
    max_all_i32,
    as_i32_slice,
    as_i32_slice_mut,
    i32,
    "i32",
    i32::MIN,
    >,
    8
);
int_extremum_all!(
    max_all_i64,
    as_i64_slice,
    as_i64_slice_mut,
    i64,
    "i64",
    i64::MIN,
    >,
    4
);

pub(crate) fn max_all_bool(tensor: &Tensor, result_data: &mut TensorData) -> Result<()> {
    let data = tensor
        .data()
        .as_bool_slice()
        .ok_or_else(|| MinitensorError::internal_error("Failed to get bool slice"))?;

    let max_val = data.par_iter().any(|&x| x);

    let result_slice = result_data
        .as_bool_slice_mut()
        .ok_or_else(|| MinitensorError::internal_error("Failed to get mutable bool slice"))?;

    result_slice[0] = max_val;
    Ok(())
}

// Similar implementations for min functions
float_extremum_all!(
    min_all_f32,
    as_f32_slice,
    as_f32_slice_mut,
    f32,
    "f32",
    f32::INFINITY,
    <,
    8
);
float_extremum_all!(
    min_all_f64,
    as_f64_slice,
    as_f64_slice_mut,
    f64,
    "f64",
    f64::INFINITY,
    <,
    4
);
int_extremum_all!(
    min_all_i32,
    as_i32_slice,
    as_i32_slice_mut,
    i32,
    "i32",
    i32::MAX,
    <,
    8
);
int_extremum_all!(
    min_all_i64,
    as_i64_slice,
    as_i64_slice_mut,
    i64,
    "i64",
    i64::MAX,
    <,
    4
);

pub(crate) fn min_all_bool(tensor: &Tensor, result_data: &mut TensorData) -> Result<()> {
    let data = tensor
        .data()
        .as_bool_slice()
        .ok_or_else(|| MinitensorError::internal_error("Failed to get bool slice"))?;

    let min_val = data.par_iter().all(|&x| x);

    let result_slice = result_data
        .as_bool_slice_mut()
        .ok_or_else(|| MinitensorError::internal_error("Failed to get mutable bool slice"))?;

    result_slice[0] = min_val;
    Ok(())
}
