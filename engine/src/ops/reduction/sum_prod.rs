// Copyright (c) Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

use crate::ops::map::{
    FOLD_PAR_BYTES, PAR_CHUNK, outputs_per_task, par_all_chunk, par_any_chunk, par_fold_chunks,
    par_map_indexed, par_out_chunks, par_out_chunks_sized, reduction_band,
};
use crate::ops::simd::*;
use crate::ops::util::check_dim;
use crate::ops::util::{
    Accumulate, RUN_SUM_CHUNK, accumulating_dtype, accurate_run_sum, pairwise_fold_vectors,
};
use crate::{
    error::{MinitensorError, Result},
    tensor::{DataType, Shape, Tensor, TensorData},
};
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
/// per column, through [`fold_slab_with`].
///
/// The natural shape for this loop -- fold a per-worker accumulator over
/// `par_chunks_exact(cols)` and reduce the partials -- lets rayon decide how
/// rows are grouped, and that grouping changes with the thread count. For
/// floating point that changes the rounding, so the same program produced
/// different sums on machines with different core counts. Here every grouping
/// of rows into partial sums follows from the shape alone.
///
/// `combine` folds one input value into an accumulator and `merge` joins two
/// accumulators. They are separate because the two are no longer the same
/// function: an accumulating reduction may gather a narrower input into a wider
/// total (see `accumulating_dtype`), so `combine` crosses types where `merge`
/// does not.
fn reduce_along_dim0<I, A, F, M>(
    input: &[I],
    out: &mut [A],
    cols: usize,
    init: A,
    combine: F,
    merge: M,
) where
    I: Copy + Send + Sync,
    A: Copy + Send + Sync,
    F: Fn(A, I) -> A + Send + Sync + Copy,
    M: Fn(A, A) -> A + Send + Sync + Copy,
{
    if cols == 0 || out.is_empty() {
        return;
    }
    out.copy_from_slice(&fold_slab_with(
        input,
        cols,
        init,
        move |acc: &mut [A], values: &[I], _| {
            for (slot, &value) in acc.iter_mut().zip(values) {
                *slot = combine(*slot, value);
            }
        },
        merge,
        true,
    ));
}

/// Reduce each column of a row-major `(rows, cols)` slab, the building block
/// every column reduction here -- `sum`, `var`, `norm` down a non-last axis --
/// goes through.
///
/// `step(acc, values, first)` folds a run of `values` into the accumulator
/// lanes `acc` beside it, lane `i` belonging to column `(first + i) % cols`.
/// It sees whole rows, several narrow rows side by side, or the part of a row
/// one column band covers, so a caller whose step needs something per column
/// (`var` needs each column's mean) indexes a copy of it tiled across
/// [`fold_lanes`] lanes by `first + i`. `merge` joins two accumulators.
///
/// With `spread`, the slab is split across the pool: into bands of rows when
/// there are enough rows, which stream the input in memory order, and into
/// bands of columns when there are not. Every band boundary follows from the
/// shape alone, and the column split does not group any column's rows
/// differently whatever its width, so the answer never depends on the thread
/// count. Without `spread` the whole slab is folded on the calling thread;
/// which one a caller asks for must follow from the shape as well.
pub(crate) fn fold_slab_with<I, A, S, M>(
    input: &[I],
    cols: usize,
    init: A,
    step: S,
    merge: M,
    spread: bool,
) -> Vec<A>
where
    I: Copy + Send + Sync,
    A: Copy + Send + Sync,
    S: Fn(&mut [A], &[I], usize) + Send + Sync + Copy,
    M: Fn(A, A) -> A + Send + Sync + Copy,
{
    let rows = input.len() / cols;
    if !spread {
        return fold_rows_with(input, cols, init, step, merge);
    }
    // Below this the pool costs more than the fold, and both splits below are
    // run the same way on the calling thread instead -- the same bands, in the
    // same order, merged the same way -- so the answer is the parallel one.
    let serial = std::mem::size_of_val(input) < FOLD_PAR_BYTES;

    // Contiguous bands of rows, when there are enough of them to go around.
    // The band boundaries come from the row count alone -- never from the
    // thread count -- because here the partition *does* decide how the
    // partial sums are grouped.
    //
    // Inside a band the rows are blocked and the blocks folded pairwise,
    // rather than run into one total. A band is up to `rows / 64` rows wide,
    // so on a few million rows it was a chain of tens of thousands of
    // roundings: summing four million squares two columns wide measured
    // 7.5e-6 relative, where the same values through a contiguous
    // `accurate_run_sum` give about 3e-7. It only shows on summands with a
    // wide relative spread -- uniform values in [0.5, 1.5] hid it at 3.5e-7 --
    // but a sum of squares, which is what `var` and `norm` feed through here,
    // has exactly that spread.
    if let Some(band) = row_band(rows) {
        let fold_band = |index: usize| {
            let start = index * band;
            let end = ((index + 1) * band).min(rows);
            fold_rows_with(&input[start * cols..end * cols], cols, init, step, merge)
        };
        let partials: Vec<Vec<A>> = if serial {
            (0..rows.div_ceil(band)).map(fold_band).collect()
        } else {
            par_map_indexed(rows.div_ceil(band), &fold_band)
        };
        // The bands merge pairwise too; a running fold over them was a second,
        // shorter chain of the same kind.
        return pairwise_fold_vectors(partials, merge);
    }

    // Too few rows to split: give each thread its own band of output columns
    // instead; see [`fold_column_band`]. Its width changes no column's answer,
    // so on one thread it is simply all of them.
    if serial {
        return fold_column_band(input, cols, 0, cols, init, step, merge);
    }
    let mut out = vec![init; cols];
    par_out_chunks(&mut out, column_band(cols), &|start, out_block| {
        let width = out_block.len();
        out_block.copy_from_slice(&fold_column_band(
            input, cols, start, width, init, step, merge,
        ));
    });
    out
}

/// The rows in one band when a `rows`-row slab is split across the pool by
/// rows, or `None` when there are too few rows to be worth it and the slab is
/// split by columns instead. It follows from the row count alone, because
/// here the partition decides how the partial sums are grouped.
pub(crate) fn row_band(rows: usize) -> Option<usize> {
    let band = rows.div_ceil(DIM0_TARGET_BANDS).max(DIM0_MIN_ROW_BAND);
    (rows.div_ceil(band) >= DIM0_MIN_BANDS).then_some(band)
}

/// How many columns one task takes when a slab with too few rows to band is
/// split by columns. Free to follow the thread count; see
/// [`fold_column_band`].
pub(crate) fn column_band(cols: usize) -> usize {
    cols.div_ceil(rayon::current_num_threads().max(1))
        .max(DIM0_MIN_BLOCK)
}

/// Fold columns `start..start + width` of a row-major slab `cols` wide, on one
/// thread, handing `step` the part of each row they cover with `first` set to
/// `start`.
///
/// Each column's rows are blocked by `FOLD_ROWS` however wide the band, so the
/// width cannot change any column's answer and is free to follow the thread
/// count. This used to be a running total down the rows, a chain as long as
/// the slab: 16 ulps on the variance of a `(512, 4096)` matrix down its rows.
pub(crate) fn fold_column_band<I, A, S, M>(
    input: &[I],
    cols: usize,
    start: usize,
    width: usize,
    init: A,
    step: S,
    merge: M,
) -> Vec<A>
where
    I: Copy,
    A: Copy,
    S: Fn(&mut [A], &[I], usize),
    M: Fn(A, A) -> A,
{
    let rows = input.len() / cols;
    let blocks: Vec<Vec<A>> = (0..rows)
        .step_by(FOLD_ROWS)
        .map(|first| {
            let mut acc = vec![init; width];
            for row in first..(first + FOLD_ROWS).min(rows) {
                let at = row * cols + start;
                step(&mut acc, &input[at..at + width], start);
            }
            acc
        })
        .collect();
    if blocks.is_empty() {
        return vec![init; width];
    }
    pairwise_fold_vectors(blocks, merge)
}

/// How many accumulator lanes [`fold_rows_with`] folds narrow rows into: a
/// whole number of rows, at least `NARROW_ROW` elements. A caller whose step
/// indexes a per-column vector tiles it to at least this length.
pub(crate) fn fold_lanes(cols: usize) -> usize {
    if cols < NARROW_ROW {
        NARROW_ROW.div_ceil(cols) * cols
    } else {
        cols
    }
}

/// [`fold_slab_with`] on one thread: blocks of rows each run into their own
/// accumulator row, and the blocks fold pairwise. A block is `FOLD_ROWS` rows,
/// or `RUN_SUM_CHUNK` elements when the rows are narrow enough that that is
/// more; either way each accumulator lane sees a chain of about `FOLD_ROWS`
/// additions. Blocking by `RUN_SUM_CHUNK` rows instead left a 512-row slab one
/// 512-long chain per column, 17 ulps from the exact sum.
///
/// A narrow row leaves the accumulate loop a few elements long, too short to
/// vectorize, so narrow rows are taken `k` at a time into [`fold_lanes`]
/// lanes and merged down to one row at the end of the block. Summing
/// `(16, 100000, 2)` over its middle axis went from 1.06ms to 0.22ms this way,
/// and a `(1000000, 2)` matrix over its rows from 0.65 to 0.14 -- and both
/// closer to the exact answer, since `k` accumulators per column is `k`
/// shorter rounding chains.
fn fold_rows_with<I, A, S, M>(input: &[I], cols: usize, init: A, step: S, merge: M) -> Vec<A>
where
    I: Copy,
    A: Copy,
    S: Fn(&mut [A], &[I], usize) + Copy,
    M: Fn(A, A) -> A + Copy,
{
    let span = fold_lanes(cols);
    let blocks: Vec<Vec<A>> = input
        .chunks((RUN_SUM_CHUNK / cols).max(FOLD_ROWS) * cols)
        .map(|block| {
            let mut wide = vec![init; span];
            let mut spans = block.chunks_exact(span);
            for group in &mut spans {
                step(&mut wide, group, 0);
            }
            let mut acc = vec![init; cols];
            for row in spans.remainder().chunks_exact(cols) {
                step(&mut acc, row, 0);
            }
            // The `k` rows' worth of lanes fold pairwise, not one after
            // another: a single column is 64 lanes, and merging them in a
            // chain left a 4096-element row's mean 6 ulps out where `sum`'s
            // own tree lands 2.
            let mut groups = span / cols;
            while groups > 1 {
                let half = groups / 2;
                let (low, high) = wide.split_at_mut(half * cols);
                for (slot, &value) in low.iter_mut().zip(&high[..half * cols]) {
                    *slot = merge(*slot, value);
                }
                if groups % 2 == 1 {
                    wide.copy_within((groups - 1) * cols..groups * cols, half * cols);
                    groups = half + 1;
                } else {
                    groups = half;
                }
            }
            for (slot, &value) in acc.iter_mut().zip(&wide[..cols]) {
                *slot = merge(value, *slot);
            }
            acc
        })
        .collect();
    if blocks.is_empty() {
        return vec![init; cols];
    }
    pairwise_fold_vectors(blocks, merge)
}

/// Rows narrower than this many elements are accumulated several at a time;
/// see [`fold_rows_with`].
const NARROW_ROW: usize = 64;

/// The rounding chain each accumulator lane in [`fold_slab_with`] is held to.
const FOLD_ROWS: usize = RUN_SUM_CHUNK / NARROW_ROW;

/// Reduce the middle axis of a row-major `(outer, len, inner)` slice, one value
/// per `(outer, inner)` position, for any rank: a 1-D input is `(1, len, 1)`,
/// and a 2-D one reduced along `dim` is `(1, rows, cols)` or `(rows, cols, 1)`.
///
/// An axis with nothing after it is `outer` contiguous runs, each summed by
/// `run` -- [`accurate_run_sum`] over the dtype's SIMD kernel. Anything else is
/// `outer` slabs of `(len, inner)` rows, reduced as [`reduce_along_dim0`]
/// reduces one: the rows streamed in memory order and blocked, the blocks
/// folded pairwise.
///
/// Before this, a rank-3 or higher input walked each output's `len` terms one
/// at a time `inner` apart: a single rounding chain as long as the axis, and a
/// strided read. Summing a `(4, 1000000, 1)` float32 tensor over its middle
/// axis landed 188 ulps from the correctly rounded answer where NumPy, which
/// sees a contiguous run there, lands one; this lands one too.
///
/// Which route a slab takes follows from the shape alone -- with at least
/// `DIM0_MIN_BANDS` slabs each is folded whole on one thread, otherwise each
/// is spread across the pool by `reduce_along_dim0` -- so, like everything
/// here, the answer does not depend on the thread count.
#[allow(clippy::too_many_arguments)]
fn reduce_axis<I, A, F, M, R>(
    input: &[I],
    out: &mut [A],
    len: usize,
    inner: usize,
    init: A,
    combine: F,
    merge: M,
    run: R,
) where
    I: Copy + Send + Sync,
    A: Copy + Send + Sync,
    F: Fn(A, I) -> A + Send + Sync + Copy,
    M: Fn(A, A) -> A + Send + Sync + Copy,
    R: Fn(&[I]) -> A + Send + Sync,
{
    if out.is_empty() {
        return;
    }
    let input_bytes = std::mem::size_of_val(input);
    if inner == 1 {
        par_out_chunks_sized(out, outputs_per_task(len), input_bytes, &|start, chunk| {
            for (offset, slot) in chunk.iter_mut().enumerate() {
                let base = (start + offset) * len;
                *slot = run(&input[base..base + len]);
            }
        });
        return;
    }
    let outer = out.len() / inner;
    let slab = len * inner;
    if outer < DIM0_MIN_BANDS {
        for (source, target) in input.chunks_exact(slab).zip(out.chunks_exact_mut(inner)) {
            reduce_along_dim0(source, target, inner, init, combine, merge);
        }
        return;
    }
    let slabs_per_task = outputs_per_task(slab).div_ceil(inner).max(1);
    par_out_chunks_sized(out, slabs_per_task * inner, input_bytes, &|start, chunk| {
        let first = start / inner;
        for (index, target) in chunk.chunks_exact_mut(inner).enumerate() {
            let source = &input[(first + index) * slab..(first + index + 1) * slab];
            target.copy_from_slice(&fold_slab_with(
                source,
                inner,
                init,
                move |acc: &mut [A], values: &[I], _| {
                    for (slot, &value) in acc.iter_mut().zip(values) {
                        *slot = combine(*slot, value);
                    }
                },
                merge,
                false,
            ));
        }
    });
}

/// Generates a sum-along-dim reduction kernel. The body is identical across
/// numeric dtypes; only the element type, the additive identity, and the SIMD
/// row-sum helper differ.
macro_rules! sum_along_dim_kernel {
    ($name:ident, $accessor:ident, $accessor_mut:ident, $tyname:literal, $acc:ty, $zero:expr,
     $simd_sum:ident) => {
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
            let Some(&len) = input_shape.get(dim) else {
                return Err(MinitensorError::dim_out_of_range(
                    dim as isize,
                    tensor.ndim(),
                ));
            };
            let inner = input_shape[dim + 1..].iter().product::<usize>();
            reduce_axis(
                input_data,
                result_slice,
                len,
                inner,
                $zero,
                |a: $acc, v| a.acc_add(v as $acc),
                |a: $acc, b| a.acc_add(b),
                |run| accurate_run_sum(run, $simd_sum),
            );
            Ok(())
        }
    };
}

/// Generates a NaN-ignoring sum-along-dim reduction kernel. Float dtypes only
/// (integer dtypes have no NaN, so they route through the plain sum kernel).
macro_rules! nansum_along_dim_kernel {
    ($name:ident, $ty:ty, $accessor:ident, $accessor_mut:ident, $tyname:literal, $zero:expr,
     $simd_nansum:ident) => {
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
            let Some(&len) = input_shape.get(dim) else {
                return Err(MinitensorError::dim_out_of_range(
                    dim as isize,
                    tensor.ndim(),
                ));
            };
            let inner = input_shape[dim + 1..].iter().product::<usize>();
            reduce_axis(
                input_data,
                result_slice,
                len,
                inner,
                $zero,
                |a: $ty, v: $ty| if v.is_nan() { a } else { a + v },
                |a: $ty, b| a + b,
                |run| accurate_run_sum(run, $simd_nansum),
            );
            Ok(())
        }
    };
}

sum_along_dim_kernel!(
    sum_along_dim_f32,
    as_f32_slice,
    as_f32_slice_mut,
    "f32",
    f32,
    0f32,
    simd_sum_f32
);

nansum_along_dim_kernel!(
    nansum_along_dim_f32,
    f32,
    as_f32_slice,
    as_f32_slice_mut,
    "f32",
    0f32,
    simd_nansum_f32
);

sum_along_dim_kernel!(
    sum_along_dim_f64,
    as_f64_slice,
    as_f64_slice_mut,
    "f64",
    f64,
    0f64,
    simd_sum_f64
);

nansum_along_dim_kernel!(
    nansum_along_dim_f64,
    f64,
    as_f64_slice,
    as_f64_slice_mut,
    "f64",
    0f64,
    simd_nansum_f64
);

sum_along_dim_kernel!(
    sum_along_dim_i32,
    as_i32_slice,
    as_i64_slice_mut,
    "i32",
    i64,
    0i64,
    simd_sum_i32_to_i64
);

sum_along_dim_kernel!(
    sum_along_dim_i64,
    as_i64_slice,
    as_i64_slice_mut,
    "i64",
    i64,
    0i64,
    simd_sum_i64
);

// Counting a mask along an axis -- how many tokens each sequence has, how many
// of each row's predictions were right. Native for the same reason the
// whole-tensor count is: reaching it by widening the mask to `int64` first
// copies eight bytes per byte of question.
sum_along_dim_kernel!(
    sum_along_dim_bool,
    as_bool_slice,
    as_i64_slice_mut,
    "bool",
    i64,
    0i64,
    simd_count_true
);

#[inline]
pub fn prod_along_dim(tensor: &Tensor, dim: usize, keepdim: bool) -> Result<Tensor> {
    check_dim(dim, tensor.ndim())?;

    let input_shape = tensor.shape().dims();
    let mut output_shape = input_shape.to_vec();
    if keepdim {
        output_shape[dim] = 1;
    } else {
        output_shape.remove(dim);
    }
    let output_shape_obj = Shape::new(output_shape);
    // `bool` has no multiplication to accumulate in, so -- like `sum` -- the
    // result lands in `Int64` and the integer path takes it from there.
    if tensor.dtype() == DataType::Bool {
        return prod_along_dim(&tensor.astype(DataType::Int64)?, dim, keepdim);
    }
    let out_dtype = accumulating_dtype(tensor.dtype());
    let mut result_data =
        TensorData::zeros_on_device(output_shape_obj.numel(), out_dtype, tensor.device());

    let len = input_shape[dim];
    let inner = input_shape[dim + 1..].iter().product::<usize>();
    macro_rules! float_prod {
        ($accessor:ident, $accessor_mut:ident) => {{
            let input = tensor
                .data()
                .$accessor()
                .ok_or_else(|| MinitensorError::internal_error("Failed to get float slice"))?;
            let out = result_data
                .$accessor_mut()
                .ok_or_else(|| MinitensorError::internal_error("Failed to get float slice"))?;
            prod_float_along_axis(input, out, len, inner);
        }};
    }
    match tensor.dtype() {
        DataType::Float32 => float_prod!(as_f32_slice, as_f32_slice_mut),
        DataType::Float64 => float_prod!(as_f64_slice, as_f64_slice_mut),
        DataType::Int32 => prod_along_dim_i32(tensor, &mut result_data, dim)?,
        DataType::Int64 => prod_along_dim_i64(tensor, &mut result_data, dim)?,
        DataType::Bool => unreachable!("bool was promoted above"),
    }

    Ok(Tensor::new(
        Arc::new(result_data),
        output_shape_obj,
        out_dtype,
        tensor.device(),
        tensor.requires_grad(),
    ))
}

/// A float dtype whose product is accumulated in `f64`.
pub(crate) trait ProdFloat: Copy + Send + Sync {
    fn widen(self) -> f64;
    fn narrow(total: f64) -> Self;
}

impl ProdFloat for f32 {
    #[inline(always)]
    fn widen(self) -> f64 {
        self as f64
    }
    #[inline(always)]
    fn narrow(total: f64) -> Self {
        total as f32
    }
}

impl ProdFloat for f64 {
    #[inline(always)]
    fn widen(self) -> f64 {
        self
    }
    #[inline(always)]
    fn narrow(total: f64) -> Self {
        total
    }
}

/// The product of a contiguous run, accumulated in `f64` across eight lanes in
/// fixed chunks of `RUN_SUM_CHUNK`, the chunks multiplied in order -- and
/// spread across the pool when the run is long enough, which cannot change
/// the answer since the chunks are fixed.
///
/// `f64` because a float32 product rounded at every step is biased: the exact
/// product of two factors near one has structured low bits, and rounding them
/// to nearest lands low on average -- about 0.05 ulp a multiplication on
/// factors within 5e-4 of one. That is a drift, not a random walk, so it grows
/// with the length: four million such factors came out 15000 ulps low. No
/// grouping helps (a pairwise product was 30 times worse); a wider accumulator
/// does, rounding once at the end. NumPy multiplies in float32 and drifts the
/// same way. A float64 input gains nothing from it but the same deterministic
/// split. An answer that is not a normal `f64` is recomputed by [`exact_prod`].
pub(crate) fn prod_run<I: ProdFloat>(run: &[I]) -> f64 {
    fn chunk_prod<I: ProdFloat>(chunk: &[I]) -> f64 {
        const LANES: usize = 8;
        let mut lanes = [1f64; LANES];
        let (blocks, rest) = chunk.as_chunks::<LANES>();
        for block in blocks {
            for lane in 0..LANES {
                lanes[lane] *= block[lane].widen();
            }
        }
        let mut total: f64 = lanes.iter().product();
        for &v in rest {
            total *= v.widen();
        }
        total
    }
    let total: f64 = if run.len() < PROD_PAR_MIN {
        run.chunks(RUN_SUM_CHUNK).map(chunk_prod).product()
    } else {
        par_map_indexed(run.len().div_ceil(RUN_SUM_CHUNK), &|index| {
            let from = index * RUN_SUM_CHUNK;
            chunk_prod(&run[from..(from + RUN_SUM_CHUNK).min(run.len())])
        })
        .into_iter()
        .product()
    };
    if total.is_normal() {
        total
    } else {
        exact_prod(run.iter().copied())
    }
}

/// Runs at least this long have their chunks spread across the pool.
const PROD_PAR_MIN: usize = 1 << 16;

/// The float product along the middle axis of `(outer, len, inner)`, through
/// the same routing `sum` takes -- contiguous runs by [`prod_run`], slabs by
/// the blocked row fold -- with every accumulator an `f64`; see [`prod_run`].
///
/// This replaced a slab loop that multiplied in the input's own dtype, one
/// column at a time along a contiguous row: `prod` along the rows of a
/// `(1024, 4096)` float32 matrix took 3.6ms against `sum`'s 0.19, and along the
/// middle of `(16, 100000, 2)` 2.0ms.
fn prod_float_along_axis<I: ProdFloat>(input: &[I], out: &mut [I], len: usize, inner: usize) {
    if out.is_empty() {
        return;
    }
    // No factors: every product is empty, and one. `reduce_axis` would cut the
    // input into zero-length slabs.
    if len == 0 {
        out.fill(I::narrow(1.0));
        return;
    }
    let mut wide = vec![1f64; out.len()];
    reduce_axis(
        input,
        &mut wide,
        len,
        inner,
        1f64,
        |a: f64, v: I| a * v.widen(),
        |a: f64, b: f64| a * b,
        prod_run,
    );
    let slab = len * inner;
    par_out_chunks(out, outputs_per_task(len), &|start, chunk| {
        for (offset, slot) in chunk.iter_mut().enumerate() {
            let at = start + offset;
            let total = wide[at];
            // A contiguous run came through `prod_run`, which checked already.
            *slot = I::narrow(if total.is_normal() || inner == 1 {
                total
            } else {
                let first = (at / inner) * slab + at % inner;
                exact_prod((0..len).map(|k| input[first + k * inner]))
            });
        }
    });
}

/// The product of `values` without a partial product ever leaving `f64`'s
/// range: the check [`prod_run`] and [`prod_float_along_axis`] fall back to
/// when their answer is not a normal `f64`.
///
/// Their lanes and blocks multiply separately, so on factors spanning a huge
/// range one partial can overflow to infinity while another underflows to
/// zero, and joining them gives NaN -- from finite, nonzero data, where the
/// running product the grouping replaced saturated instead. An answer that is
/// normal came through without that happening; anything else -- NaN, an
/// infinity, zero, a subnormal -- is recomputed here, which only data that
/// holds such values or truly overflows ever pays for.
///
/// A NaN factor, or a zero and an infinity together, is NaN; otherwise a zero
/// makes the product a zero and an infinity an infinity, signed by the parity
/// of the negative factors. Anything else is multiplied as a mantissa and a
/// separate exponent, so the only rounding is the last one.
pub(crate) fn exact_prod<I: ProdFloat>(values: impl Iterator<Item = I> + Clone) -> f64 {
    let (mut negative, mut zero, mut infinite, mut nan) = (false, false, false, false);
    for v in values.clone() {
        let v = v.widen();
        negative ^= v.is_sign_negative();
        zero |= v == 0.0;
        infinite |= v.is_infinite();
        nan |= v.is_nan();
    }
    let sign = if negative { -1.0 } else { 1.0 };
    if nan || (zero && infinite) {
        return f64::NAN;
    }
    if zero {
        return sign * 0.0;
    }
    if infinite {
        return sign * f64::INFINITY;
    }
    // (mantissa in [0.5, 1), exponent) of a positive, finite, nonzero value.
    fn split(x: f64) -> (f64, i64) {
        let (x, bias) = if x < f64::MIN_POSITIVE {
            (x * 2f64.powi(64), -64)
        } else {
            (x, 0)
        };
        let bits = x.to_bits();
        let exponent = ((bits >> 52) & 0x7ff) as i64 - 1022;
        let mantissa = f64::from_bits((bits & !(0x7ffu64 << 52)) | (1022u64 << 52));
        (mantissa, exponent + bias)
    }
    let (mut mantissa, mut exponent) = (1.0f64, 0i64);
    for v in values {
        let (m, e) = split(v.widen().abs());
        let (m, e2) = split(mantissa * m);
        mantissa = m;
        exponent += e + e2;
    }
    // `mantissa * 2^exponent`, saturating: in three steps so no power of two
    // taken on the way leaves the range, since the exponent can be far outside
    // it when the true product is an overflow or an underflow.
    let exponent = exponent.clamp(-3300, 3300) as i32;
    let third = exponent / 3;
    sign * mantissa * 2f64.powi(third) * 2f64.powi(third) * 2f64.powi(exponent - 2 * third)
}

/// Generates a product-along-dim reduction kernel. Body is identical across
/// numeric dtypes; only the element type and multiplicative identity differ.
macro_rules! prod_along_dim_kernel {
    ($name:ident, $accessor:ident, $accessor_mut:ident, $tyname:literal, $acc:ty, $one:expr) => {
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
            let outer = result_slice.len() / inner;
            par_out_chunks(
                result_slice,
                reduction_band(outer, inner),
                &|start, out_chunk| {
                    out_chunk.fill($one);
                    // A band is part of one block's row rather than all of it, so
                    // the block is fixed and the chunk starts `start % inner`
                    // columns into it. Each column still multiplies its own steps
                    // in their own order, so the cut cannot move an answer.
                    let block_base = (start / inner) * outer_stride + start % inner;
                    let width = out_chunk.len();
                    for k in 0..dim_size {
                        let slab_base = block_base + k * inner;
                        let slab = &input_data[slab_base..slab_base + width];
                        for (acc, &v) in out_chunk.iter_mut().zip(slab) {
                            *acc = acc.acc_mul(v as $acc);
                        }
                    }
                },
            );
            Ok(())
        }
    };
}

prod_along_dim_kernel!(
    prod_along_dim_i32,
    as_i32_slice,
    as_i64_slice_mut,
    "i32",
    i64,
    1i64
);

prod_along_dim_kernel!(
    prod_along_dim_i64,
    as_i64_slice,
    as_i64_slice_mut,
    "i64",
    i64,
    1i64
);

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
pub(crate) const MINMAX_CHUNK: usize = 8 * 1024;

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
///
/// Eight was not enough of them. `sum` stayed twice as quick over the same
/// buffer long after that split, and the reason is arithmetic rather than
/// anything about the kernels: `maxps` has a four-cycle latency with two
/// issuing per cycle, so eight chains are needed to keep the unit fed -- and
/// eight `f32` *lanes* is two SSE vectors, so two chains. Per element over one
/// chunk, single-threaded:
///
/// ```text
///   lanes       f32+NaN       f64+NaN
///       4      0.363 ns      0.357 ns
///       8      0.175         0.232
///      16      0.128         0.275
///      32      0.635              -
/// ```
///
/// The NaN flag is what caps it: it is a second array, so the registers run
/// out an octave sooner than they would for a bare extremum, which kept
/// improving to 32 lanes and 0.055 ns -- level with `sum`. Carrying the flag
/// costs about a tenth at the right width and the whole win at the wrong one,
/// which is why these numbers are per fold shape and not one constant.
macro_rules! float_extremum_all {
    ($name:ident, $accessor:ident, $accessor_mut:ident, $ty:ty, $tyname:literal, $identity:expr, $better:tt, $lanes:expr) => {
        pub(crate) fn $name(tensor: &Tensor, result_data: &mut TensorData) -> Result<()> {
            const LANES: usize = $lanes;

            /// One chunk's extremum and NaN flag.
            ///
            /// `#[inline(always)]` is what makes the wrapper below a second
            /// compilation rather than a call to this one: inlining into a
            /// `#[target_feature]` function rebuilds the body with that
            /// function's registers available. The same arrangement the binary
            /// kernels in `ops::simd` use.
            #[inline(always)]
            fn body(chunk: &[$ty]) -> ($ty, bool) {
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
            }

            // `avx`, not `avx2`: the comparison and select are float
            // operations and 256-bit `maxps`/`maxpd` arrived with AVX.
            #[cfg(target_arch = "x86_64")]
            #[target_feature(enable = "avx")]
            fn body_avx(chunk: &[$ty]) -> ($ty, bool) {
                body(chunk)
            }

            #[inline]
            fn fold_chunk(chunk: &[$ty]) -> ($ty, bool) {
                #[cfg(target_arch = "x86_64")]
                if crate::ops::simd::simd_capabilities().avx {
                    // SAFETY: `detect` confirmed avx on this CPU.
                    return unsafe { body_avx(chunk) };
                }
                body(chunk)
            }

            let data = tensor.data().$accessor().ok_or_else(|| {
                MinitensorError::internal_error(concat!("Failed to get ", $tyname, " slice"))
            })?;

            let (value, has_nan) = par_fold_chunks(
                data,
                MINMAX_CHUNK,
                ($identity, false),
                &|_, chunk| fold_chunk(chunk),
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
///
/// These carry one array rather than two, so they keep improving past where
/// the float ones stop. Per element over one chunk, single-threaded:
///
/// ```text
///   lanes         i32           i64
///       4            -      0.247 ns
///       8      0.183 ns      0.246
///      16      0.134         0.322
///      32      0.126         0.318
///      64      0.124              -
/// ```
///
/// `i32` takes 32 -- 64 is another 2% and spends every vector register to get
/// it. `i64` is left at 4: eight measured the same to within the noise, and a
/// count is not worth changing without a reason to.
macro_rules! int_extremum_all {
    ($name:ident, $accessor:ident, $accessor_mut:ident, $ty:ty, $tyname:literal, $identity:expr, $better:tt, $lanes:expr, $wide:expr) => {
        pub(crate) fn $name(tensor: &Tensor, result_data: &mut TensorData) -> Result<()> {
            const LANES: usize = $lanes;

            /// One chunk's extremum. See the float version for why the body is
            /// written once and inlined into a second compilation.
            #[inline(always)]
            fn body(chunk: &[$ty]) -> $ty {
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
            }

            // `avx2`, not `avx`: these are integer compares, and the 256-bit
            // `vpmaxsd` that serves them is an AVX2 instruction.
            //
            // `$wide` is false for `i64`, which is the one type that does not
            // want it. There is no packed 64-bit integer maximum before
            // AVX-512, so the wide body has to synthesise one from a compare
            // and a blend, and that measured slower than the baseline it
            // replaced -- 0.291 ns an element against 0.247. A second
            // compilation is only worth having where it is faster.
            #[cfg(target_arch = "x86_64")]
            #[target_feature(enable = "avx2")]
            #[allow(dead_code)]
            fn body_avx2(chunk: &[$ty]) -> $ty {
                body(chunk)
            }

            #[inline]
            fn fold_chunk(chunk: &[$ty]) -> $ty {
                #[cfg(target_arch = "x86_64")]
                if $wide && crate::ops::simd::simd_capabilities().avx2 {
                    // SAFETY: `detect` confirmed avx2 on this CPU.
                    return unsafe { body_avx2(chunk) };
                }
                body(chunk)
            }

            let data = tensor.data().$accessor().ok_or_else(|| {
                MinitensorError::internal_error(concat!("Failed to get ", $tyname, " slice"))
            })?;

            let value = par_fold_chunks(
                data,
                MINMAX_CHUNK,
                $identity,
                &|_, chunk| fold_chunk(chunk),
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
    16
);
float_extremum_all!(
    max_all_f64,
    as_f64_slice,
    as_f64_slice_mut,
    f64,
    "f64",
    f64::NEG_INFINITY,
    >,
    8
);
int_extremum_all!(
    max_all_i32,
    as_i32_slice,
    as_i32_slice_mut,
    i32,
    "i32",
    i32::MIN,
    >,
    32,
    true
);
int_extremum_all!(
    max_all_i64,
    as_i64_slice,
    as_i64_slice_mut,
    i64,
    "i64",
    i64::MIN,
    >,
    4,
    false
);

pub(crate) fn max_all_bool(tensor: &Tensor, result_data: &mut TensorData) -> Result<()> {
    let data = tensor
        .data()
        .as_bool_slice()
        .ok_or_else(|| MinitensorError::internal_error("Failed to get bool slice"))?;

    let max_val = par_any_chunk(data, PAR_CHUNK, &|chunk| chunk.iter().any(|&x| x));

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
    16
);
float_extremum_all!(
    min_all_f64,
    as_f64_slice,
    as_f64_slice_mut,
    f64,
    "f64",
    f64::INFINITY,
    <,
    8
);
int_extremum_all!(
    min_all_i32,
    as_i32_slice,
    as_i32_slice_mut,
    i32,
    "i32",
    i32::MAX,
    <,
    32,
    true
);
int_extremum_all!(
    min_all_i64,
    as_i64_slice,
    as_i64_slice_mut,
    i64,
    "i64",
    i64::MAX,
    <,
    4,
    false
);

pub(crate) fn min_all_bool(tensor: &Tensor, result_data: &mut TensorData) -> Result<()> {
    let data = tensor
        .data()
        .as_bool_slice()
        .ok_or_else(|| MinitensorError::internal_error("Failed to get bool slice"))?;

    let min_val = par_all_chunk(data, PAR_CHUNK, &|chunk| chunk.iter().all(|&x| x));

    let result_slice = result_data
        .as_bool_slice_mut()
        .ok_or_else(|| MinitensorError::internal_error("Failed to get mutable bool slice"))?;

    result_slice[0] = min_val;
    Ok(())
}
