// Copyright (c) Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

use super::*;
use crate::{
    error::{MinitensorError, Result},
    tensor::{DataType, Shape, Tensor, TensorData},
};
use num_traits::Float;
use rayon::prelude::*;
use std::sync::Arc;

/// A dimension reduction seen as `outer` independent slabs of
/// `dim_size * inner` elements: within a slab, position `d` along the reduced
/// dimension owns the run `[d * inner, (d + 1) * inner)`.
pub(crate) struct DimReductionLayout {
    pub(crate) output_shape: Shape,
    pub(crate) dim_size: usize,
    pub(crate) inner: usize,
    pub(crate) outer_stride: usize,
}

pub(crate) fn reduction_layout(
    tensor: &Tensor,
    dim: usize,
    keepdim: bool,
) -> Result<DimReductionLayout> {
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
    let dim_size = input_shape[dim];
    let inner = input_shape[dim + 1..].iter().product::<usize>();
    let outer_stride = dim_size * inner;

    Ok(DimReductionLayout {
        output_shape: Shape::new(output_shape),
        dim_size,
        inner,
        outer_stride,
    })
}

/// Reduced-axis width past which the memory-order path wins.
///
/// Not simply "wherever striding hurts". The blocked path parallelizes over
/// bands of the *output*, so a narrow output has few bands to hand out: at
/// width 16 it collapses to a single task and ran slower than the strided walk
/// it replaced (3.2ms against 2.3ms on a 131072x16 f32 reduction), even though
/// the strided walk touches a new cache line every step. The strided path
/// parallelizes over output elements instead, which is the better trade while
/// the output is small. Measured crossover on f32: 16 and 64 favour striding,
/// 1024 and 32768 favour blocking by 3.5x and 1.6x.
const BLOCKED_INNER_MIN: usize = 256;

/// Floor on a column band, so a narrow slab is not split into slivers whose
/// per-task overhead exceeds the work.
const BLOCKED_MIN_BAND: usize = 64;

/// Reduce `input` along a dimension into `output`, parallelizing over output
/// elements (one rayon task per output position, each walking its column of the
/// reduced dimension with a running offset). `combine` folds the accumulator
/// with each element; `short_circuit` returning `Some(v)` stops the column early
/// with `v` (used to propagate NaN and to break out of boolean any/all).
#[inline]
fn reduce_along_dim_par<T, C, S>(
    input: &[T],
    output: &mut [T],
    layout: &DimReductionLayout,
    init: T,
    combine: C,
    short_circuit: S,
) where
    T: Copy + Send + Sync,
    C: Fn(T, T) -> T + Sync,
    S: Fn(T) -> Option<T> + Sync,
{
    let inner = layout.inner;
    let dim_size = layout.dim_size;
    let outer_stride = layout.outer_stride;

    // Walking one output at a time strides the input by `inner`, so with a wide
    // reduced axis every step lands on a different cache line: `max(dim=0)` on a
    // 2048x1024 f32 matrix took 4.3ms against 0.23ms for `sum` over the same
    // axis, which walks row-major instead. The cost tracked `inner` exactly --
    // 19x at 1024, 2.7x at 64, gone by 8 -- so above that width the loops are
    // swapped: stream the input in memory order and keep `inner` accumulators
    // live. `combine` alone decides the result here; the short-circuit is an
    // optimization for the strided path, and every caller's combine is correct
    // without it.
    if inner >= BLOCKED_INNER_MIN {
        let outer = if outer_stride == 0 {
            1
        } else {
            input.len() / outer_stride.max(1)
        };
        if outer > 1 {
            output
                .par_chunks_mut(inner)
                .enumerate()
                .for_each(|(o, row)| {
                    let base = o * outer_stride;
                    row.fill(init);
                    for step in 0..dim_size {
                        let slab = &input[base + step * inner..][..inner];
                        for (acc, &value) in row.iter_mut().zip(slab) {
                            *acc = combine(*acc, value);
                        }
                    }
                });
        } else {
            // A single slab has no outer parallelism, so split the accumulator
            // range into column bands instead; each band still streams its own
            // columns in order.
            let band = inner
                .div_ceil(rayon::current_num_threads().max(1))
                .max(BLOCKED_MIN_BAND);
            output
                .par_chunks_mut(band)
                .enumerate()
                .for_each(|(index, cols)| {
                    let start = index * band;
                    let width = cols.len();
                    cols.fill(init);
                    for step in 0..dim_size {
                        let slab = &input[step * inner + start..][..width];
                        for (acc, &value) in cols.iter_mut().zip(slab) {
                            *acc = combine(*acc, value);
                        }
                    }
                });
        }
        return;
    }

    output
        .par_iter_mut()
        .enumerate()
        .for_each(|(out_idx, out)| {
            let o = out_idx / inner;
            let r = out_idx % inner;
            let mut acc = init;
            let mut idx = o * outer_stride + r;
            for _ in 0..dim_size {
                let val = input[idx];
                if let Some(sc) = short_circuit(val) {
                    acc = sc;
                    break;
                }
                acc = combine(acc, val);
                idx += inner;
            }
            *out = acc;
        });
}

/// Like [`reduce_along_dim_par`] but also records the index (along the reduced
/// dimension) of the winning element, parallelizing over output positions.
/// `better(candidate, current_best)` decides replacement using a strict
/// comparison, so the first winner keeps its index on a tie; `short(val)`
/// returning `Some(v)` finalizes the output early with value `v` at the
/// current index (NaN propagation, boolean any/all short-circuit).
#[inline]
pub(crate) fn reduce_arg_along_dim_par<T, Better, Short>(
    input: &[T],
    values: &mut [T],
    indices: &mut [i64],
    layout: &DimReductionLayout,
    init: T,
    better: Better,
    short: Short,
) where
    T: Copy + Send + Sync,
    Better: Fn(T, T) -> bool + Sync,
    Short: Fn(T) -> Option<T> + Sync,
{
    let inner = layout.inner;
    let dim_size = layout.dim_size;
    let outer_stride = layout.outer_stride;

    // Same swap as `reduce_along_dim_par`, carrying the winning index alongside
    // the value. This is the path Python's `max(dim=...)` actually takes, since
    // it returns `(values, indices)`.
    if inner >= BLOCKED_INNER_MIN {
        let outer = if outer_stride == 0 {
            1
        } else {
            input.len() / outer_stride.max(1)
        };
        let band = if outer > 1 {
            inner
        } else {
            inner
                .div_ceil(rayon::current_num_threads().max(1))
                .max(BLOCKED_MIN_BAND)
        };
        values
            .par_chunks_mut(band)
            .zip(indices.par_chunks_mut(band))
            .enumerate()
            .for_each(|(index, (vals, idxs))| {
                let flat = index * band;
                let o = flat / inner;
                let start = flat % inner;
                let width = vals.len();
                let base = o * outer_stride + start;
                vals.fill(init);
                idxs.fill(0);
                for step in 0..dim_size {
                    let slab = &input[base + step * inner..][..width];
                    for (lane, &value) in slab.iter().enumerate() {
                        if better(value, vals[lane]) {
                            vals[lane] = value;
                            idxs[lane] = step as i64;
                        }
                    }
                }
            });
        return;
    }

    values
        .par_iter_mut()
        .zip(indices.par_iter_mut())
        .enumerate()
        .for_each(|(out_idx, (vout, iout))| {
            let o = out_idx / inner;
            let r = out_idx % inner;
            let mut best = init;
            let mut best_i = 0usize;
            let mut idx = o * outer_stride + r;
            for d in 0..dim_size {
                let val = input[idx];
                if let Some(fin) = short(val) {
                    best = fin;
                    best_i = d;
                    break;
                }
                if better(val, best) {
                    best = val;
                    best_i = d;
                }
                idx += inner;
            }
            *vout = best;
            *iout = best_i as i64;
        });
}

/// Global min/max over the non-NaN elements.
///
/// Returns NaN when every element is NaN -- there is no non-NaN element to
/// report. The four dtype-specific versions this replaces each carried a
/// `(value, found)` accumulator seeded with a `±inf` sentinel;
/// `reduce_with` has no identity element, so there is no sentinel that a real
/// input value could collide with.
fn nan_extremum_all<T: Float + Send + Sync>(data: &[T], which: Extremum) -> T {
    data.par_iter()
        .copied()
        .filter(|v| !v.is_nan())
        .reduce_with(|a, b| match which {
            Extremum::Max => a.max(b),
            Extremum::Min => a.min(b),
        })
        .unwrap_or_else(T::nan)
}

/// Index of the global extremum.
///
/// Ties go to the lowest index. A NaN wins outright; ties among NaNs also go
/// to the lowest index. As
/// with [`nan_extremum_all`] there is no identity element, so the `±inf` /
/// `i32::MIN` seeds the ten dtype-specific versions carried -- each of which a
/// real input could equal -- are gone.
fn arg_extremum_all<T, IsNan, Better>(data: &[T], is_nan: IsNan, better: Better) -> usize
where
    T: Copy + Send + Sync,
    IsNan: Fn(T) -> bool + Sync,
    Better: Fn(T, T) -> bool + Sync,
{
    data.par_iter()
        .copied()
        .enumerate()
        .reduce_with(|(i1, v1), (i2, v2)| match (is_nan(v1), is_nan(v2)) {
            (true, true) => {
                if i1 <= i2 {
                    (i1, v1)
                } else {
                    (i2, v2)
                }
            }
            (true, false) => (i1, v1),
            (false, true) => (i2, v2),
            (false, false) => {
                if better(v1, v2) {
                    (i1, v1)
                } else if better(v2, v1) {
                    (i2, v2)
                } else if i1 <= i2 {
                    (i1, v1)
                } else {
                    (i2, v2)
                }
            }
        })
        .map_or(0, |(i, _)| i)
}

#[inline]
fn write_index(result_data: &mut TensorData, index: usize) -> Result<()> {
    let slot = result_data
        .as_i64_slice_mut()
        .ok_or_else(|| MinitensorError::internal_error("Failed to get mutable i64 slice"))?;
    slot[0] = index as i64;
    Ok(())
}

/// `nan{min,max}_all_{f32,f64}`: the whole-tensor NaN-skipping extremum.
macro_rules! nan_extremum_all_entry {
    ($name:ident, $which:ident, $accessor:ident, $mut_accessor:ident, $tyname:literal) => {
        pub(crate) fn $name(tensor: &Tensor, result_data: &mut TensorData) -> Result<()> {
            let data = tensor.data().$accessor().ok_or_else(|| {
                MinitensorError::internal_error(concat!("Failed to get ", $tyname, " slice"))
            })?;
            let value = nan_extremum_all(data, Extremum::$which);
            let slot = result_data.$mut_accessor().ok_or_else(|| {
                MinitensorError::internal_error(concat!(
                    "Failed to get mutable ",
                    $tyname,
                    " slice"
                ))
            })?;
            slot[0] = value;
            Ok(())
        }
    };
}

nan_extremum_all_entry!(nanmax_all_f32, Max, as_f32_slice, as_f32_slice_mut, "f32");
nan_extremum_all_entry!(nanmax_all_f64, Max, as_f64_slice, as_f64_slice_mut, "f64");
nan_extremum_all_entry!(nanmin_all_f32, Min, as_f32_slice, as_f32_slice_mut, "f32");
nan_extremum_all_entry!(nanmin_all_f64, Min, as_f64_slice, as_f64_slice_mut, "f64");

/// `arg{min,max}_all_*` for a dtype that can hold NaN.
macro_rules! arg_extremum_all_float {
    ($name:ident, $accessor:ident, $ty:ty, $tyname:literal, $better:expr) => {
        pub(crate) fn $name(tensor: &Tensor, result_data: &mut TensorData) -> Result<()> {
            let data = tensor.data().$accessor().ok_or_else(|| {
                MinitensorError::internal_error(concat!("Failed to get ", $tyname, " slice"))
            })?;
            let index = arg_extremum_all(data, |v: $ty| v.is_nan(), $better);
            write_index(result_data, index)
        }
    };
}

/// `arg{min,max}_all_*` for a dtype with no NaN.
macro_rules! arg_extremum_all_exact {
    ($name:ident, $accessor:ident, $ty:ty, $tyname:literal, $better:expr) => {
        pub(crate) fn $name(tensor: &Tensor, result_data: &mut TensorData) -> Result<()> {
            let data = tensor.data().$accessor().ok_or_else(|| {
                MinitensorError::internal_error(concat!("Failed to get ", $tyname, " slice"))
            })?;
            let index = arg_extremum_all(data, |_: $ty| false, $better);
            write_index(result_data, index)
        }
    };
}

/// Bool has two values, so the answer is the first `true` (max) or first
/// `false` (min); `position_first` finds it in parallel and short-circuits,
/// which a full reduction cannot.
macro_rules! arg_extremum_all_bool {
    ($name:ident, $wanted:literal) => {
        pub(crate) fn $name(tensor: &Tensor, result_data: &mut TensorData) -> Result<()> {
            let data = tensor
                .data()
                .as_bool_slice()
                .ok_or_else(|| MinitensorError::internal_error("Failed to get bool slice"))?;
            let index = data
                .par_iter()
                .position_first(|&x| x == $wanted)
                .unwrap_or(0);
            write_index(result_data, index)
        }
    };
}

arg_extremum_all_float!(argmax_all_f32, as_f32_slice, f32, "f32", |a, b| a > b);
arg_extremum_all_float!(argmax_all_f64, as_f64_slice, f64, "f64", |a, b| a > b);
arg_extremum_all_exact!(argmax_all_i32, as_i32_slice, i32, "i32", |a, b| a > b);
arg_extremum_all_exact!(argmax_all_i64, as_i64_slice, i64, "i64", |a, b| a > b);
arg_extremum_all_bool!(argmax_all_bool, true);

arg_extremum_all_float!(argmin_all_f32, as_f32_slice, f32, "f32", |a, b| a < b);
arg_extremum_all_float!(argmin_all_f64, as_f64_slice, f64, "f64", |a, b| a < b);
arg_extremum_all_exact!(argmin_all_i32, as_i32_slice, i32, "i32", |a, b| a < b);
arg_extremum_all_exact!(argmin_all_i64, as_i64_slice, i64, "i64", |a, b| a < b);
arg_extremum_all_bool!(argmin_all_bool, false);

/// Fold one contiguous float row to its extremum, propagating NaN.
///
/// This is [`super::sum_prod`]'s `float_extremum_all!` applied a row at a time.
/// That macro exists because a single accumulator makes the compare-and-select
/// a serial dependency chain that cannot vectorize, and splitting it across
/// `$lanes` independent accumulators measured 6.2x faster on f32 -- but it only
/// ever covered the whole-tensor reduction. The along-a-dimension fold kept the
/// scalar walk, with a NaN test and a `break` on every element, so `max` along
/// the last axis of a 4096x1024 f32 tensor took 2.86 ms where `sum` over the
/// same axis took 0.23 ms.
///
/// NaN is tracked as a separate flag so the value loop stays a bare comparison:
/// `v > best` is false for a NaN, so a NaN never displaces a real value, and the
/// flag decides the result at the end. `f32::max` would be wrong here -- it
/// returns the *non*-NaN operand, the opposite of propagation.
///
/// The lane count is fixed rather than taken from the hardware, so the fold
/// groups the same way on every machine.
macro_rules! float_extremum_row {
    ($name:ident, $ty:ty, $identity:expr, $better:tt, $lanes:expr) => {
        #[inline]
        fn $name(row: &[$ty]) -> $ty {
            const LANES: usize = $lanes;
            let mut bests = [$identity; LANES];
            let mut nans = [0u32; LANES];
            let mut blocks = row.chunks_exact(LANES);
            for block in &mut blocks {
                for lane in 0..LANES {
                    let v = block[lane];
                    if v $better bests[lane] {
                        bests[lane] = v;
                    }
                    // `as u32` rather than a bool `|=`: keeps the lane update
                    // branch-free so it vectorizes with the comparison above.
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
            if nan != 0 { <$ty>::NAN } else { best }
        }
    };
}

float_extremum_row!(max_row_f32, f32, f32::NEG_INFINITY, >, 8);
float_extremum_row!(min_row_f32, f32, f32::INFINITY, <, 8);
float_extremum_row!(max_row_f64, f64, f64::NEG_INFINITY, >, 4);
float_extremum_row!(min_row_f64, f64, f64::INFINITY, <, 4);

/// Fold `width` columns of a slab, streaming the input in memory order.
///
/// The same idea as [`float_extremum_row`] with the slab's own columns as the
/// lanes: `width` accumulators are already live and independent, so the only
/// thing stopping the loop from vectorizing was the NaN test inside the fold.
/// Carrying a NaN mask alongside keeps the comparison branchless and puts the
/// propagation back in one pass at the end. Through the generic closure this
/// cost `max` along dimension 0 of a 4096x1024 f32 tensor 3.6 ms where `sum`
/// over the same axis took 0.39 ms.
macro_rules! float_extremum_columns {
    ($name:ident, $ty:ty, $identity:expr, $better:tt) => {
        /// Fold steps `[from, to)` of the slab at `base`, over the `out.len()`
        /// columns starting at `start`.
        #[inline]
        fn $name(
            input: &[$ty],
            base: usize,
            start: usize,
            from: usize,
            to: usize,
            inner: usize,
            out: &mut [$ty],
        ) {
            let width = out.len();
            let mut nans = vec![0u32; width];
            out.fill($identity);
            for step in from..to {
                let slab = &input[base + step * inner + start..][..width];
                for ((acc, flag), &v) in out.iter_mut().zip(nans.iter_mut()).zip(slab) {
                    if v $better *acc {
                        *acc = v;
                    }
                    *flag |= (v != v) as u32;
                }
            }
            for (acc, &flag) in out.iter_mut().zip(nans.iter()) {
                if flag != 0 {
                    *acc = <$ty>::NAN;
                }
            }
        }
    };
}

float_extremum_columns!(max_columns_f32, f32, f32::NEG_INFINITY, >);
float_extremum_columns!(min_columns_f32, f32, f32::INFINITY, <);
float_extremum_columns!(max_columns_f64, f64, f64::NEG_INFINITY, >);
float_extremum_columns!(min_columns_f64, f64, f64::INFINITY, <);

/// Reduce `dim` to its extremum, without reporting where it was found.
///
/// The value-only forms of `min` and `max` differed only in their seed and
/// their fold, exactly as the value-and-index forms in `minmax_indices` did.
/// A NaN anywhere in a float slice short-circuits the whole column; bool
/// short-circuits on the first `true` (max) or `false` (min).
fn extremum_along_dim(
    tensor: &Tensor,
    dim: usize,
    keepdim: bool,
    which: Extremum,
) -> Result<Tensor> {
    let layout = reduction_layout(tensor, dim, keepdim)?;
    let mut result_data =
        TensorData::zeros_on_device(layout.output_shape.numel(), tensor.dtype(), tensor.device());
    let is_max = which == Extremum::Max;

    macro_rules! float_arm {
        ($accessor:ident, $mut_accessor:ident, $ty:ty, $tyname:literal, $row_max:ident, $row_min:ident, $col_max:ident, $col_min:ident) => {{
            let input = tensor.data().$accessor().ok_or_else(|| {
                MinitensorError::internal_error(concat!("Failed to get ", $tyname, " slice"))
            })?;
            let output = result_data.$mut_accessor().ok_or_else(|| {
                MinitensorError::internal_error(concat!(
                    "Failed to get mutable ",
                    $tyname,
                    " slice"
                ))
            })?;
            let seed = if is_max {
                <$ty>::NEG_INFINITY
            } else {
                <$ty>::INFINITY
            };
            // The reduced axis is the last one, so each output's column is a
            // contiguous run and can go through the vectorized row fold rather
            // than the general strided walk.
            if layout.inner == 1 {
                let dim_size = layout.dim_size;
                output.par_iter_mut().enumerate().for_each(|(o, out)| {
                    let row = &input[o * dim_size..][..dim_size];
                    *out = if is_max { $row_max(row) } else { $row_min(row) };
                });
            } else if layout.inner >= BLOCKED_INNER_MIN {
                // Wide reduced axis: stream the slabs in memory order with the
                // columns as accumulators. Same partition as the generic
                // blocked path in `reduce_along_dim_par`, since the choice of
                // bands is what makes that path worth taking.
                let (dim_size, inner, outer_stride) =
                    (layout.dim_size, layout.inner, layout.outer_stride);
                let outer = if outer_stride == 0 {
                    1
                } else {
                    input.len() / outer_stride.max(1)
                };
                if outer > 1 {
                    output
                        .par_chunks_mut(inner)
                        .enumerate()
                        .for_each(|(o, row)| {
                            let base = o * outer_stride;
                            if is_max {
                                $col_max(input, base, 0, 0, dim_size, inner, row);
                            } else {
                                $col_min(input, base, 0, 0, dim_size, inner, row);
                            }
                        });
                } else {
                    // One slab, so there is no outer work to hand out. Banding
                    // the columns gives each thread a narrow stripe of every
                    // row; banding the *rows* lets each stream a contiguous run
                    // and merge afterwards. That regrouping is free here in a
                    // way it is not for a sum: an extremum is exactly
                    // associative, so how the steps are grouped cannot change
                    // the answer.
                    let bands = rayon::current_num_threads().max(1);
                    let band = dim_size.div_ceil(bands).max(1);
                    let partials: Vec<Vec<$ty>> = (0..dim_size.div_ceil(band))
                        .into_par_iter()
                        .map(|b| {
                            let mut acc = vec![seed; inner];
                            let from = b * band;
                            let to = ((b + 1) * band).min(dim_size);
                            if is_max {
                                $col_max(input, 0, 0, from, to, inner, &mut acc);
                            } else {
                                $col_min(input, 0, 0, from, to, inner, &mut acc);
                            }
                            acc
                        })
                        .collect();
                    output.copy_from_slice(&partials[0]);
                    for partial in &partials[1..] {
                        for (slot, &v) in output.iter_mut().zip(partial) {
                            // A partial may already hold NaN, and a bare
                            // comparison would drop it.
                            *slot = if *slot != *slot || v != v {
                                <$ty>::NAN
                            } else if (v > *slot) == is_max && v != *slot {
                                v
                            } else {
                                *slot
                            };
                        }
                    }
                }
            } else {
                reduce_along_dim_par(
                    input,
                    output,
                    &layout,
                    seed,
                    // NaN-propagating on its own rather than relying on the
                    // short-circuit below: the blocked path in
                    // `reduce_along_dim_par` walks memory in order and has no
                    // per-element early exit to lean on. `a.max(v)` would be
                    // wrong here -- it returns the *non*-NaN operand.
                    move |a: $ty, v: $ty| {
                        if a != a || v != v {
                            <$ty>::NAN
                        } else if (v > a) == is_max && v != a {
                            v
                        } else {
                            a
                        }
                    },
                    |v: $ty| if v.is_nan() { Some(<$ty>::NAN) } else { None },
                );
            }
        }};
    }

    macro_rules! int_arm {
        ($accessor:ident, $mut_accessor:ident, $ty:ty, $tyname:literal) => {{
            let input = tensor.data().$accessor().ok_or_else(|| {
                MinitensorError::internal_error(concat!("Failed to get ", $tyname, " slice"))
            })?;
            let output = result_data.$mut_accessor().ok_or_else(|| {
                MinitensorError::internal_error(concat!(
                    "Failed to get mutable ",
                    $tyname,
                    " slice"
                ))
            })?;
            let seed = if is_max { <$ty>::MIN } else { <$ty>::MAX };
            reduce_along_dim_par(
                input,
                output,
                &layout,
                seed,
                move |a: $ty, v: $ty| if is_max { a.max(v) } else { a.min(v) },
                |_| None,
            );
        }};
    }

    match tensor.dtype() {
        DataType::Float32 => float_arm!(
            as_f32_slice,
            as_f32_slice_mut,
            f32,
            "f32",
            max_row_f32,
            min_row_f32,
            max_columns_f32,
            min_columns_f32
        ),
        DataType::Float64 => float_arm!(
            as_f64_slice,
            as_f64_slice_mut,
            f64,
            "f64",
            max_row_f64,
            min_row_f64,
            max_columns_f64,
            min_columns_f64
        ),
        DataType::Int32 => int_arm!(as_i32_slice, as_i32_slice_mut, i32, "i32"),
        DataType::Int64 => int_arm!(as_i64_slice, as_i64_slice_mut, i64, "i64"),
        DataType::Bool => {
            let input = tensor
                .data()
                .as_bool_slice()
                .ok_or_else(|| MinitensorError::internal_error("Failed to get bool slice"))?;
            let output = result_data.as_bool_slice_mut().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get mutable bool slice")
            })?;
            // max is `any`, min is `all`; either way the first element equal to
            // the target value settles the column.
            reduce_along_dim_par(
                input,
                output,
                &layout,
                !is_max,
                move |a, v| if is_max { a | v } else { a & v },
                move |v| if v == is_max { Some(is_max) } else { None },
            );
        }
    }

    Ok(Tensor::new(
        Arc::new(result_data),
        layout.output_shape,
        tensor.dtype(),
        tensor.device(),
        tensor.requires_grad(),
    ))
}

pub(crate) fn max_along_dim(tensor: &Tensor, dim: usize, keepdim: bool) -> Result<Tensor> {
    extremum_along_dim(tensor, dim, keepdim, Extremum::Max)
}

pub(crate) fn min_along_dim(tensor: &Tensor, dim: usize, keepdim: bool) -> Result<Tensor> {
    extremum_along_dim(tensor, dim, keepdim, Extremum::Min)
}
