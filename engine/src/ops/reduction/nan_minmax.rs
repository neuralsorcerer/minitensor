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
        return Err(MinitensorError::index_error(dim as isize, 0, tensor.ndim()));
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
/// comparison, so the first winner keeps its index (matches NumPy/PyTorch
/// argmax/argmin tie-breaking); `short(val)` returning `Some(v)` finalizes the
/// output early with value `v` at the current index (NaN propagation, boolean
/// any/all short-circuit).
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
/// Returns NaN when every element is NaN, matching NumPy's `nanmin`/`nanmax`.
/// The four dtype-specific versions this replaces each carried a
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
/// Ties go to the lowest index. A NaN wins outright, matching
/// `torch.argmax`/`argmin`; ties among NaNs also go to the lowest index. As
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

/// Reduce `dim` to its extremum, without reporting where it was found.
///
/// The value-only forms of `min` and `max` differed only in their seed and
/// their fold, exactly as the value-and-index forms in `minmax_indices` did.
/// A NaN anywhere in a float slice short-circuits the whole column, matching
/// `torch.min`/`torch.max`; bool short-circuits on the first `true` (max) or
/// `false` (min).
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
            let seed = if is_max {
                <$ty>::NEG_INFINITY
            } else {
                <$ty>::INFINITY
            };
            reduce_along_dim_par(
                input,
                output,
                &layout,
                seed,
                // NaN-propagating on its own rather than relying on the
                // short-circuit below: the blocked path in
                // `reduce_along_dim_par` walks memory in order and has no
                // per-element early exit to lean on. `a.max(v)` would be wrong
                // here -- it returns the *non*-NaN operand.
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
        DataType::Float32 => float_arm!(as_f32_slice, as_f32_slice_mut, f32, "f32"),
        DataType::Float64 => float_arm!(as_f64_slice, as_f64_slice_mut, f64, "f64"),
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
