// Copyright (c) Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

use super::*;
use crate::ops::map::{
    outputs_per_task, par_fold_chunks, par_map_indexed, par_out_chunks, par_out_chunks2,
    reduction_band,
};
use crate::ops::util::check_dim;
use crate::{
    error::{MinitensorError, Result},
    tensor::{DataType, Shape, Tensor, TensorData},
};
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
    check_dim(dim, tensor.ndim())?;

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

/// Most outputs a reduction can have before splitting *them* fills the machine
/// on its own.
///
/// Below this many, the split has to be over the reduced axis instead: a
/// reduction to a single value has one output however long the axis is, so
/// `max(dim=0)` of a two-million-element vector ran on one core at 2.6ms where
/// the whole-tensor form does the same work in 0.2.
const ARG_BAND_MAX_OUTPUTS: usize = 64;

/// Shortest reduced axis worth cutting into bands. Below it the whole
/// reduction is a few microseconds and the partials cost more than they save.
const ARG_BAND_MIN_LEN: usize = 1 << 15;

/// How much of the reduced axis one band covers.
///
/// From the *shape*, never from the thread pool. The combination below is a
/// comparison rather than an accumulation, so the answer does not depend on how
/// the axis was cut -- but every partition in this library derives its
/// boundaries from the shape, and keeping the rule uniform is what makes that
/// easy to check.
const ARG_BAND_LEN: usize = 1 << 14;

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
        // One outer position's columns, or a band of them when there are not
        // enough outer positions to fill the pool. The two used to be written
        // separately; they are the same loop, and `start % inner` is the only
        // thing the band case adds -- zero whenever the band is the full width.
        par_out_chunks(output, reduction_band(outer, inner), &|start, cols| {
            let base = (start / inner) * outer_stride + start % inner;
            let width = cols.len();
            cols.fill(init);
            for step in 0..dim_size {
                let slab = &input[base + step * inner..][..width];
                for (acc, &value) in cols.iter_mut().zip(slab) {
                    *acc = combine(*acc, value);
                }
            }
        });
        return;
    }

    par_out_chunks(output, outputs_per_task(dim_size), &|start, chunk| {
        for (offset, out) in chunk.iter_mut().enumerate() {
            let out_idx = start + offset;
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
        }
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
        let band = reduction_band(outer, inner);
        par_out_chunks2(values, indices, band, &|flat, vals, idxs| {
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

    // Too few outputs to split, and a long axis: band the axis instead. Each
    // band takes its own extremum and they are combined in band order, so the
    // earliest position still wins a tie and the answer is the walk's own.
    //
    // `short` is deliberately not used here, for the same reason the blocked
    // path above does not use it: the comparison the callers pass already folds
    // NaN in, which reproduces the index the break-on-first-NaN short circuit
    // produced.
    if values.len() <= ARG_BAND_MAX_OUTPUTS && dim_size >= ARG_BAND_MIN_LEN {
        let bands = dim_size.div_ceil(ARG_BAND_LEN);
        let mut partial: Vec<(T, usize)> = vec![(init, 0); values.len() * bands];
        // One band per chunk: rayon splits the range itself, and a band is
        // already `ARG_BAND_LEN` reads of work. Handing out whole *outputs*
        // instead would be one chunk again whenever there is one output, which
        // is the case this path exists for.
        par_out_chunks(&mut partial, 1, &|start, chunk| {
            for (offset, slot) in chunk.iter_mut().enumerate() {
                let flat = start + offset;
                let out_idx = flat / bands;
                let band = flat % bands;
                let o = out_idx / inner;
                let r = out_idx % inner;
                let from = band * ARG_BAND_LEN;
                let to = ((band + 1) * ARG_BAND_LEN).min(dim_size);
                // Seeded at the band's first position, so a band nothing beats
                // still names somewhere inside itself -- and the first band's
                // seed is position zero, which is what the walk reports when
                // nothing beats the initial value.
                let mut best = init;
                let mut best_i = from;
                let mut idx = o * outer_stride + r + from * inner;
                for d in from..to {
                    let val = input[idx];
                    if better(val, best) {
                        best = val;
                        best_i = d;
                    }
                    idx += inner;
                }
                *slot = (best, best_i);
            }
        });
        for (lane, (vout, iout)) in values.iter_mut().zip(indices.iter_mut()).enumerate() {
            let (mut best, mut best_i) = partial[lane * bands];
            for band in 1..bands {
                let (value, at) = partial[lane * bands + band];
                if better(value, best) {
                    best = value;
                    best_i = at;
                }
            }
            *vout = best;
            *iout = best_i as i64;
        }
        return;
    }

    par_out_chunks2(
        values,
        indices,
        outputs_per_task(dim_size),
        &|start, vchunk, ichunk| {
            for (offset, (vout, iout)) in vchunk.iter_mut().zip(ichunk.iter_mut()).enumerate() {
                let out_idx = start + offset;
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
            }
        },
    );
}

/// `nan{min,max}_all_{f32,f64}`: the whole-tensor NaN-skipping extremum.
///
/// The lane-blocked shape of [`super::sum_prod`]'s `float_extremum_all!`, with
/// the flag inverted. That one records whether a NaN was *seen*, so it can
/// propagate one; this one records whether anything that was *not* a NaN was
/// seen, so it can answer NaN only when nothing was. Neither needs a NaN test
/// in the comparison: a NaN is greater than nothing and less than nothing, so
/// `v > best` passes over it without being asked to.
///
/// Written generically over `T: Float` it did not vectorise -- one accumulator
/// serialises the compare-and-select, and the trait call blocks the lane
/// blocking that fixes it. On a million float32 that read 0.44 ms where the
/// same scan without the NaN test reads 0.11.
///
/// The identity is a value the data can hold, and that is what the flag is
/// for: a slice of nothing but `-inf` answers `-inf`, and a slice of nothing
/// but NaN answers NaN, and the extremum alone cannot tell those apart.
macro_rules! nan_extremum_all_entry {
    ($name:ident, $accessor:ident, $accessor_mut:ident, $ty:ty, $tyname:literal, $identity:expr, $better:tt, $lanes:expr) => {
        pub(crate) fn $name(tensor: &Tensor, result_data: &mut TensorData) -> Result<()> {
            let data = tensor.data().$accessor().ok_or_else(|| {
                MinitensorError::internal_error(concat!("Failed to get ", $tyname, " slice"))
            })?;

            let (value, has_real) = par_fold_chunks(
                data,
                MINMAX_CHUNK,
                ($identity, false),
                &|_, chunk| {
                    const LANES: usize = $lanes;
                    let mut bests = [$identity; LANES];
                    let mut reals = [0u32; LANES];
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
                            reals[lane] |= (v == v) as u32;
                        }
                    }
                    let mut best: $ty = $identity;
                    let mut real = 0u32;
                    for lane in 0..LANES {
                        if bests[lane] $better best {
                            best = bests[lane];
                        }
                        real |= reals[lane];
                    }
                    for &v in blocks.remainder() {
                        if v $better best {
                            best = v;
                        }
                        real |= (v == v) as u32;
                    }
                    (best, real != 0)
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

            result_slice[0] = if has_real { value } else { <$ty>::NAN };
            Ok(())
        }
    };
}

nan_extremum_all_entry!(
    nanmax_all_f32, as_f32_slice, as_f32_slice_mut, f32, "f32", f32::NEG_INFINITY, >, 8
);
nan_extremum_all_entry!(
    nanmax_all_f64, as_f64_slice, as_f64_slice_mut, f64, "f64", f64::NEG_INFINITY, >, 4
);
nan_extremum_all_entry!(
    nanmin_all_f32, as_f32_slice, as_f32_slice_mut, f32, "f32", f32::INFINITY, <, 8
);
nan_extremum_all_entry!(
    nanmin_all_f64, as_f64_slice, as_f64_slice_mut, f64, "f64", f64::INFINITY, <, 4
);
/// `arg{min,max}_all_*`: the index of the global extremum.
///
/// Ties go to the lowest index. A NaN wins outright, and ties among NaNs go to
/// the lowest index too. An empty input answers 0.
///
/// The lane-blocked shape of the extremum above, carrying a position beside
/// each lane's running best. Written with one accumulator and an `Option` it
/// did not vectorise: `argmax` over a million float32 took 0.34ms where `max`
/// over the same data -- the same scan without the index -- takes 0.065.
/// Lane-blocked it takes 0.134, against NumPy's 0.145.
///
/// Four things the lanes make delicate, and each is why the code below is
/// shaped the way it is:
///
/// * Lane `l` walks positions `l, l + LANES, l + 2 * LANES, ...`, so a later
///   lane holds *earlier* positions than an earlier lane's second block. The
///   comparison inside a lane can stay strict -- it meets its own positions in
///   order, so an equal value never displaces the one it already has -- but
///   folding the lanes together has to break ties on the index explicitly.
/// * The remainder comes after every lane block, so its positions are later
///   than all of them, and a strict comparison is again all it needs.
/// * A NaN satisfies no comparison, so it can never become a lane's best and
///   the tie-break never sees one. That is why the tie-break reads `==` rather
///   than a negated comparison: with no NaN on either side, "neither is
///   better" *is* equality, and the two candidates for a tie -- equal values,
///   and `+0.0` against `-0.0` -- both want the lower index.
/// * The hot loop only *flags* NaN, per lane and branch-free, the way the
///   value fold in `sum_prod` does; a chunk that raised its flag then locates
///   the first NaN with a short-circuiting scan. Carrying the position instead
///   costs more than everything else put together: it needs a `usize` min and
///   select per element, which for f32 is two 64-bit-lane vectors against the
///   one the values occupy. Carrying it left f32 `argmax` at 0.297ms over a
///   million elements; flagging and locating brought it to 0.134, and f64 from
///   0.363 to 0.199. The one input that loses by it is an array whose *first*
///   element is NaN, where NumPy returns immediately and we still scan: a
///   hundredth of NumPy's speed on a pathological input, for twice its speed
///   on every ordinary one.
///
/// Positions inside the loop are `u32` and relative to the chunk, which
/// `par_fold_chunks` caps at `MINMAX_CHUNK` -- 8192, so they cannot overflow.
/// That is what makes the index lanes the same width as the value lanes for a
/// 32-bit type, rather than twice it.
///
/// The seed is the type's extreme value, which a real input can equal. That
/// costs nothing here: the only way nothing beats the seed is that every
/// element equals it or is NaN, and then the answer is either the first NaN or
/// -- ties going to the lowest index -- position 0, which is what the fallback
/// gives.
macro_rules! arg_extremum_all_lanes {
    ($name:ident, $accessor:ident, $ty:ty, $tyname:literal, $identity:expr, $better:tt, $lanes:expr, $nan:expr) => {
        pub(crate) fn $name(tensor: &Tensor, result_data: &mut TensorData) -> Result<()> {
            let data = tensor.data().$accessor().ok_or_else(|| {
                MinitensorError::internal_error(concat!("Failed to get ", $tyname, " slice"))
            })?;

            let (nan_at, _, best_at) = par_fold_chunks(
                data,
                MINMAX_CHUNK,
                (usize::MAX, $identity, usize::MAX),
                &|offset, chunk| {
                    const LANES: usize = $lanes;
                    let mut bests = [$identity; LANES];
                    let mut wheres = [u32::MAX; LANES];
                    let mut nans = [0u32; LANES];
                    let mut blocks = chunk.chunks_exact(LANES);
                    let mut base = 0u32;
                    for block in &mut blocks {
                        for lane in 0..LANES {
                            let v = block[lane];
                            if v $better bests[lane] {
                                bests[lane] = v;
                                wheres[lane] = base + lane as u32;
                            }
                            if $nan {
                                nans[lane] |= (v != v) as u32;
                            }
                        }
                        base += LANES as u32;
                    }

                    let mut best: $ty = $identity;
                    let mut best_at = u32::MAX;
                    let mut nan = 0u32;
                    for lane in 0..LANES {
                        nan |= nans[lane];
                        if bests[lane] $better best
                            || (bests[lane] == best && wheres[lane] < best_at)
                        {
                            best = bests[lane];
                            best_at = wheres[lane];
                        }
                    }
                    for (step, &v) in blocks.remainder().iter().enumerate() {
                        let at = base + step as u32;
                        if v $better best {
                            best = v;
                            best_at = at;
                        }
                        if $nan {
                            nan |= (v != v) as u32;
                        }
                    }

                    let nan_at = if nan != 0 {
                        chunk
                            .iter()
                            .position(|v| v != v)
                            .map_or(usize::MAX, |at| offset + at)
                    } else {
                        usize::MAX
                    };
                    let best_at = if best_at == u32::MAX {
                        usize::MAX
                    } else {
                        offset + best_at as usize
                    };
                    (nan_at, best, best_at)
                },
                &|a, b| {
                    let nan = a.0.min(b.0);
                    if b.1 $better a.1 || (a.1 == b.1 && b.2 < a.2) {
                        (nan, b.1, b.2)
                    } else {
                        (nan, a.1, a.2)
                    }
                },
            );

            let index = if nan_at != usize::MAX {
                nan_at
            } else if best_at != usize::MAX {
                best_at
            } else {
                0
            };
            write_index(result_data, index)
        }
    };
}

#[inline]
fn write_index(result_data: &mut TensorData, index: usize) -> Result<()> {
    let slot = result_data
        .as_i64_slice_mut()
        .ok_or_else(|| MinitensorError::internal_error("Failed to get mutable i64 slice"))?;
    slot[0] = index as i64;
    Ok(())
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

arg_extremum_all_lanes!(argmax_all_f32, as_f32_slice, f32, "f32", f32::NEG_INFINITY, >, 8, true);
arg_extremum_all_lanes!(argmax_all_f64, as_f64_slice, f64, "f64", f64::NEG_INFINITY, >, 4, true);
arg_extremum_all_lanes!(argmax_all_i32, as_i32_slice, i32, "i32", i32::MIN, >, 8, false);
arg_extremum_all_lanes!(argmax_all_i64, as_i64_slice, i64, "i64", i64::MIN, >, 4, false);
arg_extremum_all_bool!(argmax_all_bool, true);

arg_extremum_all_lanes!(argmin_all_f32, as_f32_slice, f32, "f32", f32::INFINITY, <, 8, true);
arg_extremum_all_lanes!(argmin_all_f64, as_f64_slice, f64, "f64", f64::INFINITY, <, 4, true);
arg_extremum_all_lanes!(argmin_all_i32, as_i32_slice, i32, "i32", i32::MAX, <, 8, false);
arg_extremum_all_lanes!(argmin_all_i64, as_i64_slice, i64, "i64", i64::MAX, <, 4, false);
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
                par_out_chunks(output, outputs_per_task(dim_size), &|start, chunk| {
                    for (offset, out) in chunk.iter_mut().enumerate() {
                        let row = &input[(start + offset) * dim_size..][..dim_size];
                        *out = if is_max { $row_max(row) } else { $row_min(row) };
                    }
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
                    par_out_chunks(output, inner, &|start, row| {
                        let base = (start / inner) * outer_stride;
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
                    let partials: Vec<Vec<$ty>> = par_map_indexed(dim_size.div_ceil(band), &|b| {
                        let mut acc = vec![seed; inner];
                        let from = b * band;
                        let to = ((b + 1) * band).min(dim_size);
                        if is_max {
                            $col_max(input, 0, 0, from, to, inner, &mut acc);
                        } else {
                            $col_min(input, 0, 0, from, to, inner, &mut acc);
                        }
                        acc
                    });
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
