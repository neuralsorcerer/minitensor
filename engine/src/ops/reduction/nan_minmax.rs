// Copyright (c) Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

use super::sum_prod_impl::{column_band, fold_slab_with};
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
            const LANES: usize = $lanes;

            /// One chunk's extremum and whether it saw a real value. See
            /// `super::sum_prod`'s `float_extremum_all!` for why the body is
            /// written once and inlined into a second compilation.
            #[inline(always)]
            fn body(chunk: &[$ty]) -> ($ty, bool) {
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
            }

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

            let (value, has_real) = par_fold_chunks(
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

            result_slice[0] = if has_real { value } else { <$ty>::NAN };
            Ok(())
        }
    };
}

nan_extremum_all_entry!(
    nanmax_all_f32, as_f32_slice, as_f32_slice_mut, f32, "f32", f32::NEG_INFINITY, >, 16
);
nan_extremum_all_entry!(
    nanmax_all_f64, as_f64_slice, as_f64_slice_mut, f64, "f64", f64::NEG_INFINITY, >, 8
);
nan_extremum_all_entry!(
    nanmin_all_f32, as_f32_slice, as_f32_slice_mut, f32, "f32", f32::INFINITY, <, 16
);
nan_extremum_all_entry!(
    nanmin_all_f64, as_f64_slice, as_f64_slice_mut, f64, "f64", f64::INFINITY, <, 8
);
/// The first position in `data` where `test` holds, asked a block at a time.
///
/// The per-lane test accumulates into one flag rather than branching, which is
/// the shape the NaN flags in the folds above already use and vectorizes the
/// same way; the branch is once per block, and predicted not-taken until the
/// block that holds the answer. Only that block is then walked in order, which
/// is what makes the answer the *first* match rather than any of them.
///
/// Spelled as `iter().position(..)` this is a scalar loop, and for an extremum
/// sitting at the end of the data it cost more than the fold it was saving:
/// over one chunk, 0.667 ns an element against 0.214 for this.
///
/// The block is far wider than the folds above use, and deliberately: they
/// carry two accumulator arrays and stop wanting more chains once the
/// registers fill, while this carries one flag and is a plain scan, so it
/// keeps gaining. Sharing the fold's count is what left float64 `argmax` at
/// 2.5x its own `max` where float32 sat at 1.34x. With the target at the end,
/// so the whole scan runs:
///
/// ```text
///   lanes      f32       f64
///       8   0.147 ns  0.297 ns
///      16   0.078     0.288
///      32   0.082     0.138
///      64   0.049     0.092
/// ```
#[inline(always)]
fn locate_first<T: Copy>(data: &[T], test: impl Fn(T) -> bool) -> Option<usize> {
    const LANES: usize = 64;
    let mut base = 0usize;
    let (blocks, remainder) = data.as_chunks::<LANES>();
    for block in blocks {
        let mut hit = 0u32;
        for lane in 0..LANES {
            hit |= test(block[lane]) as u32;
        }
        if hit != 0 {
            for lane in 0..LANES {
                if test(block[lane]) {
                    return Some(base + lane);
                }
            }
        }
        base += LANES;
    }
    for (step, &v) in remainder.iter().enumerate() {
        if test(v) {
            return Some(base + step);
        }
    }
    None
}

/// The chunked fold both index reductions share: `(nan_at, best, best_at)`
/// over `data`, with positions absolute.
///
/// Two passes over each chunk rather than one, which is the opposite of what
/// it looks like it should want. Carrying the positions alongside the values
/// is a third accumulator array, and three arrays run out of registers where
/// two do not -- `argmax` sat at 0.424 ns an element against `max`'s 0.101 for
/// the same scan. Dropping them leaves pass one *as* `max`, at `max`'s width
/// and with `max`'s second compilation, and pass two is a search for a value
/// already known. A chunk is 32KB and is still in cache when the second pass
/// reads it.
///
/// Measured over one chunk, against carrying the positions, with the extremum
/// placed at a quarter, half and the end of the data -- the last being the
/// worst case for a search that stops at its first match:
///
/// ```text
///   0.407 -> 0.120     0.430 -> 0.157     0.510 -> 0.214
/// ```
///
/// The first match is also the tie-break the one-pass form spelled out: equal
/// values go to the lower index, which is what a forward search returns
/// without being asked. Signed zeros fall out of it too, `-0.0 == 0.0` being
/// true, so whichever came first is found first.
macro_rules! arg_lane_fold {
    ($data:expr, $ty:ty, $identity:expr, $better:tt, $lanes:expr, $nan:expr) => {{
        const LANES: usize = $lanes;

        /// Pass one: the extremum and whether a NaN is present, with no
        /// positions. This is `sum_prod`'s `float_extremum_all!` body.
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
                    if $nan {
                        nans[lane] |= (v != v) as u32;
                    }
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
                if $nan {
                    nan |= (v != v) as u32;
                }
            }
            (best, nan != 0)
        }

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

        par_fold_chunks(
            $data,
            MINMAX_CHUNK,
            (usize::MAX, $identity, usize::MAX),
            &|offset, chunk| {
                let (best, nan) = fold_chunk(chunk);
                let nan_at = if nan {
                    locate_first(chunk, |v: $ty| v != v)
                        .map_or(usize::MAX, |at| offset + at)
                } else {
                    usize::MAX
                };
                // A NaN anywhere outranks every value, so once this chunk has
                // one its extremum's position cannot be the answer and the
                // second pass is not run at all.
                let best_at = if nan_at != usize::MAX {
                    usize::MAX
                } else {
                    locate_first(chunk, |v: $ty| v == best)
                        .map_or(usize::MAX, |at| offset + at)
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
        )
    }};
}

/// `arg{min,max}_all_*`: the index of the global extremum.
///
/// Ties go to the lowest index. A NaN wins outright, and ties among NaNs go to
/// the lowest index too. An empty input answers 0.
///
/// The seed is the type's extreme value, which a real input can equal. That
/// costs nothing: the only way nothing beats the seed is that every element
/// equals it or is NaN, and then the answer is either the first NaN or -- ties
/// going to the lowest index -- position 0, which is what the fallback gives.
///
/// The one input that loses by locating the NaN rather than carrying it is an
/// array whose *first* element is NaN, where NumPy returns immediately and we
/// still scan: a hundredth of its speed on a pathological input, for twice its
/// speed on every ordinary one.
macro_rules! arg_extremum_all_lanes {
    ($name:ident, $accessor:ident, $ty:ty, $tyname:literal, $identity:expr, $better:tt, $lanes:expr, $nan:expr) => {
        pub(crate) fn $name(tensor: &Tensor, result_data: &mut TensorData) -> Result<()> {
            let data = tensor.data().$accessor().ok_or_else(|| {
                MinitensorError::internal_error(concat!("Failed to get ", $tyname, " slice"))
            })?;

            let (nan_at, _, best_at) = arg_lane_fold!(data, $ty, $identity, $better, $lanes, $nan);

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

/// `nanarg{min,max}_all_*`: the index of the global extremum among the
/// non-NaN entries.
///
/// This was `argmax(where(isnan(x), -inf, x))` behind an all-NaN check built
/// from `isnan`, a sum, an `eq` and an `any` -- seven passes over the data and
/// two full-size temporaries, 1.75ms over a million float32 where the plain
/// `argmax` underneath it takes 0.134.
///
/// None of that work was ever needed: a NaN satisfies no comparison, so the
/// fold above *already* skips it. Skipping is what `$nan = false` means, so
/// the nan-skipping index reduction is the same fold the integers run, and the
/// only thing left to decide is what to answer when nothing beat the seed.
///
/// That happens when every element is NaN or equal to the seed -- the type's
/// own infinity, which an input can hold for real. The non-NaN ones are then
/// all equal, so the answer is the first of them, and if there is none the
/// slice is all NaN and has no index to report. Both come from one
/// short-circuiting scan, which only runs in that degenerate case.
macro_rules! nanarg_extremum_all_lanes {
    ($name:ident, $accessor:ident, $ty:ty, $tyname:literal, $identity:expr, $better:tt, $lanes:expr, $what:literal) => {
        pub(crate) fn $name(tensor: &Tensor, result_data: &mut TensorData) -> Result<()> {
            let data = tensor.data().$accessor().ok_or_else(|| {
                MinitensorError::internal_error(concat!("Failed to get ", $tyname, " slice"))
            })?;

            let (_, _, best_at) = arg_lane_fold!(data, $ty, $identity, $better, $lanes, false);

            let index = match best_at {
                usize::MAX => data.iter().position(|v| !v.is_nan()).ok_or_else(|| {
                    MinitensorError::invalid_operation(concat!(
                        $what,
                        ": a slice of all-NaN values has no index to report"
                    ))
                })?,
                found => found,
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

arg_extremum_all_lanes!(argmax_all_f32, as_f32_slice, f32, "f32", f32::NEG_INFINITY, >, 16, true);
arg_extremum_all_lanes!(argmax_all_f64, as_f64_slice, f64, "f64", f64::NEG_INFINITY, >, 8, true);
arg_extremum_all_lanes!(argmax_all_i32, as_i32_slice, i32, "i32", i32::MIN, >, 16, false);
arg_extremum_all_lanes!(argmax_all_i64, as_i64_slice, i64, "i64", i64::MIN, >, 8, false);
arg_extremum_all_bool!(argmax_all_bool, true);

arg_extremum_all_lanes!(argmin_all_f32, as_f32_slice, f32, "f32", f32::INFINITY, <, 16, true);
arg_extremum_all_lanes!(argmin_all_f64, as_f64_slice, f64, "f64", f64::INFINITY, <, 8, true);
arg_extremum_all_lanes!(argmin_all_i32, as_i32_slice, i32, "i32", i32::MAX, <, 16, false);
arg_extremum_all_lanes!(argmin_all_i64, as_i64_slice, i64, "i64", i64::MAX, <, 8, false);
arg_extremum_all_bool!(argmin_all_bool, false);

nanarg_extremum_all_lanes!(
    nanargmax_all_f32, as_f32_slice, f32, "f32", f32::NEG_INFINITY, >, 16, "nanargmax"
);
nanarg_extremum_all_lanes!(
    nanargmax_all_f64, as_f64_slice, f64, "f64", f64::NEG_INFINITY, >, 8, "nanargmax"
);
nanarg_extremum_all_lanes!(
    nanargmin_all_f32, as_f32_slice, f32, "f32", f32::INFINITY, <, 16, "nanargmin"
);
nanarg_extremum_all_lanes!(
    nanargmin_all_f64, as_f64_slice, f64, "f64", f64::INFINITY, <, 8, "nanargmin"
);

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
///
/// With `$nan` false the NaN flag is not kept and a NaN is simply skipped --
/// no comparison against one is true -- which is the NaN-skipping extremum.
macro_rules! float_extremum_row {
    ($name:ident, $ty:ty, $identity:expr, $better:tt, $lanes:expr) => {
        float_extremum_row!($name, $ty, $identity, $better, $lanes, true);
    };
    ($name:ident, $ty:ty, $identity:expr, $better:tt, $lanes:expr, $nan:expr) => {
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
                    if $nan {
                        nans[lane] |= (v != v) as u32;
                    }
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
                if $nan {
                    nan |= (v != v) as u32;
                }
            }
            if nan != 0 { <$ty>::NAN } else { best }
        }
    };
}

float_extremum_row!(max_row_f32, f32, f32::NEG_INFINITY, >, 8);
float_extremum_row!(min_row_f32, f32, f32::INFINITY, <, 8);
float_extremum_row!(max_row_f64, f64, f64::NEG_INFINITY, >, 4);
float_extremum_row!(min_row_f64, f64, f64::INFINITY, <, 4);
float_extremum_row!(max_row_skip_f32, f32, f32::NEG_INFINITY, >, 8, false);
float_extremum_row!(min_row_skip_f32, f32, f32::INFINITY, <, 8, false);
float_extremum_row!(max_row_skip_f64, f64, f64::NEG_INFINITY, >, 4, false);
float_extremum_row!(min_row_skip_f64, f64, f64::INFINITY, <, 4, false);

/// The value and first index of the extremum of each contiguous row of
/// `input`, `dim_size` elements long: the indexed reduction along the last
/// axis, for a float dtype.
///
/// The generic walk this replaces carried the value, the index and a NaN
/// short-circuit through one branchy scalar loop, which could not vectorize:
/// `argmax` along the rows of a `(1024, 4096)` float32 matrix took 1.56ms
/// where NumPy takes 0.67. Here each row is the whole-tensor `argmax`'s two
/// passes in small -- the extremum by the lane fold `max` along a row uses,
/// then a search for the first element equal to it, the row still in cache.
/// The first match is the tie-break the walk had, lowest index, and signed
/// zeros compare equal so whichever came first is found. The value reported
/// is the element at that index, not the fold's, so it carries that element's
/// own sign of zero.
///
/// `nan_aware` skips NaN; an all-NaN row then reports NaN at index 0.
/// Otherwise a NaN wins and the first one is reported. A row nothing is found
/// in is empty and reports the seed at index 0, as the walk did.
macro_rules! float_arg_rows {
    ($name:ident, $ty:ty, $seed:expr, $better:tt, $fold:ident, $skip:ident) => {
        pub(crate) fn $name(
            input: &[$ty],
            values: &mut [$ty],
            indices: &mut [i64],
            dim_size: usize,
            nan_aware: bool,
        ) {
            // The extremum of one contiguous run, its position relative to
            // the run; `None` when the run holds nothing to report.
            let run_arg = |run: &[$ty]| -> Option<($ty, usize)> {
                let at = if nan_aware {
                    let best = $skip(run);
                    locate_first(run, |v: $ty| v == best)
                } else {
                    let best = $fold(run);
                    if best.is_nan() {
                        locate_first(run, |v: $ty| v != v)
                    } else {
                        locate_first(run, |v: $ty| v == best)
                    }
                };
                at.map(|at| (run[at], at))
            };
            let empty = if nan_aware { <$ty>::NAN } else { $seed };

            // A few very long rows: cut each into bands of `ARG_BAND_LEN`, so
            // the pool has something to share, and keep the first of the
            // band winners in band order -- the comparison the walk made
            // between elements, made between bands, so the same winner.
            if values.len() <= ARG_BAND_MAX_OUTPUTS && dim_size >= ARG_BAND_MIN_LEN {
                let bands = dim_size.div_ceil(ARG_BAND_LEN);
                let winners = par_map_indexed(values.len() * bands, &|task| {
                    let (row, band) = (task / bands, task % bands);
                    let from = row * dim_size + band * ARG_BAND_LEN;
                    let to = (from + ARG_BAND_LEN).min((row + 1) * dim_size);
                    run_arg(&input[from..to]).map(|(v, at)| (v, band * ARG_BAND_LEN + at))
                });
                let beats = |v: $ty, best: $ty| {
                    if nan_aware {
                        v $better best
                    } else {
                        (v != v && best == best) || v $better best
                    }
                };
                for (row, (value, index)) in values.iter_mut().zip(indices.iter_mut()).enumerate()
                {
                    let mut best: Option<($ty, usize)> = None;
                    for &found in winners[row * bands..(row + 1) * bands].iter().flatten() {
                        if best.is_none_or(|(b, _)| beats(found.0, b)) {
                            best = Some(found);
                        }
                    }
                    (*value, *index) = best.map_or((empty, 0), |(v, at)| (v, at as i64));
                }
                return;
            }

            par_out_chunks2(
                values,
                indices,
                outputs_per_task(dim_size),
                &|start, vals, idxs| {
                    for (offset, (value, index)) in vals.iter_mut().zip(idxs.iter_mut()).enumerate()
                    {
                        let first = (start + offset) * dim_size;
                        (*value, *index) = run_arg(&input[first..first + dim_size])
                            .map_or((empty, 0), |(v, at)| (v, at as i64));
                    }
                },
            );
        }
    };
}

float_arg_rows!(
    argmax_rows_f32,
    f32,
    f32::NEG_INFINITY,
    >,
    max_row_f32,
    max_row_skip_f32
);
float_arg_rows!(
    argmin_rows_f32,
    f32,
    f32::INFINITY,
    <,
    min_row_f32,
    min_row_skip_f32
);
float_arg_rows!(
    argmax_rows_f64,
    f64,
    f64::NEG_INFINITY,
    >,
    max_row_f64,
    max_row_skip_f64
);
float_arg_rows!(
    argmin_rows_f64,
    f64,
    f64::INFINITY,
    <,
    min_row_f64,
    min_row_skip_f64
);

/// The extremum of each column of `outer` slabs of `(len, inner)` rows: the
/// value-only reduction along any axis but the last, for a float dtype.
///
/// Slabs narrower than `BLOCKED_INNER_MIN` went one output at a time down the
/// generic strided walk, a branchy comparison per element `inner` apart: `max`
/// down the rows of a `(200000, 33)` float32 matrix took 12.7ms where `sum`
/// took 0.42, and NumPy 7.1. Wider ones had a column walk of their own that
/// measured 2-4x slower than this on float32. Both now go through the fold
/// `sum` uses for a slab -- rows streamed in
/// memory order, narrow ones several to an accumulator row, split across the
/// pool as `sum` splits them -- with a step that is two selects and so
/// vectorizes. The grouping cannot change the answer, since an extremum is
/// exact. It can change which of `-0.0` and `0.0` a column whose extremum is
/// zero reports, as the lane fold along a row already could.
///
/// A NaN propagates: once in a lane nothing displaces it, since no comparison
/// against it is true, and every NaN is reported as the canonical one.
fn slab_extremum<T>(
    input: &[T],
    output: &mut [T],
    layout: &DimReductionLayout,
    seed: T,
    is_max: bool,
) where
    T: num_traits::Float + Send + Sync,
{
    let inner = layout.inner;
    let slab = layout.dim_size * inner;
    let outer = output.len() / inner;
    let pick = move |a: T, v: T| {
        let wins = if is_max { v > a } else { v < a };
        if v.is_nan() || wins { v } else { a }
    };
    let fold = |source: &[T], spread: bool| {
        fold_slab_with(
            source,
            inner,
            seed,
            move |acc: &mut [T], values: &[T], _| {
                for (a, &v) in acc.iter_mut().zip(values) {
                    *a = pick(*a, v);
                }
            },
            pick,
            spread,
        )
    };
    let finish = |target: &mut [T], found: &[T]| {
        for (slot, &v) in target.iter_mut().zip(found) {
            *slot = if v.is_nan() { T::nan() } else { v };
        }
    };
    if slab == 0 {
        output.fill(seed);
    } else if outer < EXTREMUM_MIN_SLABS {
        for (source, target) in input.chunks_exact(slab).zip(output.chunks_exact_mut(inner)) {
            finish(target, &fold(source, true));
        }
    } else {
        let per_task = outputs_per_task(slab).div_ceil(inner).max(1);
        par_out_chunks(output, per_task * inner, &|start, chunk| {
            let first = start / inner;
            for (index, target) in chunk.chunks_exact_mut(inner).enumerate() {
                let at = (first + index) * slab;
                finish(target, &fold(&input[at..at + slab], false));
            }
        });
    }
}

/// Fewer slabs than this are each spread across the pool; `sum`'s threshold.
const EXTREMUM_MIN_SLABS: usize = 4;

/// The value and first index of the extremum of each column of `outer`
/// slabs of `(len, inner)` rows: the indexed reduction along any axis but the
/// last, for a float dtype.
///
/// The walks this replaces carried the value, the index and the NaN rule
/// through one branchy comparison per element -- down each output's column
/// `inner` apart when the slab was narrow, which put `max(dim=0)` of a
/// `(200000, 33)` float32 matrix at 8.8ms against `amax`'s 0.56 and NumPy's
/// 27.
///
/// Now the slab is cut into blocks of rows, and each block's column extrema
/// are taken by [`slab_extremum`]'s fold (a NaN-skipping one for
/// `nan_aware`) -- one vectorized pass over the input, split across the pool
/// when the slab is. The blocks' extrema give each column's target, and the
/// first block whose extremum matches it holds the column's first match, so
/// finding the index is a search of that block's rows for that one column --
/// a few thousand reads for the whole slab, not a second pass over it. A
/// search that knew only the targets had to walk the rows until each column
/// turned up, which on random data is half the slab at best, and cost more
/// than the fold.
///
/// "Matches" is equality, or NaN-ness when NaN propagated. The first match is
/// the tie-break the walks had, lowest index, and signed zeros compare equal,
/// so whichever came first is found; the value reported is the element at
/// that index. Under `nan_aware` an all-NaN block's extremum is the seed and
/// can equal a real target, so a block that turns out to hold no match is
/// passed over; a column with no match anywhere is all NaN, reported as NaN
/// at index 0.
pub(crate) fn slab_arg_extremum<T>(
    input: &[T],
    values: &mut [T],
    indices: &mut [i64],
    layout: &DimReductionLayout,
    is_max: bool,
    nan_aware: bool,
) where
    T: num_traits::Float + Send + Sync,
{
    let inner = layout.inner;
    let rows = layout.dim_size;
    let slab = rows * inner;
    let outer = values.len() / inner;
    let seed = if is_max {
        T::neg_infinity()
    } else {
        T::infinity()
    };
    let pick = move |a: T, v: T| {
        let wins = if is_max { v > a } else { v < a };
        if (!nan_aware && v.is_nan()) || wins {
            v
        } else {
            a
        }
    };
    let block = ARG_BLOCK_ELEMS.div_ceil(inner).max(ARG_BLOCK_MIN_ROWS);
    let blocks = rows.div_ceil(block);

    let one = |source: &[T], vals: &mut [T], idxs: &mut [i64], spread: bool| {
        let extremum = |b: usize| {
            fold_slab_with(
                &source[b * block * inner..((b + 1) * block).min(rows) * inner],
                inner,
                seed,
                move |acc: &mut [T], values: &[T], _| {
                    for (a, &v) in acc.iter_mut().zip(values) {
                        *a = pick(*a, v);
                    }
                },
                pick,
                false,
            )
        };
        let extrema: Vec<Vec<T>> = if spread {
            par_map_indexed(blocks, &extremum)
        } else {
            (0..blocks).map(extremum).collect()
        };
        let mut targets = extrema[0].clone();
        for part in &extrema[1..] {
            for (t, &v) in targets.iter_mut().zip(part) {
                *t = pick(*t, v);
            }
        }
        let locate = |start: usize, vals: &mut [T], idxs: &mut [i64]| {
            for (offset, (value, index)) in vals.iter_mut().zip(idxs.iter_mut()).enumerate() {
                let c = start + offset;
                let target = targets[c];
                let matches = |v: T| v == target || (target.is_nan() && v.is_nan());
                let found = (0..blocks)
                    .filter(|&b| matches(extrema[b][c]))
                    .find_map(|b| {
                        (b * block..((b + 1) * block).min(rows))
                            .find(|&r| matches(source[r * inner + c]))
                    });
                (*value, *index) = match found {
                    Some(r) => (source[r * inner + c], r as i64),
                    None => (T::nan(), 0),
                };
            }
        };
        if spread {
            par_out_chunks2(vals, idxs, column_band(inner), &locate);
        } else {
            locate(0, vals, idxs);
        }
    };
    if slab == 0 {
        values.fill(seed);
        indices.fill(0);
    } else if outer < EXTREMUM_MIN_SLABS {
        for ((source, vals), idxs) in input
            .chunks_exact(slab)
            .zip(values.chunks_exact_mut(inner))
            .zip(indices.chunks_exact_mut(inner))
        {
            one(source, vals, idxs, true);
        }
    } else {
        let per_task = outputs_per_task(slab).div_ceil(inner).max(1);
        par_out_chunks2(values, indices, per_task * inner, &|start, vals, idxs| {
            let first = start / inner;
            for (index, (v, i)) in vals
                .chunks_exact_mut(inner)
                .zip(idxs.chunks_exact_mut(inner))
                .enumerate()
            {
                let at = (first + index) * slab;
                one(&input[at..at + slab], v, i, false);
            }
        });
    }
}

/// Elements in one block of [`slab_arg_extremum`], and the fewest rows one
/// may hold. Fixed, so the blocks follow from the shape alone -- though here
/// they could not change the answer anyway, an extremum being exact.
const ARG_BLOCK_ELEMS: usize = 8192;
const ARG_BLOCK_MIN_ROWS: usize = 16;

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
        ($accessor:ident, $mut_accessor:ident, $ty:ty, $tyname:literal, $row_max:ident, $row_min:ident) => {{
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
            } else if layout.inner > 1 {
                slab_extremum(input, output, &layout, seed, is_max);
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
            min_row_f32
        ),
        DataType::Float64 => float_arm!(
            as_f64_slice,
            as_f64_slice_mut,
            f64,
            "f64",
            max_row_f64,
            min_row_f64
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
