// Copyright (c) Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

//! Float sums and means rounded once.
//!
//! A float64 sum added in lanes and folded pairwise is accurate, but it is not
//! the float64 nearest the exact sum: on ordinary data about half the sums
//! were off, by up to tens of ulps where the values cancel, and the mean
//! divided that rounded sum and rounded again.
//!
//! Each lane here carries its sum with the exact error of every addition --
//! a two-sum's -- added up beside it, and the absolute errors added up too,
//! which bounds what adding the errors in float64 left out. Lanes and blocks
//! merge the same way, and the result is the float64 nearest the carried
//! total when the bound places it clear of every halfway point, which is
//! nearly always. Otherwise the values are summed exactly in whole 64-bit
//! limbs and rounded from there. The answer does not depend on the order of
//! the additions, and so not on how the work was split.
//!
//! Float32 values are read into float64, which holds each exactly, and the
//! total is rounded to float32 when the bound keeps it clear of every halfway
//! point between two float32s. Summed in float64 and narrowed, a float32 sum
//! whose values cancel across a wide range lost what lay below float64's
//! last bit, and one whose float64 total fell on a float32 halfway point
//! rounded twice: `1 + 2^-24 + 2^-60` came out `1`.

use super::close::add_shifted;
use super::rounded_once::{Exact, Target, product_error, rounded, two_sum};
use crate::ops::util::pairwise_fold;

/// Lanes a run is added in, and values per block a task takes.
const LANES: usize = 8;
const BLOCK: usize = 8192;

/// Just under one half; see `evenly_spaced`.
const SHY_OF_HALF: f64 = 0.5 - 1.0 / (1u64 << 40) as f64;

/// An element a sum reads: a float32 or a float64, each exact in float64.
pub(crate) trait Element: Copy + Send + Sync + 'static {
    fn wide(self) -> f64;
    /// [`block_lanes`] in the widest build the CPU runs.
    fn block(values: &[Self]) -> Carried;
    /// [`plain_lanes`] in the widest build the CPU runs.
    fn plain_block(values: &[Self]) -> (f64, f64) {
        plain_lanes(values)
    }
}

impl Element for f64 {
    #[inline(always)]
    fn wide(self) -> f64 {
        self
    }

    fn block(values: &[Self]) -> Carried {
        #[cfg(target_arch = "x86_64")]
        if crate::ops::simd::simd_capabilities().avx {
            // SAFETY: avx was detected on this CPU.
            return unsafe { block_avx(values) };
        }
        block_lanes(values)
    }
}

impl Element for f32 {
    #[inline(always)]
    fn wide(self) -> f64 {
        f64::from(self)
    }

    fn block(values: &[Self]) -> Carried {
        #[cfg(target_arch = "x86_64")]
        if crate::ops::simd::simd_capabilities().avx {
            // SAFETY: avx was detected on this CPU.
            return unsafe { block_avx_f32(values) };
        }
        block_lanes(values)
    }

    fn plain_block(values: &[Self]) -> (f64, f64) {
        #[cfg(target_arch = "x86_64")]
        if crate::ops::simd::simd_capabilities().avx {
            // SAFETY: avx was detected on this CPU.
            return unsafe { plain_avx_f32(values) };
        }
        plain_lanes(values)
    }
}

/// A float dtype a total is rounded to.
pub(crate) trait Output: Target + Send + Sync {
    /// Whether a plain float64 sum, with a bound on its rounding, usually
    /// settles the answer: it does for float32, 29 bits narrower, and is
    /// never close enough for float64.
    const PLAIN: bool;
    /// The value nearest every number within `bound` of `value + rest`, when
    /// that is one value; `value + rest` is a two-sum's pair.
    fn settle(value: f64, rest: f64, bound: f64) -> Option<Self>;
}

impl Output for f64 {
    const PLAIN: bool = false;

    fn settle(value: f64, rest: f64, bound: f64) -> Option<Self> {
        if bound == 0.0 && rest == 0.0 {
            return value.is_finite().then_some(value);
        }
        // The gap to the float64 on the side towards zero, the smaller of the
        // two; a zero or a non-finite value leaves it NaN and the test false.
        let inward = (value - f64::from_bits(value.to_bits().wrapping_sub(1))).abs();
        (rest.abs() + bound < inward * SHY_OF_HALF && value != 0.0).then_some(value)
    }
}

impl Output for f32 {
    const PLAIN: bool = true;

    fn settle(value: f64, rest: f64, bound: f64) -> Option<Self> {
        let narrow = value as f32;
        if bound == 0.0 && rest == 0.0 {
            // `value` is the total, rounded once by the conversion.
            return value.is_finite().then_some(narrow);
        }
        if !narrow.is_finite() || narrow.abs() == f32::MAX {
            return None;
        }
        // The halfway points either side of `narrow`, which are float64s, and
        // the room the total has before each: exact differences of nearby
        // float64s, kept a hair short to stay clear of their own rounding.
        let back = f64::from(narrow);
        let above = (back + f64::from(narrow.next_up())) / 2.0;
        let below = (back + f64::from(narrow.next_down())) / 2.0;
        let room = |gap: f64| gap * (1.0 - 2f64.powi(-40));
        (rest + bound < room(above - value) && bound - rest < room(value - below)).then_some(narrow)
    }
}

/// A float64 total carried as `sum + error`, within `slack` of the exact
/// total.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Carried {
    sum: f64,
    error: f64,
    slack: f64,
}

impl Carried {
    /// The total of two. Their sums are added exactly by a two-sum; the
    /// errors are added in float64, twice, which is within `2.01u` of their
    /// magnitudes, and `slack` takes four times that.
    fn merge(self, other: Self) -> Self {
        let (sum, rounding) = two_sum(self.sum, other.sum);
        let error = (self.error + other.error) + rounding;
        let magnitude = self.error.abs() + other.error.abs() + rounding.abs();
        let slack =
            (self.slack + other.slack) * (1.0 + 2f64.powi(-50)) + magnitude * 2f64.powi(-51);
        Self { sum, error, slack }
    }

    /// The value nearest the exact total over `count`, when the bound says
    /// which it is.
    ///
    /// Divided as `evenly_spaced` divides a numerator: the quotient, its
    /// exact remainder, and the tail with the error added in.
    fn divided<T: Output>(self, count: usize) -> Option<T> {
        if self.sum == 0.0 && self.error == 0.0 && self.slack == 0.0 {
            // The values cancel exactly, or are all zeros.
            return Some(T::narrow(0.0));
        }
        let count = count as f64;
        let quotient = self.sum / count;
        let product = quotient * count;
        // `product` is within a rounding of `sum`, so the first difference
        // is exact and the remainder rounds once.
        let remainder = (self.sum - product) - product_error(quotient, count, product);
        let tail = (remainder + self.error) / count;
        let (value, rest) = two_sum(quotient, tail);
        let bound = self.slack / count * (1.0 + 2f64.powi(-50)) + tail.abs() * 2f64.powi(-50);
        // Clear of underflow in the product's split, and of a zero or a value
        // that is not finite.
        let ordinary = value.abs() >= 2f64.powi(-900) && value.abs() <= 2f64.powi(900);
        if !ordinary {
            return None;
        }
        T::settle(value, rest, bound)
    }

    /// The value nearest the exact total, when the bound says which it is.
    fn rounded<T: Output>(self) -> Option<T> {
        let (value, rest) = two_sum(self.sum, self.error);
        T::settle(value, rest, self.slack)
    }
}

/// One block's values, added in [`LANES`] lanes with each addition's exact
/// error carried beside it.
///
/// A lane's errors, added in float64 over `m` additions, are within
/// `(m - 1)u / (1 - (m - 1)u)` of the sum of their magnitudes, which the lane
/// adds up as well; `2^-52 · m` is twice that.
#[inline(always)]
fn block_lanes<I: Element>(values: &[I]) -> Carried {
    let (mut sums, mut errors, mut magnitudes) = ([0f64; LANES], [0f64; LANES], [0f64; LANES]);
    let (chunks, rest) = values.as_chunks::<LANES>();
    for chunk in chunks {
        for lane in 0..LANES {
            let (sum, error) = two_sum(sums[lane], chunk[lane].wide());
            sums[lane] = sum;
            errors[lane] += error;
            magnitudes[lane] += error.abs();
        }
    }
    finish(
        sums,
        errors,
        magnitudes,
        rest.iter().map(|value| value.wide()),
        values.len(),
    )
}

/// A block's lanes, with the values left over added to lane 0, merged into
/// one carried total.
#[inline(always)]
fn finish(
    mut sums: [f64; LANES],
    mut errors: [f64; LANES],
    mut magnitudes: [f64; LANES],
    rest: impl Iterator<Item = f64>,
    len: usize,
) -> Carried {
    // The few left over go to lane 0.
    for value in rest {
        let (sum, error) = two_sum(sums[0], value);
        sums[0] = sum;
        errors[0] += error;
        magnitudes[0] += error.abs();
    }
    let additions = (len / LANES + LANES) as f64;
    let mut total = Carried::default();
    for lane in 0..LANES {
        total = total.merge(Carried {
            sum: sums[lane],
            error: errors[lane],
            slack: magnitudes[lane] * additions * 2f64.powi(-52),
        });
    }
    total
}

/// [`block_lanes`] in 256-bit registers, for a CPU that has them: the same
/// lanes, additions and order, written out because the compiler kept the
/// two-sum's four steps scalar.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
fn block_avx(values: &[f64]) -> Carried {
    use std::arch::x86_64::_mm256_loadu_pd;
    block_avx_with(values, |chunk, half| {
        // SAFETY: four values from `4 * half` lie inside the chunk.
        unsafe { _mm256_loadu_pd(chunk.as_ptr().add(4 * half)) }
    })
}

/// [`block_avx`] for float32 values, each four read and widened at once.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
fn block_avx_f32(values: &[f32]) -> Carried {
    use std::arch::x86_64::{_mm_loadu_ps, _mm256_cvtps_pd};
    block_avx_with(values, |chunk, half| {
        // SAFETY: four values from `4 * half` lie inside the chunk.
        _mm256_cvtps_pd(unsafe { _mm_loadu_ps(chunk.as_ptr().add(4 * half)) })
    })
}

/// The lanes of [`block_lanes`] in two 256-bit registers each, `load`
/// giving half a chunk's values in float64.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
fn block_avx_with<I: Element>(
    values: &[I],
    load: impl Fn(&[I; LANES], usize) -> std::arch::x86_64::__m256d,
) -> Carried {
    use std::arch::x86_64::{
        __m256d, _mm256_add_pd, _mm256_andnot_pd, _mm256_set1_pd, _mm256_setzero_pd,
        _mm256_storeu_pd, _mm256_sub_pd,
    };
    let sign = _mm256_set1_pd(-0.0);
    let (mut sums, mut errors, mut magnitudes) = (
        [_mm256_setzero_pd(); 2],
        [_mm256_setzero_pd(); 2],
        [_mm256_setzero_pd(); 2],
    );
    let (chunks, rest) = values.as_chunks::<LANES>();
    for chunk in chunks {
        for half in 0..2 {
            let value: __m256d = load(chunk, half);
            let sum = _mm256_add_pd(sums[half], value);
            let virtual_value = _mm256_sub_pd(sum, sums[half]);
            let virtual_sum = _mm256_sub_pd(sum, virtual_value);
            let error = _mm256_add_pd(
                _mm256_sub_pd(sums[half], virtual_sum),
                _mm256_sub_pd(value, virtual_value),
            );
            sums[half] = sum;
            errors[half] = _mm256_add_pd(errors[half], error);
            magnitudes[half] = _mm256_add_pd(magnitudes[half], _mm256_andnot_pd(sign, error));
        }
    }
    let unpack = |vectors: [__m256d; 2]| {
        let mut lanes = [0f64; LANES];
        for (half, vector) in vectors.into_iter().enumerate() {
            // SAFETY: four lanes from `4 * half` lie inside the array.
            unsafe { _mm256_storeu_pd(lanes.as_mut_ptr().add(4 * half), vector) };
        }
        lanes
    };
    finish(
        unpack(sums),
        unpack(errors),
        unpack(magnitudes),
        rest.iter().map(|value| value.wide()),
        values.len(),
    )
}

/// One block's values and their magnitudes, added plainly in float64 in
/// [`LANES`] lanes, the values left over in lane 0 and the lanes in order.
///
/// No value passes through more than `len / LANES + 2 · LANES` additions,
/// which bounds the rounding by `γ` of that count times the magnitudes.
#[inline(always)]
fn plain_lanes<I: Element>(values: &[I]) -> (f64, f64) {
    let (mut sums, mut magnitudes) = ([0f64; LANES], [0f64; LANES]);
    let (chunks, rest) = values.as_chunks::<LANES>();
    for chunk in chunks {
        for lane in 0..LANES {
            let value = chunk[lane].wide();
            sums[lane] += value;
            magnitudes[lane] += value.abs();
        }
    }
    plain_finish(sums, magnitudes, rest.iter().map(|value| value.wide()))
}

#[inline(always)]
fn plain_finish(
    mut sums: [f64; LANES],
    mut magnitudes: [f64; LANES],
    rest: impl Iterator<Item = f64>,
) -> (f64, f64) {
    for value in rest {
        sums[0] += value;
        magnitudes[0] += value.abs();
    }
    let (mut sum, mut magnitude) = (0.0, 0.0);
    for lane in 0..LANES {
        sum += sums[lane];
        magnitude += magnitudes[lane];
    }
    (sum, magnitude)
}

/// [`plain_lanes`] for float32 values in 256-bit registers, each four read
/// and widened at once.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
fn plain_avx_f32(values: &[f32]) -> (f64, f64) {
    use std::arch::x86_64::{
        __m256d, _mm_loadu_ps, _mm256_add_pd, _mm256_andnot_pd, _mm256_cvtps_pd, _mm256_set1_pd,
        _mm256_setzero_pd, _mm256_storeu_pd,
    };
    let sign = _mm256_set1_pd(-0.0);
    let (mut sums, mut magnitudes) = ([_mm256_setzero_pd(); 2], [_mm256_setzero_pd(); 2]);
    let (chunks, rest) = values.as_chunks::<LANES>();
    for chunk in chunks {
        for half in 0..2 {
            // SAFETY: four values from `4 * half` lie inside the chunk.
            let value = _mm256_cvtps_pd(unsafe { _mm_loadu_ps(chunk.as_ptr().add(4 * half)) });
            sums[half] = _mm256_add_pd(sums[half], value);
            magnitudes[half] = _mm256_add_pd(magnitudes[half], _mm256_andnot_pd(sign, value));
        }
    }
    let unpack = |vectors: [__m256d; 2]| {
        let mut lanes = [0f64; LANES];
        for (half, vector) in vectors.into_iter().enumerate() {
            // SAFETY: four lanes from `4 * half` lie inside the array.
            unsafe { _mm256_storeu_pd(lanes.as_mut_ptr().add(4 * half), vector) };
        }
        lanes
    };
    plain_finish(
        unpack(sums),
        unpack(magnitudes),
        rest.iter().map(|value| value.wide()),
    )
}

/// The plain float64 sum of `values` and of their magnitudes, block by block
/// with the blocks merged pairwise, and a bound on how far the sum is from
/// the exact one.
fn plain_with<I: Element>(values: &[I], parallel: bool) -> (f64, f64) {
    let partials: Vec<(f64, f64)> = if values.len() <= BLOCK {
        vec![I::plain_block(values)]
    } else if parallel && values.len() >= crate::ops::map::PAR_THRESHOLD {
        let len = values.len();
        crate::parallel::map_tasks(len.div_ceil(BLOCK), &|nth| {
            I::plain_block(&values[nth * BLOCK..(nth * BLOCK + BLOCK).min(len)])
        })
    } else {
        values.chunks(BLOCK).map(I::plain_block).collect()
    };
    let (sum, magnitude) = pairwise_fold(partials, (0.0, 0.0), |a, b| (a.0 + b.0, a.1 + b.1));
    // Through a block's lanes, its lane merge, and the merges of up to 2^64
    // blocks: the recursive-summation bound `γ_k · Σ|x|`, doubled to cover
    // the rounding of `Σ|x|` itself.
    let depth = (values.len().min(BLOCK) / LANES + 2 * LANES + 64) as f64;
    (sum, magnitude * depth * 2f64.powi(-52))
}

/// The carried total of `values`, block by block, the blocks merged
/// pairwise: across the pool for a long run when `parallel` -- not for a
/// caller already inside one of its tasks -- with the same blocks either way.
fn carried_with<I: Element>(values: &[I], parallel: bool) -> Carried {
    if values.len() <= BLOCK {
        return I::block(values);
    }
    let partials: Vec<Carried> = if parallel && values.len() >= crate::ops::map::PAR_THRESHOLD {
        let len = values.len();
        crate::parallel::map_tasks(len.div_ceil(BLOCK), &|nth| {
            I::block(&values[nth * BLOCK..(nth * BLOCK + BLOCK).min(len)])
        })
    } else {
        values.chunks(BLOCK).map(I::block).collect()
    };
    pairwise_fold(partials, Carried::default(), Carried::merge)
}

/// The value nearest the sum of `values`.
///
/// A NaN, or infinities of both signs, sum to NaN and an infinity of one sign
/// to that infinity; finite values to the value nearest their exact sum, an
/// infinity past the largest. A sum that cancels to zero is `+0`.
pub(crate) fn sum_to<I: Element, T: Output>(values: &[I]) -> T {
    Reduction::Sum.of_values(values, true)
}

/// The value nearest the mean of `values`, the exact sum over the count; NaN
/// for no values.
pub(crate) fn mean_to<I: Element, T: Output>(values: &[I]) -> T {
    Reduction::Mean.of_values(values, true)
}

/// [`sum_to`] for float64.
pub(crate) fn sum_f64(values: &[f64]) -> f64 {
    sum_to(values)
}

/// What a reduction makes of its values' total.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Reduction {
    Sum,
    Mean,
}

impl Reduction {
    /// The answer for `values`: from their plain float64 sum where its
    /// bound settles it, from their carried total where that does, and from
    /// their exact sum otherwise.
    fn of_values<I: Element, T: Output>(self, values: &[I], parallel: bool) -> T {
        if T::PLAIN {
            let (sum, bound) = plain_with(values, parallel);
            if let Some(value) = self.settled_plain(sum, bound, values.len()) {
                return value;
            }
        }
        self.of(values, carried_with(values, parallel))
    }

    /// The answer from a plain float64 `sum` of `count` values within `bound`
    /// of the exact one, when the bound says which it is.
    fn settled_plain<T: Output>(self, sum: f64, bound: f64, count: usize) -> Option<T> {
        match self {
            Self::Sum => T::settle(sum, 0.0, bound),
            Self::Mean if count == 0 => Some(T::narrow(f64::NAN)),
            Self::Mean => {
                let count = count as f64;
                let quotient = sum / count;
                let bound =
                    bound / count * (1.0 + 2f64.powi(-50)) + quotient.abs() * 2f64.powi(-52);
                T::settle(quotient, 0.0, bound)
            }
        }
    }

    /// The answer from `values`' carried `total`, or from their exact sum
    /// where the bound cannot place it.
    fn of<I: Element, T: Output>(self, values: &[I], total: Carried) -> T {
        self.settled(total, values.len())
            .unwrap_or_else(|| self.exactly(values))
    }

    /// The answer from a carried `total` of `count` values, when the bound
    /// says which it is.
    fn settled<T: Output>(self, total: Carried, count: usize) -> Option<T> {
        match self {
            Self::Sum => total.rounded(),
            Self::Mean if count == 0 => Some(T::narrow(f64::NAN)),
            Self::Mean => total.divided(count),
        }
    }

    /// The answer from the exact sum of `values`.
    #[cold]
    #[inline(never)]
    fn exactly<I: Element, T: Output>(self, values: &[I]) -> T {
        if self == Self::Mean && values.is_empty() {
            return T::narrow(f64::NAN);
        }
        match Total::of(values) {
            Err(special) => T::narrow(special),
            Ok(total) => match self {
                Self::Sum => total.rounded(),
                Self::Mean => total.divided(values.len() as u64),
            },
        }
    }
}

/// Columns a task takes of a strided axis.
const COLUMNS: usize = 512;

/// Below this many tasks across the outputs, a long axis is cut into parts
/// as well. A count fixed by the shape, so the parts, and the answer, do not
/// depend on the threads there are.
const SPREAD: usize = 64;

/// The sum or mean of `input`, laid out `[outer, len, inner]`, along its
/// middle axis into `out` (`outer * inner`), each the float64 nearest the
/// exact one.
///
/// A run along the axis that is contiguous is carried as [`sum_f64`] carries
/// one. Strided, the values of a band of columns are added row by row into a
/// sum, an error and a magnitude per column, a loop of independent columns;
/// a long axis with few outputs is cut into parts merged afterwards, as
/// blocks are. Each output is rounded from its carried total, or from the
/// exact sum of its own values where the bound cannot decide.
pub(crate) fn reduce_axis_exactly<I: Element, T: Output>(
    input: &[I],
    out: &mut [T],
    len: usize,
    inner: usize,
    reduction: Reduction,
) {
    if out.is_empty() {
        return;
    }
    if len == 0 {
        out.fill(T::narrow(match reduction {
            Reduction::Sum => 0.0,
            Reduction::Mean => f64::NAN,
        }));
        return;
    }
    let parallel = input.len() >= crate::ops::map::PAR_THRESHOLD;
    if inner == 1 {
        // A few long runs each spread across the pool; many runs each take
        // one task. The answer is the same either way.
        if out.len() < 4 {
            for (index, slot) in out.iter_mut().enumerate() {
                let run = &input[index * len..(index + 1) * len];
                *slot = reduction.of_values(run, true);
            }
            return;
        }
        let row = |index: usize| {
            let run = &input[index * len..(index + 1) * len];
            reduction.of_values(run, false)
        };
        let per_task = (BLOCK / len).max(1);
        if parallel {
            crate::ops::map::par_out_chunks(out, per_task, &|first, chunk| {
                for (offset, slot) in chunk.iter_mut().enumerate() {
                    *slot = row(first + offset);
                }
            });
        } else {
            for (index, slot) in out.iter_mut().enumerate() {
                *slot = row(index);
            }
        }
        return;
    }
    let outer = out.len() / inner;
    let slab = len * inner;
    let bands = inner.div_ceil(COLUMNS);
    let parts = if outer * bands >= SPREAD {
        1
    } else {
        let rows = (BLOCK / inner.min(COLUMNS)).max(64);
        len.div_ceil(rows)
    };
    let rows_per_part = len.div_ceil(parts);
    let task = |nth: usize| {
        let (o, rest) = (nth / (bands * parts), nth % (bands * parts));
        let (band, part) = (rest / parts, rest % parts);
        let columns = band * COLUMNS..((band + 1) * COLUMNS).min(inner);
        let rows = part * rows_per_part..((part + 1) * rows_per_part).min(len);
        let slab = &input[o * slab..(o + 1) * slab];
        if T::PLAIN {
            columns_part::<I, true>(slab, inner, rows, columns)
        } else {
            columns_part::<I, false>(slab, inner, rows, columns)
        }
    };
    let tasks = outer * bands * parts;
    let partials: Vec<Vec<Carried>> = if parallel {
        crate::parallel::map_tasks(tasks, &task)
    } else {
        (0..tasks).map(task).collect()
    };
    let output = |index: usize| {
        let (o, column) = (index / inner, index % inner);
        let (band, offset) = (column / COLUMNS, column % COLUMNS);
        let first = (o * bands + band) * parts;
        let gathered = || -> Vec<I> {
            (0..len)
                .map(|row| input[o * slab + row * inner + column])
                .collect()
        };
        let partial = partials[first..first + parts]
            .iter()
            .map(|part| part[offset]);
        if T::PLAIN {
            // A value passes through its lane's additions -- a part's rows
            // over the rows a pass takes, and the rows left over -- the merge
            // of its column's lanes, and the merge of the parts, pairwise.
            let pairs: Vec<(f64, f64)> = partial.map(|part| (part.sum, part.slack)).collect();
            let (sum, magnitude) = pairwise_fold(pairs, (0.0, 0.0), |a, b| (a.0 + b.0, a.1 + b.1));
            let width = COLUMNS.min(inner - band * COLUMNS);
            let group = if width == inner {
                (64 / width).max(1)
            } else {
                1
            };
            let depth = (rows_per_part / group + 2 * group + parts.ilog2() as usize + 18) as f64;
            return reduction
                .settled_plain(sum, magnitude * depth * 2f64.powi(-52), len)
                .unwrap_or_else(|| reduction.of_values(&gathered(), false));
        }
        let total = partial.fold(Carried::default(), Carried::merge);
        reduction
            .settled(total, len)
            .unwrap_or_else(|| reduction.exactly(&gathered()))
    };
    if parallel {
        crate::ops::map::par_out_chunks(out, crate::ops::map::PAR_CHUNK, &|first, chunk| {
            for (offset, slot) in chunk.iter_mut().enumerate() {
                *slot = output(first + offset);
            }
        });
    } else {
        for (index, slot) in out.iter_mut().enumerate() {
            *slot = output(index);
        }
    }
}

/// The carried totals of `columns` over `rows` of a `[len, inner]` slab, in
/// the widest build the CPU runs; with `PLAIN`, each column's plain float64
/// sum and the sum of its magnitudes, in `sum` and `slack`.
fn columns_part<I: Element, const PLAIN: bool>(
    slab: &[I],
    inner: usize,
    rows: std::ops::Range<usize>,
    columns: std::ops::Range<usize>,
) -> Vec<Carried> {
    #[cfg(target_arch = "x86_64")]
    if crate::ops::simd::simd_capabilities().avx {
        // SAFETY: avx was detected on this CPU.
        return unsafe { columns_part_avx::<I, PLAIN>(slab, inner, rows, columns) };
    }
    columns_part_body::<I, PLAIN>(slab, inner, rows, columns)
}

/// [`columns_part_body`] compiled for 256-bit registers.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
fn columns_part_avx<I: Element, const PLAIN: bool>(
    slab: &[I],
    inner: usize,
    rows: std::ops::Range<usize>,
    columns: std::ops::Range<usize>,
) -> Vec<Carried> {
    columns_part_body::<I, PLAIN>(slab, inner, rows, columns)
}

/// The rows added into their columns' sums, errors and magnitudes.
///
/// A band of a few columns that is every column takes several rows at a
/// time, so that each pass is a run long enough to vectorize: lane `l` holds
/// column `l % width`, and the lanes of a column merge at the end. Rows left
/// over go to the first lanes.
#[inline(always)]
fn columns_part_body<I: Element, const PLAIN: bool>(
    slab: &[I],
    inner: usize,
    rows: std::ops::Range<usize>,
    columns: std::ops::Range<usize>,
) -> Vec<Carried> {
    let width = columns.len();
    let group = if width == inner {
        (64 / width).max(1)
    } else {
        1
    };
    let lanes = width * group;
    let (mut sums, mut errors, mut magnitudes) =
        (vec![0f64; lanes], vec![0f64; lanes], vec![0f64; lanes]);
    let mut row = rows.start;
    while row + group <= rows.end {
        let start = row * inner + columns.start;
        accumulate_columns::<I, PLAIN>(
            &mut sums,
            &mut errors,
            &mut magnitudes,
            &slab[start..start + lanes],
        );
        row += group;
    }
    while row < rows.end {
        let start = row * inner + columns.start;
        accumulate_columns::<I, PLAIN>(
            &mut sums[..width],
            &mut errors[..width],
            &mut magnitudes[..width],
            &slab[start..start + width],
        );
        row += 1;
    }
    if PLAIN {
        return (0..width)
            .map(|column| {
                let (mut sum, mut magnitude) = (0.0, 0.0);
                for nth in 0..group {
                    sum += sums[nth * width + column];
                    magnitude += magnitudes[nth * width + column];
                }
                Carried {
                    sum,
                    error: 0.0,
                    slack: magnitude,
                }
            })
            .collect();
    }
    // No lane takes more than this many additions.
    let additions = (rows.len() / group + group) as f64;
    (0..width)
        .map(|column| {
            (0..group)
                .map(|nth| {
                    let lane = nth * width + column;
                    Carried {
                        sum: sums[lane],
                        error: errors[lane],
                        slack: magnitudes[lane] * additions * 2f64.powi(-52),
                    }
                })
                .fold(Carried::default(), Carried::merge)
        })
        .collect()
}

/// Each column's value added to its sum by a two-sum, the error and its
/// magnitude beside it -- or with `PLAIN`, added plainly with its own
/// magnitude beside it: the columns are independent, so the loop vectorizes.
#[inline(always)]
fn accumulate_columns<I: Element, const PLAIN: bool>(
    sums: &mut [f64],
    errors: &mut [f64],
    magnitudes: &mut [f64],
    row: &[I],
) {
    for (((sum, error), magnitude), &value) in sums
        .iter_mut()
        .zip(errors.iter_mut())
        .zip(magnitudes.iter_mut())
        .zip(row)
    {
        let value = value.wide();
        if PLAIN {
            *sum += value;
            *magnitude += value.abs();
        } else {
            let (total, rounding) = two_sum(*sum, value);
            *sum = total;
            *error += rounding;
            *magnitude += rounding.abs();
        }
    }
}

/// One block of `a[i] · b[i]`, each product of two float32s exact in
/// float64, added plainly in [`LANES`] lanes with their magnitudes; as
/// [`plain_lanes`] is.
#[inline(always)]
fn plain_dot_lanes(a: &[f32], b: &[f32]) -> (f64, f64) {
    let (mut sums, mut magnitudes) = ([0f64; LANES], [0f64; LANES]);
    let (left, left_rest) = a.as_chunks::<LANES>();
    let (right, right_rest) = b.as_chunks::<LANES>();
    for (x, y) in left.iter().zip(right) {
        for lane in 0..LANES {
            let product = f64::from(x[lane]) * f64::from(y[lane]);
            sums[lane] += product;
            magnitudes[lane] += product.abs();
        }
    }
    let rest = left_rest
        .iter()
        .zip(right_rest)
        .map(|(&x, &y)| f64::from(x) * f64::from(y));
    plain_finish(sums, magnitudes, rest)
}

/// One block of `a[i] · b[i]` carried as [`block_lanes`] carries values.
#[inline(always)]
fn carried_dot_lanes(a: &[f32], b: &[f32]) -> Carried {
    let (mut sums, mut errors, mut magnitudes) = ([0f64; LANES], [0f64; LANES], [0f64; LANES]);
    let (left, left_rest) = a.as_chunks::<LANES>();
    let (right, right_rest) = b.as_chunks::<LANES>();
    for (x, y) in left.iter().zip(right) {
        for lane in 0..LANES {
            let (sum, error) = two_sum(sums[lane], f64::from(x[lane]) * f64::from(y[lane]));
            sums[lane] = sum;
            errors[lane] += error;
            magnitudes[lane] += error.abs();
        }
    }
    let rest = left_rest
        .iter()
        .zip(right_rest)
        .map(|(&x, &y)| f64::from(x) * f64::from(y));
    finish(sums, errors, magnitudes, rest, a.len())
}

/// [`plain_dot_lanes`] in 256-bit registers.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
fn plain_dot_avx(a: &[f32], b: &[f32]) -> (f64, f64) {
    use std::arch::x86_64::{
        __m256d, _mm_loadu_ps, _mm256_add_pd, _mm256_andnot_pd, _mm256_cvtps_pd, _mm256_mul_pd,
        _mm256_set1_pd, _mm256_setzero_pd, _mm256_storeu_pd,
    };
    let sign = _mm256_set1_pd(-0.0);
    let (mut sums, mut magnitudes) = ([_mm256_setzero_pd(); 2], [_mm256_setzero_pd(); 2]);
    let (left, left_rest) = a.as_chunks::<LANES>();
    let (right, right_rest) = b.as_chunks::<LANES>();
    for (x, y) in left.iter().zip(right) {
        for half in 0..2 {
            // SAFETY: four values from `4 * half` lie inside each chunk.
            let (x, y) = unsafe {
                (
                    _mm256_cvtps_pd(_mm_loadu_ps(x.as_ptr().add(4 * half))),
                    _mm256_cvtps_pd(_mm_loadu_ps(y.as_ptr().add(4 * half))),
                )
            };
            let product = _mm256_mul_pd(x, y);
            sums[half] = _mm256_add_pd(sums[half], product);
            magnitudes[half] = _mm256_add_pd(magnitudes[half], _mm256_andnot_pd(sign, product));
        }
    }
    let unpack = |vectors: [__m256d; 2]| {
        let mut lanes = [0f64; LANES];
        for (half, vector) in vectors.into_iter().enumerate() {
            // SAFETY: four lanes from `4 * half` lie inside the array.
            unsafe { _mm256_storeu_pd(lanes.as_mut_ptr().add(4 * half), vector) };
        }
        lanes
    };
    let rest = left_rest
        .iter()
        .zip(right_rest)
        .map(|(&x, &y)| f64::from(x) * f64::from(y));
    plain_finish(unpack(sums), unpack(magnitudes), rest)
}

/// [`carried_dot_lanes`] in 256-bit registers.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
fn carried_dot_avx(a: &[f32], b: &[f32]) -> Carried {
    use std::arch::x86_64::{
        __m256d, _mm_loadu_ps, _mm256_add_pd, _mm256_andnot_pd, _mm256_cvtps_pd, _mm256_mul_pd,
        _mm256_set1_pd, _mm256_setzero_pd, _mm256_storeu_pd, _mm256_sub_pd,
    };
    let sign = _mm256_set1_pd(-0.0);
    let (mut sums, mut errors, mut magnitudes) = (
        [_mm256_setzero_pd(); 2],
        [_mm256_setzero_pd(); 2],
        [_mm256_setzero_pd(); 2],
    );
    let (left, left_rest) = a.as_chunks::<LANES>();
    let (right, right_rest) = b.as_chunks::<LANES>();
    for (x, y) in left.iter().zip(right) {
        for half in 0..2 {
            // SAFETY: four values from `4 * half` lie inside each chunk.
            let value = unsafe {
                _mm256_mul_pd(
                    _mm256_cvtps_pd(_mm_loadu_ps(x.as_ptr().add(4 * half))),
                    _mm256_cvtps_pd(_mm_loadu_ps(y.as_ptr().add(4 * half))),
                )
            };
            let sum = _mm256_add_pd(sums[half], value);
            let virtual_value = _mm256_sub_pd(sum, sums[half]);
            let virtual_sum = _mm256_sub_pd(sum, virtual_value);
            let error = _mm256_add_pd(
                _mm256_sub_pd(sums[half], virtual_sum),
                _mm256_sub_pd(value, virtual_value),
            );
            sums[half] = sum;
            errors[half] = _mm256_add_pd(errors[half], error);
            magnitudes[half] = _mm256_add_pd(magnitudes[half], _mm256_andnot_pd(sign, error));
        }
    }
    let unpack = |vectors: [__m256d; 2]| {
        let mut lanes = [0f64; LANES];
        for (half, vector) in vectors.into_iter().enumerate() {
            // SAFETY: four lanes from `4 * half` lie inside the array.
            unsafe { _mm256_storeu_pd(lanes.as_mut_ptr().add(4 * half), vector) };
        }
        lanes
    };
    let rest = left_rest
        .iter()
        .zip(right_rest)
        .map(|(&x, &y)| f64::from(x) * f64::from(y));
    finish(
        unpack(sums),
        unpack(errors),
        unpack(magnitudes),
        rest,
        a.len(),
    )
}

/// `block` over aligned blocks of `a` and `b`, across the pool for a long
/// run, in block order either way.
fn dot_blocks<R: Send>(a: &[f32], b: &[f32], block: impl Fn(&[f32], &[f32]) -> R + Sync) -> Vec<R> {
    let len = a.len();
    if len <= BLOCK {
        return vec![block(a, b)];
    }
    let span = |nth: usize| nth * BLOCK..(nth * BLOCK + BLOCK).min(len);
    if len >= crate::ops::map::PAR_THRESHOLD {
        crate::parallel::map_tasks(len.div_ceil(BLOCK), &|nth| {
            block(&a[span(nth)], &b[span(nth)])
        })
    } else {
        (0..len.div_ceil(BLOCK))
            .map(|nth| block(&a[span(nth)], &b[span(nth)]))
            .collect()
    }
}

/// The float32 nearest `Σ a[i] · b[i]`.
///
/// Each product of two float32s is exact in float64, so this is a sum of
/// float64s, settled as a float32 sum is: plainly with a bound, carried
/// where the bound reaches a halfway point, and exactly where that does.
/// Products of an infinity and a zero, and NaNs, make NaN as they would
/// added up one by one.
pub(crate) fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let avx = {
        #[cfg(target_arch = "x86_64")]
        {
            crate::ops::simd::simd_capabilities().avx
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            false
        }
    };
    let plain_block = |x: &[f32], y: &[f32]| {
        #[cfg(target_arch = "x86_64")]
        if avx {
            // SAFETY: avx was detected on this CPU.
            return unsafe { plain_dot_avx(x, y) };
        }
        plain_dot_lanes(x, y)
    };
    let (sum, magnitude) = if a.len() <= BLOCK {
        plain_block(a, b)
    } else {
        let plain = dot_blocks(a, b, plain_block);
        pairwise_fold(plain, (0.0, 0.0), |p, q| (p.0 + q.0, p.1 + q.1))
    };
    let depth = (a.len().min(BLOCK) / LANES + 2 * LANES + 64) as f64;
    if let Some(value) = f32::settle(sum, 0.0, magnitude * depth * 2f64.powi(-52)) {
        return value;
    }
    let carried = dot_blocks(a, b, |x, y| {
        #[cfg(target_arch = "x86_64")]
        if avx {
            // SAFETY: avx was detected on this CPU.
            return unsafe { carried_dot_avx(x, y) };
        }
        carried_dot_lanes(x, y)
    });
    pairwise_fold(carried, Carried::default(), Carried::merge)
        .rounded()
        .unwrap_or_else(|| {
            let products = a.iter().zip(b).map(|(&x, &y)| f64::from(x) * f64::from(y));
            match Total::of_wide(products) {
                Err(special) => special as f32,
                Ok(total) => total.rounded(),
            }
        })
}

/// The float32 nearest every number within `bound` of `value`, when that is
/// one float32.
pub(crate) fn settle_f32(value: f64, bound: f64) -> Option<f32> {
    f32::settle(value, 0.0, bound)
}

/// The float64 nearest the mean of `values`, from their exact sum in 128
/// bits divided once; NaN for no values.
///
/// Summed in float64 an int64 past 2^53 was rounded before it was added, and
/// the rounded sum rounded again when divided.
pub(crate) fn mean_i64(values: &[i64]) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    // Fewer than 2^63 values of at most 2^63 each: below 2^126.
    let total: i128 = values.iter().map(|&value| i128::from(value)).sum();
    if total == 0 {
        return 0.0;
    }
    let magnitude = total.unsigned_abs();
    // Moved up to bit 126 the dividend leaves a quotient of at least 63 bits,
    // with whether the division was exact marked far below where it rounds.
    let shift = magnitude.leading_zeros() - 1;
    let dividend = magnitude << shift;
    let divisor = values.len() as u128;
    let quotient = dividend / divisor;
    let inexact = quotient * divisor != dividend;
    rounded(total < 0, quotient | u128::from(inexact), -(shift as i32))
}

/// The exact sum of finite floats, in base-2^64 limbs from `2^-1074` up,
/// the positive and the negative values apart.
struct Total {
    positive: Vec<u64>,
    negative: Vec<u64>,
}

/// The exponent of the least significant bit of limb `0`.
const LEAST: i32 = -1074;

impl Total {
    /// The exact sum of `values`, or the answer itself when one is not
    /// finite.
    #[cold]
    #[inline(never)]
    fn of<I: Element>(values: &[I]) -> Result<Self, f64> {
        Self::of_wide(values.iter().map(|value| value.wide()))
    }

    /// [`Total::of`] for values already in float64, each exact.
    #[cold]
    #[inline(never)]
    fn of_wide(values: impl Iterator<Item = f64> + Clone) -> Result<Self, f64> {
        let (mut nan, mut up, mut down) = (false, false, false);
        for value in values.clone() {
            nan |= value.is_nan();
            up |= value == f64::INFINITY;
            down |= value == f64::NEG_INFINITY;
        }
        if nan || (up && down) {
            return Err(f64::NAN);
        }
        if up || down {
            return Err(if up { f64::INFINITY } else { f64::NEG_INFINITY });
        }
        // Every finite float64 lies below 2^1024, which is bit 2098 here; the
        // carries of up to 2^64 values take 64 bits more.
        let limbs = (2098 + 64) / 64 + 2;
        let mut total = Self {
            positive: vec![0; limbs],
            negative: vec![0; limbs],
        };
        for value in values {
            let exact = Exact::float(value).expect("finite");
            if exact.magnitude != 0 {
                let limbs = if exact.negative {
                    &mut total.negative
                } else {
                    &mut total.positive
                };
                add_shifted(
                    limbs,
                    u128::from(exact.magnitude),
                    (exact.exponent - LEAST) as usize,
                );
            }
        }
        Ok(total)
    }

    /// The magnitude of the sum and whether it is negative.
    fn difference(self) -> (bool, Vec<u64>) {
        let negative = self.negative.iter().rev().cmp(self.positive.iter().rev())
            == std::cmp::Ordering::Greater;
        let (mut larger, smaller) = if negative {
            (self.negative, self.positive)
        } else {
            (self.positive, self.negative)
        };
        let mut borrow = false;
        for (limb, &other) in larger.iter_mut().zip(&smaller) {
            let (value, first) = limb.overflowing_sub(other);
            let (value, second) = value.overflowing_sub(u64::from(borrow));
            *limb = value;
            borrow = first || second;
        }
        (negative, larger)
    }

    /// The value nearest the sum; `+0` for a sum that cancels.
    fn rounded<T: Output>(self) -> T {
        let (negative, limbs) = self.difference();
        rounded_at(negative, &limbs, false, LEAST)
    }

    /// The float64 nearest the sum over `count`, by long division with a
    /// limb of fraction below the least subnormal, so that a remainder left
    /// over is far below where the quotient rounds.
    fn divided<T: Output>(self, count: u64) -> T {
        let (negative, mut limbs) = self.difference();
        limbs.insert(0, 0);
        let divisor = u128::from(count);
        let mut remainder = 0u128;
        for limb in limbs.iter_mut().rev() {
            let current = (remainder << 64) | u128::from(*limb);
            let quotient = current / divisor;
            remainder = current - quotient * divisor;
            *limb = quotient as u64;
        }
        rounded_at(negative, &limbs, remainder != 0, LEAST - 64)
    }
}

/// `±limbs · 2^least`, and a fraction more when `inexact`, rounded to the
/// nearest float64: the leading bits with whether anything lies below them
/// marked in the last, which is at least two places below where float64
/// rounds -- `least` is the least subnormal's exponent or below it.
fn rounded_at<T: Output>(negative: bool, limbs: &[u64], inexact: bool, least: i32) -> T {
    let Some(top) = limbs.iter().rposition(|&limb| limb != 0) else {
        // Nothing, or less than `2^least`, which is far below the least
        // subnormal.
        return T::assemble(negative, 0, 0);
    };
    if top == 0 {
        let magnitude = (u128::from(limbs[0]) << 1) | u128::from(inexact);
        return rounded(negative, magnitude, least - 1);
    }
    let leading = (u128::from(limbs[top]) << 64) | u128::from(limbs[top - 1]);
    let below = inexact || limbs[..top - 1].iter().any(|&limb| limb != 0);
    rounded(
        negative,
        leading | u128::from(below),
        least + 64 * (top as i32 - 1),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn generator(mut state: u64) -> impl FnMut() -> u64 {
        move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        }
    }

    fn mean_f64(values: &[f64]) -> f64 {
        mean_to(values)
    }

    fn exact_sum(values: &[f64]) -> f64 {
        Total::of(values).map_or_else(|special| special, |total| total.rounded())
    }

    #[test]
    fn sums_are_rounded_once() {
        assert_eq!(sum_f64(&[1.0, 1e100, 1.0, -1e100]), 2.0);
        assert_eq!(sum_f64(&[0.1; 10]), 1.0);
        assert_eq!(sum_f64(&[f64::MAX, f64::MAX, -f64::MAX]), f64::MAX);
        assert_eq!(sum_f64(&[f64::MAX, f64::MAX]), f64::INFINITY);
        assert!(sum_f64(&[f64::INFINITY, f64::NEG_INFINITY]).is_nan());
        assert_eq!(sum_f64(&[1.0, f64::NEG_INFINITY]), f64::NEG_INFINITY);
        assert!(sum_f64(&[1.0, f64::NAN]).is_nan());
        assert_eq!(sum_f64(&[]).to_bits(), 0);
        assert_eq!(sum_f64(&[1.0, -1.0]).to_bits(), 0);
        // Halfway between 1 and the float64 after it, and a hair past.
        let half = f64::EPSILON / 2.0;
        assert_eq!(sum_f64(&[1.0, half]), 1.0);
        assert_eq!(sum_f64(&[1.0, half, 1e-300]), 1.0 + f64::EPSILON);
        assert_eq!(
            sum_f64(&[f64::from_bits(1), f64::from_bits(1)]),
            f64::from_bits(2)
        );
    }

    #[test]
    fn the_carried_sum_agrees_with_the_exact_one() {
        let mut next = generator(0x6c62_272e_07bb_0142);
        for round in 0..3000 {
            let len = 1 + (next() % 20_000) as usize;
            let spread = 1 + next() % 60;
            let mut values: Vec<f64> = (0..len)
                .map(|_| {
                    let unit = (next() >> 11) as f64 / (1u64 << 53) as f64 - 0.5;
                    unit * 2f64.powi((next() % spread) as i32 - 30)
                })
                .collect();
            if round % 3 == 0 {
                // Cancelling to almost nothing.
                let total = values.iter().sum::<f64>();
                values.push(-total);
            }
            let exact = exact_sum(&values);
            assert_eq!(sum_f64(&values).to_bits(), exact.to_bits(), "{len} values");
            let mean = mean_f64(&values);
            let expected: f64 = Total::of(&values).unwrap().divided(values.len() as u64);
            assert_eq!(mean.to_bits(), expected.to_bits(), "{len} values");
        }
    }

    #[test]
    fn every_build_adds_the_same_lanes() {
        let mut next = generator(0x3c6e_f372_fe94_f82b);
        for len in [0, 1, 7, 8, 9, 100, 8191, 8192] {
            let values: Vec<f64> = (0..len)
                .map(|_| {
                    f64::from_bits(next() >> 2) * if next().is_multiple_of(2) { 1.0 } else { -1.0 }
                })
                .collect();
            let plain = block_lanes(&values);
            #[cfg(target_arch = "x86_64")]
            if crate::ops::simd::simd_capabilities().avx {
                // SAFETY: avx was detected on this CPU.
                let wide = unsafe { block_avx(&values) };
                assert_eq!(plain.sum.to_bits(), wide.sum.to_bits(), "{len}");
                assert_eq!(plain.error.to_bits(), wide.error.to_bits(), "{len}");
                assert_eq!(plain.slack.to_bits(), wide.slack.to_bits(), "{len}");
            }
            let _ = plain;
        }
    }

    #[test]
    fn every_output_of_an_axis_is_rounded_once() {
        let mut next = generator(0x1b87_3593_cc9e_2d51);
        // Rows, a band, several bands, and a long axis cut into parts.
        for (outer, len, inner) in [
            (5, 3000, 1),
            (2, 300, 7),
            (1, 50, 1500),
            (1, 200_000, 3),
            (3, 0, 4),
        ] {
            let values: Vec<f64> = (0..outer * len * inner)
                .map(|_| {
                    let unit = (next() >> 11) as f64 / (1u64 << 53) as f64 - 0.5;
                    unit * 2f64.powi((next() % 40) as i32 - 20)
                })
                .collect();
            for reduction in [Reduction::Sum, Reduction::Mean] {
                let mut out = vec![0.0f64; outer * inner];
                reduce_axis_exactly(&values, &mut out, len, inner, reduction);
                for o in 0..outer {
                    for column in 0..inner {
                        let run: Vec<f64> = (0..len)
                            .map(|row| values[o * len * inner + row * inner + column])
                            .collect();
                        let expected: f64 = reduction.exactly(&run);
                        let got = out[o * inner + column];
                        assert!(
                            got.to_bits() == expected.to_bits()
                                || (got.is_nan() && expected.is_nan()),
                            "{outer}x{len}x{inner} {reduction:?} at {o},{column}: {got:e} against {expected:e}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn float32_sums_are_rounded_once_to_float32() {
        let twice: f32 = sum_to(&[1.0f32, 2f32.powi(-24), 2f32.powi(-60)]);
        assert_eq!(twice, 1.0 + f32::EPSILON);
        let tie: f32 = sum_to(&[1.0f32, 2f32.powi(-24)]);
        assert_eq!(tie, 1.0);
        let lost: f32 = sum_to(&[1e30f32, 1e-10, -1e30]);
        assert_eq!(lost, 1e-10);
        let overflow: f32 = sum_to(&[f32::MAX, f32::MAX, -f32::MAX]);
        assert_eq!(overflow, f32::MAX);
        let mut next = generator(0x9e37_79b9_7f4a_7c15);
        for round in 0..2000 {
            let len = 1 + (next() % 5000) as usize;
            let mut values: Vec<f32> = (0..len)
                .map(|_| {
                    let unit = (next() >> 11) as f32 / (1u64 << 53) as f32 - 0.5;
                    unit * 2f32.powi((next() % 80) as i32 - 40)
                })
                .collect();
            if round % 2 == 0 {
                let mirror: Vec<f32> = values.iter().map(|&v| -v).collect();
                values.extend(&mirror[1..]);
            }
            let exact: f32 = Total::of(&values).unwrap().rounded();
            let sum: f32 = sum_to(&values);
            assert_eq!(sum.to_bits(), exact.to_bits(), "{len} values");
            let exact_mean: f32 = Total::of(&values).unwrap().divided(values.len() as u64);
            let mean: f32 = mean_to(&values);
            assert_eq!(mean.to_bits(), exact_mean.to_bits(), "{len} values");
            let wide: f64 = sum_to(&values);
            let wide_exact: f64 = Total::of(&values).unwrap().rounded();
            assert_eq!(wide.to_bits(), wide_exact.to_bits(), "{len} values");
        }
        for (outer, len, inner) in [(3, 2000, 1), (2, 300, 9), (1, 70_000, 2)] {
            let values: Vec<f32> = (0..outer * len * inner)
                .map(|_| ((next() >> 11) as f32 / (1u64 << 53) as f32 - 0.5) * 1e6)
                .collect();
            let mut out = vec![0f32; outer * inner];
            reduce_axis_exactly(&values, &mut out, len, inner, Reduction::Sum);
            for o in 0..outer {
                for column in 0..inner {
                    let run: Vec<f32> = (0..len)
                        .map(|row| values[o * len * inner + row * inner + column])
                        .collect();
                    let expected: f32 = Reduction::Sum.exactly(&run);
                    assert_eq!(out[o * inner + column].to_bits(), expected.to_bits());
                }
            }
        }
    }

    #[test]
    fn means_divide_the_exact_sum_once() {
        assert_eq!(mean_f64(&[0.1, 0.2, 0.3]), 0.2);
        assert_eq!(mean_f64(&[f64::MAX, f64::MAX]), f64::MAX);
        assert!(mean_f64(&[]).is_nan());
        let third = mean_f64(&[1.0, 0.0, 0.0]);
        assert_eq!(third, 1.0 / 3.0);
    }
}
