// Copyright (c) Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

//! Float sums, means and dots rounded once.
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
//!
//! A dot is a sum of products. A product of two float32s is exact in
//! float64, so a float32 dot is a float32 sum of those. A product of two
//! float64s is split by a fused multiply-add into the float64 nearest it and
//! the exact error of that, and both are carried as a sum's values are; the
//! exact dot sums each product's 106 bits in limbs.

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
    /// The carried total of `a[i] · b[i]`, block by block in the widest
    /// build the CPU runs, the blocks merged pairwise.
    fn carried_dot(a: &[Self], b: &[Self]) -> Carried;
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

    fn carried_dot(a: &[Self], b: &[Self]) -> Carried {
        let fused = fused();
        let block = |x: &[f64], y: &[f64]| {
            #[cfg(target_arch = "x86_64")]
            if fused {
                // SAFETY: avx2 and fma were detected on this CPU.
                return unsafe { carried_dot_fma(x, y) };
            }
            carried_dot_f64_lanes(x, y)
        };
        if a.len() <= BLOCK {
            return block(a, b);
        }
        pairwise_fold(dot_blocks(a, b, block), Carried::default(), Carried::merge)
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

    fn carried_dot(a: &[Self], b: &[Self]) -> Carried {
        let block = |x: &[f32], y: &[f32]| {
            #[cfg(target_arch = "x86_64")]
            if crate::ops::simd::simd_capabilities().avx {
                // SAFETY: avx was detected on this CPU.
                return unsafe { carried_dot_avx(x, y) };
            }
            carried_dot_lanes(x, y)
        };
        if a.len() <= BLOCK {
            return block(a, b);
        }
        pairwise_fold(dot_blocks(a, b, block), Carried::default(), Carried::merge)
    }
}

/// A float dtype a total is rounded to.
pub(crate) trait Output: Target + Send + Sync {
    /// Whether a plain float64 sum, with a bound on its rounding, usually
    /// settles the answer: it does for float32, 29 bits narrower, and is
    /// never close enough for float64.
    const PLAIN: bool;
    /// The value nearest every number within `bound` of `value + rest`, when
    /// that is one value; `value + rest` is a two-sum's pair, so `value` is
    /// the float64 nearest it. A zero bound always settles a finite pair.
    fn settle(value: f64, rest: f64, bound: f64) -> Option<Self>;
    /// [`Output::settle`] without branches, for lanes advanced side by
    /// side: the value and whether the bound settles it, a halfway point
    /// with a zero bound not among them.
    fn settle_lanes(value: f64, rest: f64, bound: f64) -> (Self, bool);
    /// Whether `sum + error` with a zero bound is rounded by
    /// [`Output::exact_pair`]: always for float64, and for float32 where
    /// the error is zero.
    fn exact_lane(error: f64) -> bool;
    /// The value nearest `sum + error`, a two-sum's pair, where
    /// [`Output::exact_lane`] says it is that simple.
    fn exact_pair(sum: f64, error: f64) -> Self;
}

impl Output for f64 {
    const PLAIN: bool = false;

    fn settle(value: f64, rest: f64, bound: f64) -> Option<Self> {
        if bound == 0.0 {
            // `value` is the pair rounded once, halfway or not.
            return value.is_finite().then_some(value);
        }
        // The gap to the float64 on the side towards zero, the smaller of the
        // two; a zero or a non-finite value leaves it NaN and the test false.
        let inward = (value - f64::from_bits(value.to_bits().wrapping_sub(1))).abs();
        (rest.abs() + bound < inward * SHY_OF_HALF && value != 0.0).then_some(value)
    }

    #[inline(always)]
    fn exact_lane(_error: f64) -> bool {
        true
    }

    #[inline(always)]
    fn exact_pair(sum: f64, error: f64) -> Self {
        sum + error
    }

    #[inline(always)]
    fn settle_lanes(value: f64, rest: f64, bound: f64) -> (Self, bool) {
        let inward = (value - f64::from_bits(value.to_bits().wrapping_sub(1))).abs();
        let clear = rest.abs() + bound < inward * SHY_OF_HALF && value != 0.0;
        (value, value.is_finite() & ((bound == 0.0) | clear))
    }
}

impl Output for f32 {
    const PLAIN: bool = true;

    fn settle(value: f64, rest: f64, bound: f64) -> Option<Self> {
        let narrow = value as f32;
        if bound == 0.0 {
            if !value.is_finite() {
                return None;
            }
            if rest == 0.0 {
                // `value` is the total, rounded once by the conversion.
                return Some(narrow);
            }
            // The total lies past `value` towards `rest`, short of the next
            // float64, so the odd one of the two is its rounding to odd at
            // float64's last bit, which narrows to float32 rounded once.
            let odd = if value.to_bits() & 1 == 1 {
                value
            } else if rest > 0.0 {
                value.next_up()
            } else {
                value.next_down()
            };
            return Some(odd as f32);
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

    #[inline(always)]
    fn exact_lane(error: f64) -> bool {
        error == 0.0
    }

    #[inline(always)]
    fn exact_pair(sum: f64, _error: f64) -> Self {
        sum as f32
    }

    #[inline(always)]
    fn settle_lanes(value: f64, rest: f64, bound: f64) -> (Self, bool) {
        // In magnitudes, the halfway points either side of the float32
        // nearest, with zero's neighbours its own reflection.
        let magnitude = value.abs();
        let narrow = magnitude as f32;
        let bits = narrow.to_bits();
        let back = f64::from(narrow);
        let above = f64::from(f32::from_bits(bits.wrapping_add(1)));
        let below = if bits == 0 {
            -above
        } else {
            f64::from(f32::from_bits(bits.wrapping_sub(1)))
        };
        let (up, down) = ((back + above) * 0.5, (back + below) * 0.5);
        let toward = if value < 0.0 { -rest } else { rest };
        let room = |gap: f64| gap * (1.0 - 2f64.powi(-40));
        let clear =
            (toward + bound < room(up - magnitude)) & (bound - toward < room(magnitude - down));
        let signed = if value < 0.0 { -narrow } else { narrow };
        (signed, (narrow < f32::MAX) & clear)
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
    let mut lanes: [Carried; LANES] = std::array::from_fn(|lane| Carried {
        sum: sums[lane],
        error: errors[lane],
        slack: magnitudes[lane] * additions * 2f64.powi(-52),
    });
    // Pairwise, so that a short block waits on three merges rather than
    // eight.
    let mut width = LANES;
    while width > 1 {
        width /= 2;
        for lane in 0..width {
            lanes[lane] = lanes[2 * lane].merge(lanes[2 * lane + 1]);
        }
    }
    lanes[0]
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
    (sum, magnitude * plain_depth(values.len()) * 2f64.powi(-52))
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
fn dot_blocks<E: Sync, R: Send>(
    a: &[E],
    b: &[E],
    block: impl Fn(&[E], &[E]) -> R + Sync,
) -> Vec<R> {
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

/// The plain float64 sum of `a[i] · b[i]` and of the products' magnitudes,
/// block by block, the blocks merged pairwise: within
/// [`plain_depth`] roundings of the exact sum.
fn plain_dot(a: &[f32], b: &[f32]) -> (f64, f64) {
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
    if a.len() <= BLOCK {
        plain_block(a, b)
    } else {
        let plain = dot_blocks(a, b, plain_block);
        pairwise_fold(plain, (0.0, 0.0), |p, q| (p.0 + q.0, p.1 + q.1))
    }
}

/// The roundings a value passes through in [`plain_dot`] or a plain sum of
/// `len` values: a block's lanes, its lane merge, and the merges of up to
/// 2^64 blocks.
fn plain_depth(len: usize) -> f64 {
    (len.min(BLOCK) / LANES + 2 * LANES + 64) as f64
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
    let (sum, magnitude) = plain_dot(a, b);
    let bound = magnitude * plain_depth(a.len()) * 2f64.powi(-52);
    if let Some(value) = f32::settle(sum, 0.0, bound) {
        return value;
    }
    f32::carried_dot(a, b).rounded().unwrap_or_else(|| {
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

/// One float64 product `x · y` added into a lane: the product split by a
/// fused multiply-add into the float64 nearest it and that rounding's
/// error, the product added to the lane's sum by a two-sum, and both errors
/// added beside it as one term.
#[inline(always)]
fn add_product(sum: &mut f64, error: &mut f64, magnitude: &mut f64, x: f64, y: f64) {
    let product = x * y;
    let product_error = x.mul_add(y, -product);
    let (total, rounding) = two_sum(*sum, product);
    *sum = total;
    *error += rounding + product_error;
    *magnitude += rounding.abs() + product_error.abs();
}

/// One block of `a[i] · b[i]` for float64, carried as [`block_lanes`]
/// carries values, with [`add_product`] for each product: a lane's error
/// terms pass through one more addition each, which `finish`'s count of
/// `LANES` past the lane's length covers.
#[inline(always)]
fn carried_dot_f64_lanes(a: &[f64], b: &[f64]) -> Carried {
    let (mut sums, mut errors, mut magnitudes) = ([0f64; LANES], [0f64; LANES], [0f64; LANES]);
    let (left, left_rest) = a.as_chunks::<LANES>();
    let (right, right_rest) = b.as_chunks::<LANES>();
    for (x, y) in left.iter().zip(right) {
        for lane in 0..LANES {
            add_product(
                &mut sums[lane],
                &mut errors[lane],
                &mut magnitudes[lane],
                x[lane],
                y[lane],
            );
        }
    }
    for (&x, &y) in left_rest.iter().zip(right_rest) {
        add_product(&mut sums[0], &mut errors[0], &mut magnitudes[0], x, y);
    }
    finish(sums, errors, magnitudes, std::iter::empty(), a.len())
}

/// [`carried_dot_f64_lanes`] in 256-bit registers with fused multiply-adds:
/// the same lanes, operations and order.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
fn carried_dot_fma(a: &[f64], b: &[f64]) -> Carried {
    use std::arch::x86_64::{
        __m256d, _mm256_add_pd, _mm256_andnot_pd, _mm256_fmsub_pd, _mm256_loadu_pd, _mm256_mul_pd,
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
            let (x, y) = unsafe {
                (
                    _mm256_loadu_pd(x.as_ptr().add(4 * half)),
                    _mm256_loadu_pd(y.as_ptr().add(4 * half)),
                )
            };
            let product = _mm256_mul_pd(x, y);
            let product_error = _mm256_fmsub_pd(x, y, product);
            let sum = _mm256_add_pd(sums[half], product);
            let virtual_value = _mm256_sub_pd(sum, sums[half]);
            let virtual_sum = _mm256_sub_pd(sum, virtual_value);
            let rounding = _mm256_add_pd(
                _mm256_sub_pd(sums[half], virtual_sum),
                _mm256_sub_pd(product, virtual_value),
            );
            sums[half] = sum;
            errors[half] = _mm256_add_pd(errors[half], _mm256_add_pd(rounding, product_error));
            let size = _mm256_add_pd(
                _mm256_andnot_pd(sign, rounding),
                _mm256_andnot_pd(sign, product_error),
            );
            magnitudes[half] = _mm256_add_pd(magnitudes[half], size);
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
    let (mut sums, mut errors, mut magnitudes) = (unpack(sums), unpack(errors), unpack(magnitudes));
    for (&x, &y) in left_rest.iter().zip(right_rest) {
        add_product(&mut sums[0], &mut errors[0], &mut magnitudes[0], x, y);
    }
    finish(sums, errors, magnitudes, std::iter::empty(), a.len())
}

/// Whether the CPU has the fused multiply-add the float64 kernels use in
/// hardware.
fn fused() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        crate::ops::simd::simd_capabilities().avx2_fma
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

/// The float64 nearest `Σ a[i] · b[i]`.
///
/// Each product is split into the float64 nearest it and the exact error of
/// that, and the halves are carried as a sum's values are; the exact dot
/// decides where the bound cannot. A NaN, a product of an infinity and a
/// zero, or infinite products of both signs make NaN, and an infinite
/// product of one sign that infinity, as they would added up one by one;
/// finite operands give the value nearest their exact dot, an infinity past
/// the largest, though a product on the way would overflow.
pub(crate) fn dot_f64(a: &[f64], b: &[f64]) -> f64 {
    debug_assert_eq!(a.len(), b.len());
    let carried = f64::carried_dot(a, b);
    settled_dot(carried, a.len(), a.iter().copied().zip(b.iter().copied()))
}

/// The float64 nearest the dot of the `count` `pairs` that `carried`
/// carries, when the bound says which it is, and the exact one otherwise.
///
/// A product's error is exact unless the product lies within `2^-968` of
/// zero, where the error may have been rounded to a subnormal -- by at most
/// half the least subnormal a product, which is allowed for as the whole of
/// it. That allowance settles every total but one that is zero or nearly;
/// such a total is settled without it when no product of two nonzero
/// operands is that small.
fn settled_dot(
    carried: Carried,
    count: usize,
    pairs: impl Iterator<Item = (f64, f64)> + Clone,
) -> f64 {
    let allowing = Carried {
        slack: carried.slack + count as f64 * f64::from_bits(1),
        ..carried
    };
    allowing
        .rounded()
        .or_else(|| {
            let tiny = pairs
                .clone()
                .any(|(x, y)| x != 0.0 && y != 0.0 && (x * y).abs() < 2f64.powi(-968));
            if tiny { None } else { carried.rounded() }
        })
        .unwrap_or_else(|| exact_dot_f64(pairs))
}

/// The float64 nearest the exact dot of `pairs`; see [`dot_f64`].
#[cold]
#[inline(never)]
fn exact_dot_f64(pairs: impl Iterator<Item = (f64, f64)> + Clone) -> f64 {
    Total::of_products(pairs).map_or_else(|special| special, |total| total.rounded())
}

/// The value nearest `Σ values[i] · weights[i] / Σ weights[i]`.
///
/// The weighted sum and the weight total are carried as a dot and a sum
/// are, and their quotient formed with its residual and a bound; where the
/// bound cannot say which value is nearest, both totals are taken exactly
/// and divided in whole limbs. A NaN, or an infinity on either side, gives
/// what dividing the two sums in float64 gives; finite operands give the
/// value nearest the exact quotient though either total is past the
/// largest float64, an infinity of the numerator's sign over a weight total
/// that is exactly zero, and NaN for zero over zero.
pub(crate) fn weighted_mean_row<I: Element, T: Output>(values: &[I], weights: &[I]) -> T {
    debug_assert_eq!(values.len(), weights.len());
    let numerator = I::carried_dot(values, weights);
    let denominator = carried_with(weights, true);
    quotient_settled(numerator, denominator).unwrap_or_else(|| {
        let pairs = values
            .iter()
            .zip(weights)
            .map(|(&value, &weight)| (value.wide(), weight.wide()));
        exact_quotient(pairs, weights.iter().map(|weight| weight.wide()))
    })
}

/// The value nearest `Σ values[i] · weights[i] / Σ weights[i]` for int64
/// operands: each product exact in 128 bits, the totals in 128 bits while
/// they fit and in whole limbs past that, and their quotient rounded once.
pub(crate) fn weighted_mean_i64<T: Output>(values: &[i64], weights: &[i64]) -> T {
    let total = |magnitude: i128| {
        let mut total = Total::empty(0, 5);
        let limbs = if magnitude < 0 {
            &mut total.negative
        } else {
            &mut total.positive
        };
        add_shifted(limbs, magnitude.unsigned_abs(), 0);
        total
    };
    // Below 2^63 a weight, and so below 2^127 for any count of them.
    let denominator = weights.iter().map(|&w| i128::from(w)).sum::<i128>();
    let products = values
        .iter()
        .zip(weights)
        .map(|(&v, &w)| i128::from(v) * i128::from(w));
    let numerator = products
        .clone()
        .try_fold(0i128, |sum, product| sum.checked_add(product))
        .map_or_else(
            || {
                // Past 2^127: each product, below 2^126, into limbs.
                let mut sum = Total::empty(0, 5);
                for product in products {
                    let limbs = if product < 0 {
                        &mut sum.negative
                    } else {
                        &mut sum.positive
                    };
                    add_shifted(limbs, product.unsigned_abs(), 0);
                }
                sum
            },
            total,
        );
    numerator.quotient(total(denominator))
}

/// The value nearest `N / D` for the totals `numerator` and `denominator`
/// carry, when the bound says which it is.
///
/// `q = n / d` of the two totals' leading parts, the residual `N - q · D`
/// formed with the product `q · d` split exactly, and the quotient
/// `q + residual / D` within a bound covering the totals' own slack, the
/// residual's few roundings, and the divisor's low part and slack.
fn quotient_settled<T: Output>(numerator: Carried, denominator: Carried) -> Option<T> {
    let (n1, n2) = two_sum(numerator.sum, numerator.error);
    let (d1, d2) = two_sum(denominator.sum, denominator.error);
    let (sn, sd) = (numerator.slack, denominator.slack);
    let spread = d2.abs() + sd;
    // The divisor known to within a part in 2^30, which also turns away a
    // NaN or an infinity anywhere in it.
    let known = spread <= d1.abs() * 2f64.powi(-30);
    if !known || d1 == 0.0 {
        return None;
    }
    if n1 == 0.0 && n2 == 0.0 && sn == 0.0 {
        // Exactly nothing over a divisor clear of zero.
        return Some(T::narrow(0.0 / d1));
    }
    // Clear of underflow in the splits and of a quotient out of range.
    let ordinary = |value: f64| (2f64.powi(-900)..=2f64.powi(900)).contains(&value.abs());
    let quotient = n1 / d1;
    if !(ordinary(n1) && ordinary(d1) && ordinary(quotient)) {
        return None;
    }
    let product = quotient * d1;
    let product_rounding = product_error(quotient, d1, product);
    // Within two roundings of `n1`, so the difference is exact.
    let leading = n1 - product;
    let first = leading - product_rounding;
    let second = first + n2;
    let third = quotient * d2;
    let residual = second - third;
    // Each of the four rounded within `u` of its result, and the last
    // product, which may be a subnormal, within the least subnormal.
    let rounding = (first.abs() + second.abs() + third.abs() + residual.abs()) * 2f64.powi(-52)
        + f64::from_bits(4);
    let off = rounding + sn + quotient.abs() * sd;
    let correction = residual / d1;
    let divisor = d1.abs() * (1.0 - 2f64.powi(-29));
    let bound = ((off + residual.abs() * spread / d1.abs()) / divisor
        + correction.abs() * 2f64.powi(-52))
        * (1.0 + 2f64.powi(-40));
    let (value, rest) = two_sum(quotient, correction);
    T::settle(value, rest, bound)
}

/// The value nearest the exact `Σ x · y / Σ w` over `pairs` and `weights`;
/// see [`weighted_mean`].
#[cold]
#[inline(never)]
fn exact_quotient<T: Output>(
    pairs: impl Iterator<Item = (f64, f64)> + Clone,
    weights: impl Iterator<Item = f64> + Clone,
) -> T {
    match (Total::of_products(pairs), Total::of_wide(weights)) {
        (Ok(numerator), Ok(denominator)) => numerator.quotient(denominator),
        (numerator, denominator) => {
            // A special on one side or both: as float64 divides them, a
            // finite total standing as itself, or as the largest float64 of
            // its sign where it is past it -- any finite value does, beside
            // an infinity or a NaN.
            let standing = |total: Result<Total, f64>| match total {
                Err(special) => special,
                Ok(total) => total.rounded::<f64>().clamp(f64::MIN, f64::MAX),
            };
            T::narrow(standing(numerator) / standing(denominator))
        }
    }
}

/// The float32 nearest `sqrt(Σ values[i]²)`.
///
/// Each square of a float32 is exact in float64, so the squares are first
/// added plainly in float64, whose bound settles the root unless it reaches
/// a halfway point between two float32s; then the root is taken of the
/// carried sum as [`root_of`] takes it.
pub(crate) fn norm_f32(values: &[f32]) -> f32 {
    let block = crate::ops::simd::simd_square_sum_f32;
    let len = values.len();
    let sum = if len <= BLOCK {
        block(values)
    } else {
        let span = |nth: usize| nth * BLOCK..(nth * BLOCK + BLOCK).min(len);
        let partials = if len >= crate::ops::map::PAR_THRESHOLD {
            crate::parallel::map_tasks(len.div_ceil(BLOCK), &|nth| block(&values[span(nth)]))
        } else {
            (0..len.div_ceil(BLOCK))
                .map(|nth| block(&values[span(nth)]))
                .collect()
        };
        pairwise_fold(partials, 0.0, |p, q| p + q)
    };
    // Sixteen lanes merged pairwise a block, within the depth a plain sum
    // allows for.
    plain_root_f32(sum, plain_depth(len)).unwrap_or_else(|| {
        root_of(
            f32::carried_dot(values, values),
            values.iter().map(|&value| f64::from(value)),
        )
    })
}

/// The float32 nearest the root of a sum of squares that the plain float64
/// `sum` is within `depth` roundings of, when the bound says which it is.
fn plain_root_f32(sum: f64, depth: f64) -> Option<f32> {
    if sum == 0.0 {
        // No square of a float32 vanishes in float64, so every value is zero.
        return Some(0.0);
    }
    // The squares' magnitudes are the squares.
    let bound = sum * depth * 2f64.powi(-52);
    if !(sum.is_finite() && sum > 2.0 * bound) {
        return None;
    }
    // The root moves by at most the bound over twice the least root within
    // it, and is rounded once itself.
    let root = sum.sqrt();
    let moved = bound / (2.0 * (sum - bound).sqrt()) + root * 2f64.powi(-52);
    f32::settle(root, 0.0, moved * (1.0 + 2f64.powi(-40)))
}

/// Columns of a block of rows a task of [`column_norms`] takes at once.
const NORM_COLUMNS: usize = 256;

/// `out[o · inner + j]`, the value nearest `sqrt(Σ_r x[o][r][j]²)` for
/// `values` laid out `[outer, len, inner]`: the norms along a middle axis,
/// read in place a tile of columns at a time rather than from a transposed
/// copy.
///
/// Float32 squares are added plainly in float64 a block of rows at a time,
/// the blocks merged pairwise, and float64 ones carried as [`norm_f64`]
/// carries them; a column the bound cannot settle is gathered and taken as
/// [`norm_f32`] or [`norm_f64`] takes a run.
pub(crate) fn column_norms<T: Squares>(
    values: &[T],
    outer: usize,
    len: usize,
    inner: usize,
    out: &mut [T],
) {
    let tiles = inner.div_ceil(NORM_COLUMNS);
    let task = |index: usize| -> Vec<T> {
        let (slab, first) = (index / tiles, index % tiles * NORM_COLUMNS);
        let width = NORM_COLUMNS.min(inner - first);
        let slab = &values[slab * len * inner..(slab + 1) * len * inner];
        let blocks: Vec<T::Partial> = (0..len.div_ceil(BLOCK).max(1))
            .map(|block| {
                let rows = block * BLOCK..(block * BLOCK + BLOCK).min(len);
                T::partial(rows.map(|row| &slab[row * inner + first..][..width]), width)
            })
            .collect();
        let merged = T::merge(blocks, width);
        (0..width)
            .map(|j| {
                let column = (0..len).map(|row| slab[row * inner + first + j]);
                T::settled(&merged, j, len, column)
            })
            .collect()
    };
    let tasks = outer * tiles;
    let parts: Vec<Vec<T>> = if tasks > 1 && outer * len * inner >= crate::ops::map::PAR_THRESHOLD {
        crate::parallel::map_tasks(tasks, &task)
    } else {
        (0..tasks).map(&task).collect()
    };
    for (index, part) in parts.into_iter().enumerate() {
        let (slab, first) = (index / tiles, index % tiles * NORM_COLUMNS);
        out[slab * inner + first..][..part.len()].copy_from_slice(&part);
    }
}

/// A float dtype [`column_norms`] reads: what it keeps of each column over
/// a block of rows, how blocks merge, and the root of the merged total.
pub(crate) trait Squares: Copy + Send + Sync + 'static {
    type Partial: Send;
    fn partial<'a>(rows: impl Iterator<Item = &'a [Self]>, width: usize) -> Self::Partial;
    fn merge(blocks: Vec<Self::Partial>, width: usize) -> Self::Partial;
    fn settled(
        merged: &Self::Partial,
        j: usize,
        len: usize,
        column: impl Iterator<Item = Self> + Clone,
    ) -> Self;
}

impl Squares for f32 {
    /// Each column's plain float64 sum of squares.
    type Partial = Vec<f64>;

    fn partial<'a>(rows: impl Iterator<Item = &'a [f32]>, width: usize) -> Vec<f64> {
        let mut sums = vec![0f64; width];
        for row in rows {
            for (sum, &value) in sums.iter_mut().zip(row) {
                let wide = f64::from(value);
                *sum += wide * wide;
            }
        }
        sums
    }

    fn merge(blocks: Vec<Vec<f64>>, width: usize) -> Vec<f64> {
        (0..width)
            .map(|j| {
                pairwise_fold(
                    blocks.iter().map(|block| block[j]).collect(),
                    0.0,
                    |p, q| p + q,
                )
            })
            .collect()
    }

    fn settled(
        merged: &Vec<f64>,
        j: usize,
        len: usize,
        column: impl Iterator<Item = f32> + Clone,
    ) -> f32 {
        // A block of rows in one chain, and the merges of up to 2^64 blocks.
        let depth = (len.min(BLOCK) + 2 + 64) as f64;
        plain_root_f32(merged[j], depth).unwrap_or_else(|| norm_f32(&column.collect::<Vec<_>>()))
    }
}

impl Squares for f64 {
    /// Each column's carried sum of squares.
    type Partial = Vec<Carried>;

    fn partial<'a>(rows: impl Iterator<Item = &'a [f64]>, width: usize) -> Vec<Carried> {
        let mut lanes = Columns {
            sums: vec![0.0; width],
            errors: vec![0.0; width],
            magnitudes: vec![0.0; width],
        };
        let mut count = 0usize;
        let fused = fused();
        for row in rows {
            count += 1;
            #[cfg(target_arch = "x86_64")]
            if fused {
                // SAFETY: avx2 and fma were detected on this CPU.
                unsafe { add_column_squares_fma(&mut lanes, row) };
                continue;
            }
            add_column_squares(&mut lanes, row);
        }
        // A column's error terms pass through a pairing and the chain.
        let additions = (count + 2) as f64;
        (0..width)
            .map(|j| Carried {
                sum: lanes.sums[j],
                error: lanes.errors[j],
                slack: lanes.magnitudes[j] * additions * 2f64.powi(-52),
            })
            .collect()
    }

    fn merge(blocks: Vec<Vec<Carried>>, width: usize) -> Vec<Carried> {
        (0..width)
            .map(|j| {
                pairwise_fold(
                    blocks.iter().map(|block| block[j]).collect(),
                    Carried::default(),
                    Carried::merge,
                )
            })
            .collect()
    }

    fn settled(
        merged: &Vec<Carried>,
        j: usize,
        _len: usize,
        column: impl Iterator<Item = f64> + Clone,
    ) -> f64 {
        root_of(merged[j], column)
    }
}

/// [`add_product`] of each value of `row` with itself into its column.
#[inline(always)]
fn add_column_squares(lanes: &mut Columns, row: &[f64]) {
    let width = lanes.sums.len();
    let (sums, errors, magnitudes) = (
        &mut lanes.sums[..width],
        &mut lanes.errors[..width],
        &mut lanes.magnitudes[..width],
    );
    let row = &row[..width];
    for j in 0..width {
        add_product(
            &mut sums[j],
            &mut errors[j],
            &mut magnitudes[j],
            row[j],
            row[j],
        );
    }
}

/// [`add_column_squares`] with fused multiply-adds in hardware.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
fn add_column_squares_fma(lanes: &mut Columns, row: &[f64]) {
    add_column_squares(lanes, row)
}

/// The float64 nearest `sqrt(Σ values[i]²)`; see [`root_of`].
pub(crate) fn norm_f64(values: &[f64]) -> f64 {
    root_of(f64::carried_dot(values, values), values.iter().copied())
}

/// The value nearest the root of the sum of the squares of `values` that
/// `carried` carries.
///
/// With the allowance [`settled_dot`] makes for squares whose errors may have
/// been rounded to subnormals, and without it where no square is that small;
/// the exact sum of squares is rooted in whole numbers where neither settles
/// it. A NaN gives NaN, and an infinity otherwise infinity.
fn root_of<T: Output>(carried: Carried, values: impl Iterator<Item = f64> + Clone) -> T {
    let count = values.clone().count() as f64;
    let allowing = Carried {
        slack: carried.slack + count * f64::from_bits(1),
        ..carried
    };
    root_settled(allowing)
        .or_else(|| {
            let tiny = values
                .clone()
                .any(|x| x != 0.0 && (x * x).abs() < 2f64.powi(-968));
            if tiny { None } else { root_settled(carried) }
        })
        .unwrap_or_else(|| exact_root(values))
}

/// The value nearest the root of the total `carried` carries, when the
/// bound says which it is.
///
/// `r = sqrt(s)` of the total's leading part, the residual `S - r²` with `r²`
/// split exactly, and `r + residual / 2r` within a bound covering the
/// total's slack, the residual's roundings, and the curvature the step
/// leaves out, at most `residual² / 8r³`.
fn root_settled<T: Output>(carried: Carried) -> Option<T> {
    let (s1, s2) = two_sum(carried.sum, carried.error);
    if s1 == 0.0 && s2 == 0.0 && carried.slack == 0.0 {
        return Some(T::narrow(0.0));
    }
    // Clear of underflow in the square's split; a NaN, an infinity or a
    // negative total fails this too.
    let ordinary = (2f64.powi(-900)..=2f64.powi(900)).contains(&s1);
    let known = s2.abs() + carried.slack <= s1 * 2f64.powi(-30);
    if !(ordinary && known) {
        return None;
    }
    let root = s1.sqrt();
    let square = root * root;
    let square_rounding = product_error(root, root, square);
    // Within three roundings of `s1`, so the difference is exact.
    let first = (s1 - square) - square_rounding;
    let residual = first + s2;
    let off = (first.abs() + residual.abs()) * 2f64.powi(-52) + f64::from_bits(4) + carried.slack;
    let correction = residual / (2.0 * root);
    let ratio = (residual.abs() + off) / root;
    let bound = (off / (2.0 * root)
        + ratio * ratio / (8.0 * root) * 1.01
        + correction.abs() * 2f64.powi(-52))
        * (1.0 + 2f64.powi(-40));
    let (value, rest) = two_sum(root, correction);
    T::settle(value, rest, bound)
}

/// The value nearest the exact root of the sum of the squares of `values`.
#[cold]
#[inline(never)]
fn exact_root<T: Output>(values: impl Iterator<Item = f64> + Clone) -> T {
    match Total::of_products(values.map(|value| (value, value))) {
        // A NaN, or an infinity: a square is never a negative one.
        Err(special) => T::narrow(special.sqrt()),
        Ok(total) => total.square_root(),
    }
}

/// The float32 nearest `sqrt(x² + y²)`.
///
/// The squares of float32s are exact in float64, so the float64 root of
/// their float64 sum is within two of its ulps of the exact root. That
/// rounds to float32 as the exact root does unless the bits it keeps below a
/// float32's last place lie within a few ulps of halfway, or the root is a
/// float32 subnormal, whose last place lies higher; then the root is taken
/// as [`hypot_of`] takes it.
#[inline(always)]
pub(crate) fn hypot_f32(x: f32, y: f32) -> f32 {
    let (x, y) = (f64::from(x), f64::from(y));
    let root = (x * x + y * y).sqrt();
    // The 29 bits below a float32's last place, against a halfway point's.
    let below = (root.to_bits() & ((1 << 29) - 1)) as i64;
    let clear = (below - (1 << 28)).abs() > 8;
    if clear && (root >= f64::from(f32::MIN_POSITIVE) || root == 0.0) {
        return root as f32;
    }
    hypot_of(x, y)
}

/// The float64 nearest `sqrt(x² + y²)`, each rounding error of a product
/// taken by a fused multiply-add where `FMA` and from halves otherwise.
///
/// Where the smaller magnitude is under `2^-28` of the larger, the exact
/// root exceeds the larger by far less than half its ulp, so the larger is
/// the answer. Otherwise both are scaled by the power of two that brings the
/// larger into `[4, 8)` -- exactly, since neither leaves the normal range --
/// rooted by [`scaled_hypot`], or exactly where its bound cannot say, and
/// the power of two taken back out, exactly unless the root passes the
/// largest float64, where the infinity is the one the exact root rounds to.
/// Operands under `2^-1020`, whose root may be subnormal and so rounded
/// again by that last step, are rooted exactly as they stand.
#[inline(always)]
pub(crate) fn hypot_f64<const FMA: bool>(x: f64, y: f64) -> f64 {
    let (x, y) = (x.abs(), y.abs());
    let (big, small) = if x < y { (y, x) } else { (x, y) };
    if !(big <= f64::MAX && small <= f64::MAX) {
        // An infinity or a NaN.
        return hypot_of(x, y);
    }
    if small < big * 2f64.powi(-28) || small == 0.0 {
        return big;
    }
    if big < 2f64.powi(-1020) {
        return exact_root([x, y].into_iter());
    }
    // For the exponent `e` of `big`, `2^(2 - e)` and `2^(e - 2)`, both
    // normal for a biased exponent from 3 to 2046.
    let biased = big.to_bits() >> 52;
    let down = f64::from_bits((2048 - biased) << 52);
    let up = f64::from_bits((biased - 2) << 52);
    let (a, b) = (big * down, small * down);
    scaled_hypot::<FMA>(a, b).unwrap_or_else(|| exact_root([a, b].into_iter())) * up
}

/// The float64 nearest `sqrt(a² + b²)` for `a` in `[4, 8)` and `b` from
/// `2^-28 a` to `a`, when the bound says which it is.
///
/// `r = sqrt(s)` of the squares' float64 sum `s`, the residual
/// `a² + b² - r²` from the exact errors of both squares, of `s` and of
/// `r²`, and `r + residual / 2r`. Every one of those errors is far inside
/// the normal range and the residual within `2^-50` of `a² + b²`, so the
/// step is within `1.5 · 2^-103 r` of the exact root -- the residual's
/// roundings, the quotient's, and the curvature `residual² / 8r³` the step
/// leaves out -- and `2^-98 r` bounds it.
#[inline(always)]
fn scaled_hypot<const FMA: bool>(a: f64, b: f64) -> Option<f64> {
    let error = |x: f64, y: f64, product: f64| {
        if FMA {
            x.mul_add(y, -product)
        } else {
            product_error(x, y, product)
        }
    };
    let (p, q) = (a * a, b * b);
    let (s, s_error) = two_sum(p, q);
    let tail = s_error + (error(a, a, p) + error(b, b, q));
    let r = s.sqrt();
    let square = r * r;
    // Within a few roundings of `s`, so the first difference is exact.
    let residual = ((s - square) - error(r, r, square)) + tail;
    let (value, rest) = two_sum(r, residual / (2.0 * r));
    f64::settle(value, rest, r * 2f64.powi(-98))
}

/// The value nearest `sqrt(x² + y²)` for float64s whose squares stay inside
/// the float64 range with their roundings' errors whole multiples of the
/// least subnormal -- any pair of float32s or of integers -- by each square
/// split exactly and the root of the four parts settled as [`root_settled`]
/// settles a sum's, or taken exactly where the bound cannot say. An infinity
/// gives `+inf` even beside a NaN, and a NaN otherwise NaN.
pub(crate) fn hypot_of<T: Output>(x: f64, y: f64) -> T {
    if x.is_infinite() || y.is_infinite() {
        return T::narrow(f64::INFINITY);
    }
    if x.is_nan() || y.is_nan() {
        return T::narrow(f64::NAN);
    }
    let (p, q) = (x * x, y * y);
    let (p_error, q_error) = (product_error(x, x, p), product_error(y, y, q));
    let (sum, rest) = two_sum(p, q);
    let errors = p_error + q_error;
    let error = rest + errors;
    // Each of the two float64 additions rounds within `u` of its result.
    let slack = (errors.abs() + error.abs()) * 2f64.powi(-52);
    root_settled(Carried { sum, error, slack }).unwrap_or_else(|| exact_root([x, y].into_iter()))
}

/// The float32 nearest `a · b - c · d`.
///
/// Each product of two float32s is exact in float64, and their difference a
/// two-sum's pair; rounded to odd at float64's last bit, which keeps whether
/// anything lay below it, the pair narrows to float32 rounded once. An
/// infinity or a NaN gives what float64 arithmetic gives.
#[inline(always)]
pub(crate) fn difference_of_products_f32(a: f32, b: f32, c: f32, d: f32) -> f32 {
    let (left, right) = (f64::from(a) * f64::from(b), f64::from(c) * f64::from(d));
    let (sum, rest) = two_sum(left, -right);
    if rest == 0.0 || !sum.is_finite() {
        return sum as f32;
    }
    // The exact value lies past `sum` towards `rest`, short of the next
    // float64, so the odd one of the two is its rounding to odd.
    let odd = if sum.to_bits() & 1 == 1 {
        sum
    } else if rest > 0.0 {
        sum.next_up()
    } else {
        sum.next_down()
    };
    odd as f32
}

/// The float64 nearest `a · b - c · d`.
///
/// Each product is split by a fused multiply-add into the float64 nearest it
/// and that rounding's error, and the four parts added with a bound on the
/// two roundings that takes; where the bound reaches a halfway point, the
/// two products are summed exactly. Where a product lies within `2^-968` of
/// zero its error may itself have been rounded, by under half the least
/// subnormal, and the bound allows for that.
#[inline(always)]
pub(crate) fn difference_of_products_f64(a: f64, b: f64, c: f64, d: f64) -> f64 {
    let (left, right) = (a * b, c * d);
    let (left_error, right_error) = (a.mul_add(b, -left), c.mul_add(d, -right));
    let (sum, rest) = two_sum(left, -right);
    let errors = left_error - right_error;
    let exact_errors =
        |product: f64, x: f64, y: f64| product.abs() >= 2f64.powi(-968) || x == 0.0 || y == 0.0;
    let exact = exact_errors(left, a, b) && exact_errors(right, c, d);
    if sum == 0.0 && exact {
        // The products are equal, so the difference is their errors'.
        return errors;
    }
    let tail = rest + errors;
    let allowance = if exact { 0.0 } else { f64::from_bits(2) };
    let bound = (errors.abs() + tail.abs()) * 2f64.powi(-52) + allowance;
    let (value, low) = two_sum(sum, tail);
    f64::settle(value, low, bound).unwrap_or_else(|| exact_difference(a, b, c, d))
}

/// The float64 nearest the exact `a · b - c · d`; see
/// [`difference_of_products_f64`].
#[cold]
#[inline(never)]
fn exact_difference(a: f64, b: f64, c: f64, d: f64) -> f64 {
    Total::of_products([(a, b), (-c, d)].into_iter())
        .map_or_else(|special| special, |total| total.rounded())
}

/// The running sums along one axis of `input`, laid out
/// `[outer, dim_size, inner]` and summed from the far end when `reverse`:
/// `output` at each position, the value nearest the exact sum of the values
/// up to it.
///
/// Each lane carries its total as `sum + error` by a two-sum a step, and adds
/// each step's error into `error` by a two-sum too, keeping the magnitudes of
/// what that drops as the bound: zero while the errors add exactly, which on
/// ordinary data is most of the time, and far below an ulp of the sum
/// otherwise. A pair on a halfway point with a zero bound is rounded by its
/// own two-sum, ties to even. A prefix the bound cannot settle is rounded
/// from the exact sum of the lane so far, kept in limbs from the first such
/// prefix on and extended only as far as it is asked for, and the lane
/// carries on from that. A NaN, or infinities of both signs, make NaN from
/// there on and an infinity of one sign that infinity; an exact zero is `-0`
/// where every value so far is, as adding them one by one gives, and `+0`
/// otherwise.
///
/// Every running sum is the one value nearest the exact sum, so how the
/// work is split cannot change it: a long run is cut into pieces across the
/// pool, short runs and columns are advanced side by side in bands.
pub(crate) fn running_sums<I: Element, T: Output>(
    output: &mut [T],
    input: &[I],
    dim_size: usize,
    inner: usize,
    reverse: bool,
) {
    let slab = dim_size * inner;
    if slab == 0 {
        return;
    }
    debug_assert_eq!(output.len(), input.len());
    let slabs = input.len() / slab;
    let order = move |step: usize| if reverse { dim_size - 1 - step } else { step };
    if inner == 1 && dim_size >= crate::ops::map::PAR_THRESHOLD {
        for (out, values) in output.chunks_mut(slab).zip(input.chunks(slab)) {
            split_running_sums(out, values, reverse);
        }
        return;
    }
    if inner == 1 && slabs == 1 {
        scan_run(
            output,
            Lane::new::<T>(),
            0..dim_size,
            |step| input[order(step)].wide(),
            order,
            || Exactly::of(std::iter::empty()),
        );
        return;
    }
    if inner == 1 {
        // Runs side by side, a band of them to a task.
        crate::ops::map::par_out_chunks(output, slab * BAND, &|start, out| {
            let values = &input[start..start + out.len()];
            scan_band(
                out.len() / slab,
                dim_size,
                |step, lanes| {
                    for (k, lane) in lanes.iter_mut().enumerate() {
                        *lane = values[k * slab + order(step)].wide();
                    }
                },
                |k, step| values[k * slab + order(step)].wide(),
                |step, sums| {
                    for (k, &sum) in sums.iter().enumerate() {
                        out[k * slab + order(step)] = sum;
                    }
                },
            );
        });
        return;
    }
    // Columns a band of one slab to a task, written where they belong.
    let bands = inner.div_ceil(BAND);
    let parts = crate::parallel::Parts::of(output);
    crate::parallel::for_each_task(slabs * bands, &|task| {
        let (base, first) = (task / bands * slab, task % bands * BAND);
        let values = &input[base..base + slab];
        let width = BAND.min(inner - first);
        let row = |step: usize| order(step) * inner + first;
        scan_band(
            width,
            dim_size,
            |step, lanes| {
                for (lane, value) in lanes.iter_mut().zip(&values[row(step)..][..width]) {
                    *lane = value.wide();
                }
            },
            |k, step| values[row(step) + k].wide(),
            |step, sums| {
                // SAFETY: a task writes only the columns of its band of its
                // slab, which no other task writes, and all lie in `output`.
                let out = unsafe {
                    std::slice::from_raw_parts_mut(parts.start().add(base + row(step)), width)
                };
                out.copy_from_slice(sums);
            },
        );
    });
}

/// Lanes [`scan_band`] advances side by side.
const BAND: usize = 64;

/// The running sums of `width` lanes, at most [`BAND`], advanced a step at
/// a time side by side: `load(step, values)` gives each lane's value at a
/// step, `read(k, step)` lane `k`'s at an earlier one, and `store(step,
/// sums)` takes the running sums. Each lane is carried; a lane its bound
/// cannot settle is rounded from its exact sum, as [`scan_run`] rounds one.
fn scan_band<T: Output>(
    width: usize,
    steps: usize,
    load: impl Fn(usize, &mut [f64]),
    read: impl Fn(usize, usize) -> f64,
    mut store: impl FnMut(usize, &[T]),
) {
    let mut band = Band {
        sums: [0.0; BAND],
        errors: [0.0; BAND],
        bounds: [0.0; BAND],
        zeros: [1; BAND],
    };
    let mut values = [0f64; BAND];
    let mut sums = [T::narrow(0.0); BAND];
    let mut settled = [0u8; BAND];
    let mut exact: Vec<Option<Box<Exactly>>> = (0..width).map(|_| None).collect();
    let wide = {
        #[cfg(target_arch = "x86_64")]
        {
            crate::ops::simd::simd_capabilities().avx2
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            false
        }
    };
    for step in 0..steps {
        load(step, &mut values[..width]);
        #[cfg(target_arch = "x86_64")]
        if wide {
            // SAFETY: avx2 was detected on this CPU.
            unsafe { advance_avx2(&mut band, &values, &mut sums, &mut settled, width) };
        } else {
            advance(&mut band, &values, &mut sums, &mut settled, width);
        }
        #[cfg(not(target_arch = "x86_64"))]
        advance(&mut band, &values, &mut sums, &mut settled, width);
        for k in 0..width {
            if settled[k] != 0 {
                continue;
            }
            let mut lane = Lane {
                sum: band.sums[k],
                error: band.errors[k],
                bound: band.bounds[k],
                carried: true,
                negative_zero: band.zeros[k] != 0,
            };
            let state = exact[k].get_or_insert_with(|| Exactly::of(std::iter::empty()));
            sums[k] = exactly(state, &mut lane, step, |earlier| read(k, earlier));
            band.sums[k] = lane.sum;
            band.errors[k] = lane.error;
            band.bounds[k] = lane.bound;
        }
        store(step, &sums[..width]);
    }
}

/// [`BAND`] carried lanes, each field apart, so that a step over them is
/// one loop the compiler can carry out in vector registers.
struct Band {
    sums: [f64; BAND],
    errors: [f64; BAND],
    /// The sums of the magnitudes of what adding up the errors dropped.
    bounds: [f64; BAND],
    /// Whether every value so far was `-0`, as 0 or 1.
    zeros: [u8; BAND],
}

/// Adds `values` to the first `width` lanes, each as [`Lane::add`] adds to a
/// carried lane, and gives each running sum in `sums` with whether the
/// bound settles it in `settled`, without a branch.
#[inline(always)]
fn advance<T: Output>(
    band: &mut Band,
    values: &[f64; BAND],
    sums: &mut [T; BAND],
    settled: &mut [u8; BAND],
    width: usize,
) {
    let width = width.min(BAND);
    let mut exact = true;
    for k in 0..width {
        let value = values[k];
        band.zeros[k] &= u8::from(value.to_bits() == (-0.0f64).to_bits());
        let (sum, rounding) = two_sum(band.sums[k], value);
        band.sums[k] = sum;
        let (error, dropped) = two_sum(band.errors[k], rounding);
        band.errors[k] = error;
        let bound = band.bounds[k] + dropped.abs();
        band.bounds[k] = bound;
        exact &= (bound == 0.0) & T::exact_lane(error);
    }
    // Every value so far `-0` makes `-0`, which the sum's `+0` is not.
    if exact {
        // Every lane's pair is its exact sum: rounded as it stands.
        for k in 0..width {
            let rounded = T::exact_pair(band.sums[k], band.errors[k]);
            sums[k] = if band.zeros[k] != 0 {
                T::narrow(-0.0)
            } else {
                rounded
            };
            settled[k] = 1;
        }
        return;
    }
    for k in 0..width {
        let (total, rest) = two_sum(band.sums[k], band.errors[k]);
        let (rounded, ok) = T::settle_lanes(total, rest, 2.0 * band.bounds[k]);
        sums[k] = if band.zeros[k] != 0 {
            T::narrow(-0.0)
        } else {
            rounded
        };
        settled[k] = u8::from(ok);
    }
}

/// [`advance`] in 256-bit registers.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
fn advance_avx2<T: Output>(
    band: &mut Band,
    values: &[f64; BAND],
    sums: &mut [T; BAND],
    settled: &mut [u8; BAND],
    width: usize,
) {
    advance(band, values, sums, settled, width)
}

/// `out[order(step)]` for the running sums of one lane over `steps`, from
/// `lane` as it stands before them; `read(step)` is the value at a step and
/// `before` the exact state of everything before the first, built only if a
/// running sum needs it.
#[inline(always)]
fn scan_run<T: Output>(
    out: &mut [T],
    mut lane: Lane,
    steps: std::ops::Range<usize>,
    read: impl Fn(usize) -> f64,
    order: impl Fn(usize) -> usize,
    before: impl FnOnce() -> Box<Exactly>,
) {
    let first = steps.start;
    let mut exact: Option<Box<Exactly>> = None;
    let mut before = Some(before);
    for step in steps {
        let at = order(step);
        out[at] = lane.add::<T>(read(step)).unwrap_or_else(|| {
            let state = exact.get_or_insert_with(|| before.take().expect("built once")());
            exactly(state, &mut lane, step - first, |earlier| {
                read(first + earlier)
            })
        });
    }
}

/// [`running_sums`] of one long run, in pieces across the pool: each
/// piece's carried total, as a scan of it carries it, merged in scan order
/// into what comes before each piece, and each piece scanned from that. A
/// piece that needs the exact sum of what comes before it adds up the exact
/// sums of the pieces before it, each taken once and kept.
fn split_running_sums<I: Element, T: Output>(output: &mut [T], input: &[I], reverse: bool) {
    const PIECE: usize = 1 << 16;
    let len = input.len();
    let pieces = len.div_ceil(PIECE);
    let span = |piece: usize| piece * PIECE..(piece * PIECE + PIECE).min(len);
    let scan_values = |piece: usize| {
        let values = &input[span(piece)];
        let count = values.len();
        (0..count).map(move |step| values[if reverse { count - 1 - step } else { step }].wide())
    };
    let totals = crate::parallel::map_tasks(pieces, &|piece| {
        let mut lane = Lane::new::<f64>();
        for value in scan_values(piece) {
            lane.carry(value);
        }
        lane
    });
    // What comes before each piece in scan order.
    let scan_order: Vec<usize> = if reverse {
        (0..pieces).rev().collect()
    } else {
        (0..pieces).collect()
    };
    let mut before = vec![Lane::new::<f64>(); pieces];
    let mut so_far = Lane::new::<f64>();
    for &piece in &scan_order {
        before[piece] = so_far;
        so_far = so_far.merge(&totals[piece]);
    }
    let exact_pieces: Vec<std::sync::OnceLock<Box<Exactly>>> =
        (0..pieces).map(|_| std::sync::OnceLock::new()).collect();
    crate::ops::map::par_out_chunks(output, PIECE, &|start, out| {
        let piece = start / PIECE;
        let values = &input[span(piece)];
        let count = values.len();
        let order = |step: usize| if reverse { count - 1 - step } else { step };
        let preceding = || {
            let mut state = Exactly::of(std::iter::empty());
            let earlier = scan_order.iter().take_while(|&&other| other != piece);
            for &other in earlier {
                let exact = exact_pieces[other].get_or_init(|| Exactly::of(scan_values(other)));
                state.absorb(exact);
            }
            state
        };
        scan_run(
            out,
            before[piece],
            0..count,
            |step| values[order(step)].wide(),
            order,
            preceding,
        );
    });
}

/// One lane of [`running_sums`].
///
/// A float32 lane starts plain: one float64 sum, each addition within `u`
/// of the total it makes, so `bound` is the sum of the totals' magnitudes
/// and `u` times it bounds the drift -- far below a float32 halfway point
/// unless the values cancel. At the first prefix that bound cannot settle
/// the lane turns carried, from the exact sum so far, which every float64
/// lane is from the start.
#[derive(Clone, Copy)]
struct Lane {
    sum: f64,
    error: f64,
    /// For a plain lane the sum of the totals' magnitudes; for a carried
    /// one, of the magnitudes of what adding up the errors dropped.
    bound: f64,
    carried: bool,
    /// Whether every value so far was `-0`.
    negative_zero: bool,
}

impl Lane {
    fn new<T: Output>() -> Self {
        Self {
            sum: 0.0,
            error: 0.0,
            bound: 0.0,
            carried: !T::PLAIN,
            negative_zero: true,
        }
    }

    /// Adds `value` to a carried lane, without giving the running sum.
    #[inline(always)]
    fn carry(&mut self, value: f64) {
        self.negative_zero &= value.to_bits() == (-0.0f64).to_bits();
        let (sum, rounding) = two_sum(self.sum, value);
        self.sum = sum;
        let (error, dropped) = two_sum(self.error, rounding);
        self.error = error;
        self.bound += dropped.abs();
    }

    /// The carried total of this lane's values followed by `next`'s, with
    /// what adding them drops added to the bound.
    fn merge(&self, next: &Self) -> Self {
        let (sum, rounding) = two_sum(self.sum, next.sum);
        let (errors, first) = two_sum(self.error, next.error);
        let (error, second) = two_sum(errors, rounding);
        Self {
            sum,
            error,
            bound: self.bound + next.bound + first.abs() + second.abs(),
            carried: true,
            negative_zero: self.negative_zero && next.negative_zero,
        }
    }

    /// Adds `value`, and gives the running sum when the bound says which
    /// value it is.
    #[inline(always)]
    fn add<T: Output>(&mut self, value: f64) -> Option<T> {
        if !self.carried {
            self.negative_zero &= value.to_bits() == (-0.0f64).to_bits();
            let sum = self.sum + value;
            self.sum = sum;
            self.bound += sum.abs();
            if sum == 0.0 && self.bound == 0.0 {
                return Some(T::narrow(if self.negative_zero { -0.0 } else { 0.0 }));
            }
            // Twice `u` covers the roundings of the bound's own sum.
            return T::settle(sum, 0.0, self.bound * 2f64.powi(-52));
        }
        self.carry(value);
        if self.bound == 0.0 && T::exact_lane(self.error) {
            // The pair is the exact sum: rounded as it stands.
            if self.negative_zero {
                return Some(T::narrow(-0.0));
            }
            return Some(T::exact_pair(self.sum, self.error));
        }
        let (value, rest) = two_sum(self.sum, self.error);
        // Twice the float64 sum of the magnitudes covers its own roundings.
        T::settle(value, rest, 2.0 * self.bound)
    }
}

/// A lane's sum so far in limbs, and the specials among its values.
struct Exactly {
    total: Total,
    done: usize,
    nan: bool,
    up: bool,
    down: bool,
}

impl Exactly {
    /// The exact state of `values`, before any step of a lane.
    fn of(values: impl Iterator<Item = f64>) -> Box<Self> {
        let mut state = Box::new(Self {
            total: Total::empty(LEAST, (2098 + 64) / 64 + 2),
            done: 0,
            nan: false,
            up: false,
            down: false,
        });
        for value in values {
            state.add(value);
        }
        state
    }

    fn add(&mut self, value: f64) {
        self.nan |= value.is_nan();
        self.up |= value == f64::INFINITY;
        self.down |= value == f64::NEG_INFINITY;
        if let Some(exact) = Exact::float(value).filter(|exact| exact.magnitude != 0) {
            let limbs = if exact.negative {
                &mut self.total.negative
            } else {
                &mut self.total.positive
            };
            add_shifted(
                limbs,
                u128::from(exact.magnitude),
                (exact.exponent - LEAST) as usize,
            );
        }
    }

    /// Adds another state's values to this one's, limb by limb.
    fn absorb(&mut self, other: &Self) {
        add_limbs(&mut self.total.positive, &other.total.positive);
        add_limbs(&mut self.total.negative, &other.total.negative);
        self.nan |= other.nan;
        self.up |= other.up;
        self.down |= other.down;
    }
}

/// `total += other`, limbs of the same length with room for the carry.
fn add_limbs(total: &mut [u64], other: &[u64]) {
    let mut carry = false;
    for (limb, &addend) in total.iter_mut().zip(other) {
        let (sum, first) = limb.overflowing_add(addend);
        let (sum, second) = sum.overflowing_add(u64::from(carry));
        *limb = sum;
        carry = first || second;
    }
}

/// `total -= value` for a finite float64.
fn take(total: &mut Total, value: f64) {
    let exact = Exact::float(value).expect("finite");
    if exact.magnitude != 0 {
        let limbs = if exact.negative {
            &mut total.positive
        } else {
            &mut total.negative
        };
        add_shifted(
            limbs,
            u128::from(exact.magnitude),
            (exact.exponent - LEAST) as usize,
        );
    }
}

/// The running sum of `lane` from the exact sum of what came before it and
/// its first `step + 1` values, which `earlier` reads in scan order, kept in
/// `state` and extended from where it was last asked for. The lane carries
/// on from the exact sum: its float64 nearest and the float64 nearest what
/// is left, with half an ulp of that as the bound unless the two hold the
/// sum exactly.
#[cold]
#[inline(never)]
fn exactly<T: Output>(
    state: &mut Exactly,
    lane: &mut Lane,
    step: usize,
    earlier: impl Fn(usize) -> f64,
) -> T {
    let negative_zero = lane.negative_zero;
    for index in state.done..=step {
        state.add(earlier(index));
    }
    state.done = step + 1;
    if state.nan || (state.up && state.down) {
        return T::narrow(f64::NAN);
    }
    if state.up || state.down {
        return T::narrow(if state.up {
            f64::INFINITY
        } else {
            f64::NEG_INFINITY
        });
    }
    // A sum of floats that is not zero is at least the least subnormal, and
    // so rounds to a value that is not zero either.
    if state.total.positive == state.total.negative {
        *lane = Lane {
            carried: true,
            negative_zero,
            ..Lane::new::<T>()
        };
        return T::narrow(if negative_zero { -0.0 } else { 0.0 });
    }
    let high: f64 = state.total.clone().rounded();
    if high.is_finite() {
        let mut rest = state.total.clone();
        take(&mut rest, high);
        let low: f64 = rest.clone().rounded();
        take(&mut rest, low);
        let exact = rest.positive == rest.negative;
        *lane = Lane {
            sum: high,
            error: low,
            // At most half an ulp of `low` is left beyond the two, and twice
            // the bound is what is allowed for.
            bound: if exact {
                0.0
            } else {
                low.abs() * 2f64.powi(-53)
            },
            carried: true,
            negative_zero,
        };
    }
    if T::PLAIN {
        // A float32 is rounded from the exact sum, not from `high`.
        state.total.clone().rounded()
    } else {
        // A float64 is `high` itself.
        T::narrow(high)
    }
}

/// Columns a task of [`dot_columns_f64`] carries at once.
const DOT_COLUMNS: usize = 256;

/// `out[j]`, the float64 nearest `Σ_s scales[s] · rows(s)[j]` for rows each
/// as long as `out`: a row times an upright matrix, every entry its dot
/// rounded once.
///
/// Each column is carried as [`dot_f64`] carries a dot, though in one chain
/// a block of rows long rather than in lanes: the rows of a block pass once
/// through a tile of columns, four at a time, and a column's blocks merge
/// after. A task takes one block of one tile, so the totals do not depend on
/// how the tasks ran.
pub(crate) fn dot_columns_f64<'a>(
    scales: &[f64],
    rows: &(dyn Fn(usize) -> &'a [f64] + Sync),
    out: &mut [f64],
) {
    let (k, n) = (scales.len(), out.len());
    let blocks = k.div_ceil(BLOCK).max(1);
    let fused = fused();
    let part = |task: usize| -> Vec<Carried> {
        let (first, block) = (task / blocks * DOT_COLUMNS, task % blocks);
        let width = DOT_COLUMNS.min(n - first);
        let steps = block * BLOCK..(block * BLOCK + BLOCK).min(k);
        let mut lanes = Columns {
            sums: vec![0.0; width],
            errors: vec![0.0; width],
            magnitudes: vec![0.0; width],
        };
        let mut start = steps.start;
        while start < steps.end {
            let count = (steps.end - start).min(4);
            let mut picked = [(0.0, &[][..]); 4];
            for (slot, step) in picked.iter_mut().zip(start..start + count) {
                *slot = (scales[step], &rows(step)[first..first + width]);
            }
            start += count;
            let picked = &picked[..count];
            #[cfg(target_arch = "x86_64")]
            if fused {
                // SAFETY: avx2 and fma were detected on this CPU.
                unsafe { add_column_products_fma(&mut lanes, picked) };
                continue;
            }
            add_column_products(&mut lanes, picked);
        }
        // A column's error terms pass through a pairing and the chain.
        let additions = (steps.len() + 2) as f64;
        (0..width)
            .map(|j| Carried {
                sum: lanes.sums[j],
                error: lanes.errors[j],
                slack: lanes.magnitudes[j] * additions * 2f64.powi(-52),
            })
            .collect()
    };
    let tasks = n.div_ceil(DOT_COLUMNS) * blocks;
    let parts: Vec<Vec<Carried>> = if tasks > 1 && k * n >= crate::ops::map::PAR_THRESHOLD {
        crate::parallel::map_tasks(tasks, &part)
    } else {
        (0..tasks).map(&part).collect()
    };
    for (tile, parts) in parts.chunks(blocks).enumerate() {
        let first = tile * DOT_COLUMNS;
        for (j, slot) in out[first..first + parts[0].len()].iter_mut().enumerate() {
            let carried = pairwise_fold(
                parts.iter().map(|part| part[j]).collect(),
                Carried::default(),
                Carried::merge,
            );
            let column = (0..k).map(|step| (scales[step], rows(step)[first + j]));
            *slot = settled_dot(carried, k, column);
        }
    }
}

/// A tile of columns' carried sums, each column's sum, error and magnitude
/// at the same index of the three.
struct Columns {
    sums: Vec<f64>,
    errors: Vec<f64>,
    magnitudes: Vec<f64>,
}

/// [`add_product`] for each column of a tile, from each of a few rows in
/// order.
#[inline(always)]
fn add_column_products(lanes: &mut Columns, rows: &[(f64, &[f64])]) {
    let width = lanes.sums.len();
    let (sums, errors, magnitudes) = (
        &mut lanes.sums[..width],
        &mut lanes.errors[..width],
        &mut lanes.magnitudes[..width],
    );
    match rows {
        [(s0, r0), (s1, r1), (s2, r2), (s3, r3)] => {
            let (r0, r1, r2, r3) = (&r0[..width], &r1[..width], &r2[..width], &r3[..width]);
            for j in 0..width {
                let (mut sum, mut error, mut magnitude) = (sums[j], errors[j], magnitudes[j]);
                add_product(&mut sum, &mut error, &mut magnitude, *s0, r0[j]);
                add_product(&mut sum, &mut error, &mut magnitude, *s1, r1[j]);
                add_product(&mut sum, &mut error, &mut magnitude, *s2, r2[j]);
                add_product(&mut sum, &mut error, &mut magnitude, *s3, r3[j]);
                (sums[j], errors[j], magnitudes[j]) = (sum, error, magnitude);
            }
        }
        _ => {
            for &(scale, row) in rows {
                let row = &row[..width];
                for j in 0..width {
                    add_product(
                        &mut sums[j],
                        &mut errors[j],
                        &mut magnitudes[j],
                        scale,
                        row[j],
                    );
                }
            }
        }
    }
}

/// [`add_column_products`] with fused multiply-adds in hardware.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
fn add_column_products_fma(lanes: &mut Columns, rows: &[(f64, &[f64])]) {
    add_column_products(lanes, rows)
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
#[derive(Clone)]
struct Total {
    positive: Vec<u64>,
    negative: Vec<u64>,
    /// The exponent of the least significant bit of limb `0`.
    least: i32,
}

/// The exponent of the least significant bit of a sum's limb `0`: the least
/// subnormal's.
const LEAST: i32 = -1074;

impl Total {
    /// Nothing yet, in `limbs` limbs from `2^least`.
    fn empty(least: i32, limbs: usize) -> Self {
        Self {
            positive: vec![0; limbs],
            negative: vec![0; limbs],
            least,
        }
    }

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
        let mut total = Self::empty(LEAST, limbs);
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

    /// The exact sum of the products of `pairs`, or the answer itself when
    /// an operand is not finite: a product with a NaN or of an infinity and
    /// a zero, or infinite products of both signs, make NaN, and infinite
    /// products of one sign that infinity.
    #[cold]
    #[inline(never)]
    fn of_products(pairs: impl Iterator<Item = (f64, f64)> + Clone) -> Result<Self, f64> {
        let (mut nan, mut up, mut down) = (false, false, false);
        for (x, y) in pairs.clone() {
            if !(x.is_finite() && y.is_finite()) {
                let product = x * y;
                nan |= product.is_nan();
                up |= product == f64::INFINITY;
                down |= product == f64::NEG_INFINITY;
            }
        }
        if nan || (up && down) {
            return Err(f64::NAN);
        }
        if up || down {
            return Err(if up { f64::INFINITY } else { f64::NEG_INFINITY });
        }
        // Every product of two finite float64s is a whole multiple of
        // `2^-2148` below `2^2048`, bit 4196 here; the carries of up to 2^64
        // products take 64 bits more.
        let least = 2 * LEAST;
        let mut total = Self::empty(least, (4196 + 64) / 64 + 2);
        for (x, y) in pairs {
            let (x, y) = (
                Exact::float(x).expect("finite"),
                Exact::float(y).expect("finite"),
            );
            let magnitude = u128::from(x.magnitude) * u128::from(y.magnitude);
            if magnitude != 0 {
                let limbs = if x.negative != y.negative {
                    &mut total.negative
                } else {
                    &mut total.positive
                };
                add_shifted(limbs, magnitude, (x.exponent + y.exponent - least) as usize);
            }
        }
        Ok(total)
    }

    /// The value nearest this sum over `divisor`'s: sixty-six bits of the
    /// quotient by long division, with whether a remainder is left marked
    /// below them. An infinity of this sum's sign over an exact zero, and NaN
    /// for zero over zero.
    fn quotient<T: Output>(self, divisor: Self) -> T {
        let (negative, numerator, least) = self.difference();
        let (divisor_negative, divisor, divisor_least) = divisor.difference();
        let bits = |limbs: &[u64]| {
            limbs
                .iter()
                .rposition(|&limb| limb != 0)
                .map_or(0, |top| 64 * top + 64 - limbs[top].leading_zeros() as usize)
        };
        let (top, bottom) = (bits(&numerator), bits(&divisor));
        if bottom == 0 {
            return T::narrow(match (top, negative) {
                (0, _) => f64::NAN,
                (_, true) => f64::NEG_INFINITY,
                (_, false) => f64::INFINITY,
            });
        }
        if top == 0 {
            return T::narrow(if divisor_negative { -0.0 } else { 0.0 });
        }
        // The dividend `bottom + 65` bits long against a divisor `bottom`
        // long, so that the quotient has 65 or 66 bits.
        let shift = bottom as i64 + 65 - top as i64;
        let (mut remainder, divisor) = if shift >= 0 {
            (shifted(&numerator, shift as usize), divisor)
        } else {
            (numerator, shifted(&divisor, (-shift) as usize))
        };
        let mut quotient = 0u128;
        for bit in (0..66).rev() {
            let step = shifted(&divisor, bit);
            if at_least(&remainder, &step) {
                subtract(&mut remainder, &step);
                quotient |= 1 << bit;
            }
        }
        let inexact = remainder.iter().any(|&limb| limb != 0);
        let exponent = i64::from(least) - i64::from(divisor_least) - shift - 1;
        rounded(
            negative != divisor_negative,
            (quotient << 1) | u128::from(inexact),
            exponent as i32,
        )
    }

    /// The value nearest the root of this sum, which is not negative: the
    /// leading 127 or 128 bits of its magnitude, at an even exponent, rooted
    /// as a whole number, with whether anything is left marked below.
    fn square_root<T: Output>(self) -> T {
        let (_, limbs, least) = self.difference();
        let Some(top) = limbs.iter().rposition(|&limb| limb != 0) else {
            return T::narrow(0.0);
        };
        let bits = 64 * top as i64 + 64 - i64::from(limbs[top].leading_zeros());
        // `magnitude · 2^shift` as the whole number to root, with
        // `least - shift` even.
        let mut shift = 128 - bits;
        if (i64::from(least) - shift).rem_euclid(2) != 0 {
            shift -= 1;
        }
        let (whole, dropped) = if shift >= 0 {
            let moved = shifted(&limbs, shift as usize);
            let low = |index: usize| u128::from(moved.get(index).copied().unwrap_or(0));
            (low(0) | (low(1) << 64), false)
        } else {
            let from = (-shift) as usize;
            let (word, bit) = (from / 64, (from % 64) as u32);
            let limb = |index: usize| u128::from(limbs.get(index).copied().unwrap_or(0));
            let window = (limb(word) | (limb(word + 1) << 64)) >> bit;
            let whole = if bit == 0 {
                window
            } else {
                window | (limb(word + 2) << (128 - bit))
            };
            let below = limbs[..word].iter().any(|&limb| limb != 0)
                || (bit != 0 && limbs[word] & ((1u64 << bit) - 1) != 0);
            (whole, below)
        };
        let root = whole.isqrt();
        let inexact = dropped || root * root != whole;
        let exponent = (i64::from(least) - shift) / 2 - 1;
        rounded(false, (root << 1) | u128::from(inexact), exponent as i32)
    }

    /// Whether the sum is negative, its magnitude, and the exponent of that
    /// magnitude's least significant bit.
    fn difference(self) -> (bool, Vec<u64>, i32) {
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
        (negative, larger, self.least)
    }

    /// The value nearest the sum; `+0` for a sum that cancels.
    fn rounded<T: Output>(self) -> T {
        let (negative, limbs, least) = self.difference();
        rounded_at(negative, &limbs, false, least)
    }

    /// The value nearest the sum over `count`, by long division with a limb
    /// of fraction below the lowest bit, so that a remainder left over is far
    /// below where the quotient rounds.
    fn divided<T: Output>(self, count: u64) -> T {
        let (negative, mut limbs, least) = self.difference();
        limbs.insert(0, 0);
        let divisor = u128::from(count);
        let mut remainder = 0u128;
        for limb in limbs.iter_mut().rev() {
            let current = (remainder << 64) | u128::from(*limb);
            let quotient = current / divisor;
            remainder = current - quotient * divisor;
            *limb = quotient as u64;
        }
        rounded_at(negative, &limbs, remainder != 0, least - 64)
    }
}

/// `limbs << bits`, in little-endian 64-bit limbs with room for every bit.
fn shifted(limbs: &[u64], bits: usize) -> Vec<u64> {
    let (words, bits) = (bits / 64, bits % 64);
    let mut out = vec![0u64; limbs.len() + words + 1];
    for (index, &limb) in limbs.iter().enumerate() {
        out[index + words] |= limb << bits;
        if bits != 0 {
            out[index + words + 1] |= limb >> (64 - bits);
        }
    }
    out
}

/// Whether `a >= b`, the shorter read as zero-extended.
fn at_least(a: &[u64], b: &[u64]) -> bool {
    let limb = |limbs: &[u64], index: usize| limbs.get(index).copied().unwrap_or(0);
    for index in (0..a.len().max(b.len())).rev() {
        let (x, y) = (limb(a, index), limb(b, index));
        if x != y {
            return x > y;
        }
    }
    true
}

/// `a -= b`, for `a >= b`.
fn subtract(a: &mut [u64], b: &[u64]) {
    let mut borrow = false;
    for (index, limb) in a.iter_mut().enumerate() {
        let other = b.get(index).copied().unwrap_or(0);
        let (value, first) = limb.overflowing_sub(other);
        let (value, second) = value.overflowing_sub(u64::from(borrow));
        *limb = value;
        borrow = first || second;
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

    fn exact_dot(a: &[f64], b: &[f64]) -> f64 {
        exact_dot_f64(a.iter().copied().zip(b.iter().copied()))
    }

    #[test]
    fn float64_dots_are_rounded_once() {
        let tiny = 2f64.powi(-600);
        let small = 2f64.powi(-475);
        // Two products of half the least subnormal: each rounds to zero,
        // together they are the least subnormal.
        assert_eq!(dot_f64(&[tiny, tiny], &[small, small]), f64::from_bits(1));
        assert_eq!(dot_f64(&[0.0; 5], &[1.0; 5]).to_bits(), 0);
        assert_eq!(dot_f64(&[], &[]).to_bits(), 0);
        // Products past the largest float64 that cancel.
        assert_eq!(dot_f64(&[1e200, 1e200, 3.0], &[1e200, -1e200, 0.5]), 1.5);
        assert_eq!(dot_f64(&[1e200], &[1e200]), f64::INFINITY);
        assert!(dot_f64(&[f64::INFINITY, 1.0], &[0.0, 1.0]).is_nan());
        assert!(dot_f64(&[f64::INFINITY, f64::INFINITY], &[1.0, -1.0]).is_nan());
        assert_eq!(
            dot_f64(&[f64::INFINITY, 1e200], &[1.0, -1e200]),
            f64::INFINITY
        );
        assert!(dot_f64(&[f64::NAN, 1.0], &[1.0, 1.0]).is_nan());
        // `0.1 · 0.1 + 0.2 · 0.2` is rounded once, not twice.
        assert_eq!(
            dot_f64(&[0.1, 0.2], &[0.1, 0.2]),
            exact_dot(&[0.1, 0.2], &[0.1, 0.2])
        );
        // A float64 halfway point, and a hair past it.
        let half = f64::EPSILON / 2.0;
        assert_eq!(dot_f64(&[1.0, half], &[1.0, 1.0]), 1.0);
        assert_eq!(
            dot_f64(&[1.0, half, 1e-300], &[1.0, 1.0, 1.0]),
            1.0 + f64::EPSILON
        );
    }

    #[test]
    fn the_carried_dot_agrees_with_the_exact_one() {
        let mut next = generator(0x9e37_79b9_7f4a_7c15);
        let mut unit = move || {
            let value = (next() >> 11) as f64 / (1u64 << 53) as f64 - 0.5;
            (value, next())
        };
        for round in 0..1500 {
            let len = 1 + (unit().1 % 20_000) as usize;
            let spread = 1 + unit().1 % 80;
            let mut a = Vec::with_capacity(len);
            let mut b = Vec::with_capacity(len);
            for _ in 0..len {
                let (x, bits) = unit();
                a.push(x * 2f64.powi((bits % spread) as i32 - 40));
                b.push(unit().0);
            }
            if round % 3 == 0 {
                // Products cancelling in pairs, nearly.
                let mirror: Vec<f64> = a.iter().map(|&v| -v).collect();
                let near: Vec<f64> = b.iter().map(|&v| v * (1.0 + 2f64.powi(-30))).collect();
                a.extend(&mirror[1..]);
                b.extend(&near[1..]);
            }
            if round % 7 == 1 {
                // Down where a product's error is a subnormal.
                for value in a.iter_mut() {
                    *value *= 2f64.powi(-1000);
                }
            }
            let exact = exact_dot(&a, &b);
            assert_eq!(dot_f64(&a, &b).to_bits(), exact.to_bits(), "{len} products");
            if round % 50 == 0 {
                // The same dots as columns of a matrix with three columns.
                let k = a.len();
                let matrix: Vec<f64> = (0..k).flat_map(|s| [b[s], -b[s], 2.0 * b[s]]).collect();
                let rows = |step: usize| &matrix[step * 3..step * 3 + 3];
                let mut out = [0.0; 3];
                dot_columns_f64(&a, &rows, &mut out);
                let doubled: Vec<f64> = b.iter().map(|&v| 2.0 * v).collect();
                let negated: Vec<f64> = b.iter().map(|&v| -v).collect();
                assert_eq!(out[0].to_bits(), exact.to_bits());
                assert_eq!(out[1].to_bits(), exact_dot(&a, &negated).to_bits());
                assert_eq!(out[2].to_bits(), exact_dot(&a, &doubled).to_bits());
            }
        }
    }

    #[test]
    fn every_build_carries_the_same_products() {
        let mut next = generator(0x2545_f491_4f6c_dd1d);
        for len in [0, 1, 7, 8, 9, 100, 8191, 8192] {
            let mut value = || ((next() >> 11) as f64 / (1u64 << 53) as f64 - 0.5) * 1e3;
            let a: Vec<f64> = (0..len).map(|_| value()).collect();
            let b: Vec<f64> = (0..len).map(|_| value()).collect();
            let bits =
                |carried: Carried| [carried.sum, carried.error, carried.slack].map(f64::to_bits);
            let scalar = bits(carried_dot_f64_lanes(&a, &b));
            #[cfg(target_arch = "x86_64")]
            {
                let capabilities = crate::ops::simd::simd_capabilities();
                if capabilities.avx2_fma {
                    // SAFETY: avx2 and fma were detected on this CPU.
                    assert_eq!(bits(unsafe { carried_dot_fma(&a, &b) }), scalar, "{len}");
                }
            }
            let _ = scalar;
        }
    }

    #[test]
    fn long_division_rounds_once() {
        let mut next = generator(0x8f1b_bcdc_bf3c_a6f7);
        for _ in 0..20_000 {
            // Whole numbers below 2^53, whose float64 quotient is rounded
            // once by the division itself.
            let n = (next() >> (11 + next() % 50)) as f64;
            let d = ((next() >> (11 + next() % 50)) | 1) as f64;
            let sign = if next().is_multiple_of(2) { 1.0 } else { -1.0 };
            if n == 0.0 {
                // An exact zero total is `+0`, whatever the sign of its zeros.
                continue;
            }
            let quotient: f64 = Total::of_wide([sign * n].into_iter())
                .unwrap()
                .quotient(Total::of_wide([d].into_iter()).unwrap());
            assert_eq!(quotient.to_bits(), (sign * n / d).to_bits(), "{n} / {d}");
        }
        let total = |values: &[f64]| Total::of_wide(values.iter().copied()).unwrap();
        assert!(total(&[0.0]).quotient::<f64>(total(&[0.0])).is_nan());
        assert_eq!(
            total(&[-2.0]).quotient::<f64>(total(&[0.0])),
            f64::NEG_INFINITY
        );
        assert_eq!(
            total(&[0.0]).quotient::<f64>(total(&[-3.0])).to_bits(),
            (-0.0f64).to_bits()
        );
        // Past the largest float64 over past the largest float64.
        let big = total(&[f64::MAX, f64::MAX, f64::MAX]);
        assert_eq!(big.quotient::<f64>(total(&[f64::MAX, f64::MAX])), 1.5);
    }

    #[test]
    fn weighted_means_divide_the_exact_totals_once() {
        let exact = |values: &[f64], weights: &[f64]| -> f64 {
            exact_quotient(
                values.iter().copied().zip(weights.iter().copied()),
                weights.iter().copied(),
            )
        };
        assert_eq!(
            weighted_mean_row::<f64, f64>(&[1.0, 2.0], &[1.0, 2.0]),
            5.0 / 3.0
        );
        assert!(weighted_mean_row::<f64, f64>(&[], &[]).is_nan());
        assert!(weighted_mean_row::<f64, f64>(&[1.0, 2.0], &[1.0, -1.0]).is_infinite());
        // Products and a weight total past the largest float64.
        let big = 1e300;
        assert_eq!(weighted_mean_row::<f64, f64>(&[big, big], &[big, big]), big);
        // An infinite weight makes an infinite weighted sum over an infinite
        // total.
        assert!(weighted_mean_row::<f64, f64>(&[1.0, 3.0], &[f64::INFINITY, 1.0]).is_nan());
        assert_eq!(
            weighted_mean_row::<f64, f64>(&[f64::INFINITY, 3.0], &[1.0, 1.0]),
            f64::INFINITY
        );
        let mut next = generator(0x5851_f42d_4c95_7f2d);
        let mut unit = move || (next() >> 11) as f64 / (1u64 << 53) as f64 - 0.5;
        for round in 0..2000 {
            let len = 1 + (round * 37) % 3000;
            let mut values: Vec<f64> = (0..len).map(|_| unit() * 1e3).collect();
            let mut weights: Vec<f64> = (0..len).map(|_| unit() + 0.25).collect();
            if round % 4 == 1 {
                // Weights that nearly cancel.
                let total: f64 = weights.iter().sum();
                weights.push(-total * (1.0 - 2f64.powi(-40)));
                values.push(unit());
            }
            if round % 4 == 2 {
                // A weighted sum that cancels across the binades.
                for (index, value) in values.iter_mut().enumerate() {
                    *value *= 2f64.powi((index % 60) as i32 - 30);
                }
            }
            let want = exact(&values, &weights);
            let got: f64 = weighted_mean_row(&values, &weights);
            assert_eq!(got.to_bits(), want.to_bits(), "{len} values, round {round}");
            let narrow: Vec<f32> = values.iter().map(|&v| v as f32).collect();
            let narrow_weights: Vec<f32> = weights.iter().map(|&w| w as f32).collect();
            let wide: Vec<f64> = narrow.iter().map(|&v| f64::from(v)).collect();
            let wide_weights: Vec<f64> = narrow_weights.iter().map(|&w| f64::from(w)).collect();
            let want: f32 = exact_quotient(
                wide.iter().copied().zip(wide_weights.iter().copied()),
                wide_weights.iter().copied(),
            );
            let got: f32 = weighted_mean_row(&narrow, &narrow_weights);
            assert_eq!(got.to_bits(), want.to_bits(), "float32, round {round}");
            let through: f32 = weighted_mean_row(&wide, &wide_weights);
            assert_eq!(
                through.to_bits(),
                want.to_bits(),
                "float32 from float64, round {round}"
            );
        }
    }

    #[test]
    fn roots_are_rounded_once() {
        let mut next = generator(0x1405_7b7e_f767_814f);
        for _ in 0..20_000 {
            // Whole numbers below 2^53 and their sums of squares below it
            // too, whose float64 root `sqrt` rounds once.
            let n = (next() >> (11 + next() % 52)) as f64;
            let root: f64 = Total::of_wide([n].into_iter()).unwrap().square_root();
            assert_eq!(root.to_bits(), n.sqrt().to_bits(), "{n}");
            let values: Vec<f64> = (0..1 + next() % 8)
                .map(|_| (next() >> 40) as f64 * if next() % 2 == 1 { -1.0 } else { 1.0 })
                .collect();
            let squares: f64 = values.iter().map(|v| v * v).sum();
            assert_eq!(norm_f64(&values).to_bits(), squares.sqrt().to_bits());
            let narrow: Vec<f32> = values.iter().map(|&v| v as f32).collect();
            assert_eq!(norm_f32(&narrow), squares.sqrt() as f32);
        }
        assert_eq!(norm_f64(&[]), 0.0);
        assert_eq!(norm_f64(&[0.0, -0.0]), 0.0);
        assert_eq!(norm_f64(&[3.0, 4.0]), 5.0);
        // Squares past the largest float64, and below the least subnormal.
        let (big, small) = (2f64.powi(600), 2f64.powi(-600));
        assert_eq!(norm_f64(&[3.0 * big, 4.0 * big]), 5.0 * big);
        assert_eq!(norm_f64(&[3.0 * small, -4.0 * small]), 5.0 * small);
        assert_eq!(norm_f64(&[f64::from_bits(1)]), f64::from_bits(1));
        assert_eq!(norm_f64(&[f64::MAX, f64::MAX]), f64::INFINITY);
        assert_eq!(norm_f64(&[f64::NEG_INFINITY, 1.0]), f64::INFINITY);
        assert!(norm_f64(&[f64::NAN, 1.0]).is_nan());
        let big = 2f32.powi(100);
        assert_eq!(norm_f32(&[3.0 * big, 4.0 * big]), 5.0 * big);
        assert_eq!(norm_f32(&[f32::from_bits(1)]), f32::from_bits(1));
        assert_eq!(norm_f32(&[f32::MAX, f32::MAX]), f32::INFINITY);
    }

    #[test]
    fn the_carried_root_agrees_with_the_exact_one() {
        let mut next = generator(0xda94_2042_e4dd_58b5);
        for round in 0..1500 {
            let len = 1 + (next() % 5000) as usize;
            let spread = 1 + next() % 80;
            let values: Vec<f64> = (0..len)
                .map(|_| {
                    let unit = (next() >> 11) as f64 / (1u64 << 53) as f64 - 0.5;
                    let scale = if round % 5 == 1 { -1040 } else { -40 };
                    unit * 2f64.powi((next() % spread) as i32 + scale)
                })
                .collect();
            let exact: f64 = exact_root(values.iter().copied());
            assert_eq!(norm_f64(&values).to_bits(), exact.to_bits(), "{len} values");
            let narrow: Vec<f32> = values.iter().map(|&v| v as f32).collect();
            let exact: f32 = exact_root(narrow.iter().map(|&v| f64::from(v)));
            assert_eq!(norm_f32(&narrow).to_bits(), exact.to_bits(), "{len} values");
        }
    }

    #[test]
    fn column_norms_agree_with_the_norms_of_runs() {
        let mut next = generator(0x6a09_e667_f3bc_c908);
        for (outer, len, inner) in [(3, 50, 7), (2, 9000, 3), (1, 5, 600), (4, 1, 2)] {
            let values: Vec<f64> = (0..outer * len * inner)
                .map(|_| {
                    let unit = (next() >> 11) as f64 / (1u64 << 53) as f64 - 0.5;
                    unit * 2f64.powi((next() % 60) as i32 - 30)
                })
                .collect();
            let column = |o: usize, j: usize| -> Vec<f64> {
                (0..len)
                    .map(|r| values[(o * len + r) * inner + j])
                    .collect()
            };
            let mut out = vec![0f64; outer * inner];
            column_norms(&values, outer, len, inner, &mut out);
            let narrow: Vec<f32> = values.iter().map(|&v| v as f32).collect();
            let mut narrow_out = vec![0f32; outer * inner];
            column_norms(&narrow, outer, len, inner, &mut narrow_out);
            for o in 0..outer {
                for j in 0..inner {
                    let run = column(o, j);
                    assert_eq!(out[o * inner + j].to_bits(), norm_f64(&run).to_bits());
                    let run: Vec<f32> = run.iter().map(|&v| v as f32).collect();
                    assert_eq!(
                        narrow_out[o * inner + j].to_bits(),
                        norm_f32(&run).to_bits()
                    );
                }
            }
        }
    }

    #[test]
    fn differences_of_products_are_rounded_once() {
        let mut next = generator(0xbb67_ae85_84ca_a73b);
        let mut value = move |spread: u64| {
            let unit = (next() >> 11) as f64 / (1u64 << 53) as f64 - 0.5;
            unit * 2f64.powi((next() % spread) as i32 - (spread / 2) as i32)
        };
        for round in 0..200_000 {
            let spread = [4, 60, 2000][round % 3];
            let (a, b, c) = (value(spread), value(spread), value(spread));
            // Every other case nearly cancels: `d` close to `a · b / c`.
            let d = if round % 2 == 0 {
                value(spread)
            } else {
                (a * b / c) * (1.0 + value(4) * 2f64.powi(-40))
            };
            // An exact zero takes the sign float arithmetic gives the
            // expression; a value that rounds to zero keeps its own.
            let signed = |want: f64, pairs: [(f64, f64); 2]| {
                let zero = Total::of_products(pairs.into_iter())
                    .is_ok_and(|total| total.positive == total.negative);
                if zero {
                    pairs[0].0 * pairs[0].1 + pairs[1].0 * pairs[1].1
                } else {
                    want
                }
            };
            let want = signed(exact_difference(a, b, c, d), [(a, b), (-c, d)]);
            let got = difference_of_products_f64(a, b, c, d);
            assert!(
                got.to_bits() == want.to_bits() || (got.is_nan() && want.is_nan()),
                "{a:e} {b:e} {c:e} {d:e}: {got:e} against {want:e}"
            );
            let (a, b, c, d) = (a as f32, b as f32, c as f32, d as f32);
            let pairs = [(f64::from(a), f64::from(b)), (-f64::from(c), f64::from(d))];
            let want: f32 = Total::of_products(pairs.into_iter())
                .map_or_else(|special| special as f32, |total| total.rounded());
            let want = signed(f64::from(want), pairs) as f32;
            let got = difference_of_products_f32(a, b, c, d);
            assert!(
                got.to_bits() == want.to_bits() || (got.is_nan() && want.is_nan()),
                "{a:e} {b:e} {c:e} {d:e}: {got:e} against {want:e}"
            );
        }
        assert_eq!(difference_of_products_f64(1e200, 1e200, 1e200, 1e200), 0.0);
        assert_eq!(
            difference_of_products_f64(1e200, 3e200, 1e200, 1e200),
            f64::INFINITY
        );
        assert!(difference_of_products_f64(f64::INFINITY, 1.0, f64::INFINITY, 1.0).is_nan());
        // Two products of half the least subnormal, each rounding to zero.
        let (x, y) = (2f64.powi(-537), 2f64.powi(-538));
        assert_eq!(difference_of_products_f64(x, y, -x, y), f64::from_bits(1));
    }

    /// Whether `r` is the value nearest `sqrt(x² + y²)` among values spaced
    /// `below` under it and `above` over it: the exact sum of the squares
    /// lies between the squares of the halfway points either side, on one
    /// only where `r` is even.
    fn nearest_root(x: f64, y: f64, r: f64, below: f64, above: f64, even: bool) -> bool {
        let sign = |pairs: &[(f64, f64)]| {
            let total = Total::of_products(pairs.iter().copied()).expect("finite");
            let (negative, limbs, _) = total.difference();
            match (limbs.iter().all(|&limb| limb == 0), negative) {
                (true, _) => 0,
                (false, true) => -1,
                (false, false) => 1,
            }
        };
        let (half_below, half_above) = (below / 2.0, above / 2.0);
        // `S - (r - below / 2)²` and `(r + above / 2)² - S`.
        let low = sign(&[
            (x, x),
            (y, y),
            (-r, r),
            (r, below),
            (-half_below, half_below),
        ]);
        let high = sign(&[
            (r, r),
            (r, above),
            (half_above, half_above),
            (-x, x),
            (-y, y),
        ]);
        let inside = |side: i32| side > 0 || (side == 0 && even);
        inside(low) && inside(high)
    }

    /// Legs below `2^bits` whose hypotenuse is an odd whole number past it,
    /// so halfway between two floats `bits` wide.
    fn halfway_triple(bits: u32) -> (f64, f64) {
        let low = 1u64 << bits;
        let ratio = 1.0 + 2f64.sqrt();
        let gcd = |mut a: u64, mut b: u64| {
            while b != 0 {
                (a, b) = (b, a % b);
            }
            a
        };
        let mut n = ((low as f64) / (4.0 + 2.0 * 2f64.sqrt())).sqrt() as u64;
        loop {
            let centre = (n as f64 * ratio) as u64;
            for m in centre - 3..=centre + 3 {
                let (a, b, c) = (m * m - n * n, 2 * m * n, m * m + n * n);
                if (m - n) % 2 == 1 && gcd(m, n) == 1 && c > low && a.max(b) < low {
                    return (a as f64, b as f64);
                }
            }
            n += 1;
        }
    }

    #[test]
    fn hypot_rounds_once() {
        let check_wide = |x: f64, y: f64| {
            let (halves, fused) = (hypot_f64::<false>(x, y), hypot_f64::<true>(x, y));
            assert_eq!(halves.to_bits(), fused.to_bits(), "hypot({x:e}, {y:e})");
            let got = halves;
            if got == f64::INFINITY {
                // The exact root is at least the largest float64 and half
                // its ulp, where an odd last bit rounds up.
                let (top, half) = (f64::MAX, 2f64.powi(970));
                let pairs = [
                    (x, x),
                    (y, y),
                    (-top, top),
                    (-top, 2.0 * half),
                    (-half, half),
                ];
                let (negative, _, _) = Total::of_products(pairs.into_iter()).unwrap().difference();
                assert!(!negative, "hypot({x:e}, {y:e}) overflowed");
                return;
            }
            if got == 0.0 {
                assert!(x == 0.0 && y == 0.0);
                return;
            }
            let below = got - f64::from_bits(got.to_bits() - 1);
            let above = if got == f64::MAX {
                2f64.powi(971)
            } else {
                f64::from_bits(got.to_bits() + 1) - got
            };
            // Below `2^-1020` everything is scaled up, exactly, so that half
            // the spacing is a float64 too.
            let scale = if got < 2f64.powi(-1020) {
                2f64.powi(600)
            } else {
                1.0
            };
            let even = got.to_bits() & 1 == 0;
            let (r, below, above) = (got * scale, below * scale, above * scale);
            assert!(
                nearest_root(x * scale, y * scale, r, below, above, even),
                "hypot({x:e}, {y:e}) = {got:e}"
            );
        };
        let check_narrow = |x: f32, y: f32| {
            let got = hypot_f32(x, y);
            if got == f32::INFINITY {
                let (top, half) = (f64::from(f32::MAX), 2f64.powi(103));
                let (x, y) = (f64::from(x), f64::from(y));
                let pairs = [
                    (x, x),
                    (y, y),
                    (-top, top),
                    (-top, 2.0 * half),
                    (-half, half),
                ];
                let (negative, _, _) = Total::of_products(pairs.into_iter()).unwrap().difference();
                assert!(!negative, "hypot({x:e}, {y:e}) overflowed");
                return;
            }
            if got == 0.0 {
                assert!(x == 0.0 && y == 0.0);
                return;
            }
            let r = f64::from(got);
            let below = r - f64::from(f32::from_bits(got.to_bits() - 1));
            let above = if got == f32::MAX {
                2f64.powi(104)
            } else {
                f64::from(f32::from_bits(got.to_bits() + 1)) - r
            };
            let even = got.to_bits() & 1 == 0;
            let (x, y) = (f64::from(x), f64::from(y));
            assert!(
                nearest_root(x, y, r, below, above, even),
                "hypot({x:e}, {y:e}) = {got:e}"
            );
        };
        let mut next = generator(0x6a09_e667_f3bc_c909);
        for round in 0..20_000 {
            let unit = |bits: u64| (bits >> 11) as f64 / (1u64 << 53) as f64 + 0.5;
            let exponent = (next() % 2099) as i32 - 1075;
            let x = unit(next()) * 2f64.powi(exponent / 2) * 2f64.powi(exponent - exponent / 2);
            let gap = (next() % 70) as i32 - 5;
            let y = x * unit(next()) * 2f64.powi(-gap);
            let (x, y) = if round % 2 == 0 { (x, -y) } else { (-y, x) };
            if x.is_finite() && y.is_finite() {
                check_wide(x, y);
            }
            if (x as f32).is_finite() && (y as f32).is_finite() {
                check_narrow(x as f32, y as f32);
            }
        }
        for (a, b) in [halfway_triple(53), halfway_triple(24)] {
            for shift in [-1070, -1000, -600, -1, 0, 1, 600, 969] {
                let scale = 2f64.powi(shift / 2) * 2f64.powi(shift - shift / 2);
                check_wide(a * scale, b * scale);
                check_wide(b * scale, -a * scale);
            }
        }
        let (a, b) = halfway_triple(24);
        for shift in [-140, -100, 0, 50, 103] {
            let scale = 2f32.powi(shift);
            check_narrow(a as f32 * scale, b as f32 * scale);
        }
        let largest = f64::MAX;
        for (x, y) in [
            (largest, largest),
            (largest, 1.0),
            (largest / 2.0, largest / 2.0),
            (largest, largest * 2f64.powi(-27)),
            (f64::from_bits(1), f64::from_bits(1)),
            (f64::from_bits(3), f64::from_bits(4)),
            (2f64.powi(-1020), 2f64.powi(-1021)),
            (2f64.powi(-1022), 2f64.powi(-1050)),
            (3.0, 4.0),
            (0.0, -0.0),
        ] {
            check_wide(x, y);
        }
        for (x, y) in [
            (f32::MAX, f32::MAX),
            (f32::MAX, 1.0),
            (f32::from_bits(1), 1e-45),
        ] {
            check_narrow(x, y);
        }
        assert_eq!(hypot_f64::<false>(f64::INFINITY, f64::NAN), f64::INFINITY);
        assert_eq!(
            hypot_f64::<true>(f64::NAN, f64::NEG_INFINITY),
            f64::INFINITY
        );
        assert!(hypot_f64::<false>(f64::NAN, 1.0).is_nan());
        assert_eq!(hypot_f32(f32::NAN, f32::INFINITY), f32::INFINITY);
        assert!(hypot_f32(1.0, f32::NAN).is_nan());
        assert_eq!(hypot_f64::<false>(-0.0, -0.0).to_bits(), 0);
        assert_eq!(hypot_f32(-0.0, -0.0).to_bits(), 0);
        // An integer beside a float32: `2^30 + 64` is a float32 halfway
        // point, and the exact root of its square plus one lies past it.
        let big = 2f64.powi(30) + 64.0;
        assert_eq!(hypot_of::<f32>(big, 1.0), 2f32.powi(30) + 128.0);
    }

    #[test]
    fn every_running_sum_is_rounded_once() {
        let mut next = generator(0x3c6e_f372_fe94_f82b);
        for round in 0..300 {
            let (len, inner) = (1 + (next() % 400) as usize, 1 + (next() % 3) as usize);
            let spread = 1 + next() % 80;
            let mut values: Vec<f64> = (0..len * inner)
                .map(|_| {
                    let unit = (next() >> 11) as f64 / (1u64 << 53) as f64 - 0.5;
                    unit * 2f64.powi((next() % spread) as i32 - 40)
                })
                .collect();
            if round % 3 == 0 {
                // Values that cancel what came before.
                for index in (inner..values.len()).step_by(2) {
                    values[index] = -values[index - inner];
                }
            }
            let reverse = round % 2 == 1;
            let mut out = vec![0f64; values.len()];
            running_sums(&mut out, &values, len, inner, reverse);
            let narrow: Vec<f32> = values.iter().map(|&v| v as f32).collect();
            let mut narrow_out = vec![0f32; values.len()];
            running_sums(&mut narrow_out, &narrow, len, inner, reverse);
            for lane in 0..inner {
                let order: Vec<usize> = if reverse {
                    (0..len).rev().collect()
                } else {
                    (0..len).collect()
                };
                for (step, &row) in order.iter().enumerate() {
                    let prefix: Vec<f64> = order[..=step]
                        .iter()
                        .map(|&r| values[r * inner + lane])
                        .collect();
                    let want: f64 = Total::of(&prefix).unwrap().rounded();
                    assert_eq!(out[row * inner + lane], want, "round {round}");
                    let prefix: Vec<f32> = order[..=step]
                        .iter()
                        .map(|&r| narrow[r * inner + lane])
                        .collect();
                    let want: f32 = Total::of(&prefix).unwrap().rounded();
                    assert_eq!(narrow_out[row * inner + lane], want, "round {round}");
                }
            }
        }
        let mut out = [0f64; 4];
        running_sums(&mut out, &[-0.0, -0.0, 1.0, -1.0], 4, 1, false);
        assert_eq!(
            out.map(f64::to_bits),
            [-0.0f64, -0.0, 1.0, 0.0].map(f64::to_bits)
        );
        running_sums(&mut out, &[1e308, 1e308, -1e308, f64::NAN], 4, 1, false);
        assert_eq!(out[..3], [1e308, f64::INFINITY, 1e308]);
        assert!(out[3].is_nan());
        running_sums(
            &mut out,
            &[1.0, f64::INFINITY, 2.0, f64::NEG_INFINITY],
            4,
            1,
            false,
        );
        assert_eq!(out[..3], [1.0, f64::INFINITY, f64::INFINITY]);
        assert!(out[3].is_nan());
    }

    #[test]
    fn long_runs_and_many_columns_round_once() {
        let mut next = generator(0xa54f_f53a_5f1d_36f1);
        let mut value = move || {
            let unit = (next() >> 11) as f64 / (1u64 << 53) as f64 - 0.5;
            unit * 2f64.powi((next() % 50) as i32 - 25)
        };
        // A run long enough to be split into pieces, with a stretch that
        // cancels everything before it.
        let len = 300_000;
        let mut values: Vec<f64> = (0..len).map(|_| value()).collect();
        let half: f64 = values[..len / 2].iter().sum();
        values[len / 2] = -half;
        for reverse in [false, true] {
            let mut out = vec![0f64; len];
            running_sums(&mut out, &values, len, 1, reverse);
            let narrow: Vec<f32> = values.iter().map(|&v| v as f32).collect();
            let mut narrow_out = vec![0f32; len];
            running_sums(&mut narrow_out, &narrow, len, 1, reverse);
            let mut exact = Exactly::of(std::iter::empty());
            let mut narrow_exact = Exactly::of(std::iter::empty());
            for step in 0..len {
                let at = if reverse { len - 1 - step } else { step };
                exact.add(values[at]);
                narrow_exact.add(f64::from(narrow[at]));
                if step % 97 == 0 || step.abs_diff(len / 2) < 3 {
                    let want: f64 = exact.total.clone().rounded();
                    assert_eq!(out[at], want, "step {step}");
                    let want: f32 = narrow_exact.total.clone().rounded();
                    assert_eq!(narrow_out[at], want, "step {step}");
                }
            }
        }
        // More columns than a band, in two slabs.
        let (outer, len, inner) = (2, 40, 150);
        let values: Vec<f64> = (0..outer * len * inner).map(|_| value()).collect();
        let mut out = vec![0f64; values.len()];
        running_sums(&mut out, &values, len, inner, false);
        for slab in 0..outer {
            for column in 0..inner {
                let mut exact = Exactly::of(std::iter::empty());
                for row in 0..len {
                    let at = (slab * len + row) * inner + column;
                    exact.add(values[at]);
                    let want: f64 = exact.total.clone().rounded();
                    assert_eq!(out[at], want);
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
