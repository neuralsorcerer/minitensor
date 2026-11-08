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
use crate::ops::util::{pairwise_fold, pairwise_fold_vectors};

#[path = "exact_covariance.rs"]
mod covariance;
pub(crate) use self::covariance::{Refusal, Sample, covariances};

/// Lanes a run is added in, and values per block a task takes.
const LANES: usize = 8;
const BLOCK: usize = 8192;

/// Just under one half; see `evenly_spaced`.
pub(super) const SHY_OF_HALF: f64 = 0.5 - 1.0 / (1u64 << 40) as f64;

/// An element a sum reads: a float32 or a float64, each exact in float64.
pub(crate) trait Element: Copy + Send + Sync + 'static {
    fn wide(self) -> f64;
    /// [`block_lanes`] in the widest build the CPU runs.
    fn block(values: &[Self]) -> Carried;
    /// [`plain_lanes`] in the widest build the CPU runs.
    fn plain_block(values: &[Self]) -> (f64, f64) {
        plain_lanes(values)
    }
    /// [`plain_squares_lanes`] in the widest build the CPU runs.
    fn plain_squares(values: &[Self], centre: f64) -> f64;
    /// [`plain_differences_lanes`] in the widest build the CPU runs.
    fn plain_differences(x: &[Self], y: &[Self], euclidean: bool, shift: f64) -> f64 {
        plain_differences_lanes(x, y, euclidean, shift)
    }
    /// The carried total of `a[i] · b[i]`, block by block in the widest
    /// build the CPU runs, the blocks merged pairwise.
    fn carried_dot(a: &[Self], b: &[Self]) -> Carried;
    /// `values` as float64s, when they are.
    fn as_wide(_values: &[Self]) -> Option<&[f64]> {
        None
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

    fn plain_squares(values: &[Self], centre: f64) -> f64 {
        #[cfg(target_arch = "x86_64")]
        if crate::ops::simd::simd_capabilities().avx {
            // SAFETY: avx was detected on this CPU.
            return unsafe { plain_squares_avx(values, centre) };
        }
        plain_squares_lanes(values, centre)
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

    fn as_wide(values: &[Self]) -> Option<&[f64]> {
        Some(values)
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

    fn plain_squares(values: &[Self], centre: f64) -> f64 {
        #[cfg(target_arch = "x86_64")]
        if crate::ops::simd::simd_capabilities().avx {
            // SAFETY: avx was detected on this CPU.
            return unsafe { plain_squares_avx_f32(values, centre) };
        }
        plain_squares_lanes(values, centre)
    }

    fn plain_differences(x: &[Self], y: &[Self], euclidean: bool, shift: f64) -> f64 {
        #[cfg(target_arch = "x86_64")]
        if crate::ops::simd::simd_capabilities().avx {
            // SAFETY: avx was detected on this CPU.
            return unsafe {
                if euclidean {
                    plain_differences_avx_f32::<true>(x, y, shift)
                } else {
                    plain_differences_avx_f32::<false>(x, y, shift)
                }
            };
        }
        plain_differences_lanes(x, y, euclidean, shift)
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
pub(crate) trait Output: Target + Send + Sync + PartialEq {
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
    /// The value as a float64, which holds it exactly.
    fn widen(self) -> f64;
    /// The value nearest an exact total within `bound` of `value + rest`,
    /// a two-sum's pair, where that reaches the halfway point between two
    /// values: `side(high, low)` says how the total compares with the
    /// halfway point `high + low` nearest the pair -- a float64 pair, the
    /// low part zero or a power of two -- or that it cannot, and a tie goes
    /// to the even value. None where the bound reaches past the two values
    /// either side of that point.
    fn split(
        value: f64,
        rest: f64,
        bound: f64,
        side: impl Fn(f64, f64) -> Option<std::cmp::Ordering>,
    ) -> Option<Self>;
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
    fn widen(self) -> f64 {
        self
    }

    #[inline(always)]
    fn settle_lanes(value: f64, rest: f64, bound: f64) -> (Self, bool) {
        let inward = (value - f64::from_bits(value.to_bits().wrapping_sub(1))).abs();
        let clear = rest.abs() + bound < inward * SHY_OF_HALF && value != 0.0;
        (value, value.is_finite() & ((bound == 0.0) | clear))
    }

    fn split(
        value: f64,
        rest: f64,
        bound: f64,
        side: impl Fn(f64, f64) -> Option<std::cmp::Ordering>,
    ) -> Option<Self> {
        if !value.is_finite() || value == 0.0 || rest == 0.0 {
            return None;
        }
        let neighbour = if rest > 0.0 {
            value.next_up()
        } else {
            value.next_down()
        };
        // Half the gap to the neighbour, a power of two; within half of it
        // of the pair, the total lies between the halfway points beyond the
        // two, `rest` being at most the gap above `value` over two.
        let half = (neighbour - value) * 0.5;
        if !neighbour.is_finite() || neighbour == 0.0 || bound >= half.abs() * 0.5 {
            return None;
        }
        Some(match side(value, half)? {
            std::cmp::Ordering::Equal if value.to_bits() & 1 == 0 => value,
            std::cmp::Ordering::Equal => neighbour,
            order if (order == std::cmp::Ordering::Greater) == (half > 0.0) => neighbour,
            _ => value,
        })
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
    fn widen(self) -> f64 {
        f64::from(self)
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
        // The sign moved by bits rather than chosen: a branch on it is a
        // guess, wrong half the time on data of both signs.
        let sign = value.to_bits() & (1 << 63);
        let toward = f64::from_bits(rest.to_bits() ^ sign);
        let room = |gap: f64| gap * (1.0 - 1.0 / (1u64 << 40) as f64);
        let clear =
            (toward + bound < room(up - magnitude)) & (bound - toward < room(magnitude - down));
        let signed = f32::from_bits(narrow.to_bits() | (sign >> 32) as u32);
        (signed, (narrow < f32::MAX) & clear)
    }

    fn split(
        value: f64,
        rest: f64,
        bound: f64,
        side: impl Fn(f64, f64) -> Option<std::cmp::Ordering>,
    ) -> Option<Self> {
        let narrow = value as f32;
        if !narrow.is_finite() || narrow.abs() == f32::MAX {
            return None;
        }
        let back = f64::from(narrow);
        // The halfway point nearest the pair, exact in float64, and the
        // float32 past it.
        let toward = (value - back) + rest;
        let other = if toward > 0.0 {
            narrow.next_up()
        } else {
            narrow.next_down()
        };
        let halfway = (back + f64::from(other)) * 0.5;
        let gap = (f64::from(other) - back).abs();
        // The total between `narrow` and `other`, so one of them.
        if (back + toward - halfway).abs() + bound >= gap * 0.5 {
            return None;
        }
        Some(match side(halfway, 0.0)? {
            std::cmp::Ordering::Equal if narrow.to_bits() & 1 == 0 => narrow,
            std::cmp::Ordering::Equal => other,
            order if (order == std::cmp::Ordering::Greater) == (f64::from(other) > back) => other,
            _ => narrow,
        })
    }
}

/// The sign of the exact sum of a few finite floats, by growing them into an
/// expansion of parts that do not overlap, smallest first, whose largest
/// nonzero part carries the sign; none for a sum past the largest float on
/// the way.
fn expansion_sign(terms: &[f64]) -> Option<std::cmp::Ordering> {
    let mut parts = [0f64; 8];
    debug_assert!(terms.len() <= parts.len());
    for (count, &term) in terms.iter().enumerate() {
        let mut carry = term;
        for part in &mut parts[..count] {
            let (sum, error) = two_sum(carry, *part);
            *part = error;
            carry = sum;
        }
        parts[count] = carry;
    }
    let parts = &parts[..terms.len()];
    if !parts.iter().all(|part| part.is_finite()) {
        return None;
    }
    let largest = parts.iter().rev().find(|&&part| part != 0.0).copied();
    Some(largest.map_or(std::cmp::Ordering::Equal, |part| {
        part.partial_cmp(&0.0).unwrap_or(std::cmp::Ordering::Equal)
    }))
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

    /// The total of two whose `slack` is twice the sum of what their error
    /// additions dropped: the sums and the errors added by two-sums, and what
    /// those drop added to it, so that it stays zero while nothing drops.
    fn merge_tracked(self, other: Self) -> Self {
        let (sum, rounding) = two_sum(self.sum, other.sum);
        let (errors, first) = two_sum(self.error, other.error);
        let (error, second) = two_sum(errors, rounding);
        let slack = (self.slack + other.slack) * (1.0 + 2f64.powi(-50))
            + 2.0 * (first.abs() + second.abs());
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
        let (value, rest, bound, ordinary) = self.quotient_parts::<false>(count as f64);
        if !ordinary {
            return None;
        }
        T::settle(value, rest, bound)
    }

    /// [`Carried::divided`] without branches, for lanes side by side, of a
    /// total that is not an exact zero: the value and whether the bound,
    /// which is not zero, settles it.
    #[inline(always)]
    fn divided_lane<T: Output, const FMA: bool>(self, count: f64) -> (T, bool) {
        let (value, rest, bound, ordinary) = self.quotient_parts::<FMA>(count);
        let (rounded, settled) = T::settle_lanes(value, rest, bound);
        (rounded, settled & ordinary)
    }

    /// The quotient by `count` as a two-sum's pair, the bound on it, and
    /// whether it is clear of underflow in the product's split, and of a
    /// zero or a value that is not finite; the product's error by a fused
    /// multiply-add where `FMA`.
    #[inline(always)]
    fn quotient_parts<const FMA: bool>(self, count: f64) -> (f64, f64, f64, bool) {
        let quotient = self.sum / count;
        let product = quotient * count;
        let rounding = if FMA {
            quotient.mul_add(count, -product)
        } else {
            product_error(quotient, count, product)
        };
        // `product` is within a rounding of `sum`, so the first difference
        // is exact and the remainder rounds once.
        let remainder = (self.sum - product) - rounding;
        let tail = (remainder + self.error) / count;
        let (value, rest) = two_sum(quotient, tail);
        let bound = self.slack / count * (1.0 + 2f64.powi(-50)) + tail.abs() * 2f64.powi(-50);
        let ordinary = (value.abs() >= 2f64.powi(-900)) & (value.abs() <= 2f64.powi(900));
        (value, rest, bound, ordinary)
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

/// [`plain_squares_lanes`] in 256-bit registers, for a CPU that has them.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
fn plain_squares_avx(values: &[f64], centre: f64) -> f64 {
    use std::arch::x86_64::_mm256_loadu_pd;
    plain_squares_avx_with(values, centre, |chunk, half| {
        // SAFETY: four values from `4 * half` lie inside the chunk.
        unsafe { _mm256_loadu_pd(chunk.as_ptr().add(4 * half)) }
    })
}

/// [`plain_squares_avx`] for float32 values, each four read and widened at
/// once.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
fn plain_squares_avx_f32(values: &[f32], centre: f64) -> f64 {
    use std::arch::x86_64::{_mm_loadu_ps, _mm256_cvtps_pd};
    plain_squares_avx_with(values, centre, |chunk, half| {
        // SAFETY: four values from `4 * half` lie inside the chunk.
        _mm256_cvtps_pd(unsafe { _mm_loadu_ps(chunk.as_ptr().add(4 * half)) })
    })
}

/// The lanes of [`plain_squares_lanes`] in two 256-bit registers, `load`
/// giving half a chunk's values in float64: each deviation and its square
/// rounded apart, as there.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
fn plain_squares_avx_with<I: Element>(
    values: &[I],
    centre: f64,
    load: impl Fn(&[I; LANES], usize) -> std::arch::x86_64::__m256d,
) -> f64 {
    use std::arch::x86_64::{
        _mm256_add_pd, _mm256_mul_pd, _mm256_set1_pd, _mm256_setzero_pd, _mm256_storeu_pd,
        _mm256_sub_pd,
    };
    let middle = _mm256_set1_pd(centre);
    let mut sums = [_mm256_setzero_pd(); 2];
    let (chunks, rest) = values.as_chunks::<LANES>();
    for chunk in chunks {
        for half in 0..2 {
            let deviation = _mm256_sub_pd(load(chunk, half), middle);
            sums[half] = _mm256_add_pd(sums[half], _mm256_mul_pd(deviation, deviation));
        }
    }
    let mut lanes = [0f64; LANES];
    for (half, vector) in sums.into_iter().enumerate() {
        // SAFETY: four lanes from `4 * half` lie inside the array.
        unsafe { _mm256_storeu_pd(lanes.as_mut_ptr().add(4 * half), vector) };
    }
    for value in rest {
        let deviation = value.wide() - centre;
        lanes[0] += deviation * deviation;
    }
    lanes.iter().sum()
}

/// The plain float64 sum of `values` and of their magnitudes, block by block
/// with the blocks merged pairwise, and a bound on how far the sum is from
/// the exact one.
fn plain_with<I: Element>(values: &[I], parallel: bool) -> (f64, f64) {
    let (sum, magnitude) = if values.len() <= BLOCK {
        I::plain_block(values)
    } else {
        let partials: Vec<(f64, f64)> =
            if parallel && values.len() >= crate::ops::map::PAR_THRESHOLD {
                let len = values.len();
                crate::parallel::map_tasks(len.div_ceil(BLOCK), &|nth| {
                    I::plain_block(&values[nth * BLOCK..(nth * BLOCK + BLOCK).min(len)])
                })
            } else {
                values.chunks(BLOCK).map(I::plain_block).collect()
            };
        pairwise_fold(partials, (0.0, 0.0), |a, b| (a.0 + b.0, a.1 + b.1))
    };
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

/// The total of the weights in each of `bins` bins, `label(i)` the bin of
/// `weights[i]` -- none for a label past the last -- each the value nearest
/// the exact total of its weights.
///
/// Each bin's weights are added by two-sums, the errors and their
/// magnitudes beside, across the pool in bands merged pairwise; a bin its
/// bound cannot settle -- a NaN, an infinity, or a total on a halfway point
/// -- has its weights gathered, all such bins in one pass, and summed as
/// [`sum_to`] sums them.
pub(crate) fn bin_totals<I: Element, T: Output>(
    weights: &[I],
    bins: usize,
    label: impl Fn(usize) -> usize + Sync,
) -> Vec<T> {
    let totals = crate::ops::search::tally_bands(
        weights.len(),
        bins,
        BinTotal::default(),
        |first, last, slots: &mut [BinTotal]| {
            for index in first..last {
                if let Some(slot) = slots.get_mut(label(index)) {
                    slot.add(weights[index].wide());
                }
            }
        },
    );
    let mut out: Vec<T> = Vec::with_capacity(bins);
    let mut open = Vec::new();
    for (bin, total) in totals.iter().enumerate() {
        // The errors pass through no more additions than were made.
        let additions = (total.additions + LANES as u64) as f64;
        let carried = Carried {
            sum: total.sum,
            error: total.error,
            slack: total.magnitude * additions * 2f64.powi(-52),
        };
        out.push(carried.rounded().unwrap_or_else(|| {
            open.push(bin);
            T::narrow(0.0)
        }));
    }
    if !open.is_empty() {
        let mut place = vec![usize::MAX; bins];
        for (nth, &bin) in open.iter().enumerate() {
            place[bin] = nth;
        }
        let mut gathered: Vec<Vec<I>> = vec![Vec::new(); open.len()];
        for (index, &weight) in weights.iter().enumerate() {
            let nth = place.get(label(index)).copied().unwrap_or(usize::MAX);
            if let Some(kept) = gathered.get_mut(nth) {
                kept.push(weight);
            }
        }
        for (bin, kept) in open.into_iter().zip(gathered) {
            out[bin] = sum_to(&kept);
        }
    }
    out
}

/// One bin's weights added so far: the sum, the exact errors of its
/// additions added up, their magnitudes, and how many additions there were.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct BinTotal {
    sum: f64,
    error: f64,
    magnitude: f64,
    additions: u64,
}

impl BinTotal {
    #[inline(always)]
    fn add(&mut self, value: f64) {
        let (sum, rounding) = two_sum(self.sum, value);
        self.sum = sum;
        self.error += rounding;
        self.magnitude += rounding.abs();
        self.additions += 1;
    }
}

impl std::ops::Add for BinTotal {
    type Output = Self;

    /// Two bands' totals: the sums by a two-sum, the errors added twice more.
    fn add(self, other: Self) -> Self {
        let (sum, rounding) = two_sum(self.sum, other.sum);
        Self {
            sum,
            error: (self.error + other.error) + rounding,
            magnitude: self.magnitude + other.magnitude + rounding.abs(),
            additions: self.additions + other.additions + 2,
        }
    }
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

/// Columns a task takes of a strided axis: rows of up to this many are read
/// whole, one stream through memory, and a band's sums, errors and
/// magnitudes stay in the first-level cache.
const COLUMNS: usize = 1024;

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
    let (merged, parts, rows_per_part) = if T::PLAIN {
        band_totals::<I, true>(input, outer, len, inner, 0..len, parallel)
    } else {
        band_totals::<I, false>(input, outer, len, inner, 0..len, parallel)
    };
    let output = |index: usize| {
        let (o, column) = (index / inner, index % inner);
        let (band, offset) = (column / COLUMNS, column % COLUMNS);
        let gathered = || -> Vec<I> {
            (0..len)
                .map(|row| input[o * slab + row * inner + column])
                .collect()
        };
        let total = merged[o * bands + band][offset];
        if T::PLAIN {
            let bound = plain_column_bound(total.slack, inner, band, parts, rows_per_part);
            return reduction
                .settled_plain(total.sum, bound, len)
                .unwrap_or_else(|| reduction.of_values(&gathered(), false));
        }
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

/// How far a plain column total of [`band_totals`] can be from the exact
/// one, from the `magnitude` it carries: a value passes through its lane's
/// additions -- a part's rows over the rows a pass takes, and the rows left
/// over -- the merge of its column's lanes, and the merge of the parts,
/// pairwise.
fn plain_column_bound(
    magnitude: f64,
    inner: usize,
    band: usize,
    parts: usize,
    rows_per_part: usize,
) -> f64 {
    let width = COLUMNS.min(inner - band * COLUMNS);
    let group = if width == inner {
        (64 / width).max(1)
    } else {
        1
    };
    let depth = (rows_per_part / group + 2 * group + parts.ilog2() as usize + 18) as f64;
    magnitude * depth * 2f64.powi(-52)
}

/// The carried totals over `rows` of every column of `input`, laid out
/// `[outer, len, inner]`, indexed by `o * bands + band` and then by column
/// within the band; with `PLAIN`, as [`columns_part`] gives them.
///
/// A task takes a band of up to [`COLUMNS`] columns, and where there are
/// too few bands to spread across the pool the rows are cut into parts,
/// merged pairwise in order. Also the number of parts and the rows a part
/// takes, which a plain total's bound counts.
fn band_totals<I: Element, const PLAIN: bool>(
    input: &[I],
    outer: usize,
    len: usize,
    inner: usize,
    rows: std::ops::Range<usize>,
    parallel: bool,
) -> (Vec<Vec<Carried>>, usize, usize) {
    let slab = len * inner;
    let bands = inner.div_ceil(COLUMNS);
    let count = rows.len();
    let parts = if outer * bands >= SPREAD {
        1
    } else {
        let per_part = (BLOCK / inner.min(COLUMNS)).max(64);
        count.div_ceil(per_part).max(1)
    };
    let rows_per_part = count.div_ceil(parts);
    let task = |nth: usize| {
        let (o, rest) = (nth / (bands * parts), nth % (bands * parts));
        let (band, part) = (rest / parts, rest % parts);
        let columns = band * COLUMNS..((band + 1) * COLUMNS).min(inner);
        let first = rows.start + part * rows_per_part;
        let part_rows = first..(first + rows_per_part).min(rows.end);
        let slab = &input[o * slab..(o + 1) * slab];
        columns_part::<I, PLAIN>(slab, inner, part_rows, columns)
    };
    let tasks = outer * bands * parts;
    let partials: Vec<Vec<Carried>> = if parallel {
        crate::parallel::map_tasks(tasks, &task)
    } else {
        (0..tasks).map(task).collect()
    };
    // Each band's parts merged pairwise, a band's columns side by side.
    let merged = if parts == 1 {
        partials
    } else {
        let mut partials = partials.into_iter();
        (0..outer * bands)
            .map(|_| {
                let group: Vec<Vec<Carried>> = partials.by_ref().take(parts).collect();
                if PLAIN {
                    pairwise_fold_vectors(group, |a, b| Carried {
                        sum: a.sum + b.sum,
                        error: 0.0,
                        slack: a.slack + b.slack,
                    })
                } else {
                    pairwise_fold_vectors(group, Carried::merge)
                }
            })
            .collect()
    };
    (merged, parts, rows_per_part)
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
    let run = |row: usize| &slab[row * inner + columns.start..][..lanes];
    let mut row = rows.start;
    // A block of runs at a time, in order, then the runs left over.
    while row + ROW_BLOCK * group <= rows.end {
        let block = std::array::from_fn(|nth| run(row + nth * group));
        accumulate_block::<I, PLAIN>(&mut sums, &mut errors, &mut magnitudes, block);
        row += ROW_BLOCK * group;
    }
    while row + group <= rows.end {
        accumulate_columns::<I, PLAIN>(&mut sums, &mut errors, &mut magnitudes, run(row));
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

/// Rows a column kernel takes into its accumulators at once.
const ROW_BLOCK: usize = 4;

/// [`accumulate_columns`] for [`ROW_BLOCK`] rows, in order: each column's
/// accumulators held in registers, [`LANES`] columns at a time, across the
/// rows, and loaded and stored once for the block rather than once a row.
/// Each column takes the same additions in the same order.
#[inline(always)]
fn accumulate_block<I: Element, const PLAIN: bool>(
    sums: &mut [f64],
    errors: &mut [f64],
    magnitudes: &mut [f64],
    rows: [&[I]; ROW_BLOCK],
) {
    let whole = sums.len() / LANES * LANES;
    for start in (0..whole).step_by(LANES) {
        let lanes = start..start + LANES;
        let mut sum: [f64; LANES] = std::array::from_fn(|lane| sums[start + lane]);
        let mut error: [f64; LANES] = std::array::from_fn(|lane| errors[start + lane]);
        let mut magnitude: [f64; LANES] = std::array::from_fn(|lane| magnitudes[start + lane]);
        for row in &rows {
            let values = &row[lanes.clone()];
            for lane in 0..LANES {
                let value = values[lane].wide();
                if PLAIN {
                    sum[lane] += value;
                    magnitude[lane] += value.abs();
                } else {
                    let (total, rounding) = two_sum(sum[lane], value);
                    sum[lane] = total;
                    error[lane] += rounding;
                    magnitude[lane] += rounding.abs();
                }
            }
        }
        sums[lanes.clone()].copy_from_slice(&sum);
        if !PLAIN {
            errors[lanes.clone()].copy_from_slice(&error);
        }
        magnitudes[lanes].copy_from_slice(&magnitude);
    }
    for row in &rows {
        accumulate_columns::<I, PLAIN>(
            &mut sums[whole..],
            &mut errors[whole..],
            &mut magnitudes[whole..],
            &row[whole..],
        );
    }
}

/// Each column's square about its centre added in, as [`deviation_lanes`]
/// adds one, for each of `rows` in order -- [`ROW_BLOCK`] of them, or one --
/// the columns' accumulators held in registers, [`LANES`] at a time, across
/// them: with `PLAIN` the plain square, otherwise the exact deviation's
/// square split, by a fused multiply-add where `FMA`, its errors and the
/// cross terms beside it, and their magnitudes.
#[inline(always)]
fn accumulate_squares<I: Element, const PLAIN: bool, const FMA: bool, const ROWS: usize>(
    sums: &mut [f64],
    errors: &mut [f64],
    magnitudes: &mut [f64],
    centres: &[f64],
    rows: [&[I]; ROWS],
) {
    let width = sums.len();
    let add = |sum: &mut f64, error: &mut f64, magnitude: &mut f64, value: f64, centre: f64| {
        if PLAIN {
            let deviation = value - centre;
            *sum += deviation * deviation;
            return;
        }
        let (head, tail) = two_sum(value, -centre);
        let square = head * head;
        let square_error = if FMA {
            head.mul_add(head, -square)
        } else {
            product_error(head, head, square)
        };
        let cross = 2.0 * head * tail + tail * tail;
        let (next, rounding) = two_sum(*sum, square);
        *sum = next;
        *error += rounding + (square_error + cross);
        *magnitude += rounding.abs() + square_error.abs() + 2.0 * cross.abs();
    };
    let whole = width / LANES * LANES;
    for start in (0..whole).step_by(LANES) {
        let lanes = start..start + LANES;
        let centre: [f64; LANES] = std::array::from_fn(|lane| centres[start + lane]);
        let mut sum: [f64; LANES] = std::array::from_fn(|lane| sums[start + lane]);
        let mut error: [f64; LANES] = std::array::from_fn(|lane| errors[start + lane]);
        let mut magnitude: [f64; LANES] = std::array::from_fn(|lane| magnitudes[start + lane]);
        for row in &rows {
            let values = &row[lanes.clone()];
            for lane in 0..LANES {
                let value = values[lane].wide();
                add(
                    &mut sum[lane],
                    &mut error[lane],
                    &mut magnitude[lane],
                    value,
                    centre[lane],
                );
            }
        }
        sums[lanes.clone()].copy_from_slice(&sum);
        if !PLAIN {
            errors[lanes.clone()].copy_from_slice(&error);
            magnitudes[lanes].copy_from_slice(&magnitude);
        }
    }
    for row in &rows {
        for column in whole..width {
            let value = row[column].wide();
            let (sum, error, magnitude) = (
                &mut sums[column],
                &mut errors[column],
                &mut magnitudes[column],
            );
            add(sum, error, magnitude, value, centres[column]);
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
pub(super) fn fused() -> bool {
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

/// A float dtype a trapezoidal integral is rounded to.
pub(crate) trait Trapezoid: Element + Output {
    /// The most that a product of two of it, or half of a float64 total of
    /// them, can lose carried in float64 below the normals: nothing for a
    /// float32, whose products and halves float64 holds exactly.
    const LOSS: f64;
}

impl Trapezoid for f32 {
    const LOSS: f64 = 0.0;
}

impl Trapezoid for f64 {
    const LOSS: f64 = f64::from_bits(1);
}

/// The value nearest the trapezoidal integral of `y`, `step` apart.
///
/// It is `step` times the samples' total with each end halved: the total
/// carried as a sum is, and the product formed with its exact rounding and a
/// bound; where the bound cannot say which value is nearest, the products
/// are taken exactly. A NaN or an infinity among the samples or as the step
/// gives what the intervals' own formula gives in float64. Across the pool
/// for a long run, as a sum is.
pub(crate) fn trapezoid_spaced<T: Trapezoid>(y: &[T], step: T) -> T {
    let n = y.len();
    if n < 2 {
        return T::narrow(0.0);
    }
    let interior = &y[1..n - 1];
    if T::PLAIN {
        // A plain float64 total, within its bound, settles a float32.
        let (sum, slack) = plain_with(interior, true);
        let plain = Carried {
            sum,
            error: 0.0,
            slack,
        };
        if let Some(value) = spaced_settled(plain, y[0], y[n - 1], step) {
            return value;
        }
    }
    spaced_settled(carried_with(interior, true), y[0], y[n - 1], step)
        .unwrap_or_else(|| spaced_exactly(y, step))
}

/// [`trapezoid_spaced`] of each column of `input`, laid out `[outer, len,
/// inner]` with `len` at least two, into `out`, laid out `[outer, inner]`:
/// each column's interior totalled as a sum along an axis totals it, a band
/// of columns row by row.
pub(crate) fn trapezoid_spaced_columns<T: Trapezoid>(
    input: &[T],
    len: usize,
    inner: usize,
    step: T,
    out: &mut [T],
) {
    debug_assert!(len >= 2 && input.len() == out.len() * len);
    let outer = out.len() / inner;
    let parallel = input.len() >= crate::ops::map::PAR_THRESHOLD;
    let rows = 1..len - 1;
    let (totals, parts, rows_per_part) = if T::PLAIN {
        band_totals::<T, true>(input, outer, len, inner, rows, parallel)
    } else {
        band_totals::<T, false>(input, outer, len, inner, rows, parallel)
    };
    let bands = inner.div_ceil(COLUMNS);
    let output = |index: usize| {
        let (o, column) = (index / inner, index % inner);
        let (band, offset) = (column / COLUMNS, column % COLUMNS);
        let start = o * len * inner + column;
        let at = |row: usize| input[start + row * inner];
        let total = totals[o * bands + band][offset];
        let interior = if T::PLAIN {
            Carried {
                sum: total.sum,
                error: 0.0,
                slack: plain_column_bound(total.slack, inner, band, parts, rows_per_part),
            }
        } else {
            total
        };
        spaced_settled(interior, at(0), at(len - 1), step).unwrap_or_else(|| {
            let run: Vec<T> = (0..len).map(at).collect();
            trapezoid_spaced(&run, step)
        })
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

/// The value nearest `step` times the carried `interior` with half of
/// `first` and of `last` added, when the bound says which it is: not for a
/// NaN or an infinity, nor a total or a step so far out of the range that
/// the product's rounding might not be exact.
fn spaced_settled<T: Trapezoid>(interior: Carried, first: T, last: T, step: T) -> Option<T> {
    let step = step.wide();
    // The ends' sum and its rounding, each halved: exactly, but for a last
    // bit below the normals, which the slack allows for.
    let (ends, rounding) = two_sum(first.wide(), last.wide());
    let half = |value: f64| {
        let halved = 0.5 * value;
        (halved, if halved * 2.0 == value { 0.0 } else { T::LOSS })
    };
    let ((sum, lost), (error, also_lost)) = (half(ends), half(rounding));
    let total = interior.merge(Carried {
        sum,
        error,
        slack: lost + also_lost,
    });
    if !(step.is_finite() && total.sum.is_finite() && total.error.is_finite()) {
        return None;
    }
    if step == 0.0 || (total.sum == 0.0 && total.error == 0.0 && total.slack == 0.0) {
        return Some(T::narrow(0.0));
    }
    let (value, rest) = two_sum(total.sum, total.error);
    let product = value * step;
    let ordinary = |v: f64| (v.abs() >= 2f64.powi(-900)) & (v.abs() <= 2f64.powi(900));
    if !(ordinary(value) && ordinary(step) && ordinary(product)) {
        return None;
    }
    // `step · (value + rest)` is `product` and its exact rounding, and the
    // rest's product, which rounds once, as does the tail they make; the
    // carried total's slack scales with the step.
    let carried = rest * step;
    let tail = product_error(value, step, product) + carried;
    let (sum, rest) = two_sum(product, tail);
    let bound = total.slack * step.abs() * (1.0 + 2f64.powi(-50))
        + (carried.abs() + tail.abs()) * 2f64.powi(-52)
        + T::LOSS;
    T::settle(sum, rest, bound)
}

/// [`trapezoid_spaced`] where the bound cannot decide: the intervals' own
/// formula in float64 for a NaN or an infinity, and otherwise the exact sum
/// of the samples' products with their weights -- half a step at each end
/// and a step inside, or for a step too small to halve, twice the integral
/// halved in whole numbers.
#[cold]
#[inline(never)]
fn spaced_exactly<T: Trapezoid>(y: &[T], step: T) -> T {
    let (n, step) = (y.len(), step.wide());
    if !(step.is_finite() && y.iter().all(|value| value.wide().is_finite())) {
        let total: f64 = y
            .windows(2)
            .map(|pair| step * ((pair[0].wide() + pair[1].wide()) * 0.5))
            .sum();
        return T::narrow(total);
    }
    let ends = [y[0], y[n - 1]].into_iter().map(Element::wide);
    let inside = y[1..n - 1].iter().map(|value| value.wide());
    let half = 0.5 * step;
    let total = if half * 2.0 == step {
        let pairs = ends
            .map(move |v| (half, v))
            .chain(inside.map(move |v| (step, v)));
        Total::of_products(pairs).map(|total| total.rounded())
    } else {
        let twice = 2.0 * step;
        let pairs = ends
            .map(move |v| (step, v))
            .chain(inside.map(move |v| (twice, v)));
        Total::of_products(pairs).map(|total| total.divided(2))
    };
    total.unwrap_or_else(T::narrow)
}

/// The value nearest the trapezoidal integral of `y` at the positions `x`.
///
/// `Σ (x[i+1] - x[i]) (y[i] + y[i+1]) / 2` telescopes to half of
/// `x[n-1] y[n-1] - x[0] y[0] + Σ (x[i+1] y[i] - x[i] y[i+1])`, every term a
/// product of a position and a sample: two dots of the samples with the
/// positions a place along, carried as a dot is, their difference halved
/// with a bound; where it cannot say which value is nearest, the products
/// are taken exactly. A NaN or an infinity among the samples or the
/// positions, whose telescoped terms would cancel, gives what the
/// intervals' own formula gives in float64.
pub(crate) fn trapezoid_at<T: Trapezoid>(y: &[T], x: &[T]) -> T {
    let n = y.len();
    debug_assert_eq!(n, x.len());
    if n < 2 {
        return T::narrow(0.0);
    }
    let ahead = T::carried_dot(&x[1..], &y[..n - 1]);
    let behind = T::carried_dot(&x[..n - 1], &y[1..]);
    let ends = T::carried_dot(&[x[n - 1], T::narrow(-x[0].wide())], &[y[n - 1], y[0]]);
    let total = ahead
        .merge(Carried {
            sum: -behind.sum,
            error: -behind.error,
            slack: behind.slack,
        })
        .merge(ends);
    if !(total.sum.is_finite() && total.error.is_finite()) {
        return at_exactly(y, x);
    }
    let halved = |slack: f64| {
        let (value, rest) = two_sum(total.sum, total.error);
        if value == 0.0 && slack == 0.0 {
            return Some(T::narrow(0.0));
        }
        // Halved exactly, but for a last bit of `rest` below the normals.
        (value.abs() >= 2f64.powi(-1000))
            .then(|| T::settle(0.5 * value, 0.5 * rest, 0.5 * slack + T::LOSS))
            .flatten()
    };
    // As for a dot: a product's error is exact unless the product lies
    // within `2^-968` of zero, where it may have lost a subnormal's last bit,
    // which the first bound allows for in every product; a total that bound
    // cannot settle is settled without it when no product is that small.
    let tiny = |a: T, b: T| {
        let (a, b) = (a.wide(), b.wide());
        a != 0.0 && b != 0.0 && (a * b).abs() < 2f64.powi(-968)
    };
    halved(total.slack + 2.0 * n as f64 * T::LOSS)
        .or_else(|| {
            let small = T::LOSS != 0.0
                && (tiny(x[0], y[0])
                    || tiny(x[n - 1], y[n - 1])
                    || (0..n - 1).any(|i| tiny(x[i + 1], y[i]) || tiny(x[i], y[i + 1])));
            if small { None } else { halved(total.slack) }
        })
        .unwrap_or_else(|| at_exactly(y, x))
}

/// [`trapezoid_at`] where the bound cannot decide: the intervals' own
/// formula in float64 for a NaN or an infinity, and otherwise the exact sum
/// of the telescoped products, halved in whole numbers.
#[cold]
#[inline(never)]
fn at_exactly<T: Trapezoid>(y: &[T], x: &[T]) -> T {
    let n = y.len();
    let finite = |values: &[T]| values.iter().all(|value| value.wide().is_finite());
    if !(finite(y) && finite(x)) {
        let total: f64 = (0..n - 1)
            .map(|i| (x[i + 1].wide() - x[i].wide()) * ((y[i].wide() + y[i + 1].wide()) * 0.5))
            .sum();
        return T::narrow(total);
    }
    let ends = [
        (x[n - 1].wide(), y[n - 1].wide()),
        (-x[0].wide(), y[0].wide()),
    ];
    let inside = (0..n - 1).flat_map(move |i| {
        [
            (x[i + 1].wide(), y[i].wide()),
            (-x[i].wide(), y[i + 1].wide()),
        ]
    });
    Total::of_products(ends.into_iter().chain(inside))
        .map(|total| total.divided(2))
        .unwrap_or_else(T::narrow)
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
    // product, which may be a subnormal, within the least subnormal -- added
    // only where it may be, since arithmetic on a subnormal is slow.
    let rounding = (first.abs() + second.abs() + third.abs() + residual.abs()) * 2f64.powi(-52)
        + underflow_allowance(d2, third);
    let off = rounding + sn + quotient.abs() * sd;
    let correction = residual / d1;
    let divisor = d1.abs() * (1.0 - 2f64.powi(-29));
    let bound = ((off + residual.abs() * spread / d1.abs()) / divisor
        + correction.abs() * 2f64.powi(-52))
        * (1.0 + 2f64.powi(-40));
    let (value, rest) = two_sum(quotient, correction);
    T::settle(value, rest, bound)
}

/// What a quotient's product with a divisor's low part `low` may have lost
/// to underflow, as `product`: up to the least subnormal four times over
/// for a product that small, and nothing otherwise.
#[inline(always)]
fn underflow_allowance(low: f64, product: f64) -> f64 {
    if low != 0.0 && product.abs() < f64::MIN_POSITIVE {
        f64::from_bits(4)
    } else {
        0.0
    }
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

/// The float64 nearest `(near · a + far · b) / d`, for `a`, `b` and `d`
/// each the exact sum of a pair, when the bound says which it is.
///
/// This is a linear interpolation between the samples `near` and `far`, `a`
/// the query's distance to the far sample's position, `b` its distance to
/// the near one's and `d` theirs apart. The leading products are split
/// exactly, the rest carried beside with a bound, and the quotient settled
/// as a weighted mean's is; none for an operand so far out of the range
/// that a product's split might not be exact. A product's error is one
/// fused multiply-add where `FMA`.
#[inline(always)]
pub(crate) fn interpolated<const FMA: bool>(
    near: f64,
    far: f64,
    a: (f64, f64),
    b: (f64, f64),
    d: (f64, f64),
) -> Option<f64> {
    let ordinary =
        |value: f64| value == 0.0 || (2f64.powi(-450)..=2f64.powi(450)).contains(&value.abs());
    if !(ordinary(near) && ordinary(far) && ordinary(a.0) && ordinary(b.0)) {
        return None;
    }
    let leading = near * a.0;
    let leading_error = crate::ops::simd::product_error::<FMA>(near, a.0, leading);
    let trailing = far * b.0;
    let trailing_error = crate::ops::simd::product_error::<FMA>(far, b.0, trailing);
    let (sum, rounding) = two_sum(leading, trailing);
    let (small_near, small_far) = (near * a.1, far * b.1);
    let tail = ((rounding + leading_error) + trailing_error) + (small_near + small_far);
    // Four additions and the two small products, each within `u` of its
    // result -- or of the least subnormal, for a product that small.
    let magnitude = rounding.abs()
        + leading_error.abs()
        + trailing_error.abs()
        + small_near.abs()
        + small_far.abs();
    let lost = |x: f64, y: f64| x != 0.0 && y != 0.0 && (x * y).abs() < 2f64.powi(-1021);
    let underflow = if lost(near, a.1) || lost(far, b.1) {
        2.0 * f64::from_bits(1)
    } else {
        0.0
    };
    let numerator = Carried {
        sum,
        error: tail,
        slack: magnitude * 2f64.powi(-50) + underflow,
    };
    ratio_settled::<f64, FMA>(numerator, d)
}

/// The value nearest `N / (d.0 + d.1)` for the total `numerator` carries
/// and a two-sum's pair `d`, when the bound says which it is.
///
/// As [`quotient_settled`] settles a quotient, with the divisor exact and
/// one division: the leading quotient from the divisor's reciprocal, within
/// two roundings of `n1 / d1`, so that `n1` less its product with `d1` is
/// exact; the residual's few roundings; and the correction, the residual by
/// the reciprocal, which is within `2.0001u` of the residual over the whole
/// divisor.
#[inline(always)]
pub(crate) fn ratio_settled<T: Output, const FMA: bool>(
    numerator: Carried,
    d: (f64, f64),
) -> Option<T> {
    let (n1, n2) = two_sum(numerator.sum, numerator.error);
    let (d1, d2) = d;
    let sn = numerator.slack;
    if !(d1.is_finite() && d1 != 0.0) || d2.abs() > d1.abs() * 2f64.powi(-52) {
        return None;
    }
    if n1 == 0.0 && n2 == 0.0 && sn == 0.0 {
        return Some(T::narrow(0.0 / d1));
    }
    let ordinary = |value: f64| (2f64.powi(-900)..=2f64.powi(900)).contains(&value.abs());
    if d2 == 0.0 && d1.to_bits() & ((1 << 52) - 1) == 0 && ordinary(d1) && ordinary(n1) {
        // A power of two: the pair over it is exact, a halfway point
        // included, which no bound above zero could settle.
        let scale = 1.0 / d1;
        let (value, rest) = (n1 * scale, n2 * scale);
        if ordinary(value) && (rest == 0.0 || rest.abs() >= f64::MIN_POSITIVE) {
            return T::settle(value, rest, sn * scale.abs());
        }
    }
    let inverse = 1.0 / d1;
    let quotient = n1 * inverse;
    if !(ordinary(n1) && ordinary(d1) && ordinary(quotient)) {
        return None;
    }
    let product = quotient * d1;
    let product_rounding = crate::ops::simd::product_error::<FMA>(quotient, d1, product);
    let leading = n1 - product;
    let first = leading - product_rounding;
    let second = first + n2;
    let third = quotient * d2;
    let residual = second - third;
    let rounding = (first.abs() + second.abs() + third.abs() + residual.abs()) * 2f64.powi(-52)
        + underflow_allowance(d2, third);
    let correction = residual * inverse;
    // At least the reciprocal of the whole divisor's magnitude.
    let scale = inverse.abs() * (1.0 + 2f64.powi(-50));
    let bound = ((rounding + sn) * scale
        + residual.abs() * scale * 2f64.powi(-50)
        + correction.abs() * 2f64.powi(-52))
        * (1.0 + 2f64.powi(-40));
    let (value, rest) = two_sum(quotient, correction);
    T::settle(value, rest, bound).or_else(|| {
        // An exact numerator over an exact divisor lands on a halfway point
        // more often than a bound can tell: which side of it the quotient
        // lies on is the sign of the numerator less the point times the
        // divisor, every product of which is exact.
        if sn != 0.0 || d2 != 0.0 {
            return None;
        }
        T::split(value, rest, bound, |high, low| {
            let product = high * d1;
            let error = crate::ops::simd::product_error::<FMA>(high, d1, product);
            let sign = expansion_sign(&[n1, n2, -product, -error, -(low * d1)])?;
            Some(if d1 < 0.0 { sign.reverse() } else { sign })
        })
    })
}

/// The value nearest `(x - y) / divisor` of finite floats, when the bound
/// says which it is: the difference a two-sum's exact pair over the
/// divisor, as [`ratio_settled`] settles it.
#[inline(always)]
pub(crate) fn difference_over<T: Output, const FMA: bool>(
    x: f64,
    y: f64,
    divisor: f64,
) -> Option<T> {
    let (sum, error) = two_sum(x, -y);
    if !sum.is_finite() {
        return None;
    }
    let difference = Carried {
        sum,
        error,
        slack: 0.0,
    };
    ratio_settled::<T, FMA>(difference, (divisor, 0.0))
}

/// A product of three exact pairs, `x.0 + x.1` each, negated where
/// `negative`: a term of [`ratio_of_products`].
#[derive(Clone, Copy, Debug)]
pub(crate) struct Product {
    pub(crate) factors: [(f64, f64); 3],
    pub(crate) negative: bool,
}

impl Product {
    pub(crate) fn new(factors: [(f64, f64); 3], negative: bool) -> Self {
        Self { factors, negative }
    }
}

/// The value nearest the exact `Σ numerator / Σ denominator`, of products
/// of pairs of finite floats: each product carried with its leading part
/// split exactly and the quotient settled by a bound, or both sums taken
/// exactly where the bound cannot say. A product's error is one fused
/// multiply-add where `FMA`.
#[inline(always)]
pub(crate) fn ratio_of_products<T: Output, const FMA: bool>(
    numerator: &[Product],
    denominator: &[Product],
) -> T {
    match (
        carried_products::<FMA>(numerator),
        carried_products::<FMA>(denominator),
    ) {
        (Some(numerator), Some(denominator)) => {
            let (d1, d2) = two_sum(denominator.sum, denominator.error);
            if denominator.slack == 0.0 && d2 == 0.0 {
                // A divisor that is one float exactly: by its reciprocal,
                // and exactly over a power of two, a halfway point included.
                ratio_settled::<T, FMA>(numerator, (d1, 0.0))
            } else {
                quotient_settled(numerator, denominator)
            }
        }
        _ => None,
    }
    .unwrap_or_else(|| exact_ratio_of_products(numerator, denominator))
}

/// The sum of `products` carried, each product's pairs normalized by a
/// two-sum: `a b c = (a1 + a2)(b1 + b2)(c1 + c2)` is its leading part
/// `a1 b1 c1` split exactly, the three terms with one low part, and what is
/// left, below `4.01u²` of the leading part, with the roundings of the
/// terms, below `26u²` of it, allowed for -- none at all for a product with
/// no low part and an exact leading pair, which is then exact, and the
/// products added with what their errors' sums drop tracked, so that exact
/// products make an exact total. None for a factor or a partial product
/// out of the range where the splits are exact.
#[inline(always)]
fn carried_products<const FMA: bool>(products: &[Product]) -> Option<Carried> {
    let ordinary =
        |value: f64| value == 0.0 || (2f64.powi(-900)..=2f64.powi(900)).contains(&value.abs());
    let mut total = Carried::default();
    for product in products {
        let [a, b, c] = product.factors.map(|(high, low)| two_sum(high, low));
        let leading = a.0 * b.0;
        let whole = leading * c.0;
        let lost = |product: f64, x: f64, y: f64| product == 0.0 && x != 0.0 && y != 0.0;
        if !(ordinary(a.0)
            && ordinary(b.0)
            && ordinary(c.0)
            && ordinary(leading)
            && ordinary(whole))
            || lost(leading, a.0, b.0)
            || lost(whole, leading, c.0)
        {
            return None;
        }
        let leading_error = crate::ops::simd::product_error::<FMA>(a.0, b.0, leading);
        let whole_error = crate::ops::simd::product_error::<FMA>(leading, c.0, whole);
        let tail = whole_error
            + leading_error * c.0
            + leading * c.1
            + (a.0 * b.1) * c.0
            + (a.1 * b.0) * c.0;
        let exact = exact_tail::<FMA>(a, b, c, leading, leading_error, whole_error, tail);
        let sign = if product.negative { -1.0 } else { 1.0 };
        total = total.merge_tracked(Carried {
            sum: sign * whole,
            error: sign * tail,
            slack: if exact {
                0.0
            } else {
                whole.abs() * 2f64.powi(-98)
            },
        });
    }
    Some(total)
}

/// Whether a product's tail in [`carried_products`] is its exact rest: so
/// where at most one of the leading pair's error and the three low parts is
/// not zero, so that nothing is dropped, and the one term that makes, and
/// its sum with the leading part's own error, round to nothing.
#[inline(always)]
fn exact_tail<const FMA: bool>(
    a: (f64, f64),
    b: (f64, f64),
    c: (f64, f64),
    leading: f64,
    leading_error: f64,
    whole_error: f64,
    tail: f64,
) -> bool {
    let lows = [leading_error, a.1, b.1, c.1];
    let exact = |x: f64, y: f64, product: f64| {
        product.abs() >= 2f64.powi(-968)
            && crate::ops::simd::product_error::<FMA>(x, y, product) == 0.0
    };
    let term = match lows.iter().filter(|&&low| low != 0.0).count() {
        0 => return tail == whole_error,
        1 => {
            if leading_error != 0.0 {
                let term = leading_error * c.0;
                exact(leading_error, c.0, term).then_some(term)
            } else if c.1 != 0.0 {
                let term = leading * c.1;
                exact(leading, c.1, term).then_some(term)
            } else {
                let (x, y) = if b.1 != 0.0 { (a.0, b.1) } else { (a.1, b.0) };
                let partial = x * y;
                let term = partial * c.0;
                (exact(x, y, partial) && exact(partial, c.0, term)).then_some(term)
            }
        }
        _ => None,
    };
    term.is_some_and(|term| two_sum(whole_error, term).1 == 0.0)
}

/// [`ratio_of_products`] where the bound cannot decide: every product of a
/// part of each pair, exactly.
#[cold]
#[inline(never)]
fn exact_ratio_of_products<T: Output>(numerator: &[Product], denominator: &[Product]) -> T {
    let expand = |products: &[Product]| {
        let mut terms = Vec::with_capacity(8 * products.len());
        for product in products {
            let [a, b, c] = product.factors;
            for x in [a.0, a.1] {
                for y in [b.0, b.1] {
                    for z in [c.0, c.1] {
                        if x != 0.0 && y != 0.0 && z != 0.0 {
                            terms.push((if product.negative { -x } else { x }, y, z));
                        }
                    }
                }
            }
        }
        Total::of_terms(&terms)
    };
    expand(numerator).quotient(expand(denominator))
}

/// The float64 nearest the exact `Σ a[i] · b[i] / Σ divisor`, finite
/// operands all; see [`exact_quotient`].
pub(crate) fn exact_ratio(a: &[f64], b: &[f64], divisor: &[f64]) -> f64 {
    debug_assert_eq!(a.len(), b.len());
    exact_quotient(
        a.iter().copied().zip(b.iter().copied()),
        divisor.iter().copied(),
    )
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
    let (value, rest, bound, ordinary) = root_parts::<false>(carried);
    if !ordinary {
        return None;
    }
    T::settle(value, rest, bound)
}

/// [`root_settled`] without branches, for lanes side by side, of a total
/// that is not an exact zero: the value and whether the bound, which is not
/// zero, settles it.
#[inline(always)]
fn root_lane<T: Output, const FMA: bool>(carried: Carried) -> (T, bool) {
    let (value, rest, bound, ordinary) = root_parts::<FMA>(carried);
    let (rounded, settled) = T::settle_lanes(value, rest, bound);
    (rounded, settled & ordinary)
}

/// The root of a carried total as a two-sum's pair, the bound on it, and
/// whether the total is clear of the ends of the range and known well
/// enough for the bound to hold; the square's error by a fused multiply-add
/// where `FMA`.
#[inline(always)]
fn root_parts<const FMA: bool>(carried: Carried) -> (f64, f64, f64, bool) {
    let (s1, s2) = two_sum(carried.sum, carried.error);
    // Clear of underflow in the square's split; a NaN, an infinity or a
    // negative total fails this too.
    let ordinary = (s1 >= 2f64.powi(-900)) & (s1 <= 2f64.powi(900));
    let known = s2.abs() + carried.slack <= s1 * 2f64.powi(-30);
    let root = s1.sqrt();
    let square = root * root;
    let square_rounding = if FMA {
        root.mul_add(root, -square)
    } else {
        product_error(root, root, square)
    };
    // Within three roundings of `s1`, so the difference is exact.
    let first = (s1 - square) - square_rounding;
    let residual = first + s2;
    let off = (first.abs() + residual.abs()) * 2f64.powi(-52) + f64::from_bits(4) + carried.slack;
    // One division: `1 / 2r` is within `u` of itself and each product with
    // it within another, so the correction is within `2u` of its own, and
    // the bound's terms take room for both.
    let half = 0.5 / root;
    let correction = residual * half;
    let room = 1.0 + 2f64.powi(-50);
    let ratio = (residual.abs() + off) * (2.0 * half) * room;
    let bound = (off * half * room
        + ratio * ratio * half * 0.25 * 1.01
        + correction.abs() * 2f64.powi(-51))
        * (1.0 + 2f64.powi(-40));
    let (value, rest) = two_sum(root, correction);
    (value, rest, bound, ordinary & known)
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

/// The value nearest the variance of `values` over `count - ddof`, or with
/// `root` the value nearest its square root: the exact
/// `(N Σx² - (Σx)²) / (N (N - ddof))`, NaN for no more values than `ddof`
/// and for any value that is not finite.
///
/// A float32 variance is first taken plainly in float64, by
/// [`plain_variance`]. Otherwise, and where that cannot say, the sum is
/// carried and `c`, the float64 nearest the mean it gives, taken as a
/// centre: the exact squares about it, `Σ (x - c)²`, are carried as
/// [`distance`] carries a 2-distance's, and [`centred_variance`] settles the
/// answer from the two -- two passes, a block of values to a task. Where
/// that cannot say, the variance is formed in whole numbers.
pub(crate) fn variance_of<I: Element, T: Output>(values: &[I], ddof: usize, root: bool) -> T {
    let count = values.len();
    if count <= ddof {
        return T::narrow(f64::NAN);
    }
    if T::PLAIN
        && let Some(value) = plain_variance(values, ddof, root)
    {
        return value;
    }
    // A NaN or an infinity among the values leaves the sum one too.
    let sum = if count < 2 * LANES {
        one_lane_sum(values)
    } else {
        carried_with(values, true)
    };
    let centre = sum.sum / count as f64;
    let settled = centre.is_finite().then(|| {
        let squares = if count <= BLOCK {
            deviation_block(values, centre)
        } else {
            let blocks = each_block(values, |block| deviation_block(block, centre));
            pairwise_fold(blocks, Carried::default(), Carried::merge)
        };
        centred_variance(sum, squares, centre, count, ddof, root)
    });
    settled
        .flatten()
        .unwrap_or_else(|| unsettled_variance(values, ddof, root))
}

/// The variance, or its root, of more than `ddof` values whose bounds did
/// not settle it: NaN for a value that is not finite, zero for equal values,
/// and otherwise formed in whole numbers.
#[cold]
#[inline(never)]
fn unsettled_variance<I: Element, T: Output>(values: &[I], ddof: usize, root: bool) -> T {
    if values.iter().any(|value| !value.wide().is_finite()) {
        return T::narrow(f64::NAN);
    }
    // Equal values, and only those, vary by nothing.
    let first = values[0].wide();
    if values.iter().all(|value| value.wide() == first) {
        return T::narrow(0.0);
    }
    exact_variance(
        values.iter().map(|value| value.wide()),
        values.len(),
        ddof,
        root,
    )
}

/// The variance, or its root, of `count` values from their carried `sum`,
/// `centre`, the float64 nearest its quotient by the count, and the carried
/// `squares` of their exact deviations from it, when the bounds settle it.
///
/// The squares about the mean `μ` are those about `c` less `N (μ - c)²`, a
/// sliver beside them, which is `d² / N` for `d = Σx - N c`: the sum's
/// leading part less `N c` is exact, both within a rounding of each other,
/// and the rest is rounded twice, within the sum's slack beside. The
/// quotient by the count is settled by its bound, or its root by
/// [`root_settled`].
fn centred_variance<T: Output>(
    sum: Carried,
    squares: Carried,
    centre: f64,
    count: usize,
    ddof: usize,
    root: bool,
) -> Option<T> {
    let (total, divisor) = (count as f64, (count - ddof) as f64);
    let (value, settled) = centred_lane::<T, false>(sum, squares, centre, total, divisor, root);
    settled.then_some(value)
}

/// [`centred_variance`] without branches, for lanes side by side, of `total`
/// values over `divisor`: the value and whether the bounds settle it, the
/// products' errors by fused multiply-adds where `FMA`.
#[inline(always)]
fn centred_lane<T: Output, const FMA: bool>(
    sum: Carried,
    squares: Carried,
    centre: f64,
    total: f64,
    divisor: f64,
    root: bool,
) -> (T, bool) {
    let product = centre * total;
    let rounding = if FMA {
        centre.mul_add(total, -product)
    } else {
        product_error(centre, total, product)
    };
    let low = sum.error - rounding;
    let distance = (sum.sum - product) + low;
    let off = sum.slack + (low.abs() + distance.abs()) * 2f64.powi(-52) + f64::from_bits(2);
    let sliver = distance * distance / total;
    // `(|d| + off)² - d²` over `N`, and the sliver's three roundings.
    let sliver_bound = (2.0 * distance.abs() + off) * off / total * (1.0 + 2f64.powi(-50))
        + sliver.abs() * 2f64.powi(-50);
    // A square's error, a cross term or the sliver rounded to the subnormal
    // range is off by under the least subnormal: fewer than four to a value.
    let allowance = 4.0 * total * f64::from_bits(1);
    let error = squares.error - sliver;
    let centred = Carried {
        sum: squares.sum,
        error,
        slack: squares.slack + sliver_bound + error.abs() * 2f64.powi(-52) + allowance,
    };
    let (value, settled) = if root {
        // The quotient carried, for its root: `q = s / k`, its exact
        // remainder, and the tail with the error added in, as `divided`
        // forms it.
        let quotient = centred.sum / divisor;
        let product = quotient * divisor;
        let rounding = if FMA {
            quotient.mul_add(divisor, -product)
        } else {
            product_error(quotient, divisor, product)
        };
        let rest = (centred.sum - product) - rounding;
        let tail = (rest + centred.error) / divisor;
        root_lane::<T, FMA>(Carried {
            sum: quotient,
            error: tail,
            slack: centred.slack / divisor * (1.0 + 2f64.powi(-50)) + tail.abs() * 2f64.powi(-50),
        })
    } else {
        centred.divided_lane::<T, FMA>(divisor)
    };
    // A sum, or a centre, past the largest float64 leaves no sliver.
    (value, settled & sliver.is_finite())
}

/// A float32 variance, or its root, from plain float64 sums, when their
/// bound settles it; see [`plain_settled`]. A NaN or an infinity, or a sum
/// past the largest float64, is left to the carried path.
fn plain_variance<I: Element, T: Output>(values: &[I], ddof: usize, root: bool) -> Option<T> {
    let (sum, bound) = plain_with(values, true);
    let centre = sum / values.len() as f64;
    if !centre.is_finite() {
        return None;
    }
    let squares = if values.len() <= BLOCK {
        I::plain_squares(values, centre)
    } else {
        let blocks = each_block(values, |block| I::plain_squares(block, centre));
        pairwise_fold(blocks, 0.0, |a, b| a + b)
    };
    let depth = plain_depth(values.len());
    plain_settled(squares, bound, centre, values.len(), depth, ddof, root)
}

/// A float32 variance, or its root, of `count` values from plain float64
/// sums, when their bound settles it; see [`plain_lane`].
fn plain_settled<T: Output>(
    squares: f64,
    bound: f64,
    centre: f64,
    count: usize,
    depth: f64,
    ddof: usize,
    root: bool,
) -> Option<T> {
    let divisor = (count - ddof) as f64;
    let (value, settled) = plain_lane(squares, bound, centre, count as f64, depth, divisor, root);
    settled.then_some(value)
}

/// A float32 variance, or with `root` its root, of `total` values over
/// `divisor` from plain float64 sums, and whether their bound settles it,
/// without branches for lanes side by side: `centre`, the quotient by the
/// count of a sum within `bound` of the exact one, and `squares`, the sum of
/// the squares about it, none through more than `depth` additions.
///
/// `c` is within `δ` of the mean `μ`, the bound over the count and the
/// quotient's rounding. Each square about `c` is rounded at most three
/// times, so `(depth + 6) u` of their sum bounds it, with twice the least
/// subnormal for each square that may have underflowed; and the squares
/// about the mean are less than those about `c` by `N (μ - c)² ≤ N δ²`. A
/// NaN or an infinity settles nothing.
#[inline(always)]
fn plain_lane<T: Output>(
    squares: f64,
    bound: f64,
    centre: f64,
    total: f64,
    depth: f64,
    divisor: f64,
    root: bool,
) -> (T, bool) {
    debug_assert!(
        T::PLAIN,
        "a plain float64 sum settles a float32 answer only"
    );
    let off = bound / total * (1.0 + 2f64.powi(-50)) + centre.abs() * 2f64.powi(-52);
    let sliver = off * off * total * (1.0 + 2f64.powi(-50));
    let spread = squares * (depth + 6.0) * 2f64.powi(-52) + total * f64::from_bits(2);
    let quotient = squares / divisor;
    let bound = (spread + sliver) / divisor * (1.0 + 2f64.powi(-50)) + quotient * 2f64.powi(-52);
    let (value, settled) = if root {
        // The root moves by at most the bound over twice the least root
        // within it, `sqrt(q - bound) ≥ r (1 - bound / q)` for a bound well
        // inside the quotient, and is rounded once itself.
        let root = quotient.sqrt();
        let moved = bound * (0.5 / root) * (1.0 + 2.0 * bound / quotient) * (1.0 + 2f64.powi(-50))
            + root * 2f64.powi(-52);
        let (value, settled) = T::settle_lanes(root, 0.0, moved * (1.0 + 2f64.powi(-40)));
        (value, settled & (quotient > 4.0 * bound))
    } else {
        T::settle_lanes(quotient, 0.0, bound)
    };
    // No square at all: each value is within 2^-536 of `c`, so the variance
    // is under 2^-1000 and its root under 2^-500, and both round to zero in
    // float32.
    let zero = squares == 0.0;
    (if zero { T::narrow(0.0) } else { value }, settled | zero)
}

/// One block's plain float64 `Σ (x - centre)²` in [`LANES`] lanes, the values
/// left over in lane 0 and the lanes in order: no square passes through more
/// than [`plain_depth`] additions.
#[inline(always)]
fn plain_squares_lanes<I: Element>(values: &[I], centre: f64) -> f64 {
    let mut lanes = [0f64; LANES];
    let (chunks, rest) = values.as_chunks::<LANES>();
    for chunk in chunks {
        for lane in 0..LANES {
            let deviation = chunk[lane].wide() - centre;
            lanes[lane] += deviation * deviation;
        }
    }
    for value in rest {
        let deviation = value.wide() - centre;
        lanes[0] += deviation * deviation;
    }
    lanes.iter().sum()
}

/// One block's plain float64 `Σ |x - y + shift|`, or with `euclidean`
/// `Σ (x - y + shift)²`, in [`LANES`] lanes, the values left over in lane 0
/// and the lanes in order: each difference rounded at most twice and its
/// square once more, and each term through at most `len / LANES + LANES`
/// additions.
#[inline(always)]
fn plain_differences_lanes<I: Element>(x: &[I], y: &[I], euclidean: bool, shift: f64) -> f64 {
    let term = |a: &I, b: &I| {
        let difference = (a.wide() - b.wide()) + shift;
        if euclidean {
            difference * difference
        } else {
            difference.abs()
        }
    };
    let mut lanes = [0f64; LANES];
    let (left, left_rest) = x.as_chunks::<LANES>();
    let (right, right_rest) = y.as_chunks::<LANES>();
    for (a, b) in left.iter().zip(right) {
        for lane in 0..LANES {
            lanes[lane] += term(&a[lane], &b[lane]);
        }
    }
    for (a, b) in left_rest.iter().zip(right_rest) {
        lanes[0] += term(a, b);
    }
    lanes.iter().sum()
}

/// [`plain_differences_lanes`] in 256-bit registers for float32 operands,
/// each four read and widened at once: the same lanes, terms and order.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
fn plain_differences_avx_f32<const EUCLIDEAN: bool>(x: &[f32], y: &[f32], shift: f64) -> f64 {
    use std::arch::x86_64::{
        _mm_loadu_ps, _mm256_add_pd, _mm256_andnot_pd, _mm256_cvtps_pd, _mm256_mul_pd,
        _mm256_set1_pd, _mm256_setzero_pd, _mm256_storeu_pd, _mm256_sub_pd,
    };
    let (sign, moved) = (_mm256_set1_pd(-0.0), _mm256_set1_pd(shift));
    let mut sums = [_mm256_setzero_pd(); 2];
    let (left, left_rest) = x.as_chunks::<LANES>();
    let (right, right_rest) = y.as_chunks::<LANES>();
    for (a, b) in left.iter().zip(right) {
        for half in 0..2 {
            // SAFETY: four values from `4 * half` lie inside each chunk.
            let (a, b) = unsafe {
                (
                    _mm256_cvtps_pd(_mm_loadu_ps(a.as_ptr().add(4 * half))),
                    _mm256_cvtps_pd(_mm_loadu_ps(b.as_ptr().add(4 * half))),
                )
            };
            let difference = _mm256_add_pd(_mm256_sub_pd(a, b), moved);
            let term = if EUCLIDEAN {
                _mm256_mul_pd(difference, difference)
            } else {
                _mm256_andnot_pd(sign, difference)
            };
            sums[half] = _mm256_add_pd(sums[half], term);
        }
    }
    let mut lanes = [0f64; LANES];
    for (half, vector) in sums.into_iter().enumerate() {
        // SAFETY: four lanes from `4 * half` lie inside the array.
        unsafe { _mm256_storeu_pd(lanes.as_mut_ptr().add(4 * half), vector) };
    }
    for (&a, &b) in left_rest.iter().zip(right_rest) {
        let difference = (f64::from(a) - f64::from(b)) + shift;
        lanes[0] += if EUCLIDEAN {
            difference * difference
        } else {
            difference.abs()
        };
    }
    lanes.iter().sum()
}

/// A short run's carried total in one lane, with nothing to merge: each
/// value's rounding passes through one addition to the error.
#[inline(always)]
fn one_lane_sum<I: Element>(values: &[I]) -> Carried {
    let (mut sum, mut error, mut magnitude) = (0.0, 0.0, 0.0);
    for value in values {
        let (next, rounding) = two_sum(sum, value.wide());
        sum = next;
        error += rounding;
        magnitude += rounding.abs();
    }
    Carried {
        sum,
        error,
        slack: magnitude * (values.len() + LANES) as f64 * 2f64.powi(-52),
    }
}

/// `block` of each [`BLOCK`] of `values` in turn, across the pool for a long
/// run.
fn each_block<E: Sync, R: Send>(values: &[E], block: impl Fn(&[E]) -> R + Sync) -> Vec<R> {
    dot_blocks(values, values, |part, _| block(part))
}

/// [`variance_of`] in whole numbers: `N Σx² - (Σx)²` over `N (N - ddof)`,
/// or the root of that.
fn exact_variance<T: Output>(
    values: impl Iterator<Item = f64> + Clone,
    count: usize,
    ddof: usize,
    root: bool,
) -> T {
    let (Ok(squares), Ok(sum)) = (
        Total::of_products(values.clone().map(|value| (value, value))),
        Total::of_wide(values),
    ) else {
        return T::narrow(f64::NAN);
    };
    // Both at the squares' least bit: the sum's square has twice the sum's.
    let (_, squares_magnitude, squares_least) = squares.difference();
    let (_, sum_magnitude, sum_least) = sum.difference();
    debug_assert_eq!(squares_least, 2 * sum_least);
    let scaled = product_of(&squares_magnitude, &[count as u64]);
    let square = product_of(&sum_magnitude, &sum_magnitude);
    let limbs = scaled.len().max(square.len()) + 1;
    let mut numerator = Total::empty(squares_least, limbs);
    numerator.positive[..scaled.len()].copy_from_slice(&scaled);
    numerator.negative[..square.len()].copy_from_slice(&square);
    let whole = (count as u128) * ((count - ddof) as u128);
    let mut divisor = Total::empty(0, 3);
    divisor.positive[0] = whole as u64;
    divisor.positive[1] = (whole >> 64) as u64;
    if root {
        numerator.quotient_root(divisor)
    } else {
        numerator.quotient(divisor)
    }
}

/// `Σ (x - centre)²` of one block, carried as [`distance_lanes`] carries a
/// 2-distance's squares, with fused multiply-adds where the CPU has them.
fn deviation_block<I: Element>(values: &[I], centre: f64) -> Carried {
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    fn fused_block<I: Element>(values: &[I], centre: f64) -> Carried {
        if values.len() >= 2 * LANES
            && let Some(values) = I::as_wide(values)
        {
            use std::arch::x86_64::_mm256_set1_pd;
            return squares_fma(values, |_| _mm256_set1_pd(centre), |_| centre).0;
        }
        deviation_lanes::<I, true>(values, centre)
    }
    #[cfg(target_arch = "x86_64")]
    if fused() {
        // SAFETY: avx2 and fma were detected on this CPU.
        return unsafe { fused_block(values, centre) };
    }
    deviation_lanes::<I, false>(values, centre)
}

/// [`distance_lanes`]' 2-distance, from every value to one `centre`.
#[inline(always)]
fn deviation_lanes<I: Element, const FMA: bool>(values: &[I], centre: f64) -> Carried {
    let (mut sums, mut errors, mut magnitudes) = ([0f64; LANES], [0f64; LANES], [0f64; LANES]);
    let mut add = |lane: usize, value: f64| {
        let (head, tail) = two_sum(value, -centre);
        let square = head * head;
        let square_error = if FMA {
            head.mul_add(head, -square)
        } else {
            product_error(head, head, square)
        };
        let cross = 2.0 * head * tail + tail * tail;
        let (sum, rounding) = two_sum(sums[lane], square);
        sums[lane] = sum;
        errors[lane] += rounding + (square_error + cross);
        magnitudes[lane] += rounding.abs() + square_error.abs() + 2.0 * cross.abs();
    };
    if values.len() < 2 * LANES {
        // A short run in one lane, with nothing to merge: each value passes
        // through three additions to the lane's error.
        for value in values {
            add(0, value.wide());
        }
        let additions = (3 * values.len() + LANES) as f64;
        return Carried {
            sum: sums[0],
            error: errors[0],
            slack: magnitudes[0] * additions * 2f64.powi(-52),
        };
    }
    let (chunks, rest) = values.as_chunks::<LANES>();
    for chunk in chunks {
        for lane in 0..LANES {
            add(lane, chunk[lane].wide());
        }
    }
    for value in rest {
        add(0, value.wide());
    }
    // A value passes through three additions to its lane's error.
    finish(
        sums,
        errors,
        magnitudes,
        std::iter::empty(),
        3 * values.len(),
    )
}

/// The variance, or with `root` its root, of each column of `values`, laid
/// out `[outer, len, inner]` -- the `inner` columns of each `[len, inner]`
/// slab -- as [`variance_of`] gives it, `[outer, inner]` of them.
///
/// Two passes run down the columns as [`reduce_axis_exactly`] runs down
/// them, a band of columns side by side row by row, each column a lane, the
/// rows cut into parts where there are few bands to share out: one for each
/// column's sum, and, from the centre that gives, one for the squares about
/// it, the parts merged pairwise. For float32 both are plain, settled as
/// [`plain_variance`] settles them; otherwise carried, settled as
/// [`variance_of`] settles them. A column neither settles is gathered and
/// handed to [`variance_of`] -- with `skip_nan`, only its values that are
/// not NaN, which a NaN among them always sends there.
pub(crate) fn column_variances<I: Element, T: Output>(
    values: &[I],
    outer: usize,
    len: usize,
    inner: usize,
    ddof: usize,
    root: bool,
    skip_nan: bool,
) -> Vec<T> {
    debug_assert_eq!(values.len(), outer * len * inner);
    let outputs = outer * inner;
    if len <= ddof || outputs == 0 {
        return vec![T::narrow(f64::NAN); outputs];
    }
    let slab = len * inner;
    let bands = inner.div_ceil(COLUMNS);
    let parts = if outer * bands >= SPREAD {
        1
    } else {
        len.div_ceil((BLOCK / inner.min(COLUMNS)).max(64))
    };
    let rows_per_part = len.div_ceil(parts);
    let parallel = values.len() >= crate::ops::map::PAR_THRESHOLD;
    let total = len as f64;
    let divisor = (len - ddof) as f64;
    // A value passes through its part's additions and the merges of the
    // parts, pairwise; a plain sum's magnitudes' rounding is covered twice.
    let depth = (rows_per_part + parts.ilog2() as usize + 20) as f64;
    // The columns from output `start` on that their bounds left open,
    // gathered a group at a time, row by row, into one buffer, column `k` of
    // a group at `k * len`, and handed to `variance_of`.
    let reopen = |start: usize, slots: &mut [T], open: Vec<usize>| {
        let width = BAND.min(((1 << 20) / len).max(1));
        let mut gathered: Vec<I> = Vec::new();
        for group in open.chunks(width) {
            let firsts: Vec<usize> = group
                .iter()
                .map(|&offset| (start + offset) / inner * slab + (start + offset) % inner)
                .collect();
            gathered.clear();
            gathered.resize(group.len() * len, values[firsts[0]]);
            let mut lengths = vec![0; group.len()];
            for row in 0..len {
                for (column, &first) in firsts.iter().enumerate() {
                    let value = values[first + row * inner];
                    if !(skip_nan && value.wide().is_nan()) {
                        gathered[column * len + lengths[column]] = value;
                        lengths[column] += 1;
                    }
                }
            }
            for (column, &offset) in group.iter().enumerate() {
                let kept = &gathered[column * len..][..lengths[column]];
                slots[offset] = if T::PLAIN || skip_nan {
                    variance_of(kept, ddof, root)
                } else {
                    unsettled_variance(kept, ddof, root)
                };
            }
        }
    };
    let mut out = vec![T::narrow(0.0); outputs];
    if parts == 1 {
        // Each band whole in one task, both passes and the settling.
        let band = |start: usize, slots: &mut [T]| {
            let (o, first) = (start / inner, start % inner);
            let columns = first..first + slots.len();
            let slab = &values[o * slab..][..slab];
            let sums = column_pass::<I, T>(slab, inner, 0..len, columns.clone(), None);
            let centres: Vec<f64> = sums.iter().map(|sum| sum.sum / total).collect();
            let squares = column_pass::<I, T>(slab, inner, 0..len, columns, Some(&centres));
            let open = column_settle(
                slots, &sums, &squares, &centres, total, depth, divisor, root,
            );
            reopen(start, slots, open);
        };
        let per_task = ((1 << 14) / len).max(COLUMNS);
        let task = |start: usize, slots: &mut [T]| {
            // Bands of up to `COLUMNS` within each slab the task's outputs meet.
            let end = start + slots.len();
            let mut index = start;
            while index < end {
                let width = (inner - index % inner).min(COLUMNS).min(end - index);
                band(index, &mut slots[index - start..][..width]);
                index += width;
            }
        };
        if parallel {
            crate::ops::map::par_out_chunks(&mut out, per_task, &task);
        } else {
            task(0, &mut out);
        }
        return out;
    }
    // A pass, each column's parts merged pairwise, columns in output order.
    let pass = |centres: Option<&[f64]>| -> Vec<Carried> {
        let task = |nth: usize| {
            let (o, rest) = (nth / (bands * parts), nth % (bands * parts));
            let (band, part) = (rest / parts, rest % parts);
            let columns = band * COLUMNS..((band + 1) * COLUMNS).min(inner);
            let rows = (part * rows_per_part).min(len)..((part + 1) * rows_per_part).min(len);
            let centres = centres.map(|centres| &centres[o * inner..][columns.clone()]);
            column_pass::<I, T>(&values[o * slab..][..slab], inner, rows, columns, centres)
        };
        let tasks = outer * bands * parts;
        let partials: Vec<Vec<Carried>> = if parallel {
            crate::parallel::map_tasks(tasks, &task)
        } else {
            (0..tasks).map(task).collect()
        };
        let mut partials = partials.into_iter();
        let mut merged = Vec::with_capacity(outputs);
        for _ in 0..outer * bands {
            let group: Vec<Vec<Carried>> = partials.by_ref().take(parts).collect();
            merged.extend(if T::PLAIN {
                pairwise_fold_vectors(group, |a, b| Carried {
                    sum: a.sum + b.sum,
                    error: 0.0,
                    slack: a.slack + b.slack,
                })
            } else {
                pairwise_fold_vectors(group, Carried::merge)
            });
        }
        merged
    };
    let sums = pass(None);
    let centres: Vec<f64> = sums.iter().map(|sum| sum.sum / total).collect();
    let squares = pass(Some(&centres));
    let settle = |start: usize, slots: &mut [T]| {
        let index = start..start + slots.len();
        let (sums, squares, centres) = (
            &sums[index.clone()],
            &squares[index.clone()],
            &centres[index],
        );
        let open = column_settle(slots, sums, squares, centres, total, depth, divisor, root);
        reopen(start, slots, open);
    };
    if parallel {
        crate::ops::map::par_out_chunks(&mut out, crate::ops::map::PAR_CHUNK, &settle);
    } else {
        settle(0, &mut out);
    }
    out
}

/// One part of a pass of [`column_variances`], in the widest build the CPU
/// runs; see [`column_part`].
fn column_pass<I: Element, T: Output>(
    slab: &[I],
    inner: usize,
    rows: std::ops::Range<usize>,
    columns: std::ops::Range<usize>,
    centres: Option<&[f64]>,
) -> Vec<Carried> {
    #[cfg(target_arch = "x86_64")]
    {
        #[target_feature(enable = "avx2,fma")]
        fn with_fma<I: Element, const PLAIN: bool>(
            slab: &[I],
            inner: usize,
            rows: std::ops::Range<usize>,
            columns: std::ops::Range<usize>,
            centres: Option<&[f64]>,
        ) -> Vec<Carried> {
            column_part::<I, PLAIN, true>(slab, inner, rows, columns, centres)
        }
        #[target_feature(enable = "avx")]
        fn with_avx<I: Element, const PLAIN: bool>(
            slab: &[I],
            inner: usize,
            rows: std::ops::Range<usize>,
            columns: std::ops::Range<usize>,
            centres: Option<&[f64]>,
        ) -> Vec<Carried> {
            column_part::<I, PLAIN, false>(slab, inner, rows, columns, centres)
        }
        if fused() {
            // SAFETY: avx2 and fma were detected on this CPU.
            return unsafe {
                if T::PLAIN {
                    with_fma::<I, true>(slab, inner, rows, columns, centres)
                } else {
                    with_fma::<I, false>(slab, inner, rows, columns, centres)
                }
            };
        }
        if crate::ops::simd::simd_capabilities().avx {
            // SAFETY: avx was detected on this CPU.
            return unsafe {
                if T::PLAIN {
                    with_avx::<I, true>(slab, inner, rows, columns, centres)
                } else {
                    with_avx::<I, false>(slab, inner, rows, columns, centres)
                }
            };
        }
    }
    if T::PLAIN {
        column_part::<I, true, false>(slab, inner, rows, columns, centres)
    } else {
        column_part::<I, false, false>(slab, inner, rows, columns, centres)
    }
}

/// The `columns` of a `[len, inner]` slab over `rows`, each column a lane
/// that runs down its values: with no centres, each column's plain sum and
/// the sum of its magnitudes, in `sum` and `slack` (`PLAIN`), or its carried
/// total; with a centre to each column, the plain sum of the squares about
/// it (`PLAIN`), or their carried total, each deviation taken exactly as
/// [`deviation_lanes`] takes it, the square's error by a fused multiply-add
/// where `FMA`. The columns are independent, so each row's loop vectorizes.
#[inline(always)]
fn column_part<I: Element, const PLAIN: bool, const FMA: bool>(
    slab: &[I],
    inner: usize,
    rows: std::ops::Range<usize>,
    columns: std::ops::Range<usize>,
    centres: Option<&[f64]>,
) -> Vec<Carried> {
    let width = columns.len();
    let (mut sums, mut errors, mut magnitudes) =
        (vec![0f64; width], vec![0f64; width], vec![0f64; width]);
    let row = |index: usize| &slab[index * inner + columns.start..][..width];
    let mut index = rows.start;
    // A block of rows at a time, in order, then the rows left over.
    while index + ROW_BLOCK <= rows.end {
        let block: [&[I]; ROW_BLOCK] = std::array::from_fn(|nth| row(index + nth));
        let (sums, errors, magnitudes) = (&mut sums, &mut errors, &mut magnitudes);
        match centres {
            Some(centres) => accumulate_squares::<I, PLAIN, FMA, ROW_BLOCK>(
                sums, errors, magnitudes, centres, block,
            ),
            None => accumulate_block::<I, PLAIN>(sums, errors, magnitudes, block),
        }
        index += ROW_BLOCK;
    }
    while index < rows.end {
        let (sums, errors, magnitudes) = (&mut sums, &mut errors, &mut magnitudes);
        match centres {
            Some(centres) => accumulate_squares::<I, PLAIN, FMA, 1>(
                sums,
                errors,
                magnitudes,
                centres,
                [row(index)],
            ),
            None => accumulate_columns::<I, PLAIN>(sums, errors, magnitudes, row(index)),
        }
        index += 1;
    }
    // A value's roundings pass through one addition to its lane's error, a
    // deviation's through three.
    let steps = if centres.is_some() { 3 } else { 1 };
    let additions = (steps * rows.len() + LANES) as f64;
    (0..width)
        .map(|lane| Carried {
            sum: sums[lane],
            error: errors[lane],
            slack: if PLAIN {
                magnitudes[lane]
            } else {
                magnitudes[lane] * additions * 2f64.powi(-52)
            },
        })
        .collect()
}

/// The columns of [`column_variances`] settled side by side into `slots`,
/// with fused multiply-adds where the CPU has them, and the offsets of those
/// their bounds did not settle.
#[allow(clippy::too_many_arguments)]
fn column_settle<T: Output>(
    slots: &mut [T],
    sums: &[Carried],
    squares: &[Carried],
    centres: &[f64],
    total: f64,
    depth: f64,
    divisor: f64,
    root: bool,
) -> Vec<usize> {
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    #[allow(clippy::too_many_arguments)]
    fn with_fma<T: Output>(
        slots: &mut [T],
        sums: &[Carried],
        squares: &[Carried],
        centres: &[f64],
        total: f64,
        depth: f64,
        divisor: f64,
        root: bool,
    ) -> Vec<usize> {
        column_settle_lanes::<T, true>(slots, sums, squares, centres, total, depth, divisor, root)
    }
    #[cfg(target_arch = "x86_64")]
    if fused() {
        // SAFETY: avx2 and fma were detected on this CPU.
        return unsafe { with_fma(slots, sums, squares, centres, total, depth, divisor, root) };
    }
    column_settle_lanes::<T, false>(slots, sums, squares, centres, total, depth, divisor, root)
}

/// [`column_settle`]'s columns: plainly, by [`plain_lane`], for float32,
/// from each sum's magnitudes and `depth`; carried, by [`centred_lane`],
/// otherwise.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn column_settle_lanes<T: Output, const FMA: bool>(
    slots: &mut [T],
    sums: &[Carried],
    squares: &[Carried],
    centres: &[f64],
    total: f64,
    depth: f64,
    divisor: f64,
    root: bool,
) -> Vec<usize> {
    let mut open = Vec::new();
    let lanes = slots.iter_mut().zip(sums).zip(squares).zip(centres);
    for (offset, (((slot, &sum), &squares), &centre)) in lanes.enumerate() {
        let (value, settled) = if T::PLAIN {
            let bound = sum.slack * depth * 2f64.powi(-52);
            plain_lane(squares.sum, bound, centre, total, depth, divisor, root)
        } else {
            centred_lane::<T, FMA>(sum, squares, centre, total, divisor, root)
        };
        *slot = value;
        if !settled {
            open.push(offset);
        }
    }
    open
}

/// The value nearest the exact distance between `x` and `y`, each difference
/// shifted by `shift`: `Σ |x - y + shift|`, or with `euclidean` the root of
/// `Σ (x - y + shift)²`.
///
/// A float32 distance is first taken plainly in float64, whose bound
/// settles it unless it reaches a halfway point between two float32s. Then,
/// and for float64, each difference is taken exactly -- a two-sum's pair,
/// three parts once shifted -- in lanes a block at a time, the blocks
/// across the pool for a long row. For the 1-distance each part takes the
/// difference's sign and is carried as [`sum_to`] carries a value; for the
/// 2-distance each leading part's square is split exactly and carried, with
/// the cross terms beside it, and the root taken as [`root_of`] takes it.
/// Where a bound cannot settle the answer, the exact differences are summed,
/// or their squares rooted, in whole numbers. A NaN, or an infinity, gives
/// what the formula gives.
pub(crate) fn distance<I: Element, T: Output>(x: &[I], y: &[I], euclidean: bool, shift: f64) -> T {
    debug_assert_eq!(x.len(), y.len());
    if let ([a], [b]) = (x, y)
        && shift == 0.0
    {
        // `|a - b|` either way, a two-sum's pair, rounded once: to odd at
        // float64's last bit for a narrower float, which keeps whether
        // anything lay below it.
        let (head, tail) = two_sum(a.wide(), -b.wide());
        if tail == 0.0 || !head.is_finite() {
            return T::narrow(head.abs());
        }
        let odd = if head.to_bits() & 1 == 1 {
            head
        } else if (tail > 0.0) == (head > 0.0) {
            f64::from_bits(head.to_bits() + 1)
        } else {
            f64::from_bits(head.to_bits() - 1)
        };
        return T::narrow(odd.abs());
    }
    if T::PLAIN
        && let Some(value) = plain_distance(x, y, euclidean, shift)
    {
        return value;
    }
    let part = if x.len() <= BLOCK {
        // One block, as most rows are, without gathering it into a list.
        distance_block(x, y, euclidean, shift)
    } else {
        let block = |xs: &[I], ys: &[I]| distance_block(xs, ys, euclidean, shift);
        let parts = dot_blocks(x, y, block);
        DistancePart {
            finite: parts.iter().all(|part| part.finite),
            unsure: parts.iter().any(|part| part.unsure),
            loose: parts.iter().map(|part| part.loose).sum(),
            carried: pairwise_fold(
                parts.into_iter().map(|part| part.carried).collect(),
                Carried::default(),
                if euclidean {
                    Carried::merge
                } else {
                    Carried::merge_tracked
                },
            ),
        }
    };
    let DistancePart {
        carried,
        loose,
        finite,
        unsure,
    } = part;
    if !finite {
        return distance_formula(x, y, euclidean, shift);
    }
    if unsure {
        return exact_distance(x, y, euclidean, shift);
    }
    if !euclidean {
        return carried
            .rounded()
            .unwrap_or_else(|| exact_distance(x, y, euclidean, shift));
    }
    // The third parts' share of the squares, and an allowance for a square's
    // error or a cross term rounded to the subnormal range, by under the
    // least subnormal each; the root needs a total clear of that range.
    let allowing = Carried {
        slack: carried.slack
            + loose * (1.0 + 2f64.powi(-50))
            + 4.0 * x.len() as f64 * f64::from_bits(1),
        ..carried
    };
    root_settled(allowing).unwrap_or_else(|| exact_distance(x, y, euclidean, shift))
}

/// A float32 distance taken plainly in float64, when its bound settles it.
///
/// Each difference is rounded at most twice and each square once more, and
/// the sum of `n` of them, in lanes, `n` times: so `(n + 6) u` of the sum
/// bounds it, with the least subnormal for each square that may have
/// underflowed. A zero is the answer only unshifted, where a rounded
/// difference is zero only if the exact one is.
fn plain_distance<I: Element, T: Output>(
    x: &[I],
    y: &[I],
    euclidean: bool,
    shift: f64,
) -> Option<T> {
    let block = |xs: &[I], ys: &[I]| I::plain_differences(xs, ys, euclidean, shift);
    let (sum, blocks) = if x.len() <= BLOCK {
        (block(x, y), 1)
    } else {
        let partials = dot_blocks(x, y, block);
        (partials.iter().sum::<f64>(), partials.len())
    };
    if !sum.is_finite() {
        return None;
    }
    if sum == 0.0 {
        return (shift == 0.0).then(|| T::narrow(0.0));
    }
    let depth = (x.len() + blocks + LANES) as f64;
    if !euclidean {
        return T::settle(sum, 0.0, sum * (depth + 4.0) * 2f64.powi(-52));
    }
    let bound = sum * (depth + 6.0) * 2f64.powi(-52) + depth * f64::from_bits(2);
    let clear = sum > 4.0 * bound;
    if !clear {
        return None;
    }
    // The root moves by at most the bound over twice the least root within
    // it, `sqrt(sum - bound) ≥ r (1 - bound / sum)`, and is rounded once
    // itself.
    let root = sum.sqrt();
    let moved = bound * (0.5 / root) * (1.0 + 2.0 * bound / sum) * (1.0 + 2f64.powi(-50))
        + root * 2f64.powi(-52);
    T::settle(root, 0.0, moved * (1.0 + 2f64.powi(-40)))
}

/// The value nearest the exact `Σ (x - y)²`, or with `taxicab`
/// `Σ |x - y|`, over `count`: a loss's sum with a count of one, and its
/// mean with the number of terms. NaN for no terms over none.
///
/// The differences are carried as [`distance`] carries them, and the total
/// over `count` settled by [`Carried::divided`]'s bound; a float32 answer is
/// first tried from plain float64 sums. A NaN or an infinity gives what the
/// formula gives; where the bound cannot say, or finite operands differ by
/// more than float64 holds, the total is formed in whole numbers and divided
/// there.
pub(crate) fn difference_total<I: Element, T: Output>(
    x: &[I],
    y: &[I],
    taxicab: bool,
    count: usize,
) -> T {
    debug_assert_eq!(x.len(), y.len());
    if count == 0 {
        return T::narrow(f64::NAN);
    }
    let euclidean = !taxicab;
    if T::PLAIN
        && let Some(value) = plain_total(x, y, euclidean, count)
    {
        return value;
    }
    let part = if x.len() <= BLOCK {
        distance_block(x, y, euclidean, 0.0)
    } else {
        let block = |xs: &[I], ys: &[I]| distance_block(xs, ys, euclidean, 0.0);
        let parts = dot_blocks(x, y, block);
        DistancePart {
            finite: parts.iter().all(|part| part.finite),
            unsure: parts.iter().any(|part| part.unsure),
            loose: parts.iter().map(|part| part.loose).sum(),
            carried: pairwise_fold(
                parts.into_iter().map(|part| part.carried).collect(),
                Carried::default(),
                if euclidean {
                    Carried::merge
                } else {
                    Carried::merge_tracked
                },
            ),
        }
    };
    if !part.finite {
        let finite = |values: &[I]| values.iter().all(|value| value.wide().is_finite());
        if finite(x) && finite(y) {
            // Finite operands a whole float64 range apart, whose mean may
            // still lie inside it.
            return exact_difference_total(x, y, euclidean, count);
        }
        let differences = x.iter().zip(y).map(|(a, b)| a.wide() - b.wide());
        let total: f64 = if euclidean {
            differences.map(|d| d * d).sum()
        } else {
            differences.map(f64::abs).sum()
        };
        return T::narrow(total / count as f64);
    }
    let settled = (!part.unsure).then(|| {
        // As for the 2-distance: the squares' third parts, and an allowance
        // for errors rounded to the subnormal range.
        let carried = if euclidean {
            Carried {
                slack: part.carried.slack
                    + part.loose * (1.0 + 2f64.powi(-50))
                    + 4.0 * x.len() as f64 * f64::from_bits(1),
                ..part.carried
            }
        } else {
            part.carried
        };
        carried.divided::<T>(count)
    });
    settled
        .flatten()
        .unwrap_or_else(|| exact_difference_total(x, y, euclidean, count))
}

/// [`difference_total`] of a float32 answer from plain float64 sums, when
/// their bound settles it: within `(n + 6)u` of the total, as for
/// [`plain_distance`], and a rounding more for the division.
fn plain_total<I: Element, T: Output>(
    x: &[I],
    y: &[I],
    euclidean: bool,
    count: usize,
) -> Option<T> {
    let block = |xs: &[I], ys: &[I]| I::plain_differences(xs, ys, euclidean, 0.0);
    let (sum, blocks) = if x.len() <= BLOCK {
        (block(x, y), 1)
    } else {
        let partials = dot_blocks(x, y, block);
        (partials.iter().sum::<f64>(), partials.len())
    };
    if !sum.is_finite() {
        return None;
    }
    if sum == 0.0 {
        // A rounded difference is zero only where the exact one is.
        return Some(T::narrow(0.0));
    }
    let depth = (x.len() + blocks + LANES) as f64;
    let value = sum / count as f64;
    let bound = value * (depth + 8.0) * 2f64.powi(-52) + depth * f64::from_bits(2);
    T::settle(value, 0.0, bound)
}

/// [`difference_total`] in whole numbers: the squares expanded into
/// products of the operands, or the differences' magnitudes, each sign taken
/// exactly, summed, and divided by `count` by long division.
#[cold]
#[inline(never)]
fn exact_difference_total<I: Element, T: Output>(
    x: &[I],
    y: &[I],
    euclidean: bool,
    count: usize,
) -> T {
    let pairs = x.iter().zip(y).map(|(a, b)| (a.wide(), b.wide()));
    let total = if euclidean {
        Total::of_products(pairs.flat_map(|(a, b)| [(a, a), (b, b), (a, -b), (a, -b)]))
    } else {
        Total::of_wide(pairs.flat_map(|(a, b)| {
            let (head, tail) = two_sum(a, -b);
            // The exact difference's sign: its head's, or its tail's at zero.
            let negative = head < 0.0 || (head == 0.0 && tail < 0.0);
            if negative { [-a, b] } else { [a, -b] }
        }))
    };
    match total {
        Err(special) => T::narrow(special / count as f64),
        Ok(total) => total.divided(count as u64),
    }
}

/// The value nearest `(a - b)²`: the difference as a two-sum's pair
/// `h + t`, its square `h² + 2ht + t²` with `h²` split exactly, settled by a
/// bound on the cross term's roundings and `t²`, and otherwise summed in
/// whole numbers. A NaN, or a difference past the range, squares as the
/// formula squares it. A float32 answer wants operands that are float32s,
/// whose squares' errors float64 holds.
pub(crate) fn squared_difference<T: Output>(a: f64, b: f64) -> T {
    let (head, tail) = two_sum(a, -b);
    if !head.is_finite() {
        return T::narrow(head * head);
    }
    let exact = || {
        Total::of_products([(head, head), (head, tail), (head, tail), (tail, tail)].into_iter())
            .map_or_else(T::narrow, |total| total.rounded())
    };
    let square = head * head;
    if !square.is_finite() {
        // Rounded past the range, where the exact square may lie inside it.
        return exact();
    }
    let error = product_error(head, head, square);
    if tail == 0.0 {
        // The difference exact, its square is the pair, rounded once.
        return T::settle(square, error, 0.0).unwrap_or_else(|| T::narrow(square));
    }
    let cross = 2.0 * head * tail;
    let low = error + cross;
    let (value, rest) = two_sum(square, low);
    // The cross term's two roundings and `t²`, and for each part rounded in
    // the subnormal range under the least subnormal.
    let bound = ((cross.abs() + low.abs()) * 2f64.powi(-53) + tail * tail) * (1.0 + 2f64.powi(-50))
        + f64::from_bits(8);
    T::settle(value, rest, bound).unwrap_or_else(exact)
}

/// [`difference_total`] of float64s `x` less integers, each the exact sum
/// of its `highs` value and, where any integer lies past 2^53, its `lows`
/// value.
pub(crate) fn integer_difference_total<T: Output>(
    x: &[f64],
    highs: &[f64],
    lows: Option<&[f64]>,
    taxicab: bool,
    count: usize,
) -> T {
    let Some(lows) = lows else {
        return difference_total(x, highs, taxicab, count);
    };
    if count == 0 {
        return T::narrow(f64::NAN);
    }
    let terms = x
        .iter()
        .zip(highs)
        .zip(lows)
        .map(|((&a, &high), &low)| (a, high, low));
    if !x.iter().all(|value| value.is_finite()) {
        // A NaN, or an infinity, as the formula gives it.
        let total: f64 = terms
            .map(|(a, high, low)| {
                let difference = (a - high) - low;
                if taxicab {
                    difference.abs()
                } else {
                    difference * difference
                }
            })
            .sum();
        return T::narrow(total / count as f64);
    }
    let total = if taxicab {
        Total::of_wide(terms.flat_map(|(a, high, low)| {
            if is_negative(&[a, -high, -low]) {
                [-a, high, low]
            } else {
                [a, -high, -low]
            }
        }))
    } else {
        // `(a - h - l)²` multiplied out.
        Total::of_products(terms.flat_map(|(a, high, low)| {
            [
                (a, a),
                (high, high),
                (low, low),
                (a, -high),
                (a, -high),
                (a, -low),
                (a, -low),
                (high, low),
                (high, low),
            ]
        }))
    };
    total.map_or_else(T::narrow, |total| total.divided(count as u64))
}

/// [`squared_difference`] of each float64 in `x` less an integer, given as
/// for [`integer_difference_total`].
pub(crate) fn integer_squared_differences<T: Output>(
    x: &[f64],
    highs: &[f64],
    lows: Option<&[f64]>,
) -> Vec<T> {
    let Some(lows) = lows else {
        return crate::ops::map::binary_map(x, highs, squared_difference::<T>);
    };
    x.iter()
        .zip(highs)
        .zip(lows)
        .map(|((&a, &high), &low)| {
            if low == 0.0 || !a.is_finite() {
                // The integer exact in float64, or a NaN or an infinity,
                // which the remainder does not move.
                return squared_difference(a, high);
            }
            Total::of_products(
                [
                    (a, a),
                    (high, high),
                    (low, low),
                    (a, -high),
                    (a, -high),
                    (a, -low),
                    (a, -low),
                    (high, low),
                    (high, low),
                ]
                .into_iter(),
            )
            .map_or_else(T::narrow, |total| total.rounded())
        })
        .collect()
}

/// Each `(x - y)²`, the value nearest the square of the exact difference:
/// [`squared_difference_lane`] a run at a time, in the widest build the CPU
/// runs and across the pool for a long operand, and a run with any left
/// open -- near a halfway point, in the subnormal range, past the range, or
/// NaN -- again by [`reopened_run`].
pub(crate) fn squared_differences<T: Squared>(x: &[T], y: &[T]) -> Vec<T> {
    use std::mem::MaybeUninit;
    assert_eq!(
        x.len(),
        y.len(),
        "squared_differences: operands differ in length"
    );
    let fill = |start: usize, block: &mut [MaybeUninit<T>]| {
        let end = start + block.len();
        const RUN: usize = 256;
        let runs = x[start..end]
            .chunks(RUN)
            .zip(y[start..end].chunks(RUN))
            .zip(block.chunks_mut(RUN));
        for ((x, y), out) in runs {
            if T::squares_run(x, y, out) {
                reopened_run(x, y, out);
            }
        }
    };
    // SAFETY: the chunks cover the output, and `fill` writes every element
    // of each.
    unsafe {
        crate::ops::map::build_vec(x.len(), |spare| {
            crate::ops::map::par_out_chunks(spare, 1 << 14, &fill)
        })
    }
}

/// A float whose squared differences [`squared_differences`] takes, the
/// operands and the answers all of it.
pub(crate) trait Squared: Element + Output {
    /// [`squares_run`] in the widest build the CPU runs: each square into
    /// `out`, and whether any was left open.
    fn squares_run(x: &[Self], y: &[Self], out: &mut [std::mem::MaybeUninit<Self>]) -> bool;
}

impl Squared for f64 {
    fn squares_run(x: &[Self], y: &[Self], out: &mut [std::mem::MaybeUninit<Self>]) -> bool {
        #[cfg(target_arch = "x86_64")]
        if fused() {
            // SAFETY: avx2 and fma were detected on this CPU.
            return unsafe { squares_run_fma(x, y, out) };
        }
        squares_run::<Self, false>(x, y, out)
    }
}

impl Squared for f32 {
    fn squares_run(x: &[Self], y: &[Self], out: &mut [std::mem::MaybeUninit<Self>]) -> bool {
        #[cfg(target_arch = "x86_64")]
        if crate::ops::simd::simd_capabilities().avx2 {
            // SAFETY: avx2 was detected on this CPU.
            return unsafe { squares_run_avx2_f32(x, y, out) };
        }
        squares_run::<Self, false>(x, y, out)
    }
}

/// [`squared_difference_lane`] over a run into `out`, and whether any was
/// left open, gathered without a branch.
#[inline(always)]
fn squares_run<T: Squared, const FMA: bool>(
    x: &[T],
    y: &[T],
    out: &mut [std::mem::MaybeUninit<T>],
) -> bool {
    let mut open = false;
    for ((slot, a), b) in out.iter_mut().zip(x).zip(y) {
        let (value, settled) = squared_difference_lane::<T, FMA>(a.wide(), b.wide());
        open |= !settled;
        slot.write(value);
    }
    open
}

/// A run with a square left open, every square again: from
/// [`squared_difference_lane`] where it settles and from
/// [`squared_difference`] where it does not.
#[cold]
#[inline(never)]
fn reopened_run<T: Squared>(x: &[T], y: &[T], out: &mut [std::mem::MaybeUninit<T>]) {
    for ((slot, a), b) in out.iter_mut().zip(x).zip(y) {
        let (a, b) = (a.wide(), b.wide());
        let (value, settled) = squared_difference_lane::<T, false>(a, b);
        slot.write(if settled {
            value
        } else {
            squared_difference(a, b)
        });
    }
}

/// [`squared_difference_lane`] for float64 in 256-bit registers with fused
/// multiply-adds, written out because the compiler kept the two-sums
/// scalar: the cross term `2ht` added to the square's error by one fused
/// rounding, and the square's pair settled as [`Output::settle_lanes`]
/// settles it, within that rounding, `t²` and an allowance for the
/// subnormal range: the least normal float64, where a subnormal operand can
/// cost a microcode assist a vector.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
fn squares_run_fma(x: &[f64], y: &[f64], out: &mut [std::mem::MaybeUninit<f64>]) -> bool {
    use std::arch::x86_64::{
        _CMP_EQ_OQ, _CMP_LT_OQ, _mm256_add_pd, _mm256_andnot_pd, _mm256_castpd_si256,
        _mm256_castsi256_pd, _mm256_cmp_pd, _mm256_fmadd_pd, _mm256_fmsub_pd, _mm256_loadu_pd,
        _mm256_movemask_pd, _mm256_mul_pd, _mm256_or_pd, _mm256_set1_epi64x, _mm256_set1_pd,
        _mm256_setzero_pd, _mm256_storeu_pd, _mm256_sub_epi64, _mm256_sub_pd,
    };
    let sign = _mm256_set1_pd(-0.0);
    let (zero, one) = (_mm256_setzero_pd(), _mm256_set1_epi64x(1));
    let (unit, widen) = (
        _mm256_set1_pd(2f64.powi(-52)),
        _mm256_set1_pd(1.0 + 2f64.powi(-50)),
    );
    let (least, shy) = (
        _mm256_set1_pd(f64::MIN_POSITIVE),
        _mm256_set1_pd(SHY_OF_HALF),
    );
    // Any lane left open, as a sign bit.
    let mut open = zero;
    let (left, left_rest) = x.as_chunks::<4>();
    let (right, right_rest) = y.as_chunks::<4>();
    let (slots, slots_rest) = out.as_chunks_mut::<4>();
    for ((a, b), slot) in left.iter().zip(right).zip(slots.iter_mut()) {
        // SAFETY: each chunk holds four values.
        let (a, b) = unsafe { (_mm256_loadu_pd(a.as_ptr()), _mm256_loadu_pd(b.as_ptr())) };
        // `a - b` and its exact error, by a two-sum.
        let head = _mm256_sub_pd(a, b);
        let virtual_b = _mm256_sub_pd(a, head);
        let virtual_a = _mm256_add_pd(head, virtual_b);
        let tail = _mm256_sub_pd(_mm256_sub_pd(a, virtual_a), _mm256_sub_pd(b, virtual_b));
        let square = _mm256_mul_pd(head, head);
        let error = _mm256_fmsub_pd(head, head, square);
        let low = _mm256_fmadd_pd(_mm256_add_pd(head, head), tail, error);
        // The pair by a fast two-sum: the square outweighs the rest.
        let value = _mm256_add_pd(square, low);
        let rest = _mm256_sub_pd(low, _mm256_sub_pd(value, square));
        let bound = _mm256_mul_pd(
            _mm256_add_pd(
                _mm256_fmadd_pd(_mm256_andnot_pd(sign, low), unit, _mm256_mul_pd(tail, tail)),
                least,
            ),
            widen,
        );
        // The gap to the float64 below, NaN for a zero or past the range.
        let below = _mm256_castsi256_pd(_mm256_sub_epi64(_mm256_castpd_si256(value), one));
        let inward = _mm256_sub_pd(value, below);
        let clear = _mm256_cmp_pd(
            _mm256_add_pd(_mm256_andnot_pd(sign, rest), bound),
            _mm256_mul_pd(inward, shy),
            _CMP_LT_OQ,
        );
        let settled = _mm256_or_pd(clear, _mm256_cmp_pd(head, zero, _CMP_EQ_OQ));
        open = _mm256_or_pd(open, _mm256_andnot_pd(settled, sign));
        // SAFETY: each chunk of the output holds four values.
        unsafe { _mm256_storeu_pd(slot.as_mut_ptr().cast(), value) };
    }
    let mut any = _mm256_movemask_pd(open) != 0;
    for ((slot, &a), &b) in slots_rest.iter_mut().zip(left_rest).zip(right_rest) {
        let (value, settled) = squared_difference_lane::<f64, true>(a, b);
        any |= !settled;
        slot.write(value);
    }
    any
}

/// [`squared_difference_lane`] for float32 in 256-bit registers, four
/// operands widened at once: each plain square, and whether it is clear of
/// a float32 halfway point and of float32's subnormals.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
fn squares_run_avx2_f32(x: &[f32], y: &[f32], out: &mut [std::mem::MaybeUninit<f32>]) -> bool {
    use std::arch::x86_64::{
        _CMP_EQ_OQ, _CMP_GE_OQ, _mm_loadu_ps, _mm_storeu_ps, _mm256_and_si256, _mm256_andnot_si256,
        _mm256_castpd_si256, _mm256_cmp_pd, _mm256_cmpeq_epi64, _mm256_cmpgt_epi64,
        _mm256_cvtpd_ps, _mm256_cvtps_pd, _mm256_mul_pd, _mm256_or_si256, _mm256_set1_epi64x,
        _mm256_set1_pd, _mm256_setzero_pd, _mm256_setzero_si256, _mm256_sub_pd, _mm256_testz_si256,
    };
    let past = _mm256_set1_epi64x((1 << 29) - 1);
    let (above, under) = (
        _mm256_set1_epi64x((1 << 28) + 8),
        _mm256_set1_epi64x((1 << 28) - 8),
    );
    let (zero, least) = (
        _mm256_setzero_pd(),
        _mm256_set1_pd(f64::from(f32::MIN_POSITIVE)),
    );
    let ones = _mm256_cmpeq_epi64(_mm256_setzero_si256(), _mm256_setzero_si256());
    // Any lane left open, all ones.
    let mut open = _mm256_setzero_si256();
    let (left, left_rest) = x.as_chunks::<4>();
    let (right, right_rest) = y.as_chunks::<4>();
    let (slots, slots_rest) = out.as_chunks_mut::<4>();
    for ((a, b), slot) in left.iter().zip(right).zip(slots.iter_mut()) {
        // SAFETY: each chunk holds four values.
        let (a, b) = unsafe {
            (
                _mm256_cvtps_pd(_mm_loadu_ps(a.as_ptr())),
                _mm256_cvtps_pd(_mm_loadu_ps(b.as_ptr())),
            )
        };
        let difference = _mm256_sub_pd(a, b);
        let square = _mm256_mul_pd(difference, difference);
        let bits = _mm256_and_si256(_mm256_castpd_si256(square), past);
        let clear = _mm256_or_si256(
            _mm256_cmpgt_epi64(bits, above),
            _mm256_cmpgt_epi64(under, bits),
        );
        let normal = _mm256_castpd_si256(_mm256_cmp_pd(square, least, _CMP_GE_OQ));
        let exact = _mm256_castpd_si256(_mm256_cmp_pd(difference, zero, _CMP_EQ_OQ));
        let settled = _mm256_or_si256(_mm256_and_si256(clear, normal), exact);
        open = _mm256_or_si256(open, _mm256_andnot_si256(settled, ones));
        // SAFETY: each chunk of the output holds four values.
        unsafe { _mm_storeu_ps(slot.as_mut_ptr().cast(), _mm256_cvtpd_ps(square)) };
    }
    let mut any = _mm256_testz_si256(open, open) == 0;
    for ((slot, &a), &b) in slots_rest.iter_mut().zip(left_rest).zip(right_rest) {
        let (value, settled) = squared_difference_lane::<f32, false>(f64::from(a), f64::from(b));
        any |= !settled;
        slot.write(value);
    }
    any
}

/// [`squared_difference`] without branches: the square of the exact
/// difference's pair with its cross terms, and whether the bound settles
/// it; for float32, the plain square and whether it is clear of a halfway
/// point. A zero difference is exact; a NaN, and a square in the answer's
/// subnormal range, are left open, and for float64 an infinite square.
#[inline(always)]
fn squared_difference_lane<T: Output, const FMA: bool>(a: f64, b: f64) -> (T, bool) {
    if T::PLAIN {
        // The difference and its square each rounded once in float64, which
        // leaves the square within three float64 units of the exact one: more
        // than eight from a float32 halfway point, the 2^28th unit past a
        // float32, and above float32's subnormals, both narrow alike.
        let difference = a - b;
        let square = difference * difference;
        let past = square.to_bits() & ((1 << 29) - 1);
        let clear = past.abs_diff(1 << 28) > 8;
        let normal = square >= f64::from(f32::MIN_POSITIVE);
        return (T::narrow(square), (clear & normal) | (difference == 0.0));
    }
    let (head, tail) = two_sum(a, -b);
    let square = head * head;
    let error = if FMA {
        head.mul_add(head, -square)
    } else {
        product_error(head, head, square)
    };
    let cross = 2.0 * head * tail;
    let low = error + cross;
    let (value, rest) = two_sum(square, low);
    let bound = ((cross.abs() + low.abs()) * 2f64.powi(-53) + tail * tail) * (1.0 + 2f64.powi(-50))
        + f64::from_bits(8);
    let (rounded, settled) = T::settle_lanes(value, rest, bound);
    (rounded, settled | (head == 0.0))
}

/// One block of a distance's exact differences, carried, with what the
/// rest of [`distance`] needs to know of it.
struct DistancePart {
    carried: Carried,
    /// For the 2-distance, a bound on the third parts' share of the squares.
    loose: f64,
    /// Whether every difference was finite.
    finite: bool,
    /// For the 1-distance, whether a difference's sign was left undecided.
    unsure: bool,
}

/// [`distance_lanes`] for one block, with fused multiply-adds where the CPU
/// has them, and without the shift's third parts where there is none.
fn distance_block<I: Element>(x: &[I], y: &[I], euclidean: bool, shift: f64) -> DistancePart {
    macro_rules! with {
        ($fma:literal, $x:ident, $y:ident, $euclidean:ident, $shift:ident) => {
            match ($euclidean, $shift == 0.0) {
                (true, true) => distance_lanes::<I, true, false, $fma>($x, $y, $shift),
                (true, false) => distance_lanes::<I, true, true, $fma>($x, $y, $shift),
                (false, true) => distance_lanes::<I, false, false, $fma>($x, $y, $shift),
                (false, false) => distance_lanes::<I, false, true, $fma>($x, $y, $shift),
            }
        };
    }
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    fn fused_block<I: Element>(x: &[I], y: &[I], euclidean: bool, shift: f64) -> DistancePart {
        with!(true, x, y, euclidean, shift)
    }
    #[cfg(target_arch = "x86_64")]
    if fused() {
        if shift == 0.0
            && x.len() >= 2 * LANES
            && let Some((x, y)) = I::as_wide(x).zip(I::as_wide(y))
        {
            // SAFETY: avx2 and fma were detected on this CPU.
            return unsafe {
                if euclidean {
                    euclidean_block_fma(x, y)
                } else {
                    taxicab_block_avx(x, y)
                }
            };
        }
        // SAFETY: avx2 and fma were detected on this CPU.
        return unsafe { fused_block(x, y, euclidean, shift) };
    }
    with!(false, x, y, euclidean, shift)
}

/// [`distance_lanes`] for the unshifted 2-distance of float64 rows, in
/// 256-bit registers with fused multiply-adds; see [`squares_fma`].
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
fn euclidean_block_fma(x: &[f64], y: &[f64]) -> DistancePart {
    use std::arch::x86_64::_mm256_loadu_pd;
    debug_assert_eq!(x.len(), y.len());
    let (carried, finite) = squares_fma(
        x,
        // SAFETY: `y` is as long as `x`, and only four values that `x` has
        // too, from `index` on, are read.
        |index| unsafe { _mm256_loadu_pd(y.as_ptr().add(index)) },
        |index| y[index],
    );
    DistancePart {
        carried,
        loose: 0.0,
        finite,
        unsure: false,
    }
}

/// The carried `Σ (x - y)²` of one block, and whether every difference was
/// finite, with `y` given four at a time from an index by `load` and one at
/// a time by `at`: [`distance_lanes`]' lanes and terms for the unshifted
/// 2-distance in 256-bit registers with fused multiply-adds, written out
/// because the compiler kept the two-sums scalar. The cross terms are one
/// fused multiply-add rounding fewer, inside the same bound.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
fn squares_fma(
    x: &[f64],
    load: impl Fn(usize) -> std::arch::x86_64::__m256d,
    at: impl Fn(usize) -> f64,
) -> (Carried, bool) {
    use std::arch::x86_64::{
        __m256d, _CMP_LT_OQ, _mm256_add_pd, _mm256_and_pd, _mm256_andnot_pd, _mm256_cmp_pd,
        _mm256_fmadd_pd, _mm256_fmsub_pd, _mm256_loadu_pd, _mm256_movemask_pd, _mm256_mul_pd,
        _mm256_set1_pd, _mm256_setzero_pd, _mm256_storeu_pd, _mm256_sub_pd,
    };
    let sign = _mm256_set1_pd(-0.0);
    let (two, infinity) = (_mm256_set1_pd(2.0), _mm256_set1_pd(f64::INFINITY));
    let (mut sums, mut errors, mut magnitudes) = (
        [_mm256_setzero_pd(); 2],
        [_mm256_setzero_pd(); 2],
        [_mm256_setzero_pd(); 2],
    );
    // All ones while every difference is finite.
    let mut finite = _mm256_cmp_pd(_mm256_setzero_pd(), infinity, _CMP_LT_OQ);
    let (chunks, rest) = x.as_chunks::<LANES>();
    for (nth, chunk) in chunks.iter().enumerate() {
        for half in 0..2 {
            // SAFETY: four values from `4 * half` lie inside the chunk.
            let a = unsafe { _mm256_loadu_pd(chunk.as_ptr().add(4 * half)) };
            let b = load(LANES * nth + 4 * half);
            // `a - b` and its exact error, by a two-sum.
            let head = _mm256_sub_pd(a, b);
            let virtual_b = _mm256_sub_pd(a, head);
            let virtual_a = _mm256_add_pd(head, virtual_b);
            let tail = _mm256_sub_pd(_mm256_sub_pd(a, virtual_a), _mm256_sub_pd(b, virtual_b));
            finite = _mm256_and_pd(
                finite,
                _mm256_cmp_pd(_mm256_andnot_pd(sign, head), infinity, _CMP_LT_OQ),
            );
            let square = _mm256_mul_pd(head, head);
            let square_error = _mm256_fmsub_pd(head, head, square);
            let cross = _mm256_fmadd_pd(_mm256_mul_pd(two, head), tail, _mm256_mul_pd(tail, tail));
            // The square into the lane's sum by a two-sum.
            let sum = _mm256_add_pd(sums[half], square);
            let virtual_value = _mm256_sub_pd(sum, sums[half]);
            let virtual_sum = _mm256_sub_pd(sum, virtual_value);
            let rounding = _mm256_add_pd(
                _mm256_sub_pd(sums[half], virtual_sum),
                _mm256_sub_pd(square, virtual_value),
            );
            sums[half] = sum;
            let extra = _mm256_add_pd(square_error, cross);
            errors[half] = _mm256_add_pd(errors[half], _mm256_add_pd(rounding, extra));
            let size = _mm256_add_pd(
                _mm256_add_pd(
                    _mm256_andnot_pd(sign, rounding),
                    _mm256_andnot_pd(sign, square_error),
                ),
                _mm256_mul_pd(two, _mm256_andnot_pd(sign, cross)),
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
    let mut all_finite = _mm256_movemask_pd(finite) == 0b1111;
    let start = LANES * chunks.len();
    for (offset, &a) in rest.iter().enumerate() {
        let (head, tail) = two_sum(a, -at(start + offset));
        all_finite &= head.is_finite();
        let square = head * head;
        let square_error = head.mul_add(head, -square);
        let cross = 2.0 * head * tail + tail * tail;
        let (sum, rounding) = two_sum(sums[0], square);
        sums[0] = sum;
        errors[0] += rounding + (square_error + cross);
        magnitudes[0] += rounding.abs() + square_error.abs() + 2.0 * cross.abs();
    }
    // A value passes through three additions to its lane's error.
    let carried = finish(sums, errors, magnitudes, std::iter::empty(), 3 * x.len());
    (carried, all_finite)
}

/// [`distance_lanes`] for the unshifted 1-distance of float64 rows, in
/// 256-bit registers, written out because the compiler kept the two-sums
/// scalar: each difference's pair `h + t` by a two-sum, `|h|` added to its
/// lane's sum by a two-sum, and `t` with `h`'s sign taken off added with
/// that rounding to the lane's error -- two additions a value, which
/// `finish` is told of. A difference's sign is its leading part's, so none
/// is left undecided.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
fn taxicab_block_avx(x: &[f64], y: &[f64]) -> DistancePart {
    use std::arch::x86_64::{
        __m256d, _CMP_LT_OQ, _mm256_add_pd, _mm256_and_pd, _mm256_andnot_pd, _mm256_cmp_pd,
        _mm256_loadu_pd, _mm256_movemask_pd, _mm256_set1_pd, _mm256_setzero_pd, _mm256_storeu_pd,
        _mm256_sub_pd, _mm256_xor_pd,
    };
    debug_assert_eq!(x.len(), y.len());
    let (sign, infinity) = (_mm256_set1_pd(-0.0), _mm256_set1_pd(f64::INFINITY));
    let (mut sums, mut errors, mut magnitudes) = (
        [_mm256_setzero_pd(); 2],
        [_mm256_setzero_pd(); 2],
        [_mm256_setzero_pd(); 2],
    );
    // All ones while every difference is finite.
    let mut finite = _mm256_cmp_pd(_mm256_setzero_pd(), infinity, _CMP_LT_OQ);
    let (left, left_rest) = x.as_chunks::<LANES>();
    let (right, right_rest) = y.as_chunks::<LANES>();
    for (a, b) in left.iter().zip(right) {
        for half in 0..2 {
            // SAFETY: four values from `4 * half` lie inside each chunk.
            let (a, b) = unsafe {
                (
                    _mm256_loadu_pd(a.as_ptr().add(4 * half)),
                    _mm256_loadu_pd(b.as_ptr().add(4 * half)),
                )
            };
            // `a - b` and its exact error, by a two-sum.
            let head = _mm256_sub_pd(a, b);
            let virtual_b = _mm256_sub_pd(a, head);
            let virtual_a = _mm256_add_pd(head, virtual_b);
            let tail = _mm256_sub_pd(_mm256_sub_pd(a, virtual_a), _mm256_sub_pd(b, virtual_b));
            let magnitude = _mm256_andnot_pd(sign, head);
            finite = _mm256_and_pd(finite, _mm256_cmp_pd(magnitude, infinity, _CMP_LT_OQ));
            let tail = _mm256_xor_pd(tail, _mm256_and_pd(sign, head));
            // The magnitude into the lane's sum by a two-sum.
            let sum = _mm256_add_pd(sums[half], magnitude);
            let virtual_value = _mm256_sub_pd(sum, sums[half]);
            let virtual_sum = _mm256_sub_pd(sum, virtual_value);
            let rounding = _mm256_add_pd(
                _mm256_sub_pd(sums[half], virtual_sum),
                _mm256_sub_pd(magnitude, virtual_value),
            );
            sums[half] = sum;
            errors[half] = _mm256_add_pd(errors[half], _mm256_add_pd(rounding, tail));
            let size = _mm256_add_pd(
                _mm256_andnot_pd(sign, rounding),
                _mm256_andnot_pd(sign, tail),
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
    let mut all_finite = _mm256_movemask_pd(finite) == 0b1111;
    for (&a, &b) in left_rest.iter().zip(right_rest) {
        let (head, tail) = two_sum(a, -b);
        all_finite &= head.is_finite();
        let tail = if head < 0.0 { -tail } else { tail };
        let (sum, rounding) = two_sum(sums[0], head.abs());
        sums[0] = sum;
        errors[0] += rounding + tail;
        magnitudes[0] += rounding.abs() + tail.abs();
    }
    // A value passes through two additions to its lane's error.
    let carried = finish(sums, errors, magnitudes, std::iter::empty(), 2 * x.len());
    DistancePart {
        carried,
        loose: 0.0,
        finite: all_finite,
        unsure: false,
    }
}

/// One block of a distance in [`LANES`] lanes, each difference taken
/// exactly by [`exact_difference_of`].
///
/// For the 2-distance, each leading part's square is split -- by a fused
/// multiply-add where `FMA` -- and added to its lane's sum by a two-sum, the
/// two errors and the cross terms `2 head tail + tail²` added to the lane's
/// error: three additions a value, which `finish` is told of, with the cross
/// terms' own two roundings in their magnitude. For the 1-distance each part
/// takes the difference's sign, decided where the leading part outweighs
/// twice the others, and is carried as a value.
#[inline(always)]
fn distance_lanes<I: Element, const EUCLIDEAN: bool, const SHIFTED: bool, const FMA: bool>(
    x: &[I],
    y: &[I],
    shift: f64,
) -> DistancePart {
    let (mut sums, mut errors, mut magnitudes) = ([0f64; LANES], [0f64; LANES], [0f64; LANES]);
    let (mut loose, mut finite, mut unsure) = ([0f64; LANES], [true; LANES], [false; LANES]);
    let mut add = |lane: usize, a: f64, b: f64| {
        let (head, tail, third) = if SHIFTED {
            exact_difference_of(a, b, shift)
        } else {
            let (head, tail) = two_sum(a, -b);
            (head, tail, 0.0)
        };
        finite[lane] &= head.is_finite();
        if EUCLIDEAN {
            let mut carry = |value: f64, extra: f64, extra_magnitude: f64| {
                let (sum, rounding) = two_sum(sums[lane], value);
                sums[lane] = sum;
                errors[lane] += rounding + extra;
                magnitudes[lane] += rounding.abs() + extra_magnitude;
            };
            let square = head * head;
            let square_error = if FMA {
                head.mul_add(head, -square)
            } else {
                product_error(head, head, square)
            };
            let cross = 2.0 * head * tail + tail * tail;
            carry(
                square,
                square_error + cross,
                square_error.abs() + 2.0 * cross.abs(),
            );
            if SHIFTED {
                loose[lane] += (2.0 * (head.abs() + tail.abs()) + third.abs()) * third.abs();
            }
        } else {
            // Each part's rounding added to the error by a two-sum too, what
            // that drops kept as the bound: zero while the errors add
            // exactly, so that a total on a halfway point settles.
            let mut carry = |value: f64| {
                let (sum, rounding) = two_sum(sums[lane], value);
                sums[lane] = sum;
                let (error, dropped) = two_sum(errors[lane], rounding);
                errors[lane] = error;
                magnitudes[lane] += dropped.abs();
            };
            let sure = head.abs() > 2.0 * (tail.abs() + third.abs());
            let zero = (head == 0.0) & (tail == 0.0);
            unsure[lane] |= !(sure | zero);
            let sign = if sure { head.signum() } else { third.signum() };
            carry(sign * head);
            carry(sign * tail);
            if SHIFTED {
                carry(sign * third);
            }
        }
    };
    if x.len() < 2 * LANES {
        // A short row in one lane, with nothing to merge: each value passes
        // through up to three additions to the lane's error.
        for (a, b) in x.iter().zip(y) {
            add(0, a.wide(), b.wide());
        }
        let additions = (3 * x.len() + LANES) as f64;
        let slack = if EUCLIDEAN {
            magnitudes[0] * additions * 2f64.powi(-52)
        } else {
            // Twice the float64 sum of what was dropped covers its roundings.
            2.0 * magnitudes[0]
        };
        return DistancePart {
            carried: Carried {
                sum: sums[0],
                error: errors[0],
                slack,
            },
            loose: loose[0] * (1.0 + 2f64.powi(-50)),
            finite: finite[0],
            unsure: unsure[0],
        };
    }
    let (left, left_rest) = x.as_chunks::<LANES>();
    let (right, right_rest) = y.as_chunks::<LANES>();
    for (a, b) in left.iter().zip(right) {
        for lane in 0..LANES {
            add(lane, a[lane].wide(), b[lane].wide());
        }
    }
    for (a, b) in left_rest.iter().zip(right_rest) {
        add(0, a.wide(), b.wide());
    }
    // A value passes through up to three additions to its lane's error.
    let values = 3 * x.len();
    let carried = if EUCLIDEAN {
        finish(sums, errors, magnitudes, std::iter::empty(), values)
    } else {
        let lanes = (0..LANES).map(|lane| Carried {
            sum: sums[lane],
            error: errors[lane],
            slack: 2.0 * magnitudes[lane],
        });
        lanes.reduce(Carried::merge_tracked).unwrap_or_default()
    };
    DistancePart {
        carried,
        loose: loose.iter().sum::<f64>() * (1.0 + 2f64.powi(-50)),
        finite: finite.iter().all(|&finite| finite),
        unsure: unsure.iter().any(|&unsure| unsure),
    }
}

/// `x - y + shift` exactly, as a leading part, a part below its last bit and
/// a third below that one's, the last zero without a shift.
#[inline(always)]
fn exact_difference_of(x: f64, y: f64, shift: f64) -> (f64, f64, f64) {
    let (head, tail) = two_sum(x, -y);
    if shift == 0.0 {
        return (head, tail, 0.0);
    }
    let (shifted, rounding) = two_sum(head, shift);
    let (low, third) = two_sum(rounding, tail);
    let (head, tail) = two_sum(shifted, low);
    (head, tail, third)
}

/// [`distance`] as its formula gives it in float64, for operands with a NaN
/// or an infinity among them.
#[cold]
#[inline(never)]
fn distance_formula<I: Element, T: Output>(x: &[I], y: &[I], euclidean: bool, shift: f64) -> T {
    let differences = x.iter().zip(y).map(|(a, b)| a.wide() - b.wide() + shift);
    T::narrow(if euclidean {
        differences.map(|d| d * d).sum::<f64>().sqrt()
    } else {
        differences.map(f64::abs).sum()
    })
}

/// [`distance`] in whole numbers: the exact differences' magnitudes summed,
/// or their squares -- each expanded into products of the operands -- rooted.
#[cold]
#[inline(never)]
fn exact_distance<I: Element, T: Output>(x: &[I], y: &[I], euclidean: bool, shift: f64) -> T {
    let pairs = x.iter().zip(y).map(|(a, b)| (a.wide(), b.wide()));
    if euclidean {
        // `(a - b + s)²`, as products of the operands.
        let products = pairs.flat_map(move |(a, b)| {
            [
                (a, a),
                (b, b),
                (shift, shift),
                (a, -b),
                (a, -b),
                (a, shift),
                (a, shift),
                (-b, shift),
                (-b, shift),
            ]
        });
        return match Total::of_products(products) {
            Err(special) => T::narrow(special.sqrt()),
            Ok(total) => total.square_root(),
        };
    }
    // `|a - b + s|`, each difference's sign taken exactly.
    let values = pairs.flat_map(move |(a, b)| {
        let sign = match Total::of_wide([a, -b, shift].into_iter()) {
            Ok(total) => {
                let (negative, limbs, _) = total.difference();
                if negative && limbs.iter().any(|&limb| limb != 0) {
                    -1.0
                } else {
                    1.0
                }
            }
            Err(_) => 1.0,
        };
        [sign * a, -sign * b, sign * shift]
    });
    match Total::of_wide(values) {
        Err(special) => T::narrow(special),
        Ok(total) => total.rounded(),
    }
}

/// Each of `values` over the greater of their 2-norm -- their 1-norm with
/// `taxicab` -- and `floor`, into `out`, each the value nearest the exact
/// quotient; `scratch` is room for the values' magnitudes scaled into range.
///
/// A quotient does not move when the values are scaled by a power of two,
/// so they are, to bring the largest magnitude into `[1, 2)`; there the norm
/// is carried as [`root_of`] or [`sum_to`] carries it, its inverse taken as
/// a pair, and each magnitude times that settled by a bound. Where the
/// bound cannot say, or a magnitude is too small for its scaled copy to be
/// exact, the quotient is formed in whole numbers: the root of the value's
/// square over the exact sum of the squares, or the value over the exact
/// sum of the magnitudes. A norm below a positive `floor` -- decided
/// exactly where the bound cannot -- leaves each value over the floor,
/// rounded once. A NaN makes NaN of every quotient, an infinity what
/// dividing by an infinite norm gives, and a zero norm with no floor NaN.
pub(crate) fn normalized<I: Element, T: Output>(
    values: &[I],
    out: &mut [T],
    taxicab: bool,
    floor: f64,
    scratch: &mut Vec<f64>,
) {
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    fn with_fma<I: Element, T: Output>(
        values: &[I],
        out: &mut [T],
        taxicab: bool,
        floor: f64,
        scratch: &mut Vec<f64>,
    ) {
        normalized_with::<I, T, true>(values, out, taxicab, floor, scratch);
    }
    // A float32 answer comes from plain float64 sums nearly always, and has
    // no use for a second build of the rest.
    #[cfg(target_arch = "x86_64")]
    if fused() && !T::PLAIN {
        // SAFETY: avx2 and fma were detected on this CPU.
        return unsafe { with_fma(values, out, taxicab, floor, scratch) };
    }
    normalized_with::<I, T, false>(values, out, taxicab, floor, scratch);
}

/// [`normalized`], the products' errors by a fused multiply-add where `FMA`.
#[inline(always)]
fn normalized_with<I: Element, T: Output, const FMA: bool>(
    values: &[I],
    out: &mut [T],
    taxicab: bool,
    floor: f64,
    scratch: &mut Vec<f64>,
) {
    debug_assert_eq!(values.len(), out.len());
    // The floor as the output's dtype holds it.
    let floor = T::narrow(floor).widen();
    let floored = floor > 0.0;
    if T::PLAIN && plain_normalized(values, out, taxicab, floor) {
        return;
    }
    // A quotient under `2^-900` may have lost bits in its scaled magnitude,
    // or its product's error be too small to hold, and is formed exactly.
    const SMALL: f64 = f64::from_bits((1023 - 900) << 52);
    // The norm carried over the values as they stand, where the sum of
    // their squares or magnitudes is far inside the range -- no NaN, no
    // infinity, no zero row, and nothing that scaling would bring nearer one.
    let direct = (values.len() < 1 << 40)
        .then(|| {
            if taxicab {
                scratch.clear();
                scratch.extend(values.iter().map(|value| value.wide().abs()));
                carried_with(scratch, false)
            } else {
                I::carried_dot(values, values)
            }
        })
        .filter(|carried| {
            let range = f64::from_bits((1023 - 800) << 52)..=f64::from_bits((1023 + 800) << 52);
            range.contains(&(carried.sum + carried.error))
        });
    // The magnitudes times a power of two, `first · second`: one for a
    // direct norm, else what brings the largest into `[1, 2)`.
    let (mut carried, first, second) = match direct {
        Some(carried) => (carried, 1.0, 1.0),
        None => {
            let (mut largest, mut special) = (0f64, false);
            for value in values {
                let magnitude = value.wide().abs();
                largest = if magnitude > largest {
                    magnitude
                } else {
                    largest
                };
                special |= !magnitude.is_finite();
            }
            if special || largest == 0.0 {
                // A norm that is NaN, infinite or zero: each quotient is what
                // dividing by it gives -- by the floor where a zero norm has
                // one.
                let nan = special && values.iter().any(|value| value.wide().is_nan());
                let divisor = if nan {
                    f64::NAN
                } else if special {
                    f64::INFINITY
                } else if floored {
                    floor
                } else {
                    0.0
                };
                for (slot, value) in out.iter_mut().zip(values) {
                    *slot = T::narrow(value.wide() / divisor);
                }
                return;
            }
            // `2^-shift` as two powers of two, each in the normal range.
            let shift = binary_exponent(largest);
            let power = |shift: i32| f64::from_bits(((shift + 1023) as u64) << 52);
            let (first, second) = (power(-shift / 2), power(-shift - (-shift / 2)));
            scratch.clear();
            scratch.extend(
                values
                    .iter()
                    .map(|value| value.wide().abs() * first * second),
            );
            let carried = if taxicab {
                carried_with(scratch, false)
            } else {
                f64::carried_dot(scratch, scratch)
            };
            (carried, first, second)
        }
    };
    let magnitude = |value: &I| value.wide().abs() * first * second;
    // What the small magnitudes could add, their bits or their squares'
    // errors lost among the subnormals.
    carried.slack += values.len() as f64 * f64::from_bits(4);
    let (norm, rest, bound) = if taxicab {
        let (norm, rest) = two_sum(carried.sum, carried.error);
        (norm, rest, carried.slack * (1.0 + 2f64.powi(-50)))
    } else {
        let (norm, rest, bound, ordinary) = root_parts::<FMA>(carried);
        if !ordinary {
            return exact_normalized(values, out, taxicab, floor);
        }
        (norm, rest, bound)
    };
    if floored {
        let limit = floor * first * second;
        let (low, high) = (
            (norm - bound) * (1.0 - 2f64.powi(-50)),
            (norm + bound) * (1.0 + 2f64.powi(-50)),
        );
        let below = if limit > high {
            true
        } else if limit < low {
            false
        } else {
            below_floor(values, taxicab, floor)
        };
        if below {
            for (slot, value) in out.iter_mut().zip(values) {
                *slot = over_floor(value.wide(), floor);
            }
            return;
        }
    }
    // `1 / (norm + rest)` as a pair, within `12u²` of itself: the
    // reciprocal's remainder is exact, and what the pair leaves out is the
    // square of terms each under `u`, and three roundings.
    let inverse = 1.0 / norm;
    let remainder = if FMA {
        (-norm).mul_add(inverse, 1.0)
    } else {
        -((norm * inverse - 1.0) + product_error(norm, inverse, norm * inverse))
    };
    let tail = inverse * (remainder - rest * inverse);
    let relative =
        (bound / norm * (1.0 + 2f64.powi(-40)) + 24.0 * 2f64.powi(-106)) * (1.0 + 2f64.powi(-40));
    // Each quotient settled without a branch, and whether any was not.
    let quotient = |value: &I| -> (T, bool) {
        let magnitude = magnitude(value);
        let value = value.wide();
        let head = magnitude * inverse;
        let error = if FMA {
            magnitude.mul_add(inverse, -head)
        } else {
            product_error(magnitude, inverse, head)
        };
        // The head is the larger by far: a fast two-sum.
        let low = error + magnitude * tail;
        let quotient = head + low;
        let residue = low - (quotient - head);
        let sign = value.to_bits() & (1 << 63);
        let signed = |part: f64| f64::from_bits(part.to_bits() ^ sign);
        let bound = quotient * relative + f64::from_bits(2);
        let (rounded, clear) = T::settle_lanes(signed(quotient), signed(residue), bound);
        // A zero over the norm is that zero, its sign kept.
        let zero = value == 0.0;
        let rounded = if zero { T::narrow(value) } else { rounded };
        (rounded, zero | (clear & (head >= SMALL)))
    };
    let mut open = false;
    for (slot, value) in out.iter_mut().zip(values) {
        let (rounded, settled) = quotient(value);
        *slot = rounded;
        open |= !settled;
    }
    if !open {
        return;
    }
    let total = norm_total(values, taxicab);
    for (slot, value) in out.iter_mut().zip(values) {
        if !quotient(value).1 {
            *slot = quotient_by_norm(value.wide(), total.clone(), taxicab);
        }
    }
}

/// [`normalized`] for a float32 answer from a plain float64 norm, the
/// floor already in the output's dtype: the sum of the squares or of the
/// magnitudes within `(n - 1)u` of itself, its root within half that and a
/// rounding more, and each quotient by the inverse within two roundings
/// more -- some thirty bits finer than a float32 place. Rounding is
/// monotonic, so a quotient whose two ends narrow alike is settled; whether
/// every one was. Where not, or where the norm is not plainly in range and
/// clear of the floor, `out` is for writing again.
#[inline(always)]
fn plain_normalized<I: Element, T: Output>(
    values: &[I],
    out: &mut [T],
    taxicab: bool,
    floor: f64,
) -> bool {
    debug_assert!(T::PLAIN);
    const U: f64 = 1.0 / (1u64 << 53) as f64;
    let mut lanes = [0f64; LANES];
    let (chunks, rest) = values.as_chunks::<LANES>();
    for chunk in chunks {
        for lane in 0..LANES {
            let value = chunk[lane].wide();
            lanes[lane] += if taxicab { value.abs() } else { value * value };
        }
    }
    for value in rest {
        let value = value.wide();
        lanes[0] += if taxicab { value.abs() } else { value * value };
    }
    let total: f64 = lanes.iter().sum();
    let range = f64::from_bits((1023 - 900) << 52)..=f64::from_bits((1023 + 900) << 52);
    if !range.contains(&total) {
        return false;
    }
    let count = values.len() as f64;
    let (norm, relative) = if taxicab {
        (total, (count + 2.0) * U)
    } else {
        (total.sqrt(), (count / 2.0 + 3.0) * U)
    };
    let relative = relative * 1.01;
    if floor > 0.0 && norm * (1.0 - 2.0 * relative) <= floor {
        // At or near the floor: the carried norm decides.
        return false;
    }
    // Each end a float64 ulp further out for its own rounding.
    let relative = relative + 2.0 * U;
    let inverse = 1.0 / norm;
    let mut settled = true;
    for (slot, value) in out.iter_mut().zip(values) {
        let quotient = value.wide() * inverse;
        let bound = quotient.abs() * relative;
        *slot = T::narrow(quotient);
        settled &= T::narrow(quotient - bound) == T::narrow(quotient + bound);
    }
    settled
}

/// `floor(log2(value))` of a positive finite value, subnormals included.
fn binary_exponent(value: f64) -> i32 {
    let biased = ((value.to_bits() >> 52) & 0x7ff) as i32;
    if biased == 0 {
        // A subnormal: scaled up into the normal range first.
        binary_exponent(value * 2f64.powi(64)) - 64
    } else {
        biased - 1023
    }
}

/// The exact sum of the squares of `values`, or with `taxicab` of their
/// magnitudes, all finite.
#[cold]
fn norm_total<I: Element>(values: &[I], taxicab: bool) -> Total {
    let total = if taxicab {
        Total::of_wide(values.iter().map(|value| value.wide().abs()))
    } else {
        Total::of_products(values.iter().map(|value| (value.wide(), value.wide())))
    };
    total.expect("finite values")
}

/// Whether the norm of `values`, all finite, lies below `floor`: the exact
/// sum of the squares against its square, or of the magnitudes against it.
#[cold]
#[inline(never)]
fn below_floor<I: Element>(values: &[I], taxicab: bool, floor: f64) -> bool {
    let total = if taxicab {
        let magnitudes = values.iter().map(|value| value.wide().abs());
        Total::of_wide(magnitudes.chain(std::iter::once(-floor)))
    } else {
        let squares = values.iter().map(|value| (value.wide(), value.wide()));
        Total::of_products(squares.chain(std::iter::once((floor, -floor))))
    };
    total.expect("finite values").difference().0
}

/// The value nearest `value / floor`: the quotient rounded once, and the
/// sign of its exact remainder for a narrower float's tie.
fn over_floor<T: Output>(value: f64, floor: f64) -> T {
    let quotient = value / floor;
    let remainder = (-quotient).mul_add(floor, value);
    T::settle(quotient, remainder, 0.0).unwrap_or_else(|| T::narrow(quotient))
}

/// The value nearest `value` over the norm whose exact sum is `total`: the
/// root of its square over the sum of the squares, signed, or the value over
/// the sum of the magnitudes.
#[cold]
#[inline(never)]
fn quotient_by_norm<T: Output>(value: f64, total: Total, taxicab: bool) -> T {
    let numerator = if taxicab {
        Total::of_wide(std::iter::once(value))
    } else {
        Total::of_products(std::iter::once((value, value)))
    };
    let numerator = numerator.expect("a finite value");
    if taxicab {
        return numerator.quotient(total);
    }
    let magnitude: T = numerator.quotient_root(total);
    if value < 0.0 {
        T::narrow(-magnitude.widen())
    } else {
        magnitude
    }
}

/// [`normalized`] of a norm too far out for the carried one, every quotient
/// formed in whole numbers.
#[cold]
#[inline(never)]
fn exact_normalized<I: Element, T: Output>(values: &[I], out: &mut [T], taxicab: bool, floor: f64) {
    if floor > 0.0 && below_floor(values, taxicab, floor) {
        for (slot, value) in out.iter_mut().zip(values) {
            *slot = over_floor(value.wide(), floor);
        }
        return;
    }
    let total = norm_total(values, taxicab);
    for (slot, value) in out.iter_mut().zip(values) {
        *slot = quotient_by_norm(value.wide(), total.clone(), taxicab);
    }
}

/// The cosine of the angle between `a` and `b`: their dot over the product
/// of their 2-norms, each norm floored at `floor` as the output's dtype
/// holds it -- no floor where that is not positive -- the value nearest the
/// exact quotient; `scratch` is room for the two scaled into range.
///
/// A float32 answer is first taken from plain float64 sums, by
/// [`plain_cosine`]. Otherwise the dot and the two sums of squares are
/// carried -- over the values as they stand where the sums of squares are
/// far inside the range, or else over copies scaled by powers of two, which
/// leave the cosine where it is, to bring each largest magnitude into
/// `[1, 2)` -- each norm rooted as [`root_parts`] roots it, each floor
/// decided, exactly where the bound cannot say, and the quotient formed as a
/// pair and settled by a bound. Where it cannot be, the cosine is formed in
/// whole numbers by [`exact_cosine`]. A NaN or an infinity among the values
/// makes NaN, a dot that is exactly zero `+0`, and a zero norm with no floor
/// NaN.
pub(crate) fn cosine<I: Element, T: Output>(
    a: &[I],
    b: &[I],
    floor: f64,
    scratch: &mut (Vec<f64>, Vec<f64>),
) -> T {
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    fn with_fma<I: Element, T: Output>(
        a: &[I],
        b: &[I],
        floor: f64,
        scratch: &mut (Vec<f64>, Vec<f64>),
    ) -> T {
        cosine_with::<I, T, true>(a, b, floor, scratch)
    }
    // As for `normalized`: one build for a float32 answer.
    #[cfg(target_arch = "x86_64")]
    if fused() && !T::PLAIN {
        // SAFETY: avx2 and fma were detected on this CPU.
        return unsafe { with_fma(a, b, floor, scratch) };
    }
    cosine_with::<I, T, false>(a, b, floor, scratch)
}

/// [`cosine`], the products' errors by a fused multiply-add where `FMA`.
#[inline(always)]
fn cosine_with<I: Element, T: Output, const FMA: bool>(
    a: &[I],
    b: &[I],
    floor: f64,
    scratch: &mut (Vec<f64>, Vec<f64>),
) -> T {
    debug_assert_eq!(a.len(), b.len());
    // The floor as the output's dtype holds it, and zero for none.
    let floor = T::narrow(floor).widen();
    let floor = if floor > 0.0 { floor } else { 0.0 };
    if T::PLAIN
        && let Some(value) = plain_cosine(a, b, floor)
    {
        return value;
    }
    let range = f64::from_bits((1023 - 800) << 52)..=f64::from_bits((1023 + 800) << 52);
    let direct = (a.len() < 1 << 40)
        .then(|| (I::carried_dot(a, a), I::carried_dot(b, b)))
        .filter(|(squares_a, squares_b)| {
            range.contains(&(squares_a.sum + squares_a.error))
                && range.contains(&(squares_b.sum + squares_b.error))
        });
    let carried = match direct {
        Some((squares_a, squares_b)) => {
            let dot = I::carried_dot(a, b);
            // The dot is exactly zero where nothing was dropped from it and
            // no product of two values that are not zero was small enough
            // to lose bits.
            if exactly_zero(dot)
                && a.iter()
                    .zip(b)
                    .all(|(x, y)| whole_product(x.wide(), y.wide()))
            {
                return T::narrow(0.0);
            }
            Scaled {
                dot,
                squares: [squares_a, squares_b],
                scales: [(1.0, 1.0); 2],
            }
        }
        None => {
            if a.len() >= 1 << 40 {
                return exact_cosine(a, b, floor);
            }
            let (largest_a, special_a) = largest(a);
            let (largest_b, special_b) = largest(b);
            if special_a || special_b {
                return T::narrow(f64::NAN);
            }
            if largest_a == 0.0 || largest_b == 0.0 {
                // A zero vector: a zero dot, over a floored norm or over
                // nothing.
                return T::narrow(if floor > 0.0 { 0.0 } else { f64::NAN });
            }
            let (left, right) = scratch;
            let shift_a = scaled_into(a, largest_a, left);
            let shift_b = scaled_into(b, largest_b, right);
            let dot = f64::carried_dot(left, right);
            // As above, a scaled copy that lost a value's bits counted as
            // too small.
            if exactly_zero(dot)
                && left
                    .iter()
                    .zip(right.iter())
                    .zip(a.iter().zip(b))
                    .all(|((x, y), (p, q))| {
                        p.wide() == 0.0 || q.wide() == 0.0 || (x * y).abs() >= 2f64.powi(-968)
                    })
            {
                return T::narrow(0.0);
            }
            Scaled {
                dot,
                squares: [f64::carried_dot(left, left), f64::carried_dot(right, right)],
                scales: [shift_a, shift_b],
            }
        }
    };
    // What the small values could add, their bits or their products'
    // errors lost among the subnormals.
    let lost = a.len() as f64 * f64::from_bits(4);
    let below = |which: usize| below_floor(if which == 0 { a } else { b }, false, floor);
    carried
        .settled::<T, FMA>(lost, floor, below)
        .unwrap_or_else(|| exact_cosine(a, b, floor))
}

/// The carried dot and sums of squares of two vectors, each scaled by a
/// power of two, the product of the two in `scales` that [`scaled_into`]
/// applied.
pub(super) struct Scaled {
    pub(super) dot: Carried,
    pub(super) squares: [Carried; 2],
    pub(super) scales: [(f64, f64); 2],
}

impl Scaled {
    /// The cosine of the two vectors these carry, when the bounds settle
    /// it: their dot over their norms, each floored at `floor`, `below(k)`
    /// saying exactly whether the `k`-th norm lies below it where the
    /// bounds cannot, and `lost` what small values may have left out of each
    /// carried total.
    ///
    /// Each norm is `n1 + n2` within its bound, or the floor scaled as its
    /// vector was, exactly; the dot `d1 + d2` within its slack. The product
    /// of the norms is taken as a pair `p1 + p2` within `9u²` of their
    /// product, and the quotient as `q1 + q2`, `q1 = d1 / p1` with its exact
    /// remainder, within `32u²` of the pair's quotient: `2^-99` covers both
    /// with room. What the carried parts leave out adds their relative
    /// bounds, each held under `2^-40` so that their products are covered
    /// by a factor `1 + 2^-37`.
    #[inline(always)]
    pub(super) fn settled<T: Output, const FMA: bool>(
        self,
        lost: f64,
        floor: f64,
        below: impl Fn(usize) -> bool,
    ) -> Option<T> {
        let norm = |squares: Carried, scale: (f64, f64), which: usize| -> Option<(f64, f64, f64)> {
            let squares = Carried {
                slack: squares.slack + lost,
                ..squares
            };
            let (norm, rest, bound, ordinary) = root_parts::<FMA>(squares);
            if !ordinary {
                return None;
            }
            let limit = floor * scale.0 * scale.1;
            let (low, high) = (
                (norm - bound) * (1.0 - 2f64.powi(-50)),
                (norm + bound) * (1.0 + 2f64.powi(-50)),
            );
            let below = if limit > high {
                true
            } else if limit < low {
                false
            } else {
                below(which)
            };
            if !below {
                return Some((norm, rest, bound));
            }
            // The floor scaled exactly, and far enough inside the range for
            // the quotient's remainders.
            (limit >= 2f64.powi(-400) && limit <= 2f64.powi(400)).then_some((limit, 0.0, 0.0))
        };
        let (n1, n2, bound_a) = norm(self.squares[0], self.scales[0], 0)?;
        let (m1, m2, bound_b) = norm(self.squares[1], self.scales[1], 1)?;
        let (d1, d2) = two_sum(self.dot.sum, self.dot.error);
        // Clear of underflow in the remainders below, and not zero.
        let clear = d1.abs() >= 2f64.powi(-900);
        if !clear {
            return None;
        }
        let relative = |bound: f64, value: f64| bound / value.abs() * (1.0 + 2f64.powi(-40));
        let parts =
            relative(self.dot.slack + lost, d1) + relative(bound_a, n1) + relative(bound_b, m1);
        let known = parts <= 2f64.powi(-40);
        if !known {
            return None;
        }
        let p1 = n1 * m1;
        let e1 = if FMA {
            n1.mul_add(m1, -p1)
        } else {
            product_error(n1, m1, p1)
        };
        let p2 = e1 + (n1 * m2 + n2 * m1);
        let q1 = d1 / p1;
        // `d1 - q1 p1`, exact with a fused multiply-add and within a
        // rounding of itself otherwise.
        let remainder = if FMA {
            (-q1).mul_add(p1, d1)
        } else {
            let product = q1 * p1;
            (d1 - product) - product_error(q1, p1, product)
        };
        let q2 = ((remainder + d2) - q1 * p2) / p1;
        let (value, rest) = two_sum(q1, q2);
        let ordinary = value.abs() >= 2f64.powi(-900);
        if !ordinary {
            return None;
        }
        let bound = value.abs()
            * (parts * (1.0 + 2f64.powi(-37)) + 2f64.powi(-99))
            * (1.0 + 2f64.powi(-40));
        T::settle(value, rest, bound)
    }
}

/// Whether a carried total is exactly what it carries, and that is zero.
fn exactly_zero(carried: Carried) -> bool {
    carried.sum == 0.0 && carried.error == 0.0 && carried.slack == 0.0
}

/// Whether the product of `x` and `y` is zero or large enough that its
/// error, split off by a fused multiply-add, is exact.
fn whole_product(x: f64, y: f64) -> bool {
    x == 0.0 || y == 0.0 || (x * y).abs() >= 2f64.powi(-968)
}

/// The largest magnitude among `values`, and whether any is NaN or
/// infinite.
fn largest<I: Element>(values: &[I]) -> (f64, bool) {
    let (mut largest, mut special) = (0f64, false);
    for value in values {
        let magnitude = value.wide().abs();
        largest = if magnitude > largest {
            magnitude
        } else {
            largest
        };
        special |= !magnitude.is_finite();
    }
    (largest, special)
}

/// `values` times the power of two that brings `largest`, theirs, into
/// `[1, 2)`, into `out`; that power, which may itself lie past the range,
/// as the two in the normal range it is applied as.
fn scaled_into<I: Element>(values: &[I], largest: f64, out: &mut Vec<f64>) -> (f64, f64) {
    let shift = binary_exponent(largest);
    let power = |shift: i32| f64::from_bits(((shift + 1023) as u64) << 52);
    let (first, second) = (power(-shift / 2), power(-shift - (-shift / 2)));
    out.clear();
    out.extend(values.iter().map(|value| value.wide() * first * second));
    (first, second)
}

/// [`cosine`] for a float32 answer from plain float64 sums, the floor
/// already in the output's dtype, where they settle it.
///
/// The dot and the sums of squares are added in lanes, each within
/// `(n + 2)u` of the sum of its terms' magnitudes -- the dot's, by Cauchy
/// and Schwarz, at most the product of the norms -- so the dot over the
/// floored norms is within `(n + 2)u` of the exact one's numerator over
/// them, and the norms move it by their own `(n / 2 + 2)u` each. Rounding is
/// monotonic, so a cosine whose two ends narrow alike, both on its side of
/// zero, is that one.
#[inline(always)]
fn plain_cosine<I: Element, T: Output>(a: &[I], b: &[I], floor: f64) -> Option<T> {
    debug_assert!(T::PLAIN);
    const U: f64 = 1.0 / (1u64 << 53) as f64;
    let (mut dot, mut left, mut right) = ([0f64; LANES], [0f64; LANES], [0f64; LANES]);
    let (a_chunks, a_rest) = a.as_chunks::<LANES>();
    let (b_chunks, b_rest) = b.as_chunks::<LANES>();
    for (x, y) in a_chunks.iter().zip(b_chunks) {
        for lane in 0..LANES {
            let (x, y) = (x[lane].wide(), y[lane].wide());
            dot[lane] += x * y;
            left[lane] += x * x;
            right[lane] += y * y;
        }
    }
    for (x, y) in a_rest.iter().zip(b_rest) {
        let (x, y) = (x.wide(), y.wide());
        dot[0] += x * y;
        left[0] += x * x;
        right[0] += y * y;
    }
    let (dot, left, right): (f64, f64, f64) =
        (dot.iter().sum(), left.iter().sum(), right.iter().sum());
    let range = f64::from_bits((1023 - 900) << 52)..=f64::from_bits((1023 + 900) << 52);
    let spread = (a.len() as f64 + 2.0) * U * 1.01;
    if !(range.contains(&left) && range.contains(&right) && spread <= 2f64.powi(-20)) {
        return None;
    }
    // Each norm within `spread / 2` and a rounding of itself; twice that
    // either side of the floor leaves no doubt which applies.
    let wobble = spread + 2.0 * U;
    let length = |squares: f64| {
        let norm = squares.sqrt();
        if floor < norm * (1.0 - 2.0 * wobble) {
            Some(norm)
        } else if floor > norm * (1.0 + 2.0 * wobble) {
            Some(floor)
        } else {
            None
        }
    };
    let quotient = dot / (length(left)? * length(right)?);
    let bound = (spread + quotient.abs() * (2.0 * wobble + 4.0 * U)) * 1.01;
    let rounded = T::narrow(quotient);
    (bound < quotient.abs()
        && T::narrow(quotient - bound) == rounded
        && T::narrow(quotient + bound) == rounded)
        .then_some(rounded)
}

/// [`cosine`] formed in whole numbers: the root of the dot's square over the
/// product of the squared norms -- each the exact sum of the squares, or the
/// floor's square where that is larger -- signed as the dot is; NaN for a
/// value that is not finite.
#[cold]
#[inline(never)]
fn exact_cosine<I: Element, T: Output>(a: &[I], b: &[I], floor: f64) -> T {
    let products =
        |x: &[I], y: &[I]| Total::of_products(x.iter().zip(y).map(|(x, y)| (x.wide(), y.wide())));
    let (Ok(dot), Ok(squares_a), Ok(squares_b)) = (products(a, b), products(a, a), products(b, b))
    else {
        return T::narrow(f64::NAN);
    };
    let floored = |values: &[I], squares: Total| {
        if floor > 0.0 && below_floor(values, false, floor) {
            Total::of_products(std::iter::once((floor, floor))).expect("a finite floor")
        } else {
            squares
        }
    };
    let (negative, dot, dot_least) = dot.difference();
    let (_, squares_a, least_a) = floored(a, squares_a).difference();
    let (_, squares_b, least_b) = floored(b, squares_b).difference();
    let (dot, dot_least) = trimmed(&dot, dot_least);
    let (squares_a, least_a) = trimmed(&squares_a, least_a);
    let (squares_b, least_b) = trimmed(&squares_b, least_b);
    let numerator = Total::of_magnitude(product_of(dot, dot), 2 * dot_least);
    let denominator = Total::of_magnitude(product_of(squares_a, squares_b), least_a + least_b);
    let magnitude: T = numerator.quotient_root(denominator);
    if negative {
        T::narrow(-magnitude.widen())
    } else {
        magnitude
    }
}

/// A magnitude's limbs from its lowest that is not zero to its highest, and
/// the exponent of the first's least significant bit.
fn trimmed(limbs: &[u64], least: i32) -> (&[u64], i32) {
    let low = limbs.iter().position(|&limb| limb != 0).unwrap_or(0);
    let high = limbs
        .iter()
        .rposition(|&limb| limb != 0)
        .map_or(low, |top| top + 1);
    (&limbs[low..high], least + 64 * low as i32)
}

/// Whether the exact sum of `terms`, all finite, is negative.
#[cold]
pub(crate) fn is_negative(terms: &[f64]) -> bool {
    let total = Total::of_wide(terms.iter().copied()).expect("finite terms");
    let (negative, limbs, _) = total.difference();
    negative && limbs.iter().any(|&limb| limb != 0)
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

/// `x · y` and its exact rounding error, by a fused multiply-add where
/// `FMA` and from halves otherwise; exact while the product is finite and
/// at least `2^-960` or zero.
#[inline(always)]
fn split_product<const FMA: bool>(x: f64, y: f64) -> (f64, f64) {
    let product = x * y;
    let error = if FMA {
        x.mul_add(y, -product)
    } else {
        product_error(x, y, product)
    };
    (product, error)
}

/// Whether a product or quotient may have lost bits of its error to the
/// subnormal range, where the quick paths below do not answer.
#[inline(always)]
fn tiny(value: f64) -> bool {
    (value != 0.0) & (value.abs() < 2f64.powi(-960))
}

/// The value nearest `value + rest`, within `bound` of the exact sum of
/// `terms`, or of that sum taken exactly; an exact zero is `formula` where
/// that is a zero, and `+0` otherwise.
#[inline(always)]
fn settled_terms<T: Output>(
    value: f64,
    rest: f64,
    bound: f64,
    formula: f64,
    exact: impl FnOnce() -> T,
) -> T {
    if value == 0.0 && rest == 0.0 && bound == 0.0 {
        return T::narrow(if formula == 0.0 { formula } else { 0.0 });
    }
    if value.is_finite()
        && let Some(settled) = T::settle(value, rest, bound)
    {
        return settled;
    }
    exact()
}

/// The value nearest the exact `Σ x · y · z` over `terms`, all finite; an
/// exact zero is `formula` where that is a zero, and `+0` otherwise.
#[cold]
#[inline(never)]
fn exact_terms<T: Output>(terms: &[(f64, f64, f64)], formula: f64) -> T {
    let total = Total::of_terms(terms);
    if total.positive == total.negative {
        return T::narrow(if formula == 0.0 { formula } else { 0.0 });
    }
    total.rounded()
}

/// A quick path's value and whether its bound settles it, without a
/// branch: `parts` are a two-sum's pair within `bound` of the exact value
/// and whether they were taken exactly. An exact zero, whose sign the
/// formula decides, is left to the full path, as is anything not finite.
#[inline(always)]
fn settled_lane<T: Output>((sum, rest, bound, exact): (f64, f64, f64, bool)) -> (T, bool) {
    let (rounded, settled) = T::settle_lanes(sum, rest, bound);
    (rounded, settled & exact & (sum != 0.0))
}

/// `start + weight · end - weight · start` as a two-sum's pair, a bound on
/// the pair's distance from the exact value, and whether both products were
/// split exactly: each product split, the five parts added with the
/// roundings of the three smallest bounded.
#[inline(always)]
fn lerp_parts<const FMA: bool>(start: f64, end: f64, weight: f64) -> (f64, f64, f64, bool) {
    let (p, p_error) = split_product::<FMA>(weight, end);
    let (q, q_error) = split_product::<FMA>(weight, start);
    let (head, head_error) = two_sum(p, -q);
    let (value, value_error) = two_sum(start, head);
    let errors = p_error - q_error;
    let tail = (value_error + head_error) + errors;
    let (sum, rest) = two_sum(value, tail);
    // Three roundings in `tail`, each within `u` of its result.
    let bound = (value_error.abs() + head_error.abs() + errors.abs() + tail.abs()) * 2f64.powi(-52);
    (sum, rest, bound, !(tiny(p) | tiny(q)))
}

/// The value nearest `start + weight · (end - start)`, the point `weight`
/// of the way from `start` to `end`.
///
/// The exact value is `start + weight · end - weight · start`, taken by
/// [`lerp_parts`] and settled by its bound, or with the products summed
/// exactly where the bound reaches a halfway point. So `weight = 0` gives
/// `start` and `weight = 1` gives `end`, and the result moves the way
/// `weight` does. An operand that is not finite gives what the formula gives
/// in float64.
#[inline(always)]
pub(crate) fn lerp_of<T: Output, const FMA: bool>(start: f64, end: f64, weight: f64) -> T {
    let formula = start + weight * (end - start);
    if !(start.is_finite() && end.is_finite() && weight.is_finite()) {
        return T::narrow(formula);
    }
    let (sum, rest, bound, exact) = lerp_parts::<FMA>(start, end, weight);
    let terms = [(start, 1.0, 1.0), (weight, end, 1.0), (-weight, start, 1.0)];
    if !exact {
        return exact_terms(&terms, formula);
    }
    settled_terms(sum, rest, bound, formula, || exact_terms(&terms, formula))
}

/// [`lerp_of`]'s quick path without a branch; see [`settled_lane`].
#[inline(always)]
pub(crate) fn lerp_lane<T: Output, const FMA: bool>(
    start: f64,
    end: f64,
    weight: f64,
) -> (T, bool) {
    settled_lane(lerp_parts::<FMA>(start, end, weight))
}

/// `input + value · first · second` as [`lerp_parts`] gives its value:
/// `first · second` split exactly and its float64 times `value` split again,
/// so the sum has five parts, the smallest rounded once.
#[inline(always)]
fn addcmul_parts<const FMA: bool>(
    input: f64,
    first: f64,
    second: f64,
    value: f64,
) -> (f64, f64, f64, bool) {
    let (a, a_error) = split_product::<FMA>(first, second);
    let (b, b_error) = split_product::<FMA>(a, value);
    let c = a_error * value;
    let (head, head_error) = two_sum(input, b);
    let small = b_error + c;
    let tail = head_error + small;
    let (sum, rest) = two_sum(head, tail);
    // `c` is rounded once, then `small` and `tail`.
    let bound = (c.abs() + small.abs() + tail.abs()) * 2f64.powi(-52);
    (sum, rest, bound, !(tiny(a) | tiny(b) | tiny(c)))
}

/// The value nearest `input + value · first · second`, from
/// [`addcmul_parts`] or, where its bound reaches a halfway point, from the
/// triple product summed exactly. An operand that is not finite gives what
/// the formula gives in float64.
#[inline(always)]
pub(crate) fn addcmul_of<T: Output, const FMA: bool>(
    input: f64,
    first: f64,
    second: f64,
    value: f64,
) -> T {
    let formula = input + first * second * value;
    if !(input.is_finite() && first.is_finite() && second.is_finite() && value.is_finite()) {
        return T::narrow(formula);
    }
    let (sum, rest, bound, exact) = addcmul_parts::<FMA>(input, first, second, value);
    let terms = [(input, 1.0, 1.0), (value, first, second)];
    if !exact {
        return exact_terms(&terms, formula);
    }
    settled_terms(sum, rest, bound, formula, || exact_terms(&terms, formula))
}

/// [`addcmul_of`]'s quick path without a branch; see [`settled_lane`].
#[inline(always)]
pub(crate) fn addcmul_lane<T: Output, const FMA: bool>(
    input: f64,
    first: f64,
    second: f64,
    value: f64,
) -> (T, bool) {
    settled_lane(addcmul_parts::<FMA>(input, first, second, value))
}

/// `input + value · first / second` as [`lerp_parts`] gives its value: the
/// quotient `q` and its exact remainder `first - q · second` make
/// `value · q`, split exactly, and `value · remainder / second`, rounded
/// twice.
#[inline(always)]
fn addcdiv_parts<const FMA: bool>(
    input: f64,
    first: f64,
    second: f64,
    value: f64,
) -> (f64, f64, f64, bool) {
    let quotient = first / second;
    let (m, m_error) = split_product::<FMA>(quotient, second);
    // `m` is within two roundings of `first`, so the first difference is
    // exact, and so is the second: a quotient's remainder is a float64.
    let remainder = (first - m) - m_error;
    let (b, b_error) = split_product::<FMA>(value, quotient);
    let c = value * (remainder / second);
    let (head, head_error) = two_sum(input, b);
    let small = b_error + c;
    let tail = head_error + small;
    let (sum, rest) = two_sum(head, tail);
    // `c` is rounded twice, then `small` and `tail`.
    let bound = (3.0 * c.abs() + small.abs() + tail.abs()) * 2f64.powi(-52);
    (
        sum,
        rest,
        bound,
        !(tiny(first) | tiny(m) | tiny(b) | tiny(c)),
    )
}

/// The value nearest `input + value · first / second`, from
/// [`addcdiv_parts`] or, where its bound reaches a halfway point, as
/// `(input · second + value · first) / second` exactly. A zero divisor, or
/// an operand that is not finite, gives what the formula gives in float64.
#[inline(always)]
pub(crate) fn addcdiv_of<T: Output, const FMA: bool>(
    input: f64,
    first: f64,
    second: f64,
    value: f64,
) -> T {
    let formula = input + first / second * value;
    if !(input.is_finite() && first.is_finite() && second.is_finite() && value.is_finite())
        || second == 0.0
    {
        return T::narrow(formula);
    }
    let (sum, rest, bound, exact) = addcdiv_parts::<FMA>(input, first, second, value);
    let exact_value = || exact_addcdiv(input, first, second, value, formula);
    if !exact {
        return exact_value();
    }
    settled_terms(sum, rest, bound, formula, exact_value)
}

/// [`addcdiv_of`]'s quick path without a branch; see [`settled_lane`]. A
/// zero divisor leaves a quotient that is not finite, and so the full path.
#[inline(always)]
pub(crate) fn addcdiv_lane<T: Output, const FMA: bool>(
    input: f64,
    first: f64,
    second: f64,
    value: f64,
) -> (T, bool) {
    settled_lane(addcdiv_parts::<FMA>(input, first, second, value))
}

/// [`lerp_lane`] for float32 operands, whose products float64 holds
/// exactly: their difference and the sum are each rounded once, within
/// `u` of the result, which settles a float32 unless it lies by a halfway
/// point. Only for float32 operands.
#[inline(always)]
pub(crate) fn lerp_lane_narrow<T: Output>(start: f64, end: f64, weight: f64) -> (T, bool) {
    let difference = weight * end - weight * start;
    let sum = start + difference;
    let bound = (difference.abs() + sum.abs()) * 2f64.powi(-52);
    settled_lane((sum, 0.0, bound, true))
}

/// [`addcmul_lane`] for float32 operands: `first · second` is exact in
/// float64, and its product with `value` and the sum are each rounded
/// once. Only for float32 operands.
#[inline(always)]
pub(crate) fn addcmul_lane_narrow<T: Output>(
    input: f64,
    first: f64,
    second: f64,
    value: f64,
) -> (T, bool) {
    let product = first * second * value;
    let sum = input + product;
    let bound = (product.abs() + sum.abs()) * 2f64.powi(-52);
    settled_lane((sum, 0.0, bound, true))
}

/// [`addcdiv_lane`] for float32 operands: the quotient, its product with
/// `value` and the sum are each rounded once. Only for float32 operands,
/// whose quotients stay clear of the subnormal range.
#[inline(always)]
pub(crate) fn addcdiv_lane_narrow<T: Output>(
    input: f64,
    first: f64,
    second: f64,
    value: f64,
) -> (T, bool) {
    let product = first / second * value;
    let sum = input + product;
    let bound = (3.0 * product.abs() + sum.abs()) * 2f64.powi(-52);
    settled_lane((sum, 0.0, bound, true))
}

/// The value nearest `(input · second + value · first) / second`, all
/// finite and `second` not zero; an exact zero is `formula` where that is a
/// zero, and `+0` otherwise.
#[cold]
#[inline(never)]
fn exact_addcdiv<T: Output>(input: f64, first: f64, second: f64, value: f64, formula: f64) -> T {
    let pairs = [(input, second), (value, first)];
    let numerator = Total::of_products(pairs.into_iter()).expect("finite");
    if numerator.positive == numerator.negative {
        return T::narrow(if formula == 0.0 { formula } else { 0.0 });
    }
    let divisor = Total::of_wide(std::iter::once(second)).expect("finite");
    numerator.quotient(divisor)
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

    /// A sum that is the magnitude `limbs`, from `2^least` up.
    fn of_magnitude(limbs: Vec<u64>, least: i32) -> Self {
        let negative = vec![0; limbs.len()];
        Self {
            positive: limbs,
            negative,
            least,
        }
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

    /// The exact sum of the products of `terms`, three finite factors each.
    #[cold]
    #[inline(never)]
    fn of_terms(terms: &[(f64, f64, f64)]) -> Self {
        // Every product of three finite float64s is a whole multiple of
        // `2^-3222` below `2^3072`, bit 6294 here; the carries take a limb.
        let least = 3 * LEAST;
        let mut total = Self::empty(least, (6294 + 64) / 64 + 2);
        for &(x, y, z) in terms {
            let exact = |value: f64| Exact::float(value).expect("finite");
            let (x, y, z) = (exact(x), exact(y), exact(z));
            let pair = u128::from(x.magnitude) * u128::from(y.magnitude);
            if pair == 0 || z.magnitude == 0 {
                continue;
            }
            let limbs = if x.negative ^ y.negative ^ z.negative {
                &mut total.negative
            } else {
                &mut total.positive
            };
            let shift = (x.exponent + y.exponent + z.exponent - least) as usize;
            // The pair's two halves, 64 bits and at most 42, each times the
            // third's at most 53.
            let third = u128::from(z.magnitude);
            add_shifted(limbs, (pair & u128::from(u64::MAX)) * third, shift);
            add_shifted(limbs, (pair >> 64) * third, shift + 64);
        }
        total
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

    /// The value nearest the root of this sum over `divisor`'s, both not
    /// negative: the quotient's leading 125 to 127 bits by long division, at
    /// an even exponent, rooted as a whole number, with whether anything is
    /// left of the division or the root marked below. NaN over zero.
    fn quotient_root<T: Output>(self, divisor: Self) -> T {
        let (_, numerator, least) = self.difference();
        let (_, divisor, divisor_least) = divisor.difference();
        let bits = |limbs: &[u64]| {
            limbs
                .iter()
                .rposition(|&limb| limb != 0)
                .map_or(0, |top| 64 * top + 64 - limbs[top].leading_zeros() as usize)
        };
        let (top, bottom) = (bits(&numerator), bits(&divisor));
        if bottom == 0 {
            return T::narrow(f64::NAN);
        }
        if top == 0 {
            return T::narrow(0.0);
        }
        // A dividend `bottom + 125` bits long, or one more where that leaves
        // the quotient's exponent odd.
        let mut shift = bottom as i64 + 125 - top as i64;
        if (i64::from(least) - i64::from(divisor_least) - shift).rem_euclid(2) != 0 {
            shift += 1;
        }
        let (mut remainder, divisor) = if shift >= 0 {
            (shifted(&numerator, shift as usize), divisor)
        } else {
            (numerator, shifted(&divisor, (-shift) as usize))
        };
        let mut quotient = 0u128;
        for bit in (0..127).rev() {
            let step = shifted(&divisor, bit);
            if at_least(&remainder, &step) {
                subtract(&mut remainder, &step);
                quotient |= 1 << bit;
            }
        }
        let root = quotient.isqrt();
        let inexact = remainder.iter().any(|&limb| limb != 0) || root * root != quotient;
        let exponent = (i64::from(least) - i64::from(divisor_least) - shift) / 2 - 1;
        rounded(false, (root << 1) | u128::from(inexact), exponent as i32)
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

/// `a · b` of two magnitudes in limbs, schoolbook.
pub(super) fn product_of(a: &[u64], b: &[u64]) -> Vec<u64> {
    let mut out = vec![0u64; a.len() + b.len() + 1];
    for (i, &x) in a.iter().enumerate() {
        if x == 0 {
            continue;
        }
        let mut carry = 0u128;
        for (j, &y) in b.iter().enumerate() {
            let sum = u128::from(out[i + j]) + u128::from(x) * u128::from(y) + carry;
            out[i + j] = sum as u64;
            carry = sum >> 64;
        }
        let mut index = i + b.len();
        while carry != 0 {
            let sum = u128::from(out[index]) + carry;
            out[index] = sum as u64;
            carry = sum >> 64;
            index += 1;
        }
    }
    out
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
pub(super) fn rounded_at<T: Output>(negative: bool, limbs: &[u64], inexact: bool, least: i32) -> T {
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

    /// Whether `got`, among values `down` below it and `up` above it, is the
    /// value nearest `Σ terms / divisor`: the exact value lies between the
    /// halfway points either side, on one only where `got` is even.
    fn nearest_value(
        terms: &[(f64, f64, f64)],
        divisor: f64,
        got: f64,
        (down, up): (f64, f64),
        even: bool,
    ) -> bool {
        let sign = |extra: [(f64, f64, f64); 2], negate: bool| {
            let mut all: Vec<(f64, f64, f64)> = terms
                .iter()
                .map(|&(x, y, z)| if negate { (-x, y, z) } else { (x, y, z) })
                .chain(extra)
                .collect();
            all.shrink_to_fit();
            let (negative, limbs, _) = Total::of_terms(&all).difference();
            let side = match (limbs.iter().all(|&limb| limb == 0), negative) {
                (true, _) => 0,
                (false, true) => -1,
                (false, false) => 1,
            };
            if divisor < 0.0 { -side } else { side }
        };
        // `Σ terms - (got - down / 2) · divisor` and
        // `(got + up / 2) · divisor - Σ terms`.
        let low = sign([(-got, divisor, 1.0), (down, divisor, 0.5)], false);
        let high = sign([(got, divisor, 1.0), (up, divisor, 0.5)], true);
        let inside = |side: i32| side > 0 || (side == 0 && even);
        inside(low) && inside(high)
    }

    /// The gaps to the float64s either side of `got`, the largest's gap
    /// above it taken as the ulp an unbounded exponent would give.
    fn gaps_wide(got: f64) -> (f64, f64) {
        let beyond = 2f64.powi(971);
        let down = if got == -f64::MAX {
            beyond
        } else {
            got - got.next_down()
        };
        let up = if got == f64::MAX {
            beyond
        } else {
            got.next_up() - got
        };
        (down, up)
    }

    /// [`gaps_wide`] for float32.
    fn gaps_narrow(got: f32) -> (f64, f64) {
        let beyond = 2f64.powi(104);
        let wide = f64::from(got);
        let down = if got == -f32::MAX {
            beyond
        } else {
            wide - f64::from(got.next_down())
        };
        let up = if got == f32::MAX {
            beyond
        } else {
            f64::from(got.next_up()) - wide
        };
        (down, up)
    }

    #[test]
    fn lerp_addcmul_and_addcdiv_round_once() {
        let mut next = generator(0x510e_527f_ade6_82d1);
        let mut draw = move |spread: u64| {
            let unit = (next() >> 11) as f64 / (1u64 << 53) as f64 - 0.5;
            let exponent = (next() % (2 * spread + 1)) as i32 - spread as i32;
            unit * 2f64.powi(exponent)
        };
        let check = |terms: &[(f64, f64, f64)], divisor: f64, wide: [f64; 2], narrow: f32| {
            assert_eq!(wide[0].to_bits(), wide[1].to_bits(), "{terms:?}");
            let got = wide[0];
            if got.is_finite() {
                let even = got.to_bits() & 1 == 0;
                assert!(
                    nearest_value(terms, divisor, got, gaps_wide(got), even),
                    "{terms:?} / {divisor:e} gave {got:e}"
                );
            }
            let _ = narrow;
        };
        let check_narrow = |terms: &[(f64, f64, f64)], divisor: f64, got: f32| {
            if got.is_finite() {
                let even = got.to_bits() & 1 == 0;
                assert!(
                    nearest_value(terms, divisor, f64::from(got), gaps_narrow(got), even),
                    "{terms:?} / {divisor:e} gave {got:e} in float32"
                );
            }
        };
        for round in 0..6000 {
            let spread = [4, 30, 300][round % 3];
            let (mut a, mut b, mut c, mut d) = (draw(spread), draw(spread), draw(4), draw(spread));
            match round % 5 {
                // Ends near each other, and a weight near the ends.
                0 => b = a * (1.0 + draw(0) * 1e-9),
                1 => c = [0.0, 1.0, 0.5, -0.0, 2.0][round / 5 % 5],
                // An input that cancels the product.
                2 => a = -(b * c * d),
                3 => d = c * b / a,
                _ => {}
            }
            if round % 7 == 0 {
                (a, b) = (-b, b);
            }
            // lerp(start = a, end = b, weight = c)
            let terms = [(a, 1.0, 1.0), (c, b, 1.0), (-c, a, 1.0)];
            check(
                &terms,
                1.0,
                [
                    lerp_of::<f64, false>(a, b, c),
                    lerp_of::<f64, true>(a, b, c),
                ],
                0.0,
            );
            // addcmul(input = a, b, d, value = c)
            let terms = [(a, 1.0, 1.0), (c, b, d)];
            check(
                &terms,
                1.0,
                [
                    addcmul_of::<f64, false>(a, b, d, c),
                    addcmul_of::<f64, true>(a, b, d, c),
                ],
                0.0,
            );
            // addcdiv(input = a, b, d, value = c)
            if d != 0.0 {
                let terms = [(a, d, 1.0), (c, b, 1.0)];
                check(
                    &terms,
                    d,
                    [
                        addcdiv_of::<f64, false>(a, b, d, c),
                        addcdiv_of::<f64, true>(a, b, d, c),
                    ],
                    0.0,
                );
            }
            for ((lane, settled), full) in [
                (
                    lerp_lane::<f64, true>(a, b, c),
                    lerp_of::<f64, true>(a, b, c),
                ),
                (
                    addcmul_lane::<f64, false>(a, b, d, c),
                    addcmul_of::<f64, false>(a, b, d, c),
                ),
                (
                    addcdiv_lane::<f64, true>(a, b, d, c),
                    addcdiv_of::<f64, true>(a, b, d, c),
                ),
            ] {
                assert!(
                    !settled || lane.to_bits() == full.to_bits(),
                    "{a:e} {b:e} {c:e} {d:e}"
                );
            }
            // The same in float32.
            let (a, b, c, d) = (
                f64::from(a as f32),
                f64::from(b as f32),
                f64::from(c as f32),
                f64::from(d as f32),
            );
            if [a, b, c, d].iter().all(|value| value.is_finite()) {
                let got = lerp_of::<f32, false>(a, b, c);
                check_narrow(&[(a, 1.0, 1.0), (c, b, 1.0), (-c, a, 1.0)], 1.0, got);
                let got = addcmul_of::<f32, true>(a, b, d, c);
                check_narrow(&[(a, 1.0, 1.0), (c, b, d)], 1.0, got);
                if d != 0.0 {
                    let got = addcdiv_of::<f32, false>(a, b, d, c);
                    check_narrow(&[(a, d, 1.0), (c, b, 1.0)], d, got);
                }
                // A lane that settles gives what the full path gives.
                let lanes = [
                    (
                        lerp_lane_narrow::<f32>(a, b, c),
                        lerp_of::<f32, false>(a, b, c),
                    ),
                    (
                        lerp_lane::<f32, false>(a, b, c),
                        lerp_of::<f32, false>(a, b, c),
                    ),
                    (
                        addcmul_lane_narrow::<f32>(a, b, d, c),
                        addcmul_of::<f32, true>(a, b, d, c),
                    ),
                    (
                        addcmul_lane::<f32, true>(a, b, d, c),
                        addcmul_of::<f32, true>(a, b, d, c),
                    ),
                    (
                        addcdiv_lane_narrow::<f32>(a, b, d, c),
                        addcdiv_of::<f32, false>(a, b, d, c),
                    ),
                    (
                        addcdiv_lane::<f32, false>(a, b, d, c),
                        addcdiv_of::<f32, false>(a, b, d, c),
                    ),
                ];
                for ((lane, settled), full) in lanes {
                    assert!(
                        !settled || lane.to_bits() == full.to_bits(),
                        "{a:e} {b:e} {c:e} {d:e}"
                    );
                }
            }
        }
        // The ends of the range.
        let largest = f64::MAX;
        assert_eq!(
            lerp_of::<f64, false>(-largest, largest, 0.75),
            largest / 2.0
        );
        assert_eq!(lerp_of::<f64, true>(-largest, largest, 1.0), largest);
        assert_eq!(lerp_of::<f64, false>(largest, largest, 3.0), largest);
        assert_eq!(lerp_of::<f64, false>(0.0, largest, 2.0), f64::INFINITY);
        // `2^1000 · 2^1000` is past the largest float64 on the way.
        let (big, small) = (2f64.powi(1000), 2f64.powi(-1000));
        assert_eq!(addcmul_of::<f64, false>(3.0, big, big, small), big);
        assert_eq!(addcmul_of::<f64, true>(3.0, big, small, small), 3.0);
        assert_eq!(
            addcmul_of::<f64, true>(-largest, largest, 2.0, 0.5),
            0.0,
            "the product cancels the input"
        );
        let tiny = f64::from_bits(1);
        assert_eq!(addcmul_of::<f64, false>(0.0, tiny, 0.5, 1.0), 0.0);
        assert_eq!(addcmul_of::<f64, false>(0.0, tiny, 0.75, 1.0), tiny);
        assert_eq!(addcdiv_of::<f64, false>(0.0, tiny, 4.0, 3.0), tiny);
        assert_eq!(addcdiv_of::<f64, true>(1.0, 1.0, 3.0, 3.0), 2.0);
        assert_eq!(addcdiv_of::<f32, false>(1.0, 1.0, 3.0, 3.0), 2.0);
        // Specials and zeros take the formula's.
        assert!(lerp_of::<f64, false>(f64::INFINITY, f64::INFINITY, 0.5).is_nan());
        assert_eq!(
            lerp_of::<f64, false>(0.0, f64::INFINITY, 0.5),
            f64::INFINITY
        );
        assert_eq!(addcdiv_of::<f64, false>(1.0, 1.0, 0.0, 1.0), f64::INFINITY);
        assert!(addcdiv_of::<f64, false>(1.0, 0.0, 0.0, 1.0).is_nan());
        let weight = -1.0f64;
        let negative_zero = lerp_of::<f64, false>(-0.0, -0.0, weight);
        let formula = -0.0f64 + weight * (-0.0f64 - -0.0);
        assert_eq!(negative_zero.to_bits(), formula.to_bits());
        assert_eq!(
            addcmul_of::<f64, false>(-0.0, -0.0, 1.0, 1.0).to_bits(),
            (-0.0f64).to_bits()
        );
        assert_eq!(addcmul_of::<f64, false>(-1.0, 1.0, 1.0, 1.0).to_bits(), 0);
    }

    /// Whether `got`, among values `down` below it and `up` above it, is the
    /// value nearest the root of `Σ terms`: the exact sum lies between the
    /// squares of the halfway points either side, on one only where `got` is
    /// even.
    fn nearest_root_of_terms(
        terms: &[(f64, f64, f64)],
        got: f64,
        (down, up): (f64, f64),
        even: bool,
    ) -> bool {
        let sign = |extra: [(f64, f64, f64); 3], negate: bool| {
            let all: Vec<(f64, f64, f64)> = terms
                .iter()
                .map(|&(x, y, z)| if negate { (-x, y, z) } else { (x, y, z) })
                .chain(extra)
                .collect();
            let (negative, limbs, _) = Total::of_terms(&all).difference();
            match (limbs.iter().all(|&limb| limb == 0), negative) {
                (true, _) => 0,
                (false, true) => -1,
                (false, false) => 1,
            }
        };
        // `S - (got - down / 2)²` and `(got + up / 2)² - S`; a root of zero
        // has nothing below it.
        let low = if got == 0.0 {
            1
        } else {
            sign(
                [(-got, got, 1.0), (got, down, 1.0), (-down, down, 0.25)],
                false,
            )
        };
        let high = sign([(got, got, 1.0), (got, up, 1.0), (up, up, 0.25)], true);
        let inside = |side: i32| side > 0 || (side == 0 && even);
        inside(low) && inside(high)
    }

    #[test]
    fn distances_round_once() {
        let mut next = generator(0x9b05_688c_2b3e_6c1f);
        let mut draw = move |spread: u64| {
            let unit = (next() >> 11) as f64 / (1u64 << 53) as f64 - 0.5;
            let exponent = (next() % (2 * spread + 1)) as i32 - spread as i32;
            unit * 2f64.powi(exponent)
        };
        for round in 0..1500 {
            let len = 1 + round % 300;
            let spread = [2, 30, 300][round % 3];
            let x: Vec<f64> = (0..len).map(|_| draw(spread)).collect();
            let mut y: Vec<f64> = (0..len).map(|_| draw(spread)).collect();
            if round % 4 == 0 {
                // Rows that nearly coincide.
                for (y, x) in y.iter_mut().zip(&x) {
                    *y = x * (1.0 + draw(0) * 1e-12);
                }
            }
            let shift = if round % 5 == 0 {
                draw(spread) * 1e-3
            } else {
                0.0
            };
            // The lanes split a square the same with halves as with a fused
            // multiply-add: both errors are exact.
            for (halves, fused) in [
                (
                    distance_lanes::<f64, true, false, false>(&x, &y, 0.0),
                    distance_lanes::<f64, true, false, true>(&x, &y, 0.0),
                ),
                (
                    distance_lanes::<f64, true, true, false>(&x, &y, shift),
                    distance_lanes::<f64, true, true, true>(&x, &y, shift),
                ),
            ] {
                let bits = |part: &DistancePart| {
                    let carried = &part.carried;
                    [carried.sum, carried.error, carried.slack].map(f64::to_bits)
                };
                assert_eq!(bits(&halves), bits(&fused), "round {round}");
            }
            // The registers' sums are the lanes', for a row long enough to
            // fill them: each square and its error are exact either way.
            #[cfg(target_arch = "x86_64")]
            if fused() && len >= 2 * LANES {
                // SAFETY: avx2 and fma were detected on this CPU.
                let registers = unsafe { euclidean_block_fma(&x, &y) };
                let lanes = distance_lanes::<f64, true, false, true>(&x, &y, 0.0);
                assert_eq!(registers.carried.sum.to_bits(), lanes.carried.sum.to_bits());
                assert_eq!(registers.finite, lanes.finite);
            }
            for euclidean in [false, true] {
                let got: f64 = distance(&x, &y, euclidean, shift);
                let narrow_x: Vec<f32> = x.iter().map(|&v| v as f32).collect();
                let narrow_y: Vec<f32> = y.iter().map(|&v| v as f32).collect();
                let narrow_shift = f64::from(shift as f32);
                let narrow: f32 = distance(&narrow_x, &narrow_y, euclidean, narrow_shift);
                let cases = [
                    (
                        x.clone(),
                        y.clone(),
                        shift,
                        got,
                        gaps_wide(got),
                        got.to_bits() & 1 == 0,
                    ),
                    (
                        narrow_x.iter().map(|&v| f64::from(v)).collect(),
                        narrow_y.iter().map(|&v| f64::from(v)).collect(),
                        narrow_shift,
                        f64::from(narrow),
                        gaps_narrow(narrow),
                        narrow.to_bits() & 1 == 0,
                    ),
                ];
                for (x, y, shift, got, gaps, even) in cases {
                    if !got.is_finite() {
                        continue;
                    }
                    let ok = if euclidean {
                        let terms: Vec<(f64, f64, f64)> = x
                            .iter()
                            .zip(&y)
                            .flat_map(|(&a, &b)| {
                                [
                                    (a, a, 1.0),
                                    (b, b, 1.0),
                                    (shift, shift, 1.0),
                                    (a, -b, 2.0),
                                    (a, shift, 2.0),
                                    (-b, shift, 2.0),
                                ]
                            })
                            .collect();
                        nearest_root_of_terms(&terms, got, gaps, even)
                    } else {
                        let terms: Vec<(f64, f64, f64)> = x
                            .iter()
                            .zip(&y)
                            .flat_map(|(&a, &b)| {
                                let parts = [(a, 1.0, 1.0), (-b, 1.0, 1.0), (shift, 1.0, 1.0)];
                                let (negative, limbs, _) = Total::of_terms(&parts).difference();
                                let sign = if negative && limbs.iter().any(|&limb| limb != 0) {
                                    -1.0
                                } else {
                                    1.0
                                };
                                [
                                    (sign * a, 1.0, 1.0),
                                    (-sign * b, 1.0, 1.0),
                                    (sign * shift, 1.0, 1.0),
                                ]
                            })
                            .collect();
                        nearest_value(&terms, 1.0, got, gaps, even)
                    };
                    assert!(ok, "round {round} euclidean {euclidean}: {got:e}");
                }
            }
        }
        // The ends of the range, where squares leave float64.
        let largest = f64::MAX;
        let got: f64 = distance(&[largest, 1.0], &[0.0, 1.0], true, 0.0);
        assert_eq!(got, largest);
        let got: f64 = distance(&[largest], &[-largest], false, 0.0);
        assert_eq!(got, f64::INFINITY);
        let tiny = f64::from_bits(1);
        let got: f64 = distance(&[3.0 * tiny, 0.0], &[0.0, -4.0 * tiny], true, 0.0);
        assert_eq!(got, 5.0 * tiny);
        let got: f64 = distance(&[1e-200, 3e-200], &[-1e-200, 0.0], false, 0.0);
        assert_eq!(got, 5e-200);
        // Specials take the formula's answer.
        assert!(distance::<f64, f64>(&[f64::NAN], &[1.0], true, 0.0).is_nan());
        let got: f64 = distance(&[f64::INFINITY, 1.0], &[0.0, 1.0], false, 0.0);
        assert_eq!(got, f64::INFINITY);
        // No elements, and equal rows, are zero apart.
        assert_eq!(distance::<f64, f64>(&[], &[], true, 0.0), 0.0);
        assert_eq!(
            distance::<f32, f32>(&[1.5, -2.0], &[1.5, -2.0], false, 0.0),
            0.0
        );
    }

    /// Whether `got` is the variance of `values` -- or with `root` its root --
    /// nearest the exact one: `N Σx² - (Σx)²` against the halfway points
    /// either side, scaled by `N (N - ddof)`.
    fn nearest_variance(
        values: &[f64],
        ddof: usize,
        root: bool,
        got: f64,
        gaps: (f64, f64),
        even: bool,
    ) -> bool {
        let count = values.len() as f64;
        let scale = count * (count - ddof as f64);
        let mut numerator: Vec<(f64, f64, f64)> = values.iter().map(|&x| (x, x, count)).collect();
        for &x in values {
            for &y in values {
                numerator.push((-x, y, 1.0));
            }
        }
        if !root {
            return nearest_value(&numerator, scale, got, gaps, even);
        }
        let (down, up) = gaps;
        let sign = |extra: [(f64, f64, f64); 3], negate: bool| {
            let all: Vec<(f64, f64, f64)> = numerator
                .iter()
                .map(|&(x, y, z)| if negate { (-x, y, z) } else { (x, y, z) })
                .chain(extra)
                .collect();
            let (negative, limbs, _) = Total::of_terms(&all).difference();
            match (limbs.iter().all(|&limb| limb == 0), negative) {
                (true, _) => 0,
                (false, true) => -1,
                (false, false) => 1,
            }
        };
        let low = if got == 0.0 {
            1
        } else {
            sign(
                [
                    (-got, got, scale),
                    (got, down, scale),
                    (-down, down, scale * 0.25),
                ],
                false,
            )
        };
        let high = sign(
            [(got, got, scale), (got, up, scale), (up, up, scale * 0.25)],
            true,
        );
        let inside = |side: i32| side > 0 || (side == 0 && even);
        inside(low) && inside(high)
    }

    #[test]
    fn variances_and_their_roots_round_once() {
        let mut next = generator(0x1f83_d9ab_fb41_bd6b);
        let mut draw = move |spread: u64| {
            let unit = (next() >> 11) as f64 / (1u64 << 53) as f64 - 0.5;
            let exponent = (next() % (2 * spread + 1)) as i32 - spread as i32;
            unit * 2f64.powi(exponent)
        };
        for round in 0..800 {
            let len = 2 + round % 70;
            let mut values: Vec<f64> = (0..len).map(|_| draw([2, 20, 200][round % 3])).collect();
            match round % 5 {
                // Offset data, nearly constant data, and whole numbers.
                0 => values.iter_mut().for_each(|v| *v += 1e6),
                1 => values.iter_mut().for_each(|v| *v = 1.0 + *v * 1e-12),
                2 => values.iter_mut().for_each(|v| *v = (*v * 64.0).round()),
                _ => {}
            }
            for (ddof, root) in [(0, false), (1, false), (0, true), (1, true)] {
                let got: f64 = variance_of(&values, ddof, root);
                let even = got.to_bits() & 1 == 0;
                assert!(
                    nearest_variance(&values, ddof, root, got, gaps_wide(got), even),
                    "round {round} ddof {ddof} root {root}: {got:e}"
                );
                let narrow: Vec<f32> = values.iter().map(|&v| v as f32).collect();
                let got: f32 = variance_of(&narrow, ddof, root);
                if got.is_finite() {
                    let wide: Vec<f64> = narrow.iter().map(|&v| f64::from(v)).collect();
                    let even = got.to_bits() & 1 == 0;
                    assert!(
                        nearest_variance(&wide, ddof, root, f64::from(got), gaps_narrow(got), even),
                        "round {round} ddof {ddof} root {root}: {got:e} in float32"
                    );
                }
            }
        }
        // Constant slices, one value, too few for the correction, and specials.
        assert_eq!(variance_of::<f64, f64>(&[2.5, 2.5, 2.5], 1, false), 0.0);
        assert_eq!(variance_of::<f64, f64>(&[-0.0, 0.0], 0, true), 0.0);
        assert_eq!(variance_of::<f64, f64>(&[7.0], 0, false), 0.0);
        assert!(variance_of::<f64, f64>(&[7.0], 1, false).is_nan());
        assert!(variance_of::<f64, f64>(&[], 0, true).is_nan());
        assert!(variance_of::<f64, f64>(&[1.0, f64::INFINITY], 1, false).is_nan());
        // Squares past the largest float64, where the variance is not.
        let big: f64 = variance_of(&[1e200, -1e200], 1, true);
        let want = (2f64).sqrt() * 1e200;
        assert!((big - want).abs() <= want * 2f64.powi(-52), "{big:e}");
        assert_eq!(variance_of::<f64, f64>(&[3e-200, -1e-200], 0, true), 2e-200);
    }

    #[test]
    fn column_variances_match_each_column_alone() {
        let mut next = generator(0x5be0_cd19_137e_2179);
        let mut draw = move |spread: u64| {
            let unit = (next() >> 11) as f64 / (1u64 << 53) as f64 - 0.5;
            let exponent = (next() % (2 * spread + 1)) as i32 - spread as i32;
            unit * 2f64.powi(exponent)
        };
        // One slab cut into parts down its rows, and many slabs whole.
        for (outer, len, inner) in [(1, 300, 40), (70, 5, 9), (3, 2, 33), (2, 1000, 17)] {
            for kind in 0..4 {
                let mut values: Vec<f64> = (0..outer * len * inner)
                    .map(|_| draw([2, 30, 300][kind % 3]))
                    .collect();
                for (index, value) in values.iter_mut().enumerate() {
                    let column = index % inner;
                    match (kind, column % 5) {
                        // Constant columns, offset ones, and specials.
                        (_, 0) => *value = column as f64 * 0.25,
                        (1, 1) => *value += 1e9,
                        (2, 2) if index / inner % len == len / 2 => *value = f64::NAN,
                        (3, 2) if index / inner % len == 0 => *value = f64::INFINITY,
                        _ => {}
                    }
                }
                let narrow: Vec<f32> = values.iter().map(|&v| v as f32).collect();
                let column = |slab: usize, index: usize| -> Vec<f64> {
                    (0..len)
                        .map(|row| values[slab * len * inner + row * inner + index])
                        .collect()
                };
                for (ddof, root) in [(0, false), (1, false), (0, true), (1, true)] {
                    let wide: Vec<f64> =
                        column_variances(&values, outer, len, inner, ddof, root, false);
                    let thin: Vec<f32> =
                        column_variances(&narrow, outer, len, inner, ddof, root, false);
                    for slab in 0..outer {
                        for index in 0..inner {
                            let alone = column(slab, index);
                            let want: f64 = variance_of(&alone, ddof, root);
                            let got = wide[slab * inner + index];
                            assert!(
                                got.to_bits() == want.to_bits() || (got.is_nan() && want.is_nan()),
                                "{outer}x{len}x{inner} kind {kind} ({slab}, {index}): {got:e} {want:e}"
                            );
                            let alone: Vec<f32> = alone.iter().map(|&v| v as f32).collect();
                            let want: f32 = variance_of(&alone, ddof, root);
                            let got = thin[slab * inner + index];
                            assert!(
                                got.to_bits() == want.to_bits() || (got.is_nan() && want.is_nan()),
                                "{outer}x{len}x{inner} kind {kind} ({slab}, {index}): {got:e} {want:e} in float32"
                            );
                        }
                    }
                }
            }
        }
        // Too few values for the correction, and no values at all.
        let none: Vec<f64> = column_variances(&[1.0, 2.0], 1, 1, 2, 1, false, false);
        assert!(none.iter().all(|value| value.is_nan()));
        let empty: Vec<f64> = column_variances::<f64, f64>(&[], 2, 0, 3, 0, true, false);
        assert_eq!(empty.len(), 6);
        assert!(empty.iter().all(|value| value.is_nan()));
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
