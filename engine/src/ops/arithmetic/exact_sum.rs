// Copyright (c) Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

//! Float64 sums and means rounded once.
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

use super::close::add_shifted;
use super::rounded_once::{Exact, product_error, rounded, two_sum};
use crate::ops::util::pairwise_fold;

/// Lanes a run is added in, and values per block a task takes.
const LANES: usize = 8;
const BLOCK: usize = 8192;

/// Just under one half; see `evenly_spaced`.
const SHY_OF_HALF: f64 = 0.5 - 1.0 / (1u64 << 40) as f64;

/// A float64 total carried as `sum + error`, within `slack` of the exact
/// total.
#[derive(Clone, Copy, Debug, Default)]
struct Carried {
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

    /// The float64 nearest the exact total, when the bound says which it is.
    fn rounded(self) -> Option<f64> {
        let (value, rest) = two_sum(self.sum, self.error);
        if self.slack == 0.0 && rest == 0.0 {
            // Every addition was exact.
            return value.is_finite().then_some(value);
        }
        // The gap to the float64 on the side towards zero, the smaller of the
        // two; a zero or a non-finite value leaves it NaN and the test false.
        let inward = (value - f64::from_bits(value.to_bits().wrapping_sub(1))).abs();
        (rest.abs() + self.slack < inward * SHY_OF_HALF && value != 0.0).then_some(value)
    }
}

/// One block's values, added in [`LANES`] lanes with each addition's exact
/// error carried beside it.
///
/// A lane's errors, added in float64 over `m` additions, are within
/// `(m - 1)u / (1 - (m - 1)u)` of the sum of their magnitudes, which the lane
/// adds up as well; `2^-52 · m` is twice that.
#[inline(always)]
fn block_lanes(values: &[f64]) -> Carried {
    let (mut sums, mut errors, mut magnitudes) = ([0f64; LANES], [0f64; LANES], [0f64; LANES]);
    let (chunks, rest) = values.as_chunks::<LANES>();
    for chunk in chunks {
        for lane in 0..LANES {
            let (sum, error) = two_sum(sums[lane], chunk[lane]);
            sums[lane] = sum;
            errors[lane] += error;
            magnitudes[lane] += error.abs();
        }
    }
    finish(sums, errors, magnitudes, rest, values.len())
}

/// A block's lanes, with the values left over added to lane 0, merged into
/// one carried total.
#[inline(always)]
fn finish(
    mut sums: [f64; LANES],
    mut errors: [f64; LANES],
    mut magnitudes: [f64; LANES],
    rest: &[f64],
    len: usize,
) -> Carried {
    // The few left over go to lane 0.
    for &value in rest {
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
    use std::arch::x86_64::{
        __m256d, _mm256_add_pd, _mm256_andnot_pd, _mm256_loadu_pd, _mm256_set1_pd,
        _mm256_setzero_pd, _mm256_storeu_pd, _mm256_sub_pd,
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
            // SAFETY: four values from `4 * half` lie inside the chunk.
            let value: __m256d = unsafe { _mm256_loadu_pd(chunk.as_ptr().add(4 * half)) };
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
        rest,
        values.len(),
    )
}

/// [`block_lanes`] in the widest build the CPU runs.
fn block(values: &[f64]) -> Carried {
    #[cfg(target_arch = "x86_64")]
    if crate::ops::simd::simd_capabilities().avx {
        // SAFETY: avx was detected on this CPU.
        return unsafe { block_avx(values) };
    }
    block_lanes(values)
}

/// The carried total of `values`, block by block, the blocks merged
/// pairwise: in parallel for a long run, with the same blocks either way.
fn carried(values: &[f64]) -> Carried {
    if values.len() <= BLOCK {
        return block(values);
    }
    let partials: Vec<Carried> = if values.len() >= crate::ops::map::PAR_THRESHOLD {
        let len = values.len();
        crate::parallel::map_tasks(len.div_ceil(BLOCK), &|nth| {
            block(&values[nth * BLOCK..(nth * BLOCK + BLOCK).min(len)])
        })
    } else {
        values.chunks(BLOCK).map(block).collect()
    };
    pairwise_fold(partials, Carried::default(), Carried::merge)
}

/// The float64 nearest the sum of `values`.
///
/// A NaN, or infinities of both signs, sum to NaN and an infinity of one sign
/// to that infinity; finite values to the float64 nearest their exact sum,
/// an infinity past the largest. A sum that cancels to zero is `+0`.
pub(crate) fn sum_f64(values: &[f64]) -> f64 {
    carried(values).rounded().unwrap_or_else(|| {
        Total::of(values).map_or_else(|special| special, |total| total.rounded())
    })
}

/// The float64 nearest the mean of `values`, the exact sum over the count;
/// NaN for no values.
///
/// The carried total is divided as `evenly_spaced` divides a numerator: the
/// quotient, its exact remainder, and the tail with the error added in.
pub(crate) fn mean_f64(values: &[f64]) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    let count = values.len() as f64;
    let total = carried(values);
    if total.sum == 0.0 && total.error == 0.0 && total.slack == 0.0 {
        // The values cancel exactly, or are all zeros.
        return 0.0;
    }
    let quotient = total.sum / count;
    let product = quotient * count;
    // `product` is within a rounding of `sum`, so the first difference is
    // exact and the remainder rounds once.
    let remainder = (total.sum - product) - product_error(quotient, count, product);
    let tail = (remainder + total.error) / count;
    let (value, rest) = two_sum(quotient, tail);
    let bound = total.slack / count * (1.0 + 2f64.powi(-50)) + tail.abs() * 2f64.powi(-50);
    let inward = (value - f64::from_bits(value.to_bits().wrapping_sub(1))).abs();
    // Clear of underflow in the product's split, and of a zero or a value
    // that is not finite.
    let ordinary = value.abs() >= 2f64.powi(-900) && value.abs() <= 2f64.powi(900);
    if ordinary && rest.abs() + bound < inward * SHY_OF_HALF {
        return value;
    }
    match Total::of(values) {
        Err(special) => special,
        Ok(total) => total.divided(values.len() as u64),
    }
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

/// The exact sum of finite float64s, in base-2^64 limbs from `2^-1074` up,
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
    fn of(values: &[f64]) -> Result<Self, f64> {
        let (mut nan, mut up, mut down) = (false, false, false);
        for &value in values {
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
        for &value in values {
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

    /// The float64 nearest the sum; `+0` for a sum that cancels.
    fn rounded(self) -> f64 {
        let (negative, limbs) = self.difference();
        rounded_at(negative, &limbs, false, LEAST)
    }

    /// The float64 nearest the sum over `count`, by long division with a
    /// limb of fraction below the least subnormal, so that a remainder left
    /// over is far below where the quotient rounds.
    fn divided(self, count: u64) -> f64 {
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
fn rounded_at(negative: bool, limbs: &[u64], inexact: bool, least: i32) -> f64 {
    let Some(top) = limbs.iter().rposition(|&limb| limb != 0) else {
        // Nothing, or less than `2^least`, which is far below the least
        // subnormal.
        return if negative { -0.0 } else { 0.0 };
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
            let expected = Total::of(&values).unwrap().divided(values.len() as u64);
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
    fn means_divide_the_exact_sum_once() {
        assert_eq!(mean_f64(&[0.1, 0.2, 0.3]), 0.2);
        assert_eq!(mean_f64(&[f64::MAX, f64::MAX]), f64::MAX);
        assert!(mean_f64(&[]).is_nan());
        let third = mean_f64(&[1.0, 0.0, 0.0]);
        assert_eq!(third, 1.0 / 3.0);
    }
}
