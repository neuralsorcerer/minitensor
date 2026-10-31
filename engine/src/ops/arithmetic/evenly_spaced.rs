// Copyright (c) Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

//! Evenly spaced points, each the exact one rounded once.
//!
//! Point `i` of the `n + 1` from `start` to `end` is
//! `(start · (n - i) + end · i) / n`. Spaced as `start + i * step`, with the
//! step rounded first, it rounded three times: `linspace(0, 1, 11)` had
//! `0.30000000000000004` for `0.3`, and a third of the points from `1e-3` to
//! `1e3` were an ulp or more off. Rounded once, the points are as even as
//! the dtype allows, never out of order, symmetric wherever the exact points
//! are -- the middle of `-1` to `1` is a zero -- and the two ends are the
//! given ones.
//!
//! Nearly every point is settled in float64 arithmetic: the two products
//! and their errors are exact, and the quotient is carried a float64 further,
//! with a bound on what that leaves out. A point the bound cannot place
//! against a halfway point is computed exactly in integer arithmetic instead.

use std::mem::MaybeUninit;

use super::rounded_once::{Exact, Target, product_error, rounded, sum_wide, two_sum};
use crate::{
    device::Device,
    ops::map::{PAR_CHUNK, build_vec, par_out_chunks},
    tensor::{DataType, TensorData},
};

/// Below this nothing in [`Spacing::near`] underflows, and above it nothing
/// overflows or splits wrongly.
const TINY: f64 = f64::from_bits((1023 - 800) << 52);
const HUGE: f64 = f64::from_bits((1023 + 900) << 52);

/// Just under one half: a point closer than this to halfway between two
/// floats is left to the exact path, which keeps the comparison clear of
/// its own rounding.
const SHY_OF_HALF: f64 = 0.5 - 1.0 / (1u64 << 40) as f64;

/// A point in float64, as [`Spacing::near`] finds it.
#[derive(Clone, Copy)]
struct Near {
    value: f64,
    /// Whether `value` is the point rounded to the nearest float64; when it
    /// is not, `value` is not to be used.
    settled: bool,
    /// Whether `value` is the point itself, which a halfway point between two
    /// narrower values can be.
    exact: bool,
}

/// What the points of one spacing share, when float64 arithmetic can settle
/// them: finite ends of ordinary size, and fewer than 2^52 intervals.
#[derive(Clone, Copy)]
struct Spacing {
    start: f64,
    end: f64,
    count: f64,
    inverse: f64,
}

impl Spacing {
    fn new(start: f64, end: f64, intervals: usize) -> Option<Self> {
        let ordinary = |value: f64| value == 0.0 || (TINY..=HUGE).contains(&value.abs());
        let fits = (2..1 << 52).contains(&(intervals as u64));
        (fits && ordinary(start) && ordinary(end)).then(|| Self {
            start,
            end,
            count: intervals as f64,
            inverse: 1.0 / intervals as f64,
        })
    }

    /// Point `index` rounded to the nearest float64, and whether it surely
    /// is.
    ///
    /// The numerator `start · (count - index) + end · index` is four exact
    /// floats: two products and their errors. Its leading float times the
    /// reciprocal is a quotient, and what that quotient leaves of the
    /// numerator, with the three small floats added in, is divided out as a
    /// tail. Quotient and tail are within `4.02u · |tail| + 5.01u² · M /
    /// count` of the point, for the unit roundoff `u` and the products'
    /// magnitude `M`, and `bound` doubles and triples those. The value is the
    /// point itself when the numerator is its leading float alone and the
    /// value times the count is exactly that. No branch is taken, so a block
    /// of points vectorizes.
    #[inline(always)]
    fn near<const FUSED: bool>(self, index: usize) -> Near {
        let right = whole(index);
        let left = self.count - right;
        let (lower, lower_error) = exact_product::<FUSED>(self.start, left);
        let (upper, upper_error) = exact_product::<FUSED>(self.end, right);
        let (lead, carry) = two_sum(lower, upper);
        let rest = (lower_error + upper_error) + carry;
        let quotient = lead * self.inverse;
        let (product, product_error) = exact_product::<FUSED>(quotient, self.count);
        // `product` is within a few roundings of `lead`, so the first
        // difference is exact and the remainder rounds once.
        let remainder = (lead - product) - product_error;
        let tail = (remainder + rest) * self.inverse;
        let (value, error) = two_sum(quotient, tail);
        let bound = (lower.abs() + upper.abs()) * self.inverse * 2f64.powi(-102)
            + tail.abs() * 2f64.powi(-50);
        // The gap to the float64 on the side towards zero, the smaller of the
        // two when they differ. A numerator cancelled almost to nothing
        // leaves the bound nothing to measure against.
        let inward = (value - f64::from_bits(value.to_bits().wrapping_sub(1))).abs();
        let ordinary = lead.abs() >= TINY;
        let (whole, whole_error) = exact_product::<FUSED>(value, self.count);
        let alone = (lower_error == 0.0) & (upper_error == 0.0) & (carry == 0.0);
        let exact = ordinary & alone & (whole == lead) & (whole_error == 0.0);
        let settled = ordinary & (error.abs() + bound < inward * SHY_OF_HALF);
        Near {
            value,
            settled: settled | exact,
            exact,
        }
    }
}

/// `index` as a float64, for an index below 2^52: its bits laid into the
/// significand of 2^52, which then comes off exactly.
#[inline(always)]
fn whole(index: usize) -> f64 {
    const TWO_TO_52: f64 = 4_503_599_627_370_496.0;
    f64::from_bits(index as u64 | TWO_TO_52.to_bits()) - TWO_TO_52
}

/// `x · y` and its exact error: a fused multiply-add in the `fma` build,
/// split halves without it.
#[inline(always)]
fn exact_product<const FUSED: bool>(x: f64, y: f64) -> (f64, f64) {
    let product = x * y;
    let error = if FUSED {
        x.mul_add(y, -product)
    } else {
        product_error(x, y, product)
    };
    (product, error)
}

/// Point `index` of `intervals + 1` evenly spaced from `start` to `end`,
/// rounded to the nearest float64; point `0` is `start` and point
/// `intervals` is `end`.
///
/// An infinite or NaN end leaves nothing to round: the weighted ends are
/// added as they are, so a NaN stays one, an infinity reaches every point
/// past `start`, and opposite infinities meet in a NaN.
pub fn evenly_spaced_point(start: f64, end: f64, index: usize, intervals: usize) -> f64 {
    if index == 0 {
        return start;
    }
    if index >= intervals {
        return end;
    }
    if !(start.is_finite() && end.is_finite()) {
        let weights = ((intervals - index) as f64, index as f64);
        return (start * weights.0 + end * weights.1) / intervals as f64;
    }
    match Spacing::new(start, end, intervals).map(|spacing| spacing.near::<false>(index)) {
        Some(near) if near.settled => near.value,
        _ => exactly(start, end, index, intervals),
    }
}

/// The point rounded once to `T`, from the exact numerator: for finite ends
/// and an interior index.
///
/// The numerator, rounded to odd at 125 bits or more, is moved up to bit
/// 126 and divided. The quotient has at least 63 bits, and with whether the
/// division left a remainder marked in its last, it is on the same side of
/// every halfway point between two `T`s as the point: those are 2^9 of its
/// units apart or more, so `intervals` times one is even in the numerator's
/// own units, and none lies between the numerator and its odd rounding.
#[cold]
#[inline(never)]
fn exactly<T: Target>(start: f64, end: f64, index: usize, intervals: usize) -> T {
    let (negative, magnitude, exponent) = numerator(start, end, index, intervals);
    if magnitude == 0 {
        return T::assemble(negative, 0, 0);
    }
    let shift = magnitude.leading_zeros() - 1;
    let lifted = magnitude << shift;
    let divisor = intervals as u128;
    let quotient = lifted / divisor;
    let inexact = quotient * divisor != lifted;
    rounded(
        negative,
        quotient | u128::from(inexact),
        exponent - shift as i32,
    )
}

/// `start · (intervals - index) + end · index` as `±magnitude · 2^exponent`,
/// exact or rounded to odd at 125 bits or more: see [`sum_wide`].
///
/// A zero is `+0`, as a sum that cancels is, unless both ends are `-0`.
fn numerator(start: f64, end: f64, index: usize, intervals: usize) -> (bool, u128, i32) {
    let weighted = |value: f64, weight: usize| {
        let value = Exact::float(value).expect("the ends are finite");
        (
            value.negative,
            u128::from(value.magnitude) * weight as u128,
            value.exponent,
        )
    };
    let lower = weighted(start, intervals - index);
    let upper = weighted(end, index);
    match (lower.1, upper.1) {
        (0, 0) => (lower.0 && upper.0, 0, 0),
        (0, _) => upper,
        (_, 0) => lower,
        _ => sum_wide(lower, upper).unwrap_or((false, 0, 0)),
    }
}

/// Whether an index is strictly between the ends of finite ones: the points
/// the exact paths are for.
fn interior(start: f64, end: f64, index: usize, intervals: usize) -> bool {
    index != 0 && index < intervals && start.is_finite() && end.is_finite()
}

/// The point rounded once to float32.
///
/// Every halfway point between two float32s is a float64, so the float64
/// point is on the same side of each as the exact one, or on it: only then
/// does narrowing round a second time, and only then is the exact one taken.
fn point_f32(start: f64, end: f64, index: usize, intervals: usize) -> f32 {
    let wide = evenly_spaced_point(start, end, index, intervals);
    let narrow = wide as f32;
    if interior(start, end, index, intervals) && on_halfway(wide, narrow) {
        exactly(start, end, index, intervals)
    } else {
        narrow
    }
}

/// Whether `wide`, which rounds to `narrow`, is halfway between two
/// float32s; an overflow to infinity counts, to be settled exactly.
#[inline(always)]
fn on_halfway(wide: f64, narrow: f32) -> bool {
    let back = f64::from(narrow);
    // The neighbour on `wide`'s side; a `wide` that is `narrow` itself is
    // never the midpoint, whichever neighbour is taken.
    let outward = (wide > back) != narrow.is_sign_negative();
    let bits = narrow.to_bits();
    let other = f32::from_bits(if outward {
        bits.wrapping_add(1)
    } else {
        bits.wrapping_sub(1)
    });
    (wide == (back + f64::from(other)) / 2.0) | narrow.is_infinite()
}

/// `wide` rounded to the nearest integer, a half away from zero, and whether
/// that is surely the exact point's: below 2^52, where every half-integer is
/// a float64 and so `wide` is on the same side of each as the exact point,
/// and only when it is not one itself -- unless `exact` says the point is
/// `wide`.
#[inline(always)]
fn nearest_integer(wide: f64, exact: bool) -> (i64, bool) {
    let truncated = wide as i64;
    let fraction = wide - truncated as f64;
    // Wrapping, since a saturated `truncated` is never settled.
    let rounded = truncated
        .wrapping_add(i64::from(fraction >= 0.5))
        .wrapping_sub(i64::from(fraction <= -0.5));
    let settled = (wide.abs() < 2f64.powi(52)) & (exact | (fraction.abs() != 0.5));
    (rounded, settled)
}

/// The point rounded to the nearest integer, a half away from zero, and
/// saturated to the int64 range; a NaN is zero.
fn point_integer(start: f64, end: f64, index: usize, intervals: usize) -> i64 {
    let wide = evenly_spaced_point(start, end, index, intervals);
    let (rounded, settled) = nearest_integer(wide, false);
    if settled {
        rounded
    } else if !interior(start, end, index, intervals) || intervals as u64 >= 1 << 53 {
        wide.round() as i64
    } else {
        integer_exactly(start, end, index, intervals)
    }
}

/// The point rounded to the nearest integer, a half away from zero, from
/// the exact numerator `N`, for fewer than 2^53 intervals.
///
/// `8N` rounded to odd at unit 1 and divided by `2 · intervals` is `4 ·
/// point` rounded to odd at unit 1 -- its multiples of `2 · intervals` are
/// even, so none lies between `8N` and its odd rounding -- and that tells a
/// quarter from a half from three quarters, or says the point is a whole or
/// half integer exactly.
#[cold]
#[inline(never)]
fn integer_exactly(start: f64, end: f64, index: usize, intervals: usize) -> i64 {
    let (negative, magnitude, exponent) = numerator(start, end, index, intervals);
    if magnitude == 0 {
        return 0;
    }
    // At 2^63 or past it the point saturates whichever way it rounds. Short
    // of that the numerator is below 2^117, and so exact wherever `8N` keeps
    // whole units: a rounded one has 125 bits.
    let leading = 128 - magnitude.leading_zeros() as i32 + exponent;
    let divisor_bits = 128 - (intervals as u128).leading_zeros() as i32;
    if leading - divisor_bits >= 64 {
        return if negative { i64::MIN } else { i64::MAX };
    }
    let unit = exponent + 3;
    let scaled = if unit >= 0 {
        magnitude << unit
    } else if unit <= -128 {
        1
    } else {
        let kept = magnitude >> -unit;
        kept | u128::from(kept << -unit != magnitude)
    };
    let divisor = 2 * intervals as u128;
    let quotient = scaled / divisor;
    let quarters = quotient | u128::from(quotient * divisor != scaled);
    let whole = (quarters >> 2) + ((quarters >> 1) & 1);
    let signed = if negative {
        -(whole as i128)
    } else {
        whole as i128
    };
    signed.clamp(i64::MIN.into(), i64::MAX.into()) as i64
}

/// Whether the point is nonzero: a float64 point of zero may be one too
/// small to hold.
fn point_nonzero(start: f64, end: f64, index: usize, intervals: usize) -> bool {
    let wide = evenly_spaced_point(start, end, index, intervals);
    if wide == 0.0 && interior(start, end, index, intervals) {
        return numerator(start, end, index, intervals).1 != 0;
    }
    wide != 0.0
}

/// Points per pass of [`fill`]: the first pass writes them all and marks
/// the ones it could not settle, the second settles those.
const BLOCK: usize = 256;

/// Writes points `first..` into `out`: `finish` takes each point found in
/// float64 to the element and says whether that is surely right, and `slow`
/// gives an element the long way.
#[inline(always)]
fn fill<U, F, S, const FUSED: bool>(
    spacing: Spacing,
    first: usize,
    out: &mut [MaybeUninit<U>],
    finish: F,
    slow: S,
) where
    F: Fn(Near) -> (U, bool),
    S: Fn(usize) -> U,
{
    for (block_index, block) in out.chunks_mut(BLOCK).enumerate() {
        let base = first + block_index * BLOCK;
        let mut unsettled = [false; BLOCK];
        let mut pending = false;
        for (offset, (slot, mark)) in block.iter_mut().zip(&mut unsettled).enumerate() {
            let (value, settled) = finish(spacing.near::<FUSED>(base + offset));
            slot.write(value);
            *mark = !settled;
            pending |= !settled;
        }
        if pending {
            for (offset, (slot, &mark)) in block.iter_mut().zip(&unsettled).enumerate() {
                if mark {
                    slot.write(slow(base + offset));
                }
            }
        }
    }
}

/// [`fill`] compiled with fused multiply-adds, for a CPU that has them.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
fn fill_fma<U, F, S>(spacing: Spacing, first: usize, out: &mut [MaybeUninit<U>], finish: F, slow: S)
where
    F: Fn(Near) -> (U, bool),
    S: Fn(usize) -> U,
{
    fill::<U, F, S, true>(spacing, first, out, finish, slow)
}

/// `steps` elements, point by point: settled in float64 blocks where the
/// spacing allows, and by `slow` otherwise.
fn build<U, F, S>(
    steps: usize,
    dtype: DataType,
    device: Device,
    spacing: Option<Spacing>,
    finish: F,
    slow: S,
) -> TensorData
where
    U: Copy + Send + 'static,
    F: Fn(Near) -> (U, bool) + Copy + Sync,
    S: Fn(usize) -> U + Copy + Sync,
{
    let chunk_of = |first: usize, chunk: &mut [MaybeUninit<U>]| {
        let Some(spacing) = spacing else {
            for (offset, slot) in chunk.iter_mut().enumerate() {
                slot.write(slow(first + offset));
            }
            return;
        };
        #[cfg(target_arch = "x86_64")]
        if crate::ops::simd::simd_capabilities().avx2_fma {
            // SAFETY: avx2 and fma were detected on this CPU.
            return unsafe { fill_fma(spacing, first, chunk, finish, slow) };
        }
        fill::<U, F, S, false>(spacing, first, chunk, finish, slow)
    };
    // SAFETY: the chunks tile the buffer and each writes every one of its
    // elements, so all `steps` are initialized.
    let data: Vec<U> =
        unsafe { build_vec::<U, _>(steps, |spare| par_out_chunks(spare, PAR_CHUNK, &chunk_of)) };
    TensorData::from_vec(data, dtype, device)
}

/// `steps` points evenly spaced from `start` to `end` inclusive, each the
/// exact point rounded once to `dtype`: to the nearest integer, a half away
/// from zero and saturated, for an integer one, and to whether it is nonzero
/// for a bool.
pub fn evenly_spaced_data(
    start: f64,
    end: f64,
    steps: usize,
    dtype: DataType,
    device: Device,
) -> TensorData {
    let intervals = steps.saturating_sub(1);
    let spacing = Spacing::new(start, end, intervals);
    match dtype {
        DataType::Float32 => build(
            steps,
            dtype,
            device,
            spacing,
            |near| {
                // A halfway point that is the point itself is a tie, which
                // the conversion rounds to even.
                let narrow = near.value as f32;
                let twice = !near.exact & on_halfway(near.value, narrow);
                (narrow, near.settled & !twice)
            },
            |index| point_f32(start, end, index, intervals),
        ),
        DataType::Float64 => build(
            steps,
            dtype,
            device,
            spacing,
            |near| (near.value, near.settled),
            |index| evenly_spaced_point(start, end, index, intervals),
        ),
        DataType::Int32 => {
            let narrow = |point: i64| point.clamp(i32::MIN.into(), i32::MAX.into()) as i32;
            build(
                steps,
                dtype,
                device,
                spacing,
                |near| {
                    let (point, settled) = nearest_integer(near.value, near.exact);
                    (narrow(point), near.settled & settled)
                },
                |index| narrow(point_integer(start, end, index, intervals)),
            )
        }
        DataType::Int64 => build(
            steps,
            dtype,
            device,
            spacing,
            |near| {
                let (point, settled) = nearest_integer(near.value, near.exact);
                (point, near.settled & settled)
            },
            |index| point_integer(start, end, index, intervals),
        ),
        DataType::Bool => build(
            steps,
            dtype,
            device,
            spacing,
            |near| (near.value != 0.0, near.settled),
            |index| point_nonzero(start, end, index, intervals),
        ),
    }
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

    #[test]
    fn tenths_are_the_nearest_floats() {
        let points: Vec<f64> = (0..=10)
            .map(|i| evenly_spaced_point(0.0, 1.0, i, 10))
            .collect();
        let tenths = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
        assert_eq!(points, tenths);
        assert_eq!(evenly_spaced_point(-1.0, 1.0, 1, 2).to_bits(), 0);
        assert_eq!(
            evenly_spaced_point(-0.0, -0.0, 1, 2).to_bits(),
            (-0.0f64).to_bits()
        );
        // Ends past half the float range, whose difference overflows.
        assert_eq!(evenly_spaced_point(-f64::MAX, f64::MAX, 1, 2), 0.0);
        assert_eq!(
            evenly_spaced_point(-f64::MAX, f64::MAX, 3, 4),
            f64::MAX / 2.0
        );
        assert_eq!(evenly_spaced_point(0.0, f64::INFINITY, 0, 2), 0.0);
        assert_eq!(evenly_spaced_point(0.0, f64::INFINITY, 1, 2), f64::INFINITY);
        assert!(evenly_spaced_point(f64::NEG_INFINITY, f64::INFINITY, 1, 2).is_nan());
    }

    #[test]
    fn ties_round_to_even_and_half_away_for_integers() {
        // Halfway between 1 and the float64 after it.
        let next = f64::from_bits(1.0f64.to_bits() + 1);
        assert_eq!(evenly_spaced_point(1.0, next, 1, 2), 1.0);
        let after = f64::from_bits(next.to_bits() + 1);
        assert_eq!(evenly_spaced_point(next, after, 1, 2), after);
        let next32 = f64::from(f32::from_bits(1.0f32.to_bits() + 1));
        assert_eq!(point_f32(1.0, next32, 1, 2), 1.0);
        // Just past halfway, where the float64 point is halfway itself.
        let past = f64::from_bits(next32.to_bits() + 1);
        assert_eq!(point_f32(1.0, past, 1, 2), next32 as f32);
        assert_eq!(point_integer(0.0, 5.0, 1, 2), 3);
        assert_eq!(point_integer(-5.0, 0.0, 1, 2), -3);
        assert_eq!(point_integer(0.0, 10.0, 1, 4), 3);
        assert_eq!(point_integer(0.0, 10.0, 3, 4), 8);
        // Past 2^53, where the float64 point is no longer the integer.
        let big = 2f64.powi(62);
        assert_eq!(point_integer(0.0, big + 2048.0, 1, 3), (1 << 62) / 3 + 683);
        assert_eq!(point_integer(-big, big, 1, 2), 0);
        assert_eq!(point_integer(0.0, 1e300, 1, 2), i64::MAX);
        assert_eq!(point_integer(-1e300, 0.0, 1, 2), i64::MIN);
        assert!(point_nonzero(0.0, f64::from_bits(1), 1, 2));
        assert!(!point_nonzero(-1.0, 1.0, 1, 2));
    }

    #[test]
    fn the_float_path_agrees_with_the_exact_one() {
        let mut next = generator(0x510e_527f_ade6_82d1);
        let (mut checked, mut exacts) = (0, 0);
        for round in 0..400_000 {
            let start = f64::from_bits(next() >> (1 + next() % 2));
            let start = if next().is_multiple_of(2) {
                start
            } else {
                -start
            };
            // Ends of any two sizes, ends that nearly cancel, and ends a few
            // ulps apart.
            let end = match round % 3 {
                0 => f64::from_bits(next() >> (1 + next() % 2)),
                1 => -start * (1.0 + (next() % 1000) as f64 * f64::EPSILON),
                _ => {
                    let ulp = 2f64.powi(start.abs().max(1e-300).log2() as i32 - 52);
                    start + (next() % 64) as f64 * ulp
                }
            };
            if !(start.is_finite() && end.is_finite()) {
                continue;
            }
            let intervals = 2 + (next() % 2000) as usize;
            let index = 1 + (next() as usize) % (intervals - 1);
            let exact: f64 = exactly(start, end, index, intervals);
            if let Some(spacing) = Spacing::new(start, end, intervals) {
                for near in [spacing.near::<false>(index), spacing.near::<true>(index)] {
                    let at = format!("{start:e} to {end:e}, {index} of {intervals}");
                    if near.settled {
                        assert_eq!(near.value.to_bits(), exact.to_bits(), "{at}");
                        checked += 1;
                    }
                    if near.exact {
                        // The numerator is the value times the count, compared
                        // with trailing zeros taken off both.
                        let reduced = |(magnitude, exponent): (u128, i32)| {
                            let zeros = magnitude.trailing_zeros();
                            (magnitude >> zeros, exponent + zeros as i32)
                        };
                        let (negative, magnitude, exponent) =
                            numerator(start, end, index, intervals);
                        let value = Exact::float(near.value).unwrap();
                        let times = u128::from(value.magnitude) * intervals as u128;
                        assert_eq!(negative, value.negative, "{at}");
                        assert_eq!(
                            reduced((magnitude, exponent)),
                            reduced((times, value.exponent)),
                            "{at}"
                        );
                        exacts += 1;
                    }
                }
            }
            let narrow: f32 = exactly(start, end, index, intervals);
            assert_eq!(
                point_f32(start, end, index, intervals).to_bits(),
                narrow.to_bits()
            );
        }
        assert!(checked > 600_000, "{checked}");
        assert!(exacts > 300, "{exacts}");
    }

    #[test]
    fn the_blocks_give_each_point_as_it_is_given_alone() {
        let mut next = generator(0x9b05_688c_2b3e_6c1f);
        let ends = [
            (0.0, 1.0),
            (-1.0, 1.0),
            (1e-3, 1e3),
            (0.0, 10.0),
            (-0.0, -0.0),
            (1.0, 1.0 + f64::EPSILON),
            (-f64::MAX, f64::MAX),
            (0.0, f64::INFINITY),
            (2f64.powi(62), -2f64.powi(61)),
        ];
        for (start, end) in ends {
            for steps in [0, 1, 2, 3, 5, 257, 1000 + (next() % 3000) as usize] {
                let data = |dtype| evenly_spaced_data(start, end, steps, dtype, Device::cpu());
                let intervals = steps.saturating_sub(1);
                let float64 = data(DataType::Float64);
                let float32 = data(DataType::Float32);
                let int64 = data(DataType::Int64);
                let int32 = data(DataType::Int32);
                let bool = data(DataType::Bool);
                for index in 0..steps {
                    let point = evenly_spaced_point(start, end, index, intervals);
                    let at = format!("{start:e} to {end:e}, {index} of {steps}");
                    let float64 = float64.as_f64_slice().unwrap()[index];
                    assert_eq!(float64.to_bits(), point.to_bits(), "{at}");
                    let float32 = float32.as_f32_slice().unwrap()[index];
                    let narrow = point_f32(start, end, index, intervals);
                    assert_eq!(float32.to_bits(), narrow.to_bits(), "{at}");
                    let integer = point_integer(start, end, index, intervals);
                    assert_eq!(int64.as_i64_slice().unwrap()[index], integer, "{at}");
                    let clamped = integer.clamp(i32::MIN.into(), i32::MAX.into()) as i32;
                    assert_eq!(int32.as_i32_slice().unwrap()[index], clamped, "{at}");
                    let nonzero = point_nonzero(start, end, index, intervals);
                    assert_eq!(bool.as_bool_slice().unwrap()[index], nonzero, "{at}");
                }
            }
        }
    }
}
