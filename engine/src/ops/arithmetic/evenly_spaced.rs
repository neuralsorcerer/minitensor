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

use super::rounded_once::{Exact, Target, odd, product_error, rounded, sum_wide, two_sum};
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

/// A point in float64, as [`Points::near`] finds it.
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

/// A run of points float64 arithmetic can settle, one index at a time.
trait Points: Copy + Send + Sync {
    /// Point `index` rounded to the nearest float64, and whether it surely
    /// is. No branch is taken, so a block of points vectorizes.
    fn near<const FUSED: bool>(self, index: usize) -> Near;
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
}

impl Points for Spacing {
    /// The numerator `start · (count - index) + end · index` is four exact
    /// floats: two products and their errors. Its leading float times the
    /// reciprocal is a quotient, and what that quotient leaves of the
    /// numerator, with the three small floats added in, is divided out as a
    /// tail. Quotient and tail are within `4.02u · |tail| + 5.01u² · M /
    /// count` of the point, for the unit roundoff `u` and the products'
    /// magnitude `M`, and `bound` doubles and triples those. The value is the
    /// point itself when the numerator is its leading float alone and the
    /// value times the count is exactly that.
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

/// 1.5 · 2^52: added to a float64 below 2^51 in magnitude, it leaves the
/// nearest integer, ties to even, in the low bits of the sum.
const SHIFTER: f64 = 6_755_399_441_055_744.0;

/// `wide` rounded to the nearest integer, ties to even, as an integer and
/// as a float64, for `wide` below 2^51 in magnitude; anything at all
/// otherwise. No conversion instruction is needed, so a block vectorizes.
#[inline(always)]
fn nearest_even(wide: f64) -> (i64, f64) {
    let shifted = wide + SHIFTER;
    let integer = (shifted.to_bits() as i64).wrapping_sub(SHIFTER.to_bits() as i64);
    (integer, shifted - SHIFTER)
}

/// `wide` rounded to the nearest integer, a half away from zero, and whether
/// that is surely the exact point's: below 2^51, where every half-integer is
/// a float64 and so `wide` is on the same side of each as the exact point,
/// and only when it is not one itself -- unless `exact` says the point is
/// `wide`.
#[inline(always)]
fn nearest_integer(wide: f64, exact: bool) -> (i64, bool) {
    let (even, rounded) = nearest_even(wide);
    let fraction = wide - rounded;
    // A tie went to even; away from zero is one further out when that was
    // towards zero.
    let away =
        i64::from((fraction == 0.5) & (wide > 0.0)) - i64::from((fraction == -0.5) & (wide < 0.0));
    let settled = (wide.abs() < 2f64.powi(51)) & (exact | (fraction.abs() != 0.5));
    (even.wrapping_add(away), settled)
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
fn fill<P, U, F, S, const FUSED: bool>(
    points: P,
    first: usize,
    out: &mut [MaybeUninit<U>],
    finish: F,
    slow: S,
) where
    P: Points,
    F: Fn(Near) -> (U, bool),
    S: Fn(usize) -> U,
{
    for (block_index, block) in out.chunks_mut(BLOCK).enumerate() {
        let base = first + block_index * BLOCK;
        let mut unsettled = [false; BLOCK];
        let mut pending = false;
        for (offset, (slot, mark)) in block.iter_mut().zip(&mut unsettled).enumerate() {
            let (value, settled) = finish(points.near::<FUSED>(base + offset));
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
fn fill_fma<P, U, F, S>(points: P, first: usize, out: &mut [MaybeUninit<U>], finish: F, slow: S)
where
    P: Points,
    F: Fn(Near) -> (U, bool),
    S: Fn(usize) -> U,
{
    fill::<P, U, F, S, true>(points, first, out, finish, slow)
}

/// `steps` elements, point by point: settled in float64 blocks where
/// `points` allows, and by `slow` otherwise.
fn build<P, U, F, S>(
    steps: usize,
    dtype: DataType,
    device: Device,
    points: Option<P>,
    finish: F,
    slow: S,
) -> TensorData
where
    P: Points,
    U: Copy + Send + 'static,
    F: Fn(Near) -> (U, bool) + Copy + Sync,
    S: Fn(usize) -> U + Copy + Sync,
{
    let chunk_of = |first: usize, chunk: &mut [MaybeUninit<U>]| {
        let Some(points) = points else {
            for (offset, slot) in chunk.iter_mut().enumerate() {
                slot.write(slow(first + offset));
            }
            return;
        };
        #[cfg(target_arch = "x86_64")]
        if crate::ops::simd::simd_capabilities().avx2_fma {
            // SAFETY: avx2 and fma were detected on this CPU.
            return unsafe { fill_fma(points, first, chunk, finish, slow) };
        }
        fill::<P, U, F, S, false>(points, first, chunk, finish, slow)
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

/// What the points of one `arange` share, when float64 arithmetic settles
/// them: a finite start and step of ordinary size, and at most 2^52 points.
#[derive(Clone, Copy)]
struct Stepping {
    start: f64,
    step: f64,
}

impl Stepping {
    fn new(start: f64, step: f64, count: usize) -> Option<Self> {
        let fits = count as u64 <= 1 << 52;
        let ordinary = start.abs() <= HUGE && (TINY..=HUGE).contains(&step.abs());
        (fits && ordinary).then_some(Self { start, step })
    }
}

impl Points for Stepping {
    /// `start + index · step` is three floats exactly: the start, and the
    /// product and its error. Two-sums take that apart into a leading float
    /// and two smaller ones, whose sum rounded to odd keeps, two places below
    /// the leading float's last bit, whether anything lies beyond it -- so
    /// adding it rounds as the exact sum does. It is the exact sum when the
    /// two smaller floats are both zero. Point `0` is the start itself, a
    /// `-0.0` included.
    #[inline(always)]
    fn near<const FUSED: bool>(self, index: usize) -> Near {
        let (product, product_error) = exact_product::<FUSED>(whole(index), self.step);
        let (total, total_error) = two_sum(self.start, product);
        let (rest, rest_error) = two_sum(total_error, product_error);
        let first = index == 0;
        let value = if first {
            self.start
        } else {
            total + odd(rest, rest_error)
        };
        let exact = first | ((rest == 0.0) & (rest_error == 0.0));
        Near {
            value,
            settled: true,
            exact,
        }
    }
}

/// Value `index` of an `arange` from `start` by `step`: `start + index ·
/// step` rounded once to the nearest float64, and `start` itself for index
/// `0`.
///
/// Computed as `start + index * step` it rounded twice, and a fifth of the
/// values of an ordinary `arange` were an ulp off the exact one rounded.
pub fn stepped_point(start: f64, step: f64, index: usize) -> f64 {
    if index == 0 {
        return start;
    }
    if !(start.is_finite() && step.is_finite()) {
        return start + index as f64 * step;
    }
    match Stepping::new(start, step, index + 1) {
        Some(stepping) => stepping.near::<false>(index).value,
        None => stepped_exactly(start, step, index),
    }
}

/// `start + index · step` as `±magnitude · 2^exponent`, exact or rounded to
/// odd at 125 bits or more: see [`sum_wide`]. A zero that cancels is `+0`.
fn stepped_sum(start: f64, step: f64, index: usize) -> (bool, u128, i32) {
    let start = Exact::float(start).expect("the start is finite");
    let step = Exact::float(step).expect("the step is finite");
    let product = (
        step.negative,
        u128::from(step.magnitude) * index as u128,
        step.exponent,
    );
    let start = (start.negative, u128::from(start.magnitude), start.exponent);
    match (start.1, product.1) {
        (0, 0) => (start.0 && product.0, 0, 0),
        (0, _) => product,
        (_, 0) => start,
        _ => sum_wide(start, product).unwrap_or((false, 0, 0)),
    }
}

/// `start + index · step` rounded once to `T`, from the exact sum, for a
/// finite start and step and an index past `0`.
#[cold]
#[inline(never)]
fn stepped_exactly<T: Target>(start: f64, step: f64, index: usize) -> T {
    let (negative, magnitude, exponent) = stepped_sum(start, step, index);
    if magnitude == 0 {
        return T::assemble(negative, 0, 0);
    }
    rounded(negative, magnitude, exponent)
}

/// Value `index` rounded once to float32; see [`point_f32`].
fn stepped_f32(start: f64, step: f64, index: usize) -> f32 {
    let wide = stepped_point(start, step, index);
    let narrow = wide as f32;
    let finite = start.is_finite() && step.is_finite();
    if index != 0 && finite && on_halfway(wide, narrow) {
        stepped_exactly(start, step, index)
    } else {
        narrow
    }
}

/// `wide` truncated towards zero, and whether that is surely the exact
/// value's: below 2^51, where every integer is a float64 and so `wide` is on
/// the same side of each as the exact value, and only when it is not one
/// itself -- unless `exact` says the value is `wide`.
#[inline(always)]
fn truncated_integer(wide: f64, exact: bool) -> (i64, bool) {
    let (even, rounded) = nearest_even(wide);
    // The nearest integer is one past the truncation when it is further
    // from zero than `wide`.
    let past =
        i64::from((wide >= 0.0) & (rounded > wide)) - i64::from((wide < 0.0) & (rounded < wide));
    let settled = (wide.abs() < 2f64.powi(51)) & (exact | (rounded != wide));
    (even.wrapping_sub(past), settled)
}

/// Value `index` truncated towards zero and saturated to the int64 range.
fn stepped_integer(start: f64, step: f64, index: usize) -> i64 {
    let wide = stepped_point(start, step, index);
    let (truncated, settled) = truncated_integer(wide, index == 0);
    if settled {
        truncated
    } else if index == 0 || !(start.is_finite() && step.is_finite()) {
        // The start itself, or no exact sum to take.
        wide as i64
    } else {
        stepped_integer_exactly(start, step, index)
    }
}

/// Value `index` truncated towards zero, from the exact sum.
///
/// Rounded to odd, the sum is on the same side of every integer as the
/// exact one wherever an integer is two of its units or more; where its
/// units are whole, it is exact or past 2^124 and saturates.
#[cold]
#[inline(never)]
fn stepped_integer_exactly(start: f64, step: f64, index: usize) -> i64 {
    let (negative, magnitude, exponent) = stepped_sum(start, step, index);
    let whole = if exponent >= 0 {
        if 128 - magnitude.leading_zeros() as i32 + exponent > 64 {
            u128::from(u64::MAX)
        } else {
            magnitude << exponent
        }
    } else if exponent <= -128 {
        0
    } else {
        magnitude >> -exponent
    };
    let signed = if negative {
        -(whole as i128)
    } else {
        whole as i128
    };
    signed.clamp(i64::MIN.into(), i64::MAX.into()) as i64
}

/// The `count` values of an `arange` from `start` by `step`, each `start +
/// index · step` rounded once to `dtype`: truncated towards zero and
/// saturated for an integer one, and to whether it is nonzero for a bool.
pub fn stepped_data(
    start: f64,
    step: f64,
    count: usize,
    dtype: DataType,
    device: Device,
) -> TensorData {
    let finite = start.is_finite() && step.is_finite();
    let stepping = finite.then(|| Stepping::new(start, step, count)).flatten();
    match dtype {
        DataType::Float32 => build(
            count,
            dtype,
            device,
            stepping,
            |near| {
                let narrow = near.value as f32;
                let twice = !near.exact & on_halfway(near.value, narrow);
                (narrow, !twice)
            },
            |index| stepped_f32(start, step, index),
        ),
        DataType::Float64 => build(
            count,
            dtype,
            device,
            stepping,
            |near| (near.value, near.settled),
            |index| stepped_point(start, step, index),
        ),
        DataType::Int32 => {
            let narrow = |value: i64| value.clamp(i32::MIN.into(), i32::MAX.into()) as i32;
            build(
                count,
                dtype,
                device,
                stepping,
                |near| {
                    let (value, settled) = truncated_integer(near.value, near.exact);
                    (narrow(value), settled)
                },
                |index| narrow(stepped_integer(start, step, index)),
            )
        }
        DataType::Int64 => build(
            count,
            dtype,
            device,
            stepping,
            |near| truncated_integer(near.value, near.exact),
            |index| stepped_integer(start, step, index),
        ),
        DataType::Bool => build(
            count,
            dtype,
            device,
            stepping,
            // A sum of multiples of the least subnormal rounds to zero only
            // when it is zero.
            |near| (near.value != 0.0, true),
            |index| stepped_point(start, step, index) != 0.0,
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

    #[test]
    fn a_step_is_rounded_once() {
        let mut next = generator(0x1f83_d9ab_fb41_bd6b);
        let mut exacts = 0;
        for round in 0..300_000 {
            let step = f64::from_bits(next() >> (1 + next() % 2));
            let step = if next().is_multiple_of(2) {
                step
            } else {
                -step
            };
            let index = 1 + (next() % 100_000) as usize;
            // Starts of any size, near the negated product, and a few ulps
            // from it.
            let product = index as f64 * step;
            let start = match round % 3 {
                0 => f64::from_bits(next() >> (1 + next() % 2)),
                1 => -product * (1.0 + (next() % 1000) as f64 * f64::EPSILON),
                _ => -product + (next() % 64) as f64 * product.abs() * f64::EPSILON,
            };
            if !(start.is_finite() && step.is_finite() && product.is_finite()) {
                continue;
            }
            let exact: f64 = stepped_exactly(start, step, index);
            let at = format!("{start:e} + {index} * {step:e}");
            if let Some(stepping) = Stepping::new(start, step, index + 1) {
                for near in [stepping.near::<false>(index), stepping.near::<true>(index)] {
                    assert_eq!(near.value.to_bits(), exact.to_bits(), "{at}");
                    if near.exact {
                        // The exact sum is the value, compared with trailing
                        // zeros taken off both.
                        let reduced = |(magnitude, exponent): (u128, i32)| {
                            let zeros = magnitude.trailing_zeros().min(127);
                            (magnitude >> zeros, exponent + zeros as i32)
                        };
                        let (negative, magnitude, exponent) = stepped_sum(start, step, index);
                        let value = Exact::float(near.value).unwrap();
                        if magnitude == 0 {
                            assert_eq!(value.magnitude, 0, "{at}");
                        } else {
                            assert_eq!(negative, value.negative, "{at}");
                            assert_eq!(
                                reduced((magnitude, exponent)),
                                reduced((u128::from(value.magnitude), value.exponent)),
                                "{at}"
                            );
                        }
                        exacts += 1;
                    }
                }
            }
            let narrow: f32 = stepped_exactly(start, step, index);
            assert_eq!(
                stepped_f32(start, step, index).to_bits(),
                narrow.to_bits(),
                "{at}"
            );
        }
        assert!(exacts > 100, "{exacts}");
        assert_eq!(stepped_point(-0.0, 1.0, 0).to_bits(), (-0.0f64).to_bits());
        // `0.1 * 3` rounds up to the float after `0.3`, and so does the sum.
        assert_eq!(stepped_point(0.0, 0.1, 3), 0.30000000000000004);
        assert_eq!(stepped_point(1.0, 0.1, 3), 1.3);
        // Truncation of a value just short of a whole number.
        let below = f64::from_bits(3.0f64.to_bits() - 1);
        assert_eq!(stepped_integer(below, 2f64.powi(-60), 1), 2);
        assert_eq!(stepped_integer(-below, -(2f64.powi(-60)), 1), -2);
        assert_eq!(stepped_integer(2f64.powi(62), 0.5, 3), (1 << 62) + 1);
        assert_eq!(stepped_integer(1e300, 1.0, 1), i64::MAX);
    }

    #[test]
    fn the_blocks_give_each_stepped_value_as_it_is_given_alone() {
        let cases = [
            (0.0, 0.1, 1000),
            (-89.62934926018048, 0.3, 700),
            (-0.0, 1.0, 5),
            (0.5, 0.25, 3000),
            (2f64.powi(62), 0.5, 300),
            (3.0, -1.0 / 3.0, 999),
            (1e300, 1e290, 50),
            (0.0, 5e-324, 20),
        ];
        for (start, step, count) in cases {
            let data = |dtype| stepped_data(start, step, count, dtype, Device::cpu());
            let (float64, float32) = (data(DataType::Float64), data(DataType::Float32));
            let (int64, int32) = (data(DataType::Int64), data(DataType::Int32));
            let bool = data(DataType::Bool);
            for index in 0..count {
                let at = format!("{start:e} + {index} * {step:e}");
                let point = stepped_point(start, step, index);
                let float64 = float64.as_f64_slice().unwrap()[index];
                assert_eq!(float64.to_bits(), point.to_bits(), "{at}");
                let narrow = stepped_f32(start, step, index);
                let float32 = float32.as_f32_slice().unwrap()[index];
                assert_eq!(float32.to_bits(), narrow.to_bits(), "{at}");
                let integer = stepped_integer(start, step, index);
                assert_eq!(int64.as_i64_slice().unwrap()[index], integer, "{at}");
                let clamped = integer.clamp(i32::MIN.into(), i32::MAX.into()) as i32;
                assert_eq!(int32.as_i32_slice().unwrap()[index], clamped, "{at}");
                assert_eq!(bool.as_bool_slice().unwrap()[index], point != 0.0, "{at}");
            }
        }
    }
}
