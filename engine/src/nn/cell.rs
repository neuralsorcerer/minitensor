// Copyright (c) Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

//! What the fused recurrent cells share.
//!
//! [`crate::nn::lstm_cell`] and [`crate::nn::gru_cell`] are the same shape of
//! kernel over different arithmetic: a `(batch, gates * hidden)` buffer
//! activated in blocks, then a few vectorizable passes. The partitioning rule,
//! the float64 fallbacks and the gates' values and slopes as their backwards
//! take them -- of the logits, in float64, as pairs or with their exponents
//! kept apart where float64 alone would not hold them -- are the parts that
//! would otherwise be written twice, so they are written here once.

use crate::ops::arithmetic::extended::{Pair, exp_pair_any_cpu};
use crate::ops::arithmetic::rounded_once::product_error;
use crate::ops::map::SIMD_PAR_CHUNK;
use crate::ops::simd::{exp_below_zero, exp_split};
use crate::ops::util::stable_sigmoid_f64;
use crate::parallel::{Parts, for_each_task};
use std::mem::MaybeUninit;

/// The chunk width to hand [`crate::ops::map::par_out_chunks`] for `total`
/// elements laid out in rows of `row_width`.
///
/// Below `threshold` the whole buffer is one chunk, which `par_out_chunks`
/// runs inline without reaching rayon at all. That is the case that matters:
/// a recurrent step's tensors are small by construction, and forking a pool
/// once per timestep would cost more than the arithmetic it splits. The
/// thresholds are the ones the composed operations used, so the point at which
/// this starts using every core is the point at which they did.
///
/// Above it the bands are whole rows. Which gate an element belongs to is
/// decided by where it sits within its row, so a split anywhere else would
/// hand a task half of one gate and half of the next -- which would not fail
/// loudly. It would activate the wrong elements with the wrong function.
#[inline]
pub(crate) fn band_width(total: usize, row_width: usize, threshold: usize) -> usize {
    if total < threshold || row_width == 0 {
        return total;
    }
    row_width * (SIMD_PAR_CHUNK / row_width).max(1)
}

/// `sigmoid` over a block of float64, element at a time.
///
/// There is no vectorized float64 transcendental here or in the elementwise
/// kernels, so the composed form these cells replace was scalar too. Same
/// function, same rounding.
#[inline]
pub(crate) fn sigmoid_block_f64(src: &[f64], dst: &mut [MaybeUninit<f64>]) {
    for (out, &x) in dst.iter_mut().zip(src) {
        out.write(stable_sigmoid_f64(x));
    }
}

/// `tanh` over a block of float64. See [`sigmoid_block_f64`].
#[inline]
pub(crate) fn tanh_block_f64(src: &[f64], dst: &mut [MaybeUninit<f64>]) {
    for (out, &x) in dst.iter_mut().zip(src) {
        out.write(x.tanh());
    }
}

/// `sigmoid(x)`, `sigmoid(-x)` and the slope `sigmoid(x) sigmoid(-x)` of a
/// logit, in float64, as `1 / (1 + t)`, `t / (1 + t)` and `t / (1 + t)^2` for
/// `t = e^-|x|`.
///
/// None of them is a difference of numbers near one. Written as `s (1 - s)`
/// of a rounded `s`, a gate's slope loses its digits as the gate saturates
/// and is zero once `s` rounds to one -- at a logit of 17 in float32 and 37
/// in float64 -- where it is still a float; so is a gate's distance from one.
#[inline(always)]
pub(crate) fn logistic(x: f64) -> (f64, f64, f64) {
    let (t, _) = exp_below_zero(-x.abs());
    let large = 1.0 / (1.0 + t);
    let small = t * large;
    let slope = small * large;
    if x >= 0.0 {
        (large, small, slope)
    } else {
        (small, large, slope)
    }
}

/// `e`, or zero where it is not finite: what is left beside a sum or a
/// product whose leading part has left the range, or whose split did, has
/// nothing finite to add.
#[inline(always)]
pub(crate) fn finite_or_zero(e: f64) -> f64 {
    if e.is_finite() { e } else { 0.0 }
}

/// `x * y` as a pair, the product's rounding error beside it.
#[inline(always)]
pub(crate) fn product_pair(x: f64, y: f64) -> Pair {
    let p = x * y;
    Pair(p, finite_or_zero(product_error(x, y, p)))
}

/// [`logistic`] at `x + e`, for an `e` far below `x` -- the rounding of a sum
/// that `x` is, say -- to first order: the values moved by the slope times
/// `e`, and the slope by its own, `sigmoid'(x) (sigmoid(-x) - sigmoid(x))`.
/// Where a logit is large its rounding would otherwise be a large part of
/// its slope's: `|x|` times what `x` is out by, relatively.
#[inline(always)]
pub(crate) fn logistic_near(x: f64, e: f64) -> (f64, f64, f64) {
    let (s, rest, slope) = logistic(x);
    let shift = slope * e;
    (s + shift, rest - shift, slope + slope * ((rest - s) * e))
}

/// `sigmoid(x)` as a pair, of a logit that is a pair: `1 / (1 + t)` or
/// `t / (1 + t)` for `t = e^-|x|` carried as one, to about 2^-100. Past
/// 708, where `t` would leave the normal range, it is [`logistic`]'s value
/// alone: the rest is below anything a product with it shows.
pub(crate) fn logistic_pair(x: Pair) -> Pair {
    if x.0.is_nan() || x.0.abs() > 708.0 {
        return Pair(logistic(x.0).0, 0.0);
    }
    let positive = x.0 >= 0.0;
    let t = exp_pair_any_cpu(if positive { x.neg() } else { x });
    let one = Pair(1.0, 0.0);
    (if positive { one } else { t }).div::<false>(one.add(t))
}

/// `tanh(x)` and its slope `1 - tanh(x)^2`, in float64, as `(1 - t) / (1 + t)`
/// and `4t / (1 + t)^2` for `t = e^-2|x|`, with `1 - t` formed beside `t`
/// rather than from it, so that a small `x` keeps its digits too.
#[inline(always)]
pub(crate) fn tanh_and_slope(x: f64) -> (f64, f64) {
    let (t, less_one) = exp_below_zero(-2.0 * x.abs());
    let scale = 1.0 / (1.0 + t);
    ((-less_one * scale).copysign(x), 4.0 * t * scale * scale)
}

/// `tanh(x)` as a pair, of an argument that is a pair: `(1 - t) / (1 + t)` for
/// `t = e^-2|x|` carried as one, to about 2^-100. Below 2^-30, where `1 - t`
/// would cancel, it is `x` itself, short of `tanh(x)` by under 2^-61 of it;
/// past 354, where `t` would leave the normal range, [`tanh_and_slope`]'s
/// value alone.
pub(crate) fn tanh_pair(x: Pair) -> Pair {
    let size = x.0.abs();
    if size.is_nan() || size > 354.0 {
        return Pair(tanh_and_slope(x.0).0, 0.0);
    }
    if size < 2f64.powi(-30) {
        return x;
    }
    let down = if x.0 >= 0.0 { x.neg() } else { x };
    let t = exp_pair_any_cpu(Pair(2.0 * down.0, 2.0 * down.1));
    let one = Pair(1.0, 0.0);
    let y = one.add(t.neg()).div::<false>(one.add(t));
    if x.0 >= 0.0 { y } else { y.neg() }
}

/// [`tanh_and_slope`] at `x + e`, for an `e` far below `x`, to first order:
/// the value moved by the slope times `e`, and the slope by its own,
/// `-2 tanh(x) (1 - tanh(x)^2)`. The slope moves by `2|e|` of itself, so a
/// cell state carried as a pair keeps a slope that its rounding would put
/// `2|x|` ulps out.
#[inline(always)]
pub(crate) fn tanh_and_slope_near(x: f64, e: f64) -> (f64, f64) {
    let (y, slope) = tanh_and_slope(x);
    (y + slope * e, slope - 2.0 * (y * slope) * e)
}

/// A float64 kept as `m 2^e`, `|m|` in `[1/2, 1)` and the exponent apart: for
/// gradients with a factor below float64's normal range -- a logit's slope
/// past 708, say -- which float64 holds there only to its last few bits,
/// while other factors bring their product back up where every bit shows.
/// A product of them rounds once, as the gradient is formed.
#[derive(Clone, Copy)]
pub(crate) struct Scaled(f64, i32);

impl Scaled {
    /// `x`, split; a zero, an infinity or a NaN as itself, with no exponent.
    pub(crate) fn of(x: f64) -> Scaled {
        if x == 0.0 || !x.is_finite() {
            return Scaled(x, 0);
        }
        let (x, shift) = if x.abs() < f64::MIN_POSITIVE {
            (x * 2f64.powi(64), -64)
        } else {
            (x, 0)
        };
        let bits = x.to_bits();
        let field = ((bits >> 52) & 0x7ff) as i32;
        let m = f64::from_bits((bits & !(0x7ff << 52)) | (1022 << 52));
        Scaled(m, field - 1022 + shift)
    }

    /// `m 2^e`: `e^v` from [`exp_split`]'s parts.
    fn split((m, e): (f64, i32)) -> Scaled {
        let s = Scaled::of(m);
        Scaled(s.0, s.1 + e)
    }

    pub(crate) fn times(self, other: Scaled) -> Scaled {
        let p = Scaled::of(self.0 * other.0);
        Scaled(p.0, p.1 + self.1 + other.1)
    }

    /// The float64 nearest `m 2^e`, rounded once.
    pub(crate) fn value(self) -> f64 {
        let (m, e) = (self.0, self.1);
        let power = |k: i32| f64::from_bits(((k + 1023) as u64) << 52);
        if m == 0.0 || !m.is_finite() {
            m
        } else if e > 1024 {
            f64::INFINITY.copysign(m)
        } else if e > 1023 {
            (m * 2.0) * power(1023)
        } else if e >= -1021 {
            m * power(e)
        } else if e >= -2095 {
            // Exact to the normal range and then one rounding into it.
            (m * power(e + 1074)) * f64::from_bits(1)
        } else {
            0.0f64.copysign(m)
        }
    }
}

/// [`logistic_near`] as [`Scaled`] values: past 708, where `t = e^-|x|` leaves
/// the normal range, `t` itself is the smaller of the two values and the slope
/// -- `1 + t` is one there -- moved by `e` as `e^-|x + e|` is.
pub(crate) fn logistic_scaled(x: f64, e: f64) -> [Scaled; 3] {
    if x.is_nan() || x.abs() <= 708.0 {
        let (s, rest, slope) = logistic_near(x, e);
        return [Scaled::of(s), Scaled::of(rest), Scaled::of(slope)];
    }
    let outward = if x > 0.0 { e } else { -e };
    let t = Scaled::split(exp_split(-x.abs())).times(Scaled::of(1.0 - outward));
    let one = Scaled::of(1.0);
    if x > 0.0 { [one, t, t] } else { [t, one, t] }
}

/// [`tanh_and_slope_near`] as [`Scaled`] values: past 354, where `e^-2|x|`
/// leaves the normal range, `tanh` is one in size and its slope `4 e^-2|x|`.
pub(crate) fn tanh_scaled(x: f64, e: f64) -> [Scaled; 2] {
    if x.is_nan() || x.abs() <= 354.0 {
        let (y, slope) = tanh_and_slope_near(x, e);
        return [Scaled::of(y), Scaled::of(slope)];
    }
    let outward = if x > 0.0 { e } else { -e };
    let slope =
        Scaled::split(exp_split(-2.0 * x.abs())).times(Scaled::of(4.0 * (1.0 - 2.0 * outward)));
    [Scaled::of(1.0f64.copysign(x)), slope]
}

/// What [`par_row_bands`] runs on each band: its first row, its count of
/// rows, and its part of each output.
type BandWork<'a, T, const N: usize> =
    &'a (dyn Fn(usize, usize, [&mut [MaybeUninit<T>]; N]) + Sync);

/// Run `work(first, count, parts)` over bands of whole rows of several
/// outputs at once, in parallel: `parts[k]` is rows `first..first + count` of
/// `outs[k]`, whose rows are `widths[k]` long. An output nobody wants is empty
/// with a width of zero. The bands are as [`band_width`] cuts them, by the
/// rows' combined width.
pub(crate) fn par_row_bands<T: Send, const N: usize>(
    outs: [&mut [MaybeUninit<T>]; N],
    widths: [usize; N],
    rows: usize,
    threshold: usize,
    work: BandWork<T, N>,
) {
    let row_width: usize = widths.iter().sum();
    if rows == 0 || row_width == 0 {
        return;
    }
    debug_assert!(
        outs.iter()
            .zip(&widths)
            .all(|(out, w)| out.len() == rows * w)
    );
    let band = band_width(rows * row_width, row_width, threshold) / row_width;
    let parts = outs.map(Parts::of);
    for_each_task(rows.div_ceil(band), &|index| {
        let first = index * band;
        let count = band.min(rows - first);
        // SAFETY: each output is `rows * widths[k]` long, and the bands' row
        // ranges are disjoint, so every task's windows are its own.
        let slices = std::array::from_fn(|k| unsafe {
            std::slice::from_raw_parts_mut(
                parts[k].start().add(first * widths[k]),
                count * widths[k],
            )
        });
        work(first, count, slices);
    });
}
