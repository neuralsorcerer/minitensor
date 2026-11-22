// Copyright (c) Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

//! The log-cosh loss's gradient, rounded once.
//!
//! The gradient of `log(cosh(p - t))` in a prediction is `tanh(p - t)`, times
//! the upstream gradient `g` and over the count `n` for a mean. It was `tanh`
//! of the rounded difference, times a rounded `1 / n`, times `g`: up to 2.3
//! ulps out. Here `tanh` is taken of the exact difference as a pair of floats
//! within 2^-97 of it, and its product with `g` and the quotient by the count
//! are carried with a bound. Where the bound does not settle the value, the
//! pair's product over the count is rounded in whole numbers; where `tanh`
//! is all but the difference, the difference less its cube over three is,
//! and where it is all but one, the exact value is.

use super::regression::{Operands, divided, in_range, rounded};
use super::{Carried, Element, Output, Total, two_sum};
use crate::ops::arithmetic::extended::{Pair, Pairs, WIDTH, error_of, expm1, expm1_lanes};
use std::mem::MaybeUninit;

/// Past this size of the difference `tanh` is within 2^-102 of one.
const SATURATED: f64 = 36.0;

/// Below this size of the difference `tanh(d)` is within `d^2 / 3`, under
/// 2^-97, of `d`: 2^-48.
const TINY: f64 = 1.0 / (1u64 << 48) as f64;

/// A run of log-cosh's gradients in its predictions, each `tanh(p - t) g /
/// count`, the operands as [`super::regression_slopes`] takes them: the
/// value of `T` nearest each where a bound settles which that is, as it has
/// for every input tried, and otherwise the value nearest one within 2^-97
/// of it -- within `2 d^4 / 15` of it where the difference `d` is below
/// 2^-48 in size, and the exact value itself past 354.
#[inline(always)]
pub(super) fn tanh_slopes<I: Element, T: Output, const FMA: bool>(
    (x, (highs, lows), upstream): Operands<'_, I>,
    count: f64,
    out: &mut [MaybeUninit<T>],
) {
    let inverse = 1.0 / count;
    let g_at = |i: usize| upstream[if upstream.len() == 1 { 0 } else { i }].wide();
    let each = |i: usize| {
        let target = (highs[i].wide(), lows.map_or(0.0, |lows| lows[i]));
        tanh_term::<T, FMA>(x[i].wide(), target, g_at(i), (count, inverse))
    };
    if T::PLAIN && lows.is_none() {
        // For float32, first plain float64 -- `tanh` within some 2^-48 of
        // it, and the difference, the reciprocal of the count and the two
        // products a rounding each -- with a bound of 2^-40: a pass without
        // branches over a block, and then the block's terms it does not
        // settle, some three in a hundred thousand.
        let plain = |p: I, t: I, g: f64| {
            let value = tanh_plain::<FMA>(p.wide() - t.wide()) * g * inverse;
            let (narrow, clear) = T::settle_lanes(value, 0.0, value.abs() * 2f64.powi(-40));
            (narrow, clear & ((value == 0.0) | in_range(value)))
        };
        for (k, block) in out.chunks_mut(512).enumerate() {
            let span = 512 * k..512 * k + block.len();
            let (xs, ts) = (&x[span.clone()], &highs[span.clone()]);
            let mut rest = false;
            if upstream.len() == 1 {
                let g = upstream[0].wide();
                for ((cell, &p), &t) in block.iter_mut().zip(xs).zip(ts) {
                    let (value, settled) = plain(p, t, g);
                    cell.write(value);
                    rest |= !settled;
                }
            } else {
                let gs = &upstream[span.clone()];
                for (((cell, &p), &t), &g) in block.iter_mut().zip(xs).zip(ts).zip(gs) {
                    let (value, settled) = plain(p, t, g.wide());
                    cell.write(value);
                    rest |= !settled;
                }
            }
            if rest {
                for (cell, i) in block.iter_mut().zip(span) {
                    if !plain(x[i], highs[i], g_at(i)).1 {
                        cell.write(each(i));
                    }
                }
            }
        }
        return;
    }
    if T::PLAIN || !FMA {
        // A float32 loss beside an integer target's parts, which is rare,
        // and a CPU without fused multiply-adds, whose products' splits the
        // lanes below would only make longer.
        for (i, cell) in out.iter_mut().enumerate() {
            cell.write(each(i));
        }
        return;
    }
    // A few terms side by side: each `tanh` is a chain of some two hundred
    // operations, which they run together.
    let target = |i: usize| (highs[i].wide(), lows.map_or(0.0, |lows| lows[i]));
    let whole = out.len() / WIDTH * WIDTH;
    let (blocks, rest) = out.split_at_mut(whole);
    for (k, block) in blocks.as_chunks_mut::<WIDTH>().0.iter_mut().enumerate() {
        let start = WIDTH * k;
        let operands: [(f64, (f64, f64), f64); WIDTH] = std::array::from_fn(|lane| {
            let i = start + lane;
            (x[i].wide(), target(i), g_at(i))
        });
        let pairs = operands.map(|(p, target, _)| difference(p, target));
        let tanhs = tanh_lanes::<FMA>(pairs);
        for (lane, cell) in block.iter_mut().enumerate() {
            let parts = (pairs[lane], tanhs[lane]);
            cell.write(finish::<T, FMA>(
                operands[lane],
                (count, inverse),
                parts.0,
                parts.1,
            ));
        }
    }
    for (k, cell) in rest.iter_mut().enumerate() {
        cell.write(each(whole + k));
    }
}

/// `tanh(d)` in float64 without a branch, within some 2^-48 of it: `u / (u +
/// 2)` for `u = expm1(2|d|)`, signed as `d`, with `|d|` held at 20, past
/// which `tanh` is within 2^-56 of one. NaN for NaN.
#[inline(always)]
fn tanh_plain<const FMA: bool>(d: f64) -> f64 {
    let u = crate::ops::simd::expm1_reduced::<FMA>(2.0 * d.abs().clamp(0.0, 20.0));
    (u / (u + 2.0)).copysign(d)
}

/// One gradient, `tanh(p - t) g / count` for the target `high + low`: see
/// [`tanh_slopes`].
#[inline(always)]
fn tanh_term<T: Output, const FMA: bool>(
    p: f64,
    (high, low): (f64, f64),
    g: f64,
    (count, inverse): (f64, f64),
) -> T {
    let (head, tail) = difference(p, (high, low));
    let tanh = tanh_pair::<FMA>(head, tail);
    finish::<T, FMA>((p, (high, low), g), (count, inverse), (head, tail), tanh)
}

/// The exact difference `p - (high + low)` as the pair `head + tail`; an
/// integer target's parts leave the tail rounded, within 2^-105 of the
/// difference.
#[inline(always)]
fn difference(p: f64, (high, low): (f64, f64)) -> (f64, f64) {
    let (first, error) = two_sum(p, -high);
    if low == 0.0 {
        return (first, error);
    }
    let (head, second) = two_sum(first, -low);
    two_sum(head, error + second)
}

/// `tanh(head + tail)` as the pair `t + t'`, within 2^-97 of it, for a head
/// that is not zero: `-m / (2 + m)` for `m = expm1(-2|d|)`, both as pairs,
/// from [`TINY`] to [`SATURATED`], the difference itself below and one past.
/// Without a branch, every piece formed and one chosen, and the pairs'
/// `expm1` of an argument in its range whichever is.
#[inline(always)]
fn tanh_pair<const FMA: bool>(head: f64, tail: f64) -> (f64, f64) {
    let size = head.abs();
    let sign = 1f64.copysign(head);
    let between = (TINY..=SATURATED).contains(&size);
    let (a, a_low) = if between {
        (size, sign * tail)
    } else {
        (1.0, 0.0)
    };
    let m = expm1::<FMA>(Pair(-2.0 * a, -2.0 * a_low));
    let th = m.neg().div::<FMA>(Pair(2.0, 0.0).add(m));
    let (t, t_low) = if size > SATURATED {
        (sign, 0.0)
    } else {
        (sign * th.0, sign * th.1)
    };
    if size < TINY {
        (head, tail)
    } else {
        (t, t_low)
    }
}

/// [`tanh_pair`] on each of [`WIDTH`] lanes, to the same bits.
#[inline(always)]
fn tanh_lanes<const FMA: bool>(pairs: [(f64, f64); WIDTH]) -> [(f64, f64); WIDTH] {
    let size = pairs.map(|(head, _)| head.abs());
    let sign = pairs.map(|(head, _)| 1f64.copysign(head));
    let m = expm1_lanes::<FMA>(Pairs::of(|lane| {
        if (TINY..=SATURATED).contains(&size[lane]) {
            Pair(-2.0 * size[lane], -2.0 * sign[lane] * pairs[lane].1)
        } else {
            Pair(-2.0, 0.0)
        }
    }));
    let th = m.neg().div::<FMA>(Pairs::of(|_| Pair(2.0, 0.0)).add(m));
    std::array::from_fn(|lane| {
        let (t, t_low) = if size[lane] > SATURATED {
            (sign[lane], 0.0)
        } else {
            (sign[lane] * th.0[lane], sign[lane] * th.1[lane])
        };
        if size[lane] < TINY {
            pairs[lane]
        } else {
            (t, t_low)
        }
    })
}

/// [`tanh_term`] from the difference's pair and its `tanh`'s. Where an
/// operand or `g` is not finite, or the difference or `g` is zero, it is the
/// formula's, but for an infinite difference's finite gradient, whose
/// `tanh` is one, rounded once.
#[inline(always)]
fn finish<T: Output, const FMA: bool>(
    (p, (high, low), g): (f64, (f64, f64), f64),
    (count, inverse): (f64, f64),
    (head, tail): (f64, f64),
    (t, t_low): (f64, f64),
) -> T {
    let d = (p - high) - low;
    let operands = p.is_finite() && high.is_finite();
    if !(operands && g.is_finite()) || d == 0.0 || g == 0.0 {
        if !operands && d.is_infinite() && g.is_finite() && g != 0.0 {
            return exact_slope((d, 0.0), g, count, 0.0);
        }
        // A zero signed as the formula signs it, NaN, or an infinity.
        return T::narrow(d.tanh() * g / count);
    }
    // Times `g`, the leading product's error exact and a bound of 2^-96 for
    // `tanh`'s and the rest's, then over the count.
    let product = t * g;
    let total = Carried {
        sum: product,
        error: error_of::<FMA>(t, g, product) + t_low * g,
        slack: product.abs() * 2f64.powi(-96),
    };
    let fine = in_range(product) & (g.abs() < 2f64.powi(990));
    let total = fine
        .then(|| divided::<FMA>(total, (count, inverse)))
        .flatten();
    match total.and_then(rounded::<T>) {
        Some(value) => value,
        None => exact_slope((head, tail), g, count, 2f64.powi(-200)),
    }
}

/// [`tanh_term`] in whole numbers, for a finite `g` and a difference `head +
/// tail`, neither zero.
///
/// Past `|d| = 354`, where `tanh` is one less 2^-1020 at most, it is the
/// quotient of `g` less `nudge` of it: a halfway point that `g / count` is
/// exactly goes toward zero, as the exact value does, and every other is
/// over 2^-107 of it away, beyond what the nudge or `tanh` moves -- and for
/// an infinite difference, whose `tanh` is one, a nudge of zero. Below
/// `|d| = 2^-48` it is `(3 d - d^3) g / (3 count)`, within `2 d^4 / 15` of
/// the exact value, which takes a halfway point `d g / count` is toward
/// zero as well; and between, the product of [`tanh_term`]'s pair.
#[cold]
#[inline(never)]
fn exact_slope<T: Output>((head, tail): (f64, f64), g: f64, count: f64, nudge: f64) -> T {
    let size = head.abs();
    let sign = 1f64.copysign(head);
    let numerator = if size > 354.0 {
        Total::of_terms(&[(sign, g, 1.0), (-sign, g, nudge)])
    } else if size < TINY {
        let (h, t) = (head, tail);
        // `(h + t)^3`, its terms three times over as one and two.
        let numerator = Total::of_quads(&[
            [h, g, 3.0, 1.0],
            [t, g, 3.0, 1.0],
            [-h, h, h, g],
            [-h, h, t, g],
            [-h, h, 2.0 * t, g],
            [-h, t, t, g],
            [-2.0 * h, t, t, g],
            [-t, t, t, g],
        ]);
        return numerator.quotient::<T>(Total::of_terms(&[(count, 3.0, 1.0)]));
    } else {
        let m = expm1::<false>(Pair(-2.0 * size, -2.0 * sign * tail));
        let th = m.neg().div::<false>(Pair(2.0, 0.0).add(m));
        Total::of_terms(&[(sign * th.0, g, 1.0), (sign * th.1, g, 1.0)])
    };
    numerator.quotient::<T>(Total::of_terms(&[(count, 1.0, 1.0)]))
}
