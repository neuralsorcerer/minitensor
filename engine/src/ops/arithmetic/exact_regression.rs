// Copyright (c) Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

//! The regression losses' gradients, each rounded once.
//!
//! A loss of the differences `d = p - t` has, in a prediction, the gradient
//! `s(d) g / n`: `s` the term's slope, `g` the upstream gradient and `n` the
//! count for a mean. Formed from the rounded difference, a rounded `1 / n` and
//! two products, it was a rounding or two out. Here the difference is exact as
//! a pair, the product is split, and each quotient is carried with a bound;
//! where that bound does not settle the value, it is the quotient of two whole
//! numbers.

use super::{Carried, Element, Output, Total, product_error, two_sum};
use std::mem::MaybeUninit;

/// A regression loss term's slope in its difference `d`.
#[derive(Clone, Copy)]
pub(crate) enum Slope {
    /// `2 d`, the squared error's.
    Squared,
    /// The sign of `d`, the absolute error's, and zero at zero.
    Absolute,
    /// `d` while `|d| <= delta` and `delta` with `d`'s sign beyond: Huber's.
    Huber(f64),
    /// `d / beta` while `|d| <= beta` and `d`'s sign beyond: smooth-L1's.
    Smooth(f64),
}

/// The slope at one difference `d`: `scale d / q`, or where `scale` is zero
/// `constant`. Numbers rather than variants, so that the piece a difference
/// falls in is chosen without a branch.
#[derive(Clone, Copy)]
struct Lead {
    scale: f64,
    constant: f64,
    q: f64,
}

impl Slope {
    /// The slope at a difference of sign `sign`, `within(limit)` saying
    /// whether its size is at most `limit`. At the limit itself the two
    /// pieces agree.
    #[inline(always)]
    fn lead(self, sign: f64, within: impl Fn(f64) -> bool) -> Lead {
        let of = |scale: f64, q: f64| Lead {
            scale,
            constant: 0.0,
            q,
        };
        let flat = |constant: f64| Lead {
            scale: 0.0,
            constant,
            q: 1.0,
        };
        match self {
            Self::Squared => of(2.0, 1.0),
            Self::Absolute => flat(sign),
            Self::Huber(delta) => {
                if within(delta) {
                    of(1.0, 1.0)
                } else {
                    flat(delta * sign)
                }
            }
            Self::Smooth(beta) => {
                if within(beta) {
                    of(1.0, beta)
                } else {
                    flat(sign)
                }
            }
        }
    }
}

/// Each `s(p - t) g / count`: the gradient of a regression loss
/// in its predictions `p`, each target `high + low` exactly -- a float, or an
/// integer as two -- and `g` one upstream gradient for every term or each
/// term's own. Each is the value of `T` nearest its exact one; where an
/// operand is not finite, it is the formula's. The products' errors are
/// taken by fused multiply-adds where the CPU has them, the same to the bit.
pub(crate) fn regression_slopes<I: Element, T: Output>(
    predictions: &[I],
    (highs, lows): (&[I], Option<&[f64]>),
    upstream: &[I],
    slope: Slope,
    count: f64,
) -> Vec<T> {
    let fma = !cfg!(target_arch = "x86_64") || super::fused();
    let fill = |out: &mut [MaybeUninit<T>]| {
        crate::ops::map::par_out_chunks(out, 1 << 13, &|start, chunk| {
            let end = start + chunk.len();
            let operands = (
                &predictions[start..end],
                (&highs[start..end], lows.map(|lows| &lows[start..end])),
                if upstream.len() == 1 {
                    upstream
                } else {
                    &upstream[start..end]
                },
            );
            if fma {
                // SAFETY: where `fma` the CPU has avx2 and fma, or the target a
                // fused multiply-add of its own.
                #[allow(unused_unsafe)]
                return unsafe { slopes_fma(operands, (slope, count), chunk) };
            }
            slopes::<I, T, false>(operands, (slope, count), chunk);
        });
    };
    // SAFETY: the chunks cover the output, and a run writes each element.
    unsafe { crate::ops::map::build_vec(predictions.len(), fill) }
}

/// The operands of a run of terms: predictions, targets and the upstream
/// gradients, one for all or one a term.
type Operands<'a, I> = (&'a [I], (&'a [I], Option<&'a [f64]>), &'a [I]);

/// [`slopes`] compiled for fused multiply-adds.
#[cfg_attr(target_arch = "x86_64", target_feature(enable = "avx2,fma"))]
fn slopes_fma<I: Element, T: Output>(
    operands: Operands<'_, I>,
    how: (Slope, f64),
    out: &mut [MaybeUninit<T>],
) {
    slopes::<I, T, true>(operands, how, out);
}

/// A run of [`term`]s.
#[inline(always)]
fn slopes<I: Element, T: Output, const FMA: bool>(
    (x, (highs, lows), upstream): Operands<'_, I>,
    (slope, count): (Slope, f64),
    out: &mut [MaybeUninit<T>],
) {
    // The divisors' reciprocals, once.
    let inverses = match slope {
        Slope::Smooth(beta) => (1.0 / beta, 1.0 / count),
        _ => (1.0, 1.0 / count),
    };
    for (i, cell) in out.iter_mut().enumerate() {
        let g = upstream[if upstream.len() == 1 { 0 } else { i }].wide();
        let target = (highs[i].wide(), lows.map_or(0.0, |lows| lows[i]));
        cell.write(term::<T, FMA>(
            x[i].wide(),
            target,
            g,
            (slope, count, inverses),
        ));
    }
}

/// One gradient: see [`regression_slopes`].
#[inline(always)]
fn term<T: Output, const FMA: bool>(
    p: f64,
    (high, low): (f64, f64),
    g: f64,
    (slope, count, (inverse_q, inverse_count)): (Slope, f64, (f64, f64)),
) -> T {
    // The formula's difference. Its sign is the exact difference's however
    // the target is held: a low part is under 2^10 beside a high one of at
    // least 2^53, and where the prediction is near the target the first
    // difference is exact.
    let d = (p - high) - low;
    let sign = if d != 0.0 && !d.is_nan() {
        1f64.copysign(d)
    } else if d == 0.0 {
        0.0
    } else {
        d
    };
    let operands = p.is_finite() && high.is_finite();
    // The exact difference as the pair `head + tail` where the target is one
    // finite float.
    let pair = operands && low == 0.0;
    let (head, tail) = if pair { two_sum(p, -high) } else { (d, 0.0) };
    let lead = if operands && low != 0.0 && g.is_finite() {
        placed(p, (high, low), slope, sign)
    } else {
        slope.lead(sign, |limit| {
            let size = head.abs();
            size < limit || (size == limit && tail * sign <= 0.0)
        })
    };
    // The slope as `(a + a') / q`.
    let difference = lead.scale != 0.0;
    let (a, a_low, q) = if difference {
        (lead.scale * head, lead.scale * tail, lead.q)
    } else {
        (lead.constant, 0.0, 1.0)
    };
    let ready = if difference {
        operands
    } else {
        lead.constant.is_finite()
    };
    if !(ready && g.is_finite()) {
        return T::narrow(a * g / q / count);
    }
    if g == 0.0 || q == f64::INFINITY || (a == 0.0 && a_low == 0.0) {
        // A zero, signed as the formula signs it; a slope past the range is
        // still a finite one.
        let a = if a.is_finite() { a } else { a.signum() };
        return T::narrow(a * g / q / count);
    }
    if !pair && difference {
        return exact((p, high, low), g, lead, count);
    }
    let product = a * g;
    let low_product = a_low * g;
    let settled = in_range(product)
        & ((a_low == 0.0) | in_range(low_product))
        & (a.abs() < 2f64.powi(990))
        & (g.abs() < 2f64.powi(990));
    let split = |sum: f64, x: f64| Carried {
        sum,
        error: if FMA {
            x.mul_add(g, -sum)
        } else {
            product_error(x, g, sum)
        },
        slack: 0.0,
    };
    let first = split(product, a);
    if T::PLAIN && settled {
        // For float32, the quotient in plain float64 with a bound on its four
        // roundings: none where the product is exact and each divisor a power
        // of two, as they mostly are for float32 operands.
        let power = |divisor: f64| divisor.to_bits() & ((1 << 52) - 1) == 0;
        let exact = (first.error == 0.0) & (a_low == 0.0) & power(q) & power(count);
        let value = (product + low_product) / q / count;
        let bound = if exact {
            0.0
        } else {
            value.abs() * 2f64.powi(-50)
        };
        if let Some(value) = in_range(value)
            .then(|| T::settle(value, 0.0, bound))
            .flatten()
        {
            return value;
        }
    }
    // `(a + a') g` exactly, each product split, in range; then over `q` and
    // the count.
    let total = settled.then(|| first.merge_tracked(split(low_product, a_low)));
    let total = total
        .and_then(|total| divided::<FMA>(total, (q, inverse_q)))
        .and_then(|total| divided::<FMA>(total, (count, inverse_count)));
    match total.and_then(rounded::<T>) {
        Some(value) => value,
        None => exact((p, high, low), g, lead, count),
    }
}

/// [`Carried::rounded`] without a branch that data of both kinds
/// mispredicts, where a zero bound settles: for float64, whose pair is
/// rounded by the two-sum, and not for float32, whose ties it is not.
#[inline(always)]
fn rounded<T: Output>(total: Carried) -> Option<T> {
    if T::PLAIN {
        return total.rounded::<T>();
    }
    let (value, rest) = two_sum(total.sum, total.error);
    let (value, settled) = T::settle_lanes(value, rest, total.slack);
    settled.then_some(value)
}

/// Whether `value` lies far enough inside the range that its product's
/// split, and what is carried beside it, stay clear of underflow.
#[inline(always)]
fn in_range(value: f64) -> bool {
    (value.abs() >= 2f64.powi(-900)) & (value.abs() <= 2f64.powi(900))
}

/// `total` over `divisor`, whose reciprocal rounded is `inverse`, carried
/// with its bound, or None where a part of it might underflow: exactly for a
/// power of two, and otherwise as [`Carried::quotient_parts`] divides.
#[inline(always)]
fn divided<const FMA: bool>(total: Carried, (divisor, inverse): (f64, f64)) -> Option<Carried> {
    if divisor == 1.0 {
        return Some(total);
    }
    // Whether a part that was `before` is zero, or `after` a normal float,
    // whose rounding is relative.
    let clear = |before: f64, after: f64| before == 0.0 || after.abs() >= f64::MIN_POSITIVE;
    let spread = total.slack * inverse;
    if divisor.to_bits() & ((1 << 52) - 1) == 0 {
        // A power of two: each part scales exactly where it stays normal.
        let error = total.error * inverse;
        let sum = total.sum * inverse;
        let fine = in_range(sum)
            & clear(total.error, error)
            & clear(total.slack, spread)
            & (divisor < 2f64.powi(990));
        return fine.then_some(Carried {
            sum,
            error,
            slack: spread,
        });
    }
    let quotient = total.sum / divisor;
    let product = quotient * divisor;
    // `product` is within a rounding of the sum, so the first difference is
    // exact, and the remainder is exact as well.
    let rounding = if FMA {
        quotient.mul_add(divisor, -product)
    } else {
        product_error(quotient, divisor, product)
    };
    let remainder = (total.sum - product) - rounding;
    // The tail over the divisor as times its reciprocal, three roundings
    // where the bound allows eight, and the slack's two short of its factor.
    let numerator = remainder + total.error;
    let tail = numerator * inverse;
    let (sum, error) = two_sum(quotient, tail);
    let slack = spread * (1.0 + 2f64.powi(-50)) + tail.abs() * 2f64.powi(-50);
    let fine = in_range(total.sum)
        & in_range(sum)
        & (divisor < 2f64.powi(990))
        & clear(numerator, tail)
        & clear(total.slack, spread);
    fine.then_some(Carried { sum, error, slack })
}

/// [`Slope::lead`] for a target of two floats: the exact difference, of
/// `sign`, placed against the limit in whole numbers.
#[cold]
#[inline(never)]
fn placed(p: f64, (high, low): (f64, f64), slope: Slope, sign: f64) -> Lead {
    slope.lead(sign, |limit| {
        // `|d| - limit`, as `sign d - limit`.
        limit == f64::INFINITY || {
            let terms = [
                (sign, p, 1.0),
                (-sign, high, 1.0),
                (-sign, low, 1.0),
                (-1.0, limit, 1.0),
            ];
            let (negative, limbs, _) = Total::of_terms(&terms).difference();
            negative || limbs.iter().all(|&limb| limb == 0)
        }
    })
}

/// [`term`] in whole numbers where it is not zero: the slope times `g`, and
/// the divisor, as exact totals, and their quotient rounded once.
#[cold]
#[inline(never)]
fn exact<T: Output>((p, high, low): (f64, f64, f64), g: f64, lead: Lead, count: f64) -> T {
    let s = lead.scale;
    let numerator = if s == 0.0 {
        Total::of_terms(&[(lead.constant, g, 1.0)])
    } else {
        Total::of_terms(&[(s, p, g), (-s, high, g), (-s, low, g)])
    };
    numerator.quotient::<T>(Total::of_terms(&[(lead.q, count, 1.0)]))
}
