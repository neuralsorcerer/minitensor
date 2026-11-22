// Copyright (c) Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

//! Binary cross entropy's gradients, rounded once.
//!
//! A term `-(t log p + (1 - t) log(1 - p))` of `binary_cross_entropy`, each
//! logarithm floored at -100 -- `log p` below `e^-100`, `log(1 - p)` at one --
//! has the slope `(p - t) / (p (1 - p))` in its prediction where neither is
//! floored, `-t / p` or `(1 - t) / (1 - p)` where one is, and `log((1 - p) /
//! p)`, the floors kept, in its target.
//! Each was formed in float64, a few roundings, and then times a rounded
//! `1 / n` and the incoming gradient: up to 2.1 ulps out in float64. Here the
//! prediction's, a ratio of whole numbers, is carried exactly but for the
//! divisor's last roundings, and rounded once -- in whole numbers where the
//! bound does not settle it -- and the target's is a pair of floats, as
//! `log1p((1 - 2p) / p)` near a half, where the two logarithms all but
//! cancel, its product with the incoming gradient and quotient by the count
//! carried with a bound.

use super::divergence::ln_pair;
use super::regression::{divided, in_range, rounded};
use super::{Carried, Element, Output, Total, product_error, ratio_settled, two_sum};
use crate::ops::arithmetic::extended::{Pair, ln_1p_pair};
use std::mem::MaybeUninit;

/// Each term's slope in its prediction, or with `of_target` in its target,
/// times `g / count`, `g` one incoming gradient for every term or each term's
/// own: in the prediction the value of `T` nearest it, and in the target the
/// value nearest it wherever a bound settles which that is, as it has for
/// every input tried, and otherwise the value nearest the slope's pair times
/// `g / count`, within 2^-96 of the slope. Where an operand is not finite, or
/// `g` is zero, it is the formula's.
pub(crate) fn bce_slopes<T: Element + Output>(
    (p, t): (&[T], &[T]),
    upstream: &[f64],
    count: f64,
    of_target: bool,
) -> Vec<T> {
    let fma = !cfg!(target_arch = "x86_64") || super::fused();
    let inverse = 1.0 / count;
    let flow = |i: usize| upstream[if upstream.len() == 1 { 0 } else { i }];
    let fill = |out: &mut [MaybeUninit<T>]| {
        crate::ops::map::par_out_chunks(out, 1 << 12, &|start, part| {
            for (k, cell) in part.iter_mut().enumerate() {
                let i = start + k;
                let (p, t, g) = (p[i].wide(), t[i].wide(), flow(i));
                cell.write(if of_target {
                    target_slope(p, g, (count, inverse), fma)
                } else {
                    prediction_slope(p, t, g, count)
                });
            }
        });
    };
    // SAFETY: the chunks cover the output, and each element is written.
    unsafe { crate::ops::map::build_vec(p.len(), fill) }
}

/// The least float whose logarithm is -100 or more: `e^-100` rounded up.
const FLOOR_EDGE: f64 = 3.720_075_976_020_836e-44;

/// Whether the forward's `log p` and `log(1 - p)` are at least its floor of
/// -100, for `p` from 0 to 1: `p` at least `e^-100`, and below one -- the
/// float below one is 2^-53 short of it, far past `e^-100`. As comparisons
/// rather than of the logarithms: the 53 float64s below `e^-100` whose
/// logarithm rounds to -100 were taken for unfloored, a slope of `1 / p`
/// where the floored term has none. NaN is neither.
#[inline(always)]
fn unfloored(p: f64) -> (bool, bool) {
    (p >= FLOOR_EDGE, p < 1.0)
}

/// The prediction's slope in float64, as the forward's derivative: a NaN
/// prediction's is NaN.
#[inline(always)]
fn prediction_formula(p: f64, t: f64) -> f64 {
    match unfloored(p) {
        (true, true) => (p - t) / (p * (1.0 - p)),
        (true, false) => -t / p,
        (false, true) => (1.0 - t) / (1.0 - p),
        (false, false) => p + t,
    }
}

/// The target's slope in float64, the floors kept, and `log1p((1 - 2p) / p)`
/// near a half; and the size a bound on it is taken of, the slope's there and
/// the logarithms' elsewhere.
#[inline(always)]
fn target_formula(p: f64) -> (f64, f64) {
    if (0.25..=0.75).contains(&p) {
        let slope = ((1.0 - 2.0 * p) / p).ln_1p();
        return (slope, slope.abs());
    }
    if p.is_nan() {
        return (p, p);
    }
    let (above, below) = unfloored(p);
    let log_p = if above { p.ln() } else { -100.0 };
    let log_q = if below { (-p).ln_1p() } else { -100.0 };
    (log_q - log_p, 1.0 + log_p.abs() + log_q.abs())
}

/// One slope in the prediction: `a g / (b n)` for the slope `a / b`, the
/// numerator exact as a pair and the divisor within 2^-104 of it, and in
/// whole numbers where the bound does not settle it.
#[inline(always)]
fn prediction_slope<T: Output>(p: f64, t: f64, g: f64, count: f64) -> T {
    let floors = unfloored(p);
    let formula = || T::narrow(prediction_formula(p, t) * g / count);
    if !(p.is_finite() && t.is_finite() && g.is_finite()) || g == 0.0 || floors == (false, false) {
        return formula();
    }
    if T::PLAIN {
        // For float32, plain float64 first: `p - t`, `1 - p` and their
        // products exact, and a few roundings besides.
        let value = prediction_formula(p, t) * g / count;
        let (narrow, clear) = T::settle_lanes(value, 0.0, value.abs() * 2f64.powi(-49));
        if clear & in_range(value) {
            return narrow;
        }
    }
    // The slope as `a / b`, both pairs: `(p - t) / (p (1 - p))`, `-t / p` or
    // `(1 - t) / (1 - p)`, each difference exact.
    let pair = |(head, tail): (f64, f64)| Pair(head, tail);
    let (a, b) = match floors {
        (true, true) => {
            let (head, tail) = two_sum(1.0, -p);
            let lead = p * head;
            let b = pair(two_sum(lead, product_error(p, head, lead) + p * tail));
            (pair(two_sum(p, -t)), b)
        }
        (true, false) => (Pair(-t, 0.0), Pair(p, 0.0)),
        _ => (pair(two_sum(1.0, -t)), pair(two_sum(1.0, -p))),
    };
    if a.0 == 0.0 {
        // A zero, signed as the formula signs it.
        return formula();
    }
    let lead = a.0 * g;
    let numerator = Carried {
        sum: lead,
        error: product_error(a.0, g, lead) + a.1 * g,
        // The trailing product's rounding, and the divisor's 2^-104.
        slack: (a.1 * g).abs() * 2f64.powi(-52) + lead.abs() * 2f64.powi(-102),
    };
    let divisor_lead = b.0 * count;
    let divisor = two_sum(
        divisor_lead,
        product_error(b.0, count, divisor_lead) + b.1 * count,
    );
    let fine = in_range(lead) & in_range(divisor_lead) & (g.abs() < 2f64.powi(990));
    match fine
        .then(|| ratio_settled::<T, false>(numerator, divisor))
        .flatten()
    {
        Some(value) => value,
        None => exact_prediction_slope(p, t, g, count, floors),
    }
}

/// [`prediction_slope`] in whole numbers.
#[cold]
#[inline(never)]
fn exact_prediction_slope<T: Output>(
    p: f64,
    t: f64,
    g: f64,
    count: f64,
    floors: (bool, bool),
) -> T {
    let (numerator, divisor) = match floors {
        (true, true) => (
            Total::of_terms(&[(p, g, 1.0), (-t, g, 1.0)]),
            Total::of_terms(&[(p, count, 1.0), (-p, p, count)]),
        ),
        (true, false) => (
            Total::of_terms(&[(-t, g, 1.0)]),
            Total::of_terms(&[(p, count, 1.0)]),
        ),
        _ => (
            Total::of_terms(&[(1.0, g, 1.0), (-t, g, 1.0)]),
            Total::of_terms(&[(1.0, count, 1.0), (-p, count, 1.0)]),
        ),
    };
    numerator.quotient::<T>(divisor)
}

/// One slope in the target: the slope as a pair, within 2^-96 of it, times
/// `g` and over the count with the bound; past it, the pair's product in
/// whole numbers.
#[inline(always)]
fn target_slope<T: Output>(p: f64, g: f64, (count, inverse): (f64, f64), fma: bool) -> T {
    let formula = || T::narrow(target_formula(p).0 * g / count);
    let floors = unfloored(p);
    if !(p.is_finite() && g.is_finite()) || g == 0.0 || p == 0.5 || floors == (false, false) {
        // A zero slope or `g`, signed as the formula signs it, or NaN.
        return formula();
    }
    let middle = (0.25..=0.75).contains(&p);
    if T::PLAIN {
        // For float32, plain float64 first: the logarithms within an ulp or
        // two and four roundings besides, with a bound of 2^-46 of the slope
        // near a half and otherwise of the logarithms' sizes.
        let (slope, size) = target_formula(p);
        let value = slope * g * inverse;
        let bound = (g * inverse).abs() * size * 2f64.powi(-46);
        let (narrow, clear) = T::settle_lanes(value, 0.0, bound);
        if clear & in_range(value) {
            return narrow;
        }
    }
    let ln_1p = |w: Pair| {
        if fma {
            ln_1p_pair::<true>(w)
        } else {
            ln_1p_pair::<false>(w)
        }
    };
    let slope = if middle {
        // `(1 - 2p) / p` as a pair, its numerator exact and its quotient's
        // remainder too.
        let numerator = 1.0 - 2.0 * p;
        let quotient = numerator / p;
        let back = quotient * p;
        let remainder = (numerator - back) - product_error(quotient, p, back);
        ln_1p(Pair(quotient, 0.0).add(Pair(remainder / p, 0.0)))
    } else {
        match floors {
            (true, true) => ln_1p(Pair(-p, 0.0)).add(ln_pair(Pair(p, 0.0), fma).neg()),
            // `log p` floored: `log(1 - p) + 100`.
            (false, true) => Pair(100.0, 0.0).add(ln_1p(Pair(-p, 0.0))),
            // `log(1 - p)` floored, at `p = 1`: `-100 - log 1`.
            _ => Pair(-100.0, 0.0),
        }
    };
    let product = slope.0 * g;
    let total = Carried {
        sum: product,
        error: product_error(slope.0, g, product) + slope.1 * g,
        slack: product.abs() * 2f64.powi(-96),
    };
    let fine = in_range(product) & (g.abs() < 2f64.powi(990));
    let total = fine
        .then(|| divided::<false>(total, (count, inverse)))
        .flatten();
    match total.and_then(rounded::<T>) {
        Some(value) => value,
        None => exact_target_slope(slope, g, count),
    }
}

/// [`target_slope`] past what the bound settles: the slope's pair times `g`
/// over the count, in whole numbers.
#[cold]
#[inline(never)]
fn exact_target_slope<T: Output>(slope: Pair, g: f64, count: f64) -> T {
    let numerator = Total::of_terms(&[(slope.0, g, 1.0), (slope.1, g, 1.0)]);
    numerator.quotient::<T>(Total::of_terms(&[(count, 1.0, 1.0)]))
}
