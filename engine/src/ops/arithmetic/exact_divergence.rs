// Copyright (c) Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

//! The Kullback-Leibler divergence's gradient in its target.
//!
//! A term `t log(t / p)` of `kl_div` has the slope `1 + log(t / p)` in its
//! target. It was `log t - log p + 1`, each logarithm rounded: where the two
//! cancel, as a good prediction's do, the slope kept the roundings of numbers
//! the size of `log t`, hundreds of ulps of it, and near `p = e t`, where the
//! slope is zero, nothing else. Here `log(t / p)` is a pair of floats -- the
//! ratio as a pair, brought within a factor of `sqrt 2` of one by a power of
//! two, the `log1p` of what is left and the power's logarithm as pairs -- one
//! is added to it exactly, and its product with the incoming gradient and the
//! quotient by the count are carried with a bound. Where that leaves the slope
//! below 2^-20, it is `log1p((e t - p) / p)` instead, the numerator exact in
//! whole numbers but for the 2^-217 of `e` its four floats leave out: the
//! slope of float64 operands comes within some 2^-112 of zero, past the reach
//! of any pair of `log 2`'s size.

use super::regression::{divided, in_range, rounded};
use super::{Carried, Element, Output, Total, product_error, two_sum};
use crate::ops::arithmetic::extended::{LN_2, Pair, ln_1p_pair};
use std::mem::MaybeUninit;

/// `e` as four floats, each the float64 nearest what the ones before leave:
/// within 2^-217 of it.
const E: [f64; 4] = [
    std::f64::consts::E,
    1.445_646_891_729_250_2e-16,
    -2.127_717_108_038_176_8e-33,
    1.515_630_159_841_219_1e-49,
];

/// Each `(1 + log(t / p)) g / count`, `g` one incoming gradient for every
/// term or each term's own: the value of `T` nearest it wherever a bound
/// settles which that is, as it has for every input tried, and otherwise the
/// value nearest the slope's pair times `g / count` -- within `2^-98 (1 +
/// |log(t / p)|)` of the slope, and within 2^-96 of it where it is below
/// 2^-20. A target or prediction that is zero or not finite takes the
/// formula's value.
pub(crate) fn kl_target_slopes<T: Element + Output>(
    (t, p): (&[T], &[T]),
    upstream: &[f64],
    count: f64,
) -> Vec<T> {
    let fma = !cfg!(target_arch = "x86_64") || super::fused();
    let inverse = 1.0 / count;
    let flow = |i: usize| upstream[if upstream.len() == 1 { 0 } else { i }];
    // For float32, first plain float64: the ratio and the slope a rounding
    // each, the logarithm within 2^-51 of it, and the products two more,
    // with a bound of 2^-44 of `1 + |log|` and 2^-48 of the value. Where the
    // slope is near zero the bound is most of it, and the pairs take it.
    let plain = |i: usize| {
        let (t, p, g) = (t[i].wide(), p[i].wide(), flow(i));
        let log = crate::ops::simd::log_core::<false, false>(t / p, 0.0);
        let value = (1.0 + log) * g * inverse;
        let bound =
            (g * inverse).abs() * (1.0 + log.abs()) * 2f64.powi(-44) + value.abs() * 2f64.powi(-48);
        let (narrow, clear) = T::settle_lanes(value, 0.0, bound);
        let ordinary = (t > 0.0) & (p > 0.0) & in_range(t / p) & in_range(value);
        (narrow, clear & ordinary)
    };
    let fill = |out: &mut [MaybeUninit<T>]| {
        crate::ops::map::par_out_chunks(out, 1 << 12, &|start, part| {
            for (k, cell) in part.iter_mut().enumerate() {
                let i = start + k;
                if T::PLAIN {
                    let (value, settled) = plain(i);
                    if settled {
                        cell.write(value);
                        continue;
                    }
                }
                let operands = (t[i].wide(), p[i].wide(), flow(i));
                cell.write(target_slope(operands, (count, inverse), fma));
            }
        });
    };
    // SAFETY: the chunks cover the output, and each element is written.
    unsafe { crate::ops::map::build_vec(t.len(), fill) }
}

/// One slope of [`kl_target_slopes`].
#[inline(always)]
fn target_slope<T: Output>(
    (t, p, g): (f64, f64, f64),
    (count, inverse): (f64, f64),
    fma: bool,
) -> T {
    let ordinary = t.is_finite() && p.is_finite() && g.is_finite() && t > 0.0 && p > 0.0;
    if !ordinary {
        return T::narrow((t.ln() - p.ln() + 1.0) * g / count);
    }
    let (log, bound) = log_ratio(t, p, fma);
    let slope = Pair(1.0, 0.0).add(log);
    let (slope, bound) = if slope.0.abs() < 2f64.powi(-20) {
        let slope = slope_near_zero(t, p, fma);
        (slope, slope.0.abs() * 2f64.powi(-96))
    } else {
        (slope, bound)
    };
    if g == 0.0 {
        // Signed as the slope and `g` sign it.
        return T::narrow(slope.0 * g / count);
    }
    let product = slope.0 * g;
    let total = Carried {
        sum: product,
        error: product_error(slope.0, g, product) + slope.1 * g,
        slack: bound * g.abs(),
    };
    let fine = in_range(product) & (g.abs() < 2f64.powi(990)) & (slope.0.abs() < 2f64.powi(990));
    let total = fine
        .then(|| divided::<false>(total, (count, inverse)))
        .flatten();
    match total.and_then(rounded::<T>) {
        Some(value) => value,
        None => exact_target_slope(slope, g, count),
    }
}

/// `log(t / p)` as a pair for `t` and `p` positive and finite, and a bound on
/// its error, with one's rounding added to it: `2^-98 (1 + |log|)` where the
/// ratio is taken as a pair -- the quotient and its exact remainder over `p`,
/// all three far enough inside the range for the remainder's product to
/// split -- and otherwise `log t - log p`, each a pair, within `2^-98 (1 +
/// |log t| + |log p|)`.
#[inline(always)]
fn log_ratio(t: f64, p: f64, fma: bool) -> (Pair, f64) {
    let ratio = t / p;
    let inside = |x: f64| (x >= 2f64.powi(-960)) & (x <= 2f64.powi(990));
    if inside(t) && inside(p) && inside(ratio) {
        let back = ratio * p;
        // `back` is within a rounding of `t`, so the first difference is exact.
        let remainder = (t - back) - product_error(ratio, p, back);
        let (head, tail) = two_sum(ratio, remainder / p);
        let log = ln_pair(Pair(head, tail), fma);
        return (log, (1.0 + log.0.abs()) * 2f64.powi(-98));
    }
    let (a, b) = (ln_pair(Pair(t, 0.0), fma), ln_pair(Pair(p, 0.0), fma));
    let bound = (1.0 + a.0.abs() + b.0.abs()) * 2f64.powi(-98);
    (a.add(b.neg()), bound)
}

/// `log(x)` as a pair for a positive finite `x = hi + lo`, a two-sum's pair:
/// `k log 2 + log1p(m - 1)` for `x = 2^k m`, `m` within a factor of `sqrt 2`
/// of one, so that `m - 1` is exact and its `log1p` small. A subnormal head
/// is moved up by `2^64` first.
#[inline(always)]
fn ln_pair(x: Pair, fma: bool) -> Pair {
    let (x, shift) = if x.0 < f64::MIN_POSITIVE {
        (Pair(x.0 * 2f64.powi(64), x.1 * 2f64.powi(64)), -64)
    } else {
        (x, 0)
    };
    let bits = x.0.to_bits();
    let mut k = (bits >> 52) as i32 - 1023;
    let mut m = f64::from_bits((bits & ((1 << 52) - 1)) | (1023 << 52));
    if m > std::f64::consts::SQRT_2 {
        m *= 0.5;
        k += 1;
    }
    let low = x.1 * 2f64.powi(-k);
    let w = Pair(m, low).add(Pair(-1.0, 0.0));
    let log = if fma {
        ln_1p_pair::<true>(w)
    } else {
        ln_1p_pair::<false>(w)
    };
    let k = f64::from(k + shift);
    let lead = k * LN_2.0;
    let power = Pair(lead, product_error(k, LN_2.0, lead) + k * LN_2.1);
    log.add(power)
}

/// `1 + log(t / p)` where it is near zero, as a pair within 2^-96 of it:
/// `log1p(w)` for `w = (e t - p) / p`, its numerator exact in whole numbers
/// but for the 2^-217 of `e` that [`E`] leaves out -- some 2^-105 of the
/// slope -- and the quotient a pair from two roundings, of it and of what it
/// leaves.
#[cold]
#[inline(never)]
fn slope_near_zero(t: f64, p: f64, fma: bool) -> Pair {
    let mut terms = vec![
        (E[0], t, 1.0),
        (E[1], t, 1.0),
        (E[2], t, 1.0),
        (E[3], t, 1.0),
    ];
    terms.push((-1.0, p, 1.0));
    let over = || Total::of_terms(&[(p, 1.0, 1.0)]);
    let head: f64 = Total::of_terms(&terms).quotient(over());
    terms.push((-head, p, 1.0));
    let tail: f64 = Total::of_terms(&terms).quotient(over());
    let (head, tail) = two_sum(head, tail);
    if fma {
        ln_1p_pair::<true>(Pair(head, tail))
    } else {
        ln_1p_pair::<false>(Pair(head, tail))
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
