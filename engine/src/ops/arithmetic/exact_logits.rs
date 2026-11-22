// Copyright (c) Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

//! Binary cross entropy's gradients from logits, rounded once.
//!
//! A term `(1 - t) x + (1 + (w - 1) t) softplus(-x)` of
//! `binary_cross_entropy_with_logits`, `w` the positive class's weight, has
//! the slope `(1 - t) s(x) - w t s(-x)` in its logit and
//! `-x + (w - 1) softplus(-x)` in its target. The first was formed in float64
//! and rounded twice more, over the count and times the incoming gradient: up
//! to 3 ulps out in float64, and hundreds where its terms all but cancel, as
//! at a soft label's own logit rounded. The second was formed in the logits'
//! dtype with a rounded `1 / n`: up to 37 ulps out with a weight, and about
//! its zero, where its terms cancel, nothing of it left.
//!
//! With `u = e^-|x|` the logit's slope is `(1 - t - w t u) / (1 + u)` for a
//! logit of zero or more and `((1 - t) u - w t) / (1 + u)` below, and the
//! target's `-x + (w - 1) log1p(u)`, or `-w x + (w - 1) log1p(u)` below: each
//! a pair of floats within 2^-97 of the sizes of its terms, its product with
//! the incoming gradient and quotient by the count carried with a bound.
//! Where that does not settle the answer, or a part leaves float64's range,
//! the quotient is taken in whole numbers from `u`, or its `log1p`, in limbs
//! of 64 bits: with what they may lack taken off and put on, and in more limbs
//! until both ends round alike.

use super::regression::{divided, in_range, rounded};
use super::{
    Carried, Element, Exact, Output, Total, product_error, product_of, shifted, subtract, two_sum,
};
use crate::ops::arithmetic::extended::{
    LOG2_E, Pair, Pairs, WIDTH, error_of, exp_lanes, exp_pair, ln_1p_lanes, ln_1p_pair,
};
use std::mem::MaybeUninit;

/// Up to this size of a logit `u` is above 2^-866, and its pair within 2^-98
/// of it; past it, `u` is below 2^-865.
const FAR: f64 = 600.0;

/// Past this size of a logit `u` is below 2^-4299: see
/// [`exact_logit_slope`].
const BEYOND: f64 = 2980.0;

/// The operands of a run of terms: logits, targets, the weights where there
/// are any, and the upstream gradients, one for all or one a term.
type Operands<'a, T> = (&'a [T], &'a [T], Option<&'a [T]>, &'a [f64]);

/// Each term's slope in its logit, or with `of_target` in its target, times
/// `g / count`, `g` one incoming gradient for every term or each term's own:
/// the value of `T` nearest it, wherever 65 limbs of `u` settle which that is,
/// as they have for every input tried, and always where the slope is a ratio
/// of whole numbers -- at a zero logit, without a weight in the target, and
/// where `1 + (w - 1) t` is zero. Where an operand is NaN, or the target or
/// the weight is infinite, it is the formula's; an infinite logit's is the
/// limit's, and a zero `g` gives a zero signed as it and the slope sign it.
pub(crate) fn logit_slopes<T: Element + Output>(
    (x, t, w): (&[T], &[T], Option<&[T]>),
    upstream: &[f64],
    count: f64,
    of_target: bool,
) -> Vec<T> {
    let fma = !cfg!(target_arch = "x86_64") || super::fused();
    let fill = |out: &mut [MaybeUninit<T>]| {
        crate::ops::map::par_out_chunks(out, 1 << 12, &|start, part| {
            let end = start + part.len();
            let operands = (
                &x[start..end],
                &t[start..end],
                w.map(|w| &w[start..end]),
                if upstream.len() == 1 {
                    upstream
                } else {
                    &upstream[start..end]
                },
            );
            // Float32 settles in plain float64 and seldom reaches the pairs.
            if fma && !T::PLAIN {
                // SAFETY: where `fma` the CPU has avx2 and fma, or the target a
                // fused multiply-add of its own.
                #[allow(unused_unsafe)]
                return unsafe { run_fma(operands, (count, of_target), part) };
            }
            run::<T, false>(operands, (count, of_target), part);
        });
    };
    // SAFETY: the chunks cover the output, and a run writes each element.
    unsafe { crate::ops::map::build_vec(x.len(), fill) }
}

/// [`run`] compiled for fused multiply-adds.
#[cfg_attr(target_arch = "x86_64", target_feature(enable = "avx2,fma"))]
fn run_fma<T: Element + Output>(
    operands: Operands<'_, T>,
    how: (f64, bool),
    out: &mut [MaybeUninit<T>],
) {
    run::<T, true>(operands, how, out);
}

/// A run of slopes: for float64 with fused multiply-adds, side by side in
/// blocks of [`WIDTH`], each pair's exponential a chain of some two hundred
/// operations, which they run together, to the same bits.
#[inline(always)]
fn run<T: Element + Output, const FMA: bool>(
    (x, t, w, upstream): Operands<'_, T>,
    (count, of_target): (f64, bool),
    out: &mut [MaybeUninit<T>],
) {
    let how = (count, 1.0 / count);
    let at = |i: usize| {
        let weight = w.map_or(1.0, |w| w[i].wide());
        let g = upstream[if upstream.len() == 1 { 0 } else { i }];
        (x[i].wide(), t[i].wide(), weight, g)
    };
    let each = |i: usize| {
        let (x, t, w, g) = at(i);
        if of_target {
            target_slope::<T, FMA>(x, w, g, how, || target_pair::<FMA>(x, w))
        } else {
            logit_slope::<T, FMA>(x, t, w, g, how, || logit_pair::<FMA>(x, t, w))
        }
    };
    if T::PLAIN || !FMA {
        for (i, cell) in out.iter_mut().enumerate() {
            cell.write(each(i));
        }
        return;
    }
    let whole = out.len() / WIDTH * WIDTH;
    let (blocks, rest) = out.split_at_mut(whole);
    for (k, block) in blocks.as_chunks_mut::<WIDTH>().0.iter_mut().enumerate() {
        let operands: [(f64, f64, f64, f64); WIDTH] =
            std::array::from_fn(|lane| at(WIDTH * k + lane));
        let pairs = if of_target {
            target_lanes::<FMA>(operands)
        } else {
            logit_lanes::<FMA>(operands)
        };
        for ((cell, (x, t, w, g)), pair) in block.iter_mut().zip(operands).zip(pairs) {
            cell.write(if of_target {
                target_slope::<T, FMA>(x, w, g, how, || pair)
            } else {
                logit_slope::<T, FMA>(x, t, w, g, how, || pair)
            });
        }
    }
    for (k, cell) in rest.iter_mut().enumerate() {
        cell.write(each(whole + k));
    }
}

/// The logit's slope in float64, and the size of its terms: within 2^-50 of
/// them where they stay in range.
#[inline(always)]
fn logit_formula(x: f64, t: f64, w: f64) -> (f64, f64) {
    let (a, b, u) = (1.0 - t, w * t, (-x.abs()).exp());
    let (first, second) = if x >= 0.0 { (a, b * u) } else { (a * u, b) };
    let divisor = 1.0 + u;
    (
        (first - second) / divisor,
        (first.abs() + second.abs()) / divisor,
    )
}

/// One slope in the logit, `pair` its [`logit_pair`]: see [`logit_slopes`].
#[inline(always)]
fn logit_slope<T: Output, const FMA: bool>(
    x: f64,
    t: f64,
    w: f64,
    g: f64,
    (count, inverse): (f64, f64),
    pair: impl FnOnce() -> (Pair, f64),
) -> T {
    let ordinary = t.is_finite() & w.is_finite() & g.is_finite() & !x.is_nan();
    if !ordinary || g == 0.0 {
        let (slope, terms) = logit_formula(x, t, w);
        if !ordinary {
            return T::narrow(slope * g / count);
        }
        return zero((slope, terms), g, || {
            exact_logit_slope::<f64>(x, t, w, 1.0, 1.0)
        });
    }
    let near =
        (x.abs() <= FAR) & ((1.0 - t).abs() <= 2f64.powi(900)) & ((w * t).abs() <= 2f64.powi(900));
    if !near {
        return beyond_logit(x, t, w, g, (count, inverse));
    }
    if T::PLAIN {
        // For float32, plain float64 first: `u` within an ulp or two and a
        // few roundings, with a bound of 2^-48 of the terms.
        let (slope, terms) = logit_formula(x, t, w);
        let value = slope * g * inverse;
        let bound = terms * (g * inverse).abs() * 2f64.powi(-48);
        let (narrow, clear) = T::settle_lanes(value, 0.0, bound);
        if clear & in_range(value) & in_range(terms) {
            return narrow;
        }
    }
    let (slope, terms) = pair();
    in_range(terms)
        .then(|| carried::<T, FMA>(slope, terms * 2f64.powi(-96), g, (count, inverse)))
        .flatten()
        .unwrap_or_else(|| exact_logit_slope(x, t, w, g, count))
}

/// The logit's slope as a pair, within 2^-97 of the size of its terms,
/// returned beside it: `u` a pair within 2^-98 of itself, `1 - t` and `w t`
/// exact, and their product, difference and quotient as pairs.
#[inline(always)]
fn logit_pair<const FMA: bool>(x: f64, t: f64, w: f64) -> (Pair, f64) {
    let u = exp_pair::<FMA>(Pair(-x.abs(), 0.0));
    let (a, a_low) = two_sum(1.0, -t);
    let b = w * t;
    let (a, b) = (Pair(a, a_low), Pair(b, error_of::<FMA>(w, t, b)));
    let (first, second) = if x >= 0.0 {
        (a, b.mul::<FMA>(u))
    } else {
        (a.mul::<FMA>(u), b)
    };
    let divisor = Pair(1.0, 0.0).add(u);
    let slope = first.add(second.neg()).div::<FMA>(divisor);
    (slope, (first.0.abs() + second.0.abs()) / divisor.0)
}

/// [`logit_pair`] on each of [`WIDTH`] lanes' logits, targets and weights,
/// to the same bits.
#[inline(always)]
fn logit_lanes<const FMA: bool>(operands: [(f64, f64, f64, f64); WIDTH]) -> [(Pair, f64); WIDTH] {
    let u = exp_lanes::<FMA>(Pairs::of(|lane| Pair(-operands[lane].0.abs(), 0.0)));
    let a = Pairs::of(|lane| {
        let (head, tail) = two_sum(1.0, -operands[lane].1);
        Pair(head, tail)
    });
    let b = Pairs::of(|lane| {
        let (_, t, w, _) = operands[lane];
        let b = w * t;
        Pair(b, error_of::<FMA>(w, t, b))
    });
    let (au, bu) = (a.mul::<FMA>(u), b.mul::<FMA>(u));
    let above = operands.map(|(x, ..)| x >= 0.0);
    let pick = |yes: Pairs, no: Pairs| {
        Pairs::of(|lane| {
            if above[lane] {
                yes.lane(lane)
            } else {
                no.lane(lane)
            }
        })
    };
    let (first, second) = (pick(a, au), pick(bu, b));
    let divisor = Pairs::of(|_| Pair(1.0, 0.0)).add(u);
    let slope = first.add(second.neg()).div::<FMA>(divisor);
    std::array::from_fn(|lane| {
        let terms = (first.0[lane].abs() + second.0[lane].abs()) / divisor.0[lane];
        (slope.lane(lane), terms)
    })
}

/// The target's slope in float64, and the size of its terms: within 2^-50 of
/// them where they stay in range.
#[inline(always)]
fn target_formula(x: f64, w: f64) -> (f64, f64) {
    let lead = if x >= 0.0 { 1.0 } else { w };
    let (linear, term) = (-lead * x, (w - 1.0) * (-x.abs()).exp().ln_1p());
    (linear + term, linear.abs() + term.abs())
}

/// One slope in the target, `pair` its [`target_pair`]: see
/// [`logit_slopes`].
#[inline(always)]
fn target_slope<T: Output, const FMA: bool>(
    x: f64,
    w: f64,
    g: f64,
    (count, inverse): (f64, f64),
    pair: impl FnOnce() -> (Pair, f64),
) -> T {
    let ordinary = w.is_finite() & g.is_finite() & x.is_finite();
    if !ordinary || g == 0.0 {
        if x.is_infinite() & w.is_finite() & !g.is_nan() {
            // The limit, `-x` or `-w x`; below zero without a weight, `-0`.
            let lead = if x > 0.0 { 1.0 } else { w };
            let slope = if lead == 0.0 { -0.0 } else { -lead * x };
            return T::narrow(slope * g / count);
        }
        let (slope, terms) = target_formula(x, w);
        if !ordinary {
            return T::narrow(slope * g / count);
        }
        return zero((slope, terms), g, || {
            exact_target_slope::<f64>(x, w, 1.0, 1.0)
        });
    }
    let near = (x.abs() <= FAR) & (w.abs() <= 2f64.powi(890));
    if !near {
        return beyond_target(x, w, g, (count, inverse));
    }
    if T::PLAIN {
        // For float32, plain float64 first: `u` and its `log1p` within an
        // ulp or two and a few roundings, with a bound of 2^-48 of the terms.
        let (slope, terms) = target_formula(x, w);
        let value = slope * g * inverse;
        let bound = terms * (g * inverse).abs() * 2f64.powi(-48);
        let (narrow, clear) = T::settle_lanes(value, 0.0, bound);
        if clear & in_range(value) & in_range(terms) {
            return narrow;
        }
    }
    let (slope, terms) = pair();
    in_range(terms)
        .then(|| carried::<T, FMA>(slope, terms * 2f64.powi(-96), g, (count, inverse)))
        .flatten()
        .unwrap_or_else(|| exact_target_slope(x, w, g, count))
}

/// The target's slope as a pair, within 2^-97 of the size of its terms,
/// returned beside it: `-x` or `-w x` and `w - 1` exact, `log1p(u)` within
/// 2^-98, and their product and sum as pairs.
#[inline(always)]
fn target_pair<const FMA: bool>(x: f64, w: f64) -> (Pair, f64) {
    let tail = ln_1p_pair::<FMA>(exp_pair::<FMA>(Pair(-x.abs(), 0.0)));
    let (spare, spare_low) = two_sum(w, -1.0);
    let term = Pair(spare, spare_low).mul::<FMA>(tail);
    let lead = if x >= 0.0 { 1.0 } else { w };
    let product = lead * x;
    let linear = Pair(-product, -error_of::<FMA>(lead, x, product));
    (linear.add(term), linear.0.abs() + term.0.abs())
}

/// [`target_pair`] on each of [`WIDTH`] lanes' logits and weights, to the
/// same bits.
#[inline(always)]
fn target_lanes<const FMA: bool>(operands: [(f64, f64, f64, f64); WIDTH]) -> [(Pair, f64); WIDTH] {
    let u = exp_lanes::<FMA>(Pairs::of(|lane| Pair(-operands[lane].0.abs(), 0.0)));
    let tail = ln_1p_lanes::<FMA>(u);
    let spare = Pairs::of(|lane| {
        let (head, low) = two_sum(operands[lane].2, -1.0);
        Pair(head, low)
    });
    let term = spare.mul::<FMA>(tail);
    std::array::from_fn(|lane| {
        let (x, _, w, _) = operands[lane];
        let lead = if x >= 0.0 { 1.0 } else { w };
        let product = lead * x;
        let linear = Pair(-product, -error_of::<FMA>(lead, x, product));
        let term = term.lane(lane);
        (linear.add(term), linear.0.abs() + term.0.abs())
    })
}

/// A zero `g` times the slope: signed as `g` and the slope sign it, the
/// slope's sign taken from `exact` where the float64 one lies within its
/// roundings of zero, or past the range. An exact zero slope is positive.
#[inline(always)]
fn zero<T: Output>((slope, terms): (f64, f64), g: f64, exact: impl FnOnce() -> f64) -> T {
    let clear = (slope.abs() > terms * 2f64.powi(-48)) & in_range(terms) & in_range(slope);
    let slope = if clear { slope } else { exact() };
    T::narrow(slope.signum() * g)
}

/// A slope's pair, within `bound` of it, times `g` and over the count: the
/// value of `T` nearest it where the bound settles which that is.
#[inline(always)]
fn carried<T: Output, const FMA: bool>(
    slope: Pair,
    bound: f64,
    g: f64,
    (count, inverse): (f64, f64),
) -> Option<T> {
    let product = slope.0 * g;
    let total = Carried {
        sum: product,
        error: error_of::<FMA>(slope.0, g, product) + slope.1 * g,
        // The bound, and the trailing product's rounding.
        slack: bound * g.abs() + product.abs() * 2f64.powi(-104),
    };
    let fine = in_range(product) & (g.abs() < 2f64.powi(990)) & (slope.0.abs() < 2f64.powi(990));
    fine.then(|| divided::<FMA>(total, (count, inverse)))
        .flatten()
        .and_then(rounded::<T>)
}

/// Whether `size / count` lies below a quarter of `T`'s least subnormal,
/// where every value rounds to a zero.
fn vanishes<T: Output>(size: f64, count: f64) -> bool {
    size * 2f64.powi(200) < count * 2f64.powi(T::LEAST - 2 + 200)
}

/// [`logit_slope`] past [`FAR`] or float64's range. For a finite logit past
/// `FAR`, `u` is below 2^-865, and the slope is its limit -- `1 - t` above
/// zero and `-w t` below -- less `c u / (1 + u)` above and plus it below,
/// `c = 1 - t + w t`: the limit with that as a bound, or where the limit is
/// zero, a zero where the bound is below what rounds to one. Otherwise, and
/// where that does not settle, in whole numbers.
#[cold]
#[inline(never)]
fn beyond_logit<T: Output>(x: f64, t: f64, w: f64, g: f64, how: (f64, f64)) -> T {
    if x.is_finite() && x.abs() > FAR {
        let reach = ((1.0 - t).abs() + (w * t).abs()) * 2f64.powi(-864);
        let limit = if x >= 0.0 {
            (t != 1.0).then(|| two_sum(1.0, -t))
        } else {
            let b = w * t;
            (w != 0.0 && t != 0.0).then(|| (-b, -product_error(w, t, b)))
        };
        match limit {
            Some((lead, low)) => {
                let value = in_range(lead)
                    .then(|| carried::<T, false>(Pair(lead, low), reach, g, how))
                    .flatten();
                if let Some(value) = value {
                    return value;
                }
            }
            None if vanishes::<T>(reach * g.abs(), how.0) => {
                // All of the slope is `c u` against it: `-w u` above zero,
                // `t` being one, and `(1 - t) u` below, `w t` being zero.
                let part = if x >= 0.0 { -w } else { 1.0 - t };
                let part = if part == 0.0 { 0.0 } else { part.signum() };
                return T::narrow(0.0f64.copysign(part) * g);
            }
            None => {}
        }
    }
    exact_logit_slope(x, t, w, g, how.0)
}

/// [`target_slope`] past [`FAR`] or float64's range. For a logit past `FAR`,
/// `u` is below 2^-865, and the slope is `-x`, or `-w x` below zero, and
/// `(w - 1) log1p(u)`, at most `|w - 1| 2^-865` in size: the first with that
/// as a bound, or where the first is zero, a zero where the bound is below
/// what rounds to one. Otherwise, and where that does not settle, in whole
/// numbers.
#[cold]
#[inline(never)]
fn beyond_target<T: Output>(x: f64, w: f64, g: f64, how: (f64, f64)) -> T {
    if x.abs() > FAR {
        let reach = (w - 1.0).abs() * 2f64.powi(-864);
        let lead = if x >= 0.0 { 1.0 } else { w };
        if lead != 0.0 {
            let product = lead * x;
            let linear = Pair(-product, -product_error(lead, x, product));
            let value = in_range(product)
                .then(|| carried::<T, false>(linear, reach, g, how))
                .flatten();
            if let Some(value) = value {
                return value;
            }
        } else if vanishes::<T>(reach * g.abs(), how.0) {
            // All of the slope is `-log1p(u)`.
            return T::narrow(-0.0 * g);
        }
    }
    exact_target_slope(x, w, g, how.0)
}

/// A positive number to some `64 l` bits, `m 2^e` with `m` in at most `l`
/// limbs, each step rounded towards zero: for the slopes' last resort, where
/// a pair of floats does not say which value is nearest.
#[derive(Clone)]
struct Wide {
    m: Vec<u64>,
    e: i64,
}

impl Wide {
    /// `m 2^e`, its leading `l` limbs kept.
    fn cut(mut m: Vec<u64>, mut e: i64, l: usize) -> Self {
        while m.last() == Some(&0) {
            m.pop();
        }
        let extra = m.len().saturating_sub(l);
        m.drain(..extra);
        e += 64 * extra as i64;
        Self { m, e }
    }

    /// A float that is not negative, exactly.
    fn of(x: f64) -> Self {
        let exact = Exact::float(x).expect("finite");
        Self::cut(vec![exact.magnitude], i64::from(exact.exponent), 1)
    }

    /// The place just above its leading bit; far below every other for zero.
    fn top(&self) -> i64 {
        self.m.last().map_or(i64::MIN / 4, |&last| {
            self.e + 64 * self.m.len() as i64 - i64::from(last.leading_zeros())
        })
    }

    fn times(&self, other: &Self, l: usize) -> Self {
        Self::cut(product_of(&self.m, &other.m), self.e + other.e, l)
    }

    /// Over a whole number, to `l` limbs and one more below: the dividend
    /// carried down that far.
    fn over(&self, n: u64, l: usize) -> Self {
        let pad = (l + 1).saturating_sub(self.m.len()).max(1);
        let mut m: Vec<u64> = std::iter::repeat_n(0, pad)
            .chain(self.m.iter().copied())
            .collect();
        let mut rest = 0u128;
        for limb in m.iter_mut().rev() {
            let current = (rest << 64) | u128::from(*limb);
            *limb = (current / u128::from(n)) as u64;
            rest = current % u128::from(n);
        }
        Self::cut(m, self.e - 64 * pad as i64, l)
    }

    /// The sum, or with `minus` the difference from the larger `self`; a term
    /// wholly below the limbs kept is left out.
    fn plus(&self, other: &Self, minus: bool, l: usize) -> Self {
        let below = 64 * (l as i64 + 1);
        if other.top() < self.top() - below {
            return self.clone();
        }
        if self.top() < other.top() - below {
            return other.clone();
        }
        let base = self.e.min(other.e);
        let mut a = shifted(&self.m, (self.e - base) as usize);
        let b = shifted(&other.m, (other.e - base) as usize);
        a.resize(a.len().max(b.len()) + 1, 0);
        if minus {
            subtract(&mut a, &b);
        } else {
            let mut carry = false;
            for (index, limb) in a.iter_mut().enumerate() {
                let (sum, first) = limb.overflowing_add(b.get(index).copied().unwrap_or(0));
                let (sum, second) = sum.overflowing_add(u64::from(carry));
                *limb = sum;
                carry = first || second;
            }
        }
        Self::cut(a, base, l)
    }

    /// Its value times `sign 2^power` as exact floats and their powers of
    /// two: each limb's halves.
    fn parts(&self, sign: f64, power: i64) -> impl Iterator<Item = (f64, i32)> + '_ {
        self.m.iter().enumerate().flat_map(move |(index, &limb)| {
            let place = self.e + 64 * index as i64 + power;
            [
                (sign * (limb >> 32) as f64, (place + 32) as i32),
                (sign * (limb & 0xffff_ffff) as f64, place as i32),
            ]
        })
    }
}

/// `2 atanh(z) = log((1 + z) / (1 - z))`, for `z` at most a third: twice
/// `z + z^3 / 3 + z^5 / 5 + ...`, to where a term falls below the limbs kept.
fn atanh_twice(z: &Wide, l: usize) -> Wide {
    let square = z.times(z, l);
    let (mut power, mut sum) = (z.clone(), z.clone());
    for k in (3u64..).step_by(2) {
        power = power.times(&square, l);
        let term = power.over(k, l);
        if term.top() < sum.top() - 64 * l as i64 - 8 {
            break;
        }
        sum = sum.plus(&term, false, l);
    }
    sum.e += 1;
    sum
}

/// `e^r`, for `r` from zero to `log 4`: `1 + r + r^2 / 2 + ...`, every term
/// positive, to where a term falls below the limbs kept.
fn exp_series(r: &Wide, l: usize) -> Wide {
    let (mut term, mut sum) = (Wide::of(1.0), Wide::of(1.0));
    for n in 1u64.. {
        term = term.times(r, l).over(n, l);
        if term.top() < sum.top() - 64 * l as i64 - 8 {
            break;
        }
        sum = sum.plus(&term, false, l);
    }
    sum
}

/// `u = e^-y` to `l` limbs, within a relative 2^-(64 l - 88) of itself, for
/// `y` above zero and at most [`BEYOND`]: `2^-k e^r`, `r = k log 2 - y` from
/// `log 2` to under `log 4` with `k` one past `y / log 2` taken upwards, and
/// `log 2` as `2 atanh(1 / 3)`. Each step's rounding is below 2^-64(l - 1)
/// of its result, and some ten thousand of them, `k` up to 4300, reach
/// 2^-(64 l - 80).
fn exp_wide(y: f64, l: usize) -> Wide {
    let k = (y * LOG2_E.0).ceil() + 1.0;
    let log_2 = atanh_twice(&Wide::of(1.0).over(3, l), l);
    let r = log_2.times(&Wide::of(k), l).plus(&Wide::of(y), true, l);
    let mut u = exp_series(&r, l);
    u.e -= k as i64;
    u
}

/// `log1p(u)` to `l` limbs, for `u` from zero to one: `2 atanh(u / (2 + u))`,
/// the reciprocal of `2 + u` by Newton's iteration from 0.4, which takes it
/// from within a fifth to the limbs kept, squaring its error at each step.
fn log1p_wide(u: &Wide, l: usize) -> Wide {
    let divisor = Wide::of(2.0).plus(u, false, l);
    let (two, mut inverse) = (Wide::of(2.0), Wide::of(0.4));
    for _ in 0..(64 * l).ilog2() + 2 {
        let product = divisor.times(&inverse, l);
        inverse = inverse.times(&two.plus(&product, true, l), l);
    }
    atanh_twice(&u.times(&inverse, l), l)
}

/// The value nearest the quotient `at` makes of a positive number's parts,
/// where `wide` gives the number to `l` limbs within a relative
/// 2^-(64 l - 88) of itself: with that taken off and put on, at more limbs
/// until both ends round alike. The slopes' quotients move one way with the
/// number, so that the exact one lies between the ends and rounds as they
/// do; and none is a halfway point, but where the number drops out. Past 65
/// limbs, which no input has needed, the value at the number itself.
fn settled<T: Output>(wide: impl Fn(usize) -> Wide, at: impl Fn(&[(f64, i32)]) -> T) -> T {
    let mut value = Wide::of(0.0);
    for l in [3, 5, 9, 17, 33, 65] {
        value = wide(l);
        let reach = 88 - 64 * l as i64;
        let [low, high] = [-1.0, 1.0].map(|side| {
            let parts: Vec<_> = value
                .parts(1.0, 0)
                .chain(value.parts(side, reach))
                .collect();
            at(&parts)
        });
        if low.widen().to_bits() == high.widen().to_bits() {
            return low;
        }
    }
    at(&value.parts(1.0, 0).collect::<Vec<_>>())
}

/// The quotient of two sums of scaled products, rounded once; a numerator
/// of zero gives `+0` times `g`.
fn quotient<T: Output>(numerator: &[([f64; 4], i32)], divisor: &[([f64; 4], i32)], g: f64) -> T {
    let numerator = Total::of_scaled(numerator);
    if numerator.positive == numerator.negative {
        return T::narrow(0.0 * g);
    }
    numerator.quotient(Total::of_scaled(divisor))
}

/// [`logit_slope`] in whole numbers: `(1 - t - w t u) g` or
/// `((1 - t) u - w t) g` over `(1 + u) n`. `u` is exact at a zero or an
/// infinite logit; past [`BEYOND`] it is below 2^-4299 and `2^-4300` stands
/// in for it: either is below 2^-4250 of the size of the slope's terms over
/// their finest place, so that both move the quotient the same way, and less
/// far than a tie of the rest lies from any other value.
#[cold]
#[inline(never)]
fn exact_logit_slope<T: Output>(x: f64, t: f64, w: f64, g: f64, count: f64) -> T {
    let at = |parts: &[(f64, i32)]| {
        let mut numerator = Vec::with_capacity(2 * parts.len() + 2);
        if x >= 0.0 {
            numerator.extend([([1.0, g, 1.0, 1.0], 0), ([-t, g, 1.0, 1.0], 0)]);
            numerator.extend(parts.iter().map(|&(part, power)| ([-w, t, part, g], power)));
        } else {
            numerator.push(([-w, t, g, 1.0], 0));
            for &(part, power) in parts {
                numerator.extend([([1.0, part, g, 1.0], power), ([-t, part, g, 1.0], power)]);
            }
        }
        let mut divisor = vec![([count, 1.0, 1.0, 1.0], 0)];
        divisor.extend(
            parts
                .iter()
                .map(|&(part, power)| ([count, part, 1.0, 1.0], power)),
        );
        quotient(&numerator, &divisor, g)
    };
    let size = x.abs();
    if size == f64::INFINITY {
        return at(&[]);
    }
    if size == 0.0 {
        return at(&[(1.0, 0)]);
    }
    if size > BEYOND {
        return at(&[(1.0, -4300)]);
    }
    settled(|l| exp_wide(size, l), at)
}

/// [`target_slope`] in whole numbers: `(-x + (w - 1) log1p(u)) g`, or with
/// `-w x` below zero, over `n`; past [`BEYOND`] `2^-4300` stands in for
/// `log1p(u)`, below `u`, as for [`exact_logit_slope`]'s `u`.
#[cold]
#[inline(never)]
fn exact_target_slope<T: Output>(x: f64, w: f64, g: f64, count: f64) -> T {
    let (spare, spare_low) = two_sum(w, -1.0);
    let lead = if x >= 0.0 { 1.0 } else { w };
    let at = |parts: &[(f64, i32)]| {
        let mut numerator = vec![([-lead, x, g, 1.0], 0)];
        for &(part, power) in parts {
            numerator.extend([
                ([spare, part, g, 1.0], power),
                ([spare_low, part, g, 1.0], power),
            ]);
        }
        quotient(&numerator, &[([count, 1.0, 1.0, 1.0], 0)], g)
    };
    let size = x.abs();
    if spare == 0.0 {
        return at(&[]);
    }
    if size > BEYOND {
        return at(&[(1.0, -4300)]);
    }
    settled(
        |l| {
            let u = if size == 0.0 {
                Wide::of(1.0)
            } else {
                exp_wide(size, l + 1)
            };
            log1p_wide(&u, l)
        },
        at,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// How far `wide` lies from `known 2^-power`, three floats within 2^-161
    /// of what they stand for, relative to it.
    fn apart(wide: &Wide, known: [f64; 3], power: i32) -> f64 {
        let reference = known.map(|part| ([part, 1.0, 1.0, 1.0], -power));
        let mut gap: Vec<_> = wide
            .parts(1.0, 0)
            .map(|(part, place)| ([part, 1.0, 1.0, 1.0], place))
            .collect();
        gap.extend(reference.map(|([part, ..], place)| ([-part, 1.0, 1.0, 1.0], place)));
        let gap = Total::of_scaled(&gap);
        if gap.positive == gap.negative {
            return 0.0;
        }
        gap.quotient::<f64>(Total::of_scaled(&reference))
    }

    #[test]
    fn wide_exponentials_and_logarithms_keep_their_digits() {
        // `e^-y 2^k` and `log1p(e^-y) 2^k` in three floats each, as mpmath
        // gives them.
        let exponentials = [
            (
                1.0,
                0,
                [
                    0.367_879_441_171_442_33,
                    -1.242_875_367_278_836_3e-17,
                    -5.830_044_851_072_742e-34,
                ],
            ),
            (
                37.25,
                53,
                [
                    0.598_577_405_216_289_7,
                    -5.306_847_486_120_429e-18,
                    1.571_449_451_695_143_3e-34,
                ],
            ),
            (
                700.0,
                1010,
                [
                    1.081_828_175_944_821,
                    9.312_926_389_907_246e-18,
                    -4.703_552_915_519_663e-34,
                ],
            ),
            (
                2979.0,
                4298,
                [
                    1.157_869_926_264_101,
                    3.848_817_374_343_011_4e-17,
                    -8.935_901_624_362_872e-34,
                ],
            ),
        ];
        let logarithms = [
            (
                0.0,
                0,
                [
                    std::f64::consts::LN_2,
                    2.319_046_813_846_299_6e-17,
                    5.707_708_438_416_212e-34,
                ],
            ),
            (
                1.0,
                0,
                [
                    0.313_261_687_518_222_8,
                    2.515_869_523_777_819e-17,
                    -8.598_673_220_141_201e-34,
                ],
            ),
            (
                37.25,
                53,
                [
                    0.598_577_405_216_289_7,
                    -2.519_621_042_248_911e-17,
                    -1.479_748_020_418_313_5e-33,
                ],
            ),
        ];
        for l in [3, 5, 9] {
            let bound = 2f64.powi(88 - 64 * l as i32).max(2f64.powi(-158));
            for (y, power, known) in exponentials {
                let gap = apart(&exp_wide(y, l), known, power);
                assert!(gap.abs() <= bound, "e^-{y} at {l}: {gap:e}");
            }
            for (y, power, known) in logarithms {
                let u = if y == 0.0 {
                    Wide::of(1.0)
                } else {
                    exp_wide(y, l + 1)
                };
                let gap = apart(&log1p_wide(&u, l), known, power);
                assert!(gap.abs() <= bound, "log1p(e^-{y}) at {l}: {gap:e}");
            }
        }
    }
}
