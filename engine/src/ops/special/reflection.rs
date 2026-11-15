// Copyright (c) Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

//! `digamma` and `lgamma` left of zero, their reflection formulas carried in
//! pairs of float64s.
//!
//! Left of zero both are reflected, `digamma(x) = digamma(-x) - 1/x -
//! pi cot(pi x)` and `lgamma(x) = ln pi - ln |x sin(pi x)| - lgamma(-x)`, and
//! between each pair of negative integers each crosses zero, where its terms
//! -- of the size of `ln |x|` -- cancel. Taken in float64 the answer there was
//! what the terms rounded: `lgamma` came out 16% low at the float nearest its
//! zero at -2.457 and `digamma` 1.8e15 ulps out next to its zero at -1.573,
//! and across (-3, -2) half of `lgamma`'s answers missed by an ulp or more.
//!
//! Here each term is carried to about 106 bits, which leaves the difference
//! some fifty even at the floats nearest the zeros, and the float64 is that
//! rounded once.

use crate::ops::arithmetic::rounded_once::{product_error, two_sum};

/// A value as the unevaluated sum of two float64s, the second within half an
/// ulp of the first.
#[derive(Clone, Copy, Debug)]
struct Pair {
    hi: f64,
    lo: f64,
}

/// `pi`, `ln 2`, `ln pi` and `ln(2 pi) / 2`, each as the float64 nearest it and
/// that float64's distance to it.
const PI: Pair = Pair {
    hi: std::f64::consts::PI,
    lo: 1.224_646_799_147_353_2e-16,
};
const LN_2: Pair = Pair {
    hi: std::f64::consts::LN_2,
    lo: 2.319_046_813_846_299_6e-17,
};
const LN_PI: Pair = Pair {
    hi: 1.144_729_885_849_400_2,
    lo: 1.026_595_116_270_782_6e-17,
};
const HALF_LN_TWO_PI: Pair = Pair {
    hi: 0.918_938_533_204_672_8,
    lo: -3.878_294_158_067_241_4e-17,
};

/// Where the asymptotic expansions take over: past it fifteen terms hold
/// both to well past the pairs' precision.
const EXPANSION_MIN: f64 = 20.0;

/// `B_2k / 2k` for `k` from one: the coefficients of `digamma(w) - ln w +
/// 1/(2w)` in `-w^-2k`, each as its numerator and denominator, both whole
/// numbers float64 holds exactly.
const DIGAMMA_SERIES: [(f64, f64); 15] = [
    (1.0, 12.0),
    (-1.0, 120.0),
    (1.0, 252.0),
    (-1.0, 240.0),
    (1.0, 132.0),
    (-691.0, 32760.0),
    (1.0, 12.0),
    (-3617.0, 8160.0),
    (43867.0, 14364.0),
    (-174611.0, 6600.0),
    (77683.0, 276.0),
    (-236364091.0, 65520.0),
    (657931.0, 12.0),
    (-3392780147.0, 3480.0),
    (1723168255201.0, 85932.0),
];

/// `B_2k / (2k (2k - 1))` for `k` from one: the coefficients of Stirling's
/// series for `lgamma(w)` in `w^(1 - 2k)`.
const LGAMMA_SERIES: [(f64, f64); 15] = [
    (1.0, 12.0),
    (-1.0, 360.0),
    (1.0, 1260.0),
    (-1.0, 1680.0),
    (1.0, 1188.0),
    (-691.0, 360360.0),
    (1.0, 156.0),
    (-3617.0, 122400.0),
    (43867.0, 244188.0),
    (-174611.0, 125400.0),
    (77683.0, 5796.0),
    (-236364091.0, 1506960.0),
    (657931.0, 300.0),
    (-3392780147.0, 93960.0),
    (1723168255201.0, 2492028.0),
];

/// `a + b` and its rounding error, for `|a|` at least `|b|`.
#[inline]
fn quick_two_sum(a: f64, b: f64) -> (f64, f64) {
    let sum = a + b;
    (sum, b - (sum - a))
}

/// `a · b` and its rounding error.
#[inline]
fn two_prod(a: f64, b: f64) -> (f64, f64) {
    let product = a * b;
    (product, product_error(a, b, product))
}

impl Pair {
    const ZERO: Self = Self { hi: 0.0, lo: 0.0 };
    const ONE: Self = Self { hi: 1.0, lo: 0.0 };

    fn of(value: f64) -> Self {
        Self { hi: value, lo: 0.0 }
    }

    /// The float64 nearest the pair.
    fn value(self) -> f64 {
        self.hi + self.lo
    }

    fn negated(self) -> Self {
        Self {
            hi: -self.hi,
            lo: -self.lo,
        }
    }

    fn plus(self, other: Self) -> Self {
        let (sum, error) = two_sum(self.hi, other.hi);
        let (low, low_error) = two_sum(self.lo, other.lo);
        let (sum, error) = quick_two_sum(sum, error + low);
        let (hi, lo) = quick_two_sum(sum, error + low_error);
        Self { hi, lo }
    }

    fn minus(self, other: Self) -> Self {
        self.plus(other.negated())
    }

    fn plus_f(self, value: f64) -> Self {
        let (sum, error) = two_sum(self.hi, value);
        let (hi, lo) = quick_two_sum(sum, error + self.lo);
        Self { hi, lo }
    }

    fn times(self, other: Self) -> Self {
        let (product, error) = two_prod(self.hi, other.hi);
        let (hi, lo) = quick_two_sum(product, error + (self.hi * other.lo + self.lo * other.hi));
        Self { hi, lo }
    }

    fn times_f(self, value: f64) -> Self {
        let (product, error) = two_prod(self.hi, value);
        let (hi, lo) = quick_two_sum(product, error + self.lo * value);
        Self { hi, lo }
    }

    fn over_f(self, value: f64) -> Self {
        let quotient = self.hi / value;
        let (product, error) = two_prod(quotient, value);
        let (remainder, remainder_error) = two_sum(self.hi, -product);
        let rest = (remainder + ((remainder_error - error) + self.lo)) / value;
        let (hi, lo) = quick_two_sum(quotient, rest);
        Self { hi, lo }
    }

    fn over(self, other: Self) -> Self {
        let first = self.hi / other.hi;
        let remainder = self.minus(other.times_f(first));
        let second = remainder.hi / other.hi;
        let remainder = remainder.minus(other.times_f(second));
        let third = remainder.hi / other.hi;
        let (hi, lo) = quick_two_sum(first, second);
        Self { hi, lo }.plus_f(third)
    }

    fn reciprocal(self) -> Self {
        Self::ONE.over(self)
    }
}

/// `ln a` of a positive, normal `a`.
fn ln(a: Pair) -> Pair {
    // `a = 2^e m`, `m` within a factor of `sqrt 2` of one, the scaling exact.
    let mut exponent = ((a.hi.to_bits() >> 52) & 0x7ff) as i32 - 1023;
    let mut scale = 2f64.powi(-exponent);
    if a.hi * scale > std::f64::consts::SQRT_2 {
        exponent += 1;
        scale *= 0.5;
    }
    let m = Pair {
        hi: a.hi * scale,
        lo: a.lo * scale,
    };
    // `ln m = 2 atanh s` with `s = (m - 1)/(m + 1)` below 0.172, and
    // `atanh s = s Σ s^2k / (2k + 1)`: twenty-one terms, those past the
    // eleventh -- each under 2^-55 of the first -- in float64.
    let s = m.plus_f(-1.0).over(m.plus_f(1.0));
    let square = s.times(s);
    let mut tail = 0.0;
    for k in (11..=20).rev() {
        tail = tail * square.hi + 1.0 / f64::from(2 * k + 1);
    }
    let mut sum = Pair::of(tail);
    for k in (0..=10).rev() {
        sum = sum
            .times(square)
            .plus(Pair::ONE.over_f(f64::from(2 * k + 1)));
    }
    LN_2.times_f(f64::from(exponent))
        .plus(s.times(sum).times_f(2.0))
}

/// `sin(pi r)` and `cos(pi r)` for `|r|` at most a half.
fn sin_cos_pi(r: f64) -> (Pair, Pair) {
    let magnitude = r.abs();
    // Past a quarter, the cosine and the sine of the complement, which is
    // exact.
    let (reduced, swapped) = if magnitude > 0.25 {
        (0.5 - magnitude, true)
    } else {
        (magnitude, false)
    };
    let t = PI.times_f(reduced);
    let minus_square = t.times(t).negated();
    // `sin t = t Σ (-t^2)^k / (2k + 1)!` and `cos t = Σ (-t^2)^k / (2k)!`
    // for `t` up to `pi/4`: fifteen terms, those past the eighth -- each
    // under 2^-53 of the first -- in float64.
    let (mut sine, mut cosine) = (Pair::ONE, Pair::ONE);
    let (mut sine_term, mut cosine_term) = (Pair::ONE, Pair::ONE);
    for k in 1..=7 {
        sine_term = sine_term
            .times(minus_square)
            .over_f(f64::from((2 * k) * (2 * k + 1)));
        cosine_term = cosine_term
            .times(minus_square)
            .over_f(f64::from((2 * k - 1) * (2 * k)));
        sine = sine.plus(sine_term);
        cosine = cosine.plus(cosine_term);
    }
    let (mut sine_small, mut cosine_small) = (sine_term.hi, cosine_term.hi);
    let (mut sine_rest, mut cosine_rest) = (0.0, 0.0);
    for k in 8..=14 {
        sine_small *= minus_square.hi / f64::from((2 * k) * (2 * k + 1));
        cosine_small *= minus_square.hi / f64::from((2 * k - 1) * (2 * k));
        sine_rest += sine_small;
        cosine_rest += cosine_small;
    }
    let sine = t.times(sine.plus_f(sine_rest));
    let cosine = cosine.plus_f(cosine_rest);
    let (sine, cosine) = if swapped {
        (cosine, sine)
    } else {
        (sine, cosine)
    };
    (if r < 0.0 { sine.negated() } else { sine }, cosine)
}

/// `Σ c_k u^k` for `k` from one to fifteen, `c_k` from `coefficients`: the
/// first five in pairs, the rest -- under 2^-40 of the first at the
/// smallest `u` here -- in float64.
fn expansion(coefficients: &[(f64, f64); 15], u: Pair) -> Pair {
    let mut tail = 0.0;
    for &(numerator, denominator) in coefficients[5..].iter().rev() {
        tail = tail * u.hi + numerator / denominator;
    }
    let mut sum = Pair::of(tail);
    for &(numerator, denominator) in coefficients[..5].iter().rev() {
        sum = sum.times(u).plus(Pair::of(numerator).over_f(denominator));
    }
    sum.times(u)
}

/// `digamma(y)` for a positive `y`: `digamma(y + n) - Σ 1/(y + i)`, with
/// `y + n` where the asymptotic expansion holds.
fn digamma_positive(y: f64) -> Pair {
    let mut w = Pair::of(y);
    let mut steps = Pair::ZERO;
    while w.hi < EXPANSION_MIN {
        steps = steps.plus(w.reciprocal());
        w = w.plus_f(1.0);
    }
    // `digamma(w) = ln w - 1/(2w) - Σ B_2k / (2k w^2k)`.
    let inverse = w.reciprocal();
    ln(w)
        .minus(inverse.times_f(0.5))
        .minus(expansion(&DIGAMMA_SERIES, inverse.times(inverse)))
        .minus(steps)
}

/// `lgamma(y)` for a positive `y`: `lgamma(y + n) - ln Π (y + i)`, with
/// `y + n` where Stirling's series holds.
fn lgamma_positive(y: f64) -> Pair {
    let mut w = Pair::of(y);
    let mut product = Pair::ONE;
    while w.hi < EXPANSION_MIN {
        product = product.times(w);
        w = w.plus_f(1.0);
    }
    // `lgamma(w) = (w - 1/2) ln w - w + ln(2 pi)/2 + Σ B_2k w^(1 - 2k) /
    // (2k (2k - 1))`.
    let inverse = w.reciprocal();
    w.plus_f(-0.5)
        .times(ln(w))
        .minus(w)
        .plus(HALF_LN_TWO_PI)
        .plus(expansion(&LGAMMA_SERIES, inverse.times(inverse)).times(w))
        .minus(ln(product))
}

/// `digamma(x)` for a negative `x` that is not an integer:
/// `digamma(-x) - 1/x - pi cot(pi x)`, its terms in pairs, rounded once.
pub(super) fn digamma_reflected(x: f64) -> f64 {
    let r = x - x.round();
    let (sine, cosine) = sin_cos_pi(r);
    let cotangent = PI.times(cosine).over(sine);
    digamma_positive(-x)
        .minus(Pair::of(x).reciprocal())
        .minus(cotangent)
        .value()
}

/// `lgamma(x)` for `x` below minus one and not an integer: `ln pi -
/// ln |x sin(pi x)| - lgamma(-x)`, its terms in pairs, rounded once.
pub(super) fn lgamma_reflected(x: f64) -> f64 {
    let r = x - x.round();
    let (sine, _) = sin_cos_pi(r);
    let product = sine.times_f(x);
    let magnitude = if product.hi < 0.0 {
        product.negated()
    } else {
        product
    };
    LN_PI
        .minus(ln(magnitude))
        .minus(lgamma_positive(-x))
        .value()
}
