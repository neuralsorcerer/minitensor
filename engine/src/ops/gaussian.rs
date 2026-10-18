// Copyright (c) Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

//! The normal distribution's CDF and its inverse, to full double precision.
//!
//! These, plus `libm::lgamma`, were the entire use this crate made of `statrs`
//! — a normal CDF and quantile for sampling a truncated normal, an inverse
//! error function for `erfinv`, and a log-gamma for one line of `polygamma`.
//! `statrs` brings `nalgebra` and `simba` with it — a linear-algebra stack and
//! its SIMD abstraction layer — for those four things.
//!
//! Replacing it with the ninety lines below is worth measuring honestly,
//! because the obvious expected win is not the one that happens. Building the
//! extension module with and without the dependency, changing nothing else:
//!
//! ```text
//!   with statrs      10,114,568 bytes    2m52s clean release build
//!   without          10,107,504 bytes    2m41s
//! ```
//!
//! Seven kilobytes. Thin LTO plus `--gc-sections` had already been discarding
//! everything in `nalgebra` that the normal CDF did not reach, so the linked
//! size was never where the cost was. The cost was the eleven seconds — about
//! 7% of a clean build — and three crates' worth of dependency surface to
//! audit, update and trust, in exchange for functions with closed forms. That
//! is still a good trade; it is just not the trade it looks like.
//!
//! Neither function is an approximation of convenience:
//!
//! * The CDF is `erfc` in disguise, and `libm::erfc` — already a dependency —
//!   is correctly rounded to within an ulp or two across the whole range,
//!   including the far tail where the naive `1 - Phi(-x)` form has no
//!   significant digits left.
//! * The quantile is Wichura's AS241 (the `PPND16` variant). Its rational
//!   approximations are accurate to about 16 significant figures over
//!   `(0, 1)`, which is as good as the double it returns.

// The AS241 coefficient tables below are quoted exactly as Wichura published
// them, at more decimal places than an `f64` can hold. Trimming them to the
// representable prefix would change nothing about the compiled constants and
// would break the one check available on a transcribed table: reading it
// against the paper, digit for digit.
#![allow(clippy::excessive_precision)]

use crate::ops::simd::product_error;
use std::f64::consts::{FRAC_1_SQRT_2, FRAC_2_SQRT_PI, SQRT_2};

/// `Phi((x - mean) / std)`: the probability a normal draw falls at or below `x`.
///
/// Written as `erfc(-z/sqrt(2))/2` rather than `(1 + erf(z/sqrt(2)))/2` because
/// the two disagree exactly where it matters. For `z = -8` the true value is
/// about `6.2e-16`; the `erf` form computes `1 + (-0.9999999999999988)`, which
/// keeps one significant digit, while `erfc` returns the small number directly
/// with full precision. Inverse-CDF sampling in the tail depends on that.
#[inline]
pub(crate) fn normal_cdf(x: f64, mean: f64, std: f64) -> f64 {
    standard_normal_cdf((x - mean) / std)
}

/// What `FRAC_1_SQRT_2` leaves out of `1/sqrt(2)`.
const FRAC_1_SQRT_2_LOW: f64 = -4.833_646_656_726_457e-17;

/// `Phi(z)`, the standard normal CDF: `erfc(-z/sqrt(2))/2`, with the rounding
/// of `-z/sqrt(2)` put back.
///
/// Relative to itself `erfc(t)` falls about `2t` per unit of `t`, so the
/// rounding of `t` -- 2^-53 of it -- came out as `2t^2` ulps of the answer: 32
/// at `z = -6` and 1250 at `z = -36`, exactly the tail the `erfc` form exists
/// to keep. The rounding is recovered exactly and taken back out through the
/// slope, `-2/sqrt(pi) e^(-t^2)`, which needs only a few digits of its own,
/// leaving the error `erfc`'s.
#[inline]
pub(crate) fn standard_normal_cdf(z: f64) -> f64 {
    let t = -z * FRAC_1_SQRT_2;
    let mut tail = libm::erfc(t);
    // Past 27 the slope underflows, and the correction with it.
    if t.abs() < 27.0 {
        let rounding = product_error::<false>(-z, FRAC_1_SQRT_2, t) - z * FRAC_1_SQRT_2_LOW;
        tail -= rounding * FRAC_2_SQRT_PI * (-t * t).exp();
    }
    0.5 * tail
}

/// `phi(z)`, the standard normal density, with the rounding of `z^2` put
/// back: `e^(-z^2/2)` turns it into `z^2/2` ulps of the answer, 450 at
/// `z = 30`.
#[inline]
pub(crate) fn standard_normal_pdf(z: f64) -> f64 {
    const FRAC_1_SQRT_2PI: f64 = 0.398_942_280_401_432_7;
    let square = z * z;
    let density = (-0.5 * square).exp() * FRAC_1_SQRT_2PI;
    // Zero past `|z| = 38.6`, where the square is too large to split, and at
    // an infinity, where its error is NaN.
    if density == 0.0 {
        return density;
    }
    density * (1.0 - 0.5 * product_error::<false>(z, z, square))
}

/// `1 / (1 + e^(-2u))` for `u = sqrt(2/pi) (x + 0.044715 x^3)`, the tanh
/// form's stand-in for `Phi(x)`.
///
/// `u` is carried to twice a double's precision -- the cube, both products
/// and their sum, and the two constants -- because the exponential turns an
/// error in `u` into a relative one in its answer, and the plain sum's
/// rounding came out as 294 ulps at `x = -20`. The low part goes in as
/// `e^(u + low) = e^u (1 + low)`, true while it is tiny.
///
/// Below zero the logistic is written `r / (1 + r)` with `r = e^(2u)`, which
/// keeps its digits where `e^(-2u)` overflows: at `x = -21` the answer is a
/// normal float near 1e-307 and the other spelling gave zero. The form is
/// odd-symmetric, so `1 - tanh_cdf(x)` is `tanh_cdf(-x)` exactly, with no
/// cancellation.
#[inline]
pub(crate) fn tanh_cdf(x: f64) -> f64 {
    // `sqrt(2/pi)` and `sqrt(2/pi) * 0.044715`, each as a double and what it
    // leaves out.
    const LINEAR: (f64, f64) = (0.797_884_560_802_865_4, -4.984_654_404_555_46e-17);
    const CUBIC: (f64, f64) = (0.035_677_408_136_300_125, -3.087_574_959_077_657_5e-19);
    let (u, low) = if x.abs() < 1e100 {
        let square = x * x;
        let cube = square * x;
        let cube_low =
            product_error::<false>(square, x, cube) + product_error::<false>(x, x, square) * x;
        let linear = LINEAR.0 * x;
        let linear_low = product_error::<false>(LINEAR.0, x, linear) + LINEAR.1 * x;
        let cubic = CUBIC.0 * cube;
        let cubic_low =
            product_error::<false>(CUBIC.0, cube, cubic) + CUBIC.0 * cube_low + CUBIC.1 * cube;
        let sum = linear + cubic;
        // The rounding of the sum, exactly: Knuth's branch-free two-sum.
        let back = sum - cubic;
        let sum_low = (linear - back) + (cubic - (sum - back));
        (sum, sum_low + linear_low + cubic_low)
    } else {
        // Past this the cube cannot be split, and the answer is 0 or 1
        // whatever its last digits.
        (LINEAR.0 * x + CUBIC.0 * (x * x * x), 0.0)
    };
    if u >= 0.0 {
        1.0 / (1.0 + (-2.0 * u).exp() * (1.0 - 2.0 * low))
    } else {
        let r = (2.0 * u).exp() * (1.0 + 2.0 * low);
        r / (1.0 + r)
    }
}

/// The value `x` with `normal_cdf(x, mean, std) == p`, for `p` in `(0, 1)`.
///
/// Outside that open interval the answer is infinite, and `p` outside `[0, 1]`
/// or NaN has no answer at all; all three return NaN rather than a plausible
/// finite number, so a caller that is about to fill a tensor finds out.
#[inline]
pub(crate) fn normal_quantile(p: f64, mean: f64, std: f64) -> f64 {
    mean + std * standard_normal_quantile(p)
}

/// [`erf_inverse`] without its Newton step: AS241 and the scaling around it,
/// about 4 ulps across the range, with the signs, the endpoints and NaN as
/// there.
///
/// What float32 asks for. Four float64 ulps are a hundred-millionth of a
/// float32 one, so the step -- an `erf` or `erfc` and an exponential, three
/// quarters of the cost -- changes no float32 answer.
pub(crate) fn erf_inverse_estimate(x: f64) -> f64 {
    if x.is_nan() || !(-1.0..=1.0).contains(&x) {
        return f64::NAN;
    }
    if x == 1.0 {
        return f64::INFINITY;
    }
    if x == -1.0 {
        return f64::NEG_INFINITY;
    }
    let size = x.abs();
    let magnitude = if size <= 2.0 * CENTRAL_HALF_WIDTH {
        size * (central_quantile_ratio(0.5 * size) * (0.5 / SQRT_2))
    } else {
        -standard_normal_quantile((1.0 - size) * 0.5) / SQRT_2
    };
    magnitude.copysign(x)
}

/// The inverse error function on `[-1, 1]`, infinite at the endpoints.
///
/// `erf` and the normal CDF are the same function under a change of variable,
/// so their inverses are too: `erf(z) = 2 Phi(z sqrt 2) - 1` rearranges to
/// `erfinv(x) = Phi^-1((1 + x) / 2) / sqrt 2`. Neither end can use that
/// spelling as it stands.
///
/// `(1 + x) / 2` for `x` near 1 is a number near 1, and doubles near 1 are
/// spaced 2.2e-16 apart — so `x = 1 - 1e-17` and `x = 1` become the same
/// argument and the quantile is asked for `Phi^-1(1)`, which is infinite. The
/// complement `(1 - x) / 2` has no such problem: for `x` in `[1/2, 1]` the
/// subtraction is exact (Sterbenz), the result is small, and a small
/// probability carries its full relative precision into AS241's lower tail.
/// So the sign is folded out first and the lower tail is what gets evaluated,
/// using `Phi^-1(1 - p) = -Phi^-1(p)`.
///
/// The complement is the wrong spelling at the middle. For small `x` it is
/// a number near one half that has already lost the low digits of `x`: at
/// `x = 1e-8` half the digits were gone, and below `1.1e-16` the argument was
/// exactly one half and the answer zero -- negative zero, at that. So for
/// `|x|` up to `0.85`, where `Phi^-1((1 + x) / 2)` is in AS241's central body,
/// that body is evaluated at `q = x / 2` directly. It is odd in `q` and a
/// multiple of it, so the answer is `x` times a ratio near `sqrt(pi) / 2`, and
/// keeps every digit down through the subnormals.
///
/// Outside `[-1, 1]` there is nothing to invert and the answer is NaN. At
/// exactly `+-1` it is `+-inf`, which is the limit and what the callers of
/// `erfinv` already expect. Either zero is itself.
#[inline]
pub(crate) fn erf_inverse(x: f64) -> f64 {
    let estimate = erf_inverse_estimate(x);
    if !estimate.is_finite() {
        return estimate;
    }
    let size = x.abs();
    let estimate = estimate.abs();
    // One Newton step on `erf(y) = |x|`. AS241 is good to about 1e-16
    // relative, which with the scaling around it came out as 4 ulps across
    // the whole range; the step brings that to about one. Below a half the
    // residual is `erf(y) - |x|`; above it `(1 - |x|) - erfc(y)`, the same
    // number, whose subtraction is exact (Sterbenz) and whose `erfc` keeps
    // the digits `erf` rounds away next to one.
    let residual = if size < 0.5 {
        libm::erf(estimate) - size
    } else {
        (1.0 - size) - libm::erfc(estimate)
    };
    let slope = FRAC_2_SQRT_PI * (-estimate * estimate).exp();
    let magnitude = estimate - residual / slope;
    magnitude.copysign(x)
}

/// Evaluate a polynomial in `x` by Horner's rule, highest coefficient first.
#[inline]
fn horner(coefficients: &[f64], x: f64) -> f64 {
    coefficients.iter().fold(0.0, |acc, &c| acc * x + c)
}

/// How far either side of one half AS241's central body reaches, in `q = p - 1/2`.
const CENTRAL_HALF_WIDTH: f64 = 0.425;

/// `Phi^-1(1/2 + q) / q` for `|q| <= 0.425`: AS241's central rational
/// function in `0.180625 - q^2`, with the factor of `q` left to the caller so
/// that a `q` too small to form exactly can be carried as a multiple instead.
#[inline]
fn central_quantile_ratio(q: f64) -> f64 {
    let r = 0.180625 - q * q;
    const A: [f64; 8] = [
        2.5090809287301226727e+3,
        3.3430575583588128105e+4,
        6.7265770927008700853e+4,
        4.5921953931549871457e+4,
        1.3731693765509461125e+4,
        1.9715909503065514427e+3,
        1.3314166789178437745e+2,
        3.3871328727963666080e+0,
    ];
    const B: [f64; 8] = [
        5.2264952788528545610e+3,
        2.8729085735721942674e+4,
        3.9307895800092710610e+4,
        2.1213794301586595867e+4,
        5.3941960214247511077e+3,
        6.8718700749205790830e+2,
        4.2313330701600911252e+1,
        1.0,
    ];
    horner(&A, r) / horner(&B, r)
}

/// AS241 `PPND16`: the standard normal quantile.
///
/// Three regions, each with its own rational approximation:
/// the central body `|p - 1/2| <= 0.425`, where the function is well behaved
/// and a ratio in `q^2` suffices; and two tail regions parameterised by
/// `sqrt(-ln r)`, split at `r = 5` because a single approximation cannot hold
/// its accuracy across the whole tail. The split points and coefficients are
/// Wichura's and are not adjustable — they were fitted together.
fn standard_normal_quantile(p: f64) -> f64 {
    if p.is_nan() || p <= 0.0 || p >= 1.0 {
        return f64::NAN;
    }

    let q = p - 0.5;
    if q.abs() <= CENTRAL_HALF_WIDTH {
        return q * central_quantile_ratio(q);
    }

    // Work with whichever tail `p` is in, then put the sign back. `r` is the
    // distance from the nearer end of the interval.
    let r = if q < 0.0 { p } else { 1.0 - p };
    let r = (-r.ln()).sqrt();

    let value = if r <= 5.0 {
        let r = r - 1.6;
        const C: [f64; 8] = [
            7.74545014278341407640e-4,
            2.27238449892691845833e-2,
            2.41780725177450611770e-1,
            1.27045825245236838258e+0,
            3.64784832476320460504e+0,
            5.76949722146069140550e+0,
            4.63033784615654529590e+0,
            1.42343711074968357734e+0,
        ];
        const D: [f64; 8] = [
            1.05075007164441684324e-9,
            5.47593808499534494600e-4,
            1.51986665636164571966e-2,
            1.48103976427480074590e-1,
            6.89767334985100004550e-1,
            1.67638483018380384940e+0,
            2.05319162663775882187e+0,
            1.0,
        ];
        horner(&C, r) / horner(&D, r)
    } else {
        let r = r - 5.0;
        const E: [f64; 8] = [
            2.01033439929228813265e-7,
            2.71155556874348757815e-5,
            1.24266094738807843860e-3,
            2.65321895265761230930e-2,
            2.96560571828504891230e-1,
            1.78482653991729133580e+0,
            5.46378491116411436990e+0,
            6.65790464350110377720e+0,
        ];
        const F: [f64; 8] = [
            2.04426310338993978564e-15,
            1.42151175831644588870e-7,
            1.84631831751005468180e-5,
            7.86869131145613259100e-4,
            1.48753612908506148525e-2,
            1.36929880922735805310e-1,
            5.99832206555887937690e-1,
            1.0,
        ];
        horner(&E, r) / horner(&F, r)
    };

    if q < 0.0 { -value } else { value }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Where the AS241 estimate alone was 3.5 to 4 ulps out; the Newton step
    /// brings each within one and a half. References are 200-bit values at
    /// the doubles passed.
    #[test]
    fn erf_inverse_is_within_an_ulp_and_a_half() {
        let cases = [
            (0.06587005607025587, 0.05844228570630508),
            (0.12454395388079771, 0.11082627821799679),
            (0.7543030212444997, 0.8208552102059348),
            (0.9835752494902221, 1.6965880954774557),
            (0.9998792938953495, 2.71860468547257),
            (5.339734293178689e-07, 4.7322163053787586e-07),
            (-0.5, -0.4769362762044699),
            (0.9999999999999805, 5.41192654423848),
        ];
        for (x, want) in cases {
            let got: f64 = erf_inverse(x);
            let ulps = (got - want).abs() / (want.abs() * f64::EPSILON / 2.0);
            assert!(
                ulps <= 3.0,
                "erf_inverse({x}) = {got}, wanted {want} ({ulps} half-ulps)"
            );
        }
    }

    /// Values from the standard normal table, to the digits a table carries.
    #[test]
    fn cdf_matches_known_values() {
        assert!((normal_cdf(0.0, 0.0, 1.0) - 0.5).abs() < 1e-15);
        assert!((normal_cdf(1.0, 0.0, 1.0) - 0.841_344_746_068_543).abs() < 1e-14);
        assert!((normal_cdf(-1.96, 0.0, 1.0) - 0.024_997_895_148_220_43).abs() < 1e-15);
        assert!((normal_cdf(2.5, 1.0, 2.0) - 0.773_372_647_623_132_4).abs() < 1e-14);
    }

    /// The tail is where a `1 - Phi(-x)` implementation loses everything, so
    /// it gets its own check against the known asymptotic values.
    #[test]
    fn cdf_keeps_precision_in_the_far_tail() {
        // Phi(-8) and Phi(-20), to the precision a double can hold.
        assert!((normal_cdf(-8.0, 0.0, 1.0) / 6.220_960_574_271_784e-16 - 1.0).abs() < 1e-12);
        assert!((normal_cdf(-20.0, 0.0, 1.0) / 2.753_624_e-89 - 1.0).abs() < 1e-5);
        // And it never underflows to a value the sampler would treat as zero
        // mass while the interval genuinely has some.
        assert!(normal_cdf(-30.0, 0.0, 1.0) > 0.0);
    }

    #[test]
    fn quantile_matches_known_values() {
        assert!((standard_normal_quantile(0.5)).abs() < 1e-15);
        assert!((standard_normal_quantile(0.975) - 1.959_963_984_540_054).abs() < 1e-13);
        assert!((standard_normal_quantile(0.025) + 1.959_963_984_540_054).abs() < 1e-13);
        assert!((standard_normal_quantile(0.99) - 2.326_347_874_040_841).abs() < 1e-13);
    }

    /// The strongest available check without a reference implementation to
    /// compare against: the two functions must invert each other, from the
    /// body all the way into the lower tail, to near machine precision.
    ///
    /// The sweep stops at zero, and not because the upper half is untested —
    /// `quantile_round_trips_through_the_cdf` covers it from the other side.
    /// It stops because `z -> Phi(z) -> z` is not a well-conditioned round trip
    /// above the mean, and no implementation could make it one. At `z = 6`,
    /// `Phi(z)` is `1 - 1e-9`: doubles near 1 are spaced 2.2e-16 apart, and the
    /// density there is 6.1e-9, so simply *storing* the probability discards
    /// about 3.6e-8 of the argument. Testing against that would be testing the
    /// float format. Below the mean the probability is small and carries its
    /// full relative precision, so the round trip is meaningful and tight.
    #[test]
    fn cdf_and_quantile_invert_each_other() {
        let mut z = -37.0;
        while z <= 0.0 {
            let p = normal_cdf(z, 0.0, 1.0);
            if p > 0.0 && p < 1.0 {
                let round_trip = standard_normal_quantile(p);
                // Relative, because `z` spans nine orders of magnitude of `p`.
                assert!(
                    (round_trip - z).abs() <= 1e-12 * z.abs().max(1.0),
                    "round trip failed at z={z}: p={p}, back to {round_trip}"
                );
            }
            z += 0.25;
        }
    }

    /// The upper tail is the lower tail reflected, and the implementation says
    /// so explicitly, so the sign handling is worth asserting rather than
    /// assuming.
    ///
    /// Only down to `p = 1e-3`, for the reason the round-trip test gives from
    /// the other direction: `1.0 - p` for tiny `p` is not `1 - p`. At
    /// `p = 1e-12` the subtraction perturbs the complement by one part in
    /// 10^4, which moves the quantile by 3e-6 — a fact about doubles, not about
    /// this function. Above 1e-3 the complement is exact to within 1e-13 of
    /// the argument and the assertion is about the code again.
    #[test]
    fn quantile_is_antisymmetric() {
        for &p in &[1e-3, 0.01, 0.2, 0.49] {
            let lower = standard_normal_quantile(p);
            let upper = standard_normal_quantile(1.0 - p);
            assert!(
                (lower + upper).abs() <= 1e-9 * lower.abs().max(1.0),
                "quantile({p}) = {lower} but quantile(1 - {p}) = {upper}"
            );
        }
    }

    #[test]
    fn quantile_round_trips_through_the_cdf() {
        for &p in &[
            1e-300,
            1e-100,
            1e-20,
            1e-8,
            0.001,
            0.1,
            0.3,
            0.5,
            0.7,
            0.9,
            0.999,
            1.0 - 1e-8,
        ] {
            let z = standard_normal_quantile(p);
            let back = normal_cdf(z, 0.0, 1.0);
            assert!((back / p - 1.0).abs() < 1e-12, "p={p} -> z={z} -> {back}");
        }
    }

    /// `erf` is in `libm`, so the inverse can be checked against the function
    /// it inverts rather than against a table -- across the body and into both
    /// tails, where the naive `(1 + x) / 2` spelling would have nothing left.
    #[test]
    fn erf_inverse_inverts_erf() {
        let mut x = -0.999_999;
        while x <= 0.999_999 {
            let z = erf_inverse(x);
            let back = libm::erf(z);
            assert!(
                (back - x).abs() < 1e-14,
                "erf_inverse({x}) = {z}, erf of that is {back}"
            );
            x += 0.001;
        }
    }

    /// The reason for the complement spelling: at `1 - 1e-17` the naive form
    /// rounds its argument to exactly 1 and returns infinity for a finite
    /// input. These are the values where that shows.
    #[test]
    fn erf_inverse_keeps_its_footing_near_one() {
        for &x in &[1.0 - 1e-10, 1.0 - 1e-15, -(1.0 - 1e-10), -(1.0 - 1e-15)] {
            let z = erf_inverse(x);
            assert!(z.is_finite(), "erf_inverse({x}) returned {z}");
            assert!(
                (libm::erf(z) - x).abs() < 1e-14,
                "erf_inverse({x}) = {z} does not invert"
            );
        }
    }

    /// `erfinv(x) = sqrt(pi)/2 (x + pi x^3 / 12 + ...)`, so below `1e-8` the
    /// cubic term is under a part in `1e16` and the leading one is the answer.
    /// The complement spelling lost half the digits at `1e-8` and returned
    /// negative zero below `1.1e-16`; the smallest subnormal rounds to itself.
    #[test]
    fn erf_inverse_keeps_its_digits_and_sign_near_zero() {
        // sqrt(pi) / 2, to more digits than a double holds.
        let lead = 0.886_226_925_452_758_013_649_083_741_671_f64;
        for &x in &[1e-8, 3e-12, 1e-16, 1e-20, 1e-300, 2.2250738585072014e-308] {
            for x in [x, -x] {
                let z = erf_inverse(x);
                let expected = lead * x;
                assert!(
                    ((z - expected) / expected).abs() < 3e-16,
                    "erf_inverse({x}) = {z}, expected {expected}"
                );
            }
        }
        assert_eq!(erf_inverse(5e-324), 5e-324);
        assert_eq!(erf_inverse(-5e-324), -5e-324);
        assert!(erf_inverse(-0.0).is_sign_negative());
        assert!(erf_inverse(0.0).is_sign_positive());
        // Through the central body and either side of the seam where it hands
        // over to the tail, against values carried to 25 digits. AS241 holds
        // a part in `1e16` or so, which is two or three ulps here.
        for &(x, expected) in &[
            (0.1, 0.088_855_990_494_257_691_974_280_41),
            (0.5, 0.476_936_276_204_469_873_381_418_4),
            (0.849_999, 1.017_899_967_206_109_391_368_224),
            (0.85, 1.017_902_464_832_027_588_150_153),
            (0.850_001, 1.017_904_962_470_645_475_953_546),
        ] {
            for (x, expected) in [(x, expected), (-x, -expected)] {
                let z = erf_inverse(x);
                assert!(
                    ((z - expected) / expected).abs() < 4.5e-16,
                    "erf_inverse({x}) = {z}, expected {expected}"
                );
            }
        }
    }

    #[test]
    fn erf_inverse_is_odd_and_zero_at_zero() {
        assert_eq!(erf_inverse(0.0), 0.0);
        for &x in &[0.1, 0.5, 0.9, 0.99] {
            assert!((erf_inverse(x) + erf_inverse(-x)).abs() < 1e-15);
        }
    }

    #[test]
    fn erf_inverse_is_infinite_at_the_endpoints_and_nan_past_them() {
        assert_eq!(erf_inverse(1.0), f64::INFINITY);
        assert_eq!(erf_inverse(-1.0), f64::NEG_INFINITY);
        assert!(erf_inverse(1.5).is_nan());
        assert!(erf_inverse(-1.000_001).is_nan());
        assert!(erf_inverse(f64::NAN).is_nan());
    }

    #[test]
    fn quantile_scales_and_shifts() {
        let x = normal_quantile(0.975, 3.0, 2.0);
        assert!((x - (3.0 + 2.0 * 1.959_963_984_540_054)).abs() < 1e-12);
    }

    /// Anything outside the open unit interval has no finite answer, and a
    /// caller filling a tensor needs to see that rather than a stray number.
    #[test]
    fn quantile_rejects_endpoints_and_nonsense() {
        assert!(standard_normal_quantile(0.0).is_nan());
        assert!(standard_normal_quantile(1.0).is_nan());
        assert!(standard_normal_quantile(-0.1).is_nan());
        assert!(standard_normal_quantile(1.5).is_nan());
        assert!(standard_normal_quantile(f64::NAN).is_nan());
    }
}
