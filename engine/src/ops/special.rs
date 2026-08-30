// Copyright (c) Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

//! The special functions: a named kernel and a named derivative each.
//!
//! `exp2`, `logit`, `sinc`, `lgamma`, `digamma` and `erfinv` have nothing in
//! common except their shape -- one value per element, one chain rule per
//! element -- which is the shape the activation units already run on. So they
//! borrow that skeleton rather than restating the dtype dispatch, the buffer
//! walk and the gradient node six more times, and what is written here is only
//! the mathematics: the function, its derivative, and the reason each is
//! evaluated the way it is.
//!
//! Two of them are worth reading closely. `sinc` cannot differentiate itself
//! from its own quotient near zero, so it switches to the series there;
//! `digamma` needs `trigamma` for its gradient, and nothing in the crate's
//! dependencies has one.

use crate::{
    error::{MinitensorError, Result},
    ops::activation::units::{
        UnitGradKernel, UnitKernel, UnitParams, unary_unit, unit_grad_kernel, unit_kernel,
    },
    tensor::Tensor,
};
use statrs::function::{
    erf::erf_inv,
    gamma::{digamma as digamma_scalar, ln_gamma},
};
use std::f64::consts::{LN_2, PI};

/// Defines a [`UnitKernel`] from a single `f64` body, with the `f32` arm
/// widening, evaluating there, and rounding once at the end.
///
/// The functions below that reach for this one are the ones whose kernel comes
/// from a library that only offers `f64`. Rounding a correctly-rounded `f64`
/// answer down to `f32` is within half an ulp of the `f32` answer, which is
/// better than a native `f32` implementation of a special function usually
/// manages, so this is the accurate arm rather than a fallback.
macro_rules! wide_kernel {
    ($(#[$meta:meta])* $name:ident, |$x:pat_param, $p:pat_param| $body:expr) => {
        $(#[$meta])*
        const $name: UnitKernel = {
            #[inline(always)]
            fn wide($x: f64, $p: UnitParams<f64>) -> f64 {
                $body
            }
            #[inline(always)]
            fn narrow(x: f32, p: UnitParams<f32>) -> f32 {
                wide(x as f64, [p[0] as f64, p[1] as f64]) as f32
            }
            (narrow, wide)
        };
    };
}

/// [`wide_kernel!`] for a chain rule, which also takes the incoming gradient.
macro_rules! wide_grad_kernel {
    ($(#[$meta:meta])* $name:ident, |$x:pat_param, $g:pat_param, $p:pat_param| $body:expr) => {
        $(#[$meta])*
        const $name: UnitGradKernel = {
            #[inline(always)]
            fn wide($x: f64, $g: f64, $p: UnitParams<f64>) -> f64 {
                $body
            }
            #[inline(always)]
            fn narrow(x: f32, g: f32, p: UnitParams<f32>) -> f32 {
                wide(x as f64, g as f64, [p[0] as f64, p[1] as f64]) as f32
            }
            (narrow, wide)
        };
    };
}

// --- exp2 ------------------------------------------------------------------

unit_kernel!(
    /// `2^x`, from the hardware's own base-2 exponential rather than
    /// `exp(x * ln 2)`, which rounds the exponent before using it.
    EXP2, |x, _p| x.exp2()
);
unit_grad_kernel!(
    /// `d/dx 2^x = 2^x ln 2`, with `ln 2` arriving as a parameter so it is
    /// converted to the working width once per call, not once per element.
    EXP2_D, |x, g, p| g * x.exp2() * p[0]
);

/// `2^x`, element-wise.
pub fn exp2(tensor: &Tensor) -> Result<Tensor> {
    unary_unit(tensor, "exp2", EXP2, EXP2_D, [LN_2, 0.0])
}

// --- logit -----------------------------------------------------------------

unit_kernel!(
    /// `log(x / (1 - x))`, the inverse of `sigmoid`.
    ///
    /// `p[0]` is the clamp: inputs are pulled into `[eps, 1 - eps]` first, so
    /// a probability that has reached 0 or 1 by rounding gives a large finite
    /// number instead of an infinity. A NaN there means no clamp was asked
    /// for -- there is no epsilon that is not a number -- and then an input
    /// outside `[0, 1]` gives NaN, as the formula says it should.
    LOGIT, |x, p| {
        let z = if p[0].is_nan() {
            x
        } else {
            x.clamp(p[0], 1.0 - p[0])
        };
        (z / (1.0 - z)).ln()
    }
);
unit_grad_kernel!(
    /// `d/dx logit(x) = 1 / (x(1 - x))`, and zero wherever the clamp took
    /// over: there the output does not vary with the input at all.
    LOGIT_D, |x, g, p| {
        if !p[0].is_nan() && (x < p[0] || x > 1.0 - p[0]) {
            0.0
        } else {
            g / (x * (1.0 - x))
        }
    }
);

/// `log(input / (1 - input))`, the inverse of `sigmoid`.
///
/// `eps` clamps the input into `[eps, 1 - eps]` before the logarithm, which
/// bounds the result for a probability that has rounded to 0 or 1. Without it
/// those give infinities, and anything outside `[0, 1]` gives NaN.
pub fn logit(tensor: &Tensor, eps: Option<f64>) -> Result<Tensor> {
    let clamp = match eps {
        None => f64::NAN,
        // A NaN eps is refused along with the out-of-range ones: it compares
        // false against both bounds, so the negation catches it, and "not a
        // number" is not the same request as "no clamp".
        Some(eps) if !(0.0..=0.5).contains(&eps) => {
            return Err(MinitensorError::invalid_argument(format!(
                "logit requires eps in [0, 0.5], got {eps}"
            )));
        }
        Some(eps) => eps,
    };
    unary_unit(tensor, "logit", LOGIT, LOGIT_D, [clamp, 0.0])
}

// --- sinc ------------------------------------------------------------------

unit_kernel!(
    /// `sin(pi x) / (pi x)`, and `1` at the origin, where the quotient is
    /// `0 / 0` but the limit is not.
    ///
    /// `p[0]` carries pi at the working width. The quotient loses nothing near
    /// zero -- `sin(u)` and `u` agree there to the last bit, so the ratio is
    /// accurate -- which is why only the derivative below needs a series.
    SINC, |x, p| {
        let u = p[0] * x;
        if u == 0.0 { 1.0 } else { u.sin() / u }
    }
);
unit_grad_kernel!(
    /// `d/dx sinc(x) = pi (u cos u - sin u) / u^2`, with `u = pi x`.
    ///
    /// The numerator is the difference of two quantities that agree to their
    /// first several digits near zero: at `u = 0.01` both are within 1e-6 of
    /// each other and their difference is 3e-7, so subtracting them throws
    /// away most of the mantissa. Below `|u| = 0.1` the series
    /// `pi u (-1/3 + u^2/30 - u^4/840 + u^6/45360)` computes the same value
    /// with no cancellation at all; its first dropped term is `u^8 / 1330560`
    /// relative to the leading one, under 1e-14 at the cutoff.
    SINC_D, |x, g, p| {
        let pi = p[0];
        let u = pi * x;
        let derivative = if u.abs() < 0.1 {
            let square = u * u;
            pi * u
                * (-1.0 / 3.0
                    + square * (1.0 / 30.0 + square * (-1.0 / 840.0 + square / 45360.0)))
        } else {
            pi * (u * u.cos() - u.sin()) / (u * u)
        };
        g * derivative
    }
);

/// `sin(pi * input) / (pi * input)`, taken as `1` at zero.
pub fn sinc(tensor: &Tensor) -> Result<Tensor> {
    unary_unit(tensor, "sinc", SINC, SINC_D, [PI, 0.0])
}

// --- lgamma and digamma ----------------------------------------------------

wide_kernel!(
    /// `log |gamma(x)|`, which stays finite where `gamma` itself overflows:
    /// `gamma(200)` is past the top of float64 and its logarithm is 857.
    LGAMMA, |x, _p| libm::lgamma(x)
);
wide_grad_kernel!(
    /// `d/dx log gamma(x) = digamma(x)`, which is the definition of digamma.
    LGAMMA_D, |x, g, _p| g * digamma_scalar(x)
);
wide_kernel!(
    /// `digamma(x)`, the logarithmic derivative of the gamma function.
    DIGAMMA, |x, _p| digamma_scalar(x)
);
wide_grad_kernel!(
    /// `d/dx digamma(x) = trigamma(x)`.
    DIGAMMA_D, |x, g, _p| g * trigamma(x)
);

/// The Bernoulli numbers `B_2j` for `j = 1..=7`, which are the coefficients of
/// the Euler-Maclaurin expansion below and the only constants it needs.
const BERNOULLI: [f64; 7] = [
    1.0 / 6.0,
    -1.0 / 30.0,
    1.0 / 42.0,
    -1.0 / 30.0,
    5.0 / 66.0,
    -691.0 / 2730.0,
    7.0 / 6.0,
];

/// Where the recurrence stops and the expansion below takes over, for a zeta
/// of order `s`.
///
/// Twelve is where the first dropped term falls under 1e-14 of the answer for
/// the small orders; the higher ones need the argument to lead the order,
/// because the expansion's terms carry a rising factorial in `s`. Stopping at
/// six, as some implementations do, leaves a thousand times more error.
fn expansion_threshold(s: f64) -> f64 {
    (s + 7.0).max(12.0)
}

/// `zeta(s, a)` divided by `a^-s`, for an `a` already large enough, by
/// Euler-Maclaurin.
///
/// The expansion is
/// `a^(1-s)/(s-1) + 1/(2 a^s) + sum_j B_2j/(2j)! (s)_(2j-1) a^-(s+2j-1)`,
/// where `(s)_m` is the rising factorial. Dividing it by the scale of its
/// leading term leaves every term of order one, which is the point: at a high
/// order, `a^-s` underflows long before the answer does, and the factorial in
/// front of it would have restored the scale had there been anything left to
/// restore. Kept apart, the two are combined in the exponent instead.
fn hurwitz_bracket(s: f64, a: f64) -> f64 {
    let mut total = a / (s - 1.0) + 0.5;
    let mut rising = s;
    let mut factorial = 1.0;
    for (index, bernoulli) in BERNOULLI.iter().enumerate() {
        let two_j = 2.0 * (index as f64 + 1.0);
        factorial *= two_j * (two_j - 1.0);
        total += bernoulli / factorial * rising * a.powf(-(two_j - 1.0));
        rising *= (s + two_j - 1.0) * (s + two_j);
    }
    total
}

/// `zeta(s, x)` divided by `|x|^-s`, and nothing else.
///
/// Everything is measured against the first term, which is the largest one for
/// a positive argument, so the sum stays near one and the scale lives in the
/// exponent outside. That is what lets a high order work at all: `n!` sits near
/// the top of the range and `x^-s` near the bottom, and multiplying them loses
/// the answer to overflow or underflow on the way to a value that fits.
fn scaled_zeta(order: u32, x: f64) -> f64 {
    let exponent = order as i32 + 1;
    let threshold = expansion_threshold(exponent as f64);
    let steps = if x < threshold {
        (threshold - x).ceil() as i64
    } else {
        0
    };
    let argument = x + steps as f64;
    let reference = x.abs();

    // `(x + k)^-s / |x|^-s`, which `powi` keeps the sign of when the argument
    // starts out negative and the recurrence has yet to walk it up.
    let mut total = 0.0;
    for step in 0..steps {
        total += (reference / (x + step as f64)).powi(exponent);
    }
    total + (reference / argument).powi(exponent) * hurwitz_bracket(exponent as f64, argument)
}

/// `polygamma(order, x)` for an order of one or more and an argument that is
/// neither a pole nor beyond [`RECURRENCE_FLOOR`].
fn polygamma_series(order: u32, x: f64) -> f64 {
    let s = order as f64 + 1.0;
    let sign = if order.is_multiple_of(2) { -1.0 } else { 1.0 };
    // `n! * |x|^-s` in the exponent rather than as a product, for the reason
    // `scaled_zeta` divides by `|x|^-s` in the first place.
    sign * (ln_gamma(s) - s * x.abs().ln()).exp() * scaled_zeta(order, x)
}

/// `trigamma(x)`, the derivative of [`digamma`].
///
/// The negative half is reflected rather than walked, which is what keeps a
/// large negative argument a single step instead of a million of them:
/// `trigamma(x) + trigamma(1 - x) = pi^2 / sin^2(pi x)`, and `1 - x` is above
/// one for every negative `x`, so this recurses exactly once.
fn trigamma(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x <= 0.0 && x.floor() == x {
        // Double poles at the non-positive integers: locally `1/(x - n)^2`,
        // which rises to `+inf` from both sides rather than changing sign.
        return f64::INFINITY;
    }
    if x < 0.0 {
        let sine = (PI * x).sin();
        return PI * PI / (sine * sine) - trigamma(1.0 - x);
    }
    polygamma_series(1, x)
}

/// The largest order this can carry.
///
/// `polygamma` is `(-1)^(n+1) n! zeta(n + 1, x)`, so the factorial has to be
/// finite -- and the gradient is the next order up, which needs one more.
/// `170!` is the last that fits a double, so `169` is the last order whose
/// derivative is still computable.
pub const POLYGAMMA_MAX_ORDER: u32 = 169;

/// `polygamma(order, x)`, the `order`-th derivative of [`digamma`].
///
/// `(-1)^(n+1) n! zeta(n + 1, x)`, with the Hurwitz zeta split the way every
/// implementation splits it: walk the argument up by the recurrence until the
/// asymptotic expansion has converged, then evaluate it there.
///
/// Orders zero and one come from `digamma` and `trigamma`, which are the same
/// function with a better route for negative arguments.
fn polygamma_scalar(order: u32, x: f64) -> f64 {
    match order {
        0 => return digamma_scalar(x),
        1 => return trigamma(x),
        _ => {}
    }
    if x.is_nan() {
        return f64::NAN;
    }
    // The sign is the one the limit takes from the right, which is what
    // `digamma` and `trigamma` already answer at their own poles.
    let sign = if order.is_multiple_of(2) { -1.0 } else { 1.0 };
    if x == 0.0 {
        return sign * f64::INFINITY;
    }
    if x < 0.0 {
        // Above order one the negative half is not computed, and NaN says so
        // rather than returning digits that are not there. Walking the
        // recurrence up from a negative argument sums terms that are enormous
        // beside the answer and alternate in sign: at order six and `x = -10.5`
        // that already costs eight digits, and by `x = -100.5` there is nothing
        // left. What avoids it is the reflection formula, and for a general
        // order that needs the `n`-th derivative of the cotangent -- a
        // polynomial in `cot(pi x)` whose coefficients overflow well before the
        // orders here do. `scipy` stops at the same place, for the same reason:
        // its `zeta(s, q)` is defined for positive `q` only.
        //
        // Orders zero and one keep the whole line, because `digamma` and
        // `trigamma` reach it by routes this one does not have.
        return f64::NAN;
    }
    polygamma_series(order, x)
}

/// `log |gamma(input)|`, element-wise.
pub fn lgamma(tensor: &Tensor) -> Result<Tensor> {
    unary_unit(tensor, "lgamma", LGAMMA, LGAMMA_D, [0.0; 2])
}

/// `digamma(input)`, the derivative of `lgamma`.
pub fn digamma(tensor: &Tensor) -> Result<Tensor> {
    unary_unit(tensor, "digamma", DIGAMMA, DIGAMMA_D, [0.0; 2])
}

wide_kernel!(
    /// `polygamma(order, x)`, with the order carried as a parameter.
    POLYGAMMA, |x, p| polygamma_scalar(p[0] as u32, x)
);
wide_grad_kernel!(
    /// `d/dx polygamma(n, x) = polygamma(n + 1, x)`.
    ///
    /// The derivative of a polygamma is the next one, so the chain rule needs
    /// no second formula and no finite difference -- which is also why the
    /// order is capped one below where the factorial stops fitting.
    POLYGAMMA_D, |x, g, p| g * polygamma_scalar(p[0] as u32 + 1, x)
);

/// `polygamma(order, input)`, the `order`-th derivative of `digamma`.
///
/// Order zero is `digamma` itself and order one is `trigamma`.
pub fn polygamma(order: i64, tensor: &Tensor) -> Result<Tensor> {
    if order < 0 {
        return Err(MinitensorError::invalid_argument(format!(
            "polygamma takes a non-negative order, got {order}"
        )));
    }
    if order > POLYGAMMA_MAX_ORDER as i64 {
        return Err(MinitensorError::invalid_argument(format!(
            "polygamma is defined here up to order {POLYGAMMA_MAX_ORDER}, where the \
             factorial in its definition and in its derivative both still fit a \
             double, got {order}"
        )));
    }
    unary_unit(
        tensor,
        "polygamma",
        POLYGAMMA,
        POLYGAMMA_D,
        [order as f64, 0.0],
    )
}

// --- erfinv ----------------------------------------------------------------

wide_kernel!(
    /// The inverse of `erf` on `[-1, 1]`, infinite at the endpoints.
    ///
    /// Outside that interval there is nothing to invert, and NaN says so.
    /// The underlying routine reports an infinity for any argument past the
    /// endpoints, which would claim `erf` reaches 2 somewhere.
    ERFINV, |x, _p| {
        if !(-1.0..=1.0).contains(&x) {
            f64::NAN
        } else {
            erf_inv(x)
        }
    }
);
wide_grad_kernel!(
    /// `d/dx erfinv(x) = (sqrt(pi) / 2) exp(erfinv(x)^2)`, from
    /// differentiating `erf(erfinv(x)) = x`.
    ERFINV_D, |x, g, _p| {
        if !(-1.0..=1.0).contains(&x) {
            f64::NAN
        } else {
            let y = erf_inv(x);
            g * (PI.sqrt() * 0.5) * (y * y).exp()
        }
    }
);

/// The inverse error function on `[-1, 1]`.
pub fn erfinv(tensor: &Tensor) -> Result<Tensor> {
    unary_unit(tensor, "erfinv", ERFINV, ERFINV_D, [0.0; 2])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::vector;
    use crate::{
        autograd::backward_collect,
        device::Device,
        tensor::{DataType, Shape, TensorData},
    };
    use std::sync::Arc;

    fn f64_tensor(data: Vec<f64>) -> Tensor {
        vector(data)
    }

    fn f32_tensor(data: Vec<f32>) -> Tensor {
        vector(data)
    }

    fn wide(tensor: &Tensor) -> Vec<f64> {
        tensor.data().as_f64_slice().unwrap().to_vec()
    }

    fn narrow(tensor: &Tensor) -> Vec<f32> {
        tensor.data().as_f32_slice().unwrap().to_vec()
    }

    /// The gradient the op reports, for a seed of ones.
    fn gradient(op: fn(&Tensor) -> Result<Tensor>, values: &[f64]) -> Vec<f64> {
        let input = f64_tensor(values.to_vec()).requires_grad_(true);
        let output = op(&input).unwrap();
        let seed = Tensor::ones(
            output.shape().clone(),
            output.dtype(),
            output.device(),
            false,
        );
        let grads = backward_collect(&output, Some(seed)).unwrap();
        wide(grads.get(&input.id()).unwrap())
    }

    /// The gradient central differences report, for the same seed.
    fn numeric_gradient(op: fn(&Tensor) -> Result<Tensor>, values: &[f64], eps: f64) -> Vec<f64> {
        values
            .iter()
            .map(|&v| {
                let up = wide(&op(&f64_tensor(vec![v + eps])).unwrap())[0];
                let down = wide(&op(&f64_tensor(vec![v - eps])).unwrap())[0];
                (up - down) / (2.0 * eps)
            })
            .collect()
    }

    fn assert_close(got: &[f64], want: &[f64], tolerance: f64, what: &str) {
        assert_eq!(got.len(), want.len(), "{what}: length");
        for (index, (&g, &w)) in got.iter().zip(want).enumerate() {
            let scale = w.abs().max(1.0);
            assert!(
                (g - w).abs() <= tolerance * scale,
                "{what}[{index}]: {g} vs {w}"
            );
        }
    }

    #[test]
    fn exp2_is_two_to_the_power_and_differentiates_to_itself_times_ln_two() {
        let values = [-3.5, -1.0, 0.0, 0.5, 4.0, 10.25];
        let got = wide(&exp2(&f64_tensor(values.to_vec())).unwrap());
        let want: Vec<f64> = values.iter().map(|v| 2f64.powf(*v)).collect();
        assert_close(&got, &want, 1e-15, "exp2");

        let analytic = gradient(exp2, &values);
        let expected: Vec<f64> = want.iter().map(|v| v * LN_2).collect();
        assert_close(&analytic, &expected, 1e-14, "exp2 gradient");
    }

    #[test]
    fn logit_inverts_sigmoid() {
        // Round-tripping is the whole claim, so it is what the test makes.
        let probabilities = [0.001, 0.1, 0.5, 0.75, 0.999];
        let logits = wide(&logit(&f64_tensor(probabilities.to_vec()), None).unwrap());
        let back = wide(&crate::ops::activation::sigmoid(&f64_tensor(logits)).unwrap());
        assert_close(&back, &probabilities, 1e-14, "sigmoid(logit(p))");
    }

    #[test]
    fn logit_without_an_eps_is_infinite_at_the_endpoints_and_nan_outside() {
        let got = wide(&logit(&f64_tensor(vec![0.0, 1.0, -0.5, 1.5]), None).unwrap());
        assert_eq!(got[0], f64::NEG_INFINITY);
        assert_eq!(got[1], f64::INFINITY);
        assert!(got[2].is_nan() && got[3].is_nan());
    }

    #[test]
    fn an_eps_bounds_the_endpoints_and_flattens_the_gradient_there() {
        let eps = 1e-6;
        let got = wide(&logit(&f64_tensor(vec![0.0, 1.0, 0.5]), Some(eps)).unwrap());
        // Each endpoint becomes the logit of the bound it was pulled to.
        let low = eps;
        let high = 1.0 - eps;
        assert!((got[0] - (low / (1.0 - low)).ln()).abs() < 1e-12);
        assert!((got[1] - (high / (1.0 - high)).ln()).abs() < 1e-12);
        // Which is symmetric about zero, to the precision `1 - eps` survives
        // at: the bounds are mirror images and the logit is an odd function
        // about a half.
        assert!((got[0] + got[1]).abs() < 1e-9);
        assert_eq!(got[2], 0.0);

        let input = f64_tensor(vec![0.0, 0.5, 1.0]).requires_grad_(true);
        let output = logit(&input, Some(eps)).unwrap();
        let seed = Tensor::ones(
            output.shape().clone(),
            output.dtype(),
            output.device(),
            false,
        );
        let grad = wide(
            backward_collect(&output, Some(seed))
                .unwrap()
                .get(&input.id())
                .unwrap(),
        );
        // Clamped on both ends, so the output does not move with the input.
        assert_eq!(grad[0], 0.0);
        assert_eq!(grad[2], 0.0);
        assert!((grad[1] - 4.0).abs() < 1e-12, "1 / (0.5 * 0.5)");
    }

    #[test]
    fn an_eps_outside_zero_to_a_half_is_refused() {
        // A clamp to `[eps, 1 - eps]` with `eps > 0.5` has an empty interval,
        // and `f64::clamp` panics on inverted bounds rather than reporting it.
        for bad in [-1e-9, 0.5 + 1e-9, 1.0, f64::NAN] {
            assert!(logit(&f64_tensor(vec![0.5]), Some(bad)).is_err(), "{bad}");
        }
        assert!(logit(&f64_tensor(vec![0.5]), Some(0.5)).is_ok());
    }

    #[test]
    fn sinc_is_one_at_the_origin_and_zero_at_every_other_integer() {
        let values = [0.0, 1.0, 2.0, -3.0, 0.5, -0.5, 1.5];
        let got = wide(&sinc(&f64_tensor(values.to_vec())).unwrap());
        let want: Vec<f64> = values
            .iter()
            .map(|&x| {
                if x == 0.0 {
                    1.0
                } else {
                    (PI * x).sin() / (PI * x)
                }
            })
            .collect();
        assert_close(&got, &want, 1e-15, "sinc");
        assert_eq!(got[0], 1.0);
        for index in [1, 2, 3] {
            assert!(got[index].abs() < 1e-15, "sinc at an integer");
        }
    }

    #[test]
    fn the_sinc_series_and_the_quotient_agree_where_they_meet() {
        // The gradient switches formula at |pi x| = 0.1. If the two disagreed
        // there the gradient would have a step in it, which no amount of
        // testing either side alone would show.
        // Straddling by 1e-15 rather than something larger: the derivative
        // itself has a slope of about -pi^2/3 there, so over that gap its true
        // value moves by 7e-15, and anything the assertion below catches is a
        // step in the formula rather than the function going about its
        // business.
        let cutoff = 0.1 / PI;
        let just_below = gradient(sinc, &[cutoff - 1e-15])[0];
        let just_above = gradient(sinc, &[cutoff + 1e-15])[0];
        assert!(
            (just_below - just_above).abs() < 1e-13,
            "{just_below} vs {just_above}"
        );

        // And at the origin the derivative is zero by symmetry, which the
        // quotient cannot produce at all -- it is 0/0 there.
        assert_eq!(gradient(sinc, &[0.0])[0], 0.0);
    }

    #[test]
    fn sinc_gradient_matches_central_differences() {
        let values = [-2.3, -0.75, -0.01, 0.02, 0.4, 1.6, 3.25];
        let analytic = gradient(sinc, &values);
        let numeric = numeric_gradient(sinc, &values, 1e-6);
        assert_close(&analytic, &numeric, 1e-8, "sinc gradient");
    }

    #[test]
    fn lgamma_matches_the_values_it_is_named_for() {
        // gamma(1) = gamma(2) = 1, gamma(1/2) = sqrt(pi), gamma(5) = 24.
        let got = wide(&lgamma(&f64_tensor(vec![1.0, 2.0, 0.5, 5.0, 200.0])).unwrap());
        assert!(got[0].abs() < 1e-15 && got[1].abs() < 1e-15);
        assert!((got[2] - PI.sqrt().ln()).abs() < 1e-14);
        assert!((got[3] - 24f64.ln()).abs() < 1e-13);
        // The point of taking the logarithm: gamma(200) is past float64's top.
        assert!(got[4].is_finite() && (got[4] - 857.933_669_825_857).abs() < 1e-9);
    }

    #[test]
    fn lgamma_differentiates_to_digamma() {
        let values = [0.3, 1.0, 2.5, 7.75];
        let analytic = gradient(lgamma, &values);
        let expected = wide(&digamma(&f64_tensor(values.to_vec())).unwrap());
        assert_close(&analytic, &expected, 1e-14, "lgamma gradient");
        assert_close(
            &analytic,
            &numeric_gradient(lgamma, &values, 1e-6),
            1e-8,
            "against central differences",
        );
    }

    #[test]
    fn digamma_matches_the_values_it_is_named_for() {
        const EULER_MASCHERONI: f64 = 0.577_215_664_901_532_9;
        let got = wide(&digamma(&f64_tensor(vec![1.0, 2.0, 0.5])).unwrap());
        assert!((got[0] + EULER_MASCHERONI).abs() < 1e-13);
        // psi(2) = 1 - gamma, from the recurrence psi(x+1) = psi(x) + 1/x.
        assert!((got[1] - (1.0 - EULER_MASCHERONI)).abs() < 1e-13);
        // psi(1/2) = -gamma - 2 log 2.
        assert!((got[2] + EULER_MASCHERONI + 2.0 * 2f64.ln()).abs() < 1e-13);
    }

    #[test]
    fn trigamma_matches_its_closed_forms() {
        let sixth_of_pi_squared = PI * PI / 6.0;
        assert!((trigamma(1.0) - sixth_of_pi_squared).abs() < 1e-14);
        // psi'(x+1) = psi'(x) - 1/x^2.
        assert!((trigamma(2.0) - (sixth_of_pi_squared - 1.0)).abs() < 1e-14);
        assert!((trigamma(0.5) - PI * PI / 2.0).abs() < 1e-13);
        // Reflection: psi'(-1/2) + psi'(3/2) = pi^2 / sin^2(-pi/2) = pi^2.
        assert!((trigamma(-0.5) - (PI * PI / 2.0 + 4.0)).abs() < 1e-12);
        // Large arguments go straight to the expansion.
        assert!((trigamma(100.0) - 0.010_050_166_663_333_571).abs() < 1e-15);
        // The poles.
        assert_eq!(trigamma(0.0), f64::INFINITY);
        assert_eq!(trigamma(-3.0), f64::INFINITY);
        assert!(trigamma(f64::NAN).is_nan());
    }

    #[test]
    fn polygamma_agrees_with_the_two_orders_that_have_their_own_route() {
        // Orders 0 and 1 are `digamma` and `trigamma`; the general path has to
        // be the same function, or the family has a seam in it.
        for &x in &[0.25_f64, 0.5, 1.0, 2.5, 7.0, 40.0] {
            assert!((polygamma_scalar(0, x) - digamma_scalar(x)).abs() < 1e-15);
            assert!((polygamma_scalar(1, x) - trigamma(x)).abs() <= 1e-14 * trigamma(x).abs());
        }
    }

    #[test]
    fn polygamma_matches_its_closed_forms() {
        // `polygamma(n, 1) = (-1)^(n+1) n! zeta(n + 1)`, and the zeta values at
        // small integers are known exactly.
        let zeta_3 = 1.202_056_903_159_594_3;
        let zeta_4 = std::f64::consts::PI.powi(4) / 90.0;
        let zeta_5 = 1.036_927_755_143_37;
        assert!((polygamma_scalar(2, 1.0) - (-2.0 * zeta_3)).abs() < 1e-14);
        assert!((polygamma_scalar(3, 1.0) - (6.0 * zeta_4)).abs() < 1e-13);
        assert!((polygamma_scalar(4, 1.0) - (-24.0 * zeta_5)).abs() < 1e-12);

        // The recurrence, which the walk uses and so cannot itself be right by
        // accident: `polygamma(n, x + 1) = polygamma(n, x) + (-1)^n n! / x^(n+1)`.
        for order in 2..=6_u32 {
            let factorial: f64 = (1..=order).map(|k| k as f64).product();
            let sign = if order.is_multiple_of(2) { 1.0 } else { -1.0 };
            for &x in &[0.3_f64, 1.7, 5.5, 30.0] {
                let term = factorial / x.powi(order as i32 + 1);
                let stepped = polygamma_scalar(order, x + 1.0);
                let expected = polygamma_scalar(order, x) + sign * term;
                // The two sides of the recurrence cancel almost entirely for a
                // small `x`, so the error that matters is measured against what
                // cancelled rather than against the little that survived.
                assert!(
                    (stepped - expected).abs() <= 1e-14 * term.abs().max(stepped.abs()),
                    "order {order} at {x}: {stepped} vs {expected}"
                );
            }
        }
    }

    #[test]
    fn polygamma_takes_the_sign_of_the_limit_from_the_right_at_a_pole() {
        // Which is what `digamma` and `trigamma` already answer at theirs, and
        // is the only sign the two sides agree on for an odd order.
        assert_eq!(polygamma_scalar(2, 0.0), f64::NEG_INFINITY);
        assert_eq!(polygamma_scalar(3, 0.0), f64::INFINITY);
        assert_eq!(polygamma_scalar(0, 0.0), f64::NEG_INFINITY);
        assert_eq!(polygamma_scalar(1, -1.0), f64::INFINITY);
        assert!(polygamma_scalar(4, f64::NAN).is_nan());
    }

    #[test]
    fn above_order_one_the_negative_half_says_nan_rather_than_guessing() {
        for &x in &[-0.5_f64, -1.0, -10.5, -1.0e6] {
            assert!(polygamma_scalar(2, x).is_nan(), "at {x}");
            assert!(polygamma_scalar(7, x).is_nan(), "at {x}");
        }
        // Orders zero and one reach the negative half by routes the general one
        // does not have, and keep it.
        assert!(trigamma(-1.0e9 + 0.5).is_finite());
        assert!(polygamma_scalar(1, -0.5).is_finite());
        assert!(polygamma_scalar(0, -0.5).is_finite());
        // Zero itself is the pole, not the negative half.
        assert_eq!(polygamma_scalar(2, 0.0), f64::NEG_INFINITY);
        assert_eq!(polygamma_scalar(3, 0.0), f64::INFINITY);
    }

    #[test]
    fn polygamma_refuses_an_order_it_cannot_carry() {
        let tensor = f64_tensor(vec![1.0]);
        assert!(polygamma(-1, &tensor).is_err());
        assert!(polygamma(POLYGAMMA_MAX_ORDER as i64 + 1, &tensor).is_err());
        // At the cap both the value and its derivative are still finite.
        assert!(polygamma_scalar(POLYGAMMA_MAX_ORDER, 1.0).is_finite());
        assert!(polygamma_scalar(POLYGAMMA_MAX_ORDER + 1, 1.0).is_finite());
    }

    #[test]
    fn polygamma_differentiates_to_the_next_order() {
        fn second(tensor: &Tensor) -> Result<Tensor> {
            polygamma(2, tensor)
        }
        let values = [0.4, 1.0, 2.5, 9.0];
        let analytic = gradient(second, &values);
        let expected: Vec<f64> = values.iter().map(|&v| polygamma_scalar(3, v)).collect();
        assert_close(&analytic, &expected, 1e-14, "polygamma gradient");
        assert_close(
            &analytic,
            &numeric_gradient(second, &values, 1e-6),
            1e-5,
            "against central differences",
        );
    }

    #[test]
    fn digamma_differentiates_to_trigamma() {
        let values = [0.4, 1.0, 3.5, 9.0];
        let analytic = gradient(digamma, &values);
        let expected: Vec<f64> = values.iter().map(|&v| trigamma(v)).collect();
        assert_close(&analytic, &expected, 1e-14, "digamma gradient");
        assert_close(
            &analytic,
            &numeric_gradient(digamma, &values, 1e-6),
            1e-7,
            "against central differences",
        );
    }

    #[test]
    fn erfinv_inverts_erf() {
        let values = [-0.99, -0.4, 0.0, 0.25, 0.9];
        let inverted = wide(&erfinv(&f64_tensor(values.to_vec())).unwrap());
        let back = wide(&crate::ops::activation::erf(&f64_tensor(inverted)).unwrap());
        assert_close(&back, &values, 1e-13, "erf(erfinv(x))");
    }

    #[test]
    fn erfinv_is_infinite_at_the_endpoints_and_nan_beyond_them() {
        let got = wide(&erfinv(&f64_tensor(vec![1.0, -1.0, 1.5, -2.0])).unwrap());
        assert_eq!(got[0], f64::INFINITY);
        assert_eq!(got[1], f64::NEG_INFINITY);
        assert!(
            got[2].is_nan() && got[3].is_nan(),
            "erf never reaches past 1, so nothing inverts to there"
        );
    }

    #[test]
    fn erfinv_gradient_matches_central_differences() {
        let values = [-0.8, -0.2, 0.0, 0.35, 0.95];
        assert_close(
            &gradient(erfinv, &values),
            &numeric_gradient(erfinv, &values, 1e-6),
            1e-7,
            "erfinv gradient",
        );
    }

    #[test]
    fn the_f32_arm_agrees_with_the_f64_one() {
        let values = [0.25f32, 0.75, 1.5, 3.0];
        let wide_values: Vec<f64> = values.iter().map(|&v| v as f64).collect();
        type Unary = fn(&Tensor) -> Result<Tensor>;
        for (name, op) in [
            ("exp2", exp2 as Unary),
            ("sinc", sinc),
            ("lgamma", lgamma),
            ("digamma", digamma),
        ] {
            let single = narrow(&op(&f32_tensor(values.to_vec())).unwrap());
            let double = wide(&op(&f64_tensor(wide_values.clone())).unwrap());
            for (index, (&s, &d)) in single.iter().zip(&double).enumerate() {
                let scale = (d.abs() as f32).max(1.0);
                assert!(
                    (s - d as f32).abs() <= 1e-6 * scale,
                    "{name}[{index}]: {s} vs {d}"
                );
            }
        }
    }

    #[test]
    fn an_integer_tensor_is_refused_by_name() {
        let integers = Tensor::new(
            Arc::new(TensorData::from_vec_i64(vec![1, 2], Device::cpu())),
            Shape::new(vec![2]),
            DataType::Int64,
            Device::cpu(),
            false,
        );
        for (name, result) in [
            ("exp2", exp2(&integers)),
            ("sinc", sinc(&integers)),
            ("lgamma", lgamma(&integers)),
            ("digamma", digamma(&integers)),
            ("erfinv", erfinv(&integers)),
            ("logit", logit(&integers, None)),
        ] {
            let error = result.unwrap_err().to_string();
            assert!(error.contains(name), "{name} is missing from {error:?}");
        }
    }
}
