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
use statrs::function::{erf::erf_inv, gamma::digamma as digamma_scalar};
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

/// `trigamma(x)`, the derivative of [`digamma`] and the only piece of this
/// family the crate's dependencies do not already have.
///
/// Two identities and one series, which is how every implementation of it
/// works: reflect the negative half onto the positive one, walk up by the
/// recurrence until the asymptotic expansion has converged, and evaluate it
/// there.
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
        // `trigamma(x) + trigamma(1 - x) = pi^2 / sin^2(pi x)`, and `1 - x` is
        // above 1 for every negative `x`, so this recurses exactly once.
        let sine = (PI * x).sin();
        return PI * PI / (sine * sine) - trigamma(1.0 - x);
    }

    // `trigamma(x) = trigamma(x + 1) + 1/x^2` walks the argument up to where
    // the expansion below is worth using. Twelve is where its first dropped
    // term falls under 1e-14 of the answer; stopping at 6, as some
    // implementations do, leaves a thousand times more error than that.
    let mut argument = x;
    let mut total = 0.0;
    while argument < 12.0 {
        total += 1.0 / (argument * argument);
        argument += 1.0;
    }

    // `1/x + 1/(2x^2) + sum B_2n / x^(2n+1)`, factored as `(1/x)(...)` and
    // evaluated by Horner in `1/x^2`.
    let inverse_square = 1.0 / (argument * argument);
    total
        + (1.0 / argument)
            * (1.0
                + 0.5 / argument
                + inverse_square
                    * (1.0 / 6.0
                        + inverse_square
                            * (-1.0 / 30.0
                                + inverse_square
                                    * (1.0 / 42.0
                                        + inverse_square
                                            * (-1.0 / 30.0 + inverse_square * (5.0 / 66.0))))))
}

/// `log |gamma(input)|`, element-wise.
pub fn lgamma(tensor: &Tensor) -> Result<Tensor> {
    unary_unit(tensor, "lgamma", LGAMMA, LGAMMA_D, [0.0; 2])
}

/// `digamma(input)`, the derivative of `lgamma`.
pub fn digamma(tensor: &Tensor) -> Result<Tensor> {
    unary_unit(tensor, "digamma", DIGAMMA, DIGAMMA_D, [0.0; 2])
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
    use crate::{
        autograd::backward_collect,
        device::Device,
        tensor::{DataType, Shape, TensorData},
    };
    use std::sync::Arc;

    fn f64_tensor(data: Vec<f64>) -> Tensor {
        let len = data.len();
        Tensor::new(
            Arc::new(TensorData::from_vec_f64(data, Device::cpu())),
            Shape::new(vec![len]),
            DataType::Float64,
            Device::cpu(),
            false,
        )
    }

    fn f32_tensor(data: Vec<f32>) -> Tensor {
        let len = data.len();
        Tensor::new(
            Arc::new(TensorData::from_vec_f32(data, Device::cpu())),
            Shape::new(vec![len]),
            DataType::Float32,
            Device::cpu(),
            false,
        )
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
