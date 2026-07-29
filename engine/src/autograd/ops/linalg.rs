// Copyright (c) Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

use super::*;
use crate::{
    error::Result,
    ops::arithmetic,
    tensor::{Shape, Tensor},
};
use libm::{erf, erff};
use rustc_hash::FxHashMap;

/// Wrap a single-input gradient into the map a [`GradientFunction`] returns.
#[inline]
fn single(input_id: TensorId, grad: Tensor) -> FxHashMap<TensorId, Tensor> {
    let mut gradients = FxHashMap::default();
    gradients.reserve(1);
    gradients.insert(input_id, grad);
    gradients
}

impl GradientFunction for SoftplusBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<FxHashMap<TensorId, Tensor>> {
        // d/dx softplus(x) = sigmoid(beta * x); above the threshold the forward
        // pass is the identity, so the gradient passes straight through.
        let beta32 = self.beta as f32;
        let thr32 = self.threshold as f32;
        let (beta64, thr64) = (self.beta, self.threshold);
        let grad = unary_chain_grad(
            &self.input,
            grad_output,
            "Softplus",
            move |x: f32, gout: f32| {
                let scaled = beta32 * x;
                if scaled > thr32 {
                    gout
                } else {
                    gout / (1.0 + (-scaled).exp())
                }
            },
            move |x: f64, gout: f64| {
                let scaled = beta64 * x;
                if scaled > thr64 {
                    gout
                } else {
                    gout / (1.0 + (-scaled).exp())
                }
            },
        )?;
        Ok(single(self.input_id, grad))
    }

    fn input_ids(&self) -> &[TensorId] {
        std::slice::from_ref(&self.input_id)
    }
}

/// Gradient function for GELU activation
pub struct GeluBackward {
    pub input_id: TensorId,
    pub input: Tensor,
    pub approximate: bool,
}

/// The cubic coefficient of GELU's tanh approximation.
const GELU_CUBIC_F32: f32 = 0.044_715;
const GELU_CUBIC_F64: f64 = 0.044_715;

impl GradientFunction for GeluBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<FxHashMap<TensorId, Tensor>> {
        // Exact:  d/dx x*Phi(x) = Phi(x) + x*phi(x).
        // Tanh approximation: differentiate 0.5*x*(1 + tanh(c*(x + a*x^3))).
        //
        // The two irrational coefficients are derived once here rather than
        // written as decimal literals, so each dtype keeps the exact rounding
        // of its own `sqrt`.
        let approximate = self.approximate;
        let coeff32 = (2.0f32 / std::f32::consts::PI).sqrt();
        let coeff64 = (2.0f64 / std::f64::consts::PI).sqrt();
        let inv_sqrt_2pi32 = 1.0f32 / (2.0f32 * std::f32::consts::PI).sqrt();
        let inv_sqrt_2pi64 = 1.0f64 / (2.0f64 * std::f64::consts::PI).sqrt();
        let grad = unary_chain_grad(
            &self.input,
            grad_output,
            "GELU",
            move |x: f32, gout: f32| {
                let local = if approximate {
                    let x2 = x * x;
                    let inner = coeff32 * (x + GELU_CUBIC_F32 * x * x2);
                    let tanh_inner = inner.tanh();
                    let sech2 = 1.0 - tanh_inner * tanh_inner;
                    0.5 * (1.0 + tanh_inner)
                        + 0.5 * x * sech2 * coeff32 * (1.0 + 3.0 * GELU_CUBIC_F32 * x2)
                } else {
                    let cdf = 0.5 * (1.0 + erff(x * std::f32::consts::FRAC_1_SQRT_2));
                    let pdf = (-0.5 * x * x).exp() * inv_sqrt_2pi32;
                    cdf + x * pdf
                };
                gout * local
            },
            move |x: f64, gout: f64| {
                let local = if approximate {
                    let x2 = x * x;
                    let inner = coeff64 * (x + GELU_CUBIC_F64 * x * x2);
                    let tanh_inner = inner.tanh();
                    let sech2 = 1.0 - tanh_inner * tanh_inner;
                    0.5 * (1.0 + tanh_inner)
                        + 0.5 * x * sech2 * coeff64 * (1.0 + 3.0 * GELU_CUBIC_F64 * x2)
                } else {
                    let cdf = 0.5 * (1.0 + erf(x * std::f64::consts::FRAC_1_SQRT_2));
                    let pdf = (-0.5 * x * x).exp() * inv_sqrt_2pi64;
                    cdf + x * pdf
                };
                gout * local
            },
        )?;
        Ok(single(self.input_id, grad))
    }

    fn input_ids(&self) -> &[TensorId] {
        std::slice::from_ref(&self.input_id)
    }
}

/// Gradient function for ELU activation
pub struct EluBackward {
    pub input_id: TensorId,
    pub output: Tensor,
    pub alpha: f64,
}

impl GradientFunction for EluBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<FxHashMap<TensorId, Tensor>> {
        // Expressed in the saved *output*: for x <= 0, elu(x) = alpha*(e^x - 1)
        // so d/dx = alpha*e^x = elu(x) + alpha.
        let alpha32 = self.alpha as f32;
        let alpha64 = self.alpha;
        let grad = unary_chain_grad(
            &self.output,
            grad_output,
            "ELU",
            move |out: f32, gout: f32| gout * if out > 0.0 { 1.0 } else { out + alpha32 },
            move |out: f64, gout: f64| gout * if out > 0.0 { 1.0 } else { out + alpha64 },
        )?;
        Ok(single(self.input_id, grad))
    }

    fn input_ids(&self) -> &[TensorId] {
        std::slice::from_ref(&self.input_id)
    }
}

/// Gradient function for SELU activation
pub struct SeluBackward {
    pub input_id: TensorId,
    pub output: Tensor,
}

/// SELU's fixed constants (Klambauer et al., 2017).
const SELU_SCALE_F32: f32 = 1.050_701;
const SELU_ALPHA_F32: f32 = 1.673_263_2;
const SELU_SCALE_F64: f64 = 1.050_700_987_355_480_5;
const SELU_ALPHA_F64: f64 = 1.673_263_242_354_377_2;

impl GradientFunction for SeluBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<FxHashMap<TensorId, Tensor>> {
        // As with ELU, in terms of the saved output: for x <= 0,
        // selu(x) = scale*alpha*(e^x - 1), so d/dx = selu(x) + scale*alpha.
        let grad = unary_chain_grad(
            &self.output,
            grad_output,
            "SELU",
            |out: f32, gout: f32| {
                gout * if out > 0.0 {
                    SELU_SCALE_F32
                } else {
                    out + SELU_SCALE_F32 * SELU_ALPHA_F32
                }
            },
            |out: f64, gout: f64| {
                gout * if out > 0.0 {
                    SELU_SCALE_F64
                } else {
                    out + SELU_SCALE_F64 * SELU_ALPHA_F64
                }
            },
        )?;
        Ok(single(self.input_id, grad))
    }

    fn input_ids(&self) -> &[TensorId] {
        std::slice::from_ref(&self.input_id)
    }
}

/// Gradient function for SiLU activation
pub struct SiluBackward {
    pub input_id: TensorId,
    pub input: Tensor,
}

impl GradientFunction for SiluBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<FxHashMap<TensorId, Tensor>> {
        // d/dx x*sigmoid(x) = sigmoid(x) * (1 + x*(1 - sigmoid(x))).
        let grad = unary_chain_grad(
            &self.input,
            grad_output,
            "SiLU",
            |x: f32, gout: f32| {
                let s = stable_sigmoid_f32(x);
                gout * (s * (1.0 + x * (1.0 - s)))
            },
            |x: f64, gout: f64| {
                let s = stable_sigmoid_f64(x);
                gout * (s * (1.0 + x * (1.0 - s)))
            },
        )?;
        Ok(single(self.input_id, grad))
    }

    fn input_ids(&self) -> &[TensorId] {
        std::slice::from_ref(&self.input_id)
    }
}

/// Sigmoid evaluated through whichever of `e^-x` / `e^x` cannot overflow, so
/// large-magnitude inputs saturate to 1/0 instead of producing inf/inf = NaN.
#[inline]
fn stable_sigmoid_f32(x: f32) -> f32 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let exp_pos = x.exp();
        exp_pos / (1.0 + exp_pos)
    }
}

#[inline]
fn stable_sigmoid_f64(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let exp_pos = x.exp();
        exp_pos / (1.0 + exp_pos)
    }
}

/// Gradient function for Softsign activation
pub struct SoftsignBackward {
    pub input_id: TensorId,
    pub input: Tensor,
}

impl GradientFunction for SoftsignBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<FxHashMap<TensorId, Tensor>> {
        // d/dx x/(1 + |x|) = 1 / (1 + |x|)^2.
        let grad = unary_chain_grad(
            &self.input,
            grad_output,
            "Softsign",
            |x: f32, gout: f32| {
                let denom = 1.0 + x.abs();
                gout / (denom * denom)
            },
            |x: f64, gout: f64| {
                let denom = 1.0 + x.abs();
                gout / (denom * denom)
            },
        )?;
        Ok(single(self.input_id, grad))
    }

    fn input_ids(&self) -> &[TensorId] {
        std::slice::from_ref(&self.input_id)
    }
}

/// Gradient function for power operation
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PowBroadcast {
    None,
    BaseScalar,
    ExponentScalar,
}
pub struct PowBackward {
    pub base: Tensor,
    pub exponent: Tensor,
    pub output: Tensor,
    pub input_ids: [TensorId; 2],
    pub base_requires_grad: bool,
    pub exp_requires_grad: bool,
    pub broadcast: PowBroadcast,
}

/// Gradient function for logaddexp
pub struct LogAddExpBackward {
    pub lhs: Tensor,
    pub rhs: Tensor,
    pub output: Tensor,
    pub input_ids: [TensorId; 2],
    pub input_shapes: [Vec<usize>; 2],
    /// Which inputs actually need a gradient; frozen inputs skip their
    /// exp/sub/mul/reduce chain entirely.
    pub input_requires_grad: [bool; 2],
}

impl GradientFunction for LogAddExpBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<FxHashMap<TensorId, Tensor>> {
        let mut gradients = FxHashMap::default();
        gradients.reserve(2);

        if self.input_requires_grad[0] {
            let lhs_diff = arithmetic::sub(&self.lhs.detach(), &self.output.detach())?;
            let lhs_term = lhs_diff.exp()?;
            let lhs_mul = arithmetic::mul(&lhs_term, grad_output)?;
            let lhs_grad = reduce_gradient_for_broadcasting(
                &lhs_mul,
                &Shape::new(self.input_shapes[0].clone()),
            )?;
            accumulate_grad(&mut gradients, self.input_ids[0], lhs_grad)?;
        }

        if self.input_requires_grad[1] {
            let rhs_diff = arithmetic::sub(&self.rhs.detach(), &self.output.detach())?;
            let rhs_term = rhs_diff.exp()?;
            let rhs_mul = arithmetic::mul(&rhs_term, grad_output)?;
            let rhs_grad = reduce_gradient_for_broadcasting(
                &rhs_mul,
                &Shape::new(self.input_shapes[1].clone()),
            )?;
            accumulate_grad(&mut gradients, self.input_ids[1], rhs_grad)?;
        }

        Ok(gradients)
    }

    fn input_ids(&self) -> &[TensorId] {
        &self.input_ids
    }
}
