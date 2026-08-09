// Copyright (c) Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

//! Gradient functions for the activation operations in
//! [`crate::ops::activation`].

use super::shape::{mask_select_into, zip_mask_into};
use super::*;
use crate::{
    error::{MinitensorError, Result},
    ops::map::binary_map,
    ops::util::{broadcast_mask_index, stable_sigmoid_f32, stable_sigmoid_f64},
    tensor::{DataType, Strides, Tensor, TensorData},
};
use libm::erfc;
use num_traits::Float;
use rayon::prelude::*;
use rustc_hash::FxHashMap;
use std::sync::Arc;

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
            // `1/(1 + exp(-s))` overflows for large negative `s` and returns a
            // zero gradient where the true one is still representable -- at
            // x = -95 it was 0 against 5.5e-42. The stable form branches on the
            // sign and keeps it.
            move |x: f32, gout: f32| {
                let scaled = beta32 * x;
                if scaled > thr32 {
                    gout
                } else {
                    gout * stable_sigmoid_f32(scaled)
                }
            },
            move |x: f64, gout: f64| {
                let scaled = beta64 * x;
                if scaled > thr64 {
                    gout
                } else {
                    gout * stable_sigmoid_f64(scaled)
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
const GELU_CUBIC_F64: f64 = 0.044_715;
/// The float32 GELU gradient, through the vectorized kernels.
///
/// Shaped like the forward pass in `ops::activation`: one backend selection,
/// then whole blocks handed to the multiversioned kernel. The `approximate`
/// branch is chosen per block, never per element.
fn gelu_backward_f32(input: &Tensor, grad_output: &Tensor, approximate: bool) -> Result<Tensor> {
    let saved = input.data().as_f32_slice().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get f32 slice from saved tensor")
    })?;
    let gout = grad_output.data().as_f32_slice().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get f32 slice from grad_output tensor")
    })?;
    if gout.len() != saved.len() {
        return Err(MinitensorError::shape_mismatch(
            grad_output.shape().dims().to_vec(),
            input.shape().dims().to_vec(),
        ));
    }

    let kernel = crate::ops::simd::F32Kernel::select();
    // SAFETY: both block kernels write every element of each block.
    let out = unsafe {
        crate::ops::map::binary_map_blocks_threshold(
            saved,
            gout,
            crate::ops::map::VECTOR_F32_PAR_THRESHOLD,
            |x, g, dst| {
                if approximate {
                    kernel.gelu_tanh_backward(x, g, dst)
                } else {
                    kernel.gelu_erf_backward(x, g, dst)
                }
            },
        )
    };
    Ok(Tensor::new(
        Arc::new(TensorData::from_vec::<f32>(
            out,
            DataType::Float32,
            input.device(),
        )),
        input.shape().clone(),
        DataType::Float32,
        input.device(),
        false,
    ))
}

impl GradientFunction for GeluBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<FxHashMap<TensorId, Tensor>> {
        // Exact:  d/dx x*Phi(x) = Phi(x) + x*phi(x).
        // Tanh approximation: differentiate 0.5*x*(1 + tanh(c*(x + a*x^3))).
        //
        // The two irrational coefficients are derived once here rather than
        // written as decimal literals, so each dtype keeps the exact rounding
        // of its own `sqrt`.
        let approximate = self.approximate;
        let coeff64 = (2.0f64 / std::f64::consts::PI).sqrt();
        let inv_sqrt_2pi64 = 1.0f64 / (2.0f64 * std::f64::consts::PI).sqrt();

        // float32 takes the vectorized kernels (`ops::simd::transcendental`),
        // which is a 7.5x saving on what was the most expensive gradient in the
        // activation set. They also carry the forward pass's cancellation fix
        // into the derivative: `1 + tanh(v)` and `sech^2(v) = 1 - tanh(v)^2`
        // both collapse for negative `v`, and the scalar form destroyed them.
        if self.input.dtype() == DataType::Float32 {
            let grad = gelu_backward_f32(&self.input, grad_output, approximate)?;
            return Ok(single(self.input_id, grad));
        }

        // Same cancellation-free spellings as the float32 kernels and as the
        // forward pass: `0.5*(1 + tanh(v))` is the logistic `s`, and
        // `sech^2(v) = 1 - tanh(v)^2` is `4*s*(1 - s)`. Written the obvious way
        // this gradient was 1% wrong at x = -10 and lost every digit past that.
        let grad = unary_chain_grad_f64(&self.input, grad_output, "GELU", move |x: f64| {
            if approximate {
                let x2 = x * x;
                let inner = coeff64 * (x + GELU_CUBIC_F64 * x * x2);
                let s = 1.0 / (1.0 + (-2.0 * inner).exp());
                let sech2 = 4.0 * s * (1.0 - s);
                s + 0.5 * x * sech2 * coeff64 * (1.0 + 3.0 * GELU_CUBIC_F64 * x2)
            } else {
                let cdf = 0.5 * erfc(-x * std::f64::consts::FRAC_1_SQRT_2);
                let pdf = (-0.5 * x * x).exp() * inv_sqrt_2pi64;
                cdf + x * pdf
            }
        })?;
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
/// The float32 SiLU gradient, through the vectorized kernel.
fn silu_backward_f32(input: &Tensor, grad_output: &Tensor) -> Result<Tensor> {
    let saved = input.data().as_f32_slice().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get f32 slice from saved tensor")
    })?;
    let gout = grad_output.data().as_f32_slice().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get f32 slice from grad_output tensor")
    })?;
    if gout.len() != saved.len() {
        return Err(MinitensorError::shape_mismatch(
            grad_output.shape().dims().to_vec(),
            input.shape().dims().to_vec(),
        ));
    }
    let kernel = crate::ops::simd::F32Kernel::select();
    // SAFETY: `silu_backward` writes every element of each block.
    let out = unsafe {
        crate::ops::map::binary_map_blocks_threshold(
            saved,
            gout,
            crate::ops::map::VECTOR_F32_PAR_THRESHOLD,
            |x, g, dst| kernel.silu_backward(x, g, dst),
        )
    };
    Ok(Tensor::new(
        Arc::new(TensorData::from_vec::<f32>(
            out,
            DataType::Float32,
            input.device(),
        )),
        input.shape().clone(),
        DataType::Float32,
        input.device(),
        false,
    ))
}

impl GradientFunction for SiluBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<FxHashMap<TensorId, Tensor>> {
        // d/dx x*sigmoid(x) = sigmoid(x) * (1 + x*(1 - sigmoid(x))).
        //
        // float32 takes the vectorized kernel, which also gets `1 - sigmoid(x)`
        // from `1/(e + 1)` rather than by subtraction -- past x = 40 the
        // subtraction returns exactly zero.
        if self.input.dtype() == DataType::Float32 {
            let grad = silu_backward_f32(&self.input, grad_output)?;
            return Ok(single(self.input_id, grad));
        }
        let grad = unary_chain_grad_f64(&self.input, grad_output, "SiLU", |x: f64| {
            let s = stable_sigmoid_f64(x);
            s * (1.0 + x * (1.0 - s))
        })?;
        Ok(single(self.input_id, grad))
    }

    fn input_ids(&self) -> &[TensorId] {
        std::slice::from_ref(&self.input_id)
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
/// Gradient function for Softplus
pub struct SoftplusBackward {
    pub input_id: TensorId,
    pub input: Tensor,
    pub beta: f64,
    pub threshold: f64,
}

/// Gradient function for Hardshrink
pub struct HardshrinkBackward {
    pub input_id: TensorId,
    pub mask: Vec<bool>,
}
impl GradientFunction for HardshrinkBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<FxHashMap<TensorId, Tensor>> {
        let mut gradients = FxHashMap::default();
        gradients.reserve(1);

        let mut grad_data = TensorData::zeros_on_device(
            grad_output.numel(),
            grad_output.dtype(),
            grad_output.device(),
        );

        match grad_output.dtype() {
            DataType::Float32 => {
                let go = grad_output.data().as_f32_slice().ok_or_else(|| {
                    MinitensorError::internal_error("Failed to get f32 slice from grad_output")
                })?;
                let grad_slice = grad_data.as_f32_slice_mut().ok_or_else(|| {
                    MinitensorError::internal_error(
                        "Failed to get mutable f32 slice from grad_data",
                    )
                })?;
                mask_select_into(grad_slice, go, &self.mask);
            }
            DataType::Float64 => {
                let go = grad_output.data().as_f64_slice().ok_or_else(|| {
                    MinitensorError::internal_error("Failed to get f64 slice from grad_output")
                })?;
                let grad_slice = grad_data.as_f64_slice_mut().ok_or_else(|| {
                    MinitensorError::internal_error(
                        "Failed to get mutable f64 slice from grad_data",
                    )
                })?;
                mask_select_into(grad_slice, go, &self.mask);
            }
            _ => {
                return Err(MinitensorError::invalid_operation(
                    "hardshrink backward only supported for floating point tensors",
                ));
            }
        }

        let grad_input = Tensor::new(
            Arc::new(grad_data),
            grad_output.shape().clone(),
            grad_output.dtype(),
            grad_output.device(),
            grad_output.requires_grad(),
        );
        gradients.insert(self.input_id, grad_input);

        Ok(gradients)
    }

    fn input_ids(&self) -> &[TensorId] {
        std::slice::from_ref(&self.input_id)
    }
}
/// Gradient function for ReLU
pub struct ReluBackward {
    pub input_id: TensorId,
    pub mask: Vec<bool>,
}
impl GradientFunction for ReluBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<FxHashMap<TensorId, Tensor>> {
        let mut gradients = FxHashMap::default();
        gradients.reserve(1);

        let mut grad_data = TensorData::zeros_on_device(
            grad_output.numel(),
            grad_output.dtype(),
            grad_output.device(),
        );

        match grad_output.dtype() {
            DataType::Float32 => {
                let go = grad_output.data().as_f32_slice().ok_or_else(|| {
                    MinitensorError::internal_error("Failed to get f32 slice from grad_output")
                })?;
                let grad_slice = grad_data.as_f32_slice_mut().ok_or_else(|| {
                    MinitensorError::internal_error(
                        "Failed to get mutable f32 slice from grad_data",
                    )
                })?;
                zip_mask_into(grad_slice, go, &self.mask, |g: f32, keep| {
                    // Multiply rather than select: a NaN gradient must stay
                    // NaN even where the mask is clear (NaN inputs are not
                    // `> 0`, and PyTorch propagates NaN through relu).
                    g * if keep { 1.0 } else { 0.0 }
                });
            }
            DataType::Float64 => {
                let go = grad_output.data().as_f64_slice().ok_or_else(|| {
                    MinitensorError::internal_error("Failed to get f64 slice from grad_output")
                })?;
                let grad_slice = grad_data.as_f64_slice_mut().ok_or_else(|| {
                    MinitensorError::internal_error(
                        "Failed to get mutable f64 slice from grad_data",
                    )
                })?;
                zip_mask_into(grad_slice, go, &self.mask, |g: f64, keep| {
                    // Multiply rather than select: a NaN gradient must stay
                    // NaN even where the mask is clear (NaN inputs are not
                    // `> 0`, and PyTorch propagates NaN through relu).
                    g * if keep { 1.0 } else { 0.0 }
                });
            }
            _ => {
                return Err(MinitensorError::invalid_operation(
                    "ReLU backward only supported for floating point tensors",
                ));
            }
        }

        let grad_input = Tensor::new(
            Arc::new(grad_data),
            grad_output.shape().clone(),
            grad_output.dtype(),
            grad_output.device(),
            grad_output.requires_grad(),
        );
        gradients.insert(self.input_id, grad_input);

        Ok(gradients)
    }

    fn input_ids(&self) -> &[TensorId] {
        std::slice::from_ref(&self.input_id)
    }
}
/// Gradient function for LeakyReLU
pub struct LeakyReluBackward {
    pub input_id: TensorId,
    pub negative_slope: f64,
    pub mask: Vec<bool>,
}
impl GradientFunction for LeakyReluBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<FxHashMap<TensorId, Tensor>> {
        let mut gradients = FxHashMap::default();
        gradients.reserve(1);

        let mut grad_data = TensorData::zeros_on_device(
            grad_output.numel(),
            grad_output.dtype(),
            grad_output.device(),
        );

        match grad_output.dtype() {
            DataType::Float32 => {
                let go = grad_output.data().as_f32_slice().ok_or_else(|| {
                    MinitensorError::internal_error("Failed to get f32 slice from grad_output")
                })?;
                let grad_slice = grad_data.as_f32_slice_mut().ok_or_else(|| {
                    MinitensorError::internal_error(
                        "Failed to get mutable f32 slice from grad_data",
                    )
                })?;
                let slope = self.negative_slope as f32;
                zip_mask_into(
                    grad_slice,
                    go,
                    &self.mask,
                    |g: f32, keep| {
                        if keep { g } else { g * slope }
                    },
                );
            }
            DataType::Float64 => {
                let go = grad_output.data().as_f64_slice().ok_or_else(|| {
                    MinitensorError::internal_error("Failed to get f64 slice from grad_output")
                })?;
                let grad_slice = grad_data.as_f64_slice_mut().ok_or_else(|| {
                    MinitensorError::internal_error(
                        "Failed to get mutable f64 slice from grad_data",
                    )
                })?;
                let slope = self.negative_slope;
                zip_mask_into(
                    grad_slice,
                    go,
                    &self.mask,
                    |g: f64, keep| {
                        if keep { g } else { g * slope }
                    },
                );
            }
            _ => {
                return Err(MinitensorError::invalid_operation(
                    "LeakyReLU backward only supported for floating point tensors",
                ));
            }
        }

        let grad_input = Tensor::new(
            Arc::new(grad_data),
            grad_output.shape().clone(),
            grad_output.dtype(),
            grad_output.device(),
            grad_output.requires_grad(),
        );
        gradients.insert(self.input_id, grad_input);

        Ok(gradients)
    }

    fn input_ids(&self) -> &[TensorId] {
        std::slice::from_ref(&self.input_id)
    }
}
/// Gradient function for softmax
pub struct SoftmaxBackward {
    pub input_id: TensorId,
    pub output: Tensor,
    pub dim: usize,
}
impl GradientFunction for SoftmaxBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<FxHashMap<TensorId, Tensor>> {
        let mut gradients = FxHashMap::default();
        gradients.reserve(1);

        // Allocate gradient buffer
        let mut grad_data = TensorData::zeros_on_device(
            self.output.numel(),
            self.output.dtype(),
            self.output.device(),
        );

        match grad_output.dtype() {
            DataType::Float32 => {
                let go = grad_output.data().as_f32_slice().ok_or_else(|| {
                    MinitensorError::internal_error("Failed to get f32 slice from grad_output")
                })?;
                let y = self.output.data().as_f32_slice().ok_or_else(|| {
                    MinitensorError::internal_error("Failed to get f32 slice from softmax output")
                })?;
                let grad_slice = grad_data.as_f32_slice_mut().ok_or_else(|| {
                    MinitensorError::internal_error(
                        "Failed to get mutable f32 slice from grad_data",
                    )
                })?;
                softmax_backward(go, y, grad_slice, self.output.shape().dims(), self.dim);
            }
            DataType::Float64 => {
                let go = grad_output.data().as_f64_slice().ok_or_else(|| {
                    MinitensorError::internal_error("Failed to get f64 slice from grad_output")
                })?;
                let y = self.output.data().as_f64_slice().ok_or_else(|| {
                    MinitensorError::internal_error("Failed to get f64 slice from softmax output")
                })?;
                let grad_slice = grad_data.as_f64_slice_mut().ok_or_else(|| {
                    MinitensorError::internal_error(
                        "Failed to get mutable f64 slice from grad_data",
                    )
                })?;
                softmax_backward(go, y, grad_slice, self.output.shape().dims(), self.dim);
            }
            _ => {
                return Err(MinitensorError::invalid_operation(
                    "Softmax backward only supported for floating point tensors",
                ));
            }
        }

        let grad_input = Tensor::new(
            Arc::new(grad_data),
            self.output.shape().clone(),
            self.output.dtype(),
            self.output.device(),
            grad_output.requires_grad(),
        );

        gradients.insert(self.input_id, grad_input);

        Ok(gradients)
    }

    fn input_ids(&self) -> &[TensorId] {
        std::slice::from_ref(&self.input_id)
    }
}
/// Gradient function for log-softmax
pub struct LogSoftmaxBackward {
    pub input_id: TensorId,
    pub output: Tensor,
    pub dim: usize,
}
impl GradientFunction for LogSoftmaxBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<FxHashMap<TensorId, Tensor>> {
        let mut gradients = FxHashMap::default();
        gradients.reserve(1);

        let mut grad_data = TensorData::zeros_on_device(
            self.output.numel(),
            self.output.dtype(),
            self.output.device(),
        );

        match grad_output.dtype() {
            DataType::Float32 => {
                let go = grad_output.data().as_f32_slice().ok_or_else(|| {
                    MinitensorError::internal_error("Failed to get f32 slice from grad_output")
                })?;
                let log_y = self.output.data().as_f32_slice().ok_or_else(|| {
                    MinitensorError::internal_error(
                        "Failed to get f32 slice from log_softmax output",
                    )
                })?;
                let grad_slice = grad_data.as_f32_slice_mut().ok_or_else(|| {
                    MinitensorError::internal_error(
                        "Failed to get mutable f32 slice from grad_data",
                    )
                })?;
                log_softmax_backward(go, log_y, grad_slice, self.output.shape().dims(), self.dim);
            }
            DataType::Float64 => {
                let go = grad_output.data().as_f64_slice().ok_or_else(|| {
                    MinitensorError::internal_error("Failed to get f64 slice from grad_output")
                })?;
                let log_y = self.output.data().as_f64_slice().ok_or_else(|| {
                    MinitensorError::internal_error(
                        "Failed to get f64 slice from log_softmax output",
                    )
                })?;
                let grad_slice = grad_data.as_f64_slice_mut().ok_or_else(|| {
                    MinitensorError::internal_error(
                        "Failed to get mutable f64 slice from grad_data",
                    )
                })?;
                log_softmax_backward(go, log_y, grad_slice, self.output.shape().dims(), self.dim);
            }
            _ => {
                return Err(MinitensorError::invalid_operation(
                    "LogSoftmax backward only supported for floating point tensors",
                ));
            }
        }

        let grad_input = Tensor::new(
            Arc::new(grad_data),
            self.output.shape().clone(),
            self.output.dtype(),
            self.output.device(),
            grad_output.requires_grad(),
        );

        gradients.insert(self.input_id, grad_input);

        Ok(gradients)
    }

    fn input_ids(&self) -> &[TensorId] {
        std::slice::from_ref(&self.input_id)
    }
}
/// Gradient function for tanh
/// Fuse an elementwise backward over the saved output and the incoming
/// gradient into one pass.
///
/// `tanh` and `sigmoid` were the only two backward kernels built by chaining
/// public tensor ops -- `Tensor::ones`, then `sub`, then `mul`, then `mul` --
/// and they were the only two whose backward cost far more than their forward.
/// Every other elementwise backward here sits at 1.2x-1.7x of its forward;
/// sigmoid was at 4.7x (32.3ms against 6.9ms on a 2048x1024 f32 tensor) because
/// each link allocated and traversed a full-size tensor. Folding the expression
/// into a single `binary_map` leaves one allocation and one pass.
fn fused_elementwise_backward<F32, F64>(
    output: &Tensor,
    grad_output: &Tensor,
    input_id: TensorId,
    gradients: &mut FxHashMap<TensorId, Tensor>,
    f32_op: F32,
    f64_op: F64,
) -> Result<()>
where
    F32: Fn(f32, f32) -> f32 + Send + Sync,
    F64: Fn(f64, f64) -> f64 + Send + Sync,
{
    let grad_data = match output.dtype() {
        DataType::Float32 => {
            let out = output.data().as_f32_slice().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get f32 slice for backward")
            })?;
            let grad = grad_output.data().as_f32_slice().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get f32 gradient slice")
            })?;
            TensorData::from_vec::<f32>(
                binary_map(out, grad, f32_op),
                DataType::Float32,
                output.device(),
            )
        }
        DataType::Float64 => {
            let out = output.data().as_f64_slice().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get f64 slice for backward")
            })?;
            let grad = grad_output.data().as_f64_slice().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get f64 gradient slice")
            })?;
            TensorData::from_vec::<f64>(
                binary_map(out, grad, f64_op),
                DataType::Float64,
                output.device(),
            )
        }
        _ => {
            return Err(MinitensorError::invalid_operation(
                "gradients are only defined for floating point tensors",
            ));
        }
    };
    gradients.insert(
        input_id,
        Tensor::new(
            Arc::new(grad_data),
            output.shape().clone(),
            output.dtype(),
            output.device(),
            false,
        ),
    );
    Ok(())
}

pub struct TanhBackward {
    pub input_id: TensorId,
    pub output: Tensor,
}
impl GradientFunction for TanhBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<FxHashMap<TensorId, Tensor>> {
        let mut gradients = FxHashMap::default();
        gradients.reserve(1);

        // d/dx(tanh(x)) = (1 - tanh²(x)) * grad_output
        fused_elementwise_backward(
            &self.output,
            grad_output,
            self.input_id,
            &mut gradients,
            |y: f32, g: f32| (1.0 - y * y) * g,
            |y: f64, g: f64| (1.0 - y * y) * g,
        )?;

        Ok(gradients)
    }

    fn input_ids(&self) -> &[TensorId] {
        std::slice::from_ref(&self.input_id)
    }
}
/// Gradient function for sigmoid
pub struct SigmoidBackward {
    pub input_id: TensorId,
    pub output: Tensor,
}
impl GradientFunction for SigmoidBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<FxHashMap<TensorId, Tensor>> {
        let mut gradients = FxHashMap::default();
        gradients.reserve(1);

        // d/dx(sigmoid(x)) = sigmoid(x) * (1 - sigmoid(x)) * grad_output
        fused_elementwise_backward(
            &self.output,
            grad_output,
            self.input_id,
            &mut gradients,
            |y: f32, g: f32| y * (1.0 - y) * g,
            |y: f64, g: f64| y * (1.0 - y) * g,
        )?;

        Ok(gradients)
    }

    fn input_ids(&self) -> &[TensorId] {
        std::slice::from_ref(&self.input_id)
    }
}
impl GradientFunction for MaskedLogSoftmaxBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<FxHashMap<TensorId, Tensor>> {
        let mut gradients = FxHashMap::default();
        gradients.reserve(1);

        let mut grad_data = TensorData::zeros_on_device(
            self.output.numel(),
            self.output.dtype(),
            self.output.device(),
        );

        let output_dims = self.output.shape().dims();
        let mask_dims = self.mask.shape().dims();
        let same_shape = output_dims == mask_dims;
        let output_strides = if same_shape {
            None
        } else {
            Some(Strides::from_shape(self.output.shape()))
        };
        let mask_strides = if same_shape {
            None
        } else {
            Some(Strides::from_shape(self.mask.shape()))
        };

        match grad_output.dtype() {
            DataType::Float32 => {
                let go = grad_output.data().as_f32_slice().ok_or_else(|| {
                    MinitensorError::internal_error("Failed to get f32 slice from grad_output")
                })?;
                let log_y = self.output.data().as_f32_slice().ok_or_else(|| {
                    MinitensorError::internal_error(
                        "Failed to get f32 slice from masked log_softmax output",
                    )
                })?;
                let mask_data = self.mask.data().as_bool_slice().ok_or_else(|| {
                    MinitensorError::internal_error("Failed to get bool slice from mask tensor")
                })?;
                let grad_slice = grad_data.as_f32_slice_mut().ok_or_else(|| {
                    MinitensorError::internal_error(
                        "Failed to get mutable f32 slice from grad_data",
                    )
                })?;
                masked_log_softmax_backward(
                    go,
                    log_y,
                    mask_data,
                    grad_slice,
                    output_dims,
                    self.dim,
                    mask_dims,
                    output_strides.as_ref().map(Strides::as_slice),
                    mask_strides.as_ref().map(Strides::as_slice),
                );
            }
            DataType::Float64 => {
                let go = grad_output.data().as_f64_slice().ok_or_else(|| {
                    MinitensorError::internal_error("Failed to get f64 slice from grad_output")
                })?;
                let log_y = self.output.data().as_f64_slice().ok_or_else(|| {
                    MinitensorError::internal_error(
                        "Failed to get f64 slice from masked log_softmax output",
                    )
                })?;
                let mask_data = self.mask.data().as_bool_slice().ok_or_else(|| {
                    MinitensorError::internal_error("Failed to get bool slice from mask tensor")
                })?;
                let grad_slice = grad_data.as_f64_slice_mut().ok_or_else(|| {
                    MinitensorError::internal_error(
                        "Failed to get mutable f64 slice from grad_data",
                    )
                })?;
                masked_log_softmax_backward(
                    go,
                    log_y,
                    mask_data,
                    grad_slice,
                    output_dims,
                    self.dim,
                    mask_dims,
                    output_strides.as_ref().map(Strides::as_slice),
                    mask_strides.as_ref().map(Strides::as_slice),
                );
            }
            _ => {
                return Err(MinitensorError::invalid_operation(
                    "Masked log_softmax backward only supported for floating point tensors",
                ));
            }
        }

        let grad_input = Tensor::new(
            Arc::new(grad_data),
            self.output.shape().clone(),
            self.output.dtype(),
            self.output.device(),
            grad_output.requires_grad(),
        );

        accumulate_grad(&mut gradients, self.input_id, grad_input)?;

        Ok(gradients)
    }

    fn input_ids(&self) -> &[TensorId] {
        std::slice::from_ref(&self.input_id)
    }
}
/// Run `block` over aligned `group`-sized blocks of the three slices,
/// sequentially below [`PAR_THRESHOLD`] and in parallel above it.
///
/// Every softmax-family backward kernel has this shape, and each one used to
/// spell both branches out with the block body copied verbatim into each --
/// twice per kernel, times two dtypes.
fn zip_blocks_maybe_par<T, F>(
    grad_output: &[T],
    saved: &[T],
    grad_input: &mut [T],
    group: usize,
    block: F,
) where
    T: Send + Sync,
    F: Fn(usize, &[T], &[T], &mut [T]) + Send + Sync,
{
    // `group` is the reduced axis times everything after it, so it is zero
    // exactly when one of those axes is empty -- and then the tensor has no
    // elements and all three slices are empty. `chunks` and `par_chunks` both
    // panic on a chunk size of zero rather than yielding nothing, and the
    // panic crossed the binding: `softmax((3, 0), dim=0).sum().backward()`
    // reached it from Python. There is no block to run either way.
    if group == 0 {
        return;
    }

    if grad_output.len() < PAR_THRESHOLD {
        for (block_idx, ((go, sv), out)) in grad_output
            .chunks(group)
            .zip(saved.chunks(group))
            .zip(grad_input.chunks_mut(group))
            .enumerate()
        {
            block(block_idx, go, sv, out);
        }
    } else {
        grad_output
            .par_chunks(group)
            .zip(saved.par_chunks(group))
            .zip(grad_input.par_chunks_mut(group))
            .enumerate()
            .for_each(|(block_idx, ((go, sv), out))| block(block_idx, go, sv, out));
    }
}
/// Geometry shared by the softmax-family backward kernels: the reduced
/// dimension's size, the number of trailing elements per slice (`after`), and
/// the size of one contiguous block spanning the reduced dimension.
///
/// `None` means there is nothing left to compute: a 0-d tensor (whose single
/// gradient element is zeroed here, the reduction being a constant) or an empty
/// reduced dimension (no output elements to write).
fn softmax_geometry<T: Float>(
    grad_input: &mut [T],
    dims: &[usize],
    dim: usize,
) -> Option<(usize, usize, usize)> {
    if dims.is_empty() {
        if let Some(first) = grad_input.first_mut() {
            *first = T::zero();
        }
        return None;
    }

    let dim_size = dims[dim];
    if dim_size == 0 {
        return None;
    }

    let after: usize = if dim + 1 >= dims.len() {
        1
    } else {
        dims[dim + 1..].iter().product()
    };
    Some((dim_size, after, dim_size * after))
}
/// `softmax` backward: `dx = y * (dy - sum_k(dy_k * y_k))` along `dim`.
pub(crate) fn softmax_backward<T: Float + Send + Sync>(
    grad_output: &[T],
    y: &[T],
    grad_input: &mut [T],
    dims: &[usize],
    dim: usize,
) {
    let Some((dim_size, after, group)) = softmax_geometry(grad_input, dims, dim) else {
        return;
    };

    zip_blocks_maybe_par(
        grad_output,
        y,
        grad_input,
        group,
        |_, go_block, y_block, out_block| {
            for base in 0..after {
                let mut dot = T::zero();
                for k in 0..dim_size {
                    let idx = base + k * after;
                    dot = dot + go_block[idx] * y_block[idx];
                }
                for k in 0..dim_size {
                    let idx = base + k * after;
                    out_block[idx] = y_block[idx] * (go_block[idx] - dot);
                }
            }
        },
    );
}
/// `log_softmax` backward: `dx = dy - exp(log_y) * sum_k(dy_k)` along `dim`.
pub(crate) fn log_softmax_backward<T: Float + Send + Sync>(
    grad_output: &[T],
    log_y: &[T],
    grad_input: &mut [T],
    dims: &[usize],
    dim: usize,
) {
    let Some((dim_size, after, group)) = softmax_geometry(grad_input, dims, dim) else {
        return;
    };

    zip_blocks_maybe_par(
        grad_output,
        log_y,
        grad_input,
        group,
        |_, go_block, log_block, out_block| {
            for base in 0..after {
                let mut sum = T::zero();
                for k in 0..dim_size {
                    sum = sum + go_block[base + k * after];
                }
                for k in 0..dim_size {
                    let idx = base + k * after;
                    out_block[idx] = go_block[idx] - log_block[idx].exp() * sum;
                }
            }
        },
    );
}
/// `log_softmax` backward restricted to the unmasked positions.
///
/// Masked entries contribute nothing to their slice's sum and receive a zero
/// gradient. `mask` may be a broadcast of the output shape, in which case
/// `output_strides`/`mask_strides` describe how to map an output position onto
/// it; when the shapes already agree both are `None` and the index is used
/// directly.
fn masked_log_softmax_backward<T: Float + Send + Sync>(
    grad_output: &[T],
    log_y: &[T],
    mask: &[bool],
    grad_input: &mut [T],
    dims: &[usize],
    dim: usize,
    mask_dims: &[usize],
    output_strides: Option<&[usize]>,
    mask_strides: Option<&[usize]>,
) {
    let Some((dim_size, after, group)) = softmax_geometry(grad_input, dims, dim) else {
        return;
    };

    // Resolving a mask position once, here, is what lets the two passes below
    // read the same whether or not the mask is broadcast.
    let is_masked = |linear_idx: usize| match (output_strides, mask_strides) {
        (Some(out_strides), Some(m_strides)) => {
            mask[broadcast_mask_index(linear_idx, dims, out_strides, mask_dims, m_strides)]
        }
        _ => mask[linear_idx],
    };

    zip_blocks_maybe_par(
        grad_output,
        log_y,
        grad_input,
        group,
        |block_idx, go_block, log_block, out_block| {
            let block_offset = block_idx * group;
            for base in 0..after {
                let mut sum = T::zero();
                for k in 0..dim_size {
                    let idx = base + k * after;
                    if !is_masked(block_offset + idx) {
                        sum = sum + go_block[idx];
                    }
                }
                for k in 0..dim_size {
                    let idx = base + k * after;
                    out_block[idx] = if is_masked(block_offset + idx) {
                        T::zero()
                    } else {
                        go_block[idx] - log_block[idx].exp() * sum
                    };
                }
            }
        },
    );
}
/// Gradient function for masked log-softmax
pub struct MaskedLogSoftmaxBackward {
    pub input_id: TensorId,
    pub output: Tensor,
    pub mask: Tensor,
    pub dim: usize,
}
