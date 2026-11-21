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
    ops::activation::{slope_zeros, units::UnitParams},
    ops::map::par_out_chunks_gated,
    ops::util::{accurate_indexed_sum, broadcast_mask_index, stable_sigmoid_f64},
    tensor::{DataType, Strides, Tensor, TensorData},
};
use num_traits::Float;
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
        // pass is the identity, so the gradient passes straight through. The
        // logistic of `-|v|` cannot overflow, and the rounding of `beta * x`,
        // which it turns into `|beta * x|` ulps of a far tail, is taken back
        // through its own slope, as the forward does. In float64 throughout,
        // the forward's threshold test included.
        let (beta, threshold) = (self.beta, self.threshold);
        let grad = unary_chain_grad_f64(&self.input, grad_output, "Softplus", &move |x| {
            let scaled = beta * x;
            if scaled > threshold {
                return 1.0;
            }
            let rounding = if scaled.abs() < 1e300 {
                crate::ops::simd::product_error::<false>(beta, x, scaled)
            } else {
                0.0
            };
            let decay = (-scaled.abs()).exp();
            let share = if scaled > 0.0 {
                1.0 / (1.0 + decay)
            } else {
                decay / (1.0 + decay)
            };
            share + share * (1.0 - share) * rounding
        })?;
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
            crate::ops::map::PAR_CHUNK,
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
        let approximate = self.approximate;
        let coeff64 = (2.0f64 / std::f64::consts::PI).sqrt();

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
        let grad = unary_chain_grad_f64(&self.input, grad_output, "GELU", &move |x: f64| {
            // The limits, where either formula would form `inf * 0`.
            if x.is_infinite() {
                return if x > 0.0 { 1.0 } else { 0.0 };
            }
            // About the slope's zero its two terms cancel; there it is its
            // series instead (see `ops::activation::slope_zeros`).
            let near = if approximate {
                slope_zeros::GELU_TANH.near(x)
            } else {
                slope_zeros::GELU_ERF.near(x)
            };
            if let Some(slope) = near {
                return slope;
            }
            // Each factor with the rounding of its argument taken back, as in
            // the forward pass; see `ops::gaussian`.
            if approximate {
                // `1 - s` is the form at `-x`, which keeps its digits where
                // the subtraction would cancel them.
                let s = crate::ops::gaussian::tanh_cdf(x);
                let sech2 = 4.0 * s * crate::ops::gaussian::tanh_cdf(-x);
                s + 0.5 * x * sech2 * coeff64 * (1.0 + 3.0 * GELU_CUBIC_F64 * x * x)
            } else {
                crate::ops::gaussian::standard_normal_cdf(x)
                    + x * crate::ops::gaussian::standard_normal_pdf(x)
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
    pub input: Tensor,
    pub alpha: f64,
}
impl GradientFunction for EluBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<FxHashMap<TensorId, Tensor>> {
        // For x <= 0, elu(x) = alpha (e^x - 1) and d/dx = alpha e^x, taken of
        // the input: as `elu(x) + alpha` it was a difference of two numbers
        // near -alpha, zero from x = -38 in float64 where e^x is not.
        let alpha = self.alpha;
        let grad = unary_chain_grad_f64(&self.input, grad_output, "ELU", &move |x| {
            if x > 0.0 { 1.0 } else { alpha * x.exp() }
        })?;
        Ok(single(self.input_id, grad))
    }

    fn input_ids(&self) -> &[TensorId] {
        std::slice::from_ref(&self.input_id)
    }
}
/// Gradient function for SELU activation
pub struct SeluBackward {
    pub input_id: TensorId,
    pub input: Tensor,
}
/// SELU's fixed constants (Klambauer et al., 2017).
const SELU_SCALE_F64: f64 = 1.050_700_987_355_480_5;
const SELU_ALPHA_F64: f64 = 1.673_263_242_354_377_2;
impl GradientFunction for SeluBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<FxHashMap<TensorId, Tensor>> {
        // As with ELU, of the input: for x <= 0, selu(x) = scale alpha
        // (e^x - 1), so d/dx = scale alpha e^x.
        let grad = unary_chain_grad_f64(&self.input, grad_output, "SELU", &|x| {
            if x > 0.0 {
                SELU_SCALE_F64
            } else {
                SELU_SCALE_F64 * SELU_ALPHA_F64 * x.exp()
            }
        })?;
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
/// A float32 gradient through one of the vectorized kernels, `op` the
/// kernel's method: SiLU's, tanh's or sigmoid's.
type KernelBackward =
    fn(crate::ops::simd::F32Kernel, &[f32], &[f32], &mut [std::mem::MaybeUninit<f32>]);
fn kernel_backward_f32(input: &Tensor, grad_output: &Tensor, op: KernelBackward) -> Result<Tensor> {
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
            crate::ops::map::PAR_CHUNK,
            |x, g, dst| op(kernel, x, g, dst),
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
            let grad = kernel_backward_f32(
                &self.input,
                grad_output,
                crate::ops::simd::F32Kernel::silu_backward,
            )?;
            return Ok(single(self.input_id, grad));
        }
        let grad = unary_chain_grad_f64(&self.input, grad_output, "SiLU", &|x: f64| {
            // The limits, where the formula would form `inf * 0`.
            if x.is_infinite() {
                return if x > 0.0 { 1.0 } else { 0.0 };
            }
            // About its zero, near -1.2785, `1 + x s(-x)` cancels; there the
            // slope is its series (see `ops::activation::slope_zeros`). Else
            // `s(-x)` is formed itself: `1 - s` would keep only the bits of
            // `s` past one's, none of them past x = 37.
            if let Some(slope) = slope_zeros::SILU.near(x) {
                return slope;
            }
            stable_sigmoid_f64(x) * (1.0 + x * stable_sigmoid_f64(-x))
        })?;
        Ok(single(self.input_id, grad))
    }

    fn input_ids(&self) -> &[TensorId] {
        std::slice::from_ref(&self.input_id)
    }
}
/// Gradient function for every activation in [`crate::ops::activation::units`].
///
/// Those units differ only in their chain rule, so this holds it as a kernel
/// rather than there being one struct per unit. The saved input is what every
/// one of them needs -- none is cheaper to differentiate from its output.
///
/// The kernel is a type parameter rather than a function pointer. Stored as a
/// pointer it was loaded and called indirectly once per element inside the
/// pool's workers, which kept the loop from being inlined or vectorized: the
/// chain rule of a clamp cost ten times the clamp.
pub struct ActivationUnitBackward<N, W> {
    pub input_id: TensorId,
    pub input: Tensor,
    pub name: &'static str,
    pub narrow: N,
    pub wide: W,
    pub params: UnitParams<f64>,
}

impl<N, W> GradientFunction for ActivationUnitBackward<N, W>
where
    N: Fn(f32, f32, UnitParams<f32>) -> f32 + Copy + Send + Sync,
    W: Fn(f64, f64, UnitParams<f64>) -> f64 + Copy + Send + Sync,
{
    fn backward(&self, grad_output: &Tensor) -> Result<FxHashMap<TensorId, Tensor>> {
        let narrow_params = [self.params[0] as f32, self.params[1] as f32];
        let wide_params = self.params;
        let (narrow, wide) = (self.narrow, self.wide);
        let grad = unary_chain_grad(
            &self.input,
            grad_output,
            self.name,
            move |x: f32, gout: f32| narrow(x, gout, narrow_params),
            move |x: f64, gout: f64| wide(x, gout, wide_params),
        )?;
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
        let grad = unary_chain_grad_f64(&self.input, grad_output, "Softsign", &|x| {
            let denom = 1.0 + x.abs();
            1.0 / (denom * denom)
        })?;
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
                    // `> 0`, and relu propagates NaN).
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
                    // `> 0`, and relu propagates NaN).
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
pub struct TanhBackward {
    pub input_id: TensorId,
    pub input: Tensor,
}
impl GradientFunction for TanhBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<FxHashMap<TensorId, Tensor>> {
        // d/dx tanh(x) = sech^2(x) = 4t / (1 + t)^2 for t = e^(-2|x|), taken of
        // the input: as `1 - tanh^2` of the rounded output it was a difference
        // of two numbers near one -- thousands of ulps out from |x| = 5 and
        // zero from |x| = 19 in float64, 9 in float32, where it is not.
        if self.input.dtype() == DataType::Float32 {
            let grad = kernel_backward_f32(
                &self.input,
                grad_output,
                crate::ops::simd::F32Kernel::tanh_backward,
            )?;
            return Ok(single(self.input_id, grad));
        }
        let grad = unary_chain_grad_f64(&self.input, grad_output, "tanh", &|x| {
            let t = (-2.0 * x.abs()).exp();
            4.0 * t / ((1.0 + t) * (1.0 + t))
        })?;
        Ok(single(self.input_id, grad))
    }

    fn input_ids(&self) -> &[TensorId] {
        std::slice::from_ref(&self.input_id)
    }
}
/// Gradient function for sigmoid
pub struct SigmoidBackward {
    pub input_id: TensorId,
    pub input: Tensor,
}
impl GradientFunction for SigmoidBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<FxHashMap<TensorId, Tensor>> {
        // d/dx sigmoid(x) = sigmoid(x) sigmoid(-x) = t / (1 + t)^2 for
        // t = e^(-|x|), taken of the input: as `y (1 - y)` of the rounded
        // output it lost the digits of `1 - y` for large x, and was zero
        // from x = 37 in float64 where it is not.
        if self.input.dtype() == DataType::Float32 {
            let grad = kernel_backward_f32(
                &self.input,
                grad_output,
                crate::ops::simd::F32Kernel::sigmoid_backward,
            )?;
            return Ok(single(self.input_id, grad));
        }
        let grad = unary_chain_grad_f64(&self.input, grad_output, "sigmoid", &|x| {
            let t = (-x.abs()).exp();
            t / ((1.0 + t) * (1.0 + t))
        })?;
        Ok(single(self.input_id, grad))
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

    par_out_chunks_gated(grad_input, group, PAR_THRESHOLD, &|start, out| {
        let span = start..start + out.len();
        block(start / group, &grad_output[span.clone()], &saved[span], out);
    });
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
pub(crate) fn log_softmax_backward<T: crate::ops::activation::ShiftedExp + Send + Sync>(
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
            // Each gradient in float64 and rounded once, against the sum of
            // the slice's in float64.
            if after == 1 {
                // Along the last axis a block is one contiguous row, so its
                // probabilities come from the forward's `exp` kernel -- the
                // vectorised one for float32 -- a piece at a time, where the
                // loop below takes a scalar `exp` per element.
                const PIECE: usize = 256;
                let sum = accurate_indexed_sum(go_block.len(), 0.0f64, |k| go_block[k].wide());
                let mut probabilities = [0.0f64; PIECE];
                for ((go, logs), out) in go_block
                    .chunks(PIECE)
                    .zip(log_block.chunks(PIECE))
                    .zip(out_block.chunks_mut(PIECE))
                {
                    let probabilities = &mut probabilities[..logs.len()];
                    T::exp_shifted_into(logs, T::zero(), probabilities);
                    for ((o, &g), &p) in out.iter_mut().zip(go).zip(probabilities.iter()) {
                        *o = T::narrow(g.wide() - p * sum);
                    }
                }
                return;
            }
            for base in 0..after {
                let sum =
                    accurate_indexed_sum(dim_size, 0.0f64, |k| go_block[base + k * after].wide());
                for k in 0..dim_size {
                    let idx = base + k * after;
                    let p = log_block[idx].wide().exp();
                    out_block[idx] = T::narrow(go_block[idx].wide() - p * sum);
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
