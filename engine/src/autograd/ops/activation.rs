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
    ops::activation::{infinite_slice_parts, slope_zeros, units::UnitParams},
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
            // Where the density leaves float64's normal range and the slope
            // does not; see `ops::gaussian`.
            if approximate {
                if x < -20.0
                    && let Some((_, slope)) = crate::ops::gaussian::gelu_tanh_far(x)
                {
                    return slope;
                }
            } else if x < crate::ops::gaussian::GELU_ERF_FAR {
                return crate::ops::gaussian::gelu_erf_far(x).1;
            }
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
/// Gradient function for log-softmax. It keeps the input rather than the
/// output: the probabilities its gradient needs are formed again from the
/// input, which `exp` of the rounded output could not give to the last bits.
pub struct LogSoftmaxBackward {
    pub input_id: TensorId,
    pub input: Tensor,
    pub dim: usize,
}
impl GradientFunction for LogSoftmaxBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<FxHashMap<TensorId, Tensor>> {
        let mut gradients = FxHashMap::default();
        gradients.reserve(1);

        let mut grad_data = TensorData::zeros_on_device(
            self.input.numel(),
            self.input.dtype(),
            self.input.device(),
        );

        match grad_output.dtype() {
            DataType::Float32 => {
                let go = grad_output.data().as_f32_slice().ok_or_else(|| {
                    MinitensorError::internal_error("Failed to get f32 slice from grad_output")
                })?;
                let input = self.input.data().as_f32_slice().ok_or_else(|| {
                    MinitensorError::internal_error(
                        "Failed to get f32 slice from log_softmax input",
                    )
                })?;
                let grad_slice = grad_data.as_f32_slice_mut().ok_or_else(|| {
                    MinitensorError::internal_error(
                        "Failed to get mutable f32 slice from grad_data",
                    )
                })?;
                log_softmax_backward(go, input, grad_slice, self.input.shape().dims(), self.dim);
            }
            DataType::Float64 => {
                let go = grad_output.data().as_f64_slice().ok_or_else(|| {
                    MinitensorError::internal_error("Failed to get f64 slice from grad_output")
                })?;
                let input = self.input.data().as_f64_slice().ok_or_else(|| {
                    MinitensorError::internal_error(
                        "Failed to get f64 slice from log_softmax input",
                    )
                })?;
                let grad_slice = grad_data.as_f64_slice_mut().ok_or_else(|| {
                    MinitensorError::internal_error(
                        "Failed to get mutable f64 slice from grad_data",
                    )
                })?;
                log_softmax_backward(go, input, grad_slice, self.input.shape().dims(), self.dim);
            }
            _ => {
                return Err(MinitensorError::invalid_operation(
                    "LogSoftmax backward only supported for floating point tensors",
                ));
            }
        }

        let grad_input = Tensor::new(
            Arc::new(grad_data),
            self.input.shape().clone(),
            self.input.dtype(),
            self.input.device(),
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
            self.input.numel(),
            self.input.dtype(),
            self.input.device(),
        );

        let output_dims = self.input.shape().dims();
        let mask_dims = self.mask.shape().dims();
        let same_shape = output_dims == mask_dims;
        let output_strides = if same_shape {
            None
        } else {
            Some(Strides::from_shape(self.input.shape()))
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
                let input = self.input.data().as_f32_slice().ok_or_else(|| {
                    MinitensorError::internal_error(
                        "Failed to get f32 slice from masked log_softmax input",
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
                    input,
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
                let input = self.input.data().as_f64_slice().ok_or_else(|| {
                    MinitensorError::internal_error(
                        "Failed to get f64 slice from masked log_softmax input",
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
                    input,
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
            self.input.shape().clone(),
            self.input.dtype(),
            self.input.device(),
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
/// `softmax` backward: `dx = y * (dy - sum_k(dy_k * y_k))` along `dim`, in
/// float64 for either dtype and rounded once.
///
/// Taken as `y_c (g_c (1 - y_c) - sum_{k != c} g_k y_k)`, with the likeliest
/// class's `1 - y` and the largest term's rest of the dot product each the sum
/// of the rest of its slice: `dy - dot` for a gradient concentrated on a
/// probability near one -- `log` after `softmax`, say -- took one from a
/// rounded near-one, and the dot product was summed in the tensor's own
/// dtype.
pub(crate) fn softmax_backward<T: crate::ops::activation::ShiftedExp + Send + Sync>(
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
                let at = |k: usize| base + k * after;
                let p = |k: usize| y_block[at(k)].wide();
                let g = |k: usize| go_block[at(k)].wide();
                let top = (1..dim_size).fold(0, |m, k| if p(k) > p(m) { k } else { m });
                let rest_p =
                    accurate_indexed_sum(dim_size, 0.0f64, |k| if k == top { 0.0 } else { p(k) });
                let dot = accurate_indexed_sum(dim_size, 0.0f64, |k| g(k) * p(k));
                let big = (1..dim_size).fold(0, |m, k| {
                    if (g(k) * p(k)).abs() > (g(m) * p(m)).abs() {
                        k
                    } else {
                        m
                    }
                });
                let rest_dot = accurate_indexed_sum(dim_size, 0.0f64, |k| {
                    if k == big { 0.0 } else { g(k) * p(k) }
                });
                for k in 0..dim_size {
                    let not_p = if k == top { rest_p } else { 1.0 - p(k) };
                    let others = if k == big {
                        rest_dot
                    } else {
                        dot - g(k) * p(k)
                    };
                    out_block[at(k)] = T::narrow(p(k) * (g(k) * not_p - others));
                }
            }
        },
    );
}
/// `g_c - p_c S` along `dim` for `p = softmax(z)` and `S` its slice's sum of
/// `g`: `log_softmax`'s backward, from its input.
///
/// The probabilities are formed again from the input, `exp(z - top)` with the
/// difference taken exactly over their total, where `exp` of the rounded
/// output turned its rounding into `|log p|` ulps of `p`: thirty at
/// `p = e^-40`. And the gradient is `g_c (1 - p_c) - p_c (S - g_c)`, with the
/// likeliest class's `1 - p` and the largest gradient's `S - g` each the sum
/// of the rest of the slice: as `g_c - p_c S` the class `nll_loss` puts its
/// gradient on took a probability near one from one, and a confident,
/// correct row's gradient there was all rounding -- 1.7e7 ulps out in
/// float32 and 9e15 in float64. In float64 for either dtype, each rounded
/// once.
pub(crate) fn log_softmax_backward<T: crate::ops::activation::ShiftedExp + Send + Sync>(
    grad_output: &[T],
    input: &[T],
    grad_input: &mut [T],
    dims: &[usize],
    dim: usize,
) {
    let Some((dim_size, after, group)) = softmax_geometry(grad_input, dims, dim) else {
        return;
    };
    zip_blocks_maybe_par(
        grad_output,
        input,
        grad_input,
        group,
        |_, go_block, z_block, out_block| {
            let mut stack = [0.0f64; 3 * SLICE_ON_STACK];
            let mut heap = Vec::new();
            let (terms, g, out) = slice_scratch(dim_size, &mut stack, &mut heap);
            let mut gathered: Vec<T> = Vec::new();
            for base in 0..after {
                let at = |k: usize| base + k * after;
                let z: &[T] = if after == 1 {
                    z_block
                } else {
                    gathered.clear();
                    gathered.extend((0..dim_size).map(|k| z_block[at(k)]));
                    &gathered
                };
                for (k, g) in g.iter_mut().enumerate() {
                    *g = go_block[at(k)].wide();
                }
                log_softmax_slice(z, g, terms, out);
                for (k, &v) in out.iter().enumerate() {
                    out_block[at(k)] = T::narrow(v);
                }
            }
        },
    );
}

/// Slices of up to this many entries take their scratch from the stack.
const SLICE_ON_STACK: usize = 64;

/// Three float64 buffers of `len` for one slice of a softmax-family
/// backward, from `stack` where they fit and `heap` where they do not.
fn slice_scratch<'a>(
    len: usize,
    stack: &'a mut [f64; 3 * SLICE_ON_STACK],
    heap: &'a mut Vec<f64>,
) -> (&'a mut [f64], &'a mut [f64], &'a mut [f64]) {
    let room: &mut [f64] = if len <= SLICE_ON_STACK {
        &mut stack[..3 * len]
    } else {
        heap.resize(3 * len, 0.0);
        heap
    };
    let (first, rest) = room.split_at_mut(len);
    let (second, third) = rest.split_at_mut(len);
    (first, second, third)
}

/// One slice of `log_softmax`'s backward, from its input `z` and incoming
/// gradient `g`: `g_c (1 - p_c) - p_c (S - g_c)` into `out`, in float64; see
/// [`log_softmax_backward`]. `terms` holds the slice's exponentials; all four
/// are the slice's length. A slice whose maximum is infinite takes the
/// forward's limits there: the `+inf`s share all of the mass evenly, a slice
/// of `-inf` has none, and a NaN makes the whole slice NaN.
fn log_softmax_slice<T: crate::ops::activation::ShiftedExp>(
    z: &[T],
    g: &[f64],
    terms: &mut [f64],
    out: &mut [f64],
) {
    let n = z.len();
    // The likeliest entry and the largest gradient, in one pass.
    let (mut top, mut big) = (0, 0);
    for k in 1..n {
        if z[k] > z[top] {
            top = k;
        }
        if g[k].abs() > g[big].abs() {
            big = k;
        }
    }
    if z[top].is_infinite() {
        let sum = accurate_indexed_sum(n, 0.0f64, |k| g[k]);
        let (infinities, nan) = infinite_slice_parts(z.iter().copied());
        let share = if nan {
            f64::NAN
        } else {
            (infinities as f64).recip()
        };
        for ((o, &v), &g) in out.iter_mut().zip(z).zip(g) {
            let p = if nan || v == T::infinity() {
                share
            } else {
                0.0
            };
            *o = g - p * sum;
        }
        return;
    }
    T::exp_shifted_into(z, z[top], terms);
    // The slice's sums, in one pass: of `g`, of `g` but the largest, and of
    // the terms but the likeliest's, which is one.
    let SliceSums([sum, rest_g, rest_terms]) = accurate_indexed_sum(n, SliceSums([0.0; 3]), |k| {
        SliceSums([
            g[k],
            if k == big { 0.0 } else { g[k] },
            if k == top { 0.0 } else { terms[k] },
        ])
    });
    let total = 1.0 + rest_terms;
    // `g (1 - p) - p (S - g)` is `g - p S`, and only at the likeliest entry,
    // where `1 - p` can be all but cancelled, and the largest gradient, where
    // `S - g` can, does the sum of the rest decide it; elsewhere it is one
    // product of the term with the slice's `S / total`.
    let share = sum / total;
    for ((o, &g), &term) in out.iter_mut().zip(g).zip(terms.iter()) {
        *o = g - term * share;
    }
    for k in [top, big] {
        let p = terms[k] / total;
        let not_p = if k == top {
            rest_terms / total
        } else {
            1.0 - p
        };
        let not_g = if k == big { rest_g } else { sum - g[k] };
        out[k] = g[k] * not_p - p * not_g;
    }
}

/// Three sums of a slice taken side by side in one pass.
#[derive(Clone, Copy)]
struct SliceSums([f64; 3]);

impl std::ops::Add for SliceSums {
    type Output = Self;

    #[inline(always)]
    fn add(self, other: Self) -> Self {
        let ([a, b, c], [x, y, z]) = (self.0, other.0);
        Self([a + x, b + y, c + z])
    }
}
/// `log_softmax` backward restricted to the unmasked positions, from the
/// input: each slice's unmasked entries as [`log_softmax_backward`] takes a
/// whole slice, and zero at the masked ones.
///
/// It took its probabilities as `exp` of the rounded output and formed
/// `g - p S` in the tensor's dtype, and a confident, correct slice's gradient
/// at its likeliest entry was all rounding, as `log_softmax`'s was.
///
/// `mask` may be a broadcast of the input shape, in which case
/// `output_strides`/`mask_strides` describe how to map a position onto it;
/// when the shapes already agree both are `None` and the index is used
/// directly.
fn masked_log_softmax_backward<T: crate::ops::activation::ShiftedExp + Send + Sync>(
    grad_output: &[T],
    input: &[T],
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

    // Resolving a mask position once, here, is what lets the pass below read
    // the same whether or not the mask is broadcast.
    let is_masked = |linear_idx: usize| match (output_strides, mask_strides) {
        (Some(out_strides), Some(m_strides)) => {
            mask[broadcast_mask_index(linear_idx, dims, out_strides, mask_dims, m_strides)]
        }
        _ => mask[linear_idx],
    };

    zip_blocks_maybe_par(
        grad_output,
        input,
        grad_input,
        group,
        |block_idx, go_block, z_block, out_block| {
            let block_offset = block_idx * group;
            // A slice's float64 scratch and its unmasked positions and
            // logits, on the stack for a slice of up to 64, as most are.
            let mut stack = [0.0f64; 3 * SLICE_ON_STACK];
            let mut heap = Vec::new();
            let (terms, g, out) = slice_scratch(dim_size, &mut stack, &mut heap);
            let (mut stack_kept, mut stack_z) =
                ([0usize; SLICE_ON_STACK], [T::zero(); SLICE_ON_STACK]);
            let (mut heap_kept, mut heap_z) = (Vec::new(), Vec::new());
            let (kept, z): (&mut [usize], &mut [T]) = if dim_size <= SLICE_ON_STACK {
                (&mut stack_kept[..dim_size], &mut stack_z[..dim_size])
            } else {
                heap_kept.resize(dim_size, 0);
                heap_z.resize(dim_size, T::zero());
                (&mut heap_kept, &mut heap_z)
            };
            for base in 0..after {
                let mut n = 0;
                for k in 0..dim_size {
                    let idx = base + k * after;
                    out_block[idx] = T::zero();
                    if !is_masked(block_offset + idx) {
                        kept[n] = idx;
                        z[n] = z_block[idx];
                        n += 1;
                    }
                }
                if n == 0 {
                    continue;
                }
                let kept = &kept[..n];
                for (g, &idx) in g.iter_mut().zip(kept) {
                    *g = go_block[idx].wide();
                }
                log_softmax_slice(&z[..n], &g[..n], &mut terms[..n], &mut out[..n]);
                for (&idx, &v) in kept.iter().zip(out.iter()) {
                    out_block[idx] = T::narrow(v);
                }
            }
        },
    );
}
/// Gradient function for masked log-softmax. It keeps the input rather than
/// the output, as [`LogSoftmaxBackward`] does.
pub struct MaskedLogSoftmaxBackward {
    pub input_id: TensorId,
    pub input: Tensor,
    pub mask: Tensor,
    pub dim: usize,
}
