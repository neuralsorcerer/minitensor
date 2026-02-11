// Copyright (c) 2026 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

impl GradientFunction for SoftplusBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<FxHashMap<TensorId, Tensor>> {
        let mut gradients = FxHashMap::default();
        gradients.reserve(1);

        match self.input.dtype() {
            DataType::Float32 => {
                let input_slice = self.input.data().as_f32_slice().ok_or_else(|| {
                    MinitensorError::internal_error("Failed to get f32 slice from input tensor")
                })?;
                let grad_out_slice = grad_output.data().as_f32_slice().ok_or_else(|| {
                    MinitensorError::internal_error(
                        "Failed to get f32 slice from grad_output tensor",
                    )
                })?;

                let mut grad_data = TensorData::uninitialized_on_device(
                    input_slice.len(),
                    DataType::Float32,
                    self.input.device(),
                );
                let grad_slice = grad_data.as_f32_slice_mut().ok_or_else(|| {
                    MinitensorError::internal_error(
                        "Failed to get mutable f32 slice from gradient tensor",
                    )
                })?;

                let beta = self.beta as f32;
                let threshold = self.threshold as f32;
                for ((grad_slot, &x), &gout) in grad_slice
                    .iter_mut()
                    .zip(input_slice.iter())
                    .zip(grad_out_slice.iter())
                {
                    let scaled = beta * x;
                    *grad_slot = if scaled > threshold {
                        gout
                    } else {
                        gout / (1.0 + (-scaled).exp())
                    };
                }

                let grad_tensor = Tensor::new(
                    Arc::new(grad_data),
                    self.input.shape().clone(),
                    DataType::Float32,
                    self.input.device(),
                    false,
                );
                gradients.insert(self.input_id, grad_tensor);
            }
            DataType::Float64 => {
                let input_slice = self.input.data().as_f64_slice().ok_or_else(|| {
                    MinitensorError::internal_error("Failed to get f64 slice from input tensor")
                })?;
                let grad_out_slice = grad_output.data().as_f64_slice().ok_or_else(|| {
                    MinitensorError::internal_error(
                        "Failed to get f64 slice from grad_output tensor",
                    )
                })?;

                let mut grad_data = TensorData::uninitialized_on_device(
                    input_slice.len(),
                    DataType::Float64,
                    self.input.device(),
                );
                let grad_slice = grad_data.as_f64_slice_mut().ok_or_else(|| {
                    MinitensorError::internal_error(
                        "Failed to get mutable f64 slice from gradient tensor",
                    )
                })?;

                let beta = self.beta;
                let threshold = self.threshold;
                for ((grad_slot, &x), &gout) in grad_slice
                    .iter_mut()
                    .zip(input_slice.iter())
                    .zip(grad_out_slice.iter())
                {
                    let scaled = beta * x;
                    *grad_slot = if scaled > threshold {
                        gout
                    } else {
                        gout / (1.0 + (-scaled).exp())
                    };
                }

                let grad_tensor = Tensor::new(
                    Arc::new(grad_data),
                    self.input.shape().clone(),
                    DataType::Float64,
                    self.input.device(),
                    false,
                );
                gradients.insert(self.input_id, grad_tensor);
            }
            _ => {
                return Err(MinitensorError::invalid_operation(
                    "Softplus gradient only defined for floating point tensors",
                ));
            }
        }

        Ok(gradients)
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

impl GradientFunction for GeluBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<FxHashMap<TensorId, Tensor>> {
        let mut gradients = FxHashMap::default();
        gradients.reserve(1);

        match self.input.dtype() {
            DataType::Float32 => {
                let input_slice = self.input.data().as_f32_slice().ok_or_else(|| {
                    MinitensorError::internal_error("Failed to get f32 slice from input tensor")
                })?;
                let grad_out_slice = grad_output.data().as_f32_slice().ok_or_else(|| {
                    MinitensorError::internal_error(
                        "Failed to get f32 slice from grad_output tensor",
                    )
                })?;

                let mut grad_data = TensorData::uninitialized_on_device(
                    input_slice.len(),
                    DataType::Float32,
                    self.input.device(),
                );
                let grad_slice = grad_data.as_f32_slice_mut().ok_or_else(|| {
                    MinitensorError::internal_error(
                        "Failed to get mutable f32 slice from gradient tensor",
                    )
                })?;

                if self.approximate {
                    let coeff = (2.0f32 / std::f32::consts::PI).sqrt();
                    for ((grad_slot, &x), &gout) in grad_slice
                        .iter_mut()
                        .zip(input_slice.iter())
                        .zip(grad_out_slice.iter())
                    {
                        let x2 = x * x;
                        let inner = coeff * (x + 0.044715f32 * x * x2);
                        let tanh_inner = inner.tanh();
                        let sech2 = 1.0f32 - tanh_inner * tanh_inner;
                        let grad_val = 0.5f32 * (1.0f32 + tanh_inner)
                            + 0.5f32 * x * sech2 * coeff * (1.0f32 + 3.0f32 * 0.044715f32 * x2);
                        *grad_slot = gout * grad_val;
                    }
                } else {
                    let inv_sqrt_2 = std::f32::consts::FRAC_1_SQRT_2;
                    let inv_sqrt_2pi = 1.0f32 / ((2.0f32 * std::f32::consts::PI).sqrt());
                    for ((grad_slot, &x), &gout) in grad_slice
                        .iter_mut()
                        .zip(input_slice.iter())
                        .zip(grad_out_slice.iter())
                    {
                        let cdf = 0.5f32 * (1.0f32 + erff(x * inv_sqrt_2));
                        let pdf = (-0.5f32 * x * x).exp() * inv_sqrt_2pi;
                        let grad_val = cdf + x * pdf;
                        *grad_slot = gout * grad_val;
                    }
                }

                let grad_tensor = Tensor::new(
                    Arc::new(grad_data),
                    self.input.shape().clone(),
                    DataType::Float32,
                    self.input.device(),
                    false,
                );
                gradients.insert(self.input_id, grad_tensor);
            }
            DataType::Float64 => {
                let input_slice = self.input.data().as_f64_slice().ok_or_else(|| {
                    MinitensorError::internal_error("Failed to get f64 slice from input tensor")
                })?;
                let grad_out_slice = grad_output.data().as_f64_slice().ok_or_else(|| {
                    MinitensorError::internal_error(
                        "Failed to get f64 slice from grad_output tensor",
                    )
                })?;

                let mut grad_data = TensorData::uninitialized_on_device(
                    input_slice.len(),
                    DataType::Float64,
                    self.input.device(),
                );
                let grad_slice = grad_data.as_f64_slice_mut().ok_or_else(|| {
                    MinitensorError::internal_error(
                        "Failed to get mutable f64 slice from gradient tensor",
                    )
                })?;

                if self.approximate {
                    let coeff = (2.0f64 / std::f64::consts::PI).sqrt();
                    for ((grad_slot, &x), &gout) in grad_slice
                        .iter_mut()
                        .zip(input_slice.iter())
                        .zip(grad_out_slice.iter())
                    {
                        let x2 = x * x;
                        let inner = coeff * (x + 0.044715f64 * x * x2);
                        let tanh_inner = inner.tanh();
                        let sech2 = 1.0f64 - tanh_inner * tanh_inner;
                        let grad_val = 0.5f64 * (1.0f64 + tanh_inner)
                            + 0.5f64 * x * sech2 * coeff * (1.0f64 + 3.0f64 * 0.044715f64 * x2);
                        *grad_slot = gout * grad_val;
                    }
                } else {
                    let inv_sqrt_2 = std::f64::consts::FRAC_1_SQRT_2;
                    let inv_sqrt_2pi = 1.0f64 / ((2.0f64 * std::f64::consts::PI).sqrt());
                    for ((grad_slot, &x), &gout) in grad_slice
                        .iter_mut()
                        .zip(input_slice.iter())
                        .zip(grad_out_slice.iter())
                    {
                        let cdf = 0.5f64 * (1.0f64 + erf(x * inv_sqrt_2));
                        let pdf = (-0.5f64 * x * x).exp() * inv_sqrt_2pi;
                        let grad_val = cdf + x * pdf;
                        *grad_slot = gout * grad_val;
                    }
                }

                let grad_tensor = Tensor::new(
                    Arc::new(grad_data),
                    self.input.shape().clone(),
                    DataType::Float64,
                    self.input.device(),
                    false,
                );
                gradients.insert(self.input_id, grad_tensor);
            }
            _ => {
                return Err(MinitensorError::invalid_operation(
                    "GELU backward only supports floating point tensors",
                ));
            }
        }

        Ok(gradients)
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
        let mut gradients = FxHashMap::default();
        gradients.reserve(1);

        match self.output.dtype() {
            DataType::Float32 => {
                let output_slice = self.output.data().as_f32_slice().ok_or_else(|| {
                    MinitensorError::internal_error("Failed to get f32 slice from output tensor")
                })?;
                let grad_out_slice = grad_output.data().as_f32_slice().ok_or_else(|| {
                    MinitensorError::internal_error(
                        "Failed to get f32 slice from grad_output tensor",
                    )
                })?;

                let mut grad_data = TensorData::uninitialized_on_device(
                    output_slice.len(),
                    DataType::Float32,
                    self.output.device(),
                );
                let grad_slice = grad_data.as_f32_slice_mut().ok_or_else(|| {
                    MinitensorError::internal_error(
                        "Failed to get mutable f32 slice from gradient tensor",
                    )
                })?;

                let alpha = self.alpha as f32;
                for ((grad_slot, &out), &gout) in grad_slice
                    .iter_mut()
                    .zip(output_slice.iter())
                    .zip(grad_out_slice.iter())
                {
                    let local_grad = if out > 0.0f32 { 1.0f32 } else { out + alpha };
                    *grad_slot = gout * local_grad;
                }

                let grad_tensor = Tensor::new(
                    Arc::new(grad_data),
                    self.output.shape().clone(),
                    DataType::Float32,
                    self.output.device(),
                    false,
                );
                gradients.insert(self.input_id, grad_tensor);
            }
            DataType::Float64 => {
                let output_slice = self.output.data().as_f64_slice().ok_or_else(|| {
                    MinitensorError::internal_error("Failed to get f64 slice from output tensor")
                })?;
                let grad_out_slice = grad_output.data().as_f64_slice().ok_or_else(|| {
                    MinitensorError::internal_error(
                        "Failed to get f64 slice from grad_output tensor",
                    )
                })?;

                let mut grad_data = TensorData::uninitialized_on_device(
                    output_slice.len(),
                    DataType::Float64,
                    self.output.device(),
                );
                let grad_slice = grad_data.as_f64_slice_mut().ok_or_else(|| {
                    MinitensorError::internal_error(
                        "Failed to get mutable f64 slice from gradient tensor",
                    )
                })?;

                for ((grad_slot, &out), &gout) in grad_slice
                    .iter_mut()
                    .zip(output_slice.iter())
                    .zip(grad_out_slice.iter())
                {
                    let local_grad = if out > 0.0f64 {
                        1.0f64
                    } else {
                        out + self.alpha
                    };
                    *grad_slot = gout * local_grad;
                }

                let grad_tensor = Tensor::new(
                    Arc::new(grad_data),
                    self.output.shape().clone(),
                    DataType::Float64,
                    self.output.device(),
                    false,
                );
                gradients.insert(self.input_id, grad_tensor);
            }
            _ => {
                return Err(MinitensorError::invalid_operation(
                    "ELU backward only supports floating point tensors",
                ));
            }
        }

        Ok(gradients)
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

impl GradientFunction for SeluBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<FxHashMap<TensorId, Tensor>> {
        let mut gradients = FxHashMap::default();
        gradients.reserve(1);

        match self.output.dtype() {
            DataType::Float32 => {
                let output_slice = self.output.data().as_f32_slice().ok_or_else(|| {
                    MinitensorError::internal_error("Failed to get f32 slice from output tensor")
                })?;
                let grad_out_slice = grad_output.data().as_f32_slice().ok_or_else(|| {
                    MinitensorError::internal_error(
                        "Failed to get f32 slice from grad_output tensor",
                    )
                })?;

                let mut grad_data = TensorData::uninitialized_on_device(
                    output_slice.len(),
                    DataType::Float32,
                    self.output.device(),
                );
                let grad_slice = grad_data.as_f32_slice_mut().ok_or_else(|| {
                    MinitensorError::internal_error(
                        "Failed to get mutable f32 slice from gradient tensor",
                    )
                })?;

                const SCALE: f32 = 1.050701;
                const ALPHA: f32 = 1.6732632;
                for ((grad_slot, &out), &gout) in grad_slice
                    .iter_mut()
                    .zip(output_slice.iter())
                    .zip(grad_out_slice.iter())
                {
                    let local_grad = if out > 0.0f32 {
                        SCALE
                    } else {
                        out + SCALE * ALPHA
                    };
                    *grad_slot = gout * local_grad;
                }

                let grad_tensor = Tensor::new(
                    Arc::new(grad_data),
                    self.output.shape().clone(),
                    DataType::Float32,
                    self.output.device(),
                    false,
                );
                gradients.insert(self.input_id, grad_tensor);
            }
            DataType::Float64 => {
                let output_slice = self.output.data().as_f64_slice().ok_or_else(|| {
                    MinitensorError::internal_error("Failed to get f64 slice from output tensor")
                })?;
                let grad_out_slice = grad_output.data().as_f64_slice().ok_or_else(|| {
                    MinitensorError::internal_error(
                        "Failed to get f64 slice from grad_output tensor",
                    )
                })?;

                let mut grad_data = TensorData::uninitialized_on_device(
                    output_slice.len(),
                    DataType::Float64,
                    self.output.device(),
                );
                let grad_slice = grad_data.as_f64_slice_mut().ok_or_else(|| {
                    MinitensorError::internal_error(
                        "Failed to get mutable f64 slice from gradient tensor",
                    )
                })?;

                const SCALE: f64 = 1.0507009873554804934193349852946;
                const ALPHA: f64 = 1.6732632423543772848170429916717;
                for ((grad_slot, &out), &gout) in grad_slice
                    .iter_mut()
                    .zip(output_slice.iter())
                    .zip(grad_out_slice.iter())
                {
                    let local_grad = if out > 0.0f64 {
                        SCALE
                    } else {
                        out + SCALE * ALPHA
                    };
                    *grad_slot = gout * local_grad;
                }

                let grad_tensor = Tensor::new(
                    Arc::new(grad_data),
                    self.output.shape().clone(),
                    DataType::Float64,
                    self.output.device(),
                    false,
                );
                gradients.insert(self.input_id, grad_tensor);
            }
            _ => {
                return Err(MinitensorError::invalid_operation(
                    "SELU backward only supports floating point tensors",
                ));
            }
        }

        Ok(gradients)
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
        let mut gradients = FxHashMap::default();
        gradients.reserve(1);

        match self.input.dtype() {
            DataType::Float32 => {
                let input_slice = self.input.data().as_f32_slice().ok_or_else(|| {
                    MinitensorError::internal_error("Failed to get f32 slice from input tensor")
                })?;
                let grad_out_slice = grad_output.data().as_f32_slice().ok_or_else(|| {
                    MinitensorError::internal_error(
                        "Failed to get f32 slice from grad_output tensor",
                    )
                })?;

                let mut grad_data = TensorData::uninitialized_on_device(
                    input_slice.len(),
                    DataType::Float32,
                    self.input.device(),
                );
                let grad_slice = grad_data.as_f32_slice_mut().ok_or_else(|| {
                    MinitensorError::internal_error(
                        "Failed to get mutable f32 slice from gradient tensor",
                    )
                })?;

                for ((grad_slot, &x), &gout) in grad_slice
                    .iter_mut()
                    .zip(input_slice.iter())
                    .zip(grad_out_slice.iter())
                {
                    let sigmoid = stable_sigmoid_f32(x);
                    let grad_val = sigmoid * (1.0f32 + x * (1.0f32 - sigmoid));
                    *grad_slot = gout * grad_val;
                }

                let grad_tensor = Tensor::new(
                    Arc::new(grad_data),
                    self.input.shape().clone(),
                    DataType::Float32,
                    self.input.device(),
                    false,
                );
                gradients.insert(self.input_id, grad_tensor);
            }
            DataType::Float64 => {
                let input_slice = self.input.data().as_f64_slice().ok_or_else(|| {
                    MinitensorError::internal_error("Failed to get f64 slice from input tensor")
                })?;
                let grad_out_slice = grad_output.data().as_f64_slice().ok_or_else(|| {
                    MinitensorError::internal_error(
                        "Failed to get f64 slice from grad_output tensor",
                    )
                })?;

                let mut grad_data = TensorData::uninitialized_on_device(
                    input_slice.len(),
                    DataType::Float64,
                    self.input.device(),
                );
                let grad_slice = grad_data.as_f64_slice_mut().ok_or_else(|| {
                    MinitensorError::internal_error(
                        "Failed to get mutable f64 slice from gradient tensor",
                    )
                })?;

                for ((grad_slot, &x), &gout) in grad_slice
                    .iter_mut()
                    .zip(input_slice.iter())
                    .zip(grad_out_slice.iter())
                {
                    let sigmoid = stable_sigmoid_f64(x);
                    let grad_val = sigmoid * (1.0f64 + x * (1.0f64 - sigmoid));
                    *grad_slot = gout * grad_val;
                }

                let grad_tensor = Tensor::new(
                    Arc::new(grad_data),
                    self.input.shape().clone(),
                    DataType::Float64,
                    self.input.device(),
                    false,
                );
                gradients.insert(self.input_id, grad_tensor);
            }
            _ => {
                return Err(MinitensorError::invalid_operation(
                    "SiLU backward only supports floating point tensors",
                ));
            }
        }

        Ok(gradients)
    }

    fn input_ids(&self) -> &[TensorId] {
        std::slice::from_ref(&self.input_id)
    }
}

#[inline]
fn stable_sigmoid_f32(x: f32) -> f32 {
    if x >= 0.0 {
        let exp_neg = (-x).exp();
        1.0 / (1.0 + exp_neg)
    } else {
        let exp_pos = x.exp();
        exp_pos / (1.0 + exp_pos)
    }
}

#[inline]
fn stable_sigmoid_f64(x: f64) -> f64 {
    if x >= 0.0 {
        let exp_neg = (-x).exp();
        1.0 / (1.0 + exp_neg)
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
        let mut gradients = FxHashMap::default();
        gradients.reserve(1);

        match self.input.dtype() {
            DataType::Float32 => {
                let input_slice = self.input.data().as_f32_slice().ok_or_else(|| {
                    MinitensorError::internal_error("Failed to get f32 slice from input tensor")
                })?;
                let grad_out_slice = grad_output.data().as_f32_slice().ok_or_else(|| {
                    MinitensorError::internal_error(
                        "Failed to get f32 slice from grad_output tensor",
                    )
                })?;

                let mut grad_data = TensorData::uninitialized_on_device(
                    input_slice.len(),
                    DataType::Float32,
                    self.input.device(),
                );
                let grad_slice = grad_data.as_f32_slice_mut().ok_or_else(|| {
                    MinitensorError::internal_error(
                        "Failed to get mutable f32 slice from gradient tensor",
                    )
                })?;

                for ((grad_slot, &x), &gout) in grad_slice
                    .iter_mut()
                    .zip(input_slice.iter())
                    .zip(grad_out_slice.iter())
                {
                    let denom = 1.0f32 + x.abs();
                    let local_grad = 1.0f32 / (denom * denom);
                    *grad_slot = gout * local_grad;
                }

                let grad_tensor = Tensor::new(
                    Arc::new(grad_data),
                    self.input.shape().clone(),
                    DataType::Float32,
                    self.input.device(),
                    false,
                );
                gradients.insert(self.input_id, grad_tensor);
            }
            DataType::Float64 => {
                let input_slice = self.input.data().as_f64_slice().ok_or_else(|| {
                    MinitensorError::internal_error("Failed to get f64 slice from input tensor")
                })?;
                let grad_out_slice = grad_output.data().as_f64_slice().ok_or_else(|| {
                    MinitensorError::internal_error(
                        "Failed to get f64 slice from grad_output tensor",
                    )
                })?;

                let mut grad_data = TensorData::uninitialized_on_device(
                    input_slice.len(),
                    DataType::Float64,
                    self.input.device(),
                );
                let grad_slice = grad_data.as_f64_slice_mut().ok_or_else(|| {
                    MinitensorError::internal_error(
                        "Failed to get mutable f64 slice from gradient tensor",
                    )
                })?;

                for ((grad_slot, &x), &gout) in grad_slice
                    .iter_mut()
                    .zip(input_slice.iter())
                    .zip(grad_out_slice.iter())
                {
                    let denom = 1.0f64 + x.abs();
                    let local_grad = 1.0f64 / (denom * denom);
                    *grad_slot = gout * local_grad;
                }

                let grad_tensor = Tensor::new(
                    Arc::new(grad_data),
                    self.input.shape().clone(),
                    DataType::Float64,
                    self.input.device(),
                    false,
                );
                gradients.insert(self.input_id, grad_tensor);
            }
            _ => {
                return Err(MinitensorError::invalid_operation(
                    "Softsign backward only supports floating point tensors",
                ));
            }
        }

        Ok(gradients)
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
}

impl GradientFunction for LogAddExpBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<FxHashMap<TensorId, Tensor>> {
        let mut gradients = FxHashMap::default();
        gradients.reserve(2);

        let lhs_diff = arithmetic::sub(&self.lhs.detach(), &self.output.detach())?;
        let lhs_term = lhs_diff.exp()?;
        let lhs_mul = arithmetic::mul(&lhs_term, grad_output)?;
        let lhs_grad =
            reduce_gradient_for_broadcasting(&lhs_mul, &Shape::new(self.input_shapes[0].clone()))?;
        gradients.insert(self.input_ids[0], lhs_grad);

        let rhs_diff = arithmetic::sub(&self.rhs.detach(), &self.output.detach())?;
        let rhs_term = rhs_diff.exp()?;
        let rhs_mul = arithmetic::mul(&rhs_term, grad_output)?;
        let rhs_grad =
            reduce_gradient_for_broadcasting(&rhs_mul, &Shape::new(self.input_shapes[1].clone()))?;
        gradients.insert(self.input_ids[1], rhs_grad);

        Ok(gradients)
    }

    fn input_ids(&self) -> &[TensorId] {
        &self.input_ids
    }
}
