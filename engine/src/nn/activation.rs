// Copyright (c) 2025 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

use super::Layer;
use crate::{
    device::Device,
    error::{MinitensorError, Result},
    operations::{
        activation::{exp, leaky_relu, relu, sigmoid, softmax, tanh},
        arithmetic,
    },
    tensor::{DataType, Shape, Tensor, TensorData},
};
use std::sync::Arc;

fn scalar_tensor(
    value: f64,
    dtype: DataType,
    device: Device,
    requires_grad: bool,
) -> Result<Tensor> {
    match dtype {
        DataType::Float32 => {
            let td = TensorData::from_vec_f32(vec![value as f32], device);
            Ok(Tensor::new(
                Arc::new(td),
                Shape::new(vec![1]),
                dtype,
                device,
                requires_grad,
            ))
        }
        DataType::Float64 => {
            let td = TensorData::from_vec_f64(vec![value], device);
            Ok(Tensor::new(
                Arc::new(td),
                Shape::new(vec![1]),
                dtype,
                device,
                requires_grad,
            ))
        }
        _ => Err(MinitensorError::invalid_argument(
            "Scalar tensors only support floating point types".to_string(),
        )),
    }
}

fn cached_scalar<'a>(
    cache: &'a mut Option<Tensor>,
    value: f64,
    dtype: DataType,
    device: Device,
) -> Result<&'a Tensor> {
    let needs_update = match cache {
        Some(t) => t.dtype() != dtype || t.device() != device,
        None => true,
    };
    if needs_update {
        *cache = Some(scalar_tensor(value, dtype, device, false)?);
    }
    Ok(cache.as_ref().unwrap())
}

/// ReLU (Rectified Linear Unit) activation layer
///
/// Applies the rectified linear unit function element-wise:
/// ReLU(x) = max(0, x)
#[derive(Clone)]
pub struct ReLU;

impl ReLU {
    /// Create a new ReLU activation layer
    pub fn new() -> Self {
        Self
    }
}

impl Default for ReLU {
    fn default() -> Self {
        Self::new()
    }
}

impl Layer for ReLU {
    fn forward(&mut self, input: &Tensor) -> Result<Tensor> {
        relu(input)
    }

    fn parameters(&self) -> Vec<&Tensor> {
        vec![] // No parameters
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        vec![] // No parameters
    }
}

/// Sigmoid activation layer
///
/// Applies the sigmoid function element-wise:
/// Sigmoid(x) = 1 / (1 + exp(-x))
#[derive(Clone)]
pub struct Sigmoid;

impl Sigmoid {
    /// Create a new Sigmoid activation layer
    pub fn new() -> Self {
        Self
    }
}

impl Default for Sigmoid {
    fn default() -> Self {
        Self::new()
    }
}

impl Layer for Sigmoid {
    fn forward(&mut self, input: &Tensor) -> Result<Tensor> {
        sigmoid(input)
    }

    fn parameters(&self) -> Vec<&Tensor> {
        vec![] // No parameters
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        vec![] // No parameters
    }
}

/// Tanh (Hyperbolic Tangent) activation layer
///
/// Applies the hyperbolic tangent function element-wise:
/// Tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
#[derive(Clone)]
pub struct Tanh;

impl Tanh {
    /// Create a new Tanh activation layer
    pub fn new() -> Self {
        Self
    }
}

impl Default for Tanh {
    fn default() -> Self {
        Self::new()
    }
}

impl Layer for Tanh {
    fn forward(&mut self, input: &Tensor) -> Result<Tensor> {
        tanh(input)
    }

    fn parameters(&self) -> Vec<&Tensor> {
        vec![] // No parameters
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        vec![] // No parameters
    }
}

/// Softmax activation layer
///
/// Applies the softmax function to an n-dimensional input tensor
/// rescaling them so that the elements of the n-dimensional output tensor
/// lie in the range [0,1] and sum to 1.
#[derive(Clone)]
pub struct Softmax {
    dim: Option<usize>,
}

impl Softmax {
    /// Create a new Softmax activation layer
    ///
    /// # Arguments
    /// * `dim` - A dimension along which Softmax will be computed (so every slice
    ///           along dim will sum to 1). Default: None (applies to the last dimension)
    pub fn new(dim: Option<usize>) -> Self {
        Self { dim }
    }

    /// Get the dimension along which softmax is computed
    pub fn dim(&self) -> Option<usize> {
        self.dim
    }
}

impl Default for Softmax {
    fn default() -> Self {
        Self::new(None)
    }
}

impl Layer for Softmax {
    fn forward(&mut self, input: &Tensor) -> Result<Tensor> {
        let dim = self.dim.or(Some(input.ndim() - 1));
        softmax(input, dim)
    }

    fn parameters(&self) -> Vec<&Tensor> {
        vec![] // No parameters
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        vec![] // No parameters
    }
}

/// LeakyReLU activation layer
///
/// Applies the leaky rectified linear unit function element-wise:
/// LeakyReLU(x) = max(negative_slope * x, x)
#[derive(Clone)]
pub struct LeakyReLU {
    negative_slope: f64,
}

impl LeakyReLU {
    /// Create a new LeakyReLU activation layer
    ///
    /// # Arguments
    /// * `negative_slope` - Controls the angle of the negative slope. Default: 0.01
    pub fn new(negative_slope: Option<f64>) -> Self {
        Self {
            negative_slope: negative_slope.unwrap_or(0.01),
        }
    }

    /// Get the negative slope parameter
    pub fn negative_slope(&self) -> f64 {
        self.negative_slope
    }
}

impl Default for LeakyReLU {
    fn default() -> Self {
        Self::new(None)
    }
}

impl Layer for LeakyReLU {
    fn forward(&mut self, input: &Tensor) -> Result<Tensor> {
        leaky_relu(input, self.negative_slope)
    }

    fn parameters(&self) -> Vec<&Tensor> {
        vec![] // No parameters
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        vec![] // No parameters
    }
}

/// ELU (Exponential Linear Unit) activation layer
///
/// Applies the exponential linear unit function element-wise:
/// ELU(x) = max(0, x) + min(0, alpha * (exp(x) - 1))
#[derive(Clone)]
pub struct ELU {
    alpha: f64,
    alpha_tensor: Option<Tensor>,
}

impl ELU {
    /// Create a new ELU activation layer
    ///
    /// # Arguments
    /// * `alpha` - The α value for the ELU formulation. Default: 1.0
    pub fn new(alpha: Option<f64>) -> Self {
        Self {
            alpha: alpha.unwrap_or(1.0),
            alpha_tensor: None,
        }
    }

    /// Get the alpha parameter
    pub fn alpha(&self) -> f64 {
        self.alpha
    }
}

impl Default for ELU {
    fn default() -> Self {
        Self::new(None)
    }
}

impl Layer for ELU {
    fn forward(&mut self, input: &Tensor) -> Result<Tensor> {
        // Positive part: max(0, x)
        let positive = relu(input)?;

        // Negative part: alpha * (exp(min(0, x)) - 1)
        // Compute negative input values (x - relu(x) gives x for x<=0 and 0 otherwise)
        let neg_input = arithmetic::sub(input, &positive)?;
        let exp_neg = exp(&neg_input)?;
        let ones = Tensor::ones(
            neg_input.shape().clone(),
            neg_input.dtype(),
            neg_input.device(),
            false,
        );
        let exp_minus_one = arithmetic::sub(&exp_neg, &ones)?;
        let alpha = cached_scalar(
            &mut self.alpha_tensor,
            self.alpha,
            input.dtype(),
            input.device(),
        )?;
        let neg_part = arithmetic::mul(&exp_minus_one, alpha)?;

        // Combine positive and negative parts
        arithmetic::add(&positive, &neg_part)
    }

    fn parameters(&self) -> Vec<&Tensor> {
        vec![] // No parameters
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        vec![] // No parameters
    }
}

/// GELU (Gaussian Error Linear Unit) activation layer
///
/// Applies the Gaussian Error Linear Unit function:
/// GELU(x) = x * theta(x)
/// where theta(x) is the Cumulative Distribution Function for Gaussian Distribution.
#[derive(Clone)]
pub struct GELU {
    half: Option<Tensor>,
    one: Option<Tensor>,
    coeff: Option<Tensor>,
    cubic_coeff: Option<Tensor>,
}

impl GELU {
    /// Create a new GELU activation layer
    pub fn new() -> Self {
        Self {
            half: None,
            one: None,
            coeff: None,
            cubic_coeff: None,
        }
    }

    fn prepare_constants(&mut self, dtype: DataType, device: Device) -> Result<()> {
        let _ = cached_scalar(&mut self.half, 0.5, dtype, device)?;
        let _ = cached_scalar(&mut self.one, 1.0, dtype, device)?;
        let _ = cached_scalar(
            &mut self.coeff,
            (2.0 / std::f64::consts::PI).sqrt(),
            dtype,
            device,
        )?;
        let _ = cached_scalar(&mut self.cubic_coeff, 0.044_715, dtype, device)?;
        Ok(())
    }
}

impl Default for GELU {
    fn default() -> Self {
        Self::new()
    }
}

impl Layer for GELU {
    fn forward(&mut self, input: &Tensor) -> Result<Tensor> {
        // Use tanh-based approximation:
        // 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715x^3)))
        self.prepare_constants(input.dtype(), input.device())?;

        let half = self.half.as_ref().unwrap();
        let one = self.one.as_ref().unwrap();
        let coeff = self.coeff.as_ref().unwrap();
        let cubic_coeff = self.cubic_coeff.as_ref().unwrap();

        let x_sq = arithmetic::mul(input, input)?;
        let x_cubed = arithmetic::mul(&x_sq, input)?;
        let scaled_cubic = arithmetic::mul(&x_cubed, cubic_coeff)?;
        let inner = arithmetic::add(input, &scaled_cubic)?;
        let inner = arithmetic::mul(&inner, coeff)?;
        let tanh_inner = tanh(&inner)?;
        let one_plus_tanh = arithmetic::add(one, &tanh_inner)?;
        let half_x = arithmetic::mul(input, half)?;
        arithmetic::mul(&half_x, &one_plus_tanh)
    }

    fn parameters(&self) -> Vec<&Tensor> {
        vec![] // No parameters
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        vec![] // No parameters
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::Device;
    use crate::tensor::{DataType, Shape, Tensor, TensorData};
    use std::sync::Arc;

    #[test]
    fn test_relu_creation() {
        let relu = ReLU::new();
        assert_eq!(relu.parameters().len(), 0);
    }

    #[test]
    fn test_sigmoid_creation() {
        let sigmoid = Sigmoid::new();
        assert_eq!(sigmoid.parameters().len(), 0);
    }

    #[test]
    fn test_tanh_creation() {
        let tanh = Tanh::new();
        assert_eq!(tanh.parameters().len(), 0);
    }

    #[test]
    fn test_softmax_creation() {
        let softmax = Softmax::new(Some(1));
        assert_eq!(softmax.dim(), Some(1));
        assert_eq!(softmax.parameters().len(), 0);

        let softmax_default = Softmax::default();
        assert_eq!(softmax_default.dim(), None);
    }

    #[test]
    fn test_leaky_relu_creation() {
        let leaky_relu = LeakyReLU::new(Some(0.02));
        assert_eq!(leaky_relu.negative_slope(), 0.02);
        assert_eq!(leaky_relu.parameters().len(), 0);

        let leaky_relu_default = LeakyReLU::default();
        assert_eq!(leaky_relu_default.negative_slope(), 0.01);
    }

    #[test]
    fn test_elu_creation() {
        let elu = ELU::new(Some(1.5));
        assert_eq!(elu.alpha(), 1.5);
        assert_eq!(elu.parameters().len(), 0);

        let elu_default = ELU::default();
        assert_eq!(elu_default.alpha(), 1.0);
    }

    #[test]
    fn test_gelu_creation() {
        let gelu = GELU::new();
        assert_eq!(gelu.parameters().len(), 0);
    }

    #[test]
    fn test_elu_forward_values() {
        let mut elu = ELU::new(Some(1.0));
        let data = TensorData::from_vec_f32(vec![-1.0, 0.0, 1.0], Device::cpu());
        let input = Tensor::new(
            Arc::new(data),
            Shape::new(vec![3]),
            DataType::Float32,
            Device::cpu(),
            false,
        );
        let output = elu.forward(&input).unwrap();
        let out_slice = output.data().as_f32_slice().unwrap();
        let expected = vec![(-1f32).exp() - 1.0, 0.0, 1.0];
        for (o, e) in out_slice.iter().zip(expected.iter()) {
            assert!((o - e).abs() < 1e-4);
        }
    }

    #[test]
    fn test_relu_forward_values() {
        let mut relu = ReLU::new();
        let data = TensorData::from_vec_f32(vec![-1.0, 0.0, 1.0], Device::cpu());
        let input = Tensor::new(
            Arc::new(data),
            Shape::new(vec![3]),
            DataType::Float32,
            Device::cpu(),
            false,
        );
        let output = relu.forward(&input).unwrap();
        let out_slice = output.data().as_f32_slice().unwrap();
        assert_eq!(out_slice, &[0.0, 0.0, 1.0]);
    }

    #[test]
    fn test_leaky_relu_forward_values() {
        let mut lr = LeakyReLU::new(Some(0.1));
        let data = TensorData::from_vec_f32(vec![-2.0, 0.0, 2.0], Device::cpu());
        let input = Tensor::new(
            Arc::new(data),
            Shape::new(vec![3]),
            DataType::Float32,
            Device::cpu(),
            false,
        );
        let output = lr.forward(&input).unwrap();
        let out_slice = output.data().as_f32_slice().unwrap();
        assert_eq!(out_slice, &[-0.2, 0.0, 2.0]);
    }

    #[test]
    fn test_gelu_forward_values() {
        let mut gelu = GELU::new();
        let data = TensorData::from_vec_f32(vec![-1.0, 0.0, 1.0], Device::cpu());
        let input = Tensor::new(
            Arc::new(data),
            Shape::new(vec![3]),
            DataType::Float32,
            Device::cpu(),
            false,
        );
        let output = gelu.forward(&input).unwrap();
        let out_slice = output.data().as_f32_slice().unwrap();
        let expected: Vec<f32> = [-1.0f32, 0.0, 1.0]
            .iter()
            .map(|&x| {
                let x3 = x * x * x;
                let inner = x + 0.044_715 * x3;
                let inner = (2.0 / std::f32::consts::PI).sqrt() * inner;
                0.5 * x * (1.0 + inner.tanh())
            })
            .collect();
        for (o, e) in out_slice.iter().zip(expected.iter()) {
            assert!((o - e).abs() < 1e-4);
        }
    }

    #[test]
    fn test_activation_forward_shapes() {
        let input = Tensor::zeros(
            Shape::new(vec![2, 3, 4]),
            DataType::Float32,
            Device::cpu(),
            false,
        );

        // Test that activations preserve input shape (when operations are implemented)
        let mut relu = ReLU::new();
        let mut sigmoid = Sigmoid::new();
        let mut tanh = Tanh::new();
        let mut softmax = Softmax::new(Some(2));
        let mut leaky_relu = LeakyReLU::new(None);
        let mut elu = ELU::new(None);
        let mut gelu = GELU::new();

        let _ = relu.forward(&input);
        let _ = sigmoid.forward(&input);
        let _ = tanh.forward(&input);
        let _ = softmax.forward(&input);
        let _ = leaky_relu.forward(&input);
        let _ = elu.forward(&input);
        let _ = gelu.forward(&input);
    }
}
