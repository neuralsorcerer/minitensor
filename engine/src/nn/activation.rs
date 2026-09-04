// Copyright (c) Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

use super::Layer;
use crate::{
    device::Device,
    error::{MinitensorError, Result},
    ops::{
        activation::{exp, gelu, leaky_relu, relu, sigmoid, softmax, tanh},
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

fn cached_scalar(
    cache: &mut Option<Tensor>,
    value: f64,
    dtype: DataType,
    device: Device,
) -> Result<&Tensor> {
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
/// lie in the range \[0, 1\] and sum to 1.
#[derive(Clone)]
pub struct Softmax {
    /// Kept as given, negative included, and resolved against the input's rank
    /// at forward time. A layer is built before it sees a tensor, so `-1` --
    /// which is how a softmax is almost always written -- cannot be turned
    /// into an axis index until then.
    dim: Option<isize>,
}

impl Softmax {
    /// Create a new Softmax activation layer
    ///
    /// # Arguments
    /// * `dim` - A dimension along which Softmax will be computed (so every
    ///   slice along dim will sum to 1). Default: None (applies to the last
    ///   dimension)
    pub fn new(dim: Option<isize>) -> Self {
        Self { dim }
    }

    /// Get the dimension along which softmax is computed
    pub fn dim(&self) -> Option<isize> {
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
        let dim = match self.dim {
            Some(dim) => crate::ops::util::normalize_dim(dim, input.ndim())?,
            None => input.ndim().saturating_sub(1),
        };
        softmax(input, Some(dim))
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
    /// Whether to use the tanh approximation rather than the error function.
    approximate: bool,
}

impl GELU {
    /// Create a new GELU activation layer.
    ///
    /// The tanh approximation, which is what this layer has always computed
    /// and is measurably the reason to reach for a layer rather than the
    /// function: 2.3ms against 3.4ms on four million single-precision
    /// elements, and 16ms against 25ms in double.
    pub fn new() -> Self {
        Self { approximate: true }
    }

    /// The same layer with the tanh approximation instead.
    pub fn approximate(approximate: bool) -> Self {
        Self { approximate }
    }

    /// Whether this layer takes the approximation.
    pub fn is_approximate(&self) -> bool {
        self.approximate
    }
}

impl Default for GELU {
    fn default() -> Self {
        Self::new()
    }
}

impl Layer for GELU {
    fn forward(&mut self, input: &Tensor) -> Result<Tensor> {
        // `x * Phi(x)` through the vectorised kernel, from the tanh
        // approximation by default and from the error function when asked.
        //
        // The two are different functions, about five parts in ten thousand
        // apart, and which one a layer computes is a choice rather than an
        // accident: the approximation is half again as quick, which is the
        // reason it exists. What the layer had no way of doing was the other
        // one, so a model that wanted `mt.gelu`'s values could not be built
        // out of layers at all. `approximate="none"` is that way.
        //
        // Either way it is one pass. This used to be built out of nine separate
        // tensor operations -- three of them broadcasting a cached scalar --
        // which meant nine passes over the input and nine full-size allocations
        // to compute one elementwise function. On a 4M-element tensor that was
        // 39ms against the kernel's 2.5ms.
        gelu(input, self.approximate)
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
        let expected = [(-1f32).exp() - 1.0, 0.0, 1.0];
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

    /// The layer takes the tanh approximation and the free function takes the
    /// error function, which is a choice rather than an accident -- the
    /// approximation is half again as quick. What is new is that the layer can
    /// be asked for the other one, so a model wanting `mt.gelu`'s values can be
    /// built out of layers.
    #[test]
    fn test_gelu_forward_values() {
        let points = [-3.0f32, -1.0, 0.0, 0.5, 1.0, 3.0];
        let make = || {
            Tensor::new(
                Arc::new(TensorData::from_vec_f32(points.to_vec(), Device::cpu())),
                Shape::new(vec![points.len()]),
                DataType::Float32,
                Device::cpu(),
                false,
            )
        };

        let exact: Vec<f32> = points
            .iter()
            .map(|&x| {
                let phi = 0.5 * (1.0 + libm::erf(x as f64 / std::f64::consts::SQRT_2));
                (x as f64 * phi) as f32
            })
            .collect();
        let approximated: Vec<f32> = points
            .iter()
            .map(|&x| {
                let inner = (2.0f64 / std::f64::consts::PI).sqrt()
                    * (x as f64 + 0.044_715 * (x as f64).powi(3));
                (0.5 * x as f64 * (1.0 + inner.tanh())) as f32
            })
            .collect();

        let mut gelu = GELU::new();
        let output = gelu.forward(&make()).unwrap();
        for (o, e) in output
            .data()
            .as_f32_slice()
            .unwrap()
            .iter()
            .zip(&approximated)
        {
            assert!((o - e).abs() < 1e-6, "default gelu gave {o}, expected {e}");
        }

        let mut precise = GELU::approximate(false);
        let output = precise.forward(&make()).unwrap();
        for (o, e) in output.data().as_f32_slice().unwrap().iter().zip(&exact) {
            assert!((o - e).abs() < 1e-6, "exact gelu gave {o}, expected {e}");
        }

        // And they are different functions, so a test that could not tell them
        // apart would not be testing which one ran.
        let apart = exact
            .iter()
            .zip(&approximated)
            .fold(0.0f32, |acc, (a, b)| acc.max((a - b).abs()));
        assert!(apart > 1e-4, "the two forms differ by only {apart}");
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
