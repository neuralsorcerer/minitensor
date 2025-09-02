// Copyright (c) 2025 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

use super::{
    init::{init_parameter, InitMethod},
    Layer,
};
use crate::{
    device::Device,
    error::{MinitensorError, Result},
    tensor::{DataType, Shape, Tensor},
};
use std::collections::HashMap;

/// 1D Batch normalization layer
///
/// Applies Batch Normalization over a 2D or 3D input (a mini-batch of 1D inputs
/// with optional additional channel dimension).
///
/// The mean and standard-deviation are calculated per-dimension over the mini-batches
/// and γ and β are learnable parameter vectors of size C (where C is the input size).
pub struct BatchNorm1d {
    weight: Tensor,       // γ (gamma) - learnable scale parameter
    bias: Tensor,         // β (beta) - learnable shift parameter
    running_mean: Tensor, // Running mean for inference
    running_var: Tensor,  // Running variance for inference
    num_features: usize,
    eps: f64,
    momentum: f64,
    training: bool,
}

impl BatchNorm1d {
    /// Create a new 1D batch normalization layer
    ///
    /// # Arguments
    /// * `num_features` - Number of features or channels C from an expected input of size (N, C) or (N, C, L)
    /// * `eps` - A value added to the denominator for numerical stability. Default: 1e-5
    /// * `momentum` - The value used for the running_mean and running_var computation. Default: 0.1
    /// * `device` - Device to place the layer parameters on
    /// * `dtype` - Data type for the layer parameters
    pub fn new(
        num_features: usize,
        eps: Option<f64>,
        momentum: Option<f64>,
        device: Device,
        dtype: DataType,
    ) -> Result<Self> {
        let eps = eps.unwrap_or(1e-5);
        let momentum = momentum.unwrap_or(0.1);

        let param_shape = Shape::new(vec![num_features]);

        // Initialize weight (gamma) to ones
        let weight = init_parameter(param_shape.clone(), InitMethod::Ones, dtype, device)?;

        // Initialize bias (beta) to zeros
        let bias = init_parameter(param_shape.clone(), InitMethod::Zeros, dtype, device)?;

        // Initialize running statistics to zeros and ones respectively
        let running_mean = Tensor::zeros(param_shape.clone(), dtype, device, false); // No gradients for running stats
        let running_var = Tensor::ones(param_shape, dtype, device, false); // No gradients for running stats

        Ok(Self {
            weight,
            bias,
            running_mean,
            running_var,
            num_features,
            eps,
            momentum,
            training: true,
        })
    }

    /// Get number of features
    pub fn num_features(&self) -> usize {
        self.num_features
    }

    /// Get epsilon value
    pub fn eps(&self) -> f64 {
        self.eps
    }

    /// Get momentum value
    pub fn momentum(&self) -> f64 {
        self.momentum
    }

    /// Check if in training mode
    pub fn is_training(&self) -> bool {
        self.training
    }

    /// Get the weight (gamma) tensor
    pub fn weight(&self) -> &Tensor {
        &self.weight
    }

    /// Get the bias (beta) tensor
    pub fn bias(&self) -> &Tensor {
        &self.bias
    }

    /// Get the running mean tensor
    pub fn running_mean(&self) -> &Tensor {
        &self.running_mean
    }

    /// Get the running variance tensor
    pub fn running_var(&self) -> &Tensor {
        &self.running_var
    }

    /// Set training mode
    pub fn train(&mut self) {
        self.training = true;
    }

    /// Set evaluation mode
    pub fn eval(&mut self) {
        self.training = false;
    }

    /// Get named parameters for this layer
    pub fn named_parameters(&self) -> HashMap<String, &Tensor> {
        let mut params = HashMap::new();
        params.insert("weight".to_string(), &self.weight);
        params.insert("bias".to_string(), &self.bias);
        params
    }

    /// Get named mutable parameters for this layer
    pub fn named_parameters_mut(&mut self) -> HashMap<String, &mut Tensor> {
        let mut params = HashMap::new();
        params.insert("weight".to_string(), &mut self.weight);
        params.insert("bias".to_string(), &mut self.bias);
        params
    }

    /// Get named buffers (non-trainable parameters) for this layer
    pub fn named_buffers(&self) -> HashMap<String, &Tensor> {
        let mut buffers = HashMap::new();
        buffers.insert("running_mean".to_string(), &self.running_mean);
        buffers.insert("running_var".to_string(), &self.running_var);
        buffers
    }
}

impl Layer for BatchNorm1d {
    fn forward(&mut self, input: &Tensor) -> Result<Tensor> {
        // Validate input dimensions - expect 2D [N, C] or 3D [N, C, L]
        if input.ndim() < 2 || input.ndim() > 3 {
            return Err(MinitensorError::invalid_operation(
                "BatchNorm1d expects 2D input [batch_size, features] or 3D input [batch_size, features, length]"
            ));
        }

        let _batch_size = input.size(0)?;
        let num_features = input.size(1)?;

        // Validate number of features
        if num_features != self.num_features {
            return Err(MinitensorError::shape_mismatch(
                vec![self.num_features],
                vec![num_features],
            ));
        }

        // Reshape for statistics computation
        let reshaped = if input.ndim() == 3 {
            let n = input.size(0)?;
            let l = input.size(2)?;
            input.view(Shape::new(vec![n * l, self.num_features]))?
        } else {
            input.clone()
        };

        let mean = reshaped.mean(Some(vec![0]), true)?; // [1,C]
        let centered = crate::operations::arithmetic::sub(&reshaped, &mean)?;
        let var =
            crate::operations::arithmetic::mul(&centered, &centered)?.mean(Some(vec![0]), true)?; // [1,C]

        let mean = if input.ndim() == 3 {
            mean.view(Shape::new(vec![1, self.num_features, 1]))?
        } else {
            mean.view(Shape::new(vec![1, self.num_features]))?
        };
        let var = if input.ndim() == 3 {
            var.view(Shape::new(vec![1, self.num_features, 1]))?
        } else {
            var.view(Shape::new(vec![1, self.num_features]))?
        };

        let centered = crate::operations::arithmetic::sub(input, &mean)?;
        let var = var;

        // Prepare epsilon tensor
        let eps_tensor = {
            use crate::tensor::TensorData;
            use std::sync::Arc;
            let mut data = TensorData::zeros_on_device(1, input.dtype(), input.device());
            match input.dtype() {
                DataType::Float32 => {
                    let slice = data.as_f32_slice_mut().unwrap();
                    slice[0] = self.eps as f32;
                }
                DataType::Float64 => {
                    let slice = data.as_f64_slice_mut().unwrap();
                    slice[0] = self.eps;
                }
                _ => unreachable!("BatchNorm only supports floating types"),
            }
            Tensor::new(
                Arc::new(data),
                Shape::new(vec![1]),
                input.dtype(),
                input.device(),
                false,
            )
        };

        let var = crate::operations::arithmetic::add(&var, &eps_tensor)?;
        let std = crate::operations::activation::sqrt(&var)?;
        let normalized = crate::operations::arithmetic::div(&centered, &std)?;

        // Scale and shift
        let mut weight = self.weight.clone();
        let mut bias = self.bias.clone();
        if input.ndim() == 3 {
            weight = weight.unsqueeze(0)?.unsqueeze(2)?;
            bias = bias.unsqueeze(0)?.unsqueeze(2)?;
        } else {
            weight = weight.unsqueeze(0)?;
            bias = bias.unsqueeze(0)?;
        }
        let output = crate::operations::arithmetic::add(
            &crate::operations::arithmetic::mul(&normalized, &weight)?,
            &bias,
        )?;

        if self.training {
            // Update running statistics: running = momentum*current + (1-momentum)*running
            let mean_flat = mean.view(Shape::new(vec![self.num_features]))?.detach();
            let var_flat = var.view(Shape::new(vec![self.num_features]))?.detach();

            let momentum = self.momentum;
            let one_minus = 1.0 - momentum;

            let m_tensor = {
                use crate::tensor::TensorData;
                use std::sync::Arc;
                let mut data = TensorData::zeros_on_device(1, input.dtype(), input.device());
                match input.dtype() {
                    DataType::Float32 => {
                        let s = data.as_f32_slice_mut().unwrap();
                        s[0] = momentum as f32;
                    }
                    DataType::Float64 => {
                        let s = data.as_f64_slice_mut().unwrap();
                        s[0] = momentum;
                    }
                    _ => unreachable!(),
                }
                Tensor::new(
                    Arc::new(data),
                    Shape::new(vec![1]),
                    input.dtype(),
                    input.device(),
                    false,
                )
            };

            let one_minus_tensor = {
                use crate::tensor::TensorData;
                use std::sync::Arc;
                let mut data = TensorData::zeros_on_device(1, input.dtype(), input.device());
                match input.dtype() {
                    DataType::Float32 => {
                        let s = data.as_f32_slice_mut().unwrap();
                        s[0] = one_minus as f32;
                    }
                    DataType::Float64 => {
                        let s = data.as_f64_slice_mut().unwrap();
                        s[0] = one_minus;
                    }
                    _ => unreachable!(),
                }
                Tensor::new(
                    Arc::new(data),
                    Shape::new(vec![1]),
                    input.dtype(),
                    input.device(),
                    false,
                )
            };

            self.running_mean = crate::operations::arithmetic::add(
                &crate::operations::arithmetic::mul(&self.running_mean, &one_minus_tensor)?,
                &crate::operations::arithmetic::mul(&mean_flat, &m_tensor)?,
            )?;
            self.running_var = crate::operations::arithmetic::add(
                &crate::operations::arithmetic::mul(&self.running_var, &one_minus_tensor)?,
                &crate::operations::arithmetic::mul(&var_flat, &m_tensor)?,
            )?;
        }

        Ok(output)
    }

    fn parameters(&self) -> Vec<&Tensor> {
        vec![&self.weight, &self.bias]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        vec![&mut self.weight, &mut self.bias]
    }

    fn train(&mut self) {
        self.training = true;
    }

    fn eval(&mut self) {
        self.training = false;
    }
}

/// 2D Batch normalization layer for convolutional layers
///
/// Applies Batch Normalization over a 4D input (a mini-batch of 2D inputs
/// with additional channel dimension).
pub struct BatchNorm2d {
    weight: Tensor,
    bias: Tensor,
    _running_mean: Tensor,
    _running_var: Tensor,
    num_features: usize,
    _eps: f64,
    _momentum: f64,
    training: bool,
}

impl BatchNorm2d {
    /// Create a new 2D batch normalization layer
    ///
    /// # Arguments
    /// * `num_features` - Number of features or channels C from an expected input of size (N, C, H, W)
    /// * `eps` - A value added to the denominator for numerical stability. Default: 1e-5
    /// * `momentum` - The value used for the running_mean and running_var computation. Default: 0.1
    /// * `device` - Device to place the layer parameters on
    /// * `dtype` - Data type for the layer parameters
    pub fn new(
        num_features: usize,
        eps: Option<f64>,
        momentum: Option<f64>,
        device: Device,
        dtype: DataType,
    ) -> Result<Self> {
        let eps = eps.unwrap_or(1e-5);
        let momentum = momentum.unwrap_or(0.1);

        let param_shape = Shape::new(vec![num_features]);

        // Initialize weight (gamma) to ones
        let weight = init_parameter(param_shape.clone(), InitMethod::Ones, dtype, device)?;

        // Initialize bias (beta) to zeros
        let bias = init_parameter(param_shape.clone(), InitMethod::Zeros, dtype, device)?;

        // Initialize running statistics
        let running_mean = Tensor::zeros(param_shape.clone(), dtype, device, false);
        let running_var = Tensor::ones(param_shape, dtype, device, false);

        Ok(Self {
            weight,
            bias,
            _running_mean: running_mean,
            _running_var: running_var,
            num_features,
            _eps: eps,
            _momentum: momentum,
            training: true,
        })
    }

    /// Get number of features
    pub fn num_features(&self) -> usize {
        self.num_features
    }

    /// Set training mode
    pub fn train(&mut self) {
        self.training = true;
    }

    /// Set evaluation mode
    pub fn eval(&mut self) {
        self.training = false;
    }
}

impl Layer for BatchNorm2d {
    fn forward(&mut self, input: &Tensor) -> Result<Tensor> {
        // Validate input dimensions - expect 4D [N, C, H, W]
        if input.ndim() != 4 {
            return Err(MinitensorError::invalid_operation(
                "BatchNorm2d expects 4D input [batch_size, channels, height, width]",
            ));
        }

        let num_features = input.size(1)?;

        // Validate number of features
        if num_features != self.num_features {
            return Err(MinitensorError::shape_mismatch(
                vec![self.num_features],
                vec![num_features],
            ));
        }

        // Reshape to compute statistics over N*H*W x C
        let n = input.size(0)?;
        let h = input.size(2)?;
        let w = input.size(3)?;
        let reshaped = input.view(Shape::new(vec![n * h * w, self.num_features]))?;

        let mean = reshaped.mean(Some(vec![0]), true)?; // [1,C]
        let centered = crate::operations::arithmetic::sub(&reshaped, &mean)?;
        let var =
            crate::operations::arithmetic::mul(&centered, &centered)?.mean(Some(vec![0]), true)?; // [1,C]

        let mean = mean.view(Shape::new(vec![1, self.num_features, 1, 1]))?;
        let var = var.view(Shape::new(vec![1, self.num_features, 1, 1]))?;

        let centered = crate::operations::arithmetic::sub(input, &mean)?;

        let eps_tensor = {
            use crate::tensor::TensorData;
            use std::sync::Arc;
            let mut data = TensorData::zeros_on_device(1, input.dtype(), input.device());
            match input.dtype() {
                DataType::Float32 => data.as_f32_slice_mut().unwrap()[0] = self._eps as f32,
                DataType::Float64 => data.as_f64_slice_mut().unwrap()[0] = self._eps,
                _ => unreachable!("BatchNorm only supports floating types"),
            }
            Tensor::new(
                Arc::new(data),
                Shape::new(vec![1]),
                input.dtype(),
                input.device(),
                false,
            )
        };

        let var = crate::operations::arithmetic::add(&var, &eps_tensor)?;
        let std = crate::operations::activation::sqrt(&var)?;
        let normalized = crate::operations::arithmetic::div(&centered, &std)?;
        let weight = self
            .weight
            .clone()
            .unsqueeze(0)?
            .unsqueeze(2)?
            .unsqueeze(3)?;
        let bias = self.bias.clone().unsqueeze(0)?.unsqueeze(2)?.unsqueeze(3)?;
        let output = crate::operations::arithmetic::add(
            &crate::operations::arithmetic::mul(&normalized, &weight)?,
            &bias,
        )?;

        Ok(output)
    }

    fn parameters(&self) -> Vec<&Tensor> {
        vec![&self.weight, &self.bias]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        vec![&mut self.weight, &mut self.bias]
    }

    fn train(&mut self) {
        self.training = true;
    }

    fn eval(&mut self) {
        self.training = false;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::Device;
    use crate::tensor::{DataType, Shape};

    #[test]
    fn test_batchnorm1d_creation() {
        let layer =
            BatchNorm1d::new(128, Some(1e-5), Some(0.1), Device::cpu(), DataType::Float32).unwrap();

        assert_eq!(layer.num_features(), 128);
        assert_eq!(layer.eps(), 1e-5);
        assert_eq!(layer.momentum(), 0.1);
        assert!(layer.is_training());
        assert_eq!(layer.weight().shape(), &Shape::new(vec![128]));
        assert_eq!(layer.bias().shape(), &Shape::new(vec![128]));
        assert_eq!(layer.running_mean().shape(), &Shape::new(vec![128]));
        assert_eq!(layer.running_var().shape(), &Shape::new(vec![128]));
    }

    #[test]
    fn test_batchnorm1d_training_mode() {
        let mut layer =
            BatchNorm1d::new(128, None, None, Device::cpu(), DataType::Float32).unwrap();

        assert!(layer.is_training());

        layer.eval();
        assert!(!layer.is_training());

        layer.train();
        assert!(layer.is_training());
    }

    #[test]
    fn test_batchnorm1d_parameters() {
        let mut layer =
            BatchNorm1d::new(128, None, None, Device::cpu(), DataType::Float32).unwrap();

        let params = layer.parameters();
        assert_eq!(params.len(), 2); // weight + bias

        let mut_params = layer.parameters_mut();
        assert_eq!(mut_params.len(), 2);

        let named_params = layer.named_parameters();
        assert_eq!(named_params.len(), 2);
        assert!(named_params.contains_key("weight"));
        assert!(named_params.contains_key("bias"));

        let buffers = layer.named_buffers();
        assert_eq!(buffers.len(), 2);
        assert!(buffers.contains_key("running_mean"));
        assert!(buffers.contains_key("running_var"));
    }

    #[test]
    fn test_batchnorm1d_forward_shape_validation() {
        let mut layer =
            BatchNorm1d::new(128, None, None, Device::cpu(), DataType::Float32).unwrap();

        // Test with correct 2D input [batch=32, features=128]
        let input_2d = Tensor::zeros(
            Shape::new(vec![32, 128]),
            DataType::Float32,
            Device::cpu(),
            false,
        );
        let output = layer.forward(&input_2d).unwrap();
        assert_eq!(output.shape(), input_2d.shape());

        // Test with correct 3D input [batch=32, features=128, length=10]
        let input_3d = Tensor::zeros(
            Shape::new(vec![32, 128, 10]),
            DataType::Float32,
            Device::cpu(),
            false,
        );
        let output = layer.forward(&input_3d).unwrap();
        assert_eq!(output.shape(), input_3d.shape());

        // Test with incorrect number of features
        let wrong_input = Tensor::zeros(
            Shape::new(vec![32, 64]),
            DataType::Float32,
            Device::cpu(),
            false,
        );
        let result = layer.forward(&wrong_input);
        assert!(result.is_err());

        // Test with wrong number of dimensions
        let wrong_dim_input = Tensor::zeros(
            Shape::new(vec![128]),
            DataType::Float32,
            Device::cpu(),
            false,
        );
        let result = layer.forward(&wrong_dim_input);
        assert!(result.is_err());
    }

    #[test]
    fn test_batchnorm2d_creation() {
        let layer =
            BatchNorm2d::new(64, Some(1e-5), Some(0.1), Device::cpu(), DataType::Float32).unwrap();

        assert_eq!(layer.num_features(), 64);
    }

    #[test]
    fn test_batchnorm2d_forward_shape_validation() {
        let mut layer = BatchNorm2d::new(64, None, None, Device::cpu(), DataType::Float32).unwrap();

        // Test with correct 4D input [batch=16, channels=64, height=32, width=32]
        let input = Tensor::zeros(
            Shape::new(vec![16, 64, 32, 32]),
            DataType::Float32,
            Device::cpu(),
            false,
        );
        let output = layer.forward(&input).unwrap();
        assert_eq!(output.shape(), input.shape());

        // Test with incorrect number of channels
        let wrong_input = Tensor::zeros(
            Shape::new(vec![16, 32, 32, 32]),
            DataType::Float32,
            Device::cpu(),
            false,
        );
        let result = layer.forward(&wrong_input);
        assert!(result.is_err());

        // Test with wrong number of dimensions
        let wrong_dim_input = Tensor::zeros(
            Shape::new(vec![16, 64, 32]),
            DataType::Float32,
            Device::cpu(),
            false,
        );
        let result = layer.forward(&wrong_dim_input);
        assert!(result.is_err());
    }
}
