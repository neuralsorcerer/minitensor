// Copyright (c) 2025 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

use super::Layer;
use crate::{
    error::{MinitensorError, Result},
    operations::arithmetic,
    tensor::{DataType, Shape, Tensor, TensorData},
};
use rand_distr::{Bernoulli, Distribution};
use std::sync::Arc;

/// Generate a dropout mask with values either `0` or `1/(1-p)`.
///
/// This utility is used by both `Dropout` and `Dropout2d` to create
/// Bernoulli masks that keep activations with probability `1-p`.
fn generate_dropout_mask(
    p: f64,
    shape: &Shape,
    dtype: DataType,
    device: crate::device::Device,
) -> Result<Tensor> {
    let keep_prob = 1.0 - p;
    let numel = shape.numel();

    match dtype {
        DataType::Float32 => {
            let mut data = Vec::with_capacity(numel);
            unsafe {
                data.set_len(numel);
            }
            if keep_prob >= 1.0 {
                data.fill(1.0);
            } else if keep_prob > 0.0 {
                let scale = 1.0f32 / keep_prob as f32;
                let mut rng = rand::rng();
                let bernoulli = Bernoulli::new(keep_prob)
                    .map_err(|e| MinitensorError::invalid_argument(e.to_string()))?;
                for (v, b) in data.iter_mut().zip(bernoulli.sample_iter(&mut rng)) {
                    *v = if b { scale } else { 0.0 };
                }
            } else {
                data.fill(0.0);
            }
            let td = TensorData::from_vec_f32(data, device);
            Ok(Tensor::new(
                Arc::new(td),
                shape.clone(),
                dtype,
                device,
                false,
            ))
        }
        DataType::Float64 => {
            let mut data = Vec::with_capacity(numel);
            unsafe {
                data.set_len(numel);
            }
            if keep_prob >= 1.0 {
                data.fill(1.0);
            } else if keep_prob > 0.0 {
                let scale = 1.0 / keep_prob;
                let mut rng = rand::rng();
                let bernoulli = Bernoulli::new(keep_prob)
                    .map_err(|e| MinitensorError::invalid_argument(e.to_string()))?;
                for (v, b) in data.iter_mut().zip(bernoulli.sample_iter(&mut rng)) {
                    *v = if b { scale } else { 0.0 };
                }
            } else {
                data.fill(0.0);
            }
            let td = TensorData::from_vec_f64(data, device);
            Ok(Tensor::new(
                Arc::new(td),
                shape.clone(),
                dtype,
                device,
                false,
            ))
        }
        _ => Err(MinitensorError::invalid_argument(
            "Dropout mask generation only supports floating point tensors".to_string(),
        )),
    }
}

/// Dropout layer for regularization
///
/// During training, randomly zeroes some of the elements of the input tensor
/// with probability p using samples from a Bernoulli distribution. Each channel
/// will be zeroed out independently on every forward call.
///
/// This has proven to be an effective technique for regularization and preventing
/// the co-adaptation of neurons as described in the paper "Improving neural networks
/// by preventing co-adaptation of feature detectors".
#[derive(Clone)]
pub struct Dropout {
    p: f64,
    training: bool,
}

impl Dropout {
    /// Create a new Dropout layer
    ///
    /// # Arguments
    /// * `p` - Probability of an element to be zeroed. Default: 0.5
    pub fn new(p: Option<f64>) -> Result<Self> {
        let p = p.unwrap_or(0.5);

        if !(0.0..=1.0).contains(&p) {
            return Err(MinitensorError::invalid_argument(format!(
                "Dropout probability must be between 0 and 1, got {}",
                p
            )));
        }

        Ok(Self { p, training: true })
    }

    /// Get the dropout probability
    pub fn p(&self) -> f64 {
        self.p
    }

    /// Check if in training mode
    pub fn is_training(&self) -> bool {
        self.training
    }

    /// Set training mode
    pub fn train(&mut self) {
        self.training = true;
    }

    /// Set evaluation mode
    pub fn eval(&mut self) {
        self.training = false;
    }

    /// Generate a dropout mask for the given shape
    ///
    /// The mask contains either `0` or `1/(1-p)` for each element, where `p` is
    /// the dropout probability. Elements with value `0` correspond to dropped
    /// activations. The scaling factor `1/(1-p)` ensures that the expected value
    /// of the activations remains the same during training.
    fn generate_mask(
        &self,
        shape: &Shape,
        dtype: DataType,
        device: crate::device::Device,
    ) -> Result<Tensor> {
        generate_dropout_mask(self.p, shape, dtype, device)
    }
}

impl Layer for Dropout {
    fn forward(&mut self, input: &Tensor) -> Result<Tensor> {
        if !self.training || self.p == 0.0 {
            // During evaluation or when p=0, return input unchanged
            return Ok(input.clone());
        }

        if self.p == 1.0 {
            // When p=1, return zeros
            return Ok(Tensor::zeros(
                input.shape().clone(),
                input.dtype(),
                input.device(),
                input.requires_grad(),
            ));
        }

        // Generate dropout mask
        let mask = self.generate_mask(input.shape(), input.dtype(), input.device())?;

        // Apply mask element-wise. The mask is scaled by `1/(1-p)` so no
        // additional scaling is required here.
        let output = arithmetic::mul(input, &mask)?;
        Ok(output)
    }

    fn parameters(&self) -> Vec<&Tensor> {
        vec![] // No learnable parameters
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        vec![] // No learnable parameters
    }

    fn train(&mut self) {
        self.training = true;
    }

    fn eval(&mut self) {
        self.training = false;
    }
}

/// 2D Dropout layer
///
/// Randomly zero out entire channels (a channel is a 2D feature map,
/// e.g., the j-th channel of the i-th sample in the batched input is a 2D tensor).
/// Each channel will be zeroed out independently on every forward call with
/// probability p using samples from a Bernoulli distribution.
#[derive(Clone)]
pub struct Dropout2d {
    p: f64,
    training: bool,
}

impl Dropout2d {
    /// Create a new Dropout2d layer
    ///
    /// # Arguments
    /// * `p` - Probability of a channel to be zeroed. Default: 0.5
    pub fn new(p: Option<f64>) -> Result<Self> {
        let p = p.unwrap_or(0.5);

        if !(0.0..=1.0).contains(&p) {
            return Err(MinitensorError::invalid_argument(format!(
                "Dropout probability must be between 0 and 1, got {}",
                p
            )));
        }

        Ok(Self { p, training: true })
    }

    /// Get the dropout probability
    pub fn p(&self) -> f64 {
        self.p
    }

    /// Check if in training mode
    pub fn is_training(&self) -> bool {
        self.training
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

impl Layer for Dropout2d {
    fn forward(&mut self, input: &Tensor) -> Result<Tensor> {
        // Validate input dimensions - expect 4D tensor [N, C, H, W]
        if input.ndim() != 4 {
            return Err(MinitensorError::invalid_operation(
                "Dropout2d expects 4D input tensor with shape [batch_size, channels, height, width]",
            ));
        }

        if !self.training || self.p == 0.0 {
            // During evaluation or when p=0, return input unchanged
            return Ok(input.clone());
        }

        if self.p == 1.0 {
            // When p=1, return zeros
            return Ok(Tensor::zeros(
                input.shape().clone(),
                input.dtype(),
                input.device(),
                input.requires_grad(),
            ));
        }

        // Generate a mask that zeroes out entire channels. The mask has
        // shape `[N, C, 1, 1]` and is broadcast across the spatial
        // dimensions when multiplied with the input.
        let mask_shape = Shape::new(vec![input.size(0)?, input.size(1)?, 1, 1]);
        let mask = generate_dropout_mask(self.p, &mask_shape, input.dtype(), input.device())?;

        let output = arithmetic::mul(input, &mask)?;
        Ok(output)
    }

    fn parameters(&self) -> Vec<&Tensor> {
        vec![] // No learnable parameters
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        vec![] // No learnable parameters
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
    fn test_dropout_creation() {
        let dropout = Dropout::new(Some(0.3)).unwrap();
        assert_eq!(dropout.p(), 0.3);
        assert!(dropout.is_training());
        assert_eq!(dropout.parameters().len(), 0);

        let dropout_default = Dropout::new(None).unwrap();
        assert_eq!(dropout_default.p(), 0.5);
    }

    #[test]
    fn test_dropout_invalid_probability() {
        let result = Dropout::new(Some(-0.1));
        assert!(result.is_err());

        let result = Dropout::new(Some(1.5));
        assert!(result.is_err());
    }

    #[test]
    fn test_dropout_training_mode() {
        let mut dropout = Dropout::new(Some(0.5)).unwrap();

        assert!(dropout.is_training());

        dropout.eval();
        assert!(!dropout.is_training());

        dropout.train();
        assert!(dropout.is_training());
    }

    #[test]
    fn test_dropout_forward() {
        let mut dropout = Dropout::new(Some(0.5)).unwrap();
        let input = Tensor::ones(
            Shape::new(vec![2, 3, 4]),
            DataType::Float32,
            Device::cpu(),
            false,
        );

        // Test training mode
        dropout.train();
        let output_train = dropout.forward(&input).unwrap();
        assert_eq!(output_train.shape(), input.shape());

        // Test evaluation mode
        dropout.eval();
        let output_eval = dropout.forward(&input).unwrap();
        assert_eq!(output_eval.shape(), input.shape());
    }

    #[test]
    fn test_dropout_edge_cases() {
        let input = Tensor::ones(
            Shape::new(vec![2, 3]),
            DataType::Float32,
            Device::cpu(),
            false,
        );

        // Test p=0 (no dropout)
        let mut dropout_zero = Dropout::new(Some(0.0)).unwrap();
        dropout_zero.train();
        let output = dropout_zero.forward(&input).unwrap();
        assert_eq!(output.shape(), input.shape());
        let in_data = input.data().as_f32_slice().unwrap();
        let out_data = output.data().as_f32_slice().unwrap();
        assert_eq!(in_data, out_data);

        // Test p=1 (complete dropout)
        let mut dropout_one = Dropout::new(Some(1.0)).unwrap();
        dropout_one.train();
        let output = dropout_one.forward(&input).unwrap();
        assert_eq!(output.shape(), input.shape());
        let out_data = output.data().as_f32_slice().unwrap();
        assert!(out_data.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_dropout2d_creation() {
        let dropout2d = Dropout2d::new(Some(0.3)).unwrap();
        assert_eq!(dropout2d.p(), 0.3);
        assert!(dropout2d.is_training());
        assert_eq!(dropout2d.parameters().len(), 0);

        let dropout2d_default = Dropout2d::new(None).unwrap();
        assert_eq!(dropout2d_default.p(), 0.5);
    }

    #[test]
    fn test_dropout2d_forward_shape_validation() {
        let mut dropout2d = Dropout2d::new(Some(0.5)).unwrap();

        // Test with correct 4D input
        let input_4d = Tensor::ones(
            Shape::new(vec![2, 3, 8, 8]),
            DataType::Float32,
            Device::cpu(),
            false,
        );
        let output = dropout2d.forward(&input_4d).unwrap();
        assert_eq!(output.shape(), input_4d.shape());

        // Test with incorrect dimensions
        let input_3d = Tensor::ones(
            Shape::new(vec![2, 3, 8]),
            DataType::Float32,
            Device::cpu(),
            false,
        );
        let result = dropout2d.forward(&input_3d);
        assert!(result.is_err());
    }

    #[test]
    fn test_dropout2d_training_mode() {
        let mut dropout2d = Dropout2d::new(Some(0.5)).unwrap();
        let input = Tensor::ones(
            Shape::new(vec![2, 3, 8, 8]),
            DataType::Float32,
            Device::cpu(),
            false,
        );

        // Test training mode
        dropout2d.train();
        let output_train = dropout2d.forward(&input).unwrap();
        assert_eq!(output_train.shape(), input.shape());

        // Test evaluation mode
        dropout2d.eval();
        let output_eval = dropout2d.forward(&input).unwrap();
        assert_eq!(output_eval.shape(), input.shape());
    }

    #[test]
    fn test_dropout2d_edge_cases() {
        let input = Tensor::ones(
            Shape::new(vec![1, 2, 4, 4]),
            DataType::Float32,
            Device::cpu(),
            false,
        );

        // p = 0 should return identical input
        let mut no_dropout = Dropout2d::new(Some(0.0)).unwrap();
        let output = no_dropout.forward(&input).unwrap();
        assert_eq!(output.shape(), input.shape());
        let out_data = output.data().as_f32_slice().unwrap();
        let in_data = input.data().as_f32_slice().unwrap();
        assert_eq!(out_data, in_data);

        // p = 1 should return all zeros
        let mut full_dropout = Dropout2d::new(Some(1.0)).unwrap();
        let output = full_dropout.forward(&input).unwrap();
        let data = output.data().as_f32_slice().unwrap();
        assert!(data.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_dropout_scaling_expectation() {
        let mut dropout = Dropout::new(Some(0.25)).unwrap();
        dropout.train();
        let input = Tensor::ones(
            Shape::new(vec![10_000]),
            DataType::Float32,
            Device::cpu(),
            false,
        );
        let output = dropout.forward(&input).unwrap();
        let data = output.data().as_f32_slice().unwrap();
        let mean: f32 = data.iter().sum::<f32>() / data.len() as f32;
        // Expected mean should remain close to 1.0 after scaling
        assert!((mean - 1.0).abs() < 0.1);
    }
}
