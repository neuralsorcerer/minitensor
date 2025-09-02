// Copyright (c) 2025 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

use super::Layer;
use crate::{
    error::{MinitensorError, Result},
    operations::arithmetic,
    tensor::{DataType, Shape, Tensor, TensorData},
};
use rand::Rng;
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
    let scale = if keep_prob == 0.0 {
        0.0
    } else {
        1.0 / keep_prob
    };
    let numel = shape.numel();
    let mut rng = rand::thread_rng();

    match dtype {
        DataType::Float32 => {
            let mut data = Vec::with_capacity(numel);
            for _ in 0..numel {
                let r: f64 = rng.gen();
                if r < keep_prob {
                    data.push(scale as f32);
                } else {
                    data.push(0.0);
                }
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
            for _ in 0..numel {
                let r: f64 = rng.gen();
                if r < keep_prob {
                    data.push(scale);
                } else {
                    data.push(0.0);
                }
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
                "Dropout2d expects 4D input tensor with shape [batch_size, channels, height, width]"
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

        // Test p=1 (complete dropout)
        let mut dropout_one = Dropout::new(Some(1.0)).unwrap();
        dropout_one.train();
        let output = dropout_one.forward(&input).unwrap();
        assert_eq!(output.shape(), input.shape());
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
}
