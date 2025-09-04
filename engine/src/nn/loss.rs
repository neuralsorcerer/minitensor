// Copyright (c) 2025 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

use crate::{
    error::Result,
    operations::loss::{
        binary_cross_entropy_loss, cross_entropy_loss, focal_loss, huber_loss, mae_loss, mse_loss,
    },
    tensor::Tensor,
};

/// Mean Squared Error loss layer
#[derive(Debug, Clone)]
pub struct MSELoss {
    reduction: String,
}

impl MSELoss {
    /// Create a new MSE loss with the specified reduction
    pub fn new(reduction: impl Into<String>) -> Self {
        Self {
            reduction: reduction.into(),
        }
    }

    /// Create MSE loss with mean reduction (default)
    pub fn mean() -> Self {
        Self::new("mean")
    }

    /// Create MSE loss with sum reduction
    pub fn sum() -> Self {
        Self::new("sum")
    }

    /// Create MSE loss with no reduction (element-wise)
    pub fn none() -> Self {
        Self::new("none")
    }

    /// Compute the MSE loss between predictions and targets
    pub fn forward(&self, predictions: &Tensor, targets: &Tensor) -> Result<Tensor> {
        mse_loss(predictions, targets, &self.reduction)
    }

    /// Get the reduction mode
    pub fn reduction(&self) -> &str {
        &self.reduction
    }

    /// Set the reduction mode
    pub fn set_reduction(&mut self, reduction: impl Into<String>) {
        self.reduction = reduction.into();
    }
}

/// Mean Absolute Error loss layer
#[derive(Debug, Clone)]
pub struct MAELoss {
    reduction: String,
}

impl MAELoss {
    /// Create a new MAE loss with the specified reduction
    pub fn new(reduction: impl Into<String>) -> Self {
        Self {
            reduction: reduction.into(),
        }
    }

    /// Create MAE loss with mean reduction (default)
    pub fn mean() -> Self {
        Self::new("mean")
    }

    /// Create MAE loss with sum reduction
    pub fn sum() -> Self {
        Self::new("sum")
    }

    /// Create MAE loss with no reduction (element-wise)
    pub fn none() -> Self {
        Self::new("none")
    }

    /// Compute the MAE loss between predictions and targets
    pub fn forward(&self, predictions: &Tensor, targets: &Tensor) -> Result<Tensor> {
        mae_loss(predictions, targets, &self.reduction)
    }

    /// Get the reduction mode
    pub fn reduction(&self) -> &str {
        &self.reduction
    }

    /// Set the reduction mode
    pub fn set_reduction(&mut self, reduction: impl Into<String>) {
        self.reduction = reduction.into();
    }
}

/// Huber loss layer for robust regression
#[derive(Debug, Clone)]
pub struct HuberLoss {
    delta: f64,
    reduction: String,
}

impl HuberLoss {
    /// Create a new Huber loss with the specified delta and reduction
    pub fn new(delta: f64, reduction: impl Into<String>) -> Self {
        Self {
            delta,
            reduction: reduction.into(),
        }
    }

    /// Create Huber loss with mean reduction (default)
    pub fn mean(delta: f64) -> Self {
        Self::new(delta, "mean")
    }

    /// Create Huber loss with sum reduction
    pub fn sum(delta: f64) -> Self {
        Self::new(delta, "sum")
    }

    /// Create Huber loss with no reduction (element-wise)
    pub fn none(delta: f64) -> Self {
        Self::new(delta, "none")
    }

    /// Compute the Huber loss between predictions and targets
    pub fn forward(&self, predictions: &Tensor, targets: &Tensor) -> Result<Tensor> {
        huber_loss(predictions, targets, self.delta, &self.reduction)
    }

    /// Get the delta parameter
    pub fn delta(&self) -> f64 {
        self.delta
    }

    /// Set the delta parameter
    pub fn set_delta(&mut self, delta: f64) {
        self.delta = delta;
    }

    /// Get the reduction mode
    pub fn reduction(&self) -> &str {
        &self.reduction
    }

    /// Set the reduction mode
    pub fn set_reduction(&mut self, reduction: impl Into<String>) {
        self.reduction = reduction.into();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        device::Device,
        tensor::{DataType, Shape, Tensor, TensorData},
    };
    use std::sync::Arc;

    fn create_test_tensor_f32(data: Vec<f32>, shape: Vec<usize>, requires_grad: bool) -> Tensor {
        let shape_obj = Shape::new(shape);
        let mut tensor_data = TensorData::zeros(shape_obj.numel(), DataType::Float32);

        if let Some(slice) = tensor_data.as_f32_slice_mut() {
            slice.copy_from_slice(&data);
        }

        Tensor::new(
            Arc::new(tensor_data),
            shape_obj,
            DataType::Float32,
            Device::cpu(),
            requires_grad,
        )
    }

    #[test]
    fn test_mse_loss_layer() {
        let mse = MSELoss::mean();
        assert_eq!(mse.reduction(), "mean");

        let predictions = create_test_tensor_f32(vec![1.0, 2.0, 3.0], vec![3], false);
        let targets = create_test_tensor_f32(vec![1.5, 2.5, 2.5], vec![3], false);

        let loss = mse.forward(&predictions, &targets).unwrap();
        let loss_data = loss.data().as_f32_slice().unwrap();

        // Expected: ((1.0-1.5)² + (2.0-2.5)² + (3.0-2.5)²) / 3 = 0.25
        assert!((loss_data[0] - 0.25).abs() < 1e-6);
    }

    #[test]
    fn test_mae_loss_layer() {
        let mae = MAELoss::mean();
        assert_eq!(mae.reduction(), "mean");

        let predictions = create_test_tensor_f32(vec![1.0, 2.0, 3.0], vec![3], false);
        let targets = create_test_tensor_f32(vec![1.5, 2.5, 2.0], vec![3], false);

        let loss = mae.forward(&predictions, &targets).unwrap();
        let loss_data = loss.data().as_f32_slice().unwrap();

        // Expected: (0.5 + 0.5 + 1.0) / 3 = 2.0/3
        assert!((loss_data[0] - (2.0 / 3.0)).abs() < 1e-6);
    }

    #[test]
    fn test_huber_loss_layer() {
        let huber = HuberLoss::mean(1.0);
        assert_eq!(huber.delta(), 1.0);
        assert_eq!(huber.reduction(), "mean");

        let predictions = create_test_tensor_f32(vec![1.0, 2.0], vec![2], false);
        let targets = create_test_tensor_f32(vec![1.2, 2.3], vec![2], false);

        let loss = huber.forward(&predictions, &targets).unwrap();
        // Just check that the loss was computed successfully
        assert_eq!(loss.shape().dims(), &[1]);
    }

    #[test]
    fn test_loss_layer_builders() {
        let mse_mean = MSELoss::mean();
        assert_eq!(mse_mean.reduction(), "mean");

        let mse_sum = MSELoss::sum();
        assert_eq!(mse_sum.reduction(), "sum");

        let mse_none = MSELoss::none();
        assert_eq!(mse_none.reduction(), "none");

        let mae_mean = MAELoss::mean();
        assert_eq!(mae_mean.reduction(), "mean");

        let huber_mean = HuberLoss::mean(0.5);
        assert_eq!(huber_mean.delta(), 0.5);
        assert_eq!(huber_mean.reduction(), "mean");
    }

    #[test]
    fn test_loss_layer_setters() {
        let mut mse = MSELoss::mean();
        mse.set_reduction("sum");
        assert_eq!(mse.reduction(), "sum");

        let mut mae = MAELoss::mean();
        mae.set_reduction("none");
        assert_eq!(mae.reduction(), "none");

        let mut huber = HuberLoss::mean(1.0);
        huber.set_delta(2.0);
        huber.set_reduction("sum");
        assert_eq!(huber.delta(), 2.0);
        assert_eq!(huber.reduction(), "sum");
    }
}

/// Cross Entropy loss layer for classification
#[derive(Debug, Clone)]
pub struct CrossEntropyLoss {
    reduction: String,
}

impl CrossEntropyLoss {
    /// Create a new Cross Entropy loss with the specified reduction
    pub fn new(reduction: impl Into<String>) -> Self {
        Self {
            reduction: reduction.into(),
        }
    }

    /// Create Cross Entropy loss with mean reduction (default)
    pub fn mean() -> Self {
        Self::new("mean")
    }

    /// Create Cross Entropy loss with sum reduction
    pub fn sum() -> Self {
        Self::new("sum")
    }

    /// Create Cross Entropy loss with no reduction (element-wise)
    pub fn none() -> Self {
        Self::new("none")
    }

    /// Compute the Cross Entropy loss between predictions (logits) and targets
    pub fn forward(&self, predictions: &Tensor, targets: &Tensor) -> Result<Tensor> {
        cross_entropy_loss(predictions, targets, &self.reduction)
    }

    /// Get the reduction mode
    pub fn reduction(&self) -> &str {
        &self.reduction
    }

    /// Set the reduction mode
    pub fn set_reduction(&mut self, reduction: impl Into<String>) {
        self.reduction = reduction.into();
    }
}

/// Binary Cross Entropy loss layer
#[derive(Debug, Clone)]
pub struct BCELoss {
    reduction: String,
}

impl BCELoss {
    /// Create a new BCE loss with the specified reduction
    pub fn new(reduction: impl Into<String>) -> Self {
        Self {
            reduction: reduction.into(),
        }
    }

    /// Create BCE loss with mean reduction (default)
    pub fn mean() -> Self {
        Self::new("mean")
    }

    /// Create BCE loss with sum reduction
    pub fn sum() -> Self {
        Self::new("sum")
    }

    /// Create BCE loss with no reduction (element-wise)
    pub fn none() -> Self {
        Self::new("none")
    }

    /// Compute the BCE loss between predictions (probabilities) and targets
    pub fn forward(&self, predictions: &Tensor, targets: &Tensor) -> Result<Tensor> {
        binary_cross_entropy_loss(predictions, targets, &self.reduction)
    }

    /// Get the reduction mode
    pub fn reduction(&self) -> &str {
        &self.reduction
    }

    /// Set the reduction mode
    pub fn set_reduction(&mut self, reduction: impl Into<String>) {
        self.reduction = reduction.into();
    }
}

/// Focal loss layer for handling class imbalance
#[derive(Debug, Clone)]
pub struct FocalLoss {
    alpha: f64,
    gamma: f64,
    reduction: String,
}

impl FocalLoss {
    /// Create a new Focal loss with the specified parameters
    pub fn new(alpha: f64, gamma: f64, reduction: impl Into<String>) -> Self {
        Self {
            alpha,
            gamma,
            reduction: reduction.into(),
        }
    }

    /// Create Focal loss with mean reduction (default)
    pub fn mean(alpha: f64, gamma: f64) -> Self {
        Self::new(alpha, gamma, "mean")
    }

    /// Create Focal loss with sum reduction
    pub fn sum(alpha: f64, gamma: f64) -> Self {
        Self::new(alpha, gamma, "sum")
    }

    /// Create Focal loss with no reduction (element-wise)
    pub fn none(alpha: f64, gamma: f64) -> Self {
        Self::new(alpha, gamma, "none")
    }

    /// Compute the Focal loss between predictions (logits) and targets
    pub fn forward(&self, predictions: &Tensor, targets: &Tensor) -> Result<Tensor> {
        focal_loss(
            predictions,
            targets,
            self.alpha,
            self.gamma,
            &self.reduction,
        )
    }

    /// Get the alpha parameter
    pub fn alpha(&self) -> f64 {
        self.alpha
    }

    /// Set the alpha parameter
    pub fn set_alpha(&mut self, alpha: f64) {
        self.alpha = alpha;
    }

    /// Get the gamma parameter
    pub fn gamma(&self) -> f64 {
        self.gamma
    }

    /// Set the gamma parameter
    pub fn set_gamma(&mut self, gamma: f64) {
        self.gamma = gamma;
    }

    /// Get the reduction mode
    pub fn reduction(&self) -> &str {
        &self.reduction
    }

    /// Set the reduction mode
    pub fn set_reduction(&mut self, reduction: impl Into<String>) {
        self.reduction = reduction.into();
    }
}

#[cfg(test)]
mod classification_tests {
    use super::*;
    use crate::{
        device::Device,
        tensor::{DataType, Shape, Tensor, TensorData},
    };
    use std::sync::Arc;

    fn create_test_tensor_f32(data: Vec<f32>, shape: Vec<usize>, requires_grad: bool) -> Tensor {
        let shape_obj = Shape::new(shape);
        let mut tensor_data = TensorData::zeros(shape_obj.numel(), DataType::Float32);

        if let Some(slice) = tensor_data.as_f32_slice_mut() {
            slice.copy_from_slice(&data);
        }

        Tensor::new(
            Arc::new(tensor_data),
            shape_obj,
            DataType::Float32,
            Device::cpu(),
            requires_grad,
        )
    }

    #[test]
    fn test_cross_entropy_loss_layer() {
        let ce_loss = CrossEntropyLoss::mean();
        assert_eq!(ce_loss.reduction(), "mean");

        // Create simple 2-class classification example
        let predictions = create_test_tensor_f32(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2], false);
        let targets = create_test_tensor_f32(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2], false);

        let loss = ce_loss.forward(&predictions, &targets);
        // Just check that the loss was computed successfully
        assert!(loss.is_ok());
    }

    #[test]
    fn test_bce_loss_layer() {
        let bce_loss = BCELoss::mean();
        assert_eq!(bce_loss.reduction(), "mean");

        // Create binary classification example with probabilities
        let predictions = create_test_tensor_f32(vec![0.8, 0.2, 0.3, 0.9], vec![4], false);
        let targets = create_test_tensor_f32(vec![1.0, 0.0, 0.0, 1.0], vec![4], false);

        let loss = bce_loss.forward(&predictions, &targets);
        // Just check that the loss was computed successfully
        assert!(loss.is_ok());
    }

    #[test]
    fn test_focal_loss_layer() {
        let focal_loss = FocalLoss::mean(0.25, 2.0);
        assert_eq!(focal_loss.alpha(), 0.25);
        assert_eq!(focal_loss.gamma(), 2.0);
        assert_eq!(focal_loss.reduction(), "mean");

        // Create simple classification example
        let predictions = create_test_tensor_f32(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2], false);
        let targets = create_test_tensor_f32(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2], false);

        let loss = focal_loss.forward(&predictions, &targets);
        // Just check that the loss was computed successfully
        assert!(loss.is_ok());
    }

    #[test]
    fn test_classification_loss_builders() {
        let ce_mean = CrossEntropyLoss::mean();
        assert_eq!(ce_mean.reduction(), "mean");

        let ce_sum = CrossEntropyLoss::sum();
        assert_eq!(ce_sum.reduction(), "sum");

        let ce_none = CrossEntropyLoss::none();
        assert_eq!(ce_none.reduction(), "none");

        let bce_mean = BCELoss::mean();
        assert_eq!(bce_mean.reduction(), "mean");

        let focal_mean = FocalLoss::mean(0.5, 1.5);
        assert_eq!(focal_mean.alpha(), 0.5);
        assert_eq!(focal_mean.gamma(), 1.5);
        assert_eq!(focal_mean.reduction(), "mean");
    }

    #[test]
    fn test_classification_loss_setters() {
        let mut ce_loss = CrossEntropyLoss::mean();
        ce_loss.set_reduction("sum");
        assert_eq!(ce_loss.reduction(), "sum");

        let mut bce_loss = BCELoss::mean();
        bce_loss.set_reduction("none");
        assert_eq!(bce_loss.reduction(), "none");

        let mut focal_loss = FocalLoss::mean(0.25, 2.0);
        focal_loss.set_alpha(0.5);
        focal_loss.set_gamma(1.0);
        focal_loss.set_reduction("sum");
        assert_eq!(focal_loss.alpha(), 0.5);
        assert_eq!(focal_loss.gamma(), 1.0);
        assert_eq!(focal_loss.reduction(), "sum");
    }
}
