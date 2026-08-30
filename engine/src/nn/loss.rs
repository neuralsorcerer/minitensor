// Copyright (c) Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

use crate::{
    error::Result,
    ops::loss::{
        binary_cross_entropy_loss, binary_cross_entropy_with_logits_loss, cross_entropy_loss,
        focal_loss, huber_loss, log_cosh_loss, mae_loss, mse_loss, smooth_l1_loss,
    },
    tensor::Tensor,
};

/// Declares a loss layer whose whole state is its reduction mode.
///
/// Five of the layers here were the same struct and the same seven methods,
/// differing only in which free function `forward` calls. That call is what
/// this takes; the constructors named for each reduction, the getter and the
/// setter are written once.
///
/// The layers with a hyperparameter of their own -- `HuberLoss`'s delta,
/// `SmoothL1Loss`'s beta, `FocalLoss`'s alpha and gamma,
/// `BCEWithLogitsLoss`'s positive-class weight -- stay written out: their
/// constructors carry that parameter, and it is the part of them worth reading.
macro_rules! reduction_only_loss {
    ($name:ident, $forward:ident, $what:literal) => {
        #[doc = concat!("The ", $what, " loss as a layer.")]
        #[derive(Debug, Clone)]
        pub struct $name {
            reduction: String,
        }

        impl $name {
            #[doc = concat!("A ", $what, " loss reducing the way `reduction` says.")]
            pub fn new(reduction: impl Into<String>) -> Self {
                Self {
                    reduction: reduction.into(),
                }
            }

            #[doc = concat!("A ", $what, " loss averaged over its elements, which is the usual choice.")]
            pub fn mean() -> Self {
                Self::new("mean")
            }

            #[doc = concat!("A ", $what, " loss summed over its elements.")]
            pub fn sum() -> Self {
                Self::new("sum")
            }

            #[doc = concat!("A ", $what, " loss left element-wise, with nothing reduced.")]
            pub fn none() -> Self {
                Self::new("none")
            }

            #[doc = concat!("The ", $what, " loss between `predictions` and `targets`.")]
            pub fn forward(&self, predictions: &Tensor, targets: &Tensor) -> Result<Tensor> {
                $forward(predictions, targets, &self.reduction)
            }

            /// The reduction mode this layer was built with.
            pub fn reduction(&self) -> &str {
                &self.reduction
            }

            /// Change the reduction mode.
            pub fn set_reduction(&mut self, reduction: impl Into<String>) {
                self.reduction = reduction.into();
            }
        }
    };
}

reduction_only_loss!(MSELoss, mse_loss, "mean squared error");
reduction_only_loss!(MAELoss, mae_loss, "mean absolute error");
reduction_only_loss!(LogCoshLoss, log_cosh_loss, "log-cosh");
reduction_only_loss!(
    CrossEntropyLoss,
    cross_entropy_loss,
    "softmax cross entropy"
);
reduction_only_loss!(BCELoss, binary_cross_entropy_loss, "binary cross entropy");

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

/// Smooth L1 loss layer
#[derive(Debug, Clone)]
pub struct SmoothL1Loss {
    reduction: String,
    beta: f64,
}

impl SmoothL1Loss {
    /// Create a new Smooth L1 loss with the specified reduction and `beta = 1`.
    pub fn new(reduction: impl Into<String>) -> Self {
        Self {
            reduction: reduction.into(),
            beta: 1.0,
        }
    }

    /// Set the threshold below which the loss is quadratic. `beta` must be
    /// positive and finite; [`Self::forward`] surfaces the error if it is not.
    pub fn with_beta(mut self, beta: f64) -> Self {
        self.beta = beta;
        self
    }

    /// The threshold below which the loss is quadratic.
    pub fn beta(&self) -> f64 {
        self.beta
    }

    /// Create Smooth L1 loss with mean reduction (default)
    pub fn mean() -> Self {
        Self::new("mean")
    }

    /// Create Smooth L1 loss with sum reduction
    pub fn sum() -> Self {
        Self::new("sum")
    }

    /// Create Smooth L1 loss with no reduction (element-wise)
    pub fn none() -> Self {
        Self::new("none")
    }

    /// Compute the Smooth L1 loss between predictions and targets
    pub fn forward(&self, predictions: &Tensor, targets: &Tensor) -> Result<Tensor> {
        smooth_l1_loss(predictions, targets, self.beta, &self.reduction)
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
    use crate::test_support::tensor_of;

    #[test]
    fn test_mse_loss_layer() {
        let mse = MSELoss::mean();
        assert_eq!(mse.reduction(), "mean");

        let predictions = tensor_of::<f32>(vec![1.0, 2.0, 3.0], vec![3], false);
        let targets = tensor_of::<f32>(vec![1.5, 2.5, 2.5], vec![3], false);

        let loss = mse.forward(&predictions, &targets).unwrap();
        let loss_data = loss.data().as_f32_slice().unwrap();

        // Expected: ((1.0-1.5)² + (2.0-2.5)² + (3.0-2.5)²) / 3 = 0.25
        assert!((loss_data[0] - 0.25).abs() < 1e-6);
    }

    #[test]
    fn test_mae_loss_layer() {
        let mae = MAELoss::mean();
        assert_eq!(mae.reduction(), "mean");

        let predictions = tensor_of::<f32>(vec![1.0, 2.0, 3.0], vec![3], false);
        let targets = tensor_of::<f32>(vec![1.5, 2.5, 2.0], vec![3], false);

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

        let predictions = tensor_of::<f32>(vec![1.0, 2.0], vec![2], false);
        let targets = tensor_of::<f32>(vec![1.2, 2.3], vec![2], false);

        let loss = huber.forward(&predictions, &targets).unwrap();
        // A reduced loss is a 0-dim scalar.
        assert_eq!(loss.shape().dims(), &[] as &[usize]);
    }

    #[test]
    fn test_smooth_l1_loss_layer() {
        let smooth = SmoothL1Loss::mean();
        assert_eq!(smooth.reduction(), "mean");

        let predictions = tensor_of::<f32>(vec![0.5, 2.0], vec![2], false);
        let targets = tensor_of::<f32>(vec![0.0, 0.0], vec![2], false);

        let loss = smooth.forward(&predictions, &targets).unwrap();
        let loss_data = loss.data().as_f32_slice().unwrap();

        // Smooth L1 with delta=1.0: (0.5*0.5^2 + (2.0 - 0.5)) / 2 = 0.8125
        assert!((loss_data[0] - 0.8125).abs() < 1e-6);
    }

    #[test]
    fn test_log_cosh_loss_layer() {
        let log_cosh = LogCoshLoss::mean();
        assert_eq!(log_cosh.reduction(), "mean");

        let predictions = tensor_of::<f32>(vec![0.0, 1.0], vec![2], false);
        let targets = tensor_of::<f32>(vec![0.0, 0.0], vec![2], false);

        let loss = log_cosh.forward(&predictions, &targets).unwrap();
        let loss_data = loss.data().as_f32_slice().unwrap();

        let expected = (0.0f32.cosh().ln() + 1.0f32.cosh().ln()) / 2.0;
        assert!((loss_data[0] - expected).abs() < 1e-6);
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

        let smooth_mean = SmoothL1Loss::mean();
        assert_eq!(smooth_mean.reduction(), "mean");

        let log_cosh_mean = LogCoshLoss::mean();
        assert_eq!(log_cosh_mean.reduction(), "mean");
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

        let mut smooth = SmoothL1Loss::mean();
        smooth.set_reduction("none");
        assert_eq!(smooth.reduction(), "none");

        let mut log_cosh = LogCoshLoss::mean();
        log_cosh.set_reduction("sum");
        assert_eq!(log_cosh.reduction(), "sum");
    }
}

/// Binary Cross Entropy loss layer that takes logits rather than probabilities.
///
/// Prefer this to `Sigmoid` followed by [`BCELoss`]: it is the same function
/// mathematically, but it stays numerically exact at logit magnitudes where the
/// two-step form has already lost its gradient. See
/// [`binary_cross_entropy_with_logits_loss`] for the details.
#[derive(Debug, Clone)]
pub struct BCEWithLogitsLoss {
    reduction: String,
    pos_weight: Option<Tensor>,
}

impl BCEWithLogitsLoss {
    /// Create a new BCE-with-logits loss with the specified reduction
    pub fn new(reduction: impl Into<String>) -> Self {
        Self {
            reduction: reduction.into(),
            pos_weight: None,
        }
    }

    /// Create the loss with a weight applied to the positive class, broadcast
    /// against the targets
    pub fn with_pos_weight(reduction: impl Into<String>, pos_weight: Tensor) -> Self {
        Self {
            reduction: reduction.into(),
            pos_weight: Some(pos_weight),
        }
    }

    /// Create BCE-with-logits loss with mean reduction (default)
    pub fn mean() -> Self {
        Self::new("mean")
    }

    /// Create BCE-with-logits loss with sum reduction
    pub fn sum() -> Self {
        Self::new("sum")
    }

    /// Create BCE-with-logits loss with no reduction (element-wise)
    pub fn none() -> Self {
        Self::new("none")
    }

    /// Compute the loss between raw logits and targets
    pub fn forward(&self, logits: &Tensor, targets: &Tensor) -> Result<Tensor> {
        binary_cross_entropy_with_logits_loss(
            logits,
            targets,
            self.pos_weight.as_ref(),
            &self.reduction,
        )
    }

    /// Get the reduction mode
    pub fn reduction(&self) -> &str {
        &self.reduction
    }

    /// Set the reduction mode
    pub fn set_reduction(&mut self, reduction: impl Into<String>) {
        self.reduction = reduction.into();
    }

    /// Get the positive-class weight, if one was set
    pub fn pos_weight(&self) -> Option<&Tensor> {
        self.pos_weight.as_ref()
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
    use crate::test_support::tensor_of;

    #[test]
    fn test_cross_entropy_loss_layer() {
        let ce_loss = CrossEntropyLoss::mean();
        assert_eq!(ce_loss.reduction(), "mean");

        // Create simple 2-class classification example
        let predictions = tensor_of::<f32>(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2], false);
        let targets = tensor_of::<f32>(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2], false);

        let loss = ce_loss.forward(&predictions, &targets);
        // Just check that the loss was computed successfully
        assert!(loss.is_ok());
    }

    #[test]
    fn test_bce_loss_layer() {
        let bce_loss = BCELoss::mean();
        assert_eq!(bce_loss.reduction(), "mean");

        // Create binary classification example with probabilities
        let predictions = tensor_of::<f32>(vec![0.8, 0.2, 0.3, 0.9], vec![4], false);
        let targets = tensor_of::<f32>(vec![1.0, 0.0, 0.0, 1.0], vec![4], false);

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
        let predictions = tensor_of::<f32>(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2], false);
        let targets = tensor_of::<f32>(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2], false);

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
