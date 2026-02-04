// Copyright (c) 2026 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

use crate::{
    autograd::{
        BCELossBackward, CrossEntropyLossBackward, FocalLossBackward, HuberLossBackward,
        KLDivLossBackward, MAELossBackward, MSELossBackward, add_to_graph,
    },
    error::{MinitensorError, Result},
    operations::{
        activation::{abs as activation_abs, exp, log_softmax, log1p},
        arithmetic::{add, mul, sub},
        reduction::{mean, sum},
    },
    tensor::{DataType, Shape, Tensor, TensorData},
};
use rayon::prelude::*;
use std::sync::Arc;

const CHUNK: usize = 1024;

/// Mean Squared Error (MSE) loss function
///
/// Computes the mean squared error between predictions and targets:
/// MSE = (1/n) * Σ(predictions - targets)²
///
/// # Arguments
/// * `predictions` - Model predictions tensor
/// * `targets` - Ground truth targets tensor
/// * `reduction` - How to reduce the loss ("mean", "sum", or "none")
///
/// # Returns
/// * `Result<Tensor>` - The computed MSE loss
pub fn mse_loss(predictions: &Tensor, targets: &Tensor, reduction: &str) -> Result<Tensor> {
    // Validate inputs
    validate_loss_inputs(predictions, targets)?;

    // Compute squared differences: (predictions - targets)²
    // Also keep the difference for gradient computation
    let diff = sub(predictions, targets)?;
    let diff_for_grad = diff.clone().detach();
    let squared_diff = mul(&diff, &diff)?;

    // Apply reduction
    let loss = match reduction {
        "mean" => {
            // Compute mean of squared differences
            let sum = sum_all_elements(&squared_diff)?;
            let n = squared_diff.numel() as f64;
            divide_by_scalar(&sum, n)?
        }
        "sum" => {
            // Sum all squared differences
            sum_all_elements(&squared_diff)?
        }
        "none" => {
            // Return element-wise squared differences
            squared_diff
        }
        _ => {
            return Err(MinitensorError::invalid_operation(format!(
                "Invalid reduction mode: {}. Must be 'mean', 'sum', or 'none'",
                reduction
            )));
        }
    };

    // Set up gradient function if needed
    if loss.requires_grad() {
        let grad_fn = Arc::new(MSELossBackward {
            predictions_shape: predictions.shape().dims().to_vec(),
            targets_shape: targets.shape().dims().to_vec(),
            input_ids: [predictions.id(), targets.id()],
            reduction: reduction.to_string(),
            diff: diff_for_grad,
        });

        let mut loss_with_grad = loss;
        loss_with_grad.set_grad_fn(Some(grad_fn.clone()));

        // Add to computation graph
        add_to_graph(&loss_with_grad, Some(grad_fn))?;

        Ok(loss_with_grad)
    } else {
        Ok(loss)
    }
}

/// Mean Absolute Error (MAE) loss function
///
/// Computes the mean absolute error between predictions and targets:
/// MAE = (1/n) * Σ|predictions - targets|
///
/// # Arguments
/// * `predictions` - Model predictions tensor
/// * `targets` - Ground truth targets tensor
/// * `reduction` - How to reduce the loss ("mean", "sum", or "none")
///
/// # Returns
/// * `Result<Tensor>` - The computed MAE loss
pub fn mae_loss(predictions: &Tensor, targets: &Tensor, reduction: &str) -> Result<Tensor> {
    // Validate inputs
    validate_loss_inputs(predictions, targets)?;

    // Compute absolute differences: |predictions - targets|
    // Also compute the sign for gradient computation
    let diff = sub(predictions, targets)?;
    let sign_diff = sign(&diff)?;
    let sign_for_grad = sign_diff.clone().detach();
    let abs_diff = activation_abs(&diff.detach())?;

    // Apply reduction
    let loss = match reduction {
        "mean" => {
            // Compute mean of absolute differences
            let sum = sum_all_elements(&abs_diff)?;
            let n = abs_diff.numel() as f64;
            divide_by_scalar(&sum, n)?
        }
        "sum" => {
            // Sum all absolute differences
            sum_all_elements(&abs_diff)?
        }
        "none" => {
            // Return element-wise absolute differences
            abs_diff
        }
        _ => {
            return Err(MinitensorError::invalid_operation(format!(
                "Invalid reduction mode: {}. Must be 'mean', 'sum', or 'none'",
                reduction
            )));
        }
    };

    // Set up gradient function if needed
    if loss.requires_grad() {
        let grad_fn = Arc::new(MAELossBackward {
            predictions_shape: predictions.shape().dims().to_vec(),
            targets_shape: targets.shape().dims().to_vec(),
            input_ids: [predictions.id(), targets.id()],
            reduction: reduction.to_string(),
            sign: sign_for_grad,
        });

        let mut loss_with_grad = loss;
        loss_with_grad.set_grad_fn(Some(grad_fn.clone()));

        // Add to computation graph
        add_to_graph(&loss_with_grad, Some(grad_fn))?;

        Ok(loss_with_grad)
    } else {
        Ok(loss)
    }
}

/// Cross Entropy loss function for classification
///
/// Computes the cross entropy loss between predictions (logits) and targets:
/// CE = -Σ(targets * log(softmax(predictions)))
///
/// # Arguments
/// * `predictions` - Model predictions (logits) tensor
/// * `targets` - Ground truth targets tensor (class indices or one-hot)
/// * `reduction` - How to reduce the loss ("mean", "sum", or "none")
///
/// # Returns
/// * `Result<Tensor>` - The computed cross entropy loss
pub fn cross_entropy_loss(
    predictions: &Tensor,
    targets: &Tensor,
    reduction: &str,
) -> Result<Tensor> {
    // Validate inputs
    validate_classification_inputs(predictions, targets, false)?;

    // Convert class indices to one-hot encoding if needed
    let targets_one_hot = prepare_classification_targets(predictions, targets)?;

    // Apply log-softmax to predictions for numerical stability
    let log_predictions = log_softmax(predictions, None)?;
    let softmax_predictions = exp(&log_predictions.detach())?;

    // Compute negative log likelihood summed over classes
    let nll = negative_log_likelihood(&log_predictions, &targets_one_hot)?;
    let per_sample = sum(&nll, Some(vec![1]), false)?;

    // Apply reduction
    let loss = match reduction {
        "mean" => {
            let sum = sum_all_elements(&per_sample)?;
            let batch = per_sample.shape().dims().first().copied().unwrap_or(1) as f64;
            divide_by_scalar(&sum, batch)?
        }
        "sum" => sum_all_elements(&per_sample)?,
        "none" => per_sample,
        _ => {
            return Err(MinitensorError::invalid_operation(format!(
                "Invalid reduction mode: {}. Must be 'mean', 'sum', or 'none'",
                reduction
            )));
        }
    };

    // Set up gradient function if needed
    if loss.requires_grad() {
        let grad_fn = Arc::new(CrossEntropyLossBackward {
            predictions_shape: predictions.shape().dims().to_vec(),
            targets_shape: targets_one_hot.shape().dims().to_vec(),
            input_ids: [predictions.id(), targets.id()],
            reduction: reduction.to_string(),
            softmax_predictions: softmax_predictions.clone().detach(),
            targets: targets_one_hot.clone().detach(),
        });

        let mut loss_with_grad = loss;
        loss_with_grad.set_grad_fn(Some(grad_fn.clone()));

        // Add to computation graph
        add_to_graph(&loss_with_grad, Some(grad_fn))?;

        Ok(loss_with_grad)
    } else {
        Ok(loss)
    }
}

/// Cross entropy loss for tensors with arbitrary shapes and class dimension.
///
/// This wrapper permutes and flattens the input so that the core
/// `cross_entropy_loss` implementation can operate on ``[N, C]`` shaped
/// tensors entirely in Rust.
pub fn cross_entropy(
    input: &Tensor,
    target: &Tensor,
    reduction: &str,
    dim: usize,
) -> Result<Tensor> {
    let ndim = input.ndim();
    if dim >= ndim {
        return Err(MinitensorError::invalid_operation(
            "dim out of range in cross_entropy",
        ));
    }

    // Move class dimension to the end using successive transposes
    let mut pred = input.clone();
    let mut tgt = target.clone();
    if dim != ndim - 1 {
        for i in dim..(ndim - 1) {
            pred = pred.transpose(i as isize, (i + 1) as isize)?;
            if target.ndim() == ndim {
                tgt = tgt.transpose(i as isize, (i + 1) as isize)?;
            }
        }
    }

    // Flatten all but the class dimension
    let flat_size: usize = pred.shape().dims().iter().take(ndim - 1).product();
    let classes = pred.shape().dims()[ndim - 1];
    let pred_2d = pred.reshape(Shape::new(vec![flat_size, classes]))?;
    let tgt_flat = if tgt.ndim() == ndim {
        tgt.reshape(Shape::new(vec![flat_size, classes]))?
    } else {
        tgt.reshape(Shape::new(vec![flat_size]))?
    };

    let loss = cross_entropy_loss(&pred_2d, &tgt_flat, reduction)?;

    if reduction == "none" {
        // Restore the original shape without the class dimension
        let out_shape: Vec<usize> = input
            .shape()
            .dims()
            .iter()
            .enumerate()
            .filter_map(|(i, &d)| if i != dim { Some(d) } else { None })
            .collect();
        loss.reshape(Shape::new(out_shape))
    } else {
        Ok(loss)
    }
}

/// Binary Cross Entropy loss function
///
/// Computes the binary cross entropy loss between predictions and targets:
/// BCE = -Σ(targets * log(predictions) + (1 - targets) * log(1 - predictions))
///
/// # Arguments
/// * `predictions` - Model predictions tensor (probabilities between 0 and 1)
/// * `targets` - Ground truth targets tensor (0 or 1)
/// * `reduction` - How to reduce the loss ("mean", "sum", or "none")
///
/// # Returns
/// * `Result<Tensor>` - The computed BCE loss
pub fn binary_cross_entropy_loss(
    predictions: &Tensor,
    targets: &Tensor,
    reduction: &str,
) -> Result<Tensor> {
    // Validate inputs
    validate_loss_inputs(predictions, targets)?;

    // Compute BCE: -[targets * log(predictions) + (1 - targets) * log(1 - predictions)]
    let log_predictions = log(predictions)?;

    let ones = Tensor::ones(
        predictions.shape().clone(),
        predictions.dtype(),
        predictions.device(),
        false,
    );
    let one_minus_targets = sub(&ones, targets)?;
    let one_minus_predictions = sub(&ones, predictions)?;
    let log_one_minus_predictions = log(&one_minus_predictions)?;

    let term1 = mul(targets, &log_predictions)?;
    let term2 = mul(&one_minus_targets, &log_one_minus_predictions)?;
    let combined = add(&term1, &term2)?;
    let zeros = Tensor::zeros(
        combined.shape().clone(),
        combined.dtype(),
        combined.device(),
        combined.requires_grad(),
    );
    let negative_bce = sub(&zeros, &combined)?;

    // Apply reduction
    let loss = match reduction {
        "mean" => {
            let sum = sum_all_elements(&negative_bce)?;
            let n = negative_bce.numel() as f64;
            divide_by_scalar(&sum, n)?
        }
        "sum" => sum_all_elements(&negative_bce)?,
        "none" => negative_bce,
        _ => {
            return Err(MinitensorError::invalid_operation(format!(
                "Invalid reduction mode: {}. Must be 'mean', 'sum', or 'none'",
                reduction
            )));
        }
    };

    // Set up gradient function if needed
    if loss.requires_grad() {
        let grad_fn = Arc::new(BCELossBackward {
            predictions_shape: predictions.shape().dims().to_vec(),
            targets_shape: targets.shape().dims().to_vec(),
            input_ids: [predictions.id(), targets.id()],
            reduction: reduction.to_string(),
            predictions: predictions.clone().detach(),
            targets: targets.clone().detach(),
        });

        let mut loss_with_grad = loss;
        loss_with_grad.set_grad_fn(Some(grad_fn.clone()));

        // Add to computation graph
        add_to_graph(&loss_with_grad, Some(grad_fn))?;

        Ok(loss_with_grad)
    } else {
        Ok(loss)
    }
}

/// Kullback-Leibler divergence loss function
///
/// Computes KL divergence between target and prediction distributions:
/// KL(target || prediction) = Σ target * (log(target) - log(prediction))
pub fn kl_div_loss(predictions: &Tensor, targets: &Tensor, reduction: &str) -> Result<Tensor> {
    // Validate inputs
    validate_loss_inputs(predictions, targets)?;

    // Compute elementwise targets * (log(targets) - log(predictions))
    let log_targets = log(targets)?;
    let log_predictions = log(predictions)?;
    let diff = sub(&log_targets, &log_predictions)?;
    let kld = mul(targets, &diff)?;

    // Apply reduction
    let loss = match reduction {
        "mean" => {
            let sum = sum_all_elements(&kld)?;
            // Compute mean over the batch dimension if present.
            // For 1D tensors (single distribution), the batch size is 1
            let batch = if predictions.shape().dims().len() > 1 {
                predictions.shape().dims()[0] as f64
            } else {
                1.0
            };
            divide_by_scalar(&sum, batch)?
        }
        "sum" => sum_all_elements(&kld)?,
        "none" => kld,
        _ => {
            return Err(MinitensorError::invalid_operation(format!(
                "Invalid reduction mode: {}. Must be 'mean', 'sum', or 'none'",
                reduction
            )));
        }
    };

    // Set up gradient function if needed
    if loss.requires_grad() {
        let grad_fn = Arc::new(KLDivLossBackward {
            predictions_shape: predictions.shape().dims().to_vec(),
            targets_shape: targets.shape().dims().to_vec(),
            input_ids: [predictions.id(), targets.id()],
            reduction: reduction.to_string(),
            predictions: predictions.clone().detach(),
            targets: targets.clone().detach(),
        });

        let mut loss_with_grad = loss;
        loss_with_grad.set_grad_fn(Some(grad_fn.clone()));

        // Add to computation graph
        add_to_graph(&loss_with_grad, Some(grad_fn))?;

        Ok(loss_with_grad)
    } else {
        Ok(loss)
    }
}

/// Focal loss function for handling class imbalance
///
/// Computes the focal loss, which is a modified cross entropy loss:
/// FL = -α * (1 - p_t)^γ * log(p_t)
/// where p_t is the predicted probability for the true class
///
/// # Arguments
/// * `predictions` - Model predictions (logits) tensor
/// * `targets` - Ground truth targets tensor
/// * `alpha` - Weighting factor for rare class (typically 0.25)
/// * `gamma` - Focusing parameter (typically 2.0)
/// * `reduction` - How to reduce the loss ("mean", "sum", or "none")
///
/// # Returns
/// * `Result<Tensor>` - The computed focal loss
pub fn focal_loss(
    predictions: &Tensor,
    targets: &Tensor,
    alpha: f64,
    gamma: f64,
    reduction: &str,
) -> Result<Tensor> {
    // Validate inputs
    validate_classification_inputs(predictions, targets, false)?;

    let targets_one_hot = prepare_classification_targets(predictions, targets)?;

    if alpha <= 0.0 || alpha >= 1.0 {
        return Err(MinitensorError::invalid_operation(
            "Alpha must be between 0 and 1 for focal loss",
        ));
    }

    if gamma < 0.0 {
        return Err(MinitensorError::invalid_operation(
            "Gamma must be non-negative for focal loss",
        ));
    }

    // Apply log-softmax to predictions for numerical stability
    let log_predictions = log_softmax(predictions, None)?;
    let softmax_predictions = exp(&log_predictions)?;
    let softmax_for_grad = softmax_predictions.clone().detach();

    // Compute focal loss components
    let ones = Tensor::ones(
        softmax_predictions.shape().clone(),
        softmax_predictions.dtype(),
        softmax_predictions.device(),
        false,
    );
    let one_minus_p = sub(&ones, &softmax_predictions)?;
    let focal_weight = power(&one_minus_p, gamma)?;

    // Compute negative log likelihood with focal weighting
    let nll = negative_log_likelihood(&log_predictions, &targets_one_hot)?;
    let alpha_tensor = create_scalar_tensor(alpha, predictions.dtype(), predictions.device())?;
    let weighted_nll = mul(&nll, &focal_weight)?;
    let focal_values = mul(&weighted_nll, &alpha_tensor)?;

    // Apply reduction
    let loss = match reduction {
        "mean" => {
            let sum = sum_all_elements(&focal_values)?;
            let n = focal_values.numel() as f64;
            divide_by_scalar(&sum, n)?
        }
        "sum" => sum_all_elements(&focal_values)?,
        "none" => focal_values,
        _ => {
            return Err(MinitensorError::invalid_operation(format!(
                "Invalid reduction mode: {}. Must be 'mean', 'sum', or 'none'",
                reduction
            )));
        }
    };

    // Set up gradient function if needed
    if loss.requires_grad() {
        let grad_fn = Arc::new(FocalLossBackward {
            predictions_shape: predictions.shape().dims().to_vec(),
            targets_shape: targets_one_hot.shape().dims().to_vec(),
            input_ids: [predictions.id(), targets.id()],
            alpha,
            gamma,
            reduction: reduction.to_string(),
            softmax_predictions: softmax_for_grad,
            targets: targets_one_hot.clone().detach(),
        });

        let mut loss_with_grad = loss;
        loss_with_grad.set_grad_fn(Some(grad_fn.clone()));

        // Add to computation graph
        add_to_graph(&loss_with_grad, Some(grad_fn))?;

        Ok(loss_with_grad)
    } else {
        Ok(loss)
    }
}

/// Huber loss function for robust regression
///
/// Combines MSE and MAE for robust regression:
/// - For |x| <= delta: 0.5 * x²
/// - For |x| > delta: delta * (|x| - 0.5 * delta)
///
/// # Arguments
/// * `predictions` - Model predictions tensor
/// * `targets` - Ground truth targets tensor
/// * `delta` - Threshold for switching between MSE and MAE behavior
/// * `reduction` - How to reduce the loss ("mean", "sum", or "none")
///
/// # Returns
/// * `Result<Tensor>` - The computed Huber loss
pub fn huber_loss(
    predictions: &Tensor,
    targets: &Tensor,
    delta: f64,
    reduction: &str,
) -> Result<Tensor> {
    // Validate inputs
    validate_loss_inputs(predictions, targets)?;

    if delta <= 0.0 {
        return Err(MinitensorError::invalid_operation(
            "Delta must be positive for Huber loss",
        ));
    }

    // Compute absolute differences: |predictions - targets|
    let diff = sub(predictions, targets)?;
    let diff_for_grad = diff.clone().detach();
    let abs_diff = activation_abs(&diff.detach())?;

    // Create delta tensor for comparison
    let delta_tensor = create_scalar_tensor(delta, predictions.dtype(), predictions.device())?;

    // Compute Huber loss element-wise
    let huber_values = compute_huber_elementwise(&abs_diff, &diff, &delta_tensor, delta)?;

    // Apply reduction
    let loss = match reduction {
        "mean" => {
            let sum = sum_all_elements(&huber_values)?;
            let n = huber_values.numel() as f64;
            divide_by_scalar(&sum, n)?
        }
        "sum" => sum_all_elements(&huber_values)?,
        "none" => huber_values,
        _ => {
            return Err(MinitensorError::invalid_operation(format!(
                "Invalid reduction mode: {}. Must be 'mean', 'sum', or 'none'",
                reduction
            )));
        }
    };

    // Set up gradient function if needed
    if loss.requires_grad() {
        let grad_fn = Arc::new(HuberLossBackward {
            predictions_shape: predictions.shape().dims().to_vec(),
            targets_shape: targets.shape().dims().to_vec(),
            input_ids: [predictions.id(), targets.id()],
            delta,
            reduction: reduction.to_string(),
            diff: diff_for_grad,
        });

        let mut loss_with_grad = loss;
        loss_with_grad.set_grad_fn(Some(grad_fn.clone()));

        // Add to computation graph
        add_to_graph(&loss_with_grad, Some(grad_fn))?;

        Ok(loss_with_grad)
    } else {
        Ok(loss)
    }
}

/// Smooth L1 loss (Huber loss with delta=1.0)
///
/// Computes Smooth L1 loss between predictions and targets:
/// SmoothL1(x) = 0.5 * x² if |x| < 1, otherwise |x| - 0.5
///
/// # Arguments
/// * `predictions` - Model predictions tensor
/// * `targets` - Ground truth targets tensor
/// * `reduction` - How to reduce the loss ("mean", "sum", or "none")
pub fn smooth_l1_loss(predictions: &Tensor, targets: &Tensor, reduction: &str) -> Result<Tensor> {
    huber_loss(predictions, targets, 1.0, reduction)
}

/// Log-cosh loss for robust regression
///
/// Computes log(cosh(x)) where x = predictions - targets using a numerically
/// stable formulation: |x| + log1p(exp(-2|x|)) - log(2).
///
/// # Arguments
/// * `predictions` - Model predictions tensor
/// * `targets` - Ground truth targets tensor
/// * `reduction` - How to reduce the loss ("mean", "sum", or "none")
pub fn log_cosh_loss(predictions: &Tensor, targets: &Tensor, reduction: &str) -> Result<Tensor> {
    validate_loss_inputs(predictions, targets)?;

    let diff = sub(predictions, targets)?;
    let diff_abs = activation_abs(&diff)?;
    let neg_two = create_scalar_tensor(-2.0, diff.dtype(), diff.device())?;
    let exp_term = exp(&mul(&diff_abs, &neg_two)?)?;
    let log1p_term = log1p(&exp_term)?;
    let log2 = create_scalar_tensor(std::f64::consts::LN_2, diff.dtype(), diff.device())?;
    let log_cosh = sub(&add(&diff_abs, &log1p_term)?, &log2)?;

    match reduction {
        "mean" => mean(&log_cosh, None, false),
        "sum" => sum(&log_cosh, None, false),
        "none" => Ok(log_cosh),
        _ => Err(MinitensorError::invalid_operation(format!(
            "Invalid reduction mode: {}. Must be 'mean', 'sum', or 'none'",
            reduction
        ))),
    }
}

// Helper functions

/// Validate that loss function inputs are compatible
fn validate_loss_inputs(predictions: &Tensor, targets: &Tensor) -> Result<()> {
    // Check device compatibility
    if predictions.device() != targets.device() {
        return Err(MinitensorError::device_mismatch(
            format!("{:?}", predictions.device()),
            format!("{:?}", targets.device()),
        ));
    }

    // Check data type compatibility
    if predictions.dtype() != targets.dtype() {
        return Err(MinitensorError::type_mismatch(
            format!("{:?}", predictions.dtype()),
            format!("{:?}", targets.dtype()),
        ));
    }

    // Check shape compatibility
    if predictions.shape() != targets.shape() {
        return Err(MinitensorError::shape_mismatch(
            predictions.shape().dims().to_vec(),
            targets.shape().dims().to_vec(),
        ));
    }

    // Check that tensors contain floating point data (required for loss computation)
    match predictions.dtype() {
        DataType::Float32 | DataType::Float64 => {}
        _ => {
            return Err(MinitensorError::invalid_operation(
                "Loss functions require floating point tensors",
            ));
        }
    }

    Ok(())
}

/// Validate that classification loss function inputs are compatible
fn validate_classification_inputs(
    predictions: &Tensor,
    targets: &Tensor,
    require_same_dtype: bool,
) -> Result<()> {
    // Check device compatibility
    if predictions.device() != targets.device() {
        return Err(MinitensorError::device_mismatch(
            format!("{:?}", predictions.device()),
            format!("{:?}", targets.device()),
        ));
    }

    // Optionally enforce data type equality
    if require_same_dtype && predictions.dtype() != targets.dtype() {
        return Err(MinitensorError::type_mismatch(
            format!("{:?}", predictions.dtype()),
            format!("{:?}", targets.dtype()),
        ));
    }

    // Predictions must be at least 2D (batch_size, num_classes)
    if predictions.ndim() < 2 {
        return Err(MinitensorError::invalid_operation(
            "Classification predictions must be at least 2D (batch_size, num_classes)",
        ));
    }

    // Predictions must be floating point
    match predictions.dtype() {
        DataType::Float32 | DataType::Float64 => {}
        _ => {
            return Err(MinitensorError::invalid_operation(
                "Classification loss functions require floating point tensors",
            ));
        }
    }

    Ok(())
}

fn prepare_classification_targets(predictions: &Tensor, targets: &Tensor) -> Result<Tensor> {
    if targets.ndim() + 1 == predictions.ndim() {
        let num_classes = predictions.size(predictions.ndim() - 1)?;
        let total = targets.numel();
        let mut data = TensorData::zeros_on_device(
            total * num_classes,
            predictions.dtype(),
            predictions.device(),
        );
        match (targets.dtype(), predictions.dtype()) {
            (DataType::Int32, DataType::Float32) => {
                let idx = targets.data().as_i32_slice().ok_or_else(|| {
                    MinitensorError::internal_error("Failed to get i32 slice from targets")
                })?;
                let out = data.as_f32_slice_mut().unwrap();
                fill_one_hot_f32(idx, out, num_classes, |val| {
                    checked_index_from_i64(i64::from(*val), num_classes)
                })?;
            }
            (DataType::Int64, DataType::Float32) => {
                let idx = targets.data().as_i64_slice().ok_or_else(|| {
                    MinitensorError::internal_error("Failed to get i64 slice from targets")
                })?;
                let out = data.as_f32_slice_mut().unwrap();
                fill_one_hot_f32(idx, out, num_classes, |val| {
                    checked_index_from_i64(*val, num_classes)
                })?;
            }
            (DataType::Int32, DataType::Float64) => {
                let idx = targets.data().as_i32_slice().ok_or_else(|| {
                    MinitensorError::internal_error("Failed to get i32 slice from targets")
                })?;
                let out = data.as_f64_slice_mut().unwrap();
                fill_one_hot_f64(idx, out, num_classes, |val| {
                    checked_index_from_i64(i64::from(*val), num_classes)
                })?;
            }
            (DataType::Int64, DataType::Float64) => {
                let idx = targets.data().as_i64_slice().ok_or_else(|| {
                    MinitensorError::internal_error("Failed to get i64 slice from targets")
                })?;
                let out = data.as_f64_slice_mut().unwrap();
                fill_one_hot_f64(idx, out, num_classes, |val| {
                    checked_index_from_i64(*val, num_classes)
                })?;
            }
            (DataType::Float32, DataType::Float32) => {
                let idx = targets.data().as_f32_slice().ok_or_else(|| {
                    MinitensorError::internal_error("Failed to get f32 slice from targets")
                })?;
                let out = data.as_f32_slice_mut().unwrap();
                fill_one_hot_f32(idx, out, num_classes, |val| {
                    checked_index_from_f32(*val, num_classes)
                })?;
            }
            (DataType::Float64, DataType::Float64) => {
                let idx = targets.data().as_f64_slice().ok_or_else(|| {
                    MinitensorError::internal_error("Failed to get f64 slice from targets")
                })?;
                let out = data.as_f64_slice_mut().unwrap();
                fill_one_hot_f64(idx, out, num_classes, |val| {
                    checked_index_from_f64(*val, num_classes)
                })?;
            }
            _ => {
                return Err(MinitensorError::invalid_operation(
                    "Unsupported target dtype for classification loss",
                ));
            }
        }
        let mut dims = targets.shape().dims().to_vec();
        dims.push(num_classes);
        Ok(Tensor::new(
            Arc::new(data),
            Shape::new(dims),
            predictions.dtype(),
            predictions.device(),
            false,
        ))
    } else if targets.ndim() == predictions.ndim() {
        if targets.shape().dims() != predictions.shape().dims() {
            return Err(MinitensorError::shape_mismatch(
                predictions.shape().dims().to_vec(),
                targets.shape().dims().to_vec(),
            ));
        }
        Ok(targets.clone())
    } else {
        Err(MinitensorError::shape_mismatch(
            predictions.shape().dims().to_vec(),
            targets.shape().dims().to_vec(),
        ))
    }
}

fn checked_index_from_i64(value: i64, num_classes: usize) -> Result<usize> {
    if value < 0 {
        return Err(MinitensorError::invalid_operation(
            "Target class index must be non-negative",
        ));
    }
    let index = value as usize;
    if index >= num_classes {
        return Err(MinitensorError::invalid_operation(
            "Target class index out of range",
        ));
    }
    Ok(index)
}

fn checked_index_from_f32(value: f32, num_classes: usize) -> Result<usize> {
    if !value.is_finite() || value.fract() != 0.0 {
        return Err(MinitensorError::invalid_operation(
            "Target class index must be a finite integer",
        ));
    }
    if value < 0.0 || value >= num_classes as f32 {
        return Err(MinitensorError::invalid_operation(
            "Target class index out of range",
        ));
    }
    Ok(value as usize)
}

fn checked_index_from_f64(value: f64, num_classes: usize) -> Result<usize> {
    if !value.is_finite() || value.fract() != 0.0 {
        return Err(MinitensorError::invalid_operation(
            "Target class index must be a finite integer",
        ));
    }
    if value < 0.0 || value >= num_classes as f64 {
        return Err(MinitensorError::invalid_operation(
            "Target class index out of range",
        ));
    }
    Ok(value as usize)
}

fn fill_one_hot_f32<T, F>(
    indices: &[T],
    out: &mut [f32],
    num_classes: usize,
    to_index: F,
) -> Result<()>
where
    F: Fn(&T) -> Result<usize>,
{
    for (i, value) in indices.iter().enumerate() {
        let class = to_index(value)?;
        out[i * num_classes + class] = 1.0;
    }
    Ok(())
}
