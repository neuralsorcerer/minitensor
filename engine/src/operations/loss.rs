// Copyright (c) 2025 Soumyadip Sarkar.
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
        arithmetic::{add, mul, sub},
        reduction::{max, sum},
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
    let abs_diff = abs(&diff)?;

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

    // Apply softmax to predictions for numerical stability
    let softmax_predictions = softmax(predictions)?;

    // Compute negative log likelihood summed over classes
    let log_predictions = log(&softmax_predictions)?;
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

    // Apply softmax to predictions
    let softmax_predictions = softmax(predictions)?;
    let softmax_for_grad = softmax_predictions.clone().detach();

    // Compute focal loss components
    let log_predictions = log(&softmax_predictions)?;
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
    let abs_diff = abs(&diff)?;

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
                for (i, &val) in idx.iter().enumerate() {
                    out[i * num_classes + val as usize] = 1.0;
                }
            }
            (DataType::Int64, DataType::Float32) => {
                let idx = targets.data().as_i64_slice().ok_or_else(|| {
                    MinitensorError::internal_error("Failed to get i64 slice from targets")
                })?;
                let out = data.as_f32_slice_mut().unwrap();
                for (i, &val) in idx.iter().enumerate() {
                    out[i * num_classes + val as usize] = 1.0;
                }
            }
            (DataType::Int32, DataType::Float64) => {
                let idx = targets.data().as_i32_slice().ok_or_else(|| {
                    MinitensorError::internal_error("Failed to get i32 slice from targets")
                })?;
                let out = data.as_f64_slice_mut().unwrap();
                for (i, &val) in idx.iter().enumerate() {
                    out[i * num_classes + val as usize] = 1.0;
                }
            }
            (DataType::Int64, DataType::Float64) => {
                let idx = targets.data().as_i64_slice().ok_or_else(|| {
                    MinitensorError::internal_error("Failed to get i64 slice from targets")
                })?;
                let out = data.as_f64_slice_mut().unwrap();
                for (i, &val) in idx.iter().enumerate() {
                    out[i * num_classes + val as usize] = 1.0;
                }
            }
            (DataType::Float32, DataType::Float32) => {
                let idx = targets.data().as_f32_slice().ok_or_else(|| {
                    MinitensorError::internal_error("Failed to get f32 slice from targets")
                })?;
                let out = data.as_f32_slice_mut().unwrap();
                for (i, &val) in idx.iter().enumerate() {
                    out[i * num_classes + val as usize] = 1.0;
                }
            }
            (DataType::Float64, DataType::Float64) => {
                let idx = targets.data().as_f64_slice().ok_or_else(|| {
                    MinitensorError::internal_error("Failed to get f64 slice from targets")
                })?;
                let out = data.as_f64_slice_mut().unwrap();
                for (i, &val) in idx.iter().enumerate() {
                    out[i * num_classes + val as usize] = 1.0;
                }
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

/// Compute absolute value of tensor elements
fn abs(tensor: &Tensor) -> Result<Tensor> {
    let mut output_data =
        TensorData::zeros_on_device(tensor.numel(), tensor.dtype(), tensor.device());

    match tensor.dtype() {
        DataType::Float32 => {
            let input_data = tensor.data().as_f32_slice().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get f32 slice from tensor")
            })?;
            let output_slice = output_data.as_f32_slice_mut().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get mutable f32 slice from output")
            })?;

            for (i, &val) in input_data.iter().enumerate() {
                output_slice[i] = val.abs();
            }
        }
        DataType::Float64 => {
            let input_data = tensor.data().as_f64_slice().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get f64 slice from tensor")
            })?;
            let output_slice = output_data.as_f64_slice_mut().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get mutable f64 slice from output")
            })?;

            for (i, &val) in input_data.iter().enumerate() {
                output_slice[i] = val.abs();
            }
        }
        _ => {
            return Err(MinitensorError::invalid_operation(
                "Absolute value only supported for floating point tensors",
            ));
        }
    }

    Ok(Tensor::new(
        Arc::new(output_data),
        tensor.shape().clone(),
        tensor.dtype(),
        tensor.device(),
        tensor.requires_grad(),
    ))
}

/// Compute the sign of each tensor element (-1.0, 0.0, or 1.0)
fn sign(tensor: &Tensor) -> Result<Tensor> {
    let mut output_data =
        TensorData::zeros_on_device(tensor.numel(), tensor.dtype(), tensor.device());

    match tensor.dtype() {
        DataType::Float32 => {
            let input_data = tensor.data().as_f32_slice().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get f32 slice from tensor")
            })?;
            let output_slice = output_data.as_f32_slice_mut().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get mutable f32 slice from output")
            })?;

            output_slice
                .par_chunks_mut(CHUNK)
                .zip(input_data.par_chunks(CHUNK))
                .for_each(|(out, inp)| unsafe {
                    let in_ptr = inp.as_ptr();
                    let out_ptr = out.as_mut_ptr();
                    for i in 0..out.len() {
                        let v = *in_ptr.add(i);
                        *out_ptr.add(i) = if v > 0.0 {
                            1.0
                        } else if v < 0.0 {
                            -1.0
                        } else {
                            0.0
                        };
                    }
                });
        }
        DataType::Float64 => {
            let input_data = tensor.data().as_f64_slice().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get f64 slice from tensor")
            })?;
            let output_slice = output_data.as_f64_slice_mut().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get mutable f64 slice from output")
            })?;

            output_slice
                .par_chunks_mut(CHUNK)
                .zip(input_data.par_chunks(CHUNK))
                .for_each(|(out, inp)| unsafe {
                    let in_ptr = inp.as_ptr();
                    let out_ptr = out.as_mut_ptr();
                    for i in 0..out.len() {
                        let v = *in_ptr.add(i);
                        *out_ptr.add(i) = if v > 0.0 {
                            1.0
                        } else if v < 0.0 {
                            -1.0
                        } else {
                            0.0
                        };
                    }
                });
        }
        _ => {
            return Err(MinitensorError::invalid_operation(
                "Sign operation only supported for floating point tensors",
            ));
        }
    }

    Ok(Tensor::new(
        Arc::new(output_data),
        tensor.shape().clone(),
        tensor.dtype(),
        tensor.device(),
        false,
    ))
}

/// Sum all elements in a tensor to produce a scalar
fn sum_all_elements(tensor: &Tensor) -> Result<Tensor> {
    let scalar_shape = Shape::new(vec![1]);
    let mut output_data = TensorData::zeros_on_device(1, tensor.dtype(), tensor.device());

    match tensor.dtype() {
        DataType::Float32 => {
            let input_data = tensor.data().as_f32_slice().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get f32 slice from tensor")
            })?;
            let output_slice = output_data.as_f32_slice_mut().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get mutable f32 slice from output")
            })?;

            let sum: f32 = input_data
                .par_chunks(CHUNK)
                .map(|chunk| unsafe {
                    let mut acc = 0f32;
                    let ptr = chunk.as_ptr();
                    for i in 0..chunk.len() {
                        acc += *ptr.add(i);
                    }
                    acc
                })
                .sum();
            output_slice[0] = sum;
        }
        DataType::Float64 => {
            let input_data = tensor.data().as_f64_slice().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get f64 slice from tensor")
            })?;
            let output_slice = output_data.as_f64_slice_mut().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get mutable f64 slice from output")
            })?;

            let sum: f64 = input_data
                .par_chunks(CHUNK)
                .map(|chunk| unsafe {
                    let mut acc = 0f64;
                    let ptr = chunk.as_ptr();
                    for i in 0..chunk.len() {
                        acc += *ptr.add(i);
                    }
                    acc
                })
                .sum();
            output_slice[0] = sum;
        }
        _ => {
            return Err(MinitensorError::invalid_operation(
                "Sum only supported for floating point tensors",
            ));
        }
    }

    Ok(Tensor::new(
        Arc::new(output_data),
        scalar_shape,
        tensor.dtype(),
        tensor.device(),
        tensor.requires_grad(),
    ))
}

/// Divide tensor by a scalar value
fn divide_by_scalar(tensor: &Tensor, scalar: f64) -> Result<Tensor> {
    let mut output_data =
        TensorData::zeros_on_device(tensor.numel(), tensor.dtype(), tensor.device());

    match tensor.dtype() {
        DataType::Float32 => {
            let input_data = tensor.data().as_f32_slice().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get f32 slice from tensor")
            })?;
            let output_slice = output_data.as_f32_slice_mut().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get mutable f32 slice from output")
            })?;

            let scalar_f32 = scalar as f32;
            output_slice
                .par_chunks_mut(CHUNK)
                .zip(input_data.par_chunks(CHUNK))
                .for_each(|(out, inp)| unsafe {
                    let in_ptr = inp.as_ptr();
                    let out_ptr = out.as_mut_ptr();
                    for i in 0..out.len() {
                        *out_ptr.add(i) = *in_ptr.add(i) / scalar_f32;
                    }
                });
        }
        DataType::Float64 => {
            let input_data = tensor.data().as_f64_slice().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get f64 slice from tensor")
            })?;
            let output_slice = output_data.as_f64_slice_mut().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get mutable f64 slice from output")
            })?;

            output_slice
                .par_chunks_mut(CHUNK)
                .zip(input_data.par_chunks(CHUNK))
                .for_each(|(out, inp)| unsafe {
                    let in_ptr = inp.as_ptr();
                    let out_ptr = out.as_mut_ptr();
                    for i in 0..out.len() {
                        *out_ptr.add(i) = *in_ptr.add(i) / scalar;
                    }
                });
        }
        _ => {
            return Err(MinitensorError::invalid_operation(
                "Division only supported for floating point tensors",
            ));
        }
    }

    Ok(Tensor::new(
        Arc::new(output_data),
        tensor.shape().clone(),
        tensor.dtype(),
        tensor.device(),
        tensor.requires_grad(),
    ))
}

/// Create a scalar tensor with the given value
fn create_scalar_tensor(
    value: f64,
    dtype: DataType,
    device: crate::device::Device,
) -> Result<Tensor> {
    let scalar_shape = Shape::new(vec![1]);
    let mut tensor_data = TensorData::zeros_on_device(1, dtype, device);

    match dtype {
        DataType::Float32 => {
            let slice = tensor_data.as_f32_slice_mut().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get mutable f32 slice")
            })?;
            slice[0] = value as f32;
        }
        DataType::Float64 => {
            let slice = tensor_data.as_f64_slice_mut().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get mutable f64 slice")
            })?;
            slice[0] = value;
        }
        _ => {
            return Err(MinitensorError::invalid_operation(
                "Scalar tensor creation only supported for floating point types",
            ));
        }
    }

    Ok(Tensor::new(
        Arc::new(tensor_data),
        scalar_shape,
        dtype,
        device,
        false,
    ))
}

/// Compute Huber loss element-wise
fn compute_huber_elementwise(
    abs_diff: &Tensor,
    diff: &Tensor,
    _delta_tensor: &Tensor,
    delta: f64,
) -> Result<Tensor> {
    let mut output_data =
        TensorData::zeros_on_device(abs_diff.numel(), abs_diff.dtype(), abs_diff.device());

    match abs_diff.dtype() {
        DataType::Float32 => {
            let abs_data = abs_diff.data().as_f32_slice().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get f32 slice from abs_diff")
            })?;
            let diff_data = diff.data().as_f32_slice().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get f32 slice from diff")
            })?;
            let output_slice = output_data.as_f32_slice_mut().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get mutable f32 slice from output")
            })?;

            let delta_f32 = delta as f32;
            output_slice
                .par_chunks_mut(CHUNK)
                .zip(abs_data.par_chunks(CHUNK).zip(diff_data.par_chunks(CHUNK)))
                .for_each(|(out, (abs_chunk, diff_chunk))| unsafe {
                    let abs_ptr = abs_chunk.as_ptr();
                    let diff_ptr = diff_chunk.as_ptr();
                    let out_ptr = out.as_mut_ptr();
                    for i in 0..out.len() {
                        let abs_val = *abs_ptr.add(i);
                        *out_ptr.add(i) = if abs_val <= delta_f32 {
                            0.5 * *diff_ptr.add(i) * *diff_ptr.add(i)
                        } else {
                            delta_f32 * (abs_val - 0.5 * delta_f32)
                        };
                    }
                });
        }
        DataType::Float64 => {
            let abs_data = abs_diff.data().as_f64_slice().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get f64 slice from abs_diff")
            })?;
            let diff_data = diff.data().as_f64_slice().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get f64 slice from diff")
            })?;
            let output_slice = output_data.as_f64_slice_mut().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get mutable f64 slice from output")
            })?;

            output_slice
                .par_chunks_mut(CHUNK)
                .zip(abs_data.par_chunks(CHUNK).zip(diff_data.par_chunks(CHUNK)))
                .for_each(|(out, (abs_chunk, diff_chunk))| unsafe {
                    let abs_ptr = abs_chunk.as_ptr();
                    let diff_ptr = diff_chunk.as_ptr();
                    let out_ptr = out.as_mut_ptr();
                    for i in 0..out.len() {
                        let abs_val = *abs_ptr.add(i);
                        *out_ptr.add(i) = if abs_val <= delta {
                            0.5 * *diff_ptr.add(i) * *diff_ptr.add(i)
                        } else {
                            delta * (abs_val - 0.5 * delta)
                        };
                    }
                });
        }
        _ => {
            return Err(MinitensorError::invalid_operation(
                "Huber loss only supported for floating point tensors",
            ));
        }
    }

    Ok(Tensor::new(
        Arc::new(output_data),
        abs_diff.shape().clone(),
        abs_diff.dtype(),
        abs_diff.device(),
        abs_diff.requires_grad(),
    ))
}

/// Apply softmax activation for numerical stability
fn softmax(tensor: &Tensor) -> Result<Tensor> {
    // Subtract the maximum value for numerical stability
    let dim = tensor.ndim() as isize - 1;
    let max_vals = max(tensor, Some(dim), true)?;
    let shifted = sub(tensor, &max_vals)?;
    let exp_tensor = exp(&shifted)?;
    let sum_exp = sum(&exp_tensor, Some(vec![dim]), true)?;
    divide_tensors(&exp_tensor, &sum_exp)
}

/// Compute exponential of tensor elements
fn exp(tensor: &Tensor) -> Result<Tensor> {
    let mut output_data =
        TensorData::zeros_on_device(tensor.numel(), tensor.dtype(), tensor.device());

    match tensor.dtype() {
        DataType::Float32 => {
            let input_data = tensor.data().as_f32_slice().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get f32 slice from tensor")
            })?;
            let output_slice = output_data.as_f32_slice_mut().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get mutable f32 slice from output")
            })?;

            for (i, &val) in input_data.iter().enumerate() {
                output_slice[i] = val.exp();
            }
        }
        DataType::Float64 => {
            let input_data = tensor.data().as_f64_slice().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get f64 slice from tensor")
            })?;
            let output_slice = output_data.as_f64_slice_mut().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get mutable f64 slice from output")
            })?;

            for (i, &val) in input_data.iter().enumerate() {
                output_slice[i] = val.exp();
            }
        }
        _ => {
            return Err(MinitensorError::invalid_operation(
                "Exponential only supported for floating point tensors",
            ));
        }
    }

    Ok(Tensor::new(
        Arc::new(output_data),
        tensor.shape().clone(),
        tensor.dtype(),
        tensor.device(),
        tensor.requires_grad(),
    ))
}

/// Compute natural logarithm of tensor elements
fn log(tensor: &Tensor) -> Result<Tensor> {
    let mut output_data =
        TensorData::zeros_on_device(tensor.numel(), tensor.dtype(), tensor.device());

    match tensor.dtype() {
        DataType::Float32 => {
            let input_data = tensor.data().as_f32_slice().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get f32 slice from tensor")
            })?;
            let output_slice = output_data.as_f32_slice_mut().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get mutable f32 slice from output")
            })?;

            for (i, &val) in input_data.iter().enumerate() {
                if val <= 0.0 {
                    output_slice[i] = f32::NEG_INFINITY;
                } else {
                    output_slice[i] = val.ln();
                }
            }
        }
        DataType::Float64 => {
            let input_data = tensor.data().as_f64_slice().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get f64 slice from tensor")
            })?;
            let output_slice = output_data.as_f64_slice_mut().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get mutable f64 slice from output")
            })?;

            for (i, &val) in input_data.iter().enumerate() {
                if val <= 0.0 {
                    output_slice[i] = f64::NEG_INFINITY;
                } else {
                    output_slice[i] = val.ln();
                }
            }
        }
        _ => {
            return Err(MinitensorError::invalid_operation(
                "Logarithm only supported for floating point tensors",
            ));
        }
    }

    Ok(Tensor::new(
        Arc::new(output_data),
        tensor.shape().clone(),
        tensor.dtype(),
        tensor.device(),
        tensor.requires_grad(),
    ))
}

/// Negate tensor elements
fn negate(tensor: &Tensor) -> Result<Tensor> {
    let mut output_data =
        TensorData::zeros_on_device(tensor.numel(), tensor.dtype(), tensor.device());

    match tensor.dtype() {
        DataType::Float32 => {
            let input_data = tensor.data().as_f32_slice().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get f32 slice from tensor")
            })?;
            let output_slice = output_data.as_f32_slice_mut().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get mutable f32 slice from output")
            })?;

            for (i, &val) in input_data.iter().enumerate() {
                output_slice[i] = -val;
            }
        }
        DataType::Float64 => {
            let input_data = tensor.data().as_f64_slice().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get f64 slice from tensor")
            })?;
            let output_slice = output_data.as_f64_slice_mut().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get mutable f64 slice from output")
            })?;

            for (i, &val) in input_data.iter().enumerate() {
                output_slice[i] = -val;
            }
        }
        _ => {
            return Err(MinitensorError::invalid_operation(
                "Negation only supported for floating point tensors",
            ));
        }
    }

    Ok(Tensor::new(
        Arc::new(output_data),
        tensor.shape().clone(),
        tensor.dtype(),
        tensor.device(),
        tensor.requires_grad(),
    ))
}

/// Compute negative log likelihood for classification
fn negative_log_likelihood(log_predictions: &Tensor, targets: &Tensor) -> Result<Tensor> {
    // Simplified implementation - multiply log predictions by targets and negate
    let likelihood = mul(log_predictions, targets)?;
    negate(&likelihood)
}

/// Raise tensor elements to a power
fn power(tensor: &Tensor, exponent: f64) -> Result<Tensor> {
    let mut output_data =
        TensorData::zeros_on_device(tensor.numel(), tensor.dtype(), tensor.device());

    match tensor.dtype() {
        DataType::Float32 => {
            let input_data = tensor.data().as_f32_slice().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get f32 slice from tensor")
            })?;
            let output_slice = output_data.as_f32_slice_mut().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get mutable f32 slice from output")
            })?;

            let exp_f32 = exponent as f32;
            for (i, &val) in input_data.iter().enumerate() {
                output_slice[i] = val.powf(exp_f32);
            }
        }
        DataType::Float64 => {
            let input_data = tensor.data().as_f64_slice().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get f64 slice from tensor")
            })?;
            let output_slice = output_data.as_f64_slice_mut().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get mutable f64 slice from output")
            })?;

            for (i, &val) in input_data.iter().enumerate() {
                output_slice[i] = val.powf(exponent);
            }
        }
        _ => {
            return Err(MinitensorError::invalid_operation(
                "Power operation only supported for floating point tensors",
            ));
        }
    }

    Ok(Tensor::new(
        Arc::new(output_data),
        tensor.shape().clone(),
        tensor.dtype(),
        tensor.device(),
        tensor.requires_grad(),
    ))
}

/// Sum along a specific dimension
/// Divide two tensors element-wise
fn divide_tensors(numerator: &Tensor, denominator: &Tensor) -> Result<Tensor> {
    // Use the existing division operation
    crate::operations::arithmetic::div(numerator, denominator)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::Device;

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
    fn test_mse_loss_mean() {
        let predictions = create_test_tensor_f32(vec![1.0, 2.0, 3.0], vec![3], false);
        let targets = create_test_tensor_f32(vec![1.5, 2.5, 2.5], vec![3], false);

        let loss = mse_loss(&predictions, &targets, "mean").unwrap();
        let loss_data = loss.data().as_f32_slice().unwrap();

        // Expected: ((1.0-1.5)² + (2.0-2.5)² + (3.0-2.5)²) / 3 = (0.25 + 0.25 + 0.25) / 3 = 0.25
        assert!((loss_data[0] - 0.25).abs() < 1e-6);
        assert_eq!(loss.shape().dims(), &[1]);
    }

    #[test]
    fn test_mse_loss_sum() {
        let predictions = create_test_tensor_f32(vec![1.0, 2.0], vec![2], false);
        let targets = create_test_tensor_f32(vec![2.0, 3.0], vec![2], false);

        let loss = mse_loss(&predictions, &targets, "sum").unwrap();
        let loss_data = loss.data().as_f32_slice().unwrap();

        // Expected: (1.0-2.0)² + (2.0-3.0)² = 1.0 + 1.0 = 2.0
        assert!((loss_data[0] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_mse_loss_none() {
        let predictions = create_test_tensor_f32(vec![1.0, 2.0], vec![2], false);
        let targets = create_test_tensor_f32(vec![2.0, 3.0], vec![2], false);

        let loss = mse_loss(&predictions, &targets, "none").unwrap();
        let loss_data = loss.data().as_f32_slice().unwrap();

        // Expected: [(1.0-2.0)², (2.0-3.0)²] = [1.0, 1.0]
        assert!((loss_data[0] - 1.0).abs() < 1e-6);
        assert!((loss_data[1] - 1.0).abs() < 1e-6);
        assert_eq!(loss.shape().dims(), &[2]);
    }

    #[test]
    fn test_mae_loss_mean() {
        let predictions = create_test_tensor_f32(vec![1.0, 2.0, 3.0], vec![3], false);
        let targets = create_test_tensor_f32(vec![1.5, 2.5, 2.0], vec![3], false);

        let loss = mae_loss(&predictions, &targets, "mean").unwrap();
        let loss_data = loss.data().as_f32_slice().unwrap();

        // Expected: (|1.0-1.5| + |2.0-2.5| + |3.0-2.0|) / 3 = (0.5 + 0.5 + 1.0) / 3 = 2.0/3 ≈ 0.667
        assert!((loss_data[0] - (2.0 / 3.0)).abs() < 1e-6);
    }

    #[test]
    fn test_huber_loss_quadratic_region() {
        let predictions = create_test_tensor_f32(vec![1.0, 2.0], vec![2], false);
        let targets = create_test_tensor_f32(vec![1.2, 2.3], vec![2], false);

        // Delta = 1.0, differences are 0.2 and 0.3, both <= 1.0, so quadratic
        let loss = huber_loss(&predictions, &targets, 1.0, "none").unwrap();
        let loss_data = loss.data().as_f32_slice().unwrap();

        // Expected: [0.5 * 0.2², 0.5 * 0.3²] = [0.02, 0.045]
        assert!((loss_data[0] - 0.02).abs() < 1e-6);
        assert!((loss_data[1] - 0.045).abs() < 1e-6);
    }

    #[test]
    fn test_huber_loss_linear_region() {
        let predictions = create_test_tensor_f32(vec![1.0, 2.0], vec![2], false);
        let targets = create_test_tensor_f32(vec![3.0, 0.0], vec![2], false);

        // Delta = 1.0, differences are 2.0 and 2.0, both > 1.0, so linear
        let loss = huber_loss(&predictions, &targets, 1.0, "none").unwrap();
        let loss_data = loss.data().as_f32_slice().unwrap();

        // Expected: [1.0 * (2.0 - 0.5 * 1.0), 1.0 * (2.0 - 0.5 * 1.0)] = [1.5, 1.5]
        assert!((loss_data[0] - 1.5).abs() < 1e-6);
        assert!((loss_data[1] - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_bce_loss_mean_and_backward() {
        let predictions = create_test_tensor_f32(vec![0.8, 0.2], vec![2], true);
        let targets = create_test_tensor_f32(vec![1.0, 0.0], vec![2], false);

        let loss = binary_cross_entropy_loss(&predictions, &targets, "mean").unwrap();
        let loss_val = loss.data().as_f32_slice().unwrap()[0];
        let expected = -((0.8f32).ln() + (0.8f32).ln()) / 2.0;
        assert!((loss_val - expected).abs() < 1e-6);

        let grads = crate::autograd::backward(&loss, None).unwrap();
        let grad = grads.get(&predictions.id()).unwrap();
        let grad_slice = grad.data().as_f32_slice().unwrap();
        let expected_grad = [-(1.0 / 0.8) / 2.0, (1.0 / 0.8) / 2.0];
        assert!((grad_slice[0] - expected_grad[0]).abs() < 1e-6);
        assert!((grad_slice[1] - expected_grad[1]).abs() < 1e-6);
    }

    #[test]
    fn test_kl_div_loss_mean_and_backward() {
        let predictions = create_test_tensor_f32(vec![0.4, 0.6], vec![2], true);
        let targets = create_test_tensor_f32(vec![0.5, 0.5], vec![2], false);

        let loss = kl_div_loss(&predictions, &targets, "mean").unwrap();
        let loss_val = loss.data().as_f32_slice().unwrap()[0];
        let expected = 0.5 * ((0.5f32.ln() - 0.4f32.ln()) + (0.5f32.ln() - 0.6f32.ln()));
        assert!((loss_val - expected).abs() < 1e-6);

        let grads = crate::autograd::backward(&loss, None).unwrap();
        let grad = grads.get(&predictions.id()).unwrap();
        let grad_slice = grad.data().as_f32_slice().unwrap();
        let expected_grad = [-(0.5 / 0.4) / 2.0, -(0.5 / 0.6) / 2.0];
        assert!((grad_slice[0] - expected_grad[0]).abs() < 1e-6);
        assert!((grad_slice[1] - expected_grad[1]).abs() < 1e-6);
    }

    #[test]
    fn test_loss_gradient_tracking() {
        let predictions = create_test_tensor_f32(vec![1.0, 2.0], vec![2], true);
        let targets = create_test_tensor_f32(vec![1.5, 2.5], vec![2], false);

        let loss = mse_loss(&predictions, &targets, "mean").unwrap();

        assert!(loss.requires_grad());
        assert!(loss.grad_fn().is_some());
    }

    #[test]
    fn test_loss_input_validation() {
        let predictions = create_test_tensor_f32(vec![1.0, 2.0], vec![2], false);
        let targets = create_test_tensor_f32(vec![1.5, 2.5, 3.5], vec![3], false);

        // Shape mismatch should fail
        let result = mse_loss(&predictions, &targets, "mean");
        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_reduction_mode() {
        let predictions = create_test_tensor_f32(vec![1.0, 2.0], vec![2], false);
        let targets = create_test_tensor_f32(vec![1.5, 2.5], vec![2], false);

        let result = mse_loss(&predictions, &targets, "invalid");
        assert!(result.is_err());
    }

    #[test]
    fn test_huber_loss_invalid_delta() {
        let predictions = create_test_tensor_f32(vec![1.0, 2.0], vec![2], false);
        let targets = create_test_tensor_f32(vec![1.5, 2.5], vec![2], false);

        let result = huber_loss(&predictions, &targets, -1.0, "mean");
        assert!(result.is_err());
    }
}
