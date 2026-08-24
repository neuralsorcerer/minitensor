// Copyright (c) Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

use super::*;
use crate::autograd::with_grad_fn;

use crate::{
    autograd::{
        BCELossBackward, BCEWithLogitsLossBackward, CrossEntropyLossBackward, FocalLossBackward,
        HuberLossBackward, KLDivLossBackward, MAELossBackward, MSELossBackward, NoGradGuard,
    },
    error::{MinitensorError, Result},
    ops::util::create_scalar_tensor,
    ops::{
        activation::{abs as activation_abs, exp, log_softmax, log1p},
        arithmetic::{add, div as divide, mul, sub},
        reduction::{mean, sum},
    },
    tensor::{DataType, Shape, Tensor, TensorData},
};
use std::sync::Arc;

pub(crate) const CHUNK: usize = 1024;

/// Whether a loss that carries its own analytical backward should attach one,
/// given the tensors it differentiates with respect to.
///
/// Every loss in this file computes its forward from ordinary tensor ops and
/// then replaces the resulting `grad_fn` with a single hand-written node. The
/// ops must therefore run with autograd recording *off*: once the final
/// `grad_fn` is replaced, the primitive nodes they would have recorded are no
/// longer reachable from the loss, so a backward pass never walks them and
/// `release_saved_subgraph` never releases them. They and every tensor they
/// saved would then sit in the graph until something cleared the whole thing.
///
/// With recording off the loss no longer propagates `requires_grad` on its
/// own, which is what this decides instead -- call it *before* opening the
/// guard, then mark the finished loss with `requires_grad_(true)`.
pub(crate) fn manual_backward_needed(inputs: &[&Tensor]) -> bool {
    crate::autograd::is_grad_enabled() && inputs.iter().any(|t| t.requires_grad())
}

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

    let needs_grad = manual_backward_needed(&[predictions, targets]);

    // Analytical backward below, so the primitive subtract/multiply/reduce
    // graph is not recorded as well -- see [`manual_backward_needed`].
    let (loss, diff_for_grad) = {
        let _guard = NoGradGuard::new();

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
        (loss, diff_for_grad)
    };

    // Set up gradient function if needed
    if needs_grad {
        let grad_fn = Arc::new(MSELossBackward {
            predictions_shape: predictions.shape().dims().to_vec(),
            targets_shape: targets.shape().dims().to_vec(),
            input_ids: [predictions.id(), targets.id()],
            input_requires_grad: [predictions.requires_grad(), targets.requires_grad()],
            reduction: reduction.to_string(),
            diff: diff_for_grad,
        });

        let mut loss_with_grad = loss.requires_grad_(true);
        loss_with_grad = with_grad_fn(loss_with_grad, grad_fn)?;

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

    // The forward is computed on detached data (the exact gradient is provided
    // by MAELossBackward from the stored sign), so gate on the inputs and
    // enable grad on the loss explicitly.
    let needs_grad = manual_backward_needed(&[predictions, targets]);

    let (loss, sign_for_grad) = {
        let _guard = NoGradGuard::new();

        // Compute absolute differences: |predictions - targets|
        // Also compute the sign for gradient computation
        let diff = sub(predictions, targets)?;
        let sign_diff = sign_tensor(&diff)?;
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
        (loss, sign_for_grad)
    };

    // Set up gradient function if needed
    if needs_grad {
        let grad_fn = Arc::new(MAELossBackward {
            predictions_shape: predictions.shape().dims().to_vec(),
            targets_shape: targets.shape().dims().to_vec(),
            input_ids: [predictions.id(), targets.id()],
            input_requires_grad: [predictions.requires_grad(), targets.requires_grad()],
            reduction: reduction.to_string(),
            sign: sign_for_grad,
        });

        let mut loss_with_grad = loss.requires_grad_(true);
        loss_with_grad = with_grad_fn(loss_with_grad, grad_fn)?;

        Ok(loss_with_grad)
    } else {
        Ok(loss)
    }
}
/// The per-sample loss `-log_predictions[i, targets[i]]`, one gather per row.
///
/// The dense path forms the same value as a class-axis reduction of
/// `one_hot * log_p`, which for class-index targets is a 4-million-element
/// multiply and reduction to read out 4096 numbers. It also needs four more
/// full-size passes first (a zeros, an `eq`, a `masked_fill` and a `negate`),
/// purely so that multiplying a zero target by an infinite log-probability
/// cannot produce NaN where there is no target mass. None of that arises here:
/// only the target class is ever touched.
fn nll_from_indices(log_predictions: &Tensor, targets: &Tensor) -> Result<Tensor> {
    let rows = targets.numel();
    let classes = log_predictions.size(log_predictions.ndim() - 1)?;
    let device = log_predictions.device();

    macro_rules! gather {
        ($pred_slice:ident, $ty:ty, $variant:ident, $idx_slice:ident, $idx_ty:ty) => {{
            let preds = log_predictions.data().$pred_slice().ok_or_else(|| {
                MinitensorError::internal_error("cross_entropy: bad prediction slice")
            })?;
            let idx = targets.data().$idx_slice().ok_or_else(|| {
                MinitensorError::internal_error("cross_entropy: bad target slice")
            })?;
            let mut out = Vec::with_capacity(rows);
            for (row, &raw) in idx.iter().enumerate().take(rows) {
                let class = checked_index_from_i64(raw as i64, classes)?;
                out.push(-preds[row * classes + class]);
            }
            TensorData::from_vec::<$ty>(out, DataType::$variant, device)
        }};
    }

    let data = match (log_predictions.dtype(), targets.dtype()) {
        (DataType::Float32, DataType::Int32) => {
            gather!(as_f32_slice, f32, Float32, as_i32_slice, i32)
        }
        (DataType::Float32, DataType::Int64) => {
            gather!(as_f32_slice, f32, Float32, as_i64_slice, i64)
        }
        (DataType::Float64, DataType::Int32) => {
            gather!(as_f64_slice, f64, Float64, as_i32_slice, i32)
        }
        (DataType::Float64, DataType::Int64) => {
            gather!(as_f64_slice, f64, Float64, as_i64_slice, i64)
        }
        _ => unreachable!("nll_from_indices is only reached for integer targets"),
    };

    Ok(Tensor::new(
        Arc::new(data),
        Shape::new(vec![rows]),
        log_predictions.dtype(),
        device,
        false,
    ))
}

/// Cross Entropy loss function for classification
///
/// Computes the cross entropy loss between predictions (logits) and targets:
/// CE = -Σ(targets * log_tensor(softmax(predictions)))
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

    // Class-index targets take a gather instead of the dense one-hot path; the
    // one-hot is still needed for the analytical backward, so it is built only
    // when a gradient will actually be taken.
    // Only integer indices take the fast path. Float-typed class indices are
    // accepted by the dense path too, and keeping them on it means this change
    // cannot alter any result it does not also speed up.
    let index_targets = targets.ndim() + 1 == predictions.ndim()
        && matches!(targets.dtype(), DataType::Int32 | DataType::Int64);
    let needs_grad = predictions.requires_grad() && crate::autograd::is_grad_enabled();
    // The backward takes indices directly, so the one-hot is only ever needed
    // for soft/dense targets now -- never materialized for class indices.
    let targets_one_hot = if index_targets {
        None
    } else {
        Some(prepare_classification_targets(predictions, targets)?)
    };

    // Cross-entropy owns an analytical backward node. Do not record the
    // primitive log-softmax/multiply/reduction graph as well: those nodes are
    // unreachable after the final grad_fn is replaced and would retain saved
    // tensors until the whole graph is cleared. Outputs still propagate
    // `requires_grad`, so the custom node below remains correctly enabled.
    let (loss, softmax_predictions) = {
        let _guard = NoGradGuard::new();
        let log_predictions = log_softmax(predictions, None)?;
        // Keep probabilities only for the analytical backward formula, and only
        // when there is one to take. The loss itself remains in log-space so
        // finite log-probabilities cannot be turned into infinity by
        // exponentiation underflow.
        let softmax_predictions = if needs_grad {
            Some(exp(&log_predictions.detach())?)
        } else {
            None
        };
        let per_sample = if index_targets {
            nll_from_indices(&log_predictions, targets)?
        } else {
            let one_hot = targets_one_hot
                .as_ref()
                .expect("dense targets prepared above");
            let nll = negative_log_likelihood(&log_predictions, one_hot)?;
            sum(&nll, Some(vec![1]), false)?
        };

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
        (loss, softmax_predictions)
    };

    // Set up gradient function if needed
    if needs_grad {
        let softmax_predictions =
            softmax_predictions.expect("probabilities computed whenever a gradient is needed");
        let targets_shape = targets_one_hot
            .as_ref()
            .map(|t| t.shape().dims().to_vec())
            .unwrap_or_else(|| predictions.shape().dims().to_vec());
        let grad_fn = Arc::new(CrossEntropyLossBackward {
            predictions_shape: predictions.shape().dims().to_vec(),
            targets_shape,
            input_ids: [predictions.id()],
            reduction: reduction.to_string(),
            softmax_predictions: softmax_predictions.detach(),
            targets: targets_one_hot.map(|t| t.detach()),
            target_indices: index_targets.then(|| targets.detach()),
        });

        let mut loss_with_grad = loss.requires_grad_(true);
        loss_with_grad = with_grad_fn(loss_with_grad, grad_fn)?;

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
/// Check the target against the input before anything reshapes either of them.
///
/// The forward below flattens both operands down to two dimensions and lets
/// the reshape fail if they disagree, which reported a mismatch between two
/// element *counts*: 4 predictions against 5 class indices came out as
/// "Shape mismatch: expected [5], got [4]", naming the target count as the
/// expectation and mentioning neither tensor's actual shape. With the class
/// axis anywhere but last the numbers stop resembling the input at all --
/// `(2, 3, 10)` against `(2, 4)` reported "expected [8], got [20]".
fn check_cross_entropy_target(input: &Tensor, target: &Tensor, dim: usize) -> Result<()> {
    let input_dims = input.shape().dims();
    let target_dims = target.shape().dims();
    let classes = input_dims[dim];

    // Dense targets carry a score per class and match the input exactly.
    if target.ndim() == input.ndim() {
        if target_dims == input_dims {
            return Ok(());
        }
        return Err(MinitensorError::invalid_argument_with_suggestion(
            format!(
                "cross_entropy: a target with the same rank as the input holds one \
                 score per class and must match it exactly, but the input is \
                 {input_dims:?} and the target is {target_dims:?}"
            ),
            format!(
                "Give a target of shape {input_dims:?}, or one rank lower holding class indices"
            ),
        ));
    }

    // Class-index targets carry one entry per prediction, so they are the
    // input's shape with the class axis taken out.
    let expected: Vec<usize> = input_dims
        .iter()
        .enumerate()
        .filter_map(|(i, &d)| (i != dim).then_some(d))
        .collect();
    if target.ndim() + 1 == input.ndim() && target_dims == expected.as_slice() {
        return Ok(());
    }
    Err(MinitensorError::invalid_argument_with_suggestion(
        format!(
            "cross_entropy: the input is {input_dims:?} with {classes} classes on \
             dim {dim}, so class-index targets must have shape {expected:?}, but the \
             target is {target_dims:?}"
        ),
        format!(
            "Pass one class index per prediction (shape {expected:?}), or a full score \
             per class (shape {input_dims:?}). `dim` selects the class axis and \
             defaults to 1"
        ),
    ))
}

pub fn cross_entropy(
    input: &Tensor,
    target: &Tensor,
    reduction: &str,
    dim: usize,
) -> Result<Tensor> {
    let ndim = input.ndim();
    if dim >= ndim {
        return Err(MinitensorError::dim_out_of_range_with_context(
            dim as isize,
            ndim,
            "cross_entropy: dim (the class axis)",
        ));
    }
    check_cross_entropy_target(input, target, dim)?;

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
/// BCE = -Σ(targets * log_tensor(predictions) + (1 - targets) * log_tensor(1 - predictions))
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

    let needs_grad = manual_backward_needed(&[predictions, targets]);

    // Analytical backward below, so the primitive graph is not recorded as
    // well -- see [`manual_backward_needed`].
    let loss = {
        let _guard = NoGradGuard::new();

        // Compute BCE: -[targets * log_tensor(predictions) + (1 - targets) * log_tensor(1 - predictions)]
        // The log outputs are clamped to >= -100 so a saturated prediction
        // (exactly 0 or 1) yields a finite loss instead of +inf.
        let log_predictions = log_tensor(predictions)?.clamp_min(-100.0)?;

        let ones = Tensor::ones(
            predictions.shape().clone(),
            predictions.dtype(),
            predictions.device(),
            false,
        );
        let one_minus_targets = sub(&ones, targets)?;
        let one_minus_predictions = sub(&ones, predictions)?;
        let log_one_minus_predictions = log_tensor(&one_minus_predictions)?.clamp_min(-100.0)?;

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
        match reduction {
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
        }
    };

    // Set up gradient function if needed
    if needs_grad {
        let grad_fn = Arc::new(BCELossBackward {
            predictions_shape: predictions.shape().dims().to_vec(),
            targets_shape: targets.shape().dims().to_vec(),
            input_ids: [predictions.id(), targets.id()],
            input_requires_grad: [predictions.requires_grad(), targets.requires_grad()],
            reduction: reduction.to_string(),
            predictions: predictions.clone().detach(),
            targets: targets.clone().detach(),
        });

        let mut loss_with_grad = loss.requires_grad_(true);
        loss_with_grad = with_grad_fn(loss_with_grad, grad_fn)?;

        Ok(loss_with_grad)
    } else {
        Ok(loss)
    }
}

/// Binary cross entropy computed directly from logits.
///
/// Mathematically this equals `binary_cross_entropy(sigmoid(logits), targets)`,
/// but the two are not interchangeable in floating point. Splitting the sigmoid
/// out lets it saturate — in f32 `sigmoid(-30)` rounds to `9.36e-14` and
/// `sigmoid(-90)` to exactly `0` — and the logarithm that follows then has to be
/// clamped to keep the loss finite. The clamp rescues the loss value but not its
/// derivative: at a logit of -30 against a target of 1, the gradient should be
/// -1 (the largest signal the loss can produce, from a confident and completely
/// wrong prediction), yet the split path returns about -0.09, and beyond -50 it
/// returns 0. Training stalls precisely on the examples it most needs to learn
/// from.
///
/// Fusing the sigmoid in keeps every intermediate in range, so the loss needs no
/// clamp and the gradient stays exact at any logit magnitude.
///
/// # Arguments
/// * `logits` - Unnormalized scores; **not** probabilities
/// * `targets` - Ground truth in [0, 1]
/// * `pos_weight` - Optional weight for the positive class, broadcast against
///   `targets`. A value above 1 trades precision for recall, which is the usual
///   reason to reach for it on an imbalanced dataset.
/// * `reduction` - How to reduce the loss ("mean", "sum", or "none")
pub fn binary_cross_entropy_with_logits_loss(
    logits: &Tensor,
    targets: &Tensor,
    pos_weight: Option<&Tensor>,
    reduction: &str,
) -> Result<Tensor> {
    validate_loss_inputs(logits, targets)?;

    // Broadcast pos_weight up front so both the forward kernel and the backward
    // see one flat, aligned buffer rather than re-deriving the mapping.
    let pos_weight = match pos_weight {
        Some(w) => {
            if w.device() != logits.device() {
                return Err(MinitensorError::device_mismatch(
                    format!("{:?}", w.device()),
                    format!("{:?}", logits.device()),
                ));
            }
            if w.dtype() != logits.dtype() {
                return Err(MinitensorError::type_mismatch(
                    format!("{:?}", w.dtype()),
                    format!("{:?}", logits.dtype()),
                ));
            }
            let target_dims: Vec<isize> =
                logits.shape().dims().iter().map(|&d| d as isize).collect();
            let expanded = w.detach().expand(target_dims).map_err(|_| {
                MinitensorError::shape_mismatch(
                    w.shape().dims().to_vec(),
                    logits.shape().dims().to_vec(),
                )
            })?;
            Some(expanded.contiguous()?)
        }
        None => None,
    };

    let logits_detached = logits.detach();
    let targets_detached = targets.detach();
    let values = compute_bce_with_logits_elementwise(
        &logits_detached,
        &targets_detached,
        pos_weight.as_ref(),
    )?;

    let loss = match reduction {
        "mean" => {
            let sum = sum_all_elements(&values)?;
            let n = values.numel() as f64;
            divide_by_scalar(&sum, n)?
        }
        "sum" => sum_all_elements(&values)?,
        "none" => values,
        _ => {
            return Err(MinitensorError::invalid_operation(format!(
                "Invalid reduction mode: {}. Must be 'mean', 'sum', or 'none'",
                reduction
            )));
        }
    };

    // The forward runs on detached data (the exact gradient comes from
    // BCEWithLogitsLossBackward), so gate on the inputs and turn grad back on
    // explicitly rather than letting it propagate.
    if (logits.requires_grad() || targets.requires_grad()) && crate::autograd::is_grad_enabled() {
        let grad_fn = Arc::new(BCEWithLogitsLossBackward {
            input_ids: [logits.id(), targets.id()],
            input_requires_grad: [logits.requires_grad(), targets.requires_grad()],
            reduction: reduction.to_string(),
            logits: logits_detached,
            targets: targets_detached,
            pos_weight,
        });

        let loss_with_grad = loss.requires_grad_(true);
        with_grad_fn(loss_with_grad, grad_fn)
    } else {
        Ok(loss)
    }
}

/// Kullback-Leibler divergence loss function
///
/// Computes KL divergence between target and prediction distributions:
/// KL(target || prediction) = Σ target * (log_tensor(target) - log_tensor(prediction))
/// Divisor for the `batchmean` reduction: the leading dimension, or 1 for a
/// single distribution stored as a 1-D tensor.
pub(crate) fn kl_div_batch_size(predictions: &Tensor) -> f64 {
    match predictions.shape().dims() {
        [batch, _rest @ ..] if !_rest.is_empty() => (*batch).max(1) as f64,
        _ => 1.0,
    }
}

pub fn kl_div_loss(predictions: &Tensor, targets: &Tensor, reduction: &str) -> Result<Tensor> {
    // Validate inputs
    validate_loss_inputs(predictions, targets)?;

    let needs_grad = manual_backward_needed(&[predictions, targets]);

    // Analytical backward below, so the primitive graph is not recorded as
    // well -- see [`manual_backward_needed`].
    let loss = {
        let _guard = NoGradGuard::new();

        // Compute elementwise targets * (log_tensor(targets) - log_tensor(predictions))
        let log_targets = log_tensor(targets)?;
        let log_predictions = log_tensor(predictions)?;
        let diff = sub(&log_targets, &log_predictions)?;
        let raw = mul(targets, &diff)?;

        // A zero in the target makes that product `0 * -inf`, which is NaN, and
        // one NaN term takes the whole reduction with it. A zero-probability
        // class is an ordinary thing to ask for -- a one-hot target is nothing
        // but zeros and a one -- so `kl_div` against one returned NaN rather
        // than a loss. The term is defined as zero there, which is also its
        // limit as the target goes to zero, so the elementwise result is masked
        // rather than the logarithm being nudged away from the singularity.
        // Masking the result rather than the log also covers a zero *and* a
        // zero prediction at the same position, where the log difference is
        // `-inf - -inf`.
        let zero = create_scalar_tensor(0.0, targets.dtype(), targets.device())?;
        let target_is_zero = targets.eq(&zero)?;
        let kld = crate::ops::selection::where_op(&target_is_zero, &zero, &raw)?;

        // Apply reduction.
        //
        // `mean` is the element-wise mean, as it is for every other loss here.
        // It used to divide by the batch dimension instead -- `batchmean` --
        // while [`KLDivLossBackward`] divided by the
        // element count, so forward and backward disagreed by a factor of
        // `numel / batch` (4x for a 3x4 input) whenever there was more than one
        // column. `batchmean` is now spelled out, and scales its gradient to match.
        match reduction {
            "mean" => {
                let sum = sum_all_elements(&kld)?;
                divide_by_scalar(&sum, predictions.numel().max(1) as f64)?
            }
            "batchmean" => {
                let sum = sum_all_elements(&kld)?;
                divide_by_scalar(&sum, kl_div_batch_size(predictions))?
            }
            "sum" => sum_all_elements(&kld)?,
            "none" => kld,
            _ => {
                return Err(MinitensorError::invalid_operation(format!(
                    "Invalid reduction mode: {}. Must be 'mean', 'batchmean', 'sum', or 'none'",
                    reduction
                )));
            }
        }
    };

    // Set up gradient function if needed
    if needs_grad {
        let grad_fn = Arc::new(KLDivLossBackward {
            predictions_shape: predictions.shape().dims().to_vec(),
            targets_shape: targets.shape().dims().to_vec(),
            input_ids: [predictions.id(), targets.id()],
            input_requires_grad: [predictions.requires_grad(), targets.requires_grad()],
            reduction: reduction.to_string(),
            predictions: predictions.clone().detach(),
            targets: targets.clone().detach(),
        });

        let mut loss_with_grad = loss.requires_grad_(true);
        loss_with_grad = with_grad_fn(loss_with_grad, grad_fn)?;

        Ok(loss_with_grad)
    } else {
        Ok(loss)
    }
}

/// Focal loss function for handling class imbalance
///
/// Computes the focal loss, which is a modified cross entropy loss:
/// FL = -α * (1 - p_t)^γ * log_tensor(p_t)
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

    let needs_grad = manual_backward_needed(&[predictions, targets]);

    // Analytical backward below, so the primitive graph is not recorded as
    // well -- see [`manual_backward_needed`].
    let (loss, softmax_for_grad) = {
        let _guard = NoGradGuard::new();

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
                // Average over samples, matching cross_entropy: only the true-class
                // term per sample is non-zero, so the denominator is the number of
                // samples (numel / num_classes), not the total element count.
                let num_classes = predictions.size(predictions.ndim() - 1)?.max(1);
                let n = (focal_values.numel() / num_classes) as f64;
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
        (loss, softmax_for_grad)
    };

    // Set up gradient function if needed
    if needs_grad {
        let grad_fn = Arc::new(FocalLossBackward {
            predictions_shape: predictions.shape().dims().to_vec(),
            targets_shape: targets_one_hot.shape().dims().to_vec(),
            input_ids: [predictions.id(), targets.id()],
            input_requires_grad: [predictions.requires_grad(), targets.requires_grad()],
            alpha,
            gamma,
            reduction: reduction.to_string(),
            softmax_predictions: softmax_for_grad,
            targets: targets_one_hot.clone().detach(),
        });

        let mut loss_with_grad = loss.requires_grad_(true);
        loss_with_grad = with_grad_fn(loss_with_grad, grad_fn)?;

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

    // `delta <= 0.0` alone is false for NaN, which then propagated through
    // every comparison and returned an all-NaN loss instead of an error.
    if !delta.is_finite() || delta <= 0.0 {
        return Err(MinitensorError::invalid_operation(
            "Delta must be positive and finite for Huber loss",
        ));
    }

    // The forward is computed on detached data (the exact gradient is provided
    // by HuberLossBackward from the stored diff), so gate on the inputs and
    // enable grad on the loss explicitly.
    let needs_grad = manual_backward_needed(&[predictions, targets]);

    let (loss, diff_for_grad) = {
        let _guard = NoGradGuard::new();

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
        (loss, diff_for_grad)
    };

    // Set up gradient function if needed
    if needs_grad {
        let grad_fn = Arc::new(HuberLossBackward {
            predictions_shape: predictions.shape().dims().to_vec(),
            targets_shape: targets.shape().dims().to_vec(),
            input_ids: [predictions.id(), targets.id()],
            input_requires_grad: [predictions.requires_grad(), targets.requires_grad()],
            delta,
            reduction: reduction.to_string(),
            diff: diff_for_grad,
        });

        let mut loss_with_grad = loss.requires_grad_(true);
        loss_with_grad = with_grad_fn(loss_with_grad, grad_fn)?;

        Ok(loss_with_grad)
    } else {
        Ok(loss)
    }
}

/// Smooth L1 loss.
///
/// `SmoothL1(x) = 0.5 * x² / beta` for `|x| < beta`, else `|x| - 0.5 * beta`.
///
/// Related to [`huber_loss`] by `huber(x, delta) == delta * smooth_l1(x, beta =
/// delta)`; the two coincide only at `1.0`, which is why this could previously
/// be a bare call to `huber_loss(.., 1.0, ..)`. The scaling goes through a
/// differentiable division so the gradient picks up the `1 / beta` factor
/// rather than being huber's.
///
/// # Arguments
/// * `predictions` - Model predictions tensor
/// * `targets` - Ground truth targets tensor
/// * `beta` - Threshold below which the loss is quadratic (must be positive)
/// * `reduction` - How to reduce the loss ("mean", "sum", or "none")
pub fn smooth_l1_loss(
    predictions: &Tensor,
    targets: &Tensor,
    beta: f64,
    reduction: &str,
) -> Result<Tensor> {
    if !beta.is_finite() || beta <= 0.0 {
        return Err(MinitensorError::invalid_argument(
            "smooth_l1_loss requires a positive, finite beta",
        ));
    }

    let loss = huber_loss(predictions, targets, beta, reduction)?;
    if beta == 1.0 {
        return Ok(loss);
    }
    let scale = create_scalar_tensor(beta, loss.dtype(), loss.device())?;
    divide(&loss, &scale)
}

/// Log-cosh loss for robust regression
///
/// Computes log_tensor(cosh(x)) where x = predictions - targets using a numerically
/// stable formulation: |x| + log1p(exp(-2|x|)) - log_tensor(2).
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

    // Class-index targets carry one entry per prediction row, so their shape
    // must be the predictions' shape without the class axis. Nothing checked
    // this: a `[4, 10]` prediction against 5 targets got as far as the
    // one-hot expansion and then failed a broadcast downstream, reporting
    // "Shape mismatch: expected [5], got [4]" -- which names the *target*
    // count as the expectation and never mentions either tensor's shape.
    if targets.ndim() + 1 == predictions.ndim() {
        let batch = &predictions.shape().dims()[..predictions.ndim() - 1];
        if targets.shape().dims() != batch {
            return Err(MinitensorError::invalid_argument_with_suggestion(
                format!(
                    "Classification loss: predictions have shape {:?} ({:?} rows over \
                     {} classes), but the class-index targets have shape {:?}",
                    predictions.shape().dims(),
                    batch,
                    predictions.shape().dims()[predictions.ndim() - 1],
                    targets.shape().dims()
                ),
                format!(
                    "Class-index targets need one entry per prediction row, so their \
                     shape must be {batch:?}"
                ),
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
        // Probability/one-hot targets participate in the same element-wise
        // kernels as the logits and must therefore use the logits dtype.
        // Normalizing here also keeps the saved target and analytical
        // backward gradient in the prediction dtype. Target gradients are not
        // part of the classification-loss contract, so detach the normalized
        // value explicitly instead of leaving an untracked cast that appears
        // differentiable.
        Ok(targets.astype(predictions.dtype())?.detach())
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
