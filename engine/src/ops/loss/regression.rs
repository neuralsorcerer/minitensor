// Copyright (c) Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

use super::*;
use crate::autograd::with_grad_fn;
use std::borrow::Cow;

use crate::ops::arithmetic::exact_sum::Slope;
use crate::ops::map::par_out_chunks_gated;
use crate::{
    autograd::{
        BCELossBackward, BCEWithLogitsLossBackward, CrossEntropyLossBackward, FocalLossBackward,
        KLDivLossBackward, NoGradGuard, RegressionLossBackward,
    },
    error::{MinitensorError, Result},
    ops::{
        activation::{
            CarriedSum, ShiftedExp, abs as activation_abs, infinite_slice_parts, less_offset,
            log_softmax, log1p_of_rest, softmax,
        },
        arithmetic::sub,
        reduction::{mean, sum},
    },
    tensor::{DataType, Shape, Tensor, TensorData},
};
use std::sync::Arc;

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
/// MSE = (1/n) * Σ(predictions - targets)², the float nearest the exact
/// value, as is the sum, and each unreduced square.
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
    let (predictions, targets) = difference_operands(predictions, targets)?;
    let (predictions, targets) = (predictions.as_ref(), targets.as_ref());

    let needs_grad = manual_backward_needed(&[predictions, targets]);

    // Analytical backward below, so the primitive subtract/multiply/reduce
    // graph is not recorded as well -- see [`manual_backward_needed`].
    let loss = {
        let _guard = NoGradGuard::new();

        // The loss is the float nearest the exact value: each square of an
        // exact difference, or their mean or sum, rounded once.
        match reduction {
            "mean" | "sum" => exact_difference_loss(predictions, targets, false, reduction)?,
            "none" => squared_differences(predictions, targets)?,
            _ => return Err(invalid_reduction(reduction, false)),
        }
    };
    if !needs_grad {
        return Ok(loss);
    }
    regression_backward(loss, (predictions, targets), Slope::Squared, reduction)
}

/// Mean Absolute Error (MAE) loss function
///
/// Computes the mean absolute error between predictions and targets:
/// MAE = (1/n) * Σ|predictions - targets|, the float nearest the exact
/// value, as is the sum.
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
    let (predictions, targets) = difference_operands(predictions, targets)?;
    let (predictions, targets) = (predictions.as_ref(), targets.as_ref());

    // The forward is computed on detached data (the gradient is provided by
    // the analytical backward from the operands), so gate on the inputs and
    // enable grad on the loss explicitly.
    let needs_grad = manual_backward_needed(&[predictions, targets]);

    let loss = {
        let _guard = NoGradGuard::new();

        // The mean or sum is of the exact magnitudes, rounded once, and each
        // unreduced magnitude that of a difference rounded once already.
        match reduction {
            "mean" | "sum" => exact_difference_loss(predictions, targets, true, reduction)?,
            "none" => activation_abs(&sub(predictions, targets)?.detach())?,
            _ => return Err(invalid_reduction(reduction, false)),
        }
    };
    if !needs_grad {
        return Ok(loss);
    }
    regression_backward(loss, (predictions, targets), Slope::Absolute, reduction)
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
    let grad_enabled = crate::autograd::is_grad_enabled();
    let predictions_need_grad = predictions.requires_grad() && grad_enabled;
    // Dense targets are scores, and a score can be learned -- a soft label
    // produced by another model, say. The loss is linear in them, so their
    // gradient is cheap; it is taken whenever they ask for one, since leaving
    // it out would read to an optimizer as "this target was never used".
    let targets_need_grad = !index_targets && targets.requires_grad() && grad_enabled;
    let needs_grad = predictions_need_grad || targets_need_grad;
    // The backward takes indices directly, so the one-hot is only ever needed
    // for soft/dense targets now -- never materialized for class indices.
    let targets_one_hot = if index_targets {
        None
    } else {
        Some(prepare_classification_targets(predictions, targets)?)
    };
    // The tensor the target gradient is recorded against: the targets in the
    // logits' dtype, through a recorded cast when they arrived in another.
    let tracked_targets = if targets_need_grad {
        Some(targets.astype(predictions.dtype())?)
    } else {
        None
    };

    // Cross-entropy owns an analytical backward node. Do not record the
    // primitive log-softmax/multiply/reduction graph as well: those nodes are
    // unreachable after the final grad_fn is replaced and would retain saved
    // tensors until the whole graph is cleared. Outputs still propagate
    // `requires_grad`, so the custom node below remains correctly enabled.
    let (loss, softmax_predictions, log_predictions) = {
        let _guard = NoGradGuard::new();
        let log_predictions = log_softmax(predictions, None)?;
        // Keep probabilities only for the analytical backward formula, and only
        // when there is one to take. The loss itself remains in log-space so
        // finite log-probabilities cannot be turned into infinity by
        // exponentiation underflow.
        // From `softmax` itself rather than as `exp` of `log_softmax`, which
        // turned the latter's rounding into `|log p|` ulps of `p`: some thirty
        // at `p = e^-40`.
        let softmax_predictions = if predictions_need_grad {
            Some(softmax(predictions, None)?)
        } else {
            None
        };
        let saved_log_predictions = targets_need_grad.then(|| log_predictions.detach());
        let per_sample = if index_targets {
            nll_from_indices(&log_predictions, targets)?
        } else {
            let one_hot = targets_one_hot
                .as_ref()
                .expect("dense targets prepared above");
            let nll = negative_log_likelihood(&log_predictions, one_hot)?;
            sum(&nll, Some(vec![1]), false)?
        };

        // Averaged over samples, not elements: only the true-class term per
        // sample is non-zero.
        let batch = per_sample.shape().dims().first().copied().unwrap_or(1) as f64;
        let loss = reduce_detached(per_sample, reduction, batch, None)?;
        (loss, softmax_predictions, saved_log_predictions)
    };

    // Set up gradient function if needed
    if needs_grad {
        let targets_shape = targets_one_hot
            .as_ref()
            .map(|t| t.shape().dims().to_vec())
            .unwrap_or_else(|| predictions.shape().dims().to_vec());
        let predictions_id = predictions_need_grad.then(|| predictions.id());
        let targets_id = tracked_targets.as_ref().map(|t| t.id());
        let grad_fn = Arc::new(CrossEntropyLossBackward {
            predictions_shape: predictions.shape().dims().to_vec(),
            targets_shape,
            input_ids: predictions_id.into_iter().chain(targets_id).collect(),
            predictions_id,
            targets_id,
            reduction: reduction.to_string(),
            softmax_predictions,
            log_predictions,
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

/// Refuse a value outside `[0, 1]` where binary cross entropy reads a
/// probability. A NaN is let through: it makes the loss NaN, which is how a
/// diverged model shows up in every loss here.
///
/// Out of range, the loss has no meaning and does not say so: a prediction of
/// 1.5 for a target of 1 scored -0.405, a loss below zero, so an output layer
/// missing its sigmoid trained against an objective that rewarded leaving the
/// unit interval.
fn ensure_probabilities(tensor: &Tensor, what: &str) -> Result<()> {
    let tensor = tensor.contiguous()?;
    // NaN is let through on purpose: it is no claim about the unit interval,
    // and it reaches the loss as NaN, as a NaN does everywhere else.
    let inside = match tensor.dtype() {
        DataType::Float32 => tensor.data().as_f32_slice().is_none_or(|values| {
            values
                .iter()
                .all(|&v| v.is_nan() || (0.0..=1.0).contains(&v))
        }),
        DataType::Float64 => tensor.data().as_f64_slice().is_none_or(|values| {
            values
                .iter()
                .all(|&v| v.is_nan() || (0.0..=1.0).contains(&v))
        }),
        _ => true,
    };
    if inside {
        Ok(())
    } else {
        Err(MinitensorError::invalid_argument(format!(
            "binary_cross_entropy requires every {what} to lie in [0, 1]; for \
             unbounded scores use binary_cross_entropy_with_logits"
        )))
    }
}

/// Reject a negative entry in a tensor of probabilities for `kl_div`.
///
/// A probability is never below zero, and the logarithm the divergence takes
/// has no value there. It used to be read as a probability of exactly zero --
/// the log of a non-positive value was taken to be `-inf` -- so a negative
/// prediction reported an infinite divergence: a plausible-looking answer to
/// a question that had none. NaN is let through, and reaches the loss as NaN.
fn ensure_non_negative_probabilities(tensor: &Tensor, what: &str) -> Result<()> {
    let tensor = tensor.contiguous()?;
    let negative = match tensor.dtype() {
        DataType::Float32 => tensor
            .data()
            .as_f32_slice()
            .and_then(|values| values.iter().copied().find(|&v| v < 0.0))
            .map(f64::from),
        DataType::Float64 => tensor
            .data()
            .as_f64_slice()
            .and_then(|values| values.iter().copied().find(|&v| v < 0.0)),
        _ => None,
    };
    match negative {
        Some(value) => Err(MinitensorError::invalid_argument(format!(
            "kl_div takes probabilities, and the {what} holds {value}; a \
             probability is never negative"
        ))),
        None => Ok(()),
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
    let (predictions, targets) = loss_operands(predictions, targets)?;
    let (predictions, targets) = (predictions.as_ref(), targets.as_ref());
    ensure_probabilities(predictions, "prediction")?;
    ensure_probabilities(targets, "target")?;

    let needs_grad = manual_backward_needed(&[predictions, targets]);

    // Analytical backward below, so the primitive graph is not recorded as
    // well -- see [`manual_backward_needed`].
    let loss = {
        let _guard = NoGradGuard::new();

        // BCE: -[targets * log(predictions) + (1 - targets) * log(1 - predictions)],
        // in float64 and narrowed once. The logs are clamped to >= -100 so a
        // saturated prediction (exactly 0 or 1) yields a finite loss instead
        // of +inf, and `log(1 - p)` is `log1p(-p)`: `1 - p` rounded first,
        // and for a small prediction against a target of 0 -- a loss near
        // `p` -- the rounding was most of the answer.
        let negative_bce = wide_terms(predictions, targets, |p, t| {
            let floor = |log: f64| if log < -100.0 { -100.0 } else { log };
            let (log_p, log_q) = (floor(p.ln()), floor((-p).ln_1p()));
            0.0 - (t * log_p + (1.0 - t) * log_q)
        })?;

        // Apply reduction
        let n = negative_bce.numel() as f64;
        reduce_detached(negative_bce, reduction, n, None)?
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

/// A loss's elementwise terms: `term(a, b)` of each pair of `a` and `b`, one
/// float dtype and shape, taken in float64 and narrowed once to that dtype.
fn wide_terms(
    a: &Tensor,
    b: &Tensor,
    term: impl Fn(f64, f64) -> f64 + Send + Sync,
) -> Result<Tensor> {
    let (a, b) = (a.contiguous()?, b.contiguous()?);
    let missing = || MinitensorError::internal_error("loss operands are not the loss's dtype");
    let data = match a.dtype() {
        DataType::Float32 => {
            let x = a.data().as_f32_slice().ok_or_else(missing)?;
            let y = b.data().as_f32_slice().ok_or_else(missing)?;
            let values = crate::ops::map::binary_map(x, y, |x: f32, y: f32| {
                term(f64::from(x), f64::from(y)) as f32
            });
            TensorData::from_vec(values, DataType::Float32, a.device())
        }
        DataType::Float64 => {
            let x = a.data().as_f64_slice().ok_or_else(missing)?;
            let y = b.data().as_f64_slice().ok_or_else(missing)?;
            let values = crate::ops::map::binary_map(x, y, &term);
            TensorData::from_vec(values, DataType::Float64, a.device())
        }
        other => {
            return Err(MinitensorError::invalid_operation(format!(
                "this loss needs floating-point inputs, got {other:?}"
            )));
        }
    };
    Ok(Tensor::new(
        Arc::new(data),
        a.shape().clone(),
        a.dtype(),
        a.device(),
        false,
    ))
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
    let (logits, targets) = loss_operands(logits, targets)?;
    let (logits, targets) = (logits.as_ref(), targets.as_ref());

    // Broadcast pos_weight up front so both the forward kernel and the backward
    // see one flat, aligned buffer rather than re-deriving the mapping.
    let pos_weight = match pos_weight {
        Some(w) => {
            // `logits` is the reference the weight has to match, so it
            // is the one named as expected.
            if w.device() != logits.device() {
                return Err(MinitensorError::device_mismatch(
                    format!("{:?}", logits.device()),
                    format!("{:?}", w.device()),
                ));
            }
            // A coefficient of the loss, as `delta` and `beta` are elsewhere,
            // so it is read at the dtype the loss is computed in rather than
            // refused for having another.
            let w = &w.detach().astype(logits.dtype())?;
            let target_dims: Vec<isize> =
                logits.shape().dims().iter().map(|&d| d as isize).collect();
            let expanded = w.expand(target_dims).map_err(|_| {
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

    let n = values.numel() as f64;
    let loss = reduce_detached(values, reduction, n, None)?;

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

/// `t log(t / p)`, a term of `kl_div` for a target `t` and a prediction `p`,
/// neither negative; zero where the target is.
///
/// The logarithm of the ratio, not the difference of two: where the two are
/// near each other, as a good prediction's are, `log t - log p` cancelled
/// down to the roundings of the two logarithms, and terms were tens of ulps
/// out. Within a factor of two of each other the difference `t - p` is exact
/// and the term is `log1p((t - p) / p)`; further apart it is the logarithm of
/// the rounded ratio, which is at least `log 2` and so keeps the ratio's
/// relative error; and where the ratio leaves the normal floats, the
/// logarithms are far enough apart for their difference.
fn kl_term(t: f64, p: f64) -> f64 {
    if t == 0.0 {
        return 0.0;
    }
    let ratio = t / p;
    let log_ratio = if (0.5..=2.0).contains(&ratio) {
        ((t - p) / p).ln_1p()
    } else if ratio.is_normal() {
        ratio.ln()
    } else {
        t.ln() - p.ln()
    };
    t * log_ratio
}

/// `kl_div`'s gradients, in its predictions where `wanted[0]` and in its
/// targets where `wanted[1]`, the reduction having divided by `divisor`:
/// `-t g / (p n)` and `(1 + log(t / p)) g / n`, `g` the incoming gradient,
/// one value or one a term. The first is the float nearest its exact value
/// where the incoming gradient shares the operands' dtype, and the second
/// wherever a bound settles which that is; see
/// `arithmetic::exact_sum::kl_target_slopes`. A zero target's term is zero
/// whatever the prediction, and so is its gradient in the prediction, signed
/// as `-t g` signs it: it was `-(0 / 0)` against a zero prediction, NaN.
pub(crate) fn kl_gradients(
    (predictions, targets): (&Tensor, &Tensor),
    grad_output: &Tensor,
    divisor: f64,
    wanted: [bool; 2],
) -> Result<[Option<Tensor>; 2]> {
    use crate::ops::arithmetic::exact_sum::{Element, Output, divisor_slopes, kl_target_slopes};
    let (p, t) = (predictions.contiguous()?, targets.contiguous()?);
    let upstream = grad_output.contiguous()?;
    let (dtype, device, n) = (p.dtype(), p.device(), p.numel());
    if upstream.numel() != 1 && upstream.numel() != n {
        return Err(MinitensorError::shape_mismatch(
            p.shape().dims().to_vec(),
            upstream.shape().dims().to_vec(),
        ));
    }
    let same = upstream.dtype() == dtype;
    // The incoming gradient's values in float64, where the target's gradient
    // or another dtype needs them.
    let flows = if wanted[1] || !same {
        widened(&upstream.astype(DataType::Float64)?)?
    } else {
        Vec::new()
    };
    /// The two gradients' values for operands of one float dtype.
    fn slopes<T: Element + Output>(
        (p, t): (&[T], &[T]),
        same: Option<&[T]>,
        (flows, divisor, wanted): (&[f64], f64, [bool; 2]),
    ) -> [Option<Vec<T>>; 2] {
        let flow = |i: usize| match same {
            Some(g) => g[if g.len() == 1 { 0 } else { i }].wide(),
            None => flows[if flows.len() == 1 { 0 } else { i }],
        };
        let by_prediction = wanted[0].then(|| {
            let mut values = match same {
                Some(g) => {
                    let g: Cow<'_, [T]> = if g.len() == 1 {
                        Cow::Owned(vec![g[0]; t.len()])
                    } else {
                        Cow::Borrowed(g)
                    };
                    divisor_slopes((t, &g), (p, Some(divisor)), 1)
                }
                // An incoming gradient of another dtype, as its values.
                None => (0..t.len())
                    .map(|i| T::narrow(-(t[i].wide() * flow(i)) / (p[i].wide() * divisor)))
                    .collect(),
            };
            for (i, value) in values.iter_mut().enumerate() {
                if t[i].wide() == 0.0 {
                    *value = T::narrow(-(t[i].wide() * flow(i)));
                }
            }
            values
        });
        let by_target = wanted[1].then(|| kl_target_slopes((t, p), flows, divisor));
        [by_prediction, by_target]
    }
    let missing = || MinitensorError::internal_error("kl_div backward: a float slice");
    let values = match dtype {
        DataType::Float32 => {
            let (x, y) = (p.data().as_f32_slice(), t.data().as_f32_slice());
            let (x, y) = (x.ok_or_else(missing)?, y.ok_or_else(missing)?);
            let g = same.then(|| upstream.data().as_f32_slice()).flatten();
            slopes((x, y), g, (&flows, divisor, wanted))
                .map(|values| values.map(|values| TensorData::from_vec(values, dtype, device)))
        }
        DataType::Float64 => {
            let (x, y) = (p.data().as_f64_slice(), t.data().as_f64_slice());
            let (x, y) = (x.ok_or_else(missing)?, y.ok_or_else(missing)?);
            let g = same.then(|| upstream.data().as_f64_slice()).flatten();
            slopes((x, y), g, (&flows, divisor, wanted))
                .map(|values| values.map(|values| TensorData::from_vec(values, dtype, device)))
        }
        other => {
            return Err(MinitensorError::invalid_operation(format!(
                "kl_div needs floating-point inputs, got {other:?}"
            )));
        }
    };
    let shape = p.shape().clone();
    Ok(values.map(|data| {
        data.map(|data| Tensor::new(Arc::new(data), shape.clone(), dtype, device, false))
    }))
}

pub fn kl_div_loss(predictions: &Tensor, targets: &Tensor, reduction: &str) -> Result<Tensor> {
    // Validate inputs
    let (predictions, targets) = loss_operands(predictions, targets)?;
    let (predictions, targets) = (predictions.as_ref(), targets.as_ref());
    ensure_non_negative_probabilities(predictions, "input")?;
    ensure_non_negative_probabilities(targets, "target")?;

    let needs_grad = manual_backward_needed(&[predictions, targets]);

    // Analytical backward below, so the primitive graph is not recorded as
    // well -- see [`manual_backward_needed`].
    let loss = {
        let _guard = NoGradGuard::new();

        // Each term `target * log(target / prediction)`, in float64 and
        // narrowed once; see [`kl_term`].
        //
        // A zero in the target makes that product `0 * -inf`, which is NaN, and
        // one NaN term takes the whole reduction with it. A zero-probability
        // class is an ordinary thing to ask for -- a one-hot target is nothing
        // but zeros and a one -- so `kl_div` against one returned NaN rather
        // than a loss. The term is defined as zero there, which is also its
        // limit as the target goes to zero, so the result is taken as zero
        // rather than the logarithm being nudged away from the singularity --
        // a zero *and* a zero prediction at the same position included.
        let kld = wide_terms(targets, predictions, kl_term)?;

        // Apply reduction.
        //
        // `mean` is the element-wise mean, as it is for every other loss here.
        // It used to divide by the batch dimension instead -- `batchmean` --
        // while [`KLDivLossBackward`] divided by the
        // element count, so forward and backward disagreed by a factor of
        // `numel / batch` (4x for a 3x4 input) whenever there was more than one
        // column. `batchmean` is now spelled out, and scales its gradient to match.
        reduce_detached(
            kld,
            reduction,
            predictions.numel().max(1) as f64,
            Some(kl_div_batch_size(predictions)),
        )?
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
/// FL = -α * (1 - p_t)^γ * log(p_t)
/// where p_t is the predicted probability for the true class, summed over the
/// classes against a distribution target. Taken from the logits in float64 and
/// rounded once per sample; see [`focal_pass`].
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

    check_focal_parameters(alpha, gamma)?;

    let needs_grad = manual_backward_needed(&[predictions, targets]);
    // A distribution target that asks for a gradient gets one, recorded
    // against its cast to the logits' dtype as `cross_entropy_loss` records it.
    let tracked_targets =
        if needs_grad && targets.requires_grad() && targets.ndim() == predictions.ndim() {
            Some(targets.astype(predictions.dtype())?)
        } else {
            None
        };

    // Analytical backward below, so the primitive graph is not recorded as
    // well -- see [`manual_backward_needed`].
    let loss = {
        let _guard = NoGradGuard::new();
        // One value per sample, the sum over its classes, as for
        // cross_entropy: `"none"` returns them, and `"mean"` averages over
        // samples. Returning the per-class terms instead made
        // `focal_loss(.., "none").mean()` a `1 / num_classes` of
        // `focal_loss(.., "mean")`.
        let per_sample = focal_values(
            predictions,
            &targets_one_hot,
            (alpha, gamma),
            &[1.0],
            Focal::Loss,
        )?;
        let n = per_sample.numel() as f64;
        reduce_detached(per_sample, reduction, n, None)?
    };

    // Set up gradient function if needed
    if needs_grad {
        let grad_fn = Arc::new(FocalLossBackward {
            predictions_shape: predictions.shape().dims().to_vec(),
            targets_shape: targets_one_hot.shape().dims().to_vec(),
            input_ids: [
                predictions.id(),
                tracked_targets.as_ref().map_or(targets.id(), |t| t.id()),
            ],
            input_requires_grad: [predictions.requires_grad(), tracked_targets.is_some()],
            alpha,
            gamma,
            reduction: reduction.to_string(),
            logits: predictions.detach(),
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
/// Each term is the float nearest its exact value for the exact difference,
/// and so are the sum and the mean; `delta` is read at the operands' width.
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
    check_huber_delta(delta)?;
    huber_shaped(predictions, targets, (delta, false), reduction)
}

/// Smooth L1 loss.
///
/// `SmoothL1(x) = 0.5 * x² / beta` for `|x| < beta`, else `|x| - 0.5 * beta`.
///
/// Related to [`huber_loss`] by `huber(x, delta) == delta * smooth_l1(x, beta =
/// delta)`; the two coincide only at `1.0`. Each term is the float nearest
/// its exact value -- the Huber term over `beta` without a rounding between
/// -- and so are the sum and the mean.
///
/// # Arguments
/// * `predictions` - Model predictions tensor
/// * `targets` - Ground truth targets tensor
/// * `beta` - Threshold below which the loss is quadratic (finite, at least
///   zero; at zero there is no quadratic region and this is `mae_loss`)
/// * `reduction` - How to reduce the loss ("mean", "sum", or "none")
pub fn smooth_l1_loss(
    predictions: &Tensor,
    targets: &Tensor,
    beta: f64,
    reduction: &str,
) -> Result<Tensor> {
    check_smooth_l1_beta(beta)?;
    // The quadratic region shrinks to nothing as `beta` goes to zero and the
    // loss becomes `|x - y|` exactly: the limit, taken directly rather than
    // dividing by zero -- as it is for a `beta` the operands' width holds
    // only as a zero.
    let common = crate::ops::binary::promote_division_dtype(predictions.dtype(), targets.dtype());
    let narrow = common == DataType::Float32 && beta as f32 == 0.0;
    if beta == 0.0 || narrow {
        return mae_loss(predictions, targets, reduction);
    }
    huber_shaped(predictions, targets, (beta, true), reduction)
}

/// A Huber-shaped loss: each term the Huber term with `delta`, over `delta`
/// where `smooth`, the float nearest its exact value, as are their sum and
/// mean, and so is each gradient.
fn huber_shaped(
    predictions: &Tensor,
    targets: &Tensor,
    (delta, smooth): (f64, bool),
    reduction: &str,
) -> Result<Tensor> {
    let (predictions, targets) = difference_operands(predictions, targets)?;
    let (predictions, targets) = (predictions.as_ref(), targets.as_ref());

    // The forward is computed on detached data, and the gradient provided by
    // the analytical backward; see [`manual_backward_needed`].
    let needs_grad = manual_backward_needed(&[predictions, targets]);
    let loss = {
        let _guard = NoGradGuard::new();
        huber_values(predictions, targets, (delta, smooth), reduction)?
    };
    if !needs_grad {
        return Ok(loss);
    }
    let slope = if smooth {
        Slope::Smooth(delta)
    } else {
        Slope::Huber(delta)
    };
    regression_backward(loss, (predictions, targets), slope, reduction)
}

/// `loss` with the backward of a regression loss of `slope` attached, which
/// keeps the operands rather than their rounded difference.
fn regression_backward(
    loss: Tensor,
    (predictions, targets): (&Tensor, &Tensor),
    slope: Slope,
    reduction: &str,
) -> Result<Tensor> {
    let grad_fn = Arc::new(RegressionLossBackward {
        input_ids: [predictions.id(), targets.id()],
        input_requires_grad: [predictions.requires_grad(), targets.requires_grad()],
        slope,
        mean: reduction == "mean",
        predictions: predictions.detach(),
        targets: targets.detach(),
    });
    with_grad_fn(loss.requires_grad_(true), grad_fn)
}

/// A regression loss's gradients, in its predictions where `wanted[0]` and in
/// its targets where `wanted[1]`: each the float nearest `s(p - t) g / n` --
/// `s` the term's slope, `g` the incoming gradient, one value or one a term,
/// and `n` the count for a `mean` -- and its negative; for log-cosh's
/// `tanh`, wherever a bound settles which that is. A float32 loss reads its
/// limit at float32, as its forward does.
pub(crate) fn regression_gradients(
    (predictions, targets): (&Tensor, &Tensor),
    grad_output: &Tensor,
    slope: Slope,
    mean: bool,
    wanted: [bool; 2],
) -> Result<[Option<Tensor>; 2]> {
    use crate::ops::arithmetic::exact_sum::regression_slopes;
    let (p, t) = (predictions.contiguous()?, targets.contiguous()?);
    let upstream = grad_output.contiguous()?;
    let (dtype, device, n) = (p.dtype(), p.device(), p.numel());
    if upstream.numel() != 1 && upstream.numel() != n {
        return Err(MinitensorError::shape_mismatch(
            p.shape().dims().to_vec(),
            upstream.shape().dims().to_vec(),
        ));
    }
    let count = if mean { n as f64 } else { 1.0 };
    // A float32 loss reads its limit at float32.
    let slope = match (slope, dtype) {
        (Slope::Huber(delta), DataType::Float32) => Slope::Huber(f64::from(delta as f32)),
        (Slope::Smooth(beta), DataType::Float32) => Slope::Smooth(f64::from(beta as f32)),
        (other, _) => other,
    };
    let missing = || MinitensorError::internal_error("loss backward: a float slice");
    let floats = |tensor: &Tensor| tensor.dtype() == dtype;
    let data = match dtype {
        // The operands as they are, where they share the predictions' dtype.
        DataType::Float32 if floats(&t) && floats(&upstream) => {
            let (x, y) = (p.data().as_f32_slice(), t.data().as_f32_slice());
            let g = upstream.data().as_f32_slice().ok_or_else(missing)?;
            let (x, y) = (x.ok_or_else(missing)?, y.ok_or_else(missing)?);
            let out: Vec<f32> = regression_slopes(x, (y, None), g, slope, count);
            TensorData::from_vec(out, dtype, device)
        }
        DataType::Float64 if floats(&t) && floats(&upstream) => {
            let (x, y) = (p.data().as_f64_slice(), t.data().as_f64_slice());
            let g = upstream.data().as_f64_slice().ok_or_else(missing)?;
            let (x, y) = (x.ok_or_else(missing)?, y.ok_or_else(missing)?);
            let out: Vec<f64> = regression_slopes(x, (y, None), g, slope, count);
            TensorData::from_vec(out, dtype, device)
        }
        // Otherwise in float64: an integer target as its parts, and an
        // incoming gradient of another dtype as its values.
        _ => {
            let x = widened(&p)?;
            let (highs, lows) = match t.dtype() {
                DataType::Int32 | DataType::Int64 => integer_parts(&t)?,
                _ => (widened(&t)?, None),
            };
            let g = widened(&upstream.astype(DataType::Float64)?)?;
            let targets = (&highs[..], lows.as_deref());
            if dtype == DataType::Float32 {
                let out: Vec<f32> = regression_slopes(&x, targets, &g, slope, count);
                TensorData::from_vec(out, dtype, device)
            } else {
                let out: Vec<f64> = regression_slopes(&x, targets, &g, slope, count);
                TensorData::from_vec(out, dtype, device)
            }
        }
    };
    let grad = Tensor::new(Arc::new(data), p.shape().clone(), dtype, device, false);
    let by_target = if wanted[1] {
        Some(crate::ops::arithmetic::neg(&grad)?)
    } else {
        None
    };
    Ok([wanted[0].then_some(grad), by_target])
}

/// Log-cosh loss for robust regression: `log(cosh(p - t))`, each term the
/// value of the exact difference in float64, an integer target's included,
/// narrowed once for float32; see [`log_cosh_term`]. The gradient, `tanh(p -
/// t)` times the incoming gradient and over the count for a mean, is the
/// float nearest it where a bound settles it; see `regression_gradients`.
///
/// It was `|d| + log1p(exp(-2|d|)) - log 2`, which is right where the
/// difference is large and cancels where it is small: the loss there is near
/// `d^2 / 2` and came out with the error of a number near `log 2`, tens of
/// thousands of ulps.
///
/// # Arguments
/// * `predictions` - Model predictions tensor
/// * `targets` - Ground truth targets tensor
/// * `reduction` - How to reduce the loss ("mean", "sum", or "none")
pub fn log_cosh_loss(predictions: &Tensor, targets: &Tensor, reduction: &str) -> Result<Tensor> {
    let (predictions, targets) = difference_operands(predictions, targets)?;
    let (predictions, targets) = (predictions.as_ref(), targets.as_ref());
    let needs_grad = manual_backward_needed(&[predictions, targets]);
    let loss = {
        let _guard = NoGradGuard::new();
        // The exact difference as a two-sum's pair.
        let values = if targets.dtype() == predictions.dtype() {
            wide_terms(predictions, targets, |p, t| {
                let (head, tail) = crate::ops::arithmetic::rounded_once::two_sum(p, -t);
                log_cosh_term(head, tail)
            })?
        } else {
            log_cosh_integer_terms(predictions, targets)?
        };
        let n = values.numel() as f64;
        reduce_detached(values, reduction, n, None)?
    };
    if !needs_grad {
        return Ok(loss);
    }
    regression_backward(loss, (predictions, targets), Slope::Tanh, reduction)
}

/// [`log_cosh_term`] of each prediction less an integer target, narrowed
/// once to the predictions' dtype: the target as its parts, and the
/// difference a pair within 2^-105 of it. It was the term of the difference
/// rounded to the predictions' dtype, which for float32 is 2^-24 of it.
fn log_cosh_integer_terms(predictions: &Tensor, targets: &Tensor) -> Result<Tensor> {
    use crate::ops::arithmetic::rounded_once::two_sum;
    let (p, t) = (predictions.contiguous()?, targets.contiguous()?);
    let x = widened(&p)?;
    let (highs, lows) = integer_parts(&t)?;
    let lows = lows.unwrap_or_else(|| vec![0.0; x.len()]);
    let term = |p: f64, high: f64, low: f64| {
        let (first, error) = two_sum(p, -high);
        if !first.is_finite() {
            // An infinite prediction's, or NaN's.
            return log_cosh_term(first, 0.0);
        }
        let (head, second) = two_sum(first, -low);
        let (head, tail) = two_sum(head, error + second);
        log_cosh_term(head, tail)
    };
    let (dtype, device) = (p.dtype(), p.device());
    let values = crate::ops::map::ternary_map(&x, &highs, &lows, term);
    let data = if dtype == DataType::Float32 {
        let values: Vec<f32> = values.iter().map(|&value| value as f32).collect();
        TensorData::from_vec(values, dtype, device)
    } else {
        TensorData::from_vec(values, dtype, device)
    };
    Ok(Tensor::new(
        Arc::new(data),
        p.shape().clone(),
        dtype,
        device,
        false,
    ))
}

/// `log(cosh(d))` for the difference `d = head + tail`, a two-sum's pair,
/// with each piece's rounding carried to one at the end.
///
/// Below `|d| = 1` it is `log1p(c)` for `c = cosh(d) - 1`, which cancels
/// nothing: `c` is the series `d^2/2 + d^4/24 + ...` as a pair, its leading
/// term formed exactly, and the pair's tail moves the logarithm of its head
/// by the slope, `1 / (1 + c)`. From `1` up it is `|d| - log 2` -- exact, as
/// a pair against `log 2` in two parts -- plus `log1p(exp(-2|d|))`, at most a
/// third of it. The tail of `d` moves the value by `tanh(d)` times itself.
///
/// It took `log1p(u^2 / (2 (1 + u)))` for `u = expm1(|d|)` below 20, which
/// doubles the error of `expm1` in `u^2` and rounds four times besides: up to
/// four ulps, more with a coarser `expm1`.
fn log_cosh_term(head: f64, tail: f64) -> f64 {
    use crate::ops::arithmetic::rounded_once::{product_error, two_sum};
    /// `log 2` less the float64 nearest it.
    const LN_2_LOW: f64 = 2.319_046_813_846_299_6e-17;
    /// `1 / (2k)!` for `k` from 2 to 9: the series of `cosh(d) - 1 - d^2/2`
    /// in `s = d^2`, to the term below `2^-53` of the whole at `|d| = 1`.
    const SERIES: [f64; 8] = [
        1.0 / 24.0,
        1.0 / 720.0,
        1.0 / 40_320.0,
        1.0 / 3_628_800.0,
        1.0 / 479_001_600.0,
        1.0 / 87_178_291_200.0,
        1.0 / 20_922_789_888_000.0,
        1.0 / 6_402_373_705_728_000.0,
    ];

    if !head.is_finite() {
        // NaN stays NaN; an infinite difference, or one past the largest
        // float, has an infinite loss.
        return head.abs();
    }
    let slope = head.tanh() * tail;
    let magnitude = head.abs();
    if magnitude >= 1.0 {
        let (excess, excess_low) = two_sum(magnitude, -std::f64::consts::LN_2);
        let rest = (-2.0 * magnitude).exp().ln_1p();
        return excess + (((excess_low - LN_2_LOW) + rest) + slope);
    }
    let half = 0.5 * magnitude;
    let leading = magnitude * half;
    if leading < 1e-290 {
        // `d^2/2` rounded once, and nothing past it within reach of the
        // float: the series' next term is `d^2/12` of this one.
        return leading + slope;
    }
    let s = magnitude * magnitude;
    let series = SERIES.iter().rev().fold(0.0, |acc, &c| acc * s + c) * s * s;
    let (c, c_low) = two_sum(leading, product_error(magnitude, half, leading) + series);
    c.ln_1p() + (c_low / (1.0 + c) + slope)
}

// Helper functions

/// The float nearest the exact `Σ (p - t)²`, or with `taxicab` `Σ |p - t|`,
/// for `"sum"`, or that over the element count for `"mean"`, as a 0-d tensor
/// of the predictions' float dtype; see
/// `arithmetic::exact_sum::difference_total`. An integer target is taken
/// exactly, beside the predictions widened to float64.
fn exact_difference_loss(
    predictions: &Tensor,
    targets: &Tensor,
    taxicab: bool,
    reduction: &str,
) -> Result<Tensor> {
    use crate::ops::arithmetic::exact_sum::{difference_total, integer_difference_total};
    let (p, t) = (predictions.contiguous()?, targets.contiguous()?);
    let count = if reduction == "mean" { p.numel() } else { 1 };
    let missing = || MinitensorError::internal_error("loss: a typed slice");
    let data = match (p.dtype(), t.dtype()) {
        (DataType::Float32, DataType::Float32) => {
            let (a, b) = (p.data().as_f32_slice(), t.data().as_f32_slice());
            let (a, b) = (a.ok_or_else(missing)?, b.ok_or_else(missing)?);
            let value: f32 = difference_total(a, b, taxicab, count);
            TensorData::from_vec(vec![value], DataType::Float32, p.device())
        }
        (DataType::Float64, DataType::Float64) => {
            let (a, b) = (p.data().as_f64_slice(), t.data().as_f64_slice());
            let (a, b) = (a.ok_or_else(missing)?, b.ok_or_else(missing)?);
            let value: f64 = difference_total(a, b, taxicab, count);
            TensorData::from_vec(vec![value], DataType::Float64, p.device())
        }
        (DataType::Float32, DataType::Int32 | DataType::Int64) => {
            let (wide, (highs, lows)) = (widened(&p)?, integer_parts(&t)?);
            let value: f32 =
                integer_difference_total(&wide, &highs, lows.as_deref(), taxicab, count);
            TensorData::from_vec(vec![value], DataType::Float32, p.device())
        }
        (DataType::Float64, DataType::Int32 | DataType::Int64) => {
            let (wide, (highs, lows)) = (widened(&p)?, integer_parts(&t)?);
            let value: f64 =
                integer_difference_total(&wide, &highs, lows.as_deref(), taxicab, count);
            TensorData::from_vec(vec![value], DataType::Float64, p.device())
        }
        (prediction, target) => {
            return Err(MinitensorError::invalid_operation(format!(
                "loss takes float predictions beside float or integer targets, got \
                 {prediction:?} and {target:?}"
            )));
        }
    };
    Ok(Tensor::new(
        Arc::new(data),
        Shape::scalar(),
        p.dtype(),
        p.device(),
        false,
    ))
}

/// Each `(p - t)²`, the float nearest the square of the exact difference;
/// see `arithmetic::exact_sum::squared_differences`. An integer target is
/// taken exactly, as by [`exact_difference_loss`].
fn squared_differences(predictions: &Tensor, targets: &Tensor) -> Result<Tensor> {
    use crate::ops::arithmetic::exact_sum::{integer_squared_differences, squared_differences};
    let (p, t) = (predictions.contiguous()?, targets.contiguous()?);
    let missing = || MinitensorError::internal_error("loss: a typed slice");
    let data = match (p.dtype(), t.dtype()) {
        (DataType::Float32, DataType::Float32) => {
            let (a, b) = (p.data().as_f32_slice(), t.data().as_f32_slice());
            let (a, b) = (a.ok_or_else(missing)?, b.ok_or_else(missing)?);
            let out: Vec<f32> = squared_differences(a, b);
            TensorData::from_vec(out, DataType::Float32, p.device())
        }
        (DataType::Float64, DataType::Float64) => {
            let (a, b) = (p.data().as_f64_slice(), t.data().as_f64_slice());
            let (a, b) = (a.ok_or_else(missing)?, b.ok_or_else(missing)?);
            let out: Vec<f64> = squared_differences(a, b);
            TensorData::from_vec(out, DataType::Float64, p.device())
        }
        (DataType::Float32, DataType::Int32 | DataType::Int64) => {
            let (wide, (highs, lows)) = (widened(&p)?, integer_parts(&t)?);
            let out: Vec<f32> = integer_squared_differences(&wide, &highs, lows.as_deref());
            TensorData::from_vec(out, DataType::Float32, p.device())
        }
        (DataType::Float64, DataType::Int32 | DataType::Int64) => {
            let (wide, (highs, lows)) = (widened(&p)?, integer_parts(&t)?);
            let out: Vec<f64> = integer_squared_differences(&wide, &highs, lows.as_deref());
            TensorData::from_vec(out, DataType::Float64, p.device())
        }
        (prediction, target) => {
            return Err(MinitensorError::invalid_operation(format!(
                "loss takes float predictions beside float or integer targets, got \
                 {prediction:?} and {target:?}"
            )));
        }
    };
    Ok(Tensor::new(
        Arc::new(data),
        p.shape().clone(),
        p.dtype(),
        p.device(),
        false,
    ))
}

/// The terms of a Huber-shaped loss of `predictions - targets`, with
/// `delta` read at the predictions' width, for `"none"`, or their sum or
/// mean as a 0-d tensor: each the float nearest its exact value; see
/// `arithmetic::exact_sum::huber_terms`. An integer target is taken exactly,
/// as by [`exact_difference_loss`].
fn huber_values(
    predictions: &Tensor,
    targets: &Tensor,
    (delta, smooth): (f64, bool),
    reduction: &str,
) -> Result<Tensor> {
    use crate::ops::arithmetic::exact_sum::Output;
    use crate::ops::arithmetic::exact_sum::{Huber, huber_terms, huber_total};
    /// The terms, or their total over the count a reduction divides by.
    fn values<T: Output>(
        (x, y, lows): (&[f64], &[f64], Option<&[f64]>),
        huber: Huber,
        reduction: &str,
    ) -> Vec<T> {
        match reduction {
            "none" => huber_terms(x, y, lows, huber),
            "mean" => vec![huber_total(x, y, lows, huber, x.len())],
            _ => vec![huber_total(x, y, lows, huber, 1)],
        }
    }
    if !matches!(reduction, "none" | "mean" | "sum") {
        return Err(invalid_reduction(reduction, false));
    }
    let (p, t) = (predictions.contiguous()?, targets.contiguous()?);
    let missing = || MinitensorError::internal_error("loss: a typed slice");
    let x: Cow<'_, [f64]> = match p.dtype() {
        DataType::Float64 => Cow::Borrowed(p.data().as_f64_slice().ok_or_else(missing)?),
        _ => Cow::Owned(widened(&p)?),
    };
    let (y, lows): (Cow<'_, [f64]>, Option<Vec<f64>>) = match t.dtype() {
        DataType::Float64 => (
            Cow::Borrowed(t.data().as_f64_slice().ok_or_else(missing)?),
            None,
        ),
        DataType::Float32 => (Cow::Owned(widened(&t)?), None),
        _ => {
            let (highs, lows) = integer_parts(&t)?;
            (Cow::Owned(highs), lows)
        }
    };
    let operands = (&x[..], &y[..], lows.as_deref());
    let data = if p.dtype() == DataType::Float32 {
        let delta = f64::from(delta as f32);
        let terms = values::<f32>(operands, Huber { delta, smooth }, reduction);
        TensorData::from_vec(terms, p.dtype(), p.device())
    } else {
        let terms = values::<f64>(operands, Huber { delta, smooth }, reduction);
        TensorData::from_vec(terms, p.dtype(), p.device())
    };
    let shape = if reduction == "none" {
        p.shape().clone()
    } else {
        Shape::scalar()
    };
    Ok(Tensor::new(
        Arc::new(data),
        shape,
        p.dtype(),
        p.device(),
        false,
    ))
}

/// A contiguous float tensor's values in float64, which holds each exactly.
fn widened(tensor: &Tensor) -> Result<Vec<f64>> {
    let missing = || MinitensorError::internal_error("loss: a float slice");
    Ok(match tensor.dtype() {
        DataType::Float32 => {
            let values = tensor.data().as_f32_slice().ok_or_else(missing)?;
            values.iter().map(|&value| f64::from(value)).collect()
        }
        _ => tensor.data().as_f64_slice().ok_or_else(missing)?.to_vec(),
    })
}

/// A contiguous integer tensor's values as float64s whose sums they are
/// exactly: each value's nearest float64, and what that leaves beside it
/// where any value lies past 2^53.
fn integer_parts(tensor: &Tensor) -> Result<(Vec<f64>, Option<Vec<f64>>)> {
    let missing = || MinitensorError::internal_error("loss: an integer slice");
    if tensor.dtype() == DataType::Int32 {
        let values = tensor.data().as_i32_slice().ok_or_else(missing)?;
        return Ok((values.iter().map(|&value| f64::from(value)).collect(), None));
    }
    let values = tensor.data().as_i64_slice().ok_or_else(missing)?;
    let highs: Vec<f64> = values.iter().map(|&value| value as f64).collect();
    let limit = 1i64 << 53;
    if values
        .iter()
        .all(|&value| (-limit..=limit).contains(&value))
    {
        return Ok((highs, None));
    }
    // The nearest float64 is within 2^10 of an int64, so the remainder is
    // exact; and as a float64 it can reach 2^63, which no int64 holds, so
    // the remainder is taken in 128 bits.
    let lows = values
        .iter()
        .zip(&highs)
        .map(|(&value, &high)| (i128::from(value) - high as i128) as f64)
        .collect();
    Ok((highs, Some(lows)))
}

/// Applies a loss's `reduction`, on data the caller has already detached.
///
/// Every loss in this file did this itself, nine times over, and mostly
/// identically: sum, divide, or hand the per-element values back. What differs
/// is only the denominator `mean` divides by -- the element count for most,
/// the sample count for `cross_entropy` and `focal`, the batch for `kl_div`'s
/// `batchmean` -- so that is what the caller supplies. The message when the
/// mode is none of them was the most-copied line of the nine.
///
/// The detached form is deliberate: these losses run their forward with
/// autograd off and attach one hand-written node afterwards (see
/// [`manual_backward_needed`]). [`reduce_loss`] is the differentiable
/// counterpart, for the losses that let autograd record the whole computation.
pub(crate) fn reduce_detached(
    values: Tensor,
    reduction: &str,
    mean_denominator: f64,
    batchmean_denominator: Option<f64>,
) -> Result<Tensor> {
    let divide_sum = |denominator: f64| -> Result<Tensor> {
        divide_by_scalar(&sum_all_elements(&values)?, denominator)
    };

    match reduction {
        "mean" => divide_sum(mean_denominator),
        "batchmean" if batchmean_denominator.is_some() => {
            divide_sum(batchmean_denominator.expect("checked by the guard"))
        }
        "sum" => sum_all_elements(&values),
        "none" => Ok(values),
        _ => Err(invalid_reduction(
            reduction,
            batchmean_denominator.is_some(),
        )),
    }
}

/// Refuse a NaN or infinite loss hyperparameter.
///
/// Neither is a setting: a NaN margin or eps made every loss NaN, and an
/// infinite one made it infinite or, for `gamma`, silently zero -- a training
/// run that went wrong at its first step with nothing to say why. The same rule
/// optimizer hyperparameters follow.
pub(crate) fn check_finite_parameter(loss: &str, name: &str, value: f64) -> Result<()> {
    if value.is_finite() {
        return Ok(());
    }
    Err(MinitensorError::invalid_argument(format!(
        "{loss} requires a finite {name}, got {value}"
    )))
}

/// Refuse a Huber `delta` that is not positive and finite.
///
/// `delta <= 0.0` alone is false for NaN, which then propagated through every
/// comparison and returned an all-NaN loss instead of an error.
pub fn check_huber_delta(delta: f64) -> Result<()> {
    if delta.is_finite() && delta > 0.0 {
        return Ok(());
    }
    Err(MinitensorError::invalid_argument(format!(
        "huber_loss requires a positive, finite delta, got {delta}"
    )))
}

/// Refuse a smooth-L1 `beta` that is not finite and at least zero. Zero is
/// the `mae_loss` limit, which `smooth_l1_loss` takes directly.
pub fn check_smooth_l1_beta(beta: f64) -> Result<()> {
    if beta.is_finite() && beta >= 0.0 {
        return Ok(());
    }
    Err(MinitensorError::invalid_argument(format!(
        "smooth_l1_loss requires a finite, non-negative beta, got {beta}"
    )))
}

/// What [`focal_pass`] gives.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Focal {
    /// Each sample's loss.
    Loss,
    /// The loss's gradient in each logit.
    Logits,
    /// Its gradient in each target.
    Targets,
}

/// [`focal_pass`] over tensors: `logits` and `targets` of one shape and dtype,
/// the classes along the last axis, and `up` one value for every sample or one
/// for each.
pub(crate) fn focal_values(
    logits: &Tensor,
    targets: &Tensor,
    params: (f64, f64),
    up: &[f64],
    what: Focal,
) -> Result<Tensor> {
    let (logits, targets) = (logits.contiguous()?, targets.contiguous()?);
    let dims = logits.shape().dims();
    let classes = dims.last().copied().unwrap_or(1);
    let samples = dims[..dims.len().saturating_sub(1)]
        .iter()
        .product::<usize>();
    if up.len() != 1 && up.len() != samples {
        return Err(MinitensorError::internal_error(format!(
            "focal_loss: {} incoming gradients for {samples} samples",
            up.len()
        )));
    }
    let shape = match what {
        Focal::Loss => Shape::new(dims[..dims.len().saturating_sub(1)].to_vec()),
        _ => logits.shape().clone(),
    };
    macro_rules! run {
        ($ty:ty, $slice:ident) => {{
            let missing =
                || MinitensorError::internal_error("focal_loss operands are not one dtype");
            let z = logits.data().$slice().ok_or_else(missing)?;
            let t = targets.data().$slice().ok_or_else(missing)?;
            let out = focal_pass(z, t, (samples, classes), params, up, what);
            TensorData::from_vec::<$ty>(out, logits.dtype(), logits.device())
        }};
    }
    let data = match logits.dtype() {
        DataType::Float32 => run!(f32, as_f32_slice),
        DataType::Float64 => run!(f64, as_f64_slice),
        _ => {
            return Err(MinitensorError::invalid_operation(
                "focal_loss requires floating point tensors",
            ));
        }
    };
    Ok(Tensor::new(
        Arc::new(data),
        shape,
        logits.dtype(),
        logits.device(),
        false,
    ))
}

/// Focal loss over rows of `classes` logits `z` against targets `t` of the
/// same shape, in float64 from the logits and rounded once into their dtype:
/// each row's loss `alpha sum_c t_c (1 - p_c)^gamma (-ln p_c)`, or its
/// gradient in each logit or each target, times the row's incoming gradient.
///
/// A row's `p`, `1 - p` and `ln p` come from [`FocalRow`], and none of them
/// is a probability near one taken from one, as `1 - exp(log_softmax(z))` was:
/// in a confident row it was all rounding -- zero in float32 once `1 - p` was
/// under `2^-25` -- and the loss and its gradient with it, 1.7e7 ulps out in
/// float32 and 3e13 in float64. The gradient in the logits is
/// `alpha (u_c (1 - p_c) - p_c sum_{k != c} u_k)` for
/// `u_k = t_k p_k f'(p_k) = -t_k (1 - p_k)^gamma (1 + gamma p_k (-ln p_k) / (1 - p_k))`,
/// whose parts have one sign, with the largest `u`'s rest of the sum the sum
/// of the rest: for a one-hot target a product at every class, and for a soft
/// one two terms that cancel only where they meet. As `u - p sum u` the
/// target's own class took `p u` from `u`, and as tensor ops in the logits'
/// dtype each step rounded.
pub(crate) fn focal_pass<T: ShiftedExp + Send + Sync>(
    z: &[T],
    t: &[T],
    (rows, classes): (usize, usize),
    (alpha, gamma): (f64, f64),
    up: &[f64],
    what: Focal,
) -> Vec<T> {
    let width = if what == Focal::Loss { 1 } else { classes };
    let mut out = vec![T::zero(); rows * width];
    if classes == 0 {
        // No classes: each row's loss is an empty sum, and there is nothing
        // to differentiate.
        return out;
    }
    // Enough rows to a task that one is worth handing out, and the split
    // gated on the exponentials every row takes, whatever it writes.
    let batch = 512usize.div_ceil(classes);
    let gate = T::SPLIT_THRESHOLD.div_ceil(classes / width);
    par_out_chunks_gated(&mut out, batch * width, gate, &|start, o| {
        let mut terms = vec![0.0f64; classes];
        let mut mass: Vec<(usize, f64)> = Vec::new();
        for (i, o) in o.chunks_mut(width).enumerate() {
            let row = start / width + i;
            let span = row * classes..(row + 1) * classes;
            let (z, t) = (&z[span.clone()], &t[span]);
            let s = FocalRow::new(z, &mut terms);
            let terms = &terms;
            let up = up[if up.len() == 1 { 0 } else { row }];
            let ln_p = |k: usize| s.ln_p(terms, z[k], k);
            match what {
                Focal::Loss => {
                    // A class without target mass adds nothing, though its
                    // `-ln p` may be infinite, and is not formed.
                    let mut sum = CarriedSum::default();
                    for (k, &tk) in t.iter().enumerate() {
                        let tk = tk.wide();
                        if tk != 0.0 {
                            sum.add(tk * s.q(terms, k).powf(gamma) * -ln_p(k));
                        }
                    }
                    // A NaN row is NaN, whatever its targets.
                    let loss = if s.top_q.is_nan() {
                        f64::NAN
                    } else {
                        sum.value()
                    };
                    o[0] = T::narrow(alpha * loss * up);
                }
                Focal::Targets => {
                    for (k, o) in o.iter_mut().enumerate() {
                        let slope = s.q(terms, k).powf(gamma) * -ln_p(k);
                        *o = T::narrow(alpha * slope * up);
                    }
                }
                Focal::Logits => {
                    // The classes with target mass and their `u`; every other
                    // class's `u` is zero.
                    mass.clear();
                    mass.extend((0..classes).filter(|&k| t[k].wide() != 0.0).map(|k| {
                        let (p, q) = (s.p(terms, k), s.q(terms, k));
                        // `gamma p (-ln p) / (1 - p)`, zero at the ends, where
                        // it tends to zero or the direct form is `0 * inf`.
                        let bent = if gamma == 0.0 || !(p > 0.0 && q > 0.0) {
                            0.0
                        } else {
                            gamma * p * (-ln_p(k) / q)
                        };
                        (k, -t[k].wide() * q.powf(gamma) * (1.0 + bent))
                    }));
                    let big = (1..mass.len()).fold(0, |m, i| {
                        if mass[i].1.abs() > mass[m].1.abs() {
                            i
                        } else {
                            m
                        }
                    });
                    let (mut total, mut rest) = (CarriedSum::default(), CarriedSum::default());
                    for (i, &(_, u)) in mass.iter().enumerate() {
                        total.add(u);
                        if i != big {
                            rest.add(u);
                        }
                    }
                    let (total, rest) = (total.value(), rest.value());
                    // A class without target mass: `-alpha p sum(u)`, its term
                    // times one factor for the row.
                    let scale = -(alpha * total * up) / s.total;
                    for (o, &term) in o.iter_mut().zip(terms) {
                        *o = T::narrow(scale * term);
                    }
                    for (i, &(k, u)) in mass.iter().enumerate() {
                        let others = if i == big { rest } else { total - u };
                        let slope = u * s.q(terms, k) - s.p(terms, k) * others;
                        o[k] = T::narrow(alpha * slope * up);
                    }
                }
            }
        }
    });
    out
}

/// One row of focal loss's softmax, in float64 from its logits: its terms
/// `exp(z - top)` of the exact differences, and what makes each class's
/// probability `p`, its complement `1 - p` and its logarithm of them, a class
/// at a time -- the likeliest class's `1 - p` as the sum of the others' terms
/// over the total, and `ln p = (z - top) - log1p(rest)` as `log_softmax` takes
/// it.
struct FocalRow {
    top: usize,
    peak: f64,
    total: f64,
    /// The likeliest class's `1 - p`, and NaN for a NaN row.
    top_q: f64,
    /// `log1p` of the others' terms, which comes off each `z - top`.
    offset: f64,
    /// Whether the row's maximum is infinite, where the terms are `softmax`'s
    /// limits themselves.
    limit: bool,
}

impl FocalRow {
    fn new<T: ShiftedExp>(z: &[T], terms: &mut [f64]) -> Self {
        let top = (1..z.len()).fold(0, |m, k| if z[k] > z[m] { k } else { m });
        let peak = z[top];
        if peak.is_infinite() {
            // The `+inf`s share all of the mass, a row of `-inf` has none,
            // and a NaN makes the row NaN.
            let (infinities, nan) = infinite_slice_parts(z.iter().copied());
            let share = if nan {
                f64::NAN
            } else {
                (infinities as f64).recip()
            };
            for (term, &v) in terms.iter_mut().zip(z) {
                *term = if nan || v == T::infinity() {
                    share
                } else {
                    0.0
                };
            }
            return Self {
                top,
                peak: peak.wide(),
                total: 1.0,
                top_q: 1.0 - terms[top],
                offset: 0.0,
                limit: true,
            };
        }
        T::exp_shifted_into(z, peak, terms);
        terms[top] = 0.0;
        let rest = T::total(terms);
        terms[top] = 1.0;
        let total = 1.0 + rest;
        Self {
            top,
            peak: peak.wide(),
            total,
            top_q: rest / total,
            offset: log1p_of_rest(rest),
            limit: false,
        }
    }

    #[inline(always)]
    fn p(&self, terms: &[f64], k: usize) -> f64 {
        terms[k] / self.total
    }

    #[inline(always)]
    fn q(&self, terms: &[f64], k: usize) -> f64 {
        if k == self.top {
            self.top_q
        } else {
            1.0 - self.p(terms, k)
        }
    }

    #[inline(always)]
    fn ln_p<T: ShiftedExp>(&self, terms: &[f64], z: T, k: usize) -> f64 {
        if self.limit {
            self.p(terms, k).ln()
        } else {
            less_offset(z.wide(), self.peak, self.offset)
        }
    }
}

/// Refuse focal loss settings outside their range: `alpha` in `(0, 1]` and
/// `gamma` finite and at least zero.
///
/// `alpha` scales every class's term alike, so `1` is no weighting at all and
/// is as valid as any smaller value. The range checks were written so a NaN
/// compared false against both bounds and passed them.
pub fn check_focal_parameters(alpha: f64, gamma: f64) -> Result<()> {
    if !(alpha > 0.0 && alpha <= 1.0) {
        return Err(MinitensorError::invalid_argument(format!(
            "focal_loss requires alpha in (0, 1], got {alpha}"
        )));
    }
    if !(gamma.is_finite() && gamma >= 0.0) {
        return Err(MinitensorError::invalid_argument(format!(
            "focal_loss requires a finite, non-negative gamma, got {gamma}"
        )));
    }
    Ok(())
}

/// Applies a loss's `reduction` through the ordinary differentiable ops, for
/// the losses whose gradients come from autograd rather than from a
/// hand-written node.
pub(crate) fn reduce_loss(values: Tensor, reduction: &str) -> Result<Tensor> {
    match reduction {
        "mean" => mean(&values, None, false),
        "sum" => sum(&values, None, false),
        "none" => Ok(values),
        _ => Err(invalid_reduction(reduction, false)),
    }
}

/// Whether a loss recognises this reduction mode, asked before any work.
///
/// The reducers above find out by trying, which is the right time for them but
/// the wrong time for a constructor: a loss *module* stores its mode and does
/// not reduce anything until its first forward, so `MSELoss("men")` used to
/// build happily and fail a training step later, where `mse_loss(..., "men")`
/// refused at the call. Same list, asked earlier.
pub fn check_reduction(reduction: &str, batchmean: bool) -> Result<()> {
    let recognised =
        matches!(reduction, "mean" | "sum" | "none") || (batchmean && reduction == "batchmean");
    if recognised {
        Ok(())
    } else {
        Err(invalid_reduction(reduction, batchmean))
    }
}

/// The one message for a reduction mode a loss does not recognise. Shared so
/// the wording cannot drift between the losses that reduce here and `ctc_loss`,
/// which validates its mode up front instead (its own reduction is
/// length-normalised per sample, so it cannot use [`reduce_detached`]).
pub(crate) fn invalid_reduction(reduction: &str, batchmean: bool) -> MinitensorError {
    let modes = if batchmean {
        "'mean', 'batchmean', 'sum', or 'none'"
    } else {
        "'mean', 'sum', or 'none'"
    };
    MinitensorError::invalid_operation(format!(
        "Invalid reduction mode: {reduction}. Must be {modes}"
    ))
}

/// The two operands of a loss, checked and at the float dtype they promote to.
///
/// The predictions are a model's activations and must be floats, as they must
/// for every layer-shaped op. The targets may be of any dtype and meet the
/// predictions as arithmetic meets them: the wider float decides, and an
/// integer or bool target takes the predictions' width. Requiring the two
/// dtypes to match refused a float32 prediction against float64 targets, and
/// every integer target -- `binary_cross_entropy(p, [1, 0])` included --
/// though each is plain arithmetic. The casts are differentiable, so the
/// gradient reaches each operand at its own dtype.
fn loss_operands<'a>(
    predictions: &'a Tensor,
    targets: &'a Tensor,
) -> Result<(Cow<'a, Tensor>, Cow<'a, Tensor>)> {
    operands(predictions, targets, false)
}

/// [`loss_operands`] for a loss of the difference alone, which leaves an
/// int32 or int64 target as it is: `predictions - targets` then rounds the
/// exact difference once, where the cast rounded the target first -- past
/// 2^24 for float32 predictions, and past 2^53 for float64 ones -- and a
/// target one above a prediction could lose nothing.
fn difference_operands<'a>(
    predictions: &'a Tensor,
    targets: &'a Tensor,
) -> Result<(Cow<'a, Tensor>, Cow<'a, Tensor>)> {
    operands(predictions, targets, true)
}

fn operands<'a>(
    predictions: &'a Tensor,
    targets: &'a Tensor,
    keep_integer_targets: bool,
) -> Result<(Cow<'a, Tensor>, Cow<'a, Tensor>)> {
    // Check device compatibility
    if predictions.device() != targets.device() {
        return Err(MinitensorError::device_mismatch(
            format!("{:?}", predictions.device()),
            format!("{:?}", targets.device()),
        ));
    }

    // Check shape compatibility
    if predictions.shape() != targets.shape() {
        return Err(MinitensorError::shape_mismatch(
            predictions.shape().dims().to_vec(),
            targets.shape().dims().to_vec(),
        ));
    }

    if !predictions.dtype().is_float() {
        return Err(MinitensorError::invalid_operation(format!(
            "Loss functions require floating point predictions, got {}",
            predictions.dtype()
        )));
    }
    let common = crate::ops::binary::promote_division_dtype(predictions.dtype(), targets.dtype());
    let cast = |tensor: &'a Tensor| -> Result<Cow<'a, Tensor>> {
        Ok(if tensor.dtype() == common {
            Cow::Borrowed(tensor)
        } else {
            Cow::Owned(tensor.astype(common)?)
        })
    };
    let integer = matches!(targets.dtype(), DataType::Int32 | DataType::Int64);
    let targets = if keep_integer_targets && integer {
        Cow::Borrowed(targets)
    } else {
        cast(targets)?
    };
    Ok((cast(predictions)?, targets))
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
        // backward gradient in the prediction dtype. This is the saved value
        // only, so it is detached; a target gradient, when one is needed, is
        // recorded by the caller against a cast of its own.
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
