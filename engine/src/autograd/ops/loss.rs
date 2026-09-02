// Copyright (c) Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

//! Gradient functions for the loss operations in [`crate::ops::loss`].

use super::*;
use crate::ops::map::par_out_chunks;
use crate::{
    error::{MinitensorError, Result},
    ops::map::{binary_map, ternary_map, unary_map_into},
    ops::util::create_scalar_tensor,
    ops::{activation, arithmetic, reduction},
    tensor::{DataType, Shape, Tensor, TensorData},
};
use rustc_hash::FxHashMap;
use std::sync::Arc;

/// Gradient function for MSE loss
pub struct MSELossBackward {
    pub predictions_shape: Vec<usize>,
    pub targets_shape: Vec<usize>,
    pub input_ids: [TensorId; 2],
    /// Which of [predictions, targets] actually need a gradient. Targets
    /// almost never do, so their gradient chain is skipped entirely.
    pub input_requires_grad: [bool; 2],
    pub reduction: String,
    pub diff: Tensor,
}
impl GradientFunction for MSELossBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<FxHashMap<TensorId, Tensor>> {
        let mut gradients = FxHashMap::default();
        gradients.reserve(2);

        // Base gradient: 2 * (predictions - targets)
        let two = create_scalar_tensor(2.0, self.diff.dtype(), self.diff.device())?;
        let mut base_grad = arithmetic::mul(&self.diff, &two)?;

        // Apply reduction scaling
        match self.reduction.as_str() {
            "mean" => {
                let n = self.diff.numel() as f64;
                let scale = create_scalar_tensor(1.0 / n, base_grad.dtype(), base_grad.device())?;
                base_grad = arithmetic::mul(&base_grad, &scale)?;
            }
            "sum" | "none" => {}
            _ => {
                return Err(MinitensorError::gradient_error(format!(
                    "Unknown reduction mode: {}",
                    self.reduction
                )));
            }
        }

        // Multiply by upstream gradient
        let pred_grad = arithmetic::mul(&base_grad, grad_output)?;
        if self.input_requires_grad[1] {
            let target_grad = arithmetic::neg(&pred_grad)?;
            accumulate_grad(&mut gradients, self.input_ids[1], target_grad)?;
        }
        if self.input_requires_grad[0] {
            accumulate_grad(&mut gradients, self.input_ids[0], pred_grad)?;
        }

        Ok(gradients)
    }

    fn input_ids(&self) -> &[TensorId] {
        &self.input_ids
    }
}
/// Gradient function for MAE loss
pub struct MAELossBackward {
    pub predictions_shape: Vec<usize>,
    pub targets_shape: Vec<usize>,
    pub input_ids: [TensorId; 2],
    /// Which of [predictions, targets] actually need a gradient. Targets
    /// almost never do, so their gradient chain is skipped entirely.
    pub input_requires_grad: [bool; 2],
    pub reduction: String,
    pub sign: Tensor,
}
impl GradientFunction for MAELossBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<FxHashMap<TensorId, Tensor>> {
        let mut gradients = FxHashMap::default();
        gradients.reserve(2);

        let mut base_grad = self.sign.clone();
        match self.reduction.as_str() {
            "mean" => {
                let n = self.sign.numel() as f64;
                let scale = create_scalar_tensor(1.0 / n, base_grad.dtype(), base_grad.device())?;
                base_grad = arithmetic::mul(&base_grad, &scale)?;
            }
            "sum" | "none" => {}
            _ => {
                return Err(MinitensorError::gradient_error(format!(
                    "Unknown reduction mode: {}",
                    self.reduction
                )));
            }
        }

        let pred_grad = arithmetic::mul(&base_grad, grad_output)?;
        if self.input_requires_grad[1] {
            let target_grad = arithmetic::neg(&pred_grad)?;
            accumulate_grad(&mut gradients, self.input_ids[1], target_grad)?;
        }
        if self.input_requires_grad[0] {
            accumulate_grad(&mut gradients, self.input_ids[0], pred_grad)?;
        }

        Ok(gradients)
    }

    fn input_ids(&self) -> &[TensorId] {
        &self.input_ids
    }
}
/// Gradient function for Huber loss
pub struct HuberLossBackward {
    pub predictions_shape: Vec<usize>,
    pub targets_shape: Vec<usize>,
    pub input_ids: [TensorId; 2],
    /// Which of [predictions, targets] actually need a gradient. Targets
    /// almost never do, so their gradient chain is skipped entirely.
    pub input_requires_grad: [bool; 2],
    pub delta: f64,
    pub reduction: String,
    pub diff: Tensor,
}
impl GradientFunction for HuberLossBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<FxHashMap<TensorId, Tensor>> {
        let mut gradients = FxHashMap::default();
        gradients.reserve(2);

        let numel = self.diff.numel();
        let dtype = self.diff.dtype();
        let device = self.diff.device();
        let mut grad_data = TensorData::zeros_on_device(numel, dtype, device);

        match dtype {
            DataType::Float32 => {
                let diff_slice = self.diff.data().as_f32_slice().ok_or_else(|| {
                    MinitensorError::internal_error("Failed to get f32 slice from diff")
                })?;
                let grad_slice = grad_data.as_f32_slice_mut().ok_or_else(|| {
                    MinitensorError::internal_error("Failed to get mutable f32 slice from grad")
                })?;
                let delta = self.delta as f32;
                unary_map_into(grad_slice, diff_slice, move |d: f32| {
                    if d.abs() <= delta {
                        d
                    } else {
                        delta * d.signum()
                    }
                });
            }
            DataType::Float64 => {
                let diff_slice = self.diff.data().as_f64_slice().ok_or_else(|| {
                    MinitensorError::internal_error("Failed to get f64 slice from diff")
                })?;
                let grad_slice = grad_data.as_f64_slice_mut().ok_or_else(|| {
                    MinitensorError::internal_error("Failed to get mutable f64 slice from grad")
                })?;
                let delta = self.delta;
                unary_map_into(grad_slice, diff_slice, move |d: f64| {
                    if d.abs() <= delta {
                        d
                    } else {
                        delta * d.signum()
                    }
                });
            }
            _ => {
                return Err(MinitensorError::invalid_operation(
                    "Huber loss only supports floating point tensors",
                ));
            }
        }

        let mut base_grad = Tensor::new(
            Arc::new(grad_data),
            Shape::new(self.predictions_shape.clone()),
            dtype,
            device,
            false,
        );

        if self.reduction == "mean" {
            let scale = create_scalar_tensor(1.0 / numel as f64, dtype, device)?;
            base_grad = arithmetic::mul(&base_grad, &scale)?;
        }

        let pred_grad = arithmetic::mul(&base_grad, grad_output)?;
        if self.input_requires_grad[1] {
            let target_grad = arithmetic::neg(&pred_grad)?;
            accumulate_grad(&mut gradients, self.input_ids[1], target_grad)?;
        }
        if self.input_requires_grad[0] {
            accumulate_grad(&mut gradients, self.input_ids[0], pred_grad)?;
        }

        Ok(gradients)
    }

    fn input_ids(&self) -> &[TensorId] {
        &self.input_ids
    }
}
/// Gradient function for Cross Entropy loss
pub struct CrossEntropyLossBackward {
    pub predictions_shape: Vec<usize>,
    pub targets_shape: Vec<usize>,
    /// Cross-entropy differentiates only with respect to logits. Targets are
    /// saved values, not graph inputs, so do not retain or traverse their tape.
    pub input_ids: [TensorId; 1],
    pub reduction: String,
    pub softmax_predictions: Tensor,
    /// One-hot (or soft) targets, for the general formula below.
    pub targets: Option<Tensor>,
    /// Class indices, when that is what the caller passed. The gradient is
    /// then `(softmax - one_hot) * scale`, which needs no one-hot to exist:
    /// subtracting 1 at one column per row is a scatter, not a dense subtract.
    /// Taking it collapses five full-size passes -- a class-axis sum, a
    /// broadcast multiply, a subtract, a scalar multiply and the grad_output
    /// multiply -- into one.
    pub target_indices: Option<Tensor>,
}

/// `grad[i, c] = scale_i * (softmax[i, c] - [c == index_i])`, in one pass.
macro_rules! cross_entropy_index_grad {
    ($name:ident, $ty:ty, $idx:ty) => {
        fn $name(probs: &[$ty], idx: &[$idx], classes: usize, scale: &[f64], out: &mut [$ty]) {
            par_out_chunks(out, classes, &|start, o| {
                let row = start / classes;
                let p = &probs[start..start + o.len()];
                {
                    let s = if scale.len() == 1 {
                        scale[0]
                    } else {
                        scale[row]
                    };
                    for (c, slot) in o.iter_mut().enumerate() {
                        *slot = (p[c] as f64 * s) as $ty;
                    }
                    let target = idx[row] as usize;
                    if target < classes {
                        o[target] = ((p[target] as f64 - 1.0) * s) as $ty;
                    }
                }
            });
        }
    };
}

cross_entropy_index_grad!(ce_index_grad_f32_i32, f32, i32);
cross_entropy_index_grad!(ce_index_grad_f32_i64, f32, i64);
cross_entropy_index_grad!(ce_index_grad_f64_i32, f64, i32);
cross_entropy_index_grad!(ce_index_grad_f64_i64, f64, i64);
impl GradientFunction for CrossEntropyLossBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<FxHashMap<TensorId, Tensor>> {
        let mut gradients = FxHashMap::default();
        gradients.reserve(1);

        // For L = -sum(t * log_softmax(z)), dL/dz is
        // softmax(z) * sum(t) - t. The commonly quoted `softmax - target`
        // assumes each target row sums to one; keeping the sum explicit makes
        // probability/weighted targets mathematically correct while reducing
        // to the usual formula for class-index and normalized one-hot targets.
        let probabilities = self.softmax_predictions.detach();

        if let Some(indices) = &self.target_indices {
            let classes = *self.predictions_shape.last().unwrap_or(&1);
            let rows = indices.numel();
            // Per-row scaling: grad_output is a scalar for `mean`/`sum` and one
            // value per sample for `none`; `mean` folds in the 1/batch.
            let go = grad_output.detach();
            let mut scale: Vec<f64> = match go.dtype() {
                DataType::Float32 => go
                    .data()
                    .as_f32_slice()
                    .map(|s| s.iter().map(|&v| v as f64).collect())
                    .unwrap_or_default(),
                DataType::Float64 => go
                    .data()
                    .as_f64_slice()
                    .map(|s| s.to_vec())
                    .unwrap_or_default(),
                _ => {
                    return Err(MinitensorError::invalid_operation(
                        "CrossEntropy backward only supports floating point tensors",
                    ));
                }
            };
            if self.reduction == "mean" {
                let batch = self.predictions_shape.first().copied().unwrap_or(1).max(1) as f64;
                for v in scale.iter_mut() {
                    *v /= batch;
                }
            }
            if scale.is_empty() {
                scale.push(0.0);
            }

            let mut out = TensorData::zeros_on_device(
                rows * classes,
                probabilities.dtype(),
                probabilities.device(),
            );
            macro_rules! run {
                ($pslice:ident, $oslice:ident, $islice:ident, $kernel:ident) => {{
                    let p = probabilities.data().$pslice().ok_or_else(|| {
                        MinitensorError::internal_error("cross_entropy backward: bad probabilities")
                    })?;
                    let i = indices.data().$islice().ok_or_else(|| {
                        MinitensorError::internal_error("cross_entropy backward: bad indices")
                    })?;
                    let o = out.$oslice().ok_or_else(|| {
                        MinitensorError::internal_error("cross_entropy backward: bad output")
                    })?;
                    $kernel(p, i, classes, &scale, o);
                }};
            }
            match (probabilities.dtype(), indices.dtype()) {
                (DataType::Float32, DataType::Int32) => {
                    run!(
                        as_f32_slice,
                        as_f32_slice_mut,
                        as_i32_slice,
                        ce_index_grad_f32_i32
                    )
                }
                (DataType::Float32, DataType::Int64) => {
                    run!(
                        as_f32_slice,
                        as_f32_slice_mut,
                        as_i64_slice,
                        ce_index_grad_f32_i64
                    )
                }
                (DataType::Float64, DataType::Int32) => {
                    run!(
                        as_f64_slice,
                        as_f64_slice_mut,
                        as_i32_slice,
                        ce_index_grad_f64_i32
                    )
                }
                (DataType::Float64, DataType::Int64) => {
                    run!(
                        as_f64_slice,
                        as_f64_slice_mut,
                        as_i64_slice,
                        ce_index_grad_f64_i64
                    )
                }
                _ => {
                    return Err(MinitensorError::invalid_operation(
                        "CrossEntropy backward: unsupported dtype pair",
                    ));
                }
            }
            let grad = Tensor::new(
                Arc::new(out),
                Shape::new(self.predictions_shape.clone()),
                probabilities.dtype(),
                probabilities.device(),
                false,
            );
            accumulate_grad(&mut gradients, self.input_ids[0], grad)?;
            return Ok(gradients);
        }

        let targets = self
            .targets
            .as_ref()
            .ok_or_else(|| {
                MinitensorError::internal_error("cross_entropy backward: no targets saved")
            })?
            .detach();
        let class_dim = (targets.ndim() - 1) as isize;
        let target_mass = reduction::sum(&targets, Some(vec![class_dim]), true)?;
        let weighted_probabilities = arithmetic::mul(&probabilities, &target_mass)?;
        let mut base_grad = arithmetic::sub(&weighted_probabilities, &targets)?;

        // Apply reduction scaling
        match self.reduction.as_str() {
            "mean" => {
                let batch = self.targets_shape[0] as f64;
                let mut scalar_data =
                    TensorData::zeros_on_device(1, base_grad.dtype(), base_grad.device());
                match base_grad.dtype() {
                    DataType::Float32 => {
                        let slice = scalar_data.as_f32_slice_mut().ok_or_else(|| {
                            MinitensorError::internal_error(
                                "Failed to get mutable f32 slice from scalar",
                            )
                        })?;
                        slice[0] = (1.0 / batch) as f32;
                    }
                    DataType::Float64 => {
                        let slice = scalar_data.as_f64_slice_mut().ok_or_else(|| {
                            MinitensorError::internal_error(
                                "Failed to get mutable f64 slice from scalar",
                            )
                        })?;
                        slice[0] = 1.0 / batch;
                    }
                    _ => {
                        return Err(MinitensorError::invalid_operation(
                            "CrossEntropy backward only supports floating point tensors",
                        ));
                    }
                }
                let scalar_tensor = Tensor::new(
                    Arc::new(scalar_data),
                    Shape::new(vec![1]),
                    base_grad.dtype(),
                    base_grad.device(),
                    false,
                );
                base_grad = arithmetic::mul(&base_grad, &scalar_tensor)?;
            }
            "sum" | "none" => {}
            _ => {
                return Err(MinitensorError::gradient_error(format!(
                    "Unknown reduction mode: {}",
                    self.reduction
                )));
            }
        }

        // `reduction="none"` returns one loss per sample, while `base_grad`
        // retains the trailing class dimension. Make that missing dimension
        // explicit: ordinary trailing-dimension broadcasting would otherwise
        // reject most batch/class combinations and can silently scale columns
        // instead of samples when the two sizes happen to match.
        let grad_output = if self.reduction == "none" {
            crate::ops::shape_ops::unsqueeze(grad_output, grad_output.ndim() as isize)?
        } else {
            grad_output.clone()
        };
        let pred_grad = arithmetic::mul(&base_grad, &grad_output)?;

        // Targets typically have no gradient
        accumulate_grad(&mut gradients, self.input_ids[0], pred_grad)?;

        Ok(gradients)
    }

    fn input_ids(&self) -> &[TensorId] {
        &self.input_ids
    }
}
/// Gradient function for Binary Cross Entropy loss
pub struct BCELossBackward {
    pub predictions_shape: Vec<usize>,
    pub targets_shape: Vec<usize>,
    pub input_ids: [TensorId; 2],
    /// Which of [predictions, targets] actually need a gradient. Only the
    /// prediction gradient is ever produced; it is skipped when frozen.
    pub input_requires_grad: [bool; 2],
    pub reduction: String,
    pub predictions: Tensor,
    pub targets: Tensor,
}
impl GradientFunction for BCELossBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<FxHashMap<TensorId, Tensor>> {
        let mut gradients = FxHashMap::default();
        if !self.input_requires_grad[0] {
            return Ok(gradients);
        }
        gradients.reserve(1);

        // BCE gradient: (predictions - targets) / (predictions * (1 - predictions))
        let one = create_scalar_tensor(1.0, self.predictions.dtype(), self.predictions.device())?;
        let one_minus_pred = arithmetic::sub(&one, &self.predictions)?;
        let numerator = arithmetic::sub(&self.predictions, &self.targets)?;
        // The denominator is clamped to EPSILON (1e-12) in
        // `binary_cross_entropy_backward`, so a saturated prediction (predictions
        // * (1 - predictions) == 0) produces a large-but-finite gradient rather
        // than inf/nan.
        let denom = arithmetic::mul(&self.predictions, &one_minus_pred)?.clamp_min(1e-12)?;
        let mut base_grad = arithmetic::div(&numerator, &denom)?;

        if self.reduction == "mean" {
            let n = self.predictions.numel() as f64;
            let scale = create_scalar_tensor(1.0 / n, base_grad.dtype(), base_grad.device())?;
            base_grad = arithmetic::mul(&base_grad, &scale)?;
        }

        let pred_grad = arithmetic::mul(&base_grad, grad_output)?;
        accumulate_grad(&mut gradients, self.input_ids[0], pred_grad)?;

        Ok(gradients)
    }

    fn input_ids(&self) -> &[TensorId] {
        &self.input_ids
    }
}
/// Gradient function for binary cross entropy computed from logits.
///
/// The whole point of fusing the sigmoid into the loss is that this gradient
/// has a closed form that never saturates: `d/dx = sigmoid(x) - target`, or
/// with a positive-class weight `w`,
///
/// ```text
/// d/dx = (1 - target) + (1 + (w - 1) * target) * (sigmoid(x) - 1)
/// ```
///
/// Computing it directly is what keeps a confidently-wrong logit learning.
/// Going through `sigmoid` and then `binary_cross_entropy` instead multiplies
/// `1/(p * (1 - p))` by `p * (1 - p)`, and at |x| >= 30 in f32 both factors
/// have already collapsed to inf/0 — the product is whatever the clamps leave
/// behind rather than the -1 the maths calls for.
pub struct BCEWithLogitsLossBackward {
    pub input_ids: [TensorId; 2],
    /// Which of [logits, targets] actually need a gradient. Only the logit
    /// gradient is ever produced; it is skipped when frozen.
    pub input_requires_grad: [bool; 2],
    pub reduction: String,
    pub logits: Tensor,
    pub targets: Tensor,
    /// Already broadcast to the logits' shape by the forward pass, so the
    /// backward is a flat elementwise walk.
    pub pos_weight: Option<Tensor>,
}
impl GradientFunction for BCEWithLogitsLossBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<FxHashMap<TensorId, Tensor>> {
        let mut gradients = FxHashMap::default();
        if !self.input_requires_grad[0] {
            return Ok(gradients);
        }
        gradients.reserve(1);

        let scale = if self.reduction == "mean" {
            1.0 / self.logits.numel() as f64
        } else {
            1.0
        };

        let base_grad =
            bce_with_logits_grad(&self.logits, &self.targets, self.pos_weight.as_ref(), scale)?;
        let logit_grad = arithmetic::mul(&base_grad, grad_output)?;
        accumulate_grad(&mut gradients, self.input_ids[0], logit_grad)?;

        Ok(gradients)
    }

    fn input_ids(&self) -> &[TensorId] {
        &self.input_ids
    }
}
/// `scale * [(1 - t) + (1 + (w - 1) * t) * (sigmoid(x) - 1)]`, elementwise.
///
/// `sigmoid` is evaluated in the numerically safe direction — `exp` of a
/// negative argument only — so the factor is exact across the whole logit
/// range instead of overflowing for large negative `x`.
fn bce_with_logits_grad(
    logits: &Tensor,
    targets: &Tensor,
    pos_weight: Option<&Tensor>,
    scale: f64,
) -> Result<Tensor> {
    macro_rules! grad_for {
        ($ty:ty, $slice:ident, $dtype:expr) => {{
            let x = logits.data().$slice().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get slice from logits")
            })?;
            let t = targets.data().$slice().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get slice from targets")
            })?;
            let scale = scale as $ty;
            let compute = |x: $ty, t: $ty, w: $ty| {
                let s: $ty = if x >= 0.0 {
                    1.0 / (1.0 + (-x).exp())
                } else {
                    let e = x.exp();
                    e / (1.0 + e)
                };
                scale * ((1.0 - t) + (1.0 + (w - 1.0) * t) * (s - 1.0))
            };
            let values = match pos_weight {
                Some(w) => {
                    let w = w.data().$slice().ok_or_else(|| {
                        MinitensorError::internal_error("Failed to get slice from pos_weight")
                    })?;
                    ternary_map(x, t, w, compute)
                }
                None => binary_map(x, t, |x: $ty, t: $ty| compute(x, t, 1.0)),
            };
            TensorData::from_vec::<$ty>(values, $dtype, logits.device())
        }};
    }

    let data = match logits.dtype() {
        DataType::Float32 => grad_for!(f32, as_f32_slice, DataType::Float32),
        DataType::Float64 => grad_for!(f64, as_f64_slice, DataType::Float64),
        _ => {
            return Err(MinitensorError::invalid_operation(
                "binary_cross_entropy_with_logits requires floating point tensors",
            ));
        }
    };

    Ok(Tensor::new(
        Arc::new(data),
        logits.shape().clone(),
        logits.dtype(),
        logits.device(),
        false,
    ))
}
/// Gradient function for KL Divergence loss
pub struct KLDivLossBackward {
    pub predictions_shape: Vec<usize>,
    pub targets_shape: Vec<usize>,
    pub input_ids: [TensorId; 2],
    /// Which of [predictions, targets] actually need a gradient. Targets
    /// almost never do, so their gradient chain is skipped entirely.
    pub input_requires_grad: [bool; 2],
    pub reduction: String,
    pub predictions: Tensor,
    pub targets: Tensor,
}
impl KLDivLossBackward {
    /// The divisor the forward pass applied, so the gradient is scaled by the
    /// same amount. Reading it off `reduction` rather than assuming `numel` is
    /// what keeps `mean` and `batchmean` from silently disagreeing with the
    /// forward, as they did when this hard-coded `numel` for `mean` while the
    /// forward divided by the batch size.
    fn reduction_divisor(&self) -> Option<f64> {
        match self.reduction.as_str() {
            "mean" => Some((self.predictions.numel().max(1)) as f64),
            "batchmean" => Some(crate::ops::loss::kl_div_batch_size(&self.predictions)),
            _ => None,
        }
    }
}

impl GradientFunction for KLDivLossBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<FxHashMap<TensorId, Tensor>> {
        let mut gradients = FxHashMap::default();
        gradients.reserve(2);

        // Gradient w.r.t predictions: -(targets / predictions)
        if self.input_requires_grad[0] {
            let mut pred_grad = arithmetic::div(&self.targets, &self.predictions)?;
            pred_grad = arithmetic::neg(&pred_grad)?;
            if let Some(n) = self.reduction_divisor() {
                let scale = create_scalar_tensor(1.0 / n, pred_grad.dtype(), pred_grad.device())?;
                pred_grad = arithmetic::mul(&pred_grad, &scale)?;
            }
            let pred_grad = arithmetic::mul(&pred_grad, grad_output)?;
            accumulate_grad(&mut gradients, self.input_ids[0], pred_grad)?;
        }

        // Gradient w.r.t targets: log(targets) - log(predictions) + 1
        if self.input_requires_grad[1] {
            let log_targets = activation::log(&self.targets)?;
            let log_preds = activation::log(&self.predictions)?;
            let diff = arithmetic::sub(&log_targets, &log_preds)?;
            let one = create_scalar_tensor(1.0, self.targets.dtype(), self.targets.device())?;
            let mut target_grad = arithmetic::add(&diff, &one)?;
            if let Some(n) = self.reduction_divisor() {
                let scale =
                    create_scalar_tensor(1.0 / n, target_grad.dtype(), target_grad.device())?;
                target_grad = arithmetic::mul(&target_grad, &scale)?;
            }
            let target_grad = arithmetic::mul(&target_grad, grad_output)?;
            accumulate_grad(&mut gradients, self.input_ids[1], target_grad)?;
        }

        Ok(gradients)
    }

    fn input_ids(&self) -> &[TensorId] {
        &self.input_ids
    }
}
/// Gradient function for Focal loss
pub struct FocalLossBackward {
    pub predictions_shape: Vec<usize>,
    pub targets_shape: Vec<usize>,
    pub input_ids: [TensorId; 2],
    /// Which of [predictions, targets] actually need a gradient. Only the
    /// prediction gradient is ever produced; it is skipped when frozen.
    pub input_requires_grad: [bool; 2],
    pub alpha: f64,
    pub gamma: f64,
    pub reduction: String,
    pub softmax_predictions: Tensor,
    pub targets: Tensor,
}
impl GradientFunction for FocalLossBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<FxHashMap<TensorId, Tensor>> {
        let mut gradients = FxHashMap::default();
        if !self.input_requires_grad[0] {
            return Ok(gradients);
        }
        gradients.reserve(1);

        // Exact gradient of FL = -alpha * (1 - p_t)^gamma * log(p_t) wrt the
        // logits, where p_t is the true-class softmax probability:
        //   dFL/dz_j = alpha * (p_j - onehot_j)
        //              * (1 - p_t)^(gamma-1) * [ (1 - p_t) - gamma * p_t * ln(p_t) ]
        // The modulating factor is a per-sample scalar (broadcast over classes).
        let p = self.softmax_predictions.detach();
        let t = self.targets.detach();
        let dtype = p.dtype();
        let device = p.device();

        // True-class probability per sample: p_t = sum(p * onehot) over classes.
        let class_dim = (p.ndim() - 1) as isize;
        let pt = reduction::sum(&arithmetic::mul(&p, &t)?, Some(vec![class_dim]), true)?;

        let one = create_scalar_tensor(1.0, dtype, device)?;
        let one_minus_pt = arithmetic::sub(&one, &pt)?;
        let log_pt = crate::ops::activation::log(&pt)?;
        let gamma_scalar = create_scalar_tensor(self.gamma, dtype, device)?;
        // bracket = (1 - p_t) - gamma * p_t * ln(p_t)
        let bracket = arithmetic::sub(
            &one_minus_pt,
            &arithmetic::mul(&arithmetic::mul(&gamma_scalar, &pt)?, &log_pt)?,
        )?;
        let modulating =
            arithmetic::mul(&tensor_power(&one_minus_pt, self.gamma - 1.0)?, &bracket)?;
        let alpha_tensor = create_scalar_tensor(self.alpha, dtype, device)?;
        let weight = arithmetic::mul(&modulating, &alpha_tensor)?; // per-sample scalar

        let mut base_grad = arithmetic::mul(&arithmetic::sub(&p, &t)?, &weight)?;

        if self.reduction == "mean" {
            let num_classes = *self.predictions_shape.last().unwrap_or(&1);
            let num_samples =
                (self.predictions_shape.iter().product::<usize>() / num_classes.max(1)) as f64;
            let scale = create_scalar_tensor(1.0 / num_samples, dtype, device)?;
            base_grad = arithmetic::mul(&base_grad, &scale)?;
        }

        let pred_grad = arithmetic::mul(&base_grad, grad_output)?;
        accumulate_grad(&mut gradients, self.input_ids[0], pred_grad)?;

        Ok(gradients)
    }

    fn input_ids(&self) -> &[TensorId] {
        &self.input_ids
    }
}
/// Raise each tensor element to the given power
fn tensor_power(tensor: &Tensor, exponent: f64) -> Result<Tensor> {
    let mut output_data =
        TensorData::zeros_on_device(tensor.numel(), tensor.dtype(), tensor.device());

    match tensor.dtype() {
        DataType::Float32 => {
            let input = tensor.data().as_f32_slice().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get f32 slice from tensor")
            })?;
            let output = output_data.as_f32_slice_mut().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get mutable f32 slice from output")
            })?;
            let exp = exponent as f32;
            unary_map_into(output, input, move |v: f32| v.powf(exp));
        }
        DataType::Float64 => {
            let input = tensor.data().as_f64_slice().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get f64 slice from tensor")
            })?;
            let output = output_data.as_f64_slice_mut().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get mutable f64 slice from output")
            })?;
            unary_map_into(output, input, move |v: f64| v.powf(exponent));
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
        false,
    ))
}

/// Gradient function for the CTC loss.
///
/// The forward-backward pass already produced the exact gradient of the loss
/// with respect to `log_probs`, with the reduction's per-sample scaling folded
/// in, so nothing is recomputed here. What is left is the upstream factor --
/// and for `"none"` that is one number per batch element, which has to meet the
/// *batch* axis of a `(steps, batch, classes)` gradient rather than the last
/// axis that broadcasting from the right would line it up with.
pub struct CtcLossBackward {
    pub input_id: TensorId,
    pub reduction: String,
    pub gradient: Tensor,
}

impl GradientFunction for CtcLossBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<FxHashMap<TensorId, Tensor>> {
        let scaled = if self.reduction == "none" {
            let batch = grad_output.numel();
            let lined_up =
                crate::ops::shape_ops::reshape(grad_output, Shape::new(vec![1, batch, 1]))?;
            arithmetic::mul(&self.gradient, &lined_up)?
        } else {
            arithmetic::mul(&self.gradient, grad_output)?
        };
        let mut gradients = FxHashMap::default();
        accumulate_grad(&mut gradients, self.input_id, scaled)?;
        Ok(gradients)
    }

    fn input_ids(&self) -> &[TensorId] {
        std::slice::from_ref(&self.input_id)
    }
}
