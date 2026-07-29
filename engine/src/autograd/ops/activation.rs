// Copyright (c) Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

use super::*;
use crate::{
    device::Device,
    error::{MinitensorError, Result},
    ops::map::{binary_map, ternary_map, unary_map_into},
    ops::{activation, arithmetic, reduction},
    tensor::{DataType, Shape, Tensor, TensorData},
};
use rayon::prelude::*;
use rustc_hash::FxHashMap;
use std::sync::Arc;

pub(crate) fn repeat_interleave_backward_impl(
    grad_output: &Tensor,
    input_shape: &[usize],
    repeats: &[usize],
    dim: usize,
) -> Result<Tensor> {
    if dim >= input_shape.len() {
        return Err(MinitensorError::index_error(
            dim as isize,
            0,
            input_shape.len(),
        ));
    }

    let dim_size = input_shape[dim];
    if repeats.len() != dim_size {
        return Err(MinitensorError::invalid_operation(
            "repeat_interleave backward: repeats must match input dimension size".to_string(),
        ));
    }

    let grad_shape_vec = input_shape.to_vec();
    let grad_shape = Shape::new(grad_shape_vec.clone());
    let numel = grad_shape.numel();
    let dtype = grad_output.dtype();
    let device = grad_output.device();
    let total_repeats: usize = repeats.iter().sum();

    let inner: usize = if dim + 1 >= input_shape.len() {
        1
    } else {
        input_shape[dim + 1..].iter().product()
    };
    let outer: usize = if dim == 0 {
        1
    } else {
        input_shape[..dim].iter().product()
    };

    if numel == 0 || total_repeats == 0 || inner == 0 || outer == 0 {
        return Ok(Tensor::zeros(
            Shape::new(grad_shape_vec),
            dtype,
            device,
            false,
        ));
    }

    let output_dims = grad_output.shape().dims();
    if output_dims.len() != input_shape.len() || output_dims[dim] != total_repeats {
        return Err(MinitensorError::shape_mismatch(
            input_shape.to_vec(),
            output_dims.to_vec(),
        ));
    }

    macro_rules! repeat_interleave_backward_impl_inner {
        ($ty:ty, $slice:ident, $from_vec:ident) => {{
            let src = grad_output.data().$slice().ok_or_else(|| {
                MinitensorError::invalid_operation(
                    "repeat_interleave backward: gradient tensor must be contiguous".to_string(),
                )
            })?;
            let mut dst = vec![<$ty>::default(); numel];
            let chunk = total_repeats * inner;
            dst.par_chunks_mut(dim_size * inner)
                .enumerate()
                .for_each(|(outer_idx, dst_chunk)| {
                    let mut src_offset = outer_idx * chunk;
                    for (i, &rep) in repeats.iter().enumerate() {
                        if rep == 0 {
                            continue;
                        }
                        let dst_start = i * inner;
                        let dst_slice = &mut dst_chunk[dst_start..dst_start + inner];
                        for _ in 0..rep {
                            let src_slice = &src[src_offset..src_offset + inner];
                            dst_slice.iter_mut().zip(src_slice.iter()).for_each(
                                |(dst_val, &src_val)| {
                                    *dst_val += src_val;
                                },
                            );
                            src_offset += inner;
                        }
                    }
                });
            TensorData::$from_vec(dst, device)
        }};
    }

    let data = match dtype {
        DataType::Float32 => {
            repeat_interleave_backward_impl_inner!(f32, as_f32_slice, from_vec_f32)
        }
        DataType::Float64 => {
            repeat_interleave_backward_impl_inner!(f64, as_f64_slice, from_vec_f64)
        }
        DataType::Int32 => repeat_interleave_backward_impl_inner!(i32, as_i32_slice, from_vec_i32),
        DataType::Int64 => repeat_interleave_backward_impl_inner!(i64, as_i64_slice, from_vec_i64),
        DataType::Bool => {
            return Ok(Tensor::zeros(grad_shape, dtype, device, false));
        }
    };

    Ok(Tensor::new(
        Arc::new(data),
        grad_shape,
        dtype,
        device,
        false,
    ))
}

/// Gradient function for expand operation which reduces broadcasted gradients
pub struct ExpandBackward {
    pub input_shape: Vec<usize>,
    pub input_id: TensorId,
}

impl GradientFunction for ExpandBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<FxHashMap<TensorId, Tensor>> {
        let mut gradients = FxHashMap::default();
        gradients.reserve(1);

        let shape = Shape::new(self.input_shape.clone());
        let grad_input = reduce_gradient_for_broadcasting(grad_output, &shape)?;
        gradients.insert(self.input_id, grad_input);
        Ok(gradients)
    }

    fn input_ids(&self) -> &[TensorId] {
        std::slice::from_ref(&self.input_id)
    }
}

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
    pub targets: Tensor,
}

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
        let targets = self.targets.detach();
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
        let one = Tensor::ones(
            Shape::new(self.predictions_shape.clone()),
            self.predictions.dtype(),
            self.predictions.device(),
            false,
        );
        let one_minus_pred = arithmetic::sub(&one, &self.predictions)?;
        let numerator = arithmetic::sub(&self.predictions, &self.targets)?;
        // PyTorch clamps the denominator to EPSILON (1e-12) in
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

impl GradientFunction for KLDivLossBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<FxHashMap<TensorId, Tensor>> {
        let mut gradients = FxHashMap::default();
        gradients.reserve(2);

        // Gradient w.r.t predictions: -(targets / predictions)
        if self.input_requires_grad[0] {
            let mut pred_grad = arithmetic::div(&self.targets, &self.predictions)?;
            pred_grad = arithmetic::neg(&pred_grad)?;
            if self.reduction == "mean" {
                let n = self.predictions.numel() as f64;
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
            let one = Tensor::ones(
                self.targets.shape().clone(),
                self.targets.dtype(),
                self.targets.device(),
                false,
            );
            let mut target_grad = arithmetic::add(&diff, &one)?;
            if self.reduction == "mean" {
                let n = self.predictions.numel() as f64;
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

/// Create a scalar tensor with the given value
pub(crate) fn create_scalar_tensor(value: f64, dtype: DataType, device: Device) -> Result<Tensor> {
    let mut data = TensorData::zeros_on_device(1, dtype, device);
    match dtype {
        DataType::Float32 => {
            let slice = data.as_f32_slice_mut().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get mutable f32 slice from scalar")
            })?;
            slice[0] = value as f32;
        }
        DataType::Float64 => {
            let slice = data.as_f64_slice_mut().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get mutable f64 slice from scalar")
            })?;
            slice[0] = value;
        }
        _ => {
            return Err(MinitensorError::invalid_operation(
                "Scalar tensors only supported for floating point types",
            ));
        }
    }

    Ok(Tensor::new(
        Arc::new(data),
        Shape::new(vec![1]),
        dtype,
        device,
        false,
    ))
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
