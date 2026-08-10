// Copyright (c) Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

use crate::autograd::{LayerNormBackward, RmsNormBackward, TensorId, add_to_graph};
use crate::device::Device;
use crate::error::{MinitensorError, Result};
use crate::tensor::{DataType, Shape, Tensor, TensorData};
use rayon::prelude::*;
use smallvec::SmallVec;
use std::sync::Arc;

fn scalar_tensor(value: f64, dtype: DataType, device: Device) -> Result<Tensor> {
    let mut data = TensorData::zeros_on_device(1, dtype, device);
    match dtype {
        DataType::Float32 => {
            let slice = data.as_f32_slice_mut().ok_or_else(|| {
                MinitensorError::internal_error(
                    "Failed to get mutable f32 slice from scalar tensor",
                )
            })?;
            slice[0] = value as f32;
        }
        DataType::Float64 => {
            let slice = data.as_f64_slice_mut().ok_or_else(|| {
                MinitensorError::internal_error(
                    "Failed to get mutable f64 slice from scalar tensor",
                )
            })?;
            slice[0] = value;
        }
        _ => {
            return Err(MinitensorError::invalid_operation(
                "Normalization operations only support floating point tensors".to_string(),
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

/// Functional batch normalization.
///
/// Normalizes the input tensor using batch statistics during training or
/// running estimates during evaluation.
///
/// * `input` - Input tensor of shape `[N, C, ...]` where the second dimension
///   is interpreted as the feature/channel dimension.
/// * `running_mean` - Optional running mean buffer updated during training.
/// * `running_var` - Optional running variance buffer updated during training.
/// * `weight` - Optional learnable scale parameter (gamma).
/// * `bias` - Optional learnable shift parameter (beta).
/// * `training` - When true, use batch statistics and update running stats.
/// * `momentum` - Momentum factor for running statistics update.
/// * `eps` - Small epsilon added to variance for numerical stability.
#[allow(clippy::too_many_arguments)]
pub fn batch_norm(
    input: &Tensor,
    running_mean: Option<&mut Tensor>,
    running_var: Option<&mut Tensor>,
    weight: Option<&Tensor>,
    bias: Option<&Tensor>,
    training: bool,
    momentum: f64,
    eps: f64,
) -> Result<Tensor> {
    if input.ndim() < 2 {
        return Err(MinitensorError::invalid_operation(
            "batch_norm expects input with at least 2 dimensions",
        ));
    }

    let num_features = input.size(1)?;

    // Validate parameter shapes
    if let Some(w) = weight
        && (w.ndim() != 1 || w.size(0)? != num_features)
    {
        return Err(MinitensorError::shape_mismatch(
            vec![num_features],
            vec![w.size(0)?],
        ));
    }
    if let Some(b) = bias
        && (b.ndim() != 1 || b.size(0)? != num_features)
    {
        return Err(MinitensorError::shape_mismatch(
            vec![num_features],
            vec![b.size(0)?],
        ));
    }
    if let Some(rm) = &running_mean
        && (rm.ndim() != 1 || rm.size(0)? != num_features)
    {
        return Err(MinitensorError::shape_mismatch(
            vec![num_features],
            vec![rm.size(0)?],
        ));
    }
    if let Some(rv) = &running_var
        && (rv.ndim() != 1 || rv.size(0)? != num_features)
    {
        return Err(MinitensorError::shape_mismatch(
            vec![num_features],
            vec![rv.size(0)?],
        ));
    }

    // Dimensions along which to compute statistics (all except channel dim)
    let axes: Vec<usize> = (0..input.ndim()).filter(|&d| d != 1).collect();
    let axes_isize: Vec<isize> = axes.iter().map(|&d| d as isize).collect();

    // Compute batch statistics only when they are actually used: during
    // training, or in eval mode when no running estimates are available.
    let use_batch_stats = training || running_mean.is_none() || running_var.is_none();
    let (mean_used, var_used, centered) = if use_batch_stats {
        let batch_mean = input.mean(Some(axes_isize.clone()), true)?; // [1, C, ...]
        let centered = crate::ops::arithmetic::sub(input, &batch_mean)?;
        let batch_var = crate::ops::arithmetic::mul(&centered, &centered)?
            .mean(Some(axes_isize.clone()), true)?;
        (batch_mean, batch_var, centered)
    } else if let (Some(rm), Some(rv)) = (running_mean.as_ref(), running_var.as_ref()) {
        // Use running statistics (reshape for broadcasting)
        let mut rm_view = (*rm).clone().unsqueeze(0)?; // [1, C]
        let mut rv_view = (*rv).clone().unsqueeze(0)?;
        for _ in 2..input.ndim() {
            rm_view = rm_view.unsqueeze(rm_view.ndim() as isize)?;
            rv_view = rv_view.unsqueeze(rv_view.ndim() as isize)?;
        }
        let centered = crate::ops::arithmetic::sub(input, &rm_view)?;
        (rm_view, rv_view, centered)
    } else {
        unreachable!("running stats checked")
    };

    // Prepare epsilon tensor
    let eps_tensor = scalar_tensor(eps, input.dtype(), input.device())?;

    let var_eps = crate::ops::arithmetic::add(&var_used, &eps_tensor)?;
    let std = crate::ops::activation::sqrt(&var_eps)?;
    let mut output = crate::ops::arithmetic::div(&centered, &std)?;

    // Scale and shift
    if let Some(w) = weight {
        let mut w_view = w.clone().unsqueeze(0)?;
        for _ in 2..input.ndim() {
            w_view = w_view.unsqueeze(w_view.ndim() as isize)?;
        }
        output = crate::ops::arithmetic::mul(&output, &w_view)?;
    }
    if let Some(b) = bias {
        let mut b_view = b.clone().unsqueeze(0)?;
        for _ in 2..input.ndim() {
            b_view = b_view.unsqueeze(b_view.ndim() as isize)?;
        }
        output = crate::ops::arithmetic::add(&output, &b_view)?;
    }

    // Update running statistics if training (mean_used/var_used hold the
    // batch statistics whenever training is true)
    if training && let (Some(rm), Some(rv)) = (running_mean, running_var) {
        let mean_flat = mean_used.view(Shape::new(vec![num_features]))?.detach();
        let var_flat = var_used.view(Shape::new(vec![num_features]))?.detach();

        // PyTorch stores the *unbiased* batch variance in running_var (Bessel's
        // correction: var_biased * n / (n - 1)), even though the normalization
        // above deliberately uses the biased estimate. Apply the correction so
        // eval-time statistics match PyTorch. `n` is the number of elements
        // reduced per channel; the degenerate n == 1 case has no unbiased
        // variance, so it is left uncorrected (PyTorch rejects it outright).
        let count = input.numel() / num_features;
        let var_flat = if count > 1 {
            let correction = count as f64 / (count as f64 - 1.0);
            let corr_tensor = scalar_tensor(correction, input.dtype(), input.device())?;
            crate::ops::arithmetic::mul(&var_flat, &corr_tensor)?
        } else {
            var_flat
        };

        let m_tensor = scalar_tensor(momentum, input.dtype(), input.device())?;
        let one_minus_tensor = scalar_tensor(1.0 - momentum, input.dtype(), input.device())?;

        *rm = crate::ops::arithmetic::add(
            &crate::ops::arithmetic::mul(rm, &one_minus_tensor)?,
            &crate::ops::arithmetic::mul(&mean_flat, &m_tensor)?,
        )?;
        *rv = crate::ops::arithmetic::add(
            &crate::ops::arithmetic::mul(rv, &one_minus_tensor)?,
            &crate::ops::arithmetic::mul(&var_flat, &m_tensor)?,
        )?;
    }

    Ok(output)
}

/// Apply layer normalization to the input tensor.
/// Fused LayerNorm forward, one row at a time.
///
/// The composed form this replaces cost six full-size tensor operations --
/// `mean`, `sub`, `mul`, `mean`, `sqrt`, `div`, then two more for weight and
/// bias -- each allocating and traversing a tensor the size of the input, and
/// three of them going through the *broadcasting* path because the statistics
/// have a trailing 1. On a 32x128x512 float32 tensor that measured 18.4ms
/// against 0.47ms for a single `mean` over the same data.
///
/// The normalized dimensions are trailing and contiguous, so the input is just
/// `[rows, norm]` in memory and each row can be reduced and written while it is
/// still in L1. The three passes over a row cost far less than one pass over
/// the whole tensor.
///
/// `normalized` and `inv_std` are produced here rather than recovered later
/// because [`crate::autograd::LayerNormBackward`] saves both, so fusing the
/// forward must not cost the backward its inputs.
macro_rules! layer_norm_rows {
    ($name:ident, $ty:ty) => {
        fn $name(
            input: &[$ty],
            norm: usize,
            weight: Option<&[$ty]>,
            bias: Option<&[$ty]>,
            eps: f64,
            out: &mut [$ty],
            normalized: Option<&mut [$ty]>,
            inv_std: &mut [$ty],
        ) {
            // Normalizing over an empty axis: every buffer is empty and
            // `par_chunks_mut(0)` would panic. The shape still has to survive,
            // which the caller handles.
            if norm == 0 {
                return;
            }
            let recip = 1.0 / norm as f64;

            /// Mean and `1/sqrt(var + eps)` for one row.
            ///
            /// Two reduction passes rather than one. `E[x^2] - E[x]^2` would
            /// halve the traffic and lose every significant digit when the mean
            /// dominates the spread, which is exactly the regime a
            /// normalization layer sits in.
            #[inline(always)]
            fn stats(row: &[$ty], recip: f64, eps: f64) -> (f64, f64) {
                let mut sum = 0.0f64;
                for &v in row {
                    sum += v as f64;
                }
                let mean = sum * recip;
                let mut sq = 0.0f64;
                for &v in row {
                    let d = v as f64 - mean;
                    sq += d * d;
                }
                (mean, 1.0 / (sq * recip + eps).sqrt())
            }

            // The normalized values are saved for the backward, so a forward
            // that will not be differentiated should not pay to write them --
            // a second full-size buffer, filled and then dropped.
            match normalized {
                Some(normalized) => {
                    out.par_chunks_mut(norm)
                        .zip(normalized.par_chunks_mut(norm))
                        .zip(inv_std.par_iter_mut())
                        .zip(input.par_chunks(norm))
                        .for_each(|(((o, n), is), row)| {
                            let (mean, scale) = stats(row, recip, eps);
                            *is = scale as $ty;
                            for i in 0..norm {
                                let z = (row[i] as f64 - mean) * scale;
                                n[i] = z as $ty;
                                let mut y = z;
                                if let Some(w) = weight {
                                    y *= w[i] as f64;
                                }
                                if let Some(b) = bias {
                                    y += b[i] as f64;
                                }
                                o[i] = y as $ty;
                            }
                        });
                }
                None => {
                    out.par_chunks_mut(norm)
                        .zip(inv_std.par_iter_mut())
                        .zip(input.par_chunks(norm))
                        .for_each(|((o, is), row)| {
                            let (mean, scale) = stats(row, recip, eps);
                            *is = scale as $ty;
                            for i in 0..norm {
                                let z = (row[i] as f64 - mean) * scale;
                                let mut y = z;
                                if let Some(w) = weight {
                                    y *= w[i] as f64;
                                }
                                if let Some(b) = bias {
                                    y += b[i] as f64;
                                }
                                o[i] = y as $ty;
                            }
                        });
                }
            }
        }
    };
}

layer_norm_rows!(layer_norm_rows_f32, f32);
layer_norm_rows!(layer_norm_rows_f64, f64);

pub fn layer_norm(
    input: &Tensor,
    normalized_shape: &[usize],
    weight: Option<&Tensor>,
    bias: Option<&Tensor>,
    eps: f64,
) -> Result<Tensor> {
    if normalized_shape.is_empty() {
        return Err(MinitensorError::invalid_argument(
            "layer_norm requires at least one normalized dimension".to_string(),
        ));
    }

    if normalized_shape.len() > input.ndim() {
        return Err(MinitensorError::invalid_operation(
            "normalized_shape rank cannot exceed input rank for layer_norm".to_string(),
        ));
    }

    match input.dtype() {
        DataType::Float32 | DataType::Float64 => {}
        _ => {
            return Err(MinitensorError::invalid_operation(
                "layer_norm only supports floating point tensors".to_string(),
            ));
        }
    }

    let axis_start = input.ndim() - normalized_shape.len();
    for (i, &expected) in normalized_shape.iter().enumerate() {
        let dim = axis_start + i;
        let actual = input.size(dim)?;
        if actual != expected {
            return Err(MinitensorError::shape_mismatch(
                vec![expected],
                vec![actual],
            ));
        }
    }

    if let Some(w) = weight {
        if w.dtype() != input.dtype() {
            return Err(MinitensorError::type_mismatch(
                input.dtype().to_string(),
                w.dtype().to_string(),
            ));
        }
        if w.device() != input.device() {
            return Err(MinitensorError::device_mismatch(
                input.device().to_string(),
                w.device().to_string(),
            ));
        }
        if w.shape().dims() != normalized_shape {
            return Err(MinitensorError::shape_mismatch(
                normalized_shape.to_vec(),
                w.shape().dims().to_vec(),
            ));
        }
    }

    if let Some(b) = bias {
        if b.dtype() != input.dtype() {
            return Err(MinitensorError::type_mismatch(
                input.dtype().to_string(),
                b.dtype().to_string(),
            ));
        }
        if b.device() != input.device() {
            return Err(MinitensorError::device_mismatch(
                input.device().to_string(),
                b.device().to_string(),
            ));
        }
        if b.shape().dims() != normalized_shape {
            return Err(MinitensorError::shape_mismatch(
                normalized_shape.to_vec(),
                b.shape().dims().to_vec(),
            ));
        }
    }

    let norm: usize = normalized_shape.iter().product();
    let total = input.numel();
    // A zero-sized normalized axis divides by zero; there are no rows in that
    // case, which is what `checked_div` reports as `None`.
    let rows = total.checked_div(norm).unwrap_or(0);

    // Statistics shape: the input's, with the normalized dims collapsed to 1.
    let mut stat_dims = input.shape().dims().to_vec();
    for d in stat_dims.iter_mut().skip(axis_start) {
        *d = 1;
    }
    let stat_shape = Shape::new(stat_dims);

    // Decided before the kernel runs, not after: the normalized values exist
    // only to be saved for the backward, so an inference forward should not
    // allocate and fill a second full-size buffer to drop it.
    let requires_grad = crate::autograd::is_grad_enabled()
        && (input.requires_grad()
            || weight.map(|w| w.requires_grad()).unwrap_or(false)
            || bias.map(|b| b.requires_grad()).unwrap_or(false));

    let (output_data, normalized_data, inv_std_data) = match input.dtype() {
        DataType::Float32 => {
            let src = input.data().as_f32_slice().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get f32 slice from input tensor")
            })?;
            let w = weight.and_then(|t| t.data().as_f32_slice());
            let b = bias.and_then(|t| t.data().as_f32_slice());
            let mut o = vec![0f32; total];
            let mut n = if requires_grad {
                vec![0f32; total]
            } else {
                Vec::new()
            };
            let mut s = vec![0f32; rows];
            layer_norm_rows_f32(
                src,
                norm,
                w,
                b,
                eps,
                &mut o,
                requires_grad.then_some(n.as_mut_slice()),
                &mut s,
            );
            (
                TensorData::from_vec::<f32>(o, DataType::Float32, input.device()),
                TensorData::from_vec::<f32>(n, DataType::Float32, input.device()),
                TensorData::from_vec::<f32>(s, DataType::Float32, input.device()),
            )
        }
        _ => {
            let src = input.data().as_f64_slice().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get f64 slice from input tensor")
            })?;
            let w = weight.and_then(|t| t.data().as_f64_slice());
            let b = bias.and_then(|t| t.data().as_f64_slice());
            let mut o = vec![0f64; total];
            let mut n = if requires_grad {
                vec![0f64; total]
            } else {
                Vec::new()
            };
            let mut s = vec![0f64; rows];
            layer_norm_rows_f64(
                src,
                norm,
                w,
                b,
                eps,
                &mut o,
                requires_grad.then_some(n.as_mut_slice()),
                &mut s,
            );
            (
                TensorData::from_vec::<f64>(o, DataType::Float64, input.device()),
                TensorData::from_vec::<f64>(n, DataType::Float64, input.device()),
                TensorData::from_vec::<f64>(s, DataType::Float64, input.device()),
            )
        }
    };

    let output = Tensor::new(
        Arc::new(output_data),
        input.shape().clone(),
        input.dtype(),
        input.device(),
        requires_grad,
    );
    if !requires_grad {
        // `normalized_data` is empty here and `inv_std` unused; building
        // tensors around them would claim a shape the buffers do not have.
        return Ok(output);
    }

    let normalized = Tensor::new(
        Arc::new(normalized_data),
        input.shape().clone(),
        input.dtype(),
        input.device(),
        false,
    );
    let inv_std = Tensor::new(
        Arc::new(inv_std_data),
        stat_shape,
        input.dtype(),
        input.device(),
        false,
    );

    let mut weight_broadcast: Option<Tensor> = None;
    if let Some(w) = weight {
        let mut view = w.clone();
        for _ in 0..axis_start {
            view = view.unsqueeze(0)?;
        }
        weight_broadcast = Some(view.detach());
    }

    let mut input_ids: SmallVec<[TensorId; 3]> = SmallVec::new();
    input_ids.push(input.id());
    if let Some(w) = weight {
        input_ids.push(w.id());
    }
    if let Some(b) = bias {
        input_ids.push(b.id());
    }

    let grad_fn = Arc::new(LayerNormBackward {
        input_ids,
        input_id: input.id(),
        weight_id: weight.map(|w| w.id()),
        bias_id: bias.map(|b| b.id()),
        normalized: normalized.detach(),
        inv_std: inv_std.detach(),
        weight_broadcast,
        normalized_shape: normalized_shape.to_vec(),
        axis_start,
        element_count: normalized_shape.iter().product(),
        input_requires_grad: input.requires_grad(),
        weight_requires_grad: weight.map(|w| w.requires_grad()).unwrap_or(false),
        bias_requires_grad: bias.map(|b| b.requires_grad()).unwrap_or(false),
    });

    let mut output_with_grad = output;
    output_with_grad.set_grad_fn(Some(grad_fn.clone()));
    add_to_graph(&output_with_grad, Some(grad_fn))?;
    Ok(output_with_grad)
}

/// Apply root-mean-square layer normalization (RMSNorm) to the input tensor.
///
/// RMSNorm (Zhang & Sennrich, 2019) normalizes activations by their root mean
/// square over the trailing `normalized_shape` dimensions — dropping LayerNorm's
/// mean subtraction and bias term — then rescales by an optional learnable
/// `weight`:
///
/// ```text
/// rms_norm(x) = x / sqrt(mean(x², over normalized dims) + eps) * weight
/// ```
///
/// It is the normalization used by LLaMA, Mistral, Gemma, Qwen and most modern
/// large language models: cheaper than LayerNorm (no mean/variance, no re-
/// centering) while matching or improving training stability. Built by composing
/// autograd-tracked primitives, so gradients — including the coupling of the
/// input through the RMS denominator — flow automatically.
/// Fused RMSNorm forward, one row at a time.
///
/// Same shape as [`layer_norm_rows`] and for the same reason: the composed form
/// cost a full-size `mul`, a reduction, a broadcast `add`, an `rsqrt` and two
/// more full-size broadcast multiplies. On a 32x128x512 float32 tensor that
/// measured 21.6ms, against 0.47ms for a single `mean` over the same data.
///
/// Only one reduction pass here -- RMSNorm has no mean to subtract, so there is
/// no cancellation to avoid and `sum(x^2)` is the whole statistic. It still
/// accumulates in float64.
macro_rules! rms_norm_rows {
    ($name:ident, $ty:ty) => {
        fn $name(
            input: &[$ty],
            norm: usize,
            weight: Option<&[$ty]>,
            eps: f64,
            out: &mut [$ty],
            inv_rms: &mut [$ty],
        ) {
            if norm == 0 {
                return;
            }
            let recip = 1.0 / norm as f64;
            out.par_chunks_mut(norm)
                .zip(inv_rms.par_iter_mut())
                .zip(input.par_chunks(norm))
                .for_each(|((o, ir), row)| {
                    let mut sq = 0.0f64;
                    for &v in row {
                        let d = v as f64;
                        sq += d * d;
                    }
                    let scale = 1.0 / (sq * recip + eps).sqrt();
                    *ir = scale as $ty;
                    for i in 0..norm {
                        let mut y = row[i] as f64 * scale;
                        if let Some(w) = weight {
                            y *= w[i] as f64;
                        }
                        o[i] = y as $ty;
                    }
                });
        }
    };
}

rms_norm_rows!(rms_norm_rows_f32, f32);
rms_norm_rows!(rms_norm_rows_f64, f64);

pub fn rms_norm(
    input: &Tensor,
    normalized_shape: &[usize],
    weight: Option<&Tensor>,
    eps: f64,
) -> Result<Tensor> {
    if normalized_shape.is_empty() {
        return Err(MinitensorError::invalid_argument(
            "rms_norm requires at least one normalized dimension".to_string(),
        ));
    }
    if normalized_shape.len() > input.ndim() {
        return Err(MinitensorError::invalid_operation(
            "normalized_shape rank cannot exceed input rank for rms_norm".to_string(),
        ));
    }
    match input.dtype() {
        DataType::Float32 | DataType::Float64 => {}
        _ => {
            return Err(MinitensorError::invalid_operation(
                "rms_norm only supports floating point tensors".to_string(),
            ));
        }
    }

    let axis_start = input.ndim() - normalized_shape.len();
    for (i, &expected) in normalized_shape.iter().enumerate() {
        let actual = input.size(axis_start + i)?;
        if actual != expected {
            return Err(MinitensorError::shape_mismatch(
                vec![expected],
                vec![actual],
            ));
        }
    }

    if let Some(w) = weight {
        if w.dtype() != input.dtype() {
            return Err(MinitensorError::type_mismatch(
                input.dtype().to_string(),
                w.dtype().to_string(),
            ));
        }
        if w.device() != input.device() {
            return Err(MinitensorError::device_mismatch(
                input.device().to_string(),
                w.device().to_string(),
            ));
        }
        if w.shape().dims() != normalized_shape {
            return Err(MinitensorError::shape_mismatch(
                normalized_shape.to_vec(),
                w.shape().dims().to_vec(),
            ));
        }
    }

    let norm: usize = normalized_shape.iter().product();
    let total = input.numel();
    // A zero-sized normalized axis divides by zero; there are no rows in that
    // case, which is what `checked_div` reports as `None`.
    let rows = total.checked_div(norm).unwrap_or(0);

    let mut stat_dims = input.shape().dims().to_vec();
    for d in stat_dims.iter_mut().skip(axis_start) {
        *d = 1;
    }
    let stat_shape = Shape::new(stat_dims);

    let (output_data, inv_rms_data) = match input.dtype() {
        DataType::Float32 => {
            let src = input.data().as_f32_slice().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get f32 slice from input tensor")
            })?;
            let w = weight.and_then(|t| t.data().as_f32_slice());
            let mut o = vec![0f32; total];
            let mut r = vec![0f32; rows];
            rms_norm_rows_f32(src, norm, w, eps, &mut o, &mut r);
            (
                TensorData::from_vec::<f32>(o, DataType::Float32, input.device()),
                TensorData::from_vec::<f32>(r, DataType::Float32, input.device()),
            )
        }
        _ => {
            let src = input.data().as_f64_slice().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get f64 slice from input tensor")
            })?;
            let w = weight.and_then(|t| t.data().as_f64_slice());
            let mut o = vec![0f64; total];
            let mut r = vec![0f64; rows];
            rms_norm_rows_f64(src, norm, w, eps, &mut o, &mut r);
            (
                TensorData::from_vec::<f64>(o, DataType::Float64, input.device()),
                TensorData::from_vec::<f64>(r, DataType::Float64, input.device()),
            )
        }
    };

    let requires_grad = input.requires_grad() || weight.map(|w| w.requires_grad()).unwrap_or(false);

    let output = Tensor::new(
        Arc::new(output_data),
        input.shape().clone(),
        input.dtype(),
        input.device(),
        requires_grad,
    );
    if !requires_grad {
        return Ok(output);
    }

    let inv_rms = Tensor::new(
        Arc::new(inv_rms_data),
        stat_shape,
        input.dtype(),
        input.device(),
        false,
    );

    let mut input_ids: SmallVec<[TensorId; 2]> = SmallVec::new();
    input_ids.push(input.id());
    if let Some(w) = weight {
        input_ids.push(w.id());
    }
    let grad_fn = Arc::new(RmsNormBackward {
        input_ids,
        input_id: input.id(),
        weight_id: weight.map(|w| w.id()),
        input: input.detach(),
        inv_rms,
        weight: weight.map(|w| w.detach()),
        normalized_shape: normalized_shape.to_vec(),
        element_count: norm,
        input_requires_grad: input.requires_grad(),
        weight_requires_grad: weight.map(|w| w.requires_grad()).unwrap_or(false),
    });

    let mut output_with_grad = output;
    output_with_grad.set_grad_fn(Some(grad_fn.clone()));
    add_to_graph(&output_with_grad, Some(grad_fn))?;
    Ok(output_with_grad)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::autograd;
    use crate::device::Device;
    use crate::tensor::{DataType, TensorData};
    use std::sync::Arc;

    fn tensor_from_vec(data: Vec<f32>, shape: Vec<usize>, requires_grad: bool) -> Tensor {
        Tensor::new(
            Arc::new(TensorData::from_vec_f32(data, Device::cpu())),
            Shape::new(shape),
            DataType::Float32,
            Device::cpu(),
            requires_grad,
        )
    }

    #[test]
    fn test_layer_norm_forward_zero_mean_unit_var() {
        let input = tensor_from_vec(vec![1.0, 2.0, 3.0, -1.0, 0.0, 4.0], vec![2, 3], false);
        let result = layer_norm(&input, &[3], None, None, 1e-5).unwrap();
        let data = result.data().as_f32_slice().unwrap();

        for row in 0..2 {
            let start = row * 3;
            let slice = &data[start..start + 3];
            let mean: f32 = slice.iter().sum::<f32>() / 3.0;
            assert!(mean.abs() < 1e-5);
            let var: f32 = slice
                .iter()
                .map(|v| {
                    let diff = *v - mean;
                    diff * diff
                })
                .sum::<f32>()
                / 3.0;
            assert!((var - 1.0).abs() < 1e-4);
        }
    }

    #[test]
    fn test_layer_norm_backward_matches_manual_gradients() {
        let input_vals = vec![1.2f32, -0.5, 2.0, 0.7, -1.3, 0.25];
        let weight_vals = vec![1.5f32, 0.75, -0.25];
        let bias_vals = vec![0.1f32, -0.2, 0.05];

        let input = tensor_from_vec(input_vals.clone(), vec![2, 3], true);
        let weight = tensor_from_vec(weight_vals.clone(), vec![3], true);
        let bias = tensor_from_vec(bias_vals.clone(), vec![3], true);

        let result = layer_norm(&input, &[3], Some(&weight), Some(&bias), 1e-5).unwrap();
        let ones = Tensor::ones(
            result.shape().clone(),
            result.dtype(),
            result.device(),
            false,
        );
        let grads = autograd::backward_collect(&result, Some(ones)).unwrap();

        let grad_input = grads.get(&input.id()).unwrap();
        let grad_weight = grads.get(&weight.id()).unwrap();
        let grad_bias = grads.get(&bias.id()).unwrap();

        let mut expected_input_grad = vec![0.0f32; input_vals.len()];
        let mut expected_weight_grad = vec![0.0f32; weight_vals.len()];
        let mut expected_bias_grad = vec![0.0f32; bias_vals.len()];
        let eps = 1e-5f32;
        let m = 3.0f32;

        for row in 0..2 {
            let start = row * 3;
            let x = &input_vals[start..start + 3];
            let mean = x.iter().sum::<f32>() / m;
            let centered: Vec<f32> = x.iter().map(|v| *v - mean).collect();
            let var = centered.iter().map(|v| v * v).sum::<f32>() / m;
            let inv_std = 1.0 / (var + eps).sqrt();
            let normalized: Vec<f32> = centered.iter().map(|v| v * inv_std).collect();

            let grad_output = [1.0f32; 3];
            let grad_output_hat: Vec<f32> = grad_output
                .iter()
                .zip(weight_vals.iter())
                .map(|(g, w)| g * *w)
                .collect();

            let sum_grad = grad_output_hat.iter().sum::<f32>();
            let sum_grad_norm = grad_output_hat
                .iter()
                .zip(normalized.iter())
                .map(|(g, n)| g * n)
                .sum::<f32>();

            for i in 0..3 {
                let numerator = grad_output_hat[i] * m - sum_grad - normalized[i] * sum_grad_norm;
                expected_input_grad[start + i] += numerator * inv_std / m;
                expected_weight_grad[i] += grad_output[i] * normalized[i];
                expected_bias_grad[i] += grad_output[i];
            }
        }

        let input_grad_vals = grad_input.data().as_f32_slice().unwrap();
        let weight_grad_vals = grad_weight.data().as_f32_slice().unwrap();
        let bias_grad_vals = grad_bias.data().as_f32_slice().unwrap();

        for (actual, expected) in input_grad_vals.iter().zip(expected_input_grad.iter()) {
            assert!((actual - expected).abs() < 1e-5);
        }

        for (actual, expected) in weight_grad_vals.iter().zip(expected_weight_grad.iter()) {
            assert!((actual - expected).abs() < 1e-5);
        }

        for (actual, expected) in bias_grad_vals.iter().zip(expected_bias_grad.iter()) {
            assert!((actual - expected).abs() < 1e-6);
        }
    }
}
