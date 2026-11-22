// Copyright (c) Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

use crate::autograd::with_grad_fn;
use crate::autograd::{GroupNormBackward, LayerNormBackward, RmsNormBackward, TensorId};
use crate::device::Device;
use crate::error::{MinitensorError, Result};
use crate::tensor::{DataType, Shape, Tensor, TensorData};
use smallvec::SmallVec;
use std::sync::Arc;

/// Refuse an `eps` that is negative or not finite.
///
/// `eps` is added to a variance before its square root, so a negative one
/// takes the root of a negative number wherever a feature varies less than it:
/// `LayerNorm(eps=-1)` answered NaN for every input of small spread. NaN or
/// infinity is never a stabilizer either. Zero is allowed; it asks for no
/// stabilization, and a constant row then divides by zero, as asked.
pub fn check_eps(op: &str, eps: f64) -> Result<()> {
    if eps.is_finite() && eps >= 0.0 {
        Ok(())
    } else {
        Err(MinitensorError::invalid_argument(format!(
            "{op} requires eps to be finite and non-negative, got {eps}"
        )))
    }
}

/// Refuse a running-statistics `momentum` outside `[0, 1]`.
///
/// The running estimate is `(1 - momentum) * running + momentum * batch`, an
/// average only while `momentum` is in `[0, 1]`: at 2 it doubles the batch's
/// share and subtracts the history, and the estimate diverges.
pub fn check_momentum(op: &str, momentum: f64) -> Result<()> {
    if (0.0..=1.0).contains(&momentum) {
        Ok(())
    } else {
        Err(MinitensorError::invalid_argument(format!(
            "{op} requires momentum to lie in [0, 1], got {momentum}"
        )))
    }
}

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
    check_eps("batch_norm", eps)?;
    check_momentum("batch_norm", momentum)?;

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
    // One value per channel has no spread to normalize by: every output came
    // back exactly zero, whatever the input, and the running variance was
    // pulled toward zero, which inflates every later output in eval mode.
    if use_batch_stats && num_features > 0 && input.numel() == num_features {
        return Err(MinitensorError::invalid_argument(format!(
            "batch_norm needs more than one value per channel to compute batch \
             statistics, and this input of shape {:?} has one; use a larger \
             batch, or evaluation mode with running statistics",
            input.shape().dims()
        )));
    }
    // A float input whose parameters share its dtype -- every layer -- takes
    // the fused kernel. What remains is composed below: an integer input,
    // which widens through the operations, mixed dtypes, which promote, and an
    // empty batch.
    let (running_mean, running_var) = match crate::ops::batch_norm::fused_batch_norm(
        input,
        running_mean,
        running_var,
        weight,
        bias,
        use_batch_stats,
        training,
        momentum,
        eps,
    ) {
        Ok(output) => return output,
        Err(stats) => stats,
    };
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
    // batch statistics whenever training is true). Not from an empty batch:
    // it says nothing about the distribution, and its mean and variance are
    // the NaN of an average over nothing -- folded into the running estimates,
    // that NaN stayed there and made every later output in eval mode NaN.
    if training
        && input.numel() > 0
        && let (Some(rm), Some(rv)) = (running_mean, running_var)
    {
        let mean_flat = mean_used.view(Shape::new(vec![num_features]))?.detach();
        let var_flat = var_used.view(Shape::new(vec![num_features]))?.detach();

        // `running_var` stores the *unbiased* batch variance (Bessel's
        // correction: var_biased * n / (n - 1)), even though the normalization
        // above deliberately uses the biased estimate. Apply the correction so
        // the eval-time statistics are the unbiased ones. `n` is the number of
        // elements reduced per channel; the degenerate n == 1 case has no
        // unbiased variance, so it is left uncorrected.
        // No channels means no elements; there is nothing to correct.
        let count = input.numel().checked_div(num_features).unwrap_or(0);
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

/// Each `norm`-wide row of `input` standardized -- less its mean where
/// `centred` -- and times `weight` plus `bias` where given, each the value
/// nearest its exact one, as a tensor's data. Each parameter is `period`
/// rows of a value for each run of `spread` columns, row `r` taking row
/// `r % period`.
fn standardized_data<T>(
    input: &Tensor,
    layout: (usize, usize, usize),
    weight: Option<&Tensor>,
    bias: Option<&Tensor>,
    (eps, centred): (f64, bool),
) -> Result<TensorData>
where
    T: crate::ops::arithmetic::exact_sum::Element
        + crate::ops::arithmetic::exact_sum::Output
        + crate::ops::batch_norm::NormFloat,
{
    fn slice<T: crate::ops::batch_norm::NormFloat>(tensor: &Tensor) -> Result<&[T]> {
        T::slice(tensor.data()).ok_or_else(|| {
            MinitensorError::internal_error("normalization: an operand of the wrong dtype")
        })
    }
    let (w, b) = (weight.map(slice).transpose()?, bias.map(slice).transpose()?);
    let mut out = vec![T::narrow(0.0); input.numel()];
    crate::ops::arithmetic::exact_sum::standardized_rows::<T, T>(
        slice(input)?,
        layout,
        w,
        b,
        (eps, centred),
        &mut out,
    );
    Ok(<T as crate::ops::batch_norm::NormFloat>::into_data(
        out,
        input.device(),
    ))
}

/// Apply layer normalization to the input tensor.
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
    check_eps("layer_norm", eps)?;

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

    let requires_grad = crate::autograd::is_grad_enabled()
        && (input.requires_grad()
            || weight.map(|w| w.requires_grad()).unwrap_or(false)
            || bias.map(|b| b.requires_grad()).unwrap_or(false));

    // Each output the value nearest its exact one; see `standardized_rows`.
    let output_data = match input.dtype() {
        DataType::Float32 => {
            standardized_data::<f32>(input, (norm, 1, 1), weight, bias, (eps, true))?
        }
        _ => standardized_data::<f64>(input, (norm, 1, 1), weight, bias, (eps, true))?,
    };

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
        input: input.detach(),
        weight: weight.map(|w| w.detach()),
        normalized_shape: normalized_shape.to_vec(),
        axis_start,
        element_count: normalized_shape.iter().product(),
        eps,
        input_requires_grad: input.requires_grad(),
        weight_requires_grad: weight.map(|w| w.requires_grad()).unwrap_or(false),
        bias_requires_grad: bias.map(|b| b.requires_grad()).unwrap_or(false),
    });

    with_grad_fn(output, grad_fn)
}

/// Group normalization of a `[N, C, ...]` input: each sample's channels in
/// `groups` runs of `C / groups`, each run -- its channels at every position
/// -- standardized over itself, and then each value times its channel's
/// weight plus its bias. Each output is the value nearest its exact one; see
/// `standardized_rows`. `instance_norm` is this with a group per channel.
///
/// It was `layer_norm` over each run and then the weight and bias as tensor
/// operations, the standardized value rounded before they were applied:
/// where the bias all but cancels the product, that rounding was what was
/// left, 1.2e9 ulps out in float32 and 3.7e17 in float64, and ordinary
/// outputs were up to 11 ulps out.
pub fn group_norm(
    input: &Tensor,
    groups: usize,
    weight: Option<&Tensor>,
    bias: Option<&Tensor>,
    eps: f64,
) -> Result<Tensor> {
    check_eps("group_norm", eps)?;
    if !matches!(input.dtype(), DataType::Float32 | DataType::Float64) {
        return Err(MinitensorError::invalid_operation(
            "group_norm only supports floating point tensors".to_string(),
        ));
    }
    let dims = input.shape().dims();
    if dims.len() < 2 {
        return Err(MinitensorError::invalid_argument(format!(
            "group_norm expects a batch and a channel axis, got {} dimensions",
            dims.len()
        )));
    }
    let channels = dims[1];
    if groups == 0 || !channels.is_multiple_of(groups) {
        return Err(MinitensorError::invalid_argument(format!(
            "group_norm needs the channel count ({channels}) to divide by the number of \
             groups ({groups})"
        )));
    }
    for parameter in [weight, bias].into_iter().flatten() {
        if parameter.dtype() != input.dtype() {
            return Err(MinitensorError::type_mismatch(
                input.dtype().to_string(),
                parameter.dtype().to_string(),
            ));
        }
        if parameter.device() != input.device() {
            return Err(MinitensorError::device_mismatch(
                input.device().to_string(),
                parameter.device().to_string(),
            ));
        }
        if parameter.numel() != channels {
            return Err(MinitensorError::shape_mismatch(
                vec![channels],
                parameter.shape().dims().to_vec(),
            ));
        }
    }
    let spatial: usize = dims[2..].iter().product();
    let layout = ((channels / groups) * spatial, groups, spatial);
    let data = match input.dtype() {
        DataType::Float32 => standardized_data::<f32>(input, layout, weight, bias, (eps, true))?,
        _ => standardized_data::<f64>(input, layout, weight, bias, (eps, true))?,
    };
    let requires_grad = crate::autograd::is_grad_enabled()
        && (input.requires_grad()
            || weight.is_some_and(|w| w.requires_grad())
            || bias.is_some_and(|b| b.requires_grad()));
    let output = Tensor::new(
        Arc::new(data),
        input.shape().clone(),
        input.dtype(),
        input.device(),
        requires_grad,
    );
    if !requires_grad {
        return Ok(output);
    }
    let input_ids: SmallVec<[TensorId; 3]> = [Some(input), weight, bias]
        .into_iter()
        .flatten()
        .map(|t| t.id())
        .collect();
    let grad_fn = Arc::new(GroupNormBackward {
        input_ids,
        input_id: input.id(),
        weight_id: weight.map(|w| w.id()),
        bias_id: bias.map(|b| b.id()),
        input: input.detach(),
        weight: weight.map(|w| w.detach()),
        bias_shape: bias.map(|b| b.shape().clone()),
        layout,
        eps,
        input_requires_grad: input.requires_grad(),
        weight_requires_grad: weight.is_some_and(|w| w.requires_grad()),
        bias_requires_grad: bias.is_some_and(|b| b.requires_grad()),
    });
    with_grad_fn(output, grad_fn)
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
/// centering) while matching or improving training stability. Each output is
/// the value nearest its exact one, as `layer_norm`'s is, and its gradient
/// comes from [`RmsNormBackward`].
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
    check_eps("rms_norm", eps)?;
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

    let output_data = match input.dtype() {
        DataType::Float32 => {
            standardized_data::<f32>(input, (norm, 1, 1), weight, None, (eps, false))?
        }
        _ => standardized_data::<f64>(input, (norm, 1, 1), weight, None, (eps, false))?,
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
        weight: weight.map(|w| w.detach()),
        normalized_shape: normalized_shape.to_vec(),
        element_count: norm,
        eps,
        input_requires_grad: input.requires_grad(),
        weight_requires_grad: weight.map(|w| w.requires_grad()).unwrap_or(false),
    });

    with_grad_fn(output, grad_fn)
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
