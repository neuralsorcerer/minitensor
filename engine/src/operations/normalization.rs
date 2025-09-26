// Copyright (c) 2025 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

use crate::autograd::{LayerNormBackward, TensorId, add_to_graph};
use crate::device::Device;
use crate::error::{MinitensorError, Result};
use crate::tensor::{DataType, Shape, Tensor, TensorData};
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
///             is interpreted as the feature/channel dimension.
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
    if let Some(w) = weight {
        if w.ndim() != 1 || w.size(0)? != num_features {
            return Err(MinitensorError::shape_mismatch(
                vec![num_features],
                vec![w.size(0)?],
            ));
        }
    }
    if let Some(b) = bias {
        if b.ndim() != 1 || b.size(0)? != num_features {
            return Err(MinitensorError::shape_mismatch(
                vec![num_features],
                vec![b.size(0)?],
            ));
        }
    }
    if let Some(rm) = &running_mean {
        if rm.ndim() != 1 || rm.size(0)? != num_features {
            return Err(MinitensorError::shape_mismatch(
                vec![num_features],
                vec![rm.size(0)?],
            ));
        }
    }
    if let Some(rv) = &running_var {
        if rv.ndim() != 1 || rv.size(0)? != num_features {
            return Err(MinitensorError::shape_mismatch(
                vec![num_features],
                vec![rv.size(0)?],
            ));
        }
    }

    // Dimensions along which to compute statistics (all except channel dim)
    let axes: Vec<usize> = (0..input.ndim()).filter(|&d| d != 1).collect();

    // Compute batch statistics if needed
    let axes_isize: Vec<isize> = axes.iter().map(|&d| d as isize).collect();
    let batch_mean = input.mean(Some(axes_isize.clone()), true)?; // [1, C, ...]
    let centered = crate::operations::arithmetic::sub(input, &batch_mean)?;
    let batch_var = crate::operations::arithmetic::mul(&centered, &centered)?
        .mean(Some(axes_isize.clone()), true)?;

    // Decide which statistics to use
    let (mean_used, var_used) = if training || running_mean.is_none() || running_var.is_none() {
        (batch_mean.clone(), batch_var.clone())
    } else if let (Some(rm), Some(rv)) = (running_mean.as_ref(), running_var.as_ref()) {
        // Use running statistics (reshape for broadcasting)
        let mut rm_view = (*rm).clone().unsqueeze(0)?; // [1, C]
        let mut rv_view = (*rv).clone().unsqueeze(0)?;
        for _ in 2..input.ndim() {
            rm_view = rm_view.unsqueeze(rm_view.ndim() as isize)?;
            rv_view = rv_view.unsqueeze(rv_view.ndim() as isize)?;
        }
        (rm_view, rv_view)
    } else {
        unreachable!("running stats checked")
    };

    // Prepare epsilon tensor
    let eps_tensor = scalar_tensor(eps, input.dtype(), input.device())?;

    let var_eps = crate::operations::arithmetic::add(&var_used, &eps_tensor)?;
    let std = crate::operations::activation::sqrt(&var_eps)?;
    let centered = crate::operations::arithmetic::sub(input, &mean_used)?;
    let mut output = crate::operations::arithmetic::div(&centered, &std)?;

    // Scale and shift
    if let Some(w) = weight {
        let mut w_view = w.clone().unsqueeze(0)?;
        for _ in 2..input.ndim() {
            w_view = w_view.unsqueeze(w_view.ndim() as isize)?;
        }
        output = crate::operations::arithmetic::mul(&output, &w_view)?;
    }
    if let Some(b) = bias {
        let mut b_view = b.clone().unsqueeze(0)?;
        for _ in 2..input.ndim() {
            b_view = b_view.unsqueeze(b_view.ndim() as isize)?;
        }
        output = crate::operations::arithmetic::add(&output, &b_view)?;
    }

    // Update running statistics if training
    if training {
        if let (Some(rm), Some(rv)) = (running_mean, running_var) {
            let mean_flat = batch_mean.view(Shape::new(vec![num_features]))?.detach();
            let var_flat = batch_var.view(Shape::new(vec![num_features]))?.detach();

            let m_tensor = scalar_tensor(momentum, input.dtype(), input.device())?;
            let one_minus_tensor = scalar_tensor(1.0 - momentum, input.dtype(), input.device())?;

            *rm = crate::operations::arithmetic::add(
                &crate::operations::arithmetic::mul(rm, &one_minus_tensor)?,
                &crate::operations::arithmetic::mul(&mean_flat, &m_tensor)?,
            )?;
            *rv = crate::operations::arithmetic::add(
                &crate::operations::arithmetic::mul(rv, &one_minus_tensor)?,
                &crate::operations::arithmetic::mul(&var_flat, &m_tensor)?,
            )?;
        }
    }

    Ok(output)
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

    let axes: Vec<usize> = (axis_start..input.ndim()).collect();
    let axes_isize: Vec<isize> = axes.iter().map(|&d| d as isize).collect();
    let mean = input.mean(Some(axes_isize.clone()), true)?;
    let centered = crate::operations::arithmetic::sub(input, &mean)?;
    let var = crate::operations::arithmetic::mul(&centered, &centered)?
        .mean(Some(axes_isize.clone()), true)?;
    let eps_tensor = scalar_tensor(eps, input.dtype(), input.device())?;
    let var_eps = crate::operations::arithmetic::add(&var, &eps_tensor)?;
    let std = crate::operations::activation::sqrt(&var_eps)?;
    let ones = Tensor::ones(std.shape().clone(), std.dtype(), std.device(), false);
    let inv_std = crate::operations::arithmetic::div(&ones, &std)?;
    let normalized = crate::operations::arithmetic::mul(&centered, &inv_std)?;

    let mut output = normalized.clone();
    let mut weight_broadcast: Option<Tensor> = None;
    if let Some(w) = weight {
        let mut view = w.clone();
        for _ in 0..axis_start {
            view = view.unsqueeze(0)?;
        }
        output = crate::operations::arithmetic::mul(&output, &view)?;
        weight_broadcast = Some(view.detach());
    }
    if let Some(b) = bias {
        let mut view = b.clone();
        for _ in 0..axis_start {
            view = view.unsqueeze(0)?;
        }
        output = crate::operations::arithmetic::add(&output, &view)?;
    }

    let requires_grad = input.requires_grad()
        || weight.map(|w| w.requires_grad()).unwrap_or(false)
        || bias.map(|b| b.requires_grad()).unwrap_or(false);

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
        let grads = autograd::backward(&result, Some(ones)).unwrap();

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
