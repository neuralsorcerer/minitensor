// Copyright (c) 2025 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

use crate::error::{MinitensorError, Result};
use crate::tensor::{DataType, Shape, Tensor, TensorData};
use std::sync::Arc;

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
    let batch_mean = input.mean(Some(axes.clone()), true)?; // [1, C, ...]
    let centered = crate::operations::arithmetic::sub(input, &batch_mean)?;
    let batch_var =
        crate::operations::arithmetic::mul(&centered, &centered)?.mean(Some(axes.clone()), true)?;

    // Decide which statistics to use
    let (mean_used, var_used) = if training || running_mean.is_none() || running_var.is_none() {
        (batch_mean.clone(), batch_var.clone())␊
    } else if let (Some(rm), Some(rv)) = (running_mean.as_ref(), running_var.as_ref()) {
        // Use running statistics (reshape for broadcasting)
        let mut rm_view = (*rm).clone().unsqueeze(0)?; // [1, C]
        let mut rv_view = (*rv).clone().unsqueeze(0)?;
        for _ in 2..input.ndim() {
            rm_view = rm_view.unsqueeze(rm_view.ndim())?;
            rv_view = rv_view.unsqueeze(rv_view.ndim())?;
        }
        (rm_view, rv_view)
    } else {
        unreachable!("running stats checked")
    };

    // Prepare epsilon tensor
    let eps_tensor = {
        let mut data = TensorData::zeros_on_device(1, input.dtype(), input.device());
        match input.dtype() {
            DataType::Float32 => {
                let slice = data.as_f32_slice_mut().unwrap();
                slice[0] = eps as f32;
            }
            DataType::Float64 => {
                let slice = data.as_f64_slice_mut().unwrap();
                slice[0] = eps;
            }
            _ => unreachable!("BatchNorm only supports floating types"),
        }
        Tensor::new(
            Arc::new(data),
            Shape::new(vec![1]),
            input.dtype(),
            input.device(),
            false,
        )
    };

    let var_eps = crate::operations::arithmetic::add(&var_used, &eps_tensor)?;
    let std = crate::operations::activation::sqrt(&var_eps)?;
    let centered = crate::operations::arithmetic::sub(input, &mean_used)?;
    let mut output = crate::operations::arithmetic::div(&centered, &std)?;

    // Scale and shift
    if let Some(w) = weight {
        let mut w_view = w.clone().unsqueeze(0)?;
        for _ in 2..input.ndim() {
            w_view = w_view.unsqueeze(w_view.ndim())?;
        }
        output = crate::operations::arithmetic::mul(&output, &w_view)?;
    }
    if let Some(b) = bias {
        let mut b_view = b.clone().unsqueeze(0)?;
        for _ in 2..input.ndim() {
            b_view = b_view.unsqueeze(b_view.ndim())?;
        }
        output = crate::operations::arithmetic::add(&output, &b_view)?;
    }

    // Update running statistics if training
    if training {
        if let (Some(rm), Some(rv)) = (running_mean, running_var) {
            let mean_flat = batch_mean.view(Shape::new(vec![num_features]))?.detach();
            let var_flat = batch_var.view(Shape::new(vec![num_features]))?.detach();

            let m_tensor = {
                let mut data = TensorData::zeros_on_device(1, input.dtype(), input.device());
                match input.dtype() {
                    DataType::Float32 => {
                        let slice = data.as_f32_slice_mut().unwrap();
                        slice[0] = momentum as f32;
                    }
                    DataType::Float64 => {
                        let slice = data.as_f64_slice_mut().unwrap();
                        slice[0] = momentum;
                    }
                    _ => unreachable!(),
                }
                Tensor::new(
                    Arc::new(data),
                    Shape::new(vec![1]),
                    input.dtype(),
                    input.device(),
                    false,
                )
            };
            let one_minus_tensor = {
                let mut data = TensorData::zeros_on_device(1, input.dtype(), input.device());
                match input.dtype() {
                    DataType::Float32 => {
                        let slice = data.as_f32_slice_mut().unwrap();
                        slice[0] = (1.0 - momentum) as f32;
                    }
                    DataType::Float64 => {
                        let slice = data.as_f64_slice_mut().unwrap();
                        slice[0] = 1.0 - momentum;
                    }
                    _ => unreachable!(),
                }
                Tensor::new(
                    Arc::new(data),
                    Shape::new(vec![1]),
                    input.dtype(),
                    input.device(),
                    false,
                )
            };

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
