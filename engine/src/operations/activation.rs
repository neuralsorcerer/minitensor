// Copyright (c) 2025 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

use crate::{
    autograd::{
        add_to_graph, CosBackward, ExpBackward, LeakyReluBackward, LogBackward, PowBackward,
        ReluBackward, SigmoidBackward, SinBackward, SoftmaxBackward, TanBackward,
        TanhBackward,
    },
    error::{MinitensorError, Result},
    tensor::{DataType, Tensor, TensorData},
};
use std::sync::Arc;

/// Exponential function with gradient support
pub fn exp(tensor: &Tensor) -> Result<Tensor> {
    // Create output tensor data
    let mut output_data =
        TensorData::zeros_on_device(tensor.numel(), tensor.dtype(), tensor.device());

    // Perform exponential based on data type
    match tensor.dtype() {
        DataType::Float32 => exp_f32(tensor, &mut output_data)?,
        DataType::Float64 => exp_f64(tensor, &mut output_data)?,
        _ => {
            return Err(MinitensorError::invalid_operation(
                "Exponential function only supported for floating point tensors",
            ))
        }
    }

    // Create output tensor
    let output = Tensor::new(
        Arc::new(output_data),
        tensor.shape().clone(),
        tensor.dtype(),
        tensor.device(),
        tensor.requires_grad(),
    );

    // Set up gradient function if needed
    if output.requires_grad() {
        let grad_fn = Arc::new(ExpBackward {
            input_id: tensor.id(),
            output: output.clone().detach(),
        });

        let mut output_with_grad = output;
        output_with_grad.set_grad_fn(Some(grad_fn.clone()));

        // Add to computation graph
        add_to_graph(&output_with_grad, Some(grad_fn))?;

        Ok(output_with_grad)
    } else {
        Ok(output)
    }
}

/// Natural logarithm function with gradient support
pub fn log(tensor: &Tensor) -> Result<Tensor> {
    // Create output tensor data
    let mut output_data =
        TensorData::zeros_on_device(tensor.numel(), tensor.dtype(), tensor.device());

    // Perform logarithm based on data type
    match tensor.dtype() {
        DataType::Float32 => log_f32(tensor, &mut output_data)?,
        DataType::Float64 => log_f64(tensor, &mut output_data)?,
        _ => {
            return Err(MinitensorError::invalid_operation(
                "Logarithm function only supported for floating point tensors",
            ))
        }
    }

    // Create output tensor
    let output = Tensor::new(
        Arc::new(output_data),
        tensor.shape().clone(),
        tensor.dtype(),
        tensor.device(),
        tensor.requires_grad(),
    );

    // Set up gradient function if needed
    if output.requires_grad() {
        let grad_fn = Arc::new(LogBackward {
            input_id: tensor.id(),
            input: tensor.clone().detach(),
        });

        let mut output_with_grad = output;
        output_with_grad.set_grad_fn(Some(grad_fn.clone()));

        // Add to computation graph
        add_to_graph(&output_with_grad, Some(grad_fn))?;

        Ok(output_with_grad)
    } else {
        Ok(output)
    }
}

/// Sine function with gradient support
pub fn sin(tensor: &Tensor) -> Result<Tensor> {
    // Create output tensor data
    let mut output_data =
        TensorData::zeros_on_device(tensor.numel(), tensor.dtype(), tensor.device());

    // Perform sine based on data type
    match tensor.dtype() {
        DataType::Float32 => sin_f32(tensor, &mut output_data)?,
        DataType::Float64 => sin_f64(tensor, &mut output_data)?,
        _ => {
            return Err(MinitensorError::invalid_operation(
                "Sine function only supported for floating point tensors",
            ))
        }
    }

    // Create output tensor
    let output = Tensor::new(
        Arc::new(output_data),
        tensor.shape().clone(),
        tensor.dtype(),
        tensor.device(),
        tensor.requires_grad(),
    );

    // Set up gradient function if needed
    if output.requires_grad() {
        let grad_fn = Arc::new(SinBackward {
            input_id: tensor.id(),
            input: tensor.clone(),
        });

        let mut output_with_grad = output;
        output_with_grad.set_grad_fn(Some(grad_fn.clone()));

        // Add to computation graph
        add_to_graph(&output_with_grad, Some(grad_fn))?;

        Ok(output_with_grad)
    } else {
        Ok(output)
    }
}

/// Cosine function with gradient support
pub fn cos(tensor: &Tensor) -> Result<Tensor> {
    // Create output tensor data
    let mut output_data =
        TensorData::zeros_on_device(tensor.numel(), tensor.dtype(), tensor.device());

    // Perform cosine based on data type
    match tensor.dtype() {
        DataType::Float32 => cos_f32(tensor, &mut output_data)?,
        DataType::Float64 => cos_f64(tensor, &mut output_data)?,
        _ => {
            return Err(MinitensorError::invalid_operation(
                "Cosine function only supported for floating point tensors",
            ))
        }
    }

    // Create output tensor
    let output = Tensor::new(
        Arc::new(output_data),
        tensor.shape().clone(),
        tensor.dtype(),
        tensor.device(),
        tensor.requires_grad(),
    );

    // Set up gradient function if needed
    if output.requires_grad() {
        let grad_fn = Arc::new(CosBackward {
            input_id: tensor.id(),
            input: tensor.clone(),
        });

        let mut output_with_grad = output;
        output_with_grad.set_grad_fn(Some(grad_fn.clone()));

        // Add to computation graph
        add_to_graph(&output_with_grad, Some(grad_fn))?;

        Ok(output_with_grad)
    } else {
        Ok(output)
    }
}

/// Tangent function with gradient support
pub fn tan(tensor: &Tensor) -> Result<Tensor> {
    // Create output tensor data
    let mut output_data =
        TensorData::zeros_on_device(tensor.numel(), tensor.dtype(), tensor.device());

    // Perform tangent based on data type
    match tensor.dtype() {
        DataType::Float32 => tan_f32(tensor, &mut output_data)?,
        DataType::Float64 => tan_f64(tensor, &mut output_data)?,
        _ => {
            return Err(MinitensorError::invalid_operation(
                "Tangent function only supported for floating point tensors",
            ))
        }
    }

    // Create output tensor
    let output = Tensor::new(
        Arc::new(output_data),
        tensor.shape().clone(),
        tensor.dtype(),
        tensor.device(),
        tensor.requires_grad(),
    );

    // Set up gradient function if needed
    if output.requires_grad() {
        let grad_fn = Arc::new(TanBackward {
            input_id: tensor.id(),
            output: output.clone().detach(),
        });

        let mut output_with_grad = output;
        output_with_grad.set_grad_fn(Some(grad_fn.clone()));

        // Add to computation graph
        add_to_graph(&output_with_grad, Some(grad_fn))?;

        Ok(output_with_grad)
    } else {
        Ok(output)
    }
}

/// Hyperbolic tangent function with gradient support
pub fn tanh(tensor: &Tensor) -> Result<Tensor> {
    // Create output tensor data
    let mut output_data =
        TensorData::zeros_on_device(tensor.numel(), tensor.dtype(), tensor.device());

    // Perform tanh based on data type
    match tensor.dtype() {
        DataType::Float32 => tanh_f32(tensor, &mut output_data)?,
        DataType::Float64 => tanh_f64(tensor, &mut output_data)?,
        _ => {
            return Err(MinitensorError::invalid_operation(
                "Tanh function only supported for floating point tensors",
            ))
        }
    }

    // Create output tensor
    let output = Tensor::new(
        Arc::new(output_data),
        tensor.shape().clone(),
        tensor.dtype(),
        tensor.device(),
        tensor.requires_grad(),
    );

    // Set up gradient function if needed
    if output.requires_grad() {
        let grad_fn = Arc::new(TanhBackward {
            input_id: tensor.id(),
            output: output.clone(),
        });

        let mut output_with_grad = output;
        output_with_grad.set_grad_fn(Some(grad_fn.clone()));

        // Add to computation graph
        add_to_graph(&output_with_grad, Some(grad_fn))?;

        Ok(output_with_grad)
    } else {
        Ok(output)
    }
}

/// Sigmoid activation function with gradient support
pub fn sigmoid(tensor: &Tensor) -> Result<Tensor> {
    // Create output tensor data
    let mut output_data =
        TensorData::zeros_on_device(tensor.numel(), tensor.dtype(), tensor.device());

    // Perform sigmoid based on data type
    match tensor.dtype() {
        DataType::Float32 => sigmoid_f32(tensor, &mut output_data)?,
        DataType::Float64 => sigmoid_f64(tensor, &mut output_data)?,
        _ => {
            return Err(MinitensorError::invalid_operation(
                "Sigmoid function only supported for floating point tensors",
            ))
        }
    }

    // Create output tensor
    let output = Tensor::new(
        Arc::new(output_data),
        tensor.shape().clone(),
        tensor.dtype(),
        tensor.device(),
        tensor.requires_grad(),
    );

    // Set up gradient function if needed
    if output.requires_grad() {
        let grad_fn = Arc::new(SigmoidBackward {
            input_id: tensor.id(),
            output: output.clone(),
        });

        let mut output_with_grad = output;
        output_with_grad.set_grad_fn(Some(grad_fn.clone()));

        // Add to computation graph
        add_to_graph(&output_with_grad, Some(grad_fn))?;

        Ok(output_with_grad)
    } else {
        Ok(output)
    }
}

/// Element-wise power with tensor exponent and gradient support
pub fn pow(base: &Tensor, exponent: &Tensor) -> Result<Tensor> {
    // Check device and dtype compatibility
    if base.device() != exponent.device() {
        return Err(MinitensorError::device_mismatch(
            format!("{:?}", base.device()),
            format!("{:?}", exponent.device()),
        ));
    }

    if base.dtype() != exponent.dtype() {
        return Err(MinitensorError::type_mismatch(
            format!("{:?}", base.dtype()),
            format!("{:?}", exponent.dtype()),
        ));
    }

    // For simplicity, require identical shapes
    if base.shape() != exponent.shape() {
        return Err(MinitensorError::shape_mismatch(
            base.shape().dims().to_vec(),
            exponent.shape().dims().to_vec(),
        ));
    }

    // Create output tensor data
    let mut output_data = TensorData::zeros_on_device(base.numel(), base.dtype(), base.device());

    match base.dtype() {
        DataType::Float32 => {
            let b = base.data().as_f32_slice().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get f32 slice from base tensor")
            })?;
            let e = exponent.data().as_f32_slice().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get f32 slice from exponent tensor")
            })?;
            let out = output_data.as_f32_slice_mut().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get mutable f32 slice from output data")
            })?;
            for i in 0..b.len() {
                out[i] = b[i].powf(e[i]);
            }
        }
        DataType::Float64 => {
            let b = base.data().as_f64_slice().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get f64 slice from base tensor")
            })?;
            let e = exponent.data().as_f64_slice().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get f64 slice from exponent tensor")
            })?;
            let out = output_data.as_f64_slice_mut().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get mutable f64 slice from output data")
            })?;
            for i in 0..b.len() {
                out[i] = b[i].powf(e[i]);
            }
        }
        _ => {
            return Err(MinitensorError::invalid_operation(
                "Power operation only supported for floating point tensors",
            ));
        }
    }

    let output = Tensor::new(
        Arc::new(output_data),
        base.shape().clone(),
        base.dtype(),
        base.device(),
        base.requires_grad() || exponent.requires_grad(),
    );

    if output.requires_grad() {
        let grad_fn = Arc::new(PowBackward {
            base: base.detach(),
            exponent: exponent.detach(),
            output: output.clone().detach(),
            input_ids: [base.id(), exponent.id()],
            base_requires_grad: base.requires_grad(),
            exp_requires_grad: exponent.requires_grad(),
        });

        let mut output_with_grad = output;
        output_with_grad.set_grad_fn(Some(grad_fn.clone()));

        add_to_graph(&output_with_grad, Some(grad_fn))?;
        Ok(output_with_grad)
    } else {
        Ok(output)
    }
}

/// Element-wise power with scalar exponent and gradient support
pub fn powf(tensor: &Tensor, exponent: f64) -> Result<Tensor> {
    // Create exponent tensor filled with scalar value
    let mut exp_data = TensorData::zeros_on_device(tensor.numel(), tensor.dtype(), tensor.device());
    match tensor.dtype() {
        DataType::Float32 => {
            let slice = exp_data.as_f32_slice_mut().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get mutable f32 slice from exponent data")
            })?;
            for val in slice.iter_mut() {
                *val = exponent as f32;
            }
        }
        DataType::Float64 => {
            let slice = exp_data.as_f64_slice_mut().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get mutable f64 slice from exponent data")
            })?;
            for val in slice.iter_mut() {
                *val = exponent;
            }
        }
        _ => {
            return Err(MinitensorError::invalid_operation(
                "Power operation only supported for floating point tensors",
            ));
        }
    }
    let exp_tensor = Tensor::new(
        Arc::new(exp_data),
        tensor.shape().clone(),
        tensor.dtype(),
        tensor.device(),
        false,
    );
    pow(tensor, &exp_tensor)
}

/// ReLU activation function with gradient support
pub fn relu(tensor: &Tensor) -> Result<Tensor> {
    // Create output tensor data
    let mut output_data =
        TensorData::zeros_on_device(tensor.numel(), tensor.dtype(), tensor.device());

    // Perform ReLU based on data type
    match tensor.dtype() {
        DataType::Float32 => relu_f32(tensor, &mut output_data)?,
        DataType::Float64 => relu_f64(tensor, &mut output_data)?,
        DataType::Int32 => relu_i32(tensor, &mut output_data)?,
        DataType::Int64 => relu_i64(tensor, &mut output_data)?,
        DataType::Bool => {
            return Err(MinitensorError::invalid_operation(
                "ReLU function not supported for boolean tensors",
            ))
        }
    }

    // Create output tensor
    let output = Tensor::new(
        Arc::new(output_data),
        tensor.shape().clone(),
        tensor.dtype(),
        tensor.device(),
        tensor.requires_grad(),
    );

    // Prepare mask for gradient computation
    let mask: Vec<bool> = match tensor.dtype() {
        DataType::Float32 => {
            let input_data = tensor.data().as_f32_slice().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get f32 slice from input tensor")
            })?;
            input_data.iter().map(|&v| v > 0.0).collect()
        }
        DataType::Float64 => {
            let input_data = tensor.data().as_f64_slice().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get f64 slice from input tensor")
            })?;
            input_data.iter().map(|&v| v > 0.0).collect()
        }
        DataType::Int32 => {
            let input_data = tensor.data().as_i32_slice().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get i32 slice from input tensor")
            })?;
            input_data.iter().map(|&v| v > 0).collect()
        }
        DataType::Int64 => {
            let input_data = tensor.data().as_i64_slice().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get i64 slice from input tensor")
            })?;
            input_data.iter().map(|&v| v > 0).collect()
        }
        DataType::Bool => {
            return Err(MinitensorError::invalid_operation(
                "ReLU function not supported for boolean tensors",
            ))
        }
    };

    // Set up gradient function if needed
    if output.requires_grad() {
        let grad_fn = Arc::new(ReluBackward {
            input_id: tensor.id(),
            mask,
        });

        let mut output_with_grad = output;
        output_with_grad.set_grad_fn(Some(grad_fn.clone()));

        // Add to computation graph
        add_to_graph(&output_with_grad, Some(grad_fn))?;

        Ok(output_with_grad)
    } else {
        Ok(output)
    }
}

/// LeakyReLU activation function with gradient support
pub fn leaky_relu(tensor: &Tensor, negative_slope: f64) -> Result<Tensor> {
    // Create output tensor data
    let mut output_data =
        TensorData::zeros_on_device(tensor.numel(), tensor.dtype(), tensor.device());

    // Perform LeakyReLU based on data type and capture mask of positive inputs
    let mask = match tensor.dtype() {
        DataType::Float32 => leaky_relu_f32(tensor, &mut output_data, negative_slope as f32)?,
        DataType::Float64 => leaky_relu_f64(tensor, &mut output_data, negative_slope)?,
        _ => {
            return Err(MinitensorError::invalid_operation(
                "LeakyReLU function only supported for floating point tensors",
            ))
        }
    };

    // Create output tensor
    let output = Tensor::new(
        Arc::new(output_data),
        tensor.shape().clone(),
        tensor.dtype(),
        tensor.device(),
        tensor.requires_grad(),
    );

    // Set up gradient function if needed
    if output.requires_grad() {
        let grad_fn = Arc::new(LeakyReluBackward {
            input_id: tensor.id(),
            negative_slope,
            mask,
        });

        let mut output_with_grad = output;
        output_with_grad.set_grad_fn(Some(grad_fn.clone()));

        // Add to computation graph
        add_to_graph(&output_with_grad, Some(grad_fn))?;

        Ok(output_with_grad)
    } else {
        Ok(output)
    }
}

/// Softmax activation function with gradient support
pub fn softmax(tensor: &Tensor, dim: Option<usize>) -> Result<Tensor> {
    let dim = dim.unwrap_or(tensor.ndim() - 1);

    if dim >= tensor.ndim() {
        return Err(MinitensorError::index_error(dim as isize, 0, tensor.ndim()));
    }

    // Create output tensor data
    let mut output_data =
        TensorData::zeros_on_device(tensor.numel(), tensor.dtype(), tensor.device());

    // Perform softmax based on data type
    match tensor.dtype() {
        DataType::Float32 => softmax_f32(tensor, &mut output_data, dim)?,
        DataType::Float64 => softmax_f64(tensor, &mut output_data, dim)?,
        _ => {
            return Err(MinitensorError::invalid_operation(
                "Softmax function only supported for floating point tensors",
            ))
        }
    }

    // Create output tensor
    let output = Tensor::new(
        Arc::new(output_data),
        tensor.shape().clone(),
        tensor.dtype(),
        tensor.device(),
        tensor.requires_grad(),
    );

    // Set up gradient function if needed
    if output.requires_grad() {
        let grad_fn = Arc::new(SoftmaxBackward {
            input_id: tensor.id(),
            output: output.detach(),
            dim,
        });

        let mut output_with_grad = output;
        output_with_grad.set_grad_fn(Some(grad_fn.clone()));

        // Add to computation graph
        add_to_graph(&output_with_grad, Some(grad_fn))?;

        Ok(output_with_grad)
    } else {
        Ok(output)
    }
}

// Helper functions for type-specific operations

fn exp_f32(tensor: &Tensor, output_data: &mut TensorData) -> Result<()> {
    let input_data = tensor.data().as_f32_slice().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get f32 slice from input tensor")
    })?;

    let output_slice = output_data.as_f32_slice_mut().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get mutable f32 slice from output data")
    })?;

    for (i, &val) in input_data.iter().enumerate() {
        output_slice[i] = val.exp();
    }

    Ok(())
}

fn exp_f64(tensor: &Tensor, output_data: &mut TensorData) -> Result<()> {
    let input_data = tensor.data().as_f64_slice().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get f64 slice from input tensor")
    })?;

    let output_slice = output_data.as_f64_slice_mut().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get mutable f64 slice from output data")
    })?;

    for (i, &val) in input_data.iter().enumerate() {
        output_slice[i] = val.exp();
    }

    Ok(())
}

fn log_f32(tensor: &Tensor, output_data: &mut TensorData) -> Result<()> {
    let input_data = tensor.data().as_f32_slice().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get f32 slice from input tensor")
    })?;

    let output_slice = output_data.as_f32_slice_mut().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get mutable f32 slice from output data")
    })?;

    for (i, &val) in input_data.iter().enumerate() {
        if val <= 0.0 {
            output_slice[i] = f32::NEG_INFINITY;
        } else {
            output_slice[i] = val.ln();
        }
    }

    Ok(())
}

fn log_f64(tensor: &Tensor, output_data: &mut TensorData) -> Result<()> {
    let input_data = tensor.data().as_f64_slice().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get f64 slice from input tensor")
    })?;

    let output_slice = output_data.as_f64_slice_mut().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get mutable f64 slice from output data")
    })?;

    for (i, &val) in input_data.iter().enumerate() {
        if val <= 0.0 {
            output_slice[i] = f64::NEG_INFINITY;
        } else {
            output_slice[i] = val.ln();
        }
    }

    Ok(())
}

fn sin_f32(tensor: &Tensor, output_data: &mut TensorData) -> Result<()> {
    let input_data = tensor.data().as_f32_slice().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get f32 slice from input tensor")
    })?;

    let output_slice = output_data.as_f32_slice_mut().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get mutable f32 slice from output data")
    })?;

    for (i, &val) in input_data.iter().enumerate() {
        output_slice[i] = val.sin();
    }

    Ok(())
}

fn sin_f64(tensor: &Tensor, output_data: &mut TensorData) -> Result<()> {
    let input_data = tensor.data().as_f64_slice().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get f64 slice from input tensor")
    })?;

    let output_slice = output_data.as_f64_slice_mut().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get mutable f64 slice from output data")
    })?;

    for (i, &val) in input_data.iter().enumerate() {
        output_slice[i] = val.sin();
    }

    Ok(())
}

fn cos_f32(tensor: &Tensor, output_data: &mut TensorData) -> Result<()> {
    let input_data = tensor.data().as_f32_slice().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get f32 slice from input tensor")
    })?;

    let output_slice = output_data.as_f32_slice_mut().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get mutable f32 slice from output data")
    })?;

    for (i, &val) in input_data.iter().enumerate() {
        output_slice[i] = val.cos();
    }

    Ok(())
}

fn cos_f64(tensor: &Tensor, output_data: &mut TensorData) -> Result<()> {
    let input_data = tensor.data().as_f64_slice().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get f64 slice from input tensor")
    })?;

    let output_slice = output_data.as_f64_slice_mut().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get mutable f64 slice from output data")
    })?;

    for (i, &val) in input_data.iter().enumerate() {
        output_slice[i] = val.cos();
    }

    Ok(())
}

fn tan_f32(tensor: &Tensor, output_data: &mut TensorData) -> Result<()> {
    let input_data = tensor.data().as_f32_slice().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get f32 slice from input tensor")
    })?;

    let output_slice = output_data.as_f32_slice_mut().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get mutable f32 slice from output data")
    })?;

    for (i, &val) in input_data.iter().enumerate() {
        output_slice[i] = val.tan();
    }

    Ok(())
}

fn tan_f64(tensor: &Tensor, output_data: &mut TensorData) -> Result<()> {
    let input_data = tensor.data().as_f64_slice().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get f64 slice from input tensor")
    })?;

    let output_slice = output_data.as_f64_slice_mut().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get mutable f64 slice from output data")
    })?;

    for (i, &val) in input_data.iter().enumerate() {
        output_slice[i] = val.tan();
    }

    Ok(())
}

fn tanh_f32(tensor: &Tensor, output_data: &mut TensorData) -> Result<()> {
    let input_data = tensor.data().as_f32_slice().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get f32 slice from input tensor")
    })?;

    let output_slice = output_data.as_f32_slice_mut().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get mutable f32 slice from output data")
    })?;

    for (i, &val) in input_data.iter().enumerate() {
        output_slice[i] = val.tanh();
    }

    Ok(())
}

fn tanh_f64(tensor: &Tensor, output_data: &mut TensorData) -> Result<()> {
    let input_data = tensor.data().as_f64_slice().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get f64 slice from input tensor")
    })?;

    let output_slice = output_data.as_f64_slice_mut().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get mutable f64 slice from output data")
    })?;

    for (i, &val) in input_data.iter().enumerate() {
        output_slice[i] = val.tanh();
    }

    Ok(())
}

fn sigmoid_f32(tensor: &Tensor, output_data: &mut TensorData) -> Result<()> {
    let input_data = tensor.data().as_f32_slice().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get f32 slice from input tensor")
    })?;

    let output_slice = output_data.as_f32_slice_mut().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get mutable f32 slice from output data")
    })?;

    for (i, &val) in input_data.iter().enumerate() {
        output_slice[i] = 1.0 / (1.0 + (-val).exp());
    }

    Ok(())
}

fn sigmoid_f64(tensor: &Tensor, output_data: &mut TensorData) -> Result<()> {
    let input_data = tensor.data().as_f64_slice().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get f64 slice from input tensor")
    })?;

    let output_slice = output_data.as_f64_slice_mut().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get mutable f64 slice from output data")
    })?;

    for (i, &val) in input_data.iter().enumerate() {
        output_slice[i] = 1.0 / (1.0 + (-val).exp());
    }

    Ok(())
}

fn relu_f32(tensor: &Tensor, output_data: &mut TensorData) -> Result<()> {
    let input_data = tensor.data().as_f32_slice().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get f32 slice from input tensor")
    })?;

    let output_slice = output_data.as_f32_slice_mut().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get mutable f32 slice from output data")
    })?;

    for (i, &val) in input_data.iter().enumerate() {
        output_slice[i] = val.max(0.0);
    }

    Ok(())
}

fn relu_f64(tensor: &Tensor, output_data: &mut TensorData) -> Result<()> {
    let input_data = tensor.data().as_f64_slice().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get f64 slice from input tensor")
    })?;

    let output_slice = output_data.as_f64_slice_mut().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get mutable f64 slice from output data")
    })?;

    for (i, &val) in input_data.iter().enumerate() {
        output_slice[i] = val.max(0.0);
    }

    Ok(())
}

fn relu_i32(tensor: &Tensor, output_data: &mut TensorData) -> Result<()> {
    let input_data = tensor.data().as_i32_slice().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get i32 slice from input tensor")
    })?;

    let output_slice = output_data.as_i32_slice_mut().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get mutable i32 slice from output data")
    })?;

    for (i, &val) in input_data.iter().enumerate() {
        output_slice[i] = val.max(0);
    }

    Ok(())
}

fn relu_i64(tensor: &Tensor, output_data: &mut TensorData) -> Result<()> {
    let input_data = tensor.data().as_i64_slice().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get i64 slice from input tensor")
    })?;

    let output_slice = output_data.as_i64_slice_mut().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get mutable i64 slice from output data")
    })?;

    for (i, &val) in input_data.iter().enumerate() {
        output_slice[i] = val.max(0);
    }

    Ok(())
}

fn leaky_relu_f32(
    tensor: &Tensor,
    output_data: &mut TensorData,
    negative_slope: f32,
) -> Result<Vec<bool>> {
    let input_data = tensor.data().as_f32_slice().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get f32 slice from input tensor")
    })?;

    let output_slice = output_data.as_f32_slice_mut().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get mutable f32 slice from output data")
    })?;

    let mut mask = Vec::with_capacity(input_data.len());
    for (i, &val) in input_data.iter().enumerate() {
        if val >= 0.0 {
            output_slice[i] = val;
            mask.push(true);
        } else {
            output_slice[i] = negative_slope * val;
            mask.push(false);
        }
    }

    Ok(mask)
}

fn leaky_relu_f64(
    tensor: &Tensor,
    output_data: &mut TensorData,
    negative_slope: f64,
) -> Result<Vec<bool>> {
    let input_data = tensor.data().as_f64_slice().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get f64 slice from input tensor")
    })?;

    let output_slice = output_data.as_f64_slice_mut().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get mutable f64 slice from output data")
    })?;

    let mut mask = Vec::with_capacity(input_data.len());
    for (i, &val) in input_data.iter().enumerate() {
        if val >= 0.0 {
            output_slice[i] = val;
            mask.push(true);
        } else {
            output_slice[i] = negative_slope * val;
            mask.push(false);
        }
    }

    Ok(mask)
}

fn softmax_f32(tensor: &Tensor, output_data: &mut TensorData, dim: usize) -> Result<()> {
    let input_data = tensor.data().as_f32_slice().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get f32 slice from input tensor")
    })?;

    let output_slice = output_data.as_f32_slice_mut().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get mutable f32 slice from output data")
    })?;

    let dims = tensor.shape().dims();
    let dim_size = dims[dim];

    // Compute the number of groups before and after the softmax dimension. This
    // allows us to iterate over all slices along `dim` for tensors of arbitrary
    // rank using row-major indexing.
    let before: usize = if dim == 0 { 1 } else { dims[..dim].iter().product() };
    let after: usize = if dim + 1 >= dims.len() {
        1
    } else {
        dims[dim + 1..].iter().product()
    };

    for b in 0..before {
        for a in 0..after {
            // Base index for this slice
            let base = b * dim_size * after + a;

            // Find max for numerical stability
            let mut max_val = f32::NEG_INFINITY;
            for k in 0..dim_size {
                let idx = base + k * after;
                max_val = max_val.max(input_data[idx]);
            }

            // Compute exp(x - max) and accumulate sum
            let mut sum = 0.0f32;
            for k in 0..dim_size {
                let idx = base + k * after;
                let val = (input_data[idx] - max_val).exp();
                output_slice[idx] = val;
                sum += val;
            }

            // Normalize
            for k in 0..dim_size {
                let idx = base + k * after;
                output_slice[idx] /= sum;
            }
        }
    }

    Ok(())
}

fn softmax_f64(tensor: &Tensor, output_data: &mut TensorData, dim: usize) -> Result<()> {
    let input_data = tensor.data().as_f64_slice().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get f64 slice from input tensor")
    })?;

    let output_slice = output_data.as_f64_slice_mut().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get mutable f64 slice from output data")
    })?;

    let dims = tensor.shape().dims();
    let dim_size = dims[dim];

    let before: usize = if dim == 0 { 1 } else { dims[..dim].iter().product() };
    let after: usize = if dim + 1 >= dims.len() {
        1
    } else {
        dims[dim + 1..].iter().product()
    };

    for b in 0..before {
        for a in 0..after {
            let base = b * dim_size * after + a;

            let mut max_val = f64::NEG_INFINITY;
            for k in 0..dim_size {
                let idx = base + k * after;
                max_val = max_val.max(input_data[idx]);
            }

            let mut sum = 0.0f64;
            for k in 0..dim_size {
                let idx = base + k * after;
                let val = (input_data[idx] - max_val).exp();
                output_slice[idx] = val;
                sum += val;
            }

            for k in 0..dim_size {
                let idx = base + k * after;
                output_slice[idx] /= sum;
            }
        }
    }

    Ok(())
}

/// Absolute value function
pub fn abs(tensor: &Tensor) -> Result<Tensor> {
    let mut output_data =
        TensorData::zeros_on_device(tensor.numel(), tensor.dtype(), tensor.device());

    match tensor.dtype() {
        DataType::Float32 => abs_f32(tensor, &mut output_data)?,
        DataType::Float64 => abs_f64(tensor, &mut output_data)?,
        DataType::Int32 => abs_i32(tensor, &mut output_data)?,
        DataType::Int64 => abs_i64(tensor, &mut output_data)?,
        DataType::Bool => {
            return Err(MinitensorError::invalid_operation(
                "Absolute value not supported for boolean tensors",
            ))
        }
    }

    let output = Tensor::new(
        Arc::new(output_data),
        tensor.shape().clone(),
        tensor.dtype(),
        tensor.device(),
        tensor.requires_grad(),
    );

    Ok(output)
}

/// Square root function
pub fn sqrt(tensor: &Tensor) -> Result<Tensor> {
    let mut output_data =
        TensorData::zeros_on_device(tensor.numel(), tensor.dtype(), tensor.device());

    match tensor.dtype() {
        DataType::Float32 => sqrt_f32(tensor, &mut output_data)?,
        DataType::Float64 => sqrt_f64(tensor, &mut output_data)?,
        _ => {
            return Err(MinitensorError::invalid_operation(
                "Square root only supported for floating point tensors",
            ))
        }
    }

    let output = Tensor::new(
        Arc::new(output_data),
        tensor.shape().clone(),
        tensor.dtype(),
        tensor.device(),
        tensor.requires_grad(),
    );

    Ok(output)
}

/// Clip tensor values to range
pub fn clip(tensor: &Tensor, min_val: Option<f64>, max_val: Option<f64>) -> Result<Tensor> {
    let mut output_data =
        TensorData::zeros_on_device(tensor.numel(), tensor.dtype(), tensor.device());

    match tensor.dtype() {
        DataType::Float32 => clip_f32(tensor, &mut output_data, min_val, max_val)?,
        DataType::Float64 => clip_f64(tensor, &mut output_data, min_val, max_val)?,
        DataType::Int32 => clip_i32(tensor, &mut output_data, min_val, max_val)?,
        DataType::Int64 => clip_i64(tensor, &mut output_data, min_val, max_val)?,
        DataType::Bool => {
            return Err(MinitensorError::invalid_operation(
                "Clip not supported for boolean tensors",
            ))
        }
    }

    let output = Tensor::new(
        Arc::new(output_data),
        tensor.shape().clone(),
        tensor.dtype(),
        tensor.device(),
        tensor.requires_grad(),
    );

    Ok(output)
}

/// Round tensor values
pub fn round(tensor: &Tensor, decimals: i32) -> Result<Tensor> {
    let mut output_data =
        TensorData::zeros_on_device(tensor.numel(), tensor.dtype(), tensor.device());

    match tensor.dtype() {
        DataType::Float32 => round_f32(tensor, &mut output_data, decimals)?,
        DataType::Float64 => round_f64(tensor, &mut output_data, decimals)?,
        _ => {
            return Err(MinitensorError::invalid_operation(
                "Round only supported for floating point tensors",
            ))
        }
    }

    let output = Tensor::new(
        Arc::new(output_data),
        tensor.shape().clone(),
        tensor.dtype(),
        tensor.device(),
        tensor.requires_grad(),
    );

    Ok(output)
}

/// Floor tensor values
pub fn floor(tensor: &Tensor) -> Result<Tensor> {
    let mut output_data =
        TensorData::zeros_on_device(tensor.numel(), tensor.dtype(), tensor.device());

    match tensor.dtype() {
        DataType::Float32 => floor_f32(tensor, &mut output_data)?,
        DataType::Float64 => floor_f64(tensor, &mut output_data)?,
        _ => {
            return Err(MinitensorError::invalid_operation(
                "Floor only supported for floating point tensors",
            ))
        }
    }

    let output = Tensor::new(
        Arc::new(output_data),
        tensor.shape().clone(),
        tensor.dtype(),
        tensor.device(),
        tensor.requires_grad(),
    );

    Ok(output)
}

/// Ceiling tensor values
pub fn ceil(tensor: &Tensor) -> Result<Tensor> {
    let mut output_data =
        TensorData::zeros_on_device(tensor.numel(), tensor.dtype(), tensor.device());

    match tensor.dtype() {
        DataType::Float32 => ceil_f32(tensor, &mut output_data)?,
        DataType::Float64 => ceil_f64(tensor, &mut output_data)?,
        _ => {
            return Err(MinitensorError::invalid_operation(
                "Ceiling only supported for floating point tensors",
            ))
        }
    }

    let output = Tensor::new(
        Arc::new(output_data),
        tensor.shape().clone(),
        tensor.dtype(),
        tensor.device(),
        tensor.requires_grad(),
    );

    Ok(output)
}

// Helper functions for the new operations

fn abs_f32(tensor: &Tensor, output_data: &mut TensorData) -> Result<()> {
    let input_data = tensor.data().as_f32_slice().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get f32 slice from input tensor")
    })?;

    let output_slice = output_data.as_f32_slice_mut().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get mutable f32 slice from output data")
    })?;

    for (i, &val) in input_data.iter().enumerate() {
        output_slice[i] = val.abs();
    }

    Ok(())
}

fn abs_f64(tensor: &Tensor, output_data: &mut TensorData) -> Result<()> {
    let input_data = tensor.data().as_f64_slice().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get f64 slice from input tensor")
    })?;

    let output_slice = output_data.as_f64_slice_mut().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get mutable f64 slice from output data")
    })?;

    for (i, &val) in input_data.iter().enumerate() {
        output_slice[i] = val.abs();
    }

    Ok(())
}

fn abs_i32(tensor: &Tensor, output_data: &mut TensorData) -> Result<()> {
    let input_data = tensor.data().as_i32_slice().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get i32 slice from input tensor")
    })?;

    let output_slice = output_data.as_i32_slice_mut().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get mutable i32 slice from output data")
    })?;

    for (i, &val) in input_data.iter().enumerate() {
        output_slice[i] = val.abs();
    }

    Ok(())
}

fn abs_i64(tensor: &Tensor, output_data: &mut TensorData) -> Result<()> {
    let input_data = tensor.data().as_i64_slice().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get i64 slice from input tensor")
    })?;

    let output_slice = output_data.as_i64_slice_mut().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get mutable i64 slice from output data")
    })?;

    for (i, &val) in input_data.iter().enumerate() {
        output_slice[i] = val.abs();
    }

    Ok(())
}

fn sqrt_f32(tensor: &Tensor, output_data: &mut TensorData) -> Result<()> {
    let input_data = tensor.data().as_f32_slice().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get f32 slice from input tensor")
    })?;

    let output_slice = output_data.as_f32_slice_mut().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get mutable f32 slice from output data")
    })?;

    for (i, &val) in input_data.iter().enumerate() {
        output_slice[i] = val.sqrt();
    }

    Ok(())
}

fn sqrt_f64(tensor: &Tensor, output_data: &mut TensorData) -> Result<()> {
    let input_data = tensor.data().as_f64_slice().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get f64 slice from input tensor")
    })?;

    let output_slice = output_data.as_f64_slice_mut().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get mutable f64 slice from output data")
    })?;

    for (i, &val) in input_data.iter().enumerate() {
        output_slice[i] = val.sqrt();
    }

    Ok(())
}

fn clip_f32(
    tensor: &Tensor,
    output_data: &mut TensorData,
    min_val: Option<f64>,
    max_val: Option<f64>,
) -> Result<()> {
    let input_data = tensor.data().as_f32_slice().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get f32 slice from input tensor")
    })?;

    let output_slice = output_data.as_f32_slice_mut().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get mutable f32 slice from output data")
    })?;

    let min_f32 = min_val.map(|v| v as f32);
    let max_f32 = max_val.map(|v| v as f32);

    for (i, &val) in input_data.iter().enumerate() {
        let mut clipped = val;
        if let Some(min) = min_f32 {
            clipped = clipped.max(min);
        }
        if let Some(max) = max_f32 {
            clipped = clipped.min(max);
        }
        output_slice[i] = clipped;
    }

    Ok(())
}

fn clip_f64(
    tensor: &Tensor,
    output_data: &mut TensorData,
    min_val: Option<f64>,
    max_val: Option<f64>,
) -> Result<()> {
    let input_data = tensor.data().as_f64_slice().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get f64 slice from input tensor")
    })?;

    let output_slice = output_data.as_f64_slice_mut().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get mutable f64 slice from output data")
    })?;

    for (i, &val) in input_data.iter().enumerate() {
        let mut clipped = val;
        if let Some(min) = min_val {
            clipped = clipped.max(min);
        }
        if let Some(max) = max_val {
            clipped = clipped.min(max);
        }
        output_slice[i] = clipped;
    }

    Ok(())
}

fn clip_i32(
    tensor: &Tensor,
    output_data: &mut TensorData,
    min_val: Option<f64>,
    max_val: Option<f64>,
) -> Result<()> {
    let input_data = tensor.data().as_i32_slice().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get i32 slice from input tensor")
    })?;

    let output_slice = output_data.as_i32_slice_mut().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get mutable i32 slice from output data")
    })?;

    let min_i32 = min_val.map(|v| v as i32);
    let max_i32 = max_val.map(|v| v as i32);

    for (i, &val) in input_data.iter().enumerate() {
        let mut clipped = val;
        if let Some(min) = min_i32 {
            clipped = clipped.max(min);
        }
        if let Some(max) = max_i32 {
            clipped = clipped.min(max);
        }
        output_slice[i] = clipped;
    }

    Ok(())
}

fn clip_i64(
    tensor: &Tensor,
    output_data: &mut TensorData,
    min_val: Option<f64>,
    max_val: Option<f64>,
) -> Result<()> {
    let input_data = tensor.data().as_i64_slice().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get i64 slice from input tensor")
    })?;

    let output_slice = output_data.as_i64_slice_mut().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get mutable i64 slice from output data")
    })?;

    let min_i64 = min_val.map(|v| v as i64);
    let max_i64 = max_val.map(|v| v as i64);

    for (i, &val) in input_data.iter().enumerate() {
        let mut clipped = val;
        if let Some(min) = min_i64 {
            clipped = clipped.max(min);
        }
        if let Some(max) = max_i64 {
            clipped = clipped.min(max);
        }
        output_slice[i] = clipped;
    }

    Ok(())
}

fn round_f32(tensor: &Tensor, output_data: &mut TensorData, decimals: i32) -> Result<()> {
    let input_data = tensor.data().as_f32_slice().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get f32 slice from input tensor")
    })?;

    let output_slice = output_data.as_f32_slice_mut().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get mutable f32 slice from output data")
    })?;

    let multiplier = 10.0_f32.powi(decimals);

    for (i, &val) in input_data.iter().enumerate() {
        output_slice[i] = (val * multiplier).round() / multiplier;
    }

    Ok(())
}

fn round_f64(tensor: &Tensor, output_data: &mut TensorData, decimals: i32) -> Result<()> {
    let input_data = tensor.data().as_f64_slice().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get f64 slice from input tensor")
    })?;

    let output_slice = output_data.as_f64_slice_mut().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get mutable f64 slice from output data")
    })?;

    let multiplier = 10.0_f64.powi(decimals);

    for (i, &val) in input_data.iter().enumerate() {
        output_slice[i] = (val * multiplier).round() / multiplier;
    }

    Ok(())
}

fn floor_f32(tensor: &Tensor, output_data: &mut TensorData) -> Result<()> {
    let input_data = tensor.data().as_f32_slice().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get f32 slice from input tensor")
    })?;

    let output_slice = output_data.as_f32_slice_mut().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get mutable f32 slice from output data")
    })?;

    for (i, &val) in input_data.iter().enumerate() {
        output_slice[i] = val.floor();
    }

    Ok(())
}

fn floor_f64(tensor: &Tensor, output_data: &mut TensorData) -> Result<()> {
    let input_data = tensor.data().as_f64_slice().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get f64 slice from input tensor")
    })?;

    let output_slice = output_data.as_f64_slice_mut().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get mutable f64 slice from output data")
    })?;

    for (i, &val) in input_data.iter().enumerate() {
        output_slice[i] = val.floor();
    }

    Ok(())
}

fn ceil_f32(tensor: &Tensor, output_data: &mut TensorData) -> Result<()> {
    let input_data = tensor.data().as_f32_slice().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get f32 slice from input tensor")
    })?;

    let output_slice = output_data.as_f32_slice_mut().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get mutable f32 slice from output data")
    })?;

    for (i, &val) in input_data.iter().enumerate() {
        output_slice[i] = val.ceil();
    }

    Ok(())
}

fn ceil_f64(tensor: &Tensor, output_data: &mut TensorData) -> Result<()> {
    let input_data = tensor.data().as_f64_slice().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get f64 slice from input tensor")
    })?;

    let output_slice = output_data.as_f64_slice_mut().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get mutable f64 slice from output data")
    })?;

    for (i, &val) in input_data.iter().enumerate() {
        output_slice[i] = val.ceil();
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        autograd,
        device::Device,
        tensor::{Shape, Tensor, TensorData},
    };

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
    fn test_exp() {
        let tensor = create_test_tensor_f32(vec![0.0, 1.0, 2.0], vec![3], false);
        let result = exp(&tensor).unwrap();
        let result_data = result.data().as_f32_slice().unwrap();

        assert!((result_data[0] - 1.0).abs() < 1e-6);
        assert!((result_data[1] - std::f32::consts::E).abs() < 1e-6);
        assert!((result_data[2] - (std::f32::consts::E * std::f32::consts::E)).abs() < 1e-5);
    }

    #[test]
    fn test_log() {
        let tensor = create_test_tensor_f32(
            vec![
                1.0,
                std::f32::consts::E,
                std::f32::consts::E * std::f32::consts::E,
            ],
            vec![3],
            false,
        );
        let result = log(&tensor).unwrap();
        let result_data = result.data().as_f32_slice().unwrap();

        assert!((result_data[0] - 0.0).abs() < 1e-6);
        assert!((result_data[1] - 1.0).abs() < 1e-6);
        assert!((result_data[2] - 2.0).abs() < 1e-5);
    }

    #[test]
    fn test_sin() {
        let tensor = create_test_tensor_f32(
            vec![0.0, std::f32::consts::PI / 2.0, std::f32::consts::PI],
            vec![3],
            false,
        );
        let result = sin(&tensor).unwrap();
        let result_data = result.data().as_f32_slice().unwrap();

        assert!((result_data[0] - 0.0).abs() < 1e-6);
        assert!((result_data[1] - 1.0).abs() < 1e-6);
        assert!(result_data[2].abs() < 1e-6); // sin(π) ≈ 0
    }

    #[test]
    fn test_cos() {
        let tensor = create_test_tensor_f32(
            vec![0.0, std::f32::consts::PI / 2.0, std::f32::consts::PI],
            vec![3],
            false,
        );
        let result = cos(&tensor).unwrap();
        let result_data = result.data().as_f32_slice().unwrap();

        assert!((result_data[0] - 1.0).abs() < 1e-6);
        assert!(result_data[1].abs() < 1e-6); // cos(π/2) ≈ 0
        assert!((result_data[2] + 1.0).abs() < 1e-6); // cos(π) ≈ -1
    }

    #[test]
    fn test_tan() {
        let tensor = create_test_tensor_f32(
            vec![0.0, std::f32::consts::PI / 4.0, -std::f32::consts::PI / 4.0],
            vec![3],
            false,
        );
        let result = tan(&tensor).unwrap();
        let result_data = result.data().as_f32_slice().unwrap();

        assert!((result_data[0] - 0.0).abs() < 1e-6);
        assert!((result_data[1] - (std::f32::consts::PI / 4.0).tan()).abs() < 1e-6);
        assert!((result_data[2] - (-std::f32::consts::PI / 4.0).tan()).abs() < 1e-6);
    }

    #[test]
    fn test_tanh() {
        let tensor = create_test_tensor_f32(vec![0.0, 1.0, -1.0], vec![3], false);
        let result = tanh(&tensor).unwrap();
        let result_data = result.data().as_f32_slice().unwrap();

        assert!((result_data[0] - 0.0).abs() < 1e-6);
        assert!((result_data[1] - 1.0_f32.tanh()).abs() < 1e-6);
        assert!((result_data[2] - (-1.0_f32).tanh()).abs() < 1e-6);
    }

    #[test]
    fn test_sigmoid() {
        let tensor = create_test_tensor_f32(vec![0.0, 1.0, -1.0], vec![3], false);
        let result = sigmoid(&tensor).unwrap();
        let result_data = result.data().as_f32_slice().unwrap();

        assert!((result_data[0] - 0.5).abs() < 1e-6);
        assert!((result_data[1] - (1.0 / (1.0 + (-1.0_f32).exp()))).abs() < 1e-6);
        assert!((result_data[2] - (1.0 / (1.0 + 1.0_f32.exp()))).abs() < 1e-6);
    }

    #[test]
    fn test_relu() {
        let tensor = create_test_tensor_f32(vec![-2.0, -1.0, 0.0, 1.0, 2.0], vec![5], false);
        let result = relu(&tensor).unwrap();
        let result_data = result.data().as_f32_slice().unwrap();

        assert_eq!(result_data, &[0.0, 0.0, 0.0, 1.0, 2.0]);
    }

    #[test]
    fn test_leaky_relu() {
        let tensor = create_test_tensor_f32(vec![-2.0, -1.0, 0.0, 1.0, 2.0], vec![5], false);
        let result = leaky_relu(&tensor, 0.1).unwrap();
        let result_data = result.data().as_f32_slice().unwrap();

        assert_eq!(result_data, &[-0.2, -0.1, 0.0, 1.0, 2.0]);
    }

    #[test]
    fn test_softmax() {
        let tensor = create_test_tensor_f32(vec![1.0, 2.0, 3.0], vec![3], false);
        let result = softmax(&tensor, None).unwrap();
        let result_data = result.data().as_f32_slice().unwrap();

        // Check that probabilities sum to 1
        let sum: f32 = result_data.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);

        // Check that all values are positive
        for &val in result_data {
            assert!(val > 0.0);
        }

        // Check that larger input values produce larger probabilities
        assert!(result_data[2] > result_data[1]);
        assert!(result_data[1] > result_data[0]);
    }

    #[test]
    fn test_powf_scalar() {
        let tensor = create_test_tensor_f32(vec![1.0, 2.0, 3.0], vec![3], false);
        let result = powf(&tensor, 2.0).unwrap();
        let data = result.data().as_f32_slice().unwrap();
        assert_eq!(data, &[1.0, 4.0, 9.0]);
    }

    #[test]
    fn test_pow_tensor() {
        let base = create_test_tensor_f32(vec![2.0, 3.0, 4.0], vec![3], false);
        let exp = create_test_tensor_f32(vec![1.0, 2.0, 0.5], vec![3], false);
        let result = pow(&base, &exp).unwrap();
        let data = result.data().as_f32_slice().unwrap();
        assert!((data[0] - 2.0).abs() < 1e-6);
        assert!((data[1] - 9.0).abs() < 1e-6);
        assert!((data[2] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_powf_gradient() {
        let tensor = create_test_tensor_f32(vec![2.0, 3.0], vec![2], true);
        let result = powf(&tensor, 3.0).unwrap();
        let ones = Tensor::ones(result.shape().clone(), result.dtype(), result.device(), false);
        let grads = autograd::backward(&result, Some(ones)).unwrap();
        let grad = grads.get(&tensor.id()).unwrap();
        let g = grad.data().as_f32_slice().unwrap();
        assert!((g[0] - 3.0 * 2.0_f32.powf(2.0)).abs() < 1e-6);
        assert!((g[1] - 3.0 * 3.0_f32.powf(2.0)).abs() < 1e-6);
    }

    #[test]
    fn test_gradient_tracking() {
        let tensor = create_test_tensor_f32(vec![1.0, 2.0, 3.0], vec![3], true);

        let result = relu(&tensor).unwrap();
        assert!(result.requires_grad());
        assert!(result.grad_fn().is_some());

        let result2 = sigmoid(&tensor).unwrap();
        assert!(result2.requires_grad());
        assert!(result2.grad_fn().is_some());
    }
}
