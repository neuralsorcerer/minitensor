// Copyright (c) 2026 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

/// Absolute value function
pub fn abs(tensor: &Tensor) -> Result<Tensor> {
    let mut output_data =
        TensorData::uninitialized_on_device(tensor.numel(), tensor.dtype(), tensor.device());

    match tensor.dtype() {
        DataType::Float32 => abs_f32(tensor, &mut output_data)?,
        DataType::Float64 => abs_f64(tensor, &mut output_data)?,
        DataType::Int32 => abs_i32(tensor, &mut output_data)?,
        DataType::Int64 => abs_i64(tensor, &mut output_data)?,
        DataType::Bool => {
            return Err(MinitensorError::invalid_operation(
                "Absolute value not supported for boolean tensors",
            ));
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

/// Element-wise sign function (-1, 0, or 1 depending on value sign)
pub fn sign(tensor: &Tensor) -> Result<Tensor> {
    let mut output_data =
        TensorData::uninitialized_on_device(tensor.numel(), tensor.dtype(), tensor.device());

    match tensor.dtype() {
        DataType::Float32 => sign_f32(tensor, &mut output_data)?,
        DataType::Float64 => sign_f64(tensor, &mut output_data)?,
        DataType::Int32 => sign_i32(tensor, &mut output_data)?,
        DataType::Int64 => sign_i64(tensor, &mut output_data)?,
        DataType::Bool => {
            return Err(MinitensorError::invalid_operation(
                "Sign operation not supported for boolean tensors",
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

/// Square root function
pub fn sqrt(tensor: &Tensor) -> Result<Tensor> {
    // Use powf implementation for gradient support: sqrt(x) = x.powf(0.5)
    powf(tensor, 0.5)
}

/// Reciprocal square root function
pub fn rsqrt(tensor: &Tensor) -> Result<Tensor> {
    // Use powf implementation for gradient support: rsqrt(x) = x.powf(-0.5)
    powf(tensor, -0.5)
}

/// Element-wise reciprocal (1/x) with gradient support
pub fn reciprocal(tensor: &Tensor) -> Result<Tensor> {
    match tensor.dtype() {
        DataType::Float32 | DataType::Float64 => powf(tensor, -1.0),
        _ => Err(MinitensorError::invalid_operation(
            "Reciprocal only supported for floating point tensors",
        )),
    }
}

/// Clip tensor values to range
pub fn clip(tensor: &Tensor, min_val: Option<f64>, max_val: Option<f64>) -> Result<Tensor> {
    let mut output_data =
        TensorData::uninitialized_on_device(tensor.numel(), tensor.dtype(), tensor.device());

    match tensor.dtype() {
        DataType::Float32 => clip_f32(tensor, &mut output_data, min_val, max_val)?,
        DataType::Float64 => clip_f64(tensor, &mut output_data, min_val, max_val)?,
        DataType::Int32 => clip_i32(tensor, &mut output_data, min_val, max_val)?,
        DataType::Int64 => clip_i64(tensor, &mut output_data, min_val, max_val)?,
        DataType::Bool => {
            return Err(MinitensorError::invalid_operation(
                "Clip not supported for boolean tensors",
            ));
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
        TensorData::uninitialized_on_device(tensor.numel(), tensor.dtype(), tensor.device());

    match tensor.dtype() {
        DataType::Float32 => round_f32(tensor, &mut output_data, decimals)?,
        DataType::Float64 => round_f64(tensor, &mut output_data, decimals)?,
        _ => {
            return Err(MinitensorError::invalid_operation(
                "Round only supported for floating point tensors",
            ));
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
        TensorData::uninitialized_on_device(tensor.numel(), tensor.dtype(), tensor.device());

    match tensor.dtype() {
        DataType::Float32 => floor_f32(tensor, &mut output_data)?,
        DataType::Float64 => floor_f64(tensor, &mut output_data)?,
        _ => {
            return Err(MinitensorError::invalid_operation(
                "Floor only supported for floating point tensors",
            ));
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
        TensorData::uninitialized_on_device(tensor.numel(), tensor.dtype(), tensor.device());

    match tensor.dtype() {
        DataType::Float32 => ceil_f32(tensor, &mut output_data)?,
        DataType::Float64 => ceil_f64(tensor, &mut output_data)?,
        _ => {
            return Err(MinitensorError::invalid_operation(
                "Ceiling only supported for floating point tensors",
            ));
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

    unary_apply(input_data, output_slice, |v: f32| v.abs());
    Ok(())
}

fn abs_f64(tensor: &Tensor, output_data: &mut TensorData) -> Result<()> {
    let input_data = tensor.data().as_f64_slice().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get f64 slice from input tensor")
    })?;

    let output_slice = output_data.as_f64_slice_mut().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get mutable f64 slice from output data")
    })?;

    unary_apply(input_data, output_slice, |v: f64| v.abs());
    Ok(())
}

fn abs_i32(tensor: &Tensor, output_data: &mut TensorData) -> Result<()> {
    let input_data = tensor.data().as_i32_slice().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get i32 slice from input tensor")
    })?;

    let output_slice = output_data.as_i32_slice_mut().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get mutable i32 slice from output data")
    })?;

    unary_apply(input_data, output_slice, |v: i32| v.abs());
    Ok(())
}

fn abs_i64(tensor: &Tensor, output_data: &mut TensorData) -> Result<()> {
    let input_data = tensor.data().as_i64_slice().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get i64 slice from input tensor")
    })?;

    let output_slice = output_data.as_i64_slice_mut().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get mutable i64 slice from output data")
    })?;

    unary_apply(input_data, output_slice, |v: i64| v.abs());
    Ok(())
}

fn sign_f32(tensor: &Tensor, output_data: &mut TensorData) -> Result<()> {
    let input_data = tensor.data().as_f32_slice().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get f32 slice from input tensor")
    })?;

    let output_slice = output_data.as_f32_slice_mut().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get mutable f32 slice from output data")
    })?;

    unary_apply(input_data, output_slice, |v: f32| {
        if v > 0.0 {
            1.0
        } else if v < 0.0 {
            -1.0
        } else {
            0.0
        }
    });
    Ok(())
}

fn sign_f64(tensor: &Tensor, output_data: &mut TensorData) -> Result<()> {
    let input_data = tensor.data().as_f64_slice().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get f64 slice from input tensor")
    })?;

    let output_slice = output_data.as_f64_slice_mut().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get mutable f64 slice from output data")
    })?;

    unary_apply(input_data, output_slice, |v: f64| {
        if v > 0.0 {
            1.0
        } else if v < 0.0 {
            -1.0
        } else {
            0.0
        }
    });
    Ok(())
}

fn sign_i32(tensor: &Tensor, output_data: &mut TensorData) -> Result<()> {
    let input_data = tensor.data().as_i32_slice().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get i32 slice from input tensor")
    })?;

    let output_slice = output_data.as_i32_slice_mut().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get mutable i32 slice from output data")
    })?;

    unary_apply(input_data, output_slice, |v: i32| {
        if v > 0 {
            1
        } else if v < 0 {
            -1
        } else {
            0
        }
    });
    Ok(())
}

fn sign_i64(tensor: &Tensor, output_data: &mut TensorData) -> Result<()> {
    let input_data = tensor.data().as_i64_slice().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get i64 slice from input tensor")
    })?;

    let output_slice = output_data.as_i64_slice_mut().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get mutable i64 slice from output data")
    })?;

    unary_apply(input_data, output_slice, |v: i64| {
        if v > 0 {
            1
        } else if v < 0 {
            -1
        } else {
            0
        }
    });
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
    unary_apply(input_data, output_slice, |val: f32| {
        let mut v = val;
        if let Some(min) = min_f32 {
            v = v.max(min);
        }
        if let Some(max) = max_f32 {
            v = v.min(max);
        }
        v
    });
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
    unary_apply(input_data, output_slice, |val: f64| {
        let mut v = val;
        if let Some(min) = min_val {
            v = v.max(min);
        }
        if let Some(max) = max_val {
            v = v.min(max);
        }
        v
    });
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
    unary_apply(input_data, output_slice, |val: i32| {
        let mut v = val;
        if let Some(min) = min_i32 {
            v = v.max(min);
        }
        if let Some(max) = max_i32 {
            v = v.min(max);
        }
        v
    });
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
    unary_apply(input_data, output_slice, |val: i64| {
        let mut v = val;
        if let Some(min) = min_i64 {
            v = v.max(min);
        }
        if let Some(max) = max_i64 {
            v = v.min(max);
        }
        v
    });
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
    unary_apply(input_data, output_slice, |val: f32| {
        (val * multiplier).round() / multiplier
    });
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
    unary_apply(input_data, output_slice, |val: f64| {
        (val * multiplier).round() / multiplier
    });
    Ok(())
}

fn floor_f32(tensor: &Tensor, output_data: &mut TensorData) -> Result<()> {
    let input_data = tensor.data().as_f32_slice().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get f32 slice from input tensor")
    })?;

    let output_slice = output_data.as_f32_slice_mut().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get mutable f32 slice from output data")
    })?;
    unary_apply(input_data, output_slice, f32::floor);
    Ok(())
}

fn floor_f64(tensor: &Tensor, output_data: &mut TensorData) -> Result<()> {
    let input_data = tensor.data().as_f64_slice().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get f64 slice from input tensor")
    })?;

    let output_slice = output_data.as_f64_slice_mut().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get mutable f64 slice from output data")
    })?;
    unary_apply(input_data, output_slice, f64::floor);
    Ok(())
}

fn ceil_f32(tensor: &Tensor, output_data: &mut TensorData) -> Result<()> {
    let input_data = tensor.data().as_f32_slice().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get f32 slice from input tensor")
    })?;

    let output_slice = output_data.as_f32_slice_mut().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get mutable f32 slice from output data")
    })?;
    unary_apply(input_data, output_slice, f32::ceil);
    Ok(())
}
