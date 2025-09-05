// Copyright (c) 2025 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

use crate::{
    autograd::{add_to_graph, AddBackward, DivBackward, MulBackward},
    error::{MinitensorError, Result},
    operations::simd::{
        can_use_simd_fast_path, simd_add_f32, simd_add_f64, simd_div_f32, simd_div_f64,
        simd_mul_f32, simd_mul_f64, simd_sub_f32, simd_sub_f64,
    },
    tensor::{DataType, Shape, Tensor, TensorData},
};
use std::sync::Arc;
use rayon::prelude::*;

/// Element-wise addition with broadcasting support
pub fn add(lhs: &Tensor, rhs: &Tensor) -> Result<Tensor> {
    // Check device compatibility
    if lhs.device() != rhs.device() {
        return Err(MinitensorError::device_mismatch(
            format!("{:?}", lhs.device()),
            format!("{:?}", rhs.device()),
        ));
    }

    // Check data type compatibility
    if lhs.dtype() != rhs.dtype() {
        return Err(MinitensorError::type_mismatch(
            format!("{:?}", lhs.dtype()),
            format!("{:?}", rhs.dtype()),
        ));
    }

    // Compute broadcasted shape
    let output_shape = lhs.shape().broadcast_with(rhs.shape())?;

    // Create output tensor data
    let mut output_data =
        TensorData::zeros_on_device(output_shape.numel(), lhs.dtype(), lhs.device());

    // Perform element-wise addition based on data type
    match lhs.dtype() {
        DataType::Float32 => add_f32_direct(lhs, rhs, &mut output_data, &output_shape)?,
        DataType::Float64 => add_f64_direct(lhs, rhs, &mut output_data, &output_shape)?,
        DataType::Int32 => add_i32_direct(lhs, rhs, &mut output_data, &output_shape)?,
        DataType::Int64 => add_i64_direct(lhs, rhs, &mut output_data, &output_shape)?,
        DataType::Bool => {
            return Err(MinitensorError::invalid_operation(
                "Addition not supported for boolean tensors",
            ))
        }
    }

    // Create output tensor
    let output = Tensor::new(
        Arc::new(output_data),
        output_shape.clone(),
        lhs.dtype(),
        lhs.device(),
        lhs.requires_grad() || rhs.requires_grad(),
    );

    // Set up gradient function if needed
    if output.requires_grad() {
        let grad_fn = Arc::new(AddBackward {
            input_shapes: [lhs.shape().dims().to_vec(), rhs.shape().dims().to_vec()],
            input_ids: [lhs.id(), rhs.id()],
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

/// Element-wise subtraction with broadcasting support
pub fn sub(lhs: &Tensor, rhs: &Tensor) -> Result<Tensor> {
    // Check device compatibility
    if lhs.device() != rhs.device() {
        return Err(MinitensorError::device_mismatch(
            format!("{:?}", lhs.device()),
            format!("{:?}", rhs.device()),
        ));
    }

    // Check data type compatibility
    if lhs.dtype() != rhs.dtype() {
        return Err(MinitensorError::type_mismatch(
            format!("{:?}", lhs.dtype()),
            format!("{:?}", rhs.dtype()),
        ));
    }

    // Compute broadcasted shape
    let output_shape = lhs.shape().broadcast_with(rhs.shape())?;

    // Create output tensor data
    let mut output_data =
        TensorData::zeros_on_device(output_shape.numel(), lhs.dtype(), lhs.device());

    // Perform element-wise subtraction based on data type
    match lhs.dtype() {
        DataType::Float32 => sub_f32_direct(lhs, rhs, &mut output_data, &output_shape)?,
        DataType::Float64 => sub_f64_direct(lhs, rhs, &mut output_data, &output_shape)?,
        DataType::Int32 => sub_i32_direct(lhs, rhs, &mut output_data, &output_shape)?,
        DataType::Int64 => sub_i64_direct(lhs, rhs, &mut output_data, &output_shape)?,
        DataType::Bool => {
            return Err(MinitensorError::invalid_operation(
                "Subtraction not supported for boolean tensors",
            ))
        }
    }

    // Create output tensor
    let output = Tensor::new(
        Arc::new(output_data),
        output_shape.clone(),
        lhs.dtype(),
        lhs.device(),
        lhs.requires_grad() || rhs.requires_grad(),
    );

    // Set up gradient function if needed (subtraction uses same gradient as addition)
    if output.requires_grad() {
        let grad_fn = Arc::new(AddBackward {
            input_shapes: [lhs.shape().dims().to_vec(), rhs.shape().dims().to_vec()],
            input_ids: [lhs.id(), rhs.id()],
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

/// Element-wise multiplication with broadcasting support
pub fn mul(lhs: &Tensor, rhs: &Tensor) -> Result<Tensor> {
    // Check device compatibility
    if lhs.device() != rhs.device() {
        return Err(MinitensorError::device_mismatch(
            format!("{:?}", lhs.device()),
            format!("{:?}", rhs.device()),
        ));
    }

    // Check data type compatibility
    if lhs.dtype() != rhs.dtype() {
        return Err(MinitensorError::type_mismatch(
            format!("{:?}", lhs.dtype()),
            format!("{:?}", rhs.dtype()),
        ));
    }

    // Compute broadcasted shape
    let output_shape = lhs.shape().broadcast_with(rhs.shape())?;

    // Create output tensor data
    let mut output_data =
        TensorData::zeros_on_device(output_shape.numel(), lhs.dtype(), lhs.device());

    // Perform element-wise multiplication based on data type
    match lhs.dtype() {
        DataType::Float32 => mul_f32_direct(lhs, rhs, &mut output_data, &output_shape)?,
        DataType::Float64 => mul_f64_direct(lhs, rhs, &mut output_data, &output_shape)?,
        DataType::Int32 => mul_i32_direct(lhs, rhs, &mut output_data, &output_shape)?,
        DataType::Int64 => mul_i64_direct(lhs, rhs, &mut output_data, &output_shape)?,
        DataType::Bool => {
            return Err(MinitensorError::invalid_operation(
                "Multiplication not supported for boolean tensors",
            ))
        }
    }

    // Create output tensor
    let output = Tensor::new(
        Arc::new(output_data),
        output_shape.clone(),
        lhs.dtype(),
        lhs.device(),
        lhs.requires_grad() || rhs.requires_grad(),
    );

    // Set up gradient function if needed
    if output.requires_grad() {
        let grad_fn = Arc::new(MulBackward {
            lhs: lhs.clone(),
            rhs: rhs.clone(),
            input_ids: [lhs.id(), rhs.id()],
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

/// Element-wise division with broadcasting support
pub fn div(lhs: &Tensor, rhs: &Tensor) -> Result<Tensor> {
    // Check device compatibility
    if lhs.device() != rhs.device() {
        return Err(MinitensorError::device_mismatch(
            format!("{:?}", lhs.device()),
            format!("{:?}", rhs.device()),
        ));
    }

    // Check data type compatibility
    if lhs.dtype() != rhs.dtype() {
        return Err(MinitensorError::type_mismatch(
            format!("{:?}", lhs.dtype()),
            format!("{:?}", rhs.dtype()),
        ));
    }

    // Compute broadcasted shape
    let output_shape = lhs.shape().broadcast_with(rhs.shape())?;

    // Create output tensor data
    let mut output_data =
        TensorData::zeros_on_device(output_shape.numel(), lhs.dtype(), lhs.device());

    // Perform element-wise division based on data type
    match lhs.dtype() {
        DataType::Float32 => div_f32_direct(lhs, rhs, &mut output_data, &output_shape)?,
        DataType::Float64 => div_f64_direct(lhs, rhs, &mut output_data, &output_shape)?,
        DataType::Int32 => div_i32_direct(lhs, rhs, &mut output_data, &output_shape)?,
        DataType::Int64 => div_i64_direct(lhs, rhs, &mut output_data, &output_shape)?,
        DataType::Bool => {
            return Err(MinitensorError::invalid_operation(
                "Division not supported for boolean tensors",
            ))
        }
    }

    // Create output tensor
    let output = Tensor::new(
        Arc::new(output_data),
        output_shape.clone(),
        lhs.dtype(),
        lhs.device(),
        lhs.requires_grad() || rhs.requires_grad(),
    );

    // Set up gradient function if needed
    if output.requires_grad() {
        let grad_fn = Arc::new(DivBackward {
            lhs: lhs.clone(),
            rhs: rhs.clone(),
            input_ids: [lhs.id(), rhs.id()],
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

fn add_f32_direct(
    lhs: &Tensor,
    rhs: &Tensor,
    output_data: &mut TensorData,
    output_shape: &Shape,
) -> Result<()> {
    let lhs_data = lhs.data().as_f32_slice().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get f32 slice from lhs tensor")
    })?;
    let rhs_data = rhs.data().as_f32_slice().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get f32 slice from rhs tensor")
    })?;

    let output_slice = output_data.as_f32_slice_mut().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get mutable f32 slice from output data")
    })?;

    // Use SIMD fast path if possible (no broadcasting, same shapes)
    if can_use_simd_fast_path(lhs.shape(), rhs.shape(), output_shape) {
        simd_add_f32(lhs_data, rhs_data, output_slice)
    } else {
        broadcast_binary_op(
            lhs_data,
            rhs_data,
            output_slice,
            lhs.shape(),
            rhs.shape(),
            output_shape,
            |a, b| a + b,
        )
    }
}

fn add_f64_direct(
    lhs: &Tensor,
    rhs: &Tensor,
    output_data: &mut TensorData,
    output_shape: &Shape,
) -> Result<()> {
    let lhs_data = lhs.data().as_f64_slice().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get f64 slice from lhs tensor")
    })?;
    let rhs_data = rhs.data().as_f64_slice().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get f64 slice from rhs tensor")
    })?;

    let output_slice = output_data.as_f64_slice_mut().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get mutable f64 slice from output data")
    })?;

    // Use SIMD fast path if possible (no broadcasting, same shapes)
    if can_use_simd_fast_path(lhs.shape(), rhs.shape(), output_shape) {
        simd_add_f64(lhs_data, rhs_data, output_slice)
    } else {
        broadcast_binary_op(
            lhs_data,
            rhs_data,
            output_slice,
            lhs.shape(),
            rhs.shape(),
            output_shape,
            |a, b| a + b,
        )
    }
}

fn add_i32_direct(
    lhs: &Tensor,
    rhs: &Tensor,
    output_data: &mut TensorData,
    output_shape: &Shape,
) -> Result<()> {
    let lhs_data = lhs.data().as_i32_slice().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get i32 slice from lhs tensor")
    })?;
    let rhs_data = rhs.data().as_i32_slice().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get i32 slice from rhs tensor")
    })?;

    let output_slice = output_data.as_i32_slice_mut().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get mutable i32 slice from output data")
    })?;

    broadcast_binary_op(
        lhs_data,
        rhs_data,
        output_slice,
        lhs.shape(),
        rhs.shape(),
        output_shape,
        |a, b| a + b,
    )
}

fn add_i64_direct(
    lhs: &Tensor,
    rhs: &Tensor,
    output_data: &mut TensorData,
    output_shape: &Shape,
) -> Result<()> {
    let lhs_data = lhs.data().as_i64_slice().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get i64 slice from lhs tensor")
    })?;
    let rhs_data = rhs.data().as_i64_slice().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get i64 slice from rhs tensor")
    })?;

    let output_slice = output_data.as_i64_slice_mut().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get mutable i64 slice from output data")
    })?;

    broadcast_binary_op(
        lhs_data,
        rhs_data,
        output_slice,
        lhs.shape(),
        rhs.shape(),
        output_shape,
        |a, b| a + b,
    )
}

fn sub_f32_direct(
    lhs: &Tensor,
    rhs: &Tensor,
    output_data: &mut TensorData,
    output_shape: &Shape,
) -> Result<()> {
    let lhs_data = lhs.data().as_f32_slice().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get f32 slice from lhs tensor")
    })?;
    let rhs_data = rhs.data().as_f32_slice().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get f32 slice from rhs tensor")
    })?;

    let output_slice = output_data.as_f32_slice_mut().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get mutable f32 slice from output data")
    })?;

    // Use SIMD fast path if possible (no broadcasting, same shapes)
    if can_use_simd_fast_path(lhs.shape(), rhs.shape(), output_shape) {
        simd_sub_f32(lhs_data, rhs_data, output_slice)
    } else {
        broadcast_binary_op(
            lhs_data,
            rhs_data,
            output_slice,
            lhs.shape(),
            rhs.shape(),
            output_shape,
            |a, b| a - b,
        )
    }
}

fn sub_f64_direct(
    lhs: &Tensor,
    rhs: &Tensor,
    output_data: &mut TensorData,
    output_shape: &Shape,
) -> Result<()> {
    let lhs_data = lhs.data().as_f64_slice().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get f64 slice from lhs tensor")
    })?;
    let rhs_data = rhs.data().as_f64_slice().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get f64 slice from rhs tensor")
    })?;

    let output_slice = output_data.as_f64_slice_mut().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get mutable f64 slice from output data")
    })?;

    // Use SIMD fast path if possible (no broadcasting, same shapes)
    if can_use_simd_fast_path(lhs.shape(), rhs.shape(), output_shape) {
        simd_sub_f64(lhs_data, rhs_data, output_slice)
    } else {
        broadcast_binary_op(
            lhs_data,
            rhs_data,
            output_slice,
            lhs.shape(),
            rhs.shape(),
            output_shape,
            |a, b| a - b,
        )
    }
}

fn sub_i32_direct(
    lhs: &Tensor,
    rhs: &Tensor,
    output_data: &mut TensorData,
    output_shape: &Shape,
) -> Result<()> {
    let lhs_data = lhs.data().as_i32_slice().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get i32 slice from lhs tensor")
    })?;
    let rhs_data = rhs.data().as_i32_slice().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get i32 slice from rhs tensor")
    })?;

    let output_slice = output_data.as_i32_slice_mut().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get mutable i32 slice from output data")
    })?;

    broadcast_binary_op(
        lhs_data,
        rhs_data,
        output_slice,
        lhs.shape(),
        rhs.shape(),
        output_shape,
        |a, b| a - b,
    )
}

fn sub_i64_direct(
    lhs: &Tensor,
    rhs: &Tensor,
    output_data: &mut TensorData,
    output_shape: &Shape,
) -> Result<()> {
    let lhs_data = lhs.data().as_i64_slice().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get i64 slice from lhs tensor")
    })?;
    let rhs_data = rhs.data().as_i64_slice().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get i64 slice from rhs tensor")
    })?;

    let output_slice = output_data.as_i64_slice_mut().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get mutable i64 slice from output data")
    })?;

    broadcast_binary_op(
        lhs_data,
        rhs_data,
        output_slice,
        lhs.shape(),
        rhs.shape(),
        output_shape,
        |a, b| a - b,
    )
}

fn mul_f32_direct(
    lhs: &Tensor,
    rhs: &Tensor,
    output_data: &mut TensorData,
    output_shape: &Shape,
) -> Result<()> {
    let lhs_data = lhs.data().as_f32_slice().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get f32 slice from lhs tensor")
    })?;
    let rhs_data = rhs.data().as_f32_slice().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get f32 slice from rhs tensor")
    })?;

    let output_slice = output_data.as_f32_slice_mut().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get mutable f32 slice from output data")
    })?;

    // Use SIMD fast path if possible (no broadcasting, same shapes)
    if can_use_simd_fast_path(lhs.shape(), rhs.shape(), output_shape) {
        simd_mul_f32(lhs_data, rhs_data, output_slice)
    } else {
        broadcast_binary_op(
            lhs_data,
            rhs_data,
            output_slice,
            lhs.shape(),
            rhs.shape(),
            output_shape,
            |a, b| a * b,
        )
    }
}

fn mul_f64_direct(
    lhs: &Tensor,
    rhs: &Tensor,
    output_data: &mut TensorData,
    output_shape: &Shape,
) -> Result<()> {
    let lhs_data = lhs.data().as_f64_slice().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get f64 slice from lhs tensor")
    })?;
    let rhs_data = rhs.data().as_f64_slice().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get f64 slice from rhs tensor")
    })?;

    let output_slice = output_data.as_f64_slice_mut().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get mutable f64 slice from output data")
    })?;

    // Use SIMD fast path if possible (no broadcasting, same shapes)
    if can_use_simd_fast_path(lhs.shape(), rhs.shape(), output_shape) {
        simd_mul_f64(lhs_data, rhs_data, output_slice)
    } else {
        broadcast_binary_op(
            lhs_data,
            rhs_data,
            output_slice,
            lhs.shape(),
            rhs.shape(),
            output_shape,
            |a, b| a * b,
        )
    }
}

fn mul_i32_direct(
    lhs: &Tensor,
    rhs: &Tensor,
    output_data: &mut TensorData,
    output_shape: &Shape,
) -> Result<()> {
    let lhs_data = lhs.data().as_i32_slice().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get i32 slice from lhs tensor")
    })?;
    let rhs_data = rhs.data().as_i32_slice().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get i32 slice from rhs tensor")
    })?;

    let output_slice = output_data.as_i32_slice_mut().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get mutable i32 slice from output data")
    })?;

    broadcast_binary_op(
        lhs_data,
        rhs_data,
        output_slice,
        lhs.shape(),
        rhs.shape(),
        output_shape,
        |a, b| a * b,
    )
}

fn mul_i64_direct(
    lhs: &Tensor,
    rhs: &Tensor,
    output_data: &mut TensorData,
    output_shape: &Shape,
) -> Result<()> {
    let lhs_data = lhs.data().as_i64_slice().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get i64 slice from lhs tensor")
    })?;
    let rhs_data = rhs.data().as_i64_slice().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get i64 slice from rhs tensor")
    })?;

    let output_slice = output_data.as_i64_slice_mut().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get mutable i64 slice from output data")
    })?;

    broadcast_binary_op(
        lhs_data,
        rhs_data,
        output_slice,
        lhs.shape(),
        rhs.shape(),
        output_shape,
        |a, b| a * b,
    )
}

fn div_f32_direct(
    lhs: &Tensor,
    rhs: &Tensor,
    output_data: &mut TensorData,
    output_shape: &Shape,
) -> Result<()> {
    let lhs_data = lhs.data().as_f32_slice().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get f32 slice from lhs tensor")
    })?;
    let rhs_data = rhs.data().as_f32_slice().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get f32 slice from rhs tensor")
    })?;

    let output_slice = output_data.as_f32_slice_mut().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get mutable f32 slice from output data")
    })?;

    // Use SIMD fast path if possible (no broadcasting, same shapes)
    if can_use_simd_fast_path(lhs.shape(), rhs.shape(), output_shape) {
        simd_div_f32(lhs_data, rhs_data, output_slice)
    } else {
        broadcast_binary_op(
            lhs_data,
            rhs_data,
            output_slice,
            lhs.shape(),
            rhs.shape(),
            output_shape,
            |a, b| {
                if b == 0.0 {
                    f32::INFINITY
                } else {
                    a / b
                }
            },
        )
    }
}

fn div_f64_direct(
    lhs: &Tensor,
    rhs: &Tensor,
    output_data: &mut TensorData,
    output_shape: &Shape,
) -> Result<()> {
    let lhs_data = lhs.data().as_f64_slice().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get f64 slice from lhs tensor")
    })?;
    let rhs_data = rhs.data().as_f64_slice().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get f64 slice from rhs tensor")
    })?;

    let output_slice = output_data.as_f64_slice_mut().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get mutable f64 slice from output data")
    })?;

    // Use SIMD fast path if possible (no broadcasting, same shapes)
    if can_use_simd_fast_path(lhs.shape(), rhs.shape(), output_shape) {
        simd_div_f64(lhs_data, rhs_data, output_slice)
    } else {
        broadcast_binary_op(
            lhs_data,
            rhs_data,
            output_slice,
            lhs.shape(),
            rhs.shape(),
            output_shape,
            |a, b| {
                if b == 0.0 {
                    f64::INFINITY
                } else {
                    a / b
                }
            },
        )
    }
}

fn div_i32_direct(
    lhs: &Tensor,
    rhs: &Tensor,
    output_data: &mut TensorData,
    output_shape: &Shape,
) -> Result<()> {
    let lhs_data = lhs.data().as_i32_slice().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get i32 slice from lhs tensor")
    })?;
    let rhs_data = rhs.data().as_i32_slice().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get i32 slice from rhs tensor")
    })?;

    let output_slice = output_data.as_i32_slice_mut().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get mutable i32 slice from output data")
    })?;

    broadcast_binary_op(
        lhs_data,
        rhs_data,
        output_slice,
        lhs.shape(),
        rhs.shape(),
        output_shape,
        |a, b| {
            if b == 0 {
                i32::MAX // Represent overflow/infinity for integers
            } else {
                a / b
            }
        },
    )
}

fn div_i64_direct(
    lhs: &Tensor,
    rhs: &Tensor,
    output_data: &mut TensorData,
    output_shape: &Shape,
) -> Result<()> {
    let lhs_data = lhs.data().as_i64_slice().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get i64 slice from lhs tensor")
    })?;
    let rhs_data = rhs.data().as_i64_slice().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get i64 slice from rhs tensor")
    })?;

    let output_slice = output_data.as_i64_slice_mut().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get mutable i64 slice from output data")
    })?;

    broadcast_binary_op(
        lhs_data,
        rhs_data,
        output_slice,
        lhs.shape(),
        rhs.shape(),
        output_shape,
        |a, b| {
            if b == 0 {
                i64::MAX // Represent overflow/infinity for integers
            } else {
                a / b
            }
        },
    )
}

/// Generic broadcasting binary operation
fn broadcast_binary_op<T, F>(
    lhs_data: &[T],
    rhs_data: &[T],
    output_data: &mut [T],
    lhs_shape: &Shape,
    rhs_shape: &Shape,
    output_shape: &Shape,
    op: F,
) -> Result<()>
where
    T: Copy + Send + Sync,
    F: Fn(T, T) -> T + Send + Sync,
{
    let output_dims = output_shape.dims().to_vec();
    let lhs_dims = lhs_shape.dims().to_vec();
    let rhs_dims = rhs_shape.dims().to_vec();

    output_data
        .par_iter_mut()
        .enumerate()
        .try_for_each(|(output_idx, out)| -> Result<()> {
            // Convert linear output index to multi-dimensional coordinates
            let mut output_coords = vec![0; output_dims.len()];
            let mut temp_idx = output_idx;

            for i in (0..output_dims.len()).rev() {
                output_coords[i] = temp_idx % output_dims[i];
                temp_idx /= output_dims[i];
            }

            // Map output coordinates to lhs coordinates (with broadcasting)
            let mut lhs_idx = 0;
            let lhs_offset = output_dims.len().saturating_sub(lhs_dims.len());

            for i in 0..lhs_dims.len() {
                let output_coord_idx = i + lhs_offset;
                let coord = if lhs_dims[i] == 1 {
                    0
                } else {
                    output_coords[output_coord_idx]
                };

                let mut stride = 1;
                for j in (i + 1)..lhs_dims.len() {
                    stride *= lhs_dims[j];
                }
                lhs_idx += coord * stride;
            }

            // Map output coordinates to rhs coordinates (with broadcasting)
            let mut rhs_idx = 0;
            let rhs_offset = output_dims.len().saturating_sub(rhs_dims.len());

            for i in 0..rhs_dims.len() {
                let output_coord_idx = i + rhs_offset;
                let coord = if rhs_dims[i] == 1 {
                    0
                } else {
                    output_coords[output_coord_idx]
                };

                let mut stride = 1;
                for j in (i + 1)..rhs_dims.len() {
                    stride *= rhs_dims[j];
                }
                rhs_idx += coord * stride;
            }

            if lhs_idx >= lhs_data.len() || rhs_idx >= rhs_data.len() {
                return Err(MinitensorError::invalid_operation(
                    "Index out of bounds in broadcasting",
                ));
            }

            *out = op(lhs_data[lhs_idx], rhs_data[rhs_idx]);
            Ok(())
        })?;

    Ok(())
}

/// Map output indices to input indices for broadcasting
#[allow(dead_code)]
fn map_broadcasted_index(
    output_indices: &[usize],
    input_shape: &Shape,
    output_shape: &Shape,
) -> Vec<usize> {
    let mut input_indices = vec![0; input_shape.ndim()];

    // Align dimensions from the right (broadcasting rule)
    let output_ndim = output_shape.ndim();
    let input_ndim = input_shape.ndim();

    for i in 0..input_ndim {
        // Map from right to left
        let input_dim_idx = input_ndim - 1 - i;
        let output_dim_idx = output_ndim - 1 - i;
        let input_dim_size = input_shape.dims()[input_dim_idx];

        if input_dim_size == 1 {
            // Broadcasting: use index 0
            input_indices[input_dim_idx] = 0;
        } else {
            // No broadcasting: use the output index
            input_indices[input_dim_idx] = output_indices[output_dim_idx];
        }
    }

    input_indices
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::Device;

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
    fn test_add_basic() {
        let a = create_test_tensor_f32(vec![1.0, 2.0, 3.0], vec![3], false);
        let b = create_test_tensor_f32(vec![4.0, 5.0, 6.0], vec![3], false);

        let result = add(&a, &b).unwrap();
        let result_data = result.data().as_f32_slice().unwrap();

        assert_eq!(result_data, &[5.0, 7.0, 9.0]);
        assert_eq!(result.shape().dims(), &[3]);
    }

    #[test]
    fn test_add_broadcasting() {
        let a = create_test_tensor_f32(vec![1.0, 2.0, 3.0], vec![3], false);
        let b = create_test_tensor_f32(vec![10.0], vec![1], false);

        let result = add(&a, &b).unwrap();
        let result_data = result.data().as_f32_slice().unwrap();

        assert_eq!(result_data, &[11.0, 12.0, 13.0]);
        assert_eq!(result.shape().dims(), &[3]);
    }

    #[test]
    fn test_sub_basic() {
        let a = create_test_tensor_f32(vec![5.0, 7.0, 9.0], vec![3], false);
        let b = create_test_tensor_f32(vec![1.0, 2.0, 3.0], vec![3], false);

        let result = sub(&a, &b).unwrap();
        let result_data = result.data().as_f32_slice().unwrap();

        assert_eq!(result_data, &[4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_mul_basic() {
        let a = create_test_tensor_f32(vec![2.0, 3.0, 4.0], vec![3], false);
        let b = create_test_tensor_f32(vec![5.0, 6.0, 7.0], vec![3], false);

        let result = mul(&a, &b).unwrap();
        let result_data = result.data().as_f32_slice().unwrap();

        assert_eq!(result_data, &[10.0, 18.0, 28.0]);
    }

    #[test]
    fn test_div_basic() {
        let a = create_test_tensor_f32(vec![10.0, 15.0, 20.0], vec![3], false);
        let b = create_test_tensor_f32(vec![2.0, 3.0, 4.0], vec![3], false);

        let result = div(&a, &b).unwrap();
        let result_data = result.data().as_f32_slice().unwrap();

        assert_eq!(result_data, &[5.0, 5.0, 5.0]);
    }

    #[test]
    fn test_gradient_tracking() {
        let a = create_test_tensor_f32(vec![1.0, 2.0], vec![2], true);
        let b = create_test_tensor_f32(vec![3.0, 4.0], vec![2], true);

        let result = add(&a, &b).unwrap();

        assert!(result.requires_grad());
        assert!(result.grad_fn().is_some());
    }

    #[test]
    fn test_device_mismatch_error() {
        let a = create_test_tensor_f32(vec![1.0, 2.0], vec![2], false);
        let b = create_test_tensor_f32(vec![3.0, 4.0], vec![2], false);

        // This would normally fail, but we can't easily create different device tensors in tests
        // So we'll just test that same device works
        let result = add(&a, &b);
        assert!(result.is_ok());
    }

    #[test]
    fn test_type_mismatch_error() {
        let a = create_test_tensor_f32(vec![1.0, 2.0], vec![2], false);

        // Create an i32 tensor
        let shape_obj = Shape::new(vec![2]);
        let mut tensor_data = TensorData::zeros(shape_obj.numel(), DataType::Int32);
        if let Some(slice) = tensor_data.as_i32_slice_mut() {
            slice.copy_from_slice(&[3, 4]);
        }
        let b = Tensor::new(
            Arc::new(tensor_data),
            shape_obj,
            DataType::Int32,
            Device::cpu(),
            false,
        );

        let result = add(&a, &b);
        assert!(result.is_err());
    }

    #[test]
    fn test_sub_broadcasting_2d() {
        let a = create_test_tensor_f32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], false);
        let b = create_test_tensor_f32(vec![1.0, 2.0], vec![1, 2], false);
        let result = sub(&a, &b).unwrap();
        let expected = vec![0.0, 0.0, 2.0, 2.0];
        assert_eq!(result.data().as_f32_slice().unwrap(), expected.as_slice());
        assert_eq!(result.shape().dims(), &[2, 2]);
    }

    #[test]
    fn test_mul_broadcasting_2d() {
        let a = create_test_tensor_f32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], false);
        let b = create_test_tensor_f32(vec![2.0], vec![1, 1], false);
        let result = mul(&a, &b).unwrap();
        assert_eq!(result.data().as_f32_slice().unwrap(), &[2.0, 4.0, 6.0, 8.0]);
        assert_eq!(result.shape().dims(), &[2, 2]);
    }

    #[test]
    fn test_div_broadcasting_2d() {
        let a = create_test_tensor_f32(vec![2.0, 4.0, 6.0, 8.0], vec![2, 2], false);
        let b = create_test_tensor_f32(vec![2.0], vec![1, 1], false);
        let result = div(&a, &b).unwrap();
        assert_eq!(result.data().as_f32_slice().unwrap(), &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(result.shape().dims(), &[2, 2]);
    }

    #[test]
    fn test_bool_arithmetic_error() {
        // Create boolean tensors
        let shape_obj = Shape::new(vec![2]);
        let mut data_a = TensorData::zeros(shape_obj.numel(), DataType::Bool);
        if let Some(slice) = data_a.as_bool_slice_mut() {
            slice.copy_from_slice(&[true, false]);
        }
        let a = Tensor::new(Arc::new(data_a), shape_obj.clone(), DataType::Bool, Device::cpu(), false);

        let mut data_b = TensorData::zeros(shape_obj.numel(), DataType::Bool);
        if let Some(slice) = data_b.as_bool_slice_mut() {
            slice.copy_from_slice(&[false, true]);
        }
        let b = Tensor::new(Arc::new(data_b), shape_obj, DataType::Bool, Device::cpu(), false);

        assert!(add(&a, &b).is_err());
        assert!(sub(&a, &b).is_err());
        assert!(mul(&a, &b).is_err());
        assert!(div(&a, &b).is_err());
    }

    #[test]
    fn test_incompatible_shapes_error() {
        let a = create_test_tensor_f32(vec![1.0, 2.0, 3.0], vec![3], false);
        let b = create_test_tensor_f32(vec![1.0, 2.0], vec![2], false);
        assert!(sub(&a, &b).is_err());
        assert!(mul(&a, &b).is_err());
        assert!(div(&a, &b).is_err());
    }

    #[test]
    fn test_division_by_zero_returns_inf() {
        let a = create_test_tensor_f32(vec![1.0, 2.0], vec![2], false);
        let b = create_test_tensor_f32(vec![0.0, 1.0], vec![2], false);
        let result = div(&a, &b).unwrap();
        let result_data = result.data().as_f32_slice().unwrap();
        assert!(result_data[0].is_infinite());
        assert_eq!(result_data[1], 2.0);
    }
}
