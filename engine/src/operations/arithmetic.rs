// Copyright (c) 2025 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

use crate::{
    autograd::{AddBackward, DivBackward, MulBackward, NegBackward, SubBackward, add_to_graph},
    error::{MinitensorError, Result},
    operations::{
        binary::{BinaryOpKind, coerce_binary_operands},
        simd::{
            can_use_simd_fast_path, simd_add_f32, simd_add_f64, simd_div_f32, simd_div_f64,
            simd_mul_f32, simd_mul_f64, simd_sub_f32, simd_sub_f64,
        },
    },
    tensor::{DataType, Shape, Strides, Tensor, TensorData},
};
use rayon::prelude::*;
use smallvec::{SmallVec, smallvec};
use std::sync::Arc;

const PAR_THRESHOLD: usize = 1 << 12; // 4096 elements

/// Element-wise addition with broadcasting support
pub fn add(lhs: &Tensor, rhs: &Tensor) -> Result<Tensor> {
    // Check device compatibility
    if lhs.device() != rhs.device() {
        return Err(MinitensorError::device_mismatch(
            format!("{:?}", lhs.device()),
            format!("{:?}", rhs.device()),
        ));
    }

    let requires_grad = lhs.requires_grad() || rhs.requires_grad();
    let (lhs_cast, rhs_cast, result_dtype) = coerce_binary_operands(lhs, rhs, BinaryOpKind::Add)?;
    let lhs_ref = lhs_cast.as_ref();
    let rhs_ref = rhs_cast.as_ref();

    // Compute broadcasted shape
    let output_shape = lhs_ref.shape().broadcast_with(rhs_ref.shape())?;

    if output_shape.numel() == 0 {
        let mut output = Tensor::empty(
            output_shape.clone(),
            result_dtype,
            lhs.device(),
            requires_grad,
        );

        if requires_grad {
            let grad_fn = Arc::new(AddBackward {
                input_shapes: [lhs.shape().dims().to_vec(), rhs.shape().dims().to_vec()],
                input_ids: [lhs.id(), rhs.id()],
            });
            output.set_grad_fn(Some(grad_fn.clone()));
            add_to_graph(&output, Some(grad_fn))?;
        }

        return Ok(output);
    }

    // Create output tensor data
    let mut output_data =
        TensorData::uninitialized_on_device(output_shape.numel(), result_dtype, lhs.device());

    // Perform element-wise addition based on data type
    match result_dtype {
        DataType::Float32 => add_f32_direct(lhs_ref, rhs_ref, &mut output_data, &output_shape)?,
        DataType::Float64 => add_f64_direct(lhs_ref, rhs_ref, &mut output_data, &output_shape)?,
        DataType::Int32 => add_i32_direct(lhs_ref, rhs_ref, &mut output_data, &output_shape)?,
        DataType::Int64 => add_i64_direct(lhs_ref, rhs_ref, &mut output_data, &output_shape)?,
        DataType::Bool => add_bool_direct(lhs_ref, rhs_ref, &mut output_data, &output_shape)?,
    }

    // Create output tensor
    let mut output = Tensor::new(
        Arc::new(output_data),
        output_shape.clone(),
        result_dtype,
        lhs.device(),
        requires_grad,
    );

    // Set up gradient function if needed
    if requires_grad {
        let grad_fn = Arc::new(AddBackward {
            input_shapes: [lhs.shape().dims().to_vec(), rhs.shape().dims().to_vec()],
            input_ids: [lhs.id(), rhs.id()],
        });

        output.set_grad_fn(Some(grad_fn.clone()));

        // Add to computation graph
        add_to_graph(&output, Some(grad_fn))?;
    }

    Ok(output)
}

/// In-place element-wise addition used for gradient accumulation
pub fn add_inplace(lhs: &mut Tensor, rhs: &Tensor) -> Result<()> {
    if lhs.shape() != rhs.shape() {
        return Err(MinitensorError::shape_mismatch(
            lhs.shape().dims().to_vec(),
            rhs.shape().dims().to_vec(),
        ));
    }
    if lhs.dtype() != rhs.dtype() {
        return Err(MinitensorError::type_mismatch(
            format!("{:?}", lhs.dtype()),
            format!("{:?}", rhs.dtype()),
        ));
    }
    if lhs.device() != rhs.device() {
        return Err(MinitensorError::device_mismatch(
            format!("{:?}", lhs.device()),
            format!("{:?}", rhs.device()),
        ));
    }
    if std::sync::Arc::strong_count(lhs.data()) > 1 {
        // Fallback to out-of-place addition if data is shared
        let tmp = add(lhs, rhs)?;
        *lhs = tmp;
        return Ok(());
    }

    match lhs.dtype() {
        DataType::Float32 => {
            let lhs_slice = lhs.data_mut().as_f32_slice_mut().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get mutable f32 slice from lhs tensor")
            })?;
            let rhs_slice = rhs.data().as_f32_slice().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get f32 slice from rhs tensor")
            })?;
            let len = lhs_slice.len();
            if len < PAR_THRESHOLD {
                for i in 0..len {
                    lhs_slice[i] += rhs_slice[i];
                }
            } else {
                let lhs_ptr = lhs_slice.as_mut_ptr() as usize;
                let rhs_ptr = rhs_slice.as_ptr() as usize;
                (0..len).into_par_iter().for_each(|i| unsafe {
                    let lhs_ptr = lhs_ptr as *mut f32;
                    let rhs_ptr = rhs_ptr as *const f32;
                    *lhs_ptr.add(i) += *rhs_ptr.add(i);
                });
            }
        }
        DataType::Float64 => {
            let lhs_slice = lhs.data_mut().as_f64_slice_mut().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get mutable f64 slice from lhs tensor")
            })?;
            let rhs_slice = rhs.data().as_f64_slice().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get f64 slice from rhs tensor")
            })?;
            let len = lhs_slice.len();
            if len < PAR_THRESHOLD {
                for i in 0..len {
                    lhs_slice[i] += rhs_slice[i];
                }
            } else {
                let lhs_ptr = lhs_slice.as_mut_ptr() as usize;
                let rhs_ptr = rhs_slice.as_ptr() as usize;
                (0..len).into_par_iter().for_each(|i| unsafe {
                    let lhs_ptr = lhs_ptr as *mut f64;
                    let rhs_ptr = rhs_ptr as *const f64;
                    *lhs_ptr.add(i) += *rhs_ptr.add(i);
                });
            }
        }
        DataType::Int32 => {
            let lhs_slice = lhs.data_mut().as_i32_slice_mut().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get mutable i32 slice from lhs tensor")
            })?;
            let rhs_slice = rhs.data().as_i32_slice().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get i32 slice from rhs tensor")
            })?;
            let len = lhs_slice.len();
            if len < PAR_THRESHOLD {
                for i in 0..len {
                    lhs_slice[i] += rhs_slice[i];
                }
            } else {
                let lhs_ptr = lhs_slice.as_mut_ptr() as usize;
                let rhs_ptr = rhs_slice.as_ptr() as usize;
                (0..len).into_par_iter().for_each(|i| unsafe {
                    let lhs_ptr = lhs_ptr as *mut i32;
                    let rhs_ptr = rhs_ptr as *const i32;
                    *lhs_ptr.add(i) += *rhs_ptr.add(i);
                });
            }
        }
        DataType::Int64 => {
            let lhs_slice = lhs.data_mut().as_i64_slice_mut().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get mutable i64 slice from lhs tensor")
            })?;
            let rhs_slice = rhs.data().as_i64_slice().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get i64 slice from rhs tensor")
            })?;
            let len = lhs_slice.len();
            if len < PAR_THRESHOLD {
                for i in 0..len {
                    lhs_slice[i] += rhs_slice[i];
                }
            } else {
                let lhs_ptr = lhs_slice.as_mut_ptr() as usize;
                let rhs_ptr = rhs_slice.as_ptr() as usize;
                (0..len).into_par_iter().for_each(|i| unsafe {
                    let lhs_ptr = lhs_ptr as *mut i64;
                    let rhs_ptr = rhs_ptr as *const i64;
                    *lhs_ptr.add(i) += *rhs_ptr.add(i);
                });
            }
        }
        DataType::Bool => {
            let lhs_slice = lhs.data_mut().as_bool_slice_mut().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get mutable bool slice from lhs tensor")
            })?;
            let rhs_slice = rhs.data().as_bool_slice().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get bool slice from rhs tensor")
            })?;
            let len = lhs_slice.len();
            if len < PAR_THRESHOLD {
                for i in 0..len {
                    lhs_slice[i] = lhs_slice[i] || rhs_slice[i];
                }
            } else {
                let lhs_ptr = lhs_slice.as_mut_ptr() as usize;
                let rhs_ptr = rhs_slice.as_ptr() as usize;
                (0..len).into_par_iter().for_each(|i| unsafe {
                    let lhs_ptr = lhs_ptr as *mut bool;
                    let rhs_ptr = rhs_ptr as *const bool;
                    *lhs_ptr.add(i) = *lhs_ptr.add(i) || *rhs_ptr.add(i);
                });
            }
        }
    }
    Ok(())
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

    let requires_grad = lhs.requires_grad() || rhs.requires_grad();
    let (lhs_cast, rhs_cast, result_dtype) = coerce_binary_operands(lhs, rhs, BinaryOpKind::Sub)?;
    let lhs_ref = lhs_cast.as_ref();
    let rhs_ref = rhs_cast.as_ref();

    // Compute broadcasted shape
    let output_shape = lhs_ref.shape().broadcast_with(rhs_ref.shape())?;

    if output_shape.numel() == 0 {
        let mut output = Tensor::empty(
            output_shape.clone(),
            result_dtype,
            lhs.device(),
            requires_grad,
        );

        if requires_grad {
            let grad_fn = Arc::new(SubBackward {
                input_shapes: [lhs.shape().dims().to_vec(), rhs.shape().dims().to_vec()],
                input_ids: [lhs.id(), rhs.id()],
            });
            output.set_grad_fn(Some(grad_fn.clone()));
            add_to_graph(&output, Some(grad_fn))?;
        }

        return Ok(output);
    }

    // Create output tensor data
    let mut output_data =
        TensorData::uninitialized_on_device(output_shape.numel(), result_dtype, lhs.device());

    // Perform element-wise subtraction based on data type
    match result_dtype {
        DataType::Float32 => sub_f32_direct(lhs_ref, rhs_ref, &mut output_data, &output_shape)?,
        DataType::Float64 => sub_f64_direct(lhs_ref, rhs_ref, &mut output_data, &output_shape)?,
        DataType::Int32 => sub_i32_direct(lhs_ref, rhs_ref, &mut output_data, &output_shape)?,
        DataType::Int64 => sub_i64_direct(lhs_ref, rhs_ref, &mut output_data, &output_shape)?,
        DataType::Bool => unreachable!("boolean subtraction should be rejected during coercion"),
    }

    // Create output tensor
    let mut output = Tensor::new(
        Arc::new(output_data),
        output_shape.clone(),
        result_dtype,
        lhs.device(),
        requires_grad,
    );

    // Set up gradient function if needed
    if requires_grad {
        let grad_fn = Arc::new(SubBackward {
            input_shapes: [lhs.shape().dims().to_vec(), rhs.shape().dims().to_vec()],
            input_ids: [lhs.id(), rhs.id()],
        });

        output.set_grad_fn(Some(grad_fn.clone()));
        add_to_graph(&output, Some(grad_fn))?;
    }

    Ok(output)
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

    let requires_grad = lhs.requires_grad() || rhs.requires_grad();
    let (lhs_cast, rhs_cast, result_dtype) = coerce_binary_operands(lhs, rhs, BinaryOpKind::Mul)?;
    let lhs_ref = lhs_cast.as_ref();
    let rhs_ref = rhs_cast.as_ref();

    // Compute broadcasted shape
    let output_shape = lhs_ref.shape().broadcast_with(rhs_ref.shape())?;

    if output_shape.numel() == 0 {
        let mut output = Tensor::empty(
            output_shape.clone(),
            result_dtype,
            lhs.device(),
            requires_grad,
        );

        if requires_grad {
            let grad_fn = Arc::new(MulBackward {
                lhs: lhs.clone(),
                rhs: rhs.clone(),
                input_ids: [lhs.id(), rhs.id()],
            });
            output.set_grad_fn(Some(grad_fn.clone()));
            add_to_graph(&output, Some(grad_fn))?;
        }

        return Ok(output);
    }

    // Create output tensor data
    let mut output_data =
        TensorData::uninitialized_on_device(output_shape.numel(), result_dtype, lhs.device());

    // Perform element-wise multiplication based on data type
    match result_dtype {
        DataType::Float32 => mul_f32_direct(lhs_ref, rhs_ref, &mut output_data, &output_shape)?,
        DataType::Float64 => mul_f64_direct(lhs_ref, rhs_ref, &mut output_data, &output_shape)?,
        DataType::Int32 => mul_i32_direct(lhs_ref, rhs_ref, &mut output_data, &output_shape)?,
        DataType::Int64 => mul_i64_direct(lhs_ref, rhs_ref, &mut output_data, &output_shape)?,
        DataType::Bool => mul_bool_direct(lhs_ref, rhs_ref, &mut output_data, &output_shape)?,
    }

    // Create output tensor
    let mut output = Tensor::new(
        Arc::new(output_data),
        output_shape.clone(),
        result_dtype,
        lhs.device(),
        requires_grad,
    );

    // Set up gradient function if needed
    if requires_grad {
        let grad_fn = Arc::new(MulBackward {
            lhs: lhs.clone(),
            rhs: rhs.clone(),
            input_ids: [lhs.id(), rhs.id()],
        });

        output.set_grad_fn(Some(grad_fn.clone()));
        add_to_graph(&output, Some(grad_fn))?;
    }

    Ok(output)
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

    let requires_grad = lhs.requires_grad() || rhs.requires_grad();
    let (lhs_cast, rhs_cast, result_dtype) = coerce_binary_operands(lhs, rhs, BinaryOpKind::Div)?;
    let lhs_ref = lhs_cast.as_ref();
    let rhs_ref = rhs_cast.as_ref();

    // Compute broadcasted shape
    let output_shape = lhs_ref.shape().broadcast_with(rhs_ref.shape())?;

    if output_shape.numel() == 0 {
        let mut output = Tensor::empty(
            output_shape.clone(),
            result_dtype,
            lhs.device(),
            requires_grad,
        );

        if requires_grad {
            let grad_fn = Arc::new(DivBackward {
                lhs: lhs.clone(),
                rhs: rhs.clone(),
                input_ids: [lhs.id(), rhs.id()],
            });
            output.set_grad_fn(Some(grad_fn.clone()));
            add_to_graph(&output, Some(grad_fn))?;
        }

        return Ok(output);
    }

    // Create output tensor data
    let mut output_data =
        TensorData::uninitialized_on_device(output_shape.numel(), result_dtype, lhs.device());

    // Perform element-wise division based on data type
    match result_dtype {
        DataType::Float32 => div_f32_direct(lhs_ref, rhs_ref, &mut output_data, &output_shape)?,
        DataType::Float64 => div_f64_direct(lhs_ref, rhs_ref, &mut output_data, &output_shape)?,
        DataType::Int32 | DataType::Int64 | DataType::Bool => {
            unreachable!("integer and boolean division should coerce to floating point")
        }
    }

    // Create output tensor
    let mut output = Tensor::new(
        Arc::new(output_data),
        output_shape.clone(),
        result_dtype,
        lhs.device(),
        requires_grad,
    );

    // Set up gradient function if needed
    if requires_grad {
        let grad_fn = Arc::new(DivBackward {
            lhs: lhs.clone(),
            rhs: rhs.clone(),
            input_ids: [lhs.id(), rhs.id()],
        });

        output.set_grad_fn(Some(grad_fn.clone()));
        add_to_graph(&output, Some(grad_fn))?;
    }

    Ok(output)
}

/// Element-wise negation
pub fn neg(tensor: &Tensor) -> Result<Tensor> {
    let mut output_data = TensorData::uninitialized_on_device(
        tensor.shape().numel(),
        tensor.dtype(),
        tensor.device(),
    );

    match tensor.dtype() {
        DataType::Float32 => {
            let input = tensor.data().as_f32_slice().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get f32 slice from tensor")
            })?;
            let output = output_data.as_f32_slice_mut().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get mutable f32 slice from output")
            })?;
            if input.len() >= PAR_THRESHOLD {
                output
                    .par_iter_mut()
                    .zip(input.par_iter())
                    .for_each(|(o, &i)| *o = -i);
            } else {
                for (o, &i) in output.iter_mut().zip(input.iter()) {
                    *o = -i;
                }
            }
        }
        DataType::Float64 => {
            let input = tensor.data().as_f64_slice().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get f64 slice from tensor")
            })?;
            let output = output_data.as_f64_slice_mut().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get mutable f64 slice from output")
            })?;
            if input.len() >= PAR_THRESHOLD {
                output
                    .par_iter_mut()
                    .zip(input.par_iter())
                    .for_each(|(o, &i)| *o = -i);
            } else {
                for (o, &i) in output.iter_mut().zip(input.iter()) {
                    *o = -i;
                }
            }
        }
        DataType::Int32 => {
            let input = tensor.data().as_i32_slice().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get i32 slice from tensor")
            })?;
            let output = output_data.as_i32_slice_mut().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get mutable i32 slice from output")
            })?;
            if input.len() >= PAR_THRESHOLD {
                output
                    .par_iter_mut()
                    .zip(input.par_iter())
                    .for_each(|(o, &i)| *o = -i);
            } else {
                for (o, &i) in output.iter_mut().zip(input.iter()) {
                    *o = -i;
                }
            }
        }
        DataType::Int64 => {
            let input = tensor.data().as_i64_slice().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get i64 slice from tensor")
            })?;
            let output = output_data.as_i64_slice_mut().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get mutable i64 slice from output")
            })?;
            if input.len() >= PAR_THRESHOLD {
                output
                    .par_iter_mut()
                    .zip(input.par_iter())
                    .for_each(|(o, &i)| *o = -i);
            } else {
                for (o, &i) in output.iter_mut().zip(input.iter()) {
                    *o = -i;
                }
            }
        }
        DataType::Bool => {
            return Err(MinitensorError::invalid_operation(
                "Negation not supported for boolean tensors",
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

    if output.requires_grad() {
        let grad_fn = Arc::new(NegBackward {
            input_id: tensor.id(),
        });
        let mut out_with_grad = output;
        out_with_grad.set_grad_fn(Some(grad_fn.clone()));
        add_to_graph(&out_with_grad, Some(grad_fn))?;
        Ok(out_with_grad)
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

fn add_bool_direct(
    lhs: &Tensor,
    rhs: &Tensor,
    output_data: &mut TensorData,
    output_shape: &Shape,
) -> Result<()> {
    let lhs_data = lhs.data().as_bool_slice().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get bool slice from lhs tensor")
    })?;
    let rhs_data = rhs.data().as_bool_slice().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get bool slice from rhs tensor")
    })?;

    let output_slice = output_data.as_bool_slice_mut().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get mutable bool slice from output data")
    })?;

    broadcast_binary_op(
        lhs_data,
        rhs_data,
        output_slice,
        lhs.shape(),
        rhs.shape(),
        output_shape,
        |a, b| a || b,
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

fn mul_bool_direct(
    lhs: &Tensor,
    rhs: &Tensor,
    output_data: &mut TensorData,
    output_shape: &Shape,
) -> Result<()> {
    let lhs_data = lhs.data().as_bool_slice().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get bool slice from lhs tensor")
    })?;
    let rhs_data = rhs.data().as_bool_slice().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get bool slice from rhs tensor")
    })?;

    let output_slice = output_data.as_bool_slice_mut().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get mutable bool slice from output data")
    })?;

    broadcast_binary_op(
        lhs_data,
        rhs_data,
        output_slice,
        lhs.shape(),
        rhs.shape(),
        output_shape,
        |a, b| a && b,
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
                if b == 0.0 { f32::INFINITY } else { a / b }
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
                if b == 0.0 { f64::INFINITY } else { a / b }
            },
        )
    }
}

/// Generic broadcasting binary operation
pub(crate) fn broadcast_binary_op<T, F>(
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
    let output_dims = output_shape.dims();
    let lhs_dims = lhs_shape.dims();
    let rhs_dims = rhs_shape.dims();
    let rank = output_dims.len();

    if output_shape.numel() == 0 || output_dims.iter().any(|&dim| dim == 0) {
        return Ok(());
    }

    // Fast path when no broadcasting is required. This avoids the
    // relatively expensive index mapping logic below and simply applies the
    // operation element-wise. We use parallel iteration for large tensors and
    // fall back to a simple loop for smaller ones to reduce rayon overhead.
    if lhs_dims == output_dims && rhs_dims == output_dims {
        if output_data.len() >= 1024 {
            output_data
                .par_iter_mut()
                .zip(lhs_data.par_iter().zip(rhs_data.par_iter()))
                .for_each(|(out, (l, r))| {
                    *out = op(*l, *r);
                });
        } else {
            for ((out, &l), &r) in output_data
                .iter_mut()
                .zip(lhs_data.iter())
                .zip(rhs_data.iter())
            {
                *out = op(l, r);
            }
        }
        return Ok(());
    }

    // Fast path when one side is a scalar and the other already matches the
    // output shape. This avoids the more expensive coordinate calculation
    // used for general broadcasting. We again switch between parallel and
    // sequential execution based on tensor size to minimize overhead.
    if lhs_data.len() == 1 && rhs_dims == output_dims {
        let lhs_val = lhs_data[0];
        if output_data.len() >= 1024 {
            output_data
                .par_iter_mut()
                .zip(rhs_data.par_iter())
                .for_each(|(out, &r)| {
                    *out = op(lhs_val, r);
                });
        } else {
            for (out, &r) in output_data.iter_mut().zip(rhs_data.iter()) {
                *out = op(lhs_val, r);
            }
        }
        return Ok(());
    }

    if rhs_data.len() == 1 && lhs_dims == output_dims {
        let rhs_val = rhs_data[0];
        if output_data.len() >= 1024 {
            output_data
                .par_iter_mut()
                .zip(lhs_data.par_iter())
                .for_each(|(out, &l)| {
                    *out = op(l, rhs_val);
                });
        } else {
            for (out, &l) in output_data.iter_mut().zip(lhs_data.iter()) {
                *out = op(l, rhs_val);
            }
        }
        return Ok(());
    }

    let lhs_contiguous = Strides::from_shape(lhs_shape);
    let rhs_contiguous = Strides::from_shape(rhs_shape);
    let lhs_strides = lhs_contiguous.as_slice();
    let rhs_strides = rhs_contiguous.as_slice();

    let mut lhs_aligned: SmallVec<[usize; 8]> = smallvec![0; rank];
    let mut rhs_aligned: SmallVec<[usize; 8]> = smallvec![0; rank];

    let lhs_offset = rank.saturating_sub(lhs_dims.len());
    for (i, &dim) in lhs_dims.iter().enumerate() {
        lhs_aligned[lhs_offset + i] = if dim == 1 { 0 } else { lhs_strides[i] };
    }

    let rhs_offset = rank.saturating_sub(rhs_dims.len());
    for (i, &dim) in rhs_dims.iter().enumerate() {
        rhs_aligned[rhs_offset + i] = if dim == 1 { 0 } else { rhs_strides[i] };
    }

    // For small tensors, a simple sequential loop is faster than spawning
    // rayon tasks. We use the same index mapping logic but without parallel
    // chunking to minimize overhead.
    if output_data.len() < 1024 {
        let lhs_ptr = lhs_data.as_ptr();
        let rhs_ptr = rhs_data.as_ptr();
        for (idx, out) in output_data.iter_mut().enumerate() {
            let mut lhs_idx = 0usize;
            let mut rhs_idx = 0usize;
            let mut tmp = idx;
            for i in (0..rank).rev() {
                let coord = tmp % output_dims[i];
                tmp /= output_dims[i];
                lhs_idx += coord * lhs_aligned[i];
                rhs_idx += coord * rhs_aligned[i];
            }
            unsafe {
                *out = op(*lhs_ptr.add(lhs_idx), *rhs_ptr.add(rhs_idx));
            }
        }
        return Ok(());
    }

    const CHUNK: usize = 1024;
    output_data
        .par_chunks_mut(CHUNK)
        .enumerate()
        .for_each(|(chunk_idx, out_chunk)| {
            let start = chunk_idx * CHUNK;
            let mut coord: SmallVec<[usize; 8]> = smallvec![0; rank];
            let mut tmp = start;
            for i in (0..rank).rev() {
                coord[i] = tmp % output_dims[i];
                tmp /= output_dims[i];
            }

            let mut lhs_idx = 0usize;
            let mut rhs_idx = 0usize;
            for i in 0..rank {
                lhs_idx += coord[i] * lhs_aligned[i];
                rhs_idx += coord[i] * rhs_aligned[i];
            }

            let lhs_ptr = lhs_data.as_ptr();
            let rhs_ptr = rhs_data.as_ptr();
            for out in out_chunk.iter_mut() {
                unsafe {
                    *out = op(*lhs_ptr.add(lhs_idx), *rhs_ptr.add(rhs_idx));
                }
                for i in (0..rank).rev() {
                    coord[i] += 1;
                    lhs_idx += lhs_aligned[i];
                    rhs_idx += rhs_aligned[i];
                    if coord[i] < output_dims[i] {
                        break;
                    }
                    coord[i] = 0;
                    lhs_idx -= lhs_aligned[i] * output_dims[i];
                    rhs_idx -= rhs_aligned[i] * output_dims[i];
                }
            }
        });

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
    fn test_neg_basic() {
        let a = create_test_tensor_f32(vec![1.0, -2.0, 3.5], vec![3], false);
        let result = neg(&a).unwrap();
        let data = result.data().as_f32_slice().unwrap();
        assert_eq!(data, &[-1.0, 2.0, -3.5]);
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
    fn test_mixed_dtype_promotion() {
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

        let result = add(&a, &b).unwrap();
        assert_eq!(result.dtype(), DataType::Float32);
        assert_eq!(result.data().as_f32_slice().unwrap(), &[4.0, 6.0]);
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
    fn test_bool_arithmetic_behaviour() {
        // Create boolean tensors
        let shape_obj = Shape::new(vec![2]);
        let mut data_a = TensorData::zeros(shape_obj.numel(), DataType::Bool);
        if let Some(slice) = data_a.as_bool_slice_mut() {
            slice.copy_from_slice(&[true, false]);
        }
        let a = Tensor::new(
            Arc::new(data_a),
            shape_obj.clone(),
            DataType::Bool,
            Device::cpu(),
            false,
        );

        let mut data_b = TensorData::zeros(shape_obj.numel(), DataType::Bool);
        if let Some(slice) = data_b.as_bool_slice_mut() {
            slice.copy_from_slice(&[false, true]);
        }
        let b = Tensor::new(
            Arc::new(data_b),
            shape_obj,
            DataType::Bool,
            Device::cpu(),
            false,
        );

        let add_result = add(&a, &b).unwrap();
        assert_eq!(add_result.dtype(), DataType::Bool);
        assert_eq!(add_result.data().as_bool_slice().unwrap(), &[true, true]);
        assert!(sub(&a, &b).is_err());
        let mul_result = mul(&a, &b).unwrap();
        assert_eq!(mul_result.dtype(), DataType::Bool);
        assert_eq!(mul_result.data().as_bool_slice().unwrap(), &[false, false]);
        let div_result = div(&a, &b).unwrap();
        assert_eq!(div_result.dtype(), DataType::Float32);
        assert_eq!(
            div_result.data().as_f32_slice().unwrap(),
            &[f32::INFINITY, 0.0]
        );
        assert!(neg(&a).is_err());
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

    #[test]
    fn test_add_handles_zero_sized_broadcast() {
        let a = create_test_tensor_f32(vec![], vec![0, 3], false);
        let b = create_test_tensor_f32(vec![1.0, 2.0, 3.0], vec![3], false);

        let result = add(&a, &b).unwrap();
        assert_eq!(result.shape().dims(), &[0, 3]);
        assert_eq!(result.data().as_f32_slice().unwrap().len(), 0);
    }

    #[test]
    fn test_add_handles_zero_sized_broadcast_from_vec() {
        use crate::tensor::TensorData;

        let a_data = TensorData::from_vec::<f32>(vec![], DataType::Float32, Device::cpu());
        let a = Tensor::new(
            Arc::new(a_data),
            Shape::new(vec![0, 3]),
            DataType::Float32,
            Device::cpu(),
            false,
        );

        let b_data =
            TensorData::from_vec::<f32>(vec![1.0_f32, 2.0, 3.0], DataType::Float32, Device::cpu());
        let b = Tensor::new(
            Arc::new(b_data),
            Shape::new(vec![3]),
            DataType::Float32,
            Device::cpu(),
            false,
        );

        let result = add(&a, &b).unwrap();
        assert_eq!(result.shape().dims(), &[0, 3]);
        assert_eq!(result.data().as_f32_slice().unwrap().len(), 0);
    }
}
