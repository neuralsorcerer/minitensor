// Copyright (c) 2026 Soumyadip Sarkar.
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
