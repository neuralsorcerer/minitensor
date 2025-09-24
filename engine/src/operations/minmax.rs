// Copyright (c) 2025 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

use crate::{
    autograd::{GradientFunction, MaximumBackward, MinimumBackward, add_to_graph},
    error::{MinitensorError, Result},
    operations::{
        arithmetic::broadcast_binary_op,
        binary::{BinaryOpKind, coerce_binary_operands},
        comparison, selection,
    },
    tensor::{DataType, Tensor, TensorData},
};
use std::sync::Arc;

pub fn maximum(lhs: &Tensor, rhs: &Tensor) -> Result<Tensor> {
    binary_minmax(lhs, rhs, BinaryOpKind::Maximum)
}

pub fn minimum(lhs: &Tensor, rhs: &Tensor) -> Result<Tensor> {
    binary_minmax(lhs, rhs, BinaryOpKind::Minimum)
}

fn binary_minmax(lhs: &Tensor, rhs: &Tensor, op: BinaryOpKind) -> Result<Tensor> {
    if lhs.device() != rhs.device() {
        return Err(MinitensorError::device_mismatch(
            format!("{:?}", lhs.device()),
            format!("{:?}", rhs.device()),
        ));
    }

    let requires_grad = lhs.requires_grad() || rhs.requires_grad();
    let (lhs_cast, rhs_cast, result_dtype) = coerce_binary_operands(lhs, rhs, op)?;
    let lhs_ref = lhs_cast.as_ref();
    let rhs_ref = rhs_cast.as_ref();

    let output_shape = lhs_ref.shape().broadcast_with(rhs_ref.shape())?;
    let mut output_data =
        TensorData::uninitialized_on_device(output_shape.numel(), result_dtype, lhs.device());

    match result_dtype {
        DataType::Float32 => {
            let lhs_slice = lhs_ref.data().as_f32_slice().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get f32 slice from lhs tensor")
            })?;
            let rhs_slice = rhs_ref.data().as_f32_slice().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get f32 slice from rhs tensor")
            })?;
            let out_slice = output_data.as_f32_slice_mut().ok_or_else(|| {
                MinitensorError::internal_error(
                    "Failed to get mutable f32 slice from min/max output",
                )
            })?;
            match op {
                BinaryOpKind::Maximum => broadcast_binary_op(
                    lhs_slice,
                    rhs_slice,
                    out_slice,
                    lhs_ref.shape(),
                    rhs_ref.shape(),
                    &output_shape,
                    |a, b| {
                        if a.is_nan() || b.is_nan() {
                            if a.is_nan() { a } else { b }
                        } else if a >= b {
                            a
                        } else {
                            b
                        }
                    },
                )?,
                BinaryOpKind::Minimum => broadcast_binary_op(
                    lhs_slice,
                    rhs_slice,
                    out_slice,
                    lhs_ref.shape(),
                    rhs_ref.shape(),
                    &output_shape,
                    |a, b| {
                        if a.is_nan() || b.is_nan() {
                            if a.is_nan() { a } else { b }
                        } else if a <= b {
                            a
                        } else {
                            b
                        }
                    },
                )?,
                _ => unreachable!(),
            }
        }
        DataType::Float64 => {
            let lhs_slice = lhs_ref.data().as_f64_slice().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get f64 slice from lhs tensor")
            })?;
            let rhs_slice = rhs_ref.data().as_f64_slice().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get f64 slice from rhs tensor")
            })?;
            let out_slice = output_data.as_f64_slice_mut().ok_or_else(|| {
                MinitensorError::internal_error(
                    "Failed to get mutable f64 slice from min/max output",
                )
            })?;
            match op {
                BinaryOpKind::Maximum => broadcast_binary_op(
                    lhs_slice,
                    rhs_slice,
                    out_slice,
                    lhs_ref.shape(),
                    rhs_ref.shape(),
                    &output_shape,
                    |a, b| {
                        if a.is_nan() || b.is_nan() {
                            if a.is_nan() { a } else { b }
                        } else if a >= b {
                            a
                        } else {
                            b
                        }
                    },
                )?,
                BinaryOpKind::Minimum => broadcast_binary_op(
                    lhs_slice,
                    rhs_slice,
                    out_slice,
                    lhs_ref.shape(),
                    rhs_ref.shape(),
                    &output_shape,
                    |a, b| {
                        if a.is_nan() || b.is_nan() {
                            if a.is_nan() { a } else { b }
                        } else if a <= b {
                            a
                        } else {
                            b
                        }
                    },
                )?,
                _ => unreachable!(),
            }
        }
        DataType::Int32 => {
            let lhs_slice = lhs_ref.data().as_i32_slice().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get i32 slice from lhs tensor")
            })?;
            let rhs_slice = rhs_ref.data().as_i32_slice().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get i32 slice from rhs tensor")
            })?;
            let out_slice = output_data.as_i32_slice_mut().ok_or_else(|| {
                MinitensorError::internal_error(
                    "Failed to get mutable i32 slice from min/max output",
                )
            })?;
            match op {
                BinaryOpKind::Maximum => broadcast_binary_op(
                    lhs_slice,
                    rhs_slice,
                    out_slice,
                    lhs_ref.shape(),
                    rhs_ref.shape(),
                    &output_shape,
                    |a, b| if a >= b { a } else { b },
                )?,
                BinaryOpKind::Minimum => broadcast_binary_op(
                    lhs_slice,
                    rhs_slice,
                    out_slice,
                    lhs_ref.shape(),
                    rhs_ref.shape(),
                    &output_shape,
                    |a, b| if a <= b { a } else { b },
                )?,
                _ => unreachable!(),
            }
        }
        DataType::Int64 => {
            let lhs_slice = lhs_ref.data().as_i64_slice().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get i64 slice from lhs tensor")
            })?;
            let rhs_slice = rhs_ref.data().as_i64_slice().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get i64 slice from rhs tensor")
            })?;
            let out_slice = output_data.as_i64_slice_mut().ok_or_else(|| {
                MinitensorError::internal_error(
                    "Failed to get mutable i64 slice from min/max output",
                )
            })?;
            match op {
                BinaryOpKind::Maximum => broadcast_binary_op(
                    lhs_slice,
                    rhs_slice,
                    out_slice,
                    lhs_ref.shape(),
                    rhs_ref.shape(),
                    &output_shape,
                    |a, b| if a >= b { a } else { b },
                )?,
                BinaryOpKind::Minimum => broadcast_binary_op(
                    lhs_slice,
                    rhs_slice,
                    out_slice,
                    lhs_ref.shape(),
                    rhs_ref.shape(),
                    &output_shape,
                    |a, b| if a <= b { a } else { b },
                )?,
                _ => unreachable!(),
            }
        }
        DataType::Bool => {
            let lhs_slice = lhs_ref.data().as_bool_slice().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get bool slice from lhs tensor")
            })?;
            let rhs_slice = rhs_ref.data().as_bool_slice().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get bool slice from rhs tensor")
            })?;
            let out_slice = output_data.as_bool_slice_mut().ok_or_else(|| {
                MinitensorError::internal_error(
                    "Failed to get mutable bool slice from min/max output",
                )
            })?;
            match op {
                BinaryOpKind::Maximum => broadcast_binary_op(
                    lhs_slice,
                    rhs_slice,
                    out_slice,
                    lhs_ref.shape(),
                    rhs_ref.shape(),
                    &output_shape,
                    |a, b| a || b,
                )?,
                BinaryOpKind::Minimum => broadcast_binary_op(
                    lhs_slice,
                    rhs_slice,
                    out_slice,
                    lhs_ref.shape(),
                    rhs_ref.shape(),
                    &output_shape,
                    |a, b| a && b,
                )?,
                _ => unreachable!(),
            }
        }
    }

    let grad_enabled = requires_grad && result_dtype.is_float();
    let mut output = Tensor::new(
        Arc::new(output_data),
        output_shape.clone(),
        result_dtype,
        lhs.device(),
        grad_enabled,
    );

    if grad_enabled {
        let grad_fn: Arc<dyn GradientFunction> = match op {
            BinaryOpKind::Maximum => Arc::new(MaximumBackward {
                lhs: lhs_ref.detach(),
                rhs: rhs_ref.detach(),
                input_shapes: [lhs.shape().dims().to_vec(), rhs.shape().dims().to_vec()],
                input_requires_grad: [lhs.requires_grad(), rhs.requires_grad()],
                input_ids: [lhs.id(), rhs.id()],
            }),
            BinaryOpKind::Minimum => Arc::new(MinimumBackward {
                lhs: lhs_ref.detach(),
                rhs: rhs_ref.detach(),
                input_shapes: [lhs.shape().dims().to_vec(), rhs.shape().dims().to_vec()],
                input_requires_grad: [lhs.requires_grad(), rhs.requires_grad()],
                input_ids: [lhs.id(), rhs.id()],
            }),
            _ => unreachable!(),
        };
        output.set_grad_fn(Some(grad_fn.clone()));
        add_to_graph(&output, Some(grad_fn))?;
    }

    Ok(output)
}

pub(crate) fn maximum_backward_mask(lhs: &Tensor, rhs: &Tensor) -> Result<Tensor> {
    comparison::ge(lhs, rhs)
}

pub(crate) fn minimum_backward_mask(lhs: &Tensor, rhs: &Tensor) -> Result<Tensor> {
    comparison::le(lhs, rhs)
}

pub(crate) fn select_with_mask(
    mask: &Tensor,
    when_true: &Tensor,
    when_false: &Tensor,
) -> Result<Tensor> {
    selection::where_op(mask, when_true, when_false)
}
