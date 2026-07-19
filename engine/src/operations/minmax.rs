// Copyright (c) 2025 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

use crate::{
    autograd::{GradientFunction, MaximumBackward, MinimumBackward, add_to_graph},
    error::{MinitensorError, Result},
    operations::{
        arithmetic::broadcast_binary_map,
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

    /// One dtype arm: fetch both slices, apply the max/min closure with
    /// broadcasting into a fresh buffer, and wrap it as `TensorData`.
    macro_rules! minmax_arm {
        ($accessor:ident, $ty:ty, $dtype:ident, $tyname:literal, $max:expr, $min:expr) => {{
            let lhs_slice = lhs_ref.data().$accessor().ok_or_else(|| {
                MinitensorError::internal_error(concat!(
                    "Failed to get ",
                    $tyname,
                    " slice from lhs tensor"
                ))
            })?;
            let rhs_slice = rhs_ref.data().$accessor().ok_or_else(|| {
                MinitensorError::internal_error(concat!(
                    "Failed to get ",
                    $tyname,
                    " slice from rhs tensor"
                ))
            })?;
            let out = match op {
                BinaryOpKind::Maximum => broadcast_binary_map(
                    lhs_slice,
                    rhs_slice,
                    lhs_ref.shape(),
                    rhs_ref.shape(),
                    &output_shape,
                    $max,
                )?,
                BinaryOpKind::Minimum => broadcast_binary_map(
                    lhs_slice,
                    rhs_slice,
                    lhs_ref.shape(),
                    rhs_ref.shape(),
                    &output_shape,
                    $min,
                )?,
                _ => unreachable!(),
            };
            TensorData::from_vec::<$ty>(out, DataType::$dtype, lhs.device())
        }};
    }

    // NaN-propagating comparisons for floats (matching the previous
    // hand-written kernels); plain ordering for ints, OR/AND for bool.
    let output_data = match result_dtype {
        DataType::Float32 => minmax_arm!(
            as_f32_slice,
            f32,
            Float32,
            "f32",
            |a: f32, b: f32| {
                if a.is_nan() || b.is_nan() {
                    if a.is_nan() { a } else { b }
                } else if a >= b {
                    a
                } else {
                    b
                }
            },
            |a: f32, b: f32| {
                if a.is_nan() || b.is_nan() {
                    if a.is_nan() { a } else { b }
                } else if a <= b {
                    a
                } else {
                    b
                }
            }
        ),
        DataType::Float64 => minmax_arm!(
            as_f64_slice,
            f64,
            Float64,
            "f64",
            |a: f64, b: f64| {
                if a.is_nan() || b.is_nan() {
                    if a.is_nan() { a } else { b }
                } else if a >= b {
                    a
                } else {
                    b
                }
            },
            |a: f64, b: f64| {
                if a.is_nan() || b.is_nan() {
                    if a.is_nan() { a } else { b }
                } else if a <= b {
                    a
                } else {
                    b
                }
            }
        ),
        DataType::Int32 => minmax_arm!(
            as_i32_slice,
            i32,
            Int32,
            "i32",
            |a: i32, b: i32| if a >= b { a } else { b },
            |a: i32, b: i32| if a <= b { a } else { b }
        ),
        DataType::Int64 => minmax_arm!(
            as_i64_slice,
            i64,
            Int64,
            "i64",
            |a: i64, b: i64| if a >= b { a } else { b },
            |a: i64, b: i64| if a <= b { a } else { b }
        ),
        DataType::Bool => minmax_arm!(
            as_bool_slice,
            bool,
            Bool,
            "bool",
            |a: bool, b: bool| a || b,
            |a: bool, b: bool| a && b
        ),
    };

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
