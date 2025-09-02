// Copyright (c) 2025 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.


use crate::{
    tensor::{Tensor, TensorData, DataType, Shape},
    error::{MinitensorError, Result},
};
use std::sync::Arc;

fn broadcast_compare_op<T: Copy>(
    lhs_data: &[T],
    rhs_data: &[T],
    output_data: &mut [bool],
    lhs_shape: &Shape,
    rhs_shape: &Shape,
    output_shape: &Shape,
    op: impl Fn(T, T) -> bool,
) -> Result<()> {
    let output_dims = output_shape.dims();
    let lhs_dims = lhs_shape.dims();
    let rhs_dims = rhs_shape.dims();

    for output_idx in 0..output_shape.numel() {
        // Convert linear index to coordinates
        let mut output_coords = vec![0; output_dims.len()];
        let mut tmp = output_idx;
        for i in (0..output_dims.len()).rev() {
            output_coords[i] = tmp % output_dims[i];
            tmp /= output_dims[i];
        }

        // Map to lhs index
        let mut lhs_idx = 0;
        let lhs_offset = output_dims.len().saturating_sub(lhs_dims.len());
        for i in 0..lhs_dims.len() {
            let coord = if lhs_dims[i] == 1 {
                0
            } else {
                output_coords[i + lhs_offset]
            };
            let mut stride = 1;
            for j in (i + 1)..lhs_dims.len() {
                stride *= lhs_dims[j];
            }
            lhs_idx += coord * stride;
        }

        // Map to rhs index
        let mut rhs_idx = 0;
        let rhs_offset = output_dims.len().saturating_sub(rhs_dims.len());
        for i in 0..rhs_dims.len() {
            let coord = if rhs_dims[i] == 1 {
                0
            } else {
                output_coords[i + rhs_offset]
            };
            let mut stride = 1;
            for j in (i + 1)..rhs_dims.len() {
                stride *= rhs_dims[j];
            }
            rhs_idx += coord * stride;
        }

        output_data[output_idx] = op(lhs_data[lhs_idx], rhs_data[rhs_idx]);
    }

    Ok(())
}

fn cmp_f32(
    lhs: &Tensor,
    rhs: &Tensor,
    output_data: &mut TensorData,
    output_shape: &Shape,
    op: impl Fn(f32, f32) -> bool,
) -> Result<()> {
    let lhs_slice = lhs
        .data()
        .as_f32_slice()
        .ok_or_else(|| MinitensorError::internal_error("Failed to get f32 slice from lhs tensor"))?;
    let rhs_slice = rhs
        .data()
        .as_f32_slice()
        .ok_or_else(|| MinitensorError::internal_error("Failed to get f32 slice from rhs tensor"))?;
    let output_slice = output_data
        .as_bool_slice_mut()
        .ok_or_else(|| MinitensorError::internal_error("Failed to get bool slice from output data"))?;
    broadcast_compare_op(lhs_slice, rhs_slice, output_slice, lhs.shape(), rhs.shape(), output_shape, op)
}

fn cmp_f64(
    lhs: &Tensor,
    rhs: &Tensor,
    output_data: &mut TensorData,
    output_shape: &Shape,
    op: impl Fn(f64, f64) -> bool,
) -> Result<()> {
    let lhs_slice = lhs
        .data()
        .as_f64_slice()
        .ok_or_else(|| MinitensorError::internal_error("Failed to get f64 slice from lhs tensor"))?;
    let rhs_slice = rhs
        .data()
        .as_f64_slice()
        .ok_or_else(|| MinitensorError::internal_error("Failed to get f64 slice from rhs tensor"))?;
    let output_slice = output_data
        .as_bool_slice_mut()
        .ok_or_else(|| MinitensorError::internal_error("Failed to get bool slice from output data"))?;
    broadcast_compare_op(lhs_slice, rhs_slice, output_slice, lhs.shape(), rhs.shape(), output_shape, op)
}

fn cmp_i32(
    lhs: &Tensor,
    rhs: &Tensor,
    output_data: &mut TensorData,
    output_shape: &Shape,
    op: impl Fn(i32, i32) -> bool,
) -> Result<()> {
    let lhs_slice = lhs
        .data()
        .as_i32_slice()
        .ok_or_else(|| MinitensorError::internal_error("Failed to get i32 slice from lhs tensor"))?;
    let rhs_slice = rhs
        .data()
        .as_i32_slice()
        .ok_or_else(|| MinitensorError::internal_error("Failed to get i32 slice from rhs tensor"))?;
    let output_slice = output_data
        .as_bool_slice_mut()
        .ok_or_else(|| MinitensorError::internal_error("Failed to get bool slice from output data"))?;
    broadcast_compare_op(lhs_slice, rhs_slice, output_slice, lhs.shape(), rhs.shape(), output_shape, op)
}

fn cmp_i64(
    lhs: &Tensor,
    rhs: &Tensor,
    output_data: &mut TensorData,
    output_shape: &Shape,
    op: impl Fn(i64, i64) -> bool,
) -> Result<()> {
    let lhs_slice = lhs
        .data()
        .as_i64_slice()
        .ok_or_else(|| MinitensorError::internal_error("Failed to get i64 slice from lhs tensor"))?;
    let rhs_slice = rhs
        .data()
        .as_i64_slice()
        .ok_or_else(|| MinitensorError::internal_error("Failed to get i64 slice from rhs tensor"))?;
    let output_slice = output_data
        .as_bool_slice_mut()
        .ok_or_else(|| MinitensorError::internal_error("Failed to get bool slice from output data"))?;
    broadcast_compare_op(lhs_slice, rhs_slice, output_slice, lhs.shape(), rhs.shape(), output_shape, op)
}

fn cmp_bool(
    lhs: &Tensor,
    rhs: &Tensor,
    output_data: &mut TensorData,
    output_shape: &Shape,
    op: impl Fn(bool, bool) -> bool,
) -> Result<()> {
    let lhs_slice = lhs
        .data()
        .as_bool_slice()
        .ok_or_else(|| MinitensorError::internal_error("Failed to get bool slice from lhs tensor"))?;
    let rhs_slice = rhs
        .data()
        .as_bool_slice()
        .ok_or_else(|| MinitensorError::internal_error("Failed to get bool slice from rhs tensor"))?;
    let output_slice = output_data
        .as_bool_slice_mut()
        .ok_or_else(|| MinitensorError::internal_error("Failed to get bool slice from output data"))?;
    broadcast_compare_op(lhs_slice, rhs_slice, output_slice, lhs.shape(), rhs.shape(), output_shape, op)
}

macro_rules! cmp_op {
    ($fn_name:ident, $op:tt, $bool_ok:expr) => {
        pub fn $fn_name(lhs: &Tensor, rhs: &Tensor) -> Result<Tensor> {
            if lhs.device() != rhs.device() {
                return Err(MinitensorError::device_mismatch(
                    format!("{:?}", lhs.device()),
                    format!("{:?}", rhs.device()),
                ));
            }
            if lhs.dtype() != rhs.dtype() {
                return Err(MinitensorError::type_mismatch(
                    format!("{:?}", lhs.dtype()),
                    format!("{:?}", rhs.dtype()),
                ));
            }

            let output_shape = lhs.shape().broadcast_with(rhs.shape())?;
            let mut output_data = TensorData::zeros_on_device(
                output_shape.numel(),
                DataType::Bool,
                lhs.device(),
            );

            match lhs.dtype() {
                DataType::Float32 => cmp_f32(lhs, rhs, &mut output_data, &output_shape, |a, b| a $op b)?,
                DataType::Float64 => cmp_f64(lhs, rhs, &mut output_data, &output_shape, |a, b| a $op b)?,
                DataType::Int32 => cmp_i32(lhs, rhs, &mut output_data, &output_shape, |a, b| a $op b)?,
                DataType::Int64 => cmp_i64(lhs, rhs, &mut output_data, &output_shape, |a, b| a $op b)?,
                DataType::Bool => {
                    if $bool_ok {
                        cmp_bool(lhs, rhs, &mut output_data, &output_shape, |a, b| a $op b)?
                    } else {
                        return Err(MinitensorError::invalid_operation(
                            "Comparison not supported for boolean tensors",
                        ));
                    }
                }
            }

            Ok(Tensor::new(
                Arc::new(output_data),
                output_shape,
                DataType::Bool,
                lhs.device(),
                false,
            ))
        }
    };
}

cmp_op!(eq, ==, true);
cmp_op!(ne, !=, true);
cmp_op!(lt, <, false);
cmp_op!(le, <=, false);
cmp_op!(gt, >, false);
cmp_op!(ge, >=, false);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{tensor::{TensorData, Shape, DataType}, device::Device};

    fn tensor_from_vec_f32(data: Vec<f32>) -> Tensor {
        let shape = Shape::new(vec![data.len()]);
        let data = TensorData::from_vec_f32(data, Device::cpu());
        Tensor::new(Arc::new(data), shape, DataType::Float32, Device::cpu(), false)
    }

    #[test]
    fn test_eq_basic() {
        let a = tensor_from_vec_f32(vec![1.0, 2.0, 3.0]);
        let b = tensor_from_vec_f32(vec![1.0, 0.0, 3.0]);
        let result = eq(&a, &b).unwrap();
        let slice = result.data().as_bool_slice().unwrap();
        assert_eq!(slice, &[true, false, true]);
    }
}
