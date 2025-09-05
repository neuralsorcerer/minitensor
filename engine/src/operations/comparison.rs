// Copyright (c) 2025 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

use crate::{
    error::{MinitensorError, Result},
    tensor::{DataType, Shape, Tensor, TensorData},
};
use rayon::prelude::*;
use std::sync::Arc;

fn compute_strides(dims: &[usize]) -> Vec<usize> {
    let mut strides = vec![1; dims.len()];
    for i in (0..dims.len()).rev() {
        if i + 1 < dims.len() {
            strides[i] = strides[i + 1] * dims[i + 1];
        }
    }
    strides
}

fn broadcast_compare_op<T: Copy + Send + Sync, F: Fn(T, T) -> bool + Sync + Send>(
    lhs_data: &[T],
    rhs_data: &[T],
    output_data: &mut [bool],
    lhs_shape: &Shape,
    rhs_shape: &Shape,
    output_shape: &Shape,
    op: F,
) -> Result<()> {
    let output_dims = output_shape.dims().to_vec();
    let lhs_dims = lhs_shape.dims().to_vec();
    let rhs_dims = rhs_shape.dims().to_vec();

    let lhs_strides = compute_strides(&lhs_dims);
    let rhs_strides = compute_strides(&rhs_dims);
    let lhs_offset = output_dims.len().saturating_sub(lhs_dims.len());
    let rhs_offset = output_dims.len().saturating_sub(rhs_dims.len());

    output_data
        .par_iter_mut()
        .enumerate()
        .for_each(|(output_idx, out)| {
            // Convert linear index to coordinates
            let mut coords = vec![0; output_dims.len()];
            let mut tmp = output_idx;
            for i in (0..output_dims.len()).rev() {
                coords[i] = tmp % output_dims[i];
                tmp /= output_dims[i];
            }

            // Map to lhs index
            let mut lhs_idx = 0;
            for i in 0..lhs_dims.len() {
                let coord = if lhs_dims[i] == 1 {
                    0
                } else {
                    coords[i + lhs_offset]
                };
                lhs_idx += coord * lhs_strides[i];
            }

            // Map to rhs index
            let mut rhs_idx = 0;
            for i in 0..rhs_dims.len() {
                let coord = if rhs_dims[i] == 1 {
                    0
                } else {
                    coords[i + rhs_offset]
                };
                rhs_idx += coord * rhs_strides[i];
            }

            *out = op(lhs_data[lhs_idx], rhs_data[rhs_idx]);
        });

    Ok(())
}

fn cmp_f32(
    lhs: &Tensor,
    rhs: &Tensor,
    output_data: &mut TensorData,
    output_shape: &Shape,
    op: impl Fn(f32, f32) -> bool + Sync + Send,
) -> Result<()> {
    let lhs_slice = lhs.data().as_f32_slice().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get f32 slice from lhs tensor")
    })?;
    let rhs_slice = rhs.data().as_f32_slice().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get f32 slice from rhs tensor")
    })?;
    let output_slice = output_data.as_bool_slice_mut().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get bool slice from output data")
    })?;
    broadcast_compare_op(
        lhs_slice,
        rhs_slice,
        output_slice,
        lhs.shape(),
        rhs.shape(),
        output_shape,
        op,
    )
}

fn cmp_f64(
    lhs: &Tensor,
    rhs: &Tensor,
    output_data: &mut TensorData,
    output_shape: &Shape,
    op: impl Fn(f64, f64) -> bool + Sync + Send,
) -> Result<()> {
    let lhs_slice = lhs.data().as_f64_slice().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get f64 slice from lhs tensor")
    })?;
    let rhs_slice = rhs.data().as_f64_slice().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get f64 slice from rhs tensor")
    })?;
    let output_slice = output_data.as_bool_slice_mut().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get bool slice from output data")
    })?;
    broadcast_compare_op(
        lhs_slice,
        rhs_slice,
        output_slice,
        lhs.shape(),
        rhs.shape(),
        output_shape,
        op,
    )
}

fn cmp_i32(
    lhs: &Tensor,
    rhs: &Tensor,
    output_data: &mut TensorData,
    output_shape: &Shape,
    op: impl Fn(i32, i32) -> bool + Sync + Send,
) -> Result<()> {
    let lhs_slice = lhs.data().as_i32_slice().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get i32 slice from lhs tensor")
    })?;
    let rhs_slice = rhs.data().as_i32_slice().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get i32 slice from rhs tensor")
    })?;
    let output_slice = output_data.as_bool_slice_mut().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get bool slice from output data")
    })?;
    broadcast_compare_op(
        lhs_slice,
        rhs_slice,
        output_slice,
        lhs.shape(),
        rhs.shape(),
        output_shape,
        op,
    )
}

fn cmp_i64(
    lhs: &Tensor,
    rhs: &Tensor,
    output_data: &mut TensorData,
    output_shape: &Shape,
    op: impl Fn(i64, i64) -> bool + Sync + Send,
) -> Result<()> {
    let lhs_slice = lhs.data().as_i64_slice().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get i64 slice from lhs tensor")
    })?;
    let rhs_slice = rhs.data().as_i64_slice().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get i64 slice from rhs tensor")
    })?;
    let output_slice = output_data.as_bool_slice_mut().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get bool slice from output data")
    })?;
    broadcast_compare_op(
        lhs_slice,
        rhs_slice,
        output_slice,
        lhs.shape(),
        rhs.shape(),
        output_shape,
        op,
    )
}

fn cmp_bool(
    lhs: &Tensor,
    rhs: &Tensor,
    output_data: &mut TensorData,
    output_shape: &Shape,
    op: impl Fn(bool, bool) -> bool + Sync + Send,
) -> Result<()> {
    let lhs_slice = lhs.data().as_bool_slice().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get bool slice from lhs tensor")
    })?;
    let rhs_slice = rhs.data().as_bool_slice().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get bool slice from rhs tensor")
    })?;
    let output_slice = output_data.as_bool_slice_mut().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get bool slice from output data")
    })?;
    broadcast_compare_op(
        lhs_slice,
        rhs_slice,
        output_slice,
        lhs.shape(),
        rhs.shape(),
        output_shape,
        op,
    )
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
    use crate::{
        device::Device,
        tensor::{DataType, Shape, TensorData},
    };

    fn tensor_from_vec_f32(data: Vec<f32>) -> Tensor {
        let shape = Shape::new(vec![data.len()]);
        let data = TensorData::from_vec_f32(data, Device::cpu());
        Tensor::new(
            Arc::new(data),
            shape,
            DataType::Float32,
            Device::cpu(),
            false,
        )
    }

    fn tensor_from_vec_f32_shape(data: Vec<f32>, shape: Vec<usize>) -> Tensor {
        let shape = Shape::new(shape);
        let data = TensorData::from_vec_f32(data, Device::cpu());
        Tensor::new(
            Arc::new(data),
            shape,
            DataType::Float32,
            Device::cpu(),
            false,
        )
    }

    fn tensor_from_vec_i32(data: Vec<i32>) -> Tensor {
        let shape = Shape::new(vec![data.len()]);
        let data = TensorData::from_vec_i32(data, Device::cpu());
        Tensor::new(
            Arc::new(data),
            shape,
            DataType::Int32,
            Device::cpu(),
            false,
        )
    }

    fn tensor_from_vec_bool(data: Vec<bool>) -> Tensor {
        let shape = Shape::new(vec![data.len()]);
        let data = TensorData::from_vec_bool(data, Device::cpu());
        Tensor::new(
            Arc::new(data),
            shape,
            DataType::Bool,
            Device::cpu(),
            false,
        )
    }

    #[test]
    fn test_eq_basic() {
        let a = tensor_from_vec_f32(vec![1.0, 2.0, 3.0]);
        let b = tensor_from_vec_f32(vec![1.0, 0.0, 3.0]);
        let result = eq(&a, &b).unwrap();
        let slice = result.data().as_bool_slice().unwrap();
        assert_eq!(slice, &[true, false, true]);
    }

    #[test]
    fn test_eq_broadcasting() {
        let a = tensor_from_vec_f32_shape(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let b = tensor_from_vec_f32_shape(vec![1.0, 4.0], vec![1, 2]);
        let result = eq(&a, &b).unwrap();
        let slice = result.data().as_bool_slice().unwrap();
        assert_eq!(slice, &[true, false, false, true]);
        assert_eq!(result.shape().dims(), &[2, 2]);
    }

    #[test]
    fn test_lt_bool_error() {
        let a = tensor_from_vec_bool(vec![true, false]);
        let b = tensor_from_vec_bool(vec![false, true]);
        assert!(lt(&a, &b).is_err());
    }

    #[test]
    fn test_gt_shape_mismatch_error() {
        let a = tensor_from_vec_f32(vec![1.0, 2.0, 3.0]);
        let b = tensor_from_vec_f32(vec![1.0, 2.0]);
        assert!(gt(&a, &b).is_err());
    }

    #[test]
    fn test_eq_type_mismatch_error() {
        let a = tensor_from_vec_f32(vec![1.0]);
        let b = tensor_from_vec_i32(vec![1]);
        assert!(eq(&a, &b).is_err());
    }
}
