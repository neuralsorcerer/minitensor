// Copyright (c) 2025 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

use crate::{
    error::{MinitensorError, Result},
    tensor::{DataType, Shape, Strides, Tensor, TensorData},
};
use rayon::prelude::*;
use std::sync::Arc;

fn broadcast_compare_op<T: Copy + Send + Sync, F: Fn(T, T) -> bool + Sync + Send>(
    lhs_data: &[T],
    rhs_data: &[T],
    output_data: &mut [bool],
    lhs_shape: &Shape,
    rhs_shape: &Shape,
    output_shape: &Shape,
    op: F,
) -> Result<()> {
    let output_dims = output_shape.dims();
    let lhs_dims = lhs_shape.dims();
    let rhs_dims = rhs_shape.dims();
    let rank = output_dims.len();

    let lhs_contiguous = Strides::from_shape(lhs_shape);
    let rhs_contiguous = Strides::from_shape(rhs_shape);
    let lhs_strides = lhs_contiguous.as_slice();
    let rhs_strides = rhs_contiguous.as_slice();

    let mut lhs_aligned = vec![0usize; rank];
    let mut rhs_aligned = vec![0usize; rank];

    let lhs_offset = rank.saturating_sub(lhs_dims.len());
    for (i, &dim) in lhs_dims.iter().enumerate() {
        lhs_aligned[lhs_offset + i] = if dim == 1 { 0 } else { lhs_strides[i] };
    }

    let rhs_offset = rank.saturating_sub(rhs_dims.len());
    for (i, &dim) in rhs_dims.iter().enumerate() {
        rhs_aligned[rhs_offset + i] = if dim == 1 { 0 } else { rhs_strides[i] };
    }

    const CHUNK: usize = 1024;
    let process_chunk = |chunk_idx: usize, out_chunk: &mut [bool]| {
        let start = chunk_idx * CHUNK;
        let mut coord = vec![0usize; rank];
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
    };

    if output_data.len() < CHUNK {
        process_chunk(0, output_data);
    } else if output_data.len() < CHUNK * 4 {
        output_data
            .chunks_mut(CHUNK)
            .enumerate()
            .for_each(|(chunk_idx, out_chunk)| process_chunk(chunk_idx, out_chunk));
    } else {
        output_data
            .par_chunks_mut(CHUNK)
            .enumerate()
            .for_each(|(chunk_idx, out_chunk)| process_chunk(chunk_idx, out_chunk));
    }

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
        Tensor::new(Arc::new(data), shape, DataType::Int32, Device::cpu(), false)
    }

    fn tensor_from_vec_bool(data: Vec<bool>) -> Tensor {
        let shape = Shape::new(vec![data.len()]);
        let data = TensorData::from_vec_bool(data, Device::cpu());
        Tensor::new(Arc::new(data), shape, DataType::Bool, Device::cpu(), false)
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
