// Copyright (c) 2025 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

use crate::{
    error::{MinitensorError, Result},
    operations::binary::{BinaryOpKind, coerce_binary_operands},
    tensor::{DataType, Shape, Tensor, TensorData},
};
use std::sync::Arc;

/// Generates a dtype-specialized comparison kernel: fetch both input slices
/// for the dtype and apply `op` element-wise with broadcasting into a fresh
/// bool buffer. The generic-output `broadcast_binary_map` replaces the
/// bool-specialized broadcast walker this file used to duplicate.
macro_rules! cmp_kernel {
    ($name:ident, $ty:ty, $accessor:ident, $tyname:literal) => {
        fn $name(
            lhs: &Tensor,
            rhs: &Tensor,
            output_shape: &Shape,
            op: impl Fn($ty, $ty) -> bool + Sync + Send,
        ) -> Result<TensorData> {
            let lhs_slice = lhs.data().$accessor().ok_or_else(|| {
                MinitensorError::internal_error(concat!(
                    "Failed to get ",
                    $tyname,
                    " slice from lhs tensor"
                ))
            })?;
            let rhs_slice = rhs.data().$accessor().ok_or_else(|| {
                MinitensorError::internal_error(concat!(
                    "Failed to get ",
                    $tyname,
                    " slice from rhs tensor"
                ))
            })?;
            let out = crate::operations::arithmetic::broadcast_binary_map(
                lhs_slice,
                rhs_slice,
                lhs.shape(),
                rhs.shape(),
                output_shape,
                op,
            )?;
            Ok(TensorData::from_vec(out, DataType::Bool, lhs.device()))
        }
    };
}

cmp_kernel!(cmp_f32, f32, as_f32_slice, "f32");
cmp_kernel!(cmp_f64, f64, as_f64_slice, "f64");
cmp_kernel!(cmp_i32, i32, as_i32_slice, "i32");
cmp_kernel!(cmp_i64, i64, as_i64_slice, "i64");
cmp_kernel!(cmp_bool, bool, as_bool_slice, "bool");

macro_rules! cmp_op {
    ($fn_name:ident, $op:tt, $bool_ok:expr) => {
        pub fn $fn_name(lhs: &Tensor, rhs: &Tensor) -> Result<Tensor> {
            if lhs.device() != rhs.device() {
                return Err(MinitensorError::device_mismatch(
                    format!("{:?}", lhs.device()),
                    format!("{:?}", rhs.device()),
                ));
            }
            let (lhs_cast, rhs_cast, common_dtype) =
                coerce_binary_operands(lhs, rhs, BinaryOpKind::Add)?;

            if matches!(common_dtype, DataType::Bool) && !$bool_ok {
                return Err(MinitensorError::invalid_operation(
                    "Comparison not supported for boolean tensors",
                ));
            }

            let lhs_ref = lhs_cast.as_ref();
            let rhs_ref = rhs_cast.as_ref();

            let output_shape = lhs_ref.shape().broadcast_with(rhs_ref.shape())?;
            let output_data = match common_dtype {
                DataType::Float32 => cmp_f32(lhs_ref, rhs_ref, &output_shape, |a, b| a $op b)?,
                DataType::Float64 => cmp_f64(lhs_ref, rhs_ref, &output_shape, |a, b| a $op b)?,
                DataType::Int32 => cmp_i32(lhs_ref, rhs_ref, &output_shape, |a, b| a $op b)?,
                DataType::Int64 => cmp_i64(lhs_ref, rhs_ref, &output_shape, |a, b| a $op b)?,
                DataType::Bool => {
                    debug_assert!($bool_ok);
                    cmp_bool(lhs_ref, rhs_ref, &output_shape, |a, b| a $op b)?
                }
            };

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

#[inline(always)]
fn isclose_f32(a: f32, b: f32, rtol: f32, atol: f32, equal_nan: bool) -> bool {
    if a == b {
        return true;
    }
    if equal_nan && a.is_nan() && b.is_nan() {
        return true;
    }
    if !a.is_finite() || !b.is_finite() {
        return false;
    }
    (a - b).abs() <= atol + rtol * b.abs()
}

#[inline(always)]
fn isclose_f64(a: f64, b: f64, rtol: f64, atol: f64, equal_nan: bool) -> bool {
    if a == b {
        return true;
    }
    if equal_nan && a.is_nan() && b.is_nan() {
        return true;
    }
    if !a.is_finite() || !b.is_finite() {
        return false;
    }
    (a - b).abs() <= atol + rtol * b.abs()
}

pub fn isclose(
    lhs: &Tensor,
    rhs: &Tensor,
    rtol: f64,
    atol: f64,
    equal_nan: bool,
) -> Result<Tensor> {
    if !rtol.is_finite() || !atol.is_finite() || rtol < 0.0 || atol < 0.0 {
        return Err(MinitensorError::invalid_operation(
            "rtol and atol must be non-negative, finite values",
        ));
    }
    if lhs.device() != rhs.device() {
        return Err(MinitensorError::device_mismatch(
            format!("{:?}", lhs.device()),
            format!("{:?}", rhs.device()),
        ));
    }

    let (lhs_cast, rhs_cast, common_dtype) = coerce_binary_operands(lhs, rhs, BinaryOpKind::Add)?;
    let lhs_ref = lhs_cast.as_ref();
    let rhs_ref = rhs_cast.as_ref();
    let output_shape = lhs_ref.shape().broadcast_with(rhs_ref.shape())?;
    let output_data = match common_dtype {
        DataType::Float32 => cmp_f32(lhs_ref, rhs_ref, &output_shape, |a, b| {
            isclose_f32(a, b, rtol as f32, atol as f32, equal_nan)
        })?,
        DataType::Float64 => cmp_f64(lhs_ref, rhs_ref, &output_shape, |a, b| {
            isclose_f64(a, b, rtol, atol, equal_nan)
        })?,
        DataType::Int32 => cmp_i32(lhs_ref, rhs_ref, &output_shape, |a, b| a == b)?,
        DataType::Int64 => cmp_i64(lhs_ref, rhs_ref, &output_shape, |a, b| a == b)?,
        DataType::Bool => cmp_bool(lhs_ref, rhs_ref, &output_shape, |a, b| a == b)?,
    };

    Ok(Tensor::new(
        Arc::new(output_data),
        output_shape,
        DataType::Bool,
        lhs.device(),
        false,
    ))
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
    fn test_eq_promotes_mixed_dtypes() {
        let a = tensor_from_vec_f32(vec![1.0, 2.0]);
        let b = tensor_from_vec_i32(vec![1, 3]);
        let result = eq(&a, &b).unwrap();
        let slice = result.data().as_bool_slice().unwrap();
        assert_eq!(slice, &[true, false]);
    }

    #[test]
    fn test_lt_promotes_bool_with_integers() {
        let a = tensor_from_vec_bool(vec![true, false]);
        let b = tensor_from_vec_i32(vec![2, -1]);
        let result = lt(&a, &b).unwrap();
        let slice = result.data().as_bool_slice().unwrap();
        assert_eq!(slice, &[true, false]);
    }
}
