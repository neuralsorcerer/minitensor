// Copyright (c) 2025 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

use crate::{
    autograd::{DotBackward, MatMulBackward, TransposeBackward, add_to_graph},
    error::{MinitensorError, Result},
    operations::{
        binary::{BinaryOpKind, coerce_binary_operands},
        reduction,
    },
    tensor::{DataType, Shape, Strides, Tensor, TensorData},
};
use rayon::prelude::*;
use std::sync::Arc;

#[cfg(feature = "blas")]
use cblas::{Layout, Transpose};

const PAR_THRESHOLD: usize = 1 << 12;

#[derive(Debug, Clone)]
pub(crate) struct DiagonalSpec {
    pub diag_len: usize,
    pub base_offset: usize,
    pub diag_stride: usize,
    pub kept_dims: Vec<usize>,
    pub output_dims: Vec<usize>,
}

fn normalize_dim(dim: isize, ndim: usize) -> Result<usize> {
    let dim = if dim < 0 { dim + ndim as isize } else { dim };
    if dim < 0 || dim >= ndim as isize {
        Err(MinitensorError::index_error(dim, 0, ndim))
    } else {
        Ok(dim as usize)
    }
}

pub(crate) fn compute_diagonal_spec(
    dims: &[usize],
    strides: &[usize],
    dim1: usize,
    dim2: usize,
    offset: isize,
) -> Result<DiagonalSpec> {
    debug_assert!(dim1 != dim2);

    let dim1_size = dims
        .get(dim1)
        .ok_or_else(|| MinitensorError::index_error(dim1 as isize, 0, dims.len()))?;
    let dim2_size = dims
        .get(dim2)
        .ok_or_else(|| MinitensorError::index_error(dim2 as isize, 0, dims.len()))?;
    let stride1 = strides
        .get(dim1)
        .ok_or_else(|| MinitensorError::index_error(dim1 as isize, 0, strides.len()))?;
    let stride2 = strides
        .get(dim2)
        .ok_or_else(|| MinitensorError::index_error(dim2 as isize, 0, strides.len()))?;

    let diag_stride = stride1.saturating_add(*stride2);

    let (diag_len, base_offset) = if offset >= 0 {
        let offset = offset as usize;
        if offset >= *dim2_size {
            (0, 0)
        } else {
            (
                (*dim1_size).min(dim2_size - offset),
                offset.saturating_mul(*stride2),
            )
        }
    } else {
        let neg = (-offset) as usize;
        if neg >= *dim1_size {
            (0, 0)
        } else {
            (
                (dim1_size - neg).min(*dim2_size),
                neg.saturating_mul(*stride1),
            )
        }
    };

    let mut kept_dims = Vec::with_capacity(dims.len().saturating_sub(2));
    let mut output_dims = Vec::with_capacity(kept_dims.capacity() + 1);
    for (idx, &size) in dims.iter().enumerate() {
        if idx == dim1 || idx == dim2 {
            continue;
        }
        kept_dims.push(idx);
        output_dims.push(size);
    }
    output_dims.push(diag_len);

    Ok(DiagonalSpec {
        diag_len,
        base_offset,
        diag_stride,
        kept_dims,
        output_dims,
    })
}

fn diagonal_copy<T: Copy + Send + Sync>(
    input: &[T],
    output: &mut [T],
    dims: &[usize],
    strides: &[usize],
    spec: &DiagonalSpec,
) {
    if output.is_empty() {
        return;
    }

    let mut axis_sizes: Vec<usize> = spec.kept_dims.iter().map(|&dim| dims[dim]).collect();
    axis_sizes.push(spec.diag_len);

    let mut axis_strides: Vec<usize> = spec.kept_dims.iter().map(|&dim| strides[dim]).collect();
    axis_strides.push(spec.diag_stride);

    let axes = axis_sizes.len();
    let mut indices = vec![0usize; axes];
    let mut out_idx = 0usize;

    loop {
        let mut input_offset = spec.base_offset;
        for axis in 0..axes {
            input_offset += indices[axis] * axis_strides[axis];
        }
        output[out_idx] = input[input_offset];
        out_idx += 1;

        let mut done = true;
        for axis in (0..axes).rev() {
            indices[axis] += 1;
            if indices[axis] < axis_sizes[axis] {
                done = false;
                break;
            }
            indices[axis] = 0;
        }
        if done {
            break;
        }
    }
}

pub(crate) fn diagonal_scatter<T>(
    grad_output: &[T],
    grad_input: &mut [T],
    dims: &[usize],
    strides: &[usize],
    spec: &DiagonalSpec,
) where
    T: Copy + Send + Sync + std::ops::AddAssign,
{
    if grad_output.is_empty() {
        return;
    }

    let mut axis_sizes: Vec<usize> = spec.kept_dims.iter().map(|&dim| dims[dim]).collect();
    axis_sizes.push(spec.diag_len);

    let mut axis_strides: Vec<usize> = spec.kept_dims.iter().map(|&dim| strides[dim]).collect();
    axis_strides.push(spec.diag_stride);

    let axes = axis_sizes.len();
    let mut indices = vec![0usize; axes];
    let mut out_idx = 0usize;

    loop {
        let mut input_offset = spec.base_offset;
        for axis in 0..axes {
            input_offset += indices[axis] * axis_strides[axis];
        }
        grad_input[input_offset] += grad_output[out_idx];
        out_idx += 1;

        let mut done = true;
        for axis in (0..axes).rev() {
            indices[axis] += 1;
            if indices[axis] < axis_sizes[axis] {
                done = false;
                break;
            }
            indices[axis] = 0;
        }
        if done {
            break;
        }
    }
}

#[cfg(feature = "blas")]
#[inline]
unsafe fn gemm_f32(m: usize, k: usize, n: usize, a: *const f32, b: *const f32, c: *mut f32) {
    cblas::sgemm(
        Layout::RowMajor,
        Transpose::None,
        Transpose::None,
        m as i32,
        n as i32,
        k as i32,
        1.0,
        a,
        k as i32,
        b,
        n as i32,
        0.0,
        c,
        n as i32,
    );
}

#[cfg(feature = "blas")]
#[inline]
unsafe fn gemm_f64(m: usize, k: usize, n: usize, a: *const f64, b: *const f64, c: *mut f64) {
    cblas::dgemm(
        Layout::RowMajor,
        Transpose::None,
        Transpose::None,
        m as i32,
        n as i32,
        k as i32,
        1.0,
        a,
        k as i32,
        b,
        n as i32,
        0.0,
        c,
        n as i32,
    );
}

#[cfg(not(feature = "blas"))]
#[inline]
unsafe fn gemm_f32(m: usize, k: usize, n: usize, a: *const f32, b: *const f32, c: *mut f32) {
    unsafe {
        matrixmultiply::sgemm(
            m, k, n, 1.0, a, k as isize, 1, b, n as isize, 1, 0.0, c, n as isize, 1,
        )
    };
}

#[cfg(not(feature = "blas"))]
#[inline]
unsafe fn gemm_f64(m: usize, k: usize, n: usize, a: *const f64, b: *const f64, c: *mut f64) {
    unsafe {
        matrixmultiply::dgemm(
            m, k, n, 1.0, a, k as isize, 1, b, n as isize, 1, 0.0, c, n as isize, 1,
        )
    };
}

/// Matrix multiplication with gradient support
pub fn matmul(lhs: &Tensor, rhs: &Tensor) -> Result<Tensor> {
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

    // Validate matrix multiplication dimensions
    if lhs.ndim() < 2 || rhs.ndim() < 2 {
        return Err(MinitensorError::invalid_operation(
            "Matrix multiplication requires tensors with at least 2 dimensions",
        ));
    }

    let lhs_shape = lhs.shape().dims();
    let rhs_shape = rhs.shape().dims();

    // Ensure batch dimensions match
    if lhs_shape[..lhs_shape.len() - 2] != rhs_shape[..rhs_shape.len() - 2] {
        return Err(MinitensorError::shape_mismatch(
            lhs_shape.to_vec(),
            rhs_shape.to_vec(),
        ));
    }

    // Get the last two dimensions for matrix multiplication
    let lhs_rows = lhs_shape[lhs_shape.len() - 2];
    let lhs_cols = lhs_shape[lhs_shape.len() - 1];
    let rhs_rows = rhs_shape[rhs_shape.len() - 2];
    let rhs_cols = rhs_shape[rhs_shape.len() - 1];

    if lhs_cols != rhs_rows {
        return Err(MinitensorError::shape_mismatch(
            vec![lhs_rows, lhs_cols],
            vec![rhs_rows, rhs_cols],
        ));
    }

    // Compute output shape
    let mut output_shape = lhs_shape[..lhs_shape.len() - 2].to_vec();
    output_shape.push(lhs_rows);
    output_shape.push(rhs_cols);
    let output_shape_obj = Shape::new(output_shape);

    // Create output tensor data
    let mut output_data =
        TensorData::zeros_on_device(output_shape_obj.numel(), lhs.dtype(), lhs.device());

    // Perform matrix multiplication based on data type
    match lhs.dtype() {
        DataType::Float32 => matmul_f32(lhs, rhs, &mut output_data, &output_shape_obj)?,
        DataType::Float64 => matmul_f64(lhs, rhs, &mut output_data, &output_shape_obj)?,
        DataType::Int32 => matmul_i32(lhs, rhs, &mut output_data, &output_shape_obj)?,
        DataType::Int64 => matmul_i64(lhs, rhs, &mut output_data, &output_shape_obj)?,
        DataType::Bool => {
            return Err(MinitensorError::invalid_operation(
                "Matrix multiplication not supported for boolean tensors",
            ));
        }
    }

    // Create output tensor
    let output = Tensor::new(
        Arc::new(output_data),
        output_shape_obj,
        lhs.dtype(),
        lhs.device(),
        lhs.requires_grad() || rhs.requires_grad(),
    );

    // Set up gradient function if needed
    if output.requires_grad() {
        let grad_fn = Arc::new(MatMulBackward {
            lhs: lhs.detach(),
            rhs: rhs.detach(),
            input_ids: [lhs.id(), rhs.id()],
            lhs_requires_grad: lhs.requires_grad(),
            rhs_requires_grad: rhs.requires_grad(),
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

/// Dot product of two 1D tensors with gradient support
pub fn dot(lhs: &Tensor, rhs: &Tensor) -> Result<Tensor> {
    if lhs.device() != rhs.device() {
        return Err(MinitensorError::device_mismatch(
            format!("{:?}", lhs.device()),
            format!("{:?}", rhs.device()),
        ));
    }

    let lhs_dims = lhs.ndim();
    let rhs_dims = rhs.ndim();
    if lhs_dims != 1 || rhs_dims != 1 {
        return Err(MinitensorError::invalid_operation(format!(
            "dot: expected 1D tensors but got {}D and {}D tensors",
            lhs_dims, rhs_dims
        )));
    }

    if lhs.numel() != rhs.numel() {
        return Err(MinitensorError::shape_mismatch(
            lhs.shape().dims().to_vec(),
            rhs.shape().dims().to_vec(),
        ));
    }

    let (lhs_cast, rhs_cast, result_dtype) = coerce_binary_operands(lhs, rhs, BinaryOpKind::Mul)?;

    if result_dtype == DataType::Bool {
        return Err(MinitensorError::invalid_operation(
            "dot does not support bool tensors",
        ));
    }

    let lhs_view = lhs_cast.as_ref();
    let rhs_view = rhs_cast.as_ref();

    let numel = lhs_view.numel();
    let device = lhs.device();
    let requires_grad = lhs.requires_grad() || rhs.requires_grad();

    let output_data = match result_dtype {
        DataType::Float32 => {
            let lhs_slice = lhs_view.data().as_f32_slice().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get f32 slice for dot input")
            })?;
            let rhs_slice = rhs_view.data().as_f32_slice().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get f32 slice for dot input")
            })?;

            let dot = if numel >= PAR_THRESHOLD {
                lhs_slice
                    .par_iter()
                    .zip(rhs_slice.par_iter())
                    .map(|(&a, &b)| a * b)
                    .sum::<f32>()
            } else {
                lhs_slice
                    .iter()
                    .zip(rhs_slice.iter())
                    .map(|(&a, &b)| a * b)
                    .sum::<f32>()
            };

            TensorData::from_vec_f32(vec![dot], device)
        }
        DataType::Float64 => {
            let lhs_slice = lhs_view.data().as_f64_slice().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get f64 slice for dot input")
            })?;
            let rhs_slice = rhs_view.data().as_f64_slice().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get f64 slice for dot input")
            })?;

            let dot = if numel >= PAR_THRESHOLD {
                lhs_slice
                    .par_iter()
                    .zip(rhs_slice.par_iter())
                    .map(|(&a, &b)| a * b)
                    .sum::<f64>()
            } else {
                lhs_slice
                    .iter()
                    .zip(rhs_slice.iter())
                    .map(|(&a, &b)| a * b)
                    .sum::<f64>()
            };

            TensorData::from_vec_f64(vec![dot], device)
        }
        DataType::Int32 => {
            let lhs_slice = lhs_view.data().as_i32_slice().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get i32 slice for dot input")
            })?;
            let rhs_slice = rhs_view.data().as_i32_slice().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get i32 slice for dot input")
            })?;

            let mut dot: i32 = 0;
            for (&a, &b) in lhs_slice.iter().zip(rhs_slice.iter()) {
                dot = dot.wrapping_add(a.wrapping_mul(b));
            }

            TensorData::from_vec_i32(vec![dot], device)
        }
        DataType::Int64 => {
            let lhs_slice = lhs_view.data().as_i64_slice().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get i64 slice for dot input")
            })?;
            let rhs_slice = rhs_view.data().as_i64_slice().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get i64 slice for dot input")
            })?;

            let mut dot: i64 = 0;
            for (&a, &b) in lhs_slice.iter().zip(rhs_slice.iter()) {
                dot = dot.wrapping_add(a.wrapping_mul(b));
            }

            TensorData::from_vec_i64(vec![dot], device)
        }
        DataType::Bool => unreachable!("Bool dtype handled earlier"),
    };

    let output_shape = Shape::new(Vec::new());
    let output = Tensor::new(
        Arc::new(output_data),
        output_shape,
        result_dtype,
        device,
        requires_grad,
    );

    if output.requires_grad() {
        let lhs_requires_grad = lhs.requires_grad();
        let rhs_requires_grad = rhs.requires_grad();
        let grad_fn = Arc::new(DotBackward {
            lhs: lhs_cast.into_owned().detach(),
            rhs: rhs_cast.into_owned().detach(),
            input_ids: [lhs.id(), rhs.id()],
            lhs_requires_grad,
            rhs_requires_grad,
        });

        let mut output_with_grad = output;
        output_with_grad.set_grad_fn(Some(grad_fn.clone()));
        add_to_graph(&output_with_grad, Some(grad_fn))?;
        Ok(output_with_grad)
    } else {
        Ok(output)
    }
}

/// Transpose operation with gradient support
pub fn transpose(tensor: &Tensor, dim0: isize, dim1: isize) -> Result<Tensor> {
    let ndim = tensor.ndim() as isize;
    let dim0 = if dim0 < 0 { dim0 + ndim } else { dim0 };
    let dim1 = if dim1 < 0 { dim1 + ndim } else { dim1 };

    if dim0 < 0 || dim0 >= ndim || dim1 < 0 || dim1 >= ndim {
        return Err(MinitensorError::index_error(
            dim0.max(dim1),
            0,
            ndim as usize,
        ));
    }

    if dim0 == dim1 {
        // No-op transpose
        return Ok(tensor.clone());
    }

    let dim0_usize = dim0 as usize;
    let dim1_usize = dim1 as usize;

    // Create new shape with swapped dimensions
    let mut new_shape = tensor.shape().dims().to_vec();
    new_shape.swap(dim0_usize, dim1_usize);
    let new_shape_obj = Shape::new(new_shape);

    // Create new strides with swapped dimensions
    let old_strides = tensor.strides().as_slice();
    let mut new_strides = old_strides.to_vec();
    new_strides.swap(dim0_usize, dim1_usize);

    // Create output tensor data by copying and rearranging
    let mut output_data =
        TensorData::zeros_on_device(tensor.numel(), tensor.dtype(), tensor.device());

    // Perform transpose based on data type
    match tensor.dtype() {
        DataType::Float32 => transpose_f32(
            tensor,
            &mut output_data,
            &new_shape_obj,
            dim0_usize,
            dim1_usize,
        )?,
        DataType::Float64 => transpose_f64(
            tensor,
            &mut output_data,
            &new_shape_obj,
            dim0_usize,
            dim1_usize,
        )?,
        DataType::Int32 => transpose_i32(
            tensor,
            &mut output_data,
            &new_shape_obj,
            dim0_usize,
            dim1_usize,
        )?,
        DataType::Int64 => transpose_i64(
            tensor,
            &mut output_data,
            &new_shape_obj,
            dim0_usize,
            dim1_usize,
        )?,
        DataType::Bool => transpose_bool(
            tensor,
            &mut output_data,
            &new_shape_obj,
            dim0_usize,
            dim1_usize,
        )?,
    }

    // Create output tensor
    let output = Tensor::new(
        Arc::new(output_data),
        new_shape_obj,
        tensor.dtype(),
        tensor.device(),
        tensor.requires_grad(),
    );

    // Set up gradient function if needed
    if output.requires_grad() {
        let grad_fn = Arc::new(TransposeBackward {
            dims: vec![dim0_usize, dim1_usize],
            input_id: tensor.id(),
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

/// Extract a diagonal from the tensor, reducing two dimensions into one.
pub fn diagonal(tensor: &Tensor, offset: isize, dim1: isize, dim2: isize) -> Result<Tensor> {
    if tensor.ndim() < 2 {
        return Err(MinitensorError::invalid_operation(
            "diagonal requires tensors with at least 2 dimensions",
        ));
    }

    if !tensor.device().is_cpu() {
        return Err(MinitensorError::invalid_operation(
            "diagonal currently supports only CPU tensors",
        ));
    }

    let ndim = tensor.ndim();
    let dim1 = normalize_dim(dim1, ndim)?;
    let dim2 = normalize_dim(dim2, ndim)?;
    if dim1 == dim2 {
        return Err(MinitensorError::invalid_operation(
            "diagonal dimensions must be distinct",
        ));
    }

    let dims = tensor.shape().dims();
    let strides = tensor.strides().as_slice();
    let spec = compute_diagonal_spec(dims, strides, dim1, dim2, offset)?;
    let out_shape = Shape::new(spec.output_dims.clone());
    let dtype = tensor.dtype();
    let device = tensor.device();
    let mut output_data = TensorData::zeros_on_device(out_shape.numel(), dtype, device);

    if out_shape.numel() > 0 {
        match dtype {
            DataType::Float32 => {
                let input = tensor.data().as_f32_slice().ok_or_else(|| {
                    MinitensorError::internal_error("Failed to get f32 slice from tensor")
                })?;
                let output = output_data.as_f32_slice_mut().ok_or_else(|| {
                    MinitensorError::internal_error(
                        "Failed to get mutable f32 slice for diagonal output",
                    )
                })?;
                diagonal_copy(input, output, dims, strides, &spec);
            }
            DataType::Float64 => {
                let input = tensor.data().as_f64_slice().ok_or_else(|| {
                    MinitensorError::internal_error("Failed to get f64 slice from tensor")
                })?;
                let output = output_data.as_f64_slice_mut().ok_or_else(|| {
                    MinitensorError::internal_error(
                        "Failed to get mutable f64 slice for diagonal output",
                    )
                })?;
                diagonal_copy(input, output, dims, strides, &spec);
            }
            DataType::Int32 => {
                let input = tensor.data().as_i32_slice().ok_or_else(|| {
                    MinitensorError::internal_error("Failed to get i32 slice from tensor")
                })?;
                let output = output_data.as_i32_slice_mut().ok_or_else(|| {
                    MinitensorError::internal_error(
                        "Failed to get mutable i32 slice for diagonal output",
                    )
                })?;
                diagonal_copy(input, output, dims, strides, &spec);
            }
            DataType::Int64 => {
                let input = tensor.data().as_i64_slice().ok_or_else(|| {
                    MinitensorError::internal_error("Failed to get i64 slice from tensor")
                })?;
                let output = output_data.as_i64_slice_mut().ok_or_else(|| {
                    MinitensorError::internal_error(
                        "Failed to get mutable i64 slice for diagonal output",
                    )
                })?;
                diagonal_copy(input, output, dims, strides, &spec);
            }
            DataType::Bool => {
                let input = tensor.data().as_bool_slice().ok_or_else(|| {
                    MinitensorError::internal_error("Failed to get bool slice from tensor")
                })?;
                let output = output_data.as_bool_slice_mut().ok_or_else(|| {
                    MinitensorError::internal_error(
                        "Failed to get mutable bool slice for diagonal output",
                    )
                })?;
                diagonal_copy(input, output, dims, strides, &spec);
            }
        }
    }

    let mut output = Tensor::new(
        Arc::new(output_data),
        out_shape,
        dtype,
        device,
        tensor.requires_grad(),
    );

    if tensor.requires_grad() {
        let grad_fn = Arc::new(crate::autograd::DiagonalBackward {
            input_shape: dims.to_vec(),
            input_strides: strides.to_vec(),
            input_dtype: dtype,
            dim1,
            dim2,
            offset,
            input_requires_grad: tensor.requires_grad(),
            input_id: tensor.id(),
        });

        output.set_grad_fn(Some(grad_fn.clone()));
        add_to_graph(&output, Some(grad_fn))?;
    }

    Ok(output)
}

/// Sum of the diagonal elements along two dimensions.
pub fn trace(tensor: &Tensor, offset: isize, dim1: isize, dim2: isize) -> Result<Tensor> {
    let diag = diagonal(tensor, offset, dim1, dim2)?;
    if diag.ndim() == 0 {
        return Ok(diag);
    }

    reduction::sum(&diag, Some(vec![-1]), false)
}

/// Return the upper triangular part of a matrix (or batch of matrices).
pub fn triu(tensor: &Tensor, diagonal: i64) -> Result<Tensor> {
    triangular_op(tensor, diagonal, true)
}

/// Return the lower triangular part of a matrix (or batch of matrices).
pub fn tril(tensor: &Tensor, diagonal: i64) -> Result<Tensor> {
    triangular_op(tensor, diagonal, false)
}

fn triangular_op(tensor: &Tensor, diagonal: i64, upper: bool) -> Result<Tensor> {
    if tensor.ndim() < 2 {
        return Err(MinitensorError::invalid_operation(
            "triangular operations require tensors with at least 2 dimensions",
        ));
    }

    let clamped_diagonal = diagonal.clamp(isize::MIN as i64, isize::MAX as i64) as isize;

    let mut output_data =
        TensorData::uninitialized_on_device(tensor.numel(), tensor.dtype(), tensor.device());

    apply_triangular_mask(tensor, &mut output_data, clamped_diagonal, upper)?;

    let mut output = Tensor::new(
        Arc::new(output_data),
        tensor.shape().clone(),
        tensor.dtype(),
        tensor.device(),
        tensor.requires_grad(),
    );

    if tensor.requires_grad() {
        let grad_fn = Arc::new(crate::autograd::TriangularBackward {
            input_shape: tensor.shape().dims().to_vec(),
            diagonal: clamped_diagonal,
            upper,
            input_requires_grad: tensor.requires_grad(),
            input_id: tensor.id(),
        });

        output.set_grad_fn(Some(grad_fn.clone()));
        add_to_graph(&output, Some(grad_fn))?;
    }

    Ok(output)
}

pub(crate) fn apply_triangular_mask(
    tensor: &Tensor,
    output_data: &mut TensorData,
    diagonal: isize,
    upper: bool,
) -> Result<()> {
    match tensor.dtype() {
        DataType::Float32 => {
            let input = tensor.data().as_f32_slice().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get f32 slice from tensor")
            })?;
            let output = output_data.as_f32_slice_mut().ok_or_else(|| {
                MinitensorError::internal_error(
                    "Failed to get mutable f32 slice for triangular output",
                )
            })?;
            copy_and_mask(input, output, tensor.shape(), diagonal, upper);
        }
        DataType::Float64 => {
            let input = tensor.data().as_f64_slice().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get f64 slice from tensor")
            })?;
            let output = output_data.as_f64_slice_mut().ok_or_else(|| {
                MinitensorError::internal_error(
                    "Failed to get mutable f64 slice for triangular output",
                )
            })?;
            copy_and_mask(input, output, tensor.shape(), diagonal, upper);
        }
        DataType::Int32 => {
            let input = tensor.data().as_i32_slice().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get i32 slice from tensor")
            })?;
            let output = output_data.as_i32_slice_mut().ok_or_else(|| {
                MinitensorError::internal_error(
                    "Failed to get mutable i32 slice for triangular output",
                )
            })?;
            copy_and_mask(input, output, tensor.shape(), diagonal, upper);
        }
        DataType::Int64 => {
            let input = tensor.data().as_i64_slice().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get i64 slice from tensor")
            })?;
            let output = output_data.as_i64_slice_mut().ok_or_else(|| {
                MinitensorError::internal_error(
                    "Failed to get mutable i64 slice for triangular output",
                )
            })?;
            copy_and_mask(input, output, tensor.shape(), diagonal, upper);
        }
        DataType::Bool => {
            let input = tensor.data().as_bool_slice().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get bool slice from tensor")
            })?;
            let output = output_data.as_bool_slice_mut().ok_or_else(|| {
                MinitensorError::internal_error(
                    "Failed to get mutable bool slice for triangular output",
                )
            })?;
            copy_and_mask(input, output, tensor.shape(), diagonal, upper);
        }
    }

    Ok(())
}

fn copy_and_mask<T: Copy + Default>(
    input: &[T],
    output: &mut [T],
    shape: &Shape,
    diagonal: isize,
    upper: bool,
) {
    if input.is_empty() {
        return;
    }

    output.copy_from_slice(input);

    let dims = shape.dims();
    debug_assert!(dims.len() >= 2);
    let rows = dims[dims.len() - 2];
    let cols = dims[dims.len() - 1];

    if rows == 0 || cols == 0 {
        return;
    }

    let batch = shape.numel() / (rows * cols);
    let zero = T::default();

    for b in 0..batch {
        let base = b * rows * cols;
        for r in 0..rows {
            let row_offset = base + r * cols;
            let row_idx = r as isize;
            for c in 0..cols {
                let col_idx = c as isize;
                let keep = if upper {
                    col_idx - row_idx >= diagonal
                } else {
                    col_idx - row_idx <= diagonal
                };
                if !keep {
                    output[row_offset + c] = zero;
                }
            }
        }
    }
}

// Helper functions for matrix multiplication

fn matmul_f32(
    lhs: &Tensor,
    rhs: &Tensor,
    output_data: &mut TensorData,
    _output_shape: &Shape,
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

    optimized_matmul_f32(lhs_data, rhs_data, output_slice, lhs.shape(), rhs.shape())
}

fn matmul_f64(
    lhs: &Tensor,
    rhs: &Tensor,
    output_data: &mut TensorData,
    _output_shape: &Shape,
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

    optimized_matmul_f64(lhs_data, rhs_data, output_slice, lhs.shape(), rhs.shape())
}

fn matmul_i32(
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

    naive_matmul(
        lhs_data,
        rhs_data,
        output_slice,
        lhs.shape(),
        rhs.shape(),
        output_shape,
    )
}

fn matmul_i64(
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

    naive_matmul(
        lhs_data,
        rhs_data,
        output_slice,
        lhs.shape(),
        rhs.shape(),
        output_shape,
    )
}

/// Naive matrix multiplication implementation (O(n^3)) with batch support
fn naive_matmul<T>(
    lhs_data: &[T],
    rhs_data: &[T],
    output_data: &mut [T],
    lhs_shape: &Shape,
    rhs_shape: &Shape,
    _output_shape: &Shape,
) -> Result<()>
where
    T: Copy + std::ops::Add<Output = T> + std::ops::Mul<Output = T> + Default + Send + Sync,
{
    let lhs_dims = lhs_shape.dims();
    let rhs_dims = rhs_shape.dims();

    let m = lhs_dims[lhs_dims.len() - 2];
    let k = lhs_dims[lhs_dims.len() - 1];
    let n = rhs_dims[rhs_dims.len() - 1];
    let batch = lhs_data.len() / (m * k);
    if batch == 1 && m * n * k < PAR_THRESHOLD {
        // For small single-batch matrices, avoid parallel overhead
        for i in 0..m {
            for j in 0..n {
                let mut sum = T::default();
                for l in 0..k {
                    let lhs_idx = i * k + l;
                    let rhs_idx = l * n + j;
                    sum = sum + lhs_data[lhs_idx] * rhs_data[rhs_idx];
                }
                output_data[i * n + j] = sum;
            }
        }
    } else {
        output_data
            .par_chunks_mut(m * n)
            .enumerate()
            .for_each(|(b, chunk)| {
                let lhs_batch = &lhs_data[b * m * k..(b + 1) * m * k];
                let rhs_batch = &rhs_data[b * k * n..(b + 1) * k * n];
                chunk.par_chunks_mut(n).enumerate().for_each(|(i, row)| {
                    for j in 0..n {
                        let mut sum = T::default();
                        for l in 0..k {
                            let lhs_idx = i * k + l;
                            let rhs_idx = l * n + j;
                            sum = sum + lhs_batch[lhs_idx] * rhs_batch[rhs_idx];
                        }
                        row[j] = sum;
                    }
                });
            });
    }

    Ok(())
}

fn optimized_matmul_f32(
    lhs_data: &[f32],
    rhs_data: &[f32],
    output_data: &mut [f32],
    lhs_shape: &Shape,
    rhs_shape: &Shape,
) -> Result<()> {
    let lhs_dims = lhs_shape.dims();
    let rhs_dims = rhs_shape.dims();
    let m = lhs_dims[lhs_dims.len() - 2];
    let k = lhs_dims[lhs_dims.len() - 1];
    let n = rhs_dims[rhs_dims.len() - 1];

    if m == 0 || k == 0 || n == 0 {
        // Nothing to compute for zero-sized dimensions
        return Ok(());
    }

    let batch = lhs_data.len() / (m * k);
    if batch == 1 {
        // Avoid parallel overhead for single matrix multiplication
        unsafe {
            gemm_f32(
                m,
                k,
                n,
                lhs_data.as_ptr(),
                rhs_data.as_ptr(),
                output_data.as_mut_ptr(),
            )
        };
    } else {
        output_data
            .par_chunks_mut(m * n)
            .enumerate()
            .for_each(|(b, chunk)| {
                let a = &lhs_data[b * m * k..(b + 1) * m * k];
                let r = &rhs_data[b * k * n..(b + 1) * k * n];
                unsafe {
                    gemm_f32(m, k, n, a.as_ptr(), r.as_ptr(), chunk.as_mut_ptr());
                }
            });
    }

    Ok(())
}

fn optimized_matmul_f64(
    lhs_data: &[f64],
    rhs_data: &[f64],
    output_data: &mut [f64],
    lhs_shape: &Shape,
    rhs_shape: &Shape,
) -> Result<()> {
    let lhs_dims = lhs_shape.dims();
    let rhs_dims = rhs_shape.dims();
    let m = lhs_dims[lhs_dims.len() - 2];
    let k = lhs_dims[lhs_dims.len() - 1];
    let n = rhs_dims[rhs_dims.len() - 1];

    if m == 0 || k == 0 || n == 0 {
        return Ok(());
    }

    let batch = lhs_data.len() / (m * k);
    if batch == 1 {
        unsafe {
            gemm_f64(
                m,
                k,
                n,
                lhs_data.as_ptr(),
                rhs_data.as_ptr(),
                output_data.as_mut_ptr(),
            )
        };
    } else {
        output_data
            .par_chunks_mut(m * n)
            .enumerate()
            .for_each(|(b, chunk)| {
                let a = &lhs_data[b * m * k..(b + 1) * m * k];
                let r = &rhs_data[b * k * n..(b + 1) * k * n];
                unsafe {
                    gemm_f64(m, k, n, a.as_ptr(), r.as_ptr(), chunk.as_mut_ptr());
                }
            });
    }

    Ok(())
}

// Helper functions for transpose operations

fn transpose_f32(
    tensor: &Tensor,
    output_data: &mut TensorData,
    output_shape: &Shape,
    dim0: usize,
    dim1: usize,
) -> Result<()> {
    let input_data = tensor.data().as_f32_slice().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get f32 slice from input tensor")
    })?;

    let output_slice = output_data.as_f32_slice_mut().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get mutable f32 slice from output data")
    })?;

    transpose_generic(
        input_data,
        output_slice,
        tensor.shape(),
        output_shape,
        dim0,
        dim1,
    )
}

fn transpose_f64(
    tensor: &Tensor,
    output_data: &mut TensorData,
    output_shape: &Shape,
    dim0: usize,
    dim1: usize,
) -> Result<()> {
    let input_data = tensor.data().as_f64_slice().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get f64 slice from input tensor")
    })?;

    let output_slice = output_data.as_f64_slice_mut().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get mutable f64 slice from output data")
    })?;

    transpose_generic(
        input_data,
        output_slice,
        tensor.shape(),
        output_shape,
        dim0,
        dim1,
    )
}

fn transpose_i32(
    tensor: &Tensor,
    output_data: &mut TensorData,
    output_shape: &Shape,
    dim0: usize,
    dim1: usize,
) -> Result<()> {
    let input_data = tensor.data().as_i32_slice().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get i32 slice from input tensor")
    })?;

    let output_slice = output_data.as_i32_slice_mut().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get mutable i32 slice from output data")
    })?;

    transpose_generic(
        input_data,
        output_slice,
        tensor.shape(),
        output_shape,
        dim0,
        dim1,
    )
}

fn transpose_i64(
    tensor: &Tensor,
    output_data: &mut TensorData,
    output_shape: &Shape,
    dim0: usize,
    dim1: usize,
) -> Result<()> {
    let input_data = tensor.data().as_i64_slice().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get i64 slice from input tensor")
    })?;

    let output_slice = output_data.as_i64_slice_mut().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get mutable i64 slice from output data")
    })?;

    transpose_generic(
        input_data,
        output_slice,
        tensor.shape(),
        output_shape,
        dim0,
        dim1,
    )
}

fn transpose_bool(
    tensor: &Tensor,
    output_data: &mut TensorData,
    output_shape: &Shape,
    dim0: usize,
    dim1: usize,
) -> Result<()> {
    let input_data = tensor.data().as_bool_slice().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get bool slice from input tensor")
    })?;

    let output_slice = output_data.as_bool_slice_mut().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get mutable bool slice from output data")
    })?;

    transpose_generic(
        input_data,
        output_slice,
        tensor.shape(),
        output_shape,
        dim0,
        dim1,
    )
}

/// Generic transpose implementation
fn transpose_generic<T: Copy + Send + Sync>(
    input_data: &[T],
    output_data: &mut [T],
    input_shape: &Shape,
    output_shape: &Shape,
    dim0: usize,
    dim1: usize,
) -> Result<()> {
    // Fast path for 2D matrix transpose
    if input_shape.ndim() == 2 && dim0 == 0 && dim1 == 1 {
        let rows = input_shape.dims()[0];
        let cols = input_shape.dims()[1];
        if rows * cols < PAR_THRESHOLD {
            for i in 0..rows {
                for j in 0..cols {
                    unsafe {
                        *output_data.get_unchecked_mut(j * rows + i) =
                            *input_data.get_unchecked(i * cols + j);
                    }
                }
            }
        } else {
            output_data
                .par_chunks_mut(rows)
                .enumerate()
                .for_each(|(j, col)| {
                    for i in 0..rows {
                        unsafe {
                            col[i] = *input_data.get_unchecked(i * cols + j);
                        }
                    }
                });
        }
        return Ok(());
    }

    let input_strides = Strides::from_shape(input_shape);
    let output_strides = Strides::from_shape(output_shape);
    let in_strides = input_strides.as_slice().to_vec();
    let out_strides = output_strides.as_slice().to_vec();
    let out_dims = output_shape.dims().to_vec();

    output_data
        .par_iter_mut()
        .enumerate()
        .for_each(|(idx, out)| {
            let mut remaining = idx;
            let mut input_linear = 0;
            for dim in 0..out_dims.len() {
                let stride = out_strides[dim];
                let coord = remaining / stride;
                remaining %= stride;
                let in_dim = if dim == dim0 {
                    dim1
                } else if dim == dim1 {
                    dim0
                } else {
                    dim
                };
                input_linear += coord * in_strides[in_dim];
            }
            *out = input_data[input_linear];
        });

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{autograd::GradientFunction, device::Device, tensor::TensorData};

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

    fn create_test_tensor_f64(data: Vec<f64>, shape: Vec<usize>, requires_grad: bool) -> Tensor {
        let shape_obj = Shape::new(shape);
        let mut tensor_data = TensorData::zeros(shape_obj.numel(), DataType::Float64);

        if let Some(slice) = tensor_data.as_f64_slice_mut() {
            slice.copy_from_slice(&data);
        }

        Tensor::new(
            Arc::new(tensor_data),
            shape_obj,
            DataType::Float64,
            Device::cpu(),
            requires_grad,
        )
    }

    fn create_test_tensor_bool(data: Vec<bool>, shape: Vec<usize>) -> Tensor {
        let shape_obj = Shape::new(shape);
        let mut tensor_data = TensorData::zeros(shape_obj.numel(), DataType::Bool);

        if let Some(slice) = tensor_data.as_bool_slice_mut() {
            slice.copy_from_slice(&data);
        }

        Tensor::new(
            Arc::new(tensor_data),
            shape_obj,
            DataType::Bool,
            Device::cpu(),
            false,
        )
    }

    fn create_test_tensor_f32_on_device(
        data: Vec<f32>,
        shape: Vec<usize>,
        device: Device,
    ) -> Tensor {
        let shape_obj = Shape::new(shape);
        let mut tensor_data =
            TensorData::zeros_on_device(shape_obj.numel(), DataType::Float32, device);

        if let Some(slice) = tensor_data.as_f32_slice_mut() {
            slice.copy_from_slice(&data);
        }

        Tensor::new(
            Arc::new(tensor_data),
            shape_obj,
            DataType::Float32,
            device,
            false,
        )
    }

    #[test]
    fn test_matmul_basic() {
        // 2x3 * 3x2 = 2x2
        let a = create_test_tensor_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3], false);
        let b = create_test_tensor_f32(vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0], vec![3, 2], false);

        let result = matmul(&a, &b).unwrap();
        let result_data = result.data().as_f32_slice().unwrap();

        // Expected: [1*7+2*9+3*11, 1*8+2*10+3*12; 4*7+5*9+6*11, 4*8+5*10+6*12]
        // = [58, 64; 139, 154]
        assert_eq!(result_data, &[58.0, 64.0, 139.0, 154.0]);
        assert_eq!(result.shape().dims(), &[2, 2]);
    }

    #[test]
    fn test_transpose_2d() {
        let a = create_test_tensor_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3], false);

        let result = transpose(&a, 0, 1).unwrap();
        let result_data = result.data().as_f32_slice().unwrap();

        // Original: [[1, 2, 3], [4, 5, 6]]
        // Transposed: [[1, 4], [2, 5], [3, 6]]
        assert_eq!(result_data, &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
        assert_eq!(result.shape().dims(), &[3, 2]);
    }

    #[test]
    fn test_matmul_dimension_mismatch() {
        let a = create_test_tensor_f32(vec![1.0, 2.0], vec![1, 2], false);
        let b = create_test_tensor_f32(vec![3.0, 4.0, 5.0], vec![3, 1], false);

        let result = matmul(&a, &b);
        assert!(result.is_err());
    }

    #[test]
    fn test_transpose_same_dim() {
        let a = create_test_tensor_f32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], false);

        let result = transpose(&a, 0, 0).unwrap();
        let result_data = result.data().as_f32_slice().unwrap();

        // Should be unchanged
        assert_eq!(result_data, &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(result.shape().dims(), &[2, 2]);
    }

    #[test]
    fn test_gradient_tracking() {
        let a = create_test_tensor_f32(vec![1.0, 2.0], vec![1, 2], true);
        let b = create_test_tensor_f32(vec![3.0, 4.0], vec![2, 1], true);

        let result = matmul(&a, &b).unwrap();

        assert!(result.requires_grad());
        assert!(result.grad_fn().is_some());
    }

    #[test]
    fn test_matmul_dtype_mismatch() {
        let a = create_test_tensor_f32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], false);
        let b = create_test_tensor_f64(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2], false);

        let result = matmul(&a, &b);
        assert!(result.is_err());
    }

    #[test]
    fn test_matmul_device_mismatch() {
        let a =
            create_test_tensor_f32_on_device(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], Device::cpu());
        let b = create_test_tensor_f32_on_device(
            vec![5.0, 6.0, 7.0, 8.0],
            vec![2, 2],
            Device::cuda(None),
        );

        let result = matmul(&a, &b);
        assert!(result.is_err());
    }

    #[test]
    fn test_matmul_bool_error() {
        let a = create_test_tensor_bool(vec![true, false, true, false], vec![2, 2]);
        let b = create_test_tensor_bool(vec![true, true, false, false], vec![2, 2]);

        let result = matmul(&a, &b);
        assert!(result.is_err());
    }

    #[test]
    fn test_matmul_requires_2d_inputs() {
        let a = create_test_tensor_f32(vec![1.0, 2.0], vec![2], false);
        let b = create_test_tensor_f32(vec![3.0, 4.0], vec![2], false);

        let result = matmul(&a, &b);
        assert!(result.is_err());
    }

    #[test]
    fn test_diagonal_main() {
        let tensor = create_test_tensor_f32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], false);
        let result = diagonal(&tensor, 0, 0, 1).unwrap();
        let data = result.data().as_f32_slice().unwrap();
        assert_eq!(data, &[1.0, 4.0]);
        assert_eq!(result.shape().dims(), &[2]);
    }

    #[test]
    fn test_diagonal_with_offset() {
        let tensor = create_test_tensor_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3], false);
        let upper = diagonal(&tensor, 1, 0, 1).unwrap();
        assert_eq!(upper.data().as_f32_slice().unwrap(), &[2.0, 6.0]);

        let lower = diagonal(&tensor, -1, 0, 1).unwrap();
        assert_eq!(lower.data().as_f32_slice().unwrap(), &[4.0]);
    }

    #[test]
    fn test_diagonal_high_dim_shape() {
        let tensor =
            create_test_tensor_f32((0..24).map(|v| v as f32).collect(), vec![2, 3, 4], false);
        let result = diagonal(&tensor, 0, 1, 2).unwrap();
        assert_eq!(result.shape().dims(), &[2, 3]);
    }

    #[test]
    fn test_diagonal_backward_gradients() {
        let tensor = create_test_tensor_f32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], true);
        let grad_output = create_test_tensor_f32(vec![1.0, 1.0], vec![2], false);

        let backward_fn = crate::autograd::DiagonalBackward {
            input_shape: tensor.shape().dims().to_vec(),
            input_strides: tensor.strides().as_slice().to_vec(),
            input_dtype: DataType::Float32,
            dim1: 0,
            dim2: 1,
            offset: 0,
            input_requires_grad: true,
            input_id: tensor.id(),
        };

        let gradients = backward_fn.backward(&grad_output).unwrap();
        let grad_tensor = gradients.get(&tensor.id()).unwrap();
        let grad = grad_tensor.data().as_f32_slice().unwrap();
        assert_eq!(grad, &[1.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn test_trace_matches_manual_sum() {
        let tensor = create_test_tensor_f32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], false);
        let traced = trace(&tensor, 0, 0, 1).unwrap();
        let value = traced.data().as_f32_slice().unwrap();
        assert_eq!(value, &[5.0]);
    }

    #[test]
    fn test_triu_basic() {
        let tensor = create_test_tensor_f32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], false);
        let result = triu(&tensor, 0).unwrap();
        let data = result.data().as_f32_slice().unwrap();
        assert_eq!(data, &[1.0, 2.0, 0.0, 4.0]);
    }

    #[test]
    fn test_triu_with_positive_diagonal() {
        let tensor = create_test_tensor_f32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], false);
        let result = triu(&tensor, 1).unwrap();
        let data = result.data().as_f32_slice().unwrap();
        assert_eq!(data, &[0.0, 2.0, 0.0, 0.0]);
    }

    #[test]
    fn test_tril_basic() {
        let tensor = create_test_tensor_f32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], false);
        let result = tril(&tensor, 0).unwrap();
        let data = result.data().as_f32_slice().unwrap();
        assert_eq!(data, &[1.0, 0.0, 3.0, 4.0]);
    }

    #[test]
    fn test_tril_with_negative_diagonal() {
        let tensor = create_test_tensor_f32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], false);
        let result = tril(&tensor, -1).unwrap();
        let data = result.data().as_f32_slice().unwrap();
        assert_eq!(data, &[0.0, 0.0, 3.0, 0.0]);
    }
}
