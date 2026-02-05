// Copyright (c) 2026 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

use crate::{
    autograd::{DotBackward, MatMulBackward, SolveBackward, TransposeBackward, add_to_graph},
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

    if lhs.dtype() == DataType::Bool {
        return Err(MinitensorError::invalid_operation(
            "Matrix multiplication not supported for boolean tensors",
        ));
    }

    // Create output tensor data
    let mut output_data =
        TensorData::zeros_on_device(output_shape_obj.numel(), lhs.dtype(), lhs.device());

    if output_shape_obj.numel() != 0 && lhs_cols != 0 {
        // Perform matrix multiplication based on data type
        match lhs.dtype() {
            DataType::Float32 => matmul_f32(lhs, rhs, &mut output_data, &output_shape_obj)?,
            DataType::Float64 => matmul_f64(lhs, rhs, &mut output_data, &output_shape_obj)?,
            DataType::Int32 => matmul_i32(lhs, rhs, &mut output_data, &output_shape_obj)?,
            DataType::Int64 => matmul_i64(lhs, rhs, &mut output_data, &output_shape_obj)?,
            DataType::Bool => unreachable!("bool dtype checked above"),
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

/// Solve a linear system of equations `AX = B` for `X`.
///
/// Both `lhs` (`A`) and `rhs` (`B`) must be float tensors that live on the CPU.
/// `lhs` must have shape `[..., n, n]` (square matrices) and `rhs` can either have
/// shape `[..., n]` (a collection of vectors) or `[..., n, k]` (multiple right
/// hand sides). Batch dimensions need to match exactly across the operands.
pub fn solve(lhs: &Tensor, rhs: &Tensor) -> Result<Tensor> {
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

    let lhs_ndim = lhs.ndim();
    if lhs_ndim < 2 {
        return Err(MinitensorError::invalid_operation(
            "solve expects lhs to have at least 2 dimensions",
        ));
    }

    let lhs_shape = lhs.shape().dims();
    let n = lhs_shape[lhs_ndim - 1];
    let m = lhs_shape[lhs_ndim - 2];
    if n != m {
        return Err(MinitensorError::invalid_operation(
            "solve expects lhs matrices to be square",
        ));
    }

    let rhs_ndim = rhs.ndim();
    if rhs_ndim < 1 {
        return Err(MinitensorError::invalid_operation(
            "solve expects rhs to have at least 1 dimension",
        ));
    }

    let rhs_shape = rhs.shape().dims();
    let (rhs_cols, rhs_batch_dims) = if rhs_ndim == lhs_ndim {
        if rhs_shape[rhs_ndim - 2] != n {
            return Err(MinitensorError::shape_mismatch(
                vec![n],
                vec![rhs_shape[rhs_ndim - 2]],
            ));
        }
        (rhs_shape[rhs_ndim - 1], &rhs_shape[..rhs_ndim - 2])
    } else if rhs_ndim + 1 == lhs_ndim {
        if rhs_shape[rhs_ndim - 1] != n {
            return Err(MinitensorError::shape_mismatch(
                vec![n],
                vec![rhs_shape[rhs_ndim - 1]],
            ));
        }
        (1usize, &rhs_shape[..rhs_ndim - 1])
    } else {
        return Err(MinitensorError::invalid_operation(
            "solve expects rhs to have either the same rank as lhs or one less",
        ));
    };

    if &lhs_shape[..lhs_ndim - 2] != rhs_batch_dims {
        return Err(MinitensorError::shape_mismatch(
            lhs_shape[..lhs_ndim - 2].to_vec(),
            rhs_batch_dims.to_vec(),
        ));
    }

    let requires_grad = lhs.requires_grad() || rhs.requires_grad();

    let output_shape = rhs_shape.to_vec();
    let output_shape = Shape::new(output_shape);

    let mut output_data =
        TensorData::zeros_on_device(output_shape.numel(), lhs.dtype(), lhs.device());

    match lhs.dtype() {
        DataType::Float32 => solve_f32(lhs, rhs, &mut output_data, rhs_cols)?,
        DataType::Float64 => solve_f64(lhs, rhs, &mut output_data, rhs_cols)?,
        _ => {
            return Err(MinitensorError::invalid_operation(
                "solve currently supports only Float32 and Float64 tensors",
            ));
        }
    }

    let mut output = Tensor::new(
        Arc::new(output_data),
        output_shape,
        lhs.dtype(),
        lhs.device(),
        requires_grad,
    );

    if output.requires_grad() {
        let grad_fn = Arc::new(SolveBackward {
            lhs: lhs.detach(),
            solution: output.detach(),
            input_ids: [lhs.id(), rhs.id()],
            lhs_requires_grad: lhs.requires_grad(),
            rhs_requires_grad: rhs.requires_grad(),
        });
        output.set_grad_fn(Some(grad_fn.clone()));
        add_to_graph(&output, Some(grad_fn))?;
    }

    Ok(output)
}

fn solve_f32(lhs: &Tensor, rhs: &Tensor, output: &mut TensorData, rhs_cols: usize) -> Result<()> {
    use std::borrow::Cow;

    let lhs_view = if lhs.is_contiguous() && lhs.data().is_contiguous() {
        Cow::Borrowed(lhs)
    } else {
        Cow::Owned(lhs.contiguous()?)
    };
    let rhs_view = if rhs.is_contiguous() && rhs.data().is_contiguous() {
        Cow::Borrowed(rhs)
    } else {
        Cow::Owned(rhs.contiguous()?)
    };

    let lhs_slice = lhs_view
        .data()
        .as_f32_slice()
        .ok_or_else(|| MinitensorError::internal_error("Failed to access f32 data for lhs"))?;
    let rhs_slice = rhs_view
        .data()
        .as_f32_slice()
        .ok_or_else(|| MinitensorError::internal_error("Failed to access f32 data for rhs"))?;
    let out_slice = output
        .as_f32_slice_mut()
        .ok_or_else(|| MinitensorError::internal_error("Failed to access f32 output slice"))?;

    solve_batched(
        lhs.shape().dims(),
        rhs_cols,
        lhs_slice,
        rhs_slice,
        out_slice,
    )
}

fn solve_f64(lhs: &Tensor, rhs: &Tensor, output: &mut TensorData, rhs_cols: usize) -> Result<()> {
    use std::borrow::Cow;

    let lhs_view = if lhs.is_contiguous() && lhs.data().is_contiguous() {
        Cow::Borrowed(lhs)
    } else {
        Cow::Owned(lhs.contiguous()?)
    };
    let rhs_view = if rhs.is_contiguous() && rhs.data().is_contiguous() {
        Cow::Borrowed(rhs)
    } else {
        Cow::Owned(rhs.contiguous()?)
    };

    let lhs_slice = lhs_view
        .data()
        .as_f64_slice()
        .ok_or_else(|| MinitensorError::internal_error("Failed to access f64 data for lhs"))?;
    let rhs_slice = rhs_view
        .data()
        .as_f64_slice()
        .ok_or_else(|| MinitensorError::internal_error("Failed to access f64 data for rhs"))?;
    let out_slice = output
        .as_f64_slice_mut()
        .ok_or_else(|| MinitensorError::internal_error("Failed to access f64 output slice"))?;

    solve_batched(
        lhs.shape().dims(),
        rhs_cols,
        lhs_slice,
        rhs_slice,
        out_slice,
    )
}

fn solve_batched<T>(
    lhs_shape: &[usize],
    rhs_cols: usize,
    lhs_slice: &[T],
    rhs_slice: &[T],
    out_slice: &mut [T],
) -> Result<()>
where
    T: Copy
        + Send
        + Sync
        + std::ops::SubAssign
        + std::ops::Mul<Output = T>
        + std::ops::Div<Output = T>
        + std::ops::Neg<Output = T>
        + PartialOrd
        + Default
        + PartialEq,
{
    let n = *lhs_shape.last().expect("lhs has at least 2 dims");
    let batch = lhs_shape[..lhs_shape.len() - 2]
        .iter()
        .copied()
        .product::<usize>()
        .max(1);
    let rhs_stride = n * rhs_cols;

    let matrix_stride = n * n;
    let mut matrix = vec![T::default(); matrix_stride];
    let mut rhs_buf = vec![T::default(); rhs_stride];

    for batch_idx in 0..batch {
        let lhs_offset = batch_idx * matrix_stride;
        let rhs_offset = batch_idx * rhs_stride;

        matrix.copy_from_slice(&lhs_slice[lhs_offset..lhs_offset + matrix_stride]);
        rhs_buf[..rhs_stride].copy_from_slice(&rhs_slice[rhs_offset..rhs_offset + rhs_stride]);

        gaussian_elimination(&mut matrix, &mut rhs_buf, n, rhs_cols)?;

        out_slice[rhs_offset..rhs_offset + rhs_stride].copy_from_slice(&rhs_buf[..rhs_stride]);
    }

    Ok(())
}

fn gaussian_elimination<T>(matrix: &mut [T], rhs: &mut [T], n: usize, rhs_cols: usize) -> Result<()>
where
    T: Copy
        + Send
        + Sync
        + std::ops::SubAssign
        + std::ops::Mul<Output = T>
        + std::ops::Div<Output = T>
        + std::ops::Neg<Output = T>
        + PartialOrd
        + Default
        + PartialEq,
{
    for k in 0..n {
        // Pivot selection
        let mut pivot_row = k;
        let mut pivot_val = abs(matrix[k * n + k]);
        for i in (k + 1)..n {
            let candidate = abs(matrix[i * n + k]);
            if candidate > pivot_val {
                pivot_val = candidate;
                pivot_row = i;
            }
        }

        if pivot_val == T::default() {
            return Err(MinitensorError::invalid_operation(
                "solve received a singular matrix",
            ));
        }

        if pivot_row != k {
            for col in 0..n {
                matrix.swap(k * n + col, pivot_row * n + col);
            }
            for col in 0..rhs_cols {
                rhs.swap(k * rhs_cols + col, pivot_row * rhs_cols + col);
            }
        }

        let pivot = matrix[k * n + k];

        for i in (k + 1)..n {
            let factor = matrix[i * n + k] / pivot;
            matrix[i * n + k] = T::default();
            for j in (k + 1)..n {
                let idx = i * n + j;
                matrix[idx] -= factor * matrix[k * n + j];
            }
            for col in 0..rhs_cols {
                let idx = i * rhs_cols + col;
                rhs[idx] -= factor * rhs[k * rhs_cols + col];
            }
        }
    }

    for i in (0..n).rev() {
        let pivot = matrix[i * n + i];
        if abs(pivot) == T::default() {
            return Err(MinitensorError::invalid_operation(
                "solve received a singular matrix",
            ));
        }
        for col in 0..rhs_cols {
            let mut value = rhs[i * rhs_cols + col];
            for j in (i + 1)..n {
                value -= matrix[i * n + j] * rhs[j * rhs_cols + col];
            }
            rhs[i * rhs_cols + col] = value / pivot;
        }
    }

    Ok(())
}

fn abs<T>(value: T) -> T
where
    T: Copy + PartialOrd + std::ops::Neg<Output = T> + Default,
{
    if value < T::default() { -value } else { value }
}

/// Batched matrix multiplication specialized for 3D tensors.
///
/// This is a thin convenience wrapper around [`matmul`] that enforces the
/// traditional batch matrix multiply constraints: both operands must be
/// rank-3 tensors with matching batch dimensions. The actual computation is
/// still delegated to the highly optimised [`matmul`] implementation so all
/// execution happens inside the Rust backend.
pub fn bmm(lhs: &Tensor, rhs: &Tensor) -> Result<Tensor> {
    if lhs.ndim() != 3 || rhs.ndim() != 3 {
        return Err(MinitensorError::invalid_operation(
            "bmm expects both inputs to be 3D tensors".to_string(),
        ));
    }

    let lhs_shape = lhs.shape().dims();
    let rhs_shape = rhs.shape().dims();

    if lhs_shape[0] != rhs_shape[0] {
        return Err(MinitensorError::shape_mismatch(
            lhs_shape.to_vec(),
            rhs_shape.to_vec(),
        ));
    }

    if lhs_shape[2] != rhs_shape[1] {
        return Err(MinitensorError::shape_mismatch(
            vec![lhs_shape[2]],
            vec![rhs_shape[1]],
        ));
    }

    matmul(lhs, rhs)
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
