// Copyright (c) 2025 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

use crate::{
    autograd::{add_to_graph, MatMulBackward, TransposeBackward},
    error::{MinitensorError, Result},
    tensor::{DataType, Shape, Strides, Tensor, TensorData},
};
use rayon::prelude::*;
use std::sync::Arc;

#[cfg(feature = "blas")]
use cblas::{Layout, Transpose};

const PAR_THRESHOLD: usize = 1 << 12;

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
    matrixmultiply::sgemm(
        m, k, n, 1.0, a, k as isize, 1, b, n as isize, 1, 0.0, c, n as isize, 1,
    );
}

#[cfg(not(feature = "blas"))]
#[inline]
unsafe fn gemm_f64(m: usize, k: usize, n: usize, a: *const f64, b: *const f64, c: *mut f64) {
    matrixmultiply::dgemm(
        m, k, n, 1.0, a, k as isize, 1, b, n as isize, 1, 0.0, c, n as isize, 1,
    );
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
            ))
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

/// Transpose operation with gradient support
pub fn transpose(tensor: &Tensor, dim0: usize, dim1: usize) -> Result<Tensor> {
    // Validate dimensions
    if dim0 >= tensor.ndim() || dim1 >= tensor.ndim() {
        return Err(MinitensorError::index_error(
            dim0.max(dim1) as isize,
            0,
            tensor.ndim(),
        ));
    }

    if dim0 == dim1 {
        // No-op transpose
        return Ok(tensor.clone());
    }

    // Create new shape with swapped dimensions
    let mut new_shape = tensor.shape().dims().to_vec();
    new_shape.swap(dim0, dim1);
    let new_shape_obj = Shape::new(new_shape);

    // Create new strides with swapped dimensions
    let old_strides = tensor.strides().as_slice();
    let mut new_strides = old_strides.to_vec();
    new_strides.swap(dim0, dim1);

    // Create output tensor data by copying and rearranging
    let mut output_data =
        TensorData::zeros_on_device(tensor.numel(), tensor.dtype(), tensor.device());

    // Perform transpose based on data type
    match tensor.dtype() {
        DataType::Float32 => transpose_f32(tensor, &mut output_data, &new_shape_obj, dim0, dim1)?,
        DataType::Float64 => transpose_f64(tensor, &mut output_data, &new_shape_obj, dim0, dim1)?,
        DataType::Int32 => transpose_i32(tensor, &mut output_data, &new_shape_obj, dim0, dim1)?,
        DataType::Int64 => transpose_i64(tensor, &mut output_data, &new_shape_obj, dim0, dim1)?,
        DataType::Bool => transpose_bool(tensor, &mut output_data, &new_shape_obj, dim0, dim1)?,
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
            dims: vec![dim0, dim1],
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
    use crate::{device::Device, tensor::TensorData};

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
}
