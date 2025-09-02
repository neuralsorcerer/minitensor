// Copyright (c) 2025 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

use crate::{
    autograd::{add_to_graph, MatMulBackward, TransposeBackward},
    error::{MinitensorError, Result},
    tensor::{DataType, Shape, Strides, Tensor, TensorData},
};
use std::sync::Arc;

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

    naive_matmul(
        lhs_data,
        rhs_data,
        output_slice,
        lhs.shape(),
        rhs.shape(),
        output_shape,
    )
}

fn matmul_f64(
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

    naive_matmul(
        lhs_data,
        rhs_data,
        output_slice,
        lhs.shape(),
        rhs.shape(),
        output_shape,
    )
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

/// Naive matrix multiplication implementation (O(n^3))
fn naive_matmul<T>(
    lhs_data: &[T],
    rhs_data: &[T],
    output_data: &mut [T],
    lhs_shape: &Shape,
    rhs_shape: &Shape,
    _output_shape: &Shape,
) -> Result<()>
where
    T: Copy + std::ops::Add<Output = T> + std::ops::Mul<Output = T> + Default,
{
    let lhs_dims = lhs_shape.dims();
    let rhs_dims = rhs_shape.dims();

    let m = lhs_dims[lhs_dims.len() - 2]; // rows of lhs
    let k = lhs_dims[lhs_dims.len() - 1]; // cols of lhs / rows of rhs
    let n = rhs_dims[rhs_dims.len() - 1]; // cols of rhs

    // For simplicity, we'll handle 2D matrices first
    if lhs_shape.ndim() == 2 && rhs_shape.ndim() == 2 {
        for i in 0..m {
            for j in 0..n {
                let mut sum = T::default();
                for l in 0..k {
                    let lhs_idx = i * k + l;
                    let rhs_idx = l * n + j;
                    sum = sum + lhs_data[lhs_idx] * rhs_data[rhs_idx];
                }
                let output_idx = i * n + j;
                output_data[output_idx] = sum;
            }
        }
    } else {
        // For higher-dimensional tensors, we need to handle batch dimensions
        // For now, we'll return an error for non-2D matrices
        return Err(MinitensorError::invalid_operation(
            "Batched matrix multiplication not yet implemented",
        ));
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
fn transpose_generic<T: Copy>(
    input_data: &[T],
    output_data: &mut [T],
    input_shape: &Shape,
    output_shape: &Shape,
    dim0: usize,
    dim1: usize,
) -> Result<()> {
    let input_strides = Strides::from_shape(input_shape);

    // For each element in the output tensor
    for i in 0..output_shape.numel() {
        // Convert linear index to multi-dimensional indices for output
        let mut output_indices = vec![0; output_shape.ndim()];
        let mut remaining = i;
        for dim_idx in 0..output_shape.ndim() {
            let stride = if dim_idx + 1 < output_shape.ndim() {
                output_shape.dims()[dim_idx + 1..].iter().product::<usize>()
            } else {
                1
            };
            output_indices[dim_idx] = remaining / stride;
            remaining %= stride;
        }

        // Create corresponding input indices by swapping the transposed dimensions
        let mut input_indices = output_indices.clone();
        input_indices.swap(dim0, dim1);

        // Compute linear indices
        let input_linear = input_strides.linear_index(&input_indices);

        // Copy the data
        output_data[i] = input_data[input_linear];
    }

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
}
