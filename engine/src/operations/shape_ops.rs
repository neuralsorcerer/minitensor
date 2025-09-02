// Copyright (c) 2025 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

use crate::{
    autograd::{add_to_graph, ReshapeBackward},
    error::{MinitensorError, Result},
    tensor::{DataType, Shape, Tensor, TensorData},
};
use std::sync::Arc;

/// Reshape operation with gradient support
pub fn reshape(tensor: &Tensor, new_shape: Shape) -> Result<Tensor> {
    // Check if the total number of elements matches
    if tensor.numel() != new_shape.numel() {
        return Err(MinitensorError::shape_mismatch(
            vec![tensor.numel()],
            vec![new_shape.numel()],
        ));
    }

    // Use the tensor's built-in view method for reshaping
    let reshaped = tensor.view(new_shape.clone())?;

    // Set up gradient function if needed
    if reshaped.requires_grad() {
        let grad_fn = Arc::new(ReshapeBackward {
            input_shape: tensor.shape().dims().to_vec(),
            input_id: tensor.id(),
        });

        let mut reshaped_with_grad = reshaped;
        reshaped_with_grad.set_grad_fn(Some(grad_fn.clone()));

        // Add to computation graph
        add_to_graph(&reshaped_with_grad, Some(grad_fn))?;

        Ok(reshaped_with_grad)
    } else {
        Ok(reshaped)
    }
}

/// Squeeze operation - remove dimensions of size 1
pub fn squeeze(tensor: &Tensor, dim: Option<usize>) -> Result<Tensor> {
    let current_shape = tensor.shape().dims();

    match dim {
        Some(d) => {
            // Squeeze specific dimension
            if d >= current_shape.len() {
                return Err(MinitensorError::index_error(
                    d as isize,
                    0,
                    current_shape.len(),
                ));
            }

            if current_shape[d] != 1 {
                return Err(MinitensorError::invalid_operation(format!(
                    "Cannot squeeze dimension {} with size {}",
                    d, current_shape[d]
                )));
            }

            let mut new_shape = current_shape.to_vec();
            new_shape.remove(d);

            reshape(tensor, Shape::new(new_shape))
        }
        None => {
            // Squeeze all dimensions of size 1
            let new_shape: Vec<usize> = current_shape
                .iter()
                .copied()
                .filter(|&size| size != 1)
                .collect();

            // Handle the case where all dimensions are 1 (scalar)
            let new_shape = if new_shape.is_empty() {
                vec![1] // Keep at least one dimension
            } else {
                new_shape
            };

            reshape(tensor, Shape::new(new_shape))
        }
    }
}

/// Unsqueeze operation - add a dimension of size 1
pub fn unsqueeze(tensor: &Tensor, dim: usize) -> Result<Tensor> {
    let current_shape = tensor.shape().dims();

    // Allow inserting at the end (dim == current_shape.len())
    if dim > current_shape.len() {
        return Err(MinitensorError::index_error(
            dim as isize,
            0,
            current_shape.len() + 1,
        ));
    }

    let mut new_shape = current_shape.to_vec();
    new_shape.insert(dim, 1);

    reshape(tensor, Shape::new(new_shape))
}

/// Concatenate tensors along a specified dimension
pub fn concatenate(tensors: &[&Tensor], dim: usize) -> Result<Tensor> {
    if tensors.is_empty() {
        return Err(MinitensorError::invalid_operation(
            "Cannot concatenate empty list of tensors",
        ));
    }

    let first_tensor = tensors[0];

    // Validate that all tensors have the same number of dimensions
    for tensor in tensors.iter().skip(1) {
        if tensor.ndim() != first_tensor.ndim() {
            return Err(MinitensorError::shape_mismatch(
                vec![first_tensor.ndim()],
                vec![tensor.ndim()],
            ));
        }

        // Check device compatibility
        if tensor.device() != first_tensor.device() {
            return Err(MinitensorError::device_mismatch(
                format!("{:?}", first_tensor.device()),
                format!("{:?}", tensor.device()),
            ));
        }

        // Check data type compatibility
        if tensor.dtype() != first_tensor.dtype() {
            return Err(MinitensorError::type_mismatch(
                format!("{:?}", first_tensor.dtype()),
                format!("{:?}", tensor.dtype()),
            ));
        }
    }

    // Validate concatenation dimension
    if dim >= first_tensor.ndim() {
        return Err(MinitensorError::index_error(
            dim as isize,
            0,
            first_tensor.ndim(),
        ));
    }

    // Validate that all dimensions except the concatenation dimension match
    for tensor in tensors.iter().skip(1) {
        for (i, (&size1, &size2)) in first_tensor
            .shape()
            .dims()
            .iter()
            .zip(tensor.shape().dims().iter())
            .enumerate()
        {
            if i != dim && size1 != size2 {
                return Err(MinitensorError::shape_mismatch(
                    first_tensor.shape().dims().to_vec(),
                    tensor.shape().dims().to_vec(),
                ));
            }
        }
    }

    if !first_tensor.device().is_cpu() {
        return Err(MinitensorError::invalid_operation(
            "concatenate currently supports only CPU tensors",
        ));
    }

    // Compute output shape
    let mut output_shape = first_tensor.shape().dims().to_vec();
    output_shape[dim] = tensors.iter().map(|t| t.shape().dims()[dim]).sum();
    let output_shape_vec = output_shape.clone();
    let output_shape_obj = Shape::new(output_shape);

    let dtype = first_tensor.dtype();
    let device = first_tensor.device();
    let requires_grad = tensors.iter().any(|t| t.requires_grad());

    let dims = first_tensor.shape().dims();
    let inner: usize = dims[dim + 1..].iter().product();
    let outer: usize = dims[..dim].iter().product();

    macro_rules! concat_impl {
        ($ty:ty, $slice:ident, $from_vec:ident) => {{
            let mut out = vec![<$ty>::default(); output_shape_obj.numel()];
            for o in 0..outer {
                let mut dst_offset = o * output_shape_vec[dim] * inner;
                for t in tensors {
                    let t_dims = t.shape().dims();
                    let src_start = o * t_dims[dim] * inner;
                    let src_len = t_dims[dim] * inner;
                    let src = t.data().$slice().ok_or_else(|| {
                        MinitensorError::invalid_operation(
                            "Tensor data access failed for concatenate",
                        )
                    })?;
                    out[dst_offset..dst_offset + src_len]
                        .copy_from_slice(&src[src_start..src_start + src_len]);
                    dst_offset += src_len;
                }
            }
            TensorData::$from_vec(out, device)
        }};
    }

    let data = match dtype {
        DataType::Float32 => concat_impl!(f32, as_f32_slice, from_vec_f32),
        DataType::Float64 => concat_impl!(f64, as_f64_slice, from_vec_f64),
        DataType::Int32 => concat_impl!(i32, as_i32_slice, from_vec_i32),
        DataType::Int64 => concat_impl!(i64, as_i64_slice, from_vec_i64),
        DataType::Bool => concat_impl!(bool, as_bool_slice, from_vec_bool),
    };

    Ok(Tensor::new(
        Arc::new(data),
        output_shape_obj,
        dtype,
        device,
        requires_grad,
    ))
}

/// Indexing operation - select elements along specified dimensions
pub fn index_select(tensor: &Tensor, dim: usize, indices: &[usize]) -> Result<Tensor> {
    if dim >= tensor.ndim() {
        return Err(MinitensorError::index_error(dim as isize, 0, tensor.ndim()));
    }

    let dim_size = tensor.shape().dims()[dim];

    // Validate indices
    for &idx in indices {
        if idx >= dim_size {
            return Err(MinitensorError::index_error(idx as isize, 0, dim_size));
        }
    }

    if !tensor.device().is_cpu() {
        return Err(MinitensorError::invalid_operation(
            "index_select currently supports only CPU tensors",
        ));
    }

    // Compute output shape
    let mut output_shape = tensor.shape().dims().to_vec();
    output_shape[dim] = indices.len();
    let output_shape_vec = output_shape.clone();
    let output_shape_obj = Shape::new(output_shape);

    let dtype = tensor.dtype();
    let device = tensor.device();
    let requires_grad = tensor.requires_grad();

    let dims = tensor.shape().dims();
    let inner: usize = dims[dim + 1..].iter().product();
    let outer: usize = dims[..dim].iter().product();

    macro_rules! index_impl {
        ($ty:ty, $slice:ident, $from_vec:ident) => {{
            let src = tensor.data().$slice().ok_or_else(|| {
                MinitensorError::invalid_operation("Tensor data access failed for index_select")
            })?;
            let mut out = vec![<$ty>::default(); output_shape_obj.numel()];
            for o in 0..outer {
                for (i, &idx) in indices.iter().enumerate() {
                    let src_start = o * dims[dim] * inner + idx * inner;
                    let dst_start = o * output_shape_vec[dim] * inner + i * inner;
                    out[dst_start..dst_start + inner]
                        .copy_from_slice(&src[src_start..src_start + inner]);
                }
            }
            TensorData::$from_vec(out, device)
        }};
    }

    let data = match dtype {
        DataType::Float32 => index_impl!(f32, as_f32_slice, from_vec_f32),
        DataType::Float64 => index_impl!(f64, as_f64_slice, from_vec_f64),
        DataType::Int32 => index_impl!(i32, as_i32_slice, from_vec_i32),
        DataType::Int64 => index_impl!(i64, as_i64_slice, from_vec_i64),
        DataType::Bool => index_impl!(bool, as_bool_slice, from_vec_bool),
    };

    Ok(Tensor::new(
        Arc::new(data),
        output_shape_obj,
        dtype,
        device,
        requires_grad,
    ))
}

/// Slicing operation - select a contiguous range of elements
pub fn slice(tensor: &Tensor, dim: usize, start: usize, end: usize, step: usize) -> Result<Tensor> {
    if dim >= tensor.ndim() {
        return Err(MinitensorError::index_error(dim as isize, 0, tensor.ndim()));
    }

    let dim_size = tensor.shape().dims()[dim];

    if start >= dim_size || end > dim_size || start >= end {
        return Err(MinitensorError::invalid_operation(format!(
            "Invalid slice range: start={}, end={}, dim_size={}",
            start, end, dim_size
        )));
    }

    if step == 0 {
        return Err(MinitensorError::invalid_operation(
            "Slice step cannot be zero",
        ));
    }

    if !tensor.device().is_cpu() {
        return Err(MinitensorError::invalid_operation(
            "slice currently supports only CPU tensors",
        ));
    }

    // Compute output shape
    let mut output_shape = tensor.shape().dims().to_vec();
    output_shape[dim] = (end - start + step - 1) / step; // Ceiling division
    let output_shape_vec = output_shape.clone();
    let output_shape_obj = Shape::new(output_shape);

    let dtype = tensor.dtype();
    let device = tensor.device();
    let requires_grad = tensor.requires_grad();

    let dims = tensor.shape().dims();
    let inner: usize = dims[dim + 1..].iter().product();
    let outer: usize = dims[..dim].iter().product();
    let count = output_shape_vec[dim];

    macro_rules! slice_impl {
        ($ty:ty, $slice:ident, $from_vec:ident) => {{
            let src = tensor.data().$slice().ok_or_else(|| {
                MinitensorError::invalid_operation("Tensor data access failed for slice")
            })?;
            let mut out = vec![<$ty>::default(); output_shape_obj.numel()];
            for o in 0..outer {
                for i in 0..count {
                    let src_idx = start + i * step;
                    let src_start = o * dims[dim] * inner + src_idx * inner;
                    let dst_start = o * count * inner + i * inner;
                    out[dst_start..dst_start + inner]
                        .copy_from_slice(&src[src_start..src_start + inner]);
                }
            }
            TensorData::$from_vec(out, device)
        }};
    }

    let data = match dtype {
        DataType::Float32 => slice_impl!(f32, as_f32_slice, from_vec_f32),
        DataType::Float64 => slice_impl!(f64, as_f64_slice, from_vec_f64),
        DataType::Int32 => slice_impl!(i32, as_i32_slice, from_vec_i32),
        DataType::Int64 => slice_impl!(i64, as_i64_slice, from_vec_i64),
        DataType::Bool => slice_impl!(bool, as_bool_slice, from_vec_bool),
    };

    Ok(Tensor::new(
        Arc::new(data),
        output_shape_obj,
        dtype,
        device,
        requires_grad,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        device::Device,
        tensor::{DataType, TensorData},
    };

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
    fn test_reshape_basic() {
        let tensor = create_test_tensor_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3], false);

        let reshaped = reshape(&tensor, Shape::new(vec![3, 2])).unwrap();

        assert_eq!(reshaped.shape().dims(), &[3, 2]);
        assert_eq!(reshaped.numel(), 6);

        let data = reshaped.data().as_f32_slice().unwrap();
        assert_eq!(data, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_reshape_invalid_size() {
        let tensor = create_test_tensor_f32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], false);

        let result = reshape(&tensor, Shape::new(vec![2, 3]));
        assert!(result.is_err());
    }

    #[test]
    fn test_squeeze_specific_dim() {
        let tensor = create_test_tensor_f32(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4, 1], false);

        let squeezed = squeeze(&tensor, Some(0)).unwrap();
        assert_eq!(squeezed.shape().dims(), &[4, 1]);

        let squeezed2 = squeeze(&squeezed, Some(1)).unwrap();
        assert_eq!(squeezed2.shape().dims(), &[4]);
    }

    #[test]
    fn test_squeeze_all() {
        let tensor = create_test_tensor_f32(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4, 1], false);

        let squeezed = squeeze(&tensor, None).unwrap();
        assert_eq!(squeezed.shape().dims(), &[4]);
    }

    #[test]
    fn test_squeeze_invalid_dim() {
        let tensor = create_test_tensor_f32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], false);

        let result = squeeze(&tensor, Some(0));
        assert!(result.is_err()); // Cannot squeeze dimension of size 2
    }

    #[test]
    fn test_unsqueeze() {
        let tensor = create_test_tensor_f32(vec![1.0, 2.0, 3.0, 4.0], vec![4], false);

        let unsqueezed = unsqueeze(&tensor, 0).unwrap();
        assert_eq!(unsqueezed.shape().dims(), &[1, 4]);

        let unsqueezed2 = unsqueeze(&tensor, 1).unwrap();
        assert_eq!(unsqueezed2.shape().dims(), &[4, 1]);
    }

    #[test]
    fn test_gradient_tracking() {
        let tensor = create_test_tensor_f32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], true);

        let reshaped = reshape(&tensor, Shape::new(vec![4])).unwrap();

        assert!(reshaped.requires_grad());
        assert!(reshaped.grad_fn().is_some());
    }

    #[test]
    fn test_concatenate_validation() {
        let tensor1 = create_test_tensor_f32(vec![1.0, 2.0], vec![2], false);
        let tensor2 = create_test_tensor_f32(vec![3.0, 4.0], vec![2], false);

        let result = concatenate(&[&tensor1, &tensor2], 0).unwrap();
        assert_eq!(result.shape().dims(), &[4]);
        let data = result.data().as_f32_slice().unwrap();
        assert_eq!(data, &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_index_select_validation() {
        let tensor = create_test_tensor_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3], false);

        let result = index_select(&tensor, 1, &[0, 2]).unwrap();
        assert_eq!(result.shape().dims(), &[2, 2]);
        let data = result.data().as_f32_slice().unwrap();
        assert_eq!(data, &[1.0, 3.0, 4.0, 6.0]);
    }

    #[test]
    fn test_slice_validation() {
        let tensor = create_test_tensor_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3], false);

        let result = slice(&tensor, 1, 0, 2, 1).unwrap();
        assert_eq!(result.shape().dims(), &[2, 2]);
        let data = result.data().as_f32_slice().unwrap();
        assert_eq!(data, &[1.0, 2.0, 4.0, 5.0]);
    }
}
