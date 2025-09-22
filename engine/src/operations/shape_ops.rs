// Copyright (c) 2025 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

use crate::{
    autograd::{RepeatInterleaveBackward, ReshapeBackward, add_to_graph},
    error::{MinitensorError, Result},
    tensor::{DataType, Shape, Tensor, TensorData},
};
use rayon::prelude::*;
use std::sync::Arc;

fn normalize_dim(dim: isize, ndim: usize) -> Result<usize> {
    let dim = if dim < 0 { dim + ndim as isize } else { dim };
    if dim < 0 || dim >= ndim as isize {
        Err(MinitensorError::index_error(dim, 0, ndim))
    } else {
        Ok(dim as usize)
    }
}

/// Reshape operation with gradient support
pub fn reshape(tensor: &Tensor, new_shape: Shape) -> Result<Tensor> {
    // Check if the total number of elements matches
    if tensor.numel() != new_shape.numel() {
        return Err(MinitensorError::shape_mismatch(
            vec![tensor.numel()],
            vec![new_shape.numel()],
        ));
    }

    // Use the tensor's built-in view method for reshaping and refresh metadata
    let mut reshaped = tensor.view(new_shape.clone())?;
    reshaped.refresh_autograd_metadata();

    // Set up gradient function if needed
    if reshaped.requires_grad() {
        let grad_fn = Arc::new(ReshapeBackward {
            input_shape: tensor.shape().dims().to_vec(),
            input_id: tensor.id(),
        });

        reshaped.set_grad_fn(Some(grad_fn.clone()));

        // Add to computation graph
        add_to_graph(&reshaped, Some(grad_fn))?;

        Ok(reshaped)
    } else {
        Ok(reshaped)
    }
}

/// This wrapper performs validation and inference for a single ``-1``
/// dimension before delegating to [`reshape`].
pub fn reshape_with_inference(tensor: &Tensor, dims: Vec<isize>) -> Result<Tensor> {
    let neg_count = dims.iter().filter(|&&d| d == -1).count();
    if neg_count > 1 {
        return Err(MinitensorError::invalid_operation(
            "can only specify one -1 dimension in reshape".to_string(),
        ));
    }

    let mut out_dims = Vec::with_capacity(dims.len());
    if neg_count == 1 {
        let mut known: usize = 1;
        for &dim in &dims {
            if dim == -1 {
                continue;
            }
            if dim < -1 {
                return Err(MinitensorError::invalid_operation(
                    "invalid negative dimension".to_string(),
                ));
            }
            known *= dim as usize;
        }
        if known == 0 {
            return Err(MinitensorError::invalid_operation(
                "cannot reshape tensor with -1 and 0 dimensions".to_string(),
            ));
        }
        if tensor.numel() % known != 0 {
            return Err(MinitensorError::invalid_operation(
                "cannot infer reshape dimension".to_string(),
            ));
        }
        let inferred = tensor.numel() / known;
        for &dim in &dims {
            if dim == -1 {
                out_dims.push(inferred);
            } else {
                out_dims.push(dim as usize);
            }
        }
    } else {
        for &dim in &dims {
            if dim < 0 {
                return Err(MinitensorError::invalid_operation(
                    "negative dimensions are not allowed".to_string(),
                ));
            }
            out_dims.push(dim as usize);
        }
    }

    reshape(tensor, Shape::new(out_dims))
}

/// Squeeze operation - remove dimensions of size 1
pub fn squeeze(tensor: &Tensor, dim: Option<isize>) -> Result<Tensor> {
    match dim {
        Some(d) => tensor.squeeze_dim(d),
        None => tensor.squeeze(),
    }
}

/// Unsqueeze operation - add a dimension of size 1
pub fn unsqueeze(tensor: &Tensor, dim: isize) -> Result<Tensor> {
    tensor.unsqueeze(dim)
}

/// Permute tensor dimensions according to `dims`
pub fn permute(tensor: &Tensor, dims: Vec<isize>) -> Result<Tensor> {
    let ndim = tensor.ndim();

    // Validate number of dimensions
    if dims.len() != ndim {
        return Err(MinitensorError::invalid_operation(
            "dims must match number of dimensions".to_string(),
        ));
    }

    // Normalise negative dimensions and validate range
    let mut normalized = Vec::with_capacity(ndim);
    for &d in &dims {
        let d = if d < 0 { d + ndim as isize } else { d };
        if d < 0 || d >= ndim as isize {
            return Err(MinitensorError::index_error(d, 0, ndim));
        }
        normalized.push(d as usize);
    }
    // Check that dims form a proper permutation
    let mut sorted = normalized.clone();
    sorted.sort_unstable();
    if sorted != (0..ndim).collect::<Vec<_>>() {
        return Err(MinitensorError::invalid_operation(
            "dims must be a permutation of dimensions".to_string(),
        ));
    }

    // Apply sequence of transposes to achieve the permutation
    let mut result = tensor.clone();
    let mut current: Vec<usize> = (0..ndim).collect();
    for i in 0..ndim {
        let target = normalized[i];
        let j = current.iter().position(|&x| x == target).unwrap();
        if i != j {
            result = result.transpose(i as isize, j as isize)?;
            current.swap(i, j);
        }
    }

    Ok(result)
}

/// Move tensor dimensions to new positions, keeping relative order of other dims
pub fn movedim(tensor: &Tensor, source: &[isize], destination: &[isize]) -> Result<Tensor> {
    let ndim = tensor.ndim();

    if source.len() != destination.len() {
        return Err(MinitensorError::invalid_operation(
            "movedim: source and destination must have the same length".to_string(),
        ));
    }

    let mut src: Vec<usize> = Vec::with_capacity(source.len());
    let mut dst: Vec<usize> = Vec::with_capacity(destination.len());

    for &s in source {
        let s = if s < 0 { s + ndim as isize } else { s };
        if s < 0 || s >= ndim as isize {
            return Err(MinitensorError::index_error(s, 0, ndim));
        }
        src.push(s as usize);
    }
    for &d in destination {
        let d = if d < 0 { d + ndim as isize } else { d };
        if d < 0 || d >= ndim as isize {
            return Err(MinitensorError::index_error(d, 0, ndim));
        }
        dst.push(d as usize);
    }

    // Check for duplicate dimensions
    let mut src_sorted = src.clone();
    src_sorted.sort_unstable();
    src_sorted.dedup();
    if src_sorted.len() != src.len() {
        return Err(MinitensorError::invalid_operation(
            "movedim: duplicate dimensions in source".to_string(),
        ));
    }
    let mut dst_sorted = dst.clone();
    dst_sorted.sort_unstable();
    dst_sorted.dedup();
    if dst_sorted.len() != dst.len() {
        return Err(MinitensorError::invalid_operation(
            "movedim: duplicate dimensions in destination".to_string(),
        ));
    }

    // Build permutation order
    let mut order: Vec<usize> = (0..ndim).filter(|i| !src.contains(i)).collect();
    let mut pairs: Vec<(usize, usize)> = dst.iter().copied().zip(src.iter().copied()).collect();
    pairs.sort_by_key(|&(d, _)| d);
    for (d, s) in pairs {
        order.insert(d, s);
    }
    let order_isize: Vec<isize> = order.into_iter().map(|v| v as isize).collect();
    permute(tensor, order_isize)
}

/// Concatenate tensors along a specified dimension
pub fn concatenate(tensors: &[&Tensor], dim: isize) -> Result<Tensor> {
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
    let dim = normalize_dim(dim, first_tensor.ndim())?;

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
    let _outer: usize = dims[..dim].iter().product();

    macro_rules! concat_impl {
        ($ty:ty, $slice:ident, $from_vec:ident) => {{
            let mut out = vec![<$ty>::default(); output_shape_obj.numel()];
            out.par_chunks_mut(output_shape_vec[dim] * inner)
                .enumerate()
                .for_each(|(o, out_chunk)| {
                    let mut dst_offset = 0;
                    for t in tensors {
                        let t_dims = t.shape().dims();
                        let src_start = o * t_dims[dim] * inner;
                        let src_len = t_dims[dim] * inner;
                        let src = t
                            .data()
                            .$slice()
                            .ok_or_else(|| {
                                MinitensorError::invalid_operation(
                                    "Tensor data access failed for concatenate",
                                )
                            })
                            .unwrap();
                        out_chunk[dst_offset..dst_offset + src_len]
                            .copy_from_slice(&src[src_start..src_start + src_len]);
                        dst_offset += src_len;
                    }
                });
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

/// Repeat `tensor` according to `repeats` along each dimension.
pub fn repeat(tensor: &Tensor, repeats: &[usize]) -> Result<Tensor> {
    if repeats.len() < tensor.ndim() {
        return Err(MinitensorError::invalid_operation(
            "number of dimensions of repeat dims can not be smaller than number of dimensions of tensor",
        ));
    }

    let mut result = tensor.clone();

    if repeats.len() > result.ndim() {
        let mut new_shape = vec![1; repeats.len() - result.ndim()];
        new_shape.extend_from_slice(result.shape().dims());
        result = result.reshape(Shape::new(new_shape))?;
    }

    if repeats.iter().any(|&r| r == 0) {
        let mut out_shape = result.shape().dims().to_vec();
        for (dim, &rep) in repeats.iter().enumerate() {
            out_shape[dim] *= rep;
        }
        let shape = Shape::new(out_shape);
        let dtype = result.dtype();
        let device = result.device();
        let data = match dtype {
            DataType::Float32 => TensorData::from_vec_f32(vec![], device),
            DataType::Float64 => TensorData::from_vec_f64(vec![], device),
            DataType::Int32 => TensorData::from_vec_i32(vec![], device),
            DataType::Int64 => TensorData::from_vec_i64(vec![], device),
            DataType::Bool => TensorData::from_vec_bool(vec![], device),
        };
        return Ok(Tensor::new(
            Arc::new(data),
            shape,
            dtype,
            device,
            result.requires_grad(),
        ));
    }

    for (dim, &rep) in repeats.iter().enumerate() {
        if rep == 1 {
            continue;
        }
        let clones: Vec<Tensor> = (0..rep).map(|_| result.clone()).collect();
        let refs: Vec<&Tensor> = clones.iter().collect();
        result = concatenate(&refs, dim as isize)?;
    }

    Ok(result)
}

/// Indexing operation - select elements along specified dimensions
pub fn index_select(tensor: &Tensor, dim: isize, indices: &[usize]) -> Result<Tensor> {
    let dim = normalize_dim(dim, tensor.ndim())?;

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
    let _outer: usize = dims[..dim].iter().product();

    macro_rules! index_impl {
        ($ty:ty, $slice:ident, $from_vec:ident) => {{
            let src = tensor.data().$slice().ok_or_else(|| {
                MinitensorError::invalid_operation("Tensor data access failed for index_select")
            })?;
            let mut out = vec![<$ty>::default(); output_shape_obj.numel()];
            out.par_chunks_mut(output_shape_vec[dim] * inner)
                .enumerate()
                .for_each(|(o, out_chunk)| {
                    for (i, &idx) in indices.iter().enumerate() {
                        let src_start = o * dims[dim] * inner + idx * inner;
                        let dst_start = i * inner;
                        out_chunk[dst_start..dst_start + inner]
                            .copy_from_slice(&src[src_start..src_start + inner]);
                    }
                });
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

/// Gather operation - collect elements along a dimension using an index tensor
pub fn gather(tensor: &Tensor, dim: isize, index: &Tensor) -> Result<Tensor> {
    let dim = normalize_dim(dim, tensor.ndim())?;

    if index.ndim() != tensor.ndim() {
        return Err(MinitensorError::invalid_operation(
            "gather index tensor must have the same number of dimensions as input",
        ));
    }

    if index.dtype() != DataType::Int64 {
        return Err(MinitensorError::invalid_operation(
            "gather indices must be int64",
        ));
    }

    let input_dims = tensor.shape().dims();
    let index_dims = index.shape().dims();

    // Validate shapes except at gather dimension
    for (i, (&idx_d, &in_d)) in index_dims.iter().zip(input_dims.iter()).enumerate() {
        if i != dim && idx_d != in_d {
            return Err(MinitensorError::shape_mismatch(
                input_dims.to_vec(),
                index_dims.to_vec(),
            ));
        }
    }

    let dim_size = input_dims[dim];

    // Validate indices
    let idx_slice = index
        .data()
        .as_i64_slice()
        .ok_or_else(|| MinitensorError::invalid_operation("gather indices must be int64"))?;
    for &v in idx_slice {
        if v < 0 || v as usize >= dim_size {
            return Err(MinitensorError::index_error(v as isize, 0, dim_size));
        }
    }

    if !tensor.device().is_cpu() {
        return Err(MinitensorError::invalid_operation(
            "gather currently supports only CPU tensors",
        ));
    }

    let outer: usize = input_dims[..dim].iter().product();
    let inner: usize = input_dims[dim + 1..].iter().product();
    let idx_dim = index_dims[dim];

    let dtype = tensor.dtype();
    let device = tensor.device();
    let requires_grad = tensor.requires_grad();
    let output_shape_obj = Shape::new(index_dims.to_vec());

    macro_rules! gather_impl {
        ($ty:ty, $slice:ident, $from_vec:ident) => {{
            let src = tensor.data().$slice().ok_or_else(|| {
                MinitensorError::invalid_operation("Tensor data access failed for gather")
            })?;
            let idx = idx_slice;
            let mut out = vec![<$ty>::default(); output_shape_obj.numel()];
            for o in 0..outer {
                for i in 0..idx_dim {
                    for j in 0..inner {
                        let idx_pos = o * idx_dim * inner + i * inner + j;
                        let gather_idx = idx[idx_pos] as usize;
                        let src_pos = o * dim_size * inner + gather_idx * inner + j;
                        out[idx_pos] = src[src_pos];
                    }
                }
            }
            TensorData::$from_vec(out, device)
        }};
    }

    let data = match dtype {
        DataType::Float32 => gather_impl!(f32, as_f32_slice, from_vec_f32),
        DataType::Float64 => gather_impl!(f64, as_f64_slice, from_vec_f64),
        DataType::Int32 => gather_impl!(i32, as_i32_slice, from_vec_i32),
        DataType::Int64 => gather_impl!(i64, as_i64_slice, from_vec_i64),
        DataType::Bool => gather_impl!(bool, as_bool_slice, from_vec_bool),
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
pub fn slice(tensor: &Tensor, dim: isize, start: usize, end: usize, step: usize) -> Result<Tensor> {
    let dim = normalize_dim(dim, tensor.ndim())?;

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
    output_shape[dim] = (end - start).div_ceil(step);
    let output_shape_vec = output_shape.clone();
    let output_shape_obj = Shape::new(output_shape);

    let dtype = tensor.dtype();
    let device = tensor.device();
    let requires_grad = tensor.requires_grad();

    let dims = tensor.shape().dims();
    let inner: usize = dims[dim + 1..].iter().product();
    let _outer: usize = dims[..dim].iter().product();
    let count = output_shape_vec[dim];

    macro_rules! slice_impl {
        ($ty:ty, $slice:ident, $from_vec:ident) => {{
            let src = tensor.data().$slice().ok_or_else(|| {
                MinitensorError::invalid_operation("Tensor data access failed for slice")
            })?;
            let mut out = vec![<$ty>::default(); output_shape_obj.numel()];
            out.par_chunks_mut(count * inner)
                .enumerate()
                .for_each(|(o, out_chunk)| {
                    for i in 0..count {
                        let src_idx = start + i * step;
                        let src_start = o * dims[dim] * inner + src_idx * inner;
                        let dst_start = i * inner;
                        out_chunk[dst_start..dst_start + inner]
                            .copy_from_slice(&src[src_start..src_start + inner]);
                    }
                });
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

/// Narrow tensor along a dimension starting at `start` for `length` elements
pub fn narrow(tensor: &Tensor, dim: isize, start: usize, length: usize) -> Result<Tensor> {
    let dim = normalize_dim(dim, tensor.ndim())?;
    let dim_size = tensor.shape().dims()[dim];

    if start > dim_size {
        return Err(MinitensorError::index_error(start as isize, 0, dim_size));
    }
    if start + length > dim_size {
        return Err(MinitensorError::index_error(
            (start + length) as isize,
            0,
            dim_size,
        ));
    }

    if length == 0 {
        let mut out_shape = tensor.shape().dims().to_vec();
        out_shape[dim] = 0;
        return Ok(Tensor::zeros(
            Shape::new(out_shape),
            tensor.dtype(),
            tensor.device(),
            tensor.requires_grad(),
        ));
    }

    slice(tensor, dim as isize, start, start + length, 1)
}

/// Flip tensor elements along specified dimensions.
pub fn flip(tensor: &Tensor, dims: &[isize]) -> Result<Tensor> {
    let ndim = tensor.ndim();
    let mut normalized = Vec::with_capacity(dims.len());
    for &d in dims {
        let dim = normalize_dim(d, ndim)?;
        if normalized.contains(&dim) {
            return Err(MinitensorError::invalid_operation(
                "dims must be unique".to_string(),
            ));
        }
        normalized.push(dim);
    }

    let mut result = tensor.clone();
    for &dim in &normalized {
        let size = result.shape().dims()[dim];
        let indices: Vec<usize> = (0..size).rev().collect();
        result = index_select(&result, dim as isize, &indices)?;
    }

    Ok(result)
}

/// Roll tensor elements along specified dimensions with wrap-around
pub fn roll(tensor: &Tensor, shifts: &[isize], dims: Option<&[isize]>) -> Result<Tensor> {
    if let Some(dims) = dims {
        if shifts.len() != dims.len() {
            return Err(MinitensorError::invalid_operation(
                "shifts and dims must have the same length".to_string(),
            ));
        }
        let mut result = tensor.clone();
        for (&shift, &dim) in shifts.iter().zip(dims.iter()) {
            let dim = normalize_dim(dim, result.ndim())?;
            let size = result.shape().dims()[dim] as isize;
            if size == 0 {
                continue;
            }
            let k = ((shift % size) + size) % size;
            if k == 0 {
                continue;
            }
            let split_point = (size - k) as usize;
            let first = slice(&result, dim as isize, 0, split_point, 1)?;
            let second = slice(&result, dim as isize, split_point, size as usize, 1)?;
            result = concatenate(&[&second, &first], dim as isize)?;
        }
        Ok(result)
    } else {
        if shifts.len() != 1 {
            return Err(MinitensorError::invalid_operation(
                "shifts must contain a single value when dims is None".to_string(),
            ));
        }
        let shift = shifts[0];
        let flat = tensor.flatten_all()?;
        let size = flat.shape().dims()[0] as isize;
        if size == 0 {
            return flat.reshape(tensor.shape().clone());
        }
        let k = ((shift % size) + size) % size;
        if k == 0 {
            return flat.reshape(tensor.shape().clone());
        }
        let split_point = (size - k) as usize;
        let first = slice(&flat, 0, 0, split_point, 1)?;
        let second = slice(&flat, 0, split_point, size as usize, 1)?;
        let rolled = concatenate(&[&second, &first], 0)?;
        rolled.reshape(tensor.shape().clone())
    }
}

/// Specification of repeat counts accepted by [`repeat_interleave`].
#[derive(Clone, Copy)]
pub enum RepeatInterleaveSpec<'a> {
    /// A single repeat value applied to every element along ``dim``.
    Scalar(usize),
    /// Explicit repeat counts provided as a slice.
    Slice(&'a [usize]),
    /// Repeat counts provided as a tensor (must contain integer data).
    Tensor(&'a Tensor),
}

fn collect_repeats_from_tensor(tensor: &Tensor, dim_size: usize) -> Result<Vec<usize>> {
    if !tensor.device().is_cpu() {
        return Err(MinitensorError::invalid_operation(
            "repeat_interleave: repeats tensor must reside on CPU".to_string(),
        ));
    }

    if tensor.numel() != dim_size {
        return Err(MinitensorError::invalid_operation(
            "repeat_interleave: repeats tensor must have the same number of elements as the selected dimension"
                .to_string(),
        ));
    }

    match tensor.dtype() {
        DataType::Int32 => {
            let slice = tensor.data().as_i32_slice().ok_or_else(|| {
                MinitensorError::invalid_operation(
                    "repeat_interleave: repeats tensor must be contiguous".to_string(),
                )
            })?;
            let mut out = Vec::with_capacity(slice.len());
            for &value in slice {
                if value < 0 {
                    return Err(MinitensorError::invalid_operation(
                        "repeat_interleave: repeats must be non-negative".to_string(),
                    ));
                }
                out.push(value as usize);
            }
            Ok(out)
        }
        DataType::Int64 => {
            let slice = tensor.data().as_i64_slice().ok_or_else(|| {
                MinitensorError::invalid_operation(
                    "repeat_interleave: repeats tensor must be contiguous".to_string(),
                )
            })?;
            let mut out = Vec::with_capacity(slice.len());
            for &value in slice {
                if value < 0 {
                    return Err(MinitensorError::invalid_operation(
                        "repeat_interleave: repeats must be non-negative".to_string(),
                    ));
                }
                out.push(value as usize);
            }
            Ok(out)
        }
        other => Err(MinitensorError::type_mismatch(
            "integral tensor",
            format!("{:?}", other),
        )),
    }
}

fn expand_repeats(spec: RepeatInterleaveSpec<'_>, dim_size: usize) -> Result<Vec<usize>> {
    match spec {
        RepeatInterleaveSpec::Scalar(value) => Ok(vec![value; dim_size]),
        RepeatInterleaveSpec::Slice(values) => {
            if values.len() == dim_size {
                Ok(values.to_vec())
            } else if values.len() == 1 {
                if dim_size == 0 {
                    Ok(Vec::new())
                } else {
                    Ok(vec![values[0]; dim_size])
                }
            } else if values.is_empty() && dim_size == 0 {
                Ok(Vec::new())
            } else {
                Err(MinitensorError::invalid_operation(
                    "repeat_interleave: repeats must be a single value or match tensor size along dim"
                        .to_string(),
                ))
            }
        }
        RepeatInterleaveSpec::Tensor(tensor) => collect_repeats_from_tensor(tensor, dim_size),
    }
}

fn build_empty_repeat_result(tensor: &Tensor, dim: usize, target: usize) -> Result<Tensor> {
    let mut out_shape = tensor.shape().dims().to_vec();
    out_shape[dim] = target;
    let shape = Shape::new(out_shape);
    let dtype = tensor.dtype();
    let device = tensor.device();
    let data = TensorData::zeros_on_device(shape.numel(), dtype, device);
    Ok(Tensor::new(
        Arc::new(data),
        shape,
        dtype,
        device,
        tensor.requires_grad(),
    ))
}

/// Repeat elements of ``tensor`` according to ``repeats`` along ``dim``.
pub fn repeat_interleave(
    tensor: &Tensor,
    repeats: RepeatInterleaveSpec<'_>,
    dim: Option<isize>,
    output_size: Option<usize>,
) -> Result<Tensor> {
    if dim.is_none() {
        let flat = tensor.flatten_all()?;
        return repeat_interleave(&flat, repeats, Some(0), output_size);
    }

    if !tensor.device().is_cpu() {
        return Err(MinitensorError::invalid_operation(
            "repeat_interleave currently supports only CPU tensors".to_string(),
        ));
    }

    let dim = normalize_dim(dim.unwrap(), tensor.ndim())?;
    let dims = tensor.shape().dims();
    let dim_size = dims[dim];
    let reps = expand_repeats(repeats, dim_size)?;
    let repeats_for_backward = reps.clone();
    let total_repeats: usize = reps.iter().sum();
    let input_shape_vec = dims.to_vec();

    if let Some(expected) = output_size {
        if expected != total_repeats {
            return Err(MinitensorError::invalid_argument(format!(
                "repeat_interleave: output_size ({expected}) must equal sum of repeats ({total_repeats})"
            )));
        }
    }

    let dtype = tensor.dtype();
    let device = tensor.device();
    let requires_grad = tensor.requires_grad();

    let target_dim = output_size.unwrap_or(total_repeats);
    let mut output_shape = input_shape_vec.clone();
    output_shape[dim] = target_dim;
    let output_shape_obj = Shape::new(output_shape.clone());
    let output_numel = output_shape_obj.numel();

    let inner: usize = dims[dim + 1..].iter().product();
    let outer: usize = if dim == 0 {
        1
    } else {
        dims[..dim].iter().product()
    };

    if target_dim == 0 || output_numel == 0 || inner == 0 || outer == 0 {
        let mut result = build_empty_repeat_result(tensor, dim, target_dim)?;
        if requires_grad {
            let grad_fn = Arc::new(RepeatInterleaveBackward {
                input_shape: input_shape_vec.clone(),
                repeats: repeats_for_backward.clone(),
                input_id: tensor.id(),
                dim,
            });
            result.set_grad_fn(Some(grad_fn.clone()));
            add_to_graph(&result, Some(grad_fn))?;
        }
        return Ok(result);
    }

    macro_rules! repeat_impl {
        ($ty:ty, $slice:ident, $from_vec:ident) => {{
            let src = tensor.data().$slice().ok_or_else(|| {
                MinitensorError::invalid_operation(
                    "repeat_interleave: tensor data access failed".to_string(),
                )
            })?;
            let mut out = vec![<$ty>::default(); output_numel];
            out.par_chunks_mut(target_dim * inner).enumerate().for_each(
                |(outer_idx, out_chunk)| {
                    let mut dst_offset = 0;
                    let base = outer_idx * dim_size * inner;
                    for (i, &rep) in reps.iter().enumerate() {
                        if rep == 0 {
                            continue;
                        }
                        let src_start = base + i * inner;
                        let src_slice = &src[src_start..src_start + inner];
                        for _ in 0..rep {
                            let end = dst_offset + inner;
                            out_chunk[dst_offset..end].copy_from_slice(src_slice);
                            dst_offset = end;
                        }
                    }
                },
            );
            TensorData::$from_vec(out, device)
        }};
    }

    let data = match dtype {
        DataType::Float32 => repeat_impl!(f32, as_f32_slice, from_vec_f32),
        DataType::Float64 => repeat_impl!(f64, as_f64_slice, from_vec_f64),
        DataType::Int32 => repeat_impl!(i32, as_i32_slice, from_vec_i32),
        DataType::Int64 => repeat_impl!(i64, as_i64_slice, from_vec_i64),
        DataType::Bool => repeat_impl!(bool, as_bool_slice, from_vec_bool),
    };

    let mut result = Tensor::new(
        Arc::new(data),
        output_shape_obj,
        dtype,
        device,
        requires_grad,
    );

    if requires_grad {
        let grad_fn = Arc::new(RepeatInterleaveBackward {
            input_shape: input_shape_vec,
            repeats: repeats_for_backward,
            input_id: tensor.id(),
            dim,
        });
        result.set_grad_fn(Some(grad_fn.clone()));
        add_to_graph(&result, Some(grad_fn))?;
    }

    Ok(result)
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
    fn test_reshape_infer_dim() {
        let tensor = create_test_tensor_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![6], false);
        let reshaped = reshape_with_inference(&tensor, vec![2, -1]).unwrap();
        assert_eq!(reshaped.shape().dims(), &[2, 3]);
    }

    #[test]
    fn test_reshape_multiple_negative_one_error() {
        let tensor = create_test_tensor_f32(vec![1.0, 2.0, 3.0, 4.0], vec![4], false);
        let result = reshape_with_inference(&tensor, vec![-1, -1]);
        assert!(result.is_err());
    }

    #[test]
    fn test_reshape_infer_mismatch_error() {
        let tensor = create_test_tensor_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0], vec![5], false);
        let result = reshape_with_inference(&tensor, vec![4, -1]);
        assert!(result.is_err());
    }

    #[test]
    fn test_reshape_zero_dim_with_inference_error() {
        let tensor = create_test_tensor_f32(vec![], vec![0], false);
        let result = reshape_with_inference(&tensor, vec![-1, 0]);
        assert!(result.is_err());
    }

    #[test]
    fn test_squeeze_specific_dim() {
        let tensor = create_test_tensor_f32(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4, 1], false);

        let s0 = squeeze(&tensor, Some(0)).unwrap();
        assert_eq!(s0.shape().dims(), &[4, 1]);

        let s1 = squeeze(&s0, Some(1)).unwrap();
        assert_eq!(s1.shape().dims(), &[4]);

        let s_neg = squeeze(&tensor, Some(-1)).unwrap();
        assert_eq!(s_neg.shape().dims(), &[1, 4]);
    }

    #[test]
    fn test_squeeze_all() {
        let tensor = create_test_tensor_f32(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4, 1], false);

        let squeezed = squeeze(&tensor, None).unwrap();
        assert_eq!(squeezed.shape().dims(), &[4]);

        let scalar = create_test_tensor_f32(vec![1.0], vec![1, 1], false);
        let s = squeeze(&scalar, None).unwrap();
        assert!(s.shape().dims().is_empty());
    }

    #[test]
    fn test_squeeze_out_of_range() {
        let tensor = create_test_tensor_f32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], false);

        assert!(squeeze(&tensor, Some(2)).is_err());
        assert!(squeeze(&tensor, Some(-3)).is_err());
    }

    #[test]
    fn test_unsqueeze() {
        let tensor = create_test_tensor_f32(vec![1.0, 2.0, 3.0, 4.0], vec![4], false);

        let u0 = unsqueeze(&tensor, 0).unwrap();
        assert_eq!(u0.shape().dims(), &[1, 4]);

        let u1 = unsqueeze(&tensor, 1).unwrap();
        assert_eq!(u1.shape().dims(), &[4, 1]);

        let u_neg = unsqueeze(&tensor, -1).unwrap();
        assert_eq!(u_neg.shape().dims(), &[4, 1]);
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

    #[test]
    fn test_repeat_basic() {
        let tensor = create_test_tensor_f32(vec![1.0, 2.0], vec![2], false);
        let repeated = repeat(&tensor, &[3]).unwrap();
        assert_eq!(repeated.shape().dims(), &[6]);
        let data = repeated.data().as_f32_slice().unwrap();
        assert_eq!(data, &[1.0, 2.0, 1.0, 2.0, 1.0, 2.0]);
    }

    #[test]
    fn test_repeat_dim_mismatch_error() {
        let tensor = create_test_tensor_f32(vec![1.0, 2.0], vec![2], false);
        assert!(repeat(&tensor, &[]).is_err());
    }
}
