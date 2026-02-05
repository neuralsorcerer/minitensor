// Copyright (c) 2026 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

use crate::{
    autograd::{RepeatInterleaveBackward, ReshapeBackward, add_to_graph},
    device::Device,
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

fn empty_tensor(shape: Shape, dtype: DataType, device: Device, requires_grad: bool) -> Tensor {
    Tensor::new(
        Arc::new(TensorData::zeros_on_device(0, dtype, device)),
        shape,
        dtype,
        device,
        requires_grad,
    )
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

    let mut src_seen = vec![false; ndim];
    let mut dst_seen = vec![false; ndim];
    let mut pairs: Vec<(usize, usize)> = Vec::with_capacity(source.len());

    for (&s, &d) in source.iter().zip(destination.iter()) {
        let s = if s < 0 { s + ndim as isize } else { s };
        if s < 0 || s >= ndim as isize {
            return Err(MinitensorError::index_error(s, 0, ndim));
        }
        let s = s as usize;
        if src_seen[s] {
            return Err(MinitensorError::invalid_operation(
                "movedim: duplicate dimensions in source".to_string(),
            ));
        }
        src_seen[s] = true;
        let d = if d < 0 { d + ndim as isize } else { d };
        if d < 0 || d >= ndim as isize {
            return Err(MinitensorError::index_error(d, 0, ndim));
        }
        let d = d as usize;
        if dst_seen[d] {
            return Err(MinitensorError::invalid_operation(
                "movedim: duplicate dimensions in destination".to_string(),
            ));
        }
        dst_seen[d] = true;
        pairs.push((d, s));
    }

    // Build permutation order
    let mut order: Vec<usize> = (0..ndim).filter(|&i| !src_seen[i]).collect();
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
    let output_shape_obj = Shape::new(output_shape);

    let dtype = first_tensor.dtype();
    let device = first_tensor.device();
    let requires_grad = tensors.iter().any(|t| t.requires_grad());

    let dims = first_tensor.shape().dims();
    let inner: usize = dims[dim + 1..].iter().product();
    let _outer: usize = dims[..dim].iter().product();

    if output_shape_obj.numel() == 0 {
        let data = TensorData::zeros_on_device(0, dtype, device);
        return Ok(Tensor::new(
            Arc::new(data),
            output_shape_obj,
            dtype,
            device,
            requires_grad,
        ));
    }

    macro_rules! concat_impl {
        ($ty:ty, $slice:ident, $from_vec:ident) => {{
            let mut sources: Vec<&[$ty]> = Vec::with_capacity(tensors.len());
            let mut dim_sizes: Vec<usize> = Vec::with_capacity(tensors.len());
            for t in tensors {
                let src = t.data().$slice().ok_or_else(|| {
                    MinitensorError::invalid_operation("Tensor data access failed for concatenate")
                })?;
                sources.push(src);
                dim_sizes.push(t.shape().dims()[dim]);
            }
            let src_strides: Vec<usize> = dim_sizes.iter().map(|&d| d * inner).collect();

            let mut out = vec![<$ty>::default(); output_shape_obj.numel()];
            let chunk_size = output_shape_obj.dims()[dim] * inner;
            out.par_chunks_mut(chunk_size)
                .enumerate()
                .for_each(|(o, out_chunk)| {
                    let mut dst_offset = 0;
                    for (src, &src_stride) in sources.iter().zip(src_strides.iter()) {
                        let src_start = o * src_stride;
                        let src_len = src_stride;
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

        return Ok(empty_tensor(
            Shape::new(out_shape),
            result.dtype(),
            result.device(),
            result.requires_grad(),
        ));
    }

    for (dim, &rep) in repeats.iter().enumerate() {
        if rep == 1 {
            continue;
        }
        let dims = result.shape().dims().to_vec();
        let dim_size = dims[dim];
        let inner: usize = dims[dim + 1..].iter().product();
        let mut output_shape = dims.clone();
        output_shape[dim] = dim_size * rep;
        let output_shape_obj = Shape::new(output_shape);
        let output_numel = output_shape_obj.numel();

        let dtype = result.dtype();
        let device = result.device();
        let requires_grad = result.requires_grad();

        if output_numel == 0 {
            result = empty_tensor(output_shape_obj, dtype, device, requires_grad);
            continue;
        }

        macro_rules! repeat_impl {
            ($ty:ty, $slice:ident, $from_vec:ident) => {{
                let src = result.data().$slice().ok_or_else(|| {
                    MinitensorError::invalid_operation("Tensor data access failed for repeat")
                })?;
                let mut out = vec![<$ty>::default(); output_numel];
                let chunk_size = dim_size * rep * inner;
                let src_chunk_size = dim_size * inner;
                out.par_chunks_mut(chunk_size)
                    .enumerate()
                    .for_each(|(o, out_chunk)| {
                        let src_start = o * src_chunk_size;
                        let src_chunk = &src[src_start..src_start + src_chunk_size];
                        for r in 0..rep {
                            let dst_start = r * src_chunk_size;
                            out_chunk[dst_start..dst_start + src_chunk_size]
                                .copy_from_slice(src_chunk);
                        }
                    });
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

        result = Tensor::new(
            Arc::new(data),
            output_shape_obj,
            dtype,
            device,
            requires_grad,
        );
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

    if output_shape_obj.numel() == 0 {
        return Ok(empty_tensor(output_shape_obj, dtype, device, requires_grad));
    }

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

    let inner: usize = input_dims[dim + 1..].iter().product();
    let idx_dim = index_dims[dim];

    let dtype = tensor.dtype();
    let device = tensor.device();
    let requires_grad = tensor.requires_grad();
    let output_shape_obj = Shape::new(index_dims.to_vec());
    let output_numel = idx_slice.len();

    if output_numel == 0 {
        return Ok(empty_tensor(output_shape_obj, dtype, device, requires_grad));
    }

    macro_rules! gather_impl {
        ($ty:ty, $slice:ident, $from_vec:ident) => {{
            let src = tensor.data().$slice().ok_or_else(|| {
                MinitensorError::invalid_operation("Tensor data access failed for gather")
            })?;
            let idx = idx_slice;
            let mut out = vec![<$ty>::default(); output_numel];
            let chunk_size = idx_dim * inner;
            if output_numel % chunk_size != 0 {
                return Err(MinitensorError::internal_error(format!(
                    "gather output length ({output_numel}) is not divisible by chunk size ({chunk_size})"
                )));
            }
            out.par_chunks_mut(chunk_size)
                .enumerate()
                .for_each(|(o, out_chunk)| {
                    let base = o * dim_size * inner;
                    let idx_chunk = &idx[o * chunk_size..(o + 1) * chunk_size];
                    for i in 0..idx_dim {
                        let idx_row = &idx_chunk[i * inner..(i + 1) * inner];
                        let dst_row = &mut out_chunk[i * inner..(i + 1) * inner];
                        for (j, &gather_val) in idx_row.iter().enumerate() {
                            let gather_idx = gather_val as usize;
                            dst_row[j] = src[base + gather_idx * inner + j];
                        }
                    }
                });
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

    if start > dim_size || end > dim_size || start > end {
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
    let output_shape_obj = Shape::new(output_shape);

    let dtype = tensor.dtype();
    let device = tensor.device();
    let requires_grad = tensor.requires_grad();

    if output_shape_obj.numel() == 0 {
        return Ok(empty_tensor(output_shape_obj, dtype, device, requires_grad));
    }

    let dims = tensor.shape().dims();
    let inner: usize = dims[dim + 1..].iter().product();
    let count = output_shape_obj.dims()[dim];

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

fn collect_repeats_from_values<I>(len: usize, values: I) -> Result<Vec<usize>>
where
    I: IntoIterator<Item = i64>,
{
    let mut out = Vec::with_capacity(len);
    for value in values {
        if value < 0 {
            return Err(MinitensorError::invalid_operation(
                "repeat_interleave: repeats must be non-negative".to_string(),
            ));
        }
        out.push(value as usize);
    }
    Ok(out)
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
            collect_repeats_from_values(slice.len(), slice.iter().map(|&value| value as i64))
        }
        DataType::Int64 => {
            let slice = tensor.data().as_i64_slice().ok_or_else(|| {
                MinitensorError::invalid_operation(
                    "repeat_interleave: repeats tensor must be contiguous".to_string(),
                )
            })?;
            collect_repeats_from_values(slice.len(), slice.iter().copied())
        }
        other => Err(MinitensorError::type_mismatch(
            "integral tensor",
            format!("{:?}", other),
        )),
    }
}
