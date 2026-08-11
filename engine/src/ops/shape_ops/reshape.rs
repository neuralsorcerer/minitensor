// Copyright (c) Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

use crate::{
    autograd::{
        ConcatBackward, GatherBackward, IndexSelectBackward, RepeatBackward, ReshapeBackward,
        add_to_graph,
    },
    device::Device,
    error::{MinitensorError, Result},
    ops::map::PAR_THRESHOLD,
    tensor::{DataType, Shape, Tensor, TensorData},
};
use rayon::prelude::*;
use std::sync::Arc;

pub(crate) use crate::ops::util::{normalize_dim, normalize_dim_named};

fn empty_tensor(shape: Shape, dtype: DataType, device: Device, requires_grad: bool) -> Tensor {
    Tensor::new(
        Arc::new(TensorData::zeros_on_device(0, dtype, device)),
        shape,
        dtype,
        device,
        requires_grad,
    )
}

fn checked_repeat_dim(size: usize, repeat: usize) -> Result<usize> {
    size.checked_mul(repeat).ok_or_else(|| {
        MinitensorError::invalid_operation(
            "repeat output dimensions exceed supported size".to_string(),
        )
    })
}

fn checked_repeat_numel(dims: &[usize]) -> Result<usize> {
    if dims.is_empty() {
        return Ok(1);
    }

    dims.iter().try_fold(1usize, |numel, &dim| {
        numel.checked_mul(dim).ok_or_else(|| {
            MinitensorError::invalid_operation(
                "repeat output dimensions exceed supported size".to_string(),
            )
        })
    })
}

fn attach_repeat_backward(mut output: Tensor, input: &Tensor, repeats: &[usize]) -> Result<Tensor> {
    if input.requires_grad() && input.dtype().is_float() && crate::autograd::is_grad_enabled() {
        output.refresh_autograd_metadata();
        let mut output = output.requires_grad_(true);
        let grad_fn = Arc::new(RepeatBackward {
            input_id: input.id(),
            input_shape: input.shape().dims().to_vec(),
            repeats: repeats.to_vec(),
        });
        output.set_grad_fn(Some(grad_fn.clone()));
        add_to_graph(&output, Some(grad_fn))?;
        Ok(output)
    } else {
        Ok(output)
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

    // Reinterpret the buffer when possible; materialise a contiguous copy for
    // non-contiguous inputs (e.g. results of `expand`) so the new shape always
    // describes real storage. The copy is made outside of autograd because the
    // ReshapeBackward node attached below already routes gradients straight to
    // the original tensor.
    // Under `no_grad`, the output must not claim to be tracked. Neither path
    // below gates on grad mode by itself: `view` clones the input (flag and
    // all), and `requires_grad_` deliberately ignores grad mode so that marking
    // a leaf trainable inside `no_grad` still works. Propagating the flag
    // through it therefore produced a tensor with `requires_grad = true` and a
    // `grad_fn`, but no graph node -- `add_to_graph` gates correctly -- so the
    // result looked tracked and back-propagated to nothing.
    let track = tensor.requires_grad() && crate::autograd::is_grad_enabled();
    let mut reshaped = if tensor.is_contiguous() {
        tensor.view(new_shape.clone())?.requires_grad_(track)
    } else {
        tensor
            .detach()
            .contiguous()?
            .view(new_shape.clone())?
            .requires_grad_(track)
    };
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
    let mut out_dims = Vec::with_capacity(dims.len());
    let mut inferred_index: Option<usize> = None;
    let mut known_product: usize = 1;

    for (index, &dim) in dims.iter().enumerate() {
        if dim == -1 {
            if inferred_index.is_some() {
                return Err(MinitensorError::invalid_operation(
                    "can only specify one -1 dimension in reshape".to_string(),
                ));
            }
            inferred_index = Some(index);
            out_dims.push(0);
            continue;
        }

        if dim < 0 {
            return Err(MinitensorError::invalid_operation(
                "invalid negative dimension".to_string(),
            ));
        }

        let dim_usize = dim as usize;
        known_product = known_product.checked_mul(dim_usize).ok_or_else(|| {
            MinitensorError::invalid_operation("reshape dimensions exceed supported size")
        })?;
        out_dims.push(dim_usize);
    }

    let total_elements = tensor.numel();
    if let Some(index) = inferred_index {
        if known_product == 0 {
            return Err(MinitensorError::invalid_operation(
                "cannot reshape tensor with -1 and 0 dimensions".to_string(),
            ));
        }

        if !total_elements.is_multiple_of(known_product) {
            return Err(MinitensorError::invalid_operation(
                "cannot infer reshape dimension".to_string(),
            ));
        }

        out_dims[index] = total_elements / known_product;
    } else if known_product != total_elements {
        return Err(MinitensorError::shape_mismatch(
            vec![total_elements],
            vec![known_product],
        ));
    }

    reshape(tensor, Shape::new(out_dims))
}

/// Squeeze operation - remove dimensions of size 1.
///
/// Routed through [`reshape`] so the result is a first-class differentiable node
/// (the plain `Tensor::squeeze` view shares the input's id and would attribute a
/// wrongly-shaped gradient to it).
pub fn squeeze(tensor: &Tensor, dim: Option<isize>) -> Result<Tensor> {
    let dims = tensor.shape().dims();
    let new_dims: Vec<usize> = match dim {
        None => dims.iter().copied().filter(|&d| d != 1).collect(),
        Some(d) => {
            let d = normalize_dim(d, tensor.ndim())?;
            if dims[d] != 1 {
                // A non-unit axis remains untouched.
                dims.to_vec()
            } else {
                let mut v = dims.to_vec();
                v.remove(d);
                v
            }
        }
    };
    reshape(tensor, Shape::new(new_dims))
}

/// Unsqueeze operation - add a dimension of size 1. See [`squeeze`] for why this
/// goes through [`reshape`] rather than the view-based `Tensor::unsqueeze`.
pub fn unsqueeze(tensor: &Tensor, dim: isize) -> Result<Tensor> {
    // An axis may be inserted one past the last, so `unsqueeze` accepts a
    // range one wider than the tensor's rank: `[-(ndim + 1), ndim]`.
    let d = normalize_dim(dim, tensor.ndim() + 1)?;
    let mut new_dims = tensor.shape().dims().to_vec();
    new_dims.insert(d, 1);
    reshape(tensor, Shape::new(new_dims))
}

/// Flatten dimensions `start_dim..=end_dim` into one. Routed through [`reshape`]
/// so gradients flow (see [`squeeze`]).
pub fn flatten(tensor: &Tensor, start_dim: isize, end_dim: isize) -> Result<Tensor> {
    let start = normalize_dim_named(start_dim, tensor.ndim(), "flatten: start_dim")?;
    let end = normalize_dim_named(end_dim, tensor.ndim(), "flatten: end_dim")?;
    if start > end {
        return Err(MinitensorError::invalid_argument(
            "start_dim must be less than or equal to end_dim",
        ));
    }

    let dims = tensor.shape().dims();
    let mut new_dims = dims[..start].to_vec();
    new_dims.push(dims[start..=end].iter().product());
    new_dims.extend_from_slice(&dims[end + 1..]);
    reshape(tensor, Shape::new(new_dims))
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
        normalized.push(normalize_dim(d, ndim)?);
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
        let s = normalize_dim_named(s, ndim, "movedim: source")?;
        if src_seen[s] {
            return Err(MinitensorError::invalid_operation(
                "movedim: duplicate dimensions in source".to_string(),
            ));
        }
        src_seen[s] = true;
        let d = normalize_dim_named(d, ndim, "movedim: destination")?;
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

    let output = Tensor::new(
        Arc::new(data),
        output_shape_obj,
        dtype,
        device,
        requires_grad,
    );

    if requires_grad && dtype.is_float() {
        let grad_fn = Arc::new(ConcatBackward {
            input_ids: tensors.iter().map(|t| t.id()).collect(),
            sizes: tensors.iter().map(|t| t.shape().dims()[dim]).collect(),
            dim,
            input_requires_grad: tensors.iter().map(|t| t.requires_grad()).collect(),
        });
        let mut output = output;
        output.set_grad_fn(Some(grad_fn.clone()));
        add_to_graph(&output, Some(grad_fn))?;
        return Ok(output);
    }

    Ok(output)
}

/// Repeat `tensor` according to `repeats` along each dimension.
pub fn repeat(tensor: &Tensor, repeats: &[usize]) -> Result<Tensor> {
    if repeats.len() < tensor.ndim() {
        return Err(MinitensorError::invalid_operation(
            "number of dimensions of repeat dims can not be smaller than number of dimensions of tensor",
        ));
    }

    // Tile on a detached view so the intermediate per-dimension copies never
    // create graph nodes; a single RepeatBackward maps the final result straight
    // back to the original input.
    let mut result = tensor.detach();

    if repeats.len() > result.ndim() {
        let mut new_shape = vec![1; repeats.len() - result.ndim()];
        new_shape.extend_from_slice(result.shape().dims());
        result = result.reshape(Shape::new(new_shape))?;
    }

    if repeats.contains(&0) {
        let mut out_shape = result.shape().dims().to_vec();
        for (dim, &rep) in repeats.iter().enumerate() {
            out_shape[dim] = checked_repeat_dim(out_shape[dim], rep)?;
        }

        let output = empty_tensor(
            Shape::new(out_shape),
            result.dtype(),
            result.device(),
            false,
        );
        return attach_repeat_backward(output, tensor, repeats);
    }

    for (dim, &rep) in repeats.iter().enumerate() {
        if rep == 1 {
            continue;
        }
        let dims = result.shape().dims().to_vec();
        let dim_size = dims[dim];
        let inner: usize = dims[dim + 1..].iter().product();
        let repeated_dim = checked_repeat_dim(dim_size, rep)?;
        let chunk_size = checked_repeat_dim(repeated_dim, inner)?;
        let src_chunk_size = checked_repeat_dim(dim_size, inner)?;
        let mut output_shape = dims.clone();
        output_shape[dim] = repeated_dim;
        let output_numel = checked_repeat_numel(&output_shape)?;
        let output_shape_obj = Shape::new(output_shape);

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

    attach_repeat_backward(result, tensor, repeats)
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

    let output = Tensor::new(
        Arc::new(data),
        output_shape_obj,
        dtype,
        device,
        requires_grad,
    );

    if requires_grad && dtype.is_float() {
        let grad_fn = Arc::new(IndexSelectBackward {
            input_id: tensor.id(),
            input_shape: tensor.shape().dims().to_vec(),
            dim,
            indices: indices.to_vec(),
        });
        let mut output = output;
        output.set_grad_fn(Some(grad_fn.clone()));
        add_to_graph(&output, Some(grad_fn))?;
        return Ok(output);
    }

    Ok(output)
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

    let output = Tensor::new(
        Arc::new(data),
        output_shape_obj,
        dtype,
        device,
        requires_grad,
    );

    if requires_grad && dtype.is_float() {
        let grad_fn = Arc::new(GatherBackward {
            input_id: tensor.id(),
            input_shape: input_dims.to_vec(),
            dim,
            index: idx_slice.to_vec(),
        });
        let mut output = output;
        output.set_grad_fn(Some(grad_fn.clone()));
        add_to_graph(&output, Some(grad_fn))?;
        return Ok(output);
    }

    Ok(output)
}

/// Slicing operation - select a contiguous range of elements
pub fn slice(tensor: &Tensor, dim: isize, start: usize, end: usize, step: usize) -> Result<Tensor> {
    let dim = normalize_dim(dim, tensor.ndim())?;

    let dim_size = tensor.shape().dims()[dim];

    if start > dim_size || end > dim_size {
        return Err(MinitensorError::invalid_operation(format!(
            "Invalid slice range: start={}, end={}, dim_size={}",
            start, end, dim_size
        )));
    }

    // An inverted range selects nothing. Clamping
    // here rather than erroring also keeps the `end - start` below from
    // underflowing `usize`.
    let end = end.max(start);

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
    let outer_stride = dims[dim] * inner;
    // One output block per position in the dimensions ahead of `dim`. Every
    // block is exactly this long because the output has the same leading dims.
    let block = count * inner;

    macro_rules! slice_impl {
        ($ty:ty, $slice:ident, $from_vec:ident) => {{
            let src = tensor.data().$slice().ok_or_else(|| {
                MinitensorError::invalid_operation("Tensor data access failed for slice")
            })?;
            let mut out = vec![<$ty>::default(); output_shape_obj.numel()];
            let fill = |o: usize, out_chunk: &mut [$ty]| {
                let block_start = o * outer_stride;
                if step == 1 {
                    // A unit step selects a run that is contiguous with
                    // everything beneath it, so a whole block is a single copy.
                    // Copying it as `count` runs of `inner` instead cost an
                    // order of magnitude whenever `inner` was 1 -- which is
                    // every slice along the last dimension, and every slice of
                    // a 1-D tensor.
                    let src_start = block_start + start * inner;
                    out_chunk.copy_from_slice(&src[src_start..src_start + out_chunk.len()]);
                } else {
                    for i in 0..count {
                        let src_start = block_start + (start + i * step) * inner;
                        let dst_start = i * inner;
                        out_chunk[dst_start..dst_start + inner]
                            .copy_from_slice(&src[src_start..src_start + inner]);
                    }
                }
            };
            if out.len() < PAR_THRESHOLD {
                out.chunks_mut(block)
                    .enumerate()
                    .for_each(|(o, out_chunk)| fill(o, out_chunk));
            } else {
                out.par_chunks_mut(block)
                    .enumerate()
                    .for_each(|(o, out_chunk)| fill(o, out_chunk));
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

    let output = Tensor::new(
        Arc::new(data),
        output_shape_obj,
        dtype,
        device,
        requires_grad,
    );

    if requires_grad && dtype.is_float() {
        // A slice selects source positions `start, start+step, ...` along `dim`;
        // its backward scatters the gradient back to exactly those positions.
        let indices: Vec<usize> = (0..count).map(|i| start + i * step).collect();
        let grad_fn = Arc::new(IndexSelectBackward {
            input_id: tensor.id(),
            input_shape: tensor.shape().dims().to_vec(),
            dim,
            indices,
        });
        let mut output = output;
        output.set_grad_fn(Some(grad_fn.clone()));
        add_to_graph(&output, Some(grad_fn))?;
        return Ok(output);
    }

    Ok(output)
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
/// Copy `tensor` with every reversed dimension applied in a single pass.
///
/// `flipped[d]` says whether dimension `d` is reversed. A flip is an index
/// remapping just as a roll is -- output position `i` along a reversed
/// dimension reads input position `size - 1 - i` -- so all of the dimensions
/// resolve while copying once. Reversing them one at a time built a whole
/// intermediate tensor per dimension, since each went through `index_select`:
/// `flip([0, 1])` on a 4096x1024 f32 tensor took 11.70 ms against a plain
/// contiguous copy of the same tensor at 1.39 ms.
///
/// The copy goes a row at a time, a row being the last dimension and therefore
/// contiguous: it is a `memcpy` when that dimension is not reversed and an
/// element-wise reverse when it is, and the leading dimensions cost only the
/// arithmetic that locates the source row.
fn flip_rows(tensor: &Tensor, flipped: &[bool]) -> Result<Tensor> {
    if !tensor.device().is_cpu() {
        return Err(MinitensorError::invalid_operation(
            "flip currently supports only CPU tensors",
        ));
    }

    let dims = tensor.shape().dims();
    let last = dims.len() - 1;
    let row_len = dims[last];
    let dtype = tensor.dtype();
    let device = tensor.device();

    // Strides of the dimensions ahead of the last, counted in rows.
    let mut row_strides = vec![0usize; last];
    let mut acc = 1usize;
    for d in (0..last).rev() {
        row_strides[d] = acc;
        acc *= dims[d];
    }
    let leading_flipped = flipped[..last].iter().any(|&f| f);
    let reverse_row = flipped[last];

    // Output row `r` decodes to leading indices `i_d`; it copies from the row at
    // the same indices with the reversed ones mirrored.
    let source_row = |r: usize| -> usize {
        if !leading_flipped {
            return r;
        }
        let mut source = 0;
        let mut rest = r;
        for d in 0..last {
            let stride = row_strides[d];
            let i = rest / stride;
            rest %= stride;
            source += if flipped[d] { dims[d] - 1 - i } else { i } * stride;
        }
        source
    };

    macro_rules! flip_impl {
        ($ty:ty, $slice:ident, $from_vec:ident) => {{
            let src = tensor.data().$slice().ok_or_else(|| {
                MinitensorError::internal_error("Tensor data access failed for flip")
            })?;
            let mut out = vec![<$ty>::default(); src.len()];
            let copy_row = |r: usize, dst: &mut [$ty]| {
                let base = source_row(r) * row_len;
                let row = &src[base..base + row_len];
                if reverse_row {
                    for (slot, value) in dst.iter_mut().zip(row.iter().rev()) {
                        *slot = *value;
                    }
                } else {
                    dst.copy_from_slice(row);
                }
            };
            if out.len() < PAR_THRESHOLD {
                out.chunks_mut(row_len)
                    .enumerate()
                    .for_each(|(r, dst)| copy_row(r, dst));
            } else {
                out.par_chunks_mut(row_len)
                    .enumerate()
                    .for_each(|(r, dst)| copy_row(r, dst));
            }
            TensorData::$from_vec(out, device)
        }};
    }

    let data = match dtype {
        DataType::Float32 => flip_impl!(f32, as_f32_slice, from_vec_f32),
        DataType::Float64 => flip_impl!(f64, as_f64_slice, from_vec_f64),
        DataType::Int32 => flip_impl!(i32, as_i32_slice, from_vec_i32),
        DataType::Int64 => flip_impl!(i64, as_i64_slice, from_vec_i64),
        DataType::Bool => flip_impl!(bool, as_bool_slice, from_vec_bool),
    };

    Ok(Tensor::new(
        Arc::new(data),
        tensor.shape().clone(),
        dtype,
        device,
        tensor.requires_grad(),
    ))
}

pub fn flip(tensor: &Tensor, dims: &[isize]) -> Result<Tensor> {
    let ndim = tensor.ndim();
    let mut flipped = vec![false; ndim];
    let mut normalized = Vec::with_capacity(dims.len());
    for &d in dims {
        let dim = normalize_dim(d, ndim)?;
        if flipped[dim] {
            return Err(MinitensorError::invalid_operation(
                "dims must be unique".to_string(),
            ));
        }
        flipped[dim] = true;
        normalized.push(d);
    }

    // Nothing to reverse, or nothing to reverse it in.
    if normalized.is_empty() || tensor.numel() == 0 {
        return Ok(tensor.clone());
    }

    // Fill the output directly on a detached view, so the copy leaves no
    // gradient edges of its own; one `FlipBackward` inverts the whole thing.
    let track_grad =
        tensor.requires_grad() && tensor.dtype().is_float() && crate::autograd::is_grad_enabled();
    let base = if track_grad {
        tensor.detach()
    } else {
        tensor.clone()
    };
    let output = flip_rows(&base, &flipped)?;

    if track_grad {
        let mut output = output;
        output.refresh_autograd_metadata();
        let mut output = output.requires_grad_(true);
        let grad_fn = Arc::new(crate::autograd::FlipBackward {
            input_id: tensor.id(),
            dims: normalized,
        });
        output.set_grad_fn(Some(grad_fn.clone()));
        add_to_graph(&output, Some(grad_fn))?;
        return Ok(output);
    }

    Ok(output)
}

/// Roll tensor elements along specified dimensions with wrap-around
pub fn roll(tensor: &Tensor, shifts: &[isize], dims: Option<&[isize]>) -> Result<Tensor> {
    // Compute the roll on a detached view so the internal slice/concatenate steps
    // (which flatten to a storage-sharing view in the `dims == None` case) never
    // build gradient edges; a single RollBackward inverts the whole operation.
    let track_grad =
        tensor.requires_grad() && tensor.dtype().is_float() && crate::autograd::is_grad_enabled();
    let base = if track_grad {
        tensor.detach()
    } else {
        tensor.clone()
    };
    let output = roll_forward(&base, shifts, dims)?;

    if track_grad {
        let mut output = output;
        output.refresh_autograd_metadata();
        let mut output = output.requires_grad_(true);
        let grad_fn = Arc::new(crate::autograd::RollBackward {
            input_id: tensor.id(),
            shifts: shifts.to_vec(),
            dims: dims.map(|d| d.to_vec()),
        });
        output.set_grad_fn(Some(grad_fn.clone()));
        add_to_graph(&output, Some(grad_fn))?;
        return Ok(output);
    }

    Ok(output)
}

/// Copy `tensor` with every rolled dimension applied in a single pass.
///
/// `shifts[d]` is the shift already reduced into `0..dims[d]`, and the shape is
/// unchanged, so the output buffer can be filled directly. Rolling one
/// dimension at a time meant a whole intermediate tensor per dimension, each
/// built as slice + slice + concatenate -- three allocations and three passes
/// apiece. A roll is only an index remapping, so all of the dimensions can be
/// resolved while copying once.
///
/// The copy goes a row at a time, a row being the last dimension and therefore
/// contiguous: the innermost dimension costs two `memcpy`s per row (one when it
/// is not rolled, since the wrapped part is then empty) and the rest cost only
/// the arithmetic that locates the source row.
fn roll_rows(tensor: &Tensor, shifts: &[usize]) -> Result<Tensor> {
    if !tensor.device().is_cpu() {
        return Err(MinitensorError::invalid_operation(
            "roll currently supports only CPU tensors",
        ));
    }

    let dims = tensor.shape().dims();
    let last = dims.len() - 1;
    let row_len = dims[last];
    let dtype = tensor.dtype();
    let device = tensor.device();

    // Strides of the dimensions ahead of the last, counted in rows.
    let mut row_strides = vec![0usize; last];
    let mut acc = 1usize;
    for d in (0..last).rev() {
        row_strides[d] = acc;
        acc *= dims[d];
    }
    let leading_rolled = shifts[..last].iter().any(|&k| k != 0);

    // Within a row the element at `split` becomes the first. An unrolled last
    // dimension gives `split == row_len`, which makes the wrapped part empty
    // and leaves a single whole-row copy.
    let split = row_len - shifts[last];
    let head = row_len - split;

    // Output row `r` decodes to leading indices `i_d`; it copies from the row
    // at the same indices stepped back by their own shifts.
    let source_row = |r: usize| -> usize {
        if !leading_rolled {
            return r;
        }
        let mut source = 0;
        let mut rest = r;
        for d in 0..last {
            let stride = row_strides[d];
            let i = rest / stride;
            rest %= stride;
            let k = shifts[d];
            source += if i >= k { i - k } else { i + dims[d] - k } * stride;
        }
        source
    };

    macro_rules! roll_impl {
        ($ty:ty, $slice:ident, $from_vec:ident) => {{
            let src = tensor.data().$slice().ok_or_else(|| {
                MinitensorError::invalid_operation("Tensor data access failed for roll")
            })?;
            let mut out = vec![<$ty>::default(); src.len()];
            let copy_row = |r: usize, dst: &mut [$ty]| {
                let base = source_row(r) * row_len;
                dst[..head].copy_from_slice(&src[base + split..base + row_len]);
                dst[head..].copy_from_slice(&src[base..base + split]);
            };
            if out.len() < PAR_THRESHOLD {
                out.chunks_mut(row_len)
                    .enumerate()
                    .for_each(|(r, dst)| copy_row(r, dst));
            } else {
                out.par_chunks_mut(row_len)
                    .enumerate()
                    .for_each(|(r, dst)| copy_row(r, dst));
            }
            TensorData::$from_vec(out, device)
        }};
    }

    let data = match dtype {
        DataType::Float32 => roll_impl!(f32, as_f32_slice, from_vec_f32),
        DataType::Float64 => roll_impl!(f64, as_f64_slice, from_vec_f64),
        DataType::Int32 => roll_impl!(i32, as_i32_slice, from_vec_i32),
        DataType::Int64 => roll_impl!(i64, as_i64_slice, from_vec_i64),
        DataType::Bool => roll_impl!(bool, as_bool_slice, from_vec_bool),
    };

    Ok(Tensor::new(
        Arc::new(data),
        tensor.shape().clone(),
        dtype,
        device,
        tensor.requires_grad(),
    ))
}

fn roll_forward(tensor: &Tensor, shifts: &[isize], dims: Option<&[isize]>) -> Result<Tensor> {
    if let Some(dims) = dims {
        if shifts.len() != dims.len() {
            return Err(MinitensorError::invalid_operation(
                "shifts and dims must have the same length".to_string(),
            ));
        }

        // Reduce every shift into its own dimension first, so that one pass can
        // apply all of them. Repeating a dimension accumulates, which is what
        // rolling it twice in succession did.
        let mut by_dim = vec![0usize; tensor.ndim()];
        let mut any = false;
        for (&shift, &dim) in shifts.iter().zip(dims.iter()) {
            let dim = normalize_dim(dim, tensor.ndim())?;
            let size = tensor.shape().dims()[dim] as isize;
            if size == 0 {
                continue;
            }
            let k = (((shift % size) + size) % size) as usize;
            by_dim[dim] = (by_dim[dim] + k) % size as usize;
            any |= by_dim[dim] != 0;
        }

        // Nothing to move: no shift survived reduction, or there is no data to
        // move in the first place.
        if !any || tensor.numel() == 0 {
            return Ok(tensor.clone());
        }
        roll_rows(tensor, &by_dim)
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
        // A flat roll is a roll of the one dimension the buffer already has.
        roll_rows(&flat, &[k as usize])?.reshape(tensor.shape().clone())
    }
}

/// Specification of repeat counts accepted by [`super::repeat_interleave`].
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

pub(crate) fn collect_repeats_from_tensor(tensor: &Tensor, dim_size: usize) -> Result<Vec<usize>> {
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

#[cfg(test)]
mod reshape_tests {
    use super::*;

    #[test]
    fn reshape_with_inference_rejects_overflowing_known_product() {
        let tensor = Tensor::zeros(Shape::new(vec![1]), DataType::Float32, Device::cpu(), false);

        let result = reshape_with_inference(&tensor, vec![isize::MAX, isize::MAX, -1]);

        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("reshape dimensions exceed supported size")
        );
    }

    #[test]
    fn reshape_with_inference_rejects_multiple_inferred_dimensions() {
        let tensor = Tensor::zeros(Shape::new(vec![4]), DataType::Float32, Device::cpu(), false);

        let result = reshape_with_inference(&tensor, vec![-1, -1]);

        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("can only specify one -1 dimension in reshape")
        );
    }

    #[test]
    fn reshape_with_inference_rejects_invalid_negative_dimension() {
        let tensor = Tensor::zeros(Shape::new(vec![1]), DataType::Float32, Device::cpu(), false);

        let result = reshape_with_inference(&tensor, vec![-2, 1]);

        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("invalid negative dimension")
        );
    }

    #[test]
    fn reshape_with_inference_rejects_overflowing_shape_without_inference() {
        let tensor = Tensor::zeros(Shape::new(vec![1]), DataType::Float32, Device::cpu(), false);

        let result = reshape_with_inference(&tensor, vec![isize::MAX, isize::MAX]);

        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("reshape dimensions exceed supported size")
        );
    }

    #[test]
    fn reshape_with_inference_rejects_zero_dimension_with_inference() {
        let tensor = Tensor::zeros(Shape::new(vec![0]), DataType::Float32, Device::cpu(), false);

        let result = reshape_with_inference(&tensor, vec![-1, 0]);

        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("cannot reshape tensor with -1 and 0 dimensions")
        );
    }

    #[test]
    fn reshape_with_inference_no_inference_shape_mismatch() {
        let tensor = Tensor::zeros(Shape::new(vec![5]), DataType::Float32, Device::cpu(), false);

        let result = reshape_with_inference(&tensor, vec![2, 2]);

        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Shape mismatch"));
    }

    #[test]
    fn reshape_with_inference_no_inference_shape_match() {
        let tensor = Tensor::zeros(Shape::new(vec![6]), DataType::Float32, Device::cpu(), false);

        let result = reshape_with_inference(&tensor, vec![2, 3]);

        assert!(result.is_ok());
        assert_eq!(
            result.expect("reshape should succeed").shape().dims(),
            &[2, 3]
        );
    }

    #[test]
    fn reshape_with_inference_infers_single_negative_dimension() {
        let tensor = Tensor::zeros(
            Shape::new(vec![12]),
            DataType::Float32,
            Device::cpu(),
            false,
        );

        let reshaped = reshape_with_inference(&tensor, vec![3, -1]).expect("reshape should work");

        assert_eq!(reshaped.shape().dims(), &[3, 4]);
    }

    #[test]
    fn slice_treats_an_inverted_range_as_empty() {
        // `end < start` selects nothing. It cannot
        // simply fall through to the size computation either: `end - start` is
        // `usize` arithmetic, so an unclamped inverted range would underflow
        // rather than produce a small result.
        let tensor = Tensor::zeros(
            Shape::new(vec![4, 3]),
            DataType::Float32,
            Device::cpu(),
            false,
        );

        for (start, end) in [(3usize, 1usize), (2, 0), (4, 0), (1, 1)] {
            let sliced = slice(&tensor, 0, start, end, 1)
                .unwrap_or_else(|err| panic!("slice({start}, {end}) failed: {err}"));
            assert_eq!(sliced.shape().dims(), &[0, 3], "slice({start}, {end})");
            assert_eq!(sliced.numel(), 0);
        }

        // Bounds that exceed the dimension are still rejected.
        assert!(slice(&tensor, 0, 0, 5, 1).is_err());
        assert!(slice(&tensor, 0, 5, 5, 1).is_err());
    }
}
