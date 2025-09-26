// Copyright (c) 2025 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

use crate::{
    autograd::{CumprodBackward, CumsumBackward, ProdBackward, SumBackward, add_to_graph},
    error::{MinitensorError, Result},
    operations::{
        activation, arithmetic, shape_ops,
        simd::{
            simd_prod_f32, simd_prod_f64, simd_prod_i32, simd_prod_i64, simd_sum_f32, simd_sum_f64,
            simd_sum_i32, simd_sum_i64,
        },
    },
    tensor::{DataType, Shape, Tensor, TensorData},
};
use rayon::prelude::*;
use std::cmp::Ordering;
use std::sync::Arc;

fn cmp_f32_desc(a: &(usize, f32), b: &(usize, f32)) -> Ordering {
    match (a.1.is_nan(), b.1.is_nan()) {
        (true, true) => a.0.cmp(&b.0),
        (true, false) => Ordering::Less,
        (false, true) => Ordering::Greater,
        (false, false) => match b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal) {
            Ordering::Equal => a.0.cmp(&b.0),
            order => order,
        },
    }
}

fn cmp_f32_asc(a: &(usize, f32), b: &(usize, f32)) -> Ordering {
    match (a.1.is_nan(), b.1.is_nan()) {
        (true, true) => a.0.cmp(&b.0),
        (true, false) => Ordering::Greater,
        (false, true) => Ordering::Less,
        (false, false) => match a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal) {
            Ordering::Equal => a.0.cmp(&b.0),
            order => order,
        },
    }
}

fn cmp_f64_desc(a: &(usize, f64), b: &(usize, f64)) -> Ordering {
    match (a.1.is_nan(), b.1.is_nan()) {
        (true, true) => a.0.cmp(&b.0),
        (true, false) => Ordering::Less,
        (false, true) => Ordering::Greater,
        (false, false) => match b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal) {
            Ordering::Equal => a.0.cmp(&b.0),
            order => order,
        },
    }
}

fn cmp_f64_asc(a: &(usize, f64), b: &(usize, f64)) -> Ordering {
    match (a.1.is_nan(), b.1.is_nan()) {
        (true, true) => a.0.cmp(&b.0),
        (true, false) => Ordering::Greater,
        (false, true) => Ordering::Less,
        (false, false) => match a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal) {
            Ordering::Equal => a.0.cmp(&b.0),
            order => order,
        },
    }
}

fn cmp_i32_desc(a: &(usize, i32), b: &(usize, i32)) -> Ordering {
    match b.1.cmp(&a.1) {
        Ordering::Equal => a.0.cmp(&b.0),
        order => order,
    }
}

fn cmp_i32_asc(a: &(usize, i32), b: &(usize, i32)) -> Ordering {
    match a.1.cmp(&b.1) {
        Ordering::Equal => a.0.cmp(&b.0),
        order => order,
    }
}

fn cmp_i64_desc(a: &(usize, i64), b: &(usize, i64)) -> Ordering {
    match b.1.cmp(&a.1) {
        Ordering::Equal => a.0.cmp(&b.0),
        order => order,
    }
}

fn cmp_i64_asc(a: &(usize, i64), b: &(usize, i64)) -> Ordering {
    match a.1.cmp(&b.1) {
        Ordering::Equal => a.0.cmp(&b.0),
        order => order,
    }
}

fn cmp_bool_desc(a: &(usize, bool), b: &(usize, bool)) -> Ordering {
    match (a.1, b.1) {
        (true, true) | (false, false) => a.0.cmp(&b.0),
        (true, false) => Ordering::Less,
        (false, true) => Ordering::Greater,
    }
}

fn cmp_bool_asc(a: &(usize, bool), b: &(usize, bool)) -> Ordering {
    match (a.1, b.1) {
        (true, true) | (false, false) => a.0.cmp(&b.0),
        (true, false) => Ordering::Greater,
        (false, true) => Ordering::Less,
    }
}

fn ensure_non_empty(numel: usize) -> Result<()> {
    if numel == 0 {
        Err(MinitensorError::invalid_argument(
            "median() does not support empty tensors".to_string(),
        ))
    } else {
        Ok(())
    }
}

pub fn median(
    tensor: &Tensor,
    dim: Option<isize>,
    keepdim: bool,
) -> Result<(Tensor, Option<Tensor>)> {
    ensure_non_empty(tensor.numel())?;

    if tensor.ndim() == 0 {
        return Ok((tensor.clone(), None));
    }

    match dim {
        None => median_all(tensor),
        Some(dim_value) => {
            let axis = if tensor.ndim() == 0 {
                if dim_value == 0 || dim_value == -1 {
                    0
                } else {
                    return Err(MinitensorError::index_error(dim_value, 0, 1));
                }
            } else {
                normalize_dim(dim_value, tensor.ndim())?
            };
            let (values, indices) = median_along_dim(tensor, axis, keepdim)?;
            Ok((values, Some(indices)))
        }
    }
}

fn median_all(tensor: &Tensor) -> Result<(Tensor, Option<Tensor>)> {
    let mut result_data = TensorData::zeros_on_device(1, tensor.dtype(), tensor.device());

    match tensor.dtype() {
        DataType::Float32 => {
            let data = tensor
                .data()
                .as_f32_slice()
                .ok_or_else(|| MinitensorError::internal_error("Failed to get f32 slice"))?;
            let mut values: Vec<f32> = data.to_vec();
            values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Greater));
            let median = values[(values.len() - 1) / 2];
            result_data.as_f32_slice_mut().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get mutable f32 slice")
            })?[0] = median;
        }
        DataType::Float64 => {
            let data = tensor
                .data()
                .as_f64_slice()
                .ok_or_else(|| MinitensorError::internal_error("Failed to get f64 slice"))?;
            let mut values: Vec<f64> = data.to_vec();
            values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Greater));
            let median = values[(values.len() - 1) / 2];
            result_data.as_f64_slice_mut().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get mutable f64 slice")
            })?[0] = median;
        }
        DataType::Int32 => {
            let data = tensor
                .data()
                .as_i32_slice()
                .ok_or_else(|| MinitensorError::internal_error("Failed to get i32 slice"))?;
            let mut values: Vec<i32> = data.to_vec();
            values.sort_unstable();
            let median = values[(values.len() - 1) / 2];
            result_data.as_i32_slice_mut().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get mutable i32 slice")
            })?[0] = median;
        }
        DataType::Int64 => {
            let data = tensor
                .data()
                .as_i64_slice()
                .ok_or_else(|| MinitensorError::internal_error("Failed to get i64 slice"))?;
            let mut values: Vec<i64> = data.to_vec();
            values.sort_unstable();
            let median = values[(values.len() - 1) / 2];
            result_data.as_i64_slice_mut().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get mutable i64 slice")
            })?[0] = median;
        }
        DataType::Bool => {
            let data = tensor
                .data()
                .as_bool_slice()
                .ok_or_else(|| MinitensorError::internal_error("Failed to get bool slice"))?;
            let mut values: Vec<bool> = data.to_vec();
            values.sort_by(|a, b| match (a, b) {
                (true, true) | (false, false) => Ordering::Equal,
                (false, true) => Ordering::Less,
                (true, false) => Ordering::Greater,
            });
            let median = values[(values.len() - 1) / 2];
            result_data.as_bool_slice_mut().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get mutable bool slice")
            })?[0] = median;
        }
    }

    let value = Tensor::new(
        Arc::new(result_data),
        Shape::scalar(),
        tensor.dtype(),
        tensor.device(),
        tensor.requires_grad(),
    );

    Ok((value, None))
}

fn median_along_dim(tensor: &Tensor, dim: usize, keepdim: bool) -> Result<(Tensor, Tensor)> {
    let dims = tensor.shape().dims();
    let dim_size = if dims.is_empty() { 1 } else { dims[dim] };

    ensure_non_empty(dim_size)?;

    let mut out_dims = if dims.is_empty() {
        vec![1]
    } else {
        dims.to_vec()
    };

    if keepdim {
        if !out_dims.is_empty() {
            out_dims[dim] = 1;
        }
    } else if !out_dims.is_empty() {
        out_dims.remove(dim);
    }

    let values_shape = Shape::new(out_dims);
    let num_out = values_shape.numel();

    let mut values_data = TensorData::zeros_on_device(num_out, tensor.dtype(), tensor.device());
    let mut indices_data = TensorData::zeros_on_device(num_out, DataType::Int64, tensor.device());

    let outer = if dims.is_empty() || dim == 0 {
        1
    } else {
        dims[..dim].iter().product()
    };
    let inner = if dims.is_empty() || dim + 1 >= dims.len() {
        1
    } else {
        dims[dim + 1..].iter().product()
    };
    let outer_stride = dim_size * inner;
    let median_pos = (dim_size - 1) / 2;

    match tensor.dtype() {
        DataType::Float32 => {
            let input = tensor
                .data()
                .as_f32_slice()
                .ok_or_else(|| MinitensorError::internal_error("Failed to get f32 slice"))?;
            let values = values_data.as_f32_slice_mut().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get mutable f32 slice")
            })?;
            let indices = indices_data.as_i64_slice_mut().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get mutable i64 slice")
            })?;

            let mut entries = Vec::with_capacity(dim_size);
            for o in 0..outer {
                for r in 0..inner {
                    entries.clear();
                    for d in 0..dim_size {
                        let idx = o * outer_stride + d * inner + r;
                        entries.push((d, input[idx]));
                    }

                    entries.sort_by(cmp_f32_asc);
                    let (index, value) = entries[median_pos];
                    let base = o * inner + r;
                    values[base] = value;
                    indices[base] = index as i64;
                }
            }
        }
        DataType::Float64 => {
            let input = tensor
                .data()
                .as_f64_slice()
                .ok_or_else(|| MinitensorError::internal_error("Failed to get f64 slice"))?;
            let values = values_data.as_f64_slice_mut().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get mutable f64 slice")
            })?;
            let indices = indices_data.as_i64_slice_mut().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get mutable i64 slice")
            })?;

            let mut entries = Vec::with_capacity(dim_size);
            for o in 0..outer {
                for r in 0..inner {
                    entries.clear();
                    for d in 0..dim_size {
                        let idx = o * outer_stride + d * inner + r;
                        entries.push((d, input[idx]));
                    }

                    entries.sort_by(cmp_f64_asc);
                    let (index, value) = entries[median_pos];
                    let base = o * inner + r;
                    values[base] = value;
                    indices[base] = index as i64;
                }
            }
        }
        DataType::Int32 => {
            let input = tensor
                .data()
                .as_i32_slice()
                .ok_or_else(|| MinitensorError::internal_error("Failed to get i32 slice"))?;
            let values = values_data.as_i32_slice_mut().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get mutable i32 slice")
            })?;
            let indices = indices_data.as_i64_slice_mut().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get mutable i64 slice")
            })?;

            let mut entries = Vec::with_capacity(dim_size);
            for o in 0..outer {
                for r in 0..inner {
                    entries.clear();
                    for d in 0..dim_size {
                        let idx = o * outer_stride + d * inner + r;
                        entries.push((d, input[idx]));
                    }

                    entries.sort_by(cmp_i32_asc);
                    let (index, value) = entries[median_pos];
                    let base = o * inner + r;
                    values[base] = value;
                    indices[base] = index as i64;
                }
            }
        }
        DataType::Int64 => {
            let input = tensor
                .data()
                .as_i64_slice()
                .ok_or_else(|| MinitensorError::internal_error("Failed to get i64 slice"))?;
            let values = values_data.as_i64_slice_mut().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get mutable i64 slice")
            })?;
            let indices = indices_data.as_i64_slice_mut().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get mutable i64 slice")
            })?;

            let mut entries = Vec::with_capacity(dim_size);
            for o in 0..outer {
                for r in 0..inner {
                    entries.clear();
                    for d in 0..dim_size {
                        let idx = o * outer_stride + d * inner + r;
                        entries.push((d, input[idx]));
                    }

                    entries.sort_by(cmp_i64_asc);
                    let (index, value) = entries[median_pos];
                    let base = o * inner + r;
                    values[base] = value;
                    indices[base] = index as i64;
                }
            }
        }
        DataType::Bool => {
            let input = tensor
                .data()
                .as_bool_slice()
                .ok_or_else(|| MinitensorError::internal_error("Failed to get bool slice"))?;
            let values = values_data.as_bool_slice_mut().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get mutable bool slice")
            })?;
            let indices = indices_data.as_i64_slice_mut().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get mutable i64 slice")
            })?;

            let mut entries = Vec::with_capacity(dim_size);
            for o in 0..outer {
                for r in 0..inner {
                    entries.clear();
                    for d in 0..dim_size {
                        let idx = o * outer_stride + d * inner + r;
                        entries.push((d, input[idx]));
                    }

                    entries.sort_by(cmp_bool_asc);
                    let (index, value) = entries[median_pos];
                    let base = o * inner + r;
                    values[base] = value;
                    indices[base] = index as i64;
                }
            }
        }
    }

    let values = Tensor::new(
        Arc::new(values_data),
        values_shape.clone(),
        tensor.dtype(),
        tensor.device(),
        tensor.requires_grad(),
    );

    let indices = Tensor::new(
        Arc::new(indices_data),
        values_shape,
        DataType::Int64,
        tensor.device(),
        false,
    );

    Ok((values, indices))
}

fn normalize_dim(dim: isize, ndim: usize) -> Result<usize> {
    let dim = if dim < 0 { dim + ndim as isize } else { dim };
    if dim < 0 || dim >= ndim as isize {
        Err(MinitensorError::index_error(dim, 0, ndim))
    } else {
        Ok(dim as usize)
    }
}

/// Sum reduction along specified dimensions
pub fn sum(tensor: &Tensor, dim: Option<Vec<isize>>, keepdim: bool) -> Result<Tensor> {
    // Normalise negative dimensions and deduplicate
    let ndim = tensor.ndim() as isize;
    let dim = match dim {
        Some(dims) => {
            let mut normalized = Vec::with_capacity(dims.len());
            for d in dims {
                let d = if d < 0 { d + ndim } else { d };
                if d < 0 || d >= ndim {
                    return Err(MinitensorError::index_error(d, 0, tensor.ndim()));
                }
                normalized.push(d as usize);
            }
            normalized.sort_unstable();
            normalized.dedup();
            Some(normalized)
        }
        None => None,
    };
    let dims_clone = dim.clone();

    let result = match dim {
        None => {
            // Sum all elements
            let result_shape = if keepdim {
                Shape::new(vec![1; tensor.ndim()])
            } else {
                Shape::scalar()
            };

            let mut result_data = TensorData::zeros_on_device(1, tensor.dtype(), tensor.device());

            match tensor.dtype() {
                DataType::Float32 => sum_all_f32(tensor, &mut result_data)?,
                DataType::Float64 => sum_all_f64(tensor, &mut result_data)?,
                DataType::Int32 => sum_all_i32(tensor, &mut result_data)?,
                DataType::Int64 => sum_all_i64(tensor, &mut result_data)?,
                DataType::Bool => {
                    return Err(MinitensorError::invalid_operation(
                        "Sum not supported for boolean tensors",
                    ));
                }
            }

            Tensor::new(
                Arc::new(result_data),
                result_shape,
                tensor.dtype(),
                tensor.device(),
                tensor.requires_grad(),
            )
        }
        Some(dims) => {
            // Sum along specific dimensions
            if dims.is_empty() {
                tensor.clone()
            } else {
                let mut result = tensor.clone();
                if keepdim {
                    for &d in &dims {
                        result = sum_along_dim(&result, d, true)?;
                    }
                } else {
                    for &d in dims.iter().rev() {
                        result = sum_along_dim(&result, d, false)?;
                    }
                }
                result
            }
        }
    };

    if result.requires_grad() {
        let grad_fn = Arc::new(SumBackward {
            input_id: tensor.id(),
            input_shape: tensor.shape().dims().to_vec(),
            dims: dims_clone,
            keepdim,
        });
        let mut result_with_grad = result;
        result_with_grad.set_grad_fn(Some(grad_fn.clone()));
        add_to_graph(&result_with_grad, Some(grad_fn))?;
        Ok(result_with_grad)
    } else {
        Ok(result)
    }
}

/// Numerically stable log-sum-exp reduction along specified dimensions
pub fn logsumexp(tensor: &Tensor, dim: Option<Vec<isize>>, keepdim: bool) -> Result<Tensor> {
    match tensor.dtype() {
        DataType::Float32 | DataType::Float64 => {}
        _ => {
            return Err(MinitensorError::invalid_operation(
                "Logsumexp only supported for floating point tensors",
            ));
        }
    }

    let ndim = tensor.ndim() as isize;
    let dims = match dim {
        Some(dims) => {
            if dims.is_empty() {
                Vec::new()
            } else {
                let mut normalized = Vec::with_capacity(dims.len());
                for d in dims {
                    let d = if d < 0 { d + ndim } else { d };
                    if d < 0 || d >= ndim {
                        return Err(MinitensorError::index_error(d, 0, tensor.ndim()));
                    }
                    normalized.push(d as usize);
                }
                normalized.sort_unstable();
                normalized.dedup();
                normalized
            }
        }
        None => (0..tensor.ndim()).collect(),
    };

    if dims.is_empty() {
        return Ok(tensor.clone());
    }

    let mut max_tensor = tensor.clone();
    for &d in &dims {
        max_tensor = max_along_dim(&max_tensor, d, true)?;
    }
    let max_tensor = max_tensor.detach();

    let shifted = arithmetic::sub(tensor, &max_tensor)?;
    let exp_shifted = activation::exp(&shifted)?;
    let dims_isize: Vec<isize> = dims.iter().map(|&d| d as isize).collect();
    let sum_exp = sum(&exp_shifted, Some(dims_isize), true)?;
    let log_sum = activation::log(&sum_exp)?;
    let mut result = arithmetic::add(&max_tensor, &log_sum)?;

    if !keepdim {
        let mut new_dims = Vec::with_capacity(result.ndim() - dims.len());
        for (idx, &size) in result.shape().dims().iter().enumerate() {
            if dims.binary_search(&idx).is_err() {
                new_dims.push(size);
            }
        }

        let target_shape = if new_dims.is_empty() {
            Shape::scalar()
        } else {
            Shape::new(new_dims)
        };

        result = shape_ops::reshape(&result, target_shape)?;
    }

    Ok(result)
}

/// Product reduction along specified dimensions
pub fn prod(tensor: &Tensor, dim: Option<Vec<isize>>, keepdim: bool) -> Result<Tensor> {
    // Normalise negative dimensions and deduplicate
    let ndim = tensor.ndim() as isize;
    let dim = match dim {
        Some(dims) => {
            let mut normalized = Vec::with_capacity(dims.len());
            for d in dims {
                let d = if d < 0 { d + ndim } else { d };
                if d < 0 || d >= ndim {
                    return Err(MinitensorError::index_error(d, 0, tensor.ndim()));
                }
                normalized.push(d as usize);
            }
            normalized.sort_unstable();
            normalized.dedup();
            Some(normalized)
        }
        None => None,
    };
    let dims_clone = dim.clone();

    let result = match dim {
        None => {
            let result_shape = if keepdim {
                Shape::new(vec![1; tensor.ndim()])
            } else {
                Shape::scalar()
            };

            let mut result_data = TensorData::zeros_on_device(1, tensor.dtype(), tensor.device());
            match tensor.dtype() {
                DataType::Float32 => prod_all_f32(tensor, &mut result_data)?,
                DataType::Float64 => prod_all_f64(tensor, &mut result_data)?,
                DataType::Int32 => prod_all_i32(tensor, &mut result_data)?,
                DataType::Int64 => prod_all_i64(tensor, &mut result_data)?,
                DataType::Bool => prod_all_bool(tensor, &mut result_data)?,
            }

            let requires_grad = tensor.requires_grad() && tensor.dtype() != DataType::Bool;
            Tensor::new(
                Arc::new(result_data),
                result_shape,
                tensor.dtype(),
                tensor.device(),
                requires_grad,
            )
        }
        Some(dims) => {
            if dims.is_empty() {
                tensor.clone()
            } else {
                let mut result = tensor.clone();
                if keepdim {
                    for &d in &dims {
                        result = prod_along_dim(&result, d, true)?;
                    }
                } else {
                    for &d in dims.iter().rev() {
                        result = prod_along_dim(&result, d, false)?;
                    }
                }
                result
            }
        }
    };

    if result.requires_grad() {
        let grad_fn = Arc::new(ProdBackward {
            input: tensor.detach(),
            result: result.clone(),
            input_id: tensor.id(),
            dims: dims_clone,
            keepdim,
        });
        let mut result_with_grad = result;
        result_with_grad.set_grad_fn(Some(grad_fn.clone()));
        add_to_graph(&result_with_grad, Some(grad_fn))?;
        Ok(result_with_grad)
    } else {
        Ok(result)
    }
}

/// Cumulative sum along a specified dimension
pub fn cumsum(tensor: &Tensor, dim: isize) -> Result<Tensor> {
    let dim = normalize_dim(dim, tensor.ndim())?;

    let mut result_data =
        TensorData::uninitialized_on_device(tensor.numel(), tensor.dtype(), tensor.device());

    match tensor.dtype() {
        DataType::Float32 => cumsum_f32(tensor, &mut result_data, dim)?,
        DataType::Float64 => cumsum_f64(tensor, &mut result_data, dim)?,
        DataType::Int32 => cumsum_i32(tensor, &mut result_data, dim)?,
        DataType::Int64 => cumsum_i64(tensor, &mut result_data, dim)?,
        DataType::Bool => {
            return Err(MinitensorError::invalid_operation(
                "Cumsum not supported for boolean tensors",
            ));
        }
    }

    let result = Tensor::new(
        Arc::new(result_data),
        tensor.shape().clone(),
        tensor.dtype(),
        tensor.device(),
        tensor.requires_grad(),
    );

    if result.requires_grad() {
        let grad_fn = Arc::new(CumsumBackward {
            input_id: tensor.id(),
            dim,
        });
        let mut result_with_grad = result;
        result_with_grad.set_grad_fn(Some(grad_fn.clone()));
        add_to_graph(&result_with_grad, Some(grad_fn))?;
        Ok(result_with_grad)
    } else {
        Ok(result)
    }
}

/// Backward helper for cumulative sum
pub fn cumsum_backward(tensor: &Tensor, dim: usize) -> Result<Tensor> {
    if dim >= tensor.ndim() {
        return Err(MinitensorError::index_error(dim as isize, 0, tensor.ndim()));
    }

    let mut result_data =
        TensorData::uninitialized_on_device(tensor.numel(), tensor.dtype(), tensor.device());

    match tensor.dtype() {
        DataType::Float32 => cumsum_backward_f32(tensor, &mut result_data, dim)?,
        DataType::Float64 => cumsum_backward_f64(tensor, &mut result_data, dim)?,
        DataType::Int32 => cumsum_backward_i32(tensor, &mut result_data, dim)?,
        DataType::Int64 => cumsum_backward_i64(tensor, &mut result_data, dim)?,
        DataType::Bool => {
            return Err(MinitensorError::invalid_operation(
                "Cumsum not supported for boolean tensors",
            ));
        }
    }

    Ok(Tensor::new(
        Arc::new(result_data),
        tensor.shape().clone(),
        tensor.dtype(),
        tensor.device(),
        false,
    ))
}

/// Cumulative product along a specified dimension
pub fn cumprod(tensor: &Tensor, dim: isize) -> Result<Tensor> {
    let dim = normalize_dim(dim, tensor.ndim())?;

    let mut result_data =
        TensorData::uninitialized_on_device(tensor.numel(), tensor.dtype(), tensor.device());

    match tensor.dtype() {
        DataType::Float32 => cumprod_f32(tensor, &mut result_data, dim)?,
        DataType::Float64 => cumprod_f64(tensor, &mut result_data, dim)?,
        DataType::Int32 => cumprod_i32(tensor, &mut result_data, dim)?,
        DataType::Int64 => cumprod_i64(tensor, &mut result_data, dim)?,
        DataType::Bool => {
            return Err(MinitensorError::invalid_operation(
                "Cumprod not supported for boolean tensors",
            ));
        }
    }

    let requires_grad =
        tensor.requires_grad() && matches!(tensor.dtype(), DataType::Float32 | DataType::Float64);

    let result = Tensor::new(
        Arc::new(result_data),
        tensor.shape().clone(),
        tensor.dtype(),
        tensor.device(),
        requires_grad,
    );

    if result.requires_grad() {
        let grad_fn = Arc::new(CumprodBackward {
            input_id: tensor.id(),
            input: tensor.clone(),
            output: result.clone(),
            dim,
        });
        let mut result_with_grad = result;
        result_with_grad.set_grad_fn(Some(grad_fn.clone()));
        add_to_graph(&result_with_grad, Some(grad_fn))?;
        Ok(result_with_grad)
    } else {
        Ok(result)
    }
}

/// Backward helper for cumulative product
pub fn cumprod_backward(
    input: &Tensor,
    output: &Tensor,
    grad: &Tensor,
    dim: usize,
) -> Result<Tensor> {
    if dim >= input.ndim() {
        return Err(MinitensorError::index_error(dim as isize, 0, input.ndim()));
    }

    let mut result_data = TensorData::zeros_on_device(input.numel(), input.dtype(), input.device());

    match input.dtype() {
        DataType::Float32 => cumprod_backward_f32(input, output, grad, &mut result_data, dim)?,
        DataType::Float64 => cumprod_backward_f64(input, output, grad, &mut result_data, dim)?,
        _ => {
            return Err(MinitensorError::invalid_operation(
                "Cumprod backward only supported for floating point tensors",
            ));
        }
    }

    Ok(Tensor::new(
        Arc::new(result_data),
        input.shape().clone(),
        input.dtype(),
        input.device(),
        false,
    ))
}

/// Mean reduction along specified dimensions
pub fn mean(tensor: &Tensor, dim: Option<Vec<isize>>, keepdim: bool) -> Result<Tensor> {
    // Normalise negative dimensions and deduplicate
    let ndim = tensor.ndim() as isize;
    let normalized = match dim {
        Some(dims) => {
            let mut normalized = Vec::with_capacity(dims.len());
            for d in dims {
                let d = if d < 0 { d + ndim } else { d };
                if d < 0 || d >= ndim {
                    return Err(MinitensorError::index_error(d, 0, tensor.ndim()));
                }
                normalized.push(d as usize);
            }
            normalized.sort_unstable();
            normalized.dedup();
            Some(normalized)
        }
        None => None,
    };

    let sum_result = sum(
        tensor,
        normalized
            .clone()
            .map(|d| d.iter().map(|&x| x as isize).collect()),
        keepdim,
    )?;

    // Compute the number of elements being averaged
    let num_elements = match &normalized {
        None => tensor.numel() as f64,
        Some(dims) => {
            if dims.is_empty() {
                return Ok(tensor.clone());
            }

            let mut count = 1.0;
            for &d in dims {
                count *= tensor.shape().dims()[d] as f64;
            }
            count
        }
    };

    // Prepare sum tensor and divisor for division
    let (sum_tensor, divisor) = match tensor.dtype() {
        DataType::Float32 => (
            sum_result,
            Tensor::new(
                Arc::new(TensorData::from_vec(
                    vec![num_elements as f32],
                    DataType::Float32,
                    tensor.device(),
                )),
                Shape::scalar(),
                DataType::Float32,
                tensor.device(),
                false,
            ),
        ),
        DataType::Float64 => (
            sum_result,
            Tensor::new(
                Arc::new(TensorData::from_vec(
                    vec![num_elements],
                    DataType::Float64,
                    tensor.device(),
                )),
                Shape::scalar(),
                DataType::Float64,
                tensor.device(),
                false,
            ),
        ),
        DataType::Int32 => (
            sum_result.astype(DataType::Float32)?,
            Tensor::new(
                Arc::new(TensorData::from_vec(
                    vec![num_elements as f32],
                    DataType::Float32,
                    tensor.device(),
                )),
                Shape::scalar(),
                DataType::Float32,
                tensor.device(),
                false,
            ),
        ),
        DataType::Int64 => (
            sum_result.astype(DataType::Float64)?,
            Tensor::new(
                Arc::new(TensorData::from_vec(
                    vec![num_elements],
                    DataType::Float64,
                    tensor.device(),
                )),
                Shape::scalar(),
                DataType::Float64,
                tensor.device(),
                false,
            ),
        ),
        DataType::Bool => {
            return Err(MinitensorError::invalid_operation(
                "Mean not supported for boolean tensors",
            ));
        }
    };

    crate::operations::arithmetic::div(&sum_tensor, &divisor)
}

/// Logical all reduction along specified dimension
pub fn all(tensor: &Tensor, dim: Option<isize>, keepdim: bool) -> Result<Tensor> {
    match dim {
        None => all_all(tensor, keepdim),
        Some(d) => {
            let d = normalize_dim(d, tensor.ndim())?;
            all_along_dim(tensor, d, keepdim)
        }
    }
}

/// Logical any reduction along specified dimension
pub fn any(tensor: &Tensor, dim: Option<isize>, keepdim: bool) -> Result<Tensor> {
    match dim {
        None => any_all(tensor, keepdim),
        Some(d) => {
            let d = normalize_dim(d, tensor.ndim())?;
            any_along_dim(tensor, d, keepdim)
        }
    }
}

fn all_all(tensor: &Tensor, keepdim: bool) -> Result<Tensor> {
    let result_shape = if keepdim {
        Shape::new(vec![1; tensor.ndim()])
    } else {
        Shape::scalar()
    };
    let mut result_data = TensorData::zeros_on_device(1, DataType::Bool, tensor.device());
    let out_slice = result_data
        .as_bool_slice_mut()
        .ok_or_else(|| MinitensorError::internal_error("Failed to get bool slice"))?;
    let all_true = match tensor.dtype() {
        DataType::Float32 => tensor
            .data()
            .as_f32_slice()
            .ok_or_else(|| MinitensorError::internal_error("Expected f32 data"))?
            .par_iter()
            .all(|&x| x != 0.0),
        DataType::Float64 => tensor
            .data()
            .as_f64_slice()
            .ok_or_else(|| MinitensorError::internal_error("Expected f64 data"))?
            .par_iter()
            .all(|&x| x != 0.0),
        DataType::Int32 => tensor
            .data()
            .as_i32_slice()
            .ok_or_else(|| MinitensorError::internal_error("Expected i32 data"))?
            .par_iter()
            .all(|&x| x != 0),
        DataType::Int64 => tensor
            .data()
            .as_i64_slice()
            .ok_or_else(|| MinitensorError::internal_error("Expected i64 data"))?
            .par_iter()
            .all(|&x| x != 0),
        DataType::Bool => tensor
            .data()
            .as_bool_slice()
            .ok_or_else(|| MinitensorError::internal_error("Expected bool data"))?
            .par_iter()
            .all(|&x| x),
    };
    out_slice[0] = all_true;
    Ok(Tensor::new(
        Arc::new(result_data),
        result_shape,
        DataType::Bool,
        tensor.device(),
        false,
    ))
}

fn any_all(tensor: &Tensor, keepdim: bool) -> Result<Tensor> {
    let result_shape = if keepdim {
        Shape::new(vec![1; tensor.ndim()])
    } else {
        Shape::scalar()
    };
    let mut result_data = TensorData::zeros_on_device(1, DataType::Bool, tensor.device());
    let out_slice = result_data
        .as_bool_slice_mut()
        .ok_or_else(|| MinitensorError::internal_error("Failed to get bool slice"))?;
    let any_true = match tensor.dtype() {
        DataType::Float32 => tensor
            .data()
            .as_f32_slice()
            .ok_or_else(|| MinitensorError::internal_error("Expected f32 data"))?
            .par_iter()
            .any(|&x| x != 0.0),
        DataType::Float64 => tensor
            .data()
            .as_f64_slice()
            .ok_or_else(|| MinitensorError::internal_error("Expected f64 data"))?
            .par_iter()
            .any(|&x| x != 0.0),
        DataType::Int32 => tensor
            .data()
            .as_i32_slice()
            .ok_or_else(|| MinitensorError::internal_error("Expected i32 data"))?
            .par_iter()
            .any(|&x| x != 0),
        DataType::Int64 => tensor
            .data()
            .as_i64_slice()
            .ok_or_else(|| MinitensorError::internal_error("Expected i64 data"))?
            .par_iter()
            .any(|&x| x != 0),
        DataType::Bool => tensor
            .data()
            .as_bool_slice()
            .ok_or_else(|| MinitensorError::internal_error("Expected bool data"))?
            .par_iter()
            .any(|&x| x),
    };
    out_slice[0] = any_true;
    Ok(Tensor::new(
        Arc::new(result_data),
        result_shape,
        DataType::Bool,
        tensor.device(),
        false,
    ))
}

fn all_along_dim(tensor: &Tensor, dim: usize, keepdim: bool) -> Result<Tensor> {
    if dim >= tensor.ndim() {
        return Err(MinitensorError::index_error(dim as isize, 0, tensor.ndim()));
    }

    let input_shape = tensor.shape().dims();
    let mut output_shape = input_shape.to_vec();
    if keepdim {
        output_shape[dim] = 1;
    } else {
        output_shape.remove(dim);
    }
    let output_shape_obj = Shape::new(output_shape.clone());
    let mut result_data =
        TensorData::zeros_on_device(output_shape_obj.numel(), DataType::Bool, tensor.device());

    let dim_size = input_shape[dim];
    let _outer = input_shape[..dim].iter().product::<usize>();
    let inner = input_shape[dim + 1..].iter().product::<usize>();
    let outer_stride = dim_size * inner;

    match tensor.dtype() {
        DataType::Float32 => {
            let input = tensor
                .data()
                .as_f32_slice()
                .ok_or_else(|| MinitensorError::internal_error("Failed to get f32 slice"))?;
            let output = result_data.as_bool_slice_mut().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get mutable bool slice")
            })?;
            output.par_iter_mut().enumerate().for_each(|(idx, out)| {
                let o = idx / inner;
                let r = idx % inner;
                let mut val = true;
                for d in 0..dim_size {
                    let in_idx = o * outer_stride + d * inner + r;
                    if input[in_idx] == 0.0 {
                        val = false;
                        break;
                    }
                }
                *out = val;
            });
        }
        DataType::Float64 => {
            let input = tensor
                .data()
                .as_f64_slice()
                .ok_or_else(|| MinitensorError::internal_error("Failed to get f64 slice"))?;
            let output = result_data.as_bool_slice_mut().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get mutable bool slice")
            })?;
            output.par_iter_mut().enumerate().for_each(|(idx, out)| {
                let o = idx / inner;
                let r = idx % inner;
                let mut val = true;
                for d in 0..dim_size {
                    let in_idx = o * outer_stride + d * inner + r;
                    if input[in_idx] == 0.0 {
                        val = false;
                        break;
                    }
                }
                *out = val;
            });
        }
        DataType::Int32 => {
            let input = tensor
                .data()
                .as_i32_slice()
                .ok_or_else(|| MinitensorError::internal_error("Failed to get i32 slice"))?;
            let output = result_data.as_bool_slice_mut().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get mutable bool slice")
            })?;
            output.par_iter_mut().enumerate().for_each(|(idx, out)| {
                let o = idx / inner;
                let r = idx % inner;
                let mut val = true;
                for d in 0..dim_size {
                    let in_idx = o * outer_stride + d * inner + r;
                    if input[in_idx] == 0 {
                        val = false;
                        break;
                    }
                }
                *out = val;
            });
        }
        DataType::Int64 => {
            let input = tensor
                .data()
                .as_i64_slice()
                .ok_or_else(|| MinitensorError::internal_error("Failed to get i64 slice"))?;
            let output = result_data.as_bool_slice_mut().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get mutable bool slice")
            })?;
            output.par_iter_mut().enumerate().for_each(|(idx, out)| {
                let o = idx / inner;
                let r = idx % inner;
                let mut val = true;
                for d in 0..dim_size {
                    let in_idx = o * outer_stride + d * inner + r;
                    if input[in_idx] == 0 {
                        val = false;
                        break;
                    }
                }
                *out = val;
            });
        }
        DataType::Bool => {
            let input = tensor
                .data()
                .as_bool_slice()
                .ok_or_else(|| MinitensorError::internal_error("Failed to get bool slice"))?;
            let output = result_data.as_bool_slice_mut().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get mutable bool slice")
            })?;
            output.par_iter_mut().enumerate().for_each(|(idx, out)| {
                let o = idx / inner;
                let r = idx % inner;
                let mut val = true;
                for d in 0..dim_size {
                    let in_idx = o * outer_stride + d * inner + r;
                    if !input[in_idx] {
                        val = false;
                        break;
                    }
                }
                *out = val;
            });
        }
    }

    Ok(Tensor::new(
        Arc::new(result_data),
        output_shape_obj,
        DataType::Bool,
        tensor.device(),
        false,
    ))
}

fn any_along_dim(tensor: &Tensor, dim: usize, keepdim: bool) -> Result<Tensor> {
    if dim >= tensor.ndim() {
        return Err(MinitensorError::index_error(dim as isize, 0, tensor.ndim()));
    }

    let input_shape = tensor.shape().dims();
    let mut output_shape = input_shape.to_vec();
    if keepdim {
        output_shape[dim] = 1;
    } else {
        output_shape.remove(dim);
    }
    let output_shape_obj = Shape::new(output_shape.clone());
    let mut result_data =
        TensorData::zeros_on_device(output_shape_obj.numel(), DataType::Bool, tensor.device());

    let dim_size = input_shape[dim];
    let _outer = input_shape[..dim].iter().product::<usize>();
    let inner = input_shape[dim + 1..].iter().product::<usize>();
    let outer_stride = dim_size * inner;

    match tensor.dtype() {
        DataType::Float32 => {
            let input = tensor
                .data()
                .as_f32_slice()
                .ok_or_else(|| MinitensorError::internal_error("Failed to get f32 slice"))?;
            let output = result_data.as_bool_slice_mut().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get mutable bool slice")
            })?;
            output.par_iter_mut().enumerate().for_each(|(idx, out)| {
                let o = idx / inner;
                let r = idx % inner;
                let mut val = false;
                for d in 0..dim_size {
                    let in_idx = o * outer_stride + d * inner + r;
                    if input[in_idx] != 0.0 {
                        val = true;
                        break;
                    }
                }
                *out = val;
            });
        }
        DataType::Float64 => {
            let input = tensor
                .data()
                .as_f64_slice()
                .ok_or_else(|| MinitensorError::internal_error("Failed to get f64 slice"))?;
            let output = result_data.as_bool_slice_mut().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get mutable bool slice")
            })?;
            output.par_iter_mut().enumerate().for_each(|(idx, out)| {
                let o = idx / inner;
                let r = idx % inner;
                let mut val = false;
                for d in 0..dim_size {
                    let in_idx = o * outer_stride + d * inner + r;
                    if input[in_idx] != 0.0 {
                        val = true;
                        break;
                    }
                }
                *out = val;
            });
        }
        DataType::Int32 => {
            let input = tensor
                .data()
                .as_i32_slice()
                .ok_or_else(|| MinitensorError::internal_error("Failed to get i32 slice"))?;
            let output = result_data.as_bool_slice_mut().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get mutable bool slice")
            })?;
            output.par_iter_mut().enumerate().for_each(|(idx, out)| {
                let o = idx / inner;
                let r = idx % inner;
                let mut val = false;
                for d in 0..dim_size {
                    let in_idx = o * outer_stride + d * inner + r;
                    if input[in_idx] != 0 {
                        val = true;
                        break;
                    }
                }
                *out = val;
            });
        }
        DataType::Int64 => {
            let input = tensor
                .data()
                .as_i64_slice()
                .ok_or_else(|| MinitensorError::internal_error("Failed to get i64 slice"))?;
            let output = result_data.as_bool_slice_mut().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get mutable bool slice")
            })?;
            output.par_iter_mut().enumerate().for_each(|(idx, out)| {
                let o = idx / inner;
                let r = idx % inner;
                let mut val = false;
                for d in 0..dim_size {
                    let in_idx = o * outer_stride + d * inner + r;
                    if input[in_idx] != 0 {
                        val = true;
                        break;
                    }
                }
                *out = val;
            });
        }
        DataType::Bool => {
            let input = tensor
                .data()
                .as_bool_slice()
                .ok_or_else(|| MinitensorError::internal_error("Failed to get bool slice"))?;
            let output = result_data.as_bool_slice_mut().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get mutable bool slice")
            })?;
            output.par_iter_mut().enumerate().for_each(|(idx, out)| {
                let o = idx / inner;
                let r = idx % inner;
                let mut val = false;
                for d in 0..dim_size {
                    let in_idx = o * outer_stride + d * inner + r;
                    if input[in_idx] {
                        val = true;
                        break;
                    }
                }
                *out = val;
            });
        }
    }

    Ok(Tensor::new(
        Arc::new(result_data),
        output_shape_obj,
        DataType::Bool,
        tensor.device(),
        false,
    ))
}

/// Maximum value along specified dimension
pub fn max(tensor: &Tensor, dim: Option<isize>, keepdim: bool) -> Result<Tensor> {
    match dim {
        None => {
            // Find global maximum
            let result_shape = if keepdim {
                Shape::new(vec![1; tensor.ndim()])
            } else {
                Shape::scalar()
            };

            let mut result_data = TensorData::zeros_on_device(1, tensor.dtype(), tensor.device());

            match tensor.dtype() {
                DataType::Float32 => max_all_f32(tensor, &mut result_data)?,
                DataType::Float64 => max_all_f64(tensor, &mut result_data)?,
                DataType::Int32 => max_all_i32(tensor, &mut result_data)?,
                DataType::Int64 => max_all_i64(tensor, &mut result_data)?,
                DataType::Bool => max_all_bool(tensor, &mut result_data)?,
            }

            Ok(Tensor::new(
                Arc::new(result_data),
                result_shape,
                tensor.dtype(),
                tensor.device(),
                tensor.requires_grad(),
            ))
        }
        Some(d) => {
            let d = normalize_dim(d, tensor.ndim())?;
            max_along_dim(tensor, d, keepdim)
        }
    }
}

/// Minimum value along specified dimension
pub fn min(tensor: &Tensor, dim: Option<isize>, keepdim: bool) -> Result<Tensor> {
    match dim {
        None => {
            // Find global minimum
            let result_shape = if keepdim {
                Shape::new(vec![1; tensor.ndim()])
            } else {
                Shape::scalar()
            };

            let mut result_data = TensorData::zeros_on_device(1, tensor.dtype(), tensor.device());

            match tensor.dtype() {
                DataType::Float32 => min_all_f32(tensor, &mut result_data)?,
                DataType::Float64 => min_all_f64(tensor, &mut result_data)?,
                DataType::Int32 => min_all_i32(tensor, &mut result_data)?,
                DataType::Int64 => min_all_i64(tensor, &mut result_data)?,
                DataType::Bool => min_all_bool(tensor, &mut result_data)?,
            }

            Ok(Tensor::new(
                Arc::new(result_data),
                result_shape,
                tensor.dtype(),
                tensor.device(),
                tensor.requires_grad(),
            ))
        }
        Some(d) => {
            let d = normalize_dim(d, tensor.ndim())?;
            min_along_dim(tensor, d, keepdim)
        }
    }
}

/// Argument of maximum value along specified dimension
pub fn argmax(tensor: &Tensor, dim: Option<isize>, keepdim: bool) -> Result<Tensor> {
    match dim {
        None => {
            // Find global argmax
            let result_shape = if keepdim {
                Shape::new(vec![1; tensor.ndim()])
            } else {
                Shape::scalar()
            };

            let mut result_data = TensorData::zeros_on_device(1, DataType::Int64, tensor.device());

            match tensor.dtype() {
                DataType::Float32 => argmax_all_f32(tensor, &mut result_data)?,
                DataType::Float64 => argmax_all_f64(tensor, &mut result_data)?,
                DataType::Int32 => argmax_all_i32(tensor, &mut result_data)?,
                DataType::Int64 => argmax_all_i64(tensor, &mut result_data)?,
                DataType::Bool => argmax_all_bool(tensor, &mut result_data)?,
            }

            Ok(Tensor::new(
                Arc::new(result_data),
                result_shape,
                DataType::Int64,
                tensor.device(),
                false, // argmax doesn't require gradients
            ))
        }
        Some(d) => {
            let d = normalize_dim(d, tensor.ndim())?;
            argmax_along_dim(tensor, d, keepdim)
        }
    }
}

/// Argument of minimum value along specified dimension
pub fn argmin(tensor: &Tensor, dim: Option<isize>, keepdim: bool) -> Result<Tensor> {
    match dim {
        None => {
            // Find global argmin
            let result_shape = if keepdim {
                Shape::new(vec![1; tensor.ndim()])
            } else {
                Shape::scalar()
            };

            let mut result_data = TensorData::zeros_on_device(1, DataType::Int64, tensor.device());

            match tensor.dtype() {
                DataType::Float32 => argmin_all_f32(tensor, &mut result_data)?,
                DataType::Float64 => argmin_all_f64(tensor, &mut result_data)?,
                DataType::Int32 => argmin_all_i32(tensor, &mut result_data)?,
                DataType::Int64 => argmin_all_i64(tensor, &mut result_data)?,
                DataType::Bool => argmin_all_bool(tensor, &mut result_data)?,
            }

            Ok(Tensor::new(
                Arc::new(result_data),
                result_shape,
                DataType::Int64,
                tensor.device(),
                false, // argmin doesn't require gradients
            ))
        }
        Some(d) => {
            let d = normalize_dim(d, tensor.ndim())?;
            argmin_along_dim(tensor, d, keepdim)
        }
    }
}

/// Return the top-``k`` values and their indices along ``dim``
pub fn topk(
    tensor: &Tensor,
    k: usize,
    dim: Option<isize>,
    largest: bool,
    sorted: bool,
) -> Result<(Tensor, Tensor)> {
    let ndim = tensor.ndim();

    let axis = if ndim == 0 {
        match dim {
            Some(d) if d == 0 || d == -1 => 0,
            Some(d) => return Err(MinitensorError::index_error(d, 0, 1)),
            None => 0,
        }
    } else {
        let dim_value = dim.unwrap_or(-1);
        normalize_dim(dim_value, ndim)?
    };

    let dims = tensor.shape().dims();
    let dim_size = if dims.is_empty() { 1 } else { dims[axis] };

    if k > dim_size {
        return Err(MinitensorError::invalid_argument(format!(
            "selected index k out of range for dimension {axis} with size {dim_size}"
        )));
    }

    let output_dims = if dims.is_empty() {
        vec![k]
    } else {
        let mut dims_vec = dims.to_vec();
        dims_vec[axis] = k;
        dims_vec
    };

    let values_shape = Shape::new(output_dims.clone());
    let indices_shape = Shape::new(output_dims);

    let num_out = values_shape.numel();
    let mut values_data = TensorData::zeros_on_device(num_out, tensor.dtype(), tensor.device());
    let mut indices_data = TensorData::zeros_on_device(num_out, DataType::Int64, tensor.device());

    if k == 0 || num_out == 0 {
        let values = Tensor::new(
            Arc::new(values_data),
            values_shape,
            tensor.dtype(),
            tensor.device(),
            tensor.requires_grad(),
        );
        let indices = Tensor::new(
            Arc::new(indices_data),
            indices_shape,
            DataType::Int64,
            tensor.device(),
            false,
        );
        return Ok((values, indices));
    }

    let outer = if dims.is_empty() || axis == 0 {
        1
    } else {
        dims[..axis].iter().product()
    };
    let inner = if dims.is_empty() || axis + 1 >= dims.len() {
        1
    } else {
        dims[axis + 1..].iter().product()
    };
    let outer_stride = dim_size * inner;

    match tensor.dtype() {
        DataType::Float32 => {
            let input = tensor
                .data()
                .as_f32_slice()
                .ok_or_else(|| MinitensorError::internal_error("Failed to get f32 slice"))?;
            let values = values_data.as_f32_slice_mut().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get mutable f32 slice")
            })?;
            let indices = indices_data.as_i64_slice_mut().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get mutable i64 slice")
            })?;

            let mut entries = Vec::with_capacity(dim_size);
            for o in 0..outer {
                for r in 0..inner {
                    entries.clear();
                    for d in 0..dim_size {
                        let idx = o * outer_stride + d * inner + r;
                        entries.push((d, input[idx]));
                    }

                    if sorted {
                        if largest {
                            entries.sort_by(cmp_f32_desc);
                        } else {
                            entries.sort_by(cmp_f32_asc);
                        }
                    } else if k < dim_size {
                        if largest {
                            entries.select_nth_unstable_by(k - 1, cmp_f32_desc);
                        } else {
                            entries.select_nth_unstable_by(k - 1, cmp_f32_asc);
                        }
                    }

                    let base = (o * inner + r) * k;
                    for j in 0..k {
                        let (index, value) = entries[j];
                        values[base + j] = value;
                        indices[base + j] = index as i64;
                    }
                }
            }
        }
        DataType::Float64 => {
            let input = tensor
                .data()
                .as_f64_slice()
                .ok_or_else(|| MinitensorError::internal_error("Failed to get f64 slice"))?;
            let values = values_data.as_f64_slice_mut().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get mutable f64 slice")
            })?;
            let indices = indices_data.as_i64_slice_mut().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get mutable i64 slice")
            })?;

            let mut entries = Vec::with_capacity(dim_size);
            for o in 0..outer {
                for r in 0..inner {
                    entries.clear();
                    for d in 0..dim_size {
                        let idx = o * outer_stride + d * inner + r;
                        entries.push((d, input[idx]));
                    }

                    if sorted {
                        if largest {
                            entries.sort_by(cmp_f64_desc);
                        } else {
                            entries.sort_by(cmp_f64_asc);
                        }
                    } else if k < dim_size {
                        if largest {
                            entries.select_nth_unstable_by(k - 1, cmp_f64_desc);
                        } else {
                            entries.select_nth_unstable_by(k - 1, cmp_f64_asc);
                        }
                    }

                    let base = (o * inner + r) * k;
                    for j in 0..k {
                        let (index, value) = entries[j];
                        values[base + j] = value;
                        indices[base + j] = index as i64;
                    }
                }
            }
        }
        DataType::Int32 => {
            let input = tensor
                .data()
                .as_i32_slice()
                .ok_or_else(|| MinitensorError::internal_error("Failed to get i32 slice"))?;
            let values = values_data.as_i32_slice_mut().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get mutable i32 slice")
            })?;
            let indices = indices_data.as_i64_slice_mut().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get mutable i64 slice")
            })?;

            let mut entries = Vec::with_capacity(dim_size);
            for o in 0..outer {
                for r in 0..inner {
                    entries.clear();
                    for d in 0..dim_size {
                        let idx = o * outer_stride + d * inner + r;
                        entries.push((d, input[idx]));
                    }

                    if sorted {
                        if largest {
                            entries.sort_by(cmp_i32_desc);
                        } else {
                            entries.sort_by(cmp_i32_asc);
                        }
                    } else if k < dim_size {
                        if largest {
                            entries.select_nth_unstable_by(k - 1, cmp_i32_desc);
                        } else {
                            entries.select_nth_unstable_by(k - 1, cmp_i32_asc);
                        }
                    }

                    let base = (o * inner + r) * k;
                    for j in 0..k {
                        let (index, value) = entries[j];
                        values[base + j] = value;
                        indices[base + j] = index as i64;
                    }
                }
            }
        }
        DataType::Int64 => {
            let input = tensor
                .data()
                .as_i64_slice()
                .ok_or_else(|| MinitensorError::internal_error("Failed to get i64 slice"))?;
            let values = values_data.as_i64_slice_mut().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get mutable i64 slice")
            })?;
            let indices = indices_data.as_i64_slice_mut().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get mutable i64 slice")
            })?;

            let mut entries = Vec::with_capacity(dim_size);
            for o in 0..outer {
                for r in 0..inner {
                    entries.clear();
                    for d in 0..dim_size {
                        let idx = o * outer_stride + d * inner + r;
                        entries.push((d, input[idx]));
                    }

                    if sorted {
                        if largest {
                            entries.sort_by(cmp_i64_desc);
                        } else {
                            entries.sort_by(cmp_i64_asc);
                        }
                    } else if k < dim_size {
                        if largest {
                            entries.select_nth_unstable_by(k - 1, cmp_i64_desc);
                        } else {
                            entries.select_nth_unstable_by(k - 1, cmp_i64_asc);
                        }
                    }

                    let base = (o * inner + r) * k;
                    for j in 0..k {
                        let (index, value) = entries[j];
                        values[base + j] = value;
                        indices[base + j] = index as i64;
                    }
                }
            }
        }
        DataType::Bool => {
            let input = tensor
                .data()
                .as_bool_slice()
                .ok_or_else(|| MinitensorError::internal_error("Failed to get bool slice"))?;
            let values = values_data.as_bool_slice_mut().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get mutable bool slice")
            })?;
            let indices = indices_data.as_i64_slice_mut().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get mutable i64 slice")
            })?;

            let mut entries = Vec::with_capacity(dim_size);
            for o in 0..outer {
                for r in 0..inner {
                    entries.clear();
                    for d in 0..dim_size {
                        let idx = o * outer_stride + d * inner + r;
                        entries.push((d, input[idx]));
                    }

                    if sorted {
                        if largest {
                            entries.sort_by(cmp_bool_desc);
                        } else {
                            entries.sort_by(cmp_bool_asc);
                        }
                    } else if k < dim_size {
                        if largest {
                            entries.select_nth_unstable_by(k - 1, cmp_bool_desc);
                        } else {
                            entries.select_nth_unstable_by(k - 1, cmp_bool_asc);
                        }
                    }

                    let base = (o * inner + r) * k;
                    for j in 0..k {
                        let (index, value) = entries[j];
                        values[base + j] = value;
                        indices[base + j] = index as i64;
                    }
                }
            }
        }
    }

    let values = Tensor::new(
        Arc::new(values_data),
        values_shape,
        tensor.dtype(),
        tensor.device(),
        tensor.requires_grad(),
    );
    let indices = Tensor::new(
        Arc::new(indices_data),
        indices_shape,
        DataType::Int64,
        tensor.device(),
        false,
    );

    Ok((values, indices))
}

pub fn sort(
    tensor: &Tensor,
    dim: Option<isize>,
    descending: bool,
    stable: bool,
) -> Result<(Tensor, Tensor)> {
    let ndim = tensor.ndim();

    let axis = if ndim == 0 {
        match dim {
            Some(d) if d == 0 || d == -1 => 0,
            Some(d) => return Err(MinitensorError::index_error(d, 0, 1)),
            None => 0,
        }
    } else {
        let dim_value = dim.unwrap_or(-1);
        normalize_dim(dim_value, ndim)?
    };

    if tensor.shape().dims().is_empty() {
        let mut values_data = TensorData::zeros_on_device(1, tensor.dtype(), tensor.device());
        let mut indices_data = TensorData::zeros_on_device(1, DataType::Int64, tensor.device());

        match tensor.dtype() {
            DataType::Float32 => {
                let src = tensor
                    .data()
                    .as_f32_slice()
                    .ok_or_else(|| MinitensorError::internal_error("Failed to get f32 slice"))?;
                let dst = values_data.as_f32_slice_mut().ok_or_else(|| {
                    MinitensorError::internal_error("Failed to get mutable f32 slice")
                })?;
                dst[0] = src[0];
            }
            DataType::Float64 => {
                let src = tensor
                    .data()
                    .as_f64_slice()
                    .ok_or_else(|| MinitensorError::internal_error("Failed to get f64 slice"))?;
                let dst = values_data.as_f64_slice_mut().ok_or_else(|| {
                    MinitensorError::internal_error("Failed to get mutable f64 slice")
                })?;
                dst[0] = src[0];
            }
            DataType::Int32 => {
                let src = tensor
                    .data()
                    .as_i32_slice()
                    .ok_or_else(|| MinitensorError::internal_error("Failed to get i32 slice"))?;
                let dst = values_data.as_i32_slice_mut().ok_or_else(|| {
                    MinitensorError::internal_error("Failed to get mutable i32 slice")
                })?;
                dst[0] = src[0];
            }
            DataType::Int64 => {
                let src = tensor
                    .data()
                    .as_i64_slice()
                    .ok_or_else(|| MinitensorError::internal_error("Failed to get i64 slice"))?;
                let dst = values_data.as_i64_slice_mut().ok_or_else(|| {
                    MinitensorError::internal_error("Failed to get mutable i64 slice")
                })?;
                dst[0] = src[0];
            }
            DataType::Bool => {
                let src = tensor
                    .data()
                    .as_bool_slice()
                    .ok_or_else(|| MinitensorError::internal_error("Failed to get bool slice"))?;
                let dst = values_data.as_bool_slice_mut().ok_or_else(|| {
                    MinitensorError::internal_error("Failed to get mutable bool slice")
                })?;
                dst[0] = src[0];
            }
        }

        let indices = indices_data
            .as_i64_slice_mut()
            .ok_or_else(|| MinitensorError::internal_error("Failed to get mutable i64 slice"))?;
        indices[0] = 0;

        let values = Tensor::new(
            Arc::new(values_data),
            Shape::scalar(),
            tensor.dtype(),
            tensor.device(),
            tensor.requires_grad(),
        );
        let indices = Tensor::new(
            Arc::new(indices_data),
            Shape::scalar(),
            DataType::Int64,
            tensor.device(),
            false,
        );
        return Ok((values, indices));
    }

    let dims = tensor.shape().dims();
    let dim_size = dims[axis];

    let mut values_data =
        TensorData::zeros_on_device(tensor.numel(), tensor.dtype(), tensor.device());
    let mut indices_data =
        TensorData::zeros_on_device(tensor.numel(), DataType::Int64, tensor.device());

    let outer = if axis == 0 {
        1
    } else {
        dims[..axis].iter().product()
    };
    let inner = if axis + 1 >= dims.len() {
        1
    } else {
        dims[axis + 1..].iter().product()
    };
    let outer_stride = dim_size * inner;

    match tensor.dtype() {
        DataType::Float32 => {
            let input = tensor
                .data()
                .as_f32_slice()
                .ok_or_else(|| MinitensorError::internal_error("Failed to get f32 slice"))?;
            let values = values_data.as_f32_slice_mut().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get mutable f32 slice")
            })?;
            let indices = indices_data.as_i64_slice_mut().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get mutable i64 slice")
            })?;

            let mut entries = Vec::with_capacity(dim_size);
            for o in 0..outer {
                for r in 0..inner {
                    entries.clear();
                    for d in 0..dim_size {
                        let idx = o * outer_stride + d * inner + r;
                        entries.push((d, input[idx]));
                    }

                    if stable {
                        if descending {
                            entries.sort_by(cmp_f32_desc);
                        } else {
                            entries.sort_by(cmp_f32_asc);
                        }
                    } else if descending {
                        entries.sort_unstable_by(cmp_f32_desc);
                    } else {
                        entries.sort_unstable_by(cmp_f32_asc);
                    }

                    let base = o * outer_stride + r;
                    for (j, (index, value)) in entries.iter().enumerate() {
                        let offset = base + j * inner;
                        values[offset] = *value;
                        indices[offset] = *index as i64;
                    }
                }
            }
        }
        DataType::Float64 => {
            let input = tensor
                .data()
                .as_f64_slice()
                .ok_or_else(|| MinitensorError::internal_error("Failed to get f64 slice"))?;
            let values = values_data.as_f64_slice_mut().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get mutable f64 slice")
            })?;
            let indices = indices_data.as_i64_slice_mut().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get mutable i64 slice")
            })?;

            let mut entries = Vec::with_capacity(dim_size);
            for o in 0..outer {
                for r in 0..inner {
                    entries.clear();
                    for d in 0..dim_size {
                        let idx = o * outer_stride + d * inner + r;
                        entries.push((d, input[idx]));
                    }

                    if stable {
                        if descending {
                            entries.sort_by(cmp_f64_desc);
                        } else {
                            entries.sort_by(cmp_f64_asc);
                        }
                    } else if descending {
                        entries.sort_unstable_by(cmp_f64_desc);
                    } else {
                        entries.sort_unstable_by(cmp_f64_asc);
                    }

                    let base = o * outer_stride + r;
                    for (j, (index, value)) in entries.iter().enumerate() {
                        let offset = base + j * inner;
                        values[offset] = *value;
                        indices[offset] = *index as i64;
                    }
                }
            }
        }
        DataType::Int32 => {
            let input = tensor
                .data()
                .as_i32_slice()
                .ok_or_else(|| MinitensorError::internal_error("Failed to get i32 slice"))?;
            let values = values_data.as_i32_slice_mut().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get mutable i32 slice")
            })?;
            let indices = indices_data.as_i64_slice_mut().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get mutable i64 slice")
            })?;

            let mut entries = Vec::with_capacity(dim_size);
            for o in 0..outer {
                for r in 0..inner {
                    entries.clear();
                    for d in 0..dim_size {
                        let idx = o * outer_stride + d * inner + r;
                        entries.push((d, input[idx]));
                    }

                    if stable {
                        if descending {
                            entries.sort_by(cmp_i32_desc);
                        } else {
                            entries.sort_by(cmp_i32_asc);
                        }
                    } else if descending {
                        entries.sort_unstable_by(cmp_i32_desc);
                    } else {
                        entries.sort_unstable_by(cmp_i32_asc);
                    }

                    let base = o * outer_stride + r;
                    for (j, (index, value)) in entries.iter().enumerate() {
                        let offset = base + j * inner;
                        values[offset] = *value;
                        indices[offset] = *index as i64;
                    }
                }
            }
        }
        DataType::Int64 => {
            let input = tensor
                .data()
                .as_i64_slice()
                .ok_or_else(|| MinitensorError::internal_error("Failed to get i64 slice"))?;
            let values = values_data.as_i64_slice_mut().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get mutable i64 slice")
            })?;
            let indices = indices_data.as_i64_slice_mut().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get mutable i64 slice")
            })?;

            let mut entries = Vec::with_capacity(dim_size);
            for o in 0..outer {
                for r in 0..inner {
                    entries.clear();
                    for d in 0..dim_size {
                        let idx = o * outer_stride + d * inner + r;
                        entries.push((d, input[idx]));
                    }

                    if stable {
                        if descending {
                            entries.sort_by(cmp_i64_desc);
                        } else {
                            entries.sort_by(cmp_i64_asc);
                        }
                    } else if descending {
                        entries.sort_unstable_by(cmp_i64_desc);
                    } else {
                        entries.sort_unstable_by(cmp_i64_asc);
                    }

                    let base = o * outer_stride + r;
                    for (j, (index, value)) in entries.iter().enumerate() {
                        let offset = base + j * inner;
                        values[offset] = *value;
                        indices[offset] = *index as i64;
                    }
                }
            }
        }
        DataType::Bool => {
            let input = tensor
                .data()
                .as_bool_slice()
                .ok_or_else(|| MinitensorError::internal_error("Failed to get bool slice"))?;
            let values = values_data.as_bool_slice_mut().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get mutable bool slice")
            })?;
            let indices = indices_data.as_i64_slice_mut().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get mutable i64 slice")
            })?;

            let mut entries = Vec::with_capacity(dim_size);
            for o in 0..outer {
                for r in 0..inner {
                    entries.clear();
                    for d in 0..dim_size {
                        let idx = o * outer_stride + d * inner + r;
                        entries.push((d, input[idx]));
                    }

                    if stable {
                        if descending {
                            entries.sort_by(cmp_bool_desc);
                        } else {
                            entries.sort_by(cmp_bool_asc);
                        }
                    } else if descending {
                        entries.sort_unstable_by(cmp_bool_desc);
                    } else {
                        entries.sort_unstable_by(cmp_bool_asc);
                    }

                    let base = o * outer_stride + r;
                    for (j, (index, value)) in entries.iter().enumerate() {
                        let offset = base + j * inner;
                        values[offset] = *value;
                        indices[offset] = *index as i64;
                    }
                }
            }
        }
    }

    let values = Tensor::new(
        Arc::new(values_data),
        tensor.shape().clone(),
        tensor.dtype(),
        tensor.device(),
        tensor.requires_grad(),
    );
    let indices = Tensor::new(
        Arc::new(indices_data),
        tensor.shape().clone(),
        DataType::Int64,
        tensor.device(),
        false,
    );

    Ok((values, indices))
}

pub fn argsort(
    tensor: &Tensor,
    dim: Option<isize>,
    descending: bool,
    stable: bool,
) -> Result<Tensor> {
    let (_, indices) = sort(tensor, dim, descending, stable)?;
    Ok(indices)
}

/// Standard deviation along specified dimension
pub fn std(tensor: &Tensor, dim: Option<isize>, keepdim: bool, unbiased: bool) -> Result<Tensor> {
    let variance = var(tensor, dim, keepdim, unbiased)?;
    crate::operations::activation::sqrt(&variance)
}

/// Variance along specified dimension
pub fn var(tensor: &Tensor, dim: Option<isize>, keepdim: bool, unbiased: bool) -> Result<Tensor> {
    if !tensor.dtype().is_float() {
        return Err(MinitensorError::invalid_operation(
            "Variance only supported for floating point tensors",
        ));
    }

    // Compute mean
    let mean_tensor = mean(tensor, dim.clone().map(|d| vec![d]), keepdim)?;

    // Compute (x - mean)^2
    let diff = crate::operations::arithmetic::sub(tensor, &mean_tensor)?;
    let squared_diff = crate::operations::arithmetic::mul(&diff, &diff)?;

    // Compute mean of squared differences
    let variance = mean(&squared_diff, dim.map(|d| vec![d]), keepdim)?;

    if !unbiased {
        return Ok(variance);
    }

    let sample_count = match dim {
        None => tensor.numel() as f64,
        Some(d) => {
            let axis = normalize_dim(d, tensor.ndim())?;
            tensor.shape().dims()[axis] as f64
        }
    };

    if sample_count == 0.0 {
        return Ok(variance);
    }

    let correction = sample_count / (sample_count - 1.0);

    let correction_tensor = match variance.dtype() {
        DataType::Float32 => Tensor::new(
            Arc::new(TensorData::from_vec_f32(
                vec![correction as f32],
                variance.device(),
            )),
            Shape::scalar(),
            DataType::Float32,
            variance.device(),
            false,
        ),
        DataType::Float64 => Tensor::new(
            Arc::new(TensorData::from_vec_f64(
                vec![correction],
                variance.device(),
            )),
            Shape::scalar(),
            DataType::Float64,
            variance.device(),
            false,
        ),
        _ => unreachable!("variance is only defined for floating point tensors"),
    };

    crate::operations::arithmetic::mul(&variance, &correction_tensor)
}

// Helper functions for type-specific operations

fn prod_all_f32(tensor: &Tensor, result_data: &mut TensorData) -> Result<()> {
    let data = tensor
        .data()
        .as_f32_slice()
        .ok_or_else(|| MinitensorError::internal_error("Failed to get f32 slice"))?;

    let prod: f32 = if data.len() >= 1024 {
        data.par_chunks(8192).map(simd_prod_f32).product::<f32>()
    } else {
        simd_prod_f32(data)
    };

    let result_slice = result_data
        .as_f32_slice_mut()
        .ok_or_else(|| MinitensorError::internal_error("Failed to get mutable f32 slice"))?;

    result_slice[0] = prod;
    Ok(())
}

fn prod_all_f64(tensor: &Tensor, result_data: &mut TensorData) -> Result<()> {
    let data = tensor
        .data()
        .as_f64_slice()
        .ok_or_else(|| MinitensorError::internal_error("Failed to get f64 slice"))?;

    let prod: f64 = if data.len() >= 1024 {
        data.par_chunks(8192).map(simd_prod_f64).product::<f64>()
    } else {
        simd_prod_f64(data)
    };

    let result_slice = result_data
        .as_f64_slice_mut()
        .ok_or_else(|| MinitensorError::internal_error("Failed to get mutable f64 slice"))?;

    result_slice[0] = prod;
    Ok(())
}

fn prod_all_i32(tensor: &Tensor, result_data: &mut TensorData) -> Result<()> {
    let data = tensor
        .data()
        .as_i32_slice()
        .ok_or_else(|| MinitensorError::internal_error("Failed to get i32 slice"))?;

    let prod: i32 = if data.len() >= 1024 {
        data.par_chunks(8192).map(simd_prod_i32).product::<i32>()
    } else {
        simd_prod_i32(data)
    };

    let result_slice = result_data
        .as_i32_slice_mut()
        .ok_or_else(|| MinitensorError::internal_error("Failed to get mutable i32 slice"))?;

    result_slice[0] = prod;
    Ok(())
}

fn prod_all_i64(tensor: &Tensor, result_data: &mut TensorData) -> Result<()> {
    let data = tensor
        .data()
        .as_i64_slice()
        .ok_or_else(|| MinitensorError::internal_error("Failed to get i64 slice"))?;

    let prod: i64 = if data.len() >= 1024 {
        data.par_chunks(8192).map(simd_prod_i64).product::<i64>()
    } else {
        simd_prod_i64(data)
    };

    let result_slice = result_data
        .as_i64_slice_mut()
        .ok_or_else(|| MinitensorError::internal_error("Failed to get mutable i64 slice"))?;

    result_slice[0] = prod;
    Ok(())
}

fn prod_all_bool(tensor: &Tensor, result_data: &mut TensorData) -> Result<()> {
    let data = tensor
        .data()
        .as_bool_slice()
        .ok_or_else(|| MinitensorError::internal_error("Failed to get bool slice"))?;

    let prod = data.par_iter().all(|&x| x);

    let result_slice = result_data
        .as_bool_slice_mut()
        .ok_or_else(|| MinitensorError::internal_error("Failed to get mutable bool slice"))?;

    result_slice[0] = prod;
    Ok(())
}

fn sum_all_f32(tensor: &Tensor, result_data: &mut TensorData) -> Result<()> {
    let data = tensor
        .data()
        .as_f32_slice()
        .ok_or_else(|| MinitensorError::internal_error("Failed to get f32 slice"))?;

    let sum: f32 = if data.len() >= 1024 {
        data.par_chunks(8192).map(simd_sum_f32).sum::<f32>()
    } else {
        simd_sum_f32(data)
    };

    let result_slice = result_data
        .as_f32_slice_mut()
        .ok_or_else(|| MinitensorError::internal_error("Failed to get mutable f32 slice"))?;

    result_slice[0] = sum;
    Ok(())
}

fn sum_all_f64(tensor: &Tensor, result_data: &mut TensorData) -> Result<()> {
    let data = tensor
        .data()
        .as_f64_slice()
        .ok_or_else(|| MinitensorError::internal_error("Failed to get f64 slice"))?;

    let sum: f64 = if data.len() >= 1024 {
        data.par_chunks(8192).map(simd_sum_f64).sum::<f64>()
    } else {
        simd_sum_f64(data)
    };

    let result_slice = result_data
        .as_f64_slice_mut()
        .ok_or_else(|| MinitensorError::internal_error("Failed to get mutable f64 slice"))?;

    result_slice[0] = sum;
    Ok(())
}

fn sum_all_i32(tensor: &Tensor, result_data: &mut TensorData) -> Result<()> {
    let data = tensor
        .data()
        .as_i32_slice()
        .ok_or_else(|| MinitensorError::internal_error("Failed to get i32 slice"))?;

    let sum: i32 = if data.len() >= 1024 {
        data.par_chunks(8192).map(simd_sum_i32).sum::<i32>()
    } else {
        simd_sum_i32(data)
    };

    let result_slice = result_data
        .as_i32_slice_mut()
        .ok_or_else(|| MinitensorError::internal_error("Failed to get mutable i32 slice"))?;

    result_slice[0] = sum;
    Ok(())
}

fn sum_all_i64(tensor: &Tensor, result_data: &mut TensorData) -> Result<()> {
    let data = tensor
        .data()
        .as_i64_slice()
        .ok_or_else(|| MinitensorError::internal_error("Failed to get i64 slice"))?;

    let sum: i64 = if data.len() >= 1024 {
        data.par_chunks(8192).map(simd_sum_i64).sum::<i64>()
    } else {
        simd_sum_i64(data)
    };

    let result_slice = result_data
        .as_i64_slice_mut()
        .ok_or_else(|| MinitensorError::internal_error("Failed to get mutable i64 slice"))?;

    result_slice[0] = sum;
    Ok(())
}

#[inline]
pub fn sum_along_dim(tensor: &Tensor, dim: usize, keepdim: bool) -> Result<Tensor> {
    if dim >= tensor.ndim() {
        return Err(MinitensorError::index_error(dim as isize, 0, tensor.ndim()));
    }

    let input_shape = tensor.shape().dims();
    let mut output_shape = input_shape.to_vec();

    if keepdim {
        output_shape[dim] = 1;
    } else {
        output_shape.remove(dim);
    }

    let output_shape_obj = Shape::new(output_shape);
    let mut result_data =
        TensorData::zeros_on_device(output_shape_obj.numel(), tensor.dtype(), tensor.device());

    match tensor.dtype() {
        DataType::Float32 => sum_along_dim_f32(tensor, &mut result_data, dim)?,
        DataType::Float64 => sum_along_dim_f64(tensor, &mut result_data, dim)?,
        DataType::Int32 => sum_along_dim_i32(tensor, &mut result_data, dim)?,
        DataType::Int64 => sum_along_dim_i64(tensor, &mut result_data, dim)?,
        DataType::Bool => {
            return Err(MinitensorError::invalid_operation(
                "Sum not supported for boolean tensors",
            ));
        }
    }

    Ok(Tensor::new(
        Arc::new(result_data),
        output_shape_obj,
        tensor.dtype(),
        tensor.device(),
        tensor.requires_grad(),
    ))
}

fn sum_along_dim_f32(tensor: &Tensor, result_data: &mut TensorData, dim: usize) -> Result<()> {
    let input_data = tensor
        .data()
        .as_f32_slice()
        .ok_or_else(|| MinitensorError::internal_error("Failed to get f32 slice"))?;

    let result_slice = result_data
        .as_f32_slice_mut()
        .ok_or_else(|| MinitensorError::internal_error("Failed to get mutable f32 slice"))?;

    let input_shape = tensor.shape().dims();

    if tensor.ndim() == 1 {
        if dim != 0 {
            return Err(MinitensorError::index_error(dim as isize, 0, tensor.ndim()));
        }
        result_slice[0] = simd_sum_f32(input_data);
    } else if tensor.ndim() == 2 {
        let cols = input_shape[1];
        match dim {
            0 => {
                let sums = input_data
                    .par_chunks_exact(cols)
                    .fold(
                        || vec![0f32; cols],
                        |mut acc, row| {
                            for (a, &v) in acc.iter_mut().zip(row) {
                                *a += v;
                            }
                            acc
                        },
                    )
                    .reduce(
                        || vec![0f32; cols],
                        |mut a, b| {
                            for (x, y) in a.iter_mut().zip(b) {
                                *x += y;
                            }
                            a
                        },
                    );
                result_slice.copy_from_slice(&sums);
            }
            1 => {
                result_slice
                    .par_iter_mut()
                    .zip(input_data.par_chunks_exact(cols))
                    .for_each(|(out, row)| {
                        *out = simd_sum_f32(row);
                    });
            }
            _ => {
                return Err(MinitensorError::index_error(dim as isize, 0, tensor.ndim()));
            }
        }
    } else {
        let dim_size = input_shape[dim];
        let inner = input_shape[dim + 1..].iter().product::<usize>();
        let outer_stride = dim_size * inner;

        result_slice
            .par_iter_mut()
            .enumerate()
            .for_each(|(idx, out)| {
                let o = idx / inner;
                let r = idx % inner;
                let mut sum_val = 0f32;
                let mut base = o * outer_stride + r;
                for _ in 0..dim_size {
                    sum_val += input_data[base];
                    base += inner;
                }
                *out = sum_val;
            });
    }

    Ok(())
}

fn sum_along_dim_f64(tensor: &Tensor, result_data: &mut TensorData, dim: usize) -> Result<()> {
    let input_data = tensor
        .data()
        .as_f64_slice()
        .ok_or_else(|| MinitensorError::internal_error("Failed to get f64 slice"))?;

    let result_slice = result_data
        .as_f64_slice_mut()
        .ok_or_else(|| MinitensorError::internal_error("Failed to get mutable f64 slice"))?;

    let input_shape = tensor.shape().dims();

    if tensor.ndim() == 1 {
        if dim != 0 {
            return Err(MinitensorError::index_error(dim as isize, 0, tensor.ndim()));
        }
        result_slice[0] = simd_sum_f64(input_data);
    } else if tensor.ndim() == 2 {
        let cols = input_shape[1];
        match dim {
            0 => {
                let sums = input_data
                    .par_chunks_exact(cols)
                    .fold(
                        || vec![0f64; cols],
                        |mut acc, row| {
                            for (a, &v) in acc.iter_mut().zip(row) {
                                *a += v;
                            }
                            acc
                        },
                    )
                    .reduce(
                        || vec![0f64; cols],
                        |mut a, b| {
                            for (x, y) in a.iter_mut().zip(b) {
                                *x += y;
                            }
                            a
                        },
                    );
                result_slice.copy_from_slice(&sums);
            }
            1 => {
                result_slice
                    .par_iter_mut()
                    .zip(input_data.par_chunks_exact(cols))
                    .for_each(|(out, row)| {
                        *out = simd_sum_f64(row);
                    });
            }
            _ => {
                return Err(MinitensorError::index_error(dim as isize, 0, tensor.ndim()));
            }
        }
    } else {
        let dim_size = input_shape[dim];
        let inner = input_shape[dim + 1..].iter().product::<usize>();
        let outer_stride = dim_size * inner;

        result_slice
            .par_iter_mut()
            .enumerate()
            .for_each(|(idx, out)| {
                let o = idx / inner;
                let r = idx % inner;
                let mut sum_val = 0f64;
                let mut base = o * outer_stride + r;
                for _ in 0..dim_size {
                    sum_val += input_data[base];
                    base += inner;
                }
                *out = sum_val;
            });
    }

    Ok(())
}

fn sum_along_dim_i32(tensor: &Tensor, result_data: &mut TensorData, dim: usize) -> Result<()> {
    let input_data = tensor
        .data()
        .as_i32_slice()
        .ok_or_else(|| MinitensorError::internal_error("Failed to get i32 slice"))?;

    let result_slice = result_data
        .as_i32_slice_mut()
        .ok_or_else(|| MinitensorError::internal_error("Failed to get mutable i32 slice"))?;

    let input_shape = tensor.shape().dims();

    if tensor.ndim() == 1 {
        if dim != 0 {
            return Err(MinitensorError::index_error(dim as isize, 0, tensor.ndim()));
        }
        result_slice[0] = simd_sum_i32(input_data);
    } else if tensor.ndim() == 2 {
        let cols = input_shape[1];
        match dim {
            0 => {
                let sums = input_data
                    .par_chunks_exact(cols)
                    .fold(
                        || vec![0i32; cols],
                        |mut acc, row| {
                            for (a, &v) in acc.iter_mut().zip(row) {
                                *a += v;
                            }
                            acc
                        },
                    )
                    .reduce(
                        || vec![0i32; cols],
                        |mut a, b| {
                            for (x, y) in a.iter_mut().zip(b) {
                                *x += y;
                            }
                            a
                        },
                    );
                result_slice.copy_from_slice(&sums);
            }
            1 => {
                result_slice
                    .par_iter_mut()
                    .zip(input_data.par_chunks_exact(cols))
                    .for_each(|(out, row)| {
                        *out = simd_sum_i32(row);
                    });
            }
            _ => {
                return Err(MinitensorError::index_error(dim as isize, 0, tensor.ndim()));
            }
        }
    } else {
        let dim_size = input_shape[dim];
        let inner = input_shape[dim + 1..].iter().product::<usize>();
        let outer_stride = dim_size * inner;

        result_slice
            .par_iter_mut()
            .enumerate()
            .for_each(|(idx, out)| {
                let o = idx / inner;
                let r = idx % inner;
                let mut sum_val = 0i32;
                let mut base = o * outer_stride + r;
                for _ in 0..dim_size {
                    sum_val += input_data[base];
                    base += inner;
                }
                *out = sum_val;
            });
    }

    Ok(())
}

fn sum_along_dim_i64(tensor: &Tensor, result_data: &mut TensorData, dim: usize) -> Result<()> {
    let input_data = tensor
        .data()
        .as_i64_slice()
        .ok_or_else(|| MinitensorError::internal_error("Failed to get i64 slice"))?;

    let result_slice = result_data
        .as_i64_slice_mut()
        .ok_or_else(|| MinitensorError::internal_error("Failed to get mutable i64 slice"))?;

    let input_shape = tensor.shape().dims();

    if tensor.ndim() == 1 {
        if dim != 0 {
            return Err(MinitensorError::index_error(dim as isize, 0, tensor.ndim()));
        }
        result_slice[0] = simd_sum_i64(input_data);
    } else if tensor.ndim() == 2 {
        let cols = input_shape[1];
        match dim {
            0 => {
                let sums = input_data
                    .par_chunks_exact(cols)
                    .fold(
                        || vec![0i64; cols],
                        |mut acc, row| {
                            for (a, &v) in acc.iter_mut().zip(row) {
                                *a += v;
                            }
                            acc
                        },
                    )
                    .reduce(
                        || vec![0i64; cols],
                        |mut a, b| {
                            for (x, y) in a.iter_mut().zip(b) {
                                *x += y;
                            }
                            a
                        },
                    );
                result_slice.copy_from_slice(&sums);
            }
            1 => {
                result_slice
                    .par_iter_mut()
                    .zip(input_data.par_chunks_exact(cols))
                    .for_each(|(out, row)| {
                        *out = simd_sum_i64(row);
                    });
            }
            _ => {
                return Err(MinitensorError::index_error(dim as isize, 0, tensor.ndim()));
            }
        }
    } else {
        let dim_size = input_shape[dim];
        let inner = input_shape[dim + 1..].iter().product::<usize>();
        let outer_stride = dim_size * inner;

        result_slice
            .par_iter_mut()
            .enumerate()
            .for_each(|(idx, out)| {
                let o = idx / inner;
                let r = idx % inner;
                let mut sum_val = 0i64;
                let mut base = o * outer_stride + r;
                for _ in 0..dim_size {
                    sum_val += input_data[base];
                    base += inner;
                }
                *out = sum_val;
            });
    }

    Ok(())
}

#[inline]
pub fn prod_along_dim(tensor: &Tensor, dim: usize, keepdim: bool) -> Result<Tensor> {
    if dim >= tensor.ndim() {
        return Err(MinitensorError::index_error(dim as isize, 0, tensor.ndim()));
    }

    let input_shape = tensor.shape().dims();
    let mut output_shape = input_shape.to_vec();
    if keepdim {
        output_shape[dim] = 1;
    } else {
        output_shape.remove(dim);
    }
    let output_shape_obj = Shape::new(output_shape);
    let mut result_data =
        TensorData::zeros_on_device(output_shape_obj.numel(), tensor.dtype(), tensor.device());

    match tensor.dtype() {
        DataType::Float32 => prod_along_dim_f32(tensor, &mut result_data, dim)?,
        DataType::Float64 => prod_along_dim_f64(tensor, &mut result_data, dim)?,
        DataType::Int32 => prod_along_dim_i32(tensor, &mut result_data, dim)?,
        DataType::Int64 => prod_along_dim_i64(tensor, &mut result_data, dim)?,
        DataType::Bool => prod_along_dim_bool(tensor, &mut result_data, dim)?,
    }

    let requires_grad = tensor.requires_grad() && tensor.dtype() != DataType::Bool;
    Ok(Tensor::new(
        Arc::new(result_data),
        output_shape_obj,
        tensor.dtype(),
        tensor.device(),
        requires_grad,
    ))
}

fn prod_along_dim_f32(tensor: &Tensor, result_data: &mut TensorData, dim: usize) -> Result<()> {
    let input_data = tensor
        .data()
        .as_f32_slice()
        .ok_or_else(|| MinitensorError::internal_error("Failed to get f32 slice"))?;
    let result_slice = result_data
        .as_f32_slice_mut()
        .ok_or_else(|| MinitensorError::internal_error("Failed to get mutable f32 slice"))?;
    let input_shape = tensor.shape().dims();
    let dim_size = input_shape[dim];
    let inner = input_shape[dim + 1..].iter().product::<usize>();
    let outer_stride = dim_size * inner;
    result_slice
        .par_iter_mut()
        .enumerate()
        .for_each(|(idx, out)| {
            let o = idx / inner;
            let r = idx % inner;
            let mut prod_val = 1f32;
            let mut base = o * outer_stride + r;
            for _ in 0..dim_size {
                prod_val *= input_data[base];
                base += inner;
            }
            *out = prod_val;
        });
    Ok(())
}

fn prod_along_dim_f64(tensor: &Tensor, result_data: &mut TensorData, dim: usize) -> Result<()> {
    let input_data = tensor
        .data()
        .as_f64_slice()
        .ok_or_else(|| MinitensorError::internal_error("Failed to get f64 slice"))?;
    let result_slice = result_data
        .as_f64_slice_mut()
        .ok_or_else(|| MinitensorError::internal_error("Failed to get mutable f64 slice"))?;
    let input_shape = tensor.shape().dims();
    let dim_size = input_shape[dim];
    let inner = input_shape[dim + 1..].iter().product::<usize>();
    let outer_stride = dim_size * inner;
    result_slice
        .par_iter_mut()
        .enumerate()
        .for_each(|(idx, out)| {
            let o = idx / inner;
            let r = idx % inner;
            let mut prod_val = 1f64;
            let mut base = o * outer_stride + r;
            for _ in 0..dim_size {
                prod_val *= input_data[base];
                base += inner;
            }
            *out = prod_val;
        });
    Ok(())
}

fn prod_along_dim_i32(tensor: &Tensor, result_data: &mut TensorData, dim: usize) -> Result<()> {
    let input_data = tensor
        .data()
        .as_i32_slice()
        .ok_or_else(|| MinitensorError::internal_error("Failed to get i32 slice"))?;
    let result_slice = result_data
        .as_i32_slice_mut()
        .ok_or_else(|| MinitensorError::internal_error("Failed to get mutable i32 slice"))?;
    let input_shape = tensor.shape().dims();
    let dim_size = input_shape[dim];
    let inner = input_shape[dim + 1..].iter().product::<usize>();
    let outer_stride = dim_size * inner;
    result_slice
        .par_iter_mut()
        .enumerate()
        .for_each(|(idx, out)| {
            let o = idx / inner;
            let r = idx % inner;
            let mut prod_val = 1i32;
            let mut base = o * outer_stride + r;
            for _ in 0..dim_size {
                prod_val *= input_data[base];
                base += inner;
            }
            *out = prod_val;
        });
    Ok(())
}

fn prod_along_dim_i64(tensor: &Tensor, result_data: &mut TensorData, dim: usize) -> Result<()> {
    let input_data = tensor
        .data()
        .as_i64_slice()
        .ok_or_else(|| MinitensorError::internal_error("Failed to get i64 slice"))?;
    let result_slice = result_data
        .as_i64_slice_mut()
        .ok_or_else(|| MinitensorError::internal_error("Failed to get mutable i64 slice"))?;
    let input_shape = tensor.shape().dims();
    let dim_size = input_shape[dim];
    let inner = input_shape[dim + 1..].iter().product::<usize>();
    let outer_stride = dim_size * inner;
    result_slice
        .par_iter_mut()
        .enumerate()
        .for_each(|(idx, out)| {
            let o = idx / inner;
            let r = idx % inner;
            let mut prod_val = 1i64;
            let mut base = o * outer_stride + r;
            for _ in 0..dim_size {
                prod_val *= input_data[base];
                base += inner;
            }
            *out = prod_val;
        });

    Ok(())
}

fn prod_along_dim_bool(tensor: &Tensor, result_data: &mut TensorData, dim: usize) -> Result<()> {
    let input_data = tensor
        .data()
        .as_bool_slice()
        .ok_or_else(|| MinitensorError::internal_error("Failed to get bool slice"))?;
    let result_slice = result_data
        .as_bool_slice_mut()
        .ok_or_else(|| MinitensorError::internal_error("Failed to get mutable bool slice"))?;
    let input_shape = tensor.shape().dims();
    let dim_size = input_shape[dim];
    let inner = input_shape[dim + 1..].iter().product::<usize>();
    let outer_stride = dim_size * inner;
    result_slice
        .par_iter_mut()
        .enumerate()
        .for_each(|(idx, out)| {
            let o = idx / inner;
            let r = idx % inner;
            let mut val = true;
            let mut base = o * outer_stride + r;
            for _ in 0..dim_size {
                val &= input_data[base];
                if !val {
                    break;
                }
                base += inner;
            }
            *out = val;
        });

    Ok(())
}

// Helper implementations for max/min operations
fn max_all_f32(tensor: &Tensor, result_data: &mut TensorData) -> Result<()> {
    let data = tensor
        .data()
        .as_f32_slice()
        .ok_or_else(|| MinitensorError::internal_error("Failed to get f32 slice"))?;

    let max_val = data.par_iter().cloned().reduce(
        || f32::NEG_INFINITY,
        |a, b| {
            if b.is_nan() { a } else { a.max(b) }
        },
    );

    let result_slice = result_data
        .as_f32_slice_mut()
        .ok_or_else(|| MinitensorError::internal_error("Failed to get mutable f32 slice"))?;

    result_slice[0] = max_val;
    Ok(())
}

fn max_all_f64(tensor: &Tensor, result_data: &mut TensorData) -> Result<()> {
    let data = tensor
        .data()
        .as_f64_slice()
        .ok_or_else(|| MinitensorError::internal_error("Failed to get f64 slice"))?;

    let max_val = data.par_iter().cloned().reduce(
        || f64::NEG_INFINITY,
        |a, b| {
            if b.is_nan() { a } else { a.max(b) }
        },
    );

    let result_slice = result_data
        .as_f64_slice_mut()
        .ok_or_else(|| MinitensorError::internal_error("Failed to get mutable f64 slice"))?;

    result_slice[0] = max_val;
    Ok(())
}

fn max_all_i32(tensor: &Tensor, result_data: &mut TensorData) -> Result<()> {
    let data = tensor
        .data()
        .as_i32_slice()
        .ok_or_else(|| MinitensorError::internal_error("Failed to get i32 slice"))?;

    let max_val = data.par_iter().copied().max().unwrap_or(i32::MIN);

    let result_slice = result_data
        .as_i32_slice_mut()
        .ok_or_else(|| MinitensorError::internal_error("Failed to get mutable i32 slice"))?;

    result_slice[0] = max_val;
    Ok(())
}

fn max_all_i64(tensor: &Tensor, result_data: &mut TensorData) -> Result<()> {
    let data = tensor
        .data()
        .as_i64_slice()
        .ok_or_else(|| MinitensorError::internal_error("Failed to get i64 slice"))?;

    let max_val = data.par_iter().copied().max().unwrap_or(i64::MIN);

    let result_slice = result_data
        .as_i64_slice_mut()
        .ok_or_else(|| MinitensorError::internal_error("Failed to get mutable i64 slice"))?;

    result_slice[0] = max_val;
    Ok(())
}

fn max_all_bool(tensor: &Tensor, result_data: &mut TensorData) -> Result<()> {
    let data = tensor
        .data()
        .as_bool_slice()
        .ok_or_else(|| MinitensorError::internal_error("Failed to get bool slice"))?;

    let max_val = data.par_iter().any(|&x| x);

    let result_slice = result_data
        .as_bool_slice_mut()
        .ok_or_else(|| MinitensorError::internal_error("Failed to get mutable bool slice"))?;

    result_slice[0] = max_val;
    Ok(())
}

// Similar implementations for min functions
fn min_all_f32(tensor: &Tensor, result_data: &mut TensorData) -> Result<()> {
    let data = tensor
        .data()
        .as_f32_slice()
        .ok_or_else(|| MinitensorError::internal_error("Failed to get f32 slice"))?;

    let min_val = data.par_iter().cloned().reduce(
        || f32::INFINITY,
        |a, b| {
            if b.is_nan() { a } else { a.min(b) }
        },
    );

    let result_slice = result_data
        .as_f32_slice_mut()
        .ok_or_else(|| MinitensorError::internal_error("Failed to get mutable f32 slice"))?;

    result_slice[0] = min_val;
    Ok(())
}

fn min_all_f64(tensor: &Tensor, result_data: &mut TensorData) -> Result<()> {
    let data = tensor
        .data()
        .as_f64_slice()
        .ok_or_else(|| MinitensorError::internal_error("Failed to get f64 slice"))?;

    let min_val = data.par_iter().cloned().reduce(
        || f64::INFINITY,
        |a, b| {
            if b.is_nan() { a } else { a.min(b) }
        },
    );

    let result_slice = result_data
        .as_f64_slice_mut()
        .ok_or_else(|| MinitensorError::internal_error("Failed to get mutable f64 slice"))?;

    result_slice[0] = min_val;
    Ok(())
}

fn min_all_i32(tensor: &Tensor, result_data: &mut TensorData) -> Result<()> {
    let data = tensor
        .data()
        .as_i32_slice()
        .ok_or_else(|| MinitensorError::internal_error("Failed to get i32 slice"))?;

    let min_val = data.par_iter().copied().min().unwrap_or(i32::MAX);

    let result_slice = result_data
        .as_i32_slice_mut()
        .ok_or_else(|| MinitensorError::internal_error("Failed to get mutable i32 slice"))?;

    result_slice[0] = min_val;
    Ok(())
}

fn min_all_i64(tensor: &Tensor, result_data: &mut TensorData) -> Result<()> {
    let data = tensor
        .data()
        .as_i64_slice()
        .ok_or_else(|| MinitensorError::internal_error("Failed to get i64 slice"))?;

    let min_val = data.par_iter().copied().min().unwrap_or(i64::MAX);

    let result_slice = result_data
        .as_i64_slice_mut()
        .ok_or_else(|| MinitensorError::internal_error("Failed to get mutable i64 slice"))?;

    result_slice[0] = min_val;
    Ok(())
}

fn min_all_bool(tensor: &Tensor, result_data: &mut TensorData) -> Result<()> {
    let data = tensor
        .data()
        .as_bool_slice()
        .ok_or_else(|| MinitensorError::internal_error("Failed to get bool slice"))?;

    let min_val = data.par_iter().all(|&x| x);

    let result_slice = result_data
        .as_bool_slice_mut()
        .ok_or_else(|| MinitensorError::internal_error("Failed to get mutable bool slice"))?;

    result_slice[0] = min_val;
    Ok(())
}

// Placeholder implementations for argmax/argmin
fn argmax_all_f32(tensor: &Tensor, result_data: &mut TensorData) -> Result<()> {
    let data = tensor
        .data()
        .as_f32_slice()
        .ok_or_else(|| MinitensorError::internal_error("Failed to get f32 slice"))?;

    let (argmax_idx, _) = data.par_iter().enumerate().map(|(i, &v)| (i, v)).reduce(
        || (0, f32::NEG_INFINITY),
        |(i1, v1), (i2, v2)| {
            if v1 >= v2 { (i1, v1) } else { (i2, v2) }
        },
    );

    let result_slice = result_data
        .as_i64_slice_mut()
        .ok_or_else(|| MinitensorError::internal_error("Failed to get mutable i64 slice"))?;

    result_slice[0] = argmax_idx as i64;
    Ok(())
}

fn argmax_all_f64(tensor: &Tensor, result_data: &mut TensorData) -> Result<()> {
    let data = tensor
        .data()
        .as_f64_slice()
        .ok_or_else(|| MinitensorError::internal_error("Failed to get f64 slice"))?;

    let (argmax_idx, _) = data.par_iter().enumerate().map(|(i, &v)| (i, v)).reduce(
        || (0, f64::NEG_INFINITY),
        |(i1, v1), (i2, v2)| {
            if v1 >= v2 { (i1, v1) } else { (i2, v2) }
        },
    );

    let result_slice = result_data
        .as_i64_slice_mut()
        .ok_or_else(|| MinitensorError::internal_error("Failed to get mutable i64 slice"))?;

    result_slice[0] = argmax_idx as i64;
    Ok(())
}

fn argmax_all_i32(tensor: &Tensor, result_data: &mut TensorData) -> Result<()> {
    let data = tensor
        .data()
        .as_i32_slice()
        .ok_or_else(|| MinitensorError::internal_error("Failed to get i32 slice"))?;

    let (argmax_idx, _) = data.par_iter().enumerate().map(|(i, &v)| (i, v)).reduce(
        || (0, i32::MIN),
        |(i1, v1), (i2, v2)| {
            if v1 >= v2 { (i1, v1) } else { (i2, v2) }
        },
    );

    let result_slice = result_data
        .as_i64_slice_mut()
        .ok_or_else(|| MinitensorError::internal_error("Failed to get mutable i64 slice"))?;

    result_slice[0] = argmax_idx as i64;
    Ok(())
}

fn argmax_all_i64(tensor: &Tensor, result_data: &mut TensorData) -> Result<()> {
    let data = tensor
        .data()
        .as_i64_slice()
        .ok_or_else(|| MinitensorError::internal_error("Failed to get i64 slice"))?;

    let (argmax_idx, _) = data.par_iter().enumerate().map(|(i, &v)| (i, v)).reduce(
        || (0, i64::MIN),
        |(i1, v1), (i2, v2)| {
            if v1 >= v2 { (i1, v1) } else { (i2, v2) }
        },
    );

    let result_slice = result_data
        .as_i64_slice_mut()
        .ok_or_else(|| MinitensorError::internal_error("Failed to get mutable i64 slice"))?;

    result_slice[0] = argmax_idx as i64;
    Ok(())
}

fn argmax_all_bool(tensor: &Tensor, result_data: &mut TensorData) -> Result<()> {
    let data = tensor
        .data()
        .as_bool_slice()
        .ok_or_else(|| MinitensorError::internal_error("Failed to get bool slice"))?;

    let argmax_idx = data.iter().position(|&x| x).unwrap_or(0);

    let result_slice = result_data
        .as_i64_slice_mut()
        .ok_or_else(|| MinitensorError::internal_error("Failed to get mutable i64 slice"))?;

    result_slice[0] = argmax_idx as i64;
    Ok(())
}

// Similar implementations for argmin
fn argmin_all_f32(tensor: &Tensor, result_data: &mut TensorData) -> Result<()> {
    let data = tensor
        .data()
        .as_f32_slice()
        .ok_or_else(|| MinitensorError::internal_error("Failed to get f32 slice"))?;

    let (argmin_idx, _) = data.par_iter().enumerate().map(|(i, &v)| (i, v)).reduce(
        || (0, f32::INFINITY),
        |(i1, v1), (i2, v2)| {
            if v1 <= v2 { (i1, v1) } else { (i2, v2) }
        },
    );

    let result_slice = result_data
        .as_i64_slice_mut()
        .ok_or_else(|| MinitensorError::internal_error("Failed to get mutable i64 slice"))?;

    result_slice[0] = argmin_idx as i64;
    Ok(())
}

fn argmin_all_f64(tensor: &Tensor, result_data: &mut TensorData) -> Result<()> {
    let data = tensor
        .data()
        .as_f64_slice()
        .ok_or_else(|| MinitensorError::internal_error("Failed to get f64 slice"))?;

    let (argmin_idx, _) = data.par_iter().enumerate().map(|(i, &v)| (i, v)).reduce(
        || (0, f64::INFINITY),
        |(i1, v1), (i2, v2)| {
            if v1 <= v2 { (i1, v1) } else { (i2, v2) }
        },
    );

    let result_slice = result_data
        .as_i64_slice_mut()
        .ok_or_else(|| MinitensorError::internal_error("Failed to get mutable i64 slice"))?;

    result_slice[0] = argmin_idx as i64;
    Ok(())
}

fn argmin_all_i32(tensor: &Tensor, result_data: &mut TensorData) -> Result<()> {
    let data = tensor
        .data()
        .as_i32_slice()
        .ok_or_else(|| MinitensorError::internal_error("Failed to get i32 slice"))?;

    let (argmin_idx, _) = data.par_iter().enumerate().map(|(i, &v)| (i, v)).reduce(
        || (0, i32::MAX),
        |(i1, v1), (i2, v2)| {
            if v1 <= v2 { (i1, v1) } else { (i2, v2) }
        },
    );

    let result_slice = result_data
        .as_i64_slice_mut()
        .ok_or_else(|| MinitensorError::internal_error("Failed to get mutable i64 slice"))?;

    result_slice[0] = argmin_idx as i64;
    Ok(())
}

fn argmin_all_i64(tensor: &Tensor, result_data: &mut TensorData) -> Result<()> {
    let data = tensor
        .data()
        .as_i64_slice()
        .ok_or_else(|| MinitensorError::internal_error("Failed to get i64 slice"))?;

    let (argmin_idx, _) = data.par_iter().enumerate().map(|(i, &v)| (i, v)).reduce(
        || (0, i64::MAX),
        |(i1, v1), (i2, v2)| {
            if v1 <= v2 { (i1, v1) } else { (i2, v2) }
        },
    );

    let result_slice = result_data
        .as_i64_slice_mut()
        .ok_or_else(|| MinitensorError::internal_error("Failed to get mutable i64 slice"))?;

    result_slice[0] = argmin_idx as i64;
    Ok(())
}

fn argmin_all_bool(tensor: &Tensor, result_data: &mut TensorData) -> Result<()> {
    let data = tensor
        .data()
        .as_bool_slice()
        .ok_or_else(|| MinitensorError::internal_error("Failed to get bool slice"))?;

    let argmin_idx = data.par_iter().position_first(|&x| !x).unwrap_or(0);

    let result_slice = result_data
        .as_i64_slice_mut()
        .ok_or_else(|| MinitensorError::internal_error("Failed to get mutable i64 slice"))?;

    result_slice[0] = argmin_idx as i64;
    Ok(())
}

// Placeholder implementations for dimensional operations
fn max_along_dim(tensor: &Tensor, dim: usize, keepdim: bool) -> Result<Tensor> {
    if dim >= tensor.ndim() {
        return Err(MinitensorError::index_error(dim as isize, 0, tensor.ndim()));
    }

    let input_shape = tensor.shape().dims();
    let mut output_shape = input_shape.to_vec();
    if keepdim {
        output_shape[dim] = 1;
    } else {
        output_shape.remove(dim);
    }
    let output_shape_obj = Shape::new(output_shape);
    let mut result_data =
        TensorData::zeros_on_device(output_shape_obj.numel(), tensor.dtype(), tensor.device());

    let dim_size = input_shape[dim];
    let outer = input_shape[..dim].iter().product::<usize>();
    let inner = input_shape[dim + 1..].iter().product::<usize>();
    let outer_stride = dim_size * inner;

    match tensor.dtype() {
        DataType::Float32 => {
            let input = tensor
                .data()
                .as_f32_slice()
                .ok_or_else(|| MinitensorError::internal_error("Failed to get f32 slice"))?;
            let output = result_data.as_f32_slice_mut().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get mutable f32 slice")
            })?;

            for o in 0..outer {
                for r in 0..inner {
                    let mut max_val = f32::NEG_INFINITY;
                    for d in 0..dim_size {
                        let idx = o * outer_stride + d * inner + r;
                        let val = input[idx];
                        if !val.is_nan() {
                            max_val = max_val.max(val);
                        }
                    }
                    output[o * inner + r] = max_val;
                }
            }
        }
        DataType::Float64 => {
            let input = tensor
                .data()
                .as_f64_slice()
                .ok_or_else(|| MinitensorError::internal_error("Failed to get f64 slice"))?;
            let output = result_data.as_f64_slice_mut().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get mutable f64 slice")
            })?;

            for o in 0..outer {
                for r in 0..inner {
                    let mut max_val = f64::NEG_INFINITY;
                    for d in 0..dim_size {
                        let idx = o * outer_stride + d * inner + r;
                        let val = input[idx];
                        if !val.is_nan() {
                            max_val = max_val.max(val);
                        }
                    }
                    output[o * inner + r] = max_val;
                }
            }
        }
        DataType::Int32 => {
            let input = tensor
                .data()
                .as_i32_slice()
                .ok_or_else(|| MinitensorError::internal_error("Failed to get i32 slice"))?;
            let output = result_data.as_i32_slice_mut().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get mutable i32 slice")
            })?;

            for o in 0..outer {
                for r in 0..inner {
                    let mut max_val = i32::MIN;
                    for d in 0..dim_size {
                        let idx = o * outer_stride + d * inner + r;
                        max_val = max_val.max(input[idx]);
                    }
                    output[o * inner + r] = max_val;
                }
            }
        }
        DataType::Int64 => {
            let input = tensor
                .data()
                .as_i64_slice()
                .ok_or_else(|| MinitensorError::internal_error("Failed to get i64 slice"))?;
            let output = result_data.as_i64_slice_mut().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get mutable i64 slice")
            })?;

            for o in 0..outer {
                for r in 0..inner {
                    let mut max_val = i64::MIN;
                    for d in 0..dim_size {
                        let idx = o * outer_stride + d * inner + r;
                        max_val = max_val.max(input[idx]);
                    }
                    output[o * inner + r] = max_val;
                }
            }
        }
        DataType::Bool => {
            let input = tensor
                .data()
                .as_bool_slice()
                .ok_or_else(|| MinitensorError::internal_error("Failed to get bool slice"))?;
            let output = result_data.as_bool_slice_mut().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get mutable bool slice")
            })?;

            for o in 0..outer {
                for r in 0..inner {
                    let mut max_val = false;
                    for d in 0..dim_size {
                        let idx = o * outer_stride + d * inner + r;
                        max_val |= input[idx];
                        if max_val {
                            break;
                        }
                    }
                    output[o * inner + r] = max_val;
                }
            }
        }
    }

    Ok(Tensor::new(
        Arc::new(result_data),
        output_shape_obj,
        tensor.dtype(),
        tensor.device(),
        tensor.requires_grad(),
    ))
}

fn min_along_dim(tensor: &Tensor, dim: usize, keepdim: bool) -> Result<Tensor> {
    if dim >= tensor.ndim() {
        return Err(MinitensorError::index_error(dim as isize, 0, tensor.ndim()));
    }

    let input_shape = tensor.shape().dims();
    let mut output_shape = input_shape.to_vec();
    if keepdim {
        output_shape[dim] = 1;
    } else {
        output_shape.remove(dim);
    }
    let output_shape_obj = Shape::new(output_shape);
    let mut result_data =
        TensorData::zeros_on_device(output_shape_obj.numel(), tensor.dtype(), tensor.device());

    let dim_size = input_shape[dim];
    let outer = input_shape[..dim].iter().product::<usize>();
    let inner = input_shape[dim + 1..].iter().product::<usize>();
    let outer_stride = dim_size * inner;

    match tensor.dtype() {
        DataType::Float32 => {
            let input = tensor
                .data()
                .as_f32_slice()
                .ok_or_else(|| MinitensorError::internal_error("Failed to get f32 slice"))?;
            let output = result_data.as_f32_slice_mut().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get mutable f32 slice")
            })?;

            for o in 0..outer {
                for r in 0..inner {
                    let mut min_val = f32::INFINITY;
                    for d in 0..dim_size {
                        let idx = o * outer_stride + d * inner + r;
                        let val = input[idx];
                        if !val.is_nan() {
                            min_val = min_val.min(val);
                        }
                    }
                    output[o * inner + r] = min_val;
                }
            }
        }
        DataType::Float64 => {
            let input = tensor
                .data()
                .as_f64_slice()
                .ok_or_else(|| MinitensorError::internal_error("Failed to get f64 slice"))?;
            let output = result_data.as_f64_slice_mut().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get mutable f64 slice")
            })?;

            for o in 0..outer {
                for r in 0..inner {
                    let mut min_val = f64::INFINITY;
                    for d in 0..dim_size {
                        let idx = o * outer_stride + d * inner + r;
                        let val = input[idx];
                        if !val.is_nan() {
                            min_val = min_val.min(val);
                        }
                    }
                    output[o * inner + r] = min_val;
                }
            }
        }
        DataType::Int32 => {
            let input = tensor
                .data()
                .as_i32_slice()
                .ok_or_else(|| MinitensorError::internal_error("Failed to get i32 slice"))?;
            let output = result_data.as_i32_slice_mut().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get mutable i32 slice")
            })?;

            for o in 0..outer {
                for r in 0..inner {
                    let mut min_val = i32::MAX;
                    for d in 0..dim_size {
                        let idx = o * outer_stride + d * inner + r;
                        min_val = min_val.min(input[idx]);
                    }
                    output[o * inner + r] = min_val;
                }
            }
        }
        DataType::Int64 => {
            let input = tensor
                .data()
                .as_i64_slice()
                .ok_or_else(|| MinitensorError::internal_error("Failed to get i64 slice"))?;
            let output = result_data.as_i64_slice_mut().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get mutable i64 slice")
            })?;

            for o in 0..outer {
                for r in 0..inner {
                    let mut min_val = i64::MAX;
                    for d in 0..dim_size {
                        let idx = o * outer_stride + d * inner + r;
                        min_val = min_val.min(input[idx]);
                    }
                    output[o * inner + r] = min_val;
                }
            }
        }
        DataType::Bool => {
            let input = tensor
                .data()
                .as_bool_slice()
                .ok_or_else(|| MinitensorError::internal_error("Failed to get bool slice"))?;
            let output = result_data.as_bool_slice_mut().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get mutable bool slice")
            })?;

            for o in 0..outer {
                for r in 0..inner {
                    let mut min_val = true;
                    for d in 0..dim_size {
                        let idx = o * outer_stride + d * inner + r;
                        min_val &= input[idx];
                        if !min_val {
                            break;
                        }
                    }
                    output[o * inner + r] = min_val;
                }
            }
        }
    }

    Ok(Tensor::new(
        Arc::new(result_data),
        output_shape_obj,
        tensor.dtype(),
        tensor.device(),
        tensor.requires_grad(),
    ))
}

fn argmax_along_dim(tensor: &Tensor, dim: usize, keepdim: bool) -> Result<Tensor> {
    if dim >= tensor.ndim() {
        return Err(MinitensorError::index_error(dim as isize, 0, tensor.ndim()));
    }

    let input_shape = tensor.shape().dims();
    let mut output_shape = input_shape.to_vec();
    if keepdim {
        output_shape[dim] = 1;
    } else {
        output_shape.remove(dim);
    }
    let output_shape_obj = Shape::new(output_shape);
    let mut result_data =
        TensorData::zeros_on_device(output_shape_obj.numel(), DataType::Int64, tensor.device());

    let dim_size = input_shape[dim];
    let outer = input_shape[..dim].iter().product::<usize>();
    let inner = input_shape[dim + 1..].iter().product::<usize>();
    let outer_stride = dim_size * inner;

    let output = result_data
        .as_i64_slice_mut()
        .ok_or_else(|| MinitensorError::internal_error("Failed to get mutable i64 slice"))?;

    match tensor.dtype() {
        DataType::Float32 => {
            let input = tensor
                .data()
                .as_f32_slice()
                .ok_or_else(|| MinitensorError::internal_error("Failed to get f32 slice"))?;
            for o in 0..outer {
                for r in 0..inner {
                    let mut max_val = f32::NEG_INFINITY;
                    let mut max_idx = 0usize;
                    for d in 0..dim_size {
                        let idx = o * outer_stride + d * inner + r;
                        let val = input[idx];
                        if val > max_val {
                            max_val = val;
                            max_idx = d;
                        }
                    }
                    output[o * inner + r] = max_idx as i64;
                }
            }
        }
        DataType::Float64 => {
            let input = tensor
                .data()
                .as_f64_slice()
                .ok_or_else(|| MinitensorError::internal_error("Failed to get f64 slice"))?;
            for o in 0..outer {
                for r in 0..inner {
                    let mut max_val = f64::NEG_INFINITY;
                    let mut max_idx = 0usize;
                    for d in 0..dim_size {
                        let idx = o * outer_stride + d * inner + r;
                        let val = input[idx];
                        if val > max_val {
                            max_val = val;
                            max_idx = d;
                        }
                    }
                    output[o * inner + r] = max_idx as i64;
                }
            }
        }
        DataType::Int32 => {
            let input = tensor
                .data()
                .as_i32_slice()
                .ok_or_else(|| MinitensorError::internal_error("Failed to get i32 slice"))?;
            for o in 0..outer {
                for r in 0..inner {
                    let mut max_val = i32::MIN;
                    let mut max_idx = 0usize;
                    for d in 0..dim_size {
                        let idx = o * outer_stride + d * inner + r;
                        let val = input[idx];
                        if val > max_val {
                            max_val = val;
                            max_idx = d;
                        }
                    }
                    output[o * inner + r] = max_idx as i64;
                }
            }
        }
        DataType::Int64 => {
            let input = tensor
                .data()
                .as_i64_slice()
                .ok_or_else(|| MinitensorError::internal_error("Failed to get i64 slice"))?;
            for o in 0..outer {
                for r in 0..inner {
                    let mut max_val = i64::MIN;
                    let mut max_idx = 0usize;
                    for d in 0..dim_size {
                        let idx = o * outer_stride + d * inner + r;
                        let val = input[idx];
                        if val > max_val {
                            max_val = val;
                            max_idx = d;
                        }
                    }
                    output[o * inner + r] = max_idx as i64;
                }
            }
        }
        DataType::Bool => {
            let input = tensor
                .data()
                .as_bool_slice()
                .ok_or_else(|| MinitensorError::internal_error("Failed to get bool slice"))?;
            for o in 0..outer {
                for r in 0..inner {
                    let mut max_idx = 0usize;
                    for d in 0..dim_size {
                        let idx = o * outer_stride + d * inner + r;
                        if input[idx] {
                            max_idx = d;
                            break;
                        }
                    }
                    output[o * inner + r] = max_idx as i64;
                }
            }
        }
    }

    Ok(Tensor::new(
        Arc::new(result_data),
        output_shape_obj,
        DataType::Int64,
        tensor.device(),
        false,
    ))
}

fn argmin_along_dim(tensor: &Tensor, dim: usize, keepdim: bool) -> Result<Tensor> {
    if dim >= tensor.ndim() {
        return Err(MinitensorError::index_error(dim as isize, 0, tensor.ndim()));
    }

    let input_shape = tensor.shape().dims();
    let mut output_shape = input_shape.to_vec();
    if keepdim {
        output_shape[dim] = 1;
    } else {
        output_shape.remove(dim);
    }
    let output_shape_obj = Shape::new(output_shape);
    let mut result_data =
        TensorData::zeros_on_device(output_shape_obj.numel(), DataType::Int64, tensor.device());

    let dim_size = input_shape[dim];
    let outer = input_shape[..dim].iter().product::<usize>();
    let inner = input_shape[dim + 1..].iter().product::<usize>();
    let outer_stride = dim_size * inner;

    let output = result_data
        .as_i64_slice_mut()
        .ok_or_else(|| MinitensorError::internal_error("Failed to get mutable i64 slice"))?;

    match tensor.dtype() {
        DataType::Float32 => {
            let input = tensor
                .data()
                .as_f32_slice()
                .ok_or_else(|| MinitensorError::internal_error("Failed to get f32 slice"))?;
            for o in 0..outer {
                for r in 0..inner {
                    let mut min_val = f32::INFINITY;
                    let mut min_idx = 0usize;
                    for d in 0..dim_size {
                        let idx = o * outer_stride + d * inner + r;
                        let val = input[idx];
                        if val < min_val {
                            min_val = val;
                            min_idx = d;
                        }
                    }
                    output[o * inner + r] = min_idx as i64;
                }
            }
        }
        DataType::Float64 => {
            let input = tensor
                .data()
                .as_f64_slice()
                .ok_or_else(|| MinitensorError::internal_error("Failed to get f64 slice"))?;
            for o in 0..outer {
                for r in 0..inner {
                    let mut min_val = f64::INFINITY;
                    let mut min_idx = 0usize;
                    for d in 0..dim_size {
                        let idx = o * outer_stride + d * inner + r;
                        let val = input[idx];
                        if val < min_val {
                            min_val = val;
                            min_idx = d;
                        }
                    }
                    output[o * inner + r] = min_idx as i64;
                }
            }
        }
        DataType::Int32 => {
            let input = tensor
                .data()
                .as_i32_slice()
                .ok_or_else(|| MinitensorError::internal_error("Failed to get i32 slice"))?;
            for o in 0..outer {
                for r in 0..inner {
                    let mut min_val = i32::MAX;
                    let mut min_idx = 0usize;
                    for d in 0..dim_size {
                        let idx = o * outer_stride + d * inner + r;
                        let val = input[idx];
                        if val < min_val {
                            min_val = val;
                            min_idx = d;
                        }
                    }
                    output[o * inner + r] = min_idx as i64;
                }
            }
        }
        DataType::Int64 => {
            let input = tensor
                .data()
                .as_i64_slice()
                .ok_or_else(|| MinitensorError::internal_error("Failed to get i64 slice"))?;
            for o in 0..outer {
                for r in 0..inner {
                    let mut min_val = i64::MAX;
                    let mut min_idx = 0usize;
                    for d in 0..dim_size {
                        let idx = o * outer_stride + d * inner + r;
                        let val = input[idx];
                        if val < min_val {
                            min_val = val;
                            min_idx = d;
                        }
                    }
                    output[o * inner + r] = min_idx as i64;
                }
            }
        }
        DataType::Bool => {
            let input = tensor
                .data()
                .as_bool_slice()
                .ok_or_else(|| MinitensorError::internal_error("Failed to get bool slice"))?;
            for o in 0..outer {
                for r in 0..inner {
                    let mut min_idx = 0usize;
                    for d in 0..dim_size {
                        let idx = o * outer_stride + d * inner + r;
                        if !input[idx] {
                            min_idx = d;
                            break;
                        }
                    }
                    output[o * inner + r] = min_idx as i64;
                }
            }
        }
    }

    Ok(Tensor::new(
        Arc::new(result_data),
        output_shape_obj,
        DataType::Int64,
        tensor.device(),
        false,
    ))
}

macro_rules! cumprod_forward {
    ($name:ident, $get:ident, $get_mut:ident, $t:ty) => {
        fn $name(tensor: &Tensor, result_data: &mut TensorData, dim: usize) -> Result<()> {
            let input_data = tensor
                .data()
                .$get()
                .ok_or_else(|| MinitensorError::internal_error("Failed to get slice"))?;
            let output = result_data
                .$get_mut()
                .ok_or_else(|| MinitensorError::internal_error("Failed to get mutable slice"))?;
            let shape = tensor.shape().dims();

            if tensor.ndim() == 1 {
                if dim != 0 {
                    return Err(MinitensorError::index_error(dim as isize, 0, tensor.ndim()));
                }
                let mut acc: $t = 1 as $t;
                for i in 0..input_data.len() {
                    acc *= input_data[i];
                    output[i] = acc;
                }
            } else if tensor.ndim() == 2 {
                let rows = shape[0];
                let cols = shape[1];
                match dim {
                    0 => {
                        let out_ptr = output.as_mut_ptr() as usize;
                        (0..cols).into_par_iter().for_each(|c| {
                            let out_ptr = out_ptr as *mut $t;
                            let mut acc: $t = 1 as $t;
                            for r in 0..rows {
                                let idx = r * cols + c;
                                acc *= input_data[idx];
                                unsafe {
                                    *out_ptr.add(idx) = acc;
                                }
                            }
                        });
                    }
                    1 => {
                        input_data
                            .par_chunks_exact(cols)
                            .zip(output.par_chunks_mut(cols))
                            .for_each(|(in_row, out_row)| {
                                let mut acc: $t = 1 as $t;
                                for i in 0..cols {
                                    acc *= in_row[i];
                                    out_row[i] = acc;
                                }
                            });
                    }
                    _ => return Err(MinitensorError::index_error(dim as isize, 0, tensor.ndim())),
                }
            } else {
                let dim_size = shape[dim];
                let inner = shape[dim + 1..].iter().product::<usize>();
                let outer = shape[..dim].iter().product::<usize>();
                let total = outer * inner;
                let out_ptr = output.as_mut_ptr() as usize;
                (0..total).into_par_iter().for_each(|idx| {
                    let out_ptr = out_ptr as *mut $t;
                    let o = idx / inner;
                    let r = idx % inner;
                    let mut acc: $t = 1 as $t;
                    let mut base = o * dim_size * inner + r;
                    for _ in 0..dim_size {
                        acc *= input_data[base];
                        unsafe {
                            *out_ptr.add(base) = acc;
                        }
                        base += inner;
                    }
                });
            }
            Ok(())
        }
    };
}

macro_rules! cumprod_backward {
    ($name:ident, $get:ident, $get_mut:ident, $t:ty) => {
        fn $name(
            input: &Tensor,
            output: &Tensor,
            grad: &Tensor,
            result_data: &mut TensorData,
            dim: usize,
        ) -> Result<()> {
            let input_data = input
                .data()
                .$get()
                .ok_or_else(|| MinitensorError::internal_error("Failed to get slice"))?;
            let out_data = output
                .data()
                .$get()
                .ok_or_else(|| MinitensorError::internal_error("Failed to get slice"))?;
            let grad_data = grad
                .data()
                .$get()
                .ok_or_else(|| MinitensorError::internal_error("Failed to get slice"))?;
            let output = result_data
                .$get_mut()
                .ok_or_else(|| MinitensorError::internal_error("Failed to get mutable slice"))?;
            let shape = input.shape().dims();

            if input.ndim() == 1 {
                if dim != 0 {
                    return Err(MinitensorError::index_error(dim as isize, 0, input.ndim()));
                }
                let len = input_data.len();
                // count zeros and index
                let mut zero_count = 0;
                let mut zero_idx = 0;
                for i in 0..len {
                    if input_data[i] == 0 as $t {
                        zero_count += 1;
                        if zero_count == 1 {
                            zero_idx = i;
                        }
                    }
                }
                if zero_count == 0 {
                    let mut s: $t = 0 as $t;
                    for i in (0..len).rev() {
                        s += grad_data[i] * out_data[i];
                        output[i] = s / input_data[i];
                    }
                } else if zero_count == 1 {
                    let mut s: $t = 0 as $t;
                    for i in (0..zero_idx).rev() {
                        s += grad_data[i] * out_data[i];
                        output[i] = s / input_data[i];
                    }
                    let mut prefix: $t = 1 as $t;
                    for i in 0..zero_idx {
                        prefix *= input_data[i];
                    }
                    let mut prod_suffix: $t = 1 as $t;
                    let mut grad_zero: $t = 0 as $t;
                    for j in zero_idx..len {
                        grad_zero += grad_data[j] * prod_suffix;
                        if j + 1 < len {
                            prod_suffix *= input_data[j + 1];
                        }
                    }
                    output[zero_idx] = grad_zero * prefix;
                    for i in zero_idx + 1..len {
                        output[i] = 0 as $t;
                    }
                } else {
                    for i in 0..len {
                        output[i] = 0 as $t;
                    }
                }
            } else if input.ndim() == 2 {
                let rows = shape[0];
                let cols = shape[1];
                match dim {
                    0 => {
                        for c in 0..cols {
                            let mut zero_count = 0;
                            let mut zero_idx = 0;
                            for r in 0..rows {
                                let idx = r * cols + c;
                                if input_data[idx] == 0 as $t {
                                    zero_count += 1;
                                    if zero_count == 1 {
                                        zero_idx = r;
                                    }
                                }
                            }
                            if zero_count == 0 {
                                let mut s: $t = 0 as $t;
                                for r in (0..rows).rev() {
                                    let idx = r * cols + c;
                                    s += grad_data[idx] * out_data[idx];
                                    output[idx] = s / input_data[idx];
                                }
                            } else if zero_count == 1 {
                                let mut s: $t = 0 as $t;
                                for r in (0..zero_idx).rev() {
                                    let idx = r * cols + c;
                                    s += grad_data[idx] * out_data[idx];
                                    output[idx] = s / input_data[idx];
                                }
                                let mut prefix: $t = 1 as $t;
                                for r in 0..zero_idx {
                                    prefix *= input_data[r * cols + c];
                                }
                                let mut prod_suffix: $t = 1 as $t;
                                let mut grad_zero: $t = 0 as $t;
                                for r in zero_idx..rows {
                                    let idx = r * cols + c;
                                    grad_zero += grad_data[idx] * prod_suffix;
                                    if r + 1 < rows {
                                        prod_suffix *= input_data[(r + 1) * cols + c];
                                    }
                                }
                                let zero_index = zero_idx * cols + c;
                                output[zero_index] = grad_zero * prefix;
                                for r in zero_idx + 1..rows {
                                    let idx = r * cols + c;
                                    output[idx] = 0 as $t;
                                }
                            } else {
                                for r in 0..rows {
                                    let idx = r * cols + c;
                                    output[idx] = 0 as $t;
                                }
                            }
                        }
                    }
                    1 => {
                        for r in 0..rows {
                            let base = r * cols;
                            let mut zero_count = 0;
                            let mut zero_idx = 0;
                            for c in 0..cols {
                                let idx = base + c;
                                if input_data[idx] == 0 as $t {
                                    zero_count += 1;
                                    if zero_count == 1 {
                                        zero_idx = c;
                                    }
                                }
                            }
                            if zero_count == 0 {
                                let mut s: $t = 0 as $t;
                                for c in (0..cols).rev() {
                                    let idx = base + c;
                                    s += grad_data[idx] * out_data[idx];
                                    output[idx] = s / input_data[idx];
                                }
                            } else if zero_count == 1 {
                                let mut s: $t = 0 as $t;
                                for c in (0..zero_idx).rev() {
                                    let idx = base + c;
                                    s += grad_data[idx] * out_data[idx];
                                    output[idx] = s / input_data[idx];
                                }
                                let mut prefix: $t = 1 as $t;
                                for c in 0..zero_idx {
                                    prefix *= input_data[base + c];
                                }
                                let mut prod_suffix: $t = 1 as $t;
                                let mut grad_zero: $t = 0 as $t;
                                for c in zero_idx..cols {
                                    let idx = base + c;
                                    grad_zero += grad_data[idx] * prod_suffix;
                                    if c + 1 < cols {
                                        prod_suffix *= input_data[base + c + 1];
                                    }
                                }
                                output[base + zero_idx] = grad_zero * prefix;
                                for c in zero_idx + 1..cols {
                                    output[base + c] = 0 as $t;
                                }
                            } else {
                                for c in 0..cols {
                                    output[base + c] = 0 as $t;
                                }
                            }
                        }
                    }
                    _ => return Err(MinitensorError::index_error(dim as isize, 0, input.ndim())),
                }
            } else {
                let dim_size = shape[dim];
                let inner = shape[dim + 1..].iter().product::<usize>();
                let outer = shape[..dim].iter().product::<usize>();
                let total = outer * inner;
                for idx in 0..total {
                    let o = idx / inner;
                    let r = idx % inner;
                    let base = o * dim_size * inner + r;
                    let mut zero_count = 0;
                    let mut zero_idx = 0;
                    for d in 0..dim_size {
                        let i = base + d * inner;
                        if input_data[i] == 0 as $t {
                            zero_count += 1;
                            if zero_count == 1 {
                                zero_idx = d;
                            }
                        }
                    }
                    if zero_count == 0 {
                        let mut s: $t = 0 as $t;
                        for d in (0..dim_size).rev() {
                            let i = base + d * inner;
                            s += grad_data[i] * out_data[i];
                            output[i] = s / input_data[i];
                        }
                    } else if zero_count == 1 {
                        let mut s: $t = 0 as $t;
                        for d in (0..zero_idx).rev() {
                            let i = base + d * inner;
                            s += grad_data[i] * out_data[i];
                            output[i] = s / input_data[i];
                        }
                        let mut prefix: $t = 1 as $t;
                        for d in 0..zero_idx {
                            prefix *= input_data[base + d * inner];
                        }
                        let mut prod_suffix: $t = 1 as $t;
                        let mut grad_zero: $t = 0 as $t;
                        for d in zero_idx..dim_size {
                            let i = base + d * inner;
                            grad_zero += grad_data[i] * prod_suffix;
                            if d + 1 < dim_size {
                                prod_suffix *= input_data[base + (d + 1) * inner];
                            }
                        }
                        let zero_index = base + zero_idx * inner;
                        output[zero_index] = grad_zero * prefix;
                        for d in zero_idx + 1..dim_size {
                            output[base + d * inner] = 0 as $t;
                        }
                    } else {
                        for d in 0..dim_size {
                            output[base + d * inner] = 0 as $t;
                        }
                    }
                }
            }
            Ok(())
        }
    };
}

macro_rules! cumsum_forward {
    ($name:ident, $get:ident, $get_mut:ident, $t:ty) => {
        fn $name(tensor: &Tensor, result_data: &mut TensorData, dim: usize) -> Result<()> {
            let input_data = tensor
                .data()
                .$get()
                .ok_or_else(|| MinitensorError::internal_error("Failed to get slice"))?;
            let output = result_data
                .$get_mut()
                .ok_or_else(|| MinitensorError::internal_error("Failed to get mutable slice"))?;
            let shape = tensor.shape().dims();

            if tensor.ndim() == 1 {
                if dim != 0 {
                    return Err(MinitensorError::index_error(dim as isize, 0, tensor.ndim()));
                }
                let mut acc: $t = 0 as $t;
                for i in 0..input_data.len() {
                    acc += input_data[i];
                    output[i] = acc;
                }
            } else if tensor.ndim() == 2 {
                let rows = shape[0];
                let cols = shape[1];
                match dim {
                    0 => {
                        let out_ptr = output.as_mut_ptr() as usize;
                        (0..cols).into_par_iter().for_each(|c| {
                            let out_ptr = out_ptr as *mut $t;
                            let mut acc: $t = 0 as $t;
                            for r in 0..rows {
                                let idx = r * cols + c;
                                acc += input_data[idx];
                                unsafe {
                                    *out_ptr.add(idx) = acc;
                                }
                            }
                        });
                    }
                    1 => {
                        input_data
                            .par_chunks_exact(cols)
                            .zip(output.par_chunks_mut(cols))
                            .for_each(|(in_row, out_row)| {
                                let mut acc: $t = 0 as $t;
                                for i in 0..cols {
                                    acc += in_row[i];
                                    out_row[i] = acc;
                                }
                            });
                    }
                    _ => return Err(MinitensorError::index_error(dim as isize, 0, tensor.ndim())),
                }
            } else {
                let dim_size = shape[dim];
                let inner = shape[dim + 1..].iter().product::<usize>();
                let outer = shape[..dim].iter().product::<usize>();
                let total = outer * inner;
                let out_ptr = output.as_mut_ptr() as usize;
                (0..total).into_par_iter().for_each(|idx| {
                    let out_ptr = out_ptr as *mut $t;
                    let o = idx / inner;
                    let r = idx % inner;
                    let mut acc: $t = 0 as $t;
                    let mut base = o * dim_size * inner + r;
                    for _ in 0..dim_size {
                        acc += input_data[base];
                        unsafe {
                            *out_ptr.add(base) = acc;
                        }
                        base += inner;
                    }
                });
            }
            Ok(())
        }
    };
}

macro_rules! cumsum_backward {
    ($name:ident, $get:ident, $get_mut:ident, $t:ty) => {
        fn $name(tensor: &Tensor, result_data: &mut TensorData, dim: usize) -> Result<()> {
            let input_data = tensor
                .data()
                .$get()
                .ok_or_else(|| MinitensorError::internal_error("Failed to get slice"))?;
            let output = result_data
                .$get_mut()
                .ok_or_else(|| MinitensorError::internal_error("Failed to get mutable slice"))?;
            let shape = tensor.shape().dims();

            if tensor.ndim() == 1 {
                if dim != 0 {
                    return Err(MinitensorError::index_error(dim as isize, 0, tensor.ndim()));
                }
                let mut acc: $t = 0 as $t;
                for i in (0..input_data.len()).rev() {
                    acc += input_data[i];
                    output[i] = acc;
                }
            } else if tensor.ndim() == 2 {
                let rows = shape[0];
                let cols = shape[1];
                match dim {
                    0 => {
                        let out_ptr = output.as_mut_ptr() as usize;
                        (0..cols).into_par_iter().for_each(|c| {
                            let out_ptr = out_ptr as *mut $t;
                            let mut acc: $t = 0 as $t;
                            for r in (0..rows).rev() {
                                let idx = r * cols + c;
                                acc += input_data[idx];
                                unsafe {
                                    *out_ptr.add(idx) = acc;
                                }
                            }
                        });
                    }
                    1 => {
                        input_data
                            .par_chunks_exact(cols)
                            .zip(output.par_chunks_mut(cols))
                            .for_each(|(in_row, out_row)| {
                                let mut acc: $t = 0 as $t;
                                for i in (0..cols).rev() {
                                    acc += in_row[i];
                                    out_row[i] = acc;
                                }
                            });
                    }
                    _ => return Err(MinitensorError::index_error(dim as isize, 0, tensor.ndim())),
                }
            } else {
                let dim_size = shape[dim];
                let inner = shape[dim + 1..].iter().product::<usize>();
                let outer = shape[..dim].iter().product::<usize>();
                let total = outer * inner;
                let out_ptr = output.as_mut_ptr() as usize;
                (0..total).into_par_iter().for_each(|idx| {
                    let out_ptr = out_ptr as *mut $t;
                    let o = idx / inner;
                    let r = idx % inner;
                    let mut acc: $t = 0 as $t;
                    let mut base = o * dim_size * inner + r + (dim_size - 1) * inner;
                    for _ in 0..dim_size {
                        acc += input_data[base];
                        unsafe {
                            *out_ptr.add(base) = acc;
                        }
                        if base >= inner {
                            base -= inner;
                        }
                    }
                });
            }
            Ok(())
        }
    };
}

cumprod_forward!(cumprod_f32, as_f32_slice, as_f32_slice_mut, f32);
cumprod_forward!(cumprod_f64, as_f64_slice, as_f64_slice_mut, f64);
cumprod_forward!(cumprod_i32, as_i32_slice, as_i32_slice_mut, i32);
cumprod_forward!(cumprod_i64, as_i64_slice, as_i64_slice_mut, i64);

cumprod_backward!(cumprod_backward_f32, as_f32_slice, as_f32_slice_mut, f32);
cumprod_backward!(cumprod_backward_f64, as_f64_slice, as_f64_slice_mut, f64);

cumsum_forward!(cumsum_f32, as_f32_slice, as_f32_slice_mut, f32);
cumsum_forward!(cumsum_f64, as_f64_slice, as_f64_slice_mut, f64);
cumsum_forward!(cumsum_i32, as_i32_slice, as_i32_slice_mut, i32);
cumsum_forward!(cumsum_i64, as_i64_slice, as_i64_slice_mut, i64);

cumsum_backward!(cumsum_backward_f32, as_f32_slice, as_f32_slice_mut, f32);
cumsum_backward!(cumsum_backward_f64, as_f64_slice, as_f64_slice_mut, f64);
cumsum_backward!(cumsum_backward_i32, as_i32_slice, as_i32_slice_mut, i32);
cumsum_backward!(cumsum_backward_i64, as_i64_slice, as_i64_slice_mut, i64);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::Device;
    use std::sync::Arc;

    fn create_tensor_f32(data: Vec<f32>, shape: Vec<usize>) -> Tensor {
        let shape_obj = Shape::new(shape.clone());
        let mut tensor_data = TensorData::zeros(shape_obj.numel(), DataType::Float32);
        tensor_data
            .as_f32_slice_mut()
            .unwrap()
            .copy_from_slice(&data);
        Tensor::new(
            Arc::new(tensor_data),
            shape_obj,
            DataType::Float32,
            Device::cpu(),
            false,
        )
    }

    fn create_tensor_i32(data: Vec<i32>, shape: Vec<usize>) -> Tensor {
        let shape_obj = Shape::new(shape.clone());
        let mut tensor_data = TensorData::zeros(shape_obj.numel(), DataType::Int32);
        tensor_data
            .as_i32_slice_mut()
            .unwrap()
            .copy_from_slice(&data);
        Tensor::new(
            Arc::new(tensor_data),
            shape_obj,
            DataType::Int32,
            Device::cpu(),
            false,
        )
    }

    fn create_tensor_bool(data: Vec<bool>, shape: Vec<usize>) -> Tensor {
        let shape_obj = Shape::new(shape.clone());
        let mut tensor_data = TensorData::zeros(shape_obj.numel(), DataType::Bool);
        tensor_data
            .as_bool_slice_mut()
            .unwrap()
            .copy_from_slice(&data);
        Tensor::new(
            Arc::new(tensor_data),
            shape_obj,
            DataType::Bool,
            Device::cpu(),
            false,
        )
    }

    #[test]
    fn test_median_global_even_length() {
        let t = create_tensor_f32(vec![3.0, 1.0, 4.0, 2.0], vec![4]);
        let (value, indices) = median(&t, None, false).unwrap();
        assert!(indices.is_none());
        assert!(value.shape().is_scalar());
        let result = value.data().as_f32_slice().unwrap();
        assert_eq!(result, &[2.0]);
    }

    #[test]
    fn test_median_with_dim_returns_indices() {
        let t = create_tensor_f32(vec![1.0, 3.0, 2.0, 4.0, 6.0, 5.0], vec![2, 3]);
        let (values, indices_opt) = median(&t, Some(1), false).unwrap();
        let indices = indices_opt.unwrap();
        assert_eq!(values.shape().dims(), &[2]);
        assert_eq!(indices.shape().dims(), &[2]);
        let values_slice = values.data().as_f32_slice().unwrap();
        let indices_slice = indices.data().as_i64_slice().unwrap();
        assert_eq!(values_slice, &[2.0, 5.0]);
        assert_eq!(indices_slice, &[2, 2]);
    }

    #[test]
    fn test_median_keepdim_preserves_rank() {
        let t = create_tensor_f32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let (values, indices_opt) = median(&t, Some(1), true).unwrap();
        let indices = indices_opt.unwrap();
        assert_eq!(values.shape().dims(), &[2, 1]);
        assert_eq!(indices.shape().dims(), &[2, 1]);
        assert_eq!(values.data().as_f32_slice().unwrap(), &[1.0, 3.0]);
        assert_eq!(indices.data().as_i64_slice().unwrap(), &[0, 0]);
    }

    #[test]
    fn test_median_empty_tensor_errors() {
        let t = create_tensor_f32(vec![], vec![0]);
        assert!(median(&t, None, false).is_err());
    }

    #[test]
    fn test_argmax_along_dim() {
        let t = create_tensor_f32(vec![1.0, 5.0, 3.0, 4.0, 2.0, 6.0], vec![2, 3]);
        let result = argmax(&t, Some(1), false).unwrap();
        let res = result.data().as_i64_slice().unwrap();
        assert_eq!(res, &[1, 2]);
    }

    #[test]
    fn test_argmin_along_dim_keepdim() {
        let t = create_tensor_f32(vec![1.0, 5.0, 3.0, 4.0, 2.0, 6.0], vec![2, 3]);
        let result = argmin(&t, Some(1), true).unwrap();
        assert_eq!(result.shape().dims(), &[2, 1]);
        let res = result.data().as_i64_slice().unwrap();
        assert_eq!(res, &[0, 1]);
    }

    #[test]
    fn test_all_any_global() {
        let t = create_tensor_i32(vec![1, 0, 2, 3], vec![2, 2]);
        let all_res = all(&t, None, false).unwrap();
        let any_res = any(&t, None, false).unwrap();
        assert_eq!(all_res.data().as_bool_slice().unwrap()[0], false);
        assert_eq!(any_res.data().as_bool_slice().unwrap()[0], true);
    }

    #[test]
    fn test_all_along_dim() {
        let t = create_tensor_bool(vec![true, false, true, true], vec![2, 2]);
        let res = all(&t, Some(1), false).unwrap();
        assert_eq!(res.data().as_bool_slice().unwrap(), &[false, true]);
    }

    #[test]
    fn test_sum_global_and_keepdim() {
        let t = create_tensor_f32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s = sum(&t, None, false).unwrap();
        assert_eq!(s.shape().dims(), &[] as &[usize]);
        assert_eq!(s.data().as_f32_slice().unwrap()[0], 10.0);
        let s_keep = sum(&t, None, true).unwrap();
        assert_eq!(s_keep.shape().dims(), &[1, 1]);
        assert_eq!(s_keep.data().as_f32_slice().unwrap()[0], 10.0);
    }

    #[test]
    fn test_topk_largest_float() {
        let t = create_tensor_f32(vec![1.0, 3.0, 2.0, 4.0, -1.0, 5.0], vec![2, 3]);
        let (values, indices) = topk(&t, 2, Some(1), true, true).unwrap();
        assert_eq!(values.shape().dims(), &[2, 2]);
        assert_eq!(indices.shape().dims(), &[2, 2]);
        let values_slice = values.data().as_f32_slice().unwrap();
        let indices_slice = indices.data().as_i64_slice().unwrap();
        assert_eq!(values_slice, &[3.0, 2.0, 5.0, 4.0]);
        assert_eq!(indices_slice, &[1, 2, 2, 0]);
    }

    #[test]
    fn test_topk_smallest_unsorted() {
        let t = create_tensor_f32(vec![1.0, -2.0, 3.5, 0.0], vec![4]);
        let (values, indices) = topk(&t, 2, None, false, false).unwrap();
        assert_eq!(values.shape().dims(), &[2]);
        let mut pairs: Vec<(i64, f32)> = indices
            .data()
            .as_i64_slice()
            .unwrap()
            .iter()
            .zip(values.data().as_f32_slice().unwrap())
            .map(|(&i, &v)| (i, v))
            .collect();
        pairs.sort_by_key(|p| p.0);
        assert_eq!(pairs, vec![(1, -2.0), (3, 0.0)]);
    }

    #[test]
    fn test_sum_along_dim() {
        let t = create_tensor_f32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let res = sum(&t, Some(vec![0]), false).unwrap();
        assert_eq!(res.shape().dims(), &[2]);
        assert_eq!(res.data().as_f32_slice().unwrap(), &[4.0, 6.0]);
    }

    #[test]
    fn test_sum_bool_error() {
        let t = create_tensor_bool(vec![true, false, true, true], vec![2, 2]);
        assert!(sum(&t, Some(vec![0]), false).is_err());
    }

    #[test]
    fn test_sum_multi_dim() {
        let t = create_tensor_f32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let res = sum(&t, Some(vec![0, 1]), false).unwrap();
        assert!(res.shape().is_scalar());
        assert_eq!(res.data().as_f32_slice().unwrap()[0], 10.0);
        let res_keep = sum(&t, Some(vec![0, 1]), true).unwrap();
        assert_eq!(res_keep.shape().dims(), &[1, 1]);
        assert_eq!(res_keep.data().as_f32_slice().unwrap()[0], 10.0);
    }

    #[test]
    fn test_prod_global_and_keepdim() {
        let t = create_tensor_f32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let p = prod(&t, None, false).unwrap();
        assert_eq!(p.data().as_f32_slice().unwrap()[0], 24.0);
        let p_keep = prod(&t, None, true).unwrap();
        assert_eq!(p_keep.shape().dims(), &[1, 1]);
        assert_eq!(p_keep.data().as_f32_slice().unwrap()[0], 24.0);
    }

    #[test]
    fn test_prod_along_dim() {
        let t = create_tensor_f32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let res = prod(&t, Some(vec![0]), false).unwrap();
        assert_eq!(res.data().as_f32_slice().unwrap(), &[3.0, 8.0]);
    }

    #[test]
    fn test_mean_along_dim() {
        let t = create_tensor_f32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let res = mean(&t, Some(vec![1]), true).unwrap();
        assert_eq!(res.shape().dims(), &[2, 1]);
        assert_eq!(res.data().as_f32_slice().unwrap(), &[1.5, 3.5]);
    }

    #[test]
    fn test_mean_int_support() {
        let t = create_tensor_i32(vec![1, 2, 3, 4], vec![2, 2]);
        let res = mean(&t, Some(vec![0isize]), false).unwrap();
        assert_eq!(res.dtype(), DataType::Float32);
        assert_eq!(res.data().as_f32_slice().unwrap(), &[2.0, 3.0]);
    }
}
