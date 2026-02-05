// Copyright (c) 2026 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

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

/// NaN-aware mean reduction along specified dimensions
pub fn nanmean(tensor: &Tensor, dim: Option<Vec<isize>>, keepdim: bool) -> Result<Tensor> {
    if !tensor.dtype().is_float() {
        return mean(tensor, dim, keepdim);
    }

    let dim = normalize_reduction_dims(dim, tensor.ndim())?;
    let dims_clone = dim.clone();
    let needs_mask =
        tensor.requires_grad() || dim.as_ref().map(|dims| !dims.is_empty()).unwrap_or(false);
    let mask = if needs_mask {
        Some(non_nan_mask(tensor)?)
    } else {
        None
    };

    if let Some(dims) = &dim {
        if dims.is_empty() {
            return Ok(tensor.clone());
        }
    }

    let (sum, count) = match dim {
        None => {
            let result_shape = if keepdim {
                Shape::new(vec![1; tensor.ndim()])
            } else {
                Shape::scalar()
            };
            let mut sum_data = TensorData::zeros_on_device(1, tensor.dtype(), tensor.device());
            let mut count_data = TensorData::zeros_on_device(1, tensor.dtype(), tensor.device());

            match tensor.dtype() {
                DataType::Float32 => nanmean_all_f32(tensor, &mut sum_data, &mut count_data)?,
                DataType::Float64 => nanmean_all_f64(tensor, &mut sum_data, &mut count_data)?,
                _ => unreachable!("nanmean only supports floating point tensors"),
            }

            (
                Tensor::new(
                    Arc::new(sum_data),
                    result_shape.clone(),
                    tensor.dtype(),
                    tensor.device(),
                    false,
                ),
                Tensor::new(
                    Arc::new(count_data),
                    result_shape,
                    tensor.dtype(),
                    tensor.device(),
                    false,
                ),
            )
        }
        Some(dims) => {
            let mask = mask.as_ref().ok_or_else(|| {
                MinitensorError::internal_error("nanmean expected mask for count computation")
            })?;
            let mut sum = tensor.clone();
            let mut count = mask.astype(tensor.dtype())?;

            if keepdim {
                for &d in &dims {
                    sum = nansum_along_dim(&sum, d, true)?;
                    count = sum_along_dim(&count, d, true)?;
                }
            } else {
                for &d in dims.iter().rev() {
                    sum = nansum_along_dim(&sum, d, false)?;
                    count = sum_along_dim(&count, d, false)?;
                }
            }
            (sum, count)
        }
    };

    let result = nanmean_from_sum_count(&sum, &count, tensor.requires_grad())?;

    if result.requires_grad() {
        let mask = mask.ok_or_else(|| {
            MinitensorError::internal_error("nanmean expected mask for gradient computation")
        })?;
        let grad_fn = Arc::new(NanMeanBackward {
            input_id: tensor.id(),
            input_shape: tensor.shape().dims().to_vec(),
            dims: dims_clone,
            keepdim,
            mask,
            count,
        });
        let mut result_with_grad = result;
        result_with_grad.set_grad_fn(Some(grad_fn.clone()));
        add_to_graph(&result_with_grad, Some(grad_fn))?;
        Ok(result_with_grad)
    } else {
        Ok(result)
    }
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
