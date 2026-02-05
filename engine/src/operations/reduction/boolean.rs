// Copyright (c) 2026 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

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

/// NaN-aware maximum value along specified dimension
pub fn nanmax(tensor: &Tensor, dim: Option<isize>, keepdim: bool) -> Result<Tensor> {
    if !tensor.dtype().is_float() {
        return max(tensor, dim, keepdim);
    }

    match dim {
        None => {
            let result_shape = if keepdim {
                Shape::new(vec![1; tensor.ndim()])
            } else {
                Shape::scalar()
            };

            let mut result_data = TensorData::zeros_on_device(1, tensor.dtype(), tensor.device());

            match tensor.dtype() {
                DataType::Float32 => nanmax_all_f32(tensor, &mut result_data)?,
                DataType::Float64 => nanmax_all_f64(tensor, &mut result_data)?,
                _ => unreachable!("nanmax only supports floating point tensors"),
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
            let (values, _) = nanmax_along_dim_with_indices(tensor, d, keepdim)?;
            Ok(values)
        }
    }
}

/// NaN-aware minimum value along specified dimension
pub fn nanmin(tensor: &Tensor, dim: Option<isize>, keepdim: bool) -> Result<Tensor> {
    if !tensor.dtype().is_float() {
        return min(tensor, dim, keepdim);
    }

    match dim {
        None => {
            let result_shape = if keepdim {
                Shape::new(vec![1; tensor.ndim()])
            } else {
                Shape::scalar()
            };

            let mut result_data = TensorData::zeros_on_device(1, tensor.dtype(), tensor.device());

            match tensor.dtype() {
                DataType::Float32 => nanmin_all_f32(tensor, &mut result_data)?,
                DataType::Float64 => nanmin_all_f64(tensor, &mut result_data)?,
                _ => unreachable!("nanmin only supports floating point tensors"),
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
            let (values, _) = nanmin_along_dim_with_indices(tensor, d, keepdim)?;
            Ok(values)
        }
    }
}

/// Maximum values and their indices along specified dimension
pub fn max_with_indices(tensor: &Tensor, dim: isize, keepdim: bool) -> Result<(Tensor, Tensor)> {
    let d = normalize_dim(dim, tensor.ndim())?;
    max_along_dim_with_indices(tensor, d, keepdim)
}

/// NaN-aware maximum values and their indices along specified dimension
pub fn nanmax_with_indices(tensor: &Tensor, dim: isize, keepdim: bool) -> Result<(Tensor, Tensor)> {
    if !tensor.dtype().is_float() {
        return max_with_indices(tensor, dim, keepdim);
    }

    let d = normalize_dim(dim, tensor.ndim())?;
    nanmax_along_dim_with_indices(tensor, d, keepdim)
}

/// Minimum values and their indices along specified dimension
pub fn min_with_indices(tensor: &Tensor, dim: isize, keepdim: bool) -> Result<(Tensor, Tensor)> {
    let d = normalize_dim(dim, tensor.ndim())?;
    min_along_dim_with_indices(tensor, d, keepdim)
}

/// NaN-aware minimum values and their indices along specified dimension
pub fn nanmin_with_indices(tensor: &Tensor, dim: isize, keepdim: bool) -> Result<(Tensor, Tensor)> {
    if !tensor.dtype().is_float() {
        return min_with_indices(tensor, dim, keepdim);
    }

    let d = normalize_dim(dim, tensor.ndim())?;
    nanmin_along_dim_with_indices(tensor, d, keepdim)
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
