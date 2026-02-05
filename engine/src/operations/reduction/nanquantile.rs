// Copyright (c) 2026 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

fn nanquantiles_along_dim(
    tensor: &Tensor,
    dim: usize,
    qs: &[f64],
    keepdim: bool,
    interpolation: QuantileInterpolation,
) -> Result<Tensor> {
    let dims = tensor.shape().dims();
    let dim_size = if dims.is_empty() { 1 } else { dims[dim] };

    if dim_size == 0 {
        return Err(MinitensorError::invalid_argument(
            "nanquantile() does not support empty slices".to_string(),
        ));
    }

    let q_len = qs.len();

    let mut out_dims = Vec::with_capacity(dims.len() + 2);
    out_dims.push(q_len);
    if !dims.is_empty() {
        out_dims.extend_from_slice(&dims[..dim]);
        if keepdim {
            out_dims.push(1);
        }
        out_dims.extend_from_slice(&dims[dim + 1..]);
    } else if keepdim {
        out_dims.push(1);
    }

    let shape = Shape::new(out_dims);
    let mut values_data =
        TensorData::zeros_on_device(shape.numel(), tensor.dtype(), tensor.device());

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

    match tensor.dtype() {
        DataType::Float32 => {
            let input = tensor
                .data()
                .as_f32_slice()
                .ok_or_else(|| MinitensorError::internal_error("Failed to get f32 slice"))?;
            let values = values_data.as_f32_slice_mut().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get mutable f32 slice")
            })?;

            if dim_size == 1 {
                fill_nanquantiles_single_f32(input, values, outer, inner, outer_stride, q_len)?;
            } else {
                let mut buffer = Vec::with_capacity(dim_size);
                let mut cached_positions: Option<(usize, Vec<QuantilePosition>)> = None;
                if q_len == 1 {
                    let q_value = qs[0];
                    for o in 0..outer {
                        for r in 0..inner {
                            buffer.clear();
                            for d in 0..dim_size {
                                let idx = o * outer_stride + d * inner + r;
                                let val = input[idx];
                                if !val.is_nan() {
                                    buffer.push(val);
                                }
                            }

                            if buffer.is_empty() {
                                return Err(MinitensorError::invalid_argument(
                                    NANQUANTILE_ALL_NAN_ERR.to_string(),
                                ));
                            }

                            let out_idx = o * inner + r;
                            values[out_idx] =
                                quantile_from_unsorted_f32(&mut buffer, q_value, interpolation);
                        }
                    }
                } else {
                    for o in 0..outer {
                        for r in 0..inner {
                            buffer.clear();
                            for d in 0..dim_size {
                                let idx = o * outer_stride + d * inner + r;
                                let val = input[idx];
                                if !val.is_nan() {
                                    buffer.push(val);
                                }
                            }

                            if buffer.is_empty() {
                                return Err(MinitensorError::invalid_argument(
                                    NANQUANTILE_ALL_NAN_ERR.to_string(),
                                ));
                            }

                            buffer.sort_by(|a, b| a.total_cmp(b));
                            let positions = match cached_positions {
                                Some((len, ref positions)) if len == buffer.len() => positions,
                                _ => {
                                    let positions = quantile_positions_for_len(buffer.len(), qs);
                                    cached_positions = Some((buffer.len(), positions));
                                    &cached_positions.as_ref().expect("positions cached").1
                                }
                            };
                            for (qi, position) in positions.iter().enumerate() {
                                let out_idx = ((qi * outer) + o) * inner + r;
                                let lower = buffer[position.lower_idx] as f64;
                                let upper = buffer[position.upper_idx] as f64;
                                values[out_idx] =
                                    interpolation.interpolate(lower, upper, position.weight) as f32;
                            }
                        }
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

            if dim_size == 1 {
                fill_nanquantiles_single_f64(input, values, outer, inner, outer_stride, q_len)?;
            } else {
                let mut buffer = Vec::with_capacity(dim_size);
                let mut cached_positions: Option<(usize, Vec<QuantilePosition>)> = None;
                if q_len == 1 {
                    let q_value = qs[0];
                    for o in 0..outer {
                        for r in 0..inner {
                            buffer.clear();
                            for d in 0..dim_size {
                                let idx = o * outer_stride + d * inner + r;
                                let val = input[idx];
                                if !val.is_nan() {
                                    buffer.push(val);
                                }
                            }

                            if buffer.is_empty() {
                                return Err(MinitensorError::invalid_argument(
                                    NANQUANTILE_ALL_NAN_ERR.to_string(),
                                ));
                            }

                            let out_idx = o * inner + r;
                            values[out_idx] =
                                quantile_from_unsorted_f64(&mut buffer, q_value, interpolation);
                        }
                    }
                } else {
                    for o in 0..outer {
                        for r in 0..inner {
                            buffer.clear();
                            for d in 0..dim_size {
                                let idx = o * outer_stride + d * inner + r;
                                let val = input[idx];
                                if !val.is_nan() {
                                    buffer.push(val);
                                }
                            }

                            if buffer.is_empty() {
                                return Err(MinitensorError::invalid_argument(
                                    NANQUANTILE_ALL_NAN_ERR.to_string(),
                                ));
                            }

                            buffer.sort_by(|a, b| a.total_cmp(b));
                            let positions = match cached_positions {
                                Some((len, ref positions)) if len == buffer.len() => positions,
                                _ => {
                                    let positions = quantile_positions_for_len(buffer.len(), qs);
                                    cached_positions = Some((buffer.len(), positions));
                                    &cached_positions.as_ref().expect("positions cached").1
                                }
                            };
                            for (qi, position) in positions.iter().enumerate() {
                                let out_idx = ((qi * outer) + o) * inner + r;
                                let lower = buffer[position.lower_idx];
                                let upper = buffer[position.upper_idx];
                                values[out_idx] =
                                    interpolation.interpolate(lower, upper, position.weight);
                            }
                        }
                    }
                }
            }
        }
        _ => unreachable!("dtype validated"),
    }

    Ok(Tensor::new(
        Arc::new(values_data),
        shape,
        tensor.dtype(),
        tensor.device(),
        tensor.requires_grad(),
    ))
}

fn quantile_from_unsorted_f32(
    values: &mut [f32],
    q: f64,
    interpolation: QuantileInterpolation,
) -> f32 {
    if values.len() == 1 {
        return values[0];
    }

    let max_index = (values.len() - 1) as f64;
    let pos = (q * max_index).clamp(0.0, max_index);
    let lower_idx = pos.floor() as usize;
    let upper_idx = pos.ceil() as usize;
    let weight = (pos - lower_idx as f64).clamp(0.0, 1.0);

    if lower_idx == upper_idx {
        return select_quantile_at_f32(values, lower_idx);
    }

    let value = match interpolation {
        QuantileInterpolation::Lower => select_quantile_at_f32(values, lower_idx) as f64,
        QuantileInterpolation::Higher => select_quantile_at_f32(values, upper_idx) as f64,
        QuantileInterpolation::Nearest => {
            let idx = if weight <= 0.5 { lower_idx } else { upper_idx };
            select_quantile_at_f32(values, idx) as f64
        }
        QuantileInterpolation::Linear | QuantileInterpolation::Midpoint => {
            let (lower, upper) = select_quantile_bounds_f32(values, lower_idx, upper_idx);
            interpolation.interpolate(lower as f64, upper as f64, weight)
        }
    };

    value as f32
}

fn quantile_from_unsorted_f64(
    values: &mut [f64],
    q: f64,
    interpolation: QuantileInterpolation,
) -> f64 {
    if values.len() == 1 {
        return values[0];
    }

    let max_index = (values.len() - 1) as f64;
    let pos = (q * max_index).clamp(0.0, max_index);
    let lower_idx = pos.floor() as usize;
    let upper_idx = pos.ceil() as usize;
    let weight = (pos - lower_idx as f64).clamp(0.0, 1.0);

    if lower_idx == upper_idx {
        return select_quantile_at_f64(values, lower_idx);
    }

    match interpolation {
        QuantileInterpolation::Lower => select_quantile_at_f64(values, lower_idx),
        QuantileInterpolation::Higher => select_quantile_at_f64(values, upper_idx),
        QuantileInterpolation::Nearest => {
            let idx = if weight <= 0.5 { lower_idx } else { upper_idx };
            select_quantile_at_f64(values, idx)
        }
        QuantileInterpolation::Linear | QuantileInterpolation::Midpoint => {
            let (lower, upper) = select_quantile_bounds_f64(values, lower_idx, upper_idx);
            interpolation.interpolate(lower, upper, weight)
        }
    }
}

fn select_quantile_at_f32(values: &mut [f32], idx: usize) -> f32 {
    let (_, pivot, _) = values.select_nth_unstable_by(idx, |a, b| a.total_cmp(b));
    *pivot
}

fn select_quantile_bounds_f32(
    values: &mut [f32],
    lower_idx: usize,
    upper_idx: usize,
) -> (f32, f32) {
    if lower_idx == upper_idx {
        let value = select_quantile_at_f32(values, lower_idx);
        return (value, value);
    }

    let (_, upper_pivot, _) = values.select_nth_unstable_by(upper_idx, |a, b| a.total_cmp(b));
    let upper = *upper_pivot;
    let (_, lower_pivot, _) =
        values[..upper_idx].select_nth_unstable_by(lower_idx, |a, b| a.total_cmp(b));
    let lower = *lower_pivot;
    (lower, upper)
}

fn select_quantile_at_f64(values: &mut [f64], idx: usize) -> f64 {
    let (_, pivot, _) = values.select_nth_unstable_by(idx, |a, b| a.total_cmp(b));
    *pivot
}

fn select_quantile_bounds_f64(
    values: &mut [f64],
    lower_idx: usize,
    upper_idx: usize,
) -> (f64, f64) {
    if lower_idx == upper_idx {
        let value = select_quantile_at_f64(values, lower_idx);
        return (value, value);
    }

    let (_, upper_pivot, _) = values.select_nth_unstable_by(upper_idx, |a, b| a.total_cmp(b));
    let upper = *upper_pivot;
    let (_, lower_pivot, _) =
        values[..upper_idx].select_nth_unstable_by(lower_idx, |a, b| a.total_cmp(b));
    let lower = *lower_pivot;
    (lower, upper)
}

fn median_all(tensor: &Tensor) -> Result<(Tensor, Option<Tensor>)> {
    let mut result_data = TensorData::zeros_on_device(1, tensor.dtype(), tensor.device());

    match tensor.dtype() {
        DataType::Float32 => {
            let data = tensor
                .data()
                .as_f32_slice()
                .ok_or_else(|| MinitensorError::internal_error("Failed to get f32 slice"))?;
            let mut values = Vec::with_capacity(data.len());
            for &value in data {
                if value.is_nan() {
                    result_data.as_f32_slice_mut().ok_or_else(|| {
                        MinitensorError::internal_error("Failed to get mutable f32 slice")
                    })?[0] = f32::NAN;
                    return Ok((
                        Tensor::new(
                            Arc::new(result_data),
                            Shape::scalar(),
                            tensor.dtype(),
                            tensor.device(),
                            tensor.requires_grad(),
                        ),
                        None,
                    ));
                }
                values.push(value);
            }
            let median_index = (values.len() - 1) / 2;
            values.select_nth_unstable_by(median_index, |a, b| a.total_cmp(b));
            let median = values[median_index];
            result_data.as_f32_slice_mut().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get mutable f32 slice")
            })?[0] = median;
        }
        DataType::Float64 => {
            let data = tensor
                .data()
                .as_f64_slice()
                .ok_or_else(|| MinitensorError::internal_error("Failed to get f64 slice"))?;
            let mut values = Vec::with_capacity(data.len());
            for &value in data {
                if value.is_nan() {
                    result_data.as_f64_slice_mut().ok_or_else(|| {
                        MinitensorError::internal_error("Failed to get mutable f64 slice")
                    })?[0] = f64::NAN;
                    return Ok((
                        Tensor::new(
                            Arc::new(result_data),
                            Shape::scalar(),
                            tensor.dtype(),
                            tensor.device(),
                            tensor.requires_grad(),
                        ),
                        None,
                    ));
                }
                values.push(value);
            }
            let median_index = (values.len() - 1) / 2;
            values.select_nth_unstable_by(median_index, |a, b| a.total_cmp(b));
            let median = values[median_index];
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
            let median_index = (values.len() - 1) / 2;
            values.select_nth_unstable(median_index);
            let median = values[median_index];
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
            let median_index = (values.len() - 1) / 2;
            values.select_nth_unstable(median_index);
            let median = values[median_index];
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
            let median_index = (values.len() - 1) / 2;
            values.select_nth_unstable(median_index);
            let median = values[median_index];
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
                    let mut has_nan = false;
                    for d in 0..dim_size {
                        let idx = o * outer_stride + d * inner + r;
                        let value = input[idx];
                        if value.is_nan() {
                            has_nan = true;
                            break;
                        }
                        entries.push((d, value));
                    }

                    let base = o * inner + r;
                    if has_nan {
                        values[base] = f32::NAN;
                        continue;
                    }

                    entries.select_nth_unstable_by(median_pos, cmp_f32_asc);
                    let (index, value) = entries[median_pos];
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
                    let mut has_nan = false;
                    for d in 0..dim_size {
                        let idx = o * outer_stride + d * inner + r;
                        let value = input[idx];
                        if value.is_nan() {
                            has_nan = true;
                            break;
                        }
                        entries.push((d, value));
                    }

                    let base = o * inner + r;
                    if has_nan {
                        values[base] = f64::NAN;
                        continue;
                    }

                    entries.select_nth_unstable_by(median_pos, cmp_f64_asc);
                    let (index, value) = entries[median_pos];
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

                    entries.select_nth_unstable_by(median_pos, cmp_i32_asc);
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

                    entries.select_nth_unstable_by(median_pos, cmp_i64_asc);
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

                    entries.select_nth_unstable_by(median_pos, cmp_bool_asc);
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

/// NaN-aware sum reduction along specified dimensions
pub fn nansum(tensor: &Tensor, dim: Option<Vec<isize>>, keepdim: bool) -> Result<Tensor> {
    if !tensor.dtype().is_float() {
        return sum(tensor, dim, keepdim);
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

    let result = match dim {
        None => {
            let result_shape = if keepdim {
                Shape::new(vec![1; tensor.ndim()])
            } else {
                Shape::scalar()
            };

            let mut result_data = TensorData::zeros_on_device(1, tensor.dtype(), tensor.device());
            match tensor.dtype() {
                DataType::Float32 => nansum_all_f32(tensor, &mut result_data)?,
                DataType::Float64 => nansum_all_f64(tensor, &mut result_data)?,
                _ => unreachable!("nansum only supports floating point tensors"),
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
            if dims.is_empty() {
                tensor.clone()
            } else {
                let mut result = tensor.clone();
                if keepdim {
                    for &d in &dims {
                        result = nansum_along_dim(&result, d, true)?;
                    }
                } else {
                    for &d in dims.iter().rev() {
                        result = nansum_along_dim(&result, d, false)?;
                    }
                }
                result
            }
        }
    };

    if result.requires_grad() {
        let mask = mask.ok_or_else(|| {
            MinitensorError::internal_error("nansum expected mask for gradient computation")
        })?;
        let grad_fn = Arc::new(NanSumBackward {
            input_id: tensor.id(),
            input_shape: tensor.shape().dims().to_vec(),
            dims: dims_clone,
            keepdim,
            mask,
        });
        let mut result_with_grad = result;
        result_with_grad.set_grad_fn(Some(grad_fn.clone()));
        add_to_graph(&result_with_grad, Some(grad_fn))?;
        Ok(result_with_grad)
    } else {
        Ok(result)
    }
}
