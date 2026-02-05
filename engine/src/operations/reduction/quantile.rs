// Copyright (c) 2026 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

fn quantile_along_dim(
    tensor: &Tensor,
    dim: usize,
    keepdim: bool,
    q: f64,
    interpolation: QuantileInterpolation,
) -> Result<Tensor> {
    let dims = tensor.shape().dims();
    let dim_size = if dims.is_empty() { 1 } else { dims[dim] };

    if dim_size == 0 {
        return Err(MinitensorError::invalid_argument(
            "quantile() does not support reductions over empty dimensions".to_string(),
        ));
    }

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
                fill_quantile_single_f32(input, values, outer, inner, outer_stride);
            } else {
                let mut buffer = Vec::with_capacity(dim_size);
                for o in 0..outer {
                    for r in 0..inner {
                        buffer.clear();
                        let mut has_nan = false;
                        for d in 0..dim_size {
                            let idx = o * outer_stride + d * inner + r;
                            let value = input[idx];
                            if value.is_nan() {
                                has_nan = true;
                                break;
                            }
                            buffer.push(value);
                        }

                        if has_nan {
                            values[o * inner + r] = f32::NAN;
                            continue;
                        }

                        values[o * inner + r] =
                            quantile_from_unsorted_f32(&mut buffer, q, interpolation);
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
                fill_quantile_single_f64(input, values, outer, inner, outer_stride);
            } else {
                let mut buffer = Vec::with_capacity(dim_size);
                for o in 0..outer {
                    for r in 0..inner {
                        buffer.clear();
                        let mut has_nan = false;
                        for d in 0..dim_size {
                            let idx = o * outer_stride + d * inner + r;
                            let value = input[idx];
                            if value.is_nan() {
                                has_nan = true;
                                break;
                            }
                            buffer.push(value);
                        }

                        if has_nan {
                            values[o * inner + r] = f64::NAN;
                            continue;
                        }

                        values[o * inner + r] =
                            quantile_from_unsorted_f64(&mut buffer, q, interpolation);
                    }
                }
            }
        }
        _ => unreachable!("dtype validated"),
    }

    Ok(Tensor::new(
        Arc::new(values_data),
        values_shape,
        tensor.dtype(),
        tensor.device(),
        tensor.requires_grad(),
    ))
}

fn nanquantile_along_dim(
    tensor: &Tensor,
    dim: usize,
    keepdim: bool,
    q: f64,
    interpolation: QuantileInterpolation,
) -> Result<Tensor> {
    let dims = tensor.shape().dims();
    let dim_size = if dims.is_empty() { 1 } else { dims[dim] };

    if dim_size == 0 {
        return Err(MinitensorError::invalid_argument(
            "nanquantile() does not support reductions over empty dimensions".to_string(),
        ));
    }

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
                fill_nanquantile_single_f32(input, values, outer, inner, outer_stride)?;
            } else {
                let mut buffer = Vec::with_capacity(dim_size);
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

                        let quant = quantile_from_unsorted_f32(&mut buffer, q, interpolation);
                        values[o * inner + r] = quant;
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
                fill_nanquantile_single_f64(input, values, outer, inner, outer_stride)?;
            } else {
                let mut buffer = Vec::with_capacity(dim_size);
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

                        let quant = quantile_from_unsorted_f64(&mut buffer, q, interpolation);
                        values[o * inner + r] = quant;
                    }
                }
            }
        }
        _ => unreachable!("dtype validated"),
    }

    Ok(Tensor::new(
        Arc::new(values_data),
        values_shape,
        tensor.dtype(),
        tensor.device(),
        tensor.requires_grad(),
    ))
}

fn quantiles_all(
    tensor: &Tensor,
    qs: &[f64],
    keepdim: bool,
    interpolation: QuantileInterpolation,
) -> Result<Tensor> {
    let q_len = qs.len();
    let output_dims = quantiles_output_dims(tensor.ndim(), q_len, keepdim);

    let shape = Shape::new(output_dims);
    let mut values_data =
        TensorData::zeros_on_device(shape.numel(), tensor.dtype(), tensor.device());

    match tensor.dtype() {
        DataType::Float32 => {
            let data = tensor
                .data()
                .as_f32_slice()
                .ok_or_else(|| MinitensorError::internal_error("Failed to get f32 slice"))?;
            let values = values_data.as_f32_slice_mut().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get mutable f32 slice")
            })?;
            if data.len() == 1 {
                fill_quantiles_all_single_f32(data[0], values);
                return Ok(Tensor::new(
                    Arc::new(values_data),
                    shape,
                    tensor.dtype(),
                    tensor.device(),
                    tensor.requires_grad(),
                ));
            }
            if q_len == 1 {
                let mut buffer = Vec::with_capacity(data.len());
                for &value in data {
                    if value.is_nan() {
                        values[0] = f32::NAN;
                        return Ok(Tensor::new(
                            Arc::new(values_data),
                            shape,
                            tensor.dtype(),
                            tensor.device(),
                            tensor.requires_grad(),
                        ));
                    }
                    buffer.push(value);
                }
                values[0] = quantile_from_unsorted_f32(&mut buffer, qs[0], interpolation);
                return Ok(Tensor::new(
                    Arc::new(values_data),
                    shape,
                    tensor.dtype(),
                    tensor.device(),
                    tensor.requires_grad(),
                ));
            }
            let positions = quantile_positions_for_len(data.len(), qs);
            let mut sorted = Vec::with_capacity(data.len());
            for &value in data {
                if value.is_nan() {
                    for out in values.iter_mut() {
                        *out = f32::NAN;
                    }
                    return Ok(Tensor::new(
                        Arc::new(values_data),
                        shape,
                        tensor.dtype(),
                        tensor.device(),
                        tensor.requires_grad(),
                    ));
                }
                sorted.push(value);
            }
            sorted.sort_by(|a, b| a.total_cmp(b));
            quantiles_from_sorted_f32(&sorted, &positions, interpolation, values);
        }
        DataType::Float64 => {
            let data = tensor
                .data()
                .as_f64_slice()
                .ok_or_else(|| MinitensorError::internal_error("Failed to get f64 slice"))?;
            let values = values_data.as_f64_slice_mut().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get mutable f64 slice")
            })?;
            if data.len() == 1 {
                fill_quantiles_all_single_f64(data[0], values);
                return Ok(Tensor::new(
                    Arc::new(values_data),
                    shape,
                    tensor.dtype(),
                    tensor.device(),
                    tensor.requires_grad(),
                ));
            }
            if q_len == 1 {
                let mut buffer = Vec::with_capacity(data.len());
                for &value in data {
                    if value.is_nan() {
                        values[0] = f64::NAN;
                        return Ok(Tensor::new(
                            Arc::new(values_data),
                            shape,
                            tensor.dtype(),
                            tensor.device(),
                            tensor.requires_grad(),
                        ));
                    }
                    buffer.push(value);
                }
                values[0] = quantile_from_unsorted_f64(&mut buffer, qs[0], interpolation);
                return Ok(Tensor::new(
                    Arc::new(values_data),
                    shape,
                    tensor.dtype(),
                    tensor.device(),
                    tensor.requires_grad(),
                ));
            }
            let positions = quantile_positions_for_len(data.len(), qs);
            let mut sorted = Vec::with_capacity(data.len());
            for &value in data {
                if value.is_nan() {
                    for out in values.iter_mut() {
                        *out = f64::NAN;
                    }
                    return Ok(Tensor::new(
                        Arc::new(values_data),
                        shape,
                        tensor.dtype(),
                        tensor.device(),
                        tensor.requires_grad(),
                    ));
                }
                sorted.push(value);
            }
            sorted.sort_by(|a, b| a.total_cmp(b));
            quantiles_from_sorted_f64(&sorted, &positions, interpolation, values);
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

fn nanquantiles_all(
    tensor: &Tensor,
    qs: &[f64],
    keepdim: bool,
    interpolation: QuantileInterpolation,
) -> Result<Tensor> {
    let q_len = qs.len();
    let output_dims = quantiles_output_dims(tensor.ndim(), q_len, keepdim);

    let shape = Shape::new(output_dims);
    let mut values_data =
        TensorData::zeros_on_device(shape.numel(), tensor.dtype(), tensor.device());

    match tensor.dtype() {
        DataType::Float32 => {
            let data = tensor
                .data()
                .as_f32_slice()
                .ok_or_else(|| MinitensorError::internal_error("Failed to get f32 slice"))?;
            let values = values_data.as_f32_slice_mut().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get mutable f32 slice")
            })?;
            if data.len() == 1 {
                fill_nanquantiles_all_single_f32(data[0], values)?;
                return Ok(Tensor::new(
                    Arc::new(values_data),
                    shape,
                    tensor.dtype(),
                    tensor.device(),
                    tensor.requires_grad(),
                ));
            }
            if q_len == 1 {
                let mut buffer: Vec<f32> = data.iter().copied().filter(|v| !v.is_nan()).collect();
                if buffer.is_empty() {
                    return Err(MinitensorError::invalid_argument(
                        NANQUANTILE_ALL_NAN_ERR.to_string(),
                    ));
                }
                values[0] = quantile_from_unsorted_f32(&mut buffer, qs[0], interpolation);
                return Ok(Tensor::new(
                    Arc::new(values_data),
                    shape,
                    tensor.dtype(),
                    tensor.device(),
                    tensor.requires_grad(),
                ));
            }
            let mut sorted: Vec<f32> = data.iter().copied().filter(|v| !v.is_nan()).collect();
            if sorted.is_empty() {
                return Err(MinitensorError::invalid_argument(
                    NANQUANTILE_ALL_NAN_ERR.to_string(),
                ));
            }
            let positions = quantile_positions_for_len(sorted.len(), qs);
            sorted.sort_by(|a, b| a.total_cmp(b));
            quantiles_from_sorted_f32(&sorted, &positions, interpolation, values);
        }
        DataType::Float64 => {
            let data = tensor
                .data()
                .as_f64_slice()
                .ok_or_else(|| MinitensorError::internal_error("Failed to get f64 slice"))?;
            let values = values_data.as_f64_slice_mut().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get mutable f64 slice")
            })?;
            if data.len() == 1 {
                fill_nanquantiles_all_single_f64(data[0], values)?;
                return Ok(Tensor::new(
                    Arc::new(values_data),
                    shape,
                    tensor.dtype(),
                    tensor.device(),
                    tensor.requires_grad(),
                ));
            }
            if q_len == 1 {
                let mut buffer: Vec<f64> = data.iter().copied().filter(|v| !v.is_nan()).collect();
                if buffer.is_empty() {
                    return Err(MinitensorError::invalid_argument(
                        NANQUANTILE_ALL_NAN_ERR.to_string(),
                    ));
                }
                values[0] = quantile_from_unsorted_f64(&mut buffer, qs[0], interpolation);
                return Ok(Tensor::new(
                    Arc::new(values_data),
                    shape,
                    tensor.dtype(),
                    tensor.device(),
                    tensor.requires_grad(),
                ));
            }
            let mut sorted: Vec<f64> = data.iter().copied().filter(|v| !v.is_nan()).collect();
            if sorted.is_empty() {
                return Err(MinitensorError::invalid_argument(
                    NANQUANTILE_ALL_NAN_ERR.to_string(),
                ));
            }
            let positions = quantile_positions_for_len(sorted.len(), qs);
            sorted.sort_by(|a, b| a.total_cmp(b));
            quantiles_from_sorted_f64(&sorted, &positions, interpolation, values);
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

fn quantiles_output_dims(tensor_ndim: usize, q_len: usize, keepdim: bool) -> Vec<usize> {
    if keepdim && tensor_ndim > 0 {
        let mut dims = vec![1; tensor_ndim + 1];
        dims[0] = q_len;
        dims
    } else {
        vec![q_len]
    }
}

fn quantiles_along_dim(
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
            "quantile() does not support empty slices".to_string(),
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
                fill_quantiles_single_f32(input, values, outer, inner, outer_stride, q_len);
            } else {
                let mut buffer = Vec::with_capacity(dim_size);
                if q_len == 1 {
                    let q_value = qs[0];
                    for o in 0..outer {
                        for r in 0..inner {
                            buffer.clear();
                            let mut has_nan = false;
                            for d in 0..dim_size {
                                let idx = o * outer_stride + d * inner + r;
                                let value = input[idx];
                                if value.is_nan() {
                                    has_nan = true;
                                    break;
                                }
                                buffer.push(value);
                            }

                            let out_idx = o * inner + r;
                            if has_nan {
                                values[out_idx] = f32::NAN;
                                continue;
                            }

                            values[out_idx] =
                                quantile_from_unsorted_f32(&mut buffer, q_value, interpolation);
                        }
                    }
                } else {
                    let positions = quantile_positions_for_len(dim_size, qs);
                    for o in 0..outer {
                        for r in 0..inner {
                            buffer.clear();
                            let mut has_nan = false;
                            for d in 0..dim_size {
                                let idx = o * outer_stride + d * inner + r;
                                let value = input[idx];
                                if value.is_nan() {
                                    has_nan = true;
                                    break;
                                }
                                buffer.push(value);
                            }

                            if has_nan {
                                for qi in 0..q_len {
                                    let out_idx = ((qi * outer) + o) * inner + r;
                                    values[out_idx] = f32::NAN;
                                }
                                continue;
                            }

                            buffer.sort_by(|a, b| a.total_cmp(b));
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
                fill_quantiles_single_f64(input, values, outer, inner, outer_stride, q_len);
            } else {
                let mut buffer = Vec::with_capacity(dim_size);
                if q_len == 1 {
                    let q_value = qs[0];
                    for o in 0..outer {
                        for r in 0..inner {
                            buffer.clear();
                            let mut has_nan = false;
                            for d in 0..dim_size {
                                let idx = o * outer_stride + d * inner + r;
                                let value = input[idx];
                                if value.is_nan() {
                                    has_nan = true;
                                    break;
                                }
                                buffer.push(value);
                            }

                            let out_idx = o * inner + r;
                            if has_nan {
                                values[out_idx] = f64::NAN;
                                continue;
                            }

                            values[out_idx] =
                                quantile_from_unsorted_f64(&mut buffer, q_value, interpolation);
                        }
                    }
                } else {
                    let positions = quantile_positions_for_len(dim_size, qs);
                    for o in 0..outer {
                        for r in 0..inner {
                            buffer.clear();
                            let mut has_nan = false;
                            for d in 0..dim_size {
                                let idx = o * outer_stride + d * inner + r;
                                let value = input[idx];
                                if value.is_nan() {
                                    has_nan = true;
                                    break;
                                }
                                buffer.push(value);
                            }

                            if has_nan {
                                for qi in 0..q_len {
                                    let out_idx = ((qi * outer) + o) * inner + r;
                                    values[out_idx] = f64::NAN;
                                }
                                continue;
                            }

                            buffer.sort_by(|a, b| a.total_cmp(b));
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
