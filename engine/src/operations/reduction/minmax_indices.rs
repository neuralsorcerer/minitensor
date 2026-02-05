// Copyright (c) 2026 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

fn min_along_dim_with_indices(
    tensor: &Tensor,
    dim: usize,
    keepdim: bool,
) -> Result<(Tensor, Tensor)> {
    let layout = reduction_layout(tensor, dim, keepdim)?;
    let mut values_data =
        TensorData::zeros_on_device(layout.output_shape.numel(), tensor.dtype(), tensor.device());
    let mut indices_data = TensorData::zeros_on_device(
        layout.output_shape.numel(),
        DataType::Int64,
        tensor.device(),
    );

    let indices = indices_data
        .as_i64_slice_mut()
        .ok_or_else(|| MinitensorError::internal_error("Failed to get mutable i64 slice"))?;

    match tensor.dtype() {
        DataType::Float32 => {
            let input = tensor
                .data()
                .as_f32_slice()
                .ok_or_else(|| MinitensorError::internal_error("Failed to get f32 slice"))?;
            let values = values_data.as_f32_slice_mut().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get mutable f32 slice")
            })?;
            for o in 0..layout.outer {
                for r in 0..layout.inner {
                    let mut min_val = f32::INFINITY;
                    let mut min_idx = 0usize;
                    for d in 0..layout.dim_size {
                        let idx = o * layout.outer_stride + d * layout.inner + r;
                        let val = input[idx];
                        if !val.is_nan() && val < min_val {
                            min_val = val;
                            min_idx = d;
                        }
                    }
                    let out_idx = o * layout.inner + r;
                    values[out_idx] = min_val;
                    indices[out_idx] = min_idx as i64;
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
            for o in 0..layout.outer {
                for r in 0..layout.inner {
                    let mut min_val = f64::INFINITY;
                    let mut min_idx = 0usize;
                    for d in 0..layout.dim_size {
                        let idx = o * layout.outer_stride + d * layout.inner + r;
                        let val = input[idx];
                        if !val.is_nan() && val < min_val {
                            min_val = val;
                            min_idx = d;
                        }
                    }
                    let out_idx = o * layout.inner + r;
                    values[out_idx] = min_val;
                    indices[out_idx] = min_idx as i64;
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
            for o in 0..layout.outer {
                for r in 0..layout.inner {
                    let mut min_val = i32::MAX;
                    let mut min_idx = 0usize;
                    for d in 0..layout.dim_size {
                        let idx = o * layout.outer_stride + d * layout.inner + r;
                        let val = input[idx];
                        if val < min_val {
                            min_val = val;
                            min_idx = d;
                        }
                    }
                    let out_idx = o * layout.inner + r;
                    values[out_idx] = min_val;
                    indices[out_idx] = min_idx as i64;
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
            for o in 0..layout.outer {
                for r in 0..layout.inner {
                    let mut min_val = i64::MAX;
                    let mut min_idx = 0usize;
                    for d in 0..layout.dim_size {
                        let idx = o * layout.outer_stride + d * layout.inner + r;
                        let val = input[idx];
                        if val < min_val {
                            min_val = val;
                            min_idx = d;
                        }
                    }
                    let out_idx = o * layout.inner + r;
                    values[out_idx] = min_val;
                    indices[out_idx] = min_idx as i64;
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
            for o in 0..layout.outer {
                for r in 0..layout.inner {
                    let mut min_val = true;
                    let mut min_idx = 0usize;
                    for d in 0..layout.dim_size {
                        let idx = o * layout.outer_stride + d * layout.inner + r;
                        if !input[idx] {
                            min_val = false;
                            min_idx = d;
                            break;
                        }
                    }
                    let out_idx = o * layout.inner + r;
                    values[out_idx] = min_val;
                    indices[out_idx] = min_idx as i64;
                }
            }
        }
    }

    Ok((
        Tensor::new(
            Arc::new(values_data),
            layout.output_shape.clone(),
            tensor.dtype(),
            tensor.device(),
            tensor.requires_grad(),
        ),
        Tensor::new(
            Arc::new(indices_data),
            layout.output_shape,
            DataType::Int64,
            tensor.device(),
            false,
        ),
    ))
}

fn nanmin_along_dim_with_indices(
    tensor: &Tensor,
    dim: usize,
    keepdim: bool,
) -> Result<(Tensor, Tensor)> {
    let layout = reduction_layout(tensor, dim, keepdim)?;
    let mut values_data =
        TensorData::zeros_on_device(layout.output_shape.numel(), tensor.dtype(), tensor.device());
    let mut indices_data = TensorData::zeros_on_device(
        layout.output_shape.numel(),
        DataType::Int64,
        tensor.device(),
    );

    let indices = indices_data
        .as_i64_slice_mut()
        .ok_or_else(|| MinitensorError::internal_error("Failed to get mutable i64 slice"))?;

    match tensor.dtype() {
        DataType::Float32 => {
            let input = tensor
                .data()
                .as_f32_slice()
                .ok_or_else(|| MinitensorError::internal_error("Failed to get f32 slice"))?;
            let values = values_data.as_f32_slice_mut().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get mutable f32 slice")
            })?;
            for o in 0..layout.outer {
                for r in 0..layout.inner {
                    let mut min_val = f32::NAN;
                    let mut min_idx = 0usize;
                    for d in 0..layout.dim_size {
                        let idx = o * layout.outer_stride + d * layout.inner + r;
                        let val = input[idx];
                        if !val.is_nan() && (min_val.is_nan() || val < min_val) {
                            min_val = val;
                            min_idx = d;
                        }
                    }
                    let out_idx = o * layout.inner + r;
                    values[out_idx] = min_val;
                    indices[out_idx] = min_idx as i64;
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
            for o in 0..layout.outer {
                for r in 0..layout.inner {
                    let mut min_val = f64::NAN;
                    let mut min_idx = 0usize;
                    for d in 0..layout.dim_size {
                        let idx = o * layout.outer_stride + d * layout.inner + r;
                        let val = input[idx];
                        if !val.is_nan() && (min_val.is_nan() || val < min_val) {
                            min_val = val;
                            min_idx = d;
                        }
                    }
                    let out_idx = o * layout.inner + r;
                    values[out_idx] = min_val;
                    indices[out_idx] = min_idx as i64;
                }
            }
        }
        _ => {
            return Err(MinitensorError::invalid_operation(
                "nanmin only supports floating point tensors",
            ));
        }
    }

    Ok((
        Tensor::new(
            Arc::new(values_data),
            layout.output_shape.clone(),
            tensor.dtype(),
            tensor.device(),
            tensor.requires_grad(),
        ),
        Tensor::new(
            Arc::new(indices_data),
            layout.output_shape,
            DataType::Int64,
            tensor.device(),
            false,
        ),
    ))
}

fn argmax_along_dim(tensor: &Tensor, dim: usize, keepdim: bool) -> Result<Tensor> {
    let layout = reduction_layout(tensor, dim, keepdim)?;
    let mut result_data = TensorData::zeros_on_device(
        layout.output_shape.numel(),
        DataType::Int64,
        tensor.device(),
    );

    let output = result_data
        .as_i64_slice_mut()
        .ok_or_else(|| MinitensorError::internal_error("Failed to get mutable i64 slice"))?;

    match tensor.dtype() {
        DataType::Float32 => {
            let input = tensor
                .data()
                .as_f32_slice()
                .ok_or_else(|| MinitensorError::internal_error("Failed to get f32 slice"))?;
            for o in 0..layout.outer {
                for r in 0..layout.inner {
                    let mut max_val = f32::NEG_INFINITY;
                    let mut max_idx = 0usize;
                    for d in 0..layout.dim_size {
                        let idx = o * layout.outer_stride + d * layout.inner + r;
                        let val = input[idx];
                        if !val.is_nan() && val > max_val {
                            max_val = val;
                            max_idx = d;
                        }
                    }
                    output[o * layout.inner + r] = max_idx as i64;
                }
            }
        }
        DataType::Float64 => {
            let input = tensor
                .data()
                .as_f64_slice()
                .ok_or_else(|| MinitensorError::internal_error("Failed to get f64 slice"))?;
            for o in 0..layout.outer {
                for r in 0..layout.inner {
                    let mut max_val = f64::NEG_INFINITY;
                    let mut max_idx = 0usize;
                    for d in 0..layout.dim_size {
                        let idx = o * layout.outer_stride + d * layout.inner + r;
                        let val = input[idx];
                        if !val.is_nan() && val > max_val {
                            max_val = val;
                            max_idx = d;
                        }
                    }
                    output[o * layout.inner + r] = max_idx as i64;
                }
            }
        }
        DataType::Int32 => {
            let input = tensor
                .data()
                .as_i32_slice()
                .ok_or_else(|| MinitensorError::internal_error("Failed to get i32 slice"))?;
            for o in 0..layout.outer {
                for r in 0..layout.inner {
                    let mut max_val = i32::MIN;
                    let mut max_idx = 0usize;
                    for d in 0..layout.dim_size {
                        let idx = o * layout.outer_stride + d * layout.inner + r;
                        let val = input[idx];
                        if val > max_val {
                            max_val = val;
                            max_idx = d;
                        }
                    }
                    output[o * layout.inner + r] = max_idx as i64;
                }
            }
        }
        DataType::Int64 => {
            let input = tensor
                .data()
                .as_i64_slice()
                .ok_or_else(|| MinitensorError::internal_error("Failed to get i64 slice"))?;
            for o in 0..layout.outer {
                for r in 0..layout.inner {
                    let mut max_val = i64::MIN;
                    let mut max_idx = 0usize;
                    for d in 0..layout.dim_size {
                        let idx = o * layout.outer_stride + d * layout.inner + r;
                        let val = input[idx];
                        if val > max_val {
                            max_val = val;
                            max_idx = d;
                        }
                    }
                    output[o * layout.inner + r] = max_idx as i64;
                }
            }
        }
        DataType::Bool => {
            let input = tensor
                .data()
                .as_bool_slice()
                .ok_or_else(|| MinitensorError::internal_error("Failed to get bool slice"))?;
            for o in 0..layout.outer {
                for r in 0..layout.inner {
                    let mut max_idx = 0usize;
                    for d in 0..layout.dim_size {
                        let idx = o * layout.outer_stride + d * layout.inner + r;
                        if input[idx] {
                            max_idx = d;
                            break;
                        }
                    }
                    output[o * layout.inner + r] = max_idx as i64;
                }
            }
        }
    }

    Ok(Tensor::new(
        Arc::new(result_data),
        layout.output_shape,
        DataType::Int64,
        tensor.device(),
        false,
    ))
}
