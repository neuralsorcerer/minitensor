// Copyright (c) 2026 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

fn nanmax_all_f64(tensor: &Tensor, result_data: &mut TensorData) -> Result<()> {
    let data = tensor
        .data()
        .as_f64_slice()
        .ok_or_else(|| MinitensorError::internal_error("Failed to get f64 slice"))?;

    let (max_val, found) = data
        .par_iter()
        .map(|&v| {
            if v.is_nan() {
                (f64::NEG_INFINITY, false)
            } else {
                (v, true)
            }
        })
        .reduce(
            || (f64::NEG_INFINITY, false),
            |(a_val, a_found), (b_val, b_found)| match (a_found, b_found) {
                (true, true) => (a_val.max(b_val), true),
                (true, false) => (a_val, true),
                (false, true) => (b_val, true),
                (false, false) => (f64::NEG_INFINITY, false),
            },
        );

    let result_slice = result_data
        .as_f64_slice_mut()
        .ok_or_else(|| MinitensorError::internal_error("Failed to get mutable f64 slice"))?;

    result_slice[0] = if found { max_val } else { f64::NAN };
    Ok(())
}

fn nanmin_all_f32(tensor: &Tensor, result_data: &mut TensorData) -> Result<()> {
    let data = tensor
        .data()
        .as_f32_slice()
        .ok_or_else(|| MinitensorError::internal_error("Failed to get f32 slice"))?;

    let (min_val, found) = data
        .par_iter()
        .map(|&v| {
            if v.is_nan() {
                (f32::INFINITY, false)
            } else {
                (v, true)
            }
        })
        .reduce(
            || (f32::INFINITY, false),
            |(a_val, a_found), (b_val, b_found)| match (a_found, b_found) {
                (true, true) => (a_val.min(b_val), true),
                (true, false) => (a_val, true),
                (false, true) => (b_val, true),
                (false, false) => (f32::INFINITY, false),
            },
        );

    let result_slice = result_data
        .as_f32_slice_mut()
        .ok_or_else(|| MinitensorError::internal_error("Failed to get mutable f32 slice"))?;

    result_slice[0] = if found { min_val } else { f32::NAN };
    Ok(())
}

fn nanmin_all_f64(tensor: &Tensor, result_data: &mut TensorData) -> Result<()> {
    let data = tensor
        .data()
        .as_f64_slice()
        .ok_or_else(|| MinitensorError::internal_error("Failed to get f64 slice"))?;

    let (min_val, found) = data
        .par_iter()
        .map(|&v| {
            if v.is_nan() {
                (f64::INFINITY, false)
            } else {
                (v, true)
            }
        })
        .reduce(
            || (f64::INFINITY, false),
            |(a_val, a_found), (b_val, b_found)| match (a_found, b_found) {
                (true, true) => (a_val.min(b_val), true),
                (true, false) => (a_val, true),
                (false, true) => (b_val, true),
                (false, false) => (f64::INFINITY, false),
            },
        );

    let result_slice = result_data
        .as_f64_slice_mut()
        .ok_or_else(|| MinitensorError::internal_error("Failed to get mutable f64 slice"))?;

    result_slice[0] = if found { min_val } else { f64::NAN };
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
            let v1 = if v1.is_nan() { f32::NEG_INFINITY } else { v1 };
            let v2 = if v2.is_nan() { f32::NEG_INFINITY } else { v2 };
            if v1 > v2 {
                (i1, v1)
            } else if v2 > v1 {
                (i2, v2)
            } else if i1 <= i2 {
                (i1, v1)
            } else {
                (i2, v2)
            }
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
            let v1 = if v1.is_nan() { f64::NEG_INFINITY } else { v1 };
            let v2 = if v2.is_nan() { f64::NEG_INFINITY } else { v2 };
            if v1 > v2 {
                (i1, v1)
            } else if v2 > v1 {
                (i2, v2)
            } else if i1 <= i2 {
                (i1, v1)
            } else {
                (i2, v2)
            }
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
            let v1 = if v1.is_nan() { f32::INFINITY } else { v1 };
            let v2 = if v2.is_nan() { f32::INFINITY } else { v2 };
            if v1 < v2 {
                (i1, v1)
            } else if v2 < v1 {
                (i2, v2)
            } else if i1 <= i2 {
                (i1, v1)
            } else {
                (i2, v2)
            }
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
            let v1 = if v1.is_nan() { f64::INFINITY } else { v1 };
            let v2 = if v2.is_nan() { f64::INFINITY } else { v2 };
            if v1 < v2 {
                (i1, v1)
            } else if v2 < v1 {
                (i2, v2)
            } else if i1 <= i2 {
                (i1, v1)
            } else {
                (i2, v2)
            }
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

struct DimReductionLayout {
    output_shape: Shape,
    dim_size: usize,
    outer: usize,
    inner: usize,
    outer_stride: usize,
}

fn reduction_layout(tensor: &Tensor, dim: usize, keepdim: bool) -> Result<DimReductionLayout> {
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
    let dim_size = input_shape[dim];
    let outer = input_shape[..dim].iter().product::<usize>();
    let inner = input_shape[dim + 1..].iter().product::<usize>();
    let outer_stride = dim_size * inner;

    Ok(DimReductionLayout {
        output_shape: Shape::new(output_shape),
        dim_size,
        outer,
        inner,
        outer_stride,
    })
}

// Placeholder implementations for dimensional operations
fn max_along_dim(tensor: &Tensor, dim: usize, keepdim: bool) -> Result<Tensor> {
    let layout = reduction_layout(tensor, dim, keepdim)?;
    let mut result_data =
        TensorData::zeros_on_device(layout.output_shape.numel(), tensor.dtype(), tensor.device());

    match tensor.dtype() {
        DataType::Float32 => {
            let input = tensor
                .data()
                .as_f32_slice()
                .ok_or_else(|| MinitensorError::internal_error("Failed to get f32 slice"))?;
            let output = result_data.as_f32_slice_mut().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get mutable f32 slice")
            })?;

            for o in 0..layout.outer {
                for r in 0..layout.inner {
                    let mut max_val = f32::NEG_INFINITY;
                    for d in 0..layout.dim_size {
                        let idx = o * layout.outer_stride + d * layout.inner + r;
                        let val = input[idx];
                        if !val.is_nan() {
                            max_val = max_val.max(val);
                        }
                    }
                    output[o * layout.inner + r] = max_val;
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

            for o in 0..layout.outer {
                for r in 0..layout.inner {
                    let mut max_val = f64::NEG_INFINITY;
                    for d in 0..layout.dim_size {
                        let idx = o * layout.outer_stride + d * layout.inner + r;
                        let val = input[idx];
                        if !val.is_nan() {
                            max_val = max_val.max(val);
                        }
                    }
                    output[o * layout.inner + r] = max_val;
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

            for o in 0..layout.outer {
                for r in 0..layout.inner {
                    let mut max_val = i32::MIN;
                    for d in 0..layout.dim_size {
                        let idx = o * layout.outer_stride + d * layout.inner + r;
                        max_val = max_val.max(input[idx]);
                    }
                    output[o * layout.inner + r] = max_val;
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

            for o in 0..layout.outer {
                for r in 0..layout.inner {
                    let mut max_val = i64::MIN;
                    for d in 0..layout.dim_size {
                        let idx = o * layout.outer_stride + d * layout.inner + r;
                        max_val = max_val.max(input[idx]);
                    }
                    output[o * layout.inner + r] = max_val;
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

            for o in 0..layout.outer {
                for r in 0..layout.inner {
                    let mut max_val = false;
                    for d in 0..layout.dim_size {
                        let idx = o * layout.outer_stride + d * layout.inner + r;
                        max_val |= input[idx];
                        if max_val {
                            break;
                        }
                    }
                    output[o * layout.inner + r] = max_val;
                }
            }
        }
    }

    Ok(Tensor::new(
        Arc::new(result_data),
        layout.output_shape,
        tensor.dtype(),
        tensor.device(),
        tensor.requires_grad(),
    ))
}

fn min_along_dim(tensor: &Tensor, dim: usize, keepdim: bool) -> Result<Tensor> {
    let layout = reduction_layout(tensor, dim, keepdim)?;
    let mut result_data =
        TensorData::zeros_on_device(layout.output_shape.numel(), tensor.dtype(), tensor.device());

    match tensor.dtype() {
        DataType::Float32 => {
            let input = tensor
                .data()
                .as_f32_slice()
                .ok_or_else(|| MinitensorError::internal_error("Failed to get f32 slice"))?;
            let output = result_data.as_f32_slice_mut().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get mutable f32 slice")
            })?;

            for o in 0..layout.outer {
                for r in 0..layout.inner {
                    let mut min_val = f32::INFINITY;
                    for d in 0..layout.dim_size {
                        let idx = o * layout.outer_stride + d * layout.inner + r;
                        let val = input[idx];
                        if !val.is_nan() {
                            min_val = min_val.min(val);
                        }
                    }
                    output[o * layout.inner + r] = min_val;
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

            for o in 0..layout.outer {
                for r in 0..layout.inner {
                    let mut min_val = f64::INFINITY;
                    for d in 0..layout.dim_size {
                        let idx = o * layout.outer_stride + d * layout.inner + r;
                        let val = input[idx];
                        if !val.is_nan() {
                            min_val = min_val.min(val);
                        }
                    }
                    output[o * layout.inner + r] = min_val;
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

            for o in 0..layout.outer {
                for r in 0..layout.inner {
                    let mut min_val = i32::MAX;
                    for d in 0..layout.dim_size {
                        let idx = o * layout.outer_stride + d * layout.inner + r;
                        min_val = min_val.min(input[idx]);
                    }
                    output[o * layout.inner + r] = min_val;
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

            for o in 0..layout.outer {
                for r in 0..layout.inner {
                    let mut min_val = i64::MAX;
                    for d in 0..layout.dim_size {
                        let idx = o * layout.outer_stride + d * layout.inner + r;
                        min_val = min_val.min(input[idx]);
                    }
                    output[o * layout.inner + r] = min_val;
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

            for o in 0..layout.outer {
                for r in 0..layout.inner {
                    let mut min_val = true;
                    for d in 0..layout.dim_size {
                        let idx = o * layout.outer_stride + d * layout.inner + r;
                        min_val &= input[idx];
                        if !min_val {
                            break;
                        }
                    }
                    output[o * layout.inner + r] = min_val;
                }
            }
        }
    }

    Ok(Tensor::new(
        Arc::new(result_data),
        layout.output_shape,
        tensor.dtype(),
        tensor.device(),
        tensor.requires_grad(),
    ))
}

fn max_along_dim_with_indices(
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
                    let out_idx = o * layout.inner + r;
                    values[out_idx] = max_val;
                    indices[out_idx] = max_idx as i64;
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
                    let out_idx = o * layout.inner + r;
                    values[out_idx] = max_val;
                    indices[out_idx] = max_idx as i64;
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
                    let out_idx = o * layout.inner + r;
                    values[out_idx] = max_val;
                    indices[out_idx] = max_idx as i64;
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
                    let out_idx = o * layout.inner + r;
                    values[out_idx] = max_val;
                    indices[out_idx] = max_idx as i64;
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
                    let mut max_val = false;
                    let mut max_idx = 0usize;
                    for d in 0..layout.dim_size {
                        let idx = o * layout.outer_stride + d * layout.inner + r;
                        if input[idx] {
                            max_val = true;
                            max_idx = d;
                            break;
                        }
                    }
                    let out_idx = o * layout.inner + r;
                    values[out_idx] = max_val;
                    indices[out_idx] = max_idx as i64;
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

fn nanmax_along_dim_with_indices(
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
                    let mut max_val = f32::NAN;
                    let mut max_idx = 0usize;
                    for d in 0..layout.dim_size {
                        let idx = o * layout.outer_stride + d * layout.inner + r;
                        let val = input[idx];
                        if !val.is_nan() && (max_val.is_nan() || val > max_val) {
                            max_val = val;
                            max_idx = d;
                        }
                    }
                    let out_idx = o * layout.inner + r;
                    values[out_idx] = max_val;
                    indices[out_idx] = max_idx as i64;
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
                    let mut max_val = f64::NAN;
                    let mut max_idx = 0usize;
                    for d in 0..layout.dim_size {
                        let idx = o * layout.outer_stride + d * layout.inner + r;
                        let val = input[idx];
                        if !val.is_nan() && (max_val.is_nan() || val > max_val) {
                            max_val = val;
                            max_idx = d;
                        }
                    }
                    let out_idx = o * layout.inner + r;
                    values[out_idx] = max_val;
                    indices[out_idx] = max_idx as i64;
                }
            }
        }
        _ => {
            return Err(MinitensorError::invalid_operation(
                "nanmax only supports floating point tensors",
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
