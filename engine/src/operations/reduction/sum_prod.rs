// Copyright (c) 2026 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

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

fn nansum_along_dim_f32(tensor: &Tensor, result_data: &mut TensorData, dim: usize) -> Result<()> {
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
        result_slice[0] = input_data.iter().filter(|v| !v.is_nan()).sum::<f32>();
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
                                if !v.is_nan() {
                                    *a += v;
                                }
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
                        *out = row.iter().filter(|v| !v.is_nan()).sum::<f32>();
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
                    let value = input_data[base];
                    if !value.is_nan() {
                        sum_val += value;
                    }
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

fn nansum_along_dim_f64(tensor: &Tensor, result_data: &mut TensorData, dim: usize) -> Result<()> {
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
        result_slice[0] = input_data.iter().filter(|v| !v.is_nan()).sum::<f64>();
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
                                if !v.is_nan() {
                                    *a += v;
                                }
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
                        *out = row.iter().filter(|v| !v.is_nan()).sum::<f64>();
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
                    let value = input_data[base];
                    if !value.is_nan() {
                        sum_val += value;
                    }
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

fn nanmax_all_f32(tensor: &Tensor, result_data: &mut TensorData) -> Result<()> {
    let data = tensor
        .data()
        .as_f32_slice()
        .ok_or_else(|| MinitensorError::internal_error("Failed to get f32 slice"))?;

    let (max_val, found) = data
        .par_iter()
        .map(|&v| {
            if v.is_nan() {
                (f32::NEG_INFINITY, false)
            } else {
                (v, true)
            }
        })
        .reduce(
            || (f32::NEG_INFINITY, false),
            |(a_val, a_found), (b_val, b_found)| match (a_found, b_found) {
                (true, true) => (a_val.max(b_val), true),
                (true, false) => (a_val, true),
                (false, true) => (b_val, true),
                (false, false) => (f32::NEG_INFINITY, false),
            },
        );

    let result_slice = result_data
        .as_f32_slice_mut()
        .ok_or_else(|| MinitensorError::internal_error("Failed to get mutable f32 slice"))?;

    result_slice[0] = if found { max_val } else { f32::NAN };
    Ok(())
}
