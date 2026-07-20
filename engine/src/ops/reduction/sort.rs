// Copyright (c) Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

use super::*;
use crate::ops::shape_ops;
use crate::ops::simd::*;
use crate::{
    error::{MinitensorError, Result},
    tensor::{DataType, Shape, Tensor, TensorData},
};
use rayon::prelude::*;
use std::sync::Arc;

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

    // `values = gather(input, axis, indices)`; scatter the gradient back.
    let values = attach_gather_like_grad(values, tensor, axis, &indices)?;

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

/// Standard deviation along specified dimensions
pub fn std(
    tensor: &Tensor,
    dim: Option<Vec<isize>>,
    keepdim: bool,
    unbiased: bool,
) -> Result<Tensor> {
    let variance = var(tensor, dim, keepdim, unbiased)?;
    crate::ops::activation::sqrt(&variance)
}

/// Variance along specified dimensions
pub fn var(
    tensor: &Tensor,
    dim: Option<Vec<isize>>,
    keepdim: bool,
    unbiased: bool,
) -> Result<Tensor> {
    if !tensor.dtype().is_float() {
        return Err(MinitensorError::invalid_operation(
            "Variance only supported for floating point tensors",
        ));
    }

    let dims = match dim {
        Some(dims) => {
            let ndim = tensor.ndim() as isize;
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

    if matches!(dims, Some(ref dims) if dims.is_empty()) {
        return Ok(tensor.clone());
    }

    let reduction_dims: Vec<usize> = dims.clone().unwrap_or_else(|| (0..tensor.ndim()).collect());
    let reduction_dims_isize: Vec<isize> = reduction_dims.iter().map(|&d| d as isize).collect();

    // Keep reduced axes while computing deviations so broadcasting is unambiguous for
    // both single-axis and multi-axis reductions.
    let mean_tensor = mean(tensor, Some(reduction_dims_isize.clone()), true)?;
    let diff = crate::ops::arithmetic::sub(tensor, &mean_tensor)?;
    let squared_diff = crate::ops::arithmetic::mul(&diff, &diff)?;
    let mut variance = mean(&squared_diff, Some(reduction_dims_isize), true)?;

    let sample_count = reduction_dims
        .iter()
        .map(|&axis| tensor.shape().dims()[axis])
        .product::<usize>();

    if unbiased {
        if sample_count <= 1 {
            let nan_count = variance.numel();
            let nan_data = match variance.dtype() {
                DataType::Float32 => {
                    TensorData::from_vec_f32(vec![f32::NAN; nan_count], variance.device())
                }
                DataType::Float64 => {
                    TensorData::from_vec_f64(vec![f64::NAN; nan_count], variance.device())
                }
                _ => unreachable!("variance is only defined for floating point tensors"),
            };
            variance = Tensor::new(
                Arc::new(nan_data),
                variance.shape().clone(),
                variance.dtype(),
                variance.device(),
                variance.requires_grad(),
            );
        } else {
            let correction = sample_count as f64 / (sample_count - 1) as f64;
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
            variance = crate::ops::arithmetic::mul(&variance, &correction_tensor)?;
        }
    }

    if keepdim {
        return Ok(variance);
    }

    let mut new_dims = Vec::with_capacity(variance.ndim().saturating_sub(reduction_dims.len()));
    for (idx, &size) in variance.shape().dims().iter().enumerate() {
        if reduction_dims.binary_search(&idx).is_err() {
            new_dims.push(size);
        }
    }
    let target_shape = if new_dims.is_empty() {
        Shape::scalar()
    } else {
        Shape::new(new_dims)
    };
    shape_ops::reshape(&variance, target_shape)
}

// Helper functions for type-specific operations

pub(crate) fn prod_all_f32(tensor: &Tensor, result_data: &mut TensorData) -> Result<()> {
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

pub(crate) fn prod_all_f64(tensor: &Tensor, result_data: &mut TensorData) -> Result<()> {
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

pub(crate) fn prod_all_i32(tensor: &Tensor, result_data: &mut TensorData) -> Result<()> {
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

pub(crate) fn prod_all_i64(tensor: &Tensor, result_data: &mut TensorData) -> Result<()> {
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

pub(crate) fn prod_all_bool(tensor: &Tensor, result_data: &mut TensorData) -> Result<()> {
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

pub(crate) fn sum_all_f32(tensor: &Tensor, result_data: &mut TensorData) -> Result<()> {
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

pub(crate) fn sum_all_f64(tensor: &Tensor, result_data: &mut TensorData) -> Result<()> {
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

pub(crate) fn sum_all_i32(tensor: &Tensor, result_data: &mut TensorData) -> Result<()> {
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

pub(crate) fn sum_all_i64(tensor: &Tensor, result_data: &mut TensorData) -> Result<()> {
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

pub(crate) fn nansum_all_f32(tensor: &Tensor, result_data: &mut TensorData) -> Result<()> {
    let data = tensor
        .data()
        .as_f32_slice()
        .ok_or_else(|| MinitensorError::internal_error("Failed to get f32 slice"))?;

    let sum: f32 = data
        .par_iter()
        .map(|&v| if v.is_nan() { 0.0 } else { v })
        .sum();

    let result_slice = result_data
        .as_f32_slice_mut()
        .ok_or_else(|| MinitensorError::internal_error("Failed to get mutable f32 slice"))?;
    result_slice[0] = sum;
    Ok(())
}

pub(crate) fn nansum_all_f64(tensor: &Tensor, result_data: &mut TensorData) -> Result<()> {
    let data = tensor
        .data()
        .as_f64_slice()
        .ok_or_else(|| MinitensorError::internal_error("Failed to get f64 slice"))?;

    let sum: f64 = data
        .par_iter()
        .map(|&v| if v.is_nan() { 0.0 } else { v })
        .sum();

    let result_slice = result_data
        .as_f64_slice_mut()
        .ok_or_else(|| MinitensorError::internal_error("Failed to get mutable f64 slice"))?;
    result_slice[0] = sum;
    Ok(())
}

pub(crate) fn nanmean_all_f32(
    tensor: &Tensor,
    sum_data: &mut TensorData,
    count_data: &mut TensorData,
) -> Result<()> {
    let data = tensor
        .data()
        .as_f32_slice()
        .ok_or_else(|| MinitensorError::internal_error("Failed to get f32 slice"))?;

    let (sum, count) = data
        .par_iter()
        .map(|&v| {
            if v.is_nan() {
                (0.0, 0usize)
            } else {
                (v, 1usize)
            }
        })
        .reduce(|| (0.0, 0usize), |(s1, c1), (s2, c2)| (s1 + s2, c1 + c2));

    let sum_slice = sum_data
        .as_f32_slice_mut()
        .ok_or_else(|| MinitensorError::internal_error("Failed to get mutable f32 slice"))?;
    let count_slice = count_data
        .as_f32_slice_mut()
        .ok_or_else(|| MinitensorError::internal_error("Failed to get mutable f32 slice"))?;

    sum_slice[0] = sum;
    count_slice[0] = count as f32;
    Ok(())
}

pub(crate) fn nanmean_all_f64(
    tensor: &Tensor,
    sum_data: &mut TensorData,
    count_data: &mut TensorData,
) -> Result<()> {
    let data = tensor
        .data()
        .as_f64_slice()
        .ok_or_else(|| MinitensorError::internal_error("Failed to get f64 slice"))?;

    let (sum, count) = data
        .par_iter()
        .map(|&v| {
            if v.is_nan() {
                (0.0, 0usize)
            } else {
                (v, 1usize)
            }
        })
        .reduce(|| (0.0, 0usize), |(s1, c1), (s2, c2)| (s1 + s2, c1 + c2));

    let sum_slice = sum_data
        .as_f64_slice_mut()
        .ok_or_else(|| MinitensorError::internal_error("Failed to get mutable f64 slice"))?;
    let count_slice = count_data
        .as_f64_slice_mut()
        .ok_or_else(|| MinitensorError::internal_error("Failed to get mutable f64 slice"))?;

    sum_slice[0] = sum;
    count_slice[0] = count as f64;
    Ok(())
}

pub(crate) fn nanmean_from_sum_count(
    sum: &Tensor,
    count: &Tensor,
    requires_grad: bool,
) -> Result<Tensor> {
    if sum.dtype() != count.dtype() || sum.shape() != count.shape() {
        return Err(MinitensorError::invalid_operation(
            "nanmean requires sum and count tensors with matching dtype and shape",
        ));
    }

    let numel = sum.numel();
    let mut result_data = TensorData::zeros_on_device(numel, sum.dtype(), sum.device());

    match sum.dtype() {
        DataType::Float32 => {
            let sum_slice = sum
                .data()
                .as_f32_slice()
                .ok_or_else(|| MinitensorError::internal_error("Failed to get f32 slice"))?;
            let count_slice = count
                .data()
                .as_f32_slice()
                .ok_or_else(|| MinitensorError::internal_error("Failed to get f32 slice"))?;
            let out = result_data
                .as_f32_slice_mut()
                .ok_or_else(|| MinitensorError::internal_error("Failed to get f32 slice"))?;
            out.par_iter_mut()
                .zip(sum_slice.par_iter().zip(count_slice.par_iter()))
                .for_each(|(dst, (&s, &c))| {
                    *dst = if c == 0.0 { f32::NAN } else { s / c };
                });
        }
        DataType::Float64 => {
            let sum_slice = sum
                .data()
                .as_f64_slice()
                .ok_or_else(|| MinitensorError::internal_error("Failed to get f64 slice"))?;
            let count_slice = count
                .data()
                .as_f64_slice()
                .ok_or_else(|| MinitensorError::internal_error("Failed to get f64 slice"))?;
            let out = result_data
                .as_f64_slice_mut()
                .ok_or_else(|| MinitensorError::internal_error("Failed to get f64 slice"))?;
            out.par_iter_mut()
                .zip(sum_slice.par_iter().zip(count_slice.par_iter()))
                .for_each(|(dst, (&s, &c))| {
                    *dst = if c == 0.0 { f64::NAN } else { s / c };
                });
        }
        _ => {
            return Err(MinitensorError::invalid_operation(
                "nanmean only supports floating point tensors",
            ));
        }
    }

    Ok(Tensor::new(
        Arc::new(result_data),
        sum.shape().clone(),
        sum.dtype(),
        sum.device(),
        requires_grad,
    ))
}

#[inline]
pub fn nansum_along_dim(tensor: &Tensor, dim: usize, keepdim: bool) -> Result<Tensor> {
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
        DataType::Float32 => nansum_along_dim_f32(tensor, &mut result_data, dim)?,
        DataType::Float64 => nansum_along_dim_f64(tensor, &mut result_data, dim)?,
        _ => {
            return Err(MinitensorError::invalid_operation(
                "nansum only supports floating point tensors",
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
