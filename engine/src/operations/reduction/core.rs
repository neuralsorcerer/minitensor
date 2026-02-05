// Copyright (c) 2026 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

use crate::{
    autograd::{
        CumprodBackward, CumsumBackward, NanMeanBackward, NanSumBackward, ProdBackward,
        SumBackward, add_to_graph,
    },
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

const NANQUANTILE_ALL_NAN_ERR: &str = "nanquantile() encountered an all-NaN slice";

/// Interpolation modes supported by the quantile reduction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QuantileInterpolation {
    Linear,
    Lower,
    Higher,
    Midpoint,
    Nearest,
}

impl QuantileInterpolation {
    #[inline(always)]
    fn interpolate(self, lower: f64, upper: f64, weight: f64) -> f64 {
        match self {
            QuantileInterpolation::Linear => lower + (upper - lower) * weight,
            QuantileInterpolation::Lower => lower,
            QuantileInterpolation::Higher => upper,
            QuantileInterpolation::Midpoint => 0.5 * (lower + upper),
            QuantileInterpolation::Nearest => {
                if weight <= 0.5 {
                    lower
                } else {
                    upper
                }
            }
        }
    }
}

fn normalize_reduction_dims(dims: Option<Vec<isize>>, ndim: usize) -> Result<Option<Vec<usize>>> {
    let ndim = ndim as isize;
    Ok(match dims {
        Some(dims) => {
            let mut normalized = Vec::with_capacity(dims.len());
            for d in dims {
                let d = if d < 0 { d + ndim } else { d };
                if d < 0 || d >= ndim {
                    return Err(MinitensorError::index_error(d, 0, ndim as usize));
                }
                normalized.push(d as usize);
            }
            normalized.sort_unstable();
            normalized.dedup();
            Some(normalized)
        }
        None => None,
    })
}

fn non_nan_mask(tensor: &Tensor) -> Result<Tensor> {
    let numel = tensor.numel();
    let mut mask = vec![false; numel];

    match tensor.dtype() {
        DataType::Float32 => {
            let data = tensor
                .data()
                .as_f32_slice()
                .ok_or_else(|| MinitensorError::internal_error("Failed to get f32 slice"))?;
            mask.par_iter_mut()
                .zip(data.par_iter())
                .for_each(|(out, &v)| {
                    *out = !v.is_nan();
                });
        }
        DataType::Float64 => {
            let data = tensor
                .data()
                .as_f64_slice()
                .ok_or_else(|| MinitensorError::internal_error("Failed to get f64 slice"))?;
            mask.par_iter_mut()
                .zip(data.par_iter())
                .for_each(|(out, &v)| {
                    *out = !v.is_nan();
                });
        }
        _ => {
            return Err(MinitensorError::invalid_operation(
                "nan reductions are only supported for floating point tensors",
            ));
        }
    }

    Ok(Tensor::new(
        Arc::new(TensorData::from_vec_bool(mask, tensor.device())),
        tensor.shape().clone(),
        DataType::Bool,
        tensor.device(),
        false,
    ))
}

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

/// Compute the q-th quantile of the tensor data.
pub fn quantile(
    tensor: &Tensor,
    q: f64,
    dim: Option<isize>,
    keepdim: bool,
    interpolation: QuantileInterpolation,
) -> Result<Tensor> {
    if tensor.numel() == 0 {
        return Err(MinitensorError::invalid_argument(
            "quantile() does not support empty tensors".to_string(),
        ));
    }

    validate_quantile_value(q)?;
    ensure_floating_point_dtype(tensor.dtype())?;

    match dim {
        None => quantile_all(tensor, q, keepdim, interpolation),
        Some(dim_value) => {
            if tensor.ndim() == 0 {
                if dim_value == 0 || dim_value == -1 {
                    return quantile_all(tensor, q, keepdim, interpolation);
                }
                return Err(MinitensorError::index_error(dim_value, 0, 1));
            }

            let axis = normalize_dim(dim_value, tensor.ndim())?;
            quantile_along_dim(tensor, axis, keepdim, q, interpolation)
        }
    }
}

/// Compute multiple quantiles of the tensor data in a single pass.
pub fn quantiles(
    tensor: &Tensor,
    qs: &[f64],
    dim: Option<isize>,
    keepdim: bool,
    interpolation: QuantileInterpolation,
) -> Result<Tensor> {
    if tensor.numel() == 0 {
        return Err(MinitensorError::invalid_argument(
            "quantile() does not support empty tensors".to_string(),
        ));
    }

    if qs.is_empty() {
        return Err(MinitensorError::invalid_argument(
            "quantile() expected at least one probability value".to_string(),
        ));
    }

    for &q in qs {
        validate_quantile_value(q)?;
    }

    ensure_floating_point_dtype(tensor.dtype())?;

    match dim {
        None => quantiles_all(tensor, qs, keepdim, interpolation),
        Some(dim_value) => {
            if tensor.ndim() == 0 {
                if dim_value == 0 || dim_value == -1 {
                    return quantiles_all(tensor, qs, keepdim, interpolation);
                }
                return Err(MinitensorError::index_error(dim_value, 0, 1));
            }

            let axis = normalize_dim(dim_value, tensor.ndim())?;
            quantiles_along_dim(tensor, axis, qs, keepdim, interpolation)
        }
    }
}

/// Compute the q-th quantile of the tensor data while ignoring NaN values.
pub fn nanquantile(
    tensor: &Tensor,
    q: f64,
    dim: Option<isize>,
    keepdim: bool,
    interpolation: QuantileInterpolation,
) -> Result<Tensor> {
    if tensor.numel() == 0 {
        return Err(MinitensorError::invalid_argument(
            "nanquantile() does not support empty tensors".to_string(),
        ));
    }

    validate_quantile_value(q)?;
    ensure_floating_point_dtype(tensor.dtype())?;

    match dim {
        None => nanquantile_all(tensor, q, keepdim, interpolation),
        Some(dim_value) => {
            if tensor.ndim() == 0 {
                if dim_value == 0 || dim_value == -1 {
                    return nanquantile_all(tensor, q, keepdim, interpolation);
                }
                return Err(MinitensorError::index_error(dim_value, 0, 1));
            }

            let axis = normalize_dim(dim_value, tensor.ndim())?;
            nanquantile_along_dim(tensor, axis, keepdim, q, interpolation)
        }
    }
}

/// Compute multiple quantiles of the tensor data in a single pass while ignoring NaN values.
pub fn nanquantiles(
    tensor: &Tensor,
    qs: &[f64],
    dim: Option<isize>,
    keepdim: bool,
    interpolation: QuantileInterpolation,
) -> Result<Tensor> {
    if tensor.numel() == 0 {
        return Err(MinitensorError::invalid_argument(
            "nanquantile() does not support empty tensors".to_string(),
        ));
    }

    if qs.is_empty() {
        return Err(MinitensorError::invalid_argument(
            "nanquantile() expected at least one probability value".to_string(),
        ));
    }

    for &q in qs {
        validate_quantile_value(q)?;
    }

    ensure_floating_point_dtype(tensor.dtype())?;

    match dim {
        None => nanquantiles_all(tensor, qs, keepdim, interpolation),
        Some(dim_value) => {
            if tensor.ndim() == 0 {
                if dim_value == 0 || dim_value == -1 {
                    return nanquantiles_all(tensor, qs, keepdim, interpolation);
                }
                return Err(MinitensorError::index_error(dim_value, 0, 1));
            }

            let axis = normalize_dim(dim_value, tensor.ndim())?;
            nanquantiles_along_dim(tensor, axis, qs, keepdim, interpolation)
        }
    }
}

fn validate_quantile_value(q: f64) -> Result<()> {
    if !q.is_finite() {
        return Err(MinitensorError::invalid_argument(
            "quantile() requires a finite probability in [0, 1]".to_string(),
        ));
    }
    if !(0.0..=1.0).contains(&q) {
        return Err(MinitensorError::invalid_argument(format!(
            "quantile() expected q in [0, 1], got {q}",
        )));
    }
    Ok(())
}

fn ensure_floating_point_dtype(dtype: DataType) -> Result<()> {
    match dtype {
        DataType::Float32 | DataType::Float64 => Ok(()),
        _ => Err(MinitensorError::invalid_operation(
            "quantile() currently supports only floating point tensors".to_string(),
        )),
    }
}

fn fill_quantile_single_f32(
    input: &[f32],
    values: &mut [f32],
    outer: usize,
    inner: usize,
    outer_stride: usize,
) {
    for o in 0..outer {
        for r in 0..inner {
            let idx = o * outer_stride + r;
            let value = input[idx];
            values[o * inner + r] = if value.is_nan() { f32::NAN } else { value };
        }
    }
}

fn fill_quantile_single_f64(
    input: &[f64],
    values: &mut [f64],
    outer: usize,
    inner: usize,
    outer_stride: usize,
) {
    for o in 0..outer {
        for r in 0..inner {
            let idx = o * outer_stride + r;
            let value = input[idx];
            values[o * inner + r] = if value.is_nan() { f64::NAN } else { value };
        }
    }
}

fn fill_quantiles_single_f32(
    input: &[f32],
    values: &mut [f32],
    outer: usize,
    inner: usize,
    outer_stride: usize,
    q_len: usize,
) {
    for o in 0..outer {
        for r in 0..inner {
            let idx = o * outer_stride + r;
            let value = input[idx];
            for qi in 0..q_len {
                let out_idx = ((qi * outer) + o) * inner + r;
                values[out_idx] = if value.is_nan() { f32::NAN } else { value };
            }
        }
    }
}

fn fill_quantiles_single_f64(
    input: &[f64],
    values: &mut [f64],
    outer: usize,
    inner: usize,
    outer_stride: usize,
    q_len: usize,
) {
    for o in 0..outer {
        for r in 0..inner {
            let idx = o * outer_stride + r;
            let value = input[idx];
            for qi in 0..q_len {
                let out_idx = ((qi * outer) + o) * inner + r;
                values[out_idx] = if value.is_nan() { f64::NAN } else { value };
            }
        }
    }
}

fn fill_nanquantile_single_f32(
    input: &[f32],
    values: &mut [f32],
    outer: usize,
    inner: usize,
    outer_stride: usize,
) -> Result<()> {
    for o in 0..outer {
        for r in 0..inner {
            let idx = o * outer_stride + r;
            let value = input[idx];
            if value.is_nan() {
                return Err(MinitensorError::invalid_argument(
                    NANQUANTILE_ALL_NAN_ERR.to_string(),
                ));
            }
            values[o * inner + r] = value;
        }
    }
    Ok(())
}

fn fill_nanquantile_single_f64(
    input: &[f64],
    values: &mut [f64],
    outer: usize,
    inner: usize,
    outer_stride: usize,
) -> Result<()> {
    for o in 0..outer {
        for r in 0..inner {
            let idx = o * outer_stride + r;
            let value = input[idx];
            if value.is_nan() {
                return Err(MinitensorError::invalid_argument(
                    NANQUANTILE_ALL_NAN_ERR.to_string(),
                ));
            }
            values[o * inner + r] = value;
        }
    }
    Ok(())
}

fn fill_nanquantiles_single_f32(
    input: &[f32],
    values: &mut [f32],
    outer: usize,
    inner: usize,
    outer_stride: usize,
    q_len: usize,
) -> Result<()> {
    for o in 0..outer {
        for r in 0..inner {
            let idx = o * outer_stride + r;
            let value = input[idx];
            if value.is_nan() {
                return Err(MinitensorError::invalid_argument(
                    NANQUANTILE_ALL_NAN_ERR.to_string(),
                ));
            }
            for qi in 0..q_len {
                let out_idx = ((qi * outer) + o) * inner + r;
                values[out_idx] = value;
            }
        }
    }
    Ok(())
}

fn fill_nanquantiles_single_f64(
    input: &[f64],
    values: &mut [f64],
    outer: usize,
    inner: usize,
    outer_stride: usize,
    q_len: usize,
) -> Result<()> {
    for o in 0..outer {
        for r in 0..inner {
            let idx = o * outer_stride + r;
            let value = input[idx];
            if value.is_nan() {
                return Err(MinitensorError::invalid_argument(
                    NANQUANTILE_ALL_NAN_ERR.to_string(),
                ));
            }
            for qi in 0..q_len {
                let out_idx = ((qi * outer) + o) * inner + r;
                values[out_idx] = value;
            }
        }
    }
    Ok(())
}

fn fill_quantiles_all_single_f32(value: f32, values: &mut [f32]) {
    if value.is_nan() {
        values.fill(f32::NAN);
    } else {
        values.fill(value);
    }
}

fn fill_quantiles_all_single_f64(value: f64, values: &mut [f64]) {
    if value.is_nan() {
        values.fill(f64::NAN);
    } else {
        values.fill(value);
    }
}

fn fill_nanquantiles_all_single_f32(value: f32, values: &mut [f32]) -> Result<()> {
    if value.is_nan() {
        return Err(MinitensorError::invalid_argument(
            NANQUANTILE_ALL_NAN_ERR.to_string(),
        ));
    }
    values.fill(value);
    Ok(())
}

fn fill_nanquantiles_all_single_f64(value: f64, values: &mut [f64]) -> Result<()> {
    if value.is_nan() {
        return Err(MinitensorError::invalid_argument(
            NANQUANTILE_ALL_NAN_ERR.to_string(),
        ));
    }
    values.fill(value);
    Ok(())
}

#[derive(Clone, Copy)]
struct QuantilePosition {
    lower_idx: usize,
    upper_idx: usize,
    weight: f64,
}

fn quantile_positions_for_len(len: usize, qs: &[f64]) -> Vec<QuantilePosition> {
    if len == 1 {
        return vec![
            QuantilePosition {
                lower_idx: 0,
                upper_idx: 0,
                weight: 0.0,
            };
            qs.len()
        ];
    }

    let max_index = (len - 1) as f64;
    qs.iter()
        .map(|&q| {
            let pos = (q * max_index).clamp(0.0, max_index);
            let lower_idx = pos.floor() as usize;
            let upper_idx = pos.ceil() as usize;
            let weight = (pos - lower_idx as f64).clamp(0.0, 1.0);
            QuantilePosition {
                lower_idx,
                upper_idx,
                weight,
            }
        })
        .collect()
}

fn quantiles_from_sorted_f32(
    values: &[f32],
    positions: &[QuantilePosition],
    interpolation: QuantileInterpolation,
    output: &mut [f32],
) {
    for (slot, position) in output.iter_mut().zip(positions.iter()) {
        let lower = values[position.lower_idx] as f64;
        let upper = values[position.upper_idx] as f64;
        *slot = interpolation.interpolate(lower, upper, position.weight) as f32;
    }
}

fn quantiles_from_sorted_f64(
    values: &[f64],
    positions: &[QuantilePosition],
    interpolation: QuantileInterpolation,
    output: &mut [f64],
) {
    for (slot, position) in output.iter_mut().zip(positions.iter()) {
        let lower = values[position.lower_idx];
        let upper = values[position.upper_idx];
        *slot = interpolation.interpolate(lower, upper, position.weight);
    }
}

fn quantile_all(
    tensor: &Tensor,
    q: f64,
    keepdim: bool,
    interpolation: QuantileInterpolation,
) -> Result<Tensor> {
    if tensor.ndim() == 0 {
        return Ok(tensor.clone());
    }

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
                    let result_shape = if keepdim {
                        Shape::new(vec![1; tensor.ndim()])
                    } else {
                        Shape::scalar()
                    };
                    return Ok(Tensor::new(
                        Arc::new(result_data),
                        result_shape,
                        tensor.dtype(),
                        tensor.device(),
                        tensor.requires_grad(),
                    ));
                }
                values.push(value);
            }
            let quant = quantile_from_unsorted_f32(&mut values, q, interpolation);
            result_data.as_f32_slice_mut().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get mutable f32 slice")
            })?[0] = quant;
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
                    let result_shape = if keepdim {
                        Shape::new(vec![1; tensor.ndim()])
                    } else {
                        Shape::scalar()
                    };
                    return Ok(Tensor::new(
                        Arc::new(result_data),
                        result_shape,
                        tensor.dtype(),
                        tensor.device(),
                        tensor.requires_grad(),
                    ));
                }
                values.push(value);
            }
            let quant = quantile_from_unsorted_f64(&mut values, q, interpolation);
            result_data.as_f64_slice_mut().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get mutable f64 slice")
            })?[0] = quant;
        }
        _ => unreachable!("dtype validated"),
    }

    let result_shape = if keepdim {
        Shape::new(vec![1; tensor.ndim()])
    } else {
        Shape::scalar()
    };

    Ok(Tensor::new(
        Arc::new(result_data),
        result_shape,
        tensor.dtype(),
        tensor.device(),
        tensor.requires_grad(),
    ))
}

fn nanquantile_all(
    tensor: &Tensor,
    q: f64,
    keepdim: bool,
    interpolation: QuantileInterpolation,
) -> Result<Tensor> {
    if tensor.ndim() == 0 {
        match tensor.dtype() {
            DataType::Float32 => {
                let value = tensor
                    .data()
                    .as_f32_slice()
                    .ok_or_else(|| MinitensorError::internal_error("Failed to get f32 slice"))?[0];
                if value.is_nan() {
                    return Err(MinitensorError::invalid_argument(
                        NANQUANTILE_ALL_NAN_ERR.to_string(),
                    ));
                }
            }
            DataType::Float64 => {
                let value = tensor
                    .data()
                    .as_f64_slice()
                    .ok_or_else(|| MinitensorError::internal_error("Failed to get f64 slice"))?[0];
                if value.is_nan() {
                    return Err(MinitensorError::invalid_argument(
                        NANQUANTILE_ALL_NAN_ERR.to_string(),
                    ));
                }
            }
            _ => unreachable!("dtype validated"),
        }
        return Ok(tensor.clone());
    }

    let mut result_data = TensorData::zeros_on_device(1, tensor.dtype(), tensor.device());

    match tensor.dtype() {
        DataType::Float32 => {
            let data = tensor
                .data()
                .as_f32_slice()
                .ok_or_else(|| MinitensorError::internal_error("Failed to get f32 slice"))?;
            let mut values: Vec<f32> = data.iter().copied().filter(|v| !v.is_nan()).collect();
            if values.is_empty() {
                return Err(MinitensorError::invalid_argument(
                    NANQUANTILE_ALL_NAN_ERR.to_string(),
                ));
            }
            let quant = quantile_from_unsorted_f32(&mut values, q, interpolation);
            result_data.as_f32_slice_mut().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get mutable f32 slice")
            })?[0] = quant;
        }
        DataType::Float64 => {
            let data = tensor
                .data()
                .as_f64_slice()
                .ok_or_else(|| MinitensorError::internal_error("Failed to get f64 slice"))?;
            let mut values: Vec<f64> = data.iter().copied().filter(|v| !v.is_nan()).collect();
            if values.is_empty() {
                return Err(MinitensorError::invalid_argument(
                    NANQUANTILE_ALL_NAN_ERR.to_string(),
                ));
            }
            let quant = quantile_from_unsorted_f64(&mut values, q, interpolation);
            result_data.as_f64_slice_mut().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get mutable f64 slice")
            })?[0] = quant;
        }
        _ => unreachable!("dtype validated"),
    }

    let result_shape = if keepdim {
        Shape::new(vec![1; tensor.ndim()])
    } else {
        Shape::scalar()
    };

    Ok(Tensor::new(
        Arc::new(result_data),
        result_shape,
        tensor.dtype(),
        tensor.device(),
        tensor.requires_grad(),
    ))
}
