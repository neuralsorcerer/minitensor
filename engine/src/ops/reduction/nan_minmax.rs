// Copyright (c) Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

use crate::{
    error::{MinitensorError, Result},
    tensor::{DataType, Shape, Tensor, TensorData},
};
use rayon::prelude::*;
use std::sync::Arc;

pub(crate) fn nanmax_all_f64(tensor: &Tensor, result_data: &mut TensorData) -> Result<()> {
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

pub(crate) fn nanmin_all_f32(tensor: &Tensor, result_data: &mut TensorData) -> Result<()> {
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

pub(crate) fn nanmin_all_f64(tensor: &Tensor, result_data: &mut TensorData) -> Result<()> {
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
pub(crate) fn argmax_all_f32(tensor: &Tensor, result_data: &mut TensorData) -> Result<()> {
    let data = tensor
        .data()
        .as_f32_slice()
        .ok_or_else(|| MinitensorError::internal_error("Failed to get f32 slice"))?;

    let (argmax_idx, _) = data.par_iter().enumerate().map(|(i, &v)| (i, v)).reduce(
        || (0, f32::NEG_INFINITY),
        |(i1, v1), (i2, v2)| match (v1.is_nan(), v2.is_nan()) {
            (true, true) => {
                if i1 <= i2 {
                    (i1, v1)
                } else {
                    (i2, v2)
                }
            }
            (true, false) => (i1, v1),
            (false, true) => (i2, v2),
            (false, false) => {
                if v1 > v2 {
                    (i1, v1)
                } else if v2 > v1 {
                    (i2, v2)
                } else if i1 <= i2 {
                    (i1, v1)
                } else {
                    (i2, v2)
                }
            }
        },
    );

    let result_slice = result_data
        .as_i64_slice_mut()
        .ok_or_else(|| MinitensorError::internal_error("Failed to get mutable i64 slice"))?;

    result_slice[0] = argmax_idx as i64;
    Ok(())
}

pub(crate) fn argmax_all_f64(tensor: &Tensor, result_data: &mut TensorData) -> Result<()> {
    let data = tensor
        .data()
        .as_f64_slice()
        .ok_or_else(|| MinitensorError::internal_error("Failed to get f64 slice"))?;

    let (argmax_idx, _) = data.par_iter().enumerate().map(|(i, &v)| (i, v)).reduce(
        || (0, f64::NEG_INFINITY),
        |(i1, v1), (i2, v2)| match (v1.is_nan(), v2.is_nan()) {
            (true, true) => {
                if i1 <= i2 {
                    (i1, v1)
                } else {
                    (i2, v2)
                }
            }
            (true, false) => (i1, v1),
            (false, true) => (i2, v2),
            (false, false) => {
                if v1 > v2 {
                    (i1, v1)
                } else if v2 > v1 {
                    (i2, v2)
                } else if i1 <= i2 {
                    (i1, v1)
                } else {
                    (i2, v2)
                }
            }
        },
    );

    let result_slice = result_data
        .as_i64_slice_mut()
        .ok_or_else(|| MinitensorError::internal_error("Failed to get mutable i64 slice"))?;

    result_slice[0] = argmax_idx as i64;
    Ok(())
}

pub(crate) fn argmax_all_i32(tensor: &Tensor, result_data: &mut TensorData) -> Result<()> {
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

pub(crate) fn argmax_all_i64(tensor: &Tensor, result_data: &mut TensorData) -> Result<()> {
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

pub(crate) fn argmax_all_bool(tensor: &Tensor, result_data: &mut TensorData) -> Result<()> {
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
pub(crate) fn argmin_all_f32(tensor: &Tensor, result_data: &mut TensorData) -> Result<()> {
    let data = tensor
        .data()
        .as_f32_slice()
        .ok_or_else(|| MinitensorError::internal_error("Failed to get f32 slice"))?;

    let (argmin_idx, _) = data.par_iter().enumerate().map(|(i, &v)| (i, v)).reduce(
        || (0, f32::INFINITY),
        |(i1, v1), (i2, v2)| match (v1.is_nan(), v2.is_nan()) {
            (true, true) => {
                if i1 <= i2 {
                    (i1, v1)
                } else {
                    (i2, v2)
                }
            }
            (true, false) => (i1, v1),
            (false, true) => (i2, v2),
            (false, false) => {
                if v1 < v2 {
                    (i1, v1)
                } else if v2 < v1 {
                    (i2, v2)
                } else if i1 <= i2 {
                    (i1, v1)
                } else {
                    (i2, v2)
                }
            }
        },
    );

    let result_slice = result_data
        .as_i64_slice_mut()
        .ok_or_else(|| MinitensorError::internal_error("Failed to get mutable i64 slice"))?;

    result_slice[0] = argmin_idx as i64;
    Ok(())
}

pub(crate) fn argmin_all_f64(tensor: &Tensor, result_data: &mut TensorData) -> Result<()> {
    let data = tensor
        .data()
        .as_f64_slice()
        .ok_or_else(|| MinitensorError::internal_error("Failed to get f64 slice"))?;

    let (argmin_idx, _) = data.par_iter().enumerate().map(|(i, &v)| (i, v)).reduce(
        || (0, f64::INFINITY),
        |(i1, v1), (i2, v2)| match (v1.is_nan(), v2.is_nan()) {
            (true, true) => {
                if i1 <= i2 {
                    (i1, v1)
                } else {
                    (i2, v2)
                }
            }
            (true, false) => (i1, v1),
            (false, true) => (i2, v2),
            (false, false) => {
                if v1 < v2 {
                    (i1, v1)
                } else if v2 < v1 {
                    (i2, v2)
                } else if i1 <= i2 {
                    (i1, v1)
                } else {
                    (i2, v2)
                }
            }
        },
    );

    let result_slice = result_data
        .as_i64_slice_mut()
        .ok_or_else(|| MinitensorError::internal_error("Failed to get mutable i64 slice"))?;

    result_slice[0] = argmin_idx as i64;
    Ok(())
}

pub(crate) fn argmin_all_i32(tensor: &Tensor, result_data: &mut TensorData) -> Result<()> {
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

pub(crate) fn argmin_all_i64(tensor: &Tensor, result_data: &mut TensorData) -> Result<()> {
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

pub(crate) fn argmin_all_bool(tensor: &Tensor, result_data: &mut TensorData) -> Result<()> {
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

pub(crate) struct DimReductionLayout {
    pub(crate) output_shape: Shape,
    pub(crate) dim_size: usize,
    pub(crate) outer: usize,
    pub(crate) inner: usize,
    pub(crate) outer_stride: usize,
}

pub(crate) fn reduction_layout(
    tensor: &Tensor,
    dim: usize,
    keepdim: bool,
) -> Result<DimReductionLayout> {
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
/// Reduce `input` along a dimension into `output`, parallelizing over output
/// elements (one rayon task per output position, each walking its column of the
/// reduced dimension with a running offset). `combine` folds the accumulator
/// with each element; `short_circuit` returning `Some(v)` stops the column early
/// with `v` (used to propagate NaN and to break out of boolean any/all).
#[inline]
fn reduce_along_dim_par<T, C, S>(
    input: &[T],
    output: &mut [T],
    layout: &DimReductionLayout,
    init: T,
    combine: C,
    short_circuit: S,
) where
    T: Copy + Send + Sync,
    C: Fn(T, T) -> T + Sync,
    S: Fn(T) -> Option<T> + Sync,
{
    let inner = layout.inner;
    let dim_size = layout.dim_size;
    let outer_stride = layout.outer_stride;
    output
        .par_iter_mut()
        .enumerate()
        .for_each(|(out_idx, out)| {
            let o = out_idx / inner;
            let r = out_idx % inner;
            let mut acc = init;
            let mut idx = o * outer_stride + r;
            for _ in 0..dim_size {
                let val = input[idx];
                if let Some(sc) = short_circuit(val) {
                    acc = sc;
                    break;
                }
                acc = combine(acc, val);
                idx += inner;
            }
            *out = acc;
        });
}

pub(crate) fn max_along_dim(tensor: &Tensor, dim: usize, keepdim: bool) -> Result<Tensor> {
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

            reduce_along_dim_par(
                input,
                output,
                &layout,
                f32::NEG_INFINITY,
                |a, v| a.max(v),
                |v| if v.is_nan() { Some(f32::NAN) } else { None },
            );
        }
        DataType::Float64 => {
            let input = tensor
                .data()
                .as_f64_slice()
                .ok_or_else(|| MinitensorError::internal_error("Failed to get f64 slice"))?;
            let output = result_data.as_f64_slice_mut().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get mutable f64 slice")
            })?;

            reduce_along_dim_par(
                input,
                output,
                &layout,
                f64::NEG_INFINITY,
                |a, v| a.max(v),
                |v| if v.is_nan() { Some(f64::NAN) } else { None },
            );
        }
        DataType::Int32 => {
            let input = tensor
                .data()
                .as_i32_slice()
                .ok_or_else(|| MinitensorError::internal_error("Failed to get i32 slice"))?;
            let output = result_data.as_i32_slice_mut().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get mutable i32 slice")
            })?;

            reduce_along_dim_par(input, output, &layout, i32::MIN, |a, v| a.max(v), |_| None);
        }
        DataType::Int64 => {
            let input = tensor
                .data()
                .as_i64_slice()
                .ok_or_else(|| MinitensorError::internal_error("Failed to get i64 slice"))?;
            let output = result_data.as_i64_slice_mut().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get mutable i64 slice")
            })?;

            reduce_along_dim_par(input, output, &layout, i64::MIN, |a, v| a.max(v), |_| None);
        }
        DataType::Bool => {
            let input = tensor
                .data()
                .as_bool_slice()
                .ok_or_else(|| MinitensorError::internal_error("Failed to get bool slice"))?;
            let output = result_data.as_bool_slice_mut().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get mutable bool slice")
            })?;

            reduce_along_dim_par(
                input,
                output,
                &layout,
                false,
                |a, v| a | v,
                |v| if v { Some(true) } else { None },
            );
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

pub(crate) fn min_along_dim(tensor: &Tensor, dim: usize, keepdim: bool) -> Result<Tensor> {
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

            reduce_along_dim_par(
                input,
                output,
                &layout,
                f32::INFINITY,
                |a, v| a.min(v),
                |v| if v.is_nan() { Some(f32::NAN) } else { None },
            );
        }
        DataType::Float64 => {
            let input = tensor
                .data()
                .as_f64_slice()
                .ok_or_else(|| MinitensorError::internal_error("Failed to get f64 slice"))?;
            let output = result_data.as_f64_slice_mut().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get mutable f64 slice")
            })?;

            reduce_along_dim_par(
                input,
                output,
                &layout,
                f64::INFINITY,
                |a, v| a.min(v),
                |v| if v.is_nan() { Some(f64::NAN) } else { None },
            );
        }
        DataType::Int32 => {
            let input = tensor
                .data()
                .as_i32_slice()
                .ok_or_else(|| MinitensorError::internal_error("Failed to get i32 slice"))?;
            let output = result_data.as_i32_slice_mut().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get mutable i32 slice")
            })?;

            reduce_along_dim_par(input, output, &layout, i32::MAX, |a, v| a.min(v), |_| None);
        }
        DataType::Int64 => {
            let input = tensor
                .data()
                .as_i64_slice()
                .ok_or_else(|| MinitensorError::internal_error("Failed to get i64 slice"))?;
            let output = result_data.as_i64_slice_mut().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get mutable i64 slice")
            })?;

            reduce_along_dim_par(input, output, &layout, i64::MAX, |a, v| a.min(v), |_| None);
        }
        DataType::Bool => {
            let input = tensor
                .data()
                .as_bool_slice()
                .ok_or_else(|| MinitensorError::internal_error("Failed to get bool slice"))?;
            let output = result_data.as_bool_slice_mut().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get mutable bool slice")
            })?;

            reduce_along_dim_par(
                input,
                output,
                &layout,
                true,
                |a, v| a & v,
                |v| if !v { Some(false) } else { None },
            );
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

/// Like [`reduce_along_dim_par`] but also records the index (along the reduced
/// dimension) of the winning element, parallelizing over output positions.
/// `better(candidate, current_best)` decides replacement using a strict
/// comparison, so the first winner keeps its index (matches NumPy/PyTorch
/// argmax/argmin tie-breaking); `short(val)` returning `Some(v)` finalizes the
/// output early with value `v` at the current index (NaN propagation, boolean
/// any/all short-circuit).
#[inline]
pub(crate) fn reduce_arg_along_dim_par<T, Better, Short>(
    input: &[T],
    values: &mut [T],
    indices: &mut [i64],
    layout: &DimReductionLayout,
    init: T,
    better: Better,
    short: Short,
) where
    T: Copy + Send + Sync,
    Better: Fn(T, T) -> bool + Sync,
    Short: Fn(T) -> Option<T> + Sync,
{
    let inner = layout.inner;
    let dim_size = layout.dim_size;
    let outer_stride = layout.outer_stride;
    values
        .par_iter_mut()
        .zip(indices.par_iter_mut())
        .enumerate()
        .for_each(|(out_idx, (vout, iout))| {
            let o = out_idx / inner;
            let r = out_idx % inner;
            let mut best = init;
            let mut best_i = 0usize;
            let mut idx = o * outer_stride + r;
            for d in 0..dim_size {
                let val = input[idx];
                if let Some(fin) = short(val) {
                    best = fin;
                    best_i = d;
                    break;
                }
                if better(val, best) {
                    best = val;
                    best_i = d;
                }
                idx += inner;
            }
            *vout = best;
            *iout = best_i as i64;
        });
}

pub(crate) fn max_along_dim_with_indices(
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
            reduce_arg_along_dim_par(
                input,
                values,
                indices,
                &layout,
                f32::NEG_INFINITY,
                |v, b| v > b,
                |v| if v.is_nan() { Some(f32::NAN) } else { None },
            );
        }
        DataType::Float64 => {
            let input = tensor
                .data()
                .as_f64_slice()
                .ok_or_else(|| MinitensorError::internal_error("Failed to get f64 slice"))?;
            let values = values_data.as_f64_slice_mut().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get mutable f64 slice")
            })?;
            reduce_arg_along_dim_par(
                input,
                values,
                indices,
                &layout,
                f64::NEG_INFINITY,
                |v, b| v > b,
                |v| if v.is_nan() { Some(f64::NAN) } else { None },
            );
        }
        DataType::Int32 => {
            let input = tensor
                .data()
                .as_i32_slice()
                .ok_or_else(|| MinitensorError::internal_error("Failed to get i32 slice"))?;
            let values = values_data.as_i32_slice_mut().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get mutable i32 slice")
            })?;
            reduce_arg_along_dim_par(
                input,
                values,
                indices,
                &layout,
                i32::MIN,
                |v, b| v > b,
                |_| None,
            );
        }
        DataType::Int64 => {
            let input = tensor
                .data()
                .as_i64_slice()
                .ok_or_else(|| MinitensorError::internal_error("Failed to get i64 slice"))?;
            let values = values_data.as_i64_slice_mut().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get mutable i64 slice")
            })?;
            reduce_arg_along_dim_par(
                input,
                values,
                indices,
                &layout,
                i64::MIN,
                |v, b| v > b,
                |_| None,
            );
        }
        DataType::Bool => {
            let input = tensor
                .data()
                .as_bool_slice()
                .ok_or_else(|| MinitensorError::internal_error("Failed to get bool slice"))?;
            let values = values_data.as_bool_slice_mut().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get mutable bool slice")
            })?;
            reduce_arg_along_dim_par(
                input,
                values,
                indices,
                &layout,
                false,
                |_, _| false,
                |v| if v { Some(true) } else { None },
            );
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

pub(crate) fn nanmax_along_dim_with_indices(
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
