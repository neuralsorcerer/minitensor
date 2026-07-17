// Copyright (c) Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

use super::*;
use crate::autograd::GatherBackward;
use crate::autograd::MinMaxBackward;
use crate::{
    autograd::add_to_graph,
    error::{MinitensorError, Result},
    tensor::{DataType, Shape, Tensor, TensorData},
};
use rayon::prelude::*;
use std::cmp::Ordering;
use std::sync::Arc;

pub(crate) fn any_along_dim(tensor: &Tensor, dim: usize, keepdim: bool) -> Result<Tensor> {
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
    let (output, norm_dim) = match dim {
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

            (
                Tensor::new(
                    Arc::new(result_data),
                    result_shape,
                    tensor.dtype(),
                    tensor.device(),
                    tensor.requires_grad(),
                ),
                None,
            )
        }
        Some(d) => {
            let d = normalize_dim(d, tensor.ndim())?;
            (max_along_dim(tensor, d, keepdim)?, Some(d))
        }
    };
    attach_minmax_grad(output, tensor, norm_dim, keepdim, true, false)
}

/// Attach a [`GatherBackward`] gradient to a value tensor that was formed by
/// gathering the input along `dim` at `indices` (`sort`/`topk`). The forward is
/// `values = gather(input, dim, indices)`, so the backward scatters the gradient
/// straight back to the selected source positions.
pub(crate) fn attach_gather_like_grad(
    values: Tensor,
    input: &Tensor,
    dim: usize,
    indices: &Tensor,
) -> Result<Tensor> {
    if !input.requires_grad() || !input.dtype().is_float() {
        return Ok(values);
    }
    let index = indices
        .data()
        .as_i64_slice()
        .ok_or_else(|| MinitensorError::internal_error("selection indices must be int64"))?
        .to_vec();
    let grad_fn = Arc::new(GatherBackward {
        input_id: input.id(),
        input_shape: input.shape().dims().to_vec(),
        dim,
        index,
    });
    let mut values = values;
    values.set_grad_fn(Some(grad_fn.clone()));
    add_to_graph(&values, Some(grad_fn))?;
    Ok(values)
}

/// Attach a [`MinMaxBackward`] gradient to a `min`/`max`/`nanmax`/`nanmin` value
/// reduction (`nan_aware` selects the NaN-ignoring recompute in the backward).
fn attach_minmax_grad(
    output: Tensor,
    input: &Tensor,
    dim: Option<usize>,
    keepdim: bool,
    is_max: bool,
    nan_aware: bool,
) -> Result<Tensor> {
    if !input.requires_grad() || !input.dtype().is_float() {
        return Ok(output);
    }
    let grad_fn = Arc::new(MinMaxBackward {
        input_id: input.id(),
        input: input.detach(),
        dim,
        keepdim,
        is_max,
        nan_aware,
    });
    let mut output = output;
    output.set_grad_fn(Some(grad_fn.clone()));
    add_to_graph(&output, Some(grad_fn))?;
    Ok(output)
}

/// Minimum value along specified dimension
pub fn min(tensor: &Tensor, dim: Option<isize>, keepdim: bool) -> Result<Tensor> {
    let (output, norm_dim) = match dim {
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

            (
                Tensor::new(
                    Arc::new(result_data),
                    result_shape,
                    tensor.dtype(),
                    tensor.device(),
                    tensor.requires_grad(),
                ),
                None,
            )
        }
        Some(d) => {
            let d = normalize_dim(d, tensor.ndim())?;
            (min_along_dim(tensor, d, keepdim)?, Some(d))
        }
    };
    attach_minmax_grad(output, tensor, norm_dim, keepdim, false, false)
}

/// NaN-aware maximum value along specified dimension
pub fn nanmax(tensor: &Tensor, dim: Option<isize>, keepdim: bool) -> Result<Tensor> {
    if !tensor.dtype().is_float() {
        return max(tensor, dim, keepdim);
    }

    let (output, norm_dim) = match dim {
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

            (
                Tensor::new(
                    Arc::new(result_data),
                    result_shape,
                    tensor.dtype(),
                    tensor.device(),
                    tensor.requires_grad(),
                ),
                None,
            )
        }
        Some(d) => {
            let d = normalize_dim(d, tensor.ndim())?;
            let (values, _) = nanmax_along_dim_with_indices(tensor, d, keepdim)?;
            (values, Some(d))
        }
    };
    attach_minmax_grad(output, tensor, norm_dim, keepdim, true, true)
}

/// NaN-aware minimum value along specified dimension
pub fn nanmin(tensor: &Tensor, dim: Option<isize>, keepdim: bool) -> Result<Tensor> {
    if !tensor.dtype().is_float() {
        return min(tensor, dim, keepdim);
    }

    let (output, norm_dim) = match dim {
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

            (
                Tensor::new(
                    Arc::new(result_data),
                    result_shape,
                    tensor.dtype(),
                    tensor.device(),
                    tensor.requires_grad(),
                ),
                None,
            )
        }
        Some(d) => {
            let d = normalize_dim(d, tensor.ndim())?;
            let (values, _) = nanmin_along_dim_with_indices(tensor, d, keepdim)?;
            (values, Some(d))
        }
    };
    attach_minmax_grad(output, tensor, norm_dim, keepdim, false, true)
}

/// Maximum values and their indices along specified dimension
pub fn max_with_indices(tensor: &Tensor, dim: isize, keepdim: bool) -> Result<(Tensor, Tensor)> {
    let d = normalize_dim(dim, tensor.ndim())?;
    let (values, indices) = max_along_dim_with_indices(tensor, d, keepdim)?;
    let values = attach_minmax_grad(values, tensor, Some(d), keepdim, true, false)?;
    Ok((values, indices))
}

/// NaN-aware maximum values and their indices along specified dimension
pub fn nanmax_with_indices(tensor: &Tensor, dim: isize, keepdim: bool) -> Result<(Tensor, Tensor)> {
    if !tensor.dtype().is_float() {
        return max_with_indices(tensor, dim, keepdim);
    }

    let d = normalize_dim(dim, tensor.ndim())?;
    let (values, indices) = nanmax_along_dim_with_indices(tensor, d, keepdim)?;
    let values = attach_minmax_grad(values, tensor, Some(d), keepdim, true, true)?;
    Ok((values, indices))
}

/// Minimum values and their indices along specified dimension
pub fn min_with_indices(tensor: &Tensor, dim: isize, keepdim: bool) -> Result<(Tensor, Tensor)> {
    let d = normalize_dim(dim, tensor.ndim())?;
    let (values, indices) = min_along_dim_with_indices(tensor, d, keepdim)?;
    let values = attach_minmax_grad(values, tensor, Some(d), keepdim, false, false)?;
    Ok((values, indices))
}

/// NaN-aware minimum values and their indices along specified dimension
pub fn nanmin_with_indices(tensor: &Tensor, dim: isize, keepdim: bool) -> Result<(Tensor, Tensor)> {
    if !tensor.dtype().is_float() {
        return min_with_indices(tensor, dim, keepdim);
    }

    let d = normalize_dim(dim, tensor.ndim())?;
    let (values, indices) = nanmin_along_dim_with_indices(tensor, d, keepdim)?;
    let values = attach_minmax_grad(values, tensor, Some(d), keepdim, false, true)?;
    Ok((values, indices))
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

#[inline]
fn select_topk_entries<T>(
    entries: &mut [(usize, T)],
    k: usize,
    sorted: bool,
    compare: fn(&(usize, T), &(usize, T)) -> Ordering,
) {
    if k == 0 || entries.is_empty() {
        return;
    }

    if k < entries.len() {
        entries.select_nth_unstable_by(k - 1, compare);
        if sorted {
            entries[..k].sort_by(compare);
        }
    } else if sorted {
        entries.sort_by(compare);
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

                    let compare = if largest { cmp_f32_desc } else { cmp_f32_asc };
                    select_topk_entries(&mut entries, k, sorted, compare);

                    // Output shape is (outer, k, inner); write row-major so a
                    // non-trailing reduction axis (inner > 1) lands correctly.
                    for j in 0..k {
                        let (index, value) = entries[j];
                        let pos = o * k * inner + j * inner + r;
                        values[pos] = value;
                        indices[pos] = index as i64;
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

                    let compare = if largest { cmp_f64_desc } else { cmp_f64_asc };
                    select_topk_entries(&mut entries, k, sorted, compare);

                    // Output shape is (outer, k, inner); write row-major so a
                    // non-trailing reduction axis (inner > 1) lands correctly.
                    for j in 0..k {
                        let (index, value) = entries[j];
                        let pos = o * k * inner + j * inner + r;
                        values[pos] = value;
                        indices[pos] = index as i64;
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

                    let compare = if largest { cmp_i32_desc } else { cmp_i32_asc };
                    select_topk_entries(&mut entries, k, sorted, compare);

                    // Output shape is (outer, k, inner); write row-major so a
                    // non-trailing reduction axis (inner > 1) lands correctly.
                    for j in 0..k {
                        let (index, value) = entries[j];
                        let pos = o * k * inner + j * inner + r;
                        values[pos] = value;
                        indices[pos] = index as i64;
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

                    let compare = if largest { cmp_i64_desc } else { cmp_i64_asc };
                    select_topk_entries(&mut entries, k, sorted, compare);

                    // Output shape is (outer, k, inner); write row-major so a
                    // non-trailing reduction axis (inner > 1) lands correctly.
                    for j in 0..k {
                        let (index, value) = entries[j];
                        let pos = o * k * inner + j * inner + r;
                        values[pos] = value;
                        indices[pos] = index as i64;
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

                    let compare = if largest { cmp_bool_desc } else { cmp_bool_asc };
                    select_topk_entries(&mut entries, k, sorted, compare);

                    // Output shape is (outer, k, inner); write row-major so a
                    // non-trailing reduction axis (inner > 1) lands correctly.
                    for j in 0..k {
                        let (index, value) = entries[j];
                        let pos = o * k * inner + j * inner + r;
                        values[pos] = value;
                        indices[pos] = index as i64;
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

    // `values = gather(input, axis, indices)`; scatter the gradient back.
    let values = attach_gather_like_grad(values, tensor, axis, &indices)?;

    Ok((values, indices))
}
