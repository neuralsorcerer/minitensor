// Copyright (c) Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

use super::*;
use crate::autograd::NanSumBackward;
use crate::autograd::SumBackward;
use crate::{
    autograd::add_to_graph,
    error::{MinitensorError, Result},
    tensor::{DataType, Shape, Tensor, TensorData},
};
use rayon::prelude::*;
use std::cmp::Ordering;
use std::sync::Arc;

pub(crate) fn median_all(tensor: &Tensor) -> Result<(Tensor, Option<Tensor>)> {
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

/// Take the median of every 1-D slice along a dimension, parallelizing over the
/// outer index.
///
/// Output is `(outer, inner)`, so each outer position owns a disjoint `inner`
/// span and `par_chunks_mut` hands them out without overlap. The selection
/// itself is `select_nth_unstable_by`, so this stays linear per slice; only the
/// outer loop used to be serial, which left `median` several times slower than
/// `quantile(0.5)` computing the same thing.
///
/// How a dtype represents and detects NaN: the value to emit for a slice that
/// contains one, and the predicate that finds it. Integer dtypes have neither.
type NanHandling<T> = Option<(T, fn(&T) -> bool)>;

/// `nan` carries the floating-point NaN handling: a NaN anywhere in a slice
/// makes that whole median NaN. Integer instantiations pass `None` and skip the
/// check entirely.
#[allow(clippy::too_many_arguments)]
fn median_along_dim_par<T>(
    input: &[T],
    values: &mut [T],
    indices: &mut [i64],
    inner: usize,
    dim_size: usize,
    outer_stride: usize,
    median_pos: usize,
    nan: NanHandling<T>,
    compare: fn(&(usize, T), &(usize, T)) -> Ordering,
) where
    T: Copy + Send + Sync,
{
    values
        .par_chunks_mut(inner)
        .zip(indices.par_chunks_mut(inner))
        .enumerate()
        .for_each(|(o, (vchunk, ichunk))| {
            let mut entries: Vec<(usize, T)> = Vec::with_capacity(dim_size);
            for r in 0..inner {
                entries.clear();
                let base = o * outer_stride + r;
                let mut saw_nan = false;
                for d in 0..dim_size {
                    let value = input[base + d * inner];
                    if let Some((_, is_nan)) = nan
                        && is_nan(&value)
                    {
                        saw_nan = true;
                        break;
                    }
                    entries.push((d, value));
                }

                if let (true, Some((nan_value, _))) = (saw_nan, nan) {
                    vchunk[r] = nan_value;
                    continue;
                }

                entries.select_nth_unstable_by(median_pos, compare);
                let (index, value) = entries[median_pos];
                vchunk[r] = value;
                ichunk[r] = index as i64;
            }
        });
}

pub(crate) fn median_along_dim(
    tensor: &Tensor,
    dim: usize,
    keepdim: bool,
) -> Result<(Tensor, Tensor)> {
    let dims = tensor.shape().dims();
    let dim_size = if dims.is_empty() { 1 } else { dims[dim] };

    ensure_non_empty(dim_size, "nanquantile")?;

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

            median_along_dim_par(
                input,
                values,
                indices,
                inner,
                dim_size,
                outer_stride,
                median_pos,
                Some((f32::NAN, |v: &f32| v.is_nan())),
                cmp_f32_asc,
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
            let indices = indices_data.as_i64_slice_mut().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get mutable i64 slice")
            })?;

            median_along_dim_par(
                input,
                values,
                indices,
                inner,
                dim_size,
                outer_stride,
                median_pos,
                Some((f64::NAN, |v: &f64| v.is_nan())),
                cmp_f64_asc,
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
            let indices = indices_data.as_i64_slice_mut().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get mutable i64 slice")
            })?;

            median_along_dim_par(
                input,
                values,
                indices,
                inner,
                dim_size,
                outer_stride,
                median_pos,
                None,
                cmp_i32_asc,
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
            let indices = indices_data.as_i64_slice_mut().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get mutable i64 slice")
            })?;

            median_along_dim_par(
                input,
                values,
                indices,
                inner,
                dim_size,
                outer_stride,
                median_pos,
                None,
                cmp_i64_asc,
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
            let indices = indices_data.as_i64_slice_mut().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get mutable i64 slice")
            })?;

            median_along_dim_par(
                input,
                values,
                indices,
                inner,
                dim_size,
                outer_stride,
                median_pos,
                None,
                cmp_bool_asc,
            );
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

pub(crate) use crate::ops::util::normalize_dim;

/// Sum reduction along specified dimensions
pub fn sum(tensor: &Tensor, dim: Option<Vec<isize>>, keepdim: bool) -> Result<Tensor> {
    // Summing a mask counts its true entries, which is the most common thing
    // anyone does with one. `bool` has no addition to accumulate in, so the
    // count lands in `int64` -- the same answer NumPy and PyTorch give -- and
    // the existing integer path takes it from there. Rejecting this outright
    // left `mask.sum()` as the one hole in the boolean reductions: `max`,
    // `min`, `all`, `any`, `argmax` and `sort` all already worked.
    if tensor.dtype() == DataType::Bool {
        return sum(&tensor.astype(DataType::Int64)?, dim, keepdim);
    }

    // Normalise negative dimensions and deduplicate
    let dim = normalize_reduction_dims(dim, tensor.ndim())?;
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
