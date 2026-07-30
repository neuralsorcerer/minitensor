// Copyright (c) Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

use super::*;
use crate::{
    error::{MinitensorError, Result},
    ops::map::PAR_CHUNK,
    tensor::{DataType, Shape, Tensor, TensorData},
};
use rayon::prelude::*;
use std::sync::Arc;

pub(crate) fn nanmedian_all(tensor: &Tensor, keepdim: bool) -> Result<Tensor> {
    let output_dims = if keepdim && tensor.ndim() > 0 {
        vec![1; tensor.ndim()]
    } else {
        Vec::new()
    };
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
            let mut buffer: Vec<f32> = data.iter().copied().filter(|v| !v.is_nan()).collect();
            values[0] = if buffer.is_empty() {
                f32::NAN
            } else {
                quantile_from_unsorted(&mut buffer, 0.5, QuantileInterpolation::Linear)
            };
        }
        DataType::Float64 => {
            let data = tensor
                .data()
                .as_f64_slice()
                .ok_or_else(|| MinitensorError::internal_error("Failed to get f64 slice"))?;
            let values = values_data.as_f64_slice_mut().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get mutable f64 slice")
            })?;
            let mut buffer: Vec<f64> = data.iter().copied().filter(|v| !v.is_nan()).collect();
            values[0] = if buffer.is_empty() {
                f64::NAN
            } else {
                quantile_from_unsorted(&mut buffer, 0.5, QuantileInterpolation::Linear)
            };
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

pub(crate) fn nanmedian_along_dim(tensor: &Tensor, dim: usize, keepdim: bool) -> Result<Tensor> {
    let dims = tensor.shape().dims();
    let dim_size = if dims.is_empty() { 1 } else { dims[dim] };

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
            let mut buffer = Vec::with_capacity(dim_size);
            for o in 0..outer {
                for r in 0..inner {
                    buffer.clear();
                    for d in 0..dim_size {
                        let value = input[o * outer_stride + d * inner + r];
                        if !value.is_nan() {
                            buffer.push(value);
                        }
                    }
                    let out_idx = o * inner + r;
                    values[out_idx] = if buffer.is_empty() {
                        f32::NAN
                    } else {
                        quantile_from_unsorted(&mut buffer, 0.5, QuantileInterpolation::Linear)
                    };
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
            let mut buffer = Vec::with_capacity(dim_size);
            for o in 0..outer {
                for r in 0..inner {
                    buffer.clear();
                    for d in 0..dim_size {
                        let value = input[o * outer_stride + d * inner + r];
                        if !value.is_nan() {
                            buffer.push(value);
                        }
                    }
                    let out_idx = o * inner + r;
                    values[out_idx] = if buffer.is_empty() {
                        f64::NAN
                    } else {
                        quantile_from_unsorted(&mut buffer, 0.5, QuantileInterpolation::Linear)
                    };
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

pub(crate) fn quantiles_all(
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

            let Some(mut buffer) = copy_or_none_if_nan(data) else {
                values.fill(f32::NAN);
                return Ok(Tensor::new(
                    Arc::new(values_data),
                    shape,
                    tensor.dtype(),
                    tensor.device(),
                    tensor.requires_grad(),
                ));
            };

            if q_len == 1 {
                values[0] = quantile_from_unsorted(&mut buffer, qs[0], interpolation);
                return Ok(Tensor::new(
                    Arc::new(values_data),
                    shape,
                    tensor.dtype(),
                    tensor.device(),
                    tensor.requires_grad(),
                ));
            }

            let positions = quantile_positions_for_len(buffer.len(), qs);
            buffer.sort_by(|a, b| a.total_cmp(b));
            quantiles_from_sorted_f32(&buffer, &positions, interpolation, values);
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

            let Some(mut buffer) = copy_or_none_if_nan(data) else {
                values.fill(f64::NAN);
                return Ok(Tensor::new(
                    Arc::new(values_data),
                    shape,
                    tensor.dtype(),
                    tensor.device(),
                    tensor.requires_grad(),
                ));
            };

            if q_len == 1 {
                values[0] = quantile_from_unsorted(&mut buffer, qs[0], interpolation);
                return Ok(Tensor::new(
                    Arc::new(values_data),
                    shape,
                    tensor.dtype(),
                    tensor.device(),
                    tensor.requires_grad(),
                ));
            }

            let positions = quantile_positions_for_len(buffer.len(), qs);
            buffer.sort_by(|a, b| a.total_cmp(b));
            quantiles_from_sorted_f64(&buffer, &positions, interpolation, values);
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

fn copy_or_none_if_nan<T: num_traits::Float>(data: &[T]) -> Option<Vec<T>> {
    let mut out = Vec::with_capacity(data.len());
    for &value in data {
        if value.is_nan() {
            return None;
        }
        out.push(value);
    }
    Some(out)
}

pub(crate) fn nanquantiles_all(
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
                fill_nanquantiles_all_single_f32(data[0], values);
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
                // All-NaN input -> NaN (NumPy/PyTorch semantics).
                values[0] = if buffer.is_empty() {
                    f32::NAN
                } else {
                    quantile_from_unsorted(&mut buffer, qs[0], interpolation)
                };
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
                values.fill(f32::NAN);
            } else {
                let positions = quantile_positions_for_len(sorted.len(), qs);
                sorted.sort_by(|a, b| a.total_cmp(b));
                quantiles_from_sorted_f32(&sorted, &positions, interpolation, values);
            }
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
                fill_nanquantiles_all_single_f64(data[0], values);
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
                // All-NaN input -> NaN (NumPy/PyTorch semantics).
                values[0] = if buffer.is_empty() {
                    f64::NAN
                } else {
                    quantile_from_unsorted(&mut buffer, qs[0], interpolation)
                };
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
                values.fill(f64::NAN);
            } else {
                let positions = quantile_positions_for_len(sorted.len(), qs);
                sorted.sort_by(|a, b| a.total_cmp(b));
                quantiles_from_sorted_f64(&sorted, &positions, interpolation, values);
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

fn quantiles_output_dims(tensor_ndim: usize, q_len: usize, keepdim: bool) -> Vec<usize> {
    if keepdim && tensor_ndim > 0 {
        let mut dims = vec![1; tensor_ndim + 1];
        dims[0] = q_len;
        dims
    } else {
        vec![q_len]
    }
}

pub(crate) fn quantiles_along_dim(
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
                                quantile_from_unsorted(&mut buffer, q_value, interpolation);
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
                                values[out_idx] = quantile_from_sorted_position_f32(
                                    &buffer,
                                    position,
                                    interpolation,
                                );
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
                                quantile_from_unsorted(&mut buffer, q_value, interpolation);
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
                                values[out_idx] = quantile_from_sorted_position_f64(
                                    &buffer,
                                    position,
                                    interpolation,
                                );
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

/// Geometry of a dimension reduction that also copes with a 0-d tensor
/// (treated as a single length-1 column).
pub(crate) struct QuantileDimLayout {
    pub(crate) values_shape: Shape,
    pub(crate) dim_size: usize,
    pub(crate) inner: usize,
    pub(crate) outer_stride: usize,
}

pub(crate) fn quantile_dim_layout(
    tensor: &Tensor,
    dim: usize,
    keepdim: bool,
    op: &str,
) -> Result<QuantileDimLayout> {
    let dims = tensor.shape().dims();
    let dim_size = if dims.is_empty() { 1 } else { dims[dim] };

    if dim_size == 0 {
        return Err(MinitensorError::invalid_argument(format!(
            "{op}() does not support reductions over empty dimensions"
        )));
    }

    let mut out_dims = if dims.is_empty() {
        vec![1]
    } else {
        dims.to_vec()
    };
    if !out_dims.is_empty() {
        if keepdim {
            out_dims[dim] = 1;
        } else {
            out_dims.remove(dim);
        }
    }

    let inner = if dims.is_empty() || dim + 1 >= dims.len() {
        1
    } else {
        dims[dim + 1..].iter().product()
    };

    Ok(QuantileDimLayout {
        values_shape: Shape::new(out_dims),
        dim_size,
        inner,
        outer_stride: dim_size * inner,
    })
}

/// One quantile along `dim`.
///
/// Output slots are independent, so they parallelize; each rayon task allocates
/// one gather buffer and reuses it across the slots it is given. (The four
/// dtype-and-NaN-mode copies this replaces each walked every slot on a single
/// thread.)
///
/// `nan_aware` decides what a NaN in a column means: with it off, one NaN makes
/// the whole quantile NaN (`torch.quantile`); with it on, NaNs are dropped and
/// only an all-NaN column yields NaN (`numpy.nanquantile`).
fn quantile_along_dim_core<T: TotalCmp + Send + Sync>(
    input: &[T],
    values: &mut [T],
    layout: &QuantileDimLayout,
    nan_aware: bool,
    q: f64,
    interpolation: QuantileInterpolation,
) {
    let QuantileDimLayout {
        dim_size,
        inner,
        outer_stride,
        ..
    } = *layout;

    // A slot's column is `dim_size` elements strided by `inner`; grouping slots
    // into chunks amortizes the buffer allocation over `PAR_CHUNK` of them.
    values
        .par_chunks_mut(PAR_CHUNK)
        .enumerate()
        .for_each(|(chunk_idx, chunk)| {
            let mut buffer: Vec<T> = Vec::with_capacity(dim_size);
            for (local, slot) in chunk.iter_mut().enumerate() {
                let out_idx = chunk_idx * PAR_CHUNK + local;
                let o = out_idx / inner;
                let r = out_idx % inner;
                buffer.clear();

                let mut saw_nan = false;
                let mut idx = o * outer_stride + r;
                for _ in 0..dim_size {
                    let value = input[idx];
                    if value.is_nan() {
                        if !nan_aware {
                            saw_nan = true;
                            break;
                        }
                    } else {
                        buffer.push(value);
                    }
                    idx += inner;
                }

                *slot = if saw_nan || buffer.is_empty() {
                    T::nan()
                } else {
                    quantile_from_unsorted(&mut buffer, q, interpolation)
                };
            }
        });
}

/// Entry points for one quantile along a dimension, per dtype and NaN mode.
macro_rules! quantile_along_dim_entry {
    ($name:ident, $nan_aware:literal, $op:literal) => {
        pub(crate) fn $name(
            tensor: &Tensor,
            dim: usize,
            keepdim: bool,
            q: f64,
            interpolation: QuantileInterpolation,
        ) -> Result<Tensor> {
            let layout = quantile_dim_layout(tensor, dim, keepdim, $op)?;
            let mut values_data = TensorData::zeros_on_device(
                layout.values_shape.numel(),
                tensor.dtype(),
                tensor.device(),
            );

            match tensor.dtype() {
                DataType::Float32 => {
                    let input = tensor.data().as_f32_slice().ok_or_else(|| {
                        MinitensorError::internal_error("Failed to get f32 slice")
                    })?;
                    let values = values_data.as_f32_slice_mut().ok_or_else(|| {
                        MinitensorError::internal_error("Failed to get mutable f32 slice")
                    })?;
                    quantile_along_dim_core(input, values, &layout, $nan_aware, q, interpolation);
                }
                DataType::Float64 => {
                    let input = tensor.data().as_f64_slice().ok_or_else(|| {
                        MinitensorError::internal_error("Failed to get f64 slice")
                    })?;
                    let values = values_data.as_f64_slice_mut().ok_or_else(|| {
                        MinitensorError::internal_error("Failed to get mutable f64 slice")
                    })?;
                    quantile_along_dim_core(input, values, &layout, $nan_aware, q, interpolation);
                }
                _ => unreachable!("dtype validated"),
            }

            Ok(Tensor::new(
                Arc::new(values_data),
                layout.values_shape,
                tensor.dtype(),
                tensor.device(),
                tensor.requires_grad(),
            ))
        }
    };
}

quantile_along_dim_entry!(quantile_along_dim, false, "quantile");
quantile_along_dim_entry!(nanquantile_along_dim, true, "nanquantile");
