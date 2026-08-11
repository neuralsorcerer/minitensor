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
            for (slot, position) in values.iter_mut().zip(positions.iter()) {
                *slot = quantile_from_sorted_position(&buffer, position, interpolation);
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
            for (slot, position) in values.iter_mut().zip(positions.iter()) {
                *slot = quantile_from_sorted_position(&buffer, position, interpolation);
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
                // All-NaN input -> NaN: there is no value to report.
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
                for (slot, position) in values.iter_mut().zip(positions.iter()) {
                    *slot = quantile_from_sorted_position(&sorted, position, interpolation);
                }
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
                // All-NaN input -> NaN: there is no value to report.
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
                for (slot, position) in values.iter_mut().zip(positions.iter()) {
                    *slot = quantile_from_sorted_position(&sorted, position, interpolation);
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

fn quantiles_output_dims(tensor_ndim: usize, q_len: usize, keepdim: bool) -> Vec<usize> {
    if keepdim && tensor_ndim > 0 {
        let mut dims = vec![1; tensor_ndim + 1];
        dims[0] = q_len;
        dims
    } else {
        vec![q_len]
    }
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
/// the whole quantile NaN; with it on, NaNs are dropped and only an all-NaN
/// column yields NaN.
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

/// Several quantiles along `dim`, in one pass over each column.
///
/// The output is laid out `[q_len, ..reduced dims..]`, so a given slot's `q_len`
/// results are `slot_count` apart. The reduction therefore writes a slot-major
/// scratch in parallel and transposes it into place, rather than having every
/// task scatter into a strided output.
///
/// For a single `q` this quickselects, which is `O(n)`; for several it sorts
/// once and reads each position out of the sorted column, which beats `q_len`
/// separate selections as soon as `q_len` approaches `log n`. `nan_aware` drops
/// NaNs instead of poisoning the column, so the sorted length varies per slot
/// and the positions are recomputed when it changes.
fn quantiles_along_dim_core<T: TotalCmp + Send + Sync>(
    input: &[T],
    values: &mut [T],
    layout: &QuantileDimLayout,
    nan_aware: bool,
    qs: &[f64],
    interpolation: QuantileInterpolation,
) {
    let QuantileDimLayout {
        dim_size,
        inner,
        outer_stride,
        ..
    } = *layout;
    let q_len = qs.len();
    debug_assert!(q_len > 0 && values.len().is_multiple_of(q_len));
    let slot_count = values.len() / q_len;
    if slot_count == 0 {
        return;
    }

    let mut slot_major: Vec<T> = vec![T::nan(); slot_count * q_len];
    slot_major
        .par_chunks_mut(PAR_CHUNK * q_len)
        .enumerate()
        .for_each(|(chunk_idx, chunk)| {
            let mut buffer: Vec<T> = Vec::with_capacity(dim_size);
            // Positions depend only on the column length, which is constant
            // unless NaNs are being dropped.
            let mut cached: Option<(usize, Vec<QuantilePosition>)> = None;
            for (local, out) in chunk.chunks_mut(q_len).enumerate() {
                let slot = chunk_idx * PAR_CHUNK + local;
                let o = slot / inner;
                let r = slot % inner;
                buffer.clear();

                let mut poisoned = false;
                let mut idx = o * outer_stride + r;
                for _ in 0..dim_size {
                    let value = input[idx];
                    if value.is_nan() {
                        if !nan_aware {
                            poisoned = true;
                            break;
                        }
                    } else {
                        buffer.push(value);
                    }
                    idx += inner;
                }

                if poisoned || buffer.is_empty() {
                    out.fill(T::nan());
                    continue;
                }

                if q_len == 1 {
                    out[0] = quantile_from_unsorted(&mut buffer, qs[0], interpolation);
                    continue;
                }

                buffer.sort_by(|a, b| a.total_order(b));
                let positions = match cached {
                    Some((len, ref positions)) if len == buffer.len() => positions,
                    _ => {
                        cached = Some((buffer.len(), quantile_positions_for_len(buffer.len(), qs)));
                        &cached.as_ref().expect("positions just cached").1
                    }
                };
                for (slot_out, position) in out.iter_mut().zip(positions.iter()) {
                    *slot_out = quantile_from_sorted_position(&buffer, position, interpolation);
                }
            }
        });

    // Slot-major -> q-major. Each destination chunk is one q's full output, so
    // the writes stay contiguous.
    values
        .par_chunks_mut(slot_count)
        .enumerate()
        .for_each(|(qi, out_q)| {
            for (slot, out) in out_q.iter_mut().enumerate() {
                *out = slot_major[slot * q_len + qi];
            }
        });
}

/// Entry points for several quantiles along a dimension, per NaN mode.
macro_rules! quantiles_along_dim_entry {
    ($name:ident, $nan_aware:literal, $op:literal) => {
        pub(crate) fn $name(
            tensor: &Tensor,
            dim: usize,
            qs: &[f64],
            keepdim: bool,
            interpolation: QuantileInterpolation,
        ) -> Result<Tensor> {
            let layout = quantile_dim_layout(tensor, dim, keepdim, $op)?;

            // The quantile axis is prepended to the reduced shape.
            let mut out_dims = Vec::with_capacity(layout.values_shape.ndim() + 1);
            out_dims.push(qs.len());
            out_dims.extend_from_slice(layout.values_shape.dims());
            let shape = Shape::new(out_dims);

            let mut values_data =
                TensorData::zeros_on_device(shape.numel(), tensor.dtype(), tensor.device());

            match tensor.dtype() {
                DataType::Float32 => {
                    let input = tensor.data().as_f32_slice().ok_or_else(|| {
                        MinitensorError::internal_error("Failed to get f32 slice")
                    })?;
                    let values = values_data.as_f32_slice_mut().ok_or_else(|| {
                        MinitensorError::internal_error("Failed to get mutable f32 slice")
                    })?;
                    quantiles_along_dim_core(input, values, &layout, $nan_aware, qs, interpolation);
                }
                DataType::Float64 => {
                    let input = tensor.data().as_f64_slice().ok_or_else(|| {
                        MinitensorError::internal_error("Failed to get f64 slice")
                    })?;
                    let values = values_data.as_f64_slice_mut().ok_or_else(|| {
                        MinitensorError::internal_error("Failed to get mutable f64 slice")
                    })?;
                    quantiles_along_dim_core(input, values, &layout, $nan_aware, qs, interpolation);
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
    };
}

quantiles_along_dim_entry!(quantiles_along_dim, false, "quantile");
quantiles_along_dim_entry!(nanquantiles_along_dim, true, "nanquantile");

/// A median is the *lower* of the two middle order statistics for an even
/// count, rather than the interpolated midpoint of the two. Selecting rank
/// `(len - 1) / 2` is exactly
/// [`QuantileInterpolation::Lower`] at `q = 0.5`, so the median reductions are
/// the quantile reductions with that interpolation pinned.
///
/// This matters beyond the value itself: [`crate::autograd::MedianBackward`]
/// routes the gradient to the input elements *equal* to the result, splitting
/// it among ties. An interpolated median equals no input element, so the tie
/// count is zero and the gradient comes back NaN. Reaching for `Linear` here
/// silently produced exactly that for every even-length input.
const MEDIAN_INTERPOLATION: QuantileInterpolation = QuantileInterpolation::Lower;

/// Median of the whole tensor, ignoring NaN. An all-NaN tensor yields NaN.
pub(crate) fn nanmedian_all(tensor: &Tensor, keepdim: bool) -> Result<Tensor> {
    let output_dims = if keepdim && tensor.ndim() > 0 {
        vec![1; tensor.ndim()]
    } else {
        Vec::new()
    };
    let shape = Shape::new(output_dims);
    let mut values_data =
        TensorData::zeros_on_device(shape.numel(), tensor.dtype(), tensor.device());

    macro_rules! median_all_arm {
        ($accessor:ident, $mut_accessor:ident, $ty:ty, $tyname:literal) => {{
            let data = tensor.data().$accessor().ok_or_else(|| {
                MinitensorError::internal_error(concat!("Failed to get ", $tyname, " slice"))
            })?;
            let values = values_data.$mut_accessor().ok_or_else(|| {
                MinitensorError::internal_error(concat!(
                    "Failed to get mutable ",
                    $tyname,
                    " slice"
                ))
            })?;
            let mut buffer: Vec<$ty> = data.iter().copied().filter(|v| !v.is_nan()).collect();
            values[0] = if buffer.is_empty() {
                <$ty>::NAN
            } else {
                quantile_from_unsorted(&mut buffer, 0.5, MEDIAN_INTERPOLATION)
            };
        }};
    }

    match tensor.dtype() {
        DataType::Float32 => median_all_arm!(as_f32_slice, as_f32_slice_mut, f32, "f32"),
        DataType::Float64 => median_all_arm!(as_f64_slice, as_f64_slice_mut, f64, "f64"),
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

/// Median along `dim`, ignoring NaN.
///
/// A column with no non-NaN value yields NaN, and unlike `nanquantile` an
/// empty reduced dimension is a NaN result rather than an error -- the median
/// of nothing is as undefined as the median of all-NaN, so both answer the
/// same way.
pub(crate) fn nanmedian_along_dim(tensor: &Tensor, dim: usize, keepdim: bool) -> Result<Tensor> {
    let dims = tensor.shape().dims();
    if !dims.is_empty() && dims[dim] == 0 {
        let mut out_dims = dims.to_vec();
        if keepdim {
            out_dims[dim] = 1;
        } else {
            out_dims.remove(dim);
        }
        let shape = Shape::new(out_dims);
        let numel = shape.numel();
        let data = match tensor.dtype() {
            DataType::Float32 => TensorData::from_vec_f32(vec![f32::NAN; numel], tensor.device()),
            DataType::Float64 => TensorData::from_vec_f64(vec![f64::NAN; numel], tensor.device()),
            _ => unreachable!("dtype validated"),
        };
        return Ok(Tensor::new(
            Arc::new(data),
            shape,
            tensor.dtype(),
            tensor.device(),
            tensor.requires_grad(),
        ));
    }

    nanquantile_along_dim(tensor, dim, keepdim, 0.5, MEDIAN_INTERPOLATION)
}
