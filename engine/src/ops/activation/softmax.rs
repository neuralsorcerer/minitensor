// Copyright (c) 2026 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

use super::*;
use crate::error::MinitensorError;
use crate::error::Result;
use crate::tensor::DataType;
use crate::tensor::Shape;
use crate::tensor::Strides;
use crate::tensor::Tensor;
use crate::tensor::TensorData;
use rayon::prelude::*;

use num_traits::Float;

pub(crate) fn logaddexp_f32(
    lhs: &Tensor,
    rhs: &Tensor,
    output_shape: &Shape,
) -> Result<TensorData> {
    let lhs_data = lhs.data().as_f32_slice().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get f32 slice from lhs tensor")
    })?;
    let rhs_data = rhs.data().as_f32_slice().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get f32 slice from rhs tensor")
    })?;

    let out = crate::ops::kernels::broadcast_binary_map(
        lhs_data,
        rhs_data,
        lhs.shape(),
        rhs.shape(),
        output_shape,
        |a: f32, b: f32| {
            if a.is_nan() || b.is_nan() {
                f32::NAN
            } else {
                let max = a.max(b);
                if max.is_infinite() {
                    max
                } else {
                    let exp_a = (a - max).exp();
                    let exp_b = (b - max).exp();
                    max + (exp_a + exp_b).ln()
                }
            }
        },
    )?;
    Ok(TensorData::from_vec::<f32>(
        out,
        DataType::Float32,
        lhs.device(),
    ))
}

pub(crate) fn logaddexp_f64(
    lhs: &Tensor,
    rhs: &Tensor,
    output_shape: &Shape,
) -> Result<TensorData> {
    let lhs_data = lhs.data().as_f64_slice().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get f64 slice from lhs tensor")
    })?;
    let rhs_data = rhs.data().as_f64_slice().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get f64 slice from rhs tensor")
    })?;

    let out = crate::ops::kernels::broadcast_binary_map(
        lhs_data,
        rhs_data,
        lhs.shape(),
        rhs.shape(),
        output_shape,
        |a: f64, b: f64| {
            if a.is_nan() || b.is_nan() {
                f64::NAN
            } else {
                let max = a.max(b);
                if max.is_infinite() {
                    max
                } else {
                    let exp_a = (a - max).exp();
                    let exp_b = (b - max).exp();
                    max + (exp_a + exp_b).ln()
                }
            }
        },
    )?;
    Ok(TensorData::from_vec::<f64>(
        out,
        DataType::Float64,
        lhs.device(),
    ))
}

pub(crate) fn tanh_f32(tensor: &Tensor) -> Result<TensorData> {
    let input_data = tensor.data().as_f32_slice().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get f32 slice from input tensor")
    })?;

    let out = unary_map(input_data, f32::tanh);
    Ok(TensorData::from_vec::<f32>(
        out,
        DataType::Float32,
        tensor.device(),
    ))
}

pub(crate) fn tanh_f64(tensor: &Tensor) -> Result<TensorData> {
    let input_data = tensor.data().as_f64_slice().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get f64 slice from input tensor")
    })?;

    let out = unary_map(input_data, f64::tanh);
    Ok(TensorData::from_vec::<f64>(
        out,
        DataType::Float64,
        tensor.device(),
    ))
}

pub(crate) fn sigmoid_f32(tensor: &Tensor) -> Result<TensorData> {
    let input_data = tensor.data().as_f32_slice().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get f32 slice from input tensor")
    })?;

    let out = unary_map(input_data, stable_sigmoid_f32);
    Ok(TensorData::from_vec::<f32>(
        out,
        DataType::Float32,
        tensor.device(),
    ))
}

pub(crate) fn sigmoid_f64(tensor: &Tensor) -> Result<TensorData> {
    let input_data = tensor.data().as_f64_slice().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get f64 slice from input tensor")
    })?;

    let out = unary_map(input_data, stable_sigmoid_f64);
    Ok(TensorData::from_vec::<f64>(
        out,
        DataType::Float64,
        tensor.device(),
    ))
}

#[inline]
fn stable_sigmoid_f32(val: f32) -> f32 {
    if val >= 0.0 {
        let exp_neg = (-val).exp();
        1.0 / (1.0 + exp_neg)
    } else {
        let exp_pos = val.exp();
        exp_pos / (1.0 + exp_pos)
    }
}

#[inline]
fn stable_sigmoid_f64(val: f64) -> f64 {
    if val >= 0.0 {
        let exp_neg = (-val).exp();
        1.0 / (1.0 + exp_neg)
    } else {
        let exp_pos = val.exp();
        exp_pos / (1.0 + exp_pos)
    }
}

pub(crate) fn relu_f32(
    tensor: &Tensor,
    store_mask: bool,
) -> Result<(TensorData, Option<Vec<bool>>)> {
    let input_data = tensor.data().as_f32_slice().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get f32 slice from input tensor")
    })?;

    // NaN propagates through ReLU; the backward mask marks strictly positive
    // inputs only. The mask is materialized only when the caller will attach
    // a gradient function (`store_mask`).
    let out = unary_map(
        input_data,
        |v: f32| if v.is_nan() || v > 0.0 { v } else { 0.0 },
    );
    let mask = store_mask.then(|| unary_map(input_data, |v: f32| v > 0.0));
    Ok((
        TensorData::from_vec::<f32>(out, DataType::Float32, tensor.device()),
        mask,
    ))
}

pub(crate) fn relu_f64(
    tensor: &Tensor,
    store_mask: bool,
) -> Result<(TensorData, Option<Vec<bool>>)> {
    let input_data = tensor.data().as_f64_slice().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get f64 slice from input tensor")
    })?;

    // NaN propagates through ReLU; the backward mask marks strictly positive
    // inputs only. The mask is materialized only when the caller will attach
    // a gradient function (`store_mask`).
    let out = unary_map(
        input_data,
        |v: f64| if v.is_nan() || v > 0.0 { v } else { 0.0 },
    );
    let mask = store_mask.then(|| unary_map(input_data, |v: f64| v > 0.0));
    Ok((
        TensorData::from_vec::<f64>(out, DataType::Float64, tensor.device()),
        mask,
    ))
}

pub(crate) fn relu_i32(
    tensor: &Tensor,
    store_mask: bool,
) -> Result<(TensorData, Option<Vec<bool>>)> {
    let input_data = tensor.data().as_i32_slice().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get i32 slice from input tensor")
    })?;

    let out = unary_map(input_data, |v: i32| if v > 0 { v } else { 0 });
    let mask = store_mask.then(|| unary_map(input_data, |v: i32| v > 0));
    Ok((
        TensorData::from_vec::<i32>(out, DataType::Int32, tensor.device()),
        mask,
    ))
}

pub(crate) fn relu_i64(
    tensor: &Tensor,
    store_mask: bool,
) -> Result<(TensorData, Option<Vec<bool>>)> {
    let input_data = tensor.data().as_i64_slice().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get i64 slice from input tensor")
    })?;

    let out = unary_map(input_data, |v: i64| if v > 0 { v } else { 0 });
    let mask = store_mask.then(|| unary_map(input_data, |v: i64| v > 0));
    Ok((
        TensorData::from_vec::<i64>(out, DataType::Int64, tensor.device()),
        mask,
    ))
}

pub(crate) fn hardshrink_f32(
    tensor: &Tensor,
    lambd: f32,
    store_mask: bool,
) -> Result<(TensorData, Option<Vec<bool>>)> {
    let input_data = tensor.data().as_f32_slice().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get f32 slice from input tensor")
    })?;

    // Phrase the dead-zone test as `-lambd <= v <= lambd` (PyTorch's form)
    // rather than its finite-value complement `v > lambd || v < -lambd`. The
    // two agree for every finite input, but for NaN the complement is false on
    // both sides and would zero the NaN; testing the dead zone leaves NaN in
    // the `else` branch so it passes through, matching PyTorch and the rest of
    // minitensor's elementwise ops.
    let out = unary_map(
        input_data,
        |v: f32| {
            if v >= -lambd && v <= lambd { 0.0 } else { v }
        },
    );
    let mask = store_mask.then(|| unary_map(input_data, |v: f32| !(v >= -lambd && v <= lambd)));
    Ok((
        TensorData::from_vec::<f32>(out, DataType::Float32, tensor.device()),
        mask,
    ))
}

pub(crate) fn hardshrink_f64(
    tensor: &Tensor,
    lambd: f64,
    store_mask: bool,
) -> Result<(TensorData, Option<Vec<bool>>)> {
    let input_data = tensor.data().as_f64_slice().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get f64 slice from input tensor")
    })?;

    // See `hardshrink_f32`: test the dead zone directly so a NaN input passes
    // through instead of being zeroed by the finite-value complement.
    let out = unary_map(
        input_data,
        |v: f64| {
            if v >= -lambd && v <= lambd { 0.0 } else { v }
        },
    );
    let mask = store_mask.then(|| unary_map(input_data, |v: f64| !(v >= -lambd && v <= lambd)));
    Ok((
        TensorData::from_vec::<f64>(out, DataType::Float64, tensor.device()),
        mask,
    ))
}

pub(crate) fn leaky_relu_f32(
    tensor: &Tensor,
    negative_slope: f32,
    store_mask: bool,
) -> Result<(TensorData, Option<Vec<bool>>)> {
    let input_data = tensor.data().as_f32_slice().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get f32 slice from input tensor")
    })?;

    // Safe chunked maps replace the previous raw-pointer parallel loop; the
    // backward mask marks non-negative inputs and is only materialized when a
    // gradient function will consume it.
    let out = unary_map(
        input_data,
        move |v: f32| {
            if v >= 0.0 { v } else { negative_slope * v }
        },
    );
    let mask = store_mask.then(|| unary_map(input_data, |v: f32| v >= 0.0));
    Ok((
        TensorData::from_vec::<f32>(out, DataType::Float32, tensor.device()),
        mask,
    ))
}

pub(crate) fn leaky_relu_f64(
    tensor: &Tensor,
    negative_slope: f64,
    store_mask: bool,
) -> Result<(TensorData, Option<Vec<bool>>)> {
    let input_data = tensor.data().as_f64_slice().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get f64 slice from input tensor")
    })?;

    // Safe chunked maps replace the previous raw-pointer parallel loop; the
    // backward mask marks non-negative inputs and is only materialized when a
    // gradient function will consume it.
    let out = unary_map(
        input_data,
        move |v: f64| {
            if v >= 0.0 { v } else { negative_slope * v }
        },
    );
    let mask = store_mask.then(|| unary_map(input_data, |v: f64| v >= 0.0));
    Ok((
        TensorData::from_vec::<f64>(out, DataType::Float64, tensor.device()),
        mask,
    ))
}

/// Column-wise softmax of a `[dim_size, after]` row-major block (`after > 1`).
///
/// The softmax dimension is the outer (row) index. Processing the block one
/// contiguous row at a time with `after`-sized max/sum accumulators makes every
/// memory access sequential, unlike the naive per-column loop which strides by
/// `after` on every element. Numerically identical to the strided version: the
/// per-column max is order-independent and the per-column sum accumulates rows
/// in the same order.
fn softmax_block_columnwise_f32(
    in_block: &[f32],
    out_block: &mut [f32],
    dim_size: usize,
    after: usize,
) {
    let mut col_max = vec![f32::NEG_INFINITY; after];
    for k in 0..dim_size {
        let row = &in_block[k * after..k * after + after];
        for (m, &v) in col_max.iter_mut().zip(row) {
            if v > *m {
                *m = v;
            }
        }
    }
    let mut col_sum = vec![0.0f32; after];
    for k in 0..dim_size {
        let in_row = &in_block[k * after..k * after + after];
        let out_row = &mut out_block[k * after..k * after + after];
        for a in 0..after {
            let m = col_max[a];
            // A column whose max is -inf is all -inf (or empty); emit 0, matching
            // the strided path's negative-infinity short-circuit.
            let e = if m == f32::NEG_INFINITY {
                0.0
            } else {
                (in_row[a] - m).exp()
            };
            out_row[a] = e;
            col_sum[a] += e;
        }
    }
    for k in 0..dim_size {
        let out_row = &mut out_block[k * after..k * after + after];
        for (o, &s) in out_row.iter_mut().zip(col_sum.iter()) {
            if s > 0.0 {
                *o /= s;
            }
        }
    }
}

/// f64 counterpart of [`softmax_block_columnwise_f32`].
fn softmax_block_columnwise_f64(
    in_block: &[f64],
    out_block: &mut [f64],
    dim_size: usize,
    after: usize,
) {
    let mut col_max = vec![f64::NEG_INFINITY; after];
    for k in 0..dim_size {
        let row = &in_block[k * after..k * after + after];
        for (m, &v) in col_max.iter_mut().zip(row) {
            if v > *m {
                *m = v;
            }
        }
    }
    let mut col_sum = vec![0.0f64; after];
    for k in 0..dim_size {
        let in_row = &in_block[k * after..k * after + after];
        let out_row = &mut out_block[k * after..k * after + after];
        for a in 0..after {
            let m = col_max[a];
            let e = if m == f64::NEG_INFINITY {
                0.0
            } else {
                (in_row[a] - m).exp()
            };
            out_row[a] = e;
            col_sum[a] += e;
        }
    }
    for k in 0..dim_size {
        let out_row = &mut out_block[k * after..k * after + after];
        for (o, &s) in out_row.iter_mut().zip(col_sum.iter()) {
            if s > 0.0 {
                *o /= s;
            }
        }
    }
}

pub(crate) fn softmax_f32(tensor: &Tensor, output_data: &mut TensorData, dim: usize) -> Result<()> {
    let input_data = tensor.data().as_f32_slice().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get f32 slice from input tensor")
    })?;

    let output_slice = output_data.as_f32_slice_mut().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get mutable f32 slice from output data")
    })?;

    let dims = tensor.shape().dims();
    let dim_size = dims[dim];

    if dim_size == 0 {
        return Ok(());
    }

    // Compute the number of groups before and after the softmax dimension. This
    // allows us to iterate over all slices along `dim` for tensors of arbitrary
    // rank using row-major indexing.
    let after: usize = if dim + 1 >= dims.len() {
        1
    } else {
        dims[dim + 1..].iter().product()
    };
    let group = dim_size * after;
    input_data
        .par_chunks(group)
        .zip(output_slice.par_chunks_mut(group))
        .for_each(|(in_block, out_block)| {
            if after == 1 {
                // Softmax over the last (contiguous) dimension: each block is a
                // single slice laid out contiguously.
                let mut max_val = f32::NEG_INFINITY;
                for &v in in_block.iter() {
                    max_val = max_val.max(v);
                }
                if max_val == f32::NEG_INFINITY {
                    out_block.fill(0.0);
                    return;
                }
                let mut sum = 0.0f32;
                for (o, &v) in out_block.iter_mut().zip(in_block.iter()) {
                    let e = (v - max_val).exp();
                    *o = e;
                    sum += e;
                }
                for o in out_block.iter_mut() {
                    *o /= sum;
                }
            } else {
                // Softmax over a non-last dimension: the block is a
                // `[dim_size, after]` row-major matrix and the reduction runs
                // down the rows. Using `after`-sized column accumulators keeps
                // every pass contiguous (cache-friendly) instead of striding by
                // `after` per element.
                softmax_block_columnwise_f32(in_block, out_block, dim_size, after);
            }
        });

    Ok(())
}

pub(crate) fn softmax_f64(tensor: &Tensor, output_data: &mut TensorData, dim: usize) -> Result<()> {
    let input_data = tensor.data().as_f64_slice().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get f64 slice from input tensor")
    })?;

    let output_slice = output_data.as_f64_slice_mut().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get mutable f64 slice from output data")
    })?;

    let dims = tensor.shape().dims();
    let dim_size = dims[dim];

    if dim_size == 0 {
        return Ok(());
    }

    let after: usize = if dim + 1 >= dims.len() {
        1
    } else {
        dims[dim + 1..].iter().product()
    };
    let group = dim_size * after;
    input_data
        .par_chunks(group)
        .zip(output_slice.par_chunks_mut(group))
        .for_each(|(in_block, out_block)| {
            if after == 1 {
                let mut max_val = f64::NEG_INFINITY;
                for &v in in_block.iter() {
                    max_val = max_val.max(v);
                }
                if max_val == f64::NEG_INFINITY {
                    out_block.fill(0.0);
                    return;
                }
                let mut sum = 0.0f64;
                for (o, &v) in out_block.iter_mut().zip(in_block.iter()) {
                    let e = (v - max_val).exp();
                    *o = e;
                    sum += e;
                }
                for o in out_block.iter_mut() {
                    *o /= sum;
                }
            } else {
                softmax_block_columnwise_f64(in_block, out_block, dim_size, after);
            }
        });

    Ok(())
}

fn broadcast_mask_index(
    linear_idx: usize,
    output_dims: &[usize],
    output_strides: &[usize],
    mask_dims: &[usize],
    mask_strides: &[usize],
) -> usize {
    if mask_dims.is_empty() {
        return 0;
    }

    let output_ndim = output_dims.len();
    let mask_ndim = mask_dims.len();
    let mut mask_index = 0usize;

    for i in 0..mask_ndim {
        let output_dim_idx = output_ndim - 1 - i;
        let mask_dim_idx = mask_ndim - 1 - i;
        let stride = output_strides[output_dim_idx];
        let coord = linear_idx
            .checked_div(stride)
            .map_or(0, |quotient| quotient % output_dims[output_dim_idx]);
        let mask_dim = mask_dims[mask_dim_idx];
        let mask_coord = if mask_dim == 1 { 0 } else { coord };
        mask_index += mask_coord * mask_strides[mask_dim_idx];
    }

    mask_index
}

pub(crate) fn masked_softmax_f32(
    tensor: &Tensor,
    mask: &Tensor,
    output_data: &mut TensorData,
    dim: usize,
) -> Result<()> {
    let input_data = tensor.data().as_f32_slice().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get f32 slice from input tensor")
    })?;
    let mask_data = mask.data().as_bool_slice().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get bool slice from mask tensor")
    })?;
    let output_slice = output_data.as_f32_slice_mut().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get mutable f32 slice from output data")
    })?;

    let dims = tensor.shape().dims();
    let mask_dims = mask.shape().dims();
    let dim_size = dims[dim];
    if dim_size == 0 {
        return Ok(());
    }

    let after: usize = if dim + 1 >= dims.len() {
        1
    } else {
        dims[dim + 1..].iter().product()
    };
    let group = dim_size * after;
    let same_shape = mask_dims == dims;
    let output_strides = if same_shape {
        None
    } else {
        Some(Strides::from_shape(tensor.shape()))
    };
    let mask_strides = if same_shape {
        None
    } else {
        Some(Strides::from_shape(mask.shape()))
    };

    input_data
        .par_chunks(group)
        .zip(output_slice.par_chunks_mut(group))
        .enumerate()
        .for_each(|(block_idx, (in_block, out_block))| {
            let block_offset = block_idx * group;
            for a in 0..after {
                let base = a;
                let mut max_val = f32::NEG_INFINITY;
                let mut has_unmasked = false;
                for k in 0..dim_size {
                    let idx = base + k * after;
                    let linear_idx = block_offset + idx;
                    let masked = if same_shape {
                        mask_data[linear_idx]
                    } else {
                        let mask_index = broadcast_mask_index(
                            linear_idx,
                            dims,
                            output_strides.as_ref().unwrap().as_slice(),
                            mask_dims,
                            mask_strides.as_ref().unwrap().as_slice(),
                        );
                        mask_data[mask_index]
                    };
                    if !masked {
                        has_unmasked = true;
                        max_val = max_val.max(in_block[idx]);
                    }
                }
                if !has_unmasked {
                    for k in 0..dim_size {
                        let idx = base + k * after;
                        out_block[idx] = 0.0;
                    }
                    continue;
                }
                if max_val.is_infinite() && max_val.is_sign_negative() {
                    for k in 0..dim_size {
                        let idx = base + k * after;
                        out_block[idx] = 0.0;
                    }
                    continue;
                }
                let mut sum = 0.0f32;
                for k in 0..dim_size {
                    let idx = base + k * after;
                    let linear_idx = block_offset + idx;
                    let masked = if same_shape {
                        mask_data[linear_idx]
                    } else {
                        let mask_index = broadcast_mask_index(
                            linear_idx,
                            dims,
                            output_strides.as_ref().unwrap().as_slice(),
                            mask_dims,
                            mask_strides.as_ref().unwrap().as_slice(),
                        );
                        mask_data[mask_index]
                    };
                    if masked {
                        out_block[idx] = 0.0;
                    } else {
                        let val = (in_block[idx] - max_val).exp();
                        out_block[idx] = val;
                        sum += val;
                    }
                }
                if sum != 0.0 {
                    for k in 0..dim_size {
                        let idx = base + k * after;
                        out_block[idx] /= sum;
                    }
                }
            }
        });

    Ok(())
}

pub(crate) fn masked_softmax_f64(
    tensor: &Tensor,
    mask: &Tensor,
    output_data: &mut TensorData,
    dim: usize,
) -> Result<()> {
    let input_data = tensor.data().as_f64_slice().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get f64 slice from input tensor")
    })?;
    let mask_data = mask.data().as_bool_slice().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get bool slice from mask tensor")
    })?;
    let output_slice = output_data.as_f64_slice_mut().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get mutable f64 slice from output data")
    })?;

    let dims = tensor.shape().dims();
    let mask_dims = mask.shape().dims();
    let dim_size = dims[dim];
    if dim_size == 0 {
        return Ok(());
    }

    let after: usize = if dim + 1 >= dims.len() {
        1
    } else {
        dims[dim + 1..].iter().product()
    };
    let group = dim_size * after;
    let same_shape = mask_dims == dims;
    let output_strides = if same_shape {
        None
    } else {
        Some(Strides::from_shape(tensor.shape()))
    };
    let mask_strides = if same_shape {
        None
    } else {
        Some(Strides::from_shape(mask.shape()))
    };

    input_data
        .par_chunks(group)
        .zip(output_slice.par_chunks_mut(group))
        .enumerate()
        .for_each(|(block_idx, (in_block, out_block))| {
            let block_offset = block_idx * group;
            for a in 0..after {
                let base = a;
                let mut max_val = f64::NEG_INFINITY;
                let mut has_unmasked = false;
                for k in 0..dim_size {
                    let idx = base + k * after;
                    let linear_idx = block_offset + idx;
                    let masked = if same_shape {
                        mask_data[linear_idx]
                    } else {
                        let mask_index = broadcast_mask_index(
                            linear_idx,
                            dims,
                            output_strides.as_ref().unwrap().as_slice(),
                            mask_dims,
                            mask_strides.as_ref().unwrap().as_slice(),
                        );
                        mask_data[mask_index]
                    };
                    if !masked {
                        has_unmasked = true;
                        max_val = max_val.max(in_block[idx]);
                    }
                }
                if !has_unmasked {
                    for k in 0..dim_size {
                        let idx = base + k * after;
                        out_block[idx] = 0.0;
                    }
                    continue;
                }
                if max_val.is_infinite() && max_val.is_sign_negative() {
                    for k in 0..dim_size {
                        let idx = base + k * after;
                        out_block[idx] = 0.0;
                    }
                    continue;
                }
                let mut sum = 0.0f64;
                for k in 0..dim_size {
                    let idx = base + k * after;
                    let linear_idx = block_offset + idx;
                    let masked = if same_shape {
                        mask_data[linear_idx]
                    } else {
                        let mask_index = broadcast_mask_index(
                            linear_idx,
                            dims,
                            output_strides.as_ref().unwrap().as_slice(),
                            mask_dims,
                            mask_strides.as_ref().unwrap().as_slice(),
                        );
                        mask_data[mask_index]
                    };
                    if masked {
                        out_block[idx] = 0.0;
                    } else {
                        let val = (in_block[idx] - max_val).exp();
                        out_block[idx] = val;
                        sum += val;
                    }
                }
                if sum != 0.0 {
                    for k in 0..dim_size {
                        let idx = base + k * after;
                        out_block[idx] /= sum;
                    }
                }
            }
        });

    Ok(())
}

fn log_softmax_core<T: Float + Send + Sync>(
    input_data: &[T],
    output_slice: &mut [T],
    dims: &[usize],
    dim: usize,
    neg_inf: T,
) -> Result<()> {
    let dim_size = dims[dim];
    if dim_size == 0 {
        return Ok(());
    }

    let after: usize = if dim + 1 >= dims.len() {
        1
    } else {
        dims[dim + 1..].iter().product()
    };
    let group = dim_size * after;
    input_data
        .par_chunks(group)
        .zip(output_slice.par_chunks_mut(group))
        .for_each(|(in_block, out_block)| {
            if after == 1 {
                // Log-softmax over the last (contiguous) dimension.
                let mut max_val = neg_inf;
                for &v in in_block.iter() {
                    if v > max_val {
                        max_val = v;
                    }
                }
                if max_val == neg_inf {
                    out_block.fill(neg_inf);
                    return;
                }
                let mut sum = T::zero();
                for &v in in_block.iter() {
                    sum = sum + (v - max_val).exp();
                }
                let logsum = sum.ln() + max_val;
                for (o, &v) in out_block.iter_mut().zip(in_block.iter()) {
                    *o = v - logsum;
                }
            } else {
                // Non-last dimension: process the `[dim_size, after]` block
                // column-wise with `after`-sized accumulators so every pass is
                // contiguous instead of striding by `after`.
                let mut col_logsum = vec![neg_inf; after];
                for k in 0..dim_size {
                    let row = &in_block[k * after..k * after + after];
                    for (m, &v) in col_logsum.iter_mut().zip(row) {
                        if v > *m {
                            *m = v;
                        }
                    }
                }
                let mut col_sum = vec![T::zero(); after];
                for k in 0..dim_size {
                    let in_row = &in_block[k * after..k * after + after];
                    for a in 0..after {
                        let m = col_logsum[a];
                        if m != neg_inf {
                            col_sum[a] = col_sum[a] + (in_row[a] - m).exp();
                        }
                    }
                }
                // Fold each column's max into log(sum) + max; -inf columns stay
                // -inf so their outputs are all -inf.
                for a in 0..after {
                    if col_logsum[a] != neg_inf {
                        col_logsum[a] = col_sum[a].ln() + col_logsum[a];
                    }
                }
                for k in 0..dim_size {
                    let in_row = &in_block[k * after..k * after + after];
                    let out_row = &mut out_block[k * after..k * after + after];
                    for a in 0..after {
                        let ls = col_logsum[a];
                        out_row[a] = if ls == neg_inf {
                            neg_inf
                        } else {
                            in_row[a] - ls
                        };
                    }
                }
            }
        });

    Ok(())
}

fn masked_log_softmax_core<T: Float + Send + Sync>(
    input_data: &[T],
    output_slice: &mut [T],
    mask_data: &[bool],
    tensor_shape: &Shape,
    mask_shape: &Shape,
    dim: usize,
    neg_inf: T,
) -> Result<()> {
    let dims = tensor_shape.dims();
    let mask_dims = mask_shape.dims();
    let dim_size = dims[dim];
    if dim_size == 0 {
        return Ok(());
    }

    let after: usize = if dim + 1 >= dims.len() {
        1
    } else {
        dims[dim + 1..].iter().product()
    };
    let group = dim_size * after;
    let same_shape = mask_dims == dims;

    if same_shape {
        input_data
            .par_chunks(group)
            .zip(output_slice.par_chunks_mut(group))
            .enumerate()
            .for_each(|(block_idx, (in_block, out_block))| {
                let block_offset = block_idx * group;
                for a in 0..after {
                    let base = a;
                    let mut max_val = neg_inf;
                    let mut has_unmasked = false;
                    for k in 0..dim_size {
                        let idx = base + k * after;
                        let linear_idx = block_offset + idx;
                        if !mask_data[linear_idx] {
                            has_unmasked = true;
                            let val = in_block[idx];
                            if val > max_val {
                                max_val = val;
                            }
                        }
                    }
                    if !has_unmasked || (max_val.is_infinite() && max_val.is_sign_negative()) {
                        for k in 0..dim_size {
                            let idx = base + k * after;
                            out_block[idx] = neg_inf;
                        }
                        continue;
                    }
                    let mut sum = T::zero();
                    for k in 0..dim_size {
                        let idx = base + k * after;
                        let linear_idx = block_offset + idx;
                        if !mask_data[linear_idx] {
                            sum = sum + (in_block[idx] - max_val).exp();
                        }
                    }
                    let logsum = sum.ln() + max_val;
                    for k in 0..dim_size {
                        let idx = base + k * after;
                        let linear_idx = block_offset + idx;
                        if mask_data[linear_idx] {
                            out_block[idx] = neg_inf;
                        } else {
                            out_block[idx] = in_block[idx] - logsum;
                        }
                    }
                }
            });
        return Ok(());
    }

    let output_strides = Strides::from_shape(tensor_shape);
    let mask_strides = Strides::from_shape(mask_shape);
    let output_stride_slice = output_strides.as_slice();
    let mask_stride_slice = mask_strides.as_slice();

    input_data
        .par_chunks(group)
        .zip(output_slice.par_chunks_mut(group))
        .enumerate()
        .for_each(|(block_idx, (in_block, out_block))| {
            let block_offset = block_idx * group;
            for a in 0..after {
                let base = a;
                let mut max_val = neg_inf;
                let mut has_unmasked = false;
                for k in 0..dim_size {
                    let idx = base + k * after;
                    let linear_idx = block_offset + idx;
                    let mask_index = broadcast_mask_index(
                        linear_idx,
                        dims,
                        output_stride_slice,
                        mask_dims,
                        mask_stride_slice,
                    );
                    if !mask_data[mask_index] {
                        has_unmasked = true;
                        let val = in_block[idx];
                        if val > max_val {
                            max_val = val;
                        }
                    }
                }
                if !has_unmasked || (max_val.is_infinite() && max_val.is_sign_negative()) {
                    for k in 0..dim_size {
                        let idx = base + k * after;
                        out_block[idx] = neg_inf;
                    }
                    continue;
                }
                let mut sum = T::zero();
                for k in 0..dim_size {
                    let idx = base + k * after;
                    let linear_idx = block_offset + idx;
                    let mask_index = broadcast_mask_index(
                        linear_idx,
                        dims,
                        output_stride_slice,
                        mask_dims,
                        mask_stride_slice,
                    );
                    if !mask_data[mask_index] {
                        sum = sum + (in_block[idx] - max_val).exp();
                    }
                }
                let logsum = sum.ln() + max_val;
                for k in 0..dim_size {
                    let idx = base + k * after;
                    let linear_idx = block_offset + idx;
                    let mask_index = broadcast_mask_index(
                        linear_idx,
                        dims,
                        output_stride_slice,
                        mask_dims,
                        mask_stride_slice,
                    );
                    if mask_data[mask_index] {
                        out_block[idx] = neg_inf;
                    } else {
                        out_block[idx] = in_block[idx] - logsum;
                    }
                }
            }
        });

    Ok(())
}

macro_rules! masked_log_softmax_impl {
    (
        $tensor:expr,
        $mask:expr,
        $output_data:expr,
        $dim:expr,
        $input_ty:ty,
        $as_input:ident,
        $as_output:ident,
        $neg_inf:expr
    ) => {{
        let input_data = $tensor.data().$as_input().ok_or_else(|| {
            MinitensorError::internal_error("Failed to get input slice from tensor")
        })?;
        let mask_data = $mask.data().as_bool_slice().ok_or_else(|| {
            MinitensorError::internal_error("Failed to get bool slice from mask tensor")
        })?;
        let output_slice = $output_data.$as_output().ok_or_else(|| {
            MinitensorError::internal_error("Failed to get mutable output slice from data")
        })?;

        masked_log_softmax_core(
            input_data,
            output_slice,
            mask_data,
            $tensor.shape(),
            $mask.shape(),
            $dim,
            $neg_inf,
        )
    }};
}

macro_rules! log_softmax_impl {
    (
        $tensor:expr,
        $output_data:expr,
        $dim:expr,
        $input_ty:ty,
        $as_input:ident,
        $as_output:ident,
        $neg_inf:expr
    ) => {{
        let input_data = $tensor.data().$as_input().ok_or_else(|| {
            MinitensorError::internal_error("Failed to get input slice from tensor")
        })?;
        let output_slice = $output_data.$as_output().ok_or_else(|| {
            MinitensorError::internal_error("Failed to get mutable output slice from data")
        })?;

        let dims = $tensor.shape().dims();
        log_softmax_core(input_data, output_slice, dims, $dim, $neg_inf)
    }};
}

pub(crate) fn masked_log_softmax_f32(
    tensor: &Tensor,
    mask: &Tensor,
    output_data: &mut TensorData,
    dim: usize,
) -> Result<()> {
    masked_log_softmax_impl!(
        tensor,
        mask,
        output_data,
        dim,
        f32,
        as_f32_slice,
        as_f32_slice_mut,
        f32::NEG_INFINITY
    )
}

pub(crate) fn masked_log_softmax_f64(
    tensor: &Tensor,
    mask: &Tensor,
    output_data: &mut TensorData,
    dim: usize,
) -> Result<()> {
    masked_log_softmax_impl!(
        tensor,
        mask,
        output_data,
        dim,
        f64,
        as_f64_slice,
        as_f64_slice_mut,
        f64::NEG_INFINITY
    )
}

pub(crate) fn log_softmax_f32(
    tensor: &Tensor,
    output_data: &mut TensorData,
    dim: usize,
) -> Result<()> {
    log_softmax_impl!(
        tensor,
        output_data,
        dim,
        f32,
        as_f32_slice,
        as_f32_slice_mut,
        f32::NEG_INFINITY
    )
}

pub(crate) fn log_softmax_f64(
    tensor: &Tensor,
    output_data: &mut TensorData,
    dim: usize,
) -> Result<()> {
    log_softmax_impl!(
        tensor,
        output_data,
        dim,
        f64,
        as_f64_slice,
        as_f64_slice_mut,
        f64::NEG_INFINITY
    )
}
