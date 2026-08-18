// Copyright (c) Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

use super::*;
use crate::error::MinitensorError;
use crate::error::Result;
use crate::ops::map::par_out_chunks;
use crate::ops::util::{
    accurate_indexed_sum, accurate_slab_sum, broadcast_mask_index, pairwise_fold_vectors,
    slab_blocks, stable_sigmoid_f64,
};
use crate::tensor::DataType;
use crate::tensor::Shape;
use crate::tensor::Strides;
use crate::tensor::Tensor;
use crate::tensor::TensorData;

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

    // Vectorized, and bit-for-bit what `tanh_promoted_f32` produced -- see
    // `ops::simd::transcendental`. Dispatch is resolved once here rather than
    // per block.
    let kernel = crate::ops::simd::F32Kernel::select();
    // SAFETY: `apply` writes every element of each block it is given.
    let out = unsafe {
        unary_map_blocks_threshold(input_data, VECTOR_F32_PAR_THRESHOLD, |src, dst| {
            kernel.tanh(src, dst)
        })
    };
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

    let out = unary_map_threshold(input_data, EXPENSIVE_PAR_THRESHOLD, f64::tanh);
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

    // Vectorized -- see `ops::simd::transcendental`. `e/(e+1)` rather than the
    // branchy stable form, which needed a sign test per element.
    let kernel = crate::ops::simd::F32Kernel::select();
    // SAFETY: `sigmoid` writes every element of each block it is given.
    let out = unsafe {
        unary_map_blocks_threshold(input_data, VECTOR_F32_PAR_THRESHOLD, |src, dst| {
            kernel.sigmoid(src, dst)
        })
    };
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

    let out = unary_map_threshold(input_data, EXPENSIVE_PAR_THRESHOLD, stable_sigmoid_f64);
    Ok(TensorData::from_vec::<f64>(
        out,
        DataType::Float64,
        tensor.device(),
    ))
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

    // Phrase the dead-zone test as `-lambd <= v <= lambd` rather than its
    // finite-value complement `v > lambd || v < -lambd`. The two agree for
    // every finite input, but for NaN the complement is false on both sides
    // and would zero the NaN; testing the dead zone leaves NaN in the `else`
    // branch so it passes through, matching the rest of minitensor's
    // elementwise ops.
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
    // backward mask marks strictly positive inputs and is only materialized
    // when a gradient function will consume it.
    //
    // `> 0`, not `>= 0`: at exactly zero the derivative is the negative
    // slope, as in `relu_f32` right above, which has always
    // used the strict comparison. The forward is unaffected -- both branches
    // give zero at zero -- so only the gradient at the kink moves.
    let out = unary_map(
        input_data,
        move |v: f32| {
            if v > 0.0 { v } else { negative_slope * v }
        },
    );
    let mask = store_mask.then(|| unary_map(input_data, |v: f32| v > 0.0));
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
    // backward mask marks strictly positive inputs and is only materialized
    // when a gradient function will consume it.
    //
    // `> 0`, not `>= 0`: at exactly zero the derivative is the negative
    // slope, as in `relu_f64` right above, which has always
    // used the strict comparison. The forward is unaffected -- both branches
    // give zero at zero -- so only the gradient at the kink moves.
    let out = unary_map(
        input_data,
        move |v: f64| {
            if v > 0.0 { v } else { negative_slope * v }
        },
    );
    let mask = store_mask.then(|| unary_map(input_data, |v: f64| v > 0.0));
    Ok((
        TensorData::from_vec::<f64>(out, DataType::Float64, tensor.device()),
        mask,
    ))
}

/// Geometry shared by the softmax-family forward kernels: the size of the
/// reduced dimension, the number of trailing elements per slice (`after`), and
/// the size of one contiguous block spanning the reduced dimension.
///
/// `None` means the reduced dimension is empty and there is nothing to write.
fn softmax_geometry(dims: &[usize], dim: usize) -> Option<(usize, usize, usize)> {
    let dim_size = dims[dim];
    if dim_size == 0 {
        return None;
    }
    let after: usize = if dim + 1 >= dims.len() {
        1
    } else {
        dims[dim + 1..].iter().product()
    };
    // An empty axis *after* the reduced one makes the group size zero, and the
    // four callers all feed that straight to `par_chunks`, which panics on a
    // chunk size of zero rather than returning an error. It was reachable from
    // Python -- `softmax((3, 0), dim=0)` -- so the panic crossed the binding,
    // where it carries no useful message. The tensor is empty whenever this
    // holds, exactly as when `dim_size` is zero, so there is nothing to compute
    // either way. A zero *before* the reduced axis is already harmless: it
    // makes the input slice empty, and `par_chunks` yields no chunks.
    if after == 0 {
        return None;
    }
    Some((dim_size, after, dim_size * after))
}

/// The output/mask stride pair needed to map output positions onto a broadcast
/// mask, or `None` when the shapes already agree and the index is direct.
fn mask_strides_for(tensor_shape: &Shape, mask_shape: &Shape) -> Option<(Strides, Strides)> {
    if mask_shape.dims() == tensor_shape.dims() {
        None
    } else {
        Some((
            Strides::from_shape(tensor_shape),
            Strides::from_shape(mask_shape),
        ))
    }
}

/// Column-wise softmax of a `[dim_size, after]` row-major block (`after > 1`).
///
/// The softmax dimension is the outer (row) index. Processing the block one
/// contiguous row at a time with `after`-sized max/sum accumulators makes every
/// memory access sequential, unlike the naive per-column loop which strides by
/// `after` on every element. The per-column max is order-independent, so it is
/// exactly the strided version's; the per-column sums walk the rows in the same
/// order but add them up in blocks (see [`slab_blocks`]), which is the only
/// reason this is more accurate than a strided walk rather than identical to it.
fn softmax_block_columnwise<T: Float>(
    in_block: &[T],
    out_block: &mut [T],
    dim_size: usize,
    after: usize,
) {
    let neg_inf = T::neg_infinity();
    let mut col_max = vec![neg_inf; after];
    for k in 0..dim_size {
        let row = &in_block[k * after..k * after + after];
        for (m, &v) in col_max.iter_mut().zip(row) {
            if v > *m {
                *m = v;
            }
        }
    }
    // The column sums are the slab form of the same accuracy problem, and they
    // are filled in the pass that writes the exponentials: each `exp` is
    // computed once, stored, and added. Splitting the two apart costs the whole
    // kernel again -- recomputing `exp` for the sum ran 64% slower on
    // `softmax(dim=0)` of a 500000x3 tensor, and reading the stored value back
    // in a second pass 26%.
    let mut partials: Vec<Vec<T>> = Vec::new();
    for steps in slab_blocks(dim_size) {
        let mut acc = vec![T::zero(); after];
        for k in steps {
            let in_row = &in_block[k * after..k * after + after];
            let out_row = &mut out_block[k * after..k * after + after];
            for a in 0..after {
                let m = col_max[a];
                // A column whose max is -inf is all -inf (or empty); emit 0,
                // matching the contiguous path's negative-infinity
                // short-circuit, and let the zero drop out of the sum.
                let e = if m == neg_inf {
                    T::zero()
                } else {
                    (in_row[a] - m).exp()
                };
                out_row[a] = e;
                acc[a] = acc[a] + e;
            }
        }
        partials.push(acc);
    }
    if partials.is_empty() {
        return;
    }
    let col_sum = pairwise_fold_vectors(partials, |a, b| a + b);
    for k in 0..dim_size {
        let out_row = &mut out_block[k * after..k * after + after];
        for (o, &s) in out_row.iter_mut().zip(col_sum.iter()) {
            if s > T::zero() {
                *o = *o / s;
            }
        }
    }
}

/// `softmax` along `dim`, shifted by the per-slice max for numerical stability.
fn softmax_core<T: Float + Send + Sync>(
    input_data: &[T],
    output_slice: &mut [T],
    dims: &[usize],
    dim: usize,
) -> Result<()> {
    let Some((dim_size, after, group)) = softmax_geometry(dims, dim) else {
        return Ok(());
    };
    let neg_inf = T::neg_infinity();

    par_out_chunks(output_slice, group, &|block_offset, out_block| {
        let in_block = &input_data[block_offset..block_offset + out_block.len()];
        if after == 1 {
            // Softmax over the last (contiguous) dimension: each block is a
            // single slice laid out contiguously.
            let mut max_val = neg_inf;
            for &v in in_block.iter() {
                if v > max_val {
                    max_val = v;
                }
            }
            if max_val == neg_inf {
                out_block.fill(T::zero());
                return;
            }
            for (o, &v) in out_block.iter_mut().zip(in_block.iter()) {
                *o = (v - max_val).exp();
            }
            // Blocked: a running total over a long axis loses the small terms,
            // and every term here but the largest *is* small. Over a 250k-class
            // vocabulary the probabilities came back summing to 1.0004 rather
            // than 1, at 4.2e-4 relative error against NumPy's 1.0e-7.
            let sum = accurate_indexed_sum(out_block.len(), T::zero(), |k| out_block[k]);
            for o in out_block.iter_mut() {
                *o = *o / sum;
            }
        } else {
            // Softmax over a non-last dimension: the block is a
            // `[dim_size, after]` row-major matrix and the reduction runs
            // down the rows. Column accumulators keep every pass contiguous
            // instead of striding by `after` per element.
            softmax_block_columnwise(in_block, out_block, dim_size, after);
        }
    });

    Ok(())
}

/// `softmax` restricted to the unmasked positions.
///
/// Masked entries take no part in the max or the sum and come out as 0. A slice
/// with no unmasked entry -- or whose unmasked entries are all `-inf` -- is all
/// zeros rather than NaN.
fn masked_softmax_core<T: Float + Send + Sync>(
    input_data: &[T],
    output_slice: &mut [T],
    mask_data: &[bool],
    tensor_shape: &Shape,
    mask_shape: &Shape,
    dim: usize,
) -> Result<()> {
    let dims = tensor_shape.dims();
    let mask_dims = mask_shape.dims();
    let Some((dim_size, after, group)) = softmax_geometry(dims, dim) else {
        return Ok(());
    };
    let neg_inf = T::neg_infinity();

    // Resolving a mask position through one closure is what lets the passes
    // below read the same whether or not the mask is broadcast; spelling the
    // lookup out inline needed six copies of it.
    let strides = mask_strides_for(tensor_shape, mask_shape);
    let is_masked = |linear_idx: usize| match &strides {
        Some((out_strides, m_strides)) => {
            mask_data[broadcast_mask_index(
                linear_idx,
                dims,
                out_strides.as_slice(),
                mask_dims,
                m_strides.as_slice(),
            )]
        }
        None => mask_data[linear_idx],
    };

    par_out_chunks(output_slice, group, &|block_offset, out_block| {
        let in_block = &input_data[block_offset..block_offset + out_block.len()];
        for base in 0..after {
            let mut max_val = neg_inf;
            let mut has_unmasked = false;
            for k in 0..dim_size {
                let idx = base + k * after;
                if !is_masked(block_offset + idx) {
                    has_unmasked = true;
                    let v = in_block[idx];
                    if v > max_val {
                        max_val = v;
                    }
                }
            }
            if !has_unmasked || max_val == neg_inf {
                for k in 0..dim_size {
                    out_block[base + k * after] = T::zero();
                }
                continue;
            }
            for k in 0..dim_size {
                let idx = base + k * after;
                out_block[idx] = if is_masked(block_offset + idx) {
                    T::zero()
                } else {
                    (in_block[idx] - max_val).exp()
                };
            }
            let sum = accurate_indexed_sum(dim_size, T::zero(), |k| out_block[base + k * after]);
            if sum != T::zero() {
                for k in 0..dim_size {
                    let idx = base + k * after;
                    out_block[idx] = out_block[idx] / sum;
                }
            }
        }
    });

    Ok(())
}

macro_rules! softmax_entry {
    ($name:ident, $core:ident, $as_input:ident, $as_output:ident) => {
        pub(crate) fn $name(
            tensor: &Tensor,
            output_data: &mut TensorData,
            dim: usize,
        ) -> Result<()> {
            let input_data = tensor.data().$as_input().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get input slice from tensor")
            })?;
            let output_slice = output_data.$as_output().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get mutable output slice from data")
            })?;
            $core(input_data, output_slice, tensor.shape().dims(), dim)
        }
    };
}

macro_rules! masked_softmax_entry {
    ($name:ident, $core:ident, $as_input:ident, $as_output:ident) => {
        pub(crate) fn $name(
            tensor: &Tensor,
            mask: &Tensor,
            output_data: &mut TensorData,
            dim: usize,
        ) -> Result<()> {
            let input_data = tensor.data().$as_input().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get input slice from tensor")
            })?;
            let mask_data = mask.data().as_bool_slice().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get bool slice from mask tensor")
            })?;
            let output_slice = output_data.$as_output().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get mutable output slice from data")
            })?;
            $core(
                input_data,
                output_slice,
                mask_data,
                tensor.shape(),
                mask.shape(),
                dim,
            )
        }
    };
}

softmax_entry!(softmax_f32, softmax_core, as_f32_slice, as_f32_slice_mut);
softmax_entry!(softmax_f64, softmax_core, as_f64_slice, as_f64_slice_mut);
masked_softmax_entry!(
    masked_softmax_f32,
    masked_softmax_core,
    as_f32_slice,
    as_f32_slice_mut
);
masked_softmax_entry!(
    masked_softmax_f64,
    masked_softmax_core,
    as_f64_slice,
    as_f64_slice_mut
);

/// `log_softmax` along `dim`, via the shifted log-sum-exp so the exponentials
/// cannot overflow.
fn log_softmax_core<T: Float + Send + Sync>(
    input_data: &[T],
    output_slice: &mut [T],
    dims: &[usize],
    dim: usize,
) -> Result<()> {
    let Some((dim_size, after, group)) = softmax_geometry(dims, dim) else {
        return Ok(());
    };
    let neg_inf = T::neg_infinity();
    par_out_chunks(output_slice, group, &|block_offset, out_block| {
        let in_block = &input_data[block_offset..block_offset + out_block.len()];
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
            let sum =
                accurate_indexed_sum(in_block.len(), T::zero(), |k| (in_block[k] - max_val).exp());
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
            let col_sum = accurate_slab_sum(dim_size, after, T::zero(), |k, acc: &mut [T]| {
                let in_row = &in_block[k * after..k * after + after];
                for a in 0..after {
                    let m = col_logsum[a];
                    if m != neg_inf {
                        acc[a] = acc[a] + (in_row[a] - m).exp();
                    }
                }
            });
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

/// `log_softmax` restricted to the unmasked positions.
///
/// Masked entries take no part in the max or the log-sum and come out as
/// `-inf`, as does every entry of a slice with no unmasked value (or whose
/// unmasked values are all `-inf`).
fn masked_log_softmax_core<T: Float + Send + Sync>(
    input_data: &[T],
    output_slice: &mut [T],
    mask_data: &[bool],
    tensor_shape: &Shape,
    mask_shape: &Shape,
    dim: usize,
) -> Result<()> {
    let dims = tensor_shape.dims();
    let mask_dims = mask_shape.dims();
    let Some((dim_size, after, group)) = softmax_geometry(dims, dim) else {
        return Ok(());
    };
    let neg_inf = T::neg_infinity();

    // One closure for the mask lookup, so the three passes below read the same
    // whether or not the mask is broadcast. Writing the branch at the top level
    // instead meant two copies of the whole kernel, each with three inlined
    // copies of this lookup.
    let strides = mask_strides_for(tensor_shape, mask_shape);
    let is_masked = |linear_idx: usize| match &strides {
        Some((out_strides, m_strides)) => {
            mask_data[broadcast_mask_index(
                linear_idx,
                dims,
                out_strides.as_slice(),
                mask_dims,
                m_strides.as_slice(),
            )]
        }
        None => mask_data[linear_idx],
    };

    par_out_chunks(output_slice, group, &|block_offset, out_block| {
        let in_block = &input_data[block_offset..block_offset + out_block.len()];
        for base in 0..after {
            let mut max_val = neg_inf;
            let mut has_unmasked = false;
            for k in 0..dim_size {
                let idx = base + k * after;
                if !is_masked(block_offset + idx) {
                    has_unmasked = true;
                    let v = in_block[idx];
                    if v > max_val {
                        max_val = v;
                    }
                }
            }
            if !has_unmasked || max_val == neg_inf {
                for k in 0..dim_size {
                    out_block[base + k * after] = neg_inf;
                }
                continue;
            }
            let sum = accurate_indexed_sum(dim_size, T::zero(), |k| {
                let idx = base + k * after;
                if is_masked(block_offset + idx) {
                    T::zero()
                } else {
                    (in_block[idx] - max_val).exp()
                }
            });
            let logsum = sum.ln() + max_val;
            for k in 0..dim_size {
                let idx = base + k * after;
                out_block[idx] = if is_masked(block_offset + idx) {
                    neg_inf
                } else {
                    in_block[idx] - logsum
                };
            }
        }
    });

    Ok(())
}

softmax_entry!(
    log_softmax_f32,
    log_softmax_core,
    as_f32_slice,
    as_f32_slice_mut
);
softmax_entry!(
    log_softmax_f64,
    log_softmax_core,
    as_f64_slice,
    as_f64_slice_mut
);
masked_softmax_entry!(
    masked_log_softmax_f32,
    masked_log_softmax_core,
    as_f32_slice,
    as_f32_slice_mut
);
masked_softmax_entry!(
    masked_log_softmax_f64,
    masked_log_softmax_core,
    as_f64_slice,
    as_f64_slice_mut
);
