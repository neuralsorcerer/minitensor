// Copyright (c) Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

use super::*;
use crate::ops::map::{outputs_per_task, par_fold_chunks, par_out_chunks, par_out_chunks2};
use crate::ops::shape_ops;
use crate::ops::simd::*;
use crate::ops::util::{
    Accumulate, accumulating_dtype, accurate_run_sum, accurate_slab_sum, deterministic_par_sum,
    pairwise_fold,
};
use crate::{
    error::{MinitensorError, Result},
    tensor::{DataType, Shape, Tensor, TensorData},
};
use rayon::prelude::*;
use std::sync::Arc;

/// Below this many elements a single slice is not worth handing to rayon: the
/// split-and-merge overhead outweighs sorting it on one core.
///
/// Deliberately conservative. Measured on four cores the crossover is somewhere
/// between 4k and 8k elements and the two paths are within noise of each other
/// across that range, where the whole sort costs well under a millisecond
/// either way. Setting it here gives up a little between 8k and 16k in exchange
/// for never regressing the small-slice path, which is the one that runs inside
/// a training loop.
const PAR_SORT_MIN_LEN: usize = 1 << 14;

/// Sort each 1-D slice along a dimension, parallelizing over the outer index.
///
/// `values`/`indices` are partitioned into one disjoint chunk per outer
/// position (`par_chunks_mut`), so the parallel writes never overlap and this
/// stays safe. Each slice gathers `(original_index, value)` pairs, sorts them
/// with `cmp` (so `indices` becomes the argsort), and scatters the result back.
///
/// Only worth calling when there are enough slices to fill the thread pool --
/// see [`sort_rows_with_parallel_sort`] for the other case.
#[allow(clippy::too_many_arguments)]
fn sort_along_dim_par<T, C>(
    input: &[T],
    values: &mut [T],
    indices: &mut [i64],
    outer: usize,
    inner: usize,
    dim_size: usize,
    outer_stride: usize,
    stable: bool,
    cmp: C,
) where
    T: Copy + Send + Sync,
    C: Fn(&(usize, T), &(usize, T)) -> std::cmp::Ordering + Sync + Copy,
{
    debug_assert_eq!(values.len(), outer * outer_stride);
    debug_assert_eq!(indices.len(), outer * outer_stride);
    // Erased at the chunk boundary only: `cmp` stays a concrete type inside the
    // body, so the sort still monomorphizes against it. Erasing the comparator
    // itself would put an indirect call on every comparison.
    par_out_chunks2(values, indices, outer_stride, &|start, vchunk, ichunk| {
        let o = start / outer_stride;
        let mut entries: Vec<(usize, T)> = Vec::with_capacity(dim_size);
        for r in 0..inner {
            entries.clear();
            let base = o * outer_stride + r;
            for d in 0..dim_size {
                entries.push((d, input[base + d * inner]));
            }
            if stable {
                entries.sort_by(cmp);
            } else {
                entries.sort_unstable_by(cmp);
            }
            for (j, (index, value)) in entries.iter().enumerate() {
                let off = r + j * inner;
                vchunk[off] = *value;
                ichunk[off] = *index as i64;
            }
        }
    });
}

/// The same sort, but parallel *within* each slice rather than across them.
///
/// `sort_along_dim_par` splits the work by outer position, which leaves most
/// of the machine idle when there are fewer slices than threads -- and a 1-D
/// tensor has exactly one, so sorting one ran entirely on a single core. The
/// same 2M elements cost 134 ns each as one slice and 16 ns each as 2048 of
/// them, a 8.3x spread on four cores that was pure scheduling.
#[allow(clippy::too_many_arguments)]
fn sort_rows_with_parallel_sort<T, C>(
    input: &[T],
    values: &mut [T],
    indices: &mut [i64],
    outer: usize,
    inner: usize,
    dim_size: usize,
    outer_stride: usize,
    stable: bool,
    cmp: C,
) where
    T: Copy + Send + Sync,
    C: Fn(&(usize, T), &(usize, T)) -> std::cmp::Ordering + Sync + Copy,
{
    let mut entries: Vec<(usize, T)> = Vec::with_capacity(dim_size);
    for o in 0..outer {
        for r in 0..inner {
            entries.clear();
            let base = o * outer_stride + r;
            for d in 0..dim_size {
                entries.push((d, input[base + d * inner]));
            }
            if stable {
                entries.par_sort_by(cmp);
            } else {
                entries.par_sort_unstable_by(cmp);
            }
            for (j, (index, value)) in entries.iter().enumerate() {
                let off = base + j * inner;
                values[off] = *value;
                indices[off] = *index as i64;
            }
        }
    }
}

/// Pick whichever of the two has parallelism to exploit.
///
/// Splitting across slices is cheaper per element when there are enough of
/// them, because each sort stays on one core with no merge step. It only wins
/// when the pool is actually filled, which is what this decides.
#[allow(clippy::too_many_arguments)]
fn sort_along_dim<T, C>(
    input: &[T],
    values: &mut [T],
    indices: &mut [i64],
    outer: usize,
    inner: usize,
    dim_size: usize,
    outer_stride: usize,
    stable: bool,
    cmp: C,
) where
    T: Copy + Send + Sync,
    C: Fn(&(usize, T), &(usize, T)) -> std::cmp::Ordering + Sync + Copy,
{
    let slices = outer.saturating_mul(inner);
    if slices < rayon::current_num_threads() && dim_size >= PAR_SORT_MIN_LEN {
        sort_rows_with_parallel_sort(
            input,
            values,
            indices,
            outer,
            inner,
            dim_size,
            outer_stride,
            stable,
            cmp,
        );
    } else {
        sort_along_dim_par(
            input,
            values,
            indices,
            outer,
            inner,
            dim_size,
            outer_stride,
            stable,
            cmp,
        );
    }
}

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
            Some(d) => return Err(MinitensorError::dim_out_of_range(d, 1)),
            None => 0,
        }
    } else {
        let dim_value = dim.unwrap_or(-1);
        normalize_dim(dim_value, ndim)?
    };

    // Nothing to order, and no storage to order it in: `sort_along_dim_par`
    // chunks by `dim_size * inner`, which is zero here, and `par_chunks_mut(0)`
    // panics. Returning the empty input back is the sensible answer, so do
    // that -- after `normalize_dim` above, so an out-of-range `dim` still
    // errors.
    if tensor.numel() == 0 {
        let values = Tensor::new(
            Arc::new(TensorData::zeros_on_device(
                0,
                tensor.dtype(),
                tensor.device(),
            )),
            tensor.shape().clone(),
            tensor.dtype(),
            tensor.device(),
            tensor.requires_grad(),
        );
        let indices = Tensor::new(
            Arc::new(TensorData::zeros_on_device(
                0,
                DataType::Int64,
                tensor.device(),
            )),
            tensor.shape().clone(),
            DataType::Int64,
            tensor.device(),
            false,
        );
        return Ok((values, indices));
    }

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

    // Dispatch to the parallel kernel with the *ascending* or *descending*
    // comparator passed as a function item (not through an `if`, which would
    // coerce to a non-inlinable function pointer and slow the sort's inner
    // comparisons). Passing the item lets the generic kernel monomorphize and
    // inline the comparator.
    macro_rules! run_sort {
        ($input:expr, $values:expr, $indices:expr, $asc:expr, $desc:expr) => {
            if descending {
                sort_along_dim(
                    $input,
                    $values,
                    $indices,
                    outer,
                    inner,
                    dim_size,
                    outer_stride,
                    stable,
                    $desc,
                );
            } else {
                sort_along_dim(
                    $input,
                    $values,
                    $indices,
                    outer,
                    inner,
                    dim_size,
                    outer_stride,
                    stable,
                    $asc,
                );
            }
        };
    }

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

            run_sort!(input, values, indices, cmp_f32_asc, cmp_f32_desc);
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

            run_sort!(input, values, indices, cmp_f64_asc, cmp_f64_desc);
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

            run_sort!(input, values, indices, cmp_i32_asc, cmp_i32_desc);
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

            run_sort!(input, values, indices, cmp_i64_asc, cmp_i64_desc);
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

            run_sort!(input, values, indices, cmp_bool_asc, cmp_bool_desc);
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

    let dims = normalize_reduction_dims(dim, tensor.ndim())?;

    if matches!(dims, Some(ref dims) if dims.is_empty()) {
        return Ok(tensor.clone());
    }

    let reduction_dims: Vec<usize> = dims.clone().unwrap_or_else(|| (0..tensor.ndim()).collect());

    // Fused fast path: single-axis variance for tensors that don't require
    // gradients. Computes mean and the sum of squared deviations in two
    // cache-friendly passes, avoiding the full-size difference/square
    // intermediates that the autograd composition below materializes. The
    // gradient path is left entirely to that composition.
    if !tensor.requires_grad()
        && reduction_dims.len() == 1
        && tensor.shape().dims()[reduction_dims[0]] >= 1
    {
        return var_fused_single_axis(tensor, reduction_dims[0], keepdim, unbiased);
    }

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
        // A single sample makes Bessel's correction `n / (n - 1)` undefined, and
        // the biased variance it scales is exactly zero, so the product is NaN
        // -- which is the honest answer for an undefined correction.
        //
        // That case goes through the same multiply as any other correction
        // rather than substituting a freshly built NaN tensor. A replacement
        // carries a new tensor id, no `grad_fn` and no graph node, so while it
        // inherited `requires_grad` it had nothing behind it: `x.var(1)` on a
        // width-1 axis reported `requires_grad = true` and then left `x` with no
        // gradient at all after `backward()`. A missing gradient reads as "this
        // parameter was not used" and an optimizer skips it silently, where the
        // NaN this now produces says plainly that something is undefined.
        let correction = if sample_count <= 1 {
            f64::NAN
        } else {
            sample_count as f64 / (sample_count - 1) as f64
        };
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

/// Rows one task takes when the reduced axis is the last one. A row is
/// `dim_size` elements read twice, so a band of these is already substantial
/// work; the point of banding at all is that one row per task is not.
const VAR_ROW_BAND: usize = 64;

/// Fused single-axis variance for tensors that do not require gradients.
///
/// Two cache-friendly slab passes per outer block (mean, then sum of squared
/// deviations), parallel over the outer index. Numerically matches the autograd
/// composition (`mean` -> `x - mean` -> square -> `mean` -> Bessel): biased
/// variance is `sum_sq_dev / n`, unbiased is `sum_sq_dev / (n - 1)` (and NaN
/// when `n <= 1`).
fn var_fused_single_axis(
    tensor: &Tensor,
    axis: usize,
    keepdim: bool,
    unbiased: bool,
) -> Result<Tensor> {
    let dims = tensor.shape().dims();
    let dim_size = dims[axis];
    let inner: usize = dims[axis + 1..].iter().product();
    let outer: usize = dims[..axis].iter().product();
    let outer_stride = dim_size * inner;
    let out_numel = outer * inner;

    let mut result_data = TensorData::zeros_on_device(out_numel, tensor.dtype(), tensor.device());

    macro_rules! fill {
        ($accessor:ident, $accessor_mut:ident, $ty:ty) => {{
            let input = tensor
                .data()
                .$accessor()
                .ok_or_else(|| MinitensorError::internal_error("Failed to get slice"))?;
            let out = result_data
                .$accessor_mut()
                .ok_or_else(|| MinitensorError::internal_error("Failed to get mutable slice"))?;
            let n = dim_size as $ty;
            let divisor = if unbiased { n - 1.0 } else { n };
            let all_nan = unbiased && dim_size <= 1;

            if all_nan {
                out.fill(<$ty>::NAN);
            } else if inner == 1 {
                // Reducing the last axis: each output owns one contiguous run
                // of `dim_size` elements, so the two passes read straight
                // through it and the running mean is a scalar.
                //
                // Going through the slab path below instead was what made
                // `var(dim=-1)` cost seventeen times a `mean` over the same
                // data. With `inner == 1` its chunks are one element wide, so
                // rayon was handed one task per output and each of those
                // allocated a one-element `col_mean` on the heap -- a thousand
                // allocations and a thousand tasks to reduce a thousand rows.
                // Both passes go through `accurate_run_sum`, for the reason
                // `sum` does: a running total over a long row accumulates one
                // rounding per element. This is the path taken only when the
                // tensor does *not* require gradients, so leaving it naive made
                // `var` answer differently depending on whether it was being
                // trained through -- and the untrained answer was the worse
                // one, by 38000x at a 4M-element axis (3.8e-3 against 1.2e-7).
                let run = |first: usize, chunk: &mut [$ty]| {
                    for (i, slot) in chunk.iter_mut().enumerate() {
                        let base = (first + i) * dim_size;
                        let row = &input[base..base + dim_size];
                        let total =
                            accurate_run_sum(row, |part: &[$ty]| part.iter().copied().sum::<$ty>());
                        let mean = total / n;
                        let acc = accurate_run_sum(row, |part: &[$ty]| {
                            part.iter()
                                .map(|&v| {
                                    let d = v - mean;
                                    d * d
                                })
                                .sum::<$ty>()
                        });
                        *slot = acc / divisor;
                    }
                };
                if tensor.numel() < crate::ops::map::PAR_THRESHOLD {
                    run(0, out);
                } else {
                    // Rows per task, not elements: a task is `VAR_ROW_BAND`
                    // whole rows of `dim_size` work each.
                    par_out_chunks(out, VAR_ROW_BAND, &run);
                }
            } else if inner != 0 {
                // The reduced axis is not the last one, so each output's
                // elements are `inner` apart. Accumulate whole slabs instead,
                // which reads the input in memory order; the running means are
                // a vector of `inner`, allocated once per outer position.
                par_out_chunks(out, inner, &|start, out_chunk| {
                    let block_base = (start / inner) * outer_stride;
                    // Both passes are blocked for the same reason the
                    // contiguous-row path above uses `accurate_run_sum`.
                    let mut col_mean =
                        accurate_slab_sum(dim_size, inner, 0.0 as $ty, |k, acc: &mut [$ty]| {
                            let base = block_base + k * inner;
                            let slab = &input[base..base + inner];
                            for (m, &v) in acc.iter_mut().zip(slab) {
                                *m += v;
                            }
                        });
                    for m in col_mean.iter_mut() {
                        *m /= n;
                    }
                    let squared =
                        accurate_slab_sum(dim_size, inner, 0.0 as $ty, |k, acc: &mut [$ty]| {
                            let base = block_base + k * inner;
                            let slab = &input[base..base + inner];
                            for ((a, &v), &m) in acc.iter_mut().zip(slab).zip(col_mean.iter()) {
                                let d = v - m;
                                *a += d * d;
                            }
                        });
                    for (acc, &v) in out_chunk.iter_mut().zip(squared.iter()) {
                        *acc = v / divisor;
                    }
                });
            }
        }};
    }

    match tensor.dtype() {
        DataType::Float32 => fill!(as_f32_slice, as_f32_slice_mut, f32),
        DataType::Float64 => fill!(as_f64_slice, as_f64_slice_mut, f64),
        _ => unreachable!("variance is only defined for floating point tensors"),
    }

    let out_shape = if keepdim {
        let mut d = dims.to_vec();
        d[axis] = 1;
        Shape::new(d)
    } else {
        let d: Vec<usize> = dims
            .iter()
            .enumerate()
            .filter_map(|(i, &s)| if i == axis { None } else { Some(s) })
            .collect();
        if d.is_empty() {
            Shape::scalar()
        } else {
            Shape::new(d)
        }
    };

    Ok(Tensor::new(
        Arc::new(result_data),
        out_shape,
        tensor.dtype(),
        tensor.device(),
        false,
    ))
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

    // Reads i32, multiplies in i64 -- see `accumulating_dtype`.
    let prod: i64 = if data.len() >= 1024 {
        par_fold_chunks(data, 8192, 1i64, &simd_prod_i32_to_i64, &|a: i64, b| {
            a.acc_mul(b)
        })
    } else {
        simd_prod_i32_to_i64(data)
    };

    let result_slice = result_data
        .as_i64_slice_mut()
        .ok_or_else(|| MinitensorError::internal_error("Failed to get mutable i64 slice"))?;

    result_slice[0] = prod;
    Ok(())
}

pub(crate) fn prod_all_i64(tensor: &Tensor, result_data: &mut TensorData) -> Result<()> {
    let data = tensor
        .data()
        .as_i64_slice()
        .ok_or_else(|| MinitensorError::internal_error("Failed to get i64 slice"))?;

    let prod: i64 = if data.len() >= 1024 {
        data.par_chunks(8192)
            .map(simd_prod_i64)
            .reduce(|| 1, |a, b| a.acc_mul(b))
    } else {
        simd_prod_i64(data)
    };

    let result_slice = result_data
        .as_i64_slice_mut()
        .ok_or_else(|| MinitensorError::internal_error("Failed to get mutable i64 slice"))?;

    result_slice[0] = prod;
    Ok(())
}

pub(crate) fn sum_all_f32(tensor: &Tensor, result_data: &mut TensorData) -> Result<()> {
    let data = tensor
        .data()
        .as_f32_slice()
        .ok_or_else(|| MinitensorError::internal_error("Failed to get f32 slice"))?;

    let sum: f32 = if data.len() >= 1024 {
        deterministic_par_sum(data, 8192, simd_sum_f32)
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
        deterministic_par_sum(data, 8192, simd_sum_f64)
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

    // Reads i32, totals in i64 -- see `accumulating_dtype`.
    let sum: i64 = if data.len() >= 1024 {
        par_fold_chunks(data, 8192, 0i64, &simd_sum_i32_to_i64, &|a: i64, b| {
            a.acc_add(b)
        })
    } else {
        simd_sum_i32_to_i64(data)
    };

    let result_slice = result_data
        .as_i64_slice_mut()
        .ok_or_else(|| MinitensorError::internal_error("Failed to get mutable i64 slice"))?;

    result_slice[0] = sum;
    Ok(())
}

pub(crate) fn sum_all_i64(tensor: &Tensor, result_data: &mut TensorData) -> Result<()> {
    let data = tensor
        .data()
        .as_i64_slice()
        .ok_or_else(|| MinitensorError::internal_error("Failed to get i64 slice"))?;

    let sum: i64 = if data.len() >= 1024 {
        data.par_chunks(8192)
            .map(simd_sum_i64)
            .reduce(|| 0, |a, b| a.acc_add(b))
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

    let sum: f32 = deterministic_par_sum(data, 8192, simd_nansum_f32);

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

    let sum: f64 = deterministic_par_sum(data, 8192, simd_nansum_f64);

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

    let partials: Vec<(f32, usize)> = data
        .par_chunks(8192)
        .map(|chunk| {
            chunk.iter().fold((0.0_f32, 0usize), |(s, c), &v| {
                if v.is_nan() { (s, c) } else { (s + v, c + 1) }
            })
        })
        .collect();
    let (sum, count) = pairwise_fold(partials, (0.0_f32, 0usize), |(s1, c1), (s2, c2)| {
        (s1 + s2, c1 + c2)
    });

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

    let partials: Vec<(f64, usize)> = data
        .par_chunks(8192)
        .map(|chunk| {
            chunk.iter().fold((0.0_f64, 0usize), |(s, c), &v| {
                if v.is_nan() { (s, c) } else { (s + v, c + 1) }
            })
        })
        .collect();
    let (sum, count) = pairwise_fold(partials, (0.0_f64, 0usize), |(s1, c1), (s2, c2)| {
        (s1 + s2, c1 + c2)
    });

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

    // `count == 0` means every element along the axis was NaN, so there is no
    // mean to report and NaN is the answer rather than a division by zero.
    macro_rules! divide {
        ($accessor:ident, $accessor_mut:ident, $ty:ty, $tyname:literal) => {{
            let missing =
                || MinitensorError::internal_error(concat!("Failed to get ", $tyname, " slice"));
            let sum_slice = sum.data().$accessor().ok_or_else(missing)?;
            let count_slice = count.data().$accessor().ok_or_else(missing)?;
            let out = result_data.$accessor_mut().ok_or_else(missing)?;
            par_out_chunks(out, outputs_per_task(1), &|start, chunk| {
                let span = start..start + chunk.len();
                for ((dst, &s), &c) in chunk
                    .iter_mut()
                    .zip(&sum_slice[span.clone()])
                    .zip(&count_slice[span])
                {
                    *dst = if c == 0.0 { <$ty>::NAN } else { s / c };
                }
            });
        }};
    }

    match sum.dtype() {
        DataType::Float32 => divide!(as_f32_slice, as_f32_slice_mut, f32, "f32"),
        DataType::Float64 => divide!(as_f64_slice, as_f64_slice_mut, f64, "f64"),
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
        return Err(MinitensorError::dim_out_of_range(
            dim as isize,
            tensor.ndim(),
        ));
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
        return Err(MinitensorError::dim_out_of_range(
            dim as isize,
            tensor.ndim(),
        ));
    }

    let input_shape = tensor.shape().dims();
    let mut output_shape = input_shape.to_vec();

    if keepdim {
        output_shape[dim] = 1;
    } else {
        output_shape.remove(dim);
    }

    let output_shape_obj = Shape::new(output_shape);
    let out_dtype = accumulating_dtype(tensor.dtype());
    let mut result_data =
        TensorData::zeros_on_device(output_shape_obj.numel(), out_dtype, tensor.device());

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
        out_dtype,
        tensor.device(),
        tensor.requires_grad(),
    ))
}

#[cfg(test)]
mod var_layout_tests {
    use super::*;
    use crate::device::Device;

    fn f32_tensor(data: Vec<f32>, shape: Vec<usize>) -> Tensor {
        let shape = Shape::new(shape);
        Tensor::new(
            Arc::new(TensorData::from_vec::<f32>(
                data,
                DataType::Float32,
                Device::cpu(),
            )),
            shape,
            DataType::Float32,
            Device::cpu(),
            false,
        )
    }

    fn values(t: &Tensor) -> Vec<f32> {
        t.data().as_f32_slice().unwrap().to_vec()
    }

    /// The fused variance takes one of two layouts depending on whether the
    /// reduced axis is the last one, and they are different code. Reducing a
    /// square tensor along each axis in turn runs both over data that is a
    /// transpose of itself, so the two must produce the same numbers.
    #[test]
    fn both_layouts_agree_on_a_transpose() {
        let n = 37;
        let data: Vec<f32> = (0..n * n).map(|i| (i % 13) as f32 * 0.5 - 3.0).collect();
        let mut transposed = vec![0.0f32; n * n];
        for r in 0..n {
            for c in 0..n {
                transposed[c * n + r] = data[r * n + c];
            }
        }
        let a = f32_tensor(data, vec![n, n]);
        let b = f32_tensor(transposed, vec![n, n]);

        for unbiased in [false, true] {
            // `a` reduced along its last axis is `b` reduced along its first.
            let last = values(&var(&a, Some(vec![1]), false, unbiased).unwrap());
            let first = values(&var(&b, Some(vec![0]), false, unbiased).unwrap());
            assert_eq!(last.len(), n);
            for (i, (x, y)) in last.iter().zip(&first).enumerate() {
                assert!(
                    (x - y).abs() <= 1e-6 * x.abs().max(1.0),
                    "row {i}: last-axis {x} vs first-axis {y} (unbiased {unbiased})"
                );
            }
        }
    }

    /// The last-axis layout bands rows across the pool above the parallel
    /// threshold. Banding cannot change a row's own two passes, so a tall
    /// tensor whose rows are all the same must come back with one value
    /// repeated -- at a height that crosses the threshold and leaves a partial
    /// band.
    #[test]
    fn the_row_band_split_leaves_every_row_alone() {
        let cols = 8;
        let rows = crate::ops::map::PAR_THRESHOLD / cols + 7;
        let row: Vec<f32> = (0..cols).map(|i| i as f32 * 1.5 - 2.0).collect();
        let data: Vec<f32> = (0..rows).flat_map(|_| row.iter().copied()).collect();
        let t = f32_tensor(data, vec![rows, cols]);

        let got = values(&var(&t, Some(vec![1]), false, false).unwrap());
        assert_eq!(got.len(), rows);

        let mean = row.iter().sum::<f32>() / cols as f32;
        let want = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / cols as f32;
        for (i, &g) in got.iter().enumerate() {
            assert!(
                (g - want).abs() <= 1e-6 * want.abs().max(1.0),
                "row {i}: {g} against {want}"
            );
        }
    }

    /// Bessel's correction has nowhere to go on a single sample, and the
    /// answer is NaN rather than a silent zero. Both layouts must say so.
    #[test]
    fn a_single_sample_is_undefined_when_unbiased() {
        let last = var(
            &f32_tensor(vec![1.0, 2.0], vec![2, 1]),
            Some(vec![1]),
            false,
            true,
        )
        .unwrap();
        assert!(values(&last).iter().all(|v| v.is_nan()));

        let first = var(
            &f32_tensor(vec![1.0, 2.0], vec![1, 2]),
            Some(vec![0]),
            false,
            true,
        )
        .unwrap();
        assert!(values(&first).iter().all(|v| v.is_nan()));

        // ...and is plain zero when it is not corrected.
        let biased = var(
            &f32_tensor(vec![1.0, 2.0], vec![2, 1]),
            Some(vec![1]),
            false,
            false,
        )
        .unwrap();
        assert_eq!(values(&biased), vec![0.0, 0.0]);
    }
}
