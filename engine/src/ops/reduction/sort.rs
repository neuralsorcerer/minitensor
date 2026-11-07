// Copyright (c) Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

use super::*;
use crate::ops::map::{
    SIMD_PAR_CHUNK, build_vec, par_fold_chunks, par_out_chunks, par_out_chunks2,
};
use crate::ops::order::{
    Entry, bool_key, float_key32, float_key64, int_key32, int_key64, pack32, pack64,
    sorts_in_parallel,
};
use crate::ops::shape_ops;
use crate::ops::simd::*;
use crate::ops::util::check_dim;
use crate::ops::util::{Accumulate, accumulating_dtype};
use crate::{
    error::{MinitensorError, Result},
    tensor::{DataType, Shape, Tensor, TensorData},
};
use rayon::prelude::*;
use std::sync::Arc;

/// Sort each 1-D slice along a dimension, parallelizing over the outer index.
///
/// `values`/`indices` are partitioned into one disjoint chunk per outer
/// position (`par_chunks_mut`), so the parallel writes never overlap and this
/// stays safe. Each slice packs its elements into [`Entry`] keys, sorts them,
/// and reads the values back out of `input` by the position each key carries
/// (so `indices` becomes the argsort).
///
/// Only worth calling when there are enough slices to fill the thread pool --
/// see [`sort_rows_with_parallel_sort`] for the other case.
#[allow(clippy::too_many_arguments)]
fn sort_along_dim_par<T, E, M>(
    input: &[T],
    values: &mut [T],
    indices: &mut [i64],
    outer: usize,
    inner: usize,
    dim_size: usize,
    outer_stride: usize,
    make: M,
) where
    T: Copy + Send + Sync,
    E: Entry,
    M: Fn(usize, T) -> E + Copy + Sync,
{
    debug_assert_eq!(values.len(), outer * outer_stride);
    debug_assert_eq!(indices.len(), outer * outer_stride);
    // One outer position a task, but only once there is enough to share
    // (see [`shares_rows`]); below that the whole sort is one task, on the
    // calling thread. Erased at the chunk boundary only: `make` stays a concrete type
    // inside the body, so the packing still monomorphizes against it.
    let chunk = if shares_rows::<E>(values.len()) {
        outer_stride
    } else {
        values.len()
    };
    par_out_chunks2(values, indices, chunk, &|start, vchunk, ichunk| {
        let mut entries: Vec<E> = Vec::with_capacity(dim_size);
        for (row, (vrow, irow)) in vchunk
            .chunks_mut(outer_stride)
            .zip(ichunk.chunks_mut(outer_stride))
            .enumerate()
        {
            let o = start / outer_stride + row;
            for r in 0..inner {
                entries.clear();
                let base = o * outer_stride + r;
                for d in 0..dim_size {
                    entries.push(make(d, input[base + d * inner]));
                }
                entries.sort_unstable();
                for (j, entry) in entries.iter().enumerate() {
                    let position = entry.position();
                    let off = r + j * inner;
                    vrow[off] = input[base + position * inner];
                    irow[off] = position as i64;
                }
            }
        }
    });
}

/// Whether `len` entries sorted as independent slices are worth sharing out.
///
/// Below [`sorts_in_parallel`]'s size, because independent slices have no
/// merge to pay for, but not far below: a pool round trip from Python costs
/// 45-80us on average on four cores and past 150 one time in ten, and a
/// sorted element about ten nanoseconds, so a sort under 64 KiB of entries
/// (8192 float32, 4096 float64) costs less on one core than the split does at
/// its worst. Measured split one row a task, `(4, 8)` took 7-48us for well
/// under one on one core, and `(8, 64)` 11 where one core takes 4.6.
fn shares_rows<E>(len: usize) -> bool {
    const SHARE_ROWS_MIN_BYTES: usize = 64 << 10;
    len.saturating_mul(std::mem::size_of::<E>()) >= SHARE_ROWS_MIN_BYTES
}

/// Pack one slice into sort keys, spread across the pool.
///
/// A two-million-element slice asks for 16MB of keys -- a fresh mapping, at
/// glibc's largest dynamic mmap threshold, whose pages fault in on first
/// write. Filling it on one core paid every one of those faults serially and
/// cost as much as half the sort that follows.
fn packed_entries<T, E, M>(
    input: &[T],
    base: usize,
    inner: usize,
    dim_size: usize,
    make: M,
) -> Vec<E>
where
    T: Copy + Send + Sync,
    E: Entry,
    M: Fn(usize, T) -> E + Copy + Sync,
{
    // SAFETY: `par_out_chunks` hands out a partition of the spare slice, and
    // the body writes every element of the chunk it is given.
    unsafe {
        build_vec(dim_size, |spare| {
            let bytes = std::mem::size_of_val(spare);
            crate::ops::map::par_out_chunks_sized(spare, SIMD_PAR_CHUNK, bytes, &|start, chunk| {
                for (offset, slot) in chunk.iter_mut().enumerate() {
                    let d = start + offset;
                    slot.write(make(d, input[base + d * inner]));
                }
            });
        })
    }
}

/// Lay a sorted slice back down the axis, spread across the pool.
///
/// Only when the destination is contiguous, which is the case this exists for:
/// the caller runs when there are fewer slices than threads, and a slice with
/// `inner > 1` is interleaved with its neighbours, so no cut of the output
/// separates the writes. That case is a handful of elements anyway -- `inner`
/// is below the thread count there -- and stays sequential.
fn scatter_entries<T, E>(
    entries: &[E],
    input: &[T],
    values: &mut [T],
    indices: &mut [i64],
    base: usize,
    inner: usize,
    dim_size: usize,
) where
    T: Copy + Send + Sync,
    E: Entry,
{
    if inner == 1 {
        let end = base + dim_size;
        par_out_chunks2(
            &mut values[base..end],
            &mut indices[base..end],
            SIMD_PAR_CHUNK,
            &|start, value_chunk, index_chunk| {
                for (offset, (value, index)) in value_chunk
                    .iter_mut()
                    .zip(index_chunk.iter_mut())
                    .enumerate()
                {
                    let position = entries[start + offset].position();
                    *value = input[base + position];
                    *index = position as i64;
                }
            },
        );
        return;
    }
    for (j, entry) in entries.iter().enumerate() {
        let position = entry.position();
        let off = base + j * inner;
        values[off] = input[base + position * inner];
        indices[off] = position as i64;
    }
}

/// The same sort, but parallel *within* each slice rather than across them.
///
/// `sort_along_dim_par` splits the work by outer position, which leaves most
/// of the machine idle when there are fewer slices than threads -- and a 1-D
/// tensor has exactly one, so sorting one ran entirely on a single core. The
/// same 2M elements cost 134 ns each as one slice and 16 ns each as 2048 of
/// them, a 8.3x spread on four cores that was pure scheduling.
#[allow(clippy::too_many_arguments)]
fn sort_rows_with_parallel_sort<T, E, M>(
    input: &[T],
    values: &mut [T],
    indices: &mut [i64],
    outer: usize,
    inner: usize,
    dim_size: usize,
    outer_stride: usize,
    make: M,
) where
    T: Copy + Send + Sync,
    E: Entry,
    M: Fn(usize, T) -> E + Copy + Sync,
{
    for o in 0..outer {
        for r in 0..inner {
            let base = o * outer_stride + r;
            let mut entries: Vec<E> = packed_entries(input, base, inner, dim_size, make);
            crate::parallel::sort_unstable_by(&mut entries, E::cmp);
            scatter_entries(&entries, input, values, indices, base, inner, dim_size);
        }
    }
}

/// The same sort again, for an axis that is not the last one.
///
/// [`sort_along_dim_par`] spreads its work by cutting the output into one
/// contiguous piece per *outer* position, and a sort along the first axis has
/// exactly one of those however large the tensor is -- so `sort(x, 0)` ran on a
/// single core no matter how many were free. The slices are there to be shared
/// out, `inner` of them, but they are interleaved a stride apart and no cut of
/// a contiguous buffer separates them.
///
/// They are contiguous in the *other* layout. Sorting into a scratch ordered
/// slice-by-slice makes each slice one contiguous run, which cuts apart the way
/// the other kernel's outer positions do, and a second pass lays the result back
/// down the axis it came from. That pass is one strided copy against a sort, and
/// it buys the whole pool: 400ms to 117ms on a 2048-by-2048 sorted down its
/// columns.
#[allow(clippy::too_many_arguments)]
fn sort_along_dim_transposed<T, E, M>(
    input: &[T],
    values: &mut [T],
    indices: &mut [i64],
    outer: usize,
    inner: usize,
    dim_size: usize,
    outer_stride: usize,
    make: M,
) where
    T: Copy + Send + Sync,
    E: Entry,
    M: Fn(usize, T) -> E + Copy + Sync,
{
    // Slice-major: `[outer][inner][dim_size]`, so one slice is one run.
    // Written before it is sorted rather than into a scratch that is then
    // copied: the packing pass is what faults the pages in, and it is parallel.
    //
    // SAFETY: `par_out_chunks` partitions the spare slice into whole runs and
    // each body writes every element of its run.
    let mut packed: Vec<E> = unsafe {
        build_vec(outer * outer_stride, |spare| {
            par_out_chunks(spare, dim_size, &|start, run| {
                let slice = start / dim_size;
                let base = (slice / inner) * outer_stride + slice % inner;
                for (d, slot) in run.iter_mut().enumerate() {
                    slot.write(make(d, input[base + d * inner]));
                }
            });
        })
    };
    par_out_chunks(&mut packed, dim_size, &|_, run| run.sort_unstable());

    // Back down the axis. One output row -- every slice's `j`th element -- is
    // contiguous, which is the cut that makes this pass safe and parallel too.
    par_out_chunks2(values, indices, inner, &|start, vrow, irow| {
        let row = start / inner;
        let o = row / dim_size;
        let base = o * outer_stride + row % dim_size;
        for (r, (value, index)) in vrow.iter_mut().zip(irow.iter_mut()).enumerate() {
            let position = packed[base + r * dim_size].position();
            *value = input[o * outer_stride + r + position * inner];
            *index = position as i64;
        }
    });
}

/// Pick whichever of the three has parallelism to exploit.
///
/// Splitting across slices is cheaper per element when there are enough of
/// them, because each sort stays on one core with no merge step. It only wins
/// when the pool is actually filled, which is what this decides.
#[allow(clippy::too_many_arguments)]
fn sort_along_dim<T, E, M>(
    input: &[T],
    values: &mut [T],
    indices: &mut [i64],
    outer: usize,
    inner: usize,
    dim_size: usize,
    outer_stride: usize,
    make: M,
) where
    T: Copy + Send + Sync,
    E: Entry,
    M: Fn(usize, T) -> E + Copy + Sync,
{
    let slices = outer.saturating_mul(inner);
    let threads = crate::parallel::current_num_threads();
    // `sort_along_dim_par` can only spread its work across `outer` positions,
    // so a tensor with few of them and many slices -- which is every sort along
    // the first axis, where `outer` is one -- leaves the pool idle. Rewriting
    // the axis as the last one costs a strided pass and buys all of it.
    if inner > 1 && outer < threads && shares_rows::<E>(slices.saturating_mul(dim_size)) {
        sort_along_dim_transposed(
            input,
            values,
            indices,
            outer,
            inner,
            dim_size,
            outer_stride,
            make,
        );
    } else if slices < threads
        && sorts_in_parallel(dim_size.saturating_mul(std::mem::size_of::<E>()))
    {
        sort_rows_with_parallel_sort(
            input,
            values,
            indices,
            outer,
            inner,
            dim_size,
            outer_stride,
            make,
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
            make,
        );
    }
}

/// Sort a dtype whose key fits in 32 bits.
///
/// The key and the position share one `u64` -- 8 bytes an element where the
/// value-and-index pair this replaced took 16 -- unless the axis is longer than
/// `u32::MAX`, which has nowhere to put the position and takes the wide form
/// instead. `descending` complements the key rather than reversing the
/// comparison, so ties still fall back to the *ascending* position and the
/// answer stays the mirror of the ascending one rather than its reverse.
#[allow(clippy::too_many_arguments)]
fn sort_by_key32<T, K>(
    input: &[T],
    values: &mut [T],
    indices: &mut [i64],
    outer: usize,
    inner: usize,
    dim_size: usize,
    outer_stride: usize,
    descending: bool,
    key: K,
) where
    T: Copy + Send + Sync,
    K: Fn(T) -> u32 + Copy + Sync,
{
    macro_rules! run {
        ($make:expr) => {
            sort_along_dim(
                input,
                values,
                indices,
                outer,
                inner,
                dim_size,
                outer_stride,
                $make,
            )
        };
    }
    if dim_size <= u32::MAX as usize {
        match descending {
            true => run!(move |d: usize, v: T| pack32(!key(v), d)),
            false => run!(move |d: usize, v: T| pack32(key(v), d)),
        }
    } else {
        match descending {
            true => run!(move |d: usize, v: T| pack64(!key(v) as u64, d)),
            false => run!(move |d: usize, v: T| pack64(key(v) as u64, d)),
        }
    }
}

/// Sort a dtype whose key fills 64 bits, which leaves only the wide entry.
#[allow(clippy::too_many_arguments)]
fn sort_by_key64<T, K>(
    input: &[T],
    values: &mut [T],
    indices: &mut [i64],
    outer: usize,
    inner: usize,
    dim_size: usize,
    outer_stride: usize,
    descending: bool,
    key: K,
) where
    T: Copy + Send + Sync,
    K: Fn(T) -> u64 + Copy + Sync,
{
    macro_rules! run {
        ($make:expr) => {
            sort_along_dim(
                input,
                values,
                indices,
                outer,
                inner,
                dim_size,
                outer_stride,
                $make,
            )
        };
    }
    match descending {
        true => run!(move |d: usize, v: T| pack64(!key(v), d)),
        false => run!(move |d: usize, v: T| pack64(key(v), d)),
    }
}

/// Sort each slice along `dim`, returning the values and the argsort.
///
/// `stable` is accepted for the shape of the API and changes nothing: this
/// sort is *always* stable. Every element's position goes into its sort key
/// (see [`Entry`]), so no two entries compare equal, there is no tie for an
/// unstable sort to resolve differently, and equal values always come back in
/// the order they went in -- on any number of threads. `stable = false`
/// therefore costs nothing and buys nothing, which is the honest answer to a
/// caller who did not need the guarantee.
pub fn sort(
    tensor: &Tensor,
    dim: Option<isize>,
    descending: bool,
    stable: bool,
) -> Result<(Tensor, Tensor)> {
    // Named in the signature above, and deliberately unread. See the doc
    // comment: the ordering is total, so both settings are the same sort.
    let _ = stable;
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
        let values = crate::autograd::empty_result_of(tensor, tensor.shape().clone())?;
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

    // The key function goes in as a function item rather than through an `if`,
    // which would coerce it to a non-inlinable function pointer and put an
    // indirect call on every element.
    //
    macro_rules! run_sort {
        ($width:ident, $input:expr, $values:expr, $indices:expr, $key:expr) => {
            $width(
                $input,
                $values,
                $indices,
                outer,
                inner,
                dim_size,
                outer_stride,
                descending,
                $key,
            )
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

            run_sort!(sort_by_key32, input, values, indices, float_key32);
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

            run_sort!(sort_by_key64, input, values, indices, float_key64);
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

            run_sort!(sort_by_key32, input, values, indices, int_key32);
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

            run_sort!(sort_by_key64, input, values, indices, int_key64);
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

            run_sort!(sort_by_key32, input, values, indices, bool_key);
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

/// Standard deviation along specified dimensions: each the float nearest
/// the exact root of its slice's variance; see [`var`]. An integer tensor's
/// is a float, float32 for int32 and float64 for int64. The gradient is the
/// variance's through the root, zero where a slice is constant.
pub fn std(
    tensor: &Tensor,
    dim: Option<Vec<isize>>,
    keepdim: bool,
    unbiased: bool,
) -> Result<Tensor> {
    spread(tensor, dim, keepdim, unbiased, true, false)
}

/// The axes a statistic over `dim` reduces, every one when `dim` is `None`.
pub(crate) fn statistic_dims(tensor: &Tensor, dim: &Option<Vec<isize>>) -> Result<Vec<usize>> {
    Ok(normalize_reduction_dims(dim.clone(), tensor.ndim())?
        .unwrap_or_else(|| (0..tensor.ndim()).collect()))
}

/// A statistic taken again by `compute` over `tensor` divided by `scales`,
/// and the scales laid out as its result is.
///
/// Dividing by a power of two is exact, and leaves the largest finite
/// magnitude of every rescued slice in `[1, 2)`, where neither a sum nor its
/// squares can leave the range: a mean is the result times the scale, a
/// variance the result times it twice and a standard deviation its root
/// times it once.
pub(crate) fn rescaled(
    tensor: &Tensor,
    scales: &Tensor,
    compute: impl FnOnce(&Tensor) -> Result<Tensor>,
) -> Result<(Tensor, Tensor)> {
    let statistic = compute(&crate::ops::arithmetic::div(tensor, scales)?)?;
    let scales = scales_for_result(scales, statistic.shape())?;
    Ok((statistic, scales))
}

/// `tensor` over a count of elements, exactly as counted.
///
/// A count past `2^24` is not a float32, and dividing by its rounding is an
/// error on top of the dividend's own; there the quotient is formed in
/// float64, where every count is exact, and rounds once on its way back.
/// Otherwise it is the division in the tensor's dtype, which is the same
/// rounding. A count of zero divides by zero.
pub(crate) fn divide_by_count(tensor: &Tensor, count: f64) -> Result<Tensor> {
    let scalar = |value: f64, dtype: DataType| {
        crate::ops::util::create_scalar_tensor(value, dtype, tensor.device())
    };
    if tensor.dtype() == DataType::Float32 && count > f64::from(1u32 << 24) {
        let quotient = crate::ops::arithmetic::div(
            &tensor.astype(DataType::Float64)?,
            &scalar(count, DataType::Float64)?,
        )?;
        return quotient.astype(DataType::Float32);
    }
    crate::ops::arithmetic::div(tensor, &scalar(count, tensor.dtype())?)
}

/// Variance along specified dimensions: each the float nearest the exact
/// `(N Σx² - (Σx)²) / (N (N - ddof))` of its slice, `ddof` one when
/// `unbiased`; see `arithmetic::exact_sum::variance_of`.
///
/// An integer argument widens rather than being refused, the rule `mean`
/// already follows -- a variance of integers is a real number -- read exactly
/// in float64 and rounded once to float32 for int32. A slice with no more
/// values than `ddof` has a variance of NaN, and so has one holding a NaN or
/// an infinity; squares past the top of the float range are taken exactly.
/// The gradient is `2 (x - mean) / (N - ddof)`.
pub fn var(
    tensor: &Tensor,
    dim: Option<Vec<isize>>,
    keepdim: bool,
    unbiased: bool,
) -> Result<Tensor> {
    spread(tensor, dim, keepdim, unbiased, false, false)
}

/// The variance over `dim`, or with `root` the standard deviation, of every
/// value or with `skip_nan` of those that are not NaN, each rounded once,
/// and its gradient: `2 (x - mean) / (N - ddof)` over the values it takes,
/// zero for a NaN it skips, through the root for a standard deviation.
pub(crate) fn spread(
    tensor: &Tensor,
    dim: Option<Vec<isize>>,
    keepdim: bool,
    unbiased: bool,
    root: bool,
    skip_nan: bool,
) -> Result<Tensor> {
    let values = rounded_variance(tensor, dim.clone(), keepdim, unbiased, root, skip_nan)?;
    if !(tensor.requires_grad() && crate::autograd::is_grad_enabled()) {
        return Ok(values);
    }
    let output = values.requires_grad_(true);
    if root {
        let variance = spread(tensor, dim, keepdim, unbiased, false, skip_nan)?;
        let grad_fn = Arc::new(crate::autograd::StdBackward {
            input_id: variance.id(),
            output: output.clone().detach(),
        });
        return crate::autograd::with_grad_fn(output, grad_fn);
    }
    let grad_fn = Arc::new(VarianceBackward {
        input: tensor.detach(),
        dims: statistic_dims(tensor, &dim)?,
        keepdim,
        ddof: usize::from(unbiased),
        skip_nan,
        input_ids: [tensor.id()],
    });
    crate::autograd::with_grad_fn(output, grad_fn)
}

/// The variance of each slice of `tensor` over `dim`, or with `root` its
/// standard deviation, of every value or with `skip_nan` of those that are
/// not NaN, rounded once; no gradient.
fn rounded_variance(
    tensor: &Tensor,
    dim: Option<Vec<isize>>,
    keepdim: bool,
    unbiased: bool,
    root: bool,
    skip_nan: bool,
) -> Result<Tensor> {
    use crate::ops::arithmetic::exact_sum::{Element, Output, column_variances, variance_of};
    let dtype = match tensor.dtype() {
        DataType::Float32 | DataType::Int32 => DataType::Float32,
        DataType::Float64 | DataType::Int64 => DataType::Float64,
        DataType::Bool => {
            return Err(MinitensorError::invalid_operation(
                "Variance only supported for floating point tensors",
            ));
        }
    };
    let dims = statistic_dims(tensor, &dim)?;
    let shape = tensor.shape().dims().to_vec();
    let kept: Vec<usize> = (0..shape.len())
        .filter(|axis| !dims.contains(axis))
        .collect();
    let run: usize = dims.iter().map(|&axis| shape[axis]).product();
    let outer: usize = kept.iter().map(|&axis| shape[axis]).product();
    // Reduced axes that are one block with axes after it are read in place,
    // as `[before, run, after]` columns, when there are columns enough to
    // share out; otherwise the reduced axes go last, so that each slice is
    // one run.
    let after: usize = dims
        .last()
        .map_or(1, |&last| shape[last + 1..].iter().product());
    let block = dims.windows(2).all(|pair| pair[1] == pair[0] + 1);
    // An integer slice is shifted by its first element first, exactly, which
    // leaves its variance and makes an int64 past 2^53 convert to float64
    // exactly.
    let mut moved = if tensor.dtype().is_float() {
        tensor.detach().contiguous()?
    } else {
        integer_deviations(&tensor.detach(), &dims)?
    };
    // Columns holding a NaN to skip are read as runs, where each one's
    // values are gathered without its NaNs at the cost of a copy, rather
    // than column by column.
    let skipping = skip_nan && {
        let data = moved.data();
        match (data.as_f32_slice(), data.as_f64_slice()) {
            (Some(values), _) => any_nan(values),
            (_, Some(values)) => any_nan(values),
            _ => false,
        }
    };
    let columns = !dims.is_empty() && block && after >= 8 && outer >= 32 && !skipping;
    let order: Vec<isize> = kept
        .iter()
        .chain(&dims)
        .map(|&axis| axis as isize)
        .collect();
    if !columns
        && order
            .iter()
            .enumerate()
            .any(|(index, &axis)| index as isize != axis)
    {
        moved = moved.permute(order)?;
    }
    let mut moved = moved.contiguous()?;
    if moved.dtype() != DataType::Float32 {
        moved = moved.astype(DataType::Float64)?;
    }
    let ddof = usize::from(unbiased);
    fn slices<I: Element, T: Output>(
        values: &[I],
        (outer, run, after): (usize, usize, usize),
        columns: bool,
        (ddof, root, skip_nan): (usize, bool, bool),
    ) -> Vec<T> {
        if columns {
            return column_variances(values, outer / after, run, after, ddof, root, skip_nan);
        }
        let mut out = vec![T::narrow(0.0); outer];
        let per_task = ((1 << 14) / run.max(1)).max(1);
        par_out_chunks(&mut out, per_task, &|start, block| {
            // One buffer a task for the slices that hold a NaN.
            let mut kept: Vec<I> = Vec::new();
            for (offset, slot) in block.iter_mut().enumerate() {
                let first = (start + offset) * run;
                let slice = &values[first..first + run];
                *slot = if skip_nan && any_nan(slice) {
                    keep_numbers(slice, &mut kept);
                    variance_of(&kept, ddof, root)
                } else {
                    variance_of(slice, ddof, root)
                };
            }
        });
        out
    }
    let layout = (outer, run, after);
    let ddof = (ddof, root, skip_nan);
    let missing = || MinitensorError::internal_error("a variance operand without its slice");
    let data = match (moved.dtype(), dtype) {
        (DataType::Float32, _) => {
            let values = moved.data().as_f32_slice().ok_or_else(missing)?;
            let out = slices::<f32, f32>(values, layout, columns, ddof);
            TensorData::from_vec_f32(out, tensor.device())
        }
        (_, DataType::Float32) => {
            let values = moved.data().as_f64_slice().ok_or_else(missing)?;
            let out = slices::<f64, f32>(values, layout, columns, ddof);
            TensorData::from_vec_f32(out, tensor.device())
        }
        _ => {
            let values = moved.data().as_f64_slice().ok_or_else(missing)?;
            let out = slices::<f64, f64>(values, layout, columns, ddof);
            TensorData::from_vec_f64(out, tensor.device())
        }
    };
    let out_shape = if keepdim {
        let mut kept_shape = shape.clone();
        for &axis in &dims {
            kept_shape[axis] = 1;
        }
        Shape::new(kept_shape)
    } else if kept.is_empty() {
        Shape::scalar()
    } else {
        Shape::new(kept.iter().map(|&axis| shape[axis]).collect())
    };
    Ok(Tensor::new(
        Arc::new(data),
        out_shape,
        dtype,
        tensor.device(),
        false,
    ))
}

/// The sum, or with `mean` the mean, of the values of each slice of `tensor`
/// over `dims` that are not NaN, each the float nearest the exact one; no
/// gradient.
///
/// With no NaN at all it is `sum` or `mean`, which are that; otherwise the
/// reduced axes go last and each slice holding a NaN is gathered without it,
/// as `nanvar` gathers one. A slice with nothing left sums to zero and has a
/// mean of NaN, and one holding infinities of both signs sums to NaN.
pub(crate) fn rounded_nan_total(
    tensor: &Tensor,
    dims: &[usize],
    keepdim: bool,
    mean: bool,
) -> Result<Tensor> {
    use crate::ops::arithmetic::exact_sum::{Element, Output, mean_to, sum_to};
    let moved = tensor.detach().contiguous()?;
    let has_nan = {
        let data = moved.data();
        match (data.as_f32_slice(), data.as_f64_slice()) {
            (Some(values), _) => any_nan(values),
            (_, Some(values)) => any_nan(values),
            _ => false,
        }
    };
    let axes: Vec<isize> = dims.iter().map(|&axis| axis as isize).collect();
    if !has_nan {
        return if mean {
            crate::ops::reduction::mean(&moved, Some(axes), keepdim)
        } else {
            super::sum(&moved, Some(axes), keepdim)
        };
    }
    let shape = tensor.shape().dims().to_vec();
    let kept: Vec<usize> = (0..shape.len())
        .filter(|axis| !dims.contains(axis))
        .collect();
    let run: usize = dims.iter().map(|&axis| shape[axis]).product();
    let outer: usize = kept.iter().map(|&axis| shape[axis]).product();
    let order: Vec<isize> = kept.iter().chain(dims).map(|&axis| axis as isize).collect();
    let moved = if order
        .iter()
        .enumerate()
        .any(|(index, &axis)| index as isize != axis)
    {
        moved.permute(order)?.contiguous()?
    } else {
        moved
    };
    fn slices<I: Element, T: Output>(values: &[I], outer: usize, run: usize, mean: bool) -> Vec<T> {
        let mut out = vec![T::narrow(0.0); outer];
        let per_task = ((1 << 14) / run.max(1)).max(1);
        par_out_chunks(&mut out, per_task, &|start, block| {
            // One buffer a task for the slices that hold a NaN.
            let mut kept: Vec<I> = Vec::new();
            for (offset, slot) in block.iter_mut().enumerate() {
                let first = (start + offset) * run;
                let mut numbers = &values[first..first + run];
                if any_nan(numbers) {
                    keep_numbers(numbers, &mut kept);
                    numbers = &kept;
                }
                *slot = if mean {
                    mean_to(numbers)
                } else {
                    sum_to(numbers)
                };
            }
        });
        out
    }
    let missing = || MinitensorError::internal_error("a NaN-skipping total without its slice");
    let data = match moved.dtype() {
        DataType::Float32 => {
            let values = moved.data().as_f32_slice().ok_or_else(missing)?;
            let out = slices::<f32, f32>(values, outer, run, mean);
            TensorData::from_vec_f32(out, tensor.device())
        }
        _ => {
            let values = moved.data().as_f64_slice().ok_or_else(missing)?;
            let out = slices::<f64, f64>(values, outer, run, mean);
            TensorData::from_vec_f64(out, tensor.device())
        }
    };
    let out_shape = if keepdim {
        let mut kept_shape = shape.clone();
        for &axis in dims {
            kept_shape[axis] = 1;
        }
        Shape::new(kept_shape)
    } else if kept.is_empty() {
        Shape::scalar()
    } else {
        Shape::new(kept.iter().map(|&axis| shape[axis]).collect())
    };
    Ok(Tensor::new(
        Arc::new(data),
        out_shape,
        tensor.dtype(),
        tensor.device(),
        false,
    ))
}

/// The product of each slice of float `tensor` over `dims`, each the float
/// nearest the exact product; see `arithmetic::exact_product`. Axes side by
/// side are read where they lie, as one axis; others go last first, so that
/// each slice is one run. No gradient.
pub(crate) fn rounded_product(tensor: &Tensor, dims: &[usize], keepdim: bool) -> Result<Tensor> {
    use crate::ops::arithmetic::exact_product::axis_products;
    use crate::ops::arithmetic::exact_sum::{Element, Output};
    let shape = tensor.shape().dims().to_vec();
    let mut reduced = dims.to_vec();
    reduced.sort_unstable();
    reduced.dedup();
    let kept: Vec<usize> = (0..shape.len())
        .filter(|axis| !reduced.contains(axis))
        .collect();
    let len: usize = reduced.iter().map(|&axis| shape[axis]).product();
    let side_by_side = reduced.windows(2).all(|pair| pair[1] == pair[0] + 1);
    let (moved, inner) = match reduced.first() {
        Some(&first) if side_by_side => {
            let last = first + reduced.len() - 1;
            (tensor.detach(), shape[last + 1..].iter().product())
        }
        _ => {
            let order: Vec<isize> = kept
                .iter()
                .chain(&reduced)
                .map(|&axis| axis as isize)
                .collect();
            (tensor.detach().permute(order)?, 1)
        }
    };
    let moved = moved.contiguous()?;
    let outer: usize = kept.iter().map(|&axis| shape[axis]).product();
    fn products<I: Element, T: Output>(
        values: &[I],
        outer: usize,
        len: usize,
        inner: usize,
    ) -> Vec<T> {
        let mut out = vec![T::narrow(1.0); outer];
        axis_products(values, &mut out, len, inner);
        out
    }
    let missing = || MinitensorError::internal_error("a product without its float slice");
    let data = match moved.dtype() {
        DataType::Float32 => {
            let values = moved.data().as_f32_slice().ok_or_else(missing)?;
            let out = products::<f32, f32>(values, outer, len, inner);
            TensorData::from_vec_f32(out, tensor.device())
        }
        _ => {
            let values = moved.data().as_f64_slice().ok_or_else(missing)?;
            let out = products::<f64, f64>(values, outer, len, inner);
            TensorData::from_vec_f64(out, tensor.device())
        }
    };
    let out_shape = if keepdim {
        let mut kept_shape = shape.clone();
        for &axis in &reduced {
            kept_shape[axis] = 1;
        }
        Shape::new(kept_shape)
    } else if kept.is_empty() {
        Shape::scalar()
    } else {
        Shape::new(kept.iter().map(|&axis| shape[axis]).collect())
    };
    Ok(Tensor::new(
        Arc::new(data),
        out_shape,
        tensor.dtype(),
        tensor.device(),
        false,
    ))
}

/// Whether any of `values` is NaN: a block at a time across the pool for a
/// long run, each block folded without an early exit, which keeps it
/// vectorized.
fn any_nan<I: crate::ops::arithmetic::exact_sum::Element>(values: &[I]) -> bool {
    use crate::ops::map::{SIMD_PAR_CHUNK, par_all_chunk};
    !par_all_chunk(values, SIMD_PAR_CHUNK, &|block| {
        block
            .iter()
            .fold(true, |clear, value| clear & !value.wide().is_nan())
    })
}

/// `values` without their NaNs, in order, in `kept`: a block at a time
/// across the pool for a long run, each block's count first and then its
/// values in place.
fn keep_numbers<I: crate::ops::arithmetic::exact_sum::Element>(values: &[I], kept: &mut Vec<I>) {
    use crate::ops::map::{PAR_THRESHOLD, SIMD_PAR_CHUNK};
    kept.clear();
    let number = |value: &&I| !value.wide().is_nan();
    if values.len() < PAR_THRESHOLD {
        kept.extend(values.iter().filter(number).copied());
        return;
    }
    let span =
        |nth: usize| &values[nth * SIMD_PAR_CHUNK..((nth + 1) * SIMD_PAR_CHUNK).min(values.len())];
    let blocks = values.len().div_ceil(SIMD_PAR_CHUNK);
    let counts = crate::parallel::map_tasks(blocks, &|nth| span(nth).iter().filter(number).count());
    let Some(&filler) = values.iter().find(number) else {
        return;
    };
    kept.resize(counts.iter().sum(), filler);
    let starts: Vec<usize> = counts
        .iter()
        .scan(0, |at, &count| {
            let start = *at;
            *at += count;
            Some(start)
        })
        .collect();
    let parts = crate::parallel::Parts::of(kept);
    crate::parallel::for_each_task(blocks, &|nth| {
        // SAFETY: block `nth` writes `starts[nth]..starts[nth] + counts[nth]`,
        // apart from every other block's and inside `kept`, which stays
        // borrowed mutably throughout.
        let part =
            unsafe { std::slice::from_raw_parts_mut(parts.start().add(starts[nth]), counts[nth]) };
        for (slot, &value) in part.iter_mut().zip(span(nth).iter().filter(number)) {
            *slot = value;
        }
    });
}

/// The gradient of a variance over `dims`: `2 (x - mean) / (N - ddof)` times
/// the incoming gradient, spread back over the reduced axes; with `skip_nan`
/// the mean and the count are of each slice's values that are not NaN, and a
/// NaN it skipped takes zero.
struct VarianceBackward {
    input: Tensor,
    dims: Vec<usize>,
    keepdim: bool,
    ddof: usize,
    skip_nan: bool,
    input_ids: [crate::autograd::TensorId; 1],
}

impl crate::autograd::GradientFunction for VarianceBackward {
    fn backward(
        &self,
        grad_output: &Tensor,
    ) -> Result<rustc_hash::FxHashMap<crate::autograd::TensorId, Tensor>> {
        use crate::ops::arithmetic::{div, mul, sub};
        use crate::ops::util::create_scalar_tensor;
        let input = &self.input;
        let scalar = |value: f64| create_scalar_tensor(value, input.dtype(), input.device());
        let axes: Vec<isize> = self.dims.iter().map(|&axis| axis as isize).collect();
        let (mean, divisor) = if self.skip_nan {
            let count = super::nan_stats_impl::non_nan_count(input, Some(axes.clone()), true)?;
            let mean = crate::ops::reduction::nanmean(input, Some(axes), true)?;
            (mean, sub(&count, &scalar(self.ddof as f64)?)?)
        } else {
            let count: usize = self
                .dims
                .iter()
                .map(|&axis| input.shape().dims()[axis])
                .product();
            let mean = crate::ops::reduction::mean(input, Some(axes), true)?;
            (mean, scalar(count as f64 - self.ddof as f64)?)
        };
        let deviations = sub(input, &mean)?;
        let grad = if self.keepdim {
            grad_output.clone()
        } else {
            let mut kept_shape = input.shape().dims().to_vec();
            for &axis in &self.dims {
                kept_shape[axis] = 1;
            }
            shape_ops::reshape(grad_output, Shape::new(kept_shape))?
        };
        let scale = div(&scalar(2.0)?, &divisor)?;
        let mut gradient = mul(&mul(&grad.astype(input.dtype())?, &deviations)?, &scale)?;
        if self.skip_nan {
            gradient = crate::ops::selection::where_op(&input.isnan()?, &scalar(0.0)?, &gradient)?;
        }
        let mut gradients = rustc_hash::FxHashMap::default();
        gradients.insert(self.input_ids[0], gradient);
        Ok(gradients)
    }

    fn input_ids(&self) -> &[crate::autograd::TensorId] {
        &self.input_ids
    }
}

// Helper functions for type-specific operations

pub(crate) fn prod_all_i32(tensor: &Tensor, result_data: &mut TensorData) -> Result<()> {
    let data = tensor
        .data()
        .as_i32_slice()
        .ok_or_else(|| MinitensorError::internal_error("Failed to get i32 slice"))?;

    // Reads i32, multiplies in i64 -- see `accumulating_dtype`.
    let prod: i64 = if data.len() >= 1024 {
        par_fold_chunks(
            data,
            8192,
            1i64,
            &|_, c| simd_prod_i32_to_i64(c),
            &|a: i64, b| a.acc_mul(b),
        )
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

    let prod: i64 = if data.len() > 8192 {
        crate::parallel::install(|| {
            data.par_chunks(8192)
                .map(simd_prod_i64)
                .reduce(|| 1, |a, b| a.acc_mul(b))
        })
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

    // The float32 nearest the exact sum; see `arithmetic::exact_sum`.
    let sum: f32 = crate::ops::arithmetic::exact_sum::sum_to(data);

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

    // The float64 nearest the exact sum; see `arithmetic::exact_sum`.
    let sum = crate::ops::arithmetic::exact_sum::sum_f64(data);

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
        par_fold_chunks(
            data,
            8192,
            0i64,
            &|_, c| simd_sum_i32_to_i64(c),
            &|a: i64, b| a.acc_add(b),
        )
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

    let sum: i64 = if data.len() > 8192 {
        crate::parallel::install(|| {
            data.par_chunks(8192)
                .map(simd_sum_i64)
                .reduce(|| 0, |a, b| a.acc_add(b))
        })
    } else {
        simd_sum_i64(data)
    };

    let result_slice = result_data
        .as_i64_slice_mut()
        .ok_or_else(|| MinitensorError::internal_error("Failed to get mutable i64 slice"))?;

    result_slice[0] = sum;
    Ok(())
}

/// The count of `true`s in a mask, which is what summing one means.
///
/// Kept native rather than reached by widening the mask to `int64` first: the
/// widening is an eightfold copy of the data to answer a question about it, and
/// it dominated everything. See [`simd_count_true`].
pub(crate) fn sum_all_bool(tensor: &Tensor, result_data: &mut TensorData) -> Result<()> {
    let data = tensor
        .data()
        .as_bool_slice()
        .ok_or_else(|| MinitensorError::internal_error("Failed to get bool slice"))?;

    let sum: i64 = if data.len() >= 1024 {
        par_fold_chunks(
            data,
            1 << 16,
            0i64,
            &|_, c| simd_count_true(c),
            &|a: i64, b| a.acc_add(b),
        )
    } else {
        simd_count_true(data)
    };

    let result_slice = result_data
        .as_i64_slice_mut()
        .ok_or_else(|| MinitensorError::internal_error("Failed to get mutable i64 slice"))?;

    result_slice[0] = sum;
    Ok(())
}

#[inline]
pub fn sum_along_dim(tensor: &Tensor, dim: usize, keepdim: bool) -> Result<Tensor> {
    check_dim(dim, tensor.ndim())?;

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
        DataType::Bool => sum_along_dim_bool(tensor, &mut result_data, dim)?,
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

    fn f64_tensor(data: Vec<f64>, shape: Vec<usize>) -> Tensor {
        let shape = Shape::new(shape);
        Tensor::new(
            Arc::new(TensorData::from_vec::<f64>(
                data,
                DataType::Float64,
                Device::cpu(),
            )),
            shape,
            DataType::Float64,
            Device::cpu(),
            false,
        )
    }

    /// Variance does not move when the data does, and the arithmetic has to
    /// keep that.
    ///
    /// This is how the one-pass and half-compensated forms are caught:
    /// shifting the data by a constant leaves the answer alone mathematically
    /// while everything the sum touches grows by the shift.
    ///
    /// The shift has to be one the data survives, or the test measures the
    /// rounding of the shift rather than the arithmetic under it. Values that
    /// are whole multiples of `2^-12` and offsets that are powers of two up to
    /// `2^40` stay exactly representable together -- 52 bits from `2^40` down
    /// to `2^-12` -- so the shifted data is the shifted data and nothing else
    /// has changed. Below `2^40` the answer is then required to be *bitwise*
    /// the same.
    ///
    /// The previous code reduced each block with a scalar `sum()` where `sum`
    /// itself uses a lane-and-tree one, which left it four orders above the
    /// floor -- 3.4e-6 against 2.5e-10 at an offset of 1e12, where the floor
    /// is the square of the mean's own rounding.
    #[test]
    fn variance_does_not_move_when_the_data_does() {
        // Deterministic, spread over the whole range, and every value a whole
        // multiple of `2^-12`.
        const STEP: f64 = 0.000_244_140_625; // 2^-12
        let base: Vec<f64> = (0..4096u64)
            .map(|i| ((i.wrapping_mul(2_654_435_761) % 16_384) as i64 - 8_192) as f64 * STEP)
            .collect();

        let variance = |data: Vec<f64>| {
            values_f64(&var(&f64_tensor(data, vec![4096]), None, false, true).unwrap())[0]
        };
        let reference = variance(base.clone());
        assert!(
            reference > 1.0,
            "the sample should have spread: {reference}"
        );

        // Exactly representable shifts, so the answer must not change at all.
        for exponent in [10u32, 20, 30] {
            // A shift, not `powi`: its precision is unspecified, and a power of
            // two that is not exactly one is not an exact shift.
            let offset = (1u64 << exponent) as f64;
            let shifted: Vec<f64> = base.iter().map(|v| v + offset).collect();
            assert!(
                shifted.iter().zip(&base).all(|(s, b)| s - offset == *b),
                "2^{exponent} should shift this data exactly"
            );
            let got = variance(shifted);
            assert_eq!(
                got.to_bits(),
                reference.to_bits(),
                "variance moved under an exact shift of 2^{exponent}: {got} against {reference}"
            );
        }

        // At `2^40` the sum is near `2^52` and the mean carries a rounding of
        // its own, which squares into the answer. That floor is 1.1e-8 here.
        let shifted: Vec<f64> = base.iter().map(|v| v + (1u64 << 40) as f64).collect();
        let got = variance(shifted);
        let relative = ((got - reference) / reference).abs();
        assert!(
            relative < 1e-7,
            "variance moved by {relative:e} under a shift of 2^40: {got} against {reference}"
        );
    }

    fn values_f64(t: &Tensor) -> Vec<f64> {
        t.data().as_f64_slice().unwrap().to_vec()
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

#[cfg(test)]
mod sort_key_tests {
    use super::*;

    /// One of the three sort kernels, as the three share a signature: input,
    /// the two outputs, the axis geometry, and the packing.
    type Kernel =
        fn(&[f32], &mut [f32], &mut [i64], usize, usize, usize, usize, fn(usize, f32) -> u64);

    /// The wide entry is only reached by an axis longer than `u32::MAX`, which
    /// no test can allocate -- so drive the kernels with it directly and check
    /// it against the narrow one on the same data.
    #[test]
    fn the_wide_entry_sorts_the_same_as_the_narrow_one() {
        let dim_size = 1000usize;
        let input: Vec<f32> = (0..dim_size)
            .map(|i| ((i * 7919) % 1000) as f32 - 500.0)
            .collect();

        let mut narrow_values = vec![0.0f32; dim_size];
        let mut narrow_indices = vec![0i64; dim_size];
        sort_along_dim(
            &input,
            &mut narrow_values,
            &mut narrow_indices,
            1,
            1,
            dim_size,
            dim_size,
            |d: usize, v: f32| ((float_key32(v) as u64) << 32) | d as u64,
        );

        let mut wide_values = vec![0.0f32; dim_size];
        let mut wide_indices = vec![0i64; dim_size];
        sort_along_dim(
            &input,
            &mut wide_values,
            &mut wide_indices,
            1,
            1,
            dim_size,
            dim_size,
            |d: usize, v: f32| ((float_key32(v) as u128) << 64) | d as u128,
        );

        assert_eq!(narrow_values, wide_values);
        assert_eq!(narrow_indices, wide_indices);
        assert!(narrow_values.windows(2).all(|w| w[0] <= w[1]));
    }

    /// All three kernels sort the same tensor, and the choice between them is
    /// a scheduling decision that must not change the answer. Driving them
    /// directly is the only way to compare them on one shape.
    #[test]
    fn the_three_kernels_agree() {
        let (outer, inner, dim_size) = (3usize, 5usize, 40usize);
        let outer_stride = dim_size * inner;
        let total = outer * outer_stride;
        let input: Vec<f32> = (0..total)
            .map(|i| (((i * 31) % 97) as f32) - 48.0)
            .collect();
        let make = |d: usize, v: f32| ((float_key32(v) as u64) << 32) | d as u64;

        let run = |kernel: Kernel| {
            let mut values = vec![0.0f32; total];
            let mut indices = vec![0i64; total];
            kernel(
                &input,
                &mut values,
                &mut indices,
                outer,
                inner,
                dim_size,
                outer_stride,
                make,
            );
            (values, indices)
        };

        let by_slice = run(sort_along_dim_par);
        let by_row = run(sort_rows_with_parallel_sort);
        let transposed = run(sort_along_dim_transposed);
        assert_eq!(by_slice, by_row);
        assert_eq!(by_slice, transposed);

        // And the answer is actually sorted along the axis.
        for o in 0..outer {
            for r in 0..inner {
                let base = o * outer_stride + r;
                for d in 1..dim_size {
                    assert!(by_slice.0[base + d * inner] >= by_slice.0[base + (d - 1) * inner]);
                }
            }
        }
    }
}
