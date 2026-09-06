// Copyright (c) Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

use super::*;
use crate::ops::map::{
    SIMD_PAR_CHUNK, build_vec, outputs_per_task, par_fold_chunks, par_out_chunks, par_out_chunks2,
    reduction_band,
};
use crate::ops::shape_ops;
use crate::ops::simd::*;
use crate::ops::util::check_dim;
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

/// One element to be sorted: its order key above its position in the slice.
///
/// The comparison is what a sort spends itself on -- forty million of them for
/// two million elements -- so the ordering rule is moved *out* of it. Every
/// dtype maps to an unsigned integer whose ascending order is that dtype's
/// ascending order (see [`float_key32`] and its neighbours), the position goes
/// in the low bits, and the sort is then a plain integer comparison with no
/// branches, no NaN test and no tie-break to fall through to.
///
/// It is also what makes the answer deterministic: the position makes every
/// entry distinct, so the total order has no ties for an unstable sort to
/// resolve differently on a different day. `sort(stable=true)` and
/// `sort(stable=false)` therefore give the same answer, and both get the
/// faster sort.
///
/// The branchy three-way comparator this replaced cost 62ms where the integer
/// one costs 18ms on the same two million float32.
trait Entry: Ord + Copy + Send + Sync {
    /// Where this element sat in the slice before sorting.
    fn position(self) -> usize;
}

impl Entry for u64 {
    #[inline(always)]
    fn position(self) -> usize {
        (self & u32::MAX as u64) as usize
    }
}

impl Entry for u128 {
    #[inline(always)]
    fn position(self) -> usize {
        (self & u64::MAX as u128) as usize
    }
}

/// The order-preserving unsigned key of a float, as its own width.
///
/// The usual bit trick: flipping the sign bit of a non-negative and every bit
/// of a negative turns IEEE-754's sign-magnitude layout into an unsigned
/// integer that compares the same way. Two families need folding first, or the
/// integer order would say things the float order does not:
///
/// * NaN compares with nothing, and this library sorts it after every number.
///   Its bit patterns straddle the range -- a negative NaN would land below
///   negative infinity -- so all of them are folded to the maximum key.
/// * `-0.0` and `0.0` are equal as floats and have different bit patterns, so
///   `-0.0` is folded to `0.0` and the two keep their input order as any other
///   pair of equals does.
macro_rules! float_key {
    ($name:ident, $float:ty, $unsigned:ty, $signed:ty, $shift:expr) => {
        #[inline(always)]
        fn $name(value: $float) -> $unsigned {
            if value.is_nan() {
                return <$unsigned>::MAX;
            }
            let folded = if value == 0.0 { 0.0 } else { value };
            let bits = folded.to_bits();
            bits ^ (((bits as $signed) >> $shift) as $unsigned | (1 << $shift))
        }
    };
}

float_key!(float_key32, f32, u32, i32, 31);
float_key!(float_key64, f64, u64, i64, 63);

/// The order-preserving unsigned key of a signed integer: shift the range so
/// the most negative value becomes zero.
#[inline(always)]
fn int_key32(value: i32) -> u32 {
    (value as u32) ^ (1 << 31)
}

#[inline(always)]
fn int_key64(value: i64) -> u64 {
    (value as u64) ^ (1 << 63)
}

#[inline(always)]
fn bool_key(value: bool) -> u32 {
    value as u32
}

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
    // Erased at the chunk boundary only: `make` stays a concrete type inside
    // the body, so the packing still monomorphizes against it.
    par_out_chunks2(values, indices, outer_stride, &|start, vchunk, ichunk| {
        let o = start / outer_stride;
        let mut entries: Vec<E> = Vec::with_capacity(dim_size);
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
                vchunk[off] = input[base + position * inner];
                ichunk[off] = position as i64;
            }
        }
    });
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
            par_out_chunks(spare, SIMD_PAR_CHUNK, &|start, chunk| {
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
            entries.par_sort_unstable();
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
/// columns, which is quicker than NumPy doing the same thing.
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
    let threads = rayon::current_num_threads();
    // `sort_along_dim_par` can only spread its work across `outer` positions,
    // so a tensor with few of them and many slices -- which is every sort along
    // the first axis, where `outer` is one -- leaves the pool idle. Rewriting
    // the axis as the last one costs a strided pass and buys all of it.
    if inner > 1 && outer < threads && slices.saturating_mul(dim_size) >= PAR_SORT_MIN_LEN {
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
    } else if slices < threads && dim_size >= PAR_SORT_MIN_LEN {
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
            true => run!(move |d: usize, v: T| ((!key(v) as u64) << 32) | d as u64),
            false => run!(move |d: usize, v: T| ((key(v) as u64) << 32) | d as u64),
        }
    } else {
        match descending {
            true => run!(move |d: usize, v: T| ((!key(v) as u128) << 64) | d as u128),
            false => run!(move |d: usize, v: T| ((key(v) as u128) << 64) | d as u128),
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
        true => run!(move |d: usize, v: T| ((!key(v) as u128) << 64) | d as u128),
        false => run!(move |d: usize, v: T| ((key(v) as u128) << 64) | d as u128),
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
        ($accessor:ident, $accessor_mut:ident, $ty:ty, $sum_by:ident) => {{
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
                //
                // Each run is reduced by the same lane-and-tree sum `sum`
                // itself uses, and not by a scalar `iter().sum()`. Blocking
                // alone was not enough: a naive total over one eight-thousand
                // element block still takes a rounding per element against a
                // magnitude that grows with the run, which left `var` four
                // orders worse than the `mean` it subtracts on data with a
                // large offset -- 3.4e-6 against 2.5e-10 at an offset of 1e12,
                // where the second figure is the floor set by the mean's own
                // rounding and is what NumPy reaches.
                let run = |first: usize, chunk: &mut [$ty]| {
                    for (i, slot) in chunk.iter_mut().enumerate() {
                        let base = (first + i) * dim_size;
                        let row = &input[base..base + dim_size];
                        let total = accurate_run_sum(row, |part: &[$ty]| $sum_by(part, |v| v));
                        let mean = total / n;
                        let acc = accurate_run_sum(row, |part: &[$ty]| {
                            $sum_by(part, |v| {
                                let d = v - mean;
                                d * d
                            })
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
                let outer = out.len() / inner;
                par_out_chunks(out, reduction_band(outer, inner), &|start, out_chunk| {
                    // With one outer position the columns are cut into bands
                    // instead, so a chunk is part of one block's row rather
                    // than all of it: the block is fixed and the chunk starts
                    // `start % inner` columns into it.
                    let block_base = (start / inner) * outer_stride + start % inner;
                    let width = out_chunk.len();
                    // Both passes are blocked for the same reason the
                    // contiguous-row path above uses `accurate_run_sum`.
                    let mut col_mean =
                        accurate_slab_sum(dim_size, width, 0.0 as $ty, |k, acc: &mut [$ty]| {
                            let base = block_base + k * inner;
                            let slab = &input[base..base + width];
                            for (m, &v) in acc.iter_mut().zip(slab) {
                                *m += v;
                            }
                        });
                    for m in col_mean.iter_mut() {
                        *m /= n;
                    }
                    let squared =
                        accurate_slab_sum(dim_size, width, 0.0 as $ty, |k, acc: &mut [$ty]| {
                            let base = block_base + k * inner;
                            let slab = &input[base..base + width];
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
        DataType::Float32 => fill!(as_f32_slice, as_f32_slice_mut, f32, simd_sum_f32_by),
        DataType::Float64 => fill!(as_f64_slice, as_f64_slice_mut, f64, simd_sum_f64_by),
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
    check_dim(dim, tensor.ndim())?;

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
    /// is the square of the mean's own rounding and is what NumPy reaches.
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
        for exponent in [10i32, 20, 30] {
            let offset = (2.0f64).powi(exponent);
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
        // its own, which squares into the answer. That floor is 1.1e-8 here,
        // and NumPy sits on it too.
        let shifted: Vec<f64> = base.iter().map(|v| v + (2.0f64).powi(40)).collect();
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

    /// The keys have to reproduce the float order exactly, including the two
    /// places the bit pattern and the numeric order disagree.
    #[test]
    fn float_keys_reproduce_the_float_order() {
        let ladder = [
            f32::NEG_INFINITY,
            -3.5,
            -1.0,
            -f32::MIN_POSITIVE,
            -0.0,
            0.0,
            f32::MIN_POSITIVE,
            1.0,
            3.5,
            f32::INFINITY,
        ];
        for pair in ladder.windows(2) {
            let (low, high) = (float_key32(pair[0]), float_key32(pair[1]));
            if pair[0] == pair[1] {
                // The two zeros: equal as floats, so equal as keys, so their
                // input order decides and nothing else can.
                assert_eq!(low, high, "{} and {} keyed apart", pair[0], pair[1]);
            } else {
                assert!(low < high, "{} keyed at or above {}", pair[0], pair[1]);
            }
        }

        // Every NaN, of either sign and any payload, is the one key above
        // every number -- which is where this library sorts them.
        for nan in [
            f32::NAN,
            -f32::NAN,
            f32::from_bits(0x7fc0_1234),
            f32::from_bits(0xffff_ffff),
        ] {
            assert_eq!(float_key32(nan), u32::MAX);
        }
        assert!(float_key32(f32::INFINITY) < u32::MAX);

        // The same at double width.
        for pair in [
            (f64::NEG_INFINITY, -1.0f64),
            (-1.0, -0.0),
            (0.0, 1.0),
            (1.0, f64::INFINITY),
        ] {
            assert!(float_key64(pair.0) < float_key64(pair.1));
        }
        assert_eq!(float_key64(-0.0), float_key64(0.0));
        assert_eq!(float_key64(f64::NAN), u64::MAX);
        assert_eq!(float_key64(-f64::NAN), u64::MAX);
    }

    #[test]
    fn integer_keys_reproduce_the_integer_order() {
        let ladder = [i32::MIN, -7, -1, 0, 1, 7, i32::MAX];
        for pair in ladder.windows(2) {
            assert!(int_key32(pair[0]) < int_key32(pair[1]));
        }
        assert_eq!(int_key32(i32::MIN), 0);
        assert_eq!(int_key32(i32::MAX), u32::MAX);

        let wide = [i64::MIN, -7, 0, 7, i64::MAX];
        for pair in wide.windows(2) {
            assert!(int_key64(pair[0]) < int_key64(pair[1]));
        }
        assert!(bool_key(false) < bool_key(true));
    }

    /// Both entry widths carry the same position back out.
    #[test]
    fn entries_give_their_position_back() {
        for position in [0usize, 1, 12345, u32::MAX as usize - 1] {
            let narrow = ((float_key32(1.5) as u64) << 32) | position as u64;
            assert_eq!(Entry::position(narrow), position);
        }
        for position in [0usize, 1, 12345, u32::MAX as usize + 1] {
            let wide = ((float_key64(1.5) as u128) << 64) | position as u128;
            assert_eq!(Entry::position(wide), position);
        }
    }

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
