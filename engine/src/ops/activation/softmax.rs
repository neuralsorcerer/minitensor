// Copyright (c) Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

use super::*;
use crate::error::MinitensorError;
use crate::error::Result;
use crate::ops::map::{PAR_THRESHOLD, par_map_indexed, par_out_chunks, par_out_chunks_mapped};
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
use std::mem::MaybeUninit;

/// The `exp(x - shift)` pass, which is where `softmax` spends about half of
/// its time.
///
/// It is behind a trait because the two float types answer it very
/// differently: float32 has a vectorized kernel for it and float64 has only
/// `libm`. Over four million elements the scalar loop measures 13.0 ms
/// against the kernel's 1.1 ms, which is worth one dispatch at the top of a
/// slice. The shift cannot be folded away -- it is what stops `exp`
/// overflowing on a large input -- so the kernel takes it rather than leaving
/// a subtraction behind for a scalar loop to do.
pub(crate) trait ShiftedExp: Float {
    /// `out[i] = exp(input[i] - shift)`, for every element of `out`.
    ///
    /// `input` and `out` are the same length and are different buffers.
    fn exp_shifted_into(input: &[Self], shift: Self, out: &mut [Self]);
}

impl ShiftedExp for f32 {
    fn exp_shifted_into(input: &[f32], shift: f32, out: &mut [f32]) {
        // The kernel writes through `MaybeUninit` because its other callers
        // hand it a fresh allocation. This one is already initialized, and an
        // initialized `T` is a valid `MaybeUninit<T>`; the kernel only writes,
        // so nothing here reads a value that is not there.
        let uninit = unsafe {
            std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut MaybeUninit<f32>, out.len())
        };
        crate::ops::simd::F32Kernel::select().exp_shifted(input, uninit, shift as f64);
    }
}

impl ShiftedExp for f64 {
    fn exp_shifted_into(input: &[f64], shift: f64, out: &mut [f64]) {
        for (o, &v) in out.iter_mut().zip(input.iter()) {
            *o = (v - shift).exp();
        }
    }
}

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
        crate::ops::util::log_add_exp::<f32>,
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
        crate::ops::util::log_add_exp::<f64>,
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

/// Where one softmax slice's mask entries live, and how far apart they are.
///
/// The lookup used to be a full linear-index decomposition per element --
/// `ndim` divisions to recover the coordinates, once in the max pass and again
/// in the exponential pass. A slice's mask entries are not scattered, though:
/// stepping one position along the softmax axis moves the mask index by a
/// constant, and that constant is *zero* whenever the mask is broadcast along
/// that axis, which is the usual case. A causal mask is `[l, s]` against
/// `[batch, heads, l, s]` scores.
///
/// So the decomposition happens once per slice rather than twice per element.
/// On eight by eight heads of 256 by 256 scores with a shared `[256, 256]`
/// mask, `masked_softmax` went from 23.4ms to 8.0 -- the same cost as the mask
/// already having the scores' own shape, which is what it should be.
struct MaskWalk<'a> {
    data: &'a [bool],
    dims: &'a [usize],
    mask_dims: &'a [usize],
    strides: Option<(Strides, Strides)>,
    step: usize,
}

impl<'a> MaskWalk<'a> {
    fn new(
        data: &'a [bool],
        tensor_shape: &'a Shape,
        mask_shape: &'a Shape,
        dim: usize,
        after: usize,
    ) -> Self {
        let dims = tensor_shape.dims();
        let mask_dims = mask_shape.dims();
        let strides = mask_strides_for(tensor_shape, mask_shape);
        let step = match &strides {
            // The mask has the tensor's own shape, so its index *is* the
            // tensor's and the axis stride is the tensor's.
            None => after,
            Some((_, mask_strides)) => {
                // A shorter mask lines up with the tensor from the right.
                let offset = dims.len() - mask_dims.len();
                match dim.checked_sub(offset) {
                    Some(axis) if mask_dims[axis] > 1 => mask_strides.as_slice()[axis],
                    // Broadcast along the softmax axis, or not covered by the
                    // mask at all: one entry serves the whole slice.
                    _ => 0,
                }
            }
        };
        Self {
            data,
            dims,
            mask_dims,
            strides,
            step,
        }
    }

    /// The mask index of position zero of the slice beginning at `linear`.
    #[inline]
    fn row(&self, linear: usize) -> usize {
        match &self.strides {
            Some((out_strides, mask_strides)) => broadcast_mask_index(
                linear,
                self.dims,
                out_strides.as_slice(),
                self.mask_dims,
                mask_strides.as_slice(),
            ),
            None => linear,
        }
    }

    /// Whether position `k` of the slice that starts at mask index `row` is
    /// masked out.
    #[inline(always)]
    fn masked(&self, row: usize, k: usize) -> bool {
        self.data[row + k * self.step]
    }
}

/// Column maxima of a `[dim_size, after]` row-major block.
///
/// Reading it a contiguous row at a time with `after`-sized accumulators makes
/// every access sequential, where the naive per-column loop strides by `after`
/// on every element. The result is the strided version's exactly: a maximum
/// does not care what order it sees its values in, NaNs included -- they fail
/// every comparison either way and so take no part.
fn block_col_max<T: Float>(in_block: &[T], after: usize) -> Vec<T> {
    let mut col_max = vec![T::neg_infinity(); after];
    for row in in_block.chunks_exact(after) {
        for (m, &v) in col_max.iter_mut().zip(row) {
            if v > *m {
                *m = v;
            }
        }
    }
    col_max
}

/// Write `exp(x - col_max)` into `out_block` and return the column sums of it.
///
/// The sums are filled in the pass that writes the exponentials: each `exp` is
/// computed once, stored, and added. Splitting the two apart costs the whole
/// kernel again -- recomputing `exp` for the sum ran 64% slower on
/// `softmax(dim=0)` of a 500000x3 tensor, and reading the stored value back in
/// a second pass 26%.
///
/// The rows are walked in order but added up in blocks (see [`slab_blocks`]),
/// which is the only reason this is more accurate than a strided walk rather
/// than identical to it.
fn block_exp_columns<T: Float>(
    in_block: &[T],
    out_block: &mut [T],
    col_max: &[T],
    dim_size: usize,
    after: usize,
) -> Vec<T> {
    let neg_inf = T::neg_infinity();
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
        return vec![T::zero(); after];
    }
    pairwise_fold_vectors(partials, |a, b| a + b)
}

/// Column-wise softmax of a `[dim_size, after]` row-major block (`after > 1`).
///
/// The softmax dimension is the outer (row) index, so the block is read down
/// its rows: maxima, then exponentials and their sums, then the division.
fn softmax_block_columnwise<T: Float>(
    in_block: &[T],
    out_block: &mut [T],
    dim_size: usize,
    after: usize,
) {
    let col_max = block_col_max(in_block, after);
    let col_sum = block_exp_columns(in_block, out_block, &col_max, dim_size, after);
    divide_columns(out_block, &col_sum, after);
}

/// Divide every row of a `[dim_size, after]` block by the column totals.
///
/// A zero total means the column had nothing to normalize -- it was empty or
/// all `-inf` -- and its zeros stay zeros. Every other total divides, a NaN
/// one included: a column holding a NaN or a `+inf` has no answer, and leaving
/// it undivided returns the raw exponentials instead, which look like an answer
/// and are not. `softmax([[1, 1], [nan, 2]], dim=0)` used to come back with
/// 1.0 in the poisoned column -- a column summing to 1 that means nothing.
fn divide_columns<T: Float>(out_block: &mut [T], col_sum: &[T], after: usize) {
    for out_row in out_block.chunks_exact_mut(after) {
        for (o, &s) in out_row.iter_mut().zip(col_sum) {
            if s != T::zero() {
                *o = *o / s;
            }
        }
    }
}

/// Fold each column's max into `log(sum) + max`, in place.
///
/// A column whose max is `-inf` has nothing to take the log of and stays
/// `-inf`, which makes every one of its outputs `-inf` in turn.
fn log_totals<T: Float>(col_max: &mut [T], col_sum: &[T]) {
    let neg_inf = T::neg_infinity();
    for (m, &s) in col_max.iter_mut().zip(col_sum) {
        if *m != neg_inf {
            *m = s.ln() + *m;
        }
    }
}

/// Every row of a `[dim_size, after]` block, less its column total.
fn subtract_columns<T: Float>(in_block: &[T], out_block: &mut [T], col_logsum: &[T], after: usize) {
    let neg_inf = T::neg_infinity();
    for (in_row, out_row) in in_block
        .chunks_exact(after)
        .zip(out_block.chunks_exact_mut(after))
    {
        for ((o, &v), &ls) in out_row.iter_mut().zip(in_row).zip(col_logsum) {
            *o = if ls == neg_inf { neg_inf } else { v - ls };
        }
    }
}

/// How many row bands to cut a softmax block into when the block is the whole
/// tensor.
///
/// The blocks of a softmax are its slices along the reduced axis, and one
/// per task is the natural decomposition -- until the reduced axis is the
/// first one, where there is exactly one block however large the tensor is and
/// the whole thing lands on one core. Its rows split cleanly instead: each band
/// reads its own rows and keeps its own column accumulators, and the bands
/// merge at the end.
///
/// The boundaries come from the block's shape alone -- never from the thread
/// count -- because here the partition decides how the column sums are grouped,
/// and a sum whose grouping follows the pool answers differently on different
/// machines. `SOFTMAX_PARTIAL_BUDGET` bounds what the accumulators cost: a band
/// holds `after` of them, so a wide block gets fewer bands rather than a
/// buffer the size of the tensor.
fn softmax_bands(dim_size: usize, after: usize) -> usize {
    /// Bands to aim for, so the pool stays fed on any machine.
    const TARGET_BANDS: usize = 64;
    /// Fewer than this is not worth the merge.
    const MIN_BANDS: usize = 4;
    /// Elements the per-band accumulators may take, all bands together.
    const PARTIAL_BUDGET: usize = 1 << 20;

    let numel = dim_size.saturating_mul(after);
    if numel < PAR_THRESHOLD {
        return 1;
    }
    let affordable = (PARTIAL_BUDGET / after.max(1)).max(MIN_BANDS);
    TARGET_BANDS.min(affordable).min(dim_size).max(1)
}

/// The column maxima of a block whose rows have been split into bands.
fn banded_col_max<T: Float + Send + Sync>(
    in_block: &[T],
    after: usize,
    band_rows: usize,
) -> Vec<T> {
    let stride = band_rows * after;
    let partials = par_map_indexed(in_block.len().div_ceil(stride), &|index| {
        let start = index * stride;
        let end = (start + stride).min(in_block.len());
        block_col_max(&in_block[start..end], after)
    });
    pairwise_fold_vectors(partials, |a, b| if b > a { b } else { a })
}

/// `softmax` and `log_softmax` share everything up to the last pass: the column
/// maxima and the column sums of `exp(x - max)`, with the exponentials left in
/// the output.
fn banded_columns<T: ShiftedExp + Send + Sync>(
    in_block: &[T],
    out_block: &mut [T],
    after: usize,
    band_rows: usize,
) -> (Vec<T>, Vec<T>) {
    let col_max = banded_col_max(in_block, after, band_rows);
    let stride = band_rows * after;
    let partials = par_out_chunks_mapped(out_block, stride, &|start, out_band| {
        let in_band = &in_block[start..start + out_band.len()];
        if after == 1 {
            // One column, so the band is a contiguous run of it and the
            // vectorized kernel applies.
            let max_val = col_max[0];
            if max_val == T::neg_infinity() {
                out_band.fill(T::zero());
                return vec![T::zero()];
            }
            T::exp_shifted_into(in_band, max_val, out_band);
            vec![accurate_indexed_sum(out_band.len(), T::zero(), |k| {
                out_band[k]
            })]
        } else {
            block_exp_columns(in_band, out_band, &col_max, out_band.len() / after, after)
        }
    });
    let col_sum = pairwise_fold_vectors(partials, |a, b| a + b);
    (col_max, col_sum)
}

/// `softmax` along `dim`, shifted by the per-slice max for numerical stability.
fn softmax_core<T: ShiftedExp + Send + Sync>(
    input_data: &[T],
    output_slice: &mut [T],
    dims: &[usize],
    dim: usize,
) -> Result<()> {
    let Some((dim_size, after, group)) = softmax_geometry(dims, dim) else {
        return Ok(());
    };
    let neg_inf = T::neg_infinity();

    // One block is one task, and a softmax along the first axis has exactly one
    // block however large the tensor is -- so it would run on one core with the
    // rest of the pool watching. Cut that block's rows up instead. Only that
    // case: with blocks to go around the split already has work for everyone,
    // and the bands cost a set of column accumulators each.
    let band_rows = dim_size.div_ceil(softmax_bands(dim_size, after));
    if output_slice.len() == group && band_rows < dim_size {
        let (_, col_sum) = banded_columns(input_data, output_slice, after, band_rows);
        par_out_chunks(output_slice, band_rows * after, &|_, out_band| {
            divide_columns(out_band, &col_sum, after);
        });
        return Ok(());
    }

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
            T::exp_shifted_into(in_block, max_val, out_block);
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
    let Some((dim_size, after, group)) = softmax_geometry(dims, dim) else {
        return Ok(());
    };
    let neg_inf = T::neg_infinity();

    // Resolving a mask position through one walker is what lets the passes
    // below read the same whether or not the mask is broadcast; spelling the
    // lookup out inline needed six copies of it.
    let walk = MaskWalk::new(mask_data, tensor_shape, mask_shape, dim, after);

    par_out_chunks(output_slice, group, &|block_offset, out_block| {
        let in_block = &input_data[block_offset..block_offset + out_block.len()];
        for base in 0..after {
            let row = walk.row(block_offset + base);
            let mut max_val = neg_inf;
            let mut has_unmasked = false;
            for k in 0..dim_size {
                if !walk.masked(row, k) {
                    has_unmasked = true;
                    let v = in_block[base + k * after];
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
                out_block[idx] = if walk.masked(row, k) {
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
fn log_softmax_core<T: ShiftedExp + Send + Sync>(
    input_data: &[T],
    output_slice: &mut [T],
    dims: &[usize],
    dim: usize,
) -> Result<()> {
    let Some((dim_size, after, group)) = softmax_geometry(dims, dim) else {
        return Ok(());
    };
    let neg_inf = T::neg_infinity();

    // The same one-block case `softmax_core` bands, for the same reason: along
    // the first axis the block split has nothing to hand out.
    let band_rows = dim_size.div_ceil(softmax_bands(dim_size, after));
    if output_slice.len() == group && band_rows < dim_size {
        // The exponentials land in the output on the way to their sums; the
        // final pass overwrites them from the input, so lending the buffer out
        // costs nothing.
        let (mut col_logsum, col_sum) = banded_columns(input_data, output_slice, after, band_rows);
        log_totals(&mut col_logsum, &col_sum);
        par_out_chunks(output_slice, band_rows * after, &|start, out_band| {
            let in_band = &input_data[start..start + out_band.len()];
            subtract_columns(in_band, out_band, &col_logsum, after);
        });
        return Ok(());
    }

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
            // The exponentials are formed into `out_block` and summed from
            // there rather than one at a time inside the sum: the vectorized
            // kernel needs somewhere to write, and the block's own final
            // values are computed from `in_block` below, so it is free to
            // borrow until then.
            T::exp_shifted_into(in_block, max_val, out_block);
            let sum = accurate_indexed_sum(out_block.len(), T::zero(), |k| out_block[k]);
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
    let Some((dim_size, after, group)) = softmax_geometry(dims, dim) else {
        return Ok(());
    };
    let neg_inf = T::neg_infinity();

    // One walker for the mask lookup, so the three passes below read the same
    // whether or not the mask is broadcast. Writing the branch at the top level
    // instead meant two copies of the whole kernel, each with three inlined
    // copies of this lookup.
    let walk = MaskWalk::new(mask_data, tensor_shape, mask_shape, dim, after);

    par_out_chunks(output_slice, group, &|block_offset, out_block| {
        let in_block = &input_data[block_offset..block_offset + out_block.len()];
        for base in 0..after {
            let row = walk.row(block_offset + base);
            let mut max_val = neg_inf;
            let mut has_unmasked = false;
            for k in 0..dim_size {
                if !walk.masked(row, k) {
                    has_unmasked = true;
                    let v = in_block[base + k * after];
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
                if walk.masked(row, k) {
                    T::zero()
                } else {
                    (in_block[base + k * after] - max_val).exp()
                }
            });
            let logsum = sum.ln() + max_val;
            for k in 0..dim_size {
                let idx = base + k * after;
                out_block[idx] = if walk.masked(row, k) {
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
