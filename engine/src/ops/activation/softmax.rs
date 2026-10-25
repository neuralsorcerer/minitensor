// Copyright (c) Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

use super::*;
use crate::error::MinitensorError;
use crate::error::Result;
use crate::ops::map::{
    EXPENSIVE_PAR_THRESHOLD, PAR_THRESHOLD, par_map_indexed, par_out_chunks, par_out_chunks_gated,
    par_out_chunks_mapped,
};
use crate::ops::util::{
    accurate_indexed_sum, broadcast_mask_index, pairwise_fold_vectors, stable_sigmoid_f64,
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
    /// Elements below which a softmax over this type runs on the calling
    /// thread: the threshold of the `exp` kernel it spends its time in.
    const SPLIT_THRESHOLD: usize;

    /// `out[i] = exp(input[i] - shift)`, for every element of `out`.
    ///
    /// `input` and `out` are the same length and are different buffers.
    fn exp_shifted_into(input: &[Self], shift: Self, out: &mut [Self]);

    /// `out[i] = exp(input[i] - shift[i])`, for every element of `out`: a shift
    /// per element rather than one for the slice, which is what reducing a
    /// non-last axis needs, where each column has its own maximum. `scratch`
    /// is working space; all four are the same length.
    fn exp_diff_into(input: &[Self], shift: &[Self], scratch: &mut [Self], out: &mut [Self]);

    /// The value in float64, where a softmax's total is carried.
    fn wide(self) -> f64;

    /// A float64 result rounded to this type, once.
    fn narrow(value: f64) -> Self;

    /// The sum of `terms` in float64, as a softmax's total is taken: summed
    /// in float32, 999 equal terms ran 125 to a lane and their total, and the
    /// log-probability over it, came out 2e-6 off.
    fn wide_sum(terms: &[Self]) -> f64;
}

impl ShiftedExp for f32 {
    const SPLIT_THRESHOLD: usize = crate::ops::map::VECTOR_F32_PAR_THRESHOLD;

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

    fn exp_diff_into(input: &[f32], shift: &[f32], scratch: &mut [f32], out: &mut [f32]) {
        for ((d, &x), &m) in scratch.iter_mut().zip(input).zip(shift) {
            *d = x - m;
        }
        // SAFETY: as in `exp_shifted_into`; `out` is initialized and the
        // kernel only writes it.
        let uninit = unsafe {
            std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut MaybeUninit<f32>, out.len())
        };
        crate::ops::simd::F32Kernel::select().exp(scratch, uninit);
    }

    #[inline(always)]
    fn wide(self) -> f64 {
        f64::from(self)
    }

    #[inline(always)]
    fn narrow(value: f64) -> f32 {
        value as f32
    }

    fn wide_sum(terms: &[f32]) -> f64 {
        crate::ops::simd::simd_sum_f32_wide(terms)
    }
}

impl ShiftedExp for f64 {
    const SPLIT_THRESHOLD: usize = crate::ops::map::EXPENSIVE_PAR_THRESHOLD;

    fn exp_shifted_into(input: &[f64], shift: f64, out: &mut [f64]) {
        for (o, &v) in out.iter_mut().zip(input.iter()) {
            *o = (v - shift).exp();
        }
    }

    fn exp_diff_into(input: &[f64], shift: &[f64], _scratch: &mut [f64], out: &mut [f64]) {
        for ((o, &x), &m) in out.iter_mut().zip(input).zip(shift) {
            *o = (x - m).exp();
        }
    }

    #[inline(always)]
    fn wide(self) -> f64 {
        self
    }

    #[inline(always)]
    fn narrow(value: f64) -> f64 {
        value
    }

    fn wide_sum(terms: &[f64]) -> f64 {
        accurate_indexed_sum(terms.len(), 0.0, |k| terms[k])
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

    // The vectorized kernel, broadcast or not, which took a million pairs from
    // 5.1ms of scalar `exp` and `ln_1p` to 1.15ms and rounds once from float64.
    let kernel = crate::ops::simd::F32Kernel::select();
    // SAFETY: the block closure is one `F32Kernel` method call, which writes
    // every element of the block it is handed.
    let out = unsafe {
        crate::ops::kernels::broadcast_binary_blocks(
            lhs_data,
            rhs_data,
            lhs.shape(),
            rhs.shape(),
            output_shape,
            crate::ops::map::VECTOR_F32_PAR_THRESHOLD,
            move |a, b, out| kernel.logaddexp(a, b, out),
        )
    };
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
    // Vectorized, and bit-for-bit what `tanh_promoted_f32` produced -- see
    // `ops::simd::transcendental`.
    // SAFETY: one `F32Kernel` method call; see `map_f32_kernel`.
    unsafe { map_f32_kernel(tensor, |kernel, src, dst| kernel.tanh(src, dst)) }
}

pub(crate) fn tanh_f64(tensor: &Tensor) -> Result<TensorData> {
    let input_data = tensor.data().as_f64_slice().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get f64 slice from input tensor")
    })?;

    let out = crate::ops::provider::offer_unary_f64(crate::ops::provider::Ufunc::Tanh, input_data)
        .unwrap_or_else(|| unary_map_threshold(input_data, EXPENSIVE_PAR_THRESHOLD, f64::tanh));
    Ok(TensorData::from_vec::<f64>(
        out,
        DataType::Float64,
        tensor.device(),
    ))
}

pub(crate) fn sigmoid_f32(tensor: &Tensor) -> Result<TensorData> {
    // Vectorized -- see `ops::simd::transcendental`. `e/(e+1)` rather than the
    // branchy stable form, which needed a sign test per element.
    // SAFETY: one `F32Kernel` method call; see `map_f32_kernel`.
    unsafe { map_f32_kernel(tensor, |kernel, src, dst| kernel.sigmoid(src, dst)) }
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
/// How many entries of a slice are `+inf`, and whether it holds a NaN.
///
/// A slice whose maximum is `+inf` has no shift that makes its exponentials
/// finite: `inf - inf` is NaN, and the whole slice came back NaN. Its softmax
/// is the limit instead -- see [`infinite_slice_value`].
#[inline]
fn infinite_slice_parts<T: Float>(values: impl Iterator<Item = T>) -> (usize, bool) {
    let mut infinities = 0;
    let mut nan = false;
    for v in values {
        if v == T::infinity() {
            infinities += 1;
        } else if v.is_nan() {
            nan = true;
        }
    }
    (infinities, nan)
}

/// One entry of the softmax (or, with `LOG`, the `log_softmax`) of a slice
/// whose maximum is `+inf`, from [`infinite_slice_parts`].
///
/// As those entries grow without bound they take all of the mass, and every
/// other entry none: `softmax([inf, 1, 2])` is `[1, 0, 0]`. Several equal
/// infinities share it evenly, as entries that compare equal do everywhere
/// else, so each of `k` gets `1/k` -- `-log k` in the log form. A NaN in the
/// slice still makes all of it NaN.
#[inline]
fn infinite_slice_value<T: Float, const LOG: bool>(v: T, infinities: usize, nan: bool) -> T {
    if nan {
        return T::nan();
    }
    if v == T::infinity() {
        let share = T::from(infinities).unwrap_or_else(T::nan);
        // `0 - log k` rather than `-log k`, which is `-0.0` for one infinity.
        if LOG {
            T::zero() - share.ln()
        } else {
            share.recip()
        }
    } else if LOG {
        T::neg_infinity()
    } else {
        T::zero()
    }
}

/// Rewrite every column of a `[rows, after]` block whose maximum is `+inf`
/// with its limit; see [`infinite_slice_value`]. Nothing to do, and one pass
/// over `col_max` to find that out, when no column has one.
fn settle_infinite_columns<T: Float, const LOG: bool>(
    in_block: &[T],
    out_block: &mut [T],
    col_max: &[T],
    after: usize,
) {
    for (a, &max) in col_max.iter().enumerate() {
        if max != T::infinity() {
            continue;
        }
        let (infinities, nan) =
            infinite_slice_parts(in_block.iter().skip(a).step_by(after).copied());
        for (o, &v) in out_block
            .iter_mut()
            .skip(a)
            .step_by(after)
            .zip(in_block.iter().skip(a).step_by(after))
        {
            *o = infinite_slice_value::<T, LOG>(v, infinities, nan);
        }
    }
}

/// The total of a slice's `exp(x - max)`, split for its logarithm: how many
/// of the terms are exactly one -- the maximum's own always is -- and the sum
/// of the rest.
///
/// A log-softmax is `x - max - log(total)`, and the total of a slice with one
/// dominant entry is `1 + rest` for a small `rest`. Taken whole, its logarithm
/// keeps only the digits of `rest` that survived being added to one, and
/// `x - (max + log(total))` cancels the rest away for the maximum's own entry:
/// the float32 log-probability of a confident prediction was 7% off, and every
/// cross-entropy built on it, and the float64 one 5e-9. `log1p(rest)` keeps
/// them, and `(x - max) - log1p(rest)` subtracts nothing that cancels.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct ExpTotal<T> {
    /// How many terms are exactly one.
    pub(crate) ones: T,
    /// The sum of the others.
    pub(crate) rest: T,
}

impl<T: Float> ExpTotal<T> {
    pub(crate) fn zero() -> Self {
        Self {
            ones: T::zero(),
            rest: T::zero(),
        }
    }

    /// One term, filed under `ones` or `rest`.
    #[inline(always)]
    pub(crate) fn of(term: T) -> Self {
        let unit = term == T::one();
        Self {
            ones: if unit { T::one() } else { T::zero() },
            rest: if unit { T::zero() } else { term },
        }
    }

    /// The whole total.
    pub(crate) fn sum(self) -> T {
        self.ones + self.rest
    }

    /// `log` of the whole total: `log1p(rest)` over a single unit term, the
    /// case where taking the total first loses digits. Over several the total
    /// is at least two and its logarithm loses nothing.
    pub(crate) fn ln(self) -> T {
        if self.ones == T::one() {
            log1p_of_rest(self.rest)
        } else {
            self.sum().ln()
        }
    }
}

/// The first position of `target` in `values`, found a block at a time: each
/// block is tested whole, which vectorizes, and only the one that holds it is
/// searched. Element by element, the early exit kept the test scalar, and on
/// rows of 64 it cost a fifth of a `logsumexp`.
#[inline]
pub(crate) fn first_position<T: Float>(values: &[T], target: T) -> Option<usize> {
    const BLOCK: usize = 16;
    for (index, block) in values.chunks(BLOCK).enumerate() {
        if block.iter().fold(false, |found, &v| found | (v == target)) {
            return block
                .iter()
                .position(|&v| v == target)
                .map(|at| index * BLOCK + at);
        }
    }
    None
}

/// `log(1 + rest)`, through `log1p` where that keeps digits -- a `rest` below
/// one, where `1 + rest` rounds part of it away -- and as the logarithm of the
/// sum above, where the sum is at least two and a logarithm of it is at least
/// `log 2`: the rounding of `1 + rest` moves it by under an ulp, and the call
/// costs half of `log1p`'s, once a row.
#[inline]
pub(crate) fn log1p_of_rest<T: Float>(rest: T) -> T {
    if rest < T::one() {
        rest.ln_1p()
    } else {
        (T::one() + rest).ln()
    }
}

impl<T: Float> std::ops::Add for ExpTotal<T> {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self {
            ones: self.ones + other.ones,
            rest: self.rest + other.rest,
        }
    }
}

/// The [`ExpTotal`] of a run of terms, in float64; with `SPLIT` false every
/// term counts as `rest`, which is the plain sum.
fn run_total<T: ShiftedExp, const SPLIT: bool>(terms: &[T]) -> ExpTotal<f64> {
    if SPLIT {
        accurate_indexed_sum(terms.len(), ExpTotal::zero(), |k| {
            ExpTotal::of(terms[k].wide())
        })
    } else {
        ExpTotal {
            ones: 0.0,
            rest: T::wide_sum(terms),
        }
    }
}

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

/// Rows are batched until one call of the exponential kernel covers at least
/// this many elements.
pub(crate) const EXP_GROUP: usize = 512;
/// The rounding chain each column accumulator lane is held to.
pub(crate) const EXP_BLOCK_ROWS: usize = 128;

/// Fold lanes laid out `cols` to a row back down to one value per column.
#[inline(always)]
pub(crate) fn lanes_to_columns<T: Copy>(
    lanes: &[T],
    cols: usize,
    merge: impl Fn(T, T) -> T,
) -> Vec<T> {
    let mut columns = lanes[..cols].to_vec();
    for tile in lanes[cols..].chunks_exact(cols) {
        for (c, &v) in columns.iter_mut().zip(tile) {
            *c = merge(*c, v);
        }
    }
    columns
}

/// `exp(x - shift[column])` over a row-major `(rows, cols)` block, written to
/// `out` when there is one, and the column sums of it.
///
/// A column whose shift is `-inf` is all `-inf` (or empty) and gets zeros,
/// which then drop out of its sum -- the contiguous paths' negative-infinity
/// short-circuit.
///
/// The exponentials go through the vectorized kernel a group of rows at a
/// time, the shift tiled to match: one kernel call per `EXP_GROUP` elements
/// however narrow the rows, where a call per row of three would cost more
/// than the arithmetic. The sums are computed in the pass that writes the
/// exponentials -- each `exp` computed once, stored, and added; splitting the
/// two apart cost `softmax(dim=0)` of a 500000x3 tensor 64% when recomputing
/// and 26% when reading the stored value back -- and each accumulator lane
/// runs `EXP_BLOCK_ROWS` additions before the blocks fold pairwise.
///
/// This replaced a scalar `exp` per element in the column paths of `softmax`,
/// `log_softmax` and `logsumexp`.
pub(crate) fn exp_columns_into<T: ShiftedExp, const SPLIT: bool>(
    input: &[T],
    out: Option<&mut [T]>,
    shift: &[T],
    cols: usize,
) -> Vec<ExpTotal<f64>> {
    // The float64 lanes are half the width of the float32 ones; built with
    // AVX2 they are as wide again, as `last_axis_rows` is built.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    fn avx2<T: ShiftedExp, const SPLIT: bool>(
        input: &[T],
        out: Option<&mut [T]>,
        shift: &[T],
        cols: usize,
    ) -> Vec<ExpTotal<f64>> {
        exp_columns_body::<T, SPLIT>(input, out, shift, cols)
    }
    #[cfg(target_arch = "x86_64")]
    if crate::ops::simd::simd_capabilities().avx2_fma {
        // SAFETY: avx2 and fma were detected on this CPU.
        return unsafe { avx2::<T, SPLIT>(input, out, shift, cols) };
    }
    exp_columns_body::<T, SPLIT>(input, out, shift, cols)
}

#[inline(always)]
fn exp_columns_body<T: ShiftedExp, const SPLIT: bool>(
    input: &[T],
    mut out: Option<&mut [T]>,
    shift: &[T],
    cols: usize,
) -> Vec<ExpTotal<f64>> {
    let group = EXP_GROUP.div_ceil(cols) * cols;
    let tiled: Vec<T> = shift.iter().copied().cycle().take(group).collect();
    let dead: Vec<usize> = (0..cols)
        .filter(|&a| shift[a] == T::neg_infinity())
        .collect();
    let mut scratch = vec![T::zero(); group];
    let mut spare = if out.is_none() {
        vec![T::zero(); group]
    } else {
        Vec::new()
    };
    let block_len = group * EXP_BLOCK_ROWS;
    // The lanes are float64 whatever the input, as every softmax total is;
    // see `ShiftedExp::wide_sum`.
    let blocks: Vec<Vec<ExpTotal<f64>>> = input
        .chunks(block_len)
        .enumerate()
        .map(|(b, block)| {
            let mut lanes = vec![0.0f64; group];
            // Counted in the input's dtype, where the selects stay as wide as
            // the terms; a block's count is at most `EXP_BLOCK_ROWS`.
            let mut units = vec![T::zero(); if SPLIT { group } else { 0 }];
            for (g, part) in block.chunks(group).enumerate() {
                let n = part.len();
                let at = b * block_len + g * group;
                let terms: &mut [T] = match out.as_deref_mut() {
                    Some(out) => &mut out[at..at + n],
                    None => &mut spare[..n],
                };
                T::exp_diff_into(part, &tiled[..n], &mut scratch[..n], terms);
                if !dead.is_empty() {
                    for row in terms.chunks_exact_mut(cols) {
                        for &a in &dead {
                            row[a] = T::zero();
                        }
                    }
                }
                if SPLIT {
                    for ((lane, unit), &t) in lanes.iter_mut().zip(units.iter_mut()).zip(&*terms) {
                        let one = t == T::one();
                        *lane += (if one { T::zero() } else { t }).wide();
                        *unit = *unit + if one { T::one() } else { T::zero() };
                    }
                } else {
                    for (lane, &t) in lanes.iter_mut().zip(terms.iter()) {
                        *lane += t.wide();
                    }
                }
            }
            let rest = lanes_to_columns(&lanes, cols, |a, b| a + b);
            let ones = if SPLIT {
                lanes_to_columns(&units, cols, |a, b| a + b)
                    .into_iter()
                    .map(T::wide)
                    .collect()
            } else {
                vec![0.0; cols]
            };
            ones.into_iter()
                .zip(rest)
                .map(|(ones, rest)| ExpTotal { ones, rest })
                .collect()
        })
        .collect();
    if blocks.is_empty() {
        vec![ExpTotal::zero(); cols]
    } else {
        pairwise_fold_vectors(blocks, |a, b| a + b)
    }
}

/// Column-wise softmax of a `[dim_size, after]` row-major block (`after > 1`).
///
/// The softmax dimension is the outer (row) index, so the block is read down
/// its rows: maxima, then exponentials and their sums, then the division.
fn softmax_block_columnwise<T: ShiftedExp>(in_block: &[T], out_block: &mut [T], after: usize) {
    let col_max = block_col_max(in_block, after);
    let col_sum: Vec<f64> =
        exp_columns_into::<T, false>(in_block, Some(out_block), &col_max, after)
            .into_iter()
            .map(ExpTotal::sum)
            .collect();
    divide_columns(out_block, &col_sum, after);
    settle_infinite_columns::<T, false>(in_block, out_block, &col_max, after);
}

/// Divide every row of a `[dim_size, after]` block by the column totals.
///
/// A zero total means the column had nothing to normalize -- it was empty or
/// all `-inf` -- and its zeros stay zeros. Every other total divides, a NaN
/// one included: a column holding a NaN or a `+inf` has no answer, and leaving
/// it undivided returns the raw exponentials instead, which look like an answer
/// and are not. `softmax([[1, 1], [nan, 2]], dim=0)` used to come back with
/// 1.0 in the poisoned column -- a column summing to 1 that means nothing.
///
/// The quotient is formed in float64 against the float64 total, and rounded
/// once.
fn divide_columns<T: ShiftedExp>(out_block: &mut [T], col_sum: &[f64], after: usize) {
    for out_row in out_block.chunks_exact_mut(after) {
        for (o, &s) in out_row.iter_mut().zip(col_sum) {
            if s != 0.0 {
                *o = T::narrow(o.wide() / s);
            }
        }
    }
}

/// Each column's `log(total)`, to come off `x - max`; a column whose max is
/// `-inf` has no total and its outputs are all `-inf`.
fn log_offsets<T: Float>(col_max: &[T], totals: &[ExpTotal<f64>]) -> Vec<f64> {
    col_max
        .iter()
        .zip(totals)
        .map(|(&m, &t)| {
            if m == T::neg_infinity() {
                f64::NEG_INFINITY
            } else {
                t.ln()
            }
        })
        .collect()
}

/// Every row of a `[dim_size, after]` block, as `(x - max) - log(total)`; see
/// [`ExpTotal`].
fn subtract_columns<T: ShiftedExp>(
    in_block: &[T],
    out_block: &mut [T],
    col_max: &[T],
    offsets: &[f64],
    after: usize,
) {
    let neg_inf = T::neg_infinity();
    let offsets: Vec<T> = offsets.iter().map(|&l| T::narrow(l)).collect();
    for (in_row, out_row) in in_block
        .chunks_exact(after)
        .zip(out_block.chunks_exact_mut(after))
    {
        for (((o, &v), &m), &l) in out_row.iter_mut().zip(in_row).zip(col_max).zip(&offsets) {
            *o = if m == neg_inf { neg_inf } else { (v - m) - l };
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
fn banded_columns<T: ShiftedExp + Send + Sync, const SPLIT: bool>(
    in_block: &[T],
    out_block: &mut [T],
    after: usize,
    band_rows: usize,
) -> (Vec<T>, Vec<ExpTotal<f64>>) {
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
                return vec![ExpTotal::zero()];
            }
            T::exp_shifted_into(in_band, max_val, out_band);
            vec![run_total::<T, SPLIT>(out_band)]
        } else {
            exp_columns_into::<T, SPLIT>(in_band, Some(out_band), &col_max, after)
        }
    });
    let col_sum = pairwise_fold_vectors(partials, |a, b| a + b);
    (col_max, col_sum)
}

/// The largest value in `values` that is not NaN, or `-inf` if there is none.
///
/// Across eight lanes, so the comparison vectorizes: one running maximum is a
/// chain of dependent compare-and-selects, and on a softmax over rows of 64 it
/// cost more than the exponentials. The maximum of a set does not depend on
/// the order it is taken in, so this is the value the single chain found.
#[inline]
pub(crate) fn slice_max<T: Float>(values: &[T]) -> T {
    const LANES: usize = 8;
    let mut lanes = [T::neg_infinity(); LANES];
    let (blocks, tail) = values.as_chunks::<LANES>();
    for block in blocks {
        for (lane, &v) in lanes.iter_mut().zip(block) {
            if v > *lane {
                *lane = v;
            }
        }
    }
    let mut max_val = T::neg_infinity();
    for &v in lanes.iter().chain(tail) {
        if v > max_val {
            max_val = v;
        }
    }
    max_val
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

    // One block is one task, and a softmax along the first axis has exactly one
    // block however large the tensor is -- so it would run on one core with the
    // rest of the pool watching. Cut that block's rows up instead. Only that
    // case: with blocks to go around the split already has work for everyone,
    // and the bands cost a set of column accumulators each.
    let band_rows = dim_size.div_ceil(softmax_bands(dim_size, after));
    if output_slice.len() == group && band_rows < dim_size {
        let (col_max, totals) =
            banded_columns::<T, false>(input_data, output_slice, after, band_rows);
        let col_sum: Vec<f64> = totals.into_iter().map(ExpTotal::sum).collect();
        par_out_chunks(output_slice, band_rows * after, &|_, out_band| {
            divide_columns(out_band, &col_sum, after);
        });
        settle_infinite_columns::<T, false>(input_data, output_slice, &col_max, after);
        return Ok(());
    }

    if after == 1 {
        // Softmax over the last (contiguous) dimension: each row is a single
        // slice laid out contiguously.
        par_out_chunks_gated(
            output_slice,
            rows_per_task(dim_size) * dim_size,
            T::SPLIT_THRESHOLD,
            &|offset, out_rows| {
                let in_rows = &input_data[offset..offset + out_rows.len()];
                last_axis_rows::<T, false>(in_rows, out_rows, dim_size);
            },
        );
        return Ok(());
    }

    par_out_chunks_gated(
        output_slice,
        group,
        T::SPLIT_THRESHOLD,
        &|block_offset, out_block| {
            let in_block = &input_data[block_offset..block_offset + out_block.len()];
            // Softmax over a non-last dimension: the block is a
            // `[dim_size, after]` row-major matrix and the reduction runs down
            // the rows. Column accumulators keep every pass contiguous instead
            // of striding by `after` per element.
            softmax_block_columnwise(in_block, out_block, after);
        },
    );

    Ok(())
}

/// Rows of `len` one task of a last-axis softmax takes: enough that a task is
/// at least [`EXP_GROUP`] elements, so a short row is not a call of its own.
fn rows_per_task(len: usize) -> usize {
    EXP_GROUP.div_ceil(len.max(1))
}

/// `softmax` (or, with `LOG`, `log_softmax`) of each `len`-long row of `input`
/// into the same row of `out`.
///
/// A row's passes -- its maximum, the shifted exponentials, their sum and the
/// division -- are each a few instructions an element, so over the rows of 64
/// an attention layer has, what they cost was mostly getting to them: a call
/// per row, and three of the four loops compiled for the baseline, where only
/// the exponential has a vectorized kernel. A run of rows in one call, built
/// with AVX2 where the CPU has it, and the maximum taken across lanes (see
/// [`slice_max`]) took a `[8192, 64]` softmax from 0.60-0.70ms to 0.42-0.45
/// on four threads, and rows of 8192 from 0.57-0.68 to 0.35-0.58. What is
/// left is the exponential and the division, which is at the rate the
/// hardware divides. Every row is computed exactly as it was one at a time,
/// to the bit.
fn last_axis_rows<T: ShiftedExp, const LOG: bool>(input: &[T], out: &mut [T], len: usize) {
    #[inline(always)]
    fn body<T: ShiftedExp, const LOG: bool>(input: &[T], out: &mut [T], len: usize) {
        for (in_row, out_row) in input.chunks_exact(len).zip(out.chunks_exact_mut(len)) {
            let max_val = slice_max(in_row);
            if max_val == T::neg_infinity() {
                out_row.fill(if LOG { T::neg_infinity() } else { T::zero() });
                continue;
            }
            if max_val == T::infinity() {
                let (infinities, nan) = infinite_slice_parts(in_row.iter().copied());
                for (o, &v) in out_row.iter_mut().zip(in_row) {
                    *o = infinite_slice_value::<T, LOG>(v, infinities, nan);
                }
                continue;
            }
            // The exponentials are formed into the output row and summed from
            // there: the vectorized kernel needs somewhere to write, and for
            // `log_softmax` the row's final values are computed from `in_row`
            // below, so it is free to borrow until then.
            T::exp_shifted_into(in_row, max_val, out_row);
            // Blocked: a running total over a long axis loses the small terms,
            // and every term here but the largest *is* small. Over a 250k-class
            // vocabulary the probabilities came back summing to 1.0004 rather
            // than 1, a relative error of 4.2e-4; blocked, it is 1.0e-7.
            if LOG {
                // The maximum's own term is `exp(0) = 1` exactly; taken out
                // of the sum, the rest is what `log1p` wants, and the max
                // comes off each value before the logarithm does. See
                // `ExpTotal`. One position of the maximum is enough: any
                // other term that is one stays in the rest, where `log1p`
                // loses nothing over it.
                let top = first_position(in_row, max_val);
                if let Some(top) = top {
                    out_row[top] = T::zero();
                }
                // The total in float64 lanes, inline: a call to the widened
                // kernel per row of 64 cost more than the row.
                let rest = accurate_indexed_sum(len, 0.0f64, |k| out_row[k].wide());
                let offset = T::narrow(log1p_of_rest(rest));
                for (o, &v) in out_row.iter_mut().zip(in_row) {
                    *o = (v - max_val) - offset;
                }
            } else {
                // Each term times the float64 reciprocal of the float64 total,
                // rounded once: as near the quotient as dividing, at the cost
                // of a multiply.
                let scale = accurate_indexed_sum(len, 0.0f64, |k| out_row[k].wide()).recip();
                for o in out_row.iter_mut() {
                    *o = T::narrow(o.wide() * scale);
                }
            }
        }
    }
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    fn body_avx2<T: ShiftedExp, const LOG: bool>(input: &[T], out: &mut [T], len: usize) {
        body::<T, LOG>(input, out, len)
    }
    #[cfg(target_arch = "x86_64")]
    if crate::ops::simd::simd_capabilities().avx2_fma {
        // SAFETY: avx2 and fma were detected on this CPU.
        return unsafe { body_avx2::<T, LOG>(input, out, len) };
    }
    body::<T, LOG>(input, out, len)
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

    par_out_chunks_gated(
        output_slice,
        group,
        EXPENSIVE_PAR_THRESHOLD,
        &|block_offset, out_block| {
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
                if max_val == T::infinity() {
                    let (infinities, nan) = infinite_slice_parts(
                        (0..dim_size)
                            .filter(|&k| !walk.masked(row, k))
                            .map(|k| in_block[base + k * after]),
                    );
                    for k in 0..dim_size {
                        let idx = base + k * after;
                        out_block[idx] = if walk.masked(row, k) {
                            T::zero()
                        } else {
                            infinite_slice_value::<T, false>(in_block[idx], infinities, nan)
                        };
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
                // Totalled and divided in float64, rounded once; see
                // `ShiftedExp::wide_sum`.
                let wide = |v: T| v.to_f64().unwrap_or(f64::NAN);
                let sum =
                    accurate_indexed_sum(dim_size, 0.0f64, |k| wide(out_block[base + k * after]));
                if sum != 0.0 {
                    for k in 0..dim_size {
                        let idx = base + k * after;
                        out_block[idx] = T::from(wide(out_block[idx]) / sum).unwrap_or_else(T::nan);
                    }
                }
            }
        },
    );

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
        let (col_max, totals) =
            banded_columns::<T, true>(input_data, output_slice, after, band_rows);
        let offsets = log_offsets(&col_max, &totals);
        par_out_chunks(output_slice, band_rows * after, &|start, out_band| {
            let in_band = &input_data[start..start + out_band.len()];
            subtract_columns(in_band, out_band, &col_max, &offsets, after);
        });
        settle_infinite_columns::<T, true>(input_data, output_slice, &col_max, after);
        return Ok(());
    }

    if after == 1 {
        // Log-softmax over the last (contiguous) dimension.
        par_out_chunks_gated(
            output_slice,
            rows_per_task(dim_size) * dim_size,
            T::SPLIT_THRESHOLD,
            &|offset, out_rows| {
                let in_rows = &input_data[offset..offset + out_rows.len()];
                last_axis_rows::<T, true>(in_rows, out_rows, dim_size);
            },
        );
        return Ok(());
    }

    par_out_chunks_gated(
        output_slice,
        group,
        T::SPLIT_THRESHOLD,
        &|block_offset, out_block| {
            let in_block = &input_data[block_offset..block_offset + out_block.len()];
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
            let col_max = col_logsum;
            let totals = exp_columns_into::<T, true>(in_block, None, &col_max, after);
            // A `-inf` column stays `-inf`, so all its outputs are `-inf`.
            let offsets = log_offsets(&col_max, &totals);
            subtract_columns(in_block, out_block, &col_max, &offsets, after);
            settle_infinite_columns::<T, true>(in_block, out_block, &col_max, after);
        },
    );

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

    par_out_chunks_gated(
        output_slice,
        group,
        EXPENSIVE_PAR_THRESHOLD,
        &|block_offset, out_block| {
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
                if max_val == T::infinity() {
                    let (infinities, nan) = infinite_slice_parts(
                        (0..dim_size)
                            .filter(|&k| !walk.masked(row, k))
                            .map(|k| in_block[base + k * after]),
                    );
                    for k in 0..dim_size {
                        let idx = base + k * after;
                        out_block[idx] = if walk.masked(row, k) {
                            neg_inf
                        } else {
                            infinite_slice_value::<T, true>(in_block[idx], infinities, nan)
                        };
                    }
                    continue;
                }
                let offset = accurate_indexed_sum(dim_size, ExpTotal::zero(), |k| {
                    if walk.masked(row, k) {
                        ExpTotal::zero()
                    } else {
                        let term = (in_block[base + k * after] - max_val).exp();
                        ExpTotal::of(term.to_f64().unwrap_or(f64::NAN))
                    }
                })
                .ln();
                let top = max_val.to_f64().unwrap_or(f64::NAN);
                for k in 0..dim_size {
                    let idx = base + k * after;
                    out_block[idx] = if walk.masked(row, k) {
                        neg_inf
                    } else {
                        let v = in_block[idx].to_f64().unwrap_or(f64::NAN);
                        T::from((v - top) - offset).unwrap_or_else(T::nan)
                    };
                }
            }
        },
    );

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
