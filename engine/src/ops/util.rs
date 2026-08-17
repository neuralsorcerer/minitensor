// Copyright (c) Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

//! Small helpers shared across op clusters and their gradient functions.
//!
//! Everything here was previously defined more than once — the same body
//! copied into a forward kernel and again into the matching backward kernel,
//! where the two copies could drift apart without anything noticing.

use crate::{
    device::Device,
    error::{MinitensorError, Result},
    tensor::{DataType, Shape, Tensor, TensorData},
};
use std::sync::Arc;

/// Resolve a possibly negative dimension index against `ndim`, erroring when
/// it falls outside `[-ndim, ndim)`. Shared by the shape, linalg, and
/// reduction clusters.
pub fn normalize_dim(dim: isize, ndim: usize) -> Result<usize> {
    let resolved = if dim < 0 { dim + ndim as isize } else { dim };
    if resolved < 0 || resolved >= ndim as isize {
        // Report what the caller passed, not what it resolved to: `-4` on a
        // 3-D tensor is a mistake about `-4`, and being told that `-1` is out
        // of bounds sends the reader looking for a different bug.
        Err(MinitensorError::dim_out_of_range(dim, ndim))
    } else {
        Ok(resolved as usize)
    }
}

/// [`normalize_dim`] for calls that take more than one dimension, naming the
/// argument at fault.
///
/// `transpose(-5, 1)` used to report the *larger* of the two after resolving
/// both, which named `1` -- the argument that was fine -- as the problem.
pub fn normalize_dim_named(dim: isize, ndim: usize, argument: &'static str) -> Result<usize> {
    let resolved = if dim < 0 { dim + ndim as isize } else { dim };
    if resolved < 0 || resolved >= ndim as isize {
        Err(MinitensorError::dim_out_of_range_with_context(
            dim, ndim, argument,
        ))
    } else {
        Ok(resolved as usize)
    }
}

/// Sum a float slice in parallel with a result that does not depend on how
/// rayon schedules the work.
///
/// `par_iter().sum()` and `par_chunks(n).map(..).sum()` both look deterministic
/// and are not. Chunking fixes the accumulation order *inside* a chunk, but
/// `sum()` on a parallel iterator folds the chunk partials together in
/// split-and-steal order, which varies between runs. Floating point addition is
/// not associative, so the last bits of the total move with it: summing 10^7
/// `f32` values here produced several distinct results across repeated calls on
/// the same input, which is enough to make a seeded training run
/// irreproducible.
///
/// Collecting the partials first pins them to chunk order -- `collect` on an
/// indexed parallel iterator is order-preserving -- so the combination step is
/// then fully determined by the input length. The extra allocation is one
/// element per chunk (about 1200 floats for a 10^7-element input), which does
/// not measurably change the timing.
///
/// `sum_chunk` may widen (`&[f32] -> f64`), which is how the gradient-norm
/// accumulator squares `f32` parameters into an `f64` total.
pub(crate) fn deterministic_par_sum<T, U, F>(data: &[T], chunk: usize, sum_chunk: F) -> U
where
    T: Sync,
    U: Copy + Send + Default + std::ops::Add<Output = U>,
    F: Fn(&[T]) -> U + Send + Sync,
{
    use rayon::prelude::*;
    let partials: Vec<U> = data.par_chunks(chunk).map(&sum_chunk).collect();
    pairwise_fold(partials, U::default(), |a, b| a + b)
}

/// The dtype an *accumulating* reduction reports.
///
/// `sum`, `nansum`, `prod`, `cumsum` and `cumprod` build a running total, so
/// the output can leave the range of the input long before the input itself is
/// remarkable: summing three billion-ish `int32` values overflows, and the
/// answer that came back was `1705032704` rather than `6000000000`. NumPy and
/// PyTorch both widen narrow integers to 64 bits for exactly these operations,
/// and this matches them.
///
/// It is only the accumulating reductions. `max`, `min`, `argmax` and the
/// quantiles report a value that was already in the input, so widening them
/// would be noise -- NumPy keeps the input dtype there too.
///
/// Floats are unchanged. Promoting `f32` to `f64` would silently alter every
/// existing result and double the memory of the most common reduction in the
/// library; NumPy does not do it either.
///
/// `Bool` was already handled this way for `sum` and `cumsum` before this
/// existed, by casting to `Int64` up front -- counting a mask is the usual
/// reason to sum one. `prod` and `cumprod` had been left behind, so
/// `mask.prod()` came back as `Bool`; routing every accumulating reduction
/// through here is what makes them agree.
pub(crate) fn accumulating_dtype(dtype: DataType) -> DataType {
    match dtype {
        DataType::Int32 | DataType::Bool => DataType::Int64,
        other => other,
    }
}

/// How a reduction or scan accumulates, per dtype.
///
/// Integer accumulation wraps; float accumulation is the ordinary operator.
/// The distinction has to live somewhere, because the kernels that do the
/// accumulating -- `sum_along_dim`, `prod_along_dim`, the cumulative scans, the
/// lane folds -- are written once and instantiated for every dtype, so a bare
/// `+` in one of them means two different things depending on which type it
/// lands on and which profile it was built under.
///
/// Rust's `+`, `-` and `*` on integers panic on overflow when overflow checks
/// are on and wrap when they are not, so `sum` of an int32 tensor that
/// overflows aborted under `cargo test` and returned a wrapped value from the
/// released wheel. Naming the wrap makes every build agree on the answer the
/// release build was already giving, which is also what an integer tensor is
/// expected to do.
///
/// The float impls are `self + other` and `self * other` exactly, so no
/// summation order and no rounding changes anywhere -- which matters, because
/// the order these kernels accumulate in is pinned by `tests/determinism.rs`.
pub(crate) trait Accumulate: Copy {
    /// `self + other`, wrapping for integers.
    fn acc_add(self, other: Self) -> Self;
    /// `self * other`, wrapping for integers.
    fn acc_mul(self, other: Self) -> Self;
}

macro_rules! impl_accumulate_float {
    ($ty:ty) => {
        impl Accumulate for $ty {
            #[inline(always)]
            fn acc_add(self, other: Self) -> Self {
                self + other
            }
            #[inline(always)]
            fn acc_mul(self, other: Self) -> Self {
                self * other
            }
        }
    };
}

macro_rules! impl_accumulate_int {
    ($ty:ty) => {
        impl Accumulate for $ty {
            #[inline(always)]
            fn acc_add(self, other: Self) -> Self {
                self.wrapping_add(other)
            }
            #[inline(always)]
            fn acc_mul(self, other: Self) -> Self {
                self.wrapping_mul(other)
            }
        }
    };
}

impl_accumulate_float!(f32);
impl_accumulate_float!(f64);
impl_accumulate_int!(i32);
impl_accumulate_int!(i64);

/// Accumulate `inner` parallel totals across `dim_size` slab steps, with the
/// error growth of a pairwise fold rather than a running sum.
///
/// The slab form of [`accurate_run_sum`], for the reductions whose axis is not
/// the last one: `out[r] += f(input[k][r])` for every `r` at once. The running
/// totals are a vector rather than a scalar, so the chunk partials are vectors
/// too -- but only `numel / RUN_SUM_CHUNK` values in total, because a block
/// covers a fixed number of *elements* rather than a fixed number of steps.
/// A four-million-element slab needs about 488 extra floats.
///
/// `add_step(k, acc)` folds step `k` of the reduced axis into `acc`, which is
/// `inner` wide. It is called once per `(block, k)` and never sees a block
/// boundary, so a caller that reads `input[base + k * inner ..]` is unaffected
/// by the blocking.
pub(crate) fn accurate_slab_sum<T>(
    dim_size: usize,
    inner: usize,
    zero: T,
    mut add_step: impl FnMut(usize, &mut [T]),
) -> Vec<T>
where
    T: Copy + std::ops::Add<Output = T>,
{
    // Blocks are counted in *steps*, not elements. Counting elements made the
    // block eight steps wide for a 1024-wide slab, which is 256 partial vectors
    // for a 2048-step axis that never needed splitting -- 48% slower on
    // `var(dim=0)` of a 2048x1024 tensor for no accuracy anyone could measure.
    // Counting steps also bounds the partials at `slab_numel / RUN_SUM_CHUNK`
    // whatever `inner` is, because a wider slab has proportionally fewer steps.
    let mut fill = |from: usize, to: usize| {
        let mut acc = vec![zero; inner];
        for k in from..to {
            add_step(k, &mut acc);
        }
        acc
    };
    if dim_size <= RUN_SUM_CHUNK {
        return fill(0, dim_size);
    }
    let block = RUN_SUM_CHUNK;

    let mut partials: Vec<Vec<T>> = (0..dim_size)
        .step_by(block)
        .map(|start| fill(start, (start + block).min(dim_size)))
        .collect();

    // `pairwise_fold` wants `Copy`, which a `Vec` is not, so the same halving
    // is spelled out here over owned buffers.
    let mut len = partials.len();
    while len > 1 {
        let mut write = 0;
        let mut read = 0;
        while read + 1 < len {
            let (left, right) = partials.split_at_mut(read + 1);
            for (a, &b) in left[read].iter_mut().zip(right[0].iter()) {
                *a = *a + b;
            }
            partials.swap(write, read);
            write += 1;
            read += 2;
        }
        if read < len {
            partials.swap(write, read);
            write += 1;
        }
        len = write;
    }
    partials.swap_remove(0)
}

/// Combine `values` with a fixed binary tree rather than a running total.
///
/// The obvious way to finish `deterministic_par_sum` is a sequential fold over
/// the partials, and it is deterministic -- but it is also a chain of one
/// rounding error per chunk, so error grows with the number of chunks instead
/// of with its logarithm. That measurably regressed accuracy: relative error on
/// a 10^7-element `f32` sum went from 1.2e-08 to 8.1e-07, because rayon's own
/// `sum()` had been reducing the partials as a tree all along. This keeps the
/// tree and drops only the scheduling dependence, so the result is both stable
/// across runs and as accurate as it was before.
///
/// `combine` is passed explicitly because not every accumulator is `Add`:
/// `nanmean` carries `(sum, count)` pairs through the same fold.
/// Chunk length for [`accurate_run_sum`]. The same 8192 `deterministic_par_sum`
/// uses, so a long run and a whole-tensor reduction accumulate the same way.
pub(crate) const RUN_SUM_CHUNK: usize = 8192;

/// Sum one contiguous run with the accuracy a whole-tensor reduction gets.
///
/// `sum()` over a tensor and `sum(dim=0)` over the same tensor viewed as one
/// axis are the same arithmetic, and they were not the same answer. The first
/// goes through `deterministic_par_sum`, which folds 8192-element partials
/// pairwise, so its error grows like `log n`. The second walked the run with a
/// single `simd_sum`, whose eight accumulator lanes leave the error growing
/// like `n`. Summing 4M positive float32 values, the axis form was **613 times**
/// less accurate than the whole-tensor form on identical data (4.4e-6 against
/// 7.1e-9), and a single long row inside a 2-D reduction had the same problem.
///
/// This is `deterministic_par_sum`'s accumulation without its parallelism,
/// because the caller is already inside a parallel loop over rows: the partials
/// are pinned to chunk order and folded pairwise, so the result depends only on
/// the run length. Short runs go straight to `sum_run`, which is every row of a
/// normal matrix -- the extra allocation only appears once a single run is long
/// enough to need it, where it is amortized over at least 8192 elements.
#[inline]
pub(crate) fn accurate_run_sum<T, U, F>(data: &[T], sum_run: F) -> U
where
    U: Copy + Default + std::ops::Add<Output = U>,
    F: Fn(&[T]) -> U,
{
    if data.len() <= RUN_SUM_CHUNK {
        return sum_run(data);
    }
    let partials: Vec<U> = data.chunks(RUN_SUM_CHUNK).map(&sum_run).collect();
    pairwise_fold(partials, U::default(), |a, b| a + b)
}

pub(crate) fn pairwise_fold<U, F>(mut values: Vec<U>, identity: U, combine: F) -> U
where
    U: Copy,
    F: Fn(U, U) -> U,
{
    if values.is_empty() {
        return identity;
    }
    let mut len = values.len();
    while len > 1 {
        let mut write = 0;
        let mut read = 0;
        while read + 1 < len {
            values[write] = combine(values[read], values[read + 1]);
            write += 1;
            read += 2;
        }
        if read < len {
            values[write] = values[read];
            write += 1;
        }
        len = write;
    }
    values[0]
}

/// Sigmoid evaluated through whichever of `e^-x` / `e^x` cannot overflow, so a
/// large-magnitude input saturates to 1 or 0 instead of producing `inf/inf`
/// (NaN). Used by the sigmoid/SiLU forward kernels and their gradients.
#[inline]
pub(crate) fn stable_sigmoid_f32(x: f32) -> f32 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let exp_pos = x.exp();
        exp_pos / (1.0 + exp_pos)
    }
}

/// [`stable_sigmoid_f32`] in double precision.
#[inline]
pub(crate) fn stable_sigmoid_f64(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let exp_pos = x.exp();
        exp_pos / (1.0 + exp_pos)
    }
}

/// Map a linear index into an output tensor onto the corresponding element of
/// a mask that broadcasts against it.
///
/// Used by the masked softmax / log-softmax kernels and their gradients, which
/// must agree exactly on which mask element governs which output position.
pub(crate) fn broadcast_mask_index(
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

/// A one-element float tensor holding `value`, for the scalar coefficients the
/// loss and gradient kernels multiply through (`1/n`, `2/n`, …).
///
/// Shape is `[1]` rather than `[]` so it broadcasts against any operand.
pub(crate) fn create_scalar_tensor(value: f64, dtype: DataType, device: Device) -> Result<Tensor> {
    let mut data = TensorData::zeros_on_device(1, dtype, device);
    match dtype {
        DataType::Float32 => {
            let slice = data.as_f32_slice_mut().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get mutable f32 slice from scalar")
            })?;
            slice[0] = value as f32;
        }
        DataType::Float64 => {
            let slice = data.as_f64_slice_mut().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get mutable f64 slice from scalar")
            })?;
            slice[0] = value;
        }
        _ => {
            return Err(MinitensorError::invalid_operation(
                "Scalar tensors only supported for floating point types",
            ));
        }
    }

    Ok(Tensor::new(
        Arc::new(data),
        Shape::new(vec![1]),
        dtype,
        device,
        false,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn negative_dims_wrap_and_bounds_error() {
        assert_eq!(normalize_dim(-1, 3).unwrap(), 2);
        assert_eq!(normalize_dim(0, 3).unwrap(), 0);
        assert_eq!(normalize_dim(2, 3).unwrap(), 2);
        assert!(normalize_dim(3, 3).is_err());
        assert!(normalize_dim(-4, 3).is_err());
        assert!(normalize_dim(0, 0).is_err());
    }

    #[test]
    fn stable_sigmoid_saturates_instead_of_producing_nan() {
        // The naive `1/(1 + e^-x)` overflows to `inf` for x <= -104 in f32 and
        // then divides inf by inf; these must be plain 0 and 1.
        assert_eq!(stable_sigmoid_f32(-200.0), 0.0);
        assert_eq!(stable_sigmoid_f32(200.0), 1.0);
        assert_eq!(stable_sigmoid_f64(-800.0), 0.0);
        assert_eq!(stable_sigmoid_f64(800.0), 1.0);
        assert_eq!(stable_sigmoid_f32(0.0), 0.5);
        assert_eq!(stable_sigmoid_f64(0.0), 0.5);
        // Symmetry: sigmoid(-x) == 1 - sigmoid(x).
        for x in [0.5f64, 1.0, 3.25, 12.0] {
            assert!((stable_sigmoid_f64(-x) - (1.0 - stable_sigmoid_f64(x))).abs() < 1e-15);
        }
    }

    #[test]
    fn broadcast_mask_index_maps_output_positions_onto_a_broadcast_mask() {
        // output [2, 3] with a [1, 3] mask: both rows read the same mask row.
        let out_dims = [2usize, 3];
        let out_strides = [3usize, 1];
        let mask_dims = [1usize, 3];
        let mask_strides = [3usize, 1];
        let got: Vec<usize> = (0..6)
            .map(|i| broadcast_mask_index(i, &out_dims, &out_strides, &mask_dims, &mask_strides))
            .collect();
        assert_eq!(got, vec![0, 1, 2, 0, 1, 2]);

        // A rank-0 mask always selects its single element.
        assert_eq!(
            broadcast_mask_index(5, &out_dims, &out_strides, &[], &[]),
            0
        );
    }

    #[test]
    fn create_scalar_tensor_builds_a_broadcastable_one_element_tensor() {
        let t = create_scalar_tensor(0.25, DataType::Float32, Device::cpu()).unwrap();
        assert_eq!(t.shape().dims(), &[1]);
        assert_eq!(t.data().as_f32_slice().unwrap(), &[0.25]);
        assert!(!t.requires_grad());

        let t = create_scalar_tensor(0.25, DataType::Float64, Device::cpu()).unwrap();
        assert_eq!(t.data().as_f64_slice().unwrap(), &[0.25]);

        assert!(create_scalar_tensor(1.0, DataType::Int64, Device::cpu()).is_err());
    }
}
