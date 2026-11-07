// Copyright (c) Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

//! Keeping a mean's sum inside the float range.
//!
//! A mean sums before it divides, and the sum can leave the float range while
//! the answer does not: the mean of `[1e308, 1e308]` is `1e308`, though the
//! sum is past the top of float64. Such a slice is taken again at a power of
//! two near its largest magnitude, which scales exactly, and scaled back.
//!
//! A float32 mean never needs it: it is carried in float64, where no float32
//! values' sum leaves the range.
//!
//! The direct computation is tried first and checked afterwards, so a mean
//! that never left the range costs nothing more than the check, and only the
//! slices that may have are computed again.

use crate::{
    error::{MinitensorError, Result},
    ops::shape_ops,
    tensor::{DataType, Shape, Tensor, TensorData},
};
use std::sync::Arc;

/// For each slice of `tensor` over `dims` whose mean `result` the direct
/// computation may have taken out of range, the power of two at or below the
/// slice's largest finite magnitude; one for every other slice. `None` when
/// no slice needs a scale.
///
/// The scales are laid out with the reduced axes kept, so they broadcast
/// against the tensor, and are detached: a scale is a constant. A slice
/// scaled by one is computed again exactly as it was, so a slice that never
/// left the range keeps its value bit for bit.
///
/// A slice is in doubt when its mean is not finite and its largest finite
/// magnitude `a`, over `n` elements, could have overflowed the sum: when
/// `2 * n * a` reaches the dtype's largest value. Below that no partial sum
/// overflows, and a non-finite mean is the input's own NaN or infinity. The
/// test is made on the result alone first, and then against a single read of
/// the tensor for any element large enough to have overflowed; only a slice
/// still in doubt costs the pass for the largest magnitudes.
///
/// The largest *finite* magnitude, so that an infinity or a NaN elsewhere in
/// a slice neither stops it being rescued nor sets its scale: the mean of
/// `[inf, -1e308, -1e308]` is infinity, though its finite part overflows the
/// other way. A slice with no finite nonzero element keeps the scale one.
pub(crate) fn rescue_scales(
    tensor: &Tensor,
    dims: &[usize],
    result: &Tensor,
) -> Result<Option<Tensor>> {
    if tensor.numel() == 0 || dims.is_empty() {
        return Ok(None);
    }
    let n = dims
        .iter()
        .map(|&d| tensor.shape().dims()[d])
        .product::<usize>() as f64;
    let missing = || MinitensorError::internal_error("rescue scales: float slice");
    let data = match tensor.dtype() {
        DataType::Float32 => {
            let bounds = Bounds::new(f32::MAX.into(), n);
            let scales = scales_in_doubt(
                tensor,
                dims,
                result.data().as_f32_slice().ok_or_else(missing)?,
                &bounds,
                |data| data.as_f32_slice(),
            )?;
            let Some(scales) = scales else {
                return Ok(None);
            };
            TensorData::from_vec_f32(
                scales.into_iter().map(|s| s as f32).collect(),
                tensor.device(),
            )
        }
        DataType::Float64 => {
            let bounds = Bounds::new(f64::MAX, n);
            let scales = scales_in_doubt(
                tensor,
                dims,
                result.data().as_f64_slice().ok_or_else(missing)?,
                &bounds,
                |data| data.as_f64_slice(),
            )?;
            let Some(scales) = scales else {
                return Ok(None);
            };
            TensorData::from_vec_f64(scales, tensor.device())
        }
        _ => return Ok(None),
    };
    let mut shape = tensor.shape().dims().to_vec();
    for &d in dims {
        shape[d] = 1;
    }
    Ok(Some(Tensor::new(
        Arc::new(data),
        Shape::new(shape),
        tensor.dtype(),
        tensor.device(),
        false,
    )))
}

/// Where a slice's mean is in doubt, in a dtype and over slices of `n`
/// elements; see [`rescue_scales`].
struct Bounds {
    /// The largest finite magnitude at or past which a non-finite mean may
    /// be an overflowed sum.
    large: f64,
}

impl Bounds {
    fn new(max: f64, n: f64) -> Self {
        Self {
            large: max / (2.0 * n),
        }
    }

    /// Whether a slice of mean `r` and largest finite magnitude `a` is in
    /// doubt.
    fn in_doubt(&self, r: f64, a: f64) -> bool {
        a != 0.0 && !r.is_finite() && a >= self.large
    }
}

/// The scale for each slice of `tensor` over `dims` whose mean in `values`
/// is in doubt, or `None` when none is. See [`rescue_scales`].
fn scales_in_doubt<T>(
    tensor: &Tensor,
    dims: &[usize],
    values: &[T],
    bounds: &Bounds,
    slice_of: fn(&TensorData) -> Option<&[T]>,
) -> Result<Option<Vec<f64>>>
where
    T: num_traits::Float + Into<f64> + Send + Sync,
{
    let missing = || MinitensorError::internal_error("rescue scales: float slice");
    let non_finite = values.iter().any(|&r| !r.is_finite());
    let data = slice_of(tensor.data()).ok_or_else(missing)?;
    if !non_finite || !any_finite_at_least(data, bounds.large) {
        return Ok(None);
    }
    let largest = largest_finite_magnitudes(tensor, dims)?;
    let largest = slice_of(largest.data()).ok_or_else(missing)?;
    let mut any = false;
    let scales = values
        .iter()
        .zip(largest)
        .map(|(&r, &a)| {
            let a: f64 = a.into();
            if bounds.in_doubt(r.into(), a) {
                any = true;
                libm::ldexp(1.0, libm::frexp(a).1 - 1)
            } else {
                1.0
            }
        })
        .collect();
    Ok(any.then_some(scales))
}

/// Whether any finite element of `data` is at least `bound` in magnitude.
///
/// Folded a block at a time without an early exit, which keeps the test
/// vectorized; a NaN or an infinity fails one comparison or the other.
fn any_finite_at_least<T>(data: &[T], bound: f64) -> bool
where
    T: num_traits::Float + Send + Sync,
{
    use crate::ops::map::{SIMD_PAR_CHUNK, par_all_chunk};
    let Some(bound) = T::from(bound) else {
        return false;
    };
    let most = T::max_value();
    !par_all_chunk(data, SIMD_PAR_CHUNK, &|block| {
        !block.iter().fold(false, |any, &x| {
            let v = x.abs();
            any | ((v >= bound) & (v <= most))
        })
    })
}

/// Each slice's largest finite magnitude over `dims`, reduced axes kept; zero
/// for a slice with no finite element.
fn largest_finite_magnitudes(tensor: &Tensor, dims: &[usize]) -> Result<Tensor> {
    use crate::ops::map::unary_map_selecting;
    let missing = || MinitensorError::internal_error("finite magnitudes: float slice");
    let data = match tensor.dtype() {
        DataType::Float32 => {
            let values = tensor.data().as_f32_slice().ok_or_else(missing)?;
            let out =
                unary_map_selecting(values, |x: f32| if x.is_finite() { x.abs() } else { 0.0 });
            TensorData::from_vec_f32(out, tensor.device())
        }
        DataType::Float64 => {
            let values = tensor.data().as_f64_slice().ok_or_else(missing)?;
            let out =
                unary_map_selecting(values, |x: f64| if x.is_finite() { x.abs() } else { 0.0 });
            TensorData::from_vec_f64(out, tensor.device())
        }
        _ => {
            return Err(MinitensorError::internal_error(
                "finite magnitudes are only defined for float tensors",
            ));
        }
    };
    let magnitudes = Tensor::new(
        Arc::new(data),
        tensor.shape().clone(),
        tensor.dtype(),
        tensor.device(),
        false,
    );
    let dims: Vec<isize> = dims.iter().map(|&d| d as isize).collect();
    super::amax(&magnitudes, Some(dims), true)
}

/// `scales`, laid out as a reduction over `dims` lays out its result: as it
/// is with `keepdim`, and with the reduced axes dropped without it.
pub(crate) fn scales_for_result(scales: &Tensor, result_shape: &Shape) -> Result<Tensor> {
    if scales.shape() == result_shape {
        return Ok(scales.clone());
    }
    shape_ops::reshape(scales, result_shape.clone())
}

/// An integer tensor's elements less the first element of their slice over
/// `dims`, exactly or to within a rounding of themselves, in float64.
///
/// A variance does not move when every element is shifted by one amount, and
/// shifted by an element of their own slice the values are as small as the
/// slice's spread, which converts exactly or to within a rounding of itself.
/// Converted first and subtracted after, two int64s above 2^53 a step apart
/// became one float64, and `var([2^53, 2^53 + 1, 2^53 + 2])` was 2, not 1.
///
/// Integers that convert exactly -- every int32, and every int64 up to `2^53`
/// in magnitude -- are converted as they stand, which is cheaper and the same
/// variance; the shift is for the rest. An int32 statistic is carried in
/// float64 and rounded to float32 once, at the end.
pub(crate) fn integer_deviations(tensor: &Tensor, dims: &[usize]) -> Result<Tensor> {
    // Checked a block at a time across the pool, each block folded without an
    // early exit: on the baseline target a 64-bit comparison is scalar either
    // way, and over one thread the check cost several times the conversion it
    // decides on.
    use crate::ops::map::{SIMD_PAR_CHUNK, par_all_chunk};
    let exact = match tensor.dtype() {
        DataType::Int32 => true,
        DataType::Int64 => tensor.data().as_i64_slice().is_some_and(|v| {
            par_all_chunk(v, SIMD_PAR_CHUNK, &|block| {
                block.iter().fold(0, |most, x| most.max(x.unsigned_abs())) <= 1 << 53
            })
        }),
        _ => false,
    };
    if exact || tensor.numel() == 0 {
        return tensor.astype(DataType::Float64);
    }
    let mut reference = tensor.clone();
    for &dim in dims {
        reference = shape_ops::narrow(&reference, dim as isize, 0, 1)?;
    }
    let out_shape = tensor.shape().clone();
    let missing = || MinitensorError::internal_error("integer deviations: operand slice");
    let (data, dtype) = match tensor.dtype() {
        DataType::Int64 => {
            let values = tensor.data().as_i64_slice().ok_or_else(missing)?;
            let origin = reference.data().as_i64_slice().ok_or_else(missing)?;
            let out = crate::ops::kernels::broadcast_binary_map(
                values,
                origin,
                tensor.shape(),
                reference.shape(),
                &out_shape,
                |a: i64, b: i64| (i128::from(a) - i128::from(b)) as f64,
            )?;
            (
                TensorData::from_vec_f64(out, tensor.device()),
                DataType::Float64,
            )
        }
        _ => {
            return Err(MinitensorError::internal_error(
                "integer deviations are only defined for integer tensors",
            ));
        }
    };
    Ok(Tensor::new(
        Arc::new(data),
        out_shape,
        dtype,
        tensor.device(),
        false,
    ))
}
