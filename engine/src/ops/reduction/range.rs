// Copyright (c) Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

//! Keeping a statistic's intermediate values inside the float range.
//!
//! A mean sums before it divides, and a variance squares before it averages,
//! and either intermediate can leave the float range while the answer does
//! not: the mean of `[1e308, 1e308]` is `1e308`, and the standard deviation
//! of the float32s `[1e20, -1e20]` is `1.4e20`, though the sum and the
//! squares are past the top of their dtypes. The squares of small values fall
//! off the bottom the same way. Each such slice is taken again at a power of
//! two near its largest magnitude, which scales exactly, and scaled back.
//!
//! The direct computation is tried first and checked afterwards, so a slice
//! that never left the range costs nothing more than the check.

use crate::{
    error::{MinitensorError, Result},
    ops::shape_ops,
    tensor::{DataType, Shape, Tensor, TensorData},
};
use std::sync::Arc;

/// Whether any element of a float result is non-finite.
pub(crate) fn any_non_finite(result: &Tensor) -> bool {
    match result.dtype() {
        DataType::Float32 => result
            .data()
            .as_f32_slice()
            .is_some_and(|v| v.iter().any(|x| !x.is_finite())),
        DataType::Float64 => result
            .data()
            .as_f64_slice()
            .is_some_and(|v| v.iter().any(|x| !x.is_finite())),
        _ => false,
    }
}

/// Whether any element of a float result is non-finite or a nonzero value
/// below the smallest normal magnitude -- where a sum of squares overflowed,
/// or lost digits to underflow. An exact zero is [`zero_may_have_underflowed`]'s.
pub(crate) fn any_out_of_normal_range(result: &Tensor) -> bool {
    match result.dtype() {
        DataType::Float32 => result.data().as_f32_slice().is_some_and(|v| {
            v.iter()
                .any(|x| !x.is_finite() || (*x != 0.0 && x.abs() < f32::MIN_POSITIVE))
        }),
        DataType::Float64 => result.data().as_f64_slice().is_some_and(|v| {
            v.iter()
                .any(|x| !x.is_finite() || (*x != 0.0 && x.abs() < f64::MIN_POSITIVE))
        }),
        _ => false,
    }
}

/// Whether a zero among the variances `result` of `tensor` over `dims` may be
/// squares that underflowed rather than a slice whose elements are all equal.
///
/// A slice's deviations from its mean square to zero only below `2^-75` in
/// float32 and `2^-537` in float64, and two distinct floats of a slice whose
/// largest magnitude is past `2^-50` (`2^-480`) are always further apart than
/// that. So a zero is only in doubt for a slice of nonzero elements that are
/// all smaller -- and a slice of constant data, the common way to get a zero,
/// costs one pass for its largest magnitude rather than the whole variance
/// again.
pub(crate) fn zero_may_have_underflowed(
    tensor: &Tensor,
    dims: &[usize],
    keepdim: bool,
    result: &Tensor,
) -> Result<bool> {
    let has_zero = match result.dtype() {
        DataType::Float32 => result
            .data()
            .as_f32_slice()
            .is_some_and(|v| v.contains(&0.0)),
        DataType::Float64 => result
            .data()
            .as_f64_slice()
            .is_some_and(|v| v.contains(&0.0)),
        _ => false,
    };
    if !has_zero {
        return Ok(false);
    }
    let magnitudes = crate::ops::activation::abs(&tensor.detach())?;
    let dims: Vec<isize> = dims.iter().map(|&d| d as isize).collect();
    let largest = super::amax(&magnitudes, Some(dims), keepdim)?;
    let missing = || MinitensorError::internal_error("variance check: result slice");
    Ok(match result.dtype() {
        DataType::Float32 => {
            let values = result.data().as_f32_slice().ok_or_else(missing)?;
            let largest = largest.data().as_f32_slice().ok_or_else(missing)?;
            let doubtful = f32::powi(2.0, -50);
            values
                .iter()
                .zip(largest)
                .any(|(&v, &a)| v == 0.0 && a > 0.0 && a < doubtful)
        }
        _ => {
            let values = result.data().as_f64_slice().ok_or_else(missing)?;
            let largest = largest.data().as_f64_slice().ok_or_else(missing)?;
            let doubtful = f64::powi(2.0, -480);
            values
                .iter()
                .zip(largest)
                .any(|(&v, &a)| v == 0.0 && a > 0.0 && a < doubtful)
        }
    })
}

/// Whether every element of a float tensor is finite. An infinite or NaN
/// input gives a non-finite statistic honestly, and is left to.
pub(crate) fn all_finite(tensor: &Tensor) -> bool {
    match tensor.dtype() {
        DataType::Float32 => tensor
            .data()
            .as_f32_slice()
            .is_some_and(|v| v.iter().all(|x| x.is_finite())),
        DataType::Float64 => tensor
            .data()
            .as_f64_slice()
            .is_some_and(|v| v.iter().all(|x| x.is_finite())),
        _ => true,
    }
}

/// For each slice over `dims`, the power of two at or below its largest
/// magnitude -- one for a slice of zeros -- with the reduced axes kept, so it
/// broadcasts against the tensor. Detached: a scale is a constant.
pub(crate) fn slice_scales(tensor: &Tensor, dims: &[usize]) -> Result<Tensor> {
    let magnitudes = crate::ops::activation::abs(&tensor.detach())?;
    let dims: Vec<isize> = dims.iter().map(|&d| d as isize).collect();
    let largest = super::amax(&magnitudes, Some(dims), true)?;
    let data = match largest.dtype() {
        DataType::Float32 => {
            let values = largest
                .data()
                .as_f32_slice()
                .ok_or_else(|| MinitensorError::internal_error("slice scales: f32 slice"))?;
            let scales = values
                .iter()
                .map(|&v| {
                    if v == 0.0 {
                        1.0
                    } else {
                        libm::ldexpf(1.0, libm::frexpf(v).1 - 1)
                    }
                })
                .collect();
            TensorData::from_vec_f32(scales, tensor.device())
        }
        DataType::Float64 => {
            let values = largest
                .data()
                .as_f64_slice()
                .ok_or_else(|| MinitensorError::internal_error("slice scales: f64 slice"))?;
            let scales = values
                .iter()
                .map(|&v| {
                    if v == 0.0 {
                        1.0
                    } else {
                        libm::ldexp(1.0, libm::frexp(v).1 - 1)
                    }
                })
                .collect();
            TensorData::from_vec_f64(scales, tensor.device())
        }
        _ => {
            return Err(MinitensorError::internal_error(
                "slice scales are only defined for float tensors",
            ));
        }
    };
    Ok(Tensor::new(
        Arc::new(data),
        largest.shape().clone(),
        largest.dtype(),
        tensor.device(),
        false,
    ))
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
/// `dims`, exactly, in the float dtype its statistics widen to: float32 for
/// int32 and float64 for int64.
///
/// A variance does not move when every element is shifted by one amount, and
/// shifted by an element of their own slice the values are as small as the
/// slice's spread, which converts exactly or to within a rounding of itself.
/// Converted first and subtracted after, two int64s above 2^53 a step apart
/// became one float64, and `var([2^53, 2^53 + 1, 2^53 + 2])` was 2, not 1.
///
/// Integers that convert exactly -- every int32 up to `2^24` in magnitude and
/// every int64 up to `2^53` -- are converted as they stand, which is cheaper
/// and the same variance; the shift is for the rest.
pub(crate) fn integer_deviations(tensor: &Tensor, dims: &[usize]) -> Result<Tensor> {
    // Checked a block at a time across the pool, each block folded without an
    // early exit: on the baseline target a 64-bit comparison is scalar either
    // way, and over one thread the check cost several times the conversion it
    // decides on.
    use crate::ops::map::{SIMD_PAR_CHUNK, par_all_chunk};
    let exact = match tensor.dtype() {
        DataType::Int32 => tensor.data().as_i32_slice().is_some_and(|v| {
            par_all_chunk(v, SIMD_PAR_CHUNK, &|block| {
                block.iter().fold(0, |most, x| most.max(x.unsigned_abs())) <= 1 << 24
            })
        }),
        DataType::Int64 => tensor.data().as_i64_slice().is_some_and(|v| {
            par_all_chunk(v, SIMD_PAR_CHUNK, &|block| {
                block.iter().fold(0, |most, x| most.max(x.unsigned_abs())) <= 1 << 53
            })
        }),
        _ => false,
    };
    if exact || tensor.numel() == 0 {
        let widened = if tensor.dtype() == DataType::Int32 {
            DataType::Float32
        } else {
            DataType::Float64
        };
        return tensor.astype(widened);
    }
    let mut reference = tensor.clone();
    for &dim in dims {
        reference = shape_ops::narrow(&reference, dim as isize, 0, 1)?;
    }
    let out_shape = tensor.shape().clone();
    let missing = || MinitensorError::internal_error("integer deviations: operand slice");
    let (data, dtype) = match tensor.dtype() {
        DataType::Int32 => {
            let values = tensor.data().as_i32_slice().ok_or_else(missing)?;
            let origin = reference.data().as_i32_slice().ok_or_else(missing)?;
            let out = crate::ops::kernels::broadcast_binary_map(
                values,
                origin,
                tensor.shape(),
                reference.shape(),
                &out_shape,
                |a: i32, b: i32| (i64::from(a) - i64::from(b)) as f32,
            )?;
            (
                TensorData::from_vec_f32(out, tensor.device()),
                DataType::Float32,
            )
        }
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
