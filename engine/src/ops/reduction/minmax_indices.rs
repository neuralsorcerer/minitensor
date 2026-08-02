// Copyright (c) Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

use super::*;
use crate::{
    error::{MinitensorError, Result},
    tensor::{DataType, Tensor, TensorData},
};
use std::sync::Arc;

/// Which extremum an indexed dimension reduction is looking for.
///
/// `min` and `max` differ only in their seed and their "is this better"
/// comparison, and the NaN-skipping forms differ from the plain ones only in
/// seeding with NaN instead of an infinity. Spelling all four out cost four
/// copies of the same dtype dispatch.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Extremum {
    Min,
    Max,
}

/// Reduce `dim` to its extremum, reporting the value and the index of the
/// first element attaining it.
///
/// A NaN anywhere in a float slice wins outright (`reduce_arg_along_dim_par`'s
/// short-circuit), matching `torch.min`/`torch.max`; `nan_aware` instead skips
/// NaNs entirely, so an all-NaN slice reports NaN at index 0 as NumPy's
/// `nanmin`/`nanmax` do. Integer and bool dtypes have no NaN and are rejected
/// in the NaN-aware form.
pub(crate) fn extremum_along_dim_with_indices(
    tensor: &Tensor,
    dim: usize,
    keepdim: bool,
    which: Extremum,
    nan_aware: bool,
) -> Result<(Tensor, Tensor)> {
    let layout = reduction_layout(tensor, dim, keepdim)?;
    let mut values_data =
        TensorData::zeros_on_device(layout.output_shape.numel(), tensor.dtype(), tensor.device());
    let mut indices_data = TensorData::zeros_on_device(
        layout.output_shape.numel(),
        DataType::Int64,
        tensor.device(),
    );

    let indices = indices_data
        .as_i64_slice_mut()
        .ok_or_else(|| MinitensorError::internal_error("Failed to get mutable i64 slice"))?;

    /// One float dtype: NaN either wins immediately or is skipped, depending on
    /// `nan_aware`.
    macro_rules! float_arm {
        ($accessor:ident, $mut_accessor:ident, $ty:ty, $tyname:literal) => {{
            let input = tensor.data().$accessor().ok_or_else(|| {
                MinitensorError::internal_error(concat!("Failed to get ", $tyname, " slice"))
            })?;
            let values = values_data.$mut_accessor().ok_or_else(|| {
                MinitensorError::internal_error(concat!(
                    "Failed to get mutable ",
                    $tyname,
                    " slice"
                ))
            })?;
            let is_max = which == Extremum::Max;
            if nan_aware {
                reduce_arg_along_dim_par(
                    input,
                    values,
                    indices,
                    &layout,
                    <$ty>::NAN,
                    move |v: $ty, b: $ty| {
                        !v.is_nan() && (b.is_nan() || if is_max { v > b } else { v < b })
                    },
                    |_| None,
                );
            } else {
                let seed = if is_max {
                    <$ty>::NEG_INFINITY
                } else {
                    <$ty>::INFINITY
                };
                reduce_arg_along_dim_par(
                    input,
                    values,
                    indices,
                    &layout,
                    seed,
                    // NaN folded in rather than left to the short-circuit, so
                    // the memory-order path in `reduce_arg_along_dim_par` --
                    // which has no per-element early exit -- stays correct. A
                    // NaN beats any real value, and a later NaN does not beat an
                    // earlier one, which reproduces the break-on-first-NaN index
                    // the short-circuit produced.
                    move |v: $ty, b: $ty| {
                        if v.is_nan() {
                            !b.is_nan()
                        } else if b.is_nan() {
                            false
                        } else if is_max {
                            v > b
                        } else {
                            v < b
                        }
                    },
                    |v: $ty| if v.is_nan() { Some(<$ty>::NAN) } else { None },
                );
            }
        }};
    }

    /// One non-float dtype: no NaN to consider, so only the plain form exists.
    macro_rules! exact_arm {
        ($accessor:ident, $mut_accessor:ident, $ty:ty, $tyname:literal, $min_seed:expr, $max_seed:expr, $short:expr) => {{
            if nan_aware {
                return Err(MinitensorError::invalid_operation(
                    "nan-aware min/max only supports floating point tensors",
                ));
            }
            let input = tensor.data().$accessor().ok_or_else(|| {
                MinitensorError::internal_error(concat!("Failed to get ", $tyname, " slice"))
            })?;
            let values = values_data.$mut_accessor().ok_or_else(|| {
                MinitensorError::internal_error(concat!(
                    "Failed to get mutable ",
                    $tyname,
                    " slice"
                ))
            })?;
            let is_max = which == Extremum::Max;
            let seed = if is_max { $max_seed } else { $min_seed };
            let short = $short;
            reduce_arg_along_dim_par(
                input,
                values,
                indices,
                &layout,
                seed,
                move |v: $ty, b: $ty| if is_max { v > b } else { v < b },
                move |v: $ty| short(v, is_max),
            );
        }};
    }

    match tensor.dtype() {
        DataType::Float32 => float_arm!(as_f32_slice, as_f32_slice_mut, f32, "f32"),
        DataType::Float64 => float_arm!(as_f64_slice, as_f64_slice_mut, f64, "f64"),
        DataType::Int32 => exact_arm!(
            as_i32_slice,
            as_i32_slice_mut,
            i32,
            "i32",
            i32::MAX,
            i32::MIN,
            |_v: i32, _is_max: bool| None
        ),
        DataType::Int64 => exact_arm!(
            as_i64_slice,
            as_i64_slice_mut,
            i64,
            "i64",
            i64::MAX,
            i64::MIN,
            |_v: i64, _is_max: bool| None
        ),
        // Bool has only two values, so the first `true` (for max) or first
        // `false` (for min) is already the answer: short-circuit and never
        // compare.
        DataType::Bool => exact_arm!(
            as_bool_slice,
            as_bool_slice_mut,
            bool,
            "bool",
            true,
            false,
            |v: bool, is_max: bool| if v == is_max { Some(v) } else { None }
        ),
    }

    Ok((
        Tensor::new(
            Arc::new(values_data),
            layout.output_shape.clone(),
            tensor.dtype(),
            tensor.device(),
            tensor.requires_grad(),
        ),
        Tensor::new(
            Arc::new(indices_data),
            layout.output_shape,
            DataType::Int64,
            tensor.device(),
            false,
        ),
    ))
}

pub(crate) fn min_along_dim_with_indices(
    tensor: &Tensor,
    dim: usize,
    keepdim: bool,
) -> Result<(Tensor, Tensor)> {
    extremum_along_dim_with_indices(tensor, dim, keepdim, Extremum::Min, false)
}

pub(crate) fn max_along_dim_with_indices(
    tensor: &Tensor,
    dim: usize,
    keepdim: bool,
) -> Result<(Tensor, Tensor)> {
    extremum_along_dim_with_indices(tensor, dim, keepdim, Extremum::Max, false)
}

pub(crate) fn nanmin_along_dim_with_indices(
    tensor: &Tensor,
    dim: usize,
    keepdim: bool,
) -> Result<(Tensor, Tensor)> {
    extremum_along_dim_with_indices(tensor, dim, keepdim, Extremum::Min, true)
}

pub(crate) fn nanmax_along_dim_with_indices(
    tensor: &Tensor,
    dim: usize,
    keepdim: bool,
) -> Result<(Tensor, Tensor)> {
    extremum_along_dim_with_indices(tensor, dim, keepdim, Extremum::Max, true)
}

/// Index of the extremum along `dim`.
///
/// The index-only reductions are the value-and-index reduction with the values
/// discarded -- identical seeds, identical comparisons, identical NaN-wins and
/// bool short-circuits. They had their own hand-written copies, each a
/// sequential five-arm dtype dispatch whose inner loop strode by `inner`;
/// routing them here parallelizes them. The discarded value buffer is
/// `outer * inner` elements, a factor of `dim_size` smaller than the input the
/// reduction already has to read.
pub(crate) fn argmax_along_dim(tensor: &Tensor, dim: usize, keepdim: bool) -> Result<Tensor> {
    let (_, indices) = extremum_along_dim_with_indices(tensor, dim, keepdim, Extremum::Max, false)?;
    Ok(indices)
}

/// See [`argmax_along_dim`].
pub(crate) fn argmin_along_dim(tensor: &Tensor, dim: usize, keepdim: bool) -> Result<Tensor> {
    let (_, indices) = extremum_along_dim_with_indices(tensor, dim, keepdim, Extremum::Min, false)?;
    Ok(indices)
}
