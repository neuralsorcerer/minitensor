// Copyright (c) Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

use super::*;
use crate::autograd::AbsBackward;
use crate::autograd::ClampBackward;
use crate::autograd::NanToNumBackward;
use crate::autograd::RsqrtBackward;
use crate::autograd::SqrtBackward;
use crate::{
    autograd::with_grad_fn,
    error::{MinitensorError, Result},
    tensor::{DataType, Tensor, TensorData},
};
use std::sync::Arc;

/// Absolute value function
pub fn abs(tensor: &Tensor) -> Result<Tensor> {
    let output_data = match tensor.dtype() {
        DataType::Float32 => abs_f32(tensor)?,
        DataType::Float64 => abs_f64(tensor)?,
        DataType::Int32 => abs_i32(tensor)?,
        DataType::Int64 => abs_i64(tensor)?,
        DataType::Bool => {
            return Err(MinitensorError::invalid_operation(
                "Absolute value not supported for boolean tensors",
            ));
        }
    };

    let output = Tensor::new(
        Arc::new(output_data),
        tensor.shape().clone(),
        tensor.dtype(),
        tensor.device(),
        tensor.requires_grad(),
    );

    if output.requires_grad() && tensor.dtype().is_float() {
        let grad_fn = Arc::new(AbsBackward {
            input_id: tensor.id(),
            input: tensor.detach(),
        });
        return with_grad_fn(output, grad_fn);
    }

    Ok(output)
}

/// Element-wise sign function (-1, 0, or 1 depending on value sign)
pub fn sign(tensor: &Tensor) -> Result<Tensor> {
    let output_data = match tensor.dtype() {
        DataType::Float32 => sign_f32(tensor)?,
        DataType::Float64 => sign_f64(tensor)?,
        DataType::Int32 => sign_i32(tensor)?,
        DataType::Int64 => sign_i64(tensor)?,
        DataType::Bool => {
            return Err(MinitensorError::invalid_operation(
                "Sign operation not supported for boolean tensors",
            ));
        }
    };

    Ok(Tensor::new(
        Arc::new(output_data),
        tensor.shape().clone(),
        tensor.dtype(),
        tensor.device(),
        // A step function: its derivative is zero wherever it exists, so no
        // gradient is worth recording. The output is a constant rather than a
        // tensor that claims `requires_grad` and then behaves as a leaf --
        // which is what propagating the input's flag without attaching a
        // gradient function produced. Matches `norm(p = 0)`.
        false,
    ))
}

/// Square root function.
///
/// A dedicated op (rather than `x.powf(0.5)`) so the forward path uses the
/// hardware `sqrt`, which follows IEEE for the edge cases `powf` gets wrong:
/// `sqrt(-inf)` and `sqrt(x<0)` are NaN, `sqrt(-0.0)` is `-0.0`. Gradients flow
/// through [`SqrtBackward`].
pub fn sqrt(tensor: &Tensor) -> Result<Tensor> {
    let output_data = match tensor.dtype() {
        DataType::Float32 => sqrt_f32(tensor)?,
        DataType::Float64 => sqrt_f64(tensor)?,
        _ => {
            return Err(MinitensorError::invalid_operation(
                "Square root only supported for floating point tensors",
            ));
        }
    };

    let output = Tensor::new(
        Arc::new(output_data),
        tensor.shape().clone(),
        tensor.dtype(),
        tensor.device(),
        tensor.requires_grad(),
    );

    if output.requires_grad() {
        let grad_fn = Arc::new(SqrtBackward {
            input_id: tensor.id(),
            output: output.clone().detach(),
        });
        return with_grad_fn(output, grad_fn);
    }

    Ok(output)
}

/// Reciprocal square root function.
///
/// Dedicated op for the same reason as [`sqrt`]: `1/sqrt(x)` gives the IEEE
/// results `powf(-0.5)` misses (`rsqrt(-inf)` is NaN, `rsqrt(-0.0)` is `-inf`).
/// Gradients flow through [`RsqrtBackward`].
pub fn rsqrt(tensor: &Tensor) -> Result<Tensor> {
    let output_data = match tensor.dtype() {
        DataType::Float32 => rsqrt_f32(tensor)?,
        DataType::Float64 => rsqrt_f64(tensor)?,
        _ => {
            return Err(MinitensorError::invalid_operation(
                "Reciprocal square root only supported for floating point tensors",
            ));
        }
    };

    let output = Tensor::new(
        Arc::new(output_data),
        tensor.shape().clone(),
        tensor.dtype(),
        tensor.device(),
        tensor.requires_grad(),
    );

    if output.requires_grad() {
        let grad_fn = Arc::new(RsqrtBackward {
            input_id: tensor.id(),
            output: output.clone().detach(),
        });
        return with_grad_fn(output, grad_fn);
    }

    Ok(output)
}

/// Element-wise reciprocal (1/x) with gradient support
pub fn reciprocal(tensor: &Tensor) -> Result<Tensor> {
    match tensor.dtype() {
        DataType::Float32 | DataType::Float64 => powf(tensor, -1.0),
        _ => Err(MinitensorError::invalid_operation(
            "Reciprocal only supported for floating point tensors",
        )),
    }
}

/// Clip tensor values to range
pub fn clip(tensor: &Tensor, min_val: Option<f64>, max_val: Option<f64>) -> Result<Tensor> {
    let output_data = match tensor.dtype() {
        DataType::Float32 => clip_f32(tensor, min_val, max_val)?,
        DataType::Float64 => clip_f64(tensor, min_val, max_val)?,
        DataType::Int32 => clip_i32(tensor, min_val, max_val)?,
        DataType::Int64 => clip_i64(tensor, min_val, max_val)?,
        DataType::Bool => {
            return Err(MinitensorError::invalid_operation(
                "Clip not supported for boolean tensors",
            ));
        }
    };

    let output = Tensor::new(
        Arc::new(output_data),
        tensor.shape().clone(),
        tensor.dtype(),
        tensor.device(),
        tensor.requires_grad(),
    );

    if output.requires_grad() && tensor.dtype().is_float() {
        let grad_fn = Arc::new(ClampBackward {
            input_id: tensor.id(),
            input: tensor.detach(),
            min: min_val,
            max: max_val,
        });
        return with_grad_fn(output, grad_fn);
    }

    Ok(output)
}

/// Replace NaN and infinity values in floating point tensors.
///
/// Exact tensors cannot contain NaN or infinity, so they are returned unchanged.
pub fn nan_to_num(
    tensor: &Tensor,
    nan: f64,
    posinf: Option<f64>,
    neginf: Option<f64>,
) -> Result<Tensor> {
    match tensor.dtype() {
        DataType::Float32 | DataType::Float64 => {}
        DataType::Int32 | DataType::Int64 | DataType::Bool => return Ok(tensor.clone()),
    }

    // The finite-mask is only needed when a gradient function will consume
    // it (mirrors `Tensor::new`'s grad gating).
    let store_mask = tensor.requires_grad() && crate::autograd::is_grad_enabled();

    let (output_data, finite_mask) = match tensor.dtype() {
        DataType::Float32 => nan_to_num_f32(tensor, nan, posinf, neginf, store_mask)?,
        DataType::Float64 => nan_to_num_f64(tensor, nan, posinf, neginf, store_mask)?,
        DataType::Int32 | DataType::Int64 | DataType::Bool => unreachable!(),
    };

    let output = Tensor::new(
        Arc::new(output_data),
        tensor.shape().clone(),
        tensor.dtype(),
        tensor.device(),
        tensor.requires_grad(),
    );

    if output.requires_grad() {
        let grad_fn = Arc::new(NanToNumBackward {
            input_id: tensor.id(),
            finite_mask,
        });

        with_grad_fn(output, grad_fn)
    } else {
        Ok(output)
    }
}

/// Round tensor values
pub fn round(tensor: &Tensor, decimals: i32) -> Result<Tensor> {
    let output_data = match tensor.dtype() {
        DataType::Float32 => round_f32(tensor, decimals)?,
        DataType::Float64 => round_f64(tensor, decimals)?,
        _ => {
            return Err(MinitensorError::invalid_operation(
                "Round only supported for floating point tensors",
            ));
        }
    };

    let output = Tensor::new(
        Arc::new(output_data),
        tensor.shape().clone(),
        tensor.dtype(),
        tensor.device(),
        // A step function: its derivative is zero wherever it exists, so no
        // gradient is worth recording. The output is a constant rather than a
        // tensor that claims `requires_grad` and then behaves as a leaf --
        // which is what propagating the input's flag without attaching a
        // gradient function produced. Matches `norm(p = 0)`.
        false,
    );

    Ok(output)
}

/// Floor tensor values
pub fn floor(tensor: &Tensor) -> Result<Tensor> {
    let output_data = match tensor.dtype() {
        DataType::Float32 => floor_f32(tensor, 0.0)?,
        DataType::Float64 => floor_f64(tensor, 0.0)?,
        _ => {
            return Err(MinitensorError::invalid_operation(
                "Floor only supported for floating point tensors",
            ));
        }
    };

    let output = Tensor::new(
        Arc::new(output_data),
        tensor.shape().clone(),
        tensor.dtype(),
        tensor.device(),
        // A step function: its derivative is zero wherever it exists, so no
        // gradient is worth recording. The output is a constant rather than a
        // tensor that claims `requires_grad` and then behaves as a leaf --
        // which is what propagating the input's flag without attaching a
        // gradient function produced. Matches `norm(p = 0)`.
        false,
    );

    Ok(output)
}

/// Ceiling tensor values
pub fn ceil(tensor: &Tensor) -> Result<Tensor> {
    let output_data = match tensor.dtype() {
        DataType::Float32 => ceil_f32(tensor, 0.0)?,
        DataType::Float64 => ceil_f64(tensor, 0.0)?,
        _ => {
            return Err(MinitensorError::invalid_operation(
                "Ceiling only supported for floating point tensors",
            ));
        }
    };

    let output = Tensor::new(
        Arc::new(output_data),
        tensor.shape().clone(),
        tensor.dtype(),
        tensor.device(),
        // A step function: its derivative is zero wherever it exists, so no
        // gradient is worth recording. The output is a constant rather than a
        // tensor that claims `requires_grad` and then behaves as a leaf --
        // which is what propagating the input's flag without attaching a
        // gradient function produced. Matches `norm(p = 0)`.
        false,
    );

    Ok(output)
}

// Helper functions for the new operations

fn sqrt_f32(tensor: &Tensor) -> Result<TensorData> {
    let input_data = tensor.data().as_f32_slice().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get f32 slice from input tensor")
    })?;
    let out = unary_map(input_data, |v: f32| v.sqrt());
    Ok(TensorData::from_vec::<f32>(
        out,
        DataType::Float32,
        tensor.device(),
    ))
}

fn sqrt_f64(tensor: &Tensor) -> Result<TensorData> {
    let input_data = tensor.data().as_f64_slice().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get f64 slice from input tensor")
    })?;
    let out = unary_map(input_data, |v: f64| v.sqrt());
    Ok(TensorData::from_vec::<f64>(
        out,
        DataType::Float64,
        tensor.device(),
    ))
}

fn rsqrt_f32(tensor: &Tensor) -> Result<TensorData> {
    let input_data = tensor.data().as_f32_slice().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get f32 slice from input tensor")
    })?;
    let out = unary_map(input_data, |v: f32| v.sqrt().recip());
    Ok(TensorData::from_vec::<f32>(
        out,
        DataType::Float32,
        tensor.device(),
    ))
}

fn rsqrt_f64(tensor: &Tensor) -> Result<TensorData> {
    let input_data = tensor.data().as_f64_slice().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get f64 slice from input tensor")
    })?;
    let out = unary_map(input_data, |v: f64| v.sqrt().recip());
    Ok(TensorData::from_vec::<f64>(
        out,
        DataType::Float64,
        tensor.device(),
    ))
}

fn abs_f32(tensor: &Tensor) -> Result<TensorData> {
    let input_data = tensor.data().as_f32_slice().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get f32 slice from input tensor")
    })?;

    let out = unary_map(input_data, |v: f32| v.abs());
    Ok(TensorData::from_vec::<f32>(
        out,
        DataType::Float32,
        tensor.device(),
    ))
}

fn abs_f64(tensor: &Tensor) -> Result<TensorData> {
    let input_data = tensor.data().as_f64_slice().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get f64 slice from input tensor")
    })?;

    let out = unary_map(input_data, |v: f64| v.abs());
    Ok(TensorData::from_vec::<f64>(
        out,
        DataType::Float64,
        tensor.device(),
    ))
}

fn abs_i32(tensor: &Tensor) -> Result<TensorData> {
    let input_data = tensor.data().as_i32_slice().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get i32 slice from input tensor")
    })?;

    let out = unary_map(input_data, |v: i32| v.abs());
    Ok(TensorData::from_vec::<i32>(
        out,
        DataType::Int32,
        tensor.device(),
    ))
}

fn abs_i64(tensor: &Tensor) -> Result<TensorData> {
    let input_data = tensor.data().as_i64_slice().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get i64 slice from input tensor")
    })?;

    let out = unary_map(input_data, |v: i64| v.abs());
    Ok(TensorData::from_vec::<i64>(
        out,
        DataType::Int64,
        tensor.device(),
    ))
}

fn sign_f32(tensor: &Tensor) -> Result<TensorData> {
    let input_data = tensor.data().as_f32_slice().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get f32 slice from input tensor")
    })?;

    let out = unary_map(input_data, |v: f32| {
        if v.is_nan() {
            v
        } else if v > 0.0 {
            1.0
        } else if v < 0.0 {
            -1.0
        } else {
            0.0
        }
    });
    Ok(TensorData::from_vec::<f32>(
        out,
        DataType::Float32,
        tensor.device(),
    ))
}

fn sign_f64(tensor: &Tensor) -> Result<TensorData> {
    let input_data = tensor.data().as_f64_slice().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get f64 slice from input tensor")
    })?;

    let out = unary_map(input_data, |v: f64| {
        if v.is_nan() {
            v
        } else if v > 0.0 {
            1.0
        } else if v < 0.0 {
            -1.0
        } else {
            0.0
        }
    });
    Ok(TensorData::from_vec::<f64>(
        out,
        DataType::Float64,
        tensor.device(),
    ))
}

fn sign_i32(tensor: &Tensor) -> Result<TensorData> {
    let input_data = tensor.data().as_i32_slice().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get i32 slice from input tensor")
    })?;

    let out = unary_map(input_data, |v: i32| {
        if v > 0 {
            1
        } else if v < 0 {
            -1
        } else {
            0
        }
    });
    Ok(TensorData::from_vec::<i32>(
        out,
        DataType::Int32,
        tensor.device(),
    ))
}

fn sign_i64(tensor: &Tensor) -> Result<TensorData> {
    let input_data = tensor.data().as_i64_slice().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get i64 slice from input tensor")
    })?;

    let out = unary_map(input_data, |v: i64| {
        if v > 0 {
            1
        } else if v < 0 {
            -1
        } else {
            0
        }
    });
    Ok(TensorData::from_vec::<i64>(
        out,
        DataType::Int64,
        tensor.device(),
    ))
}

/// Generates a dtype-specialized `clip` kernel.
///
/// Two details keep the loop in vector registers, and the previous form had
/// neither.
///
/// `if v < lo { lo } else { v }` rather than `v.max(lo)`: on floats these are
/// not the same operation. `f32::max` returns the *non*-NaN operand, so a NaN
/// input would come back as a clamp bound -- which is why the old kernel opened
/// with an `is_nan` early return. A comparison against NaN is false, so the
/// same NaN passes through here with nothing to test for, and the branch that
/// bought the correctness is gone with it. (For floats this also fixes the
/// bounds' order of preference at signed zero: `-0.0` clipped below by `0.0`
/// now reliably stays `-0.0`, where `f32::max` was free to return either.)
///
/// And the bounds are resolved once, before the loop, instead of matching two
/// `Option`s per element.
macro_rules! clip_kernel {
    ($name:ident, $ty:ty, $dtype:ident, $accessor:ident, $tyname:literal) => {
        fn $name(
            tensor: &Tensor,
            min_val: Option<f64>,
            max_val: Option<f64>,
        ) -> Result<TensorData> {
            let input_data = tensor.data().$accessor().ok_or_else(|| {
                MinitensorError::internal_error(concat!(
                    "Failed to get ",
                    $tyname,
                    " slice from input tensor"
                ))
            })?;

            // Lower bound first, then upper: with `lo > hi` the upper bound
            // wins, which is the conventional reading of a reversed interval.
            let out = match (min_val.map(|v| v as $ty), max_val.map(|v| v as $ty)) {
                (Some(lo), Some(hi)) => unary_map(input_data, move |v: $ty| {
                    let v = if v < lo { lo } else { v };
                    if v > hi { hi } else { v }
                }),
                (Some(lo), None) => {
                    unary_map(input_data, move |v: $ty| if v < lo { lo } else { v })
                }
                (None, Some(hi)) => {
                    unary_map(input_data, move |v: $ty| if v > hi { hi } else { v })
                }
                (None, None) => input_data.to_vec(),
            };

            Ok(TensorData::from_vec::<$ty>(
                out,
                DataType::$dtype,
                tensor.device(),
            ))
        }
    };
}

clip_kernel!(clip_f32, f32, Float32, as_f32_slice, "f32");
clip_kernel!(clip_f64, f64, Float64, as_f64_slice, "f64");
clip_kernel!(clip_i32, i32, Int32, as_i32_slice, "i32");
clip_kernel!(clip_i64, i64, Int64, as_i64_slice, "i64");

fn nan_to_num_f32(
    tensor: &Tensor,
    nan: f64,
    posinf: Option<f64>,
    neginf: Option<f64>,
    store_mask: bool,
) -> Result<(TensorData, Vec<bool>)> {
    let input_data = tensor.data().as_f32_slice().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get f32 slice from input tensor")
    })?;

    let (out, mask) = replace_non_finite(
        input_data,
        nan as f32,
        posinf.map_or(f32::MAX, |v| v as f32),
        neginf.map_or(f32::MIN, |v| v as f32),
        store_mask,
    );
    Ok((
        TensorData::from_vec::<f32>(out, DataType::Float32, tensor.device()),
        mask,
    ))
}

fn nan_to_num_f64(
    tensor: &Tensor,
    nan: f64,
    posinf: Option<f64>,
    neginf: Option<f64>,
    store_mask: bool,
) -> Result<(TensorData, Vec<bool>)> {
    let input_data = tensor.data().as_f64_slice().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get f64 slice from input tensor")
    })?;

    let (out, mask) = replace_non_finite(
        input_data,
        nan,
        posinf.unwrap_or(f64::MAX),
        neginf.unwrap_or(f64::MIN),
        store_mask,
    );
    Ok((
        TensorData::from_vec::<f64>(out, DataType::Float64, tensor.device()),
        mask,
    ))
}

fn replace_non_finite<T>(
    input: &[T],
    nan: T,
    posinf: T,
    neginf: T,
    store_mask: bool,
) -> (Vec<T>, Vec<bool>)
where
    T: Copy + PartialEq + Send + Sync,
    T: FloatClassify,
{
    let out = unary_map(input, |val| classify_nan_to_num(val, nan, posinf, neginf));
    let mask = if store_mask {
        unary_map(input, |val: T| val.is_finite_value())
    } else {
        Vec::new()
    };
    (out, mask)
}

#[inline(always)]
fn classify_nan_to_num<T>(val: T, nan: T, posinf: T, neginf: T) -> T
where
    T: Copy + PartialEq + FloatClassify,
{
    if val.is_nan_value() {
        nan
    } else if val.is_positive_infinity() {
        posinf
    } else if val.is_negative_infinity() {
        neginf
    } else {
        val
    }
}

// Copy-scalar helpers mirroring `f32::is_nan` and friends, which also take
// `self` by value.
#[allow(clippy::wrong_self_convention)]
trait FloatClassify {
    fn is_nan_value(self) -> bool;
    fn is_finite_value(self) -> bool;
    fn is_positive_infinity(self) -> bool;
    fn is_negative_infinity(self) -> bool;
}

impl FloatClassify for f32 {
    #[inline(always)]
    fn is_nan_value(self) -> bool {
        self.is_nan()
    }

    #[inline(always)]
    fn is_finite_value(self) -> bool {
        self.is_finite()
    }

    #[inline(always)]
    fn is_positive_infinity(self) -> bool {
        self == f32::INFINITY
    }

    #[inline(always)]
    fn is_negative_infinity(self) -> bool {
        self == f32::NEG_INFINITY
    }
}

impl FloatClassify for f64 {
    #[inline(always)]
    fn is_nan_value(self) -> bool {
        self.is_nan()
    }

    #[inline(always)]
    fn is_finite_value(self) -> bool {
        self.is_finite()
    }

    #[inline(always)]
    fn is_positive_infinity(self) -> bool {
        self == f64::INFINITY
    }

    #[inline(always)]
    fn is_negative_infinity(self) -> bool {
        self == f64::NEG_INFINITY
    }
}

/// Applies one of the multiversioned rounding-family kernels over a whole
/// tensor, in blocks, parallel above the cheap-unary threshold.
///
/// The bodies live in `ops::simd`, and have to, because rounding a vector needs
/// an instruction the x86-64 baseline does not carry. Written as a plain
/// closure through `unary_map`, every one of these compiled to a `libm` call
/// per element and ran 8.7x slower than it had to; handing over a block at a
/// time is what lets the loop sit inside the multiversioned function.
macro_rules! rounding_op {
    ($name:ident, $ty:ty, $dtype:ident, $accessor:ident, $kernel:ident, $tyname:literal) => {
        pub(crate) fn $name(tensor: &Tensor, param: $ty) -> Result<TensorData> {
            let input_data = tensor.data().$accessor().ok_or_else(|| {
                MinitensorError::internal_error(concat!(
                    "Failed to get ",
                    $tyname,
                    " slice from input tensor"
                ))
            })?;
            // SAFETY: the kernel writes every element of each block it is
            // given, and the driver's blocks tile the output.
            let out = unsafe {
                crate::ops::map::unary_map_blocks_threshold(
                    input_data,
                    crate::ops::map::PAR_THRESHOLD,
                    |src, dst| crate::ops::simd::$kernel(src, dst, param),
                )
            };
            Ok(TensorData::from_vec::<$ty>(
                out,
                DataType::$dtype,
                tensor.device(),
            ))
        }
    };
}

// Ties go to the even neighbour, as Python's built-in `round` does. Rust's
// `f32::round` rounds halves away from zero instead, which disagreed at every
// exact .5: round(0.5) gave 1 rather than 0, and round(2.5) gave 3 rather than
// 2. The kernels use `round_ties_even`, which is the mode `roundps`/`roundpd`
// implement, so the vectorized form needs no correction to agree.
rounding_op!(
    round_f32_scaled,
    f32,
    Float32,
    as_f32_slice,
    round_f32_blocks,
    "f32"
);
rounding_op!(
    round_f64_scaled,
    f64,
    Float64,
    as_f64_slice,
    round_f64_blocks,
    "f64"
);
rounding_op!(
    floor_f32,
    f32,
    Float32,
    as_f32_slice,
    floor_f32_blocks,
    "f32"
);
rounding_op!(
    floor_f64,
    f64,
    Float64,
    as_f64_slice,
    floor_f64_blocks,
    "f64"
);
rounding_op!(ceil_f32, f32, Float32, as_f32_slice, ceil_f32_blocks, "f32");
rounding_op!(ceil_f64, f64, Float64, as_f64_slice, ceil_f64_blocks, "f64");

fn round_f32(tensor: &Tensor, decimals: i32) -> Result<TensorData> {
    round_f32_scaled(tensor, 10.0_f32.powi(decimals))
}

fn round_f64(tensor: &Tensor, decimals: i32) -> Result<TensorData> {
    round_f64_scaled(tensor, 10.0_f64.powi(decimals))
}

#[cfg(test)]
mod rounding_and_clip_tests {
    use super::*;
    use crate::device::Device;
    use crate::tensor::{Shape, TensorData};
    use std::sync::Arc;

    fn f32_tensor(data: Vec<f32>) -> Tensor {
        let shape = Shape::new(vec![data.len()]);
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

    /// The rounding family runs a different compilation of its loop depending
    /// on what the CPU offers, and only one of them is exercised on any given
    /// machine. What must not vary is the answer: these are the inputs where a
    /// rounding mode can be got wrong.
    #[test]
    fn rounding_family_agrees_with_the_scalar_definition() {
        // Halves in both signs (ties-to-even is the point), values either side
        // of an integer, negatives, zeroes with their signs, and the specials.
        let values: Vec<f32> = vec![
            -3.5,
            -2.5,
            -1.5,
            -0.5,
            0.5,
            1.5,
            2.5,
            3.5,
            -1.25,
            -0.75,
            0.75,
            1.25,
            -0.0,
            0.0,
            1.0,
            -1.0,
            1e7,
            -1e7,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
        ];
        // Long enough to cross the parallel threshold and leave a partial
        // block, repeating the awkward values at every offset within one.
        let long: Vec<f32> = (0..(1 << 17) + 37)
            .map(|i| values[i % values.len()])
            .collect();

        for data in [values.clone(), long] {
            let t = f32_tensor(data.clone());

            let got = floor(&t).unwrap();
            for (i, (&g, &x)) in got
                .data()
                .as_f32_slice()
                .unwrap()
                .iter()
                .zip(&data)
                .enumerate()
            {
                let want = x.floor();
                assert!(
                    g.to_bits() == want.to_bits() || (g.is_nan() && want.is_nan()),
                    "floor({x}) = {g}, want {want} (index {i})"
                );
            }

            let got = ceil(&t).unwrap();
            for (i, (&g, &x)) in got
                .data()
                .as_f32_slice()
                .unwrap()
                .iter()
                .zip(&data)
                .enumerate()
            {
                let want = x.ceil();
                assert!(
                    g.to_bits() == want.to_bits() || (g.is_nan() && want.is_nan()),
                    "ceil({x}) = {g}, want {want} (index {i})"
                );
            }

            let got = round(&t, 0).unwrap();
            for (i, (&g, &x)) in got
                .data()
                .as_f32_slice()
                .unwrap()
                .iter()
                .zip(&data)
                .enumerate()
            {
                let want = x.round_ties_even();
                assert!(
                    g.to_bits() == want.to_bits() || (g.is_nan() && want.is_nan()),
                    "round({x}) = {g}, want {want} (index {i})"
                );
            }
        }
    }

    /// Halves round to the even neighbour, not away from zero, at every
    /// decimal place -- the property that makes this `round` agree with
    /// Python's built-in rather than with `f32::round`.
    #[test]
    fn round_breaks_ties_toward_even() {
        let t = f32_tensor(vec![0.5, 1.5, 2.5, 3.5, -0.5, -1.5, -2.5]);
        assert_eq!(
            round(&t, 0).unwrap().data().as_f32_slice().unwrap(),
            &[0.0, 2.0, 2.0, 4.0, -0.0, -2.0, -2.0]
        );
        let t = f32_tensor(vec![0.125, 0.375, -0.125, -0.375]);
        assert_eq!(
            round(&t, 2).unwrap().data().as_f32_slice().unwrap(),
            &[0.12, 0.38, -0.12, -0.38]
        );
    }

    /// NaN is not a value to be clamped into range: it has no order relative to
    /// the bounds, so it comes back untouched. This is what the kernel's
    /// comparison form gives for free, and what an `f32::max` would get wrong
    /// by returning whichever operand is not NaN.
    #[test]
    fn clip_passes_nan_through_and_honours_one_sided_bounds() {
        let data = vec![f32::NAN, -5.0, 0.0, 5.0, f32::INFINITY, f32::NEG_INFINITY];
        let t = f32_tensor(data.clone());

        let both = clip(&t, Some(-1.0), Some(1.0)).unwrap();
        let out = both.data().as_f32_slice().unwrap();
        assert!(out[0].is_nan(), "NaN was clamped to {}", out[0]);
        assert_eq!(&out[1..], &[-1.0, 0.0, 1.0, 1.0, -1.0]);

        let lower = clip(&t, Some(-1.0), None).unwrap();
        let out = lower.data().as_f32_slice().unwrap();
        assert!(out[0].is_nan());
        assert_eq!(&out[1..], &[-1.0, 0.0, 5.0, f32::INFINITY, -1.0]);

        let upper = clip(&t, None, Some(1.0)).unwrap();
        let out = upper.data().as_f32_slice().unwrap();
        assert!(out[0].is_nan());
        assert_eq!(&out[1..], &[-5.0, 0.0, 1.0, 1.0, f32::NEG_INFINITY]);

        // No bounds at all is the identity, NaN included.
        let neither = clip(&t, None, None).unwrap();
        let out = neither.data().as_f32_slice().unwrap();
        assert!(out[0].is_nan());
        assert_eq!(&out[1..], &data[1..]);
    }

    /// A reversed interval resolves to the upper bound, because the lower bound
    /// is applied first.
    #[test]
    fn clip_with_reversed_bounds_yields_the_upper_one() {
        let t = f32_tensor(vec![-10.0, 0.0, 10.0]);
        assert_eq!(
            clip(&t, Some(5.0), Some(1.0))
                .unwrap()
                .data()
                .as_f32_slice()
                .unwrap(),
            &[1.0, 1.0, 1.0]
        );
    }

    /// `reciprocal` takes a division rather than `powf(x, -1)`. The two must
    /// agree bit for bit, including where IEEE has something specific to say.
    #[test]
    fn reciprocal_matches_a_division_exactly() {
        let data: Vec<f32> = vec![
            1.0,
            -1.0,
            2.0,
            -2.0,
            0.5,
            -0.5,
            3.0,
            1e-30,
            1e30,
            0.0,
            -0.0,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            f32::MIN_POSITIVE,
        ];
        let got = reciprocal(&f32_tensor(data.clone())).unwrap();
        for (&g, &x) in got.data().as_f32_slice().unwrap().iter().zip(&data) {
            let want = 1.0f32 / x;
            assert!(
                g.to_bits() == want.to_bits() || (g.is_nan() && want.is_nan()),
                "reciprocal({x}) = {g}, want {want}"
            );
        }
    }
}
