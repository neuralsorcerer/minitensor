// Copyright (c) Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

use super::*;
use crate::autograd::AbsBackward;
use crate::autograd::ClampBackward;
use crate::autograd::IdentityBackward;
use crate::autograd::NanToNumBackward;
use crate::autograd::RsqrtBackward;
use crate::autograd::SqrtBackward;
use crate::{
    autograd::with_grad_fn,
    error::{MinitensorError, Result},
    tensor::{DataType, Tensor, TensorData},
};
use std::sync::Arc;

/// Wraps the result of a step function: same shape and dtype as the input,
/// never differentiable.
///
/// `sign`, `floor`, `ceil`, `trunc` and `round` all have a derivative that is
/// zero wherever it exists, so no gradient is worth recording. The output is a
/// constant rather than a tensor that claims `requires_grad` and then behaves
/// as a leaf -- which is what propagating the input's flag without attaching a
/// gradient function produced. Matches `norm(p = 0)`.
fn step_function_output(tensor: &Tensor, data: TensorData) -> Tensor {
    Tensor::new(
        Arc::new(data),
        tensor.shape().clone(),
        tensor.dtype(),
        tensor.device(),
        false,
    )
}

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

    Ok(step_function_output(tensor, output_data))
}

/// Square root function.
///
/// A dedicated op (rather than `x.powf(0.5)`) so the forward path uses the
/// hardware `sqrt`, which follows IEEE for the edge cases `powf` gets wrong:
/// `sqrt(-inf)` and `sqrt(x<0)` are NaN, `sqrt(-0.0)` is `-0.0`. Gradients flow
/// through [`SqrtBackward`].
pub fn sqrt(tensor: &Tensor) -> Result<Tensor> {
    // An integer argument widens rather than being refused: none of these has
    // an integer answer, and refusing one would only make every caller cast first.
    if let Some(widened) = crate::ops::util::widen_integer_input(tensor)? {
        return sqrt(&widened);
    }
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
    // An integer argument widens rather than being refused: none of these has
    // an integer answer, and refusing one would only make every caller cast first.
    if let Some(widened) = crate::ops::util::widen_integer_input(tensor)? {
        return rsqrt(&widened);
    }
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
    // `1/2` is a half, so an integer argument widens rather than being
    // refused. Reading it as integer division instead would answer 0 for every
    // magnitude above 1, a result nobody asking for a reciprocal wants; the
    // rest of this family widens too.
    if let Some(widened) = crate::ops::util::widen_integer_input(tensor)? {
        return reciprocal(&widened);
    }
    match tensor.dtype() {
        DataType::Float32 | DataType::Float64 => powf(tensor, -1.0),
        _ => Err(MinitensorError::invalid_operation(
            "Reciprocal only supported for floating point tensors",
        )),
    }
}

/// Clip tensor values to range
pub fn clip(tensor: &Tensor, min_val: Option<f64>, max_val: Option<f64>) -> Result<Tensor> {
    // The rule `hardtanh` states: a clamp answers with a bound or the value
    // itself, so an integer keeps its dtype exactly when the bounds are whole.
    // A fractional bound has no integer answer, and truncating it onto the
    // integer gave a wrong one -- `clamp([1, 5], 0, 2.5)` came back `[1, 2]` --
    // so it widens. An infinite bound clamps nothing and counts as whole.
    let fractional = |bound: Option<f64>| bound.is_some_and(|b| b.is_finite() && b.fract() != 0.0);
    if matches!(tensor.dtype(), DataType::Int32 | DataType::Int64)
        && (fractional(min_val) || fractional(max_val))
    {
        let widened =
            crate::ops::util::widen_integer_input(tensor)?.expect("an integer dtype always widens");
        return clip(&widened, min_val, max_val);
    }
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

/// One integer rounded to a multiple of `step`, halves going to the even
/// multiple.
///
/// The rule floating point `round` already follows, applied to the integers so
/// the two agree: `np.round([15, 25], -1)` is `[20, 20]`, not `[20, 30]`.
/// Everything is done in `i128` because `step` can be larger than the dtype
/// being rounded, and `2 * remainder` larger again.
#[inline]
fn round_integer_to_step(value: i128, step: i128) -> i128 {
    let quotient = value.div_euclid(step);
    let twice_remainder = 2 * value.rem_euclid(step);
    let rounded = match twice_remainder.cmp(&step) {
        std::cmp::Ordering::Greater => quotient + 1,
        std::cmp::Ordering::Less => quotient,
        // Exactly half: take whichever neighbouring multiple is even.
        std::cmp::Ordering::Equal => quotient + (quotient & 1),
    };
    rounded * step
}

/// `10^power`, or `None` once it leaves `i128`.
///
/// A step that large is bigger than twice any value being rounded, so every
/// one of them rounds to zero.
fn power_of_ten(power: u32) -> Option<i128> {
    (0..power).try_fold(1i128, |acc, _| acc.checked_mul(10))
}

/// Round tensor values.
///
/// An integer is already whole, so a non-negative `decimals` leaves it alone;
/// a negative one rounds it to a multiple of a power of ten, in the dtype it
/// came in.
pub fn round(tensor: &Tensor, decimals: i32) -> Result<Tensor> {
    if matches!(tensor.dtype(), DataType::Int32 | DataType::Int64) {
        if decimals >= 0 {
            return Ok(tensor.clone());
        }
        let step = power_of_ten(decimals.unsigned_abs());
        macro_rules! round_integer_arm {
            ($accessor:ident, $ty:ty, $dtype:ident, $tyname:literal) => {{
                let input = tensor.data().$accessor().ok_or_else(|| {
                    MinitensorError::internal_error(concat!(
                        "Failed to get ",
                        $tyname,
                        " slice from input tensor"
                    ))
                })?;
                let rounded = match step {
                    // The cast back wraps, which is the rule integer `pow`
                    // follows too: the exact answer modulo the dtype's width.
                    // Only reachable by rounding a value within a step of the
                    // dtype's limit.
                    Some(step) => crate::ops::map::unary_map(input, move |x: $ty| {
                        round_integer_to_step(x as i128, step) as $ty
                    }),
                    None => vec![0 as $ty; input.len()],
                };
                TensorData::from_vec::<$ty>(rounded, DataType::$dtype, tensor.device())
            }};
        }
        let output_data = match tensor.dtype() {
            DataType::Int32 => round_integer_arm!(as_i32_slice, i32, Int32, "i32"),
            DataType::Int64 => round_integer_arm!(as_i64_slice, i64, Int64, "i64"),
            _ => unreachable!("guarded by the matches! above"),
        };
        return Ok(step_function_output(tensor, output_data));
    }

    let output_data = match tensor.dtype() {
        DataType::Float32 => round_f32(tensor, decimals)?,
        DataType::Float64 => round_f64(tensor, decimals)?,
        _ => {
            return Err(MinitensorError::invalid_operation(
                "Round only supported for floating point and integer tensors",
            ));
        }
    };

    Ok(step_function_output(tensor, output_data))
}

/// The rounding modes that take no parameter. `round` is written out
/// separately because it takes `decimals`.
///
/// An integer is already rounded, so these are the identity on one and return
/// it unchanged rather than refusing it -- which is what `relu`, `abs` and
/// `sign` already did here. Unlike the real-valued ops there
/// is nothing to widen to: the answer is an integer and the dtype stays.
macro_rules! float_rounding_op {
    ($name:ident, $f32_kernel:ident, $f64_kernel:ident, $label:literal, $doc:literal) => {
        #[doc = $doc]
        pub fn $name(tensor: &Tensor) -> Result<Tensor> {
            let output_data = match tensor.dtype() {
                DataType::Float32 => $f32_kernel(tensor, 0.0)?,
                DataType::Float64 => $f64_kernel(tensor, 0.0)?,
                DataType::Int32 | DataType::Int64 => return Ok(tensor.clone()),
                _ => {
                    return Err(MinitensorError::invalid_operation(concat!(
                        $label,
                        " only supported for floating point and integer tensors"
                    )));
                }
            };

            Ok(step_function_output(tensor, output_data))
        }
    };
}

float_rounding_op!(
    floor,
    floor_f32,
    floor_f64,
    "Floor",
    "Round towards negative infinity."
);
float_rounding_op!(
    ceil,
    ceil_f32,
    ceil_f64,
    "Ceiling",
    "Round towards positive infinity."
);
float_rounding_op!(
    trunc,
    trunc_f32,
    trunc_f64,
    "Truncation",
    "Round towards zero, discarding the fractional part."
);

/// Element-wise fractional part, `x - trunc(x)`, which carries `x`'s sign.
///
/// The only differentiable member of the family, and for the same reason the
/// others are not: `trunc` is a step function with zero derivative, so
/// `d/dx frac(x) = 1` wherever it exists and the gradient passes straight
/// through. Non-finite inputs give NaN, since `inf - inf` is what the
/// definition asks for.
pub fn frac(tensor: &Tensor) -> Result<Tensor> {
    // An integer has no fractional part, so the answer is zeros in its own
    // dtype -- the same reading that makes `floor` the identity on one.
    if matches!(tensor.dtype(), DataType::Int32 | DataType::Int64) {
        return Ok(Tensor::zeros(
            tensor.shape().clone(),
            tensor.dtype(),
            tensor.device(),
            false,
        ));
    }
    let output_data = match tensor.dtype() {
        DataType::Float32 => frac_f32(tensor, 0.0)?,
        DataType::Float64 => frac_f64(tensor, 0.0)?,
        _ => {
            return Err(MinitensorError::invalid_operation(
                "Fractional part only supported for floating point and integer tensors",
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
        return with_grad_fn(
            output,
            Arc::new(IdentityBackward {
                input_id: tensor.id(),
            }),
        );
    }

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
    // The float32 root and reciprocal each round, and the two roundings left
    // results up to 1.43 ulp out. One Newton step in float64 corrects them:
    // with `e = 1 - x r^2`, the exact answer is `r (1 + e/2 + 3e^2/8 + ...)`,
    // and at this `e` (below 2^-21) the terms kept leave an error near 2^-53,
    // far finer than float32 can show -- so the one rounding at the end is
    // the only one that matters. Taking the root and the reciprocal in
    // float64 instead does the same, at twice the cost of this.
    let out = unary_map(input_data, |v: f32| {
        let estimate = v.sqrt().recip();
        let (x, r) = (f64::from(v), f64::from(estimate));
        // `r * r` is exact in float64; the one rounding is in `x * r^2`.
        let e = 1.0 - x * (r * r);
        let refined = r + r * e * (0.5 + 0.375 * e);
        // Zero, infinity and NaN are exact already, and the step would turn
        // `0 * inf` into NaN.
        if estimate.is_finite() && estimate > 0.0 {
            refined as f32
        } else {
            estimate
        }
    });
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

    // `wrapping_abs`, not `abs`: the magnitude of i32::MIN is one past
    // what i32 can hold, so `abs` is a panic in a build with overflow
    // checks and a wrap without them -- the same input aborting in
    // development and returning a value in production. Wrapping says which
    // of the two this is, everywhere: `abs(i32::MIN)` is `i32::MIN`, which
    // is what the release build already produced and what two's complement
    // leaves no alternative to.
    let out = unary_map(input_data, |v: i32| v.wrapping_abs());
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

    // `wrapping_abs`, not `abs`: the magnitude of i64::MIN is one past
    // what i64 can hold, so `abs` is a panic in a build with overflow
    // checks and a wrap without them -- the same input aborting in
    // development and returning a value in production. Wrapping says which
    // of the two this is, everywhere: `abs(i64::MIN)` is `i64::MIN`, which
    // is what the release build already produced and what two's complement
    // leaves no alternative to.
    let out = unary_map(input_data, |v: i64| v.wrapping_abs());
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
    // `move`: captured by reference, the three replacements could alias the
    // output as far as LLVM can tell, and the loop stayed scalar (12us at
    // 16,384 float32 elements, against 3us moved). Three selects a lane is
    // what `unary_map_selecting` is for.
    let out = crate::ops::map::unary_map_selecting(input, move |val| {
        classify_nan_to_num(val, nan, posinf, neginf)
    });
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
    round_even_f32,
    f32,
    Float32,
    as_f32_slice,
    round_even_f32_blocks,
    "f32"
);
rounding_op!(
    round_even_f64,
    f64,
    Float64,
    as_f64_slice,
    round_even_f64_blocks,
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
rounding_op!(
    trunc_f32,
    f32,
    Float32,
    as_f32_slice,
    trunc_f32_blocks,
    "f32"
);
rounding_op!(
    trunc_f64,
    f64,
    Float64,
    as_f64_slice,
    trunc_f64_blocks,
    "f64"
);
rounding_op!(frac_f32, f32, Float32, as_f32_slice, frac_f32_blocks, "f32");
rounding_op!(frac_f64, f64, Float64, as_f64_slice, frac_f64_blocks, "f64");

fn round_f32(tensor: &Tensor, decimals: i32) -> Result<TensorData> {
    if decimals == 0 {
        return round_even_f32(tensor, 0.0);
    }
    let input = tensor
        .data()
        .as_f32_slice()
        .ok_or_else(|| MinitensorError::internal_error("Failed to get f32 slice from input"))?;
    let out = crate::ops::map::unary_map(input, move |x: f32| round_decimal_f32(x, decimals));
    Ok(TensorData::from_vec::<f32>(
        out,
        DataType::Float32,
        tensor.device(),
    ))
}

fn round_f64(tensor: &Tensor, decimals: i32) -> Result<TensorData> {
    if decimals == 0 {
        return round_even_f64(tensor, 0.0);
    }
    let input = tensor
        .data()
        .as_f64_slice()
        .ok_or_else(|| MinitensorError::internal_error("Failed to get f64 slice from input"))?;
    let out = crate::ops::map::unary_map(input, move |x: f64| round_decimal_f64(x, decimals));
    Ok(TensorData::from_vec::<f64>(
        out,
        DataType::Float64,
        tensor.device(),
    ))
}

// --- rounding to a number of decimal places ----------------------------------
//
// `round(x, d)` is the float nearest `N / 10^d`, for `N` the integer nearest
// the exact value of `x * 10^d`, halves to even -- which is what Python's own
// `round` gives. Scaling in floating point, rounding, and scaling back rounded
// the product before deciding which integer was nearest, so `2.675`, which as
// a double is 2.67499999999999982236431605997495353221893310546875, came back
// as `2.68`: its product with 100 rounds to 267.5 exactly, and the tie went to
// the even neighbour. Each step below carries its rounding instead.

/// The powers of ten a double holds exactly: `5^22` is the last power of five
/// under `2^53`.
const EXACT_POWERS_OF_TEN: [f64; 23] = [
    1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10, 1e11, 1e12, 1e13, 1e14, 1e15, 1e16,
    1e17, 1e18, 1e19, 1e20, 1e21, 1e22,
];

/// `2^53`: from here up every double is an integer, and past it in a scaled
/// value the scaling is finer than the value's own spacing.
const TWO_TO_53: f64 = 9_007_199_254_740_992.0;

/// `value` rounded to an integer, halves to even, for `|value|` below `2^53`.
///
/// Adding `2^52` and taking it away again rounds in the floating-point unit's
/// own mode, which is this one. `round_ties_even` is the same function, but on
/// the baseline x86-64 target it is a call into `libm` per element, which made
/// `round(x, 2)` cost twice what the rest of it did.
#[inline(always)]
fn round_half_even(value: f64) -> f64 {
    const TWO_TO_52: f64 = 4_503_599_627_370_496.0;
    if value.abs() < TWO_TO_52 {
        let shift = TWO_TO_52.copysign(value);
        // A zero keeps its sign: `-0.4 + -2^52 - -2^52` is `+0`.
        ((value + shift) - shift).copysign(value)
    } else {
        value
    }
}

/// The integer nearest `value + error`, halves to even, where `value` is a
/// rounded result below `2^53` in magnitude and `error` is what its rounding
/// left out -- at most half an ulp of `value`.
///
/// The nearest integer to `value` itself is the answer except where `value`
/// lies exactly halfway between two integers, when the error decides which
/// side the exact value is on, or where `value` is an integer an ulp of one
/// from the next and the error is exactly a half, which is a tie of its own.
/// Anywhere else no integer boundary lies between the two.
#[inline]
fn nearest_integer_of(value: f64, error: f64) -> f64 {
    let nearest = round_half_even(value);
    let offset = value - nearest;
    if offset.abs() == 0.5 {
        if error != 0.0 && (error > 0.0) == (offset > 0.0) {
            nearest + offset.signum()
        } else {
            nearest
        }
    } else if offset == 0.0 && error.abs() == 0.5 {
        let other = nearest + error.signum();
        if (nearest * 0.5).fract() == 0.0 {
            nearest
        } else {
            other
        }
    } else {
        nearest
    }
}

/// `N` for `round(x, decimals)` with `|decimals|` at most 22; `None` when `x`
/// is already its own answer -- its scaled value past `2^53`, where the
/// decimal grid is finer than the float's spacing.
///
/// The scaled value rounded to the nearest integer is `N` except where it is
/// exactly halfway, or an integer with an ulp of one; only then is its
/// rounding error worked out, so the common case costs what the scaling did.
#[inline]
fn decimal_integer(x: f64, decimals: i32) -> Option<f64> {
    use crate::ops::simd::product_error;
    let scale = EXACT_POWERS_OF_TEN[decimals.unsigned_abs() as usize];
    // An integer already lies on every finer grid. (Not on a coarser one: a
    // negative `decimals` still rounds it.)
    if decimals > 0 && x.abs() >= TWO_TO_53 {
        return None;
    }
    let scaled = if decimals > 0 { x * scale } else { x / scale };
    if scaled.abs() >= TWO_TO_53 {
        return None;
    }
    let nearest = round_half_even(scaled);
    if (scaled - nearest).abs() != 0.5 && scaled.abs() < TWO_TO_53 * 0.5 {
        return Some(nearest);
    }
    let error = if decimals > 0 {
        product_error::<false>(x, scale, scaled)
    } else {
        // The remainder of a correctly rounded quotient is a double, and these
        // two steps form it exactly; over the divisor, its sign and whether it
        // is a half are what the decision needs, and both survive the division.
        let back = scaled * scale;
        ((x - back) - product_error::<false>(scaled, scale, back)) / scale
    };
    Some(nearest_integer_of(scaled, error))
}

/// `round(x, decimals)` in float64; see the section note.
fn round_decimal_f64(x: f64, decimals: i32) -> f64 {
    if !x.is_finite() || x == 0.0 {
        return x;
    }
    if decimals.unsigned_abs() > 22 {
        return round_decimal_by_text(x, decimals, |text| text.parse::<f64>().ok());
    }
    let Some(n) = decimal_integer(x, decimals) else {
        return x;
    };
    // One correctly rounded operation on two exact operands.
    let scale = EXACT_POWERS_OF_TEN[decimals.unsigned_abs() as usize];
    if decimals > 0 { n / scale } else { n * scale }
}

/// `round(x, decimals)` in float32: `N` found in float64, where the float32
/// is exact, and `N / 10^d` rounded to float32 once. Through float64 it would
/// round twice, which is wrong exactly where the float64 lands on a midpoint
/// between two float32s; there the float64's own rounding error says which
/// way the exact value lies.
fn round_decimal_f32(x: f32, decimals: i32) -> f32 {
    use crate::ops::simd::product_error;
    if !x.is_finite() || x == 0.0 {
        return x;
    }
    if decimals.unsigned_abs() > 22 {
        return round_decimal_by_text(x, decimals, |text| text.parse::<f32>().ok());
    }
    let Some(n) = decimal_integer(f64::from(x), decimals) else {
        return x;
    };
    let scale = EXACT_POWERS_OF_TEN[decimals.unsigned_abs() as usize];
    if decimals > 0 {
        let quotient = n / scale;
        float32_nearest(quotient, || {
            let back = quotient * scale;
            (n - back) - product_error::<false>(quotient, scale, back)
        })
    } else {
        let product = n * scale;
        float32_nearest(product, || product_error::<false>(n, scale, product))
    }
}

/// The float32 nearest the exact value `value` was rounded from, given the
/// sign of that rounding's error, which `error` works out only if asked.
///
/// Rounding `value` to float32 is right unless it is exactly halfway between
/// two float32s, both of which are float64s: no other float64 can stand
/// between the exact answer and that midpoint. Every value this sees is zero,
/// a normal float32 in range, or past the top of it -- the smallest nonzero
/// answer is `1e-22` -- and halfway between two normal float32s a float64 has
/// its low 29 significand bits exactly `1 << 28`. At a midpoint the value is
/// nudged one float64 ulp towards the exact answer, which takes it off the
/// midpoint and onto the right side of it.
#[inline]
fn float32_nearest(value: f64, error: impl FnOnce() -> f64) -> f32 {
    const BELOW_FLOAT32: u64 = (1 << 29) - 1;
    if value.to_bits() & BELOW_FLOAT32 != 1 << 28 {
        return value as f32;
    }
    let error = error();
    if error > 0.0 {
        value.next_up() as f32
    } else if error < 0.0 {
        value.next_down() as f32
    } else {
        value as f32
    }
}

/// `round(x, decimals)` for `|decimals|` past 22, where no power of ten the
/// scaling needs is a double: through the decimal text of `x`, which the
/// standard library writes exactly and reads back to the nearest float.
///
/// Past 22 decimals a value at least `1e-6` is already its own answer, its
/// scaled value being past `2^53`. Below -22 the step is at least `1e23`, and
/// every float below `2^53` is nearer zero than half of it; the rest are
/// integers, written out in full and rounded at the step's digit.
fn round_decimal_by_text<T>(x: T, decimals: i32, parse: impl Fn(&str) -> Option<T>) -> T
where
    T: num_traits::Float + std::fmt::Display,
{
    let magnitude = x.abs().to_f64().unwrap_or(f64::INFINITY);
    let zero = T::zero().copysign(x);
    if decimals > 0 {
        if magnitude >= 1e-6 {
            return x;
        }
        return parse(&format!("{:.*}", decimals as usize, x)).unwrap_or(x);
    }
    let step = decimals.unsigned_abs() as usize;
    if magnitude < TWO_TO_53 {
        return zero;
    }
    let digits = format!("{:.0}", x.abs());
    if digits.len() < step {
        return zero;
    }
    let (head, tail) = digits.split_at(digits.len() - step);
    let rest_is_zero = tail[1..].bytes().all(|b| b == b'0');
    let odd = head.bytes().last().is_some_and(|b| (b - b'0') % 2 == 1);
    let up = match tail.as_bytes()[0].cmp(&b'5') {
        std::cmp::Ordering::Greater => true,
        std::cmp::Ordering::Less => false,
        std::cmp::Ordering::Equal => !rest_is_zero || odd,
    };
    let mut head: Vec<u8> = if head.is_empty() {
        vec![b'0']
    } else {
        head.as_bytes().to_vec()
    };
    if up {
        // Add one to the decimal digits, carrying.
        let mut carry = true;
        for digit in head.iter_mut().rev() {
            if *digit == b'9' {
                *digit = b'0';
            } else {
                *digit += 1;
                carry = false;
                break;
            }
        }
        if carry {
            head.insert(0, b'1');
        }
    }
    if head.iter().all(|&b| b == b'0') {
        return zero;
    }
    let sign = if x < T::zero() { "-" } else { "" };
    let head = String::from_utf8(head).unwrap_or_default();
    parse(&format!("{sign}{head}e{step}")).unwrap_or(x)
}

#[cfg(test)]
mod rounding_and_clip_tests {
    use super::*;
    use crate::device::Device;

    #[test]
    fn decimal_rounding_decides_on_the_exact_value() {
        // As a double each of these is just below its written value, and
        // the nearest two-decimal value is the lower one.
        assert_eq!(round_decimal_f64(2.675, 2), 2.67);
        assert_eq!(round_decimal_f64(1.005, 2), 1.0);
        assert_eq!(round_decimal_f64(0.125, 2), 0.12);
        assert_eq!(round_decimal_f64(0.375, 2), 0.38);
        assert_eq!(round_decimal_f64(-2.675, 2), -2.67);
        assert!(round_decimal_f64(-0.004, 2).is_sign_negative());
        assert_eq!(round_decimal_f64(25.0, -1), 20.0);
        assert_eq!(round_decimal_f64(35.0, -1), 40.0);
        // An integer with an ulp of one, its scaled value exactly a half.
        assert_eq!(round_decimal_f64(450359962737049.75, 1), 450359962737049.8);
        // The float32 nearest 2.675 is 2.674999952..., below it.
        assert_eq!(round_decimal_f32(2.675, 2), 2.67);
    }

    #[test]
    fn a_nearest_integer_is_the_rounding_of_the_exact_sum() {
        // Halfway as rounded: the error decides.
        assert_eq!(nearest_integer_of(2.5, 1e-17), 3.0);
        assert_eq!(nearest_integer_of(2.5, -1e-17), 2.0);
        assert_eq!(nearest_integer_of(2.5, 0.0), 2.0);
        assert_eq!(nearest_integer_of(-3.5, 1e-17), -3.0);
        assert_eq!(nearest_integer_of(-3.5, -1e-17), -4.0);
        // An integer with an ulp of one and the exact value a half past it.
        let odd = 4_503_599_627_370_497.0;
        assert_eq!(nearest_integer_of(odd, 0.5), odd + 1.0);
        assert_eq!(nearest_integer_of(odd, -0.5), odd - 1.0);
        assert_eq!(nearest_integer_of(odd + 1.0, 0.5), odd + 1.0);
        assert_eq!(round_half_even(-0.4).to_bits(), (-0.0f64).to_bits());
        assert_eq!(round_half_even(0.5), 0.0);
        assert_eq!(round_half_even(1.5), 2.0);
    }

    #[test]
    fn a_float32_midpoint_is_settled_by_the_side_the_exact_value_is_on() {
        // Halfway between 1 and the float32 after it.
        let midpoint = 1.0 + f64::powi(2.0, -24);
        let above = 1.0f32.next_up();
        assert_eq!(float32_nearest(midpoint, || 1e-20), above);
        assert_eq!(float32_nearest(midpoint, || -1e-20), 1.0);
        // An exact tie goes to the even neighbour, as a cast does.
        assert_eq!(float32_nearest(midpoint, || 0.0), 1.0);
        assert_eq!(float32_nearest(-midpoint, || -1e-20), -above);
        // Off a midpoint the error is never asked for.
        assert_eq!(float32_nearest(0.1, || unreachable!()), 0.1f32);
    }

    #[test]
    fn decimals_past_the_exact_powers_round_through_the_text() {
        assert_eq!(round_decimal_f64(1.25e-30, 31), 1.2e-30);
        // The double nearest 1.35e-30 is below it; the one nearest 1.36e-30
        // is not near a half at all.
        assert_eq!(round_decimal_f64(1.35e-30, 31), 1.3e-30);
        assert_eq!(round_decimal_f64(1.36e-30, 31), 1.4e-30);
        assert_eq!(round_decimal_f64(0.5, 30), 0.5);
        // 1.5e23 is 150000000000000004194304 as a double, just past the
        // half, and 2.5e23 is just short of it.
        assert_eq!(round_decimal_f64(1.5e23, -23), 2e23);
        assert_eq!(round_decimal_f64(2.5e23, -23), 2e23);
        assert_eq!(round_decimal_f64(9.6e23, -23), 1e24);
        assert_eq!(round_decimal_f64(4.9e22, -23), 0.0);
        assert!(round_decimal_f64(-4.9e22, -23).is_sign_negative());
        // The float32 nearest 1.5e30 is 1.4999999e30, short of the half.
        assert_eq!(round_decimal_f32(1.5e30, -30), 1e30);
        assert_eq!(round_decimal_f32(1.6e30, -30), 2e30);
    }
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

    /// A positive finite float64 as an odd integer times a power of two.
    fn exact_parts(value: f64) -> (u128, i32) {
        let bits = value.to_bits();
        let exponent = ((bits >> 52) & 0x7ff) as i32;
        let fraction = bits & ((1 << 52) - 1);
        let (mut mantissa, mut power) = if exponent == 0 {
            (u128::from(fraction), -1074)
        } else {
            (u128::from(fraction | (1 << 52)), exponent - 1075)
        };
        let zeros = mantissa.trailing_zeros();
        mantissa >>= zeros;
        power += zeros as i32;
        (mantissa, power)
    }

    /// `bound^2 * x` against 1, decided exactly: `bound` a midpoint of two
    /// float32s and `x` a float32, so the product has at most 74 bits.
    fn squared_times_against_one(bound: f64, x: f32) -> std::cmp::Ordering {
        let (m, e) = exact_parts(bound);
        let (n, f) = exact_parts(f64::from(x));
        let product = m * m * n;
        let power = 2 * e + f;
        if power >= 0 {
            // An odd product times a non-negative power of two.
            (product, power).cmp(&(1, 0))
        } else if -power >= 127 {
            std::cmp::Ordering::Less
        } else {
            product.cmp(&(1u128 << (-power)))
        }
    }

    #[test]
    fn float32_rsqrt_is_correctly_rounded() {
        // Every 4099th positive float32 bit pattern, subnormals included:
        // the exact `1 / sqrt(x)` must lie within half an ulp of the answer,
        // which is to say between the midpoints either side of it.
        let inputs: Vec<f32> = (1..0x7f80_0000_u32)
            .step_by(4099)
            .map(f32::from_bits)
            .collect();
        let output = rsqrt(&f32_tensor(inputs.clone())).unwrap();
        let answers = output.data().as_f32_slice().unwrap();
        for (&x, &r) in inputs.iter().zip(answers) {
            assert!(r.is_finite() && r > 0.0, "rsqrt({x:e}) = {r:e}");
            let below = f64::from(f32::from_bits(r.to_bits() - 1));
            let above = f64::from(f32::from_bits(r.to_bits() + 1));
            let (low, high) = ((below + f64::from(r)) / 2.0, (f64::from(r) + above) / 2.0);
            // `1 / sqrt(x)` is at least `low` when `low^2 * x <= 1`, and at
            // most `high` when `high^2 * x >= 1`.
            assert!(
                squared_times_against_one(low, x).is_le()
                    && squared_times_against_one(high, x).is_ge(),
                "rsqrt({x:e}) = {r:e} is not the nearest float32"
            );
        }
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

        /// One rounding mode: its name, the tensor op, and the scalar it has
        /// to agree with element for element.
        type RoundingCase = (&'static str, fn(&Tensor) -> Result<Tensor>, fn(f32) -> f32);

        // Every member of the family, against the scalar it is named after.
        // `frac` is included because it is the one that has to agree with a
        // *difference* rather than with a libm call: `x - trunc(x)`.
        let cases: [RoundingCase; 4] = [
            ("floor", floor, f32::floor),
            ("ceil", ceil, f32::ceil),
            ("trunc", trunc, f32::trunc),
            ("frac", frac, |x| x - x.trunc()),
        ];

        for data in [values.clone(), long] {
            let t = f32_tensor(data.clone());

            let check = |name: &str, got: &Tensor, want: &dyn Fn(f32) -> f32| {
                for (i, (&g, &x)) in got
                    .data()
                    .as_f32_slice()
                    .unwrap()
                    .iter()
                    .zip(&data)
                    .enumerate()
                {
                    let want = want(x);
                    // Bit equality, so a `-0.0` answered as `0.0` fails.
                    assert!(
                        g.to_bits() == want.to_bits() || (g.is_nan() && want.is_nan()),
                        "{name}({x}) = {g}, want {want} (index {i})"
                    );
                }
            };

            for (name, op, scalar) in cases {
                check(name, &op(&t).unwrap(), &scalar);
            }
            check("round", &round(&t, 0).unwrap(), &f32::round_ties_even);
        }
    }

    /// `trunc` and `floor`/`ceil` are the same function on one side of zero
    /// each, which is the whole content of "rounds towards zero".
    #[test]
    fn trunc_is_floor_above_zero_and_ceil_below_it() {
        let positive = f32_tensor(vec![0.0, 0.25, 1.5, 2.75, 1e7, f32::INFINITY]);
        assert_eq!(
            trunc(&positive).unwrap().data().as_f32_slice().unwrap(),
            floor(&positive).unwrap().data().as_f32_slice().unwrap()
        );

        let negative = f32_tensor(vec![-0.0, -0.25, -1.5, -2.75, -1e7, f32::NEG_INFINITY]);
        assert_eq!(
            trunc(&negative).unwrap().data().as_f32_slice().unwrap(),
            ceil(&negative).unwrap().data().as_f32_slice().unwrap()
        );
    }

    /// `frac` keeps the sign of its input and reconstructs it exactly:
    /// `trunc(x) + frac(x) == x` with no rounding, for every finite `x`.
    #[test]
    fn frac_carries_the_sign_and_reconstructs_the_input() {
        let data = vec![2.75f32, -2.75, 0.5, -0.5, 0.0, -0.0, 1e7, -1e7, 3.0, -3.0];
        let t = f32_tensor(data.clone());
        let f = frac(&t).unwrap();
        let whole = trunc(&t).unwrap();

        let f = f.data().as_f32_slice().unwrap();
        let whole = whole.data().as_f32_slice().unwrap();
        for (i, &x) in data.iter().enumerate() {
            assert!(
                f[i].is_sign_negative() == x.is_sign_negative() || f[i] == 0.0,
                "frac({x}) = {} flipped the sign",
                f[i]
            );
            assert!(f[i].abs() < 1.0, "frac({x}) = {} is not a fraction", f[i]);
            assert_eq!(whole[i] + f[i], x, "trunc + frac did not rebuild {x}");
        }

        // Non-finite inputs have no finite whole part to subtract.
        let specials = f32_tensor(vec![f32::INFINITY, f32::NEG_INFINITY, f32::NAN]);
        assert!(
            frac(&specials)
                .unwrap()
                .data()
                .as_f32_slice()
                .unwrap()
                .iter()
                .all(|v| v.is_nan())
        );
    }

    /// The family is not differentiable, except `frac`, whose gradient is the
    /// one that comes through unchanged.
    #[test]
    fn only_frac_carries_a_gradient() {
        let t = f32_tensor(vec![1.5, -2.25]).requires_grad_(true);

        for op in [floor, ceil, trunc] {
            assert!(!op(&t).unwrap().requires_grad());
        }
        assert!(!round(&t, 0).unwrap().requires_grad());
        assert!(!sign(&t).unwrap().requires_grad());

        let out = frac(&t).unwrap();
        assert!(out.requires_grad());
        let seed = Tensor::ones(out.shape().clone(), out.dtype(), out.device(), false);
        let grads = crate::autograd::backward_collect(&out, Some(seed)).unwrap();
        assert_eq!(
            grads.get(&t.id()).unwrap().data().as_f32_slice().unwrap(),
            &[1.0, 1.0]
        );
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
