// Copyright (c) Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

use super::*;
use crate::autograd::MaskedLogSoftmaxBackward;
use crate::autograd::SoftmaxBackward;
use crate::ops::util::check_dim;
use crate::{
    autograd::with_grad_fn,
    error::{MinitensorError, Result},
    tensor::{DataType, Tensor, TensorData},
};
use libm::{erf, erfc};
use std::sync::Arc;

/// Masked softmax activation function with gradient support.
/// Masked positions are filled with zeros in the output.
pub fn masked_softmax(tensor: &Tensor, mask: &Tensor, dim: Option<usize>) -> Result<Tensor> {
    if mask.dtype() != DataType::Bool {
        return Err(MinitensorError::invalid_operation(
            "masked_softmax mask must have bool dtype",
        ));
    }

    if tensor.device() != mask.device() {
        return Err(MinitensorError::device_mismatch(
            format!("{:?}", tensor.device()),
            format!("{:?}", mask.device()),
        ));
    }

    let broadcast_shape = mask.shape().broadcast_with(tensor.shape())?;
    if &broadcast_shape != tensor.shape() {
        return Err(MinitensorError::shape_mismatch(
            mask.shape().dims().to_vec(),
            tensor.shape().dims().to_vec(),
        ));
    }

    if tensor.ndim() == 0 {
        let mut output_data =
            TensorData::uninitialized_on_device(tensor.numel(), tensor.dtype(), tensor.device());
        let mask_value = mask
            .data()
            .as_bool_slice()
            .ok_or_else(|| {
                MinitensorError::internal_error("Failed to get bool slice from mask tensor")
            })?
            .first()
            .copied()
            .unwrap_or(false);
        match tensor.dtype() {
            DataType::Float32 => {
                let output_slice = output_data.as_f32_slice_mut().ok_or_else(|| {
                    MinitensorError::internal_error(
                        "Failed to get mutable f32 slice from output data",
                    )
                })?;
                output_slice[0] = if mask_value { 0.0 } else { 1.0 };
            }
            DataType::Float64 => {
                let output_slice = output_data.as_f64_slice_mut().ok_or_else(|| {
                    MinitensorError::internal_error(
                        "Failed to get mutable f64 slice from output data",
                    )
                })?;
                output_slice[0] = if mask_value { 0.0 } else { 1.0 };
            }
            _ => {
                return Err(MinitensorError::invalid_operation(
                    "masked_softmax only supported for floating point tensors",
                ));
            }
        }

        let output = Tensor::new(
            Arc::new(output_data),
            tensor.shape().clone(),
            tensor.dtype(),
            tensor.device(),
            tensor.requires_grad(),
        );

        if output.requires_grad() {
            let grad_fn = Arc::new(SoftmaxBackward {
                input_id: tensor.id(),
                output: output.detach(),
                dim: 0,
            });

            return with_grad_fn(output, grad_fn);
        }

        return Ok(output);
    }

    let dim = dim.unwrap_or(tensor.ndim() - 1);

    check_dim(dim, tensor.ndim())?;

    let mut output_data =
        TensorData::uninitialized_on_device(tensor.numel(), tensor.dtype(), tensor.device());

    match tensor.dtype() {
        DataType::Float32 => masked_softmax_f32(tensor, mask, &mut output_data, dim)?,
        DataType::Float64 => masked_softmax_f64(tensor, mask, &mut output_data, dim)?,
        _ => {
            return Err(MinitensorError::invalid_operation(
                "masked_softmax only supported for floating point tensors",
            ));
        }
    }

    let output = Tensor::new(
        Arc::new(output_data),
        tensor.shape().clone(),
        tensor.dtype(),
        tensor.device(),
        tensor.requires_grad(),
    );

    if output.requires_grad() {
        let grad_fn = Arc::new(SoftmaxBackward {
            input_id: tensor.id(),
            output: output.detach(),
            dim,
        });

        with_grad_fn(output, grad_fn)
    } else {
        Ok(output)
    }
}

/// Masked log-softmax activation function with gradient support.
/// Masked positions are filled with -inf in the output.
pub fn masked_log_softmax(tensor: &Tensor, mask: &Tensor, dim: Option<usize>) -> Result<Tensor> {
    if mask.dtype() != DataType::Bool {
        return Err(MinitensorError::invalid_operation(
            "masked_log_softmax mask must have bool dtype",
        ));
    }

    if tensor.device() != mask.device() {
        return Err(MinitensorError::device_mismatch(
            format!("{:?}", tensor.device()),
            format!("{:?}", mask.device()),
        ));
    }

    let broadcast_shape = mask.shape().broadcast_with(tensor.shape())?;
    if &broadcast_shape != tensor.shape() {
        return Err(MinitensorError::shape_mismatch(
            mask.shape().dims().to_vec(),
            tensor.shape().dims().to_vec(),
        ));
    }

    if tensor.ndim() == 0 {
        let mut output_data =
            TensorData::uninitialized_on_device(tensor.numel(), tensor.dtype(), tensor.device());
        let mask_value = mask
            .data()
            .as_bool_slice()
            .ok_or_else(|| {
                MinitensorError::internal_error("Failed to get bool slice from mask tensor")
            })?
            .first()
            .copied()
            .unwrap_or(false);
        match tensor.dtype() {
            DataType::Float32 => {
                let output_slice = output_data.as_f32_slice_mut().ok_or_else(|| {
                    MinitensorError::internal_error(
                        "Failed to get mutable f32 slice from output data",
                    )
                })?;
                output_slice[0] = if mask_value { f32::NEG_INFINITY } else { 0.0 };
            }
            DataType::Float64 => {
                let output_slice = output_data.as_f64_slice_mut().ok_or_else(|| {
                    MinitensorError::internal_error(
                        "Failed to get mutable f64 slice from output data",
                    )
                })?;
                output_slice[0] = if mask_value { f64::NEG_INFINITY } else { 0.0 };
            }
            _ => {
                return Err(MinitensorError::invalid_operation(
                    "masked_log_softmax only supported for floating point tensors",
                ));
            }
        }

        let output = Tensor::new(
            Arc::new(output_data),
            tensor.shape().clone(),
            tensor.dtype(),
            tensor.device(),
            tensor.requires_grad(),
        );

        if output.requires_grad() {
            let grad_fn = Arc::new(MaskedLogSoftmaxBackward {
                input_id: tensor.id(),
                output: output.detach(),
                mask: mask.detach(),
                dim: 0,
            });

            return with_grad_fn(output, grad_fn);
        }

        return Ok(output);
    }

    let dim = dim.unwrap_or(tensor.ndim() - 1);

    check_dim(dim, tensor.ndim())?;

    let mut output_data =
        TensorData::uninitialized_on_device(tensor.numel(), tensor.dtype(), tensor.device());

    match tensor.dtype() {
        DataType::Float32 => masked_log_softmax_f32(tensor, mask, &mut output_data, dim)?,
        DataType::Float64 => masked_log_softmax_f64(tensor, mask, &mut output_data, dim)?,
        _ => {
            return Err(MinitensorError::invalid_operation(
                "masked_log_softmax only supported for floating point tensors",
            ));
        }
    }

    let output = Tensor::new(
        Arc::new(output_data),
        tensor.shape().clone(),
        tensor.dtype(),
        tensor.device(),
        tensor.requires_grad(),
    );

    if output.requires_grad() {
        let grad_fn = Arc::new(MaskedLogSoftmaxBackward {
            input_id: tensor.id(),
            output: output.detach(),
            mask: mask.detach(),
            dim,
        });

        with_grad_fn(output, grad_fn)
    } else {
        Ok(output)
    }
}

// Helper functions for type-specific operations

/// Generates a float unary elementwise kernel: fetch the input slice for the
/// dtype and map `$f` element-wise into a fresh, fully-initialized buffer (no
/// zeroing pass; see `ops::map`). Only the mapping closure differs per
/// op.
macro_rules! float_unary_kernel {
    ($name:ident, $accessor:ident, $ty:ty, $dtype:ident, $tyname:literal, $f:expr) => {
        pub(crate) fn $name(tensor: &Tensor) -> Result<TensorData> {
            let input_data = tensor.data().$accessor().ok_or_else(|| {
                MinitensorError::internal_error(concat!(
                    "Failed to get ",
                    $tyname,
                    " slice from input tensor"
                ))
            })?;
            Ok(TensorData::from_vec::<$ty>(
                unary_map_threshold(input_data, EXPENSIVE_PAR_THRESHOLD, $f),
                DataType::$dtype,
                tensor.device(),
            ))
        }
    };
}

/// Same as [`float_unary_kernel!`] but for activations that take extra runtime
/// parameters (e.g. `alpha`, `beta`). The generated function exposes the
/// parameters after `output_data`, and the mapping closure captures them from
/// scope. Only usable when a single closure covers the whole tensor — kernels
/// that select between closures once (like `gelu`'s `approximate` branch) must
/// stay hand-written so the branch is not pushed into the per-element loop.
macro_rules! float_unary_kernel_param {
    ($name:ident, $accessor:ident, $ty:ty, $dtype:ident, $tyname:literal,
     ($($pname:ident : $pty:ty),* $(,)?), $f:expr) => {
        pub(crate) fn $name(
            tensor: &Tensor,
            $($pname: $pty),*
        ) -> Result<TensorData> {
            let input_data = tensor.data().$accessor().ok_or_else(|| {
                MinitensorError::internal_error(concat!(
                    "Failed to get ",
                    $tyname,
                    " slice from input tensor"
                ))
            })?;
            Ok(TensorData::from_vec::<$ty>(
                unary_map_threshold(input_data, EXPENSIVE_PAR_THRESHOLD, $f),
                DataType::$dtype,
                tensor.device(),
            ))
        }
    };
}

// Which float32 routines stay on the scalar libm, and why.
//
// `tanh`, `sinh`, `cosh`, `expm1`, `log`, `log1p`, `erf`, `erfc` and both GELUs
// now run through `ops::simd::transcendental`, which computes each in float64
// and rounds once -- the accuracy the old promoted scalars bought, at several
// times the speed. Nothing is left promoted; promotion was a way of getting the
// float64 rounding cheaply, and the vectorized kernels get it for less.
//
// The rest stay scalar deliberately. `expf`, `sinf`, `cosf` and `cbrtf` are
// substantially faster than promoting -- `sinf` by 2.7x, `cbrtf` by 2.9x -- at
// equal accuracy, so promoting them would be a regression for nothing, and none
// has yet been worth a vectorized kernel of its own: `exp` already measures
// within 1.6x of a hand-vectorized baseline, against the 12x that `sinh` and
// `expm1` started from. `sin` and `cos` are the strongest remaining
// candidates at about 3.7x, and
// they need an argument reduction none of the existing kernels provide.

float_unary_kernel!(exp_f64, as_f64_slice, f64, Float64, "f64", f64::exp);

/// Vectorized. Bit-identical to `(x as f64).ln() as f32` on all 2^32 float32
/// inputs, where the `f32::ln` it replaces misrounds 416,909 of them.
pub(crate) fn exp_f32(tensor: &Tensor) -> Result<TensorData> {
    let input_data = tensor.data().as_f32_slice().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get f32 slice from input tensor")
    })?;
    let kernel = crate::ops::simd::F32Kernel::select();
    // SAFETY: `exp` writes every element of each block it is given.
    let out = unsafe {
        unary_map_blocks_threshold(input_data, VECTOR_F32_PAR_THRESHOLD, |src, dst| {
            kernel.exp(src, dst)
        })
    };
    Ok(TensorData::from_vec::<f32>(
        out,
        DataType::Float32,
        tensor.device(),
    ))
}

pub(crate) fn log_f32(tensor: &Tensor) -> Result<TensorData> {
    let input_data = tensor.data().as_f32_slice().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get f32 slice from input tensor")
    })?;
    let kernel = crate::ops::simd::F32Kernel::select();
    // SAFETY: `log` writes every element of each block it is given.
    let out = unsafe {
        unary_map_blocks_threshold(input_data, VECTOR_F32_PAR_THRESHOLD, |src, dst| {
            kernel.log(src, dst)
        })
    };
    Ok(TensorData::from_vec::<f32>(
        out,
        DataType::Float32,
        tensor.device(),
    ))
}

float_unary_kernel!(log_f64, as_f64_slice, f64, Float64, "f64", f64::ln);

/// Vectorized. The kernel handles `x <= -1` itself: `1 + x` is zero or
/// negative there and `log_core` maps those to -inf and NaN.
pub(crate) fn log1p_f32(tensor: &Tensor) -> Result<TensorData> {
    let input_data = tensor.data().as_f32_slice().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get f32 slice from input tensor")
    })?;
    let kernel = crate::ops::simd::F32Kernel::select();
    // SAFETY: `log1p` writes every element of each block it is given.
    let out = unsafe {
        unary_map_blocks_threshold(input_data, VECTOR_F32_PAR_THRESHOLD, |src, dst| {
            kernel.log1p(src, dst)
        })
    };
    Ok(TensorData::from_vec::<f32>(
        out,
        DataType::Float32,
        tensor.device(),
    ))
}

float_unary_kernel!(log1p_f64, as_f64_slice, f64, Float64, "f64", |val: f64| {
    if val == -1.0 {
        f64::NEG_INFINITY
    } else if val < -1.0 {
        f64::NAN
    } else {
        val.ln_1p()
    }
});

float_unary_kernel!(log2_f32, as_f32_slice, f32, Float32, "f32", f32::log2);

float_unary_kernel!(log2_f64, as_f64_slice, f64, Float64, "f64", f64::log2);

float_unary_kernel!(log10_f32, as_f32_slice, f32, Float32, "f32", f32::log10);

float_unary_kernel!(log10_f64, as_f64_slice, f64, Float64, "f64", f64::log10);

/// Vectorized -- see `ops::simd::transcendental`. Replaces `libm::erff`, which
/// it beats on both counts: 9.1x faster, and 68 of the 2^32 float32 inputs
/// misrounded by one ulp against `erff`'s 127.6 million.
pub(crate) fn erf_f32(tensor: &Tensor) -> Result<TensorData> {
    let input_data = tensor.data().as_f32_slice().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get f32 slice from input tensor")
    })?;
    let kernel = crate::ops::simd::F32Kernel::select();
    // SAFETY: `erf` writes every element of each block it is given.
    let out = unsafe {
        unary_map_blocks_threshold(input_data, VECTOR_F32_PAR_THRESHOLD, |src, dst| {
            kernel.erf(src, dst)
        })
    };
    Ok(TensorData::from_vec::<f32>(
        out,
        DataType::Float32,
        tensor.device(),
    ))
}

float_unary_kernel!(erf_f64, as_f64_slice, f64, Float64, "f64", erf);

// erfc is not `1 - erf(x)`: for large x that subtraction cancels away every
// significant digit. The vectorized kernel does not form that subtraction --
// above |x| = 2 it reads the erfc branch of `erf_parts` directly -- so it keeps
// them for the same reason libm's dedicated routine does.
pub(crate) fn erfc_f32(tensor: &Tensor) -> Result<TensorData> {
    let input_data = tensor.data().as_f32_slice().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get f32 slice from input tensor")
    })?;
    let kernel = crate::ops::simd::F32Kernel::select();
    // SAFETY: `erfc` writes every element of each block it is given.
    let out = unsafe {
        unary_map_blocks_threshold(input_data, VECTOR_F32_PAR_THRESHOLD, |src, dst| {
            kernel.erfc(src, dst)
        })
    };
    Ok(TensorData::from_vec::<f32>(
        out,
        DataType::Float32,
        tensor.device(),
    ))
}

float_unary_kernel!(erfc_f64, as_f64_slice, f64, Float64, "f64", erfc);

/// Vectorized; bit-identical to the `expm1_promoted_f32` it replaces.
pub(crate) fn expm1_f32(tensor: &Tensor) -> Result<TensorData> {
    let input_data = tensor.data().as_f32_slice().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get f32 slice from input tensor")
    })?;
    let kernel = crate::ops::simd::F32Kernel::select();
    // SAFETY: `expm1` writes every element of each block it is given.
    let out = unsafe {
        unary_map_blocks_threshold(input_data, VECTOR_F32_PAR_THRESHOLD, |src, dst| {
            kernel.expm1(src, dst)
        })
    };
    Ok(TensorData::from_vec::<f32>(
        out,
        DataType::Float32,
        tensor.device(),
    ))
}

float_unary_kernel!(expm1_f64, as_f64_slice, f64, Float64, "f64", f64::exp_m1);

/// Vectorized; bit-identical to `(x as f64).sin() as f32` on all 2^32 inputs.
pub(crate) fn sin_f32(tensor: &Tensor) -> Result<TensorData> {
    let input_data = tensor.data().as_f32_slice().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get f32 slice from input tensor")
    })?;
    let kernel = crate::ops::simd::F32Kernel::select();
    // SAFETY: `sin` writes every element of each block it is given.
    let out = unsafe {
        unary_map_blocks_threshold(input_data, VECTOR_F32_PAR_THRESHOLD, |src, dst| {
            kernel.sin(src, dst)
        })
    };
    Ok(TensorData::from_vec::<f32>(
        out,
        DataType::Float32,
        tensor.device(),
    ))
}

float_unary_kernel!(sin_f64, as_f64_slice, f64, Float64, "f64", f64::sin);

/// Vectorized; bit-identical to `(x as f64).cos() as f32` on all 2^32 inputs.
pub(crate) fn cos_f32(tensor: &Tensor) -> Result<TensorData> {
    let input_data = tensor.data().as_f32_slice().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get f32 slice from input tensor")
    })?;
    let kernel = crate::ops::simd::F32Kernel::select();
    // SAFETY: `cos` writes every element of each block it is given.
    let out = unsafe {
        unary_map_blocks_threshold(input_data, VECTOR_F32_PAR_THRESHOLD, |src, dst| {
            kernel.cos(src, dst)
        })
    };
    Ok(TensorData::from_vec::<f32>(
        out,
        DataType::Float32,
        tensor.device(),
    ))
}

float_unary_kernel!(cos_f64, as_f64_slice, f64, Float64, "f64", f64::cos);

/// Vectorized; bit-identical to `(x as f64).tan() as f32` on all 2^32 inputs.
/// Shares the reduction with `sin` and `cos`, so it costs a division more.
pub(crate) fn tan_f32(tensor: &Tensor) -> Result<TensorData> {
    let input_data = tensor.data().as_f32_slice().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get f32 slice from input tensor")
    })?;
    let kernel = crate::ops::simd::F32Kernel::select();
    // SAFETY: `tan` writes every element of each block it is given.
    let out = unsafe {
        unary_map_blocks_threshold(input_data, VECTOR_F32_PAR_THRESHOLD, |src, dst| {
            kernel.tan(src, dst)
        })
    };
    Ok(TensorData::from_vec::<f32>(
        out,
        DataType::Float32,
        tensor.device(),
    ))
}

float_unary_kernel!(tan_f64, as_f64_slice, f64, Float64, "f64", f64::tan);

/// Vectorized, sharing its reduction with `acos_f32`.
pub(crate) fn asin_f32(tensor: &Tensor) -> Result<TensorData> {
    let input_data = tensor.data().as_f32_slice().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get f32 slice from input tensor")
    })?;
    let kernel = crate::ops::simd::F32Kernel::select();
    // SAFETY: `asin` writes every element of each block it is given.
    let out = unsafe {
        unary_map_blocks_threshold(input_data, VECTOR_F32_PAR_THRESHOLD, |src, dst| {
            kernel.asin(src, dst)
        })
    };
    Ok(TensorData::from_vec::<f32>(
        out,
        DataType::Float32,
        tensor.device(),
    ))
}

float_unary_kernel!(asin_f64, as_f64_slice, f64, Float64, "f64", f64::asin);

/// Vectorized. Taken from `asin`'s reduction rather than as `pi/2 - asin(x)`,
/// which cancels near `x = 1`.
pub(crate) fn acos_f32(tensor: &Tensor) -> Result<TensorData> {
    let input_data = tensor.data().as_f32_slice().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get f32 slice from input tensor")
    })?;
    let kernel = crate::ops::simd::F32Kernel::select();
    // SAFETY: `acos` writes every element of each block it is given.
    let out = unsafe {
        unary_map_blocks_threshold(input_data, VECTOR_F32_PAR_THRESHOLD, |src, dst| {
            kernel.acos(src, dst)
        })
    };
    Ok(TensorData::from_vec::<f32>(
        out,
        DataType::Float32,
        tensor.device(),
    ))
}

float_unary_kernel!(acos_f64, as_f64_slice, f64, Float64, "f64", f64::acos);

/// Vectorized. `f32::atan` is a `libm` call, so the scalar loop it replaces
/// was the one arc function left running a lane at a time: `atan` measured
/// 2.75x NumPy's while `tan` -- the harder direction -- measured 0.85x.
pub(crate) fn atan_f32(tensor: &Tensor) -> Result<TensorData> {
    let input_data = tensor.data().as_f32_slice().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get f32 slice from input tensor")
    })?;
    let kernel = crate::ops::simd::F32Kernel::select();
    // SAFETY: `atan` writes every element of each block it is given.
    let out = unsafe {
        unary_map_blocks_threshold(input_data, VECTOR_F32_PAR_THRESHOLD, |src, dst| {
            kernel.atan(src, dst)
        })
    };
    Ok(TensorData::from_vec::<f32>(
        out,
        DataType::Float32,
        tensor.device(),
    ))
}

float_unary_kernel!(atan_f64, as_f64_slice, f64, Float64, "f64", f64::atan);

/// Vectorized; bit-identical to the `sinh_promoted_f32` it replaces.
pub(crate) fn sinh_f32(tensor: &Tensor) -> Result<TensorData> {
    let input_data = tensor.data().as_f32_slice().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get f32 slice from input tensor")
    })?;
    let kernel = crate::ops::simd::F32Kernel::select();
    // SAFETY: `sinh` writes every element of each block it is given.
    let out = unsafe {
        unary_map_blocks_threshold(input_data, VECTOR_F32_PAR_THRESHOLD, |src, dst| {
            kernel.sinh(src, dst)
        })
    };
    Ok(TensorData::from_vec::<f32>(
        out,
        DataType::Float32,
        tensor.device(),
    ))
}

float_unary_kernel!(sinh_f64, as_f64_slice, f64, Float64, "f64", f64::sinh);

/// Vectorized. Unlike its neighbours this replaces glibc's `coshf` rather
/// than a promoted scalar, so it is an accuracy gain as well: see the module
/// docs in `ops::simd::transcendental`.
pub(crate) fn cosh_f32(tensor: &Tensor) -> Result<TensorData> {
    let input_data = tensor.data().as_f32_slice().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get f32 slice from input tensor")
    })?;
    let kernel = crate::ops::simd::F32Kernel::select();
    // SAFETY: `cosh` writes every element of each block it is given.
    let out = unsafe {
        unary_map_blocks_threshold(input_data, VECTOR_F32_PAR_THRESHOLD, |src, dst| {
            kernel.cosh(src, dst)
        })
    };
    Ok(TensorData::from_vec::<f32>(
        out,
        DataType::Float32,
        tensor.device(),
    ))
}

float_unary_kernel!(cosh_f64, as_f64_slice, f64, Float64, "f64", f64::cosh);

float_unary_kernel!(asinh_f32, as_f32_slice, f32, Float32, "f32", f32::asinh);

float_unary_kernel!(asinh_f64, as_f64_slice, f64, Float64, "f64", f64::asinh);

float_unary_kernel!(acosh_f32, as_f32_slice, f32, Float32, "f32", f32::acosh);

float_unary_kernel!(acosh_f64, as_f64_slice, f64, Float64, "f64", f64::acosh);

float_unary_kernel!(atanh_f32, as_f32_slice, f32, Float32, "f32", f32::atanh);

float_unary_kernel!(atanh_f64, as_f64_slice, f64, Float64, "f64", f64::atanh);

/// Vectorized. `log1p(exp(beta*x))/beta`, with the linear tail above
/// `threshold` selected per block rather than per element.
pub(crate) fn softplus_f32(tensor: &Tensor, beta: f32, threshold: f32) -> Result<TensorData> {
    let input_data = tensor.data().as_f32_slice().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get f32 slice from input tensor")
    })?;
    let kernel = crate::ops::simd::F32Kernel::select();
    // SAFETY: `softplus` writes every element of each block it is given.
    let out = unsafe {
        unary_map_blocks_threshold(input_data, VECTOR_F32_PAR_THRESHOLD, |src, dst| {
            kernel.softplus(src, dst, beta as f64, threshold as f64)
        })
    };
    Ok(TensorData::from_vec::<f32>(
        out,
        DataType::Float32,
        tensor.device(),
    ))
}

float_unary_kernel_param!(
    softplus_f64,
    as_f64_slice,
    f64,
    Float64,
    "f64",
    (beta: f64, threshold: f64),
    |val: f64| {
        let scaled = beta * val;
        if scaled > threshold {
            val
        } else {
            scaled.exp().ln_1p() / beta
        }
    }
);

pub(crate) fn gelu_f32(tensor: &Tensor, approximate: bool) -> Result<TensorData> {
    let input_data = tensor.data().as_f32_slice().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get f32 slice from input tensor")
    })?;

    // Both variants are vectorized (`ops::simd::transcendental`); the
    // `approximate` branch is still selected outside the element loop. Each
    // now keeps the whole expression in float64 rather than rounding its
    // `erf`/`tanh` to float32 first, so both are more accurate than the scalar
    // `erff`/`tanhf` they replace as well as several times faster.
    let kernel = crate::ops::simd::F32Kernel::select();
    // SAFETY: both block kernels write every element of each block.
    let out = unsafe {
        unary_map_blocks_threshold(input_data, VECTOR_F32_PAR_THRESHOLD, |src, dst| {
            if approximate {
                kernel.gelu_tanh(src, dst)
            } else {
                kernel.gelu_erf(src, dst)
            }
        })
    };
    Ok(TensorData::from_vec(
        out,
        DataType::Float32,
        tensor.device(),
    ))
}

pub(crate) fn gelu_f64(tensor: &Tensor, approximate: bool) -> Result<TensorData> {
    let input_data = tensor.data().as_f64_slice().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get f64 slice from input tensor")
    })?;

    // The `approximate` branch is selected once, outside the element loop.
    //
    // Both are written cancellation-free, as the float32 kernels are: GELU is
    // `x` times a factor that decays to zero as `x -> -inf`, and forming that
    // factor as `1 + erf` or `1 + tanh` destroys it just where `x` is largest.
    // `1 + erf(v)` is `erfc(-v)`, and `0.5*(1 + tanh(v))` is the logistic
    // `1/(1 + exp(-2v))`; both are exact expressions, not approximations.
    let out = if approximate {
        let coeff = (2.0f64 / std::f64::consts::PI).sqrt();
        unary_map_threshold(input_data, EXPENSIVE_PAR_THRESHOLD, |x: f64| {
            let x3 = x * x * x;
            let inner = coeff * (x + 0.044715f64 * x3);
            x / (1.0f64 + (-2.0f64 * inner).exp())
        })
    } else {
        let inv_sqrt_2 = std::f64::consts::FRAC_1_SQRT_2;
        unary_map_threshold(input_data, EXPENSIVE_PAR_THRESHOLD, |x: f64| {
            0.5f64 * x * erfc(-x * inv_sqrt_2)
        })
    };
    Ok(TensorData::from_vec(
        out,
        DataType::Float64,
        tensor.device(),
    ))
}

float_unary_kernel_param!(
    elu_f32,
    as_f32_slice,
    f32,
    Float32,
    "f32",
    (alpha: f32),
    |x: f32| {
        if x > 0.0 { x } else { alpha * (x.exp() - 1.0) }
    }
);

float_unary_kernel_param!(
    elu_f64,
    as_f64_slice,
    f64,
    Float64,
    "f64",
    (alpha: f64),
    |x: f64| {
        if x > 0.0 { x } else { alpha * (x.exp() - 1.0) }
    }
);

float_unary_kernel!(selu_f32, as_f32_slice, f32, Float32, "f32", |x: f32| {
    const ALPHA: f32 = 1.6732632;
    const SCALE: f32 = 1.050701;
    if x > 0.0 {
        SCALE * x
    } else {
        SCALE * ALPHA * (x.exp() - 1.0)
    }
});

float_unary_kernel!(selu_f64, as_f64_slice, f64, Float64, "f64", |x: f64| {
    const ALPHA: f64 = 1.673_263_242_354_377_2;
    const SCALE: f64 = 1.050_700_987_355_480_5;
    if x > 0.0 {
        SCALE * x
    } else {
        SCALE * ALPHA * (x.exp() - 1.0)
    }
});

/// Vectorized. Also fixes the negative tail: the scalar form was
/// `x / (1 + exp(-x))`, and `exp(-x)` overflows float32 below about x = -89, so
/// `silu(-100)` returned -0 where -3.72e-42 is representable.
pub(crate) fn silu_f32(tensor: &Tensor) -> Result<TensorData> {
    let input_data = tensor.data().as_f32_slice().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get f32 slice from input tensor")
    })?;
    let kernel = crate::ops::simd::F32Kernel::select();
    // SAFETY: `silu` writes every element of each block it is given.
    let out = unsafe {
        unary_map_blocks_threshold(input_data, VECTOR_F32_PAR_THRESHOLD, |src, dst| {
            kernel.silu(src, dst)
        })
    };
    Ok(TensorData::from_vec::<f32>(
        out,
        DataType::Float32,
        tensor.device(),
    ))
}

float_unary_kernel!(silu_f64, as_f64_slice, f64, Float64, "f64", |x: f64| {
    // `1/(1 + exp(-x))` overflows for large negative x and loses the tail; the
    // stable form costs a branch and keeps it.
    let sigmoid = crate::ops::util::stable_sigmoid_f64(x);
    x * sigmoid
});

float_unary_kernel!(softsign_f32, as_f32_slice, f32, Float32, "f32", |x: f32| {
    let denom = 1.0 + x.abs();
    x / denom
});

float_unary_kernel!(softsign_f64, as_f64_slice, f64, Float64, "f64", |x: f64| {
    let denom = 1.0 + x.abs();
    x / denom
});
