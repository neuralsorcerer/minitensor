// Copyright (c) Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

use super::*;
use crate::autograd::MaskedLogSoftmaxBackward;
use crate::autograd::SoftmaxBackward;
use crate::{
    autograd::add_to_graph,
    error::{MinitensorError, Result},
    tensor::{DataType, Tensor, TensorData},
};
use libm::{erf, erff};
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

            let mut output_with_grad = output;
            output_with_grad.set_grad_fn(Some(grad_fn.clone()));
            add_to_graph(&output_with_grad, Some(grad_fn))?;
            return Ok(output_with_grad);
        }

        return Ok(output);
    }

    let dim = dim.unwrap_or(tensor.ndim() - 1);

    if dim >= tensor.ndim() {
        return Err(MinitensorError::index_error(dim as isize, 0, tensor.ndim()));
    }

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

        let mut output_with_grad = output;
        output_with_grad.set_grad_fn(Some(grad_fn.clone()));
        add_to_graph(&output_with_grad, Some(grad_fn))?;

        Ok(output_with_grad)
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

            let mut output_with_grad = output;
            output_with_grad.set_grad_fn(Some(grad_fn.clone()));
            add_to_graph(&output_with_grad, Some(grad_fn))?;
            return Ok(output_with_grad);
        }

        return Ok(output);
    }

    let dim = dim.unwrap_or(tensor.ndim() - 1);

    if dim >= tensor.ndim() {
        return Err(MinitensorError::index_error(dim as isize, 0, tensor.ndim()));
    }

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

        let mut output_with_grad = output;
        output_with_grad.set_grad_fn(Some(grad_fn.clone()));
        add_to_graph(&output_with_grad, Some(grad_fn))?;

        Ok(output_with_grad)
    } else {
        Ok(output)
    }
}

// Helper functions for type-specific operations

/// Generates a float unary elementwise kernel: fetch the input and output
/// slices for the dtype and apply `$f` element-wise via `unary_apply`. Body is
/// identical to the hand-written wrappers it replaces; only the mapping
/// closure differs per op.
macro_rules! float_unary_kernel {
    ($name:ident, $accessor:ident, $accessor_mut:ident, $tyname:literal, $f:expr) => {
        pub(crate) fn $name(tensor: &Tensor, output_data: &mut TensorData) -> Result<()> {
            let input_data = tensor.data().$accessor().ok_or_else(|| {
                MinitensorError::internal_error(concat!(
                    "Failed to get ",
                    $tyname,
                    " slice from input tensor"
                ))
            })?;
            let output_slice = output_data.$accessor_mut().ok_or_else(|| {
                MinitensorError::internal_error(concat!(
                    "Failed to get mutable ",
                    $tyname,
                    " slice from output data"
                ))
            })?;
            unary_apply(input_data, output_slice, $f);
            Ok(())
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
    ($name:ident, $accessor:ident, $accessor_mut:ident, $tyname:literal,
     ($($pname:ident : $pty:ty),* $(,)?), $f:expr) => {
        pub(crate) fn $name(
            tensor: &Tensor,
            output_data: &mut TensorData,
            $($pname: $pty),*
        ) -> Result<()> {
            let input_data = tensor.data().$accessor().ok_or_else(|| {
                MinitensorError::internal_error(concat!(
                    "Failed to get ",
                    $tyname,
                    " slice from input tensor"
                ))
            })?;
            let output_slice = output_data.$accessor_mut().ok_or_else(|| {
                MinitensorError::internal_error(concat!(
                    "Failed to get mutable ",
                    $tyname,
                    " slice from output data"
                ))
            })?;
            unary_apply(input_data, output_slice, $f);
            Ok(())
        }
    };
}

float_unary_kernel!(exp_f32, as_f32_slice, as_f32_slice_mut, "f32", f32::exp);

float_unary_kernel!(exp_f64, as_f64_slice, as_f64_slice_mut, "f64", f64::exp);

float_unary_kernel!(log_f32, as_f32_slice, as_f32_slice_mut, "f32", f32::ln);

float_unary_kernel!(log_f64, as_f64_slice, as_f64_slice_mut, "f64", f64::ln);

float_unary_kernel!(
    log1p_f32,
    as_f32_slice,
    as_f32_slice_mut,
    "f32",
    |val: f32| {
        if val == -1.0 {
            f32::NEG_INFINITY
        } else if val < -1.0 {
            f32::NAN
        } else {
            val.ln_1p()
        }
    }
);

float_unary_kernel!(
    log1p_f64,
    as_f64_slice,
    as_f64_slice_mut,
    "f64",
    |val: f64| {
        if val == -1.0 {
            f64::NEG_INFINITY
        } else if val < -1.0 {
            f64::NAN
        } else {
            val.ln_1p()
        }
    }
);

float_unary_kernel!(
    expm1_f32,
    as_f32_slice,
    as_f32_slice_mut,
    "f32",
    f32::exp_m1
);

float_unary_kernel!(
    expm1_f64,
    as_f64_slice,
    as_f64_slice_mut,
    "f64",
    f64::exp_m1
);

float_unary_kernel!(sin_f32, as_f32_slice, as_f32_slice_mut, "f32", f32::sin);

float_unary_kernel!(sin_f64, as_f64_slice, as_f64_slice_mut, "f64", f64::sin);

float_unary_kernel!(cos_f32, as_f32_slice, as_f32_slice_mut, "f32", f32::cos);

float_unary_kernel!(cos_f64, as_f64_slice, as_f64_slice_mut, "f64", f64::cos);

float_unary_kernel!(tan_f32, as_f32_slice, as_f32_slice_mut, "f32", f32::tan);

float_unary_kernel!(tan_f64, as_f64_slice, as_f64_slice_mut, "f64", f64::tan);

float_unary_kernel!(asin_f32, as_f32_slice, as_f32_slice_mut, "f32", f32::asin);

float_unary_kernel!(asin_f64, as_f64_slice, as_f64_slice_mut, "f64", f64::asin);

float_unary_kernel!(acos_f32, as_f32_slice, as_f32_slice_mut, "f32", f32::acos);

float_unary_kernel!(acos_f64, as_f64_slice, as_f64_slice_mut, "f64", f64::acos);

float_unary_kernel!(atan_f32, as_f32_slice, as_f32_slice_mut, "f32", f32::atan);

float_unary_kernel!(atan_f64, as_f64_slice, as_f64_slice_mut, "f64", f64::atan);

float_unary_kernel!(sinh_f32, as_f32_slice, as_f32_slice_mut, "f32", f32::sinh);

float_unary_kernel!(sinh_f64, as_f64_slice, as_f64_slice_mut, "f64", f64::sinh);

float_unary_kernel!(cosh_f32, as_f32_slice, as_f32_slice_mut, "f32", f32::cosh);

float_unary_kernel!(cosh_f64, as_f64_slice, as_f64_slice_mut, "f64", f64::cosh);

float_unary_kernel!(asinh_f32, as_f32_slice, as_f32_slice_mut, "f32", f32::asinh);

float_unary_kernel!(asinh_f64, as_f64_slice, as_f64_slice_mut, "f64", f64::asinh);

float_unary_kernel!(acosh_f32, as_f32_slice, as_f32_slice_mut, "f32", f32::acosh);

float_unary_kernel!(acosh_f64, as_f64_slice, as_f64_slice_mut, "f64", f64::acosh);

float_unary_kernel!(atanh_f32, as_f32_slice, as_f32_slice_mut, "f32", f32::atanh);

float_unary_kernel!(atanh_f64, as_f64_slice, as_f64_slice_mut, "f64", f64::atanh);

float_unary_kernel_param!(
    softplus_f32,
    as_f32_slice,
    as_f32_slice_mut,
    "f32",
    (beta: f32, threshold: f32),
    |val: f32| {
        let scaled = beta * val;
        if scaled > threshold {
            val
        } else {
            scaled.exp().ln_1p() / beta
        }
    }
);

float_unary_kernel_param!(
    softplus_f64,
    as_f64_slice,
    as_f64_slice_mut,
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

pub(crate) fn gelu_f32(
    tensor: &Tensor,
    output_data: &mut TensorData,
    approximate: bool,
) -> Result<()> {
    let input_data = tensor.data().as_f32_slice().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get f32 slice from input tensor")
    })?;

    let output_slice = output_data.as_f32_slice_mut().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get mutable f32 slice from output data")
    })?;

    if approximate {
        let coeff = (2.0f32 / std::f32::consts::PI).sqrt();
        unary_apply(input_data, output_slice, |x: f32| {
            let x3 = x * x * x;
            let inner = coeff * (x + 0.044715f32 * x3);
            0.5f32 * x * (1.0f32 + inner.tanh())
        });
    } else {
        let inv_sqrt_2 = std::f32::consts::FRAC_1_SQRT_2;
        unary_apply(input_data, output_slice, |x: f32| {
            0.5f32 * x * (1.0f32 + erff(x * inv_sqrt_2))
        });
    }
    Ok(())
}

pub(crate) fn gelu_f64(
    tensor: &Tensor,
    output_data: &mut TensorData,
    approximate: bool,
) -> Result<()> {
    let input_data = tensor.data().as_f64_slice().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get f64 slice from input tensor")
    })?;

    let output_slice = output_data.as_f64_slice_mut().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get mutable f64 slice from output data")
    })?;

    if approximate {
        let coeff = (2.0f64 / std::f64::consts::PI).sqrt();
        unary_apply(input_data, output_slice, |x: f64| {
            let x3 = x * x * x;
            let inner = coeff * (x + 0.044715f64 * x3);
            0.5f64 * x * (1.0f64 + inner.tanh())
        });
    } else {
        let inv_sqrt_2 = std::f64::consts::FRAC_1_SQRT_2;
        unary_apply(input_data, output_slice, |x: f64| {
            0.5f64 * x * (1.0f64 + erf(x * inv_sqrt_2))
        });
    }
    Ok(())
}

float_unary_kernel_param!(
    elu_f32,
    as_f32_slice,
    as_f32_slice_mut,
    "f32",
    (alpha: f32),
    |x: f32| {
        if x > 0.0 { x } else { alpha * (x.exp() - 1.0) }
    }
);

float_unary_kernel_param!(
    elu_f64,
    as_f64_slice,
    as_f64_slice_mut,
    "f64",
    (alpha: f64),
    |x: f64| {
        if x > 0.0 { x } else { alpha * (x.exp() - 1.0) }
    }
);

float_unary_kernel!(selu_f32, as_f32_slice, as_f32_slice_mut, "f32", |x: f32| {
    const ALPHA: f32 = 1.6732632;
    const SCALE: f32 = 1.050701;
    if x > 0.0 {
        SCALE * x
    } else {
        SCALE * ALPHA * (x.exp() - 1.0)
    }
});

float_unary_kernel!(selu_f64, as_f64_slice, as_f64_slice_mut, "f64", |x: f64| {
    const ALPHA: f64 = 1.6732632423543772848170429916717;
    const SCALE: f64 = 1.0507009873554804934193349852946;
    if x > 0.0 {
        SCALE * x
    } else {
        SCALE * ALPHA * (x.exp() - 1.0)
    }
});

float_unary_kernel!(silu_f32, as_f32_slice, as_f32_slice_mut, "f32", |x: f32| {
    let sigmoid = 1.0 / (1.0 + (-x).exp());
    x * sigmoid
});

float_unary_kernel!(silu_f64, as_f64_slice, as_f64_slice_mut, "f64", |x: f64| {
    let sigmoid = 1.0 / (1.0 + (-x).exp());
    x * sigmoid
});

float_unary_kernel!(
    softsign_f32,
    as_f32_slice,
    as_f32_slice_mut,
    "f32",
    |x: f32| {
        let denom = 1.0 + x.abs();
        x / denom
    }
);

float_unary_kernel!(
    softsign_f64,
    as_f64_slice,
    as_f64_slice_mut,
    "f64",
    |x: f64| {
        let denom = 1.0 + x.abs();
        x / denom
    }
);
