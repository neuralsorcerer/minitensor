// Copyright (c) Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

use super::*;
use crate::autograd::EluBackward;
use crate::autograd::GeluBackward;
use crate::autograd::HardshrinkBackward;
use crate::autograd::LeakyReluBackward;
use crate::autograd::LogAddExpBackward;
use crate::autograd::LogSoftmaxBackward;
use crate::autograd::PowBackward;
use crate::autograd::PowBroadcast;
use crate::autograd::ReluBackward;
use crate::autograd::SeluBackward;
use crate::autograd::SiluBackward;
use crate::autograd::SoftmaxBackward;
use crate::autograd::SoftplusBackward;
use crate::autograd::SoftsignBackward;
use crate::{
    autograd::add_to_graph,
    error::{MinitensorError, Result},
    tensor::{DataType, Tensor, TensorData},
};
use std::sync::Arc;

/// Element-wise power with tensor exponent and gradient support
pub fn pow(base: &Tensor, exponent: &Tensor) -> Result<Tensor> {
    // Check device and dtype compatibility
    if base.device() != exponent.device() {
        return Err(MinitensorError::device_mismatch(
            format!("{:?}", base.device()),
            format!("{:?}", exponent.device()),
        ));
    }

    if base.dtype() != exponent.dtype() {
        return Err(MinitensorError::type_mismatch(
            format!("{:?}", base.dtype()),
            format!("{:?}", exponent.dtype()),
        ));
    }

    let base_shape = base.shape().clone();
    let exponent_shape = exponent.shape().clone();
    let base_numel = base_shape.numel();
    let exp_numel = exponent_shape.numel();

    let broadcast = if base_shape == exponent_shape {
        PowBroadcast::None
    } else if base_numel == 1 {
        PowBroadcast::BaseScalar
    } else if exp_numel == 1 {
        PowBroadcast::ExponentScalar
    } else {
        // General broadcasting: materialize both operands at the broadcast
        // shape (expand and contiguous are grad-aware) and recurse into the
        // same-shape fast path.
        let out_shape = base_shape.broadcast_with(&exponent_shape)?;
        let dims: Vec<isize> = out_shape.dims().iter().map(|&d| d as isize).collect();
        let base_b = base.expand(dims.clone())?.contiguous()?;
        let exp_b = exponent.expand(dims)?.contiguous()?;
        return pow(&base_b, &exp_b);
    };

    let output_shape = match broadcast {
        PowBroadcast::None | PowBroadcast::ExponentScalar => base_shape.clone(),
        PowBroadcast::BaseScalar => exponent_shape.clone(),
    };

    /// One dtype arm: map `powf` over the operands for the detected
    /// broadcast form. Uses the shared map primitives, so `pow` now
    /// parallelizes above the same thresholds as the other element-wise ops
    /// (the previous hand-written loops were always sequential).
    macro_rules! pow_arm {
        ($accessor:ident, $ty:ty, $dtype:ident, $tyname:literal) => {{
            let b = base.data().$accessor().ok_or_else(|| {
                MinitensorError::internal_error(concat!(
                    "Failed to get ",
                    $tyname,
                    " slice from base tensor"
                ))
            })?;
            let e = exponent.data().$accessor().ok_or_else(|| {
                MinitensorError::internal_error(concat!(
                    "Failed to get ",
                    $tyname,
                    " slice from exponent tensor"
                ))
            })?;
            let out = match broadcast {
                PowBroadcast::None => crate::ops::map::binary_map(b, e, |x: $ty, y: $ty| x.powf(y)),
                PowBroadcast::BaseScalar => {
                    let base_val = b[0];
                    unary_map(e, move |y: $ty| base_val.powf(y))
                }
                PowBroadcast::ExponentScalar => {
                    let exp_val = e[0];
                    unary_map(b, move |x: $ty| x.powf(exp_val))
                }
            };
            TensorData::from_vec::<$ty>(out, DataType::$dtype, base.device())
        }};
    }

    let output_data = match base.dtype() {
        DataType::Float32 => pow_arm!(as_f32_slice, f32, Float32, "f32"),
        DataType::Float64 => pow_arm!(as_f64_slice, f64, Float64, "f64"),
        _ => {
            return Err(MinitensorError::invalid_operation(
                "Power operation only supported for floating point tensors",
            ));
        }
    };

    let output = Tensor::new(
        Arc::new(output_data),
        output_shape,
        base.dtype(),
        base.device(),
        base.requires_grad() || exponent.requires_grad(),
    );

    if output.requires_grad() {
        let grad_fn = Arc::new(PowBackward {
            base: base.detach(),
            exponent: exponent.detach(),
            output: output.clone().detach(),
            input_ids: [base.id(), exponent.id()],
            base_requires_grad: base.requires_grad(),
            exp_requires_grad: exponent.requires_grad(),
            broadcast,
        });

        let mut output_with_grad = output;
        output_with_grad.set_grad_fn(Some(grad_fn.clone()));

        add_to_graph(&output_with_grad, Some(grad_fn))?;
        Ok(output_with_grad)
    } else {
        Ok(output)
    }
}

/// Element-wise power with scalar exponent and gradient support
pub fn powf(tensor: &Tensor, exponent: f64) -> Result<Tensor> {
    // A single-element exponent is enough: `pow` detects the scalar operand
    // and takes its ExponentScalar broadcast path, so there is no reason to
    // materialize (and later save for backward) a full-size constant tensor.
    // Tensors with zero or one element keep an exponent of their own shape so
    // the same-shape fast path (and the output shape) are unchanged for them.
    let exp_shape = if tensor.numel() <= 1 {
        tensor.shape().clone()
    } else {
        crate::tensor::Shape::new(vec![1])
    };
    let exp_numel = exp_shape.numel();
    let exp_data = match tensor.dtype() {
        DataType::Float32 => TensorData::from_vec(
            vec![exponent as f32; exp_numel],
            DataType::Float32,
            tensor.device(),
        ),
        DataType::Float64 => TensorData::from_vec(
            vec![exponent; exp_numel],
            DataType::Float64,
            tensor.device(),
        ),
        _ => {
            return Err(MinitensorError::invalid_operation(
                "Power operation only supported for floating point tensors",
            ));
        }
    };
    let exp_tensor = Tensor::new(
        Arc::new(exp_data),
        exp_shape,
        tensor.dtype(),
        tensor.device(),
        false,
    );
    pow(tensor, &exp_tensor)
}

/// Numerically stable logaddexp with gradient support
pub fn logaddexp(lhs: &Tensor, rhs: &Tensor) -> Result<Tensor> {
    if lhs.device() != rhs.device() {
        return Err(MinitensorError::device_mismatch(
            format!("{:?}", lhs.device()),
            format!("{:?}", rhs.device()),
        ));
    }

    let requires_grad = lhs.requires_grad() || rhs.requires_grad();
    use crate::ops::binary::{BinaryOpKind, coerce_binary_operands};
    let (lhs_cast, rhs_cast, result_dtype) = coerce_binary_operands(lhs, rhs, BinaryOpKind::Add)?;

    let lhs_tensor = match lhs_cast {
        std::borrow::Cow::Borrowed(t) => t.clone(),
        std::borrow::Cow::Owned(t) => t,
    };
    let rhs_tensor = match rhs_cast {
        std::borrow::Cow::Borrowed(t) => t.clone(),
        std::borrow::Cow::Owned(t) => t,
    };

    if result_dtype != DataType::Float32 && result_dtype != DataType::Float64 {
        return Err(MinitensorError::invalid_operation(
            "logaddexp is only supported for floating point tensors",
        ));
    }

    let output_shape = lhs_tensor.shape().broadcast_with(rhs_tensor.shape())?;
    let output_data = match result_dtype {
        DataType::Float32 => logaddexp_f32(&lhs_tensor, &rhs_tensor, &output_shape)?,
        DataType::Float64 => logaddexp_f64(&lhs_tensor, &rhs_tensor, &output_shape)?,
        _ => unreachable!(),
    };

    let output = Tensor::new(
        Arc::new(output_data),
        output_shape.clone(),
        result_dtype,
        lhs.device(),
        requires_grad,
    );

    if requires_grad {
        let grad_fn = Arc::new(LogAddExpBackward {
            lhs: lhs_tensor.detach(),
            rhs: rhs_tensor.detach(),
            output: output.clone().detach(),
            input_ids: [lhs.id(), rhs.id()],
            input_shapes: [lhs.shape().dims().to_vec(), rhs.shape().dims().to_vec()],
            input_requires_grad: [lhs.requires_grad(), rhs.requires_grad()],
        });

        let mut output_with_grad = output;
        output_with_grad.set_grad_fn(Some(grad_fn.clone()));
        add_to_graph(&output_with_grad, Some(grad_fn))?;
        Ok(output_with_grad)
    } else {
        Ok(output)
    }
}

/// Softplus activation function with gradient support
pub fn softplus(tensor: &Tensor, beta: f64, threshold: f64) -> Result<Tensor> {
    if beta <= 0.0 {
        return Err(MinitensorError::invalid_argument(
            "softplus beta must be positive",
        ));
    }

    let output_data = match tensor.dtype() {
        DataType::Float32 => softplus_f32(tensor, beta as f32, threshold as f32)?,
        DataType::Float64 => softplus_f64(tensor, beta, threshold)?,
        _ => {
            return Err(MinitensorError::invalid_operation(
                "Softplus is only supported for floating point tensors",
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
        let grad_fn = Arc::new(SoftplusBackward {
            input_id: tensor.id(),
            input: tensor.clone().detach(),
            beta,
            threshold,
        });

        let mut output_with_grad = output;
        output_with_grad.set_grad_fn(Some(grad_fn.clone()));
        add_to_graph(&output_with_grad, Some(grad_fn))?;
        Ok(output_with_grad)
    } else {
        Ok(output)
    }
}

/// GELU activation function with optional tanh approximation
pub fn gelu(tensor: &Tensor, approximate: bool) -> Result<Tensor> {
    let output_data = match tensor.dtype() {
        DataType::Float32 => gelu_f32(tensor, approximate)?,
        DataType::Float64 => gelu_f64(tensor, approximate)?,
        _ => {
            return Err(MinitensorError::invalid_operation(
                "GELU is only supported for floating point tensors",
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
        let grad_fn = Arc::new(GeluBackward {
            input_id: tensor.id(),
            input: tensor.clone().detach(),
            approximate,
        });

        let mut output_with_grad = output;
        output_with_grad.set_grad_fn(Some(grad_fn.clone()));
        add_to_graph(&output_with_grad, Some(grad_fn))?;
        Ok(output_with_grad)
    } else {
        Ok(output)
    }
}

/// ELU activation function with configurable alpha
pub fn elu(tensor: &Tensor, alpha: f64) -> Result<Tensor> {
    let output_data = match tensor.dtype() {
        DataType::Float32 => elu_f32(tensor, alpha as f32)?,
        DataType::Float64 => elu_f64(tensor, alpha)?,
        _ => {
            return Err(MinitensorError::invalid_operation(
                "ELU is only supported for floating point tensors",
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
        let grad_fn = Arc::new(EluBackward {
            input_id: tensor.id(),
            output: output.clone().detach(),
            alpha,
        });

        let mut output_with_grad = output;
        output_with_grad.set_grad_fn(Some(grad_fn.clone()));
        add_to_graph(&output_with_grad, Some(grad_fn))?;
        Ok(output_with_grad)
    } else {
        Ok(output)
    }
}

/// SELU activation function.
pub fn selu(tensor: &Tensor) -> Result<Tensor> {
    let output_data = match tensor.dtype() {
        DataType::Float32 => selu_f32(tensor)?,
        DataType::Float64 => selu_f64(tensor)?,
        _ => {
            return Err(MinitensorError::invalid_operation(
                "SELU is only supported for floating point tensors",
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
        let grad_fn = Arc::new(SeluBackward {
            input_id: tensor.id(),
            output: output.clone().detach(),
        });

        let mut output_with_grad = output;
        output_with_grad.set_grad_fn(Some(grad_fn.clone()));
        add_to_graph(&output_with_grad, Some(grad_fn))?;
        Ok(output_with_grad)
    } else {
        Ok(output)
    }
}

/// SiLU (Swish) activation function with gradient support
pub fn silu(tensor: &Tensor) -> Result<Tensor> {
    let output_data = match tensor.dtype() {
        DataType::Float32 => silu_f32(tensor)?,
        DataType::Float64 => silu_f64(tensor)?,
        _ => {
            return Err(MinitensorError::invalid_operation(
                "SiLU is only supported for floating point tensors",
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
        let grad_fn = Arc::new(SiluBackward {
            input_id: tensor.id(),
            input: tensor.clone().detach(),
        });

        let mut output_with_grad = output;
        output_with_grad.set_grad_fn(Some(grad_fn.clone()));
        add_to_graph(&output_with_grad, Some(grad_fn))?;
        Ok(output_with_grad)
    } else {
        Ok(output)
    }
}

/// Softsign activation function with gradient support
pub fn softsign(tensor: &Tensor) -> Result<Tensor> {
    let output_data = match tensor.dtype() {
        DataType::Float32 => softsign_f32(tensor)?,
        DataType::Float64 => softsign_f64(tensor)?,
        _ => {
            return Err(MinitensorError::invalid_operation(
                "Softsign is only supported for floating point tensors",
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
        let grad_fn = Arc::new(SoftsignBackward {
            input_id: tensor.id(),
            input: tensor.clone().detach(),
        });

        let mut output_with_grad = output;
        output_with_grad.set_grad_fn(Some(grad_fn.clone()));
        add_to_graph(&output_with_grad, Some(grad_fn))?;
        Ok(output_with_grad)
    } else {
        Ok(output)
    }
}

/// ReLU activation function with gradient support
pub fn relu(tensor: &Tensor) -> Result<Tensor> {
    // The backward mask is only materialized when a gradient function will
    // actually be attached (mirrors `Tensor::new`'s grad gating), so pure
    // inference skips both the mask allocation and its fill pass.
    let store_mask = tensor.requires_grad() && crate::autograd::is_grad_enabled();

    // Perform ReLU based on data type while capturing mask of positive inputs
    let (output_data, mask) = match tensor.dtype() {
        DataType::Float32 => relu_f32(tensor, store_mask)?,
        DataType::Float64 => relu_f64(tensor, store_mask)?,
        DataType::Int32 => relu_i32(tensor, store_mask)?,
        DataType::Int64 => relu_i64(tensor, store_mask)?,
        DataType::Bool => {
            return Err(MinitensorError::invalid_operation(
                "ReLU function not supported for boolean tensors",
            ));
        }
    };

    // Create output tensor
    let output = Tensor::new(
        Arc::new(output_data),
        tensor.shape().clone(),
        tensor.dtype(),
        tensor.device(),
        tensor.requires_grad(),
    );

    // Set up gradient function if needed
    if output.requires_grad() {
        let grad_fn = Arc::new(ReluBackward {
            input_id: tensor.id(),
            mask: mask.ok_or_else(|| {
                MinitensorError::internal_error(
                    "relu mask missing despite gradients being required",
                )
            })?,
        });

        let mut output_with_grad = output;
        output_with_grad.set_grad_fn(Some(grad_fn.clone()));

        // Add to computation graph
        add_to_graph(&output_with_grad, Some(grad_fn))?;

        Ok(output_with_grad)
    } else {
        Ok(output)
    }
}

/// Hardshrink activation that thresholds values to zero within ``[-lambd, lambd]``
pub fn hardshrink(tensor: &Tensor, lambd: f64) -> Result<Tensor> {
    if lambd < 0.0 {
        return Err(MinitensorError::invalid_operation(
            "hardshrink requires lambd to be non-negative",
        ));
    }

    // Gate the mask on the same condition `Tensor::new` uses for
    // `requires_grad`, so it is neither computed for inference nor missing
    // when the gradient function needs it (the bare `requires_grad()` check
    // used previously produced a stray mask under `no_grad`).
    let store_mask = tensor.requires_grad() && crate::autograd::is_grad_enabled();
    let (output_data, mask) = match tensor.dtype() {
        DataType::Float32 => hardshrink_f32(tensor, lambd as f32, store_mask)?,
        DataType::Float64 => hardshrink_f64(tensor, lambd, store_mask)?,
        _ => {
            return Err(MinitensorError::invalid_operation(
                "hardshrink is only supported for floating point tensors",
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
        let grad_fn = Arc::new(HardshrinkBackward {
            input_id: tensor.id(),
            mask: mask.ok_or_else(|| {
                MinitensorError::internal_error(
                    "hardshrink mask missing despite gradients being required",
                )
            })?,
        });

        let mut output_with_grad = output;
        output_with_grad.set_grad_fn(Some(grad_fn.clone()));
        add_to_graph(&output_with_grad, Some(grad_fn))?;
        Ok(output_with_grad)
    } else {
        debug_assert!(mask.is_none());
        Ok(output)
    }
}

/// LeakyReLU activation function with gradient support
pub fn leaky_relu(tensor: &Tensor, negative_slope: f64) -> Result<Tensor> {
    // As with `relu`, only materialize the backward mask when a gradient
    // function will consume it.
    let store_mask = tensor.requires_grad() && crate::autograd::is_grad_enabled();

    // Perform LeakyReLU based on data type and capture mask of positive inputs
    let (output_data, mask) = match tensor.dtype() {
        DataType::Float32 => leaky_relu_f32(tensor, negative_slope as f32, store_mask)?,
        DataType::Float64 => leaky_relu_f64(tensor, negative_slope, store_mask)?,
        _ => {
            return Err(MinitensorError::invalid_operation(
                "LeakyReLU function only supported for floating point tensors",
            ));
        }
    };

    // Create output tensor
    let output = Tensor::new(
        Arc::new(output_data),
        tensor.shape().clone(),
        tensor.dtype(),
        tensor.device(),
        tensor.requires_grad(),
    );

    // Set up gradient function if needed
    if output.requires_grad() {
        let grad_fn = Arc::new(LeakyReluBackward {
            input_id: tensor.id(),
            negative_slope,
            mask: mask.ok_or_else(|| {
                MinitensorError::internal_error(
                    "leaky_relu mask missing despite gradients being required",
                )
            })?,
        });

        let mut output_with_grad = output;
        output_with_grad.set_grad_fn(Some(grad_fn.clone()));

        // Add to computation graph
        add_to_graph(&output_with_grad, Some(grad_fn))?;

        Ok(output_with_grad)
    } else {
        Ok(output)
    }
}

/// Softmax activation function with gradient support
pub fn softmax(tensor: &Tensor, dim: Option<usize>) -> Result<Tensor> {
    if tensor.ndim() == 0 {
        let mut output_data =
            TensorData::uninitialized_on_device(tensor.numel(), tensor.dtype(), tensor.device());
        match tensor.dtype() {
            DataType::Float32 => {
                let output_slice = output_data.as_f32_slice_mut().ok_or_else(|| {
                    MinitensorError::internal_error(
                        "Failed to get mutable f32 slice from output data",
                    )
                })?;
                output_slice[0] = 1.0;
            }
            DataType::Float64 => {
                let output_slice = output_data.as_f64_slice_mut().ok_or_else(|| {
                    MinitensorError::internal_error(
                        "Failed to get mutable f64 slice from output data",
                    )
                })?;
                output_slice[0] = 1.0;
            }
            _ => {
                return Err(MinitensorError::invalid_operation(
                    "Softmax function only supported for floating point tensors",
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

    // Create output tensor data
    let mut output_data =
        TensorData::uninitialized_on_device(tensor.numel(), tensor.dtype(), tensor.device());

    // Perform softmax based on data type
    match tensor.dtype() {
        DataType::Float32 => softmax_f32(tensor, &mut output_data, dim)?,
        DataType::Float64 => softmax_f64(tensor, &mut output_data, dim)?,
        _ => {
            return Err(MinitensorError::invalid_operation(
                "Softmax function only supported for floating point tensors",
            ));
        }
    }

    // Create output tensor
    let output = Tensor::new(
        Arc::new(output_data),
        tensor.shape().clone(),
        tensor.dtype(),
        tensor.device(),
        tensor.requires_grad(),
    );

    // Set up gradient function if needed
    if output.requires_grad() {
        let grad_fn = Arc::new(SoftmaxBackward {
            input_id: tensor.id(),
            output: output.detach(),
            dim,
        });

        let mut output_with_grad = output;
        output_with_grad.set_grad_fn(Some(grad_fn.clone()));

        // Add to computation graph
        add_to_graph(&output_with_grad, Some(grad_fn))?;

        Ok(output_with_grad)
    } else {
        Ok(output)
    }
}

/// Log-Softmax activation function with gradient support
pub fn log_softmax(tensor: &Tensor, dim: Option<usize>) -> Result<Tensor> {
    if tensor.ndim() == 0 {
        let mut output_data =
            TensorData::uninitialized_on_device(tensor.numel(), tensor.dtype(), tensor.device());
        match tensor.dtype() {
            DataType::Float32 => {
                let output_slice = output_data.as_f32_slice_mut().ok_or_else(|| {
                    MinitensorError::internal_error(
                        "Failed to get mutable f32 slice from output data",
                    )
                })?;
                output_slice[0] = 0.0;
            }
            DataType::Float64 => {
                let output_slice = output_data.as_f64_slice_mut().ok_or_else(|| {
                    MinitensorError::internal_error(
                        "Failed to get mutable f64 slice from output data",
                    )
                })?;
                output_slice[0] = 0.0;
            }
            _ => {
                return Err(MinitensorError::invalid_operation(
                    "LogSoftmax function only supported for floating point tensors",
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
            let grad_fn = Arc::new(LogSoftmaxBackward {
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
        DataType::Float32 => log_softmax_f32(tensor, &mut output_data, dim)?,
        DataType::Float64 => log_softmax_f64(tensor, &mut output_data, dim)?,
        _ => {
            return Err(MinitensorError::invalid_operation(
                "LogSoftmax function only supported for floating point tensors",
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
        let grad_fn = Arc::new(LogSoftmaxBackward {
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
