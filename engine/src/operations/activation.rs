// Copyright (c) 2025 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

use crate::{
    autograd::{
        CosBackward, EluBackward, ExpBackward, Expm1Backward, GeluBackward, HardshrinkBackward,
        LeakyReluBackward, Log1pBackward, LogAddExpBackward, LogBackward, LogSoftmaxBackward,
        PowBackward, PowBroadcast, ReluBackward, SeluBackward, SigmoidBackward, SiluBackward,
        SinBackward, SoftmaxBackward, SoftplusBackward, SoftsignBackward, TanBackward,
        TanhBackward, add_to_graph,
    },
    error::{MinitensorError, Result},
    tensor::{DataType, Shape, Tensor, TensorData},
};
use libm::{erf, erff};
use rayon::prelude::*;
use std::sync::Arc;

const PAR_THRESHOLD: usize = 1 << 12; // 4096 elements

#[inline(always)]
fn unary_apply<T, F>(input: &[T], output: &mut [T], op: F)
where
    T: Copy + Send + Sync,
    F: Fn(T) -> T + Sync + Send,
{
    #[inline(always)]
    fn apply_chunk<T, F>(input: &[T], output: &mut [T], op: &F)
    where
        T: Copy,
        F: Fn(T) -> T,
    {
        let len = input.len();
        let mut i = 0usize;
        let n = len.saturating_sub(len % 8);
        while i < n {
            unsafe {
                *output.get_unchecked_mut(i) = op(*input.get_unchecked(i));
                *output.get_unchecked_mut(i + 1) = op(*input.get_unchecked(i + 1));
                *output.get_unchecked_mut(i + 2) = op(*input.get_unchecked(i + 2));
                *output.get_unchecked_mut(i + 3) = op(*input.get_unchecked(i + 3));
                *output.get_unchecked_mut(i + 4) = op(*input.get_unchecked(i + 4));
                *output.get_unchecked_mut(i + 5) = op(*input.get_unchecked(i + 5));
                *output.get_unchecked_mut(i + 6) = op(*input.get_unchecked(i + 6));
                *output.get_unchecked_mut(i + 7) = op(*input.get_unchecked(i + 7));
            }
            i += 8;
        }
        for j in i..len {
            unsafe {
                *output.get_unchecked_mut(j) = op(*input.get_unchecked(j));
            }
        }
    }

    let len = input.len();
    debug_assert_eq!(len, output.len());
    if len < PAR_THRESHOLD {
        apply_chunk(input, output, &op);
    } else {
        const CHUNK: usize = 1024;
        input
            .par_chunks(CHUNK)
            .zip(output.par_chunks_mut(CHUNK))
            .for_each(|(in_chunk, out_chunk)| apply_chunk(in_chunk, out_chunk, &op));
    }
}

/// Exponential function with gradient support
pub fn exp(tensor: &Tensor) -> Result<Tensor> {
    // Create output tensor data
    let mut output_data =
        TensorData::uninitialized_on_device(tensor.numel(), tensor.dtype(), tensor.device());

    // Perform exponential based on data type
    match tensor.dtype() {
        DataType::Float32 => exp_f32(tensor, &mut output_data)?,
        DataType::Float64 => exp_f64(tensor, &mut output_data)?,
        _ => {
            return Err(MinitensorError::invalid_operation(
                "Exponential function only supported for floating point tensors",
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
        let grad_fn = Arc::new(ExpBackward {
            input_id: tensor.id(),
            output: output.clone().detach(),
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

/// Natural logarithm function with gradient support
pub fn log(tensor: &Tensor) -> Result<Tensor> {
    // Create output tensor data
    let mut output_data =
        TensorData::uninitialized_on_device(tensor.numel(), tensor.dtype(), tensor.device());

    // Perform logarithm based on data type
    match tensor.dtype() {
        DataType::Float32 => log_f32(tensor, &mut output_data)?,
        DataType::Float64 => log_f64(tensor, &mut output_data)?,
        _ => {
            return Err(MinitensorError::invalid_operation(
                "Logarithm function only supported for floating point tensors",
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
        let grad_fn = Arc::new(LogBackward {
            input_id: tensor.id(),
            input: tensor.clone().detach(),
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

/// log1p (log(1 + x)) function with gradient support
pub fn log1p(tensor: &Tensor) -> Result<Tensor> {
    let mut output_data =
        TensorData::uninitialized_on_device(tensor.numel(), tensor.dtype(), tensor.device());

    match tensor.dtype() {
        DataType::Float32 => log1p_f32(tensor, &mut output_data)?,
        DataType::Float64 => log1p_f64(tensor, &mut output_data)?,
        _ => {
            return Err(MinitensorError::invalid_operation(
                "log1p is only supported for floating point tensors",
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
        let grad_fn = Arc::new(Log1pBackward {
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

/// expm1 (exp(x) - 1) with gradient support
pub fn expm1(tensor: &Tensor) -> Result<Tensor> {
    let mut output_data =
        TensorData::uninitialized_on_device(tensor.numel(), tensor.dtype(), tensor.device());

    match tensor.dtype() {
        DataType::Float32 => expm1_f32(tensor, &mut output_data)?,
        DataType::Float64 => expm1_f64(tensor, &mut output_data)?,
        _ => {
            return Err(MinitensorError::invalid_operation(
                "expm1 is only supported for floating point tensors",
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
        let grad_fn = Arc::new(Expm1Backward {
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

/// Sine function with gradient support
pub fn sin(tensor: &Tensor) -> Result<Tensor> {
    // Create output tensor data
    let mut output_data =
        TensorData::uninitialized_on_device(tensor.numel(), tensor.dtype(), tensor.device());

    // Perform sine based on data type
    match tensor.dtype() {
        DataType::Float32 => sin_f32(tensor, &mut output_data)?,
        DataType::Float64 => sin_f64(tensor, &mut output_data)?,
        _ => {
            return Err(MinitensorError::invalid_operation(
                "Sine function only supported for floating point tensors",
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
        let grad_fn = Arc::new(SinBackward {
            input_id: tensor.id(),
            input: tensor.clone(),
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

/// Cosine function with gradient support
pub fn cos(tensor: &Tensor) -> Result<Tensor> {
    // Create output tensor data
    let mut output_data =
        TensorData::uninitialized_on_device(tensor.numel(), tensor.dtype(), tensor.device());

    // Perform cosine based on data type
    match tensor.dtype() {
        DataType::Float32 => cos_f32(tensor, &mut output_data)?,
        DataType::Float64 => cos_f64(tensor, &mut output_data)?,
        _ => {
            return Err(MinitensorError::invalid_operation(
                "Cosine function only supported for floating point tensors",
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
        let grad_fn = Arc::new(CosBackward {
            input_id: tensor.id(),
            input: tensor.clone(),
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

/// Tangent function with gradient support
pub fn tan(tensor: &Tensor) -> Result<Tensor> {
    // Create output tensor data
    let mut output_data =
        TensorData::uninitialized_on_device(tensor.numel(), tensor.dtype(), tensor.device());

    // Perform tangent based on data type
    match tensor.dtype() {
        DataType::Float32 => tan_f32(tensor, &mut output_data)?,
        DataType::Float64 => tan_f64(tensor, &mut output_data)?,
        _ => {
            return Err(MinitensorError::invalid_operation(
                "Tangent function only supported for floating point tensors",
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
        let grad_fn = Arc::new(TanBackward {
            input_id: tensor.id(),
            output: output.clone().detach(),
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

/// Hyperbolic tangent function with gradient support
pub fn tanh(tensor: &Tensor) -> Result<Tensor> {
    // Create output tensor data
    let mut output_data =
        TensorData::uninitialized_on_device(tensor.numel(), tensor.dtype(), tensor.device());

    // Perform tanh based on data type
    match tensor.dtype() {
        DataType::Float32 => tanh_f32(tensor, &mut output_data)?,
        DataType::Float64 => tanh_f64(tensor, &mut output_data)?,
        _ => {
            return Err(MinitensorError::invalid_operation(
                "Tanh function only supported for floating point tensors",
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
        let grad_fn = Arc::new(TanhBackward {
            input_id: tensor.id(),
            output: output.clone(),
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

/// Sigmoid activation function with gradient support
pub fn sigmoid(tensor: &Tensor) -> Result<Tensor> {
    // Create output tensor data
    let mut output_data =
        TensorData::uninitialized_on_device(tensor.numel(), tensor.dtype(), tensor.device());

    // Perform sigmoid based on data type
    match tensor.dtype() {
        DataType::Float32 => sigmoid_f32(tensor, &mut output_data)?,
        DataType::Float64 => sigmoid_f64(tensor, &mut output_data)?,
        _ => {
            return Err(MinitensorError::invalid_operation(
                "Sigmoid function only supported for floating point tensors",
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
        let grad_fn = Arc::new(SigmoidBackward {
            input_id: tensor.id(),
            output: output.clone(),
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
        return Err(MinitensorError::shape_mismatch(
            base_shape.dims().to_vec(),
            exponent_shape.dims().to_vec(),
        ));
    };

    let output_shape = match broadcast {
        PowBroadcast::None | PowBroadcast::ExponentScalar => base_shape.clone(),
        PowBroadcast::BaseScalar => exponent_shape.clone(),
    };

    let mut output_data =
        TensorData::uninitialized_on_device(output_shape.numel(), base.dtype(), base.device());

    match base.dtype() {
        DataType::Float32 => {
            let b = base.data().as_f32_slice().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get f32 slice from base tensor")
            })?;
            let e = exponent.data().as_f32_slice().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get f32 slice from exponent tensor")
            })?;
            let out = output_data.as_f32_slice_mut().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get mutable f32 slice from output data")
            })?;
            match broadcast {
                PowBroadcast::None => {
                    for i in 0..b.len() {
                        out[i] = b[i].powf(e[i]);
                    }
                }
                PowBroadcast::BaseScalar => {
                    let base_val = b[0];
                    for i in 0..e.len() {
                        out[i] = base_val.powf(e[i]);
                    }
                }
                PowBroadcast::ExponentScalar => {
                    let exp_val = e[0];
                    for i in 0..b.len() {
                        out[i] = b[i].powf(exp_val);
                    }
                }
            }
        }
        DataType::Float64 => {
            let b = base.data().as_f64_slice().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get f64 slice from base tensor")
            })?;
            let e = exponent.data().as_f64_slice().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get f64 slice from exponent tensor")
            })?;
            let out = output_data.as_f64_slice_mut().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get mutable f64 slice from output data")
            })?;
            match broadcast {
                PowBroadcast::None => {
                    for i in 0..b.len() {
                        out[i] = b[i].powf(e[i]);
                    }
                }
                PowBroadcast::BaseScalar => {
                    let base_val = b[0];
                    for i in 0..e.len() {
                        out[i] = base_val.powf(e[i]);
                    }
                }
                PowBroadcast::ExponentScalar => {
                    let exp_val = e[0];
                    for i in 0..b.len() {
                        out[i] = b[i].powf(exp_val);
                    }
                }
            }
        }
        _ => {
            return Err(MinitensorError::invalid_operation(
                "Power operation only supported for floating point tensors",
            ));
        }
    }

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
    // Create exponent tensor filled with scalar value
    let mut exp_data =
        TensorData::uninitialized_on_device(tensor.numel(), tensor.dtype(), tensor.device());
    match tensor.dtype() {
        DataType::Float32 => {
            let slice = exp_data.as_f32_slice_mut().ok_or_else(|| {
                MinitensorError::internal_error(
                    "Failed to get mutable f32 slice from exponent data",
                )
            })?;
            for val in slice.iter_mut() {
                *val = exponent as f32;
            }
        }
        DataType::Float64 => {
            let slice = exp_data.as_f64_slice_mut().ok_or_else(|| {
                MinitensorError::internal_error(
                    "Failed to get mutable f64 slice from exponent data",
                )
            })?;
            for val in slice.iter_mut() {
                *val = exponent;
            }
        }
        _ => {
            return Err(MinitensorError::invalid_operation(
                "Power operation only supported for floating point tensors",
            ));
        }
    }
    let exp_tensor = Tensor::new(
        Arc::new(exp_data),
        tensor.shape().clone(),
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
    use crate::operations::binary::{BinaryOpKind, coerce_binary_operands};
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
    let mut output_data =
        TensorData::uninitialized_on_device(output_shape.numel(), result_dtype, lhs.device());

    match result_dtype {
        DataType::Float32 => {
            logaddexp_f32(&lhs_tensor, &rhs_tensor, &mut output_data, &output_shape)?
        }
        DataType::Float64 => {
            logaddexp_f64(&lhs_tensor, &rhs_tensor, &mut output_data, &output_shape)?
        }
        _ => unreachable!(),
    }

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

    let mut output_data =
        TensorData::uninitialized_on_device(tensor.numel(), tensor.dtype(), tensor.device());

    match tensor.dtype() {
        DataType::Float32 => softplus_f32(tensor, &mut output_data, beta as f32, threshold as f32)?,
        DataType::Float64 => softplus_f64(tensor, &mut output_data, beta, threshold)?,
        _ => {
            return Err(MinitensorError::invalid_operation(
                "Softplus is only supported for floating point tensors",
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
    let mut output_data =
        TensorData::uninitialized_on_device(tensor.numel(), tensor.dtype(), tensor.device());

    match tensor.dtype() {
        DataType::Float32 => gelu_f32(tensor, &mut output_data, approximate)?,
        DataType::Float64 => gelu_f64(tensor, &mut output_data, approximate)?,
        _ => {
            return Err(MinitensorError::invalid_operation(
                "GELU is only supported for floating point tensors",
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
    let mut output_data =
        TensorData::uninitialized_on_device(tensor.numel(), tensor.dtype(), tensor.device());

    match tensor.dtype() {
        DataType::Float32 => elu_f32(tensor, &mut output_data, alpha as f32)?,
        DataType::Float64 => elu_f64(tensor, &mut output_data, alpha)?,
        _ => {
            return Err(MinitensorError::invalid_operation(
                "ELU is only supported for floating point tensors",
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

/// SELU activation function following PyTorch constants
pub fn selu(tensor: &Tensor) -> Result<Tensor> {
    let mut output_data =
        TensorData::uninitialized_on_device(tensor.numel(), tensor.dtype(), tensor.device());

    match tensor.dtype() {
        DataType::Float32 => selu_f32(tensor, &mut output_data)?,
        DataType::Float64 => selu_f64(tensor, &mut output_data)?,
        _ => {
            return Err(MinitensorError::invalid_operation(
                "SELU is only supported for floating point tensors",
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
    let mut output_data =
        TensorData::uninitialized_on_device(tensor.numel(), tensor.dtype(), tensor.device());

    match tensor.dtype() {
        DataType::Float32 => silu_f32(tensor, &mut output_data)?,
        DataType::Float64 => silu_f64(tensor, &mut output_data)?,
        _ => {
            return Err(MinitensorError::invalid_operation(
                "SiLU is only supported for floating point tensors",
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
    let mut output_data =
        TensorData::uninitialized_on_device(tensor.numel(), tensor.dtype(), tensor.device());

    match tensor.dtype() {
        DataType::Float32 => softsign_f32(tensor, &mut output_data)?,
        DataType::Float64 => softsign_f64(tensor, &mut output_data)?,
        _ => {
            return Err(MinitensorError::invalid_operation(
                "Softsign is only supported for floating point tensors",
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
    // Create output tensor data
    let mut output_data =
        TensorData::uninitialized_on_device(tensor.numel(), tensor.dtype(), tensor.device());

    // Perform ReLU based on data type while capturing mask of positive inputs
    let mask = match tensor.dtype() {
        DataType::Float32 => relu_f32(tensor, &mut output_data)?,
        DataType::Float64 => relu_f64(tensor, &mut output_data)?,
        DataType::Int32 => relu_i32(tensor, &mut output_data)?,
        DataType::Int64 => relu_i64(tensor, &mut output_data)?,
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
            mask,
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

    let mut output_data =
        TensorData::uninitialized_on_device(tensor.numel(), tensor.dtype(), tensor.device());

    let store_mask = tensor.requires_grad();
    let mask = match tensor.dtype() {
        DataType::Float32 => hardshrink_f32(tensor, &mut output_data, lambd as f32, store_mask)?,
        DataType::Float64 => hardshrink_f64(tensor, &mut output_data, lambd, store_mask)?,
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
    // Create output tensor data
    let mut output_data =
        TensorData::uninitialized_on_device(tensor.numel(), tensor.dtype(), tensor.device());

    // Perform LeakyReLU based on data type and capture mask of positive inputs
    let mask = match tensor.dtype() {
        DataType::Float32 => leaky_relu_f32(tensor, &mut output_data, negative_slope as f32)?,
        DataType::Float64 => leaky_relu_f64(tensor, &mut output_data, negative_slope)?,
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
            mask,
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

// Helper functions for type-specific operations

fn exp_f32(tensor: &Tensor, output_data: &mut TensorData) -> Result<()> {
    let input_data = tensor.data().as_f32_slice().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get f32 slice from input tensor")
    })?;

    let output_slice = output_data.as_f32_slice_mut().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get mutable f32 slice from output data")
    })?;
    unary_apply(input_data, output_slice, f32::exp);
    Ok(())
}

fn exp_f64(tensor: &Tensor, output_data: &mut TensorData) -> Result<()> {
    let input_data = tensor.data().as_f64_slice().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get f64 slice from input tensor")
    })?;

    let output_slice = output_data.as_f64_slice_mut().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get mutable f64 slice from output data")
    })?;
    unary_apply(input_data, output_slice, f64::exp);
    Ok(())
}

fn log_f32(tensor: &Tensor, output_data: &mut TensorData) -> Result<()> {
    let input_data = tensor.data().as_f32_slice().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get f32 slice from input tensor")
    })?;

    let output_slice = output_data.as_f32_slice_mut().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get mutable f32 slice from output data")
    })?;
    unary_apply(input_data, output_slice, |val: f32| {
        if val <= 0.0 {
            f32::NEG_INFINITY
        } else {
            val.ln()
        }
    });
    Ok(())
}

fn log_f64(tensor: &Tensor, output_data: &mut TensorData) -> Result<()> {
    let input_data = tensor.data().as_f64_slice().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get f64 slice from input tensor")
    })?;

    let output_slice = output_data.as_f64_slice_mut().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get mutable f64 slice from output data")
    })?;
    unary_apply(input_data, output_slice, |val: f64| {
        if val <= 0.0 {
            f64::NEG_INFINITY
        } else {
            val.ln()
        }
    });
    Ok(())
}

fn log1p_f32(tensor: &Tensor, output_data: &mut TensorData) -> Result<()> {
    let input_data = tensor.data().as_f32_slice().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get f32 slice from input tensor")
    })?;

    let output_slice = output_data.as_f32_slice_mut().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get mutable f32 slice from output data")
    })?;

    unary_apply(input_data, output_slice, |val: f32| {
        if val == -1.0 {
            f32::NEG_INFINITY
        } else if val < -1.0 {
            f32::NAN
        } else {
            val.ln_1p()
        }
    });
    Ok(())
}

fn log1p_f64(tensor: &Tensor, output_data: &mut TensorData) -> Result<()> {
    let input_data = tensor.data().as_f64_slice().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get f64 slice from input tensor")
    })?;

    let output_slice = output_data.as_f64_slice_mut().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get mutable f64 slice from output data")
    })?;

    unary_apply(input_data, output_slice, |val: f64| {
        if val == -1.0 {
            f64::NEG_INFINITY
        } else if val < -1.0 {
            f64::NAN
        } else {
            val.ln_1p()
        }
    });
    Ok(())
}

fn expm1_f32(tensor: &Tensor, output_data: &mut TensorData) -> Result<()> {
    let input_data = tensor.data().as_f32_slice().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get f32 slice from input tensor")
    })?;

    let output_slice = output_data.as_f32_slice_mut().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get mutable f32 slice from output data")
    })?;

    unary_apply(input_data, output_slice, f32::exp_m1);
    Ok(())
}

fn expm1_f64(tensor: &Tensor, output_data: &mut TensorData) -> Result<()> {
    let input_data = tensor.data().as_f64_slice().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get f64 slice from input tensor")
    })?;

    let output_slice = output_data.as_f64_slice_mut().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get mutable f64 slice from output data")
    })?;

    unary_apply(input_data, output_slice, f64::exp_m1);
    Ok(())
}

fn sin_f32(tensor: &Tensor, output_data: &mut TensorData) -> Result<()> {
    let input_data = tensor.data().as_f32_slice().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get f32 slice from input tensor")
    })?;

    let output_slice = output_data.as_f32_slice_mut().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get mutable f32 slice from output data")
    })?;
    unary_apply(input_data, output_slice, f32::sin);
    Ok(())
}

fn sin_f64(tensor: &Tensor, output_data: &mut TensorData) -> Result<()> {
    let input_data = tensor.data().as_f64_slice().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get f64 slice from input tensor")
    })?;

    let output_slice = output_data.as_f64_slice_mut().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get mutable f64 slice from output data")
    })?;
    unary_apply(input_data, output_slice, f64::sin);
    Ok(())
}

fn cos_f32(tensor: &Tensor, output_data: &mut TensorData) -> Result<()> {
    let input_data = tensor.data().as_f32_slice().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get f32 slice from input tensor")
    })?;

    let output_slice = output_data.as_f32_slice_mut().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get mutable f32 slice from output data")
    })?;
    unary_apply(input_data, output_slice, f32::cos);
    Ok(())
}

fn cos_f64(tensor: &Tensor, output_data: &mut TensorData) -> Result<()> {
    let input_data = tensor.data().as_f64_slice().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get f64 slice from input tensor")
    })?;

    let output_slice = output_data.as_f64_slice_mut().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get mutable f64 slice from output data")
    })?;
    unary_apply(input_data, output_slice, f64::cos);
    Ok(())
}

fn tan_f32(tensor: &Tensor, output_data: &mut TensorData) -> Result<()> {
    let input_data = tensor.data().as_f32_slice().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get f32 slice from input tensor")
    })?;

    let output_slice = output_data.as_f32_slice_mut().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get mutable f32 slice from output data")
    })?;
    unary_apply(input_data, output_slice, f32::tan);
    Ok(())
}

fn tan_f64(tensor: &Tensor, output_data: &mut TensorData) -> Result<()> {
    let input_data = tensor.data().as_f64_slice().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get f64 slice from input tensor")
    })?;

    let output_slice = output_data.as_f64_slice_mut().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get mutable f64 slice from output data")
    })?;
    unary_apply(input_data, output_slice, f64::tan);
    Ok(())
}

fn softplus_f32(
    tensor: &Tensor,
    output_data: &mut TensorData,
    beta: f32,
    threshold: f32,
) -> Result<()> {
    let input_data = tensor.data().as_f32_slice().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get f32 slice from input tensor")
    })?;

    let output_slice = output_data.as_f32_slice_mut().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get mutable f32 slice from output data")
    })?;

    unary_apply(input_data, output_slice, |val: f32| {
        let scaled = beta * val;
        if scaled > threshold {
            val
        } else {
            scaled.exp().ln_1p() / beta
        }
    });
    Ok(())
}

fn softplus_f64(
    tensor: &Tensor,
    output_data: &mut TensorData,
    beta: f64,
    threshold: f64,
) -> Result<()> {
    let input_data = tensor.data().as_f64_slice().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get f64 slice from input tensor")
    })?;

    let output_slice = output_data.as_f64_slice_mut().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get mutable f64 slice from output data")
    })?;

    unary_apply(input_data, output_slice, |val: f64| {
        let scaled = beta * val;
        if scaled > threshold {
            val
        } else {
            scaled.exp().ln_1p() / beta
        }
    });
    Ok(())
}

fn gelu_f32(tensor: &Tensor, output_data: &mut TensorData, approximate: bool) -> Result<()> {
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

fn gelu_f64(tensor: &Tensor, output_data: &mut TensorData, approximate: bool) -> Result<()> {
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

fn elu_f32(tensor: &Tensor, output_data: &mut TensorData, alpha: f32) -> Result<()> {
    let input_data = tensor.data().as_f32_slice().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get f32 slice from input tensor")
    })?;

    let output_slice = output_data.as_f32_slice_mut().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get mutable f32 slice from output data")
    })?;

    unary_apply(input_data, output_slice, |x: f32| {
        if x > 0.0 { x } else { alpha * (x.exp() - 1.0) }
    });
    Ok(())
}

fn elu_f64(tensor: &Tensor, output_data: &mut TensorData, alpha: f64) -> Result<()> {
    let input_data = tensor.data().as_f64_slice().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get f64 slice from input tensor")
    })?;

    let output_slice = output_data.as_f64_slice_mut().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get mutable f64 slice from output data")
    })?;

    unary_apply(input_data, output_slice, |x: f64| {
        if x > 0.0 { x } else { alpha * (x.exp() - 1.0) }
    });
    Ok(())
}

fn selu_f32(tensor: &Tensor, output_data: &mut TensorData) -> Result<()> {
    let input_data = tensor.data().as_f32_slice().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get f32 slice from input tensor")
    })?;

    let output_slice = output_data.as_f32_slice_mut().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get mutable f32 slice from output data")
    })?;

    const ALPHA: f32 = 1.6732632;
    const SCALE: f32 = 1.050701;
    unary_apply(input_data, output_slice, |x: f32| {
        if x > 0.0 {
            SCALE * x
        } else {
            SCALE * ALPHA * (x.exp() - 1.0)
        }
    });
    Ok(())
}

fn selu_f64(tensor: &Tensor, output_data: &mut TensorData) -> Result<()> {
    let input_data = tensor.data().as_f64_slice().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get f64 slice from input tensor")
    })?;

    let output_slice = output_data.as_f64_slice_mut().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get mutable f64 slice from output data")
    })?;

    const ALPHA: f64 = 1.6732632423543772848170429916717;
    const SCALE: f64 = 1.0507009873554804934193349852946;
    unary_apply(input_data, output_slice, |x: f64| {
        if x > 0.0 {
            SCALE * x
        } else {
            SCALE * ALPHA * (x.exp() - 1.0)
        }
    });
    Ok(())
}

fn silu_f32(tensor: &Tensor, output_data: &mut TensorData) -> Result<()> {
    let input_data = tensor.data().as_f32_slice().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get f32 slice from input tensor")
    })?;

    let output_slice = output_data.as_f32_slice_mut().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get mutable f32 slice from output data")
    })?;

    unary_apply(input_data, output_slice, |x: f32| {
        let sigmoid = 1.0 / (1.0 + (-x).exp());
        x * sigmoid
    });
    Ok(())
}

fn silu_f64(tensor: &Tensor, output_data: &mut TensorData) -> Result<()> {
    let input_data = tensor.data().as_f64_slice().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get f64 slice from input tensor")
    })?;

    let output_slice = output_data.as_f64_slice_mut().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get mutable f64 slice from output data")
    })?;

    unary_apply(input_data, output_slice, |x: f64| {
        let sigmoid = 1.0 / (1.0 + (-x).exp());
        x * sigmoid
    });
    Ok(())
}

fn softsign_f32(tensor: &Tensor, output_data: &mut TensorData) -> Result<()> {
    let input_data = tensor.data().as_f32_slice().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get f32 slice from input tensor")
    })?;

    let output_slice = output_data.as_f32_slice_mut().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get mutable f32 slice from output data")
    })?;

    unary_apply(input_data, output_slice, |x: f32| {
        let denom = 1.0 + x.abs();
        x / denom
    });
    Ok(())
}

fn softsign_f64(tensor: &Tensor, output_data: &mut TensorData) -> Result<()> {
    let input_data = tensor.data().as_f64_slice().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get f64 slice from input tensor")
    })?;

    let output_slice = output_data.as_f64_slice_mut().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get mutable f64 slice from output data")
    })?;

    unary_apply(input_data, output_slice, |x: f64| {
        let denom = 1.0 + x.abs();
        x / denom
    });
    Ok(())
}

fn logaddexp_f32(
    lhs: &Tensor,
    rhs: &Tensor,
    output_data: &mut TensorData,
    output_shape: &Shape,
) -> Result<()> {
    let lhs_data = lhs.data().as_f32_slice().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get f32 slice from lhs tensor")
    })?;
    let rhs_data = rhs.data().as_f32_slice().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get f32 slice from rhs tensor")
    })?;

    let output_slice = output_data.as_f32_slice_mut().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get mutable f32 slice from output data")
    })?;

    crate::operations::arithmetic::broadcast_binary_op(
        lhs_data,
        rhs_data,
        output_slice,
        lhs.shape(),
        rhs.shape(),
        output_shape,
        |a, b| {
            if a.is_nan() || b.is_nan() {
                f32::NAN
            } else {
                let max = a.max(b);
                if max.is_infinite() {
                    max
                } else {
                    let exp_a = (a - max).exp();
                    let exp_b = (b - max).exp();
                    max + (exp_a + exp_b).ln()
                }
            }
        },
    )
}

fn logaddexp_f64(
    lhs: &Tensor,
    rhs: &Tensor,
    output_data: &mut TensorData,
    output_shape: &Shape,
) -> Result<()> {
    let lhs_data = lhs.data().as_f64_slice().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get f64 slice from lhs tensor")
    })?;
    let rhs_data = rhs.data().as_f64_slice().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get f64 slice from rhs tensor")
    })?;

    let output_slice = output_data.as_f64_slice_mut().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get mutable f64 slice from output data")
    })?;

    crate::operations::arithmetic::broadcast_binary_op(
        lhs_data,
        rhs_data,
        output_slice,
        lhs.shape(),
        rhs.shape(),
        output_shape,
        |a, b| {
            if a.is_nan() || b.is_nan() {
                f64::NAN
            } else {
                let max = a.max(b);
                if max.is_infinite() {
                    max
                } else {
                    let exp_a = (a - max).exp();
                    let exp_b = (b - max).exp();
                    max + (exp_a + exp_b).ln()
                }
            }
        },
    )
}

fn tanh_f32(tensor: &Tensor, output_data: &mut TensorData) -> Result<()> {
    let input_data = tensor.data().as_f32_slice().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get f32 slice from input tensor")
    })?;

    let output_slice = output_data.as_f32_slice_mut().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get mutable f32 slice from output data")
    })?;
    unary_apply(input_data, output_slice, f32::tanh);
    Ok(())
}

fn tanh_f64(tensor: &Tensor, output_data: &mut TensorData) -> Result<()> {
    let input_data = tensor.data().as_f64_slice().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get f64 slice from input tensor")
    })?;

    let output_slice = output_data.as_f64_slice_mut().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get mutable f64 slice from output data")
    })?;
    unary_apply(input_data, output_slice, f64::tanh);
    Ok(())
}

fn sigmoid_f32(tensor: &Tensor, output_data: &mut TensorData) -> Result<()> {
    let input_data = tensor.data().as_f32_slice().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get f32 slice from input tensor")
    })?;

    let output_slice = output_data.as_f32_slice_mut().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get mutable f32 slice from output data")
    })?;
    unary_apply(input_data, output_slice, |val: f32| {
        1.0 / (1.0 + (-val).exp())
    });
    Ok(())
}

fn sigmoid_f64(tensor: &Tensor, output_data: &mut TensorData) -> Result<()> {
    let input_data = tensor.data().as_f64_slice().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get f64 slice from input tensor")
    })?;

    let output_slice = output_data.as_f64_slice_mut().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get mutable f64 slice from output data")
    })?;
    unary_apply(input_data, output_slice, |val: f64| {
        1.0 / (1.0 + (-val).exp())
    });
    Ok(())
}

fn relu_f32(tensor: &Tensor, output_data: &mut TensorData) -> Result<Vec<bool>> {
    let input_data = tensor.data().as_f32_slice().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get f32 slice from input tensor")
    })?;

    let output_slice = output_data.as_f32_slice_mut().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get mutable f32 slice from output data")
    })?;
    let len = input_data.len();
    let mut mask = vec![false; len];
    if len >= PAR_THRESHOLD {
        output_slice
            .par_iter_mut()
            .zip(input_data.par_iter())
            .zip(mask.par_iter_mut())
            .for_each(|((o, &v), m)| {
                if v.is_nan() {
                    *o = v;
                } else if v > 0.0 {
                    *o = v;
                    *m = true;
                } else {
                    *o = 0.0;
                }
            });
    } else {
        for ((o, &v), m) in output_slice
            .iter_mut()
            .zip(input_data.iter())
            .zip(mask.iter_mut())
        {
            if v.is_nan() {
                *o = v;
            } else if v > 0.0 {
                *o = v;
                *m = true;
            } else {
                *o = 0.0;
            }
        }
    }
    Ok(mask)
}

fn relu_f64(tensor: &Tensor, output_data: &mut TensorData) -> Result<Vec<bool>> {
    let input_data = tensor.data().as_f64_slice().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get f64 slice from input tensor")
    })?;

    let output_slice = output_data.as_f64_slice_mut().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get mutable f64 slice from output data")
    })?;
    let len = input_data.len();
    let mut mask = vec![false; len];
    if len >= PAR_THRESHOLD {
        output_slice
            .par_iter_mut()
            .zip(input_data.par_iter())
            .zip(mask.par_iter_mut())
            .for_each(|((o, &v), m)| {
                if v.is_nan() {
                    *o = v;
                } else if v > 0.0 {
                    *o = v;
                    *m = true;
                } else {
                    *o = 0.0;
                }
            });
    } else {
        for ((o, &v), m) in output_slice
            .iter_mut()
            .zip(input_data.iter())
            .zip(mask.iter_mut())
        {
            if v.is_nan() {
                *o = v;
            } else if v > 0.0 {
                *o = v;
                *m = true;
            } else {
                *o = 0.0;
            }
        }
    }
    Ok(mask)
}

fn relu_i32(tensor: &Tensor, output_data: &mut TensorData) -> Result<Vec<bool>> {
    let input_data = tensor.data().as_i32_slice().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get i32 slice from input tensor")
    })?;

    let output_slice = output_data.as_i32_slice_mut().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get mutable i32 slice from output data")
    })?;
    let len = input_data.len();
    let mut mask = vec![false; len];
    if len >= PAR_THRESHOLD {
        output_slice
            .par_iter_mut()
            .zip(input_data.par_iter())
            .zip(mask.par_iter_mut())
            .for_each(|((o, &v), m)| {
                if v > 0 {
                    *o = v;
                    *m = true;
                } else {
                    *o = 0;
                }
            });
    } else {
        for ((o, &v), m) in output_slice
            .iter_mut()
            .zip(input_data.iter())
            .zip(mask.iter_mut())
        {
            if v > 0 {
                *o = v;
                *m = true;
            } else {
                *o = 0;
            }
        }
    }
    Ok(mask)
}

fn relu_i64(tensor: &Tensor, output_data: &mut TensorData) -> Result<Vec<bool>> {
    let input_data = tensor.data().as_i64_slice().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get i64 slice from input tensor")
    })?;

    let output_slice = output_data.as_i64_slice_mut().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get mutable i64 slice from output data")
    })?;
    let len = input_data.len();
    let mut mask = vec![false; len];
    if len >= PAR_THRESHOLD {
        output_slice
            .par_iter_mut()
            .zip(input_data.par_iter())
            .zip(mask.par_iter_mut())
            .for_each(|((o, &v), m)| {
                if v > 0 {
                    *o = v;
                    *m = true;
                } else {
                    *o = 0;
                }
            });
    } else {
        for ((o, &v), m) in output_slice
            .iter_mut()
            .zip(input_data.iter())
            .zip(mask.iter_mut())
        {
            if v > 0 {
                *o = v;
                *m = true;
            } else {
                *o = 0;
            }
        }
    }
    Ok(mask)
}

fn hardshrink_f32(
    tensor: &Tensor,
    output_data: &mut TensorData,
    lambd: f32,
    store_mask: bool,
) -> Result<Option<Vec<bool>>> {
    let input_data = tensor.data().as_f32_slice().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get f32 slice from input tensor")
    })?;

    let output_slice = output_data.as_f32_slice_mut().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get mutable f32 slice from output data")
    })?;

    let mut mask = if store_mask {
        Some(Vec::with_capacity(input_data.len()))
    } else {
        None
    };

    for (&value, out_slot) in input_data.iter().zip(output_slice.iter_mut()) {
        let keep = value > lambd || value < -lambd;
        *out_slot = if keep { value } else { 0.0 };
        if let Some(ref mut mask_vec) = mask {
            mask_vec.push(keep);
        }
    }

    Ok(mask)
}

fn hardshrink_f64(
    tensor: &Tensor,
    output_data: &mut TensorData,
    lambd: f64,
    store_mask: bool,
) -> Result<Option<Vec<bool>>> {
    let input_data = tensor.data().as_f64_slice().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get f64 slice from input tensor")
    })?;

    let output_slice = output_data.as_f64_slice_mut().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get mutable f64 slice from output data")
    })?;

    let mut mask = if store_mask {
        Some(Vec::with_capacity(input_data.len()))
    } else {
        None
    };

    for (&value, out_slot) in input_data.iter().zip(output_slice.iter_mut()) {
        let keep = value > lambd || value < -lambd;
        *out_slot = if keep { value } else { 0.0 };
        if let Some(ref mut mask_vec) = mask {
            mask_vec.push(keep);
        }
    }

    Ok(mask)
}

fn leaky_relu_f32(
    tensor: &Tensor,
    output_data: &mut TensorData,
    negative_slope: f32,
) -> Result<Vec<bool>> {
    let input_data = tensor.data().as_f32_slice().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get f32 slice from input tensor")
    })?;

    let output_slice = output_data.as_f32_slice_mut().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get mutable f32 slice from output data")
    })?;

    let len = input_data.len();
    let mut mask = vec![false; len];
    let mask_ptr = mask.as_mut_ptr() as usize;
    let in_ptr = input_data.as_ptr() as usize;
    let out_ptr = output_slice.as_mut_ptr() as usize;
    (0..len).into_par_iter().for_each(|i| unsafe {
        let in_ptr = in_ptr as *const f32;
        let out_ptr = out_ptr as *mut f32;
        let mask_ptr = mask_ptr as *mut bool;
        let val = *in_ptr.add(i);
        if val >= 0.0 {
            *out_ptr.add(i) = val;
            *mask_ptr.add(i) = true;
        } else {
            *out_ptr.add(i) = negative_slope * val;
        }
    });
    Ok(mask)
}

fn leaky_relu_f64(
    tensor: &Tensor,
    output_data: &mut TensorData,
    negative_slope: f64,
) -> Result<Vec<bool>> {
    let input_data = tensor.data().as_f64_slice().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get f64 slice from input tensor")
    })?;

    let output_slice = output_data.as_f64_slice_mut().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get mutable f64 slice from output data")
    })?;

    let len = input_data.len();
    let mut mask = vec![false; len];
    let mask_ptr = mask.as_mut_ptr() as usize;
    let in_ptr = input_data.as_ptr() as usize;
    let out_ptr = output_slice.as_mut_ptr() as usize;
    (0..len).into_par_iter().for_each(|i| unsafe {
        let in_ptr = in_ptr as *const f64;
        let out_ptr = out_ptr as *mut f64;
        let mask_ptr = mask_ptr as *mut bool;
        let val = *in_ptr.add(i);
        if val >= 0.0 {
            *out_ptr.add(i) = val;
            *mask_ptr.add(i) = true;
        } else {
            *out_ptr.add(i) = negative_slope * val;
        }
    });
    Ok(mask)
}

fn softmax_f32(tensor: &Tensor, output_data: &mut TensorData, dim: usize) -> Result<()> {
    let input_data = tensor.data().as_f32_slice().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get f32 slice from input tensor")
    })?;

    let output_slice = output_data.as_f32_slice_mut().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get mutable f32 slice from output data")
    })?;

    let dims = tensor.shape().dims();
    let dim_size = dims[dim];

    // Compute the number of groups before and after the softmax dimension. This
    // allows us to iterate over all slices along `dim` for tensors of arbitrary
    // rank using row-major indexing.
    let _before: usize = if dim == 0 {
        1
    } else {
        dims[..dim].iter().product()
    };
    let after: usize = if dim + 1 >= dims.len() {
        1
    } else {
        dims[dim + 1..].iter().product()
    };
    let group = dim_size * after;
    input_data
        .par_chunks(group)
        .zip(output_slice.par_chunks_mut(group))
        .for_each(|(in_block, out_block)| {
            for a in 0..after {
                let base = a;
                let mut max_val = f32::NEG_INFINITY;
                for k in 0..dim_size {
                    let idx = base + k * after;
                    max_val = max_val.max(in_block[idx]);
                }
                let mut sum = 0.0f32;
                for k in 0..dim_size {
                    let idx = base + k * after;
                    let val = (in_block[idx] - max_val).exp();
                    out_block[idx] = val;
                    sum += val;
                }
                for k in 0..dim_size {
                    let idx = base + k * after;
                    out_block[idx] /= sum;
                }
            }
        });

    Ok(())
}

fn softmax_f64(tensor: &Tensor, output_data: &mut TensorData, dim: usize) -> Result<()> {
    let input_data = tensor.data().as_f64_slice().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get f64 slice from input tensor")
    })?;

    let output_slice = output_data.as_f64_slice_mut().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get mutable f64 slice from output data")
    })?;

    let dims = tensor.shape().dims();
    let dim_size = dims[dim];

    let _before: usize = if dim == 0 {
        1
    } else {
        dims[..dim].iter().product()
    };
    let after: usize = if dim + 1 >= dims.len() {
        1
    } else {
        dims[dim + 1..].iter().product()
    };
    let group = dim_size * after;
    input_data
        .par_chunks(group)
        .zip(output_slice.par_chunks_mut(group))
        .for_each(|(in_block, out_block)| {
            for a in 0..after {
                let base = a;
                let mut max_val = f64::NEG_INFINITY;
                for k in 0..dim_size {
                    let idx = base + k * after;
                    max_val = max_val.max(in_block[idx]);
                }
                let mut sum = 0.0f64;
                for k in 0..dim_size {
                    let idx = base + k * after;
                    let val = (in_block[idx] - max_val).exp();
                    out_block[idx] = val;
                    sum += val;
                }
                for k in 0..dim_size {
                    let idx = base + k * after;
                    out_block[idx] /= sum;
                }
            }
        });

    Ok(())
}

fn log_softmax_f32(tensor: &Tensor, output_data: &mut TensorData, dim: usize) -> Result<()> {
    let input_data = tensor.data().as_f32_slice().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get f32 slice from input tensor")
    })?;

    let output_slice = output_data.as_f32_slice_mut().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get mutable f32 slice from output data")
    })?;

    let dims = tensor.shape().dims();
    let dim_size = dims[dim];

    let after: usize = if dim + 1 >= dims.len() {
        1
    } else {
        dims[dim + 1..].iter().product()
    };
    let group = dim_size * after;

    let cutoff = (f32::MIN_POSITIVE * f32::EPSILON).ln();

    input_data
        .par_chunks(group)
        .zip(output_slice.par_chunks_mut(group))
        .for_each(|(in_block, out_block)| {
            for a in 0..after {
                let base = a;
                let mut max_val = f32::NEG_INFINITY;
                for k in 0..dim_size {
                    let idx = base + k * after;
                    max_val = max_val.max(in_block[idx]);
                }
                let mut sum = 0.0f32;
                for k in 0..dim_size {
                    let idx = base + k * after;
                    sum += (in_block[idx] - max_val).exp();
                }
                let log_sum = sum.ln();
                for k in 0..dim_size {
                    let idx = base + k * after;
                    let shifted = in_block[idx] - max_val;
                    out_block[idx] = if shifted <= cutoff {
                        f32::NEG_INFINITY
                    } else {
                        shifted - log_sum
                    };
                }
            }
        });

    Ok(())
}

fn log_softmax_f64(tensor: &Tensor, output_data: &mut TensorData, dim: usize) -> Result<()> {
    let input_data = tensor.data().as_f64_slice().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get f64 slice from input tensor")
    })?;

    let output_slice = output_data.as_f64_slice_mut().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get mutable f64 slice from output data")
    })?;

    let dims = tensor.shape().dims();
    let dim_size = dims[dim];

    let after: usize = if dim + 1 >= dims.len() {
        1
    } else {
        dims[dim + 1..].iter().product()
    };
    let group = dim_size * after;

    let cutoff = (f64::MIN_POSITIVE * f64::EPSILON).ln();

    input_data
        .par_chunks(group)
        .zip(output_slice.par_chunks_mut(group))
        .for_each(|(in_block, out_block)| {
            for a in 0..after {
                let base = a;
                let mut max_val = f64::NEG_INFINITY;
                for k in 0..dim_size {
                    let idx = base + k * after;
                    max_val = max_val.max(in_block[idx]);
                }
                let mut sum = 0.0f64;
                for k in 0..dim_size {
                    let idx = base + k * after;
                    sum += (in_block[idx] - max_val).exp();
                }
                let log_sum = sum.ln();
                for k in 0..dim_size {
                    let idx = base + k * after;
                    let shifted = in_block[idx] - max_val;
                    out_block[idx] = if shifted <= cutoff {
                        f64::NEG_INFINITY
                    } else {
                        shifted - log_sum
                    };
                }
            }
        });

    Ok(())
}

/// Absolute value function
pub fn abs(tensor: &Tensor) -> Result<Tensor> {
    let mut output_data =
        TensorData::uninitialized_on_device(tensor.numel(), tensor.dtype(), tensor.device());

    match tensor.dtype() {
        DataType::Float32 => abs_f32(tensor, &mut output_data)?,
        DataType::Float64 => abs_f64(tensor, &mut output_data)?,
        DataType::Int32 => abs_i32(tensor, &mut output_data)?,
        DataType::Int64 => abs_i64(tensor, &mut output_data)?,
        DataType::Bool => {
            return Err(MinitensorError::invalid_operation(
                "Absolute value not supported for boolean tensors",
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

    Ok(output)
}

/// Square root function
pub fn sqrt(tensor: &Tensor) -> Result<Tensor> {
    // Use powf implementation for gradient support: sqrt(x) = x.powf(0.5)
    powf(tensor, 0.5)
}

/// Reciprocal square root function
pub fn rsqrt(tensor: &Tensor) -> Result<Tensor> {
    // Use powf implementation for gradient support: rsqrt(x) = x.powf(-0.5)
    powf(tensor, -0.5)
}

/// Clip tensor values to range
pub fn clip(tensor: &Tensor, min_val: Option<f64>, max_val: Option<f64>) -> Result<Tensor> {
    let mut output_data =
        TensorData::uninitialized_on_device(tensor.numel(), tensor.dtype(), tensor.device());

    match tensor.dtype() {
        DataType::Float32 => clip_f32(tensor, &mut output_data, min_val, max_val)?,
        DataType::Float64 => clip_f64(tensor, &mut output_data, min_val, max_val)?,
        DataType::Int32 => clip_i32(tensor, &mut output_data, min_val, max_val)?,
        DataType::Int64 => clip_i64(tensor, &mut output_data, min_val, max_val)?,
        DataType::Bool => {
            return Err(MinitensorError::invalid_operation(
                "Clip not supported for boolean tensors",
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

    Ok(output)
}

/// Round tensor values
pub fn round(tensor: &Tensor, decimals: i32) -> Result<Tensor> {
    let mut output_data =
        TensorData::uninitialized_on_device(tensor.numel(), tensor.dtype(), tensor.device());

    match tensor.dtype() {
        DataType::Float32 => round_f32(tensor, &mut output_data, decimals)?,
        DataType::Float64 => round_f64(tensor, &mut output_data, decimals)?,
        _ => {
            return Err(MinitensorError::invalid_operation(
                "Round only supported for floating point tensors",
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

    Ok(output)
}

/// Floor tensor values
pub fn floor(tensor: &Tensor) -> Result<Tensor> {
    let mut output_data =
        TensorData::uninitialized_on_device(tensor.numel(), tensor.dtype(), tensor.device());

    match tensor.dtype() {
        DataType::Float32 => floor_f32(tensor, &mut output_data)?,
        DataType::Float64 => floor_f64(tensor, &mut output_data)?,
        _ => {
            return Err(MinitensorError::invalid_operation(
                "Floor only supported for floating point tensors",
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

    Ok(output)
}

/// Ceiling tensor values
pub fn ceil(tensor: &Tensor) -> Result<Tensor> {
    let mut output_data =
        TensorData::uninitialized_on_device(tensor.numel(), tensor.dtype(), tensor.device());

    match tensor.dtype() {
        DataType::Float32 => ceil_f32(tensor, &mut output_data)?,
        DataType::Float64 => ceil_f64(tensor, &mut output_data)?,
        _ => {
            return Err(MinitensorError::invalid_operation(
                "Ceiling only supported for floating point tensors",
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

    Ok(output)
}

// Helper functions for the new operations

fn abs_f32(tensor: &Tensor, output_data: &mut TensorData) -> Result<()> {
    let input_data = tensor.data().as_f32_slice().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get f32 slice from input tensor")
    })?;

    let output_slice = output_data.as_f32_slice_mut().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get mutable f32 slice from output data")
    })?;

    unary_apply(input_data, output_slice, |v: f32| v.abs());
    Ok(())
}

fn abs_f64(tensor: &Tensor, output_data: &mut TensorData) -> Result<()> {
    let input_data = tensor.data().as_f64_slice().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get f64 slice from input tensor")
    })?;

    let output_slice = output_data.as_f64_slice_mut().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get mutable f64 slice from output data")
    })?;

    unary_apply(input_data, output_slice, |v: f64| v.abs());
    Ok(())
}

fn abs_i32(tensor: &Tensor, output_data: &mut TensorData) -> Result<()> {
    let input_data = tensor.data().as_i32_slice().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get i32 slice from input tensor")
    })?;

    let output_slice = output_data.as_i32_slice_mut().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get mutable i32 slice from output data")
    })?;

    unary_apply(input_data, output_slice, |v: i32| v.abs());
    Ok(())
}

fn abs_i64(tensor: &Tensor, output_data: &mut TensorData) -> Result<()> {
    let input_data = tensor.data().as_i64_slice().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get i64 slice from input tensor")
    })?;

    let output_slice = output_data.as_i64_slice_mut().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get mutable i64 slice from output data")
    })?;

    unary_apply(input_data, output_slice, |v: i64| v.abs());
    Ok(())
}

fn clip_f32(
    tensor: &Tensor,
    output_data: &mut TensorData,
    min_val: Option<f64>,
    max_val: Option<f64>,
) -> Result<()> {
    let input_data = tensor.data().as_f32_slice().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get f32 slice from input tensor")
    })?;

    let output_slice = output_data.as_f32_slice_mut().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get mutable f32 slice from output data")
    })?;

    let min_f32 = min_val.map(|v| v as f32);
    let max_f32 = max_val.map(|v| v as f32);
    unary_apply(input_data, output_slice, |val: f32| {
        let mut v = val;
        if let Some(min) = min_f32 {
            v = v.max(min);
        }
        if let Some(max) = max_f32 {
            v = v.min(max);
        }
        v
    });
    Ok(())
}

fn clip_f64(
    tensor: &Tensor,
    output_data: &mut TensorData,
    min_val: Option<f64>,
    max_val: Option<f64>,
) -> Result<()> {
    let input_data = tensor.data().as_f64_slice().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get f64 slice from input tensor")
    })?;

    let output_slice = output_data.as_f64_slice_mut().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get mutable f64 slice from output data")
    })?;
    unary_apply(input_data, output_slice, |val: f64| {
        let mut v = val;
        if let Some(min) = min_val {
            v = v.max(min);
        }
        if let Some(max) = max_val {
            v = v.min(max);
        }
        v
    });
    Ok(())
}

fn clip_i32(
    tensor: &Tensor,
    output_data: &mut TensorData,
    min_val: Option<f64>,
    max_val: Option<f64>,
) -> Result<()> {
    let input_data = tensor.data().as_i32_slice().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get i32 slice from input tensor")
    })?;

    let output_slice = output_data.as_i32_slice_mut().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get mutable i32 slice from output data")
    })?;

    let min_i32 = min_val.map(|v| v as i32);
    let max_i32 = max_val.map(|v| v as i32);
    unary_apply(input_data, output_slice, |val: i32| {
        let mut v = val;
        if let Some(min) = min_i32 {
            v = v.max(min);
        }
        if let Some(max) = max_i32 {
            v = v.min(max);
        }
        v
    });
    Ok(())
}

fn clip_i64(
    tensor: &Tensor,
    output_data: &mut TensorData,
    min_val: Option<f64>,
    max_val: Option<f64>,
) -> Result<()> {
    let input_data = tensor.data().as_i64_slice().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get i64 slice from input tensor")
    })?;

    let output_slice = output_data.as_i64_slice_mut().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get mutable i64 slice from output data")
    })?;

    let min_i64 = min_val.map(|v| v as i64);
    let max_i64 = max_val.map(|v| v as i64);
    unary_apply(input_data, output_slice, |val: i64| {
        let mut v = val;
        if let Some(min) = min_i64 {
            v = v.max(min);
        }
        if let Some(max) = max_i64 {
            v = v.min(max);
        }
        v
    });
    Ok(())
}

fn round_f32(tensor: &Tensor, output_data: &mut TensorData, decimals: i32) -> Result<()> {
    let input_data = tensor.data().as_f32_slice().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get f32 slice from input tensor")
    })?;

    let output_slice = output_data.as_f32_slice_mut().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get mutable f32 slice from output data")
    })?;

    let multiplier = 10.0_f32.powi(decimals);
    unary_apply(input_data, output_slice, |val: f32| {
        (val * multiplier).round() / multiplier
    });
    Ok(())
}

fn round_f64(tensor: &Tensor, output_data: &mut TensorData, decimals: i32) -> Result<()> {
    let input_data = tensor.data().as_f64_slice().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get f64 slice from input tensor")
    })?;

    let output_slice = output_data.as_f64_slice_mut().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get mutable f64 slice from output data")
    })?;

    let multiplier = 10.0_f64.powi(decimals);
    unary_apply(input_data, output_slice, |val: f64| {
        (val * multiplier).round() / multiplier
    });
    Ok(())
}

fn floor_f32(tensor: &Tensor, output_data: &mut TensorData) -> Result<()> {
    let input_data = tensor.data().as_f32_slice().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get f32 slice from input tensor")
    })?;

    let output_slice = output_data.as_f32_slice_mut().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get mutable f32 slice from output data")
    })?;
    unary_apply(input_data, output_slice, f32::floor);
    Ok(())
}

fn floor_f64(tensor: &Tensor, output_data: &mut TensorData) -> Result<()> {
    let input_data = tensor.data().as_f64_slice().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get f64 slice from input tensor")
    })?;

    let output_slice = output_data.as_f64_slice_mut().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get mutable f64 slice from output data")
    })?;
    unary_apply(input_data, output_slice, f64::floor);
    Ok(())
}

fn ceil_f32(tensor: &Tensor, output_data: &mut TensorData) -> Result<()> {
    let input_data = tensor.data().as_f32_slice().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get f32 slice from input tensor")
    })?;

    let output_slice = output_data.as_f32_slice_mut().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get mutable f32 slice from output data")
    })?;
    unary_apply(input_data, output_slice, f32::ceil);
    Ok(())
}

fn ceil_f64(tensor: &Tensor, output_data: &mut TensorData) -> Result<()> {
    let input_data = tensor.data().as_f64_slice().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get f64 slice from input tensor")
    })?;

    let output_slice = output_data.as_f64_slice_mut().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get mutable f64 slice from output data")
    })?;
    unary_apply(input_data, output_slice, f64::ceil);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        autograd,
        device::Device,
        tensor::{Shape, Tensor, TensorData},
    };

    fn create_test_tensor_f32(data: Vec<f32>, shape: Vec<usize>, requires_grad: bool) -> Tensor {
        let shape_obj = Shape::new(shape);
        let mut tensor_data = TensorData::zeros(shape_obj.numel(), DataType::Float32);

        if let Some(slice) = tensor_data.as_f32_slice_mut() {
            slice.copy_from_slice(&data);
        }

        Tensor::new(
            Arc::new(tensor_data),
            shape_obj,
            DataType::Float32,
            Device::cpu(),
            requires_grad,
        )
    }

    #[test]
    fn test_exp() {
        let tensor = create_test_tensor_f32(vec![0.0, 1.0, 2.0], vec![3], false);
        let result = exp(&tensor).unwrap();
        let result_data = result.data().as_f32_slice().unwrap();

        assert!((result_data[0] - 1.0).abs() < 1e-6);
        assert!((result_data[1] - std::f32::consts::E).abs() < 1e-6);
        assert!((result_data[2] - (std::f32::consts::E * std::f32::consts::E)).abs() < 1e-5);
    }

    #[test]
    fn test_exp_invalid_dtype() {
        let shape = Shape::new(vec![3]);
        let data = TensorData::from_vec_i32(vec![1, 2, 3], Device::cpu());
        let tensor = Tensor::new(Arc::new(data), shape, DataType::Int32, Device::cpu(), false);
        assert!(exp(&tensor).is_err());
    }

    #[test]
    fn test_log() {
        let tensor = create_test_tensor_f32(
            vec![
                1.0,
                std::f32::consts::E,
                std::f32::consts::E * std::f32::consts::E,
            ],
            vec![3],
            false,
        );
        let result = log(&tensor).unwrap();
        let result_data = result.data().as_f32_slice().unwrap();

        assert!((result_data[0] - 0.0).abs() < 1e-6);
        assert!((result_data[1] - 1.0).abs() < 1e-6);
        assert!((result_data[2] - 2.0).abs() < 1e-5);
    }

    #[test]
    fn test_sin() {
        let tensor = create_test_tensor_f32(
            vec![0.0, std::f32::consts::PI / 2.0, std::f32::consts::PI],
            vec![3],
            false,
        );
        let result = sin(&tensor).unwrap();
        let result_data = result.data().as_f32_slice().unwrap();

        assert!((result_data[0] - 0.0).abs() < 1e-6);
        assert!((result_data[1] - 1.0).abs() < 1e-6);
        assert!(result_data[2].abs() < 1e-6); // sin(π) ≈ 0
    }

    #[test]
    fn test_cos() {
        let tensor = create_test_tensor_f32(
            vec![0.0, std::f32::consts::PI / 2.0, std::f32::consts::PI],
            vec![3],
            false,
        );
        let result = cos(&tensor).unwrap();
        let result_data = result.data().as_f32_slice().unwrap();

        assert!((result_data[0] - 1.0).abs() < 1e-6);
        assert!(result_data[1].abs() < 1e-6); // cos(π/2) ≈ 0
        assert!((result_data[2] + 1.0).abs() < 1e-6); // cos(π) ≈ -1
    }

    #[test]
    fn test_tan() {
        let tensor = create_test_tensor_f32(
            vec![0.0, std::f32::consts::PI / 4.0, -std::f32::consts::PI / 4.0],
            vec![3],
            false,
        );
        let result = tan(&tensor).unwrap();
        let result_data = result.data().as_f32_slice().unwrap();

        assert!((result_data[0] - 0.0).abs() < 1e-6);
        assert!((result_data[1] - (std::f32::consts::PI / 4.0).tan()).abs() < 1e-6);
        assert!((result_data[2] - (-std::f32::consts::PI / 4.0).tan()).abs() < 1e-6);
    }

    #[test]
    fn test_tanh() {
        let tensor = create_test_tensor_f32(vec![0.0, 1.0, -1.0], vec![3], false);
        let result = tanh(&tensor).unwrap();
        let result_data = result.data().as_f32_slice().unwrap();

        assert!((result_data[0] - 0.0).abs() < 1e-6);
        assert!((result_data[1] - 1.0_f32.tanh()).abs() < 1e-6);
        assert!((result_data[2] - (-1.0_f32).tanh()).abs() < 1e-6);
    }

    #[test]
    fn test_sigmoid() {
        let tensor = create_test_tensor_f32(vec![0.0, 1.0, -1.0], vec![3], false);
        let result = sigmoid(&tensor).unwrap();
        let result_data = result.data().as_f32_slice().unwrap();

        assert!((result_data[0] - 0.5).abs() < 1e-6);
        assert!((result_data[1] - (1.0 / (1.0 + (-1.0_f32).exp()))).abs() < 1e-6);
        assert!((result_data[2] - (1.0 / (1.0 + 1.0_f32.exp()))).abs() < 1e-6);
    }

    #[test]
    fn test_relu() {
        let tensor = create_test_tensor_f32(vec![-2.0, -1.0, 0.0, 1.0, 2.0], vec![5], false);
        let result = relu(&tensor).unwrap();
        let result_data = result.data().as_f32_slice().unwrap();

        assert_eq!(result_data, &[0.0, 0.0, 0.0, 1.0, 2.0]);
    }

    #[test]
    fn test_hardshrink_forward() {
        let tensor = create_test_tensor_f32(vec![-1.2, -0.2, 0.0, 0.45, 0.9], vec![5], false);
        let result = hardshrink(&tensor, 0.3).unwrap();
        let data = result.data().as_f32_slice().unwrap();
        assert_eq!(data, &[-1.2, 0.0, 0.0, 0.45, 0.9]);
    }

    #[test]
    fn test_hardshrink_backward() {
        let tensor = create_test_tensor_f32(vec![-1.2, -0.25, 0.0, 0.35, 0.8], vec![5], true);
        let result = hardshrink(&tensor, 0.3).unwrap();
        let ones = Tensor::ones(
            result.shape().clone(),
            result.dtype(),
            result.device(),
            false,
        );
        let grads = autograd::backward(&result, Some(ones)).unwrap();
        let grad = grads.get(&tensor.id()).unwrap();
        let grad_vals = grad.data().as_f32_slice().unwrap();
        assert_eq!(grad_vals, &[1.0, 0.0, 0.0, 1.0, 1.0]);
    }

    #[test]
    fn test_hardshrink_invalid_lambda() {
        let tensor = create_test_tensor_f32(vec![1.0], vec![1], false);
        assert!(hardshrink(&tensor, -0.1).is_err());
    }

    #[test]
    fn test_leaky_relu() {
        let tensor = create_test_tensor_f32(vec![-2.0, -1.0, 0.0, 1.0, 2.0], vec![5], false);
        let result = leaky_relu(&tensor, 0.1).unwrap();
        let result_data = result.data().as_f32_slice().unwrap();

        assert_eq!(result_data, &[-0.2, -0.1, 0.0, 1.0, 2.0]);
    }

    #[test]
    fn test_softmax() {
        let tensor = create_test_tensor_f32(vec![1.0, 2.0, 3.0], vec![3], false);
        let result = softmax(&tensor, None).unwrap();
        let result_data = result.data().as_f32_slice().unwrap();

        // Check that probabilities sum to 1
        let sum: f32 = result_data.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);

        // Check that all values are positive
        for &val in result_data {
            assert!(val > 0.0);
        }

        // Check that larger input values produce larger probabilities
        assert!(result_data[2] > result_data[1]);
        assert!(result_data[1] > result_data[0]);
    }

    #[test]
    fn test_powf_scalar() {
        let tensor = create_test_tensor_f32(vec![1.0, 2.0, 3.0], vec![3], false);
        let result = powf(&tensor, 2.0).unwrap();
        let data = result.data().as_f32_slice().unwrap();
        assert_eq!(data, &[1.0, 4.0, 9.0]);
    }

    #[test]
    fn test_pow_tensor() {
        let base = create_test_tensor_f32(vec![2.0, 3.0, 4.0], vec![3], false);
        let exp = create_test_tensor_f32(vec![1.0, 2.0, 0.5], vec![3], false);
        let result = pow(&base, &exp).unwrap();
        let data = result.data().as_f32_slice().unwrap();
        assert!((data[0] - 2.0).abs() < 1e-6);
        assert!((data[1] - 9.0).abs() < 1e-6);
        assert!((data[2] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_pow_shape_mismatch_error() {
        let base = create_test_tensor_f32(vec![1.0, 2.0], vec![2], false);
        let exp = create_test_tensor_f32(vec![3.0, 4.0, 5.0], vec![3], false);
        assert!(pow(&base, &exp).is_err());
    }

    #[test]
    fn test_pow_dtype_mismatch_error() {
        let base = create_test_tensor_f32(vec![1.0, 2.0], vec![2], false);
        let shape = Shape::new(vec![2]);
        let data = TensorData::from_vec_f64(vec![1.0, 2.0], Device::cpu());
        let exp = Tensor::new(
            Arc::new(data),
            shape,
            DataType::Float64,
            Device::cpu(),
            false,
        );
        assert!(pow(&base, &exp).is_err());
    }

    #[test]
    fn test_pow_device_mismatch_error() {
        let base = create_test_tensor_f32(vec![1.0, 2.0], vec![2], false);
        let shape = Shape::new(vec![2]);
        let data = TensorData::from_vec_f32(vec![1.0, 2.0], Device::cuda(Some(0)));
        let exp = Tensor::new(
            Arc::new(data),
            shape,
            DataType::Float32,
            Device::cuda(Some(0)),
            false,
        );
        assert!(pow(&base, &exp).is_err());
    }

    #[test]
    fn test_powf_gradient() {
        let tensor = create_test_tensor_f32(vec![2.0, 3.0], vec![2], true);
        let result = powf(&tensor, 3.0).unwrap();
        let ones = Tensor::ones(
            result.shape().clone(),
            result.dtype(),
            result.device(),
            false,
        );
        let grads = autograd::backward(&result, Some(ones)).unwrap();
        let grad = grads.get(&tensor.id()).unwrap();
        let g = grad.data().as_f32_slice().unwrap();
        assert!((g[0] - 3.0 * 2.0_f32.powf(2.0)).abs() < 1e-6);
        assert!((g[1] - 3.0 * 3.0_f32.powf(2.0)).abs() < 1e-6);
    }

    #[test]
    fn test_pow_base_scalar_tensor_exponent() {
        let base = create_test_tensor_f32(vec![2.0], vec![], false);
        let exp = create_test_tensor_f32(vec![1.0, 2.0, 3.0], vec![3], false);
        let result = pow(&base, &exp).unwrap();
        let data = result.data().as_f32_slice().unwrap();
        assert!((data[0] - 2.0).abs() < 1e-6);
        assert!((data[1] - 4.0).abs() < 1e-6);
        assert!((data[2] - 8.0).abs() < 1e-6);
    }

    #[test]
    fn test_pow_exponent_scalar_tensor_base() {
        let base = create_test_tensor_f32(vec![2.0, 3.0, 4.0], vec![3], false);
        let exp = create_test_tensor_f32(vec![2.0], vec![1], false);
        let result = pow(&base, &exp).unwrap();
        let data = result.data().as_f32_slice().unwrap();
        assert!((data[0] - 4.0).abs() < 1e-6);
        assert!((data[1] - 9.0).abs() < 1e-6);
        assert!((data[2] - 16.0).abs() < 1e-6);
    }

    #[test]
    fn test_pow_base_scalar_gradient() {
        let base = create_test_tensor_f32(vec![2.0], vec![], true);
        let exp = create_test_tensor_f32(vec![1.0, 2.0], vec![2], false);
        let result = pow(&base, &exp).unwrap();
        let ones = Tensor::ones(
            result.shape().clone(),
            result.dtype(),
            result.device(),
            false,
        );
        let grads = autograd::backward(&result, Some(ones)).unwrap();
        let grad = grads.get(&base.id()).unwrap();
        let g = grad.data().as_f32_slice().unwrap();
        let base_val = base.data().as_f32_slice().unwrap()[0];
        let exp_vals = exp.data().as_f32_slice().unwrap();
        let expected = exp_vals
            .iter()
            .map(|&e| e * base_val.powf(e - 1.0))
            .sum::<f32>();
        assert!((g[0] - expected).abs() < 1e-6);
    }

    #[test]
    fn test_pow_exponent_scalar_gradient() {
        let base = create_test_tensor_f32(vec![2.0, 3.0], vec![2], false);
        let exp = create_test_tensor_f32(vec![1.5], vec![1], true);
        let result = pow(&base, &exp).unwrap();
        let ones = Tensor::ones(
            result.shape().clone(),
            result.dtype(),
            result.device(),
            false,
        );
        let grads = autograd::backward(&result, Some(ones)).unwrap();
        let grad = grads.get(&exp.id()).unwrap();
        let g = grad.data().as_f32_slice().unwrap();
        let exp_val = exp.data().as_f32_slice().unwrap()[0];
        let base_vals = base.data().as_f32_slice().unwrap();
        let expected = base_vals
            .iter()
            .map(|&b| b.powf(exp_val) * b.ln())
            .sum::<f32>();
        assert!((g[0] - expected).abs() < 1e-6);
    }

    #[test]
    fn test_sqrt() {
        let tensor = create_test_tensor_f32(vec![1.0, 4.0, 9.0], vec![3], false);
        let result = sqrt(&tensor).unwrap();
        let data = result.data().as_f32_slice().unwrap();
        assert_eq!(data, &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_sqrt_gradient() {
        let tensor = create_test_tensor_f32(vec![4.0, 9.0], vec![2], true);
        let result = sqrt(&tensor).unwrap();
        let ones = Tensor::ones(
            result.shape().clone(),
            result.dtype(),
            result.device(),
            false,
        );
        let grads = autograd::backward(&result, Some(ones)).unwrap();
        let grad = grads.get(&tensor.id()).unwrap();
        let g = grad.data().as_f32_slice().unwrap();
        assert!((g[0] - 0.25).abs() < 1e-6);
        assert!((g[1] - (1.0 / 6.0)).abs() < 1e-6);
    }

    #[test]
    fn test_rsqrt() {
        let tensor = create_test_tensor_f32(vec![0.25, 1.0, 4.0], vec![3], false);
        let result = rsqrt(&tensor).unwrap();
        let data = result.data().as_f32_slice().unwrap();
        assert!((data[0] - 2.0).abs() < 1e-6);
        assert!((data[1] - 1.0).abs() < 1e-6);
        assert!((data[2] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_rsqrt_gradient() {
        let tensor = create_test_tensor_f32(vec![0.25, 4.0], vec![2], true);
        let result = rsqrt(&tensor).unwrap();
        let ones = Tensor::ones(
            result.shape().clone(),
            result.dtype(),
            result.device(),
            false,
        );
        let grads = autograd::backward(&result, Some(ones)).unwrap();
        let grad = grads.get(&tensor.id()).unwrap();
        let g = grad.data().as_f32_slice().unwrap();
        assert!((g[0] - (-0.5 * 0.25_f32.powf(-1.5))).abs() < 1e-5);
        assert!((g[1] - (-0.5 * 4.0_f32.powf(-1.5))).abs() < 1e-6);
    }

    #[test]
    fn test_softsign_forward() {
        let data = vec![-2.5f32, -0.5, 0.0, 0.25, 4.0];
        let tensor = create_test_tensor_f32(data.clone(), vec![5], false);

        let result = softsign(&tensor).unwrap();
        let values = result.data().as_f32_slice().unwrap();

        for (out, &x) in values.iter().zip(data.iter()) {
            let denom = 1.0 + x.abs();
            let expected = x / denom;
            assert!((out - expected).abs() < 1e-6);
        }
    }

    #[test]
    fn test_softsign_gradient() {
        let data = vec![-1.5f32, -0.25, 0.5, 3.0];
        let tensor = create_test_tensor_f32(data.clone(), vec![4], true);

        let result = softsign(&tensor).unwrap();
        let ones = Tensor::ones(
            result.shape().clone(),
            result.dtype(),
            result.device(),
            false,
        );
        let grads = autograd::backward(&result, Some(ones)).unwrap();
        let grad_tensor = grads.get(&tensor.id()).unwrap();
        let grad_data = grad_tensor.data().as_f32_slice().unwrap();

        for ((&grad, &x), idx) in grad_data.iter().zip(data.iter()).zip(0..) {
            let denom = 1.0 + x.abs();
            let expected = 1.0 / (denom * denom);
            assert!(
                (grad - expected).abs() < 1e-5,
                "gradient mismatch at index {idx}: got {grad}, expected {expected}"
            );
        }
    }

    #[test]
    fn test_gradient_tracking() {
        let tensor = create_test_tensor_f32(vec![1.0, 2.0, 3.0], vec![3], true);

        let result = relu(&tensor).unwrap();
        assert!(result.requires_grad());
        assert!(result.grad_fn().is_some());

        let result2 = sigmoid(&tensor).unwrap();
        assert!(result2.requires_grad());
        assert!(result2.grad_fn().is_some());
    }
}
