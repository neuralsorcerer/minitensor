// Copyright (c) 2026 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

use crate::{
    autograd::{
        AcosBackward, AcoshBackward, AsinBackward, AsinhBackward, AtanBackward, AtanhBackward,
        CosBackward, CoshBackward, EluBackward, ExpBackward, Expm1Backward, GeluBackward,
        HardshrinkBackward, LeakyReluBackward, Log1pBackward, LogAddExpBackward, LogBackward,
        LogSoftmaxBackward, MaskedLogSoftmaxBackward, PowBackward, PowBroadcast, ReluBackward,
        SeluBackward, SigmoidBackward, SiluBackward, SinBackward, SinhBackward, SoftmaxBackward,
        SoftplusBackward, SoftsignBackward, TanBackward, TanhBackward, add_to_graph,
    },
    error::{MinitensorError, Result},
    tensor::{DataType, Shape, Strides, Tensor, TensorData},
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

/// Inverse sine function with gradient support
pub fn asin(tensor: &Tensor) -> Result<Tensor> {
    let mut output_data =
        TensorData::uninitialized_on_device(tensor.numel(), tensor.dtype(), tensor.device());

    match tensor.dtype() {
        DataType::Float32 => asin_f32(tensor, &mut output_data)?,
        DataType::Float64 => asin_f64(tensor, &mut output_data)?,
        _ => {
            return Err(MinitensorError::invalid_operation(
                "Inverse sine only supported for floating point tensors",
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
        let grad_fn = Arc::new(AsinBackward {
            input_id: tensor.id(),
            input: tensor.clone(),
        });

        let mut output_with_grad = output;
        output_with_grad.set_grad_fn(Some(grad_fn.clone()));
        add_to_graph(&output_with_grad, Some(grad_fn))?;
        Ok(output_with_grad)
    } else {
        Ok(output)
    }
}

/// Inverse cosine function with gradient support
pub fn acos(tensor: &Tensor) -> Result<Tensor> {
    let mut output_data =
        TensorData::uninitialized_on_device(tensor.numel(), tensor.dtype(), tensor.device());

    match tensor.dtype() {
        DataType::Float32 => acos_f32(tensor, &mut output_data)?,
        DataType::Float64 => acos_f64(tensor, &mut output_data)?,
        _ => {
            return Err(MinitensorError::invalid_operation(
                "Inverse cosine only supported for floating point tensors",
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
        let grad_fn = Arc::new(AcosBackward {
            input_id: tensor.id(),
            input: tensor.clone(),
        });

        let mut output_with_grad = output;
        output_with_grad.set_grad_fn(Some(grad_fn.clone()));
        add_to_graph(&output_with_grad, Some(grad_fn))?;
        Ok(output_with_grad)
    } else {
        Ok(output)
    }
}

/// Inverse tangent function with gradient support
pub fn atan(tensor: &Tensor) -> Result<Tensor> {
    let mut output_data =
        TensorData::uninitialized_on_device(tensor.numel(), tensor.dtype(), tensor.device());

    match tensor.dtype() {
        DataType::Float32 => atan_f32(tensor, &mut output_data)?,
        DataType::Float64 => atan_f64(tensor, &mut output_data)?,
        _ => {
            return Err(MinitensorError::invalid_operation(
                "Inverse tangent only supported for floating point tensors",
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
        let grad_fn = Arc::new(AtanBackward {
            input_id: tensor.id(),
            input: tensor.clone(),
        });

        let mut output_with_grad = output;
        output_with_grad.set_grad_fn(Some(grad_fn.clone()));
        add_to_graph(&output_with_grad, Some(grad_fn))?;
        Ok(output_with_grad)
    } else {
        Ok(output)
    }
}

/// Hyperbolic sine with gradient support
pub fn sinh(tensor: &Tensor) -> Result<Tensor> {
    let mut output_data =
        TensorData::uninitialized_on_device(tensor.numel(), tensor.dtype(), tensor.device());

    match tensor.dtype() {
        DataType::Float32 => sinh_f32(tensor, &mut output_data)?,
        DataType::Float64 => sinh_f64(tensor, &mut output_data)?,
        _ => {
            return Err(MinitensorError::invalid_operation(
                "sinh is only supported for floating point tensors",
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
        let grad_fn = Arc::new(SinhBackward {
            input_id: tensor.id(),
            input: tensor.clone(),
        });

        let mut output_with_grad = output;
        output_with_grad.set_grad_fn(Some(grad_fn.clone()));
        add_to_graph(&output_with_grad, Some(grad_fn))?;
        Ok(output_with_grad)
    } else {
        Ok(output)
    }
}

/// Hyperbolic cosine with gradient support
pub fn cosh(tensor: &Tensor) -> Result<Tensor> {
    let mut output_data =
        TensorData::uninitialized_on_device(tensor.numel(), tensor.dtype(), tensor.device());

    match tensor.dtype() {
        DataType::Float32 => cosh_f32(tensor, &mut output_data)?,
        DataType::Float64 => cosh_f64(tensor, &mut output_data)?,
        _ => {
            return Err(MinitensorError::invalid_operation(
                "cosh is only supported for floating point tensors",
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
        let grad_fn = Arc::new(CoshBackward {
            input_id: tensor.id(),
            input: tensor.clone(),
        });

        let mut output_with_grad = output;
        output_with_grad.set_grad_fn(Some(grad_fn.clone()));
        add_to_graph(&output_with_grad, Some(grad_fn))?;
        Ok(output_with_grad)
    } else {
        Ok(output)
    }
}

/// Inverse hyperbolic sine with gradient support
pub fn asinh(tensor: &Tensor) -> Result<Tensor> {
    let mut output_data =
        TensorData::uninitialized_on_device(tensor.numel(), tensor.dtype(), tensor.device());

    match tensor.dtype() {
        DataType::Float32 => asinh_f32(tensor, &mut output_data)?,
        DataType::Float64 => asinh_f64(tensor, &mut output_data)?,
        _ => {
            return Err(MinitensorError::invalid_operation(
                "asinh is only supported for floating point tensors",
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
        let grad_fn = Arc::new(AsinhBackward {
            input_id: tensor.id(),
            input: tensor.clone(),
        });

        let mut output_with_grad = output;
        output_with_grad.set_grad_fn(Some(grad_fn.clone()));
        add_to_graph(&output_with_grad, Some(grad_fn))?;
        Ok(output_with_grad)
    } else {
        Ok(output)
    }
}

/// Inverse hyperbolic cosine with gradient support
pub fn acosh(tensor: &Tensor) -> Result<Tensor> {
    let mut output_data =
        TensorData::uninitialized_on_device(tensor.numel(), tensor.dtype(), tensor.device());

    match tensor.dtype() {
        DataType::Float32 => acosh_f32(tensor, &mut output_data)?,
        DataType::Float64 => acosh_f64(tensor, &mut output_data)?,
        _ => {
            return Err(MinitensorError::invalid_operation(
                "acosh is only supported for floating point tensors",
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
        let grad_fn = Arc::new(AcoshBackward {
            input_id: tensor.id(),
            input: tensor.clone(),
        });

        let mut output_with_grad = output;
        output_with_grad.set_grad_fn(Some(grad_fn.clone()));
        add_to_graph(&output_with_grad, Some(grad_fn))?;
        Ok(output_with_grad)
    } else {
        Ok(output)
    }
}

/// Inverse hyperbolic tangent with gradient support
pub fn atanh(tensor: &Tensor) -> Result<Tensor> {
    let mut output_data =
        TensorData::uninitialized_on_device(tensor.numel(), tensor.dtype(), tensor.device());

    match tensor.dtype() {
        DataType::Float32 => atanh_f32(tensor, &mut output_data)?,
        DataType::Float64 => atanh_f64(tensor, &mut output_data)?,
        _ => {
            return Err(MinitensorError::invalid_operation(
                "atanh is only supported for floating point tensors",
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
        let grad_fn = Arc::new(AtanhBackward {
            input_id: tensor.id(),
            input: tensor.clone(),
        });

        let mut output_with_grad = output;
        output_with_grad.set_grad_fn(Some(grad_fn.clone()));
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
