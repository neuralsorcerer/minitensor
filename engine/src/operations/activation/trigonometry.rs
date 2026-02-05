// Copyright (c) 2026 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

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
