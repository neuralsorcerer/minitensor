// Copyright (c) Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

use super::*;

use crate::{
    autograd::{
        AcosBackward, AcoshBackward, AsinBackward, AsinhBackward, AtanBackward, AtanhBackward,
        CosBackward, CoshBackward, ErfBackward, ExpBackward, Expm1Backward, Log1pBackward,
        LogBackward, LogBaseBackward, SigmoidBackward, SinBackward, SinhBackward, TanBackward,
        TanhBackward, add_to_graph,
    },
    error::{MinitensorError, Result},
    tensor::{DataType, Tensor, TensorData},
};
use std::sync::Arc;

// The map primitives live in `ops::map`; re-exported here so every file in the
// activation cluster picks them up through `use super::*`. Transcendentals use
// `unary_map_threshold` with `EXPENSIVE_PAR_THRESHOLD` rather than the default
// `unary_map`: their per-element cost repays the fixed parallel-region entry
// cost far sooner than a `relu`'s does.
pub(crate) use crate::ops::map::{
    EXPENSIVE_PAR_THRESHOLD, TANH_F32_PAR_THRESHOLD, unary_map, unary_map_blocks_threshold,
    unary_map_threshold,
};

/// Exponential function with gradient support
pub fn exp(tensor: &Tensor) -> Result<Tensor> {
    // Create output tensor data
    let output_data = match tensor.dtype() {
        DataType::Float32 => exp_f32(tensor)?,
        DataType::Float64 => exp_f64(tensor)?,
        _ => {
            return Err(MinitensorError::invalid_operation(
                "Exponential function only supported for floating point tensors",
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
    let output_data = match tensor.dtype() {
        DataType::Float32 => log_f32(tensor)?,
        DataType::Float64 => log_f64(tensor)?,
        _ => {
            return Err(MinitensorError::invalid_operation(
                "Logarithm function only supported for floating point tensors",
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
    let output_data = match tensor.dtype() {
        DataType::Float32 => log1p_f32(tensor)?,
        DataType::Float64 => log1p_f64(tensor)?,
        _ => {
            return Err(MinitensorError::invalid_operation(
                "log1p is only supported for floating point tensors",
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

/// Base-2 logarithm with gradient support
pub fn log2(tensor: &Tensor) -> Result<Tensor> {
    log_base(tensor, log2_f32, log2_f64, std::f64::consts::LN_2, "log2")
}

/// Base-10 logarithm with gradient support
pub fn log10(tensor: &Tensor) -> Result<Tensor> {
    log_base(
        tensor,
        log10_f32,
        log10_f64,
        std::f64::consts::LN_10,
        "log10",
    )
}

/// Shared body for the fixed-base logarithms. They differ only in which kernel
/// runs and in the `ln(base)` their derivative divides by, so both are passed in
/// rather than re-derived from the constant.
fn log_base(
    tensor: &Tensor,
    kernel_f32: fn(&Tensor) -> Result<TensorData>,
    kernel_f64: fn(&Tensor) -> Result<TensorData>,
    ln_base: f64,
    name: &str,
) -> Result<Tensor> {
    let output_data = match tensor.dtype() {
        DataType::Float32 => kernel_f32(tensor)?,
        DataType::Float64 => kernel_f64(tensor)?,
        _ => {
            return Err(MinitensorError::invalid_operation(format!(
                "{name} is only supported for floating point tensors"
            )));
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
        let grad_fn = Arc::new(LogBaseBackward {
            input_id: tensor.id(),
            input: tensor.clone().detach(),
            ln_base,
        });
        let mut output_with_grad = output;
        output_with_grad.set_grad_fn(Some(grad_fn.clone()));
        add_to_graph(&output_with_grad, Some(grad_fn))?;
        Ok(output_with_grad)
    } else {
        Ok(output)
    }
}

/// Gauss error function with gradient support
pub fn erf(tensor: &Tensor) -> Result<Tensor> {
    erf_family(tensor, false)
}

/// Complementary error function, `1 - erf(x)`, with gradient support.
///
/// Computed by a dedicated routine rather than as `1 - erf(x)`: once `erf(x)`
/// rounds to 1 that subtraction returns exactly 0 and every significant digit
/// of the tail is lost, which is precisely the regime `erfc` exists to serve.
pub fn erfc(tensor: &Tensor) -> Result<Tensor> {
    erf_family(tensor, true)
}

fn erf_family(tensor: &Tensor, complementary: bool) -> Result<Tensor> {
    let output_data = match (tensor.dtype(), complementary) {
        (DataType::Float32, false) => erf_f32(tensor)?,
        (DataType::Float64, false) => erf_f64(tensor)?,
        (DataType::Float32, true) => erfc_f32(tensor)?,
        (DataType::Float64, true) => erfc_f64(tensor)?,
        (_, _) => {
            return Err(MinitensorError::invalid_operation(
                "erf is only supported for floating point tensors",
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
        // 2/sqrt(pi), negated for the complementary form.
        let magnitude = 2.0 / std::f64::consts::PI.sqrt();
        let grad_fn = Arc::new(ErfBackward {
            input_id: tensor.id(),
            input: tensor.clone().detach(),
            scale: if complementary { -magnitude } else { magnitude },
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
    let output_data = match tensor.dtype() {
        DataType::Float32 => expm1_f32(tensor)?,
        DataType::Float64 => expm1_f64(tensor)?,
        _ => {
            return Err(MinitensorError::invalid_operation(
                "expm1 is only supported for floating point tensors",
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
    let output_data = match tensor.dtype() {
        DataType::Float32 => sin_f32(tensor)?,
        DataType::Float64 => sin_f64(tensor)?,
        _ => {
            return Err(MinitensorError::invalid_operation(
                "Sine function only supported for floating point tensors",
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
    let output_data = match tensor.dtype() {
        DataType::Float32 => cos_f32(tensor)?,
        DataType::Float64 => cos_f64(tensor)?,
        _ => {
            return Err(MinitensorError::invalid_operation(
                "Cosine function only supported for floating point tensors",
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
    let output_data = match tensor.dtype() {
        DataType::Float32 => tan_f32(tensor)?,
        DataType::Float64 => tan_f64(tensor)?,
        _ => {
            return Err(MinitensorError::invalid_operation(
                "Tangent function only supported for floating point tensors",
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
    let output_data = match tensor.dtype() {
        DataType::Float32 => asin_f32(tensor)?,
        DataType::Float64 => asin_f64(tensor)?,
        _ => {
            return Err(MinitensorError::invalid_operation(
                "Inverse sine only supported for floating point tensors",
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
    let output_data = match tensor.dtype() {
        DataType::Float32 => acos_f32(tensor)?,
        DataType::Float64 => acos_f64(tensor)?,
        _ => {
            return Err(MinitensorError::invalid_operation(
                "Inverse cosine only supported for floating point tensors",
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
    let output_data = match tensor.dtype() {
        DataType::Float32 => atan_f32(tensor)?,
        DataType::Float64 => atan_f64(tensor)?,
        _ => {
            return Err(MinitensorError::invalid_operation(
                "Inverse tangent only supported for floating point tensors",
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
    let output_data = match tensor.dtype() {
        DataType::Float32 => sinh_f32(tensor)?,
        DataType::Float64 => sinh_f64(tensor)?,
        _ => {
            return Err(MinitensorError::invalid_operation(
                "sinh is only supported for floating point tensors",
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
    let output_data = match tensor.dtype() {
        DataType::Float32 => cosh_f32(tensor)?,
        DataType::Float64 => cosh_f64(tensor)?,
        _ => {
            return Err(MinitensorError::invalid_operation(
                "cosh is only supported for floating point tensors",
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
    let output_data = match tensor.dtype() {
        DataType::Float32 => asinh_f32(tensor)?,
        DataType::Float64 => asinh_f64(tensor)?,
        _ => {
            return Err(MinitensorError::invalid_operation(
                "asinh is only supported for floating point tensors",
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
    let output_data = match tensor.dtype() {
        DataType::Float32 => acosh_f32(tensor)?,
        DataType::Float64 => acosh_f64(tensor)?,
        _ => {
            return Err(MinitensorError::invalid_operation(
                "acosh is only supported for floating point tensors",
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
    let output_data = match tensor.dtype() {
        DataType::Float32 => atanh_f32(tensor)?,
        DataType::Float64 => atanh_f64(tensor)?,
        _ => {
            return Err(MinitensorError::invalid_operation(
                "atanh is only supported for floating point tensors",
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
    let output_data = match tensor.dtype() {
        DataType::Float32 => tanh_f32(tensor)?,
        DataType::Float64 => tanh_f64(tensor)?,
        _ => {
            return Err(MinitensorError::invalid_operation(
                "Tanh function only supported for floating point tensors",
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
    let output_data = match tensor.dtype() {
        DataType::Float32 => sigmoid_f32(tensor)?,
        DataType::Float64 => sigmoid_f64(tensor)?,
        _ => {
            return Err(MinitensorError::invalid_operation(
                "Sigmoid function only supported for floating point tensors",
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

/// Gated Linear Unit (Dauphin et al., 2017). Splits `input` into two equal
/// halves `(a, b)` along `dim` and returns `a * sigmoid(b)`. This gating is the
/// basis of the GLU-family feed-forward blocks (GEGLU, SwiGLU) used throughout
/// modern Transformers. Built from autograd-tracked slice / sigmoid / multiply,
/// so the gradient is exact. The split dimension must have even length.
pub fn glu(input: &Tensor, dim: isize) -> Result<Tensor> {
    if !matches!(input.dtype(), DataType::Float32 | DataType::Float64) {
        return Err(MinitensorError::invalid_operation(
            "glu only supports floating point tensors",
        ));
    }
    let ndim = input.ndim() as isize;
    let axis = if dim < 0 { dim + ndim } else { dim };
    if axis < 0 || axis >= ndim {
        return Err(MinitensorError::invalid_argument(format!(
            "glu dim {} is out of range for a {}-dimensional tensor",
            dim, ndim
        )));
    }
    let size = input.shape().dims()[axis as usize];
    if !size.is_multiple_of(2) {
        return Err(MinitensorError::invalid_argument(format!(
            "glu requires an even split dimension, but dim {} has length {}",
            dim, size
        )));
    }
    let half = size / 2;
    let a = crate::ops::shape_ops::narrow(input, axis, 0, half)?;
    let b = crate::ops::shape_ops::narrow(input, axis, half, half)?;
    let gate = sigmoid(&b)?;
    crate::ops::arithmetic::mul(&a, &gate)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::Device;
    use crate::tensor::{Shape, TensorData};

    type TensorOp = fn(&Tensor) -> Result<Tensor>;

    fn f64_tensor(data: Vec<f64>, requires_grad: bool) -> Tensor {
        let len = data.len();
        Tensor::new(
            Arc::new(TensorData::from_vec_f64(data, Device::cpu())),
            Shape::new(vec![len]),
            DataType::Float64,
            Device::cpu(),
            requires_grad,
        )
    }

    #[test]
    fn test_log2_and_log10_match_the_natural_log_rescaled() {
        let values = vec![0.05, 0.5, 1.0, 2.0, 8.0, 1e6];
        let t = f64_tensor(values.clone(), false);

        for (got, &v) in log2(&t)
            .unwrap()
            .data()
            .as_f64_slice()
            .unwrap()
            .iter()
            .zip(values.iter())
        {
            assert!((got - v.log2()).abs() <= 1e-12 * v.log2().abs().max(1.0));
        }
        for (got, &v) in log10(&t)
            .unwrap()
            .data()
            .as_f64_slice()
            .unwrap()
            .iter()
            .zip(values.iter())
        {
            assert!((got - v.log10()).abs() <= 1e-12 * v.log10().abs().max(1.0));
        }
    }

    #[test]
    fn test_erfc_keeps_the_tail_that_one_minus_erf_loses() {
        // Beyond about x = 6 in f64, erf(x) rounds to 1 and `1 - erf(x)` is
        // exactly zero. The dedicated routine is the difference between an
        // answer and none.
        let t = f64_tensor(vec![6.0, 10.0, 20.0], false);
        let erf_values = erf(&t).unwrap();
        let erfc_values = erfc(&t).unwrap();

        for (&e, &c) in erf_values
            .data()
            .as_f64_slice()
            .unwrap()
            .iter()
            .zip(erfc_values.data().as_f64_slice().unwrap())
        {
            assert_eq!(1.0 - e, 0.0, "1 - erf no longer cancels; revisit this");
            assert!(c > 0.0, "erfc lost the tail: {c}");
        }
    }

    #[test]
    fn test_erf_saturates_and_propagates_nan() {
        let t = f64_tensor(vec![f64::NEG_INFINITY, 0.0, f64::INFINITY, f64::NAN], false);
        let e = erf(&t).unwrap();
        let e = e.data().as_f64_slice().unwrap();
        assert_eq!(&e[..3], &[-1.0, 0.0, 1.0]);
        assert!(e[3].is_nan());

        let c = erfc(&t).unwrap();
        let c = c.data().as_f64_slice().unwrap();
        assert_eq!(&c[..3], &[2.0, 1.0, 0.0]);
        assert!(c[3].is_nan());
    }

    #[test]
    fn test_gradients_match_central_differences() {
        /// name, the op under test, a scalar reference for it, sample points
        struct Case(&'static str, TensorOp, fn(f64) -> f64, &'static [f64]);

        const LOGS: &[f64] = &[0.3, 1.0, 2.5, 4.0];
        const ERFS: &[f64] = &[-2.0, -0.5, 0.0, 1.5];
        let cases = [
            Case("log2", log2, |v: f64| v.log2(), LOGS),
            Case("log10", log10, |v: f64| v.log10(), LOGS),
            Case("erf", erf, libm::erf, ERFS),
            Case("erfc", erfc, libm::erfc, ERFS),
        ];

        for Case(name, op, reference, values) in cases {
            let t = f64_tensor(values.to_vec(), true);
            let out = op(&t).unwrap();
            let seed = f64_tensor(vec![1.0; values.len()], false);
            let grads = crate::autograd::backward_collect(&out, Some(seed)).unwrap();
            let analytic = grads[&t.id()].data().as_f64_slice().unwrap();

            let h = 1e-6;
            for (i, &v) in values.iter().enumerate() {
                let central = (reference(v + h) - reference(v - h)) / (2.0 * h);
                assert!(
                    (analytic[i] - central).abs() < 1e-7,
                    "{name} at {v}: {} != {central}",
                    analytic[i]
                );
            }
        }
    }

    #[test]
    fn test_erfc_gradient_is_the_negation_of_erf() {
        // erfc = 1 - erf, so a dropped sign is the likely bug here and a
        // tolerance-based check alone would not pin it down.
        for &x in &[-1.0f64, 0.0, 1.5] {
            let a = f64_tensor(vec![x], true);
            let ga = crate::autograd::backward_collect(&erf(&a).unwrap(), None).unwrap();
            let b = f64_tensor(vec![x], true);
            let gb = crate::autograd::backward_collect(&erfc(&b).unwrap(), None).unwrap();

            let da = ga[&a.id()].data().as_f64_slice().unwrap()[0];
            let db = gb[&b.id()].data().as_f64_slice().unwrap()[0];
            assert!((da + db).abs() < 1e-15, "{da} is not the negation of {db}");
        }

        // d/dx erf(0) = 2/sqrt(pi)
        let origin = f64_tensor(vec![0.0], true);
        let grads = crate::autograd::backward_collect(&erf(&origin).unwrap(), None).unwrap();
        let slope = grads[&origin.id()].data().as_f64_slice().unwrap()[0];
        assert!((slope - 2.0 / std::f64::consts::PI.sqrt()).abs() < 1e-15);
    }

    #[test]
    fn test_rejects_non_float_dtypes() {
        let ints = Tensor::new(
            Arc::new(TensorData::from_vec_i64(vec![1, 2], Device::cpu())),
            Shape::new(vec![2]),
            DataType::Int64,
            Device::cpu(),
            false,
        );
        assert!(log2(&ints).is_err());
        assert!(log10(&ints).is_err());
        assert!(erf(&ints).is_err());
        assert!(erfc(&ints).is_err());
    }
}
