// Copyright (c) Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

//! Every operation here is a *working* implementation, not a sketch. They are
//! registered into the process-wide registry under their real names, so a
//! caller reaching `execute_custom_op("gelu", ...)` gets GELU; and they are the
//! worked examples someone writes their own operation from, so a backward that
//! returns ones would be teaching the one thing that is hardest to debug.
//!
//! The gradients are written out in terms of the saved inputs and output rather
//! than delegated to the built-in autograd nodes, since demonstrating that is
//! the point of the custom-op API.

use super::*;
use crate::{
    error::Result,
    ops::{activation, arithmetic, normalization, reduction},
    tensor::{DataType, Tensor},
};
use rustc_hash::FxHashMap;
use std::sync::Arc;

/// Gradient contribution for a single input, keyed for the registry.
fn single_grad(ctx: &BackwardContext<'_>, grad: Tensor) -> FxHashMap<TensorId, Tensor> {
    let mut gradients = FxHashMap::default();
    if let Some(&input_id) = ctx.input_ids.first() {
        gradients.insert(input_id, grad);
    }
    gradients
}

/// A tensor of ones shaped like `like`, for the `1 - t` forms below.
fn ones_like(like: &Tensor) -> Tensor {
    Tensor::ones(like.shape().clone(), like.dtype(), like.device(), false)
}

/// A broadcastable one-element tensor holding `value`, matching `like`'s dtype
/// and device.
fn scalar_like(like: &Tensor, value: f64) -> Result<Tensor> {
    crate::ops::util::create_scalar_tensor(value, like.dtype(), like.device())
}

/// Reject an empty input, which every activation here would silently accept.
fn reject_empty(inputs: &[&Tensor]) -> Result<()> {
    if inputs[0].numel() == 0 {
        return Err(MinitensorError::invalid_argument(
            "Input tensor cannot be empty",
        ));
    }
    Ok(())
}

/// Example: Swish / SiLU, `f(x) = x * sigmoid(x)`.
pub fn create_swish_op() -> Result<Arc<dyn CustomOp>> {
    CustomOpBuilder::new("swish", 1)
        .forward(|inputs| {
            let x = inputs[0];
            arithmetic::mul(x, &activation::sigmoid(x)?)
        })
        .backward(|ctx| {
            // f'(x) = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
            let x = match ctx.input(0) {
                Some(x) => x,
                None => return Ok(FxHashMap::default()),
            };
            let s = activation::sigmoid(x)?;
            let one_minus_s = arithmetic::sub(&ones_like(&s), &s)?;
            let inner = arithmetic::add(&ones_like(&s), &arithmetic::mul(x, &one_minus_s)?)?;
            let local = arithmetic::mul(&s, &inner)?;
            Ok(single_grad(ctx, arithmetic::mul(ctx.grad_output, &local)?))
        })
        .validate(reject_empty)
        .build()
}

/// Example: GELU, `f(x) = x * Phi(x)` with the exact Gaussian CDF.
pub fn create_gelu_op() -> Result<Arc<dyn CustomOp>> {
    CustomOpBuilder::new("gelu", 1)
        .forward(|inputs| activation::gelu(inputs[0], false))
        .backward(|ctx| {
            // f'(x) = Phi(x) + x * phi(x). Recovering Phi from the forward
            // value would need `f(x) / x`, which is undefined at zero, so both
            // factors are built from erf and exp directly.
            let x = match ctx.input(0) {
                Some(x) => x,
                None => return Ok(FxHashMap::default()),
            };
            let ones = ones_like(x);
            // Phi(x) = 0.5 * (1 + erf(x / sqrt(2)))
            let cdf = arithmetic::mul(
                &scalar_like(x, 0.5)?,
                &arithmetic::add(
                    &ones,
                    &activation::erf(&arithmetic::mul(
                        x,
                        &scalar_like(x, std::f64::consts::FRAC_1_SQRT_2)?,
                    )?)?,
                )?,
            )?;
            // phi(x) = exp(-x^2 / 2) / sqrt(2 pi)
            let pdf = arithmetic::mul(
                &activation::exp(&arithmetic::mul(
                    &arithmetic::mul(x, x)?,
                    &scalar_like(x, -0.5)?,
                )?)?,
                &scalar_like(x, 1.0 / (2.0 * std::f64::consts::PI).sqrt())?,
            )?;
            let local = arithmetic::add(&cdf, &arithmetic::mul(x, &pdf)?)?;
            Ok(single_grad(ctx, arithmetic::mul(ctx.grad_output, &local)?))
        })
        .validate(reject_empty)
        .build()
}

/// Example: Mish, `f(x) = x * tanh(softplus(x))`.
pub fn create_mish_op() -> Result<Arc<dyn CustomOp>> {
    CustomOpBuilder::new("mish", 1)
        .forward(|inputs| {
            let x = inputs[0];
            let sp = activation::softplus(x, 1.0, 20.0)?;
            arithmetic::mul(x, &activation::tanh(&sp)?)
        })
        .backward(|ctx| {
            // f'(x) = tanh(sp) + x * (1 - tanh(sp)^2) * sigmoid(x),
            // since d/dx softplus(x) = sigmoid(x).
            let x = match ctx.input(0) {
                Some(x) => x,
                None => return Ok(FxHashMap::default()),
            };
            let t = activation::tanh(&activation::softplus(x, 1.0, 20.0)?)?;
            let sech2 = arithmetic::sub(&ones_like(&t), &arithmetic::mul(&t, &t)?)?;
            let local = arithmetic::add(
                &t,
                &arithmetic::mul(x, &arithmetic::mul(&sech2, &activation::sigmoid(x)?)?)?,
            )?;
            Ok(single_grad(ctx, arithmetic::mul(ctx.grad_output, &local)?))
        })
        .validate(reject_empty)
        .build()
}

/// Example: element-wise power, `f(x, y) = x^y`.
pub fn create_power_op() -> Result<Arc<dyn CustomOp>> {
    CustomOpBuilder::new("power", 2)
        .forward(|inputs| activation::pow(inputs[0], inputs[1]))
        .backward(|ctx| {
            // d/dx x^y = y * x^(y-1);  d/dy x^y = x^y * ln(x).
            let (base, exponent) = match (ctx.input(0), ctx.input(1)) {
                (Some(b), Some(e)) => (b, e),
                _ => return Ok(FxHashMap::default()),
            };
            let mut gradients = FxHashMap::default();

            if let Some(&base_id) = ctx.input_ids.first() {
                let reduced = arithmetic::sub(exponent, &ones_like(exponent))?;
                let local = arithmetic::mul(exponent, &activation::pow(base, &reduced)?)?;
                gradients.insert(base_id, arithmetic::mul(ctx.grad_output, &local)?);
            }
            if let Some(&exp_id) = ctx.input_ids.get(1) {
                // ln(x) is -inf for x <= 0, which is the true derivative there:
                // x^y is not differentiable in y for a non-positive base.
                let local = arithmetic::mul(ctx.output, &activation::log(base)?)?;
                gradients.insert(exp_id, arithmetic::mul(ctx.grad_output, &local)?);
            }
            Ok(gradients)
        })
        .validate(|inputs| {
            if inputs[0].shape() != inputs[1].shape() {
                return Err(MinitensorError::shape_mismatch(
                    inputs[0].shape().dims().to_vec(),
                    inputs[1].shape().dims().to_vec(),
                ));
            }
            Ok(())
        })
        .output_shape(|input_shapes| Ok(input_shapes[0].clone()))
        .output_dtype(|input_dtypes| {
            // Return the higher precision dtype
            match (input_dtypes[0], input_dtypes[1]) {
                (DataType::Float64, _) | (_, DataType::Float64) => Ok(DataType::Float64),
                (DataType::Float32, _) | (_, DataType::Float32) => Ok(DataType::Float32),
                _ => Ok(input_dtypes[0]),
            }
        })
        .build()
}

/// Example: layer normalization over the last dimension, with weight and bias.
pub fn create_layer_norm_op() -> Result<Arc<dyn CustomOp>> {
    CustomOpBuilder::new("layer_norm", 3) // input, weight, bias
        .forward(|inputs| {
            let (input, weight, bias) = (inputs[0], inputs[1], inputs[2]);
            let last = *input
                .shape()
                .dims()
                .last()
                .expect("validate rejects rank-0 input");
            normalization::layer_norm(input, &[last], Some(weight), Some(bias), 1e-5)
        })
        .backward(|ctx| {
            // With `xhat = (x - mean) / sigma` over the last axis of size N,
            // `y = xhat * w + b`:
            //
            //   dL/db    = sum over the leading axes of g
            //   dL/dw    = sum over the leading axes of g * xhat
            //   dL/dx    = (gw - mean(gw) - xhat * mean(gw * xhat)) / sigma,
            //              with gw = g * w
            //
            // Written out rather than delegated to the engine's own layer-norm
            // node, because a gradient function runs with recording disabled --
            // a nested forward would build no graph, and reading gradients back
            // off it would silently yield nothing.
            let (input, weight) = match (ctx.input(0), ctx.input(1)) {
                (Some(i), Some(w)) => (i, w),
                _ => return Ok(FxHashMap::default()),
            };
            let g = ctx.grad_output;
            let ndim = input.ndim();
            let last_axis = vec![ndim as isize - 1];
            let leading: Vec<isize> = (0..ndim as isize - 1).collect();

            let mean = reduction::mean(input, Some(last_axis.clone()), true)?;
            let centered = arithmetic::sub(input, &mean)?;
            let variance = reduction::mean(
                &arithmetic::mul(&centered, &centered)?,
                Some(last_axis.clone()),
                true,
            )?;
            let sigma = activation::sqrt(&arithmetic::add(&variance, &scalar_like(input, 1e-5)?)?)?;
            let xhat = arithmetic::div(&centered, &sigma)?;

            let mut gradients = FxHashMap::default();

            if let Some(&input_id) = ctx.input_ids.first() {
                let gw = arithmetic::mul(g, weight)?;
                let mean_gw = reduction::mean(&gw, Some(last_axis.clone()), true)?;
                let mean_gw_xhat =
                    reduction::mean(&arithmetic::mul(&gw, &xhat)?, Some(last_axis.clone()), true)?;
                let numerator = arithmetic::sub(
                    &arithmetic::sub(&gw, &mean_gw)?,
                    &arithmetic::mul(&xhat, &mean_gw_xhat)?,
                )?;
                gradients.insert(input_id, arithmetic::div(&numerator, &sigma)?);
            }

            // A rank-1 input has no leading axes, so the weight and bias
            // gradients are the per-element terms themselves.
            let reduce_leading = |t: &Tensor| -> Result<Tensor> {
                if leading.is_empty() {
                    Ok(t.clone())
                } else {
                    reduction::sum(t, Some(leading.clone()), false)
                }
            };
            if let Some(&weight_id) = ctx.input_ids.get(1) {
                gradients.insert(weight_id, reduce_leading(&arithmetic::mul(g, &xhat)?)?);
            }
            if let Some(&bias_id) = ctx.input_ids.get(2) {
                gradients.insert(bias_id, reduce_leading(g)?);
            }

            Ok(gradients)
        })
        .validate(|inputs| {
            let input_shape = inputs[0].shape();
            let weight_shape = inputs[1].shape();
            let bias_shape = inputs[2].shape();

            if input_shape.dims().is_empty() {
                return Err(MinitensorError::invalid_argument(
                    "layer_norm input must have at least one dimension",
                ));
            }
            if weight_shape.dims().len() != 1 || bias_shape.dims().len() != 1 {
                return Err(MinitensorError::invalid_argument(
                    "Weight and bias must be 1-dimensional",
                ));
            }

            let last_dim = input_shape.dims().last().unwrap();
            if weight_shape.dims()[0] != *last_dim || bias_shape.dims()[0] != *last_dim {
                return Err(MinitensorError::shape_mismatch(
                    vec![*last_dim],
                    weight_shape.dims().to_vec(),
                ));
            }

            Ok(())
        })
        .output_shape(|input_shapes| Ok(input_shapes[0].clone()))
        .build()
}

/// Register all example custom operations.
///
/// Idempotent: examples that are already registered are left in place instead
/// of failing the call. The registry is global and process-wide, so callers
/// cannot know which subset is currently installed — re-running a notebook
/// cell, or restoring the set after unregistering one example, would otherwise
/// abort on the first survivor. Registering a *user* operation whose name is
/// already taken is still an error (see [`register_custom_op`]).
pub fn register_example_ops() -> Result<()> {
    for op in [
        create_swish_op()?,
        create_gelu_op()?,
        create_mish_op()?,
        create_power_op()?,
        create_layer_norm_op()?,
    ] {
        crate::custom_ops::register_custom_op_if_absent(op)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::Device;

    #[test]
    fn test_swish_op() {
        let op = create_swish_op().unwrap();
        assert_eq!(op.name(), "swish");
        assert_eq!(op.num_inputs(), 1);
    }

    #[test]
    fn test_gelu_op() {
        let op = create_gelu_op().unwrap();
        assert_eq!(op.name(), "gelu");
        assert_eq!(op.num_inputs(), 1);
    }

    #[test]
    fn test_power_op() {
        let op = create_power_op().unwrap();
        assert_eq!(op.name(), "power");
        assert_eq!(op.num_inputs(), 2);

        // Test validation
        let tensor1 = Tensor::ones(
            Shape::new(vec![2, 2]),
            DataType::Float32,
            Device::cpu(),
            false,
        );
        let tensor2 = Tensor::ones(
            Shape::new(vec![3, 3]),
            DataType::Float32,
            Device::cpu(),
            false,
        );

        let result = op.validate_inputs(&[&tensor1, &tensor2]);
        assert!(result.is_err());
    }

    #[test]
    fn test_layer_norm_op() {
        let op = create_layer_norm_op().unwrap();
        assert_eq!(op.name(), "layer_norm");
        assert_eq!(op.num_inputs(), 3);
    }

    const EXAMPLE_OP_NAMES: [&str; 5] = ["swish", "gelu", "mish", "power", "layer_norm"];

    /// The custom-op registry is process-wide, and Rust runs tests in parallel.
    /// The tests below both register the example set and — in one case —
    /// unregister a name from it, so without serialising them one test can
    /// observe another's temporary removal and fail intermittently. Every test
    /// that mutates the registry must hold this.
    static REGISTRY: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// Take the registry lock, ignoring poisoning: a panic in one test must
    /// surface as that test's failure, not as an unrelated cascade here.
    fn registry_guard() -> std::sync::MutexGuard<'static, ()> {
        REGISTRY.lock().unwrap_or_else(|e| e.into_inner())
    }

    #[test]
    fn test_register_example_ops() {
        let _guard = registry_guard();
        // This test ensures all example operations can be created and registered
        let result = register_example_ops();
        assert!(result.is_ok());

        // Check that operations are registered
        for name in EXAMPLE_OP_NAMES {
            assert!(is_custom_op_registered(name).unwrap(), "{name}");
        }
    }

    #[test]
    fn test_register_example_ops_is_idempotent() {
        let _guard = registry_guard();
        // The registry is process-wide, so callers cannot know which examples
        // are already installed. Re-registering the set must succeed — both
        // when every example survives and when only some do (the pattern the
        // custom-ops notebook uses to restore the set after unregistering
        // one), which used to fail on the first name still present.
        register_example_ops().unwrap();
        register_example_ops().expect("re-registering the full set must succeed");

        crate::custom_ops::unregister_custom_op("swish").unwrap();
        assert!(!is_custom_op_registered("swish").unwrap());
        register_example_ops().expect("restoring a partially removed set must succeed");
        for name in EXAMPLE_OP_NAMES {
            assert!(is_custom_op_registered(name).unwrap(), "{name}");
        }
    }

    #[test]
    fn test_register_custom_op_still_rejects_duplicate_user_names() {
        let _guard = registry_guard();
        // Idempotence is scoped to the bundled examples; a user registering a
        // name that is already taken must still get an error.
        register_example_ops().unwrap();
        let err = crate::custom_ops::register_custom_op(create_swish_op().unwrap()).unwrap_err();
        assert!(
            err.to_string().contains("already registered"),
            "unexpected error: {err}"
        );
    }
}
