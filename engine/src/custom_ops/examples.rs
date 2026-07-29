// Copyright (c) Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

use super::*;
use crate::{
    error::Result,
    ops::{activation, arithmetic},
    tensor::{DataType, Shape, Tensor},
};
use rustc_hash::FxHashMap;
use std::sync::Arc;

/// Example: Custom Swish activation function (x * sigmoid(x))
pub fn create_swish_op() -> Result<Arc<dyn CustomOp>> {
    CustomOpBuilder::new("swish", 1)
        .forward(|inputs| {
            let x = inputs[0];
            let sigmoid_x = activation::sigmoid(x)?;
            arithmetic::mul(x, &sigmoid_x)
        })
        .backward(|ctx| {
            // Swish gradient: sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
            // For simplicity, we'll approximate this
            let mut gradients = FxHashMap::default();

            if let (Some(&input_id), Some(input_shape), Some(input_dtype), Some(input_device)) = (
                ctx.input_ids.first(),
                ctx.input_shape(0),
                ctx.input_dtype(0),
                ctx.input_device(0),
            ) {
                // Create a gradient tensor (simplified implementation)
                let grad = Tensor::ones(
                    Shape::new(input_shape.to_vec()),
                    input_dtype,
                    input_device,
                    false,
                );
                gradients.insert(input_id, grad);
            }

            Ok(gradients)
        })
        .validate(|inputs| {
            if inputs[0].numel() == 0 {
                return Err(MinitensorError::invalid_argument(
                    "Input tensor cannot be empty",
                ));
            }
            Ok(())
        })
        .build()
}

/// Example: Custom GELU activation function (Gaussian Error Linear Unit)
pub fn create_gelu_op() -> Result<Arc<dyn CustomOp>> {
    CustomOpBuilder::new("gelu", 1)
        .forward(|inputs| {
            let x = inputs[0];
            // GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
            // Simplified implementation using existing operations
            let tanh_x = activation::tanh(x)?;
            let one = Tensor::ones(x.shape().clone(), x.dtype(), x.device(), false);

            let one_plus_tanh = arithmetic::add(&one, &tanh_x)?;
            arithmetic::mul(x, &one_plus_tanh)
        })
        .backward(|ctx| {
            let mut gradients = FxHashMap::default();

            if let (Some(&input_id), Some(input_shape), Some(input_dtype), Some(input_device)) = (
                ctx.input_ids.first(),
                ctx.input_shape(0),
                ctx.input_dtype(0),
                ctx.input_device(0),
            ) {
                let grad = Tensor::ones(
                    Shape::new(input_shape.to_vec()),
                    input_dtype,
                    input_device,
                    false,
                );
                gradients.insert(input_id, grad);
            }

            Ok(gradients)
        })
        .build()
}

/// Example: Custom Mish activation function (x * tanh(softplus(x)))
pub fn create_mish_op() -> Result<Arc<dyn CustomOp>> {
    CustomOpBuilder::new("mish", 1)
        .forward(|inputs| {
            let x = inputs[0];
            // Mish(x) = x * tanh(ln(1 + exp(x)))
            // Simplified: x * tanh(x) for demonstration
            let tanh_x = activation::tanh(x)?;
            arithmetic::mul(x, &tanh_x)
        })
        .backward(|ctx| {
            let mut gradients = FxHashMap::default();

            if let Some(&input_id) = ctx.input_ids.first() {
                let grad = ctx.grad_output.clone();
                gradients.insert(input_id, grad);
            }

            Ok(gradients)
        })
        .build()
}

/// Example: Custom element-wise power operation (x^y)
pub fn create_power_op() -> Result<Arc<dyn CustomOp>> {
    CustomOpBuilder::new("power", 2)
        .forward(|inputs| {
            let base = inputs[0];
            let exponent = inputs[1];

            // For demonstration, we'll use a simplified power operation
            // In practice, this would use proper mathematical functions
            arithmetic::mul(base, exponent) // Simplified
        })
        .backward(|ctx| {
            let mut gradients = FxHashMap::default();

            // Power gradient: d/dx(x^y) = y * x^(y-1), d/dy(x^y) = x^y * ln(x)
            // Simplified implementation
            for (i, &input_id) in ctx.input_ids.iter().enumerate() {
                if let (Some(input_shape), Some(input_dtype), Some(input_device)) =
                    (ctx.input_shape(i), ctx.input_dtype(i), ctx.input_device(i))
                {
                    let grad = Tensor::ones(
                        Shape::new(input_shape.to_vec()),
                        input_dtype,
                        input_device,
                        false,
                    );
                    gradients.insert(input_id, grad);
                }
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

/// Example: Custom layer normalization operation
pub fn create_layer_norm_op() -> Result<Arc<dyn CustomOp>> {
    CustomOpBuilder::new("layer_norm", 3) // input, weight, bias
        .forward(|inputs| {
            let input = inputs[0];
            let _weight = inputs[1];
            let _bias = inputs[2];

            // Simplified layer normalization
            // In practice, this would compute mean and variance along specified dimensions
            Ok(input.clone())
        })
        .backward(|ctx| {
            let mut gradients = FxHashMap::default();

            // Layer norm has gradients for input, weight, and bias
            for (i, &input_id) in ctx.input_ids.iter().enumerate() {
                if let (Some(input_shape), Some(input_dtype), Some(input_device)) =
                    (ctx.input_shape(i), ctx.input_dtype(i), ctx.input_device(i))
                {
                    let grad = if i == 0 {
                        // Input gradient
                        ctx.grad_output.clone()
                    } else {
                        // Weight and bias gradients
                        Tensor::ones(
                            Shape::new(input_shape.to_vec()),
                            input_dtype,
                            input_device,
                            false,
                        )
                    };
                    gradients.insert(input_id, grad);
                }
            }

            Ok(gradients)
        })
        .validate(|inputs| {
            let input_shape = inputs[0].shape();
            let weight_shape = inputs[1].shape();
            let bias_shape = inputs[2].shape();

            // Check that weight and bias have compatible shapes with input
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
