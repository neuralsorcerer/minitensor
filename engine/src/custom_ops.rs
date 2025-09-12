// Copyright (c) 2025 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

pub mod examples;

use crate::{
    autograd::{GradientFunction, TensorId, add_to_graph},
    device::Device,
    error::{MinitensorError, Result},
    tensor::{DataType, Shape, Tensor},
};
use rustc_hash::FxHashMap;
use std::sync::{Arc, RwLock};

// Type aliases to keep function signatures manageable and avoid repeated
// complex trait bounds that hurt compile times and readability.
type ForwardFn = Arc<dyn Fn(&[&Tensor]) -> Result<Tensor> + Send + Sync>;
type BackwardFn = Arc<
    dyn Fn(
            &Tensor,
            &[TensorId],
            &[Vec<usize>],
            &[DataType],
            &[Device],
        ) -> Result<FxHashMap<TensorId, Tensor>>
        + Send
        + Sync,
>;
type ValidateFn = Arc<dyn Fn(&[&Tensor]) -> Result<()> + Send + Sync>;
type OutputShapeFn = Arc<dyn Fn(&[&Shape]) -> Result<Shape> + Send + Sync>;
type OutputDtypeFn = Arc<dyn Fn(&[DataType]) -> Result<DataType> + Send + Sync>;
type OutputDeviceFn = Arc<dyn Fn(&[&Device]) -> Result<Device> + Send + Sync>;

/// Trait for custom operations that can be registered with the system
pub trait CustomOp: Send + Sync {
    /// The name of the operation (must be unique)
    fn name(&self) -> &str;

    /// Validate input tensors before execution
    fn validate_inputs(&self, inputs: &[&Tensor]) -> Result<()>;

    /// Execute the forward pass of the operation
    fn forward(&self, inputs: &[&Tensor]) -> Result<Tensor>;

    /// Create a gradient function for the backward pass
    fn create_gradient_function(
        &self,
        inputs: &[&Tensor],
        output: &Tensor,
    ) -> Option<Arc<dyn GradientFunction>>;

    /// Get the expected number of input tensors
    fn num_inputs(&self) -> usize;

    /// Get the expected output shape given input shapes
    fn output_shape(&self, input_shapes: &[&Shape]) -> Result<Shape>;

    /// Get the expected output data type given input data types
    fn output_dtype(&self, input_dtypes: &[DataType]) -> Result<DataType>;

    /// Get the expected output device given input devices
    fn output_device(&self, input_devices: &[&Device]) -> Result<Device>;
}

/// Registry for custom operations
pub struct CustomOpRegistry {
    operations: RwLock<FxHashMap<String, Arc<dyn CustomOp>>>,
}

impl CustomOpRegistry {
    /// Create a new custom operation registry
    pub fn new() -> Self {
        Self {
            operations: RwLock::new(FxHashMap::default()),
        }
    }

    /// Register a custom operation
    pub fn register(&self, op: Arc<dyn CustomOp>) -> Result<()> {
        let name = op.name().to_string();

        // Validate operation name
        if name.is_empty() {
            return Err(MinitensorError::invalid_argument(
                "Operation name cannot be empty",
            ));
        }

        let mut ops = self.operations.write().map_err(|_| {
            MinitensorError::internal_error("Failed to acquire registry write lock")
        })?;

        // Check for duplicate names
        if ops.contains_key(&name) {
            return Err(MinitensorError::invalid_argument(format!(
                "Operation '{}' is already registered",
                name
            )));
        }

        ops.insert(name, op);
        Ok(())
    }

    /// Unregister a custom operation
    pub fn unregister(&self, name: &str) -> Result<()> {
        let mut ops = self.operations.write().map_err(|_| {
            MinitensorError::internal_error("Failed to acquire registry write lock")
        })?;

        if ops.remove(name).is_none() {
            return Err(MinitensorError::invalid_argument(format!(
                "Operation '{}' is not registered",
                name
            )));
        }

        Ok(())
    }

    /// Get a registered operation by name
    pub fn get(&self, name: &str) -> Result<Arc<dyn CustomOp>> {
        let ops = self
            .operations
            .read()
            .map_err(|_| MinitensorError::internal_error("Failed to acquire registry read lock"))?;

        ops.get(name).cloned().ok_or_else(|| {
            MinitensorError::invalid_argument(format!("Operation '{}' is not registered", name))
        })
    }

    /// List all registered operation names
    pub fn list_operations(&self) -> Result<Vec<String>> {
        let ops = self
            .operations
            .read()
            .map_err(|_| MinitensorError::internal_error("Failed to acquire registry read lock"))?;

        Ok(ops.keys().cloned().collect())
    }

    /// Check if an operation is registered
    pub fn is_registered(&self, name: &str) -> Result<bool> {
        let ops = self
            .operations
            .read()
            .map_err(|_| MinitensorError::internal_error("Failed to acquire registry read lock"))?;

        Ok(ops.contains_key(name))
    }

    /// Execute a registered custom operation
    pub fn execute(&self, name: &str, inputs: &[&Tensor]) -> Result<Tensor> {
        let op = self.get(name)?;

        // Validate inputs
        op.validate_inputs(inputs)?;

        // Execute forward pass
        let output = op.forward(inputs)?;

        // Set up gradient tracking if any input requires gradients
        let requires_grad = inputs.iter().any(|t| t.requires_grad());
        if requires_grad {
            if let Some(grad_fn) = op.create_gradient_function(inputs, &output) {
                add_to_graph(&output, Some(grad_fn))?;
            }
        }

        Ok(output)
    }
}

impl Default for CustomOpRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Global custom operation registry
static GLOBAL_REGISTRY: std::sync::LazyLock<CustomOpRegistry> =
    std::sync::LazyLock::new(CustomOpRegistry::new);

/// Register a custom operation globally
pub fn register_custom_op(op: Arc<dyn CustomOp>) -> Result<()> {
    GLOBAL_REGISTRY.register(op)
}

/// Unregister a custom operation globally
pub fn unregister_custom_op(name: &str) -> Result<()> {
    GLOBAL_REGISTRY.unregister(name)
}

/// Execute a custom operation globally
pub fn execute_custom_op(name: &str, inputs: &[&Tensor]) -> Result<Tensor> {
    GLOBAL_REGISTRY.execute(name, inputs)
}

/// List all registered custom operations
pub fn list_custom_ops() -> Result<Vec<String>> {
    GLOBAL_REGISTRY.list_operations()
}

/// Check if a custom operation is registered
pub fn is_custom_op_registered(name: &str) -> Result<bool> {
    GLOBAL_REGISTRY.is_registered(name)
}

/// Gradient function for custom operations
pub struct CustomOpBackward {
    pub op_name: String,
    pub input_ids: Vec<TensorId>,
    pub input_shapes: Vec<Vec<usize>>,
    pub input_dtypes: Vec<DataType>,
    pub input_devices: Vec<Device>,
    pub backward_fn: BackwardFn,
}

impl GradientFunction for CustomOpBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<FxHashMap<TensorId, Tensor>> {
        (self.backward_fn)(
            grad_output,
            &self.input_ids,
            &self.input_shapes,
            &self.input_dtypes,
            &self.input_devices,
        )
    }

    fn input_ids(&self) -> &[TensorId] {
        &self.input_ids
    }
}

/// Builder for creating custom operations with validation
pub struct CustomOpBuilder {
    name: String,
    num_inputs: usize,
    forward_fn: Option<ForwardFn>,
    backward_fn: Option<BackwardFn>,
    validate_fn: Option<ValidateFn>,
    output_shape_fn: Option<OutputShapeFn>,
    output_dtype_fn: Option<OutputDtypeFn>,
    output_device_fn: Option<OutputDeviceFn>,
}

impl CustomOpBuilder {
    /// Create a new custom operation builder
    pub fn new(name: &str, num_inputs: usize) -> Self {
        Self {
            name: name.to_string(),
            num_inputs,
            forward_fn: None,
            backward_fn: None,
            validate_fn: None,
            output_shape_fn: None,
            output_dtype_fn: None,
            output_device_fn: None,
        }
    }

    /// Set the forward function
    pub fn forward<F>(mut self, f: F) -> Self
    where
        F: Fn(&[&Tensor]) -> Result<Tensor> + Send + Sync + 'static,
    {
        self.forward_fn = Some(Arc::new(f));
        self
    }

    /// Set the backward function
    pub fn backward<F>(mut self, f: F) -> Self
    where
        F: Fn(
                &Tensor,
                &[TensorId],
                &[Vec<usize>],
                &[DataType],
                &[Device],
            ) -> Result<FxHashMap<TensorId, Tensor>>
            + Send
            + Sync
            + 'static,
    {
        self.backward_fn = Some(Arc::new(f));
        self
    }

    /// Set the validation function
    pub fn validate<F>(mut self, f: F) -> Self
    where
        F: Fn(&[&Tensor]) -> Result<()> + Send + Sync + 'static,
    {
        self.validate_fn = Some(Arc::new(f));
        self
    }

    /// Set the output shape function
    pub fn output_shape<F>(mut self, f: F) -> Self
    where
        F: Fn(&[&Shape]) -> Result<Shape> + Send + Sync + 'static,
    {
        self.output_shape_fn = Some(Arc::new(f));
        self
    }

    /// Set the output dtype function
    pub fn output_dtype<F>(mut self, f: F) -> Self
    where
        F: Fn(&[DataType]) -> Result<DataType> + Send + Sync + 'static,
    {
        self.output_dtype_fn = Some(Arc::new(f));
        self
    }

    /// Set the output device function
    pub fn output_device<F>(mut self, f: F) -> Self
    where
        F: Fn(&[&Device]) -> Result<Device> + Send + Sync + 'static,
    {
        self.output_device_fn = Some(Arc::new(f));
        self
    }

    /// Build the custom operation
    pub fn build(self) -> Result<Arc<dyn CustomOp>> {
        let forward_fn = self
            .forward_fn
            .ok_or_else(|| MinitensorError::invalid_argument("Forward function is required"))?;

        Ok(Arc::new(BuiltCustomOp {
            name: self.name,
            num_inputs: self.num_inputs,
            forward_fn,
            backward_fn: self.backward_fn,
            validate_fn: self.validate_fn,
            output_shape_fn: self.output_shape_fn,
            output_dtype_fn: self.output_dtype_fn,
            output_device_fn: self.output_device_fn,
        }))
    }
}

/// Built custom operation from the builder
struct BuiltCustomOp {
    name: String,
    num_inputs: usize,
    forward_fn: ForwardFn,
    backward_fn: Option<BackwardFn>,
    validate_fn: Option<ValidateFn>,
    output_shape_fn: Option<OutputShapeFn>,
    output_dtype_fn: Option<OutputDtypeFn>,
    output_device_fn: Option<OutputDeviceFn>,
}

impl CustomOp for BuiltCustomOp {
    fn name(&self) -> &str {
        &self.name
    }

    fn validate_inputs(&self, inputs: &[&Tensor]) -> Result<()> {
        // Check number of inputs
        if inputs.len() != self.num_inputs {
            return Err(MinitensorError::invalid_argument(format!(
                "Operation '{}' expects {} inputs, got {}",
                self.name,
                self.num_inputs,
                inputs.len()
            )));
        }

        // Run custom validation if provided
        if let Some(validate_fn) = &self.validate_fn {
            validate_fn(inputs)?;
        }

        Ok(())
    }

    fn forward(&self, inputs: &[&Tensor]) -> Result<Tensor> {
        (self.forward_fn)(inputs)
    }

    fn create_gradient_function(
        &self,
        inputs: &[&Tensor],
        _output: &Tensor,
    ) -> Option<Arc<dyn GradientFunction>> {
        if let Some(backward_fn) = &self.backward_fn {
            let input_ids: Vec<TensorId> = inputs.iter().map(|t| t.id()).collect();
            let input_shapes: Vec<Vec<usize>> =
                inputs.iter().map(|t| t.shape().dims().to_vec()).collect();
            let input_dtypes: Vec<DataType> = inputs.iter().map(|t| t.dtype()).collect();
            let input_devices: Vec<Device> = inputs.iter().map(|t| t.device()).collect();

            Some(Arc::new(CustomOpBackward {
                op_name: self.name.clone(),
                input_ids,
                input_shapes,
                input_dtypes,
                input_devices,
                backward_fn: backward_fn.clone(),
            }))
        } else {
            None
        }
    }

    fn num_inputs(&self) -> usize {
        self.num_inputs
    }

    fn output_shape(&self, input_shapes: &[&Shape]) -> Result<Shape> {
        if let Some(output_shape_fn) = &self.output_shape_fn {
            output_shape_fn(input_shapes)
        } else {
            // Default: use the shape of the first input
            if input_shapes.is_empty() {
                Err(MinitensorError::invalid_argument(
                    "No input shapes provided",
                ))
            } else {
                Ok(input_shapes[0].clone())
            }
        }
    }

    fn output_dtype(&self, input_dtypes: &[DataType]) -> Result<DataType> {
        if let Some(output_dtype_fn) = &self.output_dtype_fn {
            output_dtype_fn(input_dtypes)
        } else {
            // Default: use the dtype of the first input
            if input_dtypes.is_empty() {
                Err(MinitensorError::invalid_argument(
                    "No input dtypes provided",
                ))
            } else {
                Ok(input_dtypes[0])
            }
        }
    }

    fn output_device(&self, input_devices: &[&Device]) -> Result<Device> {
        if let Some(output_device_fn) = &self.output_device_fn {
            output_device_fn(input_devices)
        } else {
            // Default: use the device of the first input
            if input_devices.is_empty() {
                Err(MinitensorError::invalid_argument(
                    "No input devices provided",
                ))
            } else {
                Ok(input_devices[0].clone())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_custom_op_registry() {
        let registry = CustomOpRegistry::new();

        // Create a simple custom operation
        let op = CustomOpBuilder::new("test_add", 2)
            .forward(|inputs| {
                // Simple addition operation
                crate::operations::arithmetic::add(inputs[0], inputs[1])
            })
            .build()
            .unwrap();

        // Register the operation
        registry.register(op).unwrap();

        // Check if it's registered
        assert!(registry.is_registered("test_add").unwrap());

        // List operations
        let ops = registry.list_operations().unwrap();
        assert!(ops.contains(&"test_add".to_string()));

        // Unregister the operation
        registry.unregister("test_add").unwrap();
        assert!(!registry.is_registered("test_add").unwrap());
    }

    #[test]
    fn test_custom_op_builder() {
        let op = CustomOpBuilder::new("test_mul", 2)
            .forward(|inputs| crate::operations::arithmetic::mul(inputs[0], inputs[1]))
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
            .build()
            .unwrap();

        assert_eq!(op.name(), "test_mul");
        assert_eq!(op.num_inputs(), 2);
    }

    #[test]
    fn test_global_registry() {
        let op = CustomOpBuilder::new("global_test", 1)
            .forward(|inputs| Ok(inputs[0].clone()))
            .build()
            .unwrap();

        register_custom_op(op).unwrap();
        assert!(is_custom_op_registered("global_test").unwrap());

        let ops = list_custom_ops().unwrap();
        assert!(ops.contains(&"global_test".to_string()));

        unregister_custom_op("global_test").unwrap();
        assert!(!is_custom_op_registered("global_test").unwrap());
    }
}
