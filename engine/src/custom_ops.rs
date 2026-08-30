// Copyright (c) Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

pub mod examples;

use crate::{
    autograd::{GradientFunction, TensorId, with_grad_fn},
    device::Device,
    error::{MinitensorError, Result},
    tensor::{DataType, Shape, Tensor},
};
use rustc_hash::FxHashMap;
use std::sync::{Arc, RwLock};

// Type aliases to keep function signatures manageable and avoid repeated
// complex trait bounds that hurt compile times and readability.
type ForwardFn = Arc<dyn Fn(&[&Tensor]) -> Result<Tensor> + Send + Sync>;
type BackwardFn =
    Arc<dyn Fn(&BackwardContext) -> Result<FxHashMap<TensorId, Tensor>> + Send + Sync>;
type ValidateFn = Arc<dyn Fn(&[&Tensor]) -> Result<()> + Send + Sync>;
type OutputShapeFn = Arc<dyn Fn(&[&Shape]) -> Result<Shape> + Send + Sync>;
type OutputDtypeFn = Arc<dyn Fn(&[DataType]) -> Result<DataType> + Send + Sync>;
type OutputDeviceFn = Arc<dyn Fn(&[&Device]) -> Result<Device> + Send + Sync>;

/// What a custom operation's backward pass knows about the forward call it is
/// differentiating.
///
/// The saved `inputs` and `output` are what make a correct derivative
/// expressible: anything non-linear needs the values it was evaluated at, and
/// without them a backward function can only return constants or pass
/// `grad_output` straight through. Saving them costs no copies — tensors share
/// their storage — but it does keep those buffers alive until the graph is
/// released, which is the usual price of autograd.
pub struct BackwardContext<'a> {
    /// Gradient flowing into the operation's output.
    pub grad_output: &'a Tensor,
    /// The inputs the forward pass received, in order.
    pub inputs: &'a [Tensor],
    /// The value the forward pass produced. Useful when a derivative is cheaper
    /// in terms of the output, as in `sigmoid'(x) = y(1 - y)`.
    pub output: &'a Tensor,
    /// Identifier of each input, in the same order as `inputs`. Returned
    /// gradients are keyed by these.
    pub input_ids: &'a [TensorId],
}

impl BackwardContext<'_> {
    /// The `i`th input, if the operation received one.
    pub fn input(&self, i: usize) -> Option<&Tensor> {
        self.inputs.get(i)
    }

    /// Shape of the `i`th input, as dimensions.
    pub fn input_shape(&self, i: usize) -> Option<&[usize]> {
        self.inputs.get(i).map(|t| t.shape().dims())
    }

    /// Data type of the `i`th input.
    pub fn input_dtype(&self, i: usize) -> Option<DataType> {
        self.inputs.get(i).map(|t| t.dtype())
    }

    /// Device of the `i`th input.
    pub fn input_device(&self, i: usize) -> Option<Device> {
        self.inputs.get(i).map(|t| t.device())
    }
}

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

    /// Register a custom operation unless one with the same name is already
    /// present, returning whether it was inserted.
    ///
    /// [`Self::register`] rejects duplicates so that a genuine name collision
    /// in user code is caught. This variant is for callers that are
    /// (re)installing a known set and cannot know which members already exist
    /// — see [`crate::custom_ops::examples::register_example_ops`]. The check
    /// and the insert share one write lock, so concurrent callers cannot both
    /// decide the name is free.
    pub fn register_if_absent(&self, op: Arc<dyn CustomOp>) -> Result<bool> {
        let name = op.name().to_string();

        if name.is_empty() {
            return Err(MinitensorError::invalid_argument(
                "Operation name cannot be empty",
            ));
        }

        let mut ops = self.operations.write().map_err(|_| {
            MinitensorError::internal_error("Failed to acquire registry write lock")
        })?;

        if ops.contains_key(&name) {
            return Ok(false);
        }

        ops.insert(name, op);
        Ok(true)
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
        if requires_grad && let Some(grad_fn) = op.create_gradient_function(inputs, &output) {
            // `with_grad_fn`, not `add_to_graph`: the node has to go *on the
            // output* as well as into the graph, and doing only the second
            // leaves a tensor the walk treats as a leaf -- `backward()` reached
            // it, found no gradient function, and stopped, so the custom
            // gradient never ran and the inputs came back with none.
            //
            // The flag is set first for the same reason. A forward that
            // supplies its own gradient computes the value with recording
            // *off*, since the operations inside it are an implementation
            // detail rather than the graph, so the tensor it hands back carries
            // no flag of its own. It is differentiable because this node says
            // so, and the flag has to say so too.
            return with_grad_fn(output.requires_grad_(true), grad_fn);
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
static GLOBAL_REGISTRY: std::sync::LazyLock<Arc<CustomOpRegistry>> =
    std::sync::LazyLock::new(|| Arc::new(CustomOpRegistry::new()));

/// Handle to the process-wide custom operation registry
///
/// Subsystems that need to own a registry reference — the plugin manager above
/// all — must take it from here rather than building their own, so that
/// everything they register lands in the one namespace the `*_custom_op`
/// functions and the Python bindings read from.
pub fn global_registry() -> Arc<CustomOpRegistry> {
    Arc::clone(&GLOBAL_REGISTRY)
}

/// Register a custom operation globally
pub fn register_custom_op(op: Arc<dyn CustomOp>) -> Result<()> {
    GLOBAL_REGISTRY.register(op)
}

/// Register a custom operation globally unless the name is already taken,
/// returning whether it was inserted. See
/// [`CustomOpRegistry::register_if_absent`].
pub fn register_custom_op_if_absent(op: Arc<dyn CustomOp>) -> Result<bool> {
    GLOBAL_REGISTRY.register_if_absent(op)
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
    /// Inputs saved from the forward pass, so the backward function can
    /// evaluate a derivative at the point it was computed.
    pub inputs: Vec<Tensor>,
    /// Output saved from the forward pass.
    pub output: Tensor,
    pub backward_fn: BackwardFn,
}

impl GradientFunction for CustomOpBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<FxHashMap<TensorId, Tensor>> {
        (self.backward_fn)(&BackwardContext {
            grad_output,
            inputs: &self.inputs,
            output: &self.output,
            input_ids: &self.input_ids,
        })
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
    ///
    /// The closure receives a [`BackwardContext`] carrying the incoming
    /// gradient along with the inputs and output saved from the forward pass,
    /// and returns a gradient per input keyed by
    /// [`input_ids`](BackwardContext::input_ids). Omitting an input's entry
    /// means no gradient flows to it.
    pub fn backward<F>(mut self, f: F) -> Self
    where
        F: Fn(&BackwardContext) -> Result<FxHashMap<TensorId, Tensor>> + Send + Sync + 'static,
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
        output: &Tensor,
    ) -> Option<Arc<dyn GradientFunction>> {
        if let Some(backward_fn) = &self.backward_fn {
            let input_ids: Vec<TensorId> = inputs.iter().map(|t| t.id()).collect();
            // Cloning a tensor shares its storage, so saving the forward values
            // costs a refcount rather than a copy.
            let saved_inputs: Vec<Tensor> = inputs.iter().map(|t| (*t).clone()).collect();

            Some(Arc::new(CustomOpBackward {
                op_name: self.name.clone(),
                input_ids,
                inputs: saved_inputs,
                output: output.clone(),
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
                Ok(*input_devices[0])
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
                crate::ops::arithmetic::add(inputs[0], inputs[1])
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
    fn test_backward_can_differentiate_a_nonlinear_operation() {
        // The backward function used to receive only shapes, dtypes and devices,
        // so anything non-linear could return constants or pass `grad_output`
        // through, but never its actual derivative. With the forward inputs
        // saved, `d(x^3)/dx = 3x^2` is expressible — check it against central
        // differences, as the custom-op docs tell authors to do.
        let registry = CustomOpRegistry::new();
        let op = CustomOpBuilder::new("test_cube", 1)
            .forward(|inputs| {
                let sq = crate::ops::arithmetic::mul(inputs[0], inputs[0])?;
                crate::ops::arithmetic::mul(&sq, inputs[0])
            })
            .backward(|ctx| {
                let mut gradients = FxHashMap::default();
                let (Some(&id), Some(x)) = (ctx.input_ids.first(), ctx.input(0)) else {
                    return Ok(gradients);
                };
                // 3x^2, without needing a scalar-constant helper.
                let sq = crate::ops::arithmetic::mul(x, x)?;
                let two_sq = crate::ops::arithmetic::add(&sq, &sq)?;
                let dydx = crate::ops::arithmetic::add(&two_sq, &sq)?;
                gradients.insert(id, crate::ops::arithmetic::mul(ctx.grad_output, &dydx)?);
                Ok(gradients)
            })
            .build()
            .unwrap();
        registry.register(op).unwrap();

        let values = [-1.75f32, -0.5, 0.25, 2.0];
        let x = Tensor::new(
            Arc::new(crate::tensor::TensorData::from_vec_f32(
                values.to_vec(),
                Device::cpu(),
            )),
            Shape::new(vec![values.len()]),
            DataType::Float32,
            Device::cpu(),
            true,
        );

        let y = registry.execute("test_cube", &[&x]).unwrap();
        let forward = y.data().as_f32_slice().unwrap();
        for (i, v) in values.iter().enumerate() {
            assert!((forward[i] - v * v * v).abs() < 1e-5, "cube at {v}");
        }

        let loss = crate::ops::reduction::sum(&y, None, false).unwrap();
        loss.backward(None).unwrap();
        let grad = crate::autograd::get_gradient(&x).expect("input gradient");
        let grad = grad.data().as_f32_slice().unwrap();

        // d(sum of x^3)/dx_i = 3 x_i^2, compared against central differences.
        let h = 1e-2f32;
        for (i, v) in values.iter().enumerate() {
            let numeric = ((v + h).powi(3) - (v - h).powi(3)) / (2.0 * h);
            assert!(
                (grad[i] - numeric).abs() <= 1e-3 * numeric.abs().max(1.0),
                "gradient at {v}: analytic {} vs numeric {numeric}",
                grad[i]
            );
        }
    }

    #[test]
    fn test_backward_receives_the_forward_output() {
        // `output` is saved too, so derivatives are also expressible in terms of
        // the value the forward pass produced.
        let registry = CustomOpRegistry::new();
        let op = CustomOpBuilder::new("test_output_visible", 1)
            .forward(|inputs| crate::ops::arithmetic::mul(inputs[0], inputs[0]))
            .backward(|ctx| {
                let mut gradients = FxHashMap::default();
                if let Some(&id) = ctx.input_ids.first() {
                    // Hand back the output itself so the test can observe it.
                    gradients.insert(id, ctx.output.clone());
                }
                Ok(gradients)
            })
            .build()
            .unwrap();
        registry.register(op).unwrap();

        let x = Tensor::new(
            Arc::new(crate::tensor::TensorData::from_vec_f32(
                vec![2.0, 3.0],
                Device::cpu(),
            )),
            Shape::new(vec![2]),
            DataType::Float32,
            Device::cpu(),
            true,
        );
        let y = registry.execute("test_output_visible", &[&x]).unwrap();
        let loss = crate::ops::reduction::sum(&y, None, false).unwrap();
        loss.backward(None).unwrap();
        let grad = crate::autograd::get_gradient(&x).expect("input gradient");
        assert_eq!(grad.data().as_f32_slice().unwrap(), &[4.0, 9.0]);
    }

    #[test]
    fn test_custom_op_builder() {
        let op = CustomOpBuilder::new("test_mul", 2)
            .forward(|inputs| crate::ops::arithmetic::mul(inputs[0], inputs[1]))
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
