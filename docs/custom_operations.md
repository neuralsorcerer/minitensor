# Custom Operations System

The minitensor library provides a powerful extensibility framework that lets
users define their own tensor operations with full automatic differentiation
support. This system enables developers to extend the library's functionality
without modifying the core codebase.

## Overview

The custom operations system consists of several key components:

1. **CustomOp Trait**: Defines the interface for custom operations
2. **CustomOpRegistry**: Manages registration and execution of custom operations
3. **CustomOpBuilder**: Provides a convenient builder pattern for creating operations
4. **Automatic Differentiation Integration**: Seamless integration with the gradient computation system
5. **Python Bindings**: Full Python API for registering and executing custom operations

## Core Components

### CustomOp Trait

The `CustomOp` trait defines the interface that all custom operations must implement:

```rust
pub trait CustomOp: Send + Sync {
    fn name(&self) -> &str;
    fn validate_inputs(&self, inputs: &[&Tensor]) -> Result<()>;
    fn forward(&self, inputs: &[&Tensor]) -> Result<Tensor>;
    fn create_gradient_function(&self, inputs: &[&Tensor], output: &Tensor) -> Option<Arc<dyn GradientFunction>>;
    fn num_inputs(&self) -> usize;
    fn output_shape(&self, input_shapes: &[&Shape]) -> Result<Shape>;
    fn output_dtype(&self, input_dtypes: &[DataType]) -> Result<DataType>;
    fn output_device(&self, input_devices: &[&Device]) -> Result<Device>;
}
```

### CustomOpBuilder

The builder pattern provides a convenient way to create custom operations:

```rust
let op = CustomOpBuilder::new("my_operation", 2)
    .forward(|inputs| {
        // Forward pass implementation
        Ok(result_tensor)
    })
    .backward(|grad_output, input_ids, input_shapes, input_dtypes, input_devices| {
        // Backward pass implementation
        Ok(gradients_map)
    })
    .validate(|inputs| {
        // Input validation
        Ok(())
    })
    .build()?;
```

## Example Custom Operations

The library includes several example custom operations to demonstrate the system.
They can be registered from Python via `mt.register_example_custom_ops()`:

### 1. Swish Activation Function

```rust
pub fn create_swish_op() -> Result<Arc<dyn CustomOp>> {
    CustomOpBuilder::new("swish", 1)
        .forward(|inputs| {
            let x = inputs[0];
            let sigmoid_x = activation::sigmoid(x)?;
            arithmetic::mul(x, &sigmoid_x)
        })
        .backward(|grad_output, input_ids, input_shapes, input_dtypes, input_devices| {
            // Swish gradient implementation
            let mut gradients = HashMap::new();
            // ... gradient computation logic
            Ok(gradients)
        })
        .build()
}
```

### 2. GELU Activation Function

```rust
pub fn create_gelu_op() -> Result<Arc<dyn CustomOp>> {
    CustomOpBuilder::new("gelu", 1)
        .forward(|inputs| {
            let x = inputs[0];
            // GELU implementation using existing operations
            let tanh_x = activation::tanh(x)?;
            let one = Tensor::ones(x.shape().clone(), x.dtype(), x.device(), false);
            let one_plus_tanh = arithmetic::add(&one, &tanh_x)?;
            arithmetic::mul(x, &one_plus_tanh)
        })
        .build()
}
```

### 3. Element-wise Power Operation

```rust
pub fn create_power_op() -> Result<Arc<dyn CustomOp>> {
    CustomOpBuilder::new("power", 2)
        .forward(|inputs| {
            let base = inputs[0];
            let exponent = inputs[1];
            // Power operation implementation
            arithmetic::mul(base, exponent) // Simplified
        })
        .validate(|inputs| {
            if inputs[0].shape() != inputs[1].shape() {
                return Err(MinitensorError::shape_mismatch(
                    inputs[0].shape().dims().to_vec(),
                    inputs[1].shape().dims().to_vec()
                ));
            }
            Ok(())
        })
        .build()
}
```

## Python API

The custom operations system is fully accessible from Python:

### Registration

```python
import minitensor as mt

# Register example custom operations
mt.register_example_custom_ops()

# List registered operations
ops = mt.list_custom_ops_py()
print("Available operations:", ops)

# Check if an operation is registered
is_registered = mt.is_custom_op_registered_py("swish")
print("Swish registered:", is_registered)
```

### Execution

```python
import minitensor as mt

# Create input tensor
x = mt.Tensor([[1.0, 2.0, -1.0]], requires_grad=True)

# Execute custom operation (result is a core tensor)
res_core = mt.execute_custom_op_py("swish", [x._tensor])
result = mt.Tensor.__new__(mt.Tensor)
result._tensor = res_core
print("Swish result:", result)
```

### Advanced Usage

```python
# Power operation with two inputs
base = mt.Tensor([[2.0, 3.0], [4.0, 5.0]], requires_grad=True)
exponent = mt.Tensor([[2.0, 2.0], [3.0, 2.0]], requires_grad=True)

power_core = mt.execute_custom_op_py("power", [base._tensor, exponent._tensor])
power = mt.Tensor.__new__(mt.Tensor)
power._tensor = power_core
print("Power result:", power)

# Layer normalization with three inputs
input_tensor = mt.Tensor([[1.0, 2.0, 3.0, 4.0]], requires_grad=True)
weight = mt.Tensor([1.0, 1.0, 1.0, 1.0], requires_grad=True)
bias = mt.Tensor([0.0, 0.0, 0.0, 0.0], requires_grad=True)

norm_core = mt.execute_custom_op_py(
    "layer_norm", [input_tensor._tensor, weight._tensor, bias._tensor]
)
normalized = mt.Tensor.__new__(mt.Tensor)
normalized._tensor = norm_core
print("Layer norm result:", normalized)
```

## Features

### 1. Automatic Differentiation Integration

Custom operations seamlessly integrate with the automatic differentiation system:

- Operations can define custom gradient functions
- Gradients flow through custom operations just like built-in operations
- Support for complex gradient computations with multiple inputs/outputs

### 2. Type Safety and Validation

The system provides comprehensive validation:

- Input tensor count validation
- Shape compatibility checking
- Data type validation
- Device compatibility verification
- Custom validation logic support

### 3. Performance Optimization

Custom operations are designed for performance:

- Zero-copy tensor passing where possible
- Efficient memory management
- Integration with SIMD and GPU backends
- Lazy evaluation support

### 4. Error Handling

Robust error handling throughout the system:

- Clear error messages with actionable suggestions
- Proper error propagation from Rust to Python
- Validation errors with detailed context
- Runtime error detection and reporting

## Best Practices

### 1. Operation Design

- Keep operations focused and composable
- Use descriptive names that avoid conflicts
- Implement proper input validation
- Consider numerical stability

### 2. Gradient Implementation

- Ensure gradient correctness through testing
- Handle edge cases (zeros, infinities, NaNs)
- Consider gradient clipping for stability
- Test gradient computation with finite differences

### 3. Performance Considerations

- Minimize memory allocations in hot paths
- Reuse tensors where possible
- Consider vectorization opportunities
- Profile custom operations for bottlenecks

### 4. Testing

- Test with various input shapes and types
- Verify gradient correctness
- Test error conditions and edge cases
- Include performance benchmarks

## Future Extensions

The custom operations system is designed to be extensible:

1. **Plugin System**: Load operations from external libraries
2. **JIT Compilation**: Compile custom operations for better performance
3. **Graph Optimization**: Fuse custom operations with built-in operations
4. **Distributed Operations**: Support for multi-device custom operations
5. **Serialization**: Save and load custom operations with models
