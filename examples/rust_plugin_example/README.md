# Rust Plugin Example

This is an example of how to create a minitensor plugin in Rust that can be dynamically loaded.

## Features

This plugin provides three custom operations:

1. **rust_abs**: Element-wise absolute value operation
2. **rust_clamp**: Clamp values between min and max bounds
3. **rust_gelu**: GELU (Gaussian Error Linear Unit) activation function

## Building

To build the plugin as a shared library:

```bash
cargo build --release
```

This will create a shared library file:

- Linux: `target/release/librust_plugin_example.so`
- macOS: `target/release/librust_plugin_example.dylib`
- Windows: `target/release/rust_plugin_example.dll`

## Usage

### Loading the Plugin

```python
import minitensor.plugins as plugins

# Load the plugin (requires dynamic-loading feature)
plugins.load_plugin("./target/release/librust_plugin_example.so")

# Check if loaded
if plugins.is_plugin_loaded("rust_example_plugin"):
    print("Plugin loaded successfully!")

# List plugin info
info = plugins.get_plugin_info("rust_example_plugin")
print(f"Plugin: {info.name} v{info.version}")
print(f"Author: {info.author}")
print(f"Description: {info.description}")
```

### Using Custom Operations

```python
import minitensor as mt

# Create test tensor
x = mt.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])

# Use absolute value operation
abs_result = mt.execute_custom_op("rust_abs", [x])
print(f"abs({x}) = {abs_result}")

# Use clamp operation
min_val = mt.tensor([-1.0])
max_val = mt.tensor([1.0])
clamp_result = mt.execute_custom_op("rust_clamp", [x, min_val, max_val])
print(f"clamp({x}, -1, 1) = {clamp_result}")

# Use GELU activation
gelu_result = mt.execute_custom_op("rust_gelu", [x])
print(f"gelu({x}) = {gelu_result}")
```

### Unloading the Plugin

```python
# Unload when done
plugins.unload_plugin("rust_example_plugin")
```

## Plugin Interface

The plugin implements the `Plugin` trait with the following methods:

- `info()`: Returns plugin metadata
- `initialize()`: Called when the plugin is loaded
- `cleanup()`: Called when the plugin is unloaded
- `custom_operations()`: Returns list of custom operations provided

## Custom Operations

Each custom operation implements the `CustomOp` trait:

- `name()`: Operation name
- `validate_inputs()`: Input validation
- `forward()`: Forward pass computation
- `create_gradient_function()`: Backward pass for automatic differentiation
- `num_inputs()`: Expected number of inputs
- `output_shape()`: Output shape computation
- `output_dtype()`: Output data type
- `output_device()`: Output device

## Safety Considerations

- All operations validate their inputs
- Memory management is handled safely through Rust's ownership system
- Error handling provides clear messages
- The plugin isolates errors to prevent crashes

## Testing

Run the plugin tests:

```bash
cargo test
```

## Development Notes

- The plugin uses simplified implementations for demonstration
- Real-world plugins should use optimized kernels for performance
- Gradient computations are simplified - production code needs proper derivatives
- Consider using SIMD instructions for better performance
- Add comprehensive error handling and input validation
