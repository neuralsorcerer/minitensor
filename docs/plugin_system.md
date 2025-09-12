# Plugin System Documentation

The minitensor plugin system allows you to extend the library with custom operations, layers, and functionality while maintaining safety and version compatibility.

## Overview

The plugin system provides:

- **Safe Plugin Loading**: Version compatibility checking and error isolation
- **Custom Operations**: Extend minitensor with your own tensor operations
- **Custom Layers**: Create new neural network layers that integrate seamlessly
- **Python Integration**: Write plugins in Python or load compiled extensions
- **Version Management**: Automatic compatibility checking between plugins and minitensor

## Core Concepts

### Version Compatibility

Plugins specify minimum and maximum minitensor versions they support. The system automatically checks compatibility when loading plugins:

```python
import minitensor.plugins as plugins

# Check current minitensor version
current_version = plugins.VersionInfo.current()
print(f"Minitensor version: {current_version}")

# Create version info
plugin_version = plugins.VersionInfo(1, 0, 0)
min_required = plugins.VersionInfo(0, 1, 0)

# Check compatibility
if current_version.is_compatible_with(min_required):
    print("Compatible!")
```

### Plugin Information

Every plugin must provide metadata:

```python
import minitensor.plugins as plugins

plugin = (
    plugins.PluginBuilder()
    .name("my_custom_plugin")
    .version(plugins.VersionInfo(1, 0, 0))
    .description("A custom plugin for special operations")
    .author("Your Name")
    .min_minitensor_version(plugins.VersionInfo(0, 1, 0))
    .build()
)
info = plugin.info
print(f"Plugin: {info.name} v{info.version}")
```

## Creating Python Plugins

### Basic Plugin Structure

```python
import minitensor as mt
import minitensor.plugins as plugins

# Create a plugin using the builder pattern
plugin = (plugins.PluginBuilder()
    .name("example_plugin")
    .version(plugins.VersionInfo(1, 0, 0))
    .description("An example plugin demonstrating custom operations")
    .author("Plugin Developer")
    .min_minitensor_version(plugins.VersionInfo(0, 1, 0))
    .build())

# Define initialization function
def initialize_plugin(registry):
    print("Plugin initialized!")
    # Register custom operations here
    pass

# Define cleanup function
def cleanup_plugin(registry):
    print("Plugin cleaned up!")
    # Unregister operations here
    pass

# Define custom operations
def get_custom_operations():
    # Return list of custom operations
    return []

# Set plugin functions
plugin.set_initialize_fn(initialize_plugin)
plugin.set_cleanup_fn(cleanup_plugin)
plugin.set_custom_operations_fn(get_custom_operations)
```

### Custom Operations in Plugins

```python
import minitensor as mt
import minitensor.plugins as plugins

def create_square_operation():
    """Create a custom square operation"""

    def forward(inputs):
        # Forward pass: square the input
        x = inputs[0]
        return x * x

    def backward(grad_output, input_ids, input_shapes, input_dtypes, input_devices):
        # Backward pass: gradient of x^2 is 2x
        gradients = {}
        if input_ids:
            gradients[input_ids[0]] = grad_output * 2
        return gradients

    def validate(inputs):
        if len(inputs) != 1:
            raise ValueError("square operation requires a single input")

    # Create the custom operation
    op = (
        mt.CustomOpBuilder("square", 1)
        .forward(forward)
        .backward(backward)
        .validate(validate)
        .build()
    )

    return op

# Plugin with custom operation
def create_math_plugin():
    plugin = (plugins.PluginBuilder()
        .name("math_extensions")
        .version(plugins.VersionInfo(1, 0, 0))
        .description("Additional mathematical operations")
        .author("Math Team")
        .min_minitensor_version(plugins.VersionInfo(0, 1, 0))
        .build())

    def get_operations():
        return [create_square_operation()]

    plugin.set_custom_operations_fn(get_operations)
    return plugin
```

## Plugin Management

### Loading and Managing Plugins

```python
import minitensor.plugins as plugins

# Create plugin registry
registry = plugins.PluginRegistry()

# Register a plugin
plugin = create_math_plugin()
registry.register(plugin)

# List registered plugins
for plugin_info in registry.list_plugins():
    print(f"Plugin: {plugin_info.name} v{plugin_info.version}")
    print(f"  Author: {plugin_info.author}")
    print(f"  Description: {plugin_info.description}")

# Check if plugin is registered
if registry.is_registered("math_extensions"):
    print("Math extensions plugin is available")

# Get specific plugin
math_plugin = registry.get_plugin("math_extensions")

# Unregister plugin
registry.unregister("math_extensions")
```

### Global Plugin Management

```python
import minitensor.plugins as plugins

# Load plugin from shared library (requires dynamic-loading feature)
try:
    plugins.load_plugin("./my_plugin.so")
    print("Plugin loaded successfully")
except Exception as e:
    print(f"Failed to load plugin: {e}")

# List all loaded plugins
for plugin_info in plugins.list_plugins():
    print(f"Loaded: {plugin_info}")

# Get plugin information
try:
    info = plugins.get_plugin_info("my_plugin")
    print(f"Plugin info: {info}")
except Exception as e:
    print(f"Plugin not found: {e}")

# Check if plugin is loaded
if plugins.is_plugin_loaded("my_plugin"):
    print("Plugin is active")

# Unload plugin
try:
    plugins.unload_plugin("my_plugin")
    print("Plugin unloaded")
except Exception as e:
    print(f"Failed to unload: {e}")
```

## Dynamic Plugin Loading (C/C++/Rust)

For compiled plugins, you need to implement the plugin interface in your native code:

### Rust Plugin Example

```rust
// plugin_example/src/lib.rs
use minitensor_engine::{Plugin, PluginInfo, VersionInfo, CustomOp, CustomOpRegistry, Result};
use std::sync::Arc;

pub struct ExamplePlugin {
    info: PluginInfo,
}

impl ExamplePlugin {
    pub fn new() -> Self {
        Self {
            info: PluginInfo {
                name: "example_rust_plugin".to_string(),
                version: VersionInfo::new(1, 0, 0),
                description: "Example Rust plugin".to_string(),
                author: "Rust Developer".to_string(),
                min_minitensor_version: VersionInfo::new(0, 1, 0),
                max_minitensor_version: None,
            },
        }
    }
}

impl Plugin for ExamplePlugin {
    fn info(&self) -> &PluginInfo {
        &self.info
    }

    fn initialize(&self, _registry: &CustomOpRegistry) -> Result<()> {
        println!("Rust plugin initialized!");
        Ok(())
    }

    fn cleanup(&self, _registry: &CustomOpRegistry) -> Result<()> {
        println!("Rust plugin cleaned up!");
        Ok(())
    }

    fn custom_operations(&self) -> Vec<Arc<dyn CustomOp>> {
        // Return your custom operations here
        vec![]
    }
}

// Export function for dynamic loading
#[no_mangle]
pub extern "C" fn create_plugin() -> *mut dyn Plugin {
    let plugin = ExamplePlugin::new();
    Box::into_raw(Box::new(plugin))
}
```

### Building the Plugin

```toml
# plugin_example/Cargo.toml
[package]
name = "plugin_example"
version = "0.1.0"
edition = "2024"

[lib]
crate-type = ["cdylib"]

[dependencies]
minitensor-engine = { path = "../engine" }
```

Build with:

```bash
cargo build --release
```

## Best Practices

### Plugin Development

1. **Version Compatibility**: Always specify minimum and maximum supported versions
2. **Error Handling**: Provide clear error messages and handle failures gracefully
3. **Documentation**: Document your plugin's operations and usage
4. **Testing**: Include comprehensive tests for your plugin functionality
5. **Performance**: Profile your operations and optimize for common use cases

### Safety Considerations

1. **Input Validation**: Always validate tensor inputs in custom operations
2. **Memory Management**: Ensure proper cleanup in plugin cleanup functions
3. **Thread Safety**: Make sure your operations are thread-safe
4. **Error Isolation**: Don't let plugin errors crash the main application

### Example Plugin Structure

```
my_plugin/
├── src/
│   ├── lib.rs          # Main plugin implementation
│   ├── operations.rs   # Custom operations
│   └── layers.rs       # Custom layers
├── tests/
│   └── integration.rs  # Plugin tests
├── examples/
│   └── usage.py        # Usage examples
├── Cargo.toml          # Rust configuration
├── README.md           # Plugin documentation
└── plugin.json         # Plugin metadata
```

## Troubleshooting

### Common Issues

1. **Version Mismatch**: Ensure your plugin supports the current minitensor version
2. **Loading Failures**: Check that shared libraries are in the correct format
3. **Missing Dependencies**: Verify all required dependencies are available
4. **Permission Errors**: Ensure proper file permissions for plugin files

### Debugging

```python
import minitensor.plugins as plugins

# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Try loading with error handling
try:
    plugins.load_plugin("./my_plugin.so")
except Exception as e:
    print(f"Error details: {e}")
    # Check plugin compatibility
    current = plugins.VersionInfo.current()
    print(f"Current minitensor version: {current}")
```

## API Reference

### Classes

- `VersionInfo`: Version information and compatibility checking
- `PluginInfo`: Plugin metadata and information
- `CustomPlugin`: Python plugin implementation
- `PluginRegistry`: Plugin registration and management
- `CustomLayer`: Base class for custom neural network layers
- `PluginBuilder`: Builder pattern for creating plugins

### Functions

- `load_plugin(path)`: Load plugin from shared library
- `unload_plugin(name)`: Unload plugin by name
- `list_plugins()`: List all loaded plugins
- `get_plugin_info(name)`: Get plugin information
- `is_plugin_loaded(name)`: Check if plugin is loaded

### Features

- `dynamic-loading`: Enable dynamic plugin loading from shared libraries

This plugin system provides a powerful and safe way to extend minitensor while maintaining compatibility and performance.
