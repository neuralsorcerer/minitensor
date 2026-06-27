# Plugin System Documentation

MiniTensor's plugin module provides version metadata, Python-side plugin
registries, lightweight custom-layer wrappers, and optional native shared-library
loading. It is separate from the Rust custom-operation registry described in
[the custom operations guide](custom_operations.md): the current Python plugin
API can store callbacks and metadata, but it does not expose a Python API that
turns arbitrary Python functions into engine-level tensor kernels.

## Core concepts

### Version compatibility

Plugins use semantic-version-like `VersionInfo` values. A plugin can declare the
minimum supported MiniTensor version and, optionally, a maximum supported
version.

```python
import minitensor.plugins as plugins

current_version = plugins.VersionInfo.current()
minimum = plugins.VersionInfo(0, 1, 0)

if current_version.is_compatible_with(minimum):
    print(f"MiniTensor {current_version} satisfies the minimum requirement")
```

### Plugin metadata

`PluginBuilder` creates a `CustomPlugin` after all required metadata fields are
provided.

```python
import minitensor.plugins as plugins

plugin = (
    plugins.PluginBuilder()
    .name("my_custom_plugin")
    .version(plugins.VersionInfo(1, 0, 0))
    .description("A custom plugin for project-specific extensions")
    .author("Your Name")
    .min_minitensor_version(plugins.VersionInfo(0, 1, 0))
    .build()
)

info = plugin.info
print(info.name, info.version, info.author)
```

If `name`, `version`, `description`, `author`, or `min_minitensor_version` is
missing, `build()` raises `ValueError`.

## Python-side plugins

A `CustomPlugin` can hold Python callbacks for initialization, cleanup, and a
custom-operations list. `PluginRegistry` stores these Python plugin objects by
name and rejects duplicate registrations.

```python
import minitensor.plugins as plugins

plugin = (
    plugins.PluginBuilder()
    .name("example_plugin")
    .version(plugins.VersionInfo(1, 0, 0))
    .description("Demonstrates Python plugin metadata and callbacks")
    .author("Plugin Developer")
    .min_minitensor_version(plugins.VersionInfo(0, 1, 0))
    .build()
)


def initialize_plugin(registry):
    print("Plugin initialized")


def cleanup_plugin(registry):
    print("Plugin cleaned up")


def get_custom_operations():
    # The current Python API stores this callback but does not automatically
    # convert Python callables into Rust-engine CustomOp registrations.
    return []


plugin.set_initialize_fn(initialize_plugin)
plugin.set_cleanup_fn(cleanup_plugin)
plugin.set_custom_operations_fn(get_custom_operations)

registry = plugins.PluginRegistry()
registry.register(plugin)
assert registry.is_registered("example_plugin")
print(registry.get_plugin("example_plugin").info)
registry.unregister("example_plugin")
```

## Custom layers in Python

`CustomLayer` is a small Python-callable wrapper. It stores named parameters and
calls a user-provided forward function with the input list supplied to
`forward(...)`.

```python
import minitensor as mt
import minitensor.plugins as plugins

layer = plugins.CustomLayer("scale")
layer.add_parameter("weight", mt.Tensor([2.0]))


def forward(inputs):
    x = inputs[0]
    weight = layer.get_parameter("weight")
    return x * weight


layer.set_forward(forward)
out = layer.forward([mt.Tensor([3.0])])
print(out)
```

If no forward function is set, `forward(...)` raises `NotImplementedError`. If a
parameter name is missing, `get_parameter(name)` raises `KeyError`.

## Native dynamic plugin loading

The `plugins.load_plugin(path)` function is available in the Python module, but
it only loads shared libraries when the extension was compiled with the
`dynamic-loading` Cargo feature. Without that feature it raises
`NotImplementedError`.

```python
import minitensor.plugins as plugins

try:
    plugins.load_plugin("./my_plugin.so")
except NotImplementedError:
    print("This MiniTensor build does not enable dynamic plugin loading")
```

Other global helpers delegate to the Rust engine's native plugin registry:

- `list_plugins()` returns loaded native plugin metadata.
- `get_plugin_info(name)` returns metadata for one loaded native plugin.
- `is_plugin_loaded(name)` checks native registry membership.
- `unload_plugin(name)` unloads a native plugin by name.

## Native Rust plugin shape

A compiled plugin implements the Rust `Plugin` trait and exports a constructor
symbol. The exact ABI is controlled by the engine crate and by whether the
consumer build enables dynamic loading.

```rust
use minitensor_engine::{CustomOp, CustomOpRegistry, Plugin, PluginInfo, Result, VersionInfo};
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
        Ok(())
    }

    fn cleanup(&self, _registry: &CustomOpRegistry) -> Result<()> {
        Ok(())
    }

    fn custom_operations(&self) -> Vec<Arc<dyn CustomOp>> {
        vec![]
    }
}

#[no_mangle]
pub extern "C" fn create_plugin() -> *mut dyn Plugin {
    Box::into_raw(Box::new(ExamplePlugin::new()))
}
```

A typical Cargo manifest uses a `cdylib` crate type and depends on the engine
crate from an appropriate path or published package:

```toml
[lib]
crate-type = ["cdylib"]

[dependencies]
minitensor-engine = { path = "../engine" }
```

## Best practices

- Treat plugin names as globally unique registry keys.
- Declare realistic minimum and maximum supported MiniTensor versions.
- Keep initialization and cleanup idempotent where possible.
- Validate tensor shapes, dtypes, and devices in Rust `CustomOp` code.
- Test duplicate registration, missing plugin names, missing parameters, and
  builds without the `dynamic-loading` feature.
- Document clearly whether code is a Python metadata plugin, a `CustomLayer`, or
  a native plugin that can register engine custom operations.

## API reference

### Classes

- `VersionInfo(major, minor, patch)` with `parse(...)`, `current()`,
  `is_compatible_with(...)`, and read-only `major`, `minor`, `patch` fields.
- `PluginInfo` with `name`, `version`, `description`, `author`,
  `min_minitensor_version`, and optional `max_minitensor_version`.
- `CustomPlugin` with `set_initialize_fn(...)`, `set_cleanup_fn(...)`,
  `set_custom_operations_fn(...)`, and `info`.
- `PluginRegistry` with `register(...)`, `unregister(...)`, `list_plugins()`,
  `get_plugin(...)`, and `is_registered(...)`.
- `CustomLayer` with `set_forward(...)`, `add_parameter(...)`,
  `get_parameter(...)`, `list_parameters()`, `name`, and `forward(...)`.
- `PluginBuilder` with fluent metadata setters and `build()`.

### Functions

- `load_plugin(path)`
- `unload_plugin(name)`
- `list_plugins()`
- `get_plugin_info(name)`
- `is_plugin_loaded(name)`
