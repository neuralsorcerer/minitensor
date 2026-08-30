# Plugin System Documentation

MiniTensor's plugin module provides version metadata, Python-side plugin
registries, lightweight custom-layer wrappers, and optional native
shared-library loading. It is separate from the operation registry described in
[the custom operations guide](custom_operations.md), and answers a different
question: a plugin is a *bundle* -- a name, a version, a compatibility range,
and setup and teardown callbacks -- where an operation is a single function with
a gradient. Reach for `register_custom_op` to add one operation; reach for a
plugin to ship a set of them together with the metadata that says which
MiniTensor versions they work with.

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
`NotImplementedError`; with it, a missing or incompatible library raises
`RuntimeError` from `dlopen`. Both are worth distinguishing — the first means
"rebuild the extension", the second means "check the plugin".

```python
import minitensor.plugins as plugins

try:
    plugins.load_plugin("./my_plugin.so")
except NotImplementedError:
    print("This MiniTensor build does not enable dynamic plugin loading")
except RuntimeError as exc:
    print(f"Plugin could not be loaded: {exc}")
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

The plugin manager owns registration of the operations a plugin declares: it
registers everything `custom_operations()` returns once `initialize` succeeds,
and unregisters them after `cleanup` on unload. So `initialize` must not
register those operations itself — that is a double registration and the load
fails with `Operation '<name>' is already registered`. `initialize` receives the
registry so a plugin *can* add operations beyond the ones it declares; those it
must unregister itself in `cleanup`.

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
- Return the same operation names from `custom_operations()` on every call — the
  manager uses it both to register on load and to unregister on unload.
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

## Verifying the plugin path

Plugin loading needs both halves built, and the test suite skips itself if
either is missing:

```bash
cargo build --release --manifest-path examples/rust_plugin_example/Cargo.toml
maturin develop --release --features dynamic-loading
pytest tests/test_plugin_loading.py
```

Without those two commands `tests/test_plugin_loading.py` reports eleven
skipped tests, not eleven passing ones. CI runs this as its own job
(`dynamic plugin loading` in test_ubuntu.yml) and treats a skip as a failure,
because a green run that exercised nothing is what let this path go unverified
in the first place.
