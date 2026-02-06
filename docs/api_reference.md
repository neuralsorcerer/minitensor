# MiniTensor API Reference

This document consolidates the public MiniTensor surface area available through
`minitensor` and its submodules, using the Python bindings and the Rust engine
as the source of truth. It is intentionally exhaustive and meant to complement
existing guides such as `custom_operations.md`, `plugin_system.md`, and
`performance.md`.

## 1) Top-level module (`minitensor`)

### Core exports

MiniTensor’s top-level module re-exports the Rust-backed core API and a handful
of convenience aliases.

| Export | Description |
| --- | --- |
| `Tensor` / `tensor` | Core tensor type (constructor + alias). |
| `Device` / `device` | Device handle type (CPU/GPU). |
| `cpu`, `cuda` | Convenience constructors for CPU/GPU devices. |
| `functional` | Functional API module (stateless ops). |
| `nn` | Neural network modules and losses. |
| `optim` | Optimizers. |
| `numpy_compat` | NumPy-style helpers (if built). |
| `plugins` | Plugin registry and utilities (if built). |
| `serialization` | Model serialization utilities (if built). |

### Versioning

- `__version__` reflects the backend version exposed by the Rust core (if
  available) or a default fallback version.
- `__version_tuple__` mirrors the structured version tuple.

### Global configuration & graph controls

| Function | Purpose |
| --- | --- |
| `get_default_dtype()` | Return the global default dtype string. |
| `set_default_dtype(dtype)` | Set the global default dtype. |
| `default_dtype(dtype)` | Context manager for temporary dtype overrides. |
| `manual_seed(seed)` | Seed the RNG used by random ops. |
| `get_gradient(tensor)` | Access a tensor’s gradient in the global graph. |
| `clear_autograd_graph()` | Clear the global autograd graph. |
| `is_autograd_graph_consumed()` | Inspect whether a graph has been consumed. |
| `mark_autograd_graph_consumed()` | Mark the current graph as consumed. |
| `available_submodules()` | Return availability of optional submodules. |
| `list_public_api()` | Return public API symbol lists by module. |
| `api_summary()` | Return version and API counts by module. |
| `search_api(query, module=None)` | Search available symbols by name. |
| `describe_api(symbol)` | Return a one-line description for a symbol. |
| `help()` | Render a formatted MiniTensor API reference. |

### Custom operations (Python API)

The custom-ops system is exposed at the top level:

- `execute_custom_op_py(name, inputs)`
- `is_custom_op_registered_py(name)`
- `list_custom_ops_py()`
- `register_example_custom_ops()`
- `unregister_custom_op_py(name)`

## 2) Tensor creation API

Every creation helper is available as either `mt.<name>(...)` or
`Tensor.<name>(...)`.

### Random + distribution-based

- `rand`, `rand_like`
- `randn`, `randn_like`
- `truncated_normal`, `truncated_normal_like`
- `uniform`, `uniform_like`
- `randint`, `randint_like`
- `randperm`

### Initialization schemes

- `xavier_uniform`, `xavier_uniform_like`
- `xavier_normal`, `xavier_normal_like`
- `he_uniform`, `he_uniform_like`
- `he_normal`, `he_normal_like`
- `lecun_uniform`, `lecun_uniform_like`
- `lecun_normal`, `lecun_normal_like`

### Deterministic / structured

- `zeros`, `zeros_like`
- `ones`, `ones_like`
- `empty`, `empty_like`
- `full`, `full_like`
- `eye`
- `arange`
- `linspace`
- `logspace`

### NumPy interop

- `from_numpy(array)`
- `from_numpy_shared(array)`
- `as_tensor(obj, dtype=None, requires_grad=None, copy=False)`

## 3) Tensor properties & conversion helpers

Frequently used tensor attributes:

- `tensor.shape` / `tensor.ndim`
- `tensor.dtype`
- `tensor.device`
- `tensor.requires_grad`

Conversion helpers:

- `tensor.numpy()` → NumPy array
- `tensor.item()` → Python scalar (for 0-d tensors)
- `tensor.tolist()` → Python list
- `tensor.astype(dtype)` → dtype conversion

## 4) Tensor instance methods (non-exhaustive but observable in tests)

The following instance methods are exercised by the test suite and are available
on `Tensor` objects (many also have functional/top-level equivalents):

### Shape and layout

- `reshape`, `view`, `transpose`, `permute`
- `movedim`, `moveaxis`, `swapaxes`, `swapdims`
- `squeeze`, `unsqueeze`, `expand`
- `flatten`, `ravel`

### Indexing & reordering

- `index_select`, `gather`, `narrow`
- `flip`, `roll`

### Linear algebra & matrix ops

- `dot`, `bmm`
- `solve`
- `diagonal`, `trace`
- `triu`, `tril`

### Reductions & statistics

- `sum`, `mean`, `median`, `quantile`, `nanquantile`
- `nansum`, `nanmean`, `nanmax`, `nanmin`
- `logsumexp`

### Elementwise math & activation

- `softmax`, `log_softmax`
- `softsign`, `rsqrt`, `reciprocal`, `sign`
- `clip`, `clamp`, `clamp_min`, `clamp_max`
- `round`, `floor`, `ceil`
- `sin`, `cos`, `tan`
- `asin`, `acos`, `atan`
- `sinh`, `cosh`, `asinh`, `acosh`, `atanh`
- `softplus`, `gelu`, `elu`, `selu`, `silu`
- `hardshrink`

### Normalization

- `layer_norm(shape, weight=None, bias=None, eps=1e-5)`

### Autograd + in-place

- `backward()` to trigger gradient computation.
- `fill_(value)` for in-place fills.

## 5) Functional API (`minitensor.functional`)

MiniTensor provides stateless functional variants that mirror `Tensor` methods.

### Forwarders exported at top level

Each of the following names is accessible from:

- `minitensor.<name>`
- `minitensor.functional.<name>`

```
cat, stack, split, chunk, index_select, gather, narrow, topk, sort, argsort,
median, quantile, nanquantile, nansum, nanmean, nanmax, nanmin, logsumexp,
softmax, log_softmax, masked_softmax, masked_log_softmax, softsign, rsqrt,
reciprocal, sign, reshape, view, triu, tril, diagonal, trace, solve, flatten,
ravel, transpose, permute, movedim, moveaxis, swapaxes, swapdims, squeeze,
unsqueeze, expand, repeat, repeat_interleave, flip, roll, clip, clamp,
clamp_min, clamp_max, round, floor, ceil, sin, cos, tan, asin, acos, atan,
sinh, cosh, asinh, acosh, atanh, where, masked_fill
```

### Tensor-centric math helpers

The `functional` namespace also exposes:

- `dot`
- `bmm`

### Cross-pollination with `nn`

Lower-case callable symbols from `minitensor.nn` are mirrored into
`minitensor.functional` for convenience (for example, activation functions that
have a functional signature).

## 6) Neural network module (`minitensor.nn`)

### Layers & containers

- `Module` (base class)
- `DenseLayer`
- `Conv2d`
- `BatchNorm1d`
- `BatchNorm2d`
- `Dropout`, `Dropout2d`
- `Sequential` (container of modules)

### Activations

- `ReLU`
- `LeakyReLU`
- `Sigmoid`
- `Tanh`
- `GELU`
- `ELU`
- `Softmax`

### Losses

- `MSELoss`
- `MAELoss`
- `HuberLoss`
- `LogCoshLoss`
- `SmoothL1Loss`
- `CrossEntropyLoss`
- `BCELoss`
- `FocalLoss`

### Common utilities

- `layer.parameters()` returns tensors for optimizers.
- `layer.zero_grad()` clears gradients for trainable tensors.

## 7) Optimizers (`minitensor.optim`)

### Built-in optimizers

- `SGD`
- `Adam`
- `AdamW`
- `RMSprop`

### Base optimizer API

All optimizer classes share a common interface:

- `step()` — apply parameter updates and clear the global autograd graph.
- `zero_grad(set_to_none: bool = False)` — reset gradients.
- `lr` property — read/write learning rate.

## 8) NumPy compatibility module (`minitensor.numpy_compat`)

### Array creation

- `asarray(data, dtype=None, requires_grad=False)`
- `zeros_like`, `ones_like`, `empty_like`, `full_like`

### Array manipulation

- `concatenate`, `stack`, `vstack`, `hstack`
- `split`, `hsplit`, `vsplit`

### Math & comparisons

- `dot`, `matmul`, `cross`, `where`
- `allclose`, `array_equal`

### Statistics

- `mean`, `nanmean`, `std`, `var`, `prod`, `sum`, `nansum`
- `max`, `min`, `nanmax`, `nanmin`

## 9) Serialization (`minitensor.serialization`)

### Core types

- `ModelVersion` — semantic version for serialized models.
- `ModelMetadata` — name, description, architecture, shapes, custom metadata.
- `SerializationFormat` — `json()`, `binary()`, `messagepack()`.
- `SerializedModel` — metadata + state dict.
- `StateDict` — tensor parameters/buffers.
- `DeploymentModel` — compact model format for inference.
- `ModelSerializer` — `save()` / `load()` helpers.

### Convenience functions

- `save_model(model, path, format=None)`
- `load_model(path, format=None)`

## 10) Plugin system (`minitensor.plugins`)

### Versioning and metadata

- `VersionInfo` (parse, current, compatibility checks)
- `PluginInfo` (name, version, author, min/max supported versions)

### Python-side plugins

- `CustomPlugin` — plugin object with init/cleanup/custom-op callbacks.
- `PluginRegistry` — register/unregister/list Python plugins.
- `CustomLayer` — define custom layers in Python.
- `PluginBuilder` — fluent builder for plugin metadata.

### Dynamic loading (if compiled)

- `load_plugin(path)`
- `unload_plugin(name)`
- `list_plugins()`
- `get_plugin_info(name)`
- `is_plugin_loaded(name)`

## 11) Custom operations

MiniTensor supports custom ops in both Rust and Python. Refer to
`docs/custom_operations.md` for:

- The `CustomOp` trait and builder pattern.
- Python registration and execution (`execute_custom_op_py`, etc.).
- Example custom ops (Swish, GELU, power).

## 12) Notes on devices & backends

The core engine supports CPU execution and can be compiled with CUDA, Metal, or
OpenCL backends where applicable. Device selection flows through the `Device`
API and tensor creation functions.

## 13) Where to go next

- `docs/custom_operations.md` — custom ops and autograd integration.
- `docs/plugin_system.md` — plugin registry and compatibility handling.
- `docs/performance.md` — performance tuning and profiling.
- `examples/` and `examples/notebooks/` — end-to-end usage patterns.
