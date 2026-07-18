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
| `minitensor.tensor` | Compatibility module containing tensor constructors and dtype helpers. |

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
| `no_grad()` | Context manager: disable gradient recording (results are detached leaves; nothing is saved for backward). |
| `enable_grad()` | Context manager: re-enable gradient recording inside a `no_grad()` block. |
| `is_grad_enabled()` | Query the thread-local gradient recording mode. |
| `set_grad_enabled(enabled)` | Set the gradient recording mode, returning the previous mode. |
| `available_submodules()` | Return availability of optional submodules. |
| `list_public_api()` | Return public API symbol lists by module. |
| `api_summary()` | Return version and API counts by module. |
| `search_api(query, module=None)` | Search available symbols by name. |
| `describe_api(symbol)` | Return a one-line description for a symbol. |
| `help()` | Render a formatted MiniTensor API reference. |
| `broadcast_to(input, shape)` | Broadcast one tensor-like input to an explicit target shape. |
| `broadcast_shapes(*shapes)` | Compute the NumPy/PyTorch-style broadcast result for shape-like inputs without constructing tensors. |
| `broadcast_tensors(*inputs)` | Convert tensor-like inputs and broadcast them to a shared shape, returning materialized contiguous tensors. |
| `can_broadcast(*shapes)` | Return whether shape-like inputs are broadcast-compatible. |
| `atleast_1d(*inputs)` | Convert one or more tensor-like inputs to tensors with at least one dimension. |
| `atleast_2d(*inputs)` | Convert one or more tensor-like inputs to tensors with at least two dimensions. |
| `atleast_3d(*inputs)` | Convert one or more tensor-like inputs to tensors with at least three dimensions. |
| `meshgrid(*inputs, indexing="xy", sparse=False, copy=False)` | Build coordinate grids from scalar or 1-D tensor-like coordinates. |

### Shape compatibility helpers

`broadcast_shapes(*shapes)` computes the shape that would result from
NumPy/PyTorch-style broadcasting without creating input tensors. Each argument
may be a non-negative integer-like scalar dimension (including objects with
`__index__`, such as NumPy integer scalars) or an iterable shape such as a
Python tuple/list or `tensor.shape`. Scalar tensor shapes are represented by an
empty iterable, for example `broadcast_shapes((), (2, 3)) == (2, 3)`.

`broadcast_tensors(*inputs)` applies those same compatibility rules to actual
tensor-like inputs. It converts non-`Tensor` inputs with `as_tensor`, computes
the shared target shape once, prepends singleton dimensions when needed, and
expands each input to the target shape, materializing the result with
contiguous storage so every downstream operation behaves identically to a
dense tensor. If a valid broadcast changes a length-one axis to a length-zero
axis, MiniTensor returns a correctly shaped empty tensor preserving the source
dtype, device, and `requires_grad` metadata because that result has no
addressable elements. Inputs that already have the target shape are returned
unchanged.

`broadcast_to(input, shape)` is the single-input counterpart for cases where
the target shape is already known. It uses the same validation and expansion
path as `broadcast_tensors`, so it preserves dtype, device, and
`requires_grad` metadata for zero-sized results and returns the original tensor
unchanged when no broadcast is needed.

Validation and edge cases:

- Boolean dimensions are rejected even though Python `bool` is integer-like.
- Negative dimensions raise `ValueError`; non-integer dimensions raise
  `TypeError`.
- Zero-sized dimensions follow NumPy broadcasting rules: they can broadcast
  with missing dimensions or `1`, but not with another non-one positive size.
- Incompatible shapes raise `ValueError`. Use `can_broadcast(*shapes)` when a
  boolean compatibility check is preferable to exception handling.

Example:

```python
import minitensor as mt

shape = mt.broadcast_shapes((5, 1, 4), (1, 3, 1), (3, 4))
assert shape == (5, 3, 4)
assert mt.broadcast_shapes(mt.zeros(2, 1, 4).shape, (3, 4)) == (2, 3, 4)

row, column, scalar = mt.broadcast_tensors(
    mt.Tensor([[1.0, 2.0, 3.0]]),
    mt.Tensor([[10.0], [20.0]]),
    5.0,
)
assert row.shape == column.shape == scalar.shape == (2, 3)

empty, already_empty = mt.broadcast_tensors(mt.ones(1), mt.ones(0))
assert empty.shape == already_empty.shape == (0,)

column = mt.broadcast_to(mt.Tensor([[1.0], [2.0]]), (2, 3))
assert column.shape == (2, 3)

assert mt.can_broadcast((1, 3), (2, 3))
assert not mt.can_broadcast((2, 3), (4, 3))
```

`meshgrid(*inputs, indexing="xy", sparse=False, copy=False)` constructs coordinate
grids from scalar or one-dimensional tensor-like coordinates. With the default
`indexing="xy"`, the first two output axes are swapped to follow Cartesian
plotting conventions; `indexing="ij"` preserves matrix-indexing order for all
axes. Dense outputs are materialized broadcast grids, while `sparse=True`
returns only reshaped coordinate vectors that can still broadcast together
lazily inside later operations. Set `copy=True` when callers need storage
independent of the returned grid objects. Calling `meshgrid()` with no inputs returns `()`.

Validation and edge cases:

- Each coordinate input must be scalar or one-dimensional after conversion with
  `as_tensor`; higher-rank inputs raise `ValueError`.
- Scalar inputs are promoted to length-one coordinates.
- `indexing` must be either `"xy"` or `"ij"`; invalid strings raise
  `ValueError`, and non-string values raise `TypeError`.
- `sparse` and `copy` must be booleans.
- Empty coordinate vectors are supported. Dense grids involving empty vectors
  follow the same zero-sized broadcast behavior as `broadcast_to`.

Example:

```python
import minitensor as mt

x = mt.Tensor([1.0, 2.0, 3.0])
y = mt.Tensor([10.0, 20.0])

grid_x, grid_y = mt.meshgrid(x, y)
assert grid_x.shape == grid_y.shape == (2, 3)

sparse_x, sparse_y = mt.meshgrid(x, y, indexing="ij", sparse=True)
assert sparse_x.shape == (3, 1)
assert sparse_y.shape == (1, 2)

(singleton,) = mt.meshgrid(5.0, copy=True)
assert singleton.shape == (1,)
```

`atleast_1d(*inputs)`, `atleast_2d(*inputs)`, and `atleast_3d(*inputs)` mirror
NumPy's `atleast_*` shape conventions while returning MiniTensor tensors.
Existing `Tensor` inputs are preserved when they already satisfy the requested
rank; lower-rank inputs use lightweight reshape/unsqueeze operations. Supplying
one input returns a single `Tensor`, while supplying multiple inputs returns a
tuple of tensors in the same order.

Shape rules and validation:

- `atleast_1d` reshapes scalar inputs to `(1,)` and leaves rank-1-or-higher
  tensors unchanged.
- `atleast_2d` reshapes scalars to `(1, 1)`, promotes vectors to row tensors of
  shape `(1, N)`, and leaves matrices and higher-rank tensors unchanged.
- `atleast_3d` reshapes scalars to `(1, 1, 1)`, promotes vectors to
  `(1, N, 1)`, appends one trailing singleton dimension to matrices, and leaves
  rank-3-or-higher tensors unchanged.
- Empty vectors and matrices follow the same shape rules, for example
  `atleast_2d(mt.Tensor([])).shape == (1, 0)`.
- Calling any `atleast_*` helper without inputs raises `TypeError`.

Example:

```python
import minitensor as mt

scalar = mt.atleast_1d(3.5)
row = mt.atleast_2d(mt.Tensor([1.0, 2.0, 3.0]))
matrix_3d = mt.atleast_3d(mt.Tensor([[1.0, 2.0], [3.0, 4.0]]))
first, second = mt.atleast_3d(1.0, mt.zeros(0, 2))

assert scalar.shape == (1,)
assert row.shape == (1, 3)
assert matrix_3d.shape == (2, 2, 1)
assert first.shape == (1, 1, 1)
assert second.shape == (0, 2, 1)
```

### Compatibility tensor module

`minitensor.tensor` is a lightweight compatibility module populated by the
Python package. It exposes `Tensor`, the top-level tensor creation helpers,
`get_default_dtype()`, `set_default_dtype()`, `manual_seed()`, and the
`default_dtype(...)` context manager. Prefer top-level imports in new examples,
but keep this module in mind when maintaining older code that imports from
`minitensor.tensor`.

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
- `from_numpy_shared(array)` — currently copies like `from_numpy`; writes to
  the source array after construction are not visible through the tensor
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
- `float(tensor)` / `int(tensor)` → Python scalar (one-element tensors only;
  `int` truncates, bool converts to 1/0)

Python numeric protocol: tensors support `+`, `-`, `*`, `/`, `//`, `%`, `@`,
`**`, unary `-`/`+`, `abs()`, `~` (bool/int only), the comparison operators,
and the in-place forms (`+=`, `-=`, …), with scalars accepted on either side.

## 4) Tensor instance methods

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

`__getitem__` supports basic indexing (ints, slices with positive steps,
`None`/`np.newaxis`, and `...`/Ellipsis) plus NumPy-style fancy forms:

- **Boolean masks** — `t[mask]` where the mask's shape equals `t`'s leading
  `mask.ndim` dimensions selects the trailing blocks: a full-shape mask
  yields a 1-D tensor of elements, a 1-D mask over a matrix yields rows,
  and a 0-d mask adds a leading axis. Masks may be bool tensors, bool
  ndarrays, or (nested) lists of bools, and selection is differentiable.
- **Integer lists** — `t[[2, 0, -1]]` (or a 1-D int ndarray/tensor) selects
  rows along dim 0 with negative-index wrapping.

`__setitem__` additionally supports `t[mask] = value` where `value` is a
scalar or anything broadcastable to the selection shape
`[n_true] + trailing`; values are cast to the tensor's dtype and written in
place. Masks inside mixed index tuples (e.g. `t[0, mask]`) are not
supported and raise.

### Linear algebra & matrix ops

- `dot`, `bmm`
- `solve`
- `diagonal`, `trace`
- `triu`, `tril`

### Reductions, statistics, and equality

- `sum`, `mean`, `median`, `nanmedian`, `quantile`, `nanquantile`
  (the `nan*` reductions return NaN for all-NaN slices, matching NumPy)
- `std(dim=None, unbiased=True, keepdim=False)`
- `var(dim=None, unbiased=True, keepdim=False)`
- `nansum`, `nanmean`, `nanmax`, `nanmin`
- `logsumexp`
- `isclose(other, rtol=1e-5, atol=1e-8, equal_nan=False)`
- `array_equal(other)`
- `allclose(other, rtol=1e-5, atol=1e-8, equal_nan=False)`

`std` and `var` accept the same dimension forms as multi-axis reductions such
as `sum` and `mean`: `None` reduces all axes, an integer reduces one axis, and
a sequence such as a tuple/list reduces multiple axes. Negative axes are
normalized, duplicate axes are treated as a single axis, and invalid axes raise
`IndexError`. `keepdim=True` preserves reduced axes with length one; otherwise
those axes are removed after the reduction. `unbiased=True` applies the sample
variance correction (`N / (N - 1)`) over the total number of reduced elements,
and reductions with one or fewer samples return `NaN` rather than emitting a
Python warning.

`nanmedian(dim=None, keepdim=False)` is available as a tensor method,
functional helper, and top-level helper. It ignores `NaN` values in floating
point tensors, returns `NaN` for all-NaN or empty reduced slices without
emitting a Python warning, and rejects non-floating tensors. `dim` accepts a
single integer axis or `None`; use `keepdim=True` to preserve the reduced axis
with length one.

Example:

```python
import minitensor as mt

x = mt.arange(24, dtype="float32").reshape(2, 3, 4)
channel_var = x.var(dim=(1, 2), unbiased=False, keepdim=True)
row_std = x.std(dim=-1, unbiased=False)

assert channel_var.shape == (2, 1, 1)
assert row_std.shape == (2, 3)
```

### Elementwise math & activation

- `softmax`, `log_softmax`
- `softsign`, `rsqrt`, `reciprocal`, `sign`
- `isnan`, `isinf`, `isfinite`
- `clip`, `clamp`, `clamp_min`, `clamp_max`
- `round`, `floor`, `ceil`
- `sin`, `cos`, `tan`
- `asin`, `acos`, `atan`
- `sinh`, `cosh`, `asinh`, `acosh`, `atanh`
- `maximum`, `minimum`
- `softplus`, `gelu`, `elu`, `selu`, `silu`
- `hardshrink`
- `floor_divide` / `//` — Python floor division (rounds toward negative
  infinity; integer operands stay integral, integer zero divisors raise,
  not differentiable)
- `remainder` / `%` — Python-style remainder (takes the divisor's sign;
  `a == (a // b) * b + a % b` holds for every dtype; differentiable for
  float dtypes)
- `bitwise_not` / `~` — logical NOT for bool, two's complement NOT for
  ints; rejected for floats

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
median, nanmedian, quantile, nanquantile, nansum, nanmean, nanmax, nanmin, isnan,
isinf, isfinite, nan_to_num, logsumexp, softmax, log_softmax,
masked_softmax, masked_log_softmax, sum, prod,
mean, all, any, max, min, argmax, argmin, cumsum, cumprod, std, var, relu,
hardshrink, sigmoid, softplus, gelu, elu, selu, silu, softsign, tanh,
layer_norm, rsqrt, reciprocal, sign, reshape, view, triu, tril, diagonal,
trace, solve, flatten, ravel, transpose, permute, movedim, moveaxis, swapaxes,
swapdims, squeeze, unsqueeze, expand, repeat, repeat_interleave, flip, roll,
clip, clamp, clamp_min, clamp_max, round, floor, ceil, sin, cos, tan, asin,
acos, atan, sinh, cosh, asinh, acosh, atanh, log1p, expm1, logaddexp, maximum,
minimum, isclose, array_equal, allclose, where, one_hot, bincount, masked_fill
```

### Finite and NaN predicates

`isnan(input)`, `isinf(input)`, and `isfinite(input)` are available as both
top-level helpers (`minitensor.isnan`, `minitensor.isinf`,
`minitensor.isfinite`) and functional helpers (`minitensor.functional.isnan`,
`minitensor.functional.isinf`, `minitensor.functional.isfinite`). They mirror
the corresponding `Tensor.isnan()`, `Tensor.isinf()`, and `Tensor.isfinite()`
methods and always return a boolean tensor with the same shape and device as the
input.

Behavior and validation:

- Floating-point tensors are classified elementwise using the underlying Rust
  floating-point predicates.
- Integer and boolean tensors cannot contain NaN or infinite values, so
  `isnan` and `isinf` return all-false masks for those dtypes.
- Integer and boolean tensors are always finite, so `isfinite` returns an
  all-true mask for those dtypes.
- Empty tensors preserve their empty shape and return an empty boolean mask.
- Predicate outputs do not require gradients.

Example:

```python
import minitensor as mt

x = mt.Tensor([float("nan"), float("inf"), -1.5])

assert mt.isnan(x).tolist() == [True, False, False]
assert mt.functional.isinf(x).tolist() == [False, True, False]
assert mt.isfinite(x).tolist() == [False, False, True]
assert mt.isfinite(mt.Tensor([1, 2], dtype="int32")).tolist() == [True, True]
```

### Elementwise extrema

`maximum(input, other)` and `minimum(input, other)` are available as both
top-level helpers (`minitensor.maximum`, `minitensor.minimum`) and functional
helpers (`minitensor.functional.maximum`, `minitensor.functional.minimum`).
They mirror the corresponding `Tensor.maximum(other)` and
`Tensor.minimum(other)` methods.

Behavior and validation:

- Inputs follow the same Python-to-tensor conversion, dtype-promotion, device,
  and broadcasting rules as tensor binary operations.
- Python scalars, Python sequences, NumPy arrays, and MiniTensor tensors are
  accepted for `other`; `input` should be a MiniTensor tensor or tensor wrapper,
  matching the rest of the tensor-centric functional binary helpers.
- Boolean inputs use logical OR for `maximum` and logical AND for `minimum`.
- Floating-point NaNs are propagated when either operand at an element is NaN.
- Incompatible shapes raise the normal MiniTensor shape/broadcasting error.

Example:

```python
import minitensor as mt

x = mt.Tensor([[1.0, -2.0, 3.0], [4.0, 0.5, -6.0]])
y = mt.Tensor([[0.0, 2.0, 2.5]])

assert mt.maximum(x, y).shape == (2, 3)
assert mt.functional.minimum(x, -1.0).tolist() == [
    [-1.0, -2.0, -1.0],
    [-1.0, -1.0, -6.0],
]
```

### Equality helpers

`isclose(input, other, rtol=1e-5, atol=1e-8, equal_nan=False)`,
`array_equal(input, other)`, and `allclose(input, other, rtol=1e-5,
atol=1e-8, equal_nan=False)` are available as both top-level helpers and
functional helpers. They accept MiniTensor tensors and tensor-like Python inputs
(such as Python scalars/sequences and NumPy arrays) through the normal
Python-to-tensor conversion path. `isclose` returns an elementwise boolean
tensor, while `array_equal` and `allclose` return Python `bool` values.

Behavior and validation:

- `isclose` broadcasts compatible shapes, promotes compatible numeric dtypes,
  and returns a boolean tensor mask with the broadcasted shape.
- `array_equal` requires equal shapes, promotes compatible numeric dtypes, and
  returns a Python `bool` indicating exact element equality after promotion.
- `isclose` and `allclose` promote compatible numeric dtypes and apply
  `abs(a - b) <= atol + rtol * abs(b)` for finite unequal floating-point values.
- Exact equality is accepted before tolerance checks, so signed zeros and
  matching infinities compare as close. Opposite infinities and finite/non-finite
  mismatches compare as not close.
- NaNs compare as not close unless `equal_nan=True`, in which case paired NaNs
  at the same positions are accepted.
- `rtol` and `atol` must be finite, non-negative numbers.

Example:

```python
import minitensor as mt

mask = mt.isclose([[1.0, 2.0]], [1.0 + 1e-6, 3.0], rtol=1e-5)
assert mask.tolist() == [[True, False]]
assert mt.array_equal([1, 2], mt.tensor([1.0, 2.0], dtype="float32"))
assert mt.allclose([0.0, float("inf")], [-0.0, float("inf")])
assert mt.allclose([float("nan")], [float("nan")], equal_nan=True)
```

### One-hot encoding

`one_hot(input, num_classes=None, dtype="float32")` converts integer or boolean
labels to a one-hot tensor whose final dimension is the class dimension. The
helper is available as both `minitensor.one_hot(...)` and
`minitensor.functional.one_hot(...)`.

Supported label inputs:

- `Tensor` values with `int32`, `int64`, or `bool` dtype.
- Python integer scalars and nested Python integer/bool sequences.
- NumPy integer/bool arrays through the existing Python-to-tensor conversion
  path.

Behavior and validation:

- If `num_classes` is omitted, MiniTensor infers it as `max(label) + 1`; empty
  inputs therefore require an explicit `num_classes`.
- `num_classes` must be non-negative when provided, and every label must be in
  `[0, num_classes)`.
- Negative labels and floating-point label tensors/scalars are rejected.
- `dtype` controls the encoded output dtype and accepts the standard MiniTensor
  dtype strings: `float32`, `float64`, `int32`, `int64`, and `bool`.

Example:

```python
import minitensor as mt

labels = mt.Tensor([[0, 2], [1, 2]], dtype="int64")
encoded = mt.one_hot(labels, dtype="int32")
assert encoded.shape_vec() == [2, 2, 3]
```

### Bin counting

`bincount(input, weights=None, minlength=0)` counts occurrences of non-negative
integer or boolean labels in a 1-D input tensor. The helper is available as both
`minitensor.bincount(...)` and `minitensor.functional.bincount(...)`.

Supported inputs:

- `Tensor` values with `int32`, `int64`, or `bool` dtype on CPU.
- Python integer/bool sequences and NumPy integer/bool arrays through the normal
  Python-to-tensor conversion path.
- Optional `weights` as a MiniTensor/tensor-like CPU tensor with the exact same
  shape as `input` and floating-point dtype (`float32` or `float64`).

Behavior and validation:

- `input` must be exactly 1-D; scalar and multidimensional inputs are rejected.
- Labels must be non-negative. Unweighted output has `int64` dtype.
- With `weights`, output dtype follows the weight dtype and each bin contains
  the sum of weights for positions assigned to that label.
- Output length is `max(max(input) + 1, minlength)`, or `minlength` for empty
  inputs. `minlength` must be non-negative.
- `bincount` is currently CPU-only and rejects non-CPU label or weight tensors.

Example:

```python
import minitensor as mt

labels = mt.Tensor([0, 2, 1, 2, 2], dtype="int64")
assert mt.bincount(labels).tolist() == [1, 1, 3]

weights = mt.Tensor([0.5, 1.0, 2.0, 3.0, -1.0], dtype="float32")
weighted = mt.functional.bincount(labels, weights=weights, minlength=4)
assert weighted.tolist() == [0.5, 2.0, 3.0, 0.0]
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

- `step()` -- apply parameter updates and clear the global autograd graph.
- `zero_grad(set_to_none: bool = False)` -- reset gradients.
- `lr` property -- read/write learning rate.

## 8) NumPy compatibility module (`minitensor.numpy_compat`)

### Array creation

- `asarray(data, dtype=None, requires_grad=False)`
- `zeros_like`, `ones_like`, `empty_like`, `full_like`

### Array manipulation

- `concatenate`, `stack`, `vstack`, `hstack`
- `split`, `hsplit`, `vsplit`

### Math & comparisons

- `dot`, `matmul`, `cross`, `where`
- `allclose(a, b, rtol=None, atol=None, equal_nan=False)`, `array_equal(a, b)`

### Statistics

- `mean`, `nanmean`, `std`, `var`, `prod`, `sum`, `nansum`
- `max`, `min`, `nanmax`, `nanmin`

`numpy_compat.std(tensor, axis=None, keepdims=None, ddof=None)` and
`numpy_compat.var(tensor, axis=None, keepdims=None, ddof=None)` accept a single
integer axis or `None`; `ddof=0` maps to population statistics and `ddof=1` maps
to unbiased sample statistics. Values outside `0` and `1` are rejected because
the current tensor engine exposes a boolean unbiased flag rather than arbitrary
correction values.

## 9) Serialization (`minitensor.serialization`)

### Core types

- `ModelVersion` -- semantic version for serialized models.
- `ModelMetadata` -- name, description, architecture, shapes, custom metadata.
- `SerializationFormat` -- `json()`, `binary()`, `messagepack()`.
- `SerializedModel` -- metadata + state dict.
- `StateDict` -- tensor parameters/buffers.
- `DeploymentModel` -- compact model format for inference.
- `ModelSerializer` -- `save()` / `load()` helpers.

### Convenience functions

- `save_model(model, path, format=None)`
- `load_model(path, format=None)`

## 10) Plugin system (`minitensor.plugins`)

### Versioning and metadata

- `VersionInfo` (parse, current, compatibility checks)
- `PluginInfo` (name, version, author, min/max supported versions)

### Python-side plugins

- `CustomPlugin` -- plugin object with init/cleanup/custom-op callbacks.
- `PluginRegistry` -- register/unregister/list Python plugins.
- `CustomLayer` -- define custom layers in Python.
- `PluginBuilder` -- fluent builder for plugin metadata.

### Dynamic loading (if compiled)

- `load_plugin(path)`
- `unload_plugin(name)`
- `list_plugins()`
- `get_plugin_info(name)`
- `is_plugin_loaded(name)`

## 11) Debug utilities (`minitensor._core.debug`)

The compiled extension registers a debug submodule for backend diagnostics. The
high-level Python package does not re-export it as `minitensor.debug`; access it
through the core extension when needed by advanced diagnostics or tests. Debug
APIs are intended for development and troubleshooting rather than stable
end-user workflows.

## 12) Custom operations

MiniTensor supports custom ops in both Rust and Python. Refer to
`docs/custom_operations.md` for:

- The `CustomOp` trait and builder pattern.
- Python registration and execution (`execute_custom_op_py`, etc.).
- Example custom ops (Swish, GELU, power).

## 13) Notes on devices & backends

The core engine supports CPU execution and can be compiled with CUDA, Metal, or
OpenCL backends where applicable. Device selection flows through the `Device`
API and tensor creation functions.

## 14) Documentation maintenance

When public functionality changes, update this reference together with the
focused guide for that area. The runtime helpers `list_public_api()`,
`search_api(...)`, `describe_api(...)`, and `help()` are useful for auditing the
compiled API after rebuilding the extension.

## 15) Where to go next

- [`docs/index.md`](./index.md) -- documentation map and maintenance checklist.
- [`docs/development.md`](./development.md) -- contributor setup, validation, and PR workflow.
- [`docs/custom_operations.md`](./custom_operations.md) -- custom ops and autograd integration.
- [`docs/plugin_system.md`](./plugin_system.md) -- plugin registry and compatibility handling.
- [`docs/performance.md`](./performance.md) -- performance tuning and profiling.
- `examples/` and `examples/notebooks/` -- end-to-end usage patterns.
