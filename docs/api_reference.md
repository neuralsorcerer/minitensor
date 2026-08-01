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
| `clear_autograd_graph()` | Clear the global autograd graph, releasing every stored gradient. Required in any loop that calls `backward()` without an optimizer step -- see below. |
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

#### Which constructor keeps the source dtype

The two families differ, and the difference is easy to trip over:

| Constructor | dtype when `dtype=` is omitted |
| --- | --- |
| `Tensor(obj)`, `tensor(obj)` | always `float32` |
| `from_numpy(array)`, `as_tensor(obj)` | taken from the source |

So `Tensor(np.arange(3))` is `float32` while `from_numpy(np.arange(3))` is
`int64`, and a float64 array loses precision through `Tensor` but not through
`from_numpy`. `Tensor` matches `torch.Tensor`, which likewise ignores the
source dtype and uses the default; note that `tensor` does **not** match
`torch.tensor`, which infers. Pass `dtype=` explicitly, or use `from_numpy` /
`as_tensor`, whenever the source dtype is the one you want.

```python
import numpy as np

import minitensor as mt

mt.manual_seed(0)  # make the random helpers reproducible

# Deterministic / structured
print(mt.zeros(2, 3).shape, mt.ones(2).tolist(), mt.eye(3).shape)
print(mt.arange(0.0, 1.0, 0.25).tolist())
print(mt.linspace(0.0, 1.0, 3).tolist())
print(mt.full((2,), 7.0).tolist())

# Random + initialization schemes (shapes are deterministic; values are not)
print(mt.randn(2, 3).shape, mt.rand(4).shape)
print(mt.xavier_uniform(4, 4).shape, mt.he_normal(3, 3).shape)

# `*_like` variants copy shape/dtype/device from an existing tensor
base = mt.zeros(2, 2)
print(mt.ones_like(base).tolist(), mt.randn_like(base).shape)

# NumPy interop
array = np.array([1.0, 2.0, 3.0], dtype="float32")
print(mt.from_numpy(array).tolist())
print(mt.as_tensor([1, 2, 3], dtype="float32").tolist())
```

```text
Shape([2, 3]) [1.0, 1.0] Shape([3, 3])
[0.0, 0.25, 0.5, 0.75]
[0.0, 0.5, 1.0]
[7.0, 7.0]
Shape([2, 3]) Shape([4])
Shape([4, 4]) Shape([3, 3])
[[1.0, 1.0], [1.0, 1.0]] Shape([2, 2])
[1.0, 2.0, 3.0]
[1.0, 2.0, 3.0]
```

```{note}
`from_numpy_shared` currently copies exactly like `from_numpy`. Writing to the
source NumPy array after construction does **not** change the tensor.
```

## 3) Tensor properties & conversion helpers

Frequently used tensor properties (accessed without parentheses):

- `tensor.shape` -- a `Shape` object that compares equal to the equivalent tuple
- `tensor.dtype` -- dtype name, e.g. `"float32"`
- `tensor.device` -- device name, e.g. `"cpu"`
- `tensor.requires_grad` -- whether autograd tracks this tensor
- `tensor.grad` -- accumulated gradient, or `None`
- `tensor.size` -- total number of elements
- `tensor.strides`, `tensor.itemsize`, `tensor.nbytes` -- layout and storage size

Rank and element count are **methods**, so they must be called:

- `tensor.ndim()` -- number of dimensions
- `tensor.numel()` -- number of elements (same value as the `size` property)

```{warning}
`ndim` and `numel` are methods, not properties. Writing `tensor.ndim == 2`
compares a bound method against an integer and is always `False`; use
`tensor.ndim() == 2`.
```

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

```python
import minitensor as mt

t = mt.Tensor([[1.5, 2.5], [3.5, 4.5]], requires_grad=True)

print(t.shape, t.dtype, t.device, t.requires_grad)
print(t.ndim(), t.numel(), t.size)          # methods vs property
print(t.shape == (2, 2))                    # Shape compares equal to a tuple
print(t.tolist(), t.astype("float64").dtype)
print(float(t[0][0]), int(mt.Tensor([2.7])))  # int() truncates

doubled = (t * 2 + 1).numpy()               # operators return tensors
print(doubled.tolist())
```

```text
Shape([2, 2]) float32 cpu True
2 4 4
True
[[1.5, 2.5], [3.5, 4.5]] float64
1.5 2
[[4.0, 6.0], [8.0, 10.0]]
```

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
- `scatter(dim, index, src)`, `scatter_add(dim, index, src)`
- `flip`, `roll`

`scatter` and `scatter_add` write into a copy of the tensor at the positions
named by `index`, overwriting and accumulating respectively. `index` and `src`
must have the same shape, and must match the tensor on every axis except `dim` —
the same rule `gather` uses, which makes `scatter_add` its adjoint:
`<gather(x, i), v>` equals `<x, zeros.scatter_add(i, v)>` for every `x` and `v`.

Duplicate indices are the interesting case, and the two behave differently:

- `scatter_add` accumulates every value addressed at a destination, which is
  what makes it the natural spelling for segment sums and embedding-style
  gradient accumulation. Because float addition is not associative, the
  accumulation order is fixed rather than left to thread scheduling, so repeated
  runs are bit-for-bit identical.
- `scatter` keeps the last write. PyTorch leaves this case explicitly
  non-deterministic; here the order is defined, and the gradient follows it —
  a source element whose value was overwritten receives exactly zero, as does
  an input slot that was written over.

`scatter_add` is not defined for boolean tensors, since bool has no addition.

```python
import minitensor as mt

counts = mt.Tensor([[0.0, 0.0, 0.0]])
index = mt.Tensor([[0, 0, 2]], dtype="int64")
updates = mt.Tensor([[1.0, 2.0, 3.0]])
print(counts.scatter(1, index, updates).tolist())
print(counts.scatter_add(1, index, updates).tolist())
```

```text
[[2.0, 0.0, 3.0]]
[[3.0, 0.0, 3.0]]
```

Assignment (`t[key] = value`) writes in place. Whether another handle on the
same tensor sees the write follows the rule every in-place operation here uses:

- A **live handle on a leaf parameter** — what `layer.parameters()` and the
  `weight`/`bias` getters return — shares the layer's storage, so assigning to
  one updates the layer. This is the same path optimizers use to apply their
  steps.
- An **explicit copy** never aliases. `clone()` deep-copies, `detach()` yields a
  non-gradient tensor, and a reshape or any other view carries a `grad_fn`; all
  three copy on write, so assigning to them leaves the original untouched.

`load_state_dict` remains the tidier way to set a whole module's parameters.

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
- `norm(p=2, dim=None, keepdim=False)`
- `isclose(other, rtol=1e-5, atol=1e-8, equal_nan=False)`
- `array_equal(other)`
- `allclose(other, rtol=1e-5, atol=1e-8, equal_nan=False)`

`norm(p=2, dim=None, keepdim=False)` takes the vector p-norm over `dim`, or over
the flattened tensor when `dim` is `None`. It accepts the same dimension forms as
`sum`. Supported orders are any finite `p > 0`, `float("inf")` (largest
magnitude), `float("-inf")` (smallest magnitude), `0` (count of non-zeros), and
the string `"fro"` as an alias for `p=2`. Finite negative orders raise
`ValueError`.

Two behaviours are worth knowing:

- The p-norm has a corner at the origin, so it has no derivative there. `norm`
  reports a gradient of `0` — the subgradient of least magnitude, matching
  PyTorch. Building the same quantity out of `(x * x).sum().sqrt()` yields
  `0 / 0 = NaN` instead, which then spreads to everything downstream.
- The 2-norm is computed by scaling with the largest magnitude rather than
  summing squares directly, so it stays finite for inputs whose squares would
  overflow. `mt.Tensor([1e20, 1e20]).norm()` returns `1.41e20`, where
  `(x * x).sum().sqrt()` returns `inf`. Since detecting a blow-up is the usual
  reason to take a norm, saturating exactly then would defeat the purpose.

```python
import minitensor as mt

grid = mt.Tensor([[3.0, 4.0], [5.0, 12.0]])
print(grid.norm().item())
print(grid.norm(1.0, 1).tolist())
print(grid.norm(float("inf"), 1).tolist())

point = mt.Tensor([3.0, 4.0], requires_grad=True)
point.norm().backward()
print(point.grad.tolist())
```

```text
13.928388595581055
[7.0, 17.0]
[4.0, 12.0]
[0.6000000238418579, 0.800000011920929]
```

`std` and `var` accept the same dimension forms as multi-axis reductions such
as `sum` and `mean`: `None` reduces all axes, an integer reduces one axis, and
a sequence such as a tuple/list reduces multiple axes. Negative axes are
normalized, duplicate axes are treated as a single axis, and invalid axes raise
`IndexError`. `keepdim=True` preserves reduced axes with length one; otherwise
those axes are removed after the reduction. `unbiased=True` applies the sample
variance correction (`N / (N - 1)`) over the total number of reduced elements,
and reductions with one or fewer samples return `NaN` rather than emitting a
Python warning.

### Reducing an empty axis

Whether an empty reduction has an answer depends on whether the operation has
an identity element:

| Operation | Reducing a length-zero axis |
| --- | --- |
| `sum`, `nansum` | `0` |
| `prod` | `1` |
| `mean`, `nanmean`, `std`, `var` | `NaN` (0/0) |
| `logsumexp` | `-inf` (`log 0`) |
| `max`, `min`, `nanmax`, `nanmin`, `argmax`, `argmin` | raises |
| `median`, `quantile`, `nanquantile` | raises |
| `sort`, `argsort` | returns the empty input unchanged |

The extrema raise because they have no identity to fall back on. Returning the
fold identity — which is what they used to do — gives `-inf` for floats and
`iinfo.min` for integers, and the integer case is indistinguishable from a
genuine maximum, since `iinfo.min` is a value a real tensor can hold. `argmax`
returned index `0`, which is not a valid index into an axis with no elements.
NumPy and PyTorch both raise here as well.

Only the **reduced** axis has to be non-empty, so an empty batch still flows
through a reduction over some other axis:

```python
import minitensor as mt
import numpy as np

batch = mt.from_numpy(np.empty((0, 3), dtype=np.float32))

values, indices = batch.max(dim=1)   # each row has 3 elements; there are 0 rows
print(tuple(values.shape), tuple(indices.shape))

try:
    batch.max(dim=0)                 # would need 3 values from 0 elements
except ValueError as exc:
    print(exc)
```

```text
(0,) (0,)
Invalid argument: max() does not support empty tensors
```

`median` and `nanmedian` follow PyTorch: with an even number of elements they
return the **lower** of the two middle values rather than averaging them the
way `numpy.median` does. That is also what lets `median(dim=...)` report the
index of the element it selected. Use `quantile(0.5)` / `nanquantile(0.5)` when
you want the interpolated, NumPy-compatible definition.

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
- `abs`, `sqrt`, `exp`, `log`, `pow`, `matmul`
- `eq`, `ne`, `lt`, `le`, `gt`, `ge` — free-function forms of the comparison
  methods, returning bool tensors
- `floor_divide`, `remainder`, `bitwise_not`
- `softsign`, `rsqrt`, `reciprocal`, `sign`
- `leaky_relu(input, negative_slope=0.01)` — the gradient at exactly `0` is
  `negative_slope`, the same side `relu` takes, matching PyTorch
- `isnan`, `isinf`, `isfinite`
- `clip`, `clamp`, `clamp_min`, `clamp_max`
- `round`, `floor`, `ceil` — `round` sends halves to the even neighbour
  (`round(0.5) == 0`, `round(2.5) == 2`), matching NumPy, PyTorch and Python's
  built-in `round`. It takes an optional `decimals` argument.
- `log2`, `log10` — the natural log rescaled; they share `log`'s behaviour at
  `0`, negatives, infinities and NaN
- `erf`, `erfc` — the Gauss error function and its complement. `erfc` uses a
  dedicated routine rather than computing `1 - erf(x)`: past about `x = 6` in
  float64 `erf(x)` rounds to 1 and that subtraction returns exactly zero, which
  is precisely the tail `erfc` exists to give you (`erfc(20) ≈ 5.4e-176`).
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
- `rms_norm(shape, weight=None, eps=1e-6)` -- root-mean-square normalization
  (no mean subtraction, no bias)

### Comparison methods

Method forms of the comparison operators, all returning a `bool` tensor:
`eq`, `ne`, `lt`, `le`, `gt`, `ge`. Each accepts a tensor or a scalar. The
`*_from_py` variants (`eq_from_py`, `ne_from_py`, `lt_from_py`, `le_from_py`,
`gt_from_py`, `ge_from_py`) take a Python scalar directly and exist so the
operator protocol can dispatch without building a temporary tensor.

### Tensor-like constructors

Create a new tensor that inherits dtype and device from an existing one:

- `new_zeros(shape)`, `new_ones(shape)`, `new_empty(shape)`
- `new_full(shape, value)`
- `new_tensor(data)`

### Autograd + in-place

- `backward()` to trigger gradient computation.
- `fill_(value)` for in-place fills.
- `copy_(other)` copies another tensor's values in place.
- `detach()` returns a view that autograd does not track; `detach_()` detaches
  in place and returns `None`.
- `requires_grad_(flag)` sets gradient tracking and returns the tensor, so it
  chains.
- `grad` holds the accumulated gradient; `has_grad` is a **property** reporting
  whether one is present.

### Layout and conversion extras

- `is_contiguous()` reports whether the storage is contiguous.
- `numpy_copy()` returns a NumPy array that never shares storage.
- `split_with_sections(sections, dim)` splits into explicitly sized chunks.

```python
import minitensor as mt

t = mt.Tensor([[1.0, 2.0], [3.0, 4.0]])

print(t.exp().tolist()[0][0] > 2.7, t.pow(2.0).tolist())
print(t.lt(3.0).tolist())                    # scalars are accepted
print(t.new_zeros([2]).tolist(), t.new_full([2], 5.0).tolist())
print(t.is_contiguous(), type(t.numpy_copy()).__name__)
print([x.shape for x in t.split_with_sections([1, 1], 0)])

target = mt.Tensor([1.0, 2.0])
target.copy_(mt.Tensor([9.0, 9.0]))
print(target.tolist())

param = mt.Tensor([1.0]).requires_grad_(True)   # chains
print(param.requires_grad, param.has_grad)
(param * 2).sum().backward()
print(param.has_grad, param.detach().requires_grad)
```

```text
True [[1.0, 4.0], [9.0, 16.0]]
[[True, True], [False, False]]
[0.0, 0.0] [5.0, 5.0]
True ndarray
[Shape([1, 2]), Shape([1, 2])]
[9.0, 9.0]
True False
True False
```

## 5) Functional API (`minitensor.functional`)

MiniTensor provides stateless functional variants that mirror `Tensor` methods.

### Forwarders exported at top level

Each of the following names is accessible from:

- `minitensor.<name>`
- `minitensor.functional.<name>`

```
cat, stack, split, chunk, index_select, gather, scatter, scatter_add, narrow,
topk, sort, argsort,
median, nanmedian, quantile, nanquantile, nansum, nanmean, nanmax, nanmin, isnan,
isinf, isfinite, nan_to_num, logsumexp, norm, softmax, log_softmax,
masked_softmax, masked_log_softmax, sum, prod,
mean, all, any, max, min, argmax, argmin, cumsum, cumprod, std, var, relu,
hardshrink, sigmoid, softplus, gelu, elu, selu, silu, softsign, tanh,
layer_norm, rms_norm, scaled_dot_product_attention, rope, glu,
rsqrt, reciprocal, sign, abs, sqrt, exp, log, pow, matmul, leaky_relu,
eq, ne, lt, le, gt, ge, floor_divide, remainder, bitwise_not,
reshape, view, triu, tril, diagonal,
trace, solve, flatten, ravel, transpose, permute, movedim, moveaxis, swapaxes,
swapdims, squeeze, unsqueeze, expand, repeat, repeat_interleave, flip, roll,
clip, clamp, clamp_min, clamp_max, round, floor, ceil, sin, cos, tan, asin,
acos, atan, sinh, cosh, asinh, acosh, atanh, log1p, log2, log10, erf, erfc,
expm1, logaddexp, maximum,
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

### Transformer primitives

Stateless building blocks for Transformer-style models. All are assembled from
autograd-tracked operations, so gradients flow through them without any special
handling.

`rms_norm(input, normalized_shape, weight=None, eps=1e-6)` -- root-mean-square
normalization: rescales by the RMS over the trailing `normalized_shape`
dimensions and applies an optional gain. Unlike `layer_norm` there is no mean
subtraction and no bias.

`scaled_dot_product_attention(query, key, value, attn_mask=None, is_causal=False, scale=None)`
-- computes `softmax(Q Kᵀ / sqrt(E) + bias) V` over the key axis. Shapes are
`query (..., L, E)`, `key (..., S, E)`, `value (..., S, Ev)`, returning
`(..., L, Ev)`. Leading batch axes broadcast, so a multi-head layout
`(batch, heads, seq, dim)` works directly. `attn_mask` broadcasts to the scores
`(..., L, S)`: a float mask is added to them (use `-inf` to disallow a position,
or supply a relative-position bias), while a bool mask keeps `True` positions
and disables `False` ones. `is_causal=True` restricts query `i` to keys `j <= i`,
aligned to the bottom right when `L != S`; combining it with an explicit
`attn_mask` is rejected. `scale` overrides the default `1/sqrt(E)`.

`rope(x, base=10000.0, offset=0)` -- rotary position embedding. Rotates pairs of
features of an `(..., seq, head_dim)` input by position-dependent angles,
injecting *relative* position information with no learned parameters;
`head_dim` must be even. `offset` shifts the starting position, which is what
incremental (KV-cache) decoding needs, and `base` sets the frequency spectrum.

`glu(input, dim=-1)` -- gated linear unit. Splits `input` into halves `(a, b)`
along `dim` and returns `a * sigmoid(b)`; `dim` must have even length. This is
the gate underlying GLU-family feed-forward blocks.

```python
import minitensor as mt
from minitensor import functional as F

# One causal attention head with rotary positions.
q = mt.randn(2, 8, 4, 16)  # (batch, heads, seq, head_dim)
k = mt.randn(2, 8, 4, 16)
v = mt.randn(2, 8, 4, 16)
out = F.scaled_dot_product_attention(F.rope(q), F.rope(k), v, is_causal=True)
```

### Tensor-centric math helpers

The `functional` namespace also exposes:

- `dot`
- `bmm`

### Cross-pollination with `nn`

Lower-case callable symbols from `minitensor.nn` are mirrored into
`minitensor.functional`, so each of the following is reachable as both
`nn.<name>` and `functional.<name>` -- they are the same function object. These
are the stateless counterparts of the layers and losses in section 6, useful
when you already hold the weights and do not want a module:

| Function | Purpose |
| --- | --- |
| `dense_layer(input, weight, bias=None)` | Fully connected transform `x @ Wᵀ + b`. |
| `conv2d(input, weight, bias=None, stride=None, padding=None)` | 2-D convolution. Float32 or Float64 CPU tensors; input, weight and bias must share a dtype. |
| `conv1d(input, weight, bias=None, stride=1, padding=0)` | 1-D convolution over `[N, C_in, L]` with a `[C_out, C_in, K]` kernel. Same dtype and device support as `conv2d`, which it is built on. |
| `max_pool1d(input, kernel_size, stride=None, padding=0)` | 1-D max pooling; `stride` defaults to `kernel_size`. |
| `avg_pool1d(input, kernel_size, stride=None, padding=0, count_include_pad=True)` | 1-D average pooling. |
| `batch_norm(input, running_mean=None, running_var=None, weight=None, bias=None, training=True, momentum=0.1, eps=1e-5)` | Batch normalization; updates the running buffers in place when `training=True`. |
| `dropout2d(input, p)` | Channel-wise dropout. |
| `mse_loss(predictions, targets, reduction="mean")` | Mean squared error. |
| `l1_loss(predictions, targets, reduction="mean")` | Mean absolute error. |
| `smooth_l1_loss(predictions, targets, reduction="mean", beta=1.0)` | Smooth L1: quadratic below `beta`, linear above. `beta` must be positive and finite. |
| `huber_loss(predictions, targets, reduction="mean", delta=1.0)` | Huber loss. Related to the above by `huber(x, d) == d * smooth_l1(x, beta=d)`, so the two agree only at `1.0`. |
| `log_cosh_loss(predictions, targets, ...)` | Log-cosh loss. |
| `kl_div(predictions, targets, reduction="mean")` | KL divergence over probabilities (not log-probabilities). `reduction="mean"` is the element-wise mean, as for every other loss here; `"batchmean"` divides by the leading dimension, which is the divisor that makes the result a true KL divergence per sample. |
| `focal_loss(input, target, alpha=0.25, gamma=2.0, reduction="mean")` | Multi-class focal loss over logits, with one-hot or index targets. `alpha` must lie strictly in `(0, 1)`. |
| `binary_cross_entropy(predictions, targets, ...)` | Binary cross entropy over probabilities. |
| `binary_cross_entropy_with_logits(input, target, pos_weight=None, reduction="mean")` | Binary cross entropy over raw logits, with the sigmoid fused in. Prefer this to `sigmoid` followed by `binary_cross_entropy`: it is the same function mathematically but keeps its gradient at logit magnitudes where the two-step form has already lost it. `pos_weight` is broadcast against the targets and weights the positive class. |
| `cross_entropy(input, target, reduction="mean", dim=1)` | Softmax cross entropy over `dim`. |

```python
import minitensor as mt
from minitensor import functional as F
from minitensor import nn

# The nn and functional names are the same object.
print(nn.mse_loss is F.mse_loss)

x = mt.Tensor([[1.0, 2.0], [3.0, 4.0]])
weight = mt.Tensor([[1.0, 0.0], [0.0, 1.0]])
bias = mt.Tensor([0.5, 0.5])
print(F.dense_layer(x, weight, bias).tolist())

predictions = mt.Tensor([[0.2, 0.8]])
targets = mt.Tensor([[0.0, 1.0]])
print(round(float(F.mse_loss(predictions, targets).numpy()), 4))
print(round(float(F.binary_cross_entropy(predictions, targets).numpy()), 4))

logits = mt.Tensor([[2.0, 1.0, 0.1]])
labels = mt.Tensor([[1.0, 0.0, 0.0]])
print(round(float(F.cross_entropy(logits, labels).numpy()), 4))

# A logit of -30 against a target of 1 is a confident, completely wrong
# prediction, so the gradient should be -1 -- the strongest signal the loss
# can give. Taking sigmoid first would round it to zero and lose that.
binary_logit = mt.Tensor([[-30.0]], requires_grad=True)
binary_loss = F.binary_cross_entropy_with_logits(binary_logit, mt.Tensor([[1.0]]))
print(round(float(binary_loss.numpy()), 4))
binary_loss.backward()
print(binary_logit.grad.tolist())
```

```text
True
[[1.5, 2.5], [3.5, 4.5]]
0.04
0.2231
0.417
30.0
[[-1.0]]
```

## 6) Neural network module (`minitensor.nn`)

### Layers & containers

- `Module` (base class)
- `DenseLayer`
- `Conv2d`
- `Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True, device=None, dtype=None)`
- `MaxPool2d(kernel_size, stride=None, padding=None)`
- `AvgPool2d(kernel_size, stride=None, padding=None, count_include_pad=True)`
- `MaxPool1d(kernel_size, stride=None, padding=0)`
- `AvgPool1d(kernel_size, stride=None, padding=0, count_include_pad=True)`
- `BatchNorm1d`
- `BatchNorm2d`
- `LayerNorm(normalized_shape, eps=1e-5, elementwise_affine=True, device=None, dtype=None)`
- `RMSNorm(normalized_shape, eps=1e-6, elementwise_affine=True, device=None, dtype=None)`
- `Embedding(num_embeddings, embedding_dim, padding_idx=None, device=None, dtype=None)`
- `LSTM(input_size, hidden_size, num_layers=1, bias=True, batch_first=False, bidirectional=False, device=None, dtype=None)`
- `GRU(input_size, hidden_size, num_layers=1, bias=True, batch_first=False, bidirectional=False, device=None, dtype=None)`
- `MultiheadAttention(embed_dim, num_heads, bias=True, is_causal=False, device=None, dtype=None)`
- `Dropout`, `Dropout2d`
- `Sequential` (container of modules)

#### Pooling layers

`MaxPool2d` and `AvgPool2d` take `[N, C, H, W]` inputs and reduce each
`kernel_size` window of every channel. `stride` defaults to `kernel_size` --
pooling's convention, and unlike convolution, which defaults to 1 -- so
`MaxPool2d((2, 2))` halves both spatial dimensions. `padding` may not exceed
half the window, since a window lying entirely in the padding has no defined
maximum.

For `AvgPool2d`, `count_include_pad` decides whether padded cells count towards
the divisor. The functional forms are `functional.max_pool2d(input,
kernel_size, stride=None, padding=None)` and `functional.avg_pool2d(input,
kernel_size, stride=None, padding=None, count_include_pad=True)`.

```python
import minitensor as mt

features = mt.nn.Sequential(
    [
        mt.nn.Conv2d(1, 8, (3, 3), padding=(1, 1)),
        mt.nn.ReLU(),
        mt.nn.MaxPool2d((2, 2)),
    ]
)
out = features(mt.Tensor([[[[0.0] * 8] * 8]]))
print(out.shape)
```

```text
Shape([1, 8, 4, 4])
```

#### Transformer layers

`LayerNorm` and `RMSNorm` normalize over the trailing `normalized_shape`
dimensions, given as an int or a sequence of ints. `LayerNorm` learns a scale
and a shift; `RMSNorm` learns only a gain, matching its no-mean-subtraction
definition. Setting `elementwise_affine=False` drops the learned parameters
entirely.

`Embedding` maps integer token ids (int32 or int64) to rows of a learned
`[num_embeddings, embedding_dim]` matrix; output shape is the input shape with
`embedding_dim` appended. Ids are range-checked. A token given as `padding_idx`
keeps a fixed zero embedding and receives no gradient.

`MultiheadAttention` takes batch-first `(batch, seq, embed_dim)` input, where
`embed_dim` must be divisible by `num_heads`. Calling the layer performs
self-attention; `is_causal=True` makes it autoregressive. For cross-attention
use:

- `forward_qkv(query, key, value, attn_mask=None, is_causal=False)` -- `key` and
  `value` must share a batch size and sequence length, while `query` may have
  its own; the output follows the query's length. `attn_mask` broadcasts to the
  per-head scores `(batch, heads, query_seq, key_seq)`.

```python
import minitensor as mt
from minitensor import nn

embed = nn.Embedding(32000, 512)
norm = nn.RMSNorm(512)
attn = nn.MultiheadAttention(512, 8, is_causal=True)

tokens = mt.Tensor([[1, 42, 7]], dtype="int64")
hidden = embed(tokens)
hidden = hidden + attn(norm(hidden))  # pre-norm residual block
```

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
- `BCEWithLogitsLoss(reduction="mean", pos_weight=None)`
- `FocalLoss`

The 1-D convolution and pooling operations are implemented by giving the signal
a singleton height and deferring to their 2-D counterparts, so there is one
implementation of the window arithmetic and one backward pass rather than two to
keep in agreement. The reshapes are autograd-aware, so gradients flow normally.
As for the 2-D forms, `stride` defaults to `kernel_size` for pooling but to 1
for convolution — the same value produces different output lengths through the
two, so it is worth being explicit.

`LSTM` and `GRU` take `(seq, batch, input_size)`, or `(batch, seq, input_size)`
when `batch_first=True`. Calling the layer returns just the output sequence;
`forward_with_state(input, hx=None, cx=None)` also returns the final states,
shaped `(num_layers, batch, hidden_size)` regardless of `batch_first`:

- `LSTM` returns `(output, (h_n, c_n))` and accepts both `hx` and `cx`.
- `GRU` returns `(output, h_n)` and rejects a cell state.

States default to zeros when omitted. Parameters are drawn from
`U(-1/sqrt(hidden_size), 1/sqrt(hidden_size))`, biases included — the convention
for recurrent layers, unlike the zero biases elsewhere in `nn`. Gate blocks are
packed along the first axis of each weight matrix in the order `i, f, g, o` for
LSTM and `r, z, n` for GRU.

The GRU candidate is `tanh(W_in x + r * (W_hn h + b_hn))` — the reset gate scales
the *projected* hidden term, so it also scales `b_hn`. That is not the same
function as `tanh(W_in x + W_hn (r * h) + b_hn)`, and the difference is large
rather than a rounding artefact.

With `bidirectional=True` each layer also runs over the reversed sequence and
the two passes are concatenated along the feature axis, so `output_size` becomes
`2 * hidden_size` and every layer after the first consumes that width. The
reverse pass is realigned onto the input's timeline before being joined, so
position `t` of the output always pairs the two directions' states *for that
timestep*. State tensors gain a row per direction — `(num_layers * directions,
batch, hidden_size)` — ordered layer-0-forward, layer-0-reverse, layer-1-forward
and so on, which matches how PyTorch names `*_l{k}` and `*_l{k}_reverse`.

Both layers are built from ordinary autograd-aware operations rather than a
fused kernel, so the backward pass through the unrolled sequence is derived by
the existing graph rather than hand-written. That is the safer choice for a
recurrence; a fused kernel would be faster and is the obvious later change.
Packed (variable-length) sequences are not implemented.

```python
import minitensor as mt
from minitensor import nn

lstm = nn.LSTM(input_size=3, hidden_size=4, num_layers=2)
sequence = mt.zeros(5, 2, 3)
output, (h_n, c_n) = lstm.forward_with_state(sequence)
print(output.shape, h_n.shape, c_n.shape)

gru = nn.GRU(input_size=3, hidden_size=4, batch_first=True)
print(gru(mt.zeros(2, 5, 3)).shape)

# Bidirectional: the feature axis carries both directions.
bi = nn.GRU(input_size=3, hidden_size=4, bidirectional=True)
output, h_n = bi.forward_with_state(mt.zeros(5, 2, 3))
print(output.shape, h_n.shape, bi.output_size)
```

```text
Shape([5, 2, 4]) Shape([2, 2, 4]) Shape([2, 2, 4])
Shape([2, 5, 4])
Shape([5, 2, 8]) Shape([2, 2, 4]) 8
```

### Common utilities

- `layer.parameters()` returns tensors for optimizers.
- `layer.zero_grad()` clears gradients for trainable tensors.

## 7) Optimizers (`minitensor.optim`)

### Built-in optimizers

- `SGD(params, lr, momentum=0.0, dampening=0.0, weight_decay=0.0, nesterov=False)`
- `Adam(params, lr=1e-3, betas=None, beta1=None, beta2=None, epsilon=1e-8, weight_decay=0.0, amsgrad=False)`
- `AdamW`
- `RMSprop(params, lr, alpha=0.99, epsilon=1e-8, weight_decay=0.0, momentum=0.0, centered=False)`
- `NAdam(params, lr=0.002, beta1=0.9, beta2=0.999, epsilon=1e-8, weight_decay=0.0, momentum_decay=0.004)`
- `Adagrad(params, lr=0.01, lr_decay=0.0, weight_decay=0.0, initial_accumulator_value=0.0, epsilon=1e-10)`
- `Lion(params, lr=1e-4, betas=None, beta1=None, beta2=None, weight_decay=0.0)`

`SGD`'s `dampening` scales the incoming gradient by `1 - dampening` before it
enters the momentum buffer, so the buffer leans further on its history. The
first step is exempt: the buffer is seeded with the gradient itself, as PyTorch
does, rather than being damped from nothing. `nesterov=True` requires
`dampening=0` — the lookahead `grad + momentum * buf` is only the right
extrapolation when `buf` accumulated the undamped gradient.

`NAdam` (Dozat, 2016) is Adam with Nesterov momentum: the step uses the momentum
the *next* iterate will carry rather than the current one, so it begins
decelerating before overshooting rather than after. Its momentum coefficient is
scheduled rather than fixed —
`mu_t = beta1 * (1 - 0.5 * 0.96^(t * momentum_decay))` — starting near
`beta1 / 2` and rising toward `beta1`, which damps the first steps while the
moment estimates are still poor. The running product of every `mu` so far
replaces Adam's `beta1^t` in the bias correction, so it is optimizer state rather
than something recomputed per step.

`Adagrad` accumulates a running *sum* of squared gradients where RMSprop keeps
an exponential moving average. The denominator therefore never shrinks, and each
parameter's effective step `lr / (sqrt(sum) + eps)` decays monotonically — under
a constant gradient, exactly as `1/sqrt(t)`. That is the point of the method: a
coordinate whose gradient is rare accumulates little and keeps moving at close to
the full learning rate, which is what makes it suit sparse features. It is also
why Adagrad stalls on long runs, and why the moving-average methods exist.
`lr_decay` shrinks the rate further as `lr / (1 + (t - 1) * lr_decay)`, and
`initial_accumulator_value` starts the sum above zero to damp the first steps.
Its `epsilon` default is `1e-10` rather than the `1e-8` used elsewhere, because
it floors a quantity that only grows.

`Lion` (Chen et al., 2023) updates parameters by the *sign* of an interpolated
momentum, so every parameter moves by exactly `lr` regardless of gradient
magnitude, and it stores one momentum buffer per parameter instead of Adam's
two. Because the step size is uniform, a Lion learning rate is typically 3-10x
smaller than the AdamW one, with a correspondingly larger `weight_decay`. Its
beta defaults are `(0.9, 0.99)` -- not Adam's `(0.9, 0.999)`.

`Adam`'s `amsgrad=True` keeps the running *maximum* of the second moment
instead of its current value, so the denominator never shrinks and the step
size is monotonically non-increasing per coordinate. It is the fix for Adam's
non-convergence on problems where a rare large gradient would otherwise be
forgotten by the moving average.

### Gradient clipping

In `minitensor.nn`, where PyTorch puts them (`torch.nn.utils.clip_grad_norm_`):

- `clip_grad_norm_(parameters, max_norm)` -- scales every gradient in place so
  their combined L2 norm is at most `max_norm`, and returns the norm *before*
  clipping so a training loop can log it. The coefficient is
  `max_norm / (total_norm + 1e-6)`, matching PyTorch.
- `clip_grad_value_(parameters, clip_value)` -- clamps to
  `[-clip_value, clip_value]`. Pass `min_value=`/`max_value=` instead for an
  asymmetric range.
- `grad_norm(parameters)` -- the same norm, without modifying anything.
- `count_parameters_with_gradients(parameters)` -- how many currently hold one.

Parameters without a gradient are skipped rather than rejected, so passing
`model.parameters()` before the first `backward()` is a no-op. Only float
gradients participate.

```python
import minitensor as mt
from minitensor import nn, optim

parameter = mt.zeros((1,), requires_grad=True)
optimizer = optim.SGD([parameter], 1.0)

optimizer.zero_grad(True)
(parameter * mt.full((1,), 1000.0)).sum().backward()
print(round(nn.clip_grad_norm_([parameter], 2.0), 4))
optimizer.step()
print(round(abs(float(parameter.numpy()[0])), 4))
```

```text
1000.0
2.0
```

### Learning-rate schedulers

A scheduler wraps an optimizer, owns the step counter, and writes each step's
rate back to `optimizer.lr`. Constructing one applies the schedule's step-0
value immediately, as PyTorch does, so `LinearWarmupLR` starts at zero.

- `ConstantLR(optimizer)` -- holds the rate.
- `StepLR(optimizer, step_size, gamma=0.1)` -- `base_lr * gamma ** (t // step_size)`.
- `ExponentialLR(optimizer, gamma)` -- `base_lr * gamma ** t`.
- `CosineAnnealingLR(optimizer, t_max, eta_min=0.0)` -- half cosine from `base_lr` down to `eta_min` over `t_max` steps, then held.
- `LinearWarmupLR(optimizer, warmup_steps)` -- linear ramp from 0 to `base_lr`, then held.
- `PolynomialDecayLR(optimizer, decay_steps, end_lr=0.0, power=1.0)` -- `(base_lr - end_lr) * (1 - t/decay_steps) ** power + end_lr`, then held at `end_lr`.
- `MultiStepLR(optimizer, milestones, gamma=0.1)` -- multiplies by `gamma` once per milestone passed.

Every schedule is relative to the learning rate the optimizer had when the
scheduler was constructed (`scheduler.base_lr`); assigning to `optimizer.lr`
afterwards does not move the schedule. `get_lr(step)` evaluates the schedule at
any step without applying it, and `get_last_lr()` returns what was last written.

```python
import minitensor as mt
from minitensor import optim

parameter = mt.zeros((2,), requires_grad=True)
optimizer = optim.SGD([parameter], 1.0)
scheduler = optim.StepLR(optimizer, step_size=2, gamma=0.5)

rates = [optimizer.lr]
for _ in range(4):
    scheduler.step()
    rates.append(optimizer.lr)
print(rates)
```

```text
[1.0, 1.0, 0.5, 0.5, 0.25]
```

### Base optimizer API

`Optimizer` is the shared base class; every optimizer above subclasses it, so
`isinstance(opt, optim.Optimizer)` identifies any of them.

All optimizer classes share a common interface:

- `step()` -- apply parameter updates and clear the global autograd graph.
- `zero_grad(set_to_none: bool = False)` -- reset gradients.
- `lr` property -- read/write learning rate.

Every optimizer takes an iterable of parameter tensors, which is what
`model.parameters()` returns. A training step is always the same four calls:
zero the gradients, compute the loss, back-propagate, then step.

```python
import minitensor as mt
from minitensor import nn, optim

mt.manual_seed(0)

model = nn.DenseLayer(4, 2)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

inputs = mt.randn(8, 4)
targets = mt.randn(8, 2)

first = None
for _ in range(20):
    optimizer.zero_grad()
    loss = criterion(model(inputs), targets)
    loss.backward()
    optimizer.step()
    if first is None:
        first = float(loss.numpy())

print(len(model.parameters()), optimizer.lr)
print(float(loss.numpy()) < first)   # the loss went down

optimizer.lr = 0.001                 # schedules can write the learning rate
print(optimizer.lr)
```

```text
2 0.01
True
0.001
```

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

- `mean`, `nanmean`, `tensor_std`, `var`, `prod`, `sum`, `nansum`
- `max`, `min`, `nanmax`, `nanmin`

The standard-deviation helper is exported as `tensor_std`, not `std`, so it does
not shadow the builtin-shaped name when the module is star-imported.

`numpy_compat.tensor_std(tensor, axis=None, keepdims=None, ddof=None)` and
`numpy_compat.var(tensor, axis=None, keepdims=None, ddof=None)` accept a single
integer axis or `None`; `ddof=0` maps to population statistics and `ddof=1` maps
to unbiased sample statistics. Values outside `0` and `1` are rejected because
the current tensor engine exposes a boolean unbiased flag rather than arbitrary
correction values.

```python
from minitensor import numpy_compat as npc

a = npc.asarray([[1.0, 2.0], [3.0, 4.0]])
b = npc.asarray([[5.0, 6.0], [7.0, 8.0]])

print(npc.vstack([a, b]).shape, npc.hstack([a, b]).shape)
print(npc.concatenate([a, b], axis=0).shape)
print(npc.matmul(a, b).tolist())
print(npc.sum(a).tolist(), npc.mean(a).tolist(), npc.max(a).tolist())

# ddof selects population (0) vs unbiased sample (1) statistics
print(npc.var(a, ddof=0).tolist(), round(npc.var(a, ddof=1).tolist(), 4))
print(npc.allclose(a, a), npc.array_equal(a, b))

try:
    npc.tensor_std(a, ddof=2)
except ValueError as exc:
    print("ddof=2 rejected:", isinstance(exc, ValueError))
```

```text
Shape([4, 2]) Shape([2, 4])
Shape([4, 2])
[[19.0, 22.0], [43.0, 50.0]]
10.0 2.5 4.0
1.25 1.6667
True False
ddof=2 rejected: True
```

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

Any module can write its own weights with `module.save(path, format=...)` and
read a saved file back with `Module.load_state_from(path, format=...)`, which
returns a `StateDict` that `load_state_dict` applies. Passing `format=None`
picks the format from the file extension.

Parameters and buffers are keyed by name: `weight` / `bias` for the dense,
convolution and normalization layers, `running_mean` / `running_var` for
BatchNorm's buffers, `q_proj` / `k_proj` / `v_proj` / `out_proj` (and their
biases) for attention, and PyTorch's `weight_ih_l{k}` / `weight_hh_l{k}` /
`bias_ih_l{k}` / `bias_hh_l{k}` for the recurrent layers, with `_reverse`
appended for the backward direction of a bidirectional stack. `Sequential`
prefixes each child with its index (`0.weight`, `2.bias`), recursing so a
nested layer keeps its own names.

A custom `Layer` that does not override `named_parameters` falls back to
positional keys (`param_0`, `param_1`, ...), which still load correctly but
cannot be inspected or reordered.

```python
import os
import tempfile

import minitensor as mt
from minitensor import nn

mt.manual_seed(0)
model = nn.DenseLayer(3, 2)
probe = mt.ones(1, 3)
before = model(probe).tolist()

state = model.state_dict()
print(sorted(state.parameter_names()), sorted(state.buffer_names()))

with tempfile.TemporaryDirectory() as folder:
    path = os.path.join(folder, "model.json")
    model.save(path, format="json")

    restored = nn.Module.load_state_from(path, format="json")
    model.load_state_dict(restored)

print(model(probe).tolist() == before)
```

```text
['bias', 'weight'] []
True
```

## 10) Plugin system (`minitensor.plugins`)

### Versioning and metadata

- `VersionInfo` -- `VersionInfo.parse("1.2.3")`, `VersionInfo.current()`,
  `major` / `minor` / `patch` properties, and `is_compatible_with(other)`.
- `PluginInfo` -- read-only descriptor with `name`, `version`, `author`,
  `description`, `min_minitensor_version` and `max_minitensor_version`
  properties. It is produced by the library rather than constructed directly.

### Python-side plugins

- `PluginBuilder` -- fluent builder. Takes no constructor arguments; chain
  `name`, `version`, `author`, `description` and `min_minitensor_version`, then
  call `build()`, which returns a **`CustomPlugin`** (not a `PluginInfo`).
  A minimum supported version is mandatory -- `build()` raises `ValueError`
  without it.
- `CustomPlugin` -- plugin object exposing an `info` property and the
  `set_initialize_fn` / `set_cleanup_fn` / `set_custom_operations_fn` callback
  hooks.
- `PluginRegistry` -- `register(plugin)`, `unregister(name)`, `list_plugins()`,
  `is_registered(name)`, `get_plugin(name)`.
- `CustomLayer` -- define custom layers in Python: `add_parameter(name, tensor)`,
  `get_parameter(name)`, `list_parameters()`, `set_forward(fn)`, `forward(...)`,
  and a `name` property.

```python
import minitensor as mt
from minitensor.plugins import CustomLayer, PluginBuilder, PluginRegistry, VersionInfo

plugin = (
    PluginBuilder()
    .name("demo")
    .version(VersionInfo.parse("0.1.0"))
    .author("me")
    .description("a demo plugin")
    .min_minitensor_version(VersionInfo.parse("0.1.0"))
    .build()
)

info = plugin.info  # a property, not a method
print(info.name, info.author, f"{info.version.major}.{info.version.minor}")

registry = PluginRegistry()
registry.register(plugin)
print(registry.is_registered("demo"))
registry.unregister("demo")
print(registry.list_plugins())

layer = CustomLayer("mylayer")
layer.add_parameter("w", mt.ones(2, 2))
print(layer.name, layer.list_parameters())
```

```text
demo me 0.1
True
[]
mylayer ['w']
```

### Dynamic loading (if compiled)

These operate on the process-wide registry of natively compiled plugins, which
is empty unless a shared library has been loaded.

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

Available types are `TensorDebugger` (`get_info`, `inspect`, `compare`,
`health_check`), `TensorInfo` (shape, dtype, device, stride, `numel`,
`memory_usage_bytes`, `is_leaf`, `summary`, `detailed`), `MemoryTracker`,
`OperationProfiler` and `Timer`.

```python
import minitensor as mt
from minitensor import _core

debugger = _core.debug.TensorDebugger()
info = debugger.get_info(mt.ones(2, 3))

print(info.shape, info.dtype, info.numel)
print(info.requires_grad, info.is_leaf)
print(debugger.health_check(mt.ones(2, 3)))
```

```text
[2, 3] float32 6
False True
['✅ No issues detected']
```

## 12) Custom operations

MiniTensor supports custom ops in both Rust and Python. Refer to
`docs/custom_operations.md` for:

- The `CustomOp` trait and builder pattern.
- Python registration and execution (`execute_custom_op_py`, etc.).
- Example custom ops (Swish, GELU, power).

### NumPy dtypes and memory layout

The engine carries five dtypes. NumPy arrays of other dtypes are accepted when
the cast is *exact* -- every value round-trips, so the conversion cannot change
a number:

| NumPy dtype | becomes |
| --- | --- |
| `float16` | `float32` |
| `int8`, `int16`, `uint8`, `uint16` | `int32` |
| `uint32` | `int64` |

`uint64` and `longdouble` are refused rather than rounded: values above
`int64`'s maximum, and mantissas wider than `float64`'s, cannot survive the
cast. Choose the cast yourself (`arr.astype('int64')`) so the handling of
values that do not fit is yours.

Memory layout is normalised on the way in, so transposes, Fortran-ordered
arrays, strided slices, negative strides and broadcast views all convert
correctly. An already C-contiguous array is not copied.

### What `backward()` retains

Unlike PyTorch, MiniTensor exposes `.grad` on interior (non-leaf) tensors after
a backward pass:

```python
import minitensor as mt

x = mt.ones((4, 4))
w = mt.ones((4, 4), requires_grad=True)
h = mt.matmul(x, w)              # interior tensor
mt.sum(mt.tanh(h)).backward()
print(h.grad is not None)        # PyTorch would give None here
```

```text
True
```

Keeping that available means the gradient map holds an entry per interior
tensor until something resets it. Both `optimizer.step()` and
`clear_autograd_graph()` do, so an ordinary training loop is bounded:

```python
import minitensor as mt

w = mt.ones((4, 4), requires_grad=True)
optimizer = mt.optim.SGD([w], 1e-3)

for _ in range(3):               # bounded: step() clears the graph
    optimizer.zero_grad()
    h = mt.matmul(mt.ones((4, 4)), w)
    mt.sum(h).backward()
    optimizer.step()

print(mt.get_gradient(h) is None)
```

```text
True
```

A loop that backpropagates *without* stepping an optimizer -- gradient
inspection, a custom optimizer written in Python, accumulating many
micro-batches before one step -- has to clear the graph itself, or memory grows
by roughly one intermediate tensor per iteration (measured at ~65 KB per
iteration for a 256x32 intermediate, ~141 KB for 256x128):

```python
import minitensor as mt

w = mt.ones((4, 4), requires_grad=True)

for _ in range(3):
    h = mt.matmul(mt.ones((4, 4)), w)
    mt.sum(h).backward()
    _ = w.grad                   # inspect it, no optimizer involved
    mt.clear_autograd_graph()    # without this the graph grows without bound

print(mt.get_gradient(h) is None)
```

```text
True
```

Note `zero_grad()` is not a substitute: it clears the gradients of the
parameters it was given, not the interior entries.

## 13) Notes on devices & backends

**Execution is CPU-only.** Every kernel in the engine reads host memory, so CPU
is the only device a tensor can live on. `Device.is_available()` reports this,
and every placement argument enforces it:

```python
import minitensor as mt

print(mt.Device("cpu").is_available())
print(mt.Device("cuda").is_available())

try:
    mt.zeros((2, 2), device=mt.Device("cuda"))
except RuntimeError as err:
    print(err)

try:
    mt.ones((2, 2)).to("metal")
except RuntimeError as err:
    print(err)
```

```text
True
False
device 'cuda:0' is not available: minitensor executes on the CPU only, so tensors cannot be placed on cuda. Use device='cpu' (the default).
device 'metal' is not available: minitensor executes on the CPU only, so tensors cannot be placed on metal. Use device='cpu' (the default).
```

The request fails where the device was named rather than producing a tensor
that reports `device=cuda:0` and then fails in every operation applied to it.

`DeviceType` still models CUDA, Metal, and OpenCL, and the `cuda` / `metal` /
`opencl` Cargo features compile `engine::backends` — device contexts,
allocators, and a small standalone kernel set (add, mul, matmul, relu,
sigmoid) per backend. Nothing in `engine::ops` dispatches to any of it, and
`engine::backends::get_backend` has no callers outside its own tests, so
enabling a backend feature does not change what `is_available()` returns or
what any tensor operation does. Treat those modules as groundwork for GPU
execution, not as a GPU execution path.

Feature flags for the `engine` crate:

| Feature | Default | Notes |
| --- | --- | --- |
| `cpu` | yes | CPU execution. |
| `hardware` | yes | `engine::hardware` profiling/detection. |
| `debug` | yes | `engine::debug` inspector/profiler. |
| `cuda` | no | CUDA allocation scaffolding via `cudarc`. |
| `metal` | no | Metal scaffolding. **Apple targets only** — the dependency is declared under `cfg(target_vendor = "apple")`, so on other platforms the feature resolves to nothing rather than failing the build. |
| `opencl` | no | OpenCL scaffolding via `opencl3`. |
| `gpu` | no | `cuda` + `metal` + `opencl`; builds on every platform, contributing whichever of the three that platform can have. |
| `blas` | no | Routes GEMM through a system OpenBLAS (`libopenblas-dev` or equivalent). `openblas-src` is pinned to `system`, so the build links the installed library rather than downloading and compiling OpenBLAS itself. |
| `dynamic-loading` | no | Runtime plugin loading (`docs/plugin_system.md`). |

Neither `hardware` nor `debug` is used by any tensor or autograd execution
path, so a Rust consumer embedding the engine can drop both:
`--no-default-features --features cpu` builds a 30.4 MB rlib against 46.0 MB
for the default set (measured, release profile). The dependency graph is
unchanged at 132 crates either way — both features are pure Rust code, not
extra dependencies, so the saving is compiled volume rather than build inputs.
The Python extension always needs them: `mt._core.debug` and the hardware
introspection API are part of the published surface.

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
