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
| `nn.init` | Weight initialization schemes. |
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
| `clear_autograd_graph()` | Clear the global autograd graph, releasing every stored gradient — leaves included, so not for use between the backward passes of one accumulation. Training and backpropagating loops stay bounded without it; see below. |
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
| `hstack(tensors)` | Join along the second axis, or the first for 1-D inputs. |
| `vstack(tensors)` | Join along the first axis, after promoting 1-D inputs to rows. |
| `dstack(tensors)` | Join along the third axis, after promoting lower-rank inputs to it. |
| `column_stack(tensors)` | Join as columns: 1-D inputs become columns, the rest stack along axis 1. |
| `tile(input, reps)` | Repeat along each axis. Unlike `repeat`, `reps` may be shorter than the rank; the missing leading entries are taken as 1, which is why both spellings exist. |
| `unbind(input, dim=0)` | Every slice along `dim`, with that dimension removed -- the inverse of `stack`, as `split` is the inverse of `cat`. |
| `tensor_split(input, indices_or_sections, dim=0)` | Split into a *count* of balanced parts, or at explicit indices. `split` takes a piece *size* and leaves a short tail: ten by three is `[3, 3, 3, 1]` for `split` and `[4, 3, 3]` here. |
| `fliplr(input)` | Reverse the columns; needs at least two dimensions. |
| `flipud(input)` | Reverse the rows. |
| `rot90(input, k=1, dims=(0, 1))` | Rotate `k` quarter turns in the plane `dims` spans. |
| `outer(input, other)` | The outer product of two flattened tensors. |
| `vdot(input, other)` | The inner product of two flattened tensors, of any matching shape. `dot` insists on 1-D operands; this flattens first. |
| `kron(input, other)` | The Kronecker product: each element of `input` scaling a copy of `other`. Ranks need not match; the shorter is padded with leading 1s. |
| `dist(input, other, p=2.0)` | The `p`-norm of the difference. |
| `cdist(input, other, p=2.0)` | Every pairwise `p`-distance between the rows of two batches: `(..., n, d)` and `(..., m, d)` give `(..., n, m)`. Forms the difference in full, so it costs `n * m * d` elements. |
| `histogramdd(input, bins=10, range=None, weight=None, density=False)` | The joint histogram of a `(points, dimensions)` sample, as `(counts, edges)`. `histogram` counts along a line; this counts in a box. `bins` is one count for every dimension, one count each, or the edges themselves; `range` bounds each dimension, as a sequence of pairs or the flat `2 * dimensions` form `torch.histogramdd` takes. The axes are bucketed separately and folded into one flat position, so nothing iterates over cells -- the cost is the samples, not the grid, which is the part that grows exponentially. A point outside any dimension is dropped and each last bin holds its own right edge, as in `histogram`. |
| `normalize(input, p=2.0, dim=1, eps=1e-12)` | `input` scaled so each slice along `dim` has unit `p`-norm. `eps` is a floor *under* the norm rather than a term added to it, so a zero vector comes back as zero and every other vector is exactly unit length -- adding `eps` would shrink all of them. |
| `pairwise_distance(input, other, p=2.0, eps=1e-6, keepdim=False)` | The `p`-distance between corresponding rows: the diagonal of `cdist`, at `n` distances rather than `n * m`. Operands broadcast. `eps` is added to the difference, biasing every distance up by `eps * d ** (1 / p)`; it matches `torch.nn.functional.pairwise_distance` and is a compatibility default, not a requirement -- PyTorch needs it to avoid a NaN gradient where two rows coincide, and this library's `norm` answers zero there, so `eps=0.0` gives the true distance safely. |
| `pdist(input, p=2.0)` | The `p`-distance between every pair of rows without the repeats: `n * (n - 1) / 2` values, ordered by row then column -- the strict upper triangle of `cdist(x, x)`. Built from the pairs rather than from that matrix, so it forms half as many differences as it would discard. |
| `diff(input, n=1, dim=-1)` | The `n`-th discrete difference along `dim`. |
| `trapezoid(y, x=None, dx=1.0, dim=-1)` | The trapezoidal integral along `dim`, with uneven spacing when `x` is given. `trapz` is the same function. |
| `cov(input, correction=1, fweights=None, aweights=None)` | The covariance matrix of the *rows*: each row a variable, each column an observation. A 1-D input is one variable, so the result is its scalar variance. `fweights` counts repeated observations; `aweights` weights their reliability and shrinks the effective sample size rather than the count. |
| `corrcoef(input)` | The Pearson correlation matrix of the rows, clamped to `[-1, 1]` -- the division is exact in theory and lands a hair outside it in floating point. |
| `take(input, index)` | The elements at flat positions `index`, shaped like `index`. Reads the tensor in row-major order whatever its shape; negative positions count from the end. |
| `put(input, index, source, accumulate=False)` | The write direction of `take`: `input` with `source` written at the flat positions `index` names, read row-major over the whole tensor and counting negative positions from the end. `accumulate` adds into the target instead of overwriting, which is also what decides what a repeated position means. |
| `take_along_dim(input, indices, dim=None)` | One element per position, its `dim` coordinate coming from `indices`. With `dim` omitted both are flattened first, which is what makes `take_along_dim(x, x.argsort())` reorder a whole tensor. |
| `index_add(input, dim, index, source, alpha=1.0)` | Add `alpha * source` into the slices `index` names. Repeated indices accumulate. |
| `index_copy(input, dim, index, source)` | Write `source` over the slices `index` names. A repeated index leaves whichever write landed last. |
| `index_fill(input, dim, index, value)` | Set the slices `index` names to `value`. |
| `masked_scatter(input, mask, source)` | Fill the positions `mask` selects with the leading elements of a flattened `source`, in order -- a positional write, where `masked_fill` writes one value everywhere. A source of another dtype is refused rather than promoted, so the write cannot change the dtype of the tensor it writes into. |
| `slice_scatter(input, src, dim=0, start=None, end=None, step=1)` | `input` with `src` written into the slice along `dim`, as a new tensor -- the functional form of `x[..., start:end:step, ...] = src`, for when the write has to be an expression or the tensor has to keep its place in the graph. `start`, `end` and `step` are resolved by a Python slice, so negative steps and out-of-range bounds mean what they mean there. The gradient reaches `src` at the positions it landed on and `input` everywhere else. |
| `select_scatter(input, src, dim, index)` | The same write against one position rather than a range, so `src` has one axis fewer -- lined up with what `select(input, dim, index)` returns, where `slice_scatter` lines up with `narrow`. |
| `diagonal_scatter(input, src, offset=0)` | `input` with `src` written onto the diagonal, where `src` has the shape `diagonal(input, offset)` returns. An `offset` that runs off the matrix writes nothing rather than raising, which is what `diagonal` does for it too. |
| `select(input, dim, index)` | One slice along `dim`, with that dimension removed. `narrow` keeps the axis at length one; this is what makes `select(t, 0, i)` the same as `t[i]`. |
| `flatnonzero(input)` | The flat positions of every non-zero element, as a 1-D int64 tensor. |
| `argwhere(input)` | The indices of every non-zero element, one row each -- the same answer `nonzero` gives, under the name NumPy users reach for. |
| `isin(elements, test_elements, assume_unique=False, invert=False)` | Whether each element appears in `test_elements`. Sorts the test set once and binary-searches it, so it costs `(n + m) log m` time and `n + m` memory rather than the `n * m` of comparing everything against everything. |
| `tril_indices(row, col, offset=0)` | The `[2, n]` indices of a matrix's lower triangle. `offset` moves the boundary off the main diagonal. |
| `triu_indices(row, col, offset=0)` | The `[2, n]` indices of a matrix's upper triangle. |
| `diag_indices(n, ndim=2)` | The `[ndim, n]` indices of the main diagonal of an `n`-sided cube -- every row the same range, since the main diagonal is where the coordinates agree. Shaped like `tril_indices` and `triu_indices`, so the three are interchangeable. |
| `unravel_index(indices, shape)` | The coordinates of flat positions `indices` in a tensor of `shape`, one tensor per axis -- the form NumPy and PyTorch both return. `stack` them on a new leading axis for the `[ndim, n]` layout the index builders use. Positions are checked against the shape, because one out of range names the wrong element rather than failing. |
| `ravel_multi_index(multi_index, dims)` | The flat position of each coordinate, the inverse of `unravel_index`. Takes either one tensor per axis or a single tensor whose *leading* axis is the coordinate -- which is what `tril_indices`, `triu_indices` and `diag_indices` produce, so their output can be handed straight in. |
| `diagflat(input, offset=0)` | A square matrix with the flattened `input` on its `offset` diagonal. `diag` does this for a vector; this does it for any shape. |
| `block_diag(*tensors)` | Arrange the inputs down the diagonal of one larger matrix, zero elsewhere. A 1-D input is a row and a scalar a one-by-one block. |
| `cartesian_prod(*tensors)` | Every combination of one element from each 1-D input, one row each. A single input comes back unchanged. |
| `t(input)` | The transpose of a matrix, and anything of lower rank unchanged. Declines a rank above two rather than guessing which axes were meant -- name them with `transpose`. |
| `numel(input)` | How many elements the tensor holds, as a Python int. |
| `mm(input, mat2)` | The product of two matrices. `matmul` also broadcasts batches and promotes vectors; this rejects anything that is not two matrices, which is the point of asking by this name. |
| `mv(input, vec)` | A matrix times a vector. |
| `inner(input, other)` | The sum-product over the last axis of each operand -- the dot product for two vectors, and every pair of trailing rows contracted above that. |
| `tensordot(input, other, dims=2)` | Contract over the axes `dims` names, as an integer count or a pair of axis lists. Done by moving the contracted axes to the ends, flattening each side into a matrix and calling `matmul` once: a general contraction *is* a matrix product with the axes rearranged, so this inherits the blocked matmul rather than looping over indices. |
| `addmm(input, mat1, mat2, beta=1, alpha=1)` | `beta * input + alpha * (mat1 @ mat2)`, the fused form a linear layer is written in. `baddbmm(...)` is the batched one. |
| `inverse(input)` | The inverse of each square matrix in the stack -- the `torch` spelling of `inv`. For `inverse(A) @ b`, ask `solve(A, b)` instead -- same answer, without forming the inverse, faster and better conditioned. |
| `pinverse(input, rcond=1e-15)` | The Moore-Penrose pseudo-inverse of each matrix in the stack -- the `torch` spelling of `pinv`, keeping that name's threshold of `1e-15` rather than `pinv`'s own `max(m, n) * eps`. The threshold is what makes it a pseudo-inverse rather than a division by nearly zero. |
| `matrix_exp(input)` | The matrix exponential `sum_k A**k / k!` of each square matrix -- the solution operator of `dx/dt = A x`, not `exp` applied elementwise. Scaling and squaring with a Pade approximant, at the degree and halving count Higham's 2005 analysis gives for the input's precision, so float32 takes a shorter route rather than the same one at a worse answer. Every step is a `matmul`, a `solve` or a scalar multiply, so the gradient is the exact derivative of the approximant that was evaluated. A batch shares one scaling, chosen from the largest norm in it. |
| `matrix_norm(input, ord="fro", keepdim=False)` | A norm of each matrix over its last two axes. `"fro"` is the elementwise 2-norm and `"nuc"` the sum of the singular values; `1` and `inf` are the induced norms (largest absolute column and row sum) and `2` the largest singular value, with each negative order the same quantity minimised. The axes are the last two, as for `inverse`, `diagonal` and `svd` -- `permute` first to use others. A condition number in an order other than 2 is `matrix_norm(a, ord) * matrix_norm(inverse(a), ord)`; `cond` is the 2-norm one, which needs no inverse. |
| `tensorsolve(a, b, axes=None)` | Solve `a x = b` where the contraction runs over several axes at once: `a` has the shape of `b` followed by the shape of the answer, and the system is the square one that flattening each half gives. `axes` names axes of `a` to move to the end first. |
| `tensorinv(a, ind=2)` | The inverse of `a` seen as a matrix split at axis `ind` -- axes before it are the rows, axes after are the columns, and the result has them the other way round, which is what makes `tensordot(tensorinv(a), a, ind)` the identity of that shape. |
| `logdet(input)` | The log of the determinant, `-inf` where it is not positive. Taken from `slogdet`, because the determinant of a large matrix leaves float64's range long before its logarithm becomes uninteresting. |
| `renorm(input, p, dim, maxnorm)` | Scale down the sub-tensors along `dim` whose `p`-norm exceeds `maxnorm`, leaving the rest bit-for-bit unchanged -- which is what makes it usable as an embedding constraint applied every step. |
| `vander(x, N=None, increasing=False)` | The Vandermonde matrix: each row a geometric series in one entry, descending by default so `vander(x) @ c` evaluates a polynomial with `c` in the order people write coefficients. |
| `real(input)`, `conj(input)` | The input itself: every dtype here is real. The names exist because code written against NumPy asks for them defensively. |
| `imag(input)` | Zero everywhere, as a detached constant -- written as `input * 0` it would answer NaN for an infinite input. |
| `angle(input)` | `0` for a positive element and `pi` for a negative one. Reads the sign *bit*, so `angle(-0.0)` is `pi`; a NaN has no argument and stays NaN. Piecewise constant, so it carries no gradient. |
| `unflatten(input, dim, sizes)` | Split one axis into several -- the inverse of `flatten`. One entry of `sizes` may be `-1`. `reshape` can do the same thing only by restating every other dimension, which is the mistake this exists to stop. |
| `msort(input)` | Sort along the first dimension, values only. |
| `hsplit(input, indices_or_sections)`, `vsplit(...)`, `dsplit(...)` | `tensor_split` along the second, first and third axes. `hsplit` takes the first axis of a 1-D input, which is the only one it has to split horizontally. |
| `kthvalue(input, k, dim=-1, keepdim=False)` | The `k`-th smallest value along `dim` and where it came from. `k` counts from one, so `kthvalue(x, 1)` is the minimum. |
| `combinations(input, r=2, with_replacement=False)` | Every combination of `r` elements of a 1-D input, one row each, in the order `itertools.combinations` gives them. |
| `gradient(input, spacing=1.0, dim=None, edge_order=1)` | The numerical derivative of *data* -- second-order accurate in the interior, `edge_order`-accurate at the ends. `spacing` is a step or a coordinate vector, per axis, and the coordinates need not be evenly spaced. Not the autograd gradient: for that, call `backward`. |
| `bernoulli(input)` | A 0/1 draw per element, `input` giving each element's probability. |
| `normal(mean=0.0, std=1.0, size=None)` | A normal draw shifted by `mean` and scaled by `std`. With `size` omitted the shape comes from whichever of the two is a tensor. |
| `multinomial(input, num_samples, replacement=False)` | Draw indices with probability proportional to `input`, a row of weights or a batch of them; they need not sum to one. With replacement it is one `searchsorted` in the cumulative distribution; without, it is the top `k` of `log(w) + Gumbel noise`, which is exactly a weighted sample without replacement in one sort rather than a removal loop. |

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

- `register_custom_op(name, forward, backward=None, num_inputs=1)`
- `execute_custom_op_py(name, inputs)`
- `is_custom_op_registered_py(name)`
- `list_custom_ops_py()`
- `register_example_custom_ops()`
- `unregister_custom_op_py(name)`

`register_custom_op` is the extension point: an operation the library does not
have becomes one it does, participating in autograd on the same terms as a
built-in one, with no Rust toolchain and no rebuild.

`forward` is called with the input tensors as positional arguments and returns
a tensor. `backward`, when given, is called with `(grad_output, inputs, output)`
-- the incoming gradient, a tuple of the saved inputs, and the saved output --
and returns one gradient per input: a bare tensor when there is one input,
otherwise a sequence, in which `None` means no gradient flows to that input.

Whether you write a `backward` decides which of two things you get, and they
are the only two a caller can sensibly mean:

- **Without one**, the forward is recorded like any other Python function and
  the operation differentiates by composition. Use this when the forward is
  ordinary tensor code and its own derivative is the one you want.
- **With one**, the forward runs with gradient recording *off* and the gradient
  is whatever `backward` says. Use this when the true derivative is not the
  useful one -- a straight-through estimator through a step function, say,
  whose real gradient is zero everywhere -- or when you can write the
  derivative more cheaply or more stably than the chain rule would derive it.

```python
import minitensor as mt


# A straight-through estimator: the true derivative of a step is zero
# everywhere, so the backward hands back the identity instead.
def step(x):
    return (x > 0.0).astype("float64")


mt.register_custom_op("step_through", step, lambda grad, inputs, output: grad)

x = mt.Tensor([-1.0, 0.5, 2.0], dtype="float64", requires_grad=True)
mt.execute_custom_op_py("step_through", [x]).sum().backward()
print(x.grad.tolist())  # [1.0, 1.0, 1.0]

mt.clear_autograd_graph()
mt.unregister_custom_op_py("step_through")
```

A gradient whose shape or dtype does not match its input is refused rather than
accumulated into a buffer it does not fit, and an exception raised inside
either callable is reported with its own message, so a traceback from your own
code is what you see.

## 2) Tensor creation API

Every creation helper is available as either `mt.<name>(...)` or
`Tensor.<name>(...)`.

Each `*_like` form copies the source tensor's shape, dtype, device *and*
`requires_grad` — so `zeros_like(parameter)` is itself trainable, which differs
from `torch.zeros_like`, where the flag defaults to `False`. Pass
`requires_grad=` explicitly to say otherwise:

```python
import minitensor as mt

parameter = mt.zeros(3, requires_grad=True)
print(mt.zeros_like(parameter).requires_grad)                      # True
print(mt.zeros_like(parameter, requires_grad=False).requires_grad)  # False
```

The result is a leaf either way: it is built from the source's *metadata*, not
its values, so nothing flows back to the source through it.

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

`astype` between two float dtypes is differentiable: a cast is the identity on
values, so the gradient passes straight through and comes back at whatever
precision the input was held in. That is what lets a parameter be kept in one
precision and used in another:

```python
import minitensor as mt

weights = mt.Tensor([[1.0, 2.0], [3.0, 4.0]], dtype="float64", requires_grad=True)
batch = mt.Tensor([[1.0], [1.0]], dtype="float32")

weights.astype("float32").matmul(batch).sum().backward()
print(mt.get_gradient(weights).dtype)   # float64 -- the parameter's own precision
```

It is also the route mixed-dtype arithmetic takes: operands are promoted to a
common dtype through the same conversion, so `float32_tensor * float64_tensor`
back-propagates to both sides.

Promotion follows PyTorch rather than NumPy where the two differ: an integer
operand takes the float operand's width (`int64 + float32` is `float32`, not
`float64`), and `/` always produces a float (`int64 / int64` is `float32`).
A `bool` operand promotes to whatever it is paired with.

Whether an operation accepts a boolean operand is decided by that promoted
dtype, not by the operands. `-`, `//` and `%` have no boolean result to land
in, so they are rejected when *both* sides are `bool` — as they are in NumPy —
and accepted for every mixed pair, where the mask promotes and the operation is
ordinary arithmetic (`counts - mask`). The ordered comparisons `lt`, `le`, `gt`
and `ge` accept booleans with `False < True`, the same ordering `minimum` and
`maximum` apply to them.

Casting to `int32`, `int64` or `bool` returns a tensor with
`requires_grad=False`. Those dtypes cannot carry a gradient, and reporting
`True` for one would describe a tensor that looks tracked and then contributes
nothing to a backward pass.

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

`squeeze(dim)` follows PyTorch rather than NumPy for an axis that is not
length 1: it returns the tensor unchanged instead of raising. `squeeze()` with
no argument drops every length-1 axis.

### Splitting an axis

- `split(size_or_sections, dim)` cuts into pieces of `size` each, the last one
  shorter if the axis does not divide evenly, or into explicitly given sizes.
- `chunk(sections, dim)` cuts into `sections` pieces of equal size.
- `split_with_sections(sections, dim)` takes the sizes explicitly.

`chunk` requires the axis length to be a multiple of `sections` and raises
otherwise. This is stricter than PyTorch's `chunk`, which shortens the last
piece (and can return fewer pieces than asked for); use `split` when the axis
may not divide evenly. All three round-trip through `cat` along the same
dimension, including for a zero-length axis, which yields one empty piece.

### Indexing & reordering

- `index_select`, `gather`, `narrow`
- `scatter(dim, index, src)`, `scatter_add(dim, index, src)`
- `scatter_reduce(input, dim, index, src, reduce, include_self=True)`
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

`scatter_reduce(input, dim, index, src, reduce, include_self=True)` is the
general form the two of them are cases of. `reduce` is `"sum"`, `"prod"`,
`"amax"`, `"amin"` or `"mean"`; `scatter` is replacement and `scatter_add` is
summation, and they keep their own names because those two are what most callers
want.

`include_self` decides whether a destination's existing value takes part in the
reduction. With it off, a written destination starts from the reduction's
identity — zero for a sum, one for a product, an infinity for an extremum —
rather than from what it held. It changes nothing for replacement, which
overwrites either way, nor for summation, where starting from zero and adding to
a zero agree. A destination *nothing* writes to keeps its value regardless: it is
not seeded, not averaged, and not reduced.

`"mean"` over an integer tensor is refused rather than truncated. Every other
reduction works on integers.

All of them are differentiable. `"amax"`/`"amin"` route each destination's
gradient to the contributor that won it, with a tie going to the first — the
rule `max`, `mode` and `cummax` follow here, where PyTorch spreads a tie evenly.
`"mean"` divides by the same count the forward divided by. `"prod"` needs the
product of every contribution *except* each one, and computes it by counting
zeros rather than dividing the total: `total / factor` is the obvious form and
it is wrong exactly when a factor is zero, which is when the other factors still
have gradients worth reporting.

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

Assignment through a basic subscript broadcasts the same way, matching the
value against the selection right-aligned: each of the value's dimensions must
equal the selection's or be `1`, and extra leading dimensions of the value must
be `1`. A value whose *shape* does not broadcast is rejected even when it holds
the right number of elements — `t[0] = m` with `m` shaped `(4, 3)` into a
`(3, 4)` selection raises rather than storing `m`'s elements row-major, as
NumPy does.

A **negative slice step** is rejected, as it is in PyTorch — `t[::-1]` raises
rather than reversing. Use `flip`, which reverses every requested axis in one
pass, and follow it with a positive stride if the step was not `-1`:

```python
import minitensor as mt

t = mt.Tensor([[0.0, 1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0]])

print(t.flip(1).tolist())        # what t[:, ::-1] would give
print(t.flip(1)[:, ::2].tolist())  # what t[:, ::-2] would give
```

The error names the axis and spells out the equivalent call, so the
substitution does not have to be worked out from the rule.

### Distinct values, runs, and the most common one

- `unique(input, return_inverse=False, return_counts=False)` — the distinct
  values, ascending. The input is flattened: this asks which values occur, not
  where. `return_inverse` gives, in the input's shape, the position of each
  element's value in the output, so indexing the output by it rebuilds the
  input. `return_counts` gives how many times each value occurred. With no flags
  the values come back on their own rather than in a one-tuple.
- `unique_consecutive(input, return_inverse=False, return_counts=False)` — the
  same, but collapsing only *adjacent* runs and sorting nothing, so a value that
  recurs after something else appears again. This is run-length encoding, and it
  is what `unique` would destroy.
- `mode(input, dim=-1, keepdim=False)` — `(values, indices)`: the value
  occurring most often along `dim`, and where it is. A tie goes to the smaller
  value and the index is its *first* position along `dim`. Both are choices — a
  tie has no natural winner and a repeated value has no natural occurrence — so
  both are fixed and tested rather than left to fall out of the sort.

`NaN` gets the treatment NumPy gives it, and it is worth being explicit because
the two obvious implementations are both wrong in different ways. `NaN` is not
ordered against anything, so a comparison sort over raw floating-point order has
no defined result; and `NaN != NaN`, so a run detector over `==` emits every
`NaN` as its own distinct value. Here one comparison puts `NaN` after every
number and calls it equal to itself, so `unique` answers `[1.0, nan]` for
`[nan, 1.0, nan]`.

None of these is differentiable: `unique` returns a subset of its input and
which subset changes discontinuously as values collide, and `mode` returns a
value that jumps as counts cross.

```python
import minitensor as mt

symbols = mt.Tensor([7.0, 3.0, 7.0, 9.0, 3.0, 7.0])
vocabulary, encoded = mt.unique(symbols, return_inverse=True)
print(vocabulary.numpy())
print(encoded.numpy())

labels = mt.Tensor([0.0, 0.0, 0.0, 1.0, 1.0, 0.0])
runs, lengths = mt.unique_consecutive(labels, return_counts=True)
print(runs.numpy(), lengths.numpy())
```

```text
[3. 7. 9.]
[1 0 1 2 0 1]
[0. 1. 0.] [3 2 1]
```

### Sorted search, bucketing and histograms

Four operations and one binary search, none of which composes out of the rest of
the library: comparing every value against every boundary is
`O(values × boundaries)` and still leaves the counting to do.

- `searchsorted(sorted_sequence, values, right=False)` — where each value would
  be inserted to keep the sequence sorted, as `int64`. `right` chooses which end
  of a run of equal elements the insertion lands on. A one-dimensional sequence
  is searched by every value; a batched one is matched row for row along its
  last axis. The sequence is *assumed* sorted and never checked, since checking
  costs the linear scan the search exists to avoid.
- `bucketize(input, boundaries, right=False)` — the same call with the arguments
  the other way round, which is the reading that fits when the sequence is a
  fixed set of bucket boundaries.
- `histogram(input, bins=10, range=None, weights=None, density=False)` —
  `(counts, edges)`. `bins` is a count or a one-dimensional tensor of edges. The
  input is flattened; a histogram is a question about a collection of numbers,
  not about their arrangement.
- `histc(input, bins=100, min=0.0, max=0.0)` — PyTorch's spelling: counts alone,
  and equal bounds mean "span the data" rather than "an empty range".

Two conventions worth stating because they are easy to get backwards. Values
outside the outermost edges are **dropped, not clamped** — clamping would pile
everything below the range into the first bin, which is a different answer. And
the last bin is **closed on the right** while every other bin is half-open,
without which the largest value in the data would fall out of its own histogram.

None of these is differentiable, and not for convenience: the result is an index
or a count, an integer that moves in jumps as a value crosses a boundary. There
is no derivative to report, so they return `int64` (or `float64` counts) and
detach.

```python
import minitensor as mt

table = mt.Tensor([0.0, 1.0, 2.0, 3.0])
print(mt.searchsorted(table, mt.Tensor([-1.0, 1.0, 2.5, 9.0])).numpy())
print(mt.searchsorted(table, mt.Tensor([1.0]), right=True).numpy())

counts, edges = mt.histogram(mt.Tensor([0.0, 0.5, 1.0]), bins=2, range=(0.0, 1.0))
print(counts.numpy())          # the last bin is closed on the right
```

```text
[0 1 3 4]
[2]
[1. 2.]
```

### Einstein summation

`einsum(equation, *operands)` is one notation for every product-and-sum over
axes. Name each operand's axes with a letter, name the axes the result keeps,
and every axis you do not keep is summed:

```python
import minitensor as mt

a = mt.ones((3, 3))
b = mt.ones((3, 4))
u, v = mt.ones((3,)), mt.ones((4,))
q, k = mt.ones((2, 5, 7, 4)), mt.ones((2, 5, 9, 4))

print(mt.einsum("ij,jk->ik", a, b).shape)           # a matrix product
print(mt.einsum("ij->ji", b).shape)                 # a transpose
print(mt.einsum("ii->i", a).shape)                  # the diagonal
print(mt.einsum("ii", a).shape)                     # the trace: `i` is not kept
print(mt.einsum("i,j->ij", u, v).shape)             # an outer product
print(mt.einsum("ij,ij->", b, b).shape)             # a Frobenius inner product
print(mt.einsum("...ij,...jk->...ik", a, b).shape)  # batched over leading axes
print(mt.einsum("bhqd,bhkd->bhqk", q, k).shape)     # attention scores
```

```text
Shape([3, 4])
Shape([4, 3])
Shape([3])
Shape([])
Shape([3, 4])
Shape([])
Shape([3, 4])
Shape([2, 5, 7, 9])
```

The last line is why this exists rather than being a convenience: a contraction
over four axes with two of them batched has no other spelling here, only a chain
of permutes and reshapes the caller has to get right.

Omit `->` and the result keeps every subscript used exactly once, ordered by how
the letters sort — NumPy's rule, so `"ij,jk"` is a matrix product and `"ii"` is a
trace. `...` stands for any number of leading axes and broadcasts across a rank
mismatch, aligned from the right as broadcasting is everywhere else. A subscript
repeated within one operand takes its diagonal; a size-1 axis broadcasts against
a longer one with the same subscript.

Operands are contracted a pair at a time, each pair permuted into
`(batch, left, contracted)` against `(batch, contracted, right)` and handed to
the same matrix multiply as `matmul` — so `"ij,jk->ik"` never builds the
`i × j × k` intermediate the naive reading would, and runs at the speed of the
matrix product it is. Every step is an operation that already carries a
gradient, so `einsum` is differentiable in every operand without a backward pass
of its own.

### Linear algebra & matrix ops

- `matmul`, `dot`, `bmm`
- `solve`, `inv`, `det`, `slogdet`
- `diagonal`, `trace`, `diag`, `diag_embed`
- `triu`, `tril`
- `matrix_power(input, power)` -- by repeated squaring, so a large exponent
  costs `log2(power)` products rather than `power` of them. Zero gives the
  identity; a negative power inverts the *base* before squaring, since
  inverting the result would invert its condition number too.

`diag` is NumPy's: a vector in gives a matrix with it on the diagonal, a matrix
in gives its diagonal. `diagonal` and `diag_embed` are the batched forms that
take an axis pair and an offset, and they are each other's inverse -- and each
other's derivative.

#### Factorisations

Each takes a stack of matrices — anything with two or more dimensions — and
factors every matrix in it. All of them are `float32` or `float64` only.

- `cholesky(input, upper=False)` — `A = L @ L.T` for symmetric positive-definite
  `A`, or `A = U.T @ U` with `upper=True`. Reads only the lower triangle, and
  reports which leading minor failed when the matrix is not positive definite.
- `qr(input, mode="reduced")` — `A = Q @ R` with `Q` orthonormal and `R` upper
  triangular. `mode="complete"` returns a square `Q`; that shape is not
  differentiable when there are more rows than columns, because the extra
  columns are an arbitrary completion of the basis. `mode="r"` returns `R`
  alone with an `[m, 0]` `Q` beside it: `R` falls out of the reduction while
  `Q` has to be built back out of the reflectors afterwards, so skipping it is
  about twice as fast. It is not differentiable either, since the gradient of
  `R` is written in terms of the `Q` it does not compute.
- `eigh(input)` — `(w, V)` for a symmetric matrix, with `w` **ascending** and
  `A @ V == V @ diag(w)`. Reads only the lower triangle.
  `eigvalsh(input)` returns the eigenvalues alone and skips accumulating the
  vectors, which is most of the work.
- `svd(input, full_matrices=True)` — `(U, s, Vh)` with
  `A == U @ diag(s) @ Vh` and `s` **descending** and non-negative. With
  `full_matrices=False` the two orthogonal factors are cut to the `min(m, n)`
  columns that carry a singular value, which is the shape that reconstructs `A`.
  `svdvals(input)` returns the singular values alone, and skips building the
  two factors rather than building and discarding them -- the same values to
  the last bit, several times faster (36ms against 189 at 400 by 400), which
  `matrix_rank`, `cond` and `matrix_norm` at orders `2`, `-2` and `"nuc"`
  all inherit.
- `lu_factor(input)` — `(LU, pivots)`: the packed factorisation of a general
  square matrix, with `L` unit lower triangular strictly below the diagonal and
  `U` on and above it. `pivots` is `int64` and **zero-based**: step `i` exchanged
  row `i` with row `pivots[..., i]`. `lu(input)` spells the same thing out as
  `(P, L, U)` with `A == P @ L @ U`, built from the packed form rather than
  computed separately.

The orders differ on purpose: ascending eigenvalues and descending singular
values are what LAPACK, NumPy and PyTorch all return.

The `LU` factors come back detached. A pivoted factorisation's derivative is not
implemented here; `solve`, `det`, `slogdet` and `inv` carry theirs and run this
very factorisation, so they are the differentiable way to ask about a general
square matrix.

#### Solving against a factorisation you already have

- `lu_solve(lu, pivots, b)` — solve `A X = B` from what `lu_factor` returned,
  without factorising again. That is the reason the packed form is worth
  keeping: several right-hand sides against one matrix cost one factorisation
  and a pair of substitutions each, rather than a full elimination every time.
- `solve_triangular(a, b, upper=False, left=True, unitriangular=False)` — solve
  `A X = B` for triangular `A`, reading **only** the named triangle. Whatever is
  in the other half is ignored rather than checked, which is what lets a packed
  factorisation be passed straight in. `unitriangular=True` additionally ignores
  the diagonal and treats it as ones. `left=False` solves `X A = B` instead,
  which is the same routine on transposes.
- `cholesky_solve(b, factor, upper=False)` — solve `A X = B` given the Cholesky
  factor of `A` rather than `A` itself. Two triangular solves and nothing else,
  written as exactly that composition, which is why it needs no kernel of its
  own and no gradient of its own.

`solve_triangular` is differentiable in both arguments, and `cholesky_solve`
inherits that by composition. The matrix gradient is zero in the triangle that
was never read — the half that did not touch the answer cannot be told to
change.

`b` may be a matrix of right-hand sides, `(..., n, k)`, or a single vector
written without the trailing one, `(..., n)`, and the result matches whichever
was given.

#### What a decomposition is usually for

Four operations are a singular value decomposition read out a particular way,
and none of them is reachable without one. `inv` and `solve` need a square
non-singular matrix and `qr` needs full column rank; these need nothing.

- `pinv(input, rcond=None)` -- the Moore-Penrose pseudo-inverse, the unique
  matrix satisfying the four Penrose conditions. For an invertible square matrix
  it is the inverse; for anything else it inverts the directions that are
  invertible and sends the rest to zero, which is what makes it the
  least-squares answer rather than a failure.
- `lstsq(a, b, rcond=None)` -- the `x` minimising `||a @ x - b||`, and the one of
  smallest norm when there are many. `b` may be a matrix of right-hand sides or
  a single vector, and the result matches.
- `matrix_rank(input, tol=None)` -- how many singular values are
  distinguishable from zero, as `int64`. The only numerically meaningful rank
  for inexact entries: a matrix one rounding away from rank three has rank
  three, whatever exact arithmetic on its stored digits would say.
- `cond(input)` -- the 2-norm condition number, the largest singular value over
  the smallest. A singular matrix gives infinity rather than an error, because
  that is the true answer and a caller comparing against a threshold should not
  have to catch it.

They share one tolerance rule. A singular value is never exactly zero in
floating point, so each of these has to decide which ones count, and the answer
is `max(m, n) * eps` relative to the largest -- the accuracy the factorisation
itself offers. `rcond` and `tol` override it, `rcond` relatively and `tol`
absolutely.

`pinv` and `lstsq` are differentiable; `matrix_rank` counts and so is an integer,
and `cond` is a ratio of two extreme singular values whose gradient is a
subgradient at best, so both detach rather than hand back something that looks
differentiable and is not.

```python
import minitensor as mt
import numpy as np

rng = np.random.default_rng(0)
a = mt.Tensor(rng.standard_normal((5, 5)))

# One factorisation, then a solve per right-hand side.
packed, pivots = mt.lu_factor(a)
first = mt.lu_solve(packed, pivots, mt.Tensor(rng.standard_normal((5, 2))))
print(first.numpy().shape)

# The factors spelled out, and the claim they make.
p, l, u = mt.lu(a)
print(bool(np.allclose((p @ l @ u).numpy(), a.numpy())))

# Only the named triangle is read, so the packed form goes straight in.
identity = mt.Tensor(np.eye(5))
below = mt.solve_triangular(packed, identity, unitriangular=True)
print(bool(np.allclose(below.numpy(), np.linalg.inv(np.tril(packed.numpy(), -1) + np.eye(5)))))
```

```text
(5, 2)
True
True
```

```python
import minitensor as mt
import numpy as np

a = mt.Tensor(np.random.default_rng(0).standard_normal((9, 4)))
b = mt.Tensor(np.random.default_rng(1).standard_normal((9,)))

print(mt.matrix_rank(a).item())          # 4
print(mt.lstsq(a, b).numpy().shape)      # (4,)
print(mt.pinv(a).numpy().shape)          # (4, 9) -- the transposed shape
```

```text
4
(4,)
(4, 9)
```

Eigenvectors and singular vectors are determined only up to the sign of each
column, and within a repeated value's subspace only up to a rotation. Nothing
here imposes a convention, so compare what the vectors *do* —
`A @ V == V @ diag(w)`, `V.T @ V == I` — rather than comparing them against
another implementation's elementwise.

That freedom is also where the gradients stop existing. Differentiating the
vectors of a matrix with a repeated eigenvalue or singular value gives
infinities, because any rotation inside the shared subspace is as good and there
is no derivative to report; the *values* stay perfectly well defined there.
`svd` additionally divides by the singular values in the terms that reach
outside the column space, so a rank-deficient **rectangular** matrix has no
vector gradient either — a square one never forms those terms.

```python
import minitensor as mt

a = mt.Tensor([[3.0, 1.0], [0.0, 2.0]], dtype="float64")
u, s, vh = a.svd(full_matrices=False)
print(s.numpy())                                # descending
print(a.allclose(u @ mt.diag_embed(s) @ vh))    # back to a

cov = mt.Tensor([[2.0, 1.0], [1.0, 2.0]], dtype="float64")
w, v = cov.eigh()
print(w.numpy())                          # ascending: [1., 3.]
```

```text
[3.25661654 1.84240298]
True
[1. 3.]
```

### Reductions, statistics, and equality

- `sum`, `mean`, `median`, `nanmedian`, `quantile`, `nanquantile`
  (the `nan*` reductions return NaN for all-NaN slices, matching NumPy)
- `std(dim=None, unbiased=True, keepdim=False)`
- `var(dim=None, unbiased=True, keepdim=False)`
- `nansum`, `nanmean`, `nanmax`, `nanmin`, `nanprod`
- `nanvar(dim=None, unbiased=True, keepdim=False)`, `nanstd(...)`
- `nanargmax(dim=None, keepdim=False)`, `nanargmin(...)`
- `logsumexp`
- `norm(p=2, dim=None, keepdim=False)`
- `isclose(other, rtol=1e-5, atol=1e-8, equal_nan=False)`
- `array_equal(other)`
- `allclose(other, rtol=1e-5, atol=1e-8, equal_nan=False)`

#### Running totals that are not sums

`cumsum` and `cumprod` accumulate with `+` and `*`. Three more accumulate with
something else:

- `cummax(input, dim=-1)` and `cummin(input, dim=-1)` return
  `(values, indices)`: the running extremum and the position it came from. A tie
  keeps the *earliest* position, and a `NaN` takes over the running extremum and
  holds it -- including its index, so a later `NaN` does not quietly move the
  index while leaving the value alone. The gradient goes to the winning
  positions, and a value that wins several prefixes collects all of them.
- `logcumsumexp(input, dim=-1)` is the running `log(sum(exp(x)))`.

`logcumsumexp` is not `cumsum` of `exp`, and the difference is the reason it
exists. `exp` of a log-probability underflows to zero long before a real
sequence ends, so the naive reading stops accumulating and reports `-inf` for
everything after that point. Four thousand steps at `-800` is `-inf` everywhere
under `cumsum` of `exp`, and `-800 + log(4000)` here. Accumulating in the log
domain keeps every step representable.

```python
import minitensor as mt
import numpy as np

scores = mt.Tensor([3.0, 1.0, 4.0, 1.0, 5.0], dtype="float64")
values, positions = mt.cummax(scores)
print(values.numpy())
print(positions.numpy())

# A distribution's total is one, so the log of its running total ends at zero.
weights = mt.Tensor(np.log(np.full(2000, 1 / 2000)), dtype="float64")
# `abs` only so the printed sign is stable: the last log-add lands a few
# ulp either side of zero.
print(abs(round(float(mt.logcumsumexp(weights).numpy()[-1]), 12)))
```

```text
[3. 3. 4. 4. 5.]
[0 0 2 2 4]
0.0
```

#### What dtype a reduction comes back in

The rule turns on whether a reduction *accumulates*.

`sum`, `nansum`, `prod`, `cumsum` and `cumprod` build a running total, so the
result can leave the range of the input even when the input is unremarkable.
These widen a narrow integer to `int64`: `bool` and `int32` inputs come back as
`int64`, matching NumPy and PyTorch. Floats are unchanged — `float32` sums in
`float32` — because promoting to `float64` would alter every existing result
and double the memory of the most common reduction in the library, which is
also NumPy's reasoning.

Reductions that *select* keep the input dtype, because they report a value that
was already there: `max`, `min`, `amax`, `amin`, `sort`, `topk` and the
quantiles. `argmax` and `argmin` return `int64` positions.

`max(dim=...)` and `min(dim=...)` return `(values, indices)`. `amax`/`amin`
return the values alone, and are considerably cheaper for it: carrying the
winning position turns a vectorized compare-and-select into a fold that has to
branch to update an index. Reducing a 2048x1024 `float32` matrix along its last
axis measured 0.109ms for `amax` against 0.833ms for the pair, so
`t.max(dim=1)[0]` — the obvious way to write "row maxima" — costs about 7.6
times what it needs to. `nanamax`/`nanamin` are the NaN-skipping forms; `amax`
and `amin` propagate NaN, as `max` and `min` do. The names are NumPy's and
PyTorch's.

#### The NaN-skipping statistics

`nansum`, `nanmean`, `nanmax`, `nanmin`, `nanamax` and `nanamin` each carry a
kernel that walks the buffer once and tests as it goes. The rest of the family
needs no kernel of its own, because each is an arrangement of operations that
already exist:

- `nanprod(input, dim=None, keepdim=False)` — the product with NaN read as the
  multiplicative identity, so an all-NaN slice gives `1`, exactly as a genuinely
  empty one does.
- `nanvar(input, dim=None, unbiased=True, keepdim=False)` and `nanstd(...)` —
  the mean of the squared deviations from `nanmean`, over the entries that are
  not NaN. `unbiased` divides by the non-NaN count less one, so a slice with a
  single finite entry gives `NaN` and one with none gives `NaN` as well, which
  is what NumPy reports for the same input.
- `nanargmax(input, dim=None, keepdim=False)` and `nanargmin(...)` — the index
  of the largest or smallest entry that is not NaN. A slice of nothing but NaN
  raises: every index it could name points at a NaN, so there is no answer to
  give, and NumPy raises here too.

Writing them this way is one definition rather than two, and it is what makes
their gradients the gradients of the operations underneath — `nanvar` is
differentiable because `sub`, `mul` and `sum` are, with the deviation zeroed at
the NaN positions before it is squared so no `0 * NaN` reaches the chain rule.

`nanvar`, `nanstd`, `nanargmax` and `nanargmin` reduce one dimension at a time,
since the non-NaN count they divide by comes from a single-axis
`count_nonzero`; passing more than one dimension raises rather than reducing
the wrong count.

`mean` over an integer tensor returns a float: `float32` for `int32`,
`float64` for `int64`. On a `bool` tensor `mean`, `var`, `std`, `norm` and
`logsumexp` raise — what they should return for a mask is a design question
rather than an obvious one.

The widening is what makes an integer total trustworthy. Accumulation is still
two's-complement once it reaches `int64`, so an extreme input can still wrap,
but the everyday case no longer does: summing pixel values in `0..=255` used to
overflow past about 8.4 million elements.

```python
import minitensor as mt

counts = mt.Tensor([[1, 2], [3, 4]], dtype="int32")
print(counts.sum().dtype, counts.sum().item())

big = mt.full((3,), 2_000_000_000, dtype="int32")
print(big.sum().item())                  # exact, not 1705032704

mask = counts.gt(2)                      # a mask counts into int64 too
print(mask.sum().item(), mask.sum().dtype)
```

```text
int64 10
6000000000
2 int64
```

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
| `prod`, `nanprod` | `1` |
| `mean`, `nanmean`, `std`, `var`, `nanvar`, `nanstd` | `NaN` (0/0) |
| `logsumexp` | `-inf` (`log 0`) |
| `max`, `min`, `amax`, `amin`, `nanmax`, `nanmin`, `nanamax`, `nanamin`, `argmax`, `argmin`, `nanargmax`, `nanargmin` | raises |
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
- `floor_divide`, `remainder`, `fmod`
- `add(input, other, alpha=1)`, `sub(...)`, `mul`, `div(input, other, rounding_mode=None)`,
  `neg` — the operators as free functions, and as methods. `a + b` always
  worked; `mt.add(a, b)` and `a.add(b)` are what most code that moves between
  array libraries actually writes. `alpha` scales the second operand,
  `rounding_mode` picks `"floor"` or `"trunc"` instead of the exact quotient.
- `square(input)` — a product, not `pow(input, 2)`: the general power goes
  through `exp(2 log x)` for a non-integral exponent and is both slower and less
  exact than a multiplication, which is exact for every input.
- `lerp(input, end, weight)` — `input + weight * (end - input)`, written as a
  step from `input` so `weight = 0` and `weight = 1` return the endpoints
  exactly rather than approximately.
- `addcmul(input, t1, t2, value=1)`, `addcdiv(...)` — `input + value * t1 * t2`
  and the same with a division.
- `deg2rad`, `rad2deg` — a multiplication by `pi/180` and its inverse.
- `float_power(input, exponent)` — the power computed in float64 whatever the
  inputs are, since an integer power overflows silently once the answer leaves
  the dtype's range.
- `logaddexp2(input, other)` — the base-2 `logaddexp`, computed by rescaling it
  rather than by a second stable implementation of the same shift-and-add.
- `ldexp(input, other)` — `input * 2**other`. Computed as the product, so an
  `other` large enough to overflow `2**other` gives infinity even where the
  product would have been finite; the exponent itself is exact.
- `fmax(input, other)`, `fmin(...)` — the extrema *ignoring* NaN, where
  `maximum`/`minimum` propagate it. NaN survives only where both operands are
  NaN and there is genuinely nothing to compare.
- `isposinf`, `isneginf` — the two halves of `isinf`.
- `isreal` — true everywhere, NaN included: every dtype here is real. The name
  exists because code written against NumPy asks, and a missing attribute is a
  worse answer than the correct one.
- `signbit(input)` — whether the sign *bit* is set, which is not `input < 0`:
  negative zero is not less than zero but carries the bit, and telling the two
  zeros apart is the only reason to ask. Reads the bit through `copysign`.
- `sgn` — the same function as `sign` for real numbers, under the name used
  where a complex version would differ.
- Second spellings, one object under two names: `absolute` (`abs`), `subtract`,
  `multiply`, `divide` and `true_divide` (`div`), `negative` (`neg`), `concat`
  (`cat`), `greater` (`gt`), `greater_equal` (`ge`), `less` (`lt`),
  `less_equal` (`le`), `not_equal` (`ne`). NumPy and PyTorch each settled on a
  different one for several of these, and code moving between them writes
  whichever it learned.
- `bitwise_and`, `bitwise_or`, `bitwise_xor`, `bitwise_not`,
  `bitwise_left_shift`, `bitwise_right_shift`
- `logical_and`, `logical_or`, `logical_xor`, `logical_not`
- `softsign`, `rsqrt`, `reciprocal`, `sign`
- `leaky_relu(input, negative_slope=0.01)` — the gradient at exactly `0` is
  `negative_slope`, the same side `relu` takes, matching PyTorch
- `isnan`, `isinf`, `isfinite`
- `clip`, `clamp`, `clamp_min`, `clamp_max`
- `round`, `floor`, `ceil`, `trunc`, `frac` — `trunc` rounds towards zero and
  `frac` is what it leaves behind, `x - trunc(x)`, carrying `x`'s sign. `frac`
  is the only differentiable one (its gradient is 1); the rest are step
  functions, so they return a constant. `round` sends halves to the even neighbour
  (`round(0.5) == 0`, `round(2.5) == 2`), matching NumPy, PyTorch and Python's
  built-in `round`. It takes an optional `decimals` argument.
- `log2`, `log10` — the natural log rescaled; they share `log`'s behaviour at
  `0`, negatives, infinities and NaN
- `erf`, `erfc` — the Gauss error function and its complement. `erfc` uses a
  dedicated routine rather than computing `1 - erf(x)`: past about `x = 6` in
  float64 `erf(x)` rounds to 1 and that subtraction returns exactly zero, which
  is precisely the tail `erfc` exists to give you (`erfc(20) ≈ 5.4e-176`).
- `erfinv(input)` — the inverse of `erf` on `[-1, 1]`, infinite at the
  endpoints. Anything outside that interval gives NaN: `erf` never reaches
  there, so nothing inverts to it.
- `exp2(input)` — `2 ** x` from the hardware's own base-2 exponential.
  `exp(x * log(2))` rounds the exponent before using it, which costs the last
  few bits of every answer and all of them for a large `x`.
- `sinc(input)` — `sin(pi x) / (pi x)`, taken as `1` at zero. NumPy's
  normalized convention, so the zeros sit on the non-zero integers.
- `lgamma(input)`, `digamma(input)` — `log |gamma(x)|` and its derivative.
  `lgamma` is finite where `gamma` overflows: `gamma(200)` is past the top of
  float64 and `lgamma(200)` is 858. `digamma` differentiates to `trigamma`,
  which the library computes itself.
- `polygamma(order, input)` — the `order`-th derivative of `digamma`, so order
  0 is `digamma` and order 1 is `trigamma`; both are computed by their own
  routes and the general one agrees with them. Its own derivative is the next
  order, exactly, with no second formula and no finite difference.

  `polygamma(n, x)` is `(-1)^(n+1) n! zeta(n+1, x)`, and the two factors sit at
  opposite ends of the range: at order 169 the factorial is `4e304` while the
  zeta at a large argument is far below the smallest double. They are combined
  as logarithms rather than as a product, which is what makes the high orders
  work — `polygamma(169, 1)` is `169!`, and `polygamma(169, 1e4)` is a perfectly
  ordinary small number that forming the product would have lost.

  Orders above 1 take non-negative arguments only, and give NaN below zero.
  Reaching a positive argument by the recurrence sums terms that are enormous
  beside the answer and alternate in sign — by order 6 at `x = -100` there is
  nothing left of it — and the reflection formula that avoids that needs the
  `n`-th derivative of the cotangent, whose coefficients overflow well before
  these orders do. `scipy` stops in the same place. Orders 0 and 1 keep the
  whole line. The order itself must be at most 169, which is where the
  factorial in the derivative stops fitting a double.
- `i0(input)`, `i1(input)`, `i0e(input)`, `i1e(input)` — the modified Bessel
  functions of the first kind, orders zero and one, and each scaled by
  `exp(-|x|)`. `i0` and `i1` grow like `exp(x)` and overflow a double a little
  past 713; the things they are wanted for do not. A Kaiser window is a ratio of
  two `i0`s and a von Mises density divides by one, and in both the
  exponentials cancel — the scaled forms are that cancellation done before it
  can overflow rather than after. `i0` differentiates to `i1`, and `i1` to
  `i0 - i1/x`, which is `1/2` at the origin where the quotient is not.

  A power series below thirty, where the terms are all positive so nothing
  cancels, and an asymptotic series above it taken to a fixed sixteen terms.
  Fixed rather than stopped where the terms turn, which is the usual rule and
  makes the truncation depend on the last bit of the argument — two inputs a
  billionth apart would then differ in the tenth digit.
- `erfcx(input)` — `exp(x**2) erfc(x)`. `erfc` underflows to zero a little past
  26 and `exp(x**2)` overflows a little past 26.6, so above there the product is
  `inf * 0` — while the value it reaches for is ordinary: `erfcx(30)` is
  `0.0188` and `erfcx(1e100)` is `5.6e-101`. Every Gaussian tail computation
  that divides one by another needs it, a Mills ratio being the plainest.
  Its derivative is `2 x erfcx(x) - 2/sqrt(pi)`, where the constant is constant
  because `erfc`'s own derivative cancels the scaling exactly.
- `logit(input, eps=None)` — `log(x / (1 - x))`, the inverse of `sigmoid`.
  With `eps` the input is first pulled into `[eps, 1 - eps]`, which bounds the
  answer for a probability that has rounded to 0 or 1 and flattens the gradient
  there; without it those give infinities and anything outside `[0, 1]` gives
  NaN.
- `sin`, `cos`, `tan`
- `asin`, `acos`, `atan`
- `atan2(input, other)` — the angle of `(other, input)` from the positive
  x-axis, in `(-pi, pi]`. Unlike `atan(input / other)` it keeps the quadrant,
  and it answers on the y-axis instead of dividing by zero.
- `sinh`, `cosh`, `asinh`, `acosh`, `atanh`
- `maximum`, `minimum`
- `hypot(input, other)` — `sqrt(input**2 + other**2)` without forming either
  square, so it answers for operands whose squares would overflow or flush to
  zero
- `copysign(input, other)` — the magnitude of `input` with the sign of
  `other`. Reads the sign *bit*, so `copysign(1, -0.0)` is `-1`; the gradient
  reaches `input` only, since nothing differentiable depends on `other`.
- `xlogy(input, other)` — `input * log(other)`, taken as `0` wherever `input`
  is zero. That is the limit entropy and cross-entropy need, where the plain
  product gives `0 * -inf = NaN`. A NaN `other` still propagates.
- `heaviside(input, other)` — the unit step: `0` below zero, `1` above it, and
  `other` at exactly zero, which is the value no two conventions agree on and
  the reason it takes a second operand. NaN stays NaN, being on neither side.
  The step is flat wherever it is defined, so the gradient reaches `other`
  alone, and only where the input is exactly zero.
- `nextafter(input, other)` — the next representable value after each element
  in the direction of `other`. Steps by one bit pattern, so it crosses zero
  into the smallest subnormal and stops at the largest finite value on its way
  in from infinity. As a real-valued function it differs from the identity by
  one ulp, and that is the gradient it reports.
- `cosine_similarity(input, other, dim=1, eps=1e-8)` — the cosine of the angle
  between them along `dim`. Each norm is floored at `eps` on its own rather
  than their product being floored once, which is what keeps a zero vector
  paired with a long one inside `[-1, 1]`.
- `softplus`, `gelu`, `elu`, `selu`, `silu`
- `hardshrink(input, lambd=0.5)` — zeroes the band `[-lambd, lambd]` and
  leaves the rest where it is; `softshrink(input, lambd=0.5)` zeroes the same
  band but subtracts `lambd` from the rest, so it stays continuous
- `tanhshrink(input)` — `x - tanh(x)`
- `threshold(input, threshold, value)` — `x` where it exceeds `threshold`,
  the constant `value` elsewhere
- `hardtanh(input, min_val=-1.0, max_val=1.0)`, and `relu6(input)` for the
  `[0, 6]` case that quantized networks use
- `hardsigmoid(input)`, `hardswish(input)` — `sigmoid` and `silu` with the
  exponential replaced by three straight lines, for the same reason
- `mish(input)` — `x * tanh(softplus(x))`
- `celu(input, alpha=1.0)` — `elu` rescaled so its slope is continuous at zero
  for every `alpha`, which `elu`'s is only at `alpha = 1`
- `logsigmoid(input)` — `log(sigmoid(x))`, evaluated as `-softplus(-x)`. The
  direct form underflows to `-inf` once `sigmoid(x)` rounds to zero, around
  `x = -104` in float64; this one stays exact and simply returns `x`.
- `softmin(input, dim=None)` — `softmax` of the negated input
- `floor_divide` / `//` — Python floor division (rounds toward negative
  infinity; integer operands stay integral, integer zero divisors raise,
  not differentiable)
- `remainder` / `%` — Python-style remainder (takes the divisor's sign;
  `a == (a // b) * b + a % b` holds for every dtype; differentiable for
  float dtypes)
- `fmod` — the same remainder taking the *dividend's* sign, which is what C's
  `fmod` and Rust's `%` mean: `fmod(-7, 3)` is `-1` where `remainder(-7, 3)` is
  `2`. They agree whenever the operands share a sign, and are one computation
  with one correction step between them — the quotient rounds towards zero
  instead of towards negative infinity, in the value and in `d/dy` alike.
  Integer operands stay integral for both, and both are differentiable for
  float dtypes.
- `bitwise_and` / `&`, `bitwise_or` / `|`, `bitwise_xor` / `^`,
  `bitwise_not` / `~` — bit operations on integers, the matching truth table on
  bools; rejected for floats. A bool paired with an integer promotes to that
  integer dtype, as it does for `+`.
- `gcd`, `lcm` — integers only, and always non-negative: a common divisor of
  `-12` and `8` is a common divisor of `12` and `8`, and the positive one is
  what every library reports. `gcd(x, 0)` is the magnitude of `x`, since every
  integer divides zero; `lcm(x, 0)` is `0`, since zero is the least of the
  multiples of zero. `lcm` divides before it multiplies, because the product of
  two operands can leave the dtype even when their multiple does not.
- `bitwise_left_shift` / `<<`, `bitwise_right_shift` / `>>` — integers only
  (two bools have no bits to move). The right shift is arithmetic, so it
  preserves sign and floors. Counts at or past the dtype's width are undefined
  in C, and so in NumPy; here they give the limit of the operation — `0`,
  or `-1` for a right-shifted negative. Negative counts raise.
- `logical_and`, `logical_or`, `logical_xor`, `logical_not` — the same truth
  tables over *truth values* rather than bits, so they accept every dtype and
  always return bool. Each operand is reduced by `x != 0`, which makes every
  non-zero float true (NaN included) and both signed zeros false.

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

`fill_`, `copy_` and index assignment (`t[i] = v`) refuse to write to a leaf
that a pending backward pass still needs, raising rather than corrupting it. Backward nodes hold their
operands by reference, so overwriting one between the forward and the backward
would change what the backward reads — and the damage lands on the *other*
operand's gradient, as a plausible wrong number rather than an error:

```python
import minitensor as mt

a = mt.Tensor([2.0], requires_grad=True)
b = mt.Tensor([3.0], requires_grad=True)
loss = mt.sum(a * b)

try:
    a.fill_(99.0)             # `a` is an operand of a live backward node
except ValueError as exc:
    print("refused:", "pending backward" in str(exc))

loss.backward()
print(b.grad.tolist())        # d/db of a*b is a, the forward value
```

```text
refused: True
[2.0]
```

The orderings you actually want are all still allowed: writing a parameter
before any forward has consumed it, mutating and *then* building the graph, and
clamping between training steps — `backward()` releases the subgraph it walked,
and `clear_autograd_graph()` releases everything. Non-leaf tensors are unaffected
either way, since they copy on write.

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

The list is grouped for reading only; the two spellings are the same object
either way. `test_documented_top_level_names_are_exactly_the_forwarded_ones`
checks it against the exports, so a name here is one the library really has.

```
# Creation and shape
reshape, view, flatten, ravel, transpose, permute, movedim, moveaxis, swapaxes,
swapdims, squeeze, unsqueeze, expand, repeat, repeat_interleave, flip, roll,
pad,

# Matrix products, inverses and rescalings
t, numel, mm, mv, inner, tensordot, tensorsolve, tensorinv, addmm, baddbmm,
inverse, pinverse, matrix_exp, matrix_norm, logdet, renorm, vander, real, conj,
imag, angle,

# Joining, splitting and indexing
cat, stack, split, chunk, index_select, gather, narrow, scatter, scatter_add,
scatter_reduce, masked_fill, masked_select, nonzero, count_nonzero, where,
one_hot,

# Reductions and statistics
sum, prod, mean, std, var, all, any, max, min, amax, amin, argmax, argmin,
median, nanmedian, quantile, nanquantile, nansum, nanmean, nanmax, nanmin,
nanamax, nanamin, nanprod, nanvar, nanstd, nanargmax, nanargmin, logsumexp,
norm, bincount, mode,

# Scans
cumsum, cumprod, cummax, cummin, logcumsumexp,

# Ordering and search
sort, argsort, topk, unique, unique_consecutive, searchsorted, bucketize,
histogram, histc,

# Elementwise arithmetic and rounding
abs, sqrt, exp, log, pow, rsqrt, reciprocal, sign, floor_divide, remainder,
fmod, round, floor, ceil, trunc, frac, clip, clamp, clamp_min, clamp_max,
maximum, minimum, log1p, log2, log10, exp2, expm1, logaddexp, logaddexp2, erf,
erfc, hypot, copysign, xlogy, heaviside, nextafter, fmax, fmin, square,
float_power, ldexp, lerp, addcmul, addcdiv, deg2rad, rad2deg, signbit, sgn,

# The operators as free functions, and second spellings of existing names
add, sub, mul, div, neg, absolute, subtract, multiply, divide, true_divide,
negative, concat, greater, greater_equal, less, less_equal, not_equal,

# Special functions
erfinv, erfcx, sinc, lgamma, digamma, polygamma, logit, i0, i1, i0e, i1e,

# Trigonometry and hyperbolics
sin, cos, tan, asin, acos, atan, atan2, sinh, cosh, asinh, acosh, atanh,

# Comparison, bitwise and logic
eq, ne, lt, le, gt, ge, isclose, allclose, array_equal, cosine_similarity,
isnan, isinf, isfinite, isposinf, isneginf, isreal,
nan_to_num, bitwise_and, bitwise_or, bitwise_xor, bitwise_not,
bitwise_left_shift, bitwise_right_shift, logical_and, logical_or, logical_xor,
logical_not, gcd, lcm,

# Activations and normalization
relu, relu6, leaky_relu, hardtanh, hardshrink, softshrink, tanhshrink,
threshold, sigmoid, hardsigmoid, logsigmoid, softplus, gelu, elu, celu, selu,
silu, hardswish, mish, softsign, tanh, glu, softmax, log_softmax, softmin,
masked_softmax, masked_log_softmax, layer_norm, rms_norm,
scaled_dot_product_attention, rope,

# Linear algebra
matmul, solve, solve_triangular, trace, diagonal, diag, diag_embed, triu, tril,
det, slogdet, inv, pinv, matrix_rank, matrix_power, cond, lstsq, einsum,
cholesky, cholesky_solve, qr, svd, svdvals, eigh, eigvalsh, lu, lu_factor,
lu_solve
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

When the question is "does this tensor contain *any* NaN?" rather than "which
elements are NaN?", `Tensor.has_nan()` and `Tensor.has_inf()` answer it
directly. They return a Python `bool`, build no intermediate tensor, and stop at
the first non-finite element they find -- which is what makes them worth
reaching for on a large gradient, where `isnan(g).any()` reads all of it
regardless. On a 16M-element tensor the two are within 5% of each other when
there is nothing to find; when a NaN is present near the start, the scan stops
there.

```python
import minitensor as mt
from minitensor import nn

mt.manual_seed(0)
model = nn.DenseLayer(3, 2)
loss = nn.mse_loss(model(mt.randn(4, 3)), mt.randn(4, 2))
loss.backward()

print(loss.has_nan(), loss.has_inf())
print(any(mt.get_gradient(p).has_nan() for p in model.parameters()))
```

```text
False False
False
```

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

#### Which way round is a boolean mask?

The two mask-taking families use **opposite** polarity. Both follow PyTorch,
which is itself inconsistent here, so the convention is worth stating rather
than guessing:

| Function | `True` means |
| --- | --- |
| `scaled_dot_product_attention(attn_mask=...)` | **keep** this position |
| `masked_softmax`, `masked_log_softmax`, `masked_fill` | **exclude** this position |

Passing the wrong polarity does not raise — it silently attends to exactly the
positions you meant to hide, so this is worth checking rather than assuming.

A row that ends up with nothing to attend to has no defined softmax (`0/0`).
`masked_softmax` and `scaled_dot_product_attention` return zeros for such a row,
and `masked_log_softmax` returns `-inf`. Note that PyTorch returns `NaN` for a
fully-masked attention row; zeros propagate quietly, so a fully-masked row is
still a bug worth catching upstream.

```python
import minitensor as mt
import numpy as np

scores = mt.from_numpy(np.array([[1.0, 2.0, 3.0]], dtype=np.float32))
keep_first_only = mt.from_numpy(np.array([[False, True, True]]))

# True excludes, so this keeps position 0 alone and it takes all the weight.
print(mt.masked_softmax(scores, keep_first_only, -1).numpy())
```

```text
[[1. 0. 0.]]
```

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
| `max_pool1d(input, kernel_size, stride=None, padding=0, return_indices=False)` | 1-D max pooling; `stride` defaults to `kernel_size`. With `return_indices` the result is `(values, indices)`, each index the position along the axis its maximum came from. |
| `max_unpool1d(input, indices, kernel_size, stride=None, padding=0, output_size=None)` | Scatter a pooled signal back to the positions `max_pool1d(..., return_indices=True)` reported, zero elsewhere. |
| `avg_pool1d(input, kernel_size, stride=None, padding=0, count_include_pad=True)` | 1-D average pooling. |
| `conv3d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1)` | 3-D cross-correlation with a `(out, in / groups, kD, kH, kW)` kernel. Written as `kD` two-dimensional convolutions -- each depth tap of the kernel applied to the depth slices it reads, summed -- so the arithmetic stays with the `conv2d` kernel and the memory stays at its cost. Laying the volume out as columns instead would take twenty-seven times the volume for a 3x3x3 kernel over eight channels. |
| `max_unpool2d(input, indices, kernel_size, stride=None, padding=0, output_size=None)` | Scatter a pooled image back to where `max_pool2d` found each maximum, zero elsewhere. A partial inverse in two ways, both deliberate: it cannot restore what pooling discarded -- an unpooling decoder wants the shape and the locations back, not the values -- and it cannot know the input's size, because a seven-wide input and a six-wide one pool to the same three columns. `output_size` is how the caller says which. |
| `max_pool3d(input, kernel_size, stride=None, padding=0)` | The largest value in each 3-D window; stride defaults to the window. The maximum over a stack is the maximum of the maxima, so this is `max_pool2d` per depth tap and an elementwise maximum across them. Depth padding is negative infinity, so a padded position never wins. |
| `avg_pool3d(input, kernel_size, stride=None, padding=0, count_include_pad=True)` | The mean of each 3-D window; stride defaults to the window. Without `count_include_pad` the divisor is only the positions really there, computed by running the same pipeline over a volume of ones -- so the numerator and the divisor cannot disagree about which positions those are. |
| `batch_norm(input, running_mean=None, running_var=None, weight=None, bias=None, training=True, momentum=0.1, eps=1e-5)` | Batch normalization; updates the running buffers in place when `training=True`. |
| `group_norm(input, num_groups, weight=None, bias=None, eps=1e-5)` | Normalize over each group of channels and all of their positions. Between `layer_norm`, which takes every channel together, and `instance_norm`, which takes each alone -- `num_groups` says how finely to divide them and those two are the ends of the range. The statistics never cross the batch, so a sample's result does not depend on which others it was computed with, which is what makes it work at a batch size of one. `weight` and `bias` are per channel, not per group. |
| `instance_norm(input, running_mean=None, running_var=None, weight=None, bias=None, use_input_stats=True, momentum=0.1, eps=1e-5)` | Normalize each channel of each sample over its own positions -- `group_norm` with one group per channel, and written as that rather than twice. What it adds is the running buffers: updated from the batch when `use_input_stats`, used instead of it when not. The buffers take the *unbiased* variance while the normalization takes the biased one, as `batch_norm` does. |
| `local_response_norm(input, size, alpha=1e-4, beta=0.75, k=1.0)` | `x / (k + alpha * mean(x**2 over `size` neighbouring channels)) ** beta` -- AlexNet's normalization, where a strong response suppresses the same position in the channels beside it. An even window reaches one further below than above, as in `torch`. Built as `avg_pool3d` with the channel axis in the depth slot, so no kernel averages over channels. |
| `dropout2d(input, p)` | Channel-wise dropout. |
| `dropout1d(input, p=0.5, training=True)` | Zero whole channels of a `(batch, channels, positions)` input -- `dropout2d` for a signal. Adjacent positions in a feature map are correlated, so zeroing scattered elements leaves each recoverable from its neighbours and regularizes little; dropping the channel does not. |
| `dropout3d(input, p=0.5, training=True)` | The same over `(batch, channels, depth, height, width)`. |
| `alpha_dropout(input, p=0.5, training=True)` | Dropout that leaves a self-normalizing network self-normalizing. Ordinary dropout zeroes an element and rescales the rest, keeping the mean and moving the variance -- fine after a rectifier, wrong after `selu`, whose premise is a mean of zero and a variance of one from layer to layer. This drops to `selu`'s own saturation value and applies the affine correction that restores both moments, so a standard normal comes back standard normal at any `p`. |
| `feature_alpha_dropout(input, p=0.5, training=True)` | `alpha_dropout` over whole channels, as `dropout2d` is over `dropout`: the same correction and saturation value, one draw per channel rather than per element. |
| `rrelu(input, lower=0.125, upper=0.333..., training=True)` | A leaky rectifier whose negative slope is drawn uniformly from `[lower, upper]` per element while training, and is the midpoint in evaluation so the network sees the average of what it trained against. Bit-exact on the positive side, and its derivative at the origin is the negative side's, agreeing with `leaky_relu` and `prelu`. |
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
| `nll_loss(input, target, weight=None, ignore_index=-100, reduction="mean")` | Negative log-likelihood over *log*-probabilities -- what `log_softmax` produces. Pairing the two gives what `cross_entropy` gives; they are separate so a model that already carries its own log-probabilities does not have them recomputed. `weight` scales each class, and with `reduction="mean"` the divisor becomes the total weight rather than the count, which is what makes a weighted mean an average and not a scaled sum. `ignore_index` drops positions from both. |
| `ctc_loss(log_probs, targets, input_lengths, target_lengths, blank=0, reduction="mean", zero_infinity=False)` | Connectionist temporal classification. See below -- it takes more explaining than a table row allows. |
| `margin_ranking_loss(input1, input2, target, margin=0.0, reduction="mean")` | `max(0, -target * (input1 - input2) + margin)`. `target` is `+1` where `input1` should rank higher and `-1` where `input2` should, so the loss is zero exactly when the ranking is right by at least `margin`. |
| `hinge_embedding_loss(input, target, margin=1.0, reduction="mean")` | The distance itself where `target` is `+1`, `max(0, margin - distance)` where it is `-1`. |
| `cosine_embedding_loss(input1, input2, target, margin=0.0, reduction="mean")` | `1 - cos` for a similar pair, `max(0, cos - margin)` for a dissimilar one. |
| `triplet_margin_loss(anchor, positive, negative, margin=1.0, p=2.0, eps=1e-6, swap=False, reduction="mean")` | `max(0, d(a, p) - d(a, n) + margin)`. With `swap`, the negative distance is the smaller of `d(a, n)` and `d(p, n)`, so a triplet whose positive sits closest to the negative still counts as a violation. `eps` is added inside the norm, which is what keeps the gradient of two identical points from being NaN. |
| `soft_margin_loss(input, target, reduction="mean")` | `log(1 + exp(-target * input))`, the smooth hinge, for a `target` of `+1` or `-1`. |
| `poisson_nll_loss(input, target, log_input=True, full=False, eps=1e-8, reduction="mean")` | Negative log-likelihood of a Poisson observation. `log_input` says whether `input` is the log of the rate (the numerically kind form) or the rate itself; `full` adds the Stirling term, which changes no gradient since it depends only on `target`. |
| `grid_sample(input, grid, mode="bilinear", padding_mode="zeros", align_corners=False)` | Read `input` at the coordinates in `grid`, differentiably in both. See below. |
| `prelu(input, weight)` | `max(x, 0) + weight * min(x, 0)`: a leaky rectifier whose slope is learned. `weight` is one value shared by every channel or one per channel, lined up with dimension 1. The gradient reaches it, which is the entire point of the op. |
| `gumbel_softmax(logits, tau=1.0, hard=False, dim=-1, eps=1e-20)` | A differentiable sample from a categorical distribution: Gumbel noise added to the logits, then a softmax at temperature `tau`. As `tau` falls the result approaches a one-hot draw and stays differentiable at every `tau`, which sampling itself is not. With `hard=True` the value is one-hot and the gradient is still the soft one -- the straight-through estimator. |
| `pixel_shuffle(input, upscale_factor)` | Trade `r**2` channels for that much height and width: `(n, c * r * r, h, w)` becomes `(n, c, h * r, w * r)`. The last layer of a super-resolution network -- upsampling by rearrangement costs nothing and invents nothing, where a transposed convolution does both. |
| `pixel_unshuffle(input, downscale_factor)` | The inverse: `(n, c, h * r, w * r)` back to `(n, c * r * r, h, w)`. |
| `embedding(input, weight, padding_idx=None)` | The rows of `weight` that `input` names, one per index, keeping the index's shape and adding the feature axis. `nn.Embedding` with the table owned by the caller -- a frozen one, or one shared between models. `padding_idx` names a row that takes no gradient; the value it holds is returned unchanged, as in torch. |
| `embedding_bag(input, weight, offsets=None, mode="mean", per_sample_weights=None, include_last_offset=False, padding_idx=None)` | One vector per bag of indices: `embedding` followed by a reduction, with the `(total, dim)` intermediate never named -- which is the whole reason the fused operation exists elsewhere. A two-dimensional `input` is one bag per row; a one-dimensional one needs `offsets` saying where each bag starts, which is what allows the bags to differ, and with `include_last_offset` the final entry is the end rather than a start. `mode` is `"sum"`, `"mean"` or `"max"`, and an empty bag reduces to zero in all three. `per_sample_weights` scales each row before it is summed and is meaningful only for `"sum"`. |
| `channel_shuffle(input, groups)` | Read `(n, g * c, ...)` as `(n, g, c, ...)`, swap the two, flatten back. A grouped convolution never mixes its groups, so stacking two leaves two networks side by side; one shuffle between them makes it one, at the cost of a permutation and no parameters. |
| `lp_pool1d(input, norm_type, kernel_size, stride=None)` | The `p`-norm of each window rather than its mean or its largest -- a norm type of 1 is the sum of magnitudes and a large one approaches the maximum, so this is the family `avg_pool` and `max_pool` are the ends of, with a gradient reaching every element. `abs` is taken before the power, so an odd norm type is a real norm here where `torch` would take the root of a negative number. |
| `lp_pool2d(input, norm_type, kernel_size, stride=None)` | The same over 2-D windows. |
| `affine_grid(theta, size, align_corners=False)` | The sampling grid an affine transform describes, for `grid_sample`. `theta` is `(n, 2, 3)` over an `(n, c, h, w)` output or `(n, 3, 4)` over an `(n, c, d, h, w)` one. Feeding the result to `grid_sample` is a spatial transformer, and the gradient reaches `theta`, which is what lets the transform be learned. `align_corners` must match what `grid_sample` is then given. |
| `unfold(input, kernel_size, dilation=1, padding=0, stride=1)` | Every sliding block of `input`, one per column: `(n, c, *spatial)` becomes `(n, c * taps, blocks)`. im2col -- what turns a convolution into a single matrix product, so a convolution variant the library does not ship is two lines rather than a kernel. Any number of spatial axes, not only the two `torch.nn.functional.unfold` takes, so a 3-D convolution is the same product with a rank-three kernel. |
| `fold(input, output_size, kernel_size, dilation=1, padding=0, stride=1)` | Sum the sliding blocks back into one `output_size` plane -- the adjoint of `unfold`, and bit-identical to its gradient, because the backward of a gather is a scatter-add over the positions it read. Overlapping positions are summed, not averaged; fold a tensor of ones and divide to average. |

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

`unfold` and `fold` are the pair that lets a caller write a convolution the
library does not ship. Every sliding window becomes a column, so the
convolution is one matrix product, and `fold` is the same map run backwards:

```python
import minitensor as mt
from minitensor import functional as F

image = mt.Tensor([[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]]])
weight = mt.Tensor([[[[1.0, 2.0], [3.0, 4.0]]]])

# One column per window, laid out channel-major then kernel-major.
columns = F.unfold(image, 2)
print(columns.tolist())

built = F.matmul(weight.reshape(1, -1), columns).reshape(1, 1, 2, 2)
print(built.tolist())
print(F.conv2d(image, weight).tolist())

# Overlapping blocks add rather than overwrite, so folding ones counts how many
# blocks read each position -- the divisor that turns a fold into an average.
print(F.fold(mt.Tensor.ones_like(columns), (3, 3), 2).tolist())
```

```text
[[[1.0, 2.0, 4.0, 5.0], [2.0, 3.0, 5.0, 6.0], [4.0, 5.0, 7.0, 8.0], [5.0, 6.0, 8.0, 9.0]]]
[[[[37.0, 47.0], [67.0, 77.0]]]]
[[[[37.0, 47.0], [67.0, 77.0]]]]
[[[[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]]]]
```

### Sampling at coordinates

`grid_sample` reads its input at the normalised coordinates in a grid, and
differentiates with respect to *those coordinates* as well as the input. That
is the whole point: it is what makes a spatial transformer, an optical-flow
warp or a deformable convolution trainable, because the network can learn where
to look rather than only what to do with what it found. `interpolate` resamples
onto a regular grid it works out for you; this takes the grid as an argument.

- `input` is `(batch, channels, height, width)` or
  `(batch, channels, depth, height, width)`. `grid` matches its rank and holds
  one coordinate per spatial axis in its last position; the output takes its
  shape from the grid, `(batch, channels, ...grid spatial)`.
- The grid's last axis is in `x, y` order -- and `x, y, z` for a volume -- which
  is the **reverse** of the `H, W` (or `D, H, W`) it indexes. Every framework
  does this and reading it the other way transposes the output silently.
- Coordinates run from `-1` to `1` across the input, whatever its size, so one
  grid works against several resolutions. `align_corners` decides whether those
  two values name the centres of the corner samples or their outer edges.
  Neither is more correct; they differ by half a pixel, and a model trained
  under one reads the wrong place under the other.
- `padding_mode` says what lies outside: `"zeros"` reads nothing there,
  `"border"` holds the edge value, `"reflection"` folds back inside. The last
  two move the coordinate before any neighbour is chosen; `"zeros"` instead
  drops the individual neighbours that fall outside, which is why a coordinate
  half a pixel past the edge still reads half of the edge sample and half of
  nothing.
- `mode="nearest"` takes the single closest sample. Its gradient in the
  coordinate is exactly zero -- rounding is flat between samples -- so a model
  that has to learn *where* to look needs `"bilinear"`.

Folding and clamping are not smooth, and the coordinate gradient says so: one
held against an edge by `"border"` moves the output not at all, and one folded
back by `"reflection"` moves it the other way. Both are invisible in the
forward pass, which is why they are worth stating.

Bicubic sampling is not here, for the same reason `interpolate` stops at
linear: a third resampling rule belongs in both or in neither.

```python
import minitensor as mt
from minitensor import nn

image = mt.Tensor([[[[0.0, 1.0, 2.0, 3.0]]]], dtype="float64")

# x = -1 is sample 0's centre when the corners are aligned; -0.5 is three
# quarters of the way from sample 0 to sample 1.
grid = mt.Tensor([[[[-0.5, 0.0]]]], dtype="float64", requires_grad=True)
read = nn.grid_sample(image, grid, align_corners=True)
print(round(float(read.numpy()[0, 0, 0, 0]), 4))

# The gradient in the coordinate is the local slope of the image, carried
# through the normalisation: one unit per sample, times 3/2 samples per unit
# of coordinate.
read.sum().backward()
print(round(float(grid.grad.numpy()[0, 0, 0, 0]), 4))
```

```text
0.75
1.5
```

### Connectionist temporal classification

`ctc_loss` is for a model whose output is longer than its target and unaligned
with it: speech against a transcript, handwriting against characters. Nothing
says which input step produced which symbol, so the loss is the total
probability of *every* alignment that collapses to the target, where collapsing
merges adjacent equal classes and then deletes the blank. That is exponentially
many paths, summed by a dynamic program over the time axis -- which is why this
is the one loss here that is not an expression over tensors.

- `log_probs` is `(steps, batch, classes)` and is expected to hold log
  probabilities already, the output of a `log_softmax` over the class axis.
  Nothing normalises it here: a caller who has a numerically careful
  log-softmax should not have it undone and redone.
- `targets` is either a padded `(batch, length)` block, read up to each row's
  own target length, or the rows concatenated into a vector -- the second is
  what a caller with wildly uneven targets wants. It may not contain `blank`,
  which stands for emitting nothing and so has no reading inside a target.
- `input_lengths` and `target_lengths` are integer vectors -- `int32` or
  `int64`, one entry per batch element. Steps beyond a sample's input length take no part in its loss and
  receive no gradient.
- `reduction="mean"` divides each loss by its *own* target length before
  averaging, which is what makes the number comparable across batches with
  different targets -- so it is not the mean of what `reduction="none"` returns.
- `zero_infinity=True` replaces the infinite loss of a target too long to fit
  its input, and its gradient, with zero. That is a data problem rather than a
  modelling one, and left alone a single such sample takes the whole batch's
  gradient with it.

Everything runs in the log domain and in `float64` regardless of the input's
dtype. A path probability is a product of `steps` numbers below one, so at the
few thousand steps of a real utterance it underflows `float64` many times over;
an implementation that multiplied probabilities would report that every path
had probability zero.

```python
import minitensor as mt
from minitensor import nn

# Six input steps for a three-symbol target, over an alphabet of four
# symbols plus the blank at index 0.
scores = mt.Tensor(
    [[[0.1, 0.9, 0.2, 0.1, 0.0]]] * 6, dtype="float64", requires_grad=True
)
log_probs = mt.log_softmax(scores, -1)
targets = mt.Tensor([[1, 2, 1]], dtype="int64")
inputs = mt.Tensor([6], dtype="int64")
lengths = mt.Tensor([3], dtype="int64")

each = nn.ctc_loss(log_probs, targets, inputs, lengths, reduction="none")
print(round(float(each.numpy()[0]), 4))

# "mean" divides by the target length, so multiplying it back by the three
# symbols recovers the same number for this one-element batch.
averaged = nn.ctc_loss(log_probs, targets, inputs, lengths, reduction="mean")
print(round(float(averaged.numpy()) * 3, 4))

# The gradient with respect to the log probabilities sums to -1 at every
# step. Carried back through the softmax that produced them it sums to
# zero instead, since shifting every score by the same amount changes
# nothing.
each.sum().backward()
gradient = scores.grad.numpy()
print(max(abs(float(gradient[t].sum())) for t in range(6)) < 1e-12)
```

```text
3.8573
3.8573
True
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
kernel_size, stride=None, padding=None, return_indices=False)` and
`functional.avg_pool2d(input, kernel_size, stride=None, padding=None,
count_include_pad=True)`.

With `return_indices` the max pooling returns `(values, indices)` rather than
one tensor, each index a flat offset into the unpadded input plane. The kernel
finds that position anyway -- the backward pass has to send the gradient to the
element that won -- so asking for it costs one copy of a vector that already
exists, and only when asked. It is what `max_unpool2d` scatters back into, and
because the padding may not exceed half the window, every index names a real
element.

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
`embed_dim` must be divisible by `num_heads`. That is the other way round from
`LSTM` and `GRU`, which are `(seq, batch, input_size)` unless constructed with
`batch_first`. Feeding this one a sequence-first tensor is not an error -- the
shapes fit, the output comes back the right size, and it is attention over the
wrong axis -- so a stack that mixes the two needs a transpose between them. Calling the layer performs
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
- `GELU(approximate="tanh")` — the tanh approximation by default, which is what
  the layer is for: half again as quick as the error function. The free `gelu`
  defaults the other way, and the two differ by about `5e-4`, so a model that
  wants those values builds its layers with `approximate="none"`.
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

### Weight initialization (`minitensor.nn.init`)

The built-in layers initialize their own weights. These are for parameters you
create yourself -- a `plugins.CustomLayer`, or a tensor you drive through the
functional API.

They are **factories**, not PyTorch's in-place `xavier_uniform_(tensor)`: each
takes a shape and returns a new tensor, which is also how every layer in `nn`
builds its parameters. To re-initialize an existing parameter, build a new
tensor and assign it.

Six of the fan-based schemes are also available as top-level tensor
constructors -- `mt.xavier_uniform(shape)`, `mt.he_normal(shape)` and so on,
each with a `_like` variant that takes a reference tensor instead of a shape.
Those are the same schemes drawing from the same code, with **one deliberate
difference**: they default to `requires_grad=False`, because they sit beside
`mt.zeros` and `mt.randn` as ways to make a tensor. The `nn.init` spelling
defaults to `requires_grad=True`, because it exists to make a *parameter*, and
a parameter created without it does not train and says nothing. Both accept the
argument explicitly, so pass it when it matters.

| Function | Distribution |
| --- | --- |
| `zeros(shape)` / `ones(shape)` / `constant(shape, value)` | Fixed value. |
| `uniform(shape, a=0.0, b=1.0)` | Uniform over `[a, b)`. |
| `normal(shape, mean=0.0, std=1.0)` | Normal. |
| `truncated_normal(shape, mean=0.0, std=1.0, lower=None, upper=None)` | Normal confined to `[lower, upper]`, defaulting to two deviations either side. |
| `xavier_uniform(shape)` / `xavier_normal(shape)` | Glorot & Bengio (2010); scale from `fan_in + fan_out`. |
| `he_uniform(shape)` / `he_normal(shape)` | He et al. (2015); scale from `fan_in`. For ReLU networks. |
| `lecun_uniform(shape)` / `lecun_normal(shape)` | LeCun et al. (1998); scale from `fan_in`. |
| `calculate_fan_in_and_fan_out(shape)` | The `(fan_in, fan_out)` the schemes above derive their scale from. |

`kaiming_uniform`/`kaiming_normal` and `glorot_uniform`/`glorot_normal` are the
same functions under the other spelling of each name.

All of them take `dtype="float32"`, `device=None` and `requires_grad=True`.
Everything but `constant` (and its `zeros`/`ones` shorthands) draws from a
continuous distribution and so requires a float dtype.

A weight is stored `[out_features, in_features]`, so `fan_in` is the **trailing**
dimension; a convolution weight `[out_channels, in_channels, kh, kw]` scales
both fans by the receptive field. Reading those the wrong way round is the
usual way a hand-rolled initializer goes wrong, which is what
`calculate_fan_in_and_fan_out` is exposed for:

```python
from minitensor.nn import init

print(init.calculate_fan_in_and_fan_out([512, 256]))
print(init.calculate_fan_in_and_fan_out([32, 3, 5, 5]))

weight = init.he_normal([8, 4])
print(tuple(weight.shape), weight.dtype, weight.requires_grad)
```

```text
(256, 512)
(75, 800)
(8, 4) float32 True
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
- `Adadelta(params, lr=1.0, rho=0.9, eps=1e-6, weight_decay=0.0)`
- `Adamax(params, lr=0.002, betas=None, beta1=None, beta2=None, eps=1e-8, weight_decay=0.0)`
- `RAdam(params, lr=0.001, betas=None, beta1=None, beta2=None, eps=1e-8, weight_decay=0.0)`
- `Rprop(params, lr=0.01, etas=(0.5, 1.2), step_sizes=(1e-6, 50.0))`

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

`Adadelta` (Zeiler, 2012) is the answer to a dimensional complaint about the
methods above it: a gradient divided by a running magnitude of gradients is a
pure number, so their step is measured in nothing at all and the learning rate
has to carry the parameter's units. Adadelta multiplies by a running magnitude
of its own past *steps* as well, so the answer comes out in the parameter's
units and `lr` defaults to 1 — a multiplier rather than the scale that decides
whether the run converges. Gradients six orders of magnitude apart give steps
within a factor of four of each other. `rho` decays both running averages and
`eps` sits under both square roots, so it is also what seeds the very first
step.

`Adamax` (Kingma & Ba, 2015) is Adam with the second moment measured by a
decaying infinity norm — `u = max(beta2 * u, |g|)` — rather than a mean of
squares. One enormous gradient sets the denominator and then leaves it at
exactly `beta2` per step, where squaring it into an average takes far longer to
forget. There is also no second bias correction: a maximum of a decaying
sequence is not shrunk towards zero by starting at zero the way a mean is.

`RAdam` (Liu et al., 2020) scales Adam's early steps by the variance its
second-moment estimate actually has. In the first few steps that estimate is
built from almost no samples, so its variance is enormous and the steps are
wild — which is what a linear warmup schedule exists to paper over. RAdam
computes the estimate's effective sample count and applies the correction that
variance implies, so the warmup falls out of the method rather than being tuned
into it. Below five effective samples there is no usable estimate at all and it
takes a plain, non-adaptive step.

`Rprop` (Riedmiller & Braun, 1993) reads only the *sign* of the gradient. It
keeps a step size per parameter and moves by exactly that: a step whose
direction agrees with the last grows by `etas[1]`, one that reverses shrinks by
`etas[0]` and is not taken at all, because reversing means the last step went
past a minimum. Gradients twelve orders of magnitude apart but agreeing in sign
take an identical path. That makes it immune to badly scaled gradients and
useless on mini-batches, where a noisy sign flips for reasons that have nothing
to do with the surface — it is a full-batch method.

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

- `step()` -- apply parameter updates and consume the gradients it applied.
  Gradients belonging to parameters this optimizer does not hold are left
  alone, so several optimizers can step off one backward pass.
- `zero_grad(set_to_none: bool = False)` -- reset gradients.
- `lr` property -- read/write learning rate.
- `step_count` property -- how many steps have been applied.
- `state_dict()` / `load_state_dict(state)` -- snapshot and restore the
  optimizer's own state.
- `save(path)` / `load(path)` -- the same, through a file.

### Checkpointing a training run

Saving the model saves the weights. It does not save the optimizer, and for
every optimizer here except plain SGD that is only half the run: momentum
buffers, squared-gradient averages and the step count all live in the
optimizer. Reconstructing one from scratch restarts Adam's bias correction
from `t = 0`, so the first step after the resume is an outsized one -- on a
small regression it moved the parameters 2.05x as far as the step it was
supposed to be continuing.

Save both, and the resumed run is bit-identical to the uninterrupted one:

```python
import os
import tempfile

import minitensor as mt
from minitensor import nn, optim


def build():
    return nn.Sequential([nn.DenseLayer(4, 8), nn.ReLU(), nn.DenseLayer(8, 2)])


model = build()
optimizer = optim.Adam(model.parameters(), lr=0.01)

inputs, targets = mt.randn(16, 4), mt.randn(16, 2)
for _ in range(10):
    optimizer.zero_grad()
    nn.mse_loss(model(inputs), targets).backward()
    optimizer.step()

with tempfile.TemporaryDirectory() as folder:
    weights = os.path.join(folder, "model.bin")
    state = os.path.join(folder, "optimizer.bin")
    model.save(weights)
    optimizer.save(state)

    # ... later, in a new process ...
    restored = build()
    restored.load_state_dict(type(restored).load_state_from(weights))
    resumed = optim.Adam(restored.parameters(), lr=0.01)
    resumed.load(state)

print(resumed.step_count)
```

```text
10
```

If the run uses a learning-rate scheduler, that has a position too. Every
schedule here is a pure function of `(last_epoch, base_lr)`, so its whole state
is those two numbers, and `scheduler.state_dict()` returns them as a plain
dict -- JSON-serialisable, to go wherever the rest of the checkpoint goes.
`load_state_dict` writes the restored rate to the optimizer immediately, so
the step right after a resume runs at the right rate rather than the next one:

```python
import minitensor as mt
from minitensor import nn, optim

model = nn.DenseLayer(4, 2)
optimizer = optim.Adam(model.parameters(), lr=1.0)
scheduler = optim.StepLR(optimizer, step_size=2, gamma=0.5)
for _ in range(5):
    scheduler.step()

saved = scheduler.state_dict()
print(saved, round(optimizer.lr, 6))

resumed_optimizer = optim.Adam(model.parameters(), lr=1.0)
resumed = optim.StepLR(resumed_optimizer, step_size=2, gamma=0.5)
print(round(resumed_optimizer.lr, 6))      # a fresh schedule starts over
resumed.load_state_dict(saved)
print(round(resumed_optimizer.lr, 6))      # ... and now it does not
```

```text
{'base_lr': 1.0, 'last_epoch': 5} 0.25
1.0
0.25
```

Per-parameter state is matched by **position**, so the optimizer has to be
constructed over the same parameters in the same order as when it was saved.
Loading a state saved by a different algorithm, for a different number of
parameters, or for differently shaped ones is refused rather than silently
partially applied -- including Adam into AdamW, which share a buffer layout
but not an update rule.

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
- `StateDict` -- tensor parameters/buffers, readable and writable by name.
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

`load_state_dict` requires every one of those names to be present and shaped
like the slot it lands in, and raises naming all the problems at once if not:

```text
load_state_dict: missing from the state dict: 0.bias; wrong shape: 1.weight
(expected [5], got [3])
```

Nothing is written unless every entry checks out, so a load that raises leaves
the module exactly as it was — a caller that catches the error and falls back
gets the model it had, not one holding half a checkpoint. There is no partial
or non-strict mode; to load a subset deliberately, assign the tensors you want
through the parameter itself.

A custom `Layer` that does not override `named_parameters` falls back to
positional keys (`param_0`, `param_1`, ...), which still load correctly but
cannot be inspected or reordered.

### Reading and building a `StateDict`

A state dict behaves as a mapping from name to tensor:

- `state[name]` -- the tensor under `name`, checking parameters then buffers.
- `keys()`, `values()`, `items()`, `len()`, `in`, and iteration -- the usual
  mapping views, so `dict(state)`, `{**state}` and
  `for name, tensor in state.items()` all work. Every one of them spans
  parameters and buffers together.
- `get_parameter(name)` / `get_buffer(name)` -- one namespace at a time. The
  two are distinct in the file format, so a name may appear in both.
- `parameter_names()` / `buffer_names()` -- the names on one side only.
- `parameters()` / `buffers()` -- every entry on one side, as a plain dict.
- `add_parameter(name, tensor)` / `add_buffer(name, tensor)` -- record one,
  replacing any already under that name.

Together those let a checkpoint be inspected rather than only replayed --
reading a saved weight, copying one model's weights into another, or assembling
a state dict from tensors you already hold.

Names come back sorted, parameters before buffers, and that order does not
change between runs. It is also the order they are written to a checkpoint in,
which makes a saved file a function of the weights alone: saving one model
twice gives byte-identical output apart from the `created_at` timestamp, so
checkpoints can be compared by digest and diffed without spurious reordering.

```python
import minitensor as mt
from minitensor import nn

mt.manual_seed(0)
model = nn.Sequential([nn.DenseLayer(4, 3), nn.BatchNorm1d(3)])
state = model.state_dict()

print(state["0.weight"].shape)
print(sorted(state.buffers()))

built = mt.serialization.StateDict()
built.add_parameter("weight", mt.zeros(2, 2))
print(len(built), "weight" in built)
```

```text
Shape([3, 4])
['1.running_mean', '1.running_var']
1 True
```

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

#### Writing a custom layer that trains

`set_forward` takes a callable receiving a **list** of tensors, not a single
tensor — that is what makes multi-input layers expressible. It may return a
list or a bare tensor. Calling `forward` before `set_forward` raises
`NotImplementedError`.

The layer's forward runs as ordinary Python, so anything built from tensor ops
records onto the autograd graph and its parameters train like any other:

```python
import minitensor as mt
from minitensor.plugins import CustomLayer

scale = CustomLayer("scale")
gain = mt.ones(3, requires_grad=True)
scale.add_parameter("gain", gain)
scale.set_forward(lambda inputs: [inputs[0] * scale.get_parameter("gain")])

x = mt.Tensor([[1.0, 2.0, 3.0]])
optimizer = mt.optim.SGD([gain], lr=0.05)

start = mt.sum(scale.forward([x])[0] ** 2).item()
for _ in range(60):
    optimizer.zero_grad()
    loss = mt.sum(scale.forward([x])[0] ** 2)
    loss.backward()
    optimizer.step()

print(gain.grad is not None)
print(bool(loss.item() < start / 1000))
```

```text
True
True
```

A layer taking two inputs and returning two outputs is written the same way:

```python
import minitensor as mt
from minitensor.plugins import CustomLayer

pair = CustomLayer("pair")
pair.set_forward(lambda inputs: [inputs[0] + inputs[1], inputs[0] * inputs[1]])

total, product = pair.forward([mt.Tensor([1.0, 2.0]), mt.Tensor([3.0, 4.0])])
print(total.tolist(), product.tolist())
```

```text
[4.0, 6.0] [3.0, 8.0]
```

#### What does not compose

Three limits are worth knowing before designing around containers:

- `CustomLayer` is **not** an `nn.Module`, so it cannot be placed inside
  `nn.Sequential`. Chain it in a plain Python function instead, as above —
  gradients flow across the boundary either way, and an optimizer just needs
  the parameters passed to it explicitly.
- `nn.Sequential` cannot contain another `nn.Sequential`; both the constructor
  and `add_module` reject it. Build one flat container, or compose in Python.
- `nn.Module` cannot be subclassed from Python. Custom behaviour goes through
  `CustomLayer` or a compiled plugin.

Composing a custom layer with built-in ones in a Python function works, and one
optimizer trains both sides:

```python
import minitensor as mt
from minitensor import nn
from minitensor.plugins import CustomLayer

mt.manual_seed(0)
dense = nn.DenseLayer(3, 3, dtype="float64")
scale = CustomLayer("scale")
gain = mt.Tensor([2.0, 2.0, 2.0], dtype="float64", requires_grad=True)
scale.add_parameter("gain", gain)
scale.set_forward(lambda inputs: [inputs[0] * scale.get_parameter("gain")])

x = mt.Tensor([[1.0, 1.0, 1.0]], dtype="float64")
optimizer = mt.optim.SGD(list(dense.parameters()) + [gain], lr=0.01)

for _ in range(40):
    optimizer.zero_grad()
    loss = mt.sum(scale.forward([dense(x)])[0] ** 2)
    loss.backward()
    optimizer.step()

print(gain.grad is not None)
print(bool(loss.item() < 1e-6))
```

```text
True
True
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
tensor -- but only for one pass. Each `backward()` releases the interior
gradients the previous one left, so an ordinary training loop is bounded, and
so is a loop that accumulates over several backward passes before stepping:

```python
import minitensor as mt

w = mt.ones((4, 4), requires_grad=True)
optimizer = mt.optim.SGD([w], 1e-3)

for _ in range(3):
    optimizer.zero_grad()
    h = mt.matmul(mt.ones((4, 4)), w)
    mt.sum(h).backward()
    optimizer.step()

# The last pass's interior gradient is still readable; earlier ones are not.
print(mt.get_gradient(h) is not None)
```

```text
True
```

A loop that backpropagates *without* stepping an optimizer — gradient
inspection, a custom optimizer written in Python, accumulating many
micro-batches before one step — does not need to clear anything either. Each
`backward()` releases the interior gradients the previous one left behind, so
what the loop holds is one pass's worth however long it runs. Fifty backward
passes over a five-layer model sit at the same handful of entries as the first
one.

```python
import minitensor as mt

w = mt.ones((4, 4), requires_grad=True)

sizes = []
for _ in range(5):
    h = mt.matmul(mt.ones((4, 4)), w)
    mt.sum(h).backward()
    _ = w.grad                   # inspect it, no optimizer involved
    sizes.append(mt.autograd_graph_size())

print(len(set(sizes)) == 1)
```

```text
True
```

**Do not reach for `clear_autograd_graph()` inside such a loop.** It releases
*every* stored gradient, leaves included, so calling it between the backward
passes of a gradient accumulation discards exactly the running total the
accumulation exists to build — `.grad` comes back `None` and the step that
follows does nothing. It is for tearing down between unrelated pieces of work,
not for use within one.

Note `zero_grad()` is not the same thing either: it clears the gradients of the
parameters it was given, and leaves interior entries alone.

### Run inference inside `no_grad()`

A loop that only calls *forward* grows too, and for a different reason: every
forward records its intermediates, and nothing releases them until a
`backward()` walks the graph or `clear_autograd_graph()` empties it. Neither
condition happens in an inference loop, so it grows without bound.

Two things that look like they should help do not. **Discarding the output does
not release the graph** — recording is held in a graph the module owns, not by
the output tensor, so a loop that keeps nothing still accumulates. And
`model.eval()` does not either: it switches Dropout off and freezes BatchNorm's
running statistics, which is about what the layers compute, not about whether
the computation is recorded.

Resident memory over 300 forwards of a two-layer `Sequential` with a 256-row
batch, discarding every output, each row measured in its own process (a second
measurement in the same process reads near zero — the allocator has already
grown the heap and does not need to ask for more):

| Width | Per forward | Over 300 forwards |
| --- | --- | --- |
| 32 | ~42 KB | ~12 MB |
| 128 | ~162 KB | ~48 MB |
| 512 | ~642 KB | ~188 MB |

`no_grad()` removes it completely — no graph entries, no growth — and an
inference loop has no use for the recording anyway:

```python
import minitensor as mt
from minitensor import nn

mt.manual_seed(0)
model = nn.Sequential([nn.DenseLayer(32, 32), nn.ReLU()])
x = mt.ones((8, 32))

mt.clear_autograd_graph()
for _ in range(10):
    model(x)                       # the output is discarded every time
print(mt.autograd_graph_size()[0])

model.eval()                       # about layer behaviour, not recording
mt.clear_autograd_graph()
for _ in range(10):
    model(x)
print(mt.autograd_graph_size()[0])

mt.clear_autograd_graph()
with mt.no_grad():
    for _ in range(10):
        model(x)
print(mt.autograd_graph_size()[0])
```

```text
33
33
0
```

A training loop needs neither guard: `backward()` releases the subgraph it
walked, so forward-then-backward stays flat on its own.

### The graph is thread-local, and a missing graph is silent

Each thread records into its own autograd graph, so concurrent threads do not
interfere: one thread's `clear_autograd_graph()` cannot disturb a graph another
thread is still building, and independent training loops in separate threads
produce correct, independent gradients. `is_grad_enabled()` and the
graph-consumed flag are thread-local for the same reason.

The consequence is that **a graph has to be backpropagated on the thread that
built it**. Building a loss in a worker thread and calling `backward()` on it
from the main thread does not raise — it quietly does nothing, leaving `.grad`
as `None`:

```python
import threading

import minitensor as mt

state = {}


def build():
    w = mt.ones(3, requires_grad=True)
    state["w"] = w
    state["loss"] = mt.sum(w * 5.0)


worker = threading.Thread(target=build)
worker.start()
worker.join()

state["loss"].backward()          # different thread: no error, no gradient
print(state["w"].grad is None)
print(state["loss"].requires_grad)
```

```text
True
True
```

Note the tensor still reports `requires_grad=True`, so nothing about it
signals the problem. The same silence applies within a single thread after
`clear_autograd_graph()` — `backward()` on a released graph is a no-op rather
than an error. If gradients come back `None` when you expect values, check
which thread built the graph and whether it was cleared in between.

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

### Asking for a tensor that will not fit

A shape whose dimensions multiply past what `usize` can address is refused with
a `ValueError` before anything is allocated — `mt.zeros(2**32, 2**32)`,
`x.expand([2**32, 2**32])` and the rest of the shape arguments all report
`has more elements than this platform can represent`.

A shape that *is* addressable but larger than available memory is a different
matter, and worth knowing about because it does not behave like NumPy:
**the allocation failure aborts the process.** NumPy raises `MemoryError` and
PyTorch raises a `RuntimeError`, both catchable; here the Rust allocator's
failure path terminates the interpreter, so no `try`/`except` can intervene.

```text
np.zeros(12 * 10**9, dtype=np.float32)   # MemoryError, caller continues
mt.zeros(12 * 10**9)                     # process aborts
```

The same applies to any operation whose *output* is too large — `repeat`,
`contiguous` on a wide `expand`, `arange` — not only to explicit construction.
Size the allocation before requesting it if a graceful failure matters; the
tensor constructors return their storage directly rather than a `Result`, so
propagating an allocation error would have to change every one of them.

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
| `blas` | no | Routes GEMM through a system OpenBLAS (`libopenblas-dev` or equivalent). `openblas-src` is pinned to `system`, so the build links the installed library rather than downloading and compiling OpenBLAS itself. Worth roughly 1.4x-2.0x on square `f32` GEMM — see below. |
| `dynamic-loading` | no | Runtime plugin loading (`docs/plugin_system.md`). |

### Building the Python extension with BLAS

The feature table above describes the engine crate, but the same names work
when building the Python extension — they are forwarded by the bindings:

```text
sudo apt-get install libopenblas-dev        # or your platform's equivalent
maturin develop --release --features blas
```

Without it, GEMM uses `matrixmultiply`, which is pure Rust and needs no system
library. With it, square `f32` matmul is roughly at parity with NumPy, which
links OpenBLAS itself; without it MiniTensor is 1.5x-2.4x behind. Measured on
x86-64 with OpenBLAS 0.3.26, each timing taken in its own process so the two
thread pools do not contend:

| size | default | `--features blas` | NumPy |
| --- | --- | --- | --- |
| 256 | 0.41 ms | 0.21 ms | 0.11 ms |
| 512 | 1.76 ms | 0.94 ms | 0.67 ms |
| 1024 | 9.22 ms | 5.18 ms | 4.82 ms |
| 2048 | 55.2 ms | 37.1 ms | 36.6 ms |

The gap that remains at 256 is fixed per-call overhead, not GEMM throughput; it
stops mattering by 1024. Everything else in the library is unaffected, so this
is worth enabling only if matmul dominates your workload and you are willing to
carry the system dependency.

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
