# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""Indexing, scattering and index-producing helpers, in terms of the kernels.

Almost every function here rearranges what `index_select`, `gather`, `scatter`,
`nonzero`, `searchsorted` and `pad` already do. None of those needs a kernel of
its own: `take` is `index_select` over a flattened view, `index_add` is
`scatter_add` with the index broadcast, and `isin` is a sorted lookup. Writing
them here rather than in the extension keeps the shipped binary the size of the
operations that genuinely need one, and each inherits the bounds checking, the
dtype rules and the gradients of the kernel underneath instead of restating
them.

The `*_indices` builders are the exception, and they are the exception for the
reason set out under "Where an operation belongs" in `docs/development.md`:
their arguments are Python integers rather than tensors, so there is no device
to stay on and no gradient to carry, and NumPy already computes the answer.
"""

from __future__ import annotations

import operator as _operator

import numpy as _np

from . import _core as _C
from ._shape import (
    _atleast_tensor,
    _element_count,
    _index_tensor,
    _normalize_axis,
    _normalize_shape_argument,
    broadcast_to,
)
from ._shape import meshgrid as _meshgrid

Tensor = _C.Tensor
_F = _C.functional


def _as_index(value: object, name: str) -> Tensor:
    """`value` as an int64 index tensor."""

    tensor = _atleast_tensor(value)
    dtype = str(tensor.dtype)
    if "int" not in dtype or "bool" in dtype:
        raise TypeError(f"{name} requires integer indices, got {tensor.dtype}")
    return tensor if "int64" in dtype else tensor.astype("int64")


def _wrap_negative(indices: Tensor, length: int) -> Tensor:
    """Bring negative positions round to the far end, as Python's own
    indexing does. The kernels take non-negative positions only, so this is
    where `-1` becomes `length - 1`."""

    if length == 0:
        return indices
    return _F.where(indices < 0, indices + length, indices)


def _promoted_dtype(left: Tensor, right: Tensor) -> str:
    """The dtype the library's own promotion gives these two.

    Asked by doing the promotion on nothing: an empty add touches no elements
    and answers exactly what a full one would, which beats restating the
    promotion table here where it could drift from the real one.
    """

    empty_left = _F.narrow(left.reshape(-1), 0, 0, 0)
    empty_right = _F.narrow(right.reshape(-1), 0, 0, 0)
    return str((empty_left + empty_right).dtype)


def take(input: object, index: object) -> Tensor:
    """The elements at flat positions `index`, shaped like `index`.

    The tensor is read in row-major order regardless of its shape, so this is
    `index_select` over a flattened view. Negative positions count from the
    end.
    """

    tensor = _atleast_tensor(input)
    indices = _as_index(index, "take")
    flat = tensor.reshape(-1)
    positions = _wrap_negative(indices.reshape(-1), flat.shape[0])
    return _F.index_select(flat, 0, positions).reshape(list(indices.shape))


def take_along_dim(input: object, indices: object, dim: int | None = None) -> Tensor:
    """One element per position, its `dim` coordinate coming from `indices`.

    With `dim` omitted both operands are flattened first, which is what makes
    `take_along_dim(x, x.argsort(...))` reorder a whole tensor. The index is
    broadcast against the input's other axes, so a `(n, 1)` index selects one
    column per row.
    """

    tensor = _atleast_tensor(input)
    index = _as_index(indices, "take_along_dim")

    if dim is None:
        flat = tensor.reshape(-1)
        return _F.index_select(
            flat, 0, _wrap_negative(index.reshape(-1), flat.shape[0])
        )

    axis = _normalize_axis(dim, tensor.ndim(), "take_along_dim")
    target = list(tensor.shape)
    target[axis] = index.shape[axis] if index.ndim() == tensor.ndim() else 1
    if index.ndim() != tensor.ndim():
        raise ValueError(
            f"take_along_dim needs an index of the same rank as the input, "
            f"got {index.ndim()} and {tensor.ndim()}"
        )
    return _F.gather(
        tensor, axis, _wrap_negative(broadcast_to(index, target), tensor.shape[axis])
    )


def _slice_index(index: Tensor, dim: int, shape: list[int], name: str) -> Tensor:
    """A 1-D `index` over `dim`, spread to the shape a scatter wants.

    `scatter` and `scatter_add` address every element individually, so an index
    that names whole slices has to be repeated across the other axes first.
    """

    if index.ndim() != 1:
        raise ValueError(f"{name} requires a 1-D index, got {index.ndim()} dimensions")
    spread = [1] * len(shape)
    spread[dim] = index.shape[0]
    target = list(shape)
    target[dim] = index.shape[0]
    return broadcast_to(index.reshape(spread), target)


def index_add(
    input: object, dim: int, index: object, source: object, alpha: float = 1.0
) -> Tensor:
    """Add `alpha * source` into the slices of `input` that `index` names.

    Repeated indices accumulate, which is the difference between this and
    `index_copy` and the reason it is the one used to build histograms and
    scatter gradients.
    """

    tensor = _atleast_tensor(input)
    values = _atleast_tensor(source)
    axis = _normalize_axis(dim, tensor.ndim(), "index_add")
    positions = _wrap_negative(_as_index(index, "index_add"), tensor.shape[axis])
    spread = _slice_index(positions, axis, list(values.shape), "index_add")
    scaled = values if alpha == 1.0 else values * alpha
    return _F.scatter_add(tensor, axis, spread, scaled)


def index_copy(input: object, dim: int, index: object, source: object) -> Tensor:
    """Write the slices of `source` over the slices of `input` that `index`
    names. A repeated index leaves whichever write landed last."""

    tensor = _atleast_tensor(input)
    values = _atleast_tensor(source)
    axis = _normalize_axis(dim, tensor.ndim(), "index_copy")
    positions = _wrap_negative(_as_index(index, "index_copy"), tensor.shape[axis])
    spread = _slice_index(positions, axis, list(values.shape), "index_copy")
    return _F.scatter(tensor, axis, spread, values)


def index_fill(input: object, dim: int, index: object, value: float) -> Tensor:
    """Set the slices of `input` that `index` names to `value`."""

    tensor = _atleast_tensor(input)
    axis = _normalize_axis(dim, tensor.ndim(), "index_fill")
    positions = _wrap_negative(_as_index(index, "index_fill"), tensor.shape[axis])
    filled = list(tensor.shape)
    filled[axis] = positions.shape[0]
    spread = _slice_index(positions, axis, filled, "index_fill")
    return _F.scatter(
        tensor, axis, spread, Tensor.full(filled, value, dtype=str(tensor.dtype))
    )


def masked_scatter(input: object, mask: object, source: object) -> Tensor:
    """Fill the positions `mask` selects with the leading elements of `source`.

    The `n`-th selected position takes the `n`-th element of a flattened
    `source`, in row-major order -- so this is a *positional* write, unlike
    `masked_fill`, which writes one value everywhere. The running count of
    selected positions is what turns the mask into those positions, and
    `cumsum` is that count.
    """

    tensor = _atleast_tensor(input)
    selected = _atleast_tensor(mask)
    values = _atleast_tensor(source).reshape(-1)

    shape = list(tensor.shape)
    flat_mask = broadcast_to(selected, shape).reshape(-1).astype("bool")
    needed = int(_F.count_nonzero(flat_mask).item())
    if needed > values.shape[0]:
        raise ValueError(
            f"masked_scatter needs at least {needed} source elements for the "
            f"positions the mask selects, got {values.shape[0]}"
        )
    if needed == 0:
        return tensor.reshape(shape)

    # The count *before* each position, which is the index into `source` that
    # position should read. Positions the mask skips read a stale index, and
    # the final `where` throws those away.
    running = _F.cumsum(flat_mask.astype("int64"), 0) - 1
    picked = _F.index_select(values, 0, _F.clamp(running, 0, values.shape[0] - 1))
    return _F.where(flat_mask, picked, tensor.reshape(-1)).reshape(shape)


def flatnonzero(input: object) -> Tensor:
    """The flat positions of every non-zero element, as a 1-D int64 tensor."""

    tensor = _atleast_tensor(input)
    return _F.nonzero(tensor.reshape(-1)).reshape(-1)


def argwhere(input: object) -> Tensor:
    """The indices of every non-zero element, one row each.

    The same answer `nonzero` gives; the name is the one NumPy users reach for
    when they want the rows rather than a tuple of coordinate arrays.
    """

    return _F.nonzero(_atleast_tensor(input))


def isin(
    elements: object,
    test_elements: object,
    assume_unique: bool = False,
    invert: bool = False,
) -> Tensor:
    """Whether each element of `elements` appears in `test_elements`.

    Done by sorting the test set once and binary-searching it, so the cost is
    `(n + m) log m` and the memory is `n + m`. The obvious alternative --
    comparing every element against every test element -- is `n * m` of both,
    which is what makes it unusable exactly when a membership test is worth
    reaching for.

    `assume_unique` is accepted for signature compatibility and changes
    nothing: the search does not care whether the test set repeats itself.
    """

    del assume_unique
    values = _atleast_tensor(elements)
    tests = _atleast_tensor(test_elements).reshape(-1)

    shape = list(values.shape)
    if tests.shape[0] == 0:
        # Nothing to be a member of.
        present = Tensor.full(shape, False, dtype="bool")
        return _F.logical_not(present) if invert else present

    dtype = _promoted_dtype(values, tests)
    flat = values.reshape(-1).astype(dtype)
    ordered = _F.sort(tests.astype(dtype))[0]

    # `searchsorted` reports where a value *would* go; the value is present
    # exactly when what is already there equals it. The clamp keeps a value
    # past the end of the test set reading a real position rather than one off
    # the end, and the comparison then rejects it.
    slot = _F.clamp(_F.searchsorted(ordered, flat), 0, ordered.shape[0] - 1)
    present = (_F.index_select(ordered, 0, slot) == flat).reshape(shape)
    return _F.logical_not(present) if invert else present


def _triangle_indices(row: int, col: int, offset: int, lower: bool) -> Tensor:
    name = "tril_indices" if lower else "triu_indices"
    rows = _operator.index(row)
    cols = _operator.index(col)
    if rows < 0 or cols < 0:
        raise ValueError(f"{name} requires non-negative sizes, got {row}, {col}")

    # Three Python integers in, an index pair out: nothing here has a device to
    # stay on, a dtype to agree with or a gradient to carry, which is exactly
    # the case where NumPy is the better engine. Doing it with kernels means
    # two ranges, a broadcast subtraction, a comparison, a `nonzero` and a
    # transpose -- six allocations to reach integers NumPy produces in one
    # call. NumPy takes the offset where the column count goes, so the
    # arguments are reordered rather than passed through.
    build = _np.tril_indices if lower else _np.triu_indices
    pair = _np.array(build(rows, _operator.index(offset), cols), dtype=_np.int64)
    return Tensor.from_numpy(pair)


def tril_indices(row: int, col: int, offset: int = 0) -> Tensor:
    """The `[2, n]` indices of the lower triangle of a `row` by `col` matrix.

    `offset` moves the boundary: 0 keeps the main diagonal, a positive value
    keeps that many more diagonals above it, a negative one drops that many.
    """

    return _triangle_indices(row, col, offset, lower=True)


def triu_indices(row: int, col: int, offset: int = 0) -> Tensor:
    """The `[2, n]` indices of the upper triangle of a `row` by `col` matrix."""

    return _triangle_indices(row, col, offset, lower=False)


def diagflat(input: object, offset: int = 0) -> Tensor:
    """A square matrix with the flattened `input` on its `offset` diagonal.

    `diag` does this for a vector already; this is the same thing for an input
    of any shape, which is the only difference between the two names.
    """

    return _F.diag(_atleast_tensor(input).reshape(-1), _operator.index(offset))


def select(input: object, dim: int, index: int) -> Tensor:
    """One slice along `dim`, with that dimension removed.

    `narrow` keeps the axis at length one; this drops it, which is what makes
    `select(t, 0, i)` the same as `t[i]`.
    """

    tensor = _atleast_tensor(input)
    axis = _normalize_axis(dim, tensor.ndim(), "select")
    length = tensor.shape[axis]
    position = _operator.index(index)
    if position < 0:
        position += length
    if not 0 <= position < length:
        raise IndexError(
            f"select index {index} is out of range for dimension {dim} of size {length}"
        )
    return _F.squeeze(_F.narrow(tensor, axis, position, 1), axis)


def block_diag(*tensors: object) -> Tensor:
    """Arrange the inputs down the diagonal of one larger matrix, zero
    elsewhere.

    A 1-D input is taken as a single row, and a scalar as a one-by-one block,
    matching NumPy's and PyTorch's reading. Built by padding each block out to
    the full width and joining the rows, so the zeros are never materialised
    twice.
    """

    if not tensors:
        return Tensor.zeros([1, 0], dtype="float32")

    blocks = []
    for value in tensors:
        block = _atleast_tensor(value)
        if block.ndim() == 0:
            block = block.reshape(1, 1)
        elif block.ndim() == 1:
            block = block.reshape(1, -1)
        elif block.ndim() != 2:
            raise ValueError(
                f"block_diag takes tensors of at most two dimensions, got {block.ndim()}"
            )
        blocks.append(block)

    width = sum(block.shape[1] for block in blocks)
    rows = []
    left = 0
    for block in blocks:
        right = width - left - block.shape[1]
        # `pad` is innermost-axis-first, so `[left, right]` pads the columns.
        rows.append(_F.pad(block, [left, right]) if width else block)
        left += block.shape[1]
    return _F.cat(rows, 0)


def cartesian_prod(*tensors: object) -> Tensor:
    """Every combination of one element from each input, one row each.

    With a single input the result is that input, which is what
    `itertools.product` of one sequence gives and what PyTorch returns.
    """

    if not tensors:
        raise ValueError("cartesian_prod requires at least one tensor")

    vectors = []
    for value in tensors:
        vector = _atleast_tensor(value)
        if vector.ndim() > 1:
            raise ValueError(
                f"cartesian_prod takes 1-D tensors, got {vector.ndim()} dimensions"
            )
        vectors.append(vector.reshape(-1))

    if len(vectors) == 1:
        return vectors[0]

    grids = _meshgrid(*vectors, indexing="ij")
    return _F.stack([grid.reshape(-1) for grid in grids], 1)


def _scatter_into(
    op: str, tensor: Tensor, source: object, positions: "_np.ndarray"
) -> Tensor:
    """`tensor` with `source` written at `positions`, as a new tensor.

    `positions` holds the flat offset of every element the operation writes,
    shaped the way the caller sees the region. The source is lined up against
    it by broadcasting, the way an assignment lines its right-hand side up,
    rather than by demanding an exact shape.

    The gradient is `scatter`'s: it reaches `source` at the positions it landed
    on and `tensor` everywhere else, which is what makes these expressions and
    not in-place writes.
    """

    values = _atleast_tensor(source)
    region = tuple(int(size) for size in positions.shape)
    if tuple(values.shape) != region:
        try:
            values = broadcast_to(values, region)
        except (ValueError, RuntimeError):
            raise ValueError(
                f"{op} writes a region of shape {region}, and a source of shape "
                f"{tuple(values.shape)} does not broadcast to it"
            ) from None

    written = _F.scatter(
        tensor.reshape(_element_count(tensor)),
        0,
        _index_tensor(positions.reshape(-1), tensor),
        values.reshape(int(positions.size)),
    )
    return written.reshape(list(tensor.shape))


def _axis_positions(
    shape: tuple[int, ...], axis: int, along: "_np.ndarray"
) -> "_np.ndarray":
    """The flat offsets of `along` on `axis`, with every other axis in full.

    `ix_` opens the per-axis ranges into a mesh and `ravel_multi_index` folds
    the mesh into offsets, so only the selected region is ever materialised --
    where slicing a full index template would allocate one integer per element
    of the whole tensor. Both are arithmetic on shapes, which is why NumPy does
    it; see "Where an operation belongs" in `docs/development.md`.
    """

    ranges = [_np.arange(size) for size in shape]
    ranges[axis] = along
    return _np.ravel_multi_index(_np.ix_(*ranges), shape)


def slice_scatter(
    input: object,
    src: object,
    dim: int = 0,
    start: int | None = None,
    end: int | None = None,
    step: int = 1,
) -> Tensor:
    """`input` with `src` written into the slice along `dim`, as a new tensor.

    The functional form of `x[..., start:end:step, ...] = src`, for the cases
    an assignment cannot serve: inside a larger expression, or on a tensor that
    has to keep the place it already holds in the graph. The gradient goes to
    both operands -- to `src` at the positions it landed on, and to `input`
    everywhere else.

    `start`, `end` and `step` mean exactly what they mean in a Python slice,
    negative and out-of-range values included, because a Python slice is what
    computes them.
    """

    tensor = _atleast_tensor(input)
    axis = _normalize_axis(dim, tensor.ndim(), "slice_scatter")
    shape = tuple(int(size) for size in tensor.shape)
    try:
        bounds = slice(start, end, _operator.index(step)).indices(shape[axis])
    except ValueError as exc:
        raise ValueError(f"slice_scatter: {exc}") from None
    return _scatter_into(
        "slice_scatter", tensor, src, _axis_positions(shape, axis, _np.arange(*bounds))
    )


def select_scatter(input: object, src: object, dim: int, index: int) -> Tensor:
    """`input` with `src` written over the slice `select` would return.

    The other direction of `select`, and shaped to match it: `src` has one axis
    fewer than `input`, because the axis being written to is a single position
    rather than a range. That is the difference from `slice_scatter`, which
    keeps the axis.
    """

    tensor = _atleast_tensor(input)
    axis = _normalize_axis(dim, tensor.ndim(), "select_scatter")
    shape = tuple(int(size) for size in tensor.shape)
    length = shape[axis]
    position = _operator.index(index)
    if position < 0:
        position += length
    if not 0 <= position < length:
        raise IndexError(
            f"select_scatter index {index} is out of range for dimension {dim} "
            f"of size {length}"
        )
    positions = _axis_positions(shape, axis, _np.array([position]))
    return _scatter_into(
        "select_scatter",
        tensor,
        src,
        positions.reshape(shape[:axis] + shape[axis + 1 :]),
    )


def diagonal_scatter(input: object, src: object, offset: int = 0) -> Tensor:
    """`input` with `src` written onto the diagonal `diagonal` would return.

    The two line up by construction: `src` has the shape of
    `diagonal(input, offset)`, which reads the last two axes and leaves the
    diagonal as a new trailing one. Everything off the diagonal keeps its value
    and its gradient.

    An `offset` that runs the diagonal off the matrix leaves nothing to write,
    which is a length of zero rather than an error -- the same answer
    `diagonal` gives for it.
    """

    tensor = _atleast_tensor(input)
    if tensor.ndim() < 2:
        raise ValueError(
            f"diagonal_scatter needs at least two dimensions, got {tensor.ndim()}"
        )
    shape = tuple(int(size) for size in tensor.shape)
    rows, columns = shape[-2:]
    displacement = _operator.index(offset)
    down, across = max(0, -displacement), max(0, displacement)
    length = max(0, min(rows - down, columns - across))

    # One range per leading axis and one for the diagonal; the two matrix axes
    # share that last one, offset against each other by where it starts.
    grids = _np.ix_(*[_np.arange(size) for size in shape[:-2]], _np.arange(length))
    coordinates = [*grids[:-1], grids[-1] + down, grids[-1] + across]
    return _scatter_into(
        "diagonal_scatter", tensor, src, _np.ravel_multi_index(coordinates, shape)
    )


def put(
    input: object, index: object, source: object, accumulate: bool = False
) -> Tensor:
    """`input` with `source` written at the flat positions `index` names.

    The write direction of `take`, and read the same way: row-major over the
    whole tensor whatever its shape, with negative positions counting from the
    end. `accumulate` adds into the target instead of overwriting, which is
    also what decides what a repeated position means -- the sum of what landed
    there, or whichever write came last.

    The gradient reaches `source` at the positions it landed on and `input`
    everywhere else, or everywhere when accumulating, since an addition leaves
    what was already there.
    """

    tensor = _atleast_tensor(input)
    values = _atleast_tensor(source)
    indices = _as_index(index, "put")

    flat = tensor.reshape(_element_count(tensor))
    positions = _wrap_negative(indices.reshape(-1), int(flat.shape[0]))
    written = _atleast_tensor(broadcast_to(values, tuple(indices.shape))).reshape(
        int(positions.shape[0])
    )
    scatter = _F.scatter_add if accumulate else _F.scatter
    return scatter(flat, 0, positions, written).reshape(list(tensor.shape))


def diag_indices(n: int, ndim: int = 2) -> Tensor:
    """The `[ndim, n]` indices of the main diagonal of an `n`-sided cube.

    Every row is the same range, because the main diagonal is where all the
    coordinates agree. `tril_indices` and `triu_indices` shape their answers the
    same way, so the three can be used interchangeably.
    """

    side = _operator.index(n)
    rank = _operator.index(ndim)
    if side < 0:
        raise ValueError(f"diag_indices requires a non-negative size, got {n}")
    if rank < 1:
        raise ValueError(f"diag_indices requires at least one dimension, got {ndim}")
    return Tensor.from_numpy(_np.array(_np.diag_indices(side, rank), dtype=_np.int64))


def _shape_argument(shape: object, name: str) -> tuple[int, ...]:
    """`shape` as sizes, whether it arrived as a sequence or a single integer.

    The same normalisation `broadcast_shapes` and the creation functions use,
    so a shape that one of them accepts is a shape all of them accept, plus the
    one thing those allow and a coordinate conversion cannot: a shape with no
    axes at all, which has no coordinates to convert.
    """

    sizes = _normalize_shape_argument(shape, name)
    if not sizes:
        raise ValueError(f"{name} expects a shape with at least one axis")
    return sizes


def _bounds(tensor: Tensor) -> tuple[int, int]:
    """The smallest and largest value in `tensor`, as Python integers."""

    if _element_count(tensor) == 0:
        return 0, -1
    return int(tensor.min().item()), int(tensor.max().item())


def unravel_index(indices: object, shape: object) -> tuple[Tensor, ...]:
    """The coordinates of flat positions `indices` in a tensor of `shape`.

    One tensor per axis, each shaped like `indices` -- the form NumPy and
    PyTorch both return, so `input[unravel_index(k, input.shape)]` reads the
    way it does there. `stack` them on a new leading axis to get the `[ndim, n]`
    layout `tril_indices` and `diag_indices` use.

    The strides come from `shape` alone, so they are computed once in Python
    and applied to the whole index tensor at once, rather than a division at a
    time down the axes.

    Positions are checked against the tensor they claim to index, which costs
    one pass over `indices` -- a flat position that is out of range does not
    fail on its own, it silently names the wrong element.
    """

    sizes = _shape_argument(shape, "unravel_index")
    flat = _as_index(indices, "unravel_index")
    total = _element_count(sizes)

    low, high = _bounds(flat)
    if low < 0 or high >= total:
        raise IndexError(
            f"unravel_index was given positions in [{low}, {high}] for a shape "
            f"of {sizes}, which holds {total}"
        )

    # `strides[i]` is how far one step along axis `i` moves in row-major order.
    strides = _np.append(_np.cumprod(sizes[:0:-1])[::-1], 1)
    return tuple(
        (flat // int(stride)) % int(size) for stride, size in zip(strides, sizes)
    )


def ravel_multi_index(multi_index: object, dims: object) -> Tensor:
    """The flat position of each coordinate, the inverse of `unravel_index`.

    `multi_index` is one tensor per axis, or a single tensor whose *leading*
    axis is the coordinate -- which is the layout `tril_indices`,
    `triu_indices` and `diag_indices` produce, so their output can be handed
    straight here.

    Each coordinate is checked against the axis it indexes, for the same reason
    `unravel_index` checks: an out-of-range coordinate produces a position that
    is wrong rather than one that fails.
    """

    sizes = _shape_argument(dims, "ravel_multi_index")
    if isinstance(multi_index, (tuple, list)):
        coordinates = [_as_index(part, "ravel_multi_index") for part in multi_index]
    else:
        stacked = _as_index(multi_index, "ravel_multi_index")
        if stacked.ndim() < 1:
            raise ValueError(
                "ravel_multi_index needs one coordinate per axis, and a scalar "
                "carries none"
            )
        coordinates = [
            _F.squeeze(_F.narrow(stacked, 0, axis, 1), 0)
            for axis in range(int(stacked.shape[0]))
        ]

    if len(coordinates) != len(sizes):
        raise ValueError(
            f"ravel_multi_index was given {len(coordinates)} coordinate(s) for a "
            f"shape of {sizes}"
        )
    for axis, (coordinate, size) in enumerate(zip(coordinates, sizes)):
        low, high = _bounds(coordinate)
        if low < 0 or high >= size:
            raise IndexError(
                f"ravel_multi_index was given coordinates in [{low}, {high}] for "
                f"axis {axis} of size {size}"
            )

    strides = _np.append(_np.cumprod(sizes[:0:-1])[::-1], 1)
    position = coordinates[0] * int(strides[0])
    for coordinate, stride in zip(coordinates[1:], strides[1:]):
        position = position + coordinate * int(stride)
    return position
