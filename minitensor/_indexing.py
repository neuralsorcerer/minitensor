# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""Indexing, scattering and index-producing helpers, in terms of the kernels.

Every function here rearranges what `index_select`, `gather`, `scatter`,
`nonzero`, `searchsorted` and `pad` already do. None of them needs a kernel of
its own: `take` is `index_select` over a flattened view, `index_add` is
`scatter_add` with the index broadcast, `isin` is a sorted lookup, and the
`*_indices` builders are a comparison run through `nonzero`. Writing them here
rather than in the extension keeps the shipped binary the size of the
operations that genuinely need one, and each inherits the bounds checking, the
dtype rules and the gradients of the kernel underneath instead of restating
them.
"""

from __future__ import annotations

import operator as _operator

from . import _core as _C
from ._shape import (
    _atleast_tensor,
    _normalize_axis,
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
    rows = _operator.index(row)
    cols = _operator.index(col)
    if rows < 0 or cols < 0:
        raise ValueError(f"tril_indices requires non-negative sizes, got {row}, {col}")

    down = Tensor.arange(0, rows, 1, dtype="int64").reshape(rows, 1)
    across = Tensor.arange(0, cols, 1, dtype="int64").reshape(1, cols)
    diagonal = across - down
    selected = (
        diagonal <= _operator.index(offset)
        if lower
        else diagonal >= _operator.index(offset)
    )
    # `nonzero` gives one row per position; the convention for an index pair is
    # one row per *axis*, so it is transposed.
    return _F.transpose(_F.nonzero(selected), 0, 1)


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
