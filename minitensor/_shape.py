# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""Small Python shape helpers for MiniTensor's public API."""

from __future__ import annotations

import operator as _operator

import numpy as _np

from . import _core as _C

Tensor = _C.Tensor
as_tensor = Tensor.as_tensor


def _normalize_dimension(dim: object, name: str) -> int:
    if isinstance(dim, bool):
        raise TypeError(f"{name} dimensions must be integers, not bool")

    try:
        normalized = _operator.index(dim)
    except TypeError as exc:
        raise TypeError(f"{name} dimensions must be integers") from exc

    if normalized < 0:
        raise ValueError(f"{name} dimensions must be non-negative")
    return normalized


def _normalize_shape_argument(shape: object, name: str) -> tuple[int, ...]:
    if isinstance(shape, bool):
        raise TypeError(f"{name} dimensions must be integers, not bool")

    try:
        return (_normalize_dimension(shape, name),)
    except TypeError:
        pass

    try:
        dims = tuple(shape)  # type: ignore[arg-type]
    except TypeError as exc:
        raise TypeError(f"{name} must be an int or an iterable of ints") from exc

    return tuple(_normalize_dimension(dim, name) for dim in dims)


def broadcast_shapes(*shapes: object) -> tuple[int, ...]:
    """Return the shape produced by broadcasting.

    Each argument may be a single non-negative integer dimension or an iterable
    of non-negative integer dimensions. Scalar shapes are represented by an
    empty iterable, e.g. ``broadcast_shapes((), (2, 3)) == (2, 3)``.
    """

    if not shapes:
        return ()

    normalized_shapes = [
        _normalize_shape_argument(shape, f"shapes[{index}]")
        for index, shape in enumerate(shapes)
    ]
    try:
        return tuple(int(dim) for dim in _np.broadcast_shapes(*normalized_shapes))
    except ValueError as exc:
        raise ValueError(
            "shapes cannot be broadcast together: "
            + ", ".join(str(shape) for shape in normalized_shapes)
        ) from exc


def can_broadcast(*shapes: object) -> bool:
    """Return ``True`` when shapes can broadcast without creating tensors."""

    try:
        broadcast_shapes(*shapes)
    except (TypeError, ValueError):
        return False
    return True


def broadcast_tensors(*inputs: object) -> tuple[Tensor, ...]:
    """Broadcast tensor-like inputs to a shared shape.

    Inputs are converted with :func:`as_tensor`, then reshaped and expanded
    to a shared shape. The returned tensors are
    materialized with contiguous storage so they behave identically to
    dense tensors in every operation. Valid zero-sized broadcasts return
    empty tensors preserving dtype, device, and ``requires_grad`` metadata.
    """

    if not inputs:
        raise TypeError("broadcast_tensors requires at least one input")

    tensors = tuple(_atleast_tensor(input) for input in inputs)
    target_shape = broadcast_shapes(*(tuple(tensor.shape) for tensor in tensors))

    return tuple(_broadcast_tensor_to(tensor, target_shape) for tensor in tensors)


def broadcast_to(input: object, shape: object) -> Tensor:
    """Broadcast a tensor-like input to an explicit target shape.

    The input is converted with :func:`as_tensor`, and ``shape`` accepts the
    same validated shape-like values as :func:`broadcast_shapes`. The returned
    tensor is the original tensor when it already has the requested shape, a
    materialized contiguous broadcast otherwise, or a metadata-preserving
    empty tensor for valid zero-sized broadcasts.
    """

    tensor = _atleast_tensor(input)
    target_shape = _normalize_shape_argument(shape, "shape")
    # Reuse the shared broadcast validator so error behavior is identical to
    # broadcast_shapes/broadcast_tensors before asking the backend to expand.
    broadcasted_shape = broadcast_shapes(tuple(tensor.shape), target_shape)
    if broadcasted_shape != target_shape:
        raise ValueError(
            f"input shape {tuple(tensor.shape)} cannot be broadcast to {target_shape}"
        )
    return _broadcast_tensor_to(tensor, target_shape)


def _broadcast_tensor_to(tensor: Tensor, target_shape: tuple[int, ...]) -> Tensor:
    current_shape = tuple(tensor.shape)
    if current_shape == target_shape:
        return tensor

    if _requires_zero_size_materialization(current_shape, target_shape):
        return Tensor.empty(
            target_shape,
            dtype=tensor.dtype,
            device=_C.Device(tensor.device),
            requires_grad=tensor.requires_grad,
        )

    rank_delta = len(target_shape) - len(current_shape)
    reshaped = tensor
    if rank_delta:
        reshaped = tensor.reshape((1,) * rank_delta + current_shape)
    return reshaped.expand(*target_shape)


def _requires_zero_size_materialization(
    current_shape: tuple[int, ...], target_shape: tuple[int, ...]
) -> bool:
    """Return whether broadcasting must create an empty tensor.

    The Rust backend can expand existing zero-sized axes, but it cannot model
    an axis that changes from length one to zero as a view because that shape
    has no addressable elements. The Python helper returns a correctly shaped
    empty tensor for that edge case.
    """

    if 0 not in target_shape:
        return False

    padded_shape = (1,) * (len(target_shape) - len(current_shape)) + current_shape
    return any(
        current_dim == 1 and target_dim == 0
        for current_dim, target_dim in zip(padded_shape, target_shape)
    )


def _atleast_tensor(input: object) -> Tensor:
    """Convert an input to a Tensor while preserving existing Tensor objects."""

    if isinstance(input, Tensor):
        return input
    return as_tensor(input)


def _return_atleast_result(results: list[Tensor]) -> Tensor | tuple[Tensor, ...]:
    if len(results) == 1:
        return results[0]
    return tuple(results)


def meshgrid(
    *inputs: object, indexing: str = "xy", sparse: bool = False, copy: bool = False
) -> tuple[Tensor, ...]:
    """Return coordinate matrices from one-dimensional coordinate tensors.

    This helper accepts tensor-like 1-D inputs and returns broadcasted coordinate
    grids. ``indexing="ij"`` preserves input axis
    order, while ``indexing="xy"`` swaps the first two axes for Cartesian
    plotting conventions. With ``sparse=True`` the function returns reshaped
    coordinate vectors that broadcast lazily instead of materializing full
    grids. Set ``copy=True`` when independent dense tensor storage is required.
    """

    if not inputs:
        return ()

    if not isinstance(indexing, str):
        raise TypeError("indexing must be a string")
    if indexing not in {"xy", "ij"}:
        raise ValueError('indexing must be either "xy" or "ij"')
    if not isinstance(sparse, bool):
        raise TypeError("sparse must be a bool")
    if not isinstance(copy, bool):
        raise TypeError("copy must be a bool")

    vectors = tuple(
        _meshgrid_vector(input, index) for index, input in enumerate(inputs)
    )
    ndim = len(vectors)
    lengths = [int(vector.shape[0]) for vector in vectors]
    if indexing == "xy" and ndim > 1:
        lengths[0], lengths[1] = lengths[1], lengths[0]

    results: list[Tensor] = []
    for axis, vector in enumerate(vectors):
        output_axis = _meshgrid_output_axis(axis, ndim, indexing)
        view_shape = [1] * ndim
        view_shape[output_axis] = int(vector.shape[0])
        reshaped = vector.reshape(*view_shape)
        if not sparse:
            reshaped = broadcast_to(reshaped, tuple(lengths))
        results.append(reshaped.clone() if copy else reshaped)
    return tuple(results)


def _meshgrid_vector(input: object, index: int) -> Tensor:
    tensor = _atleast_tensor(input)
    ndim = tensor.ndim()
    if ndim == 0:
        return tensor.reshape(1)
    if ndim != 1:
        raise ValueError(
            f"meshgrid inputs must be scalars or 1-D tensors; input {index} has ndim {ndim}"
        )
    return tensor


def _meshgrid_output_axis(axis: int, ndim: int, indexing: str) -> int:
    if indexing == "xy" and ndim > 1:
        if axis == 0:
            return 1
        if axis == 1:
            return 0
    return axis


def atleast_1d(*inputs: object) -> Tensor | tuple[Tensor, ...]:
    """Convert inputs to tensors with at least one dimension.

    Scalar inputs are reshaped to ``(1,)``. Inputs that are already at least
    one-dimensional are returned as tensors without adding dimensions.
    Multiple inputs return a tuple of tensors.
    """

    if not inputs:
        raise TypeError("atleast_1d requires at least one input")

    results: list[Tensor] = []
    for input in inputs:
        tensor = _atleast_tensor(input)
        results.append(tensor.reshape(1) if tensor.ndim() == 0 else tensor)
    return _return_atleast_result(results)


def atleast_2d(*inputs: object) -> Tensor | tuple[Tensor, ...]:
    """Convert inputs to tensors with at least two dimensions.

    Scalars become shape ``(1, 1)`` and one-dimensional tensors become row
    tensors of shape ``(1, N)``. Higher-rank tensors are returned unchanged.
    """

    if not inputs:
        raise TypeError("atleast_2d requires at least one input")

    results: list[Tensor] = []
    for input in inputs:
        tensor = _atleast_tensor(input)
        ndim = tensor.ndim()
        if ndim == 0:
            results.append(tensor.reshape(1, 1))
        elif ndim == 1:
            results.append(tensor.unsqueeze(0))
        else:
            results.append(tensor)
    return _return_atleast_result(results)


def atleast_3d(*inputs: object) -> Tensor | tuple[Tensor, ...]:
    """Convert inputs to tensors with at least three dimensions.

    Scalars become ``(1, 1, 1)``, one-dimensional tensors become
    ``(1, N, 1)``, and two-dimensional tensors gain a trailing singleton
    dimension. Higher-rank tensors are returned unchanged.
    """

    if not inputs:
        raise TypeError("atleast_3d requires at least one input")

    results: list[Tensor] = []
    for input in inputs:
        tensor = _atleast_tensor(input)
        ndim = tensor.ndim()
        if ndim == 0:
            results.append(tensor.reshape(1, 1, 1))
        elif ndim == 1:
            results.append(tensor.reshape(1, tensor.shape[0], 1))
        elif ndim == 2:
            results.append(tensor.unsqueeze(2))
        else:
            results.append(tensor)
    return _return_atleast_result(results)


def _stack_inputs(tensors: object, name: str) -> list[Tensor]:
    """The sequence a stacking helper was handed, as tensors.

    A single tensor is not a sequence of them: `vstack(t)` is a mistake worth
    naming rather than an iteration over `t`'s rows.
    """

    if isinstance(tensors, Tensor):
        raise TypeError(f"{name} takes a sequence of tensors, not one tensor")

    try:
        items = list(tensors)  # type: ignore[arg-type]
    except TypeError as exc:
        raise TypeError(f"{name} takes a sequence of tensors") from exc

    if not items:
        raise ValueError(f"{name} needs at least one tensor")
    return [_atleast_tensor(item) for item in items]


def hstack(tensors: object) -> Tensor:
    """Join along the second axis, or the first for 1-D inputs.

    "Horizontally", which for a 1-D tensor means end to end, since it has no
    second axis to grow.
    """

    items = _stack_inputs(tensors, "hstack")
    axis = 0 if all(item.ndim() <= 1 for item in items) else 1
    return _C.functional.cat([atleast_1d(item) for item in items], axis)


def vstack(tensors: object) -> Tensor:
    """Join along the first axis, after promoting 1-D inputs to rows."""

    items = _stack_inputs(tensors, "vstack")
    return _C.functional.cat([atleast_2d(item) for item in items], 0)


def dstack(tensors: object) -> Tensor:
    """Join along the third axis, after promoting lower-rank inputs to it."""

    items = _stack_inputs(tensors, "dstack")
    return _C.functional.cat([atleast_3d(item) for item in items], 2)


def column_stack(tensors: object) -> Tensor:
    """Join as columns: 1-D inputs become columns, the rest stack along axis 1."""

    items = _stack_inputs(tensors, "column_stack")
    promoted = [
        item.reshape(item.shape[0], 1) if item.ndim() == 1 else atleast_2d(item)
        for item in items
    ]
    return _C.functional.cat(promoted, 1)


def tile(input: object, reps: object) -> Tensor:
    """Repeat the tensor `reps` times along each axis.

    Unlike `repeat`, `reps` may be shorter than the tensor's rank; the missing
    leading entries are taken as 1, which is NumPy's rule and the reason both
    spellings exist.
    """

    tensor = _atleast_tensor(input)
    counts = list(_normalize_shape_argument(reps, "tile"))
    if len(counts) < tensor.ndim():
        counts = [1] * (tensor.ndim() - len(counts)) + counts
    return tensor.repeat(counts)


def unbind(input: object, dim: int = 0) -> tuple[Tensor, ...]:
    """Every slice along `dim`, with that dimension removed.

    The inverse of `stack`, as `split` is the inverse of `cat`: what comes back
    has one dimension fewer, not a length-1 one.
    """

    tensor = _atleast_tensor(input)
    if tensor.ndim() == 0:
        raise ValueError("unbind requires a tensor with at least one dimension")

    axis = _normalize_axis(dim, tensor.ndim(), "unbind")
    return tuple(
        _C.functional.narrow(tensor, axis, index, 1).squeeze(axis)
        for index in range(tensor.shape[axis])
    )


def tensor_split(
    input: object, indices_or_sections: object, dim: int = 0
) -> tuple[Tensor, ...]:
    """Split into `n` parts, or at the given indices, without requiring an
    even division.

    `split` takes a piece *size* and leaves whatever is left over as a short
    final piece -- ten split by three is `[3, 3, 3, 1]`. This one takes a
    *count* and balances, spreading the remainder one element at a time over
    the leading parts: ten into three is `[4, 3, 3]`.
    """

    tensor = _atleast_tensor(input)
    if tensor.ndim() == 0:
        raise ValueError("tensor_split requires a tensor with at least one dimension")

    axis = _normalize_axis(dim, tensor.ndim(), "tensor_split")
    length = tensor.shape[axis]

    if isinstance(indices_or_sections, Tensor):
        raise TypeError("tensor_split takes an int or a sequence of ints")

    try:
        sections = _operator.index(indices_or_sections)
    except TypeError:
        bounds = [_operator.index(index) for index in indices_or_sections]  # type: ignore[union-attr]
        edges = [0, *(min(max(index, 0), length) for index in bounds), length]
    else:
        if sections <= 0:
            raise ValueError(
                f"tensor_split requires a positive number of sections, got {sections}"
            )
        base, extra = divmod(length, sections)
        edges = [0]
        for part in range(sections):
            edges.append(edges[-1] + base + (1 if part < extra else 0))

    return tuple(
        _C.functional.narrow(tensor, axis, start, max(stop - start, 0))
        for start, stop in zip(edges, edges[1:])
    )


def _normalize_axis(dim: object, ndim: int, name: str) -> int:
    try:
        axis = _operator.index(dim)
    except TypeError as exc:
        raise TypeError(f"{name} requires an integer dim") from exc

    if axis < 0:
        axis += ndim
    if not 0 <= axis < ndim:
        raise ValueError(
            f"{name} dim {dim} is out of range for a {ndim}-dimensional tensor"
        )
    return axis


def fliplr(input: object) -> Tensor:
    """Reverse the columns: `flip` on axis 1, which needs a second axis."""

    tensor = _atleast_tensor(input)
    if tensor.ndim() < 2:
        raise ValueError("fliplr requires a tensor with at least two dimensions")
    return _C.functional.flip(tensor, [1])


def flipud(input: object) -> Tensor:
    """Reverse the rows: `flip` on axis 0."""

    tensor = _atleast_tensor(input)
    if tensor.ndim() < 1:
        raise ValueError("flipud requires a tensor with at least one dimension")
    return _C.functional.flip(tensor, [0])


def rot90(input: object, k: int = 1, dims: object = (0, 1)) -> Tensor:
    """Rotate by 90 degrees `k` times in the plane `dims` spans.

    A rotation is a transpose and a flip; which of the two axes is flipped is
    what makes it a rotation rather than a reflection, so the direction of `k`
    decides that rather than the order of `dims`.
    """

    tensor = _atleast_tensor(input)
    if tensor.ndim() < 2:
        raise ValueError("rot90 requires a tensor with at least two dimensions")

    try:
        first, second = (_operator.index(axis) for axis in dims)  # type: ignore[misc]
    except (TypeError, ValueError) as exc:
        raise TypeError("rot90 dims must be a pair of integers") from exc

    first = _normalize_axis(first, tensor.ndim(), "rot90")
    second = _normalize_axis(second, tensor.ndim(), "rot90")
    if first == second:
        raise ValueError("rot90 dims must name two different axes")

    quarters = _operator.index(k) % 4
    if quarters == 0:
        return tensor
    if quarters == 2:
        return _C.functional.flip(tensor, [first, second])

    transposed = _C.functional.transpose(tensor, first, second)
    flipped = first if quarters == 1 else second
    return _C.functional.flip(transposed, [flipped])


def unflatten(input: object, dim: int, sizes: object) -> Tensor:
    """Split one axis into several, the inverse of `flatten`.

    One entry of `sizes` may be `-1`, and is worked out from the length of the
    axis being split. `reshape` can do the same thing, but only by restating
    every other dimension of the tensor -- which is the mistake this exists to
    stop.
    """

    tensor = _atleast_tensor(input)
    axis = _normalize_axis(dim, tensor.ndim(), "unflatten")
    parts = [_operator.index(size) for size in sizes]

    inferred = [i for i, size in enumerate(parts) if size == -1]
    if len(inferred) > 1:
        raise ValueError("unflatten can infer at most one dimension")
    if any(size < 0 and size != -1 for size in parts):
        raise ValueError(f"unflatten sizes must be non-negative or -1, got {parts}")

    length = tensor.shape[axis]
    if inferred:
        known = 1
        for size in parts:
            if size != -1:
                known *= size
        if known == 0 or length % known:
            raise ValueError(f"unflatten cannot split an axis of {length} into {parts}")
        parts[inferred[0]] = length // known
    else:
        total = 1
        for size in parts:
            total *= size
        if total != length:
            raise ValueError(
                f"unflatten sizes {parts} multiply to {total}, not the axis's {length}"
            )

    dims = list(tensor.shape)
    return tensor.reshape(dims[:axis] + parts + dims[axis + 1 :])


def msort(input: object) -> Tensor:
    """Sort along the first dimension, values only.

    `sort` returns the indices as well and defaults to the last dimension;
    this is the shorthand NumPy and PyTorch both spell this way.
    """

    return _C.functional.sort(_atleast_tensor(input), 0)[0]


def _split_along(
    input: object, indices_or_sections: object, axis: int, name: str, minimum: int
):
    tensor = _atleast_tensor(input)
    if tensor.ndim() < minimum:
        raise ValueError(
            f"{name} requires at least {minimum} dimensions, got {tensor.ndim()}"
        )
    return tensor_split(tensor, indices_or_sections, axis)


def hsplit(input: object, indices_or_sections: object) -> tuple[Tensor, ...]:
    """Split along the second axis, or the first for a 1-D input.

    A vector has only one axis to split horizontally, so that is the one taken.
    """

    tensor = _atleast_tensor(input)
    axis = 0 if tensor.ndim() == 1 else 1
    return _split_along(tensor, indices_or_sections, axis, "hsplit", 1)


def vsplit(input: object, indices_or_sections: object) -> tuple[Tensor, ...]:
    """Split along the first axis. Needs at least two dimensions: a vector has
    no rows to split."""

    return _split_along(input, indices_or_sections, 0, "vsplit", 2)


def dsplit(input: object, indices_or_sections: object) -> tuple[Tensor, ...]:
    """Split along the third axis."""

    return _split_along(input, indices_or_sections, 2, "dsplit", 3)


def kthvalue(
    input: object, k: int, dim: int = -1, keepdim: bool = False
) -> tuple[Tensor, Tensor]:
    """The `k`-th smallest value along `dim`, and where it came from.

    `k` counts from one, as it does in every other library that offers this,
    so `kthvalue(x, 1)` is the minimum and `kthvalue(x, n)` the maximum.
    """

    tensor = _atleast_tensor(input)
    if tensor.ndim() == 0:
        raise ValueError("kthvalue requires a tensor with at least one dimension")
    axis = _normalize_axis(dim, tensor.ndim(), "kthvalue")
    position = _operator.index(k)
    length = tensor.shape[axis]
    if not 1 <= position <= length:
        raise ValueError(
            f"kthvalue requires 1 <= k <= {length} for an axis of that length, got {k}"
        )

    values, indices = _C.functional.sort(tensor, axis)
    picked = _C.functional.narrow(values, axis, position - 1, 1)
    where = _C.functional.narrow(indices, axis, position - 1, 1)
    if keepdim:
        return picked, where
    return (
        _C.functional.squeeze(picked, axis),
        _C.functional.squeeze(where, axis),
    )


def combinations(input: object, r: int = 2, with_replacement: bool = False) -> Tensor:
    """Every combination of `r` elements of a 1-D `input`, one row each.

    In lexicographic order over positions, as `itertools.combinations` gives
    them, so a caller can line the rows up against that without sorting. The
    row count is the binomial coefficient, which grows fast enough that this
    builds the index list in Python rather than as a tensor operation: at the
    sizes where the tensor version would pay, the answer does not fit in
    memory anyway.
    """

    import itertools as _itertools

    tensor = _atleast_tensor(input)
    if tensor.ndim() != 1:
        raise ValueError(f"combinations requires a 1-D tensor, got {tensor.ndim()}")
    count = _operator.index(r)
    if count < 0:
        raise ValueError(f"combinations requires a non-negative r, got {r}")

    choose = (
        _itertools.combinations_with_replacement
        if with_replacement
        else _itertools.combinations
    )
    rows = list(choose(range(tensor.shape[0]), count))
    if not rows or count == 0:
        # No rows, or rows with nothing in them: either way there is nothing to
        # select, and the shape is the whole answer.
        return _C.Tensor.zeros([len(rows), count], dtype=str(tensor.dtype))

    flat = _np.asarray(rows, dtype=_np.int64).reshape(-1)
    picked = _C.functional.index_select(tensor, 0, as_tensor(flat))
    return picked.reshape(len(rows), count)
