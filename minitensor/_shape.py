# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""Small Python shape helpers for MiniTensor's public API."""

from __future__ import annotations

import math as _math
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


def _element_count(sizes: object) -> int:
    """How many elements a shape holds, or a tensor does.

    Spelled out rather than inferred with `-1` in a reshape: an axis of length
    zero leaves the inference nothing to divide by, and the failure then reads
    as a reshape error rather than as the empty answer it should be.
    """

    return _math.prod(int(size) for size in getattr(sizes, "shape", sizes))


def _index_tensor(array: "_np.ndarray", like: Tensor) -> Tensor:
    """A NumPy index as an int64 tensor beside the data it will address.

    The other half of the rule in "Where an operation belongs"
    (`docs/development.md`): NumPy computes the positions, and this is where
    they cross back, onto the device holding what they index.
    """

    return Tensor.from_numpy(_np.ascontiguousarray(array, dtype=_np.int64)).to(
        _C.Device(like.device)
    )


def _constant_like(array: "_np.ndarray", like: Tensor) -> Tensor:
    """A NumPy array as a tensor in `like`'s dtype and on its device.

    For a constant computed from shapes rather than from data -- a mask, a
    coordinate grid -- which must join an expression without changing its
    precision or moving it off its device.
    """

    return (
        Tensor.from_numpy(_np.ascontiguousarray(array, dtype=_np.float64))
        .astype(str(like.dtype))
        .to(_C.Device(like.device))
    )


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


def _as_written_values(value: object, into: Tensor) -> Tensor:
    """`value` as something the write family can put into `into`.

    An operand that carries a dtype of its own -- a tensor, an array -- keeps
    it, and a disagreement with the destination is the engine's to refuse: two
    typed operands really do disagree. A Python number or list carries no
    dtype, so it takes the destination's, the way `x[i] = 7.0` does and the way
    `index_fill` and `masked_fill` already do with the value they are handed.

    Without this a literal was built at the default dtype and then rejected
    against anything else, so `put(x, i, 7.0)` worked on a float32 tensor and
    raised on a float64 one -- the same expression, refused for the dtype of
    the tensor it was writing into rather than for anything about the value.
    """

    if isinstance(value, (Tensor, _np.ndarray)):
        return value if isinstance(value, Tensor) else as_tensor(value)
    return as_tensor(value, dtype=str(into.dtype))


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


def _promoted_dtype(left: Tensor, right: Tensor, operation: object = None) -> str:
    """The dtype the library's own promotion gives these two.

    Asked by doing the promotion on nothing: an empty operation touches no
    elements and answers exactly what a full one would, which beats restating
    the promotion table here where it could drift from the real one.

    `operation` picks which table: the default is addition's, and the
    float-valued ops pass division, whose promotion is the one that always
    lands on a float.
    """

    empty_left = _C.functional.narrow(left.reshape(-1), 0, 0, 0)
    empty_right = _C.functional.narrow(right.reshape(-1), 0, 0, 0)
    if operation is None:
        return str((empty_left + empty_right).dtype)
    return str(operation(empty_left, empty_right).dtype)


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


def _normalize_axis_tuple(dim: object, ndim: int, name: str) -> tuple[int, ...]:
    """One axis or a sequence of them, each brought into range, none repeated.

    A repeated axis is rejected rather than folded away: an op that trims or
    reduces one axis twice would do the second pass on the answer of the first,
    so asking for it is a mistake, not a shorthand.
    """

    try:
        return (_normalize_axis(dim, ndim, name),)
    except TypeError:
        pass

    try:
        entries = tuple(dim)  # type: ignore[call-overload]
    except TypeError as exc:
        raise TypeError(
            f"{name} requires an integer dim or a sequence of them"
        ) from exc

    axes = tuple(_normalize_axis(entry, ndim, name) for entry in entries)
    if len(set(axes)) != len(axes):
        raise ValueError(f"{name} was given a repeated axis in {dim!r}")
    return axes


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

    # A sort answers this and answers far more than was asked: it orders the
    # whole axis to be told about one position in it. `topk` stops once `k`
    # elements are settled, so asking it for the `k` smallest and taking the
    # last is the same answer for a fraction of the work -- 18ms against 75ms
    # for the hundredth of 2048, and 3ms against 75ms for the first.
    #
    # The `k` smallest, never the `n - k + 1` largest from the other end, even
    # though that is cheaper still when `k` is large. The two ends disagree
    # about which of several equal elements to name, and the index this reports
    # for a tie would change with `k`, which is a worse thing to be than slow.
    # Past the halfway point the sort is the cheaper of the two anyway.
    if position * 2 <= length:
        values, indices = _C.functional.topk(tensor, position, axis, False, True)
    else:
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


def _partition_positions(kth: object, length: int, name: str) -> list[int]:
    """The `kth` argument as a list of positions, checked against the axis."""

    if isinstance(kth, (Tensor, _np.ndarray, list, tuple)):
        raw = (
            [
                int(v)
                for v in _np.asarray(
                    kth.numpy() if isinstance(kth, Tensor) else kth
                ).reshape(-1)
            ]
            if not isinstance(kth, (list, tuple))
            else [_operator.index(v) for v in kth]
        )
    else:
        raw = [_operator.index(kth)]
    if not raw:
        raise ValueError(f"{name} needs at least one position to partition around")
    for position in raw:
        wrapped = position + length if position < 0 else position
        if not 0 <= wrapped < length:
            raise IndexError(
                f"{name} position {position} is out of bounds for an axis of {length}"
            )
    return raw


def partition(input: object, kth: object, dim: int = -1) -> Tensor:
    """Each slice along `dim` rearranged so position `kth` holds what a sort
    would put there, with everything before it no greater and everything after
    no less.

    The rest of the order is unspecified, and that is the point: the selection
    is linear in the slice where a sort is `n log n`, so asking "what are the
    ten smallest" costs a pass rather than an ordering. Reach for `sort` when
    the order of the rest matters.

    `kth` may be several positions, each of which lands where a sort would put
    it, and may count from the end. `dim=None` partitions the flattened tensor.
    NaN sorts after every number, as it does for `sort`.
    """

    tensor = _atleast_tensor(input)
    if dim is None:
        flat = tensor.reshape(-1)
        positions = _partition_positions(kth, flat.shape[0], "partition")
        return _C.functional.partition(flat, positions, 0, False)[0]
    axis = _normalize_axis(dim, max(tensor.ndim(), 1), "partition")
    length = tensor.shape[axis] if tensor.ndim() else 1
    positions = _partition_positions(kth, length, "partition")
    return _C.functional.partition(tensor, positions, axis, False)[0]


def argpartition(input: object, kth: object, dim: int = -1) -> Tensor:
    """Where the elements `partition` would produce came from.

    The same selection, reporting positions instead of values, so
    `take_along_dim(x, argpartition(x, k), dim)` is a partition of the same
    data around the same `k`: position `k` holds what a sort would leave there
    and the two sides hold the same values.

    Not the *same arrangement* as `partition` when values repeat. Both are
    valid answers -- the order of everything but `k` is unspecified, which is
    what makes a selection cheaper than a sort -- and the two take different
    routes to it: reporting positions means carrying them through the
    selection, and carrying them costs enough that the value-only form does
    without.
    """

    tensor = _atleast_tensor(input)
    if dim is None:
        flat = tensor.reshape(-1)
        positions = _partition_positions(kth, flat.shape[0], "argpartition")
        return _C.functional.partition(flat, positions, 0, True)[1]
    axis = _normalize_axis(dim, max(tensor.ndim(), 1), "argpartition")
    length = tensor.shape[axis] if tensor.ndim() else 1
    positions = _partition_positions(kth, length, "argpartition")
    return _C.functional.partition(tensor, positions, axis, True)[1]


def lexsort(keys: object, dim: int = -1) -> Tensor:
    """The order that sorts by several keys at once, last key first.

    The last key is the primary one and earlier keys break its ties, which is
    NumPy's convention and the one that reads correctly when the keys are
    written in the order a table's columns are.

    Done as one stable sort per key, least significant first: a stable sort
    leaves the order the previous keys established wherever the current one
    ties, so `k` passes settle `k` keys. Sorting by a composite key instead
    would need the keys to be commensurable, which they are not.
    """

    if isinstance(keys, Tensor):
        columns = [keys] if keys.ndim() == 1 else list(unbind(keys, 0))
    else:
        columns = [_atleast_tensor(key) for key in keys]
    if not columns:
        raise ValueError("lexsort needs at least one key")

    shape = list(columns[0].shape)
    for key in columns[1:]:
        if list(key.shape) != shape:
            raise ValueError(
                f"lexsort needs every key to have the same shape, got "
                f"{tuple(key.shape)} and {tuple(shape)}"
            )
    if not shape:
        raise ValueError("lexsort requires keys with at least one dimension")

    axis = _normalize_axis(dim, len(shape), "lexsort")
    length = shape[axis]
    spread = [1] * len(shape)
    spread[axis] = length
    order = broadcast_to(
        Tensor.arange(0, length, 1, dtype="int64").reshape(spread), shape
    )
    for key in columns:
        ranked = _C.functional.gather(key, axis, order)
        order = _C.functional.gather(
            order, axis, _C.functional.argsort(ranked, axis, False, True)
        )
    return order


def expand_dims(input: object, dim: object) -> Tensor:
    """A view with a length-1 axis inserted at each position in `dim`.

    `unsqueeze` for one axis; this takes several at once, and the positions
    refer to the *result*, so `expand_dims(x, (0, 2))` puts new axes at 0 and 2
    of the four-dimensional answer rather than at 0 and 2 of the original.
    """

    tensor = _atleast_tensor(input)
    positions = (
        [_operator.index(dim)]
        if isinstance(dim, (int, _np.integer))
        else [_operator.index(value) for value in dim]
    )
    rank = tensor.ndim() + len(positions)
    resolved = sorted(
        _normalize_axis(position, rank, "expand_dims") for position in positions
    )
    if len(set(resolved)) != len(resolved):
        raise ValueError(
            f"expand_dims was given the same axis twice: {tuple(positions)}"
        )
    for position in resolved:
        tensor = tensor.unsqueeze(position)
    return tensor


def permute_dims(input: object, axes: object) -> Tensor:
    """The array API's spelling of `permute`."""

    tensor = _atleast_tensor(input)
    order = [_operator.index(axis) for axis in axes]
    return tensor.permute(order)


def matrix_transpose(input: object) -> Tensor:
    """The last two axes swapped, leaving any batch axes alone.

    What `transpose(-2, -1)` says, under the array API's name for it. A tensor
    with fewer than two axes has no matrix to transpose and is refused rather
    than returned unchanged.
    """

    tensor = _atleast_tensor(input)
    if tensor.ndim() < 2:
        raise ValueError(
            f"matrix_transpose requires at least two dimensions, got {tensor.ndim()}"
        )
    return tensor.transpose(-2, -1)


def unstack(input: object, dim: int = 0) -> tuple[Tensor, ...]:
    """The array API's spelling of `unbind`: the slices along `dim`, as a tuple."""

    return unbind(input, dim)


def array_split(
    input: object, indices_or_sections: object, dim: int = 0
) -> tuple[Tensor, ...]:
    """NumPy's name for `tensor_split`: split into pieces that need not divide
    the axis evenly."""

    return tensor_split(input, indices_or_sections, dim)


def append(input: object, values: object, dim: int | None = None) -> Tensor:
    """`values` joined onto the end of `input` along `dim`.

    With no `dim` both are flattened first, which is what makes
    `append(x, 1.0)` mean what it looks like whatever shape `x` has. Every
    call copies -- there is no room at the end of a tensor to grow into, which
    is why this is a poor way to build one up element by element.
    """

    tensor = _atleast_tensor(input)
    extra = _atleast_tensor(values)
    if str(extra.dtype) != str(tensor.dtype):
        extra = extra.astype(str(tensor.dtype))
    if dim is None:
        return _C.functional.cat([tensor.reshape(-1), extra.reshape(-1)], 0)
    axis = _normalize_axis(dim, tensor.ndim(), "append")
    return _C.functional.cat([tensor, extra], axis)


def _positions_along(
    obj: object, length: int, name: str, past_the_end: bool = False
) -> list[int]:
    """`obj` as a list of positions along an axis of `length`.

    `past_the_end` allows `length` itself, which `insert` needs and `delete`
    must not have: inserting *before* the end is a real place to insert, and
    deleting the element after the last one is not a real place to delete.
    Negative positions always wrap against `length`, so `-1` is the last
    element either way.
    """

    if isinstance(obj, slice):
        return list(range(*obj.indices(length)))
    if isinstance(obj, Tensor):
        obj = obj.numpy()
    if isinstance(obj, _np.ndarray):
        if obj.dtype == bool:
            if obj.size != length:
                raise ValueError(
                    f"{name} was given a {obj.size}-element mask for an axis of {length}"
                )
            return [int(position) for position in _np.flatnonzero(obj)]
        obj = obj.reshape(-1).tolist()
    raw = (
        [_operator.index(obj)]
        if isinstance(obj, (int, _np.integer))
        else [_operator.index(value) for value in obj]
    )
    limit = length + 1 if past_the_end else length
    resolved = []
    for position in raw:
        wrapped = position + length if position < 0 else position
        if not 0 <= wrapped < limit:
            raise IndexError(
                f"{name} position {position} is out of bounds for an axis of {length}"
            )
        resolved.append(wrapped)
    return resolved


def delete(input: object, obj: object, dim: int | None = None) -> Tensor:
    """`input` without the positions `obj` names along `dim`.

    `obj` may be one position, several, a slice or a boolean mask. With no
    `dim` the tensor is flattened first. The result is a new tensor: nothing is
    removed in place, because the remaining elements have to move.
    """

    tensor = _atleast_tensor(input)
    if dim is None:
        tensor = tensor.reshape(-1)
        axis = 0
    else:
        axis = _normalize_axis(dim, tensor.ndim(), "delete")

    length = tensor.shape[axis]
    dropped = set(_positions_along(obj, length, "delete"))
    kept = [position for position in range(length) if position not in dropped]
    return _C.functional.index_select(
        tensor, axis, _index_tensor(_np.asarray(kept, dtype=_np.int64), tensor)
    )


def insert(
    input: object, obj: object, values: object, dim: int | None = None
) -> Tensor:
    """`values` placed *before* the positions `obj` names along `dim`.

    The positions refer to the original tensor, so `insert(x, [1, 1], [a, b])`
    puts both before the element that was at 1, in that order. Several
    positions are filled in position order, with each value following the
    position it was paired with.
    """

    tensor = _atleast_tensor(input)
    if dim is None:
        tensor = tensor.reshape(-1)
        axis = 0
    else:
        axis = _normalize_axis(dim, tensor.ndim(), "insert")

    length = tensor.shape[axis]
    at = _positions_along(obj, length, "insert", past_the_end=True)
    if not at:
        return tensor

    extra = _atleast_tensor(values)
    if str(extra.dtype) != str(tensor.dtype):
        extra = extra.astype(str(tensor.dtype))
    # One value per position, shaped like a slice of the axis.
    slice_shape = list(tensor.shape)
    slice_shape[axis] = 1
    if extra.ndim() == 0 or extra.numel() == 1:
        pieces = [broadcast_to(extra.reshape([1] * tensor.ndim()), slice_shape)] * len(
            at
        )
    else:
        spread = list(tensor.shape)
        spread[axis] = len(at)
        if list(extra.shape) != spread:
            extra = broadcast_to(
                extra.reshape([1] * (tensor.ndim() - extra.ndim()) + list(extra.shape)),
                spread,
            )
        pieces = list(unbind(extra, axis))
        pieces = [piece.unsqueeze(axis) for piece in pieces]

    order = sorted(range(len(at)), key=lambda index: at[index])
    parts: list[Tensor] = []
    previous = 0
    for index in order:
        position = at[index]
        if position > previous:
            parts.append(
                _C.functional.narrow(tensor, axis, previous, position - previous)
            )
        parts.append(pieces[index])
        previous = position
    if previous < length:
        parts.append(_C.functional.narrow(tensor, axis, previous, length - previous))
    return _C.functional.cat(parts, axis)


def resize(input: object, shape: object) -> Tensor:
    """`input`'s elements laid out in `shape`, repeating them to fill it.

    NumPy's `resize` rather than PyTorch's: the free function that returns a
    new tensor and *repeats* rather than zero-filling when the new shape is
    larger. An empty input has nothing to repeat, so it fills with zeros.
    """

    tensor = _atleast_tensor(input)
    sizes = _normalize_shape_argument(shape, "resize")
    total = _element_count(sizes)
    flat = tensor.reshape(-1)
    if flat.shape[0] == 0:
        return Tensor.zeros(list(sizes), dtype=str(tensor.dtype))
    if total == 0:
        return _C.functional.narrow(flat, 0, 0, 0).reshape(list(sizes))
    repeats = -(-total // flat.shape[0])
    filled = tile(flat, (repeats,)) if repeats > 1 else flat
    return _C.functional.narrow(filled, 0, 0, total).reshape(list(sizes))


def block(arrays: object) -> Tensor:
    """Assemble a tensor from nested lists of blocks.

    The innermost list is joined along the last axis, the one outside it along
    the second-to-last, and so on -- so a list of lists builds a matrix out of
    its blocks the way it is written on the page. Blocks are promoted to the
    depth of the nesting first, which is what lets a row vector sit next to a
    matrix.
    """

    def depth_of(item: object) -> int:
        if isinstance(item, (list, tuple)):
            if not item:
                raise ValueError("block was given an empty list")
            depths = {depth_of(entry) for entry in item}
            if len(depths) != 1:
                raise ValueError("block needs every list at a level to nest equally")
            return depths.pop() + 1
        return 0

    def build(item: object, depth: int) -> Tensor:
        if not isinstance(item, (list, tuple)):
            tensor = _atleast_tensor(item)
            missing = depth - tensor.ndim()
            if missing > 0:
                tensor = tensor.reshape([1] * missing + list(tensor.shape))
            return tensor
        parts = [build(entry, depth - 1) for entry in item]
        rank = max(part.ndim() for part in parts)
        parts = [
            (
                part.reshape([1] * (rank - part.ndim()) + list(part.shape))
                if part.ndim() < rank
                else part
            )
            for part in parts
        ]
        return _C.functional.cat(parts, -depth)

    nesting = depth_of(arrays)
    if nesting == 0:
        return _atleast_tensor(arrays)
    return build(arrays, nesting)


def cumulative_sum(
    input: object, dim: int | None = None, include_initial: bool = False
) -> Tensor:
    """The array API's `cumsum`, with the option of an initial zero.

    `include_initial` prepends the empty sum, so the result is one longer than
    the axis and `out[i]` is the total of everything *before* `i` -- which is
    the form an exclusive scan wants and the one `cumsum` cannot give.
    """

    tensor = _atleast_tensor(input)
    if dim is None:
        if tensor.ndim() > 1:
            raise ValueError(
                "cumulative_sum needs a dim for a tensor with more than one axis"
            )
        tensor = tensor.reshape(-1)
        axis = 0
    else:
        axis = _normalize_axis(dim, tensor.ndim(), "cumulative_sum")

    running = _C.functional.cumsum(tensor, axis)
    if not include_initial:
        return running
    lead = list(tensor.shape)
    lead[axis] = 1
    return _C.functional.cat(
        [Tensor.zeros(lead, dtype=str(running.dtype)), running], axis
    )


def broadcast_arrays(*inputs: object) -> tuple[Tensor, ...]:
    """NumPy's name for `broadcast_tensors`: every input at their common shape."""

    return broadcast_tensors(*inputs)


def geomspace(
    start: float,
    end: float,
    steps: int,
    dtype: object = None,
    device: object = None,
    requires_grad: bool = False,
) -> Tensor:
    """`steps` values spaced evenly on a *log* scale, from `start` to `end`.

    `logspace` takes the exponents; this takes the values themselves, which is
    the form a caller who knows the two ends wants. Both ends must be non-zero
    and of the same sign -- there is no geometric path from a positive number
    to a negative one, or through zero.

    The ends are set exactly rather than left to the exponential, which would
    otherwise land a rounding away from the numbers that were asked for.
    """

    if steps < 0:
        raise ValueError(
            f"geomspace requires a non-negative number of steps, got {steps}"
        )
    first, last = float(start), float(end)
    if first == 0.0 or last == 0.0:
        raise ValueError("geomspace cannot start or end at zero")
    if (first < 0.0) != (last < 0.0):
        raise ValueError(
            f"geomspace needs both ends to have the same sign, got {start} and {end}"
        )

    sign = -1.0 if first < 0.0 else 1.0
    values = Tensor.logspace(
        _math.log10(abs(first)),
        _math.log10(abs(last)),
        steps,
        base=10.0,
        dtype="float64",
        device=device,
    )
    if steps:
        # The exponential of a logarithm is not the number that went in, and
        # the two ends are the two the caller named. One step is the *start*
        # alone, so it must not be overwritten with the end.
        exact = _np.asarray(values.numpy(), dtype=_np.float64)
        exact[0] = abs(first)
        if steps > 1:
            exact[-1] = abs(last)
        values = Tensor.from_numpy(exact)
    result = values * sign
    if dtype is not None and str(result.dtype) != str(dtype):
        result = result.astype(str(dtype))
    if requires_grad:
        result.requires_grad_(True)
    return result


def tri(
    n: int,
    m: int | None = None,
    k: int = 0,
    dtype: object = None,
    device: object = None,
) -> Tensor:
    """An `n` by `m` matrix of ones on and below the `k`-th diagonal.

    The mask `tril` applies, as a tensor -- for multiplying by rather than
    selecting with.
    """

    columns = n if m is None else m
    ones = Tensor.ones(
        [n, columns], dtype=str(dtype) if dtype is not None else None, device=device
    )
    return _C.functional.tril(ones, k)


def indices(shape: object, sparse: bool = False):
    """The index grids of a tensor of `shape`.

    `indices((2, 3))[0]` holds each element's row and `[1]` its column, both in
    the tensor's own shape -- which is what turns a formula over positions into
    one tensor expression. `sparse` gives them with the other axes left at one,
    which broadcasts to the same thing at a fraction of the memory.
    """

    sizes = _normalize_shape_argument(shape, "indices")
    rank = len(sizes)
    grids = []
    for axis, size in enumerate(sizes):
        spread = [1] * rank
        spread[axis] = size
        line = Tensor.arange(0, size, 1, dtype="int64").reshape(spread)
        grids.append(line if sparse else broadcast_to(line, list(sizes)))
    if sparse:
        return tuple(grids)
    if not grids:
        return Tensor.zeros([0], dtype="int64")
    return _C.functional.stack(grids, 0)


def ix_(*sequences: object) -> tuple[Tensor, ...]:
    """Index arrays shaped so that together they select an open mesh.

    Each sequence gets its own axis and length one everywhere else, so
    `x[ix_(rows, cols)]` is the sub-matrix of those rows and columns rather
    than the elements they pair up into. Boolean sequences are turned into the
    positions they select, as NumPy does.
    """

    grids = []
    count = len(sequences)
    for axis, sequence in enumerate(sequences):
        tensor = _atleast_tensor(sequence).reshape(-1)
        if "bool" in str(tensor.dtype):
            tensor = _C.functional.nonzero(tensor).reshape(-1)
        elif "int" not in str(tensor.dtype):
            raise TypeError(
                f"ix_ takes integer or boolean sequences, got {tensor.dtype}"
            )
        spread = [1] * count
        spread[axis] = tensor.shape[0]
        grids.append(tensor.astype("int64").reshape(spread))
    return tuple(grids)


def fromfunction(function: object, shape: object, dtype: object = None) -> Tensor:
    """`function` called once with the index grids, not once per element.

    The function is handed one tensor per axis, each holding that axis'
    coordinate -- so it is written as an expression over whole tensors and
    evaluated in one pass. A function that cannot be written that way does not
    belong here.
    """

    sizes = _normalize_shape_argument(shape, "fromfunction")
    grids = indices(sizes, sparse=True)
    result = _atleast_tensor(function(*grids))
    if list(result.shape) != list(sizes):
        result = broadcast_to(result, list(sizes))
    if dtype is not None and str(result.dtype) != str(dtype):
        result = result.astype(str(dtype))
    return result


def _bit_axis(rank: int, dim: object, name: str) -> int:
    """The axis `dim` names, wrapped against `rank` and checked."""

    axis = _operator.index(dim)
    if axis < 0:
        axis += rank
    if not 0 <= axis < rank:
        raise IndexError(f"{name} got dim {dim} for a tensor of rank {rank}")
    return axis


#: The eight bit weights, most significant first. `packbits` multiplies by
#: these and `unpackbits` shifts by their exponents, so the two orders are
#: described in one place and cannot drift apart.
_BIT_WEIGHTS = (128, 64, 32, 16, 8, 4, 2, 1)


def packbits(input: object, dim: object = None, bitorder: str = "big") -> Tensor:
    """Pack groups of eight truth values along `dim` into one integer each.

    NumPy answers in `uint8`; this library has no unsigned byte, so the values
    come back as `int32` -- the same numbers in a wider box. `.numpy()` then
    `.astype(numpy.uint8)` recovers NumPy's array exactly, and that cast is the
    only place the difference shows.

    The axis is zero-padded up to a multiple of eight at its *end*, which is
    what makes `unpackbits` the inverse only when it is told the original
    length. `bitorder` decides whether the first element of each group is the
    high bit (`'big'`, the default) or the low one (`'little'`).
    """

    if bitorder not in ("big", "little"):
        raise ValueError(f"packbits takes bitorder 'big' or 'little', got {bitorder!r}")
    tensor = _atleast_tensor(input)
    dtype = str(tensor.dtype)
    if "int" not in dtype and "bool" not in dtype:
        raise TypeError(
            f"packbits takes boolean or integer tensors, got {tensor.dtype}"
        )

    bits = (tensor != 0).astype("int32")
    if dim is None:
        bits = bits.reshape([-1])
        axis = 0
    else:
        axis = _bit_axis(len(bits.shape), dim, "packbits")
        bits = _C.functional.movedim(bits, axis, -1)

    length = bits.shape[-1]
    remainder = length % 8
    if remainder:
        padding = list(bits.shape)
        padding[-1] = 8 - remainder
        bits = _C.functional.cat(
            [
                bits,
                Tensor.zeros(padding, dtype="int32", device=_C.Device(bits.device)),
            ],
            -1,
        )

    grouped = bits.reshape(list(bits.shape[:-1]) + [bits.shape[-1] // 8, 8])
    order = _BIT_WEIGHTS if bitorder == "big" else _BIT_WEIGHTS[::-1]
    weights = _constant_like(_np.asarray(order), bits)
    # The reduction widens to int64; the values are bytes, so it comes back.
    packed = (grouped * weights).sum(dim=-1).astype("int32")
    if dim is None:
        return packed
    return _C.functional.movedim(packed, -1, axis)


def unpackbits(
    input: object,
    dim: object = None,
    count: object = None,
    bitorder: str = "big",
) -> Tensor:
    """Expand each element along `dim` into its eight bits.

    The inverse of `packbits`, and its input is what `packbits` produced: an
    `int32` tensor of byte values rather than NumPy's `uint8`. A value outside
    `0..255` is refused rather than truncated -- there is no eight-bit answer
    for it, and quietly giving the low byte would make the round trip lie.

    `count` cuts the result to length: a non-negative count keeps that many
    bits, a negative one trims that many from the end, which is how the padding
    `packbits` added is undone.
    """

    if bitorder not in ("big", "little"):
        raise ValueError(
            f"unpackbits takes bitorder 'big' or 'little', got {bitorder!r}"
        )
    tensor = _atleast_tensor(input)
    dtype = str(tensor.dtype)
    if "int" not in dtype and "bool" not in dtype:
        raise TypeError(
            f"unpackbits takes boolean or integer tensors, got {tensor.dtype}"
        )
    bytes_ = tensor.astype("int32")
    if bytes_.numel():
        low, high = int(bytes_.min().item()), int(bytes_.max().item())
        if low < 0 or high > 255:
            raise ValueError(
                f"unpackbits takes byte values in 0..255, got {low} to {high}"
            )

    if dim is None:
        bytes_ = bytes_.reshape([-1])
        axis = 0
    else:
        axis = _bit_axis(len(bytes_.shape), dim, "unpackbits")
        bytes_ = _C.functional.movedim(bytes_, axis, -1)

    shifts = [7, 6, 5, 4, 3, 2, 1, 0] if bitorder == "big" else [0, 1, 2, 3, 4, 5, 6, 7]
    offsets = _constant_like(_np.asarray(shifts), bytes_)
    one = _constant_like(_np.asarray([1]), bytes_)
    spread = bytes_.reshape(list(bytes_.shape) + [1])
    planes = _C.functional.bitwise_and(
        _C.functional.bitwise_right_shift(spread, offsets), one
    )
    bits = planes.reshape(list(bytes_.shape[:-1]) + [bytes_.shape[-1] * 8])

    if count is not None:
        wanted = _operator.index(count)
        length = bits.shape[-1]
        keep = length + wanted if wanted < 0 else wanted
        if keep < 0:
            raise ValueError(
                f"unpackbits cannot trim {-wanted} bits from an axis of {length}"
            )
        if keep < length:
            bits = _C.functional.narrow(bits, -1, 0, keep)
        elif keep > length:
            # NumPy pads rather than refuses, and the zeros are the bits a
            # longer byte string would have held.
            padding = list(bits.shape)
            padding[-1] = keep - length
            bits = _C.functional.cat(
                [
                    bits,
                    Tensor.zeros(padding, dtype="int32", device=_C.Device(bits.device)),
                ],
                -1,
            )

    if dim is None:
        return bits
    return _C.functional.movedim(bits, -1, axis)
