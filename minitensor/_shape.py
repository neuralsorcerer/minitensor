# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""Small Python shape helpers for MiniTensor's public API."""

from __future__ import annotations

import builtins as _builtins
import operator as _operator

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
    """Return the shape produced by NumPy/PyTorch-style broadcasting.

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
    result_reversed: list[int] = []
    max_rank = _builtins.max(len(shape) for shape in normalized_shapes)

    for axis_from_end in range(max_rank):
        resolved = 1
        for shape in normalized_shapes:
            if axis_from_end >= len(shape):
                continue
            dim = shape[-1 - axis_from_end]
            if dim == 1 or dim == resolved:
                continue
            if resolved == 1:
                resolved = dim
                continue
            raise ValueError(
                "shapes cannot be broadcast together: "
                + ", ".join(str(shape) for shape in normalized_shapes)
            )
        result_reversed.append(resolved)

    return tuple(reversed(result_reversed))


def can_broadcast(*shapes: object) -> bool:
    """Return ``True`` when shapes can broadcast without creating tensors."""

    try:
        broadcast_shapes(*shapes)
    except (TypeError, ValueError):
        return False
    return True


def _atleast_tensor(input: object) -> Tensor:
    """Convert an input to a Tensor while preserving existing Tensor objects."""

    if isinstance(input, Tensor):
        return input
    return as_tensor(input)


def _return_atleast_result(results: list[Tensor]) -> Tensor | tuple[Tensor, ...]:
    if len(results) == 1:
        return results[0]
    return tuple(results)


def atleast_1d(*inputs: object) -> Tensor | tuple[Tensor, ...]:
    """Convert inputs to tensors with at least one dimension.

    Scalar inputs are reshaped to ``(1,)``. Inputs that are already at least
    one-dimensional are returned as tensors without adding dimensions.
    Multiple inputs return a tuple of tensors, matching NumPy's convention.
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
