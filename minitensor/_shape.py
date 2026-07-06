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
    according to NumPy/PyTorch broadcasting rules. The returned tensors are
    views when the backend can represent the expansion without copying. Valid
    zero-sized broadcasts that cannot be represented as views return empty
    tensors preserving dtype, device, and ``requires_grad`` metadata.
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
    tensor is the original tensor when it already has the requested shape, an
    expanded view when possible, or a metadata-preserving empty tensor for the
    valid zero-sized broadcasts that cannot be represented as views.
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
    has no addressable elements. NumPy treats this as a valid broadcast, so the
    Python helper returns a correctly shaped empty tensor for that edge case.
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

    This NumPy-compatible helper accepts tensor-like 1-D inputs and returns
    broadcasted coordinate grids. ``indexing="ij"`` preserves input axis
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
