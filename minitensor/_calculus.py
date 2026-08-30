# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""`gradient`: the numerical derivative of a sampled function.

Not the autograd gradient -- that is `backward`. This is NumPy's `gradient`,
the second-order central difference over data you already have, for the case
where the function was measured rather than written down.

The interior formula and both edge formulas are the non-uniform ones, with
uniform spacing handled by building the coordinates it implies. One code path
rather than two, and the uniform case comes out bit-identical to the familiar
`(f[i+1] - f[i-1]) / (2h)` because that is what the general formula reduces to
when the two spacings are equal.
"""

from __future__ import annotations

import operator as _operator

from . import _core as _C
from ._shape import _atleast_tensor, _normalize_axis

Tensor = _C.Tensor
_F = _C.functional


def _coordinates(spacing: object, length: int, dtype: str, name: str) -> Tensor:
    """The sample positions along one axis, from either a step or a vector."""

    if isinstance(spacing, (int, float)) and not isinstance(spacing, bool):
        step = float(spacing)
        if step == 0.0:
            raise ValueError(f"{name} requires a non-zero spacing")
        return _C.Tensor.arange(0, length, 1, dtype=dtype) * step

    positions = _atleast_tensor(spacing)
    if positions.ndim() != 1 or positions.shape[0] != length:
        raise ValueError(
            f"{name} requires a scalar step or one coordinate per sample "
            f"({length}), got {list(positions.shape)}"
        )
    return positions.astype(dtype)


def _broadcastable(vector: Tensor, axis: int, rank: int) -> Tensor:
    """A 1-D vector reshaped to line up with `axis` of a rank-`rank` tensor."""

    shape = [1] * rank
    shape[axis] = vector.shape[0]
    return vector.reshape(shape)


def _slice(tensor: Tensor, axis: int, start: int, length: int) -> Tensor:
    return _F.narrow(tensor, axis, start, length)


def _one_axis(values: Tensor, positions: Tensor, axis: int, edge_order: int) -> Tensor:
    """The derivative along one axis, from the samples and their coordinates."""

    length = values.shape[axis]
    rank = values.ndim()
    if length < edge_order + 1:
        raise ValueError(
            f"gradient needs at least {edge_order + 1} samples along an axis for "
            f"edge_order={edge_order}, got {length}"
        )

    # The two gaps around each interior point.
    behind = _slice(positions, 0, 1, length - 1) - _slice(positions, 0, 0, length - 1)
    interior_behind = _slice(behind, 0, 0, length - 2)
    interior_ahead = _slice(behind, 0, 1, length - 2)
    span = interior_behind + interior_ahead

    # NumPy's second-order non-uniform stencil. With equal gaps the outer
    # coefficients collapse to +-1/(2h) and the middle one to zero, which is
    # the familiar central difference.
    before = -interior_ahead / (interior_behind * span)
    middle = (interior_ahead - interior_behind) / (interior_behind * interior_ahead)
    after = interior_behind / (interior_ahead * span)

    parts = [
        _broadcastable(before, axis, rank) * _slice(values, axis, 0, length - 2),
        _broadcastable(middle, axis, rank) * _slice(values, axis, 1, length - 2),
        _broadcastable(after, axis, rank) * _slice(values, axis, 2, length - 2),
    ]
    interior = parts[0] + parts[1] + parts[2]

    if edge_order == 1:
        first = (
            _slice(values, axis, 1, 1) - _slice(values, axis, 0, 1)
        ) / _broadcastable(_slice(behind, 0, 0, 1), axis, rank)
        last = (
            _slice(values, axis, length - 1, 1) - _slice(values, axis, length - 2, 1)
        ) / _broadcastable(_slice(behind, 0, length - 2, 1), axis, rank)
    else:
        # The second-order one-sided stencils, again NumPy's.
        first_gap = _slice(behind, 0, 0, 1)
        second_gap = _slice(behind, 0, 1, 1)
        total = first_gap + second_gap
        first = (
            _broadcastable(
                -(2.0 * first_gap + second_gap) / (first_gap * total), axis, rank
            )
            * _slice(values, axis, 0, 1)
            + _broadcastable(total / (first_gap * second_gap), axis, rank)
            * _slice(values, axis, 1, 1)
            + _broadcastable(-first_gap / (second_gap * total), axis, rank)
            * _slice(values, axis, 2, 1)
        )

        last_gap = _slice(behind, 0, length - 2, 1)
        penultimate_gap = _slice(behind, 0, length - 3, 1)
        total = penultimate_gap + last_gap
        last = (
            _broadcastable(last_gap / (penultimate_gap * total), axis, rank)
            * _slice(values, axis, length - 3, 1)
            + _broadcastable(-total / (penultimate_gap * last_gap), axis, rank)
            * _slice(values, axis, length - 2, 1)
            + _broadcastable(
                (2.0 * last_gap + penultimate_gap) / (last_gap * total), axis, rank
            )
            * _slice(values, axis, length - 1, 1)
        )

    return _F.cat([first, interior, last], axis)


def gradient(
    input: object,
    spacing: object = 1.0,
    dim: object = None,
    edge_order: int = 1,
) -> Tensor | tuple[Tensor, ...]:
    """The numerical derivative of `input` along each axis in `dim`.

    Second-order accurate in the interior and `edge_order`-accurate at the two
    ends. `spacing` is a step, a coordinate vector, or one of either per axis;
    the coordinates need not be evenly spaced.

    Returns one tensor when a single axis is asked for and a tuple otherwise,
    which is what `numpy.gradient` does. This is the derivative of *data*, not
    of a computation -- for that, call `backward`.
    """

    values = _atleast_tensor(input)
    if values.ndim() == 0:
        raise ValueError("gradient requires a tensor with at least one dimension")
    if "float" not in str(values.dtype):
        values = values.astype("float64")
    dtype = str(values.dtype)

    order = _operator.index(edge_order)
    if order not in (1, 2):
        raise ValueError(f"gradient supports edge_order 1 or 2, got {edge_order}")

    single = isinstance(dim, int) and not isinstance(dim, bool)
    if dim is None:
        axes = list(range(values.ndim()))
    elif single:
        axes = [_normalize_axis(dim, values.ndim(), "gradient")]
    else:
        axes = [_normalize_axis(a, values.ndim(), "gradient") for a in dim]

    # A bare number or a single coordinate vector applies to every axis; a
    # sequence of them is one per axis, in the order `dim` names them.
    if isinstance(spacing, (int, float)) and not isinstance(spacing, bool):
        spacings = [spacing] * len(axes)
    elif (
        isinstance(spacing, (list, tuple))
        and len(spacing) == len(axes)
        and len(axes) != 1
    ):
        spacings = list(spacing)
    else:
        spacings = [spacing] * len(axes)

    results = tuple(
        _one_axis(
            values,
            _coordinates(step, values.shape[axis], dtype, "gradient"),
            axis,
            order,
        )
        for axis, step in zip(axes, spacings)
    )
    return results[0] if len(results) == 1 else results
