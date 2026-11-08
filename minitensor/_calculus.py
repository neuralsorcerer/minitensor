# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""`gradient`: the numerical derivative of a sampled function.

Not the autograd gradient -- that is `backward`. This is the second-order
central difference over data you already have, for the case where the
function was measured rather than written down.

Each derivative is the float nearest the exact value of its stencil. Evenly
spaced, the stencil inside is `(f[i+1] - f[i-1]) / 2h`, which never reads the
sample it is centred on, so a NaN there stays where it was put:
`gradient([1, nan, 3, 4])` is `[nan, 1, nan, 1]`. Coordinates whose gaps are
all one exact step are that step, and the same answer to the last bit; any
other coordinates take the general second-order stencil of the two gaps
either side, whose middle weight is not zero and does read the centre.
"""

from __future__ import annotations

import operator as _operator

import numpy as _np

from . import _core as _C
from ._shape import _atleast_tensor, _normalize_axis
from .autograd import Function as _Function

Tensor = _C.Tensor
_F = _C.functional


def _is_step(spacing: object) -> bool:
    """Whether `spacing` is a single step rather than a coordinate vector."""

    return isinstance(spacing, (int, float)) and not isinstance(spacing, bool)


def _checked_step(spacing: object) -> float:
    """A scalar step, rejected if it is zero. Divided by rather than built into
    a coordinate vector, so the check has to live here as well."""

    step = float(spacing)
    if step == 0.0:
        raise ValueError("gradient requires a non-zero spacing")
    return step


def _coordinates(spacing: object, length: int, dtype: str, name: str) -> Tensor:
    """The sample positions along one axis, from either a step or a vector."""

    if _is_step(spacing):
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


def _require_samples(length: int, edge_order: int) -> None:
    """Refuse an axis too short for the stencils `edge_order` asks for."""

    if length < edge_order + 1:
        raise ValueError(
            f"gradient needs at least {edge_order + 1} samples along an axis for "
            f"edge_order={edge_order}, got {length}"
        )


def _broadcastable(vector: Tensor, axis: int, rank: int) -> Tensor:
    """A 1-D vector reshaped to line up with `axis` of a rank-`rank` tensor."""

    shape = [1] * rank
    shape[axis] = vector.shape[0]
    return vector.reshape(shape)


def _slice(tensor: Tensor, axis: int, start: int, length: int) -> Tensor:
    return _F.narrow(tensor, axis, start, length)


class _Gradient(_Function):
    """The finite differences along an axis, each rounded once, with the
    gradient of the exact stencils for the samples and the coordinates."""

    @staticmethod
    def forward(ctx, values, positions, step, axis, edge_order):
        result = _F._gradient(values, positions, step, axis, edge_order)
        ctx.save_for_backward(values)
        ctx.positions, ctx.step, ctx.axis = positions, step, axis
        ctx.edge_order, ctx.result = edge_order, result
        return result

    @staticmethod
    def backward(ctx, grad_output):
        (values,) = ctx.saved_tensors
        axis, order = ctx.axis, ctx.edge_order
        rank, count = values.ndim(), values.shape[axis]
        dtype = str(values.dtype)
        grad = grad_output.astype("float64")
        if ctx.positions is None:
            # The step at the samples' width, as the stencil read it.
            step = float(_np.dtype(dtype).type(ctx.step))
            gaps = Tensor.full([count - 1], step, dtype="float64")
        else:
            places = ctx.positions.detach().astype("float64")
            gaps = _slice(places, 0, 1, count - 1) - _slice(places, 0, 0, count - 1)

        def along(vector):
            return _broadcastable(vector, axis, rank)

        def place(piece, offset):
            # `piece` along the axis at `offset`, zeros either side.
            parts, length = [], piece.shape[axis]
            for size, part in (
                (offset, None),
                (length, piece),
                (count - offset - length, None),
            ):
                if part is not None:
                    parts.append(part)
                elif size:
                    shape = list(piece.shape)
                    shape[axis] = size
                    parts.append(Tensor.full(shape, 0.0, dtype="float64"))
            return _F.cat(parts, axis) if len(parts) > 1 else parts[0]

        def gap(index):
            return float(_slice(gaps, 0, index, 1).item())

        first, last = _slice(grad, axis, 0, 1), _slice(grad, axis, count - 1, 1)
        grad_values = None
        pieces = []
        if count > 2:
            behind, ahead = _slice(gaps, 0, 0, count - 2), _slice(gaps, 0, 1, count - 2)
            span = behind + ahead
            inside = _slice(grad, axis, 1, count - 2)
            # Half the intervals either side of a sample, weighted as the
            # second-order stencil weights them.
            pieces += [
                place(inside * along(-ahead / (behind * span)), 0),
                place(inside * along((ahead - behind) / (behind * ahead)), 1),
                place(inside * along(behind / (ahead * span)), 2),
            ]
        if order == 1:
            start, end = gap(0), gap(count - 2)
            pieces += [
                place(first * (-1.0 / start), 0),
                place(first * (1.0 / start), 1),
                place(last * (-1.0 / end), count - 2),
                place(last * (1.0 / end), count - 1),
            ]
        else:
            one, two = gap(0), gap(1)
            total = one + two
            pieces += [
                place(first * (-(2.0 * one + two) / (one * total)), 0),
                place(first * (total / (one * two)), 1),
                place(first * (-one / (two * total)), 2),
            ]
            one, two = gap(count - 3), gap(count - 2)
            total = one + two
            pieces += [
                place(last * (two / (one * total)), count - 3),
                place(last * (-total / (one * two)), count - 2),
                place(last * ((2.0 * two + one) / (two * total)), count - 1),
            ]
        grad_values = pieces[0]
        for piece in pieces[1:]:
            grad_values = grad_values + piece

        grad_positions = None
        if ctx.positions is not None and ctx.needs_input_grad[1]:
            grad_positions = _positions_gradient(
                values.detach().astype("float64"),
                ctx.result.detach().astype("float64"),
                grad,
                gaps,
                axis,
                order,
            ).astype(str(ctx.positions.dtype))
        return grad_values.astype(dtype), grad_positions, None, None, None


def _positions_gradient(
    f: Tensor, out: Tensor, grad: Tensor, gaps: Tensor, axis: int, order: int
) -> Tensor:
    """How the derivatives move with each coordinate, by the gaps either side.

    A stencil is `N / D` in its gaps, so a gap moves it by
    `(dN - out · dD) / D`; a coordinate moves the gap before it one way and
    the gap after it the other.
    """

    rank, count = f.ndim(), f.shape[axis]
    others = [index for index in range(rank) if index != axis]

    def along(vector):
        return _broadcastable(vector, axis, rank)

    def total(tensor):
        return (_F.sum(tensor, others) if others else tensor).reshape([-1])

    def edge(index):
        return _slice(f, axis, index, 1)

    moves = [Tensor.full([count], 0.0, dtype="float64")]

    def add(values, offset):
        size = values.shape[0]
        parts = [
            Tensor.full([offset], 0.0, dtype="float64"),
            values,
            Tensor.full([count - offset - size], 0.0, dtype="float64"),
        ]
        moves.append(_F.cat([part for part in parts if part.shape[0]], 0))

    def gap_moves(d_one, d_two, offset):
        # A stencil of the gaps at `offset` and `offset + 1` moved by its
        # three coordinates.
        add(total(-d_one), offset)
        add(total(d_one - d_two), offset + 1)
        add(total(d_two), offset + 2)

    if count > 2:
        one = along(_slice(gaps, 0, 0, count - 2))
        two = along(_slice(gaps, 0, 1, count - 2))
        before = _slice(f, axis, 0, count - 2)
        centre = _slice(f, axis, 1, count - 2)
        after = _slice(f, axis, 2, count - 2)
        value = _slice(out, axis, 1, count - 2)
        weight = _slice(grad, axis, 1, count - 2) / (one * two * (one + two))
        d_one = (
            2.0 * one * (after - centre) - value * (2.0 * one * two + two * two)
        ) * weight
        d_two = (
            2.0 * two * (centre - before) - value * (one * one + 2.0 * one * two)
        ) * weight
        gap_moves(d_one, d_two, 0)
    first, last = _slice(grad, axis, 0, 1), _slice(grad, axis, count - 1, 1)
    start_value, end_value = _slice(out, axis, 0, 1), _slice(out, axis, count - 1, 1)
    if order == 1:
        for g, value, offset in ((first, start_value, 0), (last, end_value, count - 2)):
            width = float(_slice(gaps, 0, offset, 1).item())
            move = total(g * value / width)
            add(move, offset)
            add(-move, offset + 1)
    else:
        one, two = (float(_slice(gaps, 0, index, 1).item()) for index in (0, 1))
        span = one + two
        f0, f1, f2 = edge(0), edge(1), edge(2)
        weight = first / (one * two * span)
        d_one = (2.0 * span * f1 - 2.0 * two * f0 - 2.0 * one * f2) - start_value * (
            two * (span + one)
        )
        d_two = (2.0 * span * f1 - 2.0 * span * f0) - start_value * (one * (span + two))
        gap_moves(d_one * weight, d_two * weight, 0)
        one, two = (
            float(_slice(gaps, 0, index, 1).item()) for index in (count - 3, count - 2)
        )
        span = one + two
        f0, f1, f2 = edge(count - 3), edge(count - 2), edge(count - 1)
        weight = last / (one * two * span)
        d_one = (2.0 * span * f2 - 2.0 * span * f1) - end_value * (two * (span + one))
        d_two = (2.0 * two * f0 - 2.0 * span * f1 + 2.0 * one * f2) - end_value * (
            one * (span + two)
        )
        gap_moves(d_one * weight, d_two * weight, count - 3)
    result = moves[0]
    for move in moves[1:]:
        result = result + move
    return result


def _along(
    values: Tensor, spacing: object, axis: int, edge_order: int, dtype: str
) -> Tensor:
    """The derivative along one axis, `spacing` a step or one coordinate a
    sample; coordinates one exact step apart are that step."""

    length = values.shape[axis]
    _require_samples(length, edge_order)
    if _is_step(spacing):
        return _Gradient.apply(values, None, _checked_step(spacing), axis, edge_order)
    positions = _coordinates(spacing, length, dtype, "gradient")
    return _Gradient.apply(values, positions, 0.0, axis, edge_order)


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

    Returns one tensor when a single axis is asked for and a tuple otherwise.
    This is the derivative of *data*, not
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
        _along(values, step, axis, order, dtype) for axis, step in zip(axes, spacings)
    )
    return results[0] if len(results) == 1 else results
