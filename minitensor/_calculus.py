# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""`gradient`: the numerical derivative of a sampled function.

Not the autograd gradient -- that is `backward`. This is NumPy's `gradient`,
the second-order central difference over data you already have, for the case
where the function was measured rather than written down.

The edge formulas are the non-uniform ones, with uniform spacing handled by
building the coordinates it implies: one code path, and it comes out
bit-identical to the familiar one because that is what the general formula
reduces to when the two spacings are equal.

The *interior* needs the second path anyway, and not for speed. Its general
stencil weights the middle sample by `(ahead - behind) / (behind * ahead)`,
which is exactly zero on a uniform grid -- and `0 * x` is zero for every `x`
the arithmetic has, except the two it does not: `0 * NaN` and `0 * inf` are
both NaN. So a single NaN in the data spread to its neighbours, which the
central difference never touches: `gradient([1, nan, 3, 4])` answered
`[nan, nan, nan, 1]` where the derivative at the second point is `(3 - 1) / 2`,
a number NumPy reports. NumPy separates the uniform case for the same reason,
and the two now agree on NaN and infinity as well as on values. Non-uniform
coordinates keep the general stencil, where the middle weight is not zero and
the contamination is real: NumPy gives NaN there too.
"""

from __future__ import annotations

import operator as _operator

from . import _core as _C
from ._shape import _atleast_tensor, _normalize_axis

Tensor = _C.Tensor
_F = _C.functional


def _is_step(spacing: object) -> bool:
    """Whether `spacing` is a single step rather than a coordinate vector."""

    return isinstance(spacing, (int, float)) and not isinstance(spacing, bool)


def _constant_step(positions: Tensor) -> float | None:
    """The single step a coordinate vector describes, if its gaps are equal.

    NumPy does this before it starts, with the comment that it "brings a
    consistent speedup", and it decides more than speed: the uniform stencil
    is the one that does not read the point it is centred on, so reducing to
    it is what keeps a missing sample from spreading. Without the check,
    `gradient(x, h)` and `gradient(x, h * arange(n))` -- the same grid written
    two ways -- answered differently at a NaN, and differed in the last bit
    everywhere else.
    """

    gaps = _slice(positions, 0, 1, positions.shape[0] - 1) - _slice(
        positions, 0, 0, positions.shape[0] - 1
    )
    first = _slice(gaps, 0, 0, 1)
    if not bool(_F.all(gaps == first).item()):
        return None
    return float(first.item())


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


def _broadcastable(vector: Tensor, axis: int, rank: int) -> Tensor:
    """A 1-D vector reshaped to line up with `axis` of a rank-`rank` tensor."""

    shape = [1] * rank
    shape[axis] = vector.shape[0]
    return vector.reshape(shape)


def _slice(tensor: Tensor, axis: int, start: int, length: int) -> Tensor:
    return _F.narrow(tensor, axis, start, length)


def _along(
    values: Tensor, spacing: object, axis: int, edge_order: int, dtype: str
) -> Tensor:
    """The derivative along one axis, by whichever of the two paths fits.

    A scalar spacing is a uniform grid by construction. A coordinate vector
    may describe one as well, and is reduced to its step when it does -- which
    is what NumPy does, and what makes the two spellings of one grid agree.
    """

    if _is_step(spacing):
        return _one_axis_uniform(values, _checked_step(spacing), axis, edge_order)

    positions = _coordinates(spacing, values.shape[axis], dtype, "gradient")
    step = _constant_step(positions)
    if step is not None:
        return _one_axis_uniform(values, step, axis, edge_order)
    return _one_axis(values, positions, axis, edge_order)


def _one_axis(values: Tensor, positions: Tensor, axis: int, edge_order: int) -> Tensor:
    """The derivative along one axis, from the samples and their coordinates.

    The general case: the gaps differ, so every weight is a vector derived
    from them. `_one_axis_uniform` is the other half, for the spacing given as
    a single step.
    """

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


def _one_axis_uniform(
    values: Tensor, step: float, axis: int, edge_order: int
) -> Tensor:
    """The derivative along one axis of an evenly spaced grid.

    Every weight the general stencil derives is a constant here, so none of
    them is built: no coordinate vector, no vector of gaps, and no five-term
    interior. What is left is the central difference and the two one-sided
    stencils, each divided by a Python float.

    The middle weight is the reason this is a separate path rather than a
    faster one. It is exactly zero on a uniform grid, and `0 * x` is zero for
    every `x` the arithmetic has except `NaN` and `inf`, where it is `NaN` --
    so multiplying it through spread a single missing sample to both of its
    neighbours, which a central difference never reads.
    """

    length = values.shape[axis]
    if length < edge_order + 1:
        raise ValueError(
            f"gradient needs at least {edge_order + 1} samples along an axis for "
            f"edge_order={edge_order}, got {length}"
        )

    interior = (
        _slice(values, axis, 2, length - 2) - _slice(values, axis, 0, length - 2)
    ) / (2.0 * step)

    if edge_order == 1:
        first = (_slice(values, axis, 1, 1) - _slice(values, axis, 0, 1)) / step
        last = (
            _slice(values, axis, length - 1, 1) - _slice(values, axis, length - 2, 1)
        ) / step
    else:
        # NumPy's second-order one-sided stencils with equal gaps: the general
        # weights collapse to -3/2, 2, -1/2 and their mirror. Each is divided
        # by the step *before* it multiplies, which is the order NumPy uses --
        # dividing the sum instead rounds differently, and the two answers
        # then differ in the last bit.
        first = (
            (-1.5 / step) * _slice(values, axis, 0, 1)
            + (2.0 / step) * _slice(values, axis, 1, 1)
            + (-0.5 / step) * _slice(values, axis, 2, 1)
        )
        last = (
            (0.5 / step) * _slice(values, axis, length - 3, 1)
            + (-2.0 / step) * _slice(values, axis, length - 2, 1)
            + (1.5 / step) * _slice(values, axis, length - 1, 1)
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
        _along(values, step, axis, order, dtype) for axis, step in zip(axes, spacings)
    )
    return results[0] if len(results) == 1 else results
