# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""`affine_grid`'s entries, and its gradient in `theta`, are rounded once.

A grid coordinate is a whole number over the length, `(2i + 1 - L) / L`, or
over `L - 1` with the corners aligned. The grid was the matrix product of
`theta` with those coordinates rounded to the dtype, so each entry carried the
coordinates' roundings into its products: where the terms cancel, entries were
many ulps from the exact value. Each entry is now the float nearest `theta`
applied to the exact coordinates, and so is each entry of the gradient in
`theta`, a sum over the grid points of the gradient times a coordinate. An
integer `theta` widens, where it took the grid's coordinates as integers and
returned zeros.
"""

from __future__ import annotations

import math
from fractions import Fraction

import numpy as np
import pytest

import minitensor as mt
from minitensor import functional as F


def _float32(value: Fraction) -> float:
    """The float32 nearest `value`, ties to even."""

    if value == 0:
        return 0.0
    sign = -1.0 if value < 0 else 1.0
    magnitude = abs(value)
    exponent = magnitude.numerator.bit_length() - magnitude.denominator.bit_length()
    if Fraction(2) ** exponent > magnitude:
        exponent -= 1
    unit = Fraction(2) ** max(exponent - 23, -149)
    rounded = round(magnitude / unit) * unit
    return sign * (math.inf if rounded >= Fraction(2) ** 128 else float(rounded))


def _rounded(value: Fraction, dtype: str) -> float:
    return _float32(value) if dtype == "float32" else float(value)


def _coordinates(length: int, align_corners: bool) -> list:
    if align_corners:
        if length == 1:
            return [Fraction(0)]
        return [Fraction(2 * i + 1 - length, length - 1) for i in range(length)]
    return [Fraction(2 * i + 1 - length, length) for i in range(length)]


def _points(spatial, align_corners):
    """Each grid point's exact coordinates, `x` first, in the grid's order."""

    axes = [_coordinates(length, align_corners) for length in spatial]
    return [tuple(reversed(point)) for point in np.ndindex(*spatial)], axes


def _exact_grid(theta, spatial, align_corners, dtype):
    indices, axes = _points(spatial, align_corners)
    rank = len(spatial)
    out = []
    for matrix in theta:
        rows = [[Fraction(float(v)) for v in row] for row in matrix]
        for index in indices:
            c = [axes[rank - 1 - k][index[k]] for k in range(rank)]
            out.append(
                [
                    _rounded(sum(row[k] * c[k] for k in range(rank)) + row[rank], dtype)
                    for row in rows
                ]
            )
    return out


def _exact_gradient(grad, spatial, align_corners, dtype):
    indices, axes = _points(spatial, align_corners)
    rank = len(spatial)
    batch = grad.shape[0]
    flat = grad.reshape(batch, -1, rank)
    out = np.zeros((batch, rank, rank + 1))
    for b in range(batch):
        for i in range(rank):
            column = [Fraction(float(v)) for v in flat[b, :, i]]
            for k in range(rank + 1):
                if k == rank:
                    total = sum(column)
                else:
                    total = sum(
                        g * axes[rank - 1 - k][index[k]]
                        for g, index in zip(column, indices)
                    )
                out[b, i, k] = _rounded(total, dtype)
    return out


def _draw(rng, shape, dtype):
    values = rng.standard_normal(shape) * 10.0 ** rng.uniform(-3, 3, shape)
    return values.astype(dtype)


SHAPES = [(2, 1, 5, 7), (1, 1, 1, 6), (1, 2, 3, 4, 5), (2, 1, 1, 1, 2)]


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("align_corners", [False, True])
@pytest.mark.parametrize("size", SHAPES)
def test_every_entry_is_rounded_once(dtype, align_corners, size):
    rank = len(size) - 2
    rng = np.random.default_rng(1093 + rank + 3 * align_corners + size[-1])
    theta = _draw(rng, (size[0], rank, rank + 1), dtype)
    # A translation that cancels a coordinate's term at some points.
    theta[0, 0, rank] = -theta[0, 0, 0] * float(
        _coordinates(size[-1], align_corners)[-1]
    )
    grid = F.affine_grid(mt.Tensor(theta, dtype=dtype), size, align_corners)
    assert str(grid.dtype) == dtype
    assert tuple(grid.shape) == (size[0], *size[2:], rank)
    got = grid.numpy().reshape(-1, rank).tolist()
    assert got == _exact_grid(theta, size[2:], align_corners, dtype)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("align_corners", [False, True])
@pytest.mark.parametrize("size", SHAPES)
def test_every_gradient_entry_is_rounded_once(dtype, align_corners, size):
    rank = len(size) - 2
    rng = np.random.default_rng(1097 + rank + 3 * align_corners + size[-1])
    theta = mt.Tensor(
        _draw(rng, (size[0], rank, rank + 1), dtype), dtype=dtype, requires_grad=True
    )
    grid = F.affine_grid(theta, size, align_corners)
    weights = _draw(rng, tuple(grid.shape), dtype)
    (grid * mt.Tensor(weights, dtype=dtype)).sum().backward()
    want = _exact_gradient(weights, size[2:], align_corners, dtype)
    assert theta.grad.numpy().tolist() == want.tolist()


def test_cancelling_terms_are_exact():
    # The coordinate 1/3 at the second of four aligned points times 3, less
    # one: exactly zero, where the rounded coordinate left an ulp.
    theta = mt.Tensor([[[3.0, 0.0, -1.0], [0.0, 0.0, 0.0]]], dtype="float64")
    grid = F.affine_grid(theta, (1, 1, 1, 4), True).numpy()[0, 0, :, 0]
    assert grid.tolist() == [-4.0, -2.0, 0.0, 2.0]
    theta = mt.Tensor([[[0.1, 0.0, -0.1 * 0.6], [0.0, 0.0, 0.0]]], dtype="float64")
    grid = F.affine_grid(theta, (1, 1, 1, 5), False).numpy()[0, 0, :, 0]
    want = [
        float(Fraction(0.1) * c + Fraction(-0.1 * 0.6)) for c in _coordinates(5, False)
    ]
    assert grid.tolist() == want


def test_an_integer_theta_widens():
    theta = mt.Tensor(np.array([[[1, 0, 0], [0, 1, 0]]], dtype=np.int64))
    grid = F.affine_grid(theta, (1, 1, 2, 2))
    assert str(grid.dtype) == "float64"
    assert grid.tolist() == [[[[-0.5, -0.5], [0.5, -0.5]], [[-0.5, 0.5], [0.5, 0.5]]]]
    narrow = F.affine_grid(
        mt.Tensor(np.array([[[1, 0, 0], [0, 1, 0]]], dtype=np.int32)), (1, 1, 1, 2)
    )
    assert str(narrow.dtype) == "float32"
    with pytest.raises(ValueError, match="floating point"):
        F.affine_grid(mt.Tensor(np.zeros((1, 2, 3), dtype=bool)), (1, 1, 2, 2))


def test_infinities_are_what_products_make():
    inf, nan = math.inf, math.nan
    # An infinity times the middle coordinate of three, which is zero, is NaN;
    # times the others, an infinity of their sign.
    theta = mt.Tensor([[[inf, 0.0, 0.0], [0.0, 0.0, nan]]], dtype="float64")
    grid = F.affine_grid(theta, (1, 1, 1, 3), False).numpy()[0, 0]
    assert grid[0, 0] == -inf and math.isnan(grid[1, 0]) and grid[2, 0] == inf
    assert all(math.isnan(v) for v in grid[:, 1])
