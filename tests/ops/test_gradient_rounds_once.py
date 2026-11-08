# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""`gradient` is the float nearest each exact stencil.

The differences, the weights, their products and the sums each rounded, and
of random samples about one derivative in four came out other than the float
nearest the exact value. Each is that float now, evenly spaced or at uneven
coordinates, inside and at both ends to either order, along any axis.
"""

from __future__ import annotations

import math
from fractions import Fraction

import numpy as np
import pytest

import minitensor as mt


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
    if dtype == "float32":
        return _float32(value)
    try:
        return float(value)
    except OverflowError:
        return math.inf if value > 0 else -math.inf


def _stencils(f, x, edge_order) -> list[Fraction]:
    """The exact derivatives of the samples `f` at the positions `x`."""

    n = len(f)
    out = []
    for i in range(n):
        if 0 < i < n - 1:
            one, two = x[i] - x[i - 1], x[i + 1] - x[i]
            numerator = one * one * (f[i + 1] - f[i]) + two * two * (f[i] - f[i - 1])
            out.append(numerator / (one * two * (one + two)))
        elif edge_order == 1:
            a, b = (0, 1) if i == 0 else (n - 2, n - 1)
            out.append((f[b] - f[a]) / (x[b] - x[a]))
        elif i == 0:
            one, two = x[1] - x[0], x[2] - x[1]
            t = one + two
            numerator = (
                t * t * f[1] - (2 * one * two + two * two) * f[0] - one * one * f[2]
            )
            out.append(numerator / (one * two * t))
        else:
            one, two = x[n - 2] - x[n - 3], x[n - 1] - x[n - 2]
            t = one + two
            numerator = (
                two * two * f[n - 3]
                - t * t * f[n - 2]
                + (2 * one * two + one * one) * f[n - 1]
            )
            out.append(numerator / (one * two * t))
    return out


def _want(values, dtype, edge_order, step=None, coordinates=None, axis=-1):
    """Each slice's exact derivatives along `axis`, rounded once."""

    values = np.moveaxis(np.asarray(values), axis, -1)
    n = values.shape[-1]
    if coordinates is None:
        h = Fraction(float(np.dtype(dtype).type(step)))
        places = [h * k for k in range(n)]
    else:
        places = [Fraction(float(v)) for v in coordinates]
    out = np.empty(values.shape, dtype=np.float64)
    for index in np.ndindex(*values.shape[:-1]):
        f = [Fraction(float(v)) for v in values[index]]
        out[index] = [_rounded(v, dtype) for v in _stencils(f, places, edge_order)]
    return np.moveaxis(out, -1, axis).tolist()


def _draw(rng, shape, dtype: str):
    return (rng.normal(size=shape) * 10.0 ** rng.uniform(-3, 3, shape)).astype(dtype)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("edge_order", [1, 2])
def test_evenly_spaced(dtype, edge_order):
    rng = np.random.default_rng(853)
    for trial in range(100):
        n = int(rng.integers(edge_order + 1, 30))
        f = _draw(rng, n, dtype)
        step = float(rng.uniform(0.01, 3.0))
        got = mt.gradient(mt.Tensor(f, dtype=dtype), step, edge_order=edge_order)
        assert str(got.dtype) == dtype
        assert got.tolist() == _want(f, dtype, edge_order, step), (trial, n)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("edge_order", [1, 2])
def test_uneven_coordinates(dtype, edge_order):
    rng = np.random.default_rng(857)
    for trial in range(100):
        n = int(rng.integers(edge_order + 1, 30))
        f = _draw(rng, n, dtype)
        x = np.cumsum(rng.uniform(0.01, 2.0, n) * 10.0 ** rng.uniform(-2, 2)).astype(
            dtype
        )
        if np.any(np.diff(x) <= 0):
            continue
        got = mt.gradient(
            mt.Tensor(f, dtype=dtype), mt.Tensor(x, dtype=dtype), edge_order=edge_order
        )
        assert got.tolist() == _want(f, dtype, edge_order, coordinates=x), (trial, n)


def test_even_coordinates_are_their_step():
    rng = np.random.default_rng(859)
    f = _draw(rng, 12, "float64")
    for step in (0.5, 3.0, 2.0**-30):
        coordinates = step * np.arange(12, dtype=np.float64)
        for order in (1, 2):
            by_step = mt.gradient(mt.Tensor(f), step, edge_order=order).tolist()
            by_place = mt.gradient(
                mt.Tensor(f), mt.Tensor(coordinates), edge_order=order
            )
            assert by_place.tolist() == by_step
    # Steps of 0.1 are not one exact step apart, so they are the exact
    # stencils of the coordinates as given.
    coordinates = 0.1 * np.arange(12, dtype=np.float64)
    got = mt.gradient(mt.Tensor(f), mt.Tensor(coordinates), edge_order=2).tolist()
    assert got == _want(f, "float64", 2, coordinates=coordinates)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_every_axis(dtype):
    rng = np.random.default_rng(863)
    values = _draw(rng, (4, 5, 6), dtype)
    tensor = mt.Tensor(values, dtype=dtype)
    got = mt.gradient(tensor, 0.5, edge_order=2)
    for axis in range(3):
        assert got[axis].tolist() == _want(values, dtype, 2, 0.5, axis=axis), axis
    x = np.sort(_draw(rng, 5, dtype))
    got = mt.gradient(tensor, mt.Tensor(x, dtype=dtype), dim=1)
    assert got.tolist() == _want(values, dtype, 1, coordinates=x, axis=1)


def test_cancellation_and_range():
    # Samples far apart in size, whose plain differences lose the small.
    f = [1e17, 1.0, -1e17, 3.0, 2.0**-40, 7.0]
    got = mt.gradient(mt.tensor(f, dtype="float64"), 0.3, edge_order=2).tolist()
    assert got == _want(f, "float64", 2, 0.3)
    x = [0.0, 1e-8, 1.0, 1.0 + 2.0**-40, 5.0, 1e10]
    got = mt.gradient(mt.tensor(f, dtype="float64"), mt.tensor(x, dtype="float64"))
    assert got.tolist() == _want(f, "float64", 1, coordinates=x)
    # Coordinates whose products with the samples pass the largest float.
    big = 1e300
    x = [big, big * (1 + 2.0**-40), big * (1 + 2.0**-39), big * (1 + 2.0**-38)]
    f = [1e10, -3e10, 7.0, 2e10]
    got = mt.gradient(
        mt.tensor(f, dtype="float64"), mt.tensor(x, dtype="float64"), edge_order=2
    )
    assert got.tolist() == _want(f, "float64", 2, coordinates=x)


def test_a_missing_sample_on_an_even_grid():
    got = mt.gradient(mt.tensor([1.0, math.nan, 3.0, 4.0], dtype="float64")).tolist()
    assert math.isnan(got[0]) and got[1] == 1.0 and math.isnan(got[2]) and got[3] == 1.0
    got = mt.gradient(mt.tensor([1.0, math.inf, 3.0, 4.0], dtype="float64")).tolist()
    assert got[0] == math.inf and got[1] == 1.0 and got[2] == -math.inf


def test_gradients_match_central_differences():
    rng = np.random.default_rng(877)
    base_f = rng.normal(size=(3, 7))
    base_x = np.cumsum(rng.uniform(0.5, 1.5, 7))
    weights = rng.normal(size=(3, 7))
    for order in (1, 2):
        for spacing in ("step", "coordinates"):

            def total(f, x):
                argument = 0.7 if spacing == "step" else mt.Tensor(x)
                got = mt.gradient(mt.Tensor(f), argument, dim=1, edge_order=order)
                return float((got.numpy() * weights).sum())

            f = mt.Tensor(base_f.copy(), requires_grad=True)
            x = mt.Tensor(base_x.copy(), requires_grad=True)
            argument = 0.7 if spacing == "step" else x
            got = mt.gradient(f, argument, dim=1, edge_order=order)
            (got * mt.Tensor(weights)).sum().backward()
            eps = 1e-6
            checks = [(f, base_f, 0)] + ([(x, base_x, 1)] if spacing != "step" else [])
            for tensor, base, which in checks:
                for index in np.ndindex(*base.shape):
                    up = [base_f.copy(), base_x.copy()]
                    down = [base_f.copy(), base_x.copy()]
                    up[which][index] += eps
                    down[which][index] -= eps
                    numeric = (total(*up) - total(*down)) / (2 * eps)
                    assert np.asarray(tensor.grad.tolist())[index] == pytest.approx(
                        numeric, rel=1e-5, abs=1e-7
                    ), (order, spacing, which, index)
