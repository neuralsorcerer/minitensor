# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""`prod` and `cumprod` are the float nearest the exact product.

A float product was multiplied out a factor at a time, rounding at each: of
two hundred float64 products of a few factors near one, more than half were
an ulp or more out, a running product drifted further at every step, and a
product whose partial products left the range was zero or infinite where the
answer was not. Each product and each running product is now the exact one
rounded once, in every layout -- over every element, along rows, down
columns, over several axes -- for float32 and float64.
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


def _running(values, dtype: str) -> list[float]:
    product, out = Fraction(1), []
    for value in np.ravel(values):
        product *= Fraction(float(value))
        out.append(_rounded(product, dtype))
    return out


def _product(values, dtype: str) -> float:
    product = Fraction(1)
    for value in np.ravel(values):
        product *= Fraction(float(value))
    return _rounded(product, dtype)


def _draw(rng, shape, dtype: str, kind: str):
    if kind == "near one":
        values = 1 + rng.normal(size=shape) * 1e-3
    elif kind == "spread":
        values = (1 + rng.normal(size=shape) * 0.5) * 2.0 ** rng.integers(-3, 4, shape)
    else:
        # Wide enough that partial products leave the range.
        values = rng.normal(size=shape) * 2.0 ** rng.integers(-60, 61, shape)
    return values.astype(dtype)


KINDS = ["near one", "spread", "wide"]


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("kind", KINDS)
def test_every_element_and_running(dtype, kind):
    rng = np.random.default_rng(557)
    for trial in range(30):
        n = int(rng.integers(1, 120))
        values = _draw(rng, n, dtype, kind)
        tensor = mt.Tensor(values, dtype=dtype)
        want = _running(values, dtype)
        assert tensor.prod().item() == want[-1], trial
        assert tensor.cumprod(0).tolist() == want, trial


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("kind", KINDS)
def test_rows_columns_and_several_axes(dtype, kind):
    rng = np.random.default_rng(563)
    values = _draw(rng, (23, 37), dtype, kind)
    tensor = mt.Tensor(values, dtype=dtype)
    assert tensor.prod(dim=1).tolist() == [_product(row, dtype) for row in values]
    assert tensor.prod(dim=0).tolist() == [
        _product(values[:, j], dtype) for j in range(37)
    ]
    got = tensor.prod(dim=0, keepdim=True)
    assert got.shape == (1, 37)
    assert got.tolist()[0] == [_product(values[:, j], dtype) for j in range(37)]
    assert tensor.cumprod(1).tolist() == [_running(row, dtype) for row in values]
    columns = [_running(values[:, j], dtype) for j in range(37)]
    assert tensor.cumprod(0).tolist() == [list(row) for row in zip(*columns)]
    cube = _draw(rng, (5, 6, 7), dtype, kind)
    tensor = mt.Tensor(cube, dtype=dtype)
    # Axes side by side, the middle one, and two apart.
    assert tensor.prod(dim=[0, 1]).tolist() == [
        _product(cube[:, :, k], dtype) for k in range(7)
    ]
    assert tensor.prod(dim=1).tolist() == [
        [_product(cube[i, :, k], dtype) for k in range(7)] for i in range(5)
    ]
    assert tensor.prod(dim=[0, 2]).tolist() == [
        _product(cube[:, j, :], dtype) for j in range(6)
    ]
    assert tensor.cumprod(1).tolist() == [
        [list(row) for row in zip(*[_running(cube[i, :, k], dtype) for k in range(7)])]
        for i in range(5)
    ]


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_long_runs(dtype):
    # Long enough that a run is cut into pieces whose running products start
    # from the product of the pieces before them.
    rng = np.random.default_rng(569)
    values = _draw(rng, 9000, dtype, "near one")
    values[4321] = -values[4321]
    tensor = mt.Tensor(values, dtype=dtype)
    want = _running(values, dtype)
    assert tensor.cumprod(0).tolist() == want
    assert tensor.prod().item() == want[-1]
    # Long and wide: partial products past the range, and back.
    values = _draw(rng, 5000, dtype, "wide")
    got = mt.Tensor(values, dtype=dtype).cumprod(0).tolist()
    assert got == _running(values, dtype)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_specials_and_the_ends_of_the_range(dtype):
    def prod(values):
        return mt.tensor(values, dtype=dtype).prod().item()

    def cumprod(values):
        return mt.tensor(values, dtype=dtype).cumprod(0).tolist()

    nan, inf = math.nan, math.inf
    big = 1e200 if dtype == "float64" else 1e30
    # Partial products past the range where the product is not.
    want = _product(np.array([big, big, 1 / big, 1 / big], dtype=dtype), dtype)
    assert prod([big, big, 1 / big, 1 / big]) == want
    assert prod([big, big]) == inf and prod([-big, big]) == -inf
    assert prod([1 / big, 1 / big]) == 0.0
    assert math.copysign(1.0, prod([-1 / big, 1 / big])) == -1.0
    # A zero, an infinity and a NaN settle it by their flags and the sign.
    assert math.isnan(prod([0.0, inf])) and math.isnan(prod([2.0, nan]))
    assert prod([-0.0, 2.0]) == 0.0 and math.copysign(1.0, prod([-0.0, 2.0])) == -1.0
    assert prod([-1.0, inf]) == -inf
    got = cumprod([2.0, -0.0, 3.0, inf])
    assert got[:3] == [2.0, -0.0, -0.0] and math.isnan(got[3])
    assert math.copysign(1.0, got[1]) == -1.0
    # The empty product, and a product over no axes.
    assert mt.tensor([], dtype=dtype).prod().item() == 1.0
    assert mt.tensor(3.5, dtype=dtype).prod().item() == 3.5
    assert mt.zeros((0, 3), dtype=dtype).prod(dim=0).tolist() == [1.0, 1.0, 1.0]
    assert mt.tensor([1.5, -2.0], dtype=dtype).prod(dim=[]).tolist() == [1.5, -2.0]


def test_subnormal_results_and_ties():
    # A float64 product among the subnormals, rounded on their grid.
    values = [2.0**-600, 2.0**-470, 1.0 + 2.0**-30, 3.0]
    exact = _product(np.array(values), "float64")
    assert exact != 0.0 and abs(exact) < 2.0**-1022
    assert mt.tensor(values, dtype="float64").prod().item() == exact
    # 4097 squared is halfway between two float32s, and stays so times one.
    values = [4097.0, 4097.0] + [1.0] * 3000
    got = mt.tensor(values, dtype="float32").cumprod(0).tolist()
    assert got[0] == 4097.0 and set(got[1:]) == {16785408.0}
    assert mt.tensor(values, dtype="float32").prod().item() == 16785408.0
    columns = np.ones((3000, 20), dtype=np.float32)
    columns[:2] = 4097.0
    got = mt.Tensor(columns, dtype="float32").cumprod(0).tolist()
    assert all(value == 16785408.0 for row in got[1:] for value in row)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_gradients(dtype):
    x = mt.tensor([[1.5, -2.0, 4.0], [3.0, 0.5, 2.0]], dtype=dtype, requires_grad=True)
    x.prod(dim=1).sum().backward()
    want = [[-8.0, 6.0, -3.0], [1.0, 6.0, 1.5]]
    assert np.allclose(x.grad.tolist(), want)
    assert str(x.grad.dtype) == dtype
    x.grad = None
    x.cumprod(1).sum().backward()
    # The running products' sum, each factor's share of every product after.
    want = [[1 + -2.0 + -8.0, 1.5 + 1.5 * 4.0, -3.0], [1 + 0.5 + 1.0, 3.0 + 6.0, 1.5]]
    assert np.allclose(x.grad.tolist(), want)
