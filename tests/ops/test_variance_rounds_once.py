# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""Each variance and standard deviation is the float nearest the exact one.

A float64 variance came from a mean rounded once, deviations rounded again,
and squares rounded and summed: of four hundred short slices of ordinary data
nearly half were an ulp or more from the exact variance, and a quarter of the
standard deviations. Each is now the float nearest
`(N Σx² - (Σx)²) / (N (N - ddof))`, or nearest its root, in every layout --
over every element, along rows, down columns, over several axes -- for float32
and float64, and for integers, which are read exactly.
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
    exponent = value.numerator.bit_length() - value.denominator.bit_length()
    if Fraction(2) ** exponent > value:
        exponent -= 1
    unit = Fraction(2) ** max(exponent - 23, -149)
    rounded = round(value / unit) * unit
    return math.inf if rounded >= Fraction(2) ** 128 else float(rounded)


def _rounded(value: Fraction, dtype: str) -> float:
    if dtype == "float32":
        return _float32(value)
    try:
        return float(value)
    except OverflowError:
        return math.inf


def _variance(values, ddof: int) -> Fraction:
    exact = [
        Fraction(int(v)) if isinstance(v, int) else Fraction(float(v)) for v in values
    ]
    n = len(exact)
    total = sum(exact)
    return (n * sum(v * v for v in exact) - total * total) / (n * (n - ddof))


def _root(value: Fraction) -> Fraction:
    """`sqrt(value)` to far more bits than either float keeps, with whatever
    is left over marked strictly inside the last place."""

    if value == 0:
        return Fraction(0)
    places = 1200
    whole = value.numerator * value.denominator * 4**places
    root = math.isqrt(whole)
    scale = value.denominator * 2**places
    result = Fraction(root, scale)
    if root * root != whole:
        result += Fraction(1, 2 * scale)
    return result


def _expected(values, ddof: int, root: bool, dtype: str) -> float:
    values = list(values)
    if len(values) <= ddof:
        return math.nan
    variance = _variance(values, ddof)
    return _rounded(_root(variance) if root else variance, dtype)


def _draw(rng, shape, dtype, kind):
    values = rng.normal(size=shape) * 2.0 ** rng.integers(-12, 13, size=shape)
    if kind == "offset":
        values = 1e5 + rng.normal(size=shape)
    elif kind == "nearly constant":
        values = 1.0 + rng.normal(size=shape) * 1e-6
    elif kind == "whole":
        values = np.round(rng.normal(size=shape) * 50)
    return values.astype(dtype)


def _same(got, want):
    return got == want or (math.isnan(got) and math.isnan(want))


KINDS = ["spread", "offset", "nearly constant", "whole"]


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("unbiased", [True, False])
def test_every_element(dtype, unbiased):
    rng = np.random.default_rng(491)
    ddof = int(unbiased)
    for trial in range(40):
        n = int(rng.integers(1, 400))
        values = _draw(rng, n, dtype, KINDS[trial % 4])
        tensor = mt.Tensor(values, dtype=dtype)
        for root, got in (
            (False, tensor.var(unbiased=unbiased)),
            (True, tensor.std(unbiased=unbiased)),
        ):
            want = _expected(values, ddof, root, dtype)
            assert _same(got.item(), want), (trial, root, got.item(), want)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("kind", KINDS)
def test_rows_and_columns(dtype, kind):
    rng = np.random.default_rng(499)
    values = _draw(rng, (60, 40), dtype, kind)
    tensor = mt.Tensor(values, dtype=dtype)
    for unbiased in (True, False):
        ddof = int(unbiased)
        for root in (False, True):
            reduce = tensor.std if root else tensor.var
            rows = reduce(dim=1, unbiased=unbiased).tolist()
            assert rows == [_expected(values[i], ddof, root, dtype) for i in range(60)]
            columns = reduce(dim=0, unbiased=unbiased).tolist()
            want = [_expected(values[:, j], ddof, root, dtype) for j in range(40)]
            assert all(_same(g, w) for g, w in zip(columns, want)), (columns, want)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_several_axes_and_keepdim(dtype):
    rng = np.random.default_rng(503)
    values = _draw(rng, (6, 5, 16), dtype, "spread")
    tensor = mt.Tensor(values, dtype=dtype)
    # Axes side by side, the middle one, and two apart.
    got = tensor.var(dim=[0, 1]).tolist()
    assert got == [
        _expected(values[:, :, k].ravel(), 1, False, dtype) for k in range(16)
    ]
    got = tensor.std(dim=1, unbiased=False).tolist()
    want = [
        [_expected(values[i, :, k], 0, True, dtype) for k in range(16)]
        for i in range(6)
    ]
    assert got == want
    got = tensor.var(dim=[0, 2], keepdim=True)
    assert got.shape == (1, 5, 1)
    want = [_expected(values[:, j, :].ravel(), 1, False, dtype) for j in range(5)]
    assert [v[0] for v in got.tolist()[0]] == want


def test_integers_are_read_exactly():
    # Past 2^53 an int64 is not a float64, and read as one would lose its
    # low bits.
    big = 2**60
    values = [big + 1, big - 3, big + 7, big]
    tensor = mt.tensor(values, dtype="int64")
    assert str(tensor.var().dtype) == "float64"
    assert tensor.var().item() == _expected(values, 1, False, "float64")
    assert tensor.std(unbiased=False).item() == _expected(values, 0, True, "float64")
    small = (
        np.random.default_rng(509).integers(-(2**31), 2**31, (7, 33)).astype(np.int32)
    )
    tensor = mt.Tensor(small, dtype="int32")
    got = tensor.var(dim=0).tolist()
    assert str(tensor.var(dim=0).dtype) == "float32"
    want = [
        _expected([int(v) for v in small[:, j]], 1, False, "float32") for j in range(33)
    ]
    assert got == want


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_specials_and_the_ends_of_the_range(dtype):
    def var(values, **kwargs):
        return mt.tensor(values, dtype=dtype).var(**kwargs).item()

    def std(values, **kwargs):
        return mt.tensor(values, dtype=dtype).std(**kwargs).item()

    assert math.isnan(var([1.0, math.nan, 2.0]))
    assert math.isnan(var([1.0, math.inf]))
    assert math.isnan(std([math.inf, math.inf]))
    # No more values than the correction takes away.
    assert math.isnan(var([3.0]))
    assert var([3.0], unbiased=False) == 0.0
    assert var([2.5, 2.5, 2.5]) == 0.0
    assert std([-0.0, 0.0]) == 0.0
    # Squares past the largest float, where the root is not.
    big = 1e200 if dtype == "float64" else 1e30
    assert var([big, -big]) == math.inf
    assert std([big, -big]) == _expected([big, -big], 1, True, dtype)
    tiny = 3e-200 if dtype == "float64" else 3e-40
    values = np.array([tiny, -tiny / 3], dtype=dtype)
    assert std(values.tolist(), unbiased=False) == _expected(values, 0, True, dtype)
    assert var(values.tolist(), unbiased=False) == _expected(values, 0, False, dtype)


def test_gradients():
    x = mt.tensor(
        [[1.0, 2.0, 4.0], [3.0, 3.0, 3.0]], dtype="float64", requires_grad=True
    )
    x.var(dim=1).sum().backward()
    # `2 (x - mean) / (N - 1)`.
    assert np.allclose(x.grad.tolist(), [[-4 / 3, -1 / 3, 5 / 3], [0.0, 0.0, 0.0]])
    x.grad = None
    x.std(dim=1, keepdim=True).sum().backward()
    deviation = np.array([1.0, 2.0, 4.0]) - 7 / 3
    want = deviation / (2 * np.sqrt((deviation**2).sum() / 2))
    # A constant slice's root has no slope to give: zero, not NaN.
    assert np.allclose(x.grad.tolist(), [want.tolist(), [0.0, 0.0, 0.0]])
    y = mt.tensor([1.0, 5.0], dtype="float32", requires_grad=True)
    y.var(unbiased=False).backward()
    assert y.grad.tolist() == [-2.0, 2.0]
    assert str(y.grad.dtype) == "float32"
