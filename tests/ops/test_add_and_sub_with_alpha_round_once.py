# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""`add` and `sub` with an `alpha` are the floats nearest the exact values.

`input + alpha * other` was the product rounded and then the sum, and of
random float operands one element in seven to one in three came out other
than the float nearest the exact value. Each float element is that float
now, `alpha` read at the result's width as a literal beside it is; an
integer literal the width does not hold is read exactly, as arithmetic
reads it.
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


def _want(x, y, alpha: float, sign: int, dtype: str) -> list[float]:
    alpha = Fraction(float(np.float32(alpha)) if dtype == "float32" else alpha)
    return [
        _rounded(Fraction(float(a)) + sign * alpha * Fraction(float(b)), dtype)
        for a, b in zip(np.ravel(x), np.ravel(y))
    ]


def _same(got, want) -> bool:
    got = np.ravel(got).tolist()
    return len(got) == len(want) and all(
        (math.isnan(a) and math.isnan(b))
        or (a == b and math.copysign(1.0, a) == math.copysign(1.0, b))
        for a, b in zip(got, want)
    )


def _draw(rng, shape, dtype: str):
    values = rng.normal(size=shape) * 10.0 ** rng.uniform(-6, 6, shape)
    return values.astype(dtype)


ALPHAS = [0.3, -2.5, 1e-3, 7.0, 1 / 3]


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("name, sign", [("add", 1), ("sub", -1)])
def test_same_shapes(dtype, name, sign):
    rng = np.random.default_rng(691)
    for alpha in ALPHAS:
        x, y = _draw(rng, 3000, dtype), _draw(rng, 3000, dtype)
        got = getattr(mt, name)(
            mt.Tensor(x, dtype=dtype), mt.Tensor(y, dtype=dtype), alpha=alpha
        )
        assert str(got.dtype) == dtype
        assert _same(got.tolist(), _want(x, y, alpha, sign, dtype)), alpha


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("name, sign", [("add", 1), ("sub", -1)])
def test_scalars_and_broadcasts(dtype, name, sign):
    rng = np.random.default_rng(701)
    function = getattr(mt, name)
    x = _draw(rng, (40, 30), dtype)
    tensor = mt.Tensor(x, dtype=dtype)
    for alpha in ALPHAS:
        # A Python number, read at the tensor's width.
        value = float(_draw(rng, 1, dtype)[0])
        got = function(tensor, value, alpha=alpha).tolist()
        assert _same(got, _want(x, np.full(x.shape, value), alpha, sign, dtype))
        # A row and a column, and the tensor broadcast against a larger one.
        row = _draw(rng, 30, dtype)
        got = function(tensor, mt.Tensor(row, dtype=dtype), alpha=alpha).tolist()
        assert _same(got, _want(x, np.broadcast_to(row, x.shape), alpha, sign, dtype))
        column = _draw(rng, (40, 1), dtype)
        got = function(tensor, mt.Tensor(column, dtype=dtype), alpha=alpha).tolist()
        want = _want(x, np.broadcast_to(column, x.shape), alpha, sign, dtype)
        assert _same(got, want)
        got = function(mt.Tensor(row, dtype=dtype), tensor, alpha=alpha).tolist()
        assert _same(got, _want(np.broadcast_to(row, x.shape), x, alpha, sign, dtype))


def test_long_operands_and_specials():
    rng = np.random.default_rng(709)
    for dtype in ("float32", "float64"):
        x, y = _draw(rng, 50_000, dtype), _draw(rng, 50_000, dtype)
        info = np.finfo(dtype)
        specials = [0.0, -0.0, math.inf, -math.inf, math.nan, float(info.max)]
        x[: len(specials)] = specials
        y[: len(specials)] = specials[::-1]
        y[10:20] = float(info.smallest_subnormal)
        got = mt.add(mt.Tensor(x, dtype=dtype), mt.Tensor(y, dtype=dtype), alpha=0.75)
        got = got.tolist()
        want = []
        for a, b in zip(x, y):
            a, b = float(a), float(b)
            if not (math.isfinite(a) and math.isfinite(b)):
                want.append(a + float(np.dtype(dtype).type(0.75)) * b)
            else:
                want.append(_want([a], [b], 0.75, 1, dtype)[0])
        assert _same(got, want), dtype


def test_halfway_sums_of_float32():
    # Sums of two float32s that land on a float32 halfway point exactly, and
    # those a product's last bits move off one.
    x = np.array([1.0, 1.0, 3.0, 1 + 2.0**-23], dtype=np.float32)
    y = np.array([2.0**-24, 2.0**-24 + 2.0**-40, 2.0**-23, 2.0**-24], dtype=np.float32)
    for alpha in (1.5, -1.0, 0.5):
        got = mt.add(mt.Tensor(x), mt.Tensor(y), alpha=alpha).tolist()
        assert _same(got, _want(x, y, alpha, 1, "float32")), alpha


def test_integers_and_mixed_operands():
    big = mt.tensor([2.0**24], dtype="float32")
    # An integer literal the float32 does not hold, read exactly.
    assert mt.add(big, 2**24 + 1, alpha=-1).tolist() == [-1.0]
    assert mt.sub(big, 2**24 + 1, alpha=2).tolist() == [-16777218.0]
    assert mt.add(
        2**24 + 1, mt.tensor([1.0], dtype="float32"), alpha=0.5
    ).tolist() == [16777218.0]
    # An int64 tensor beside a float32 one.
    got = mt.add(big, mt.tensor([2**24 + 1], dtype="int64"), alpha=1.5)
    assert str(got.dtype) == "float32"
    assert got.tolist() == [_float32(Fraction(2**24) + Fraction(3, 2) * (2**24 + 1))]
    # Integers keep their dtype and their exact arithmetic.
    got = mt.add(mt.tensor([1, 2]), mt.tensor([3, 4]), alpha=2)
    assert str(got.dtype) == "int64" and got.tolist() == [7, 10]
    got = mt.add(
        mt.tensor([1, 2], dtype="int32"), mt.tensor([3, 4], dtype="int32"), alpha=-3
    )
    assert str(got.dtype) == "int32" and got.tolist() == [-8, -10]
    # An integral alpha beside int64s past 2^53, exact as the operator is.
    x = mt.tensor([64.0], dtype="float64")
    want = float(Fraction(64) - (2**60 + 1))
    assert mt.sub(x, mt.tensor([2**60 + 1], dtype="int64"), alpha=1).tolist() == [want]
    assert mt.sub(x, 2**60 + 1, alpha=1).tolist() == [want]


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_gradients_and_methods(dtype):
    x = mt.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=dtype, requires_grad=True)
    row = mt.tensor([0.5, -1.0, 2.0], dtype=dtype, requires_grad=True)
    (mt.sub(x, row, alpha=3) * mt.tensor([[1.0], [2.0]], dtype=dtype)).sum().backward()
    assert x.grad.tolist() == [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]
    assert row.grad.tolist() == [-9.0, -9.0, -9.0]
    a = mt.tensor([1.0, 2.0], dtype=dtype, requires_grad=True)
    b = mt.tensor([3.0, 4.0], dtype=dtype, requires_grad=True)
    a.add(b, alpha=0.5).sum().backward()
    assert a.grad.tolist() == [1.0, 1.0] and b.grad.tolist() == [0.5, 0.5]
    got = a.detach().sub(b.detach(), alpha=0.1).tolist()
    want = _want([1.0, 2.0], [3.0, 4.0], 0.1, -1, dtype)
    assert _same(got, want)
