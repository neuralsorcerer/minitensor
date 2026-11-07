# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""`lerp`, `addcmul` and `addcdiv` are each the float nearest the exact value.

Written as their formulas, they rounded at every step: `lerp` at the
difference, the product and the sum, and of three hundred short runs of
ordinary float64 data nearly every one held an element an ulp or more out.
`lerp(1, 1e-20, 1)` gave `0` rather than the `1e-20` its endpoint promises,
and two ends of opposite sign past half the largest float overflowed the
difference. Each element is now rounded once from the exact value -- the
products split exactly, the parts bounded, and summed exactly where the bound
cannot say.
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


def _exact(values) -> list[Fraction]:
    return [Fraction(float(v)) for v in np.asarray(values, dtype=np.float64).ravel()]


def _spreads(dtype):
    """Powers of two to spread values over, inside the dtype's range."""

    return (2, 20, 40) if dtype == "float32" else (2, 20, 200)


def _draw(rng, dtype, n, spread):
    values = rng.normal(size=n) * 2.0 ** rng.integers(-spread, spread + 1, size=n)
    return values.astype(dtype)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_lerp_rounds_once(dtype):
    rng = np.random.default_rng(457)
    for spread in _spreads(dtype):
        start, end = _draw(rng, dtype, 400, spread), _draw(rng, dtype, 400, spread)
        end[::3] = start[::3] * (1 + rng.normal(size=134).astype(dtype) * 1e-6)
        weight = rng.uniform(-0.5, 1.5, size=400).astype(dtype)
        weight[::5] = rng.choice([0.0, 1.0, 0.5, 2.0], size=80)
        got = mt.lerp(
            mt.Tensor(start, dtype=dtype),
            mt.Tensor(end, dtype=dtype),
            mt.Tensor(weight, dtype=dtype),
        ).tolist()
        want = [
            _rounded(s + w * (e - s), dtype)
            for s, e, w in zip(_exact(start), _exact(end), _exact(weight))
        ]
        assert got == want, spread


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_addcmul_and_addcdiv_round_once(dtype):
    rng = np.random.default_rng(461)
    for spread in _spreads(dtype):
        x, a, b = (_draw(rng, dtype, 400, spread) for _ in range(3))
        # Inputs that nearly cancel the product.
        x[::4] = -(a[::4] * b[::4]) * (1 + rng.normal(size=100).astype(dtype) * 1e-7)
        value = float(np.asarray(rng.normal(), dtype=dtype))
        tensors = [mt.Tensor(v, dtype=dtype) for v in (x, a, b)]
        got = mt.addcmul(*tensors, value=value).tolist()
        v = Fraction(value)
        want = [
            _rounded(p + v * q * r, dtype)
            for p, q, r in zip(_exact(x), _exact(a), _exact(b))
        ]
        assert got == want, spread
        got = mt.addcdiv(*tensors, value=value).tolist()
        want = [
            _rounded(p + v * q / r, dtype)
            for p, q, r in zip(_exact(x), _exact(a), _exact(b))
        ]
        assert got == want, spread


def test_the_endpoints_hold_where_the_difference_rounds():
    for dtype in ("float32", "float64"):
        start = mt.tensor([1.0, -3.0, 1e30], dtype=dtype)
        end = mt.tensor([1e-20, 1e-30, -1e-30], dtype=dtype)
        assert mt.lerp(start, end, 1.0).tolist() == end.tolist()
        assert mt.lerp(start, end, 0.0).tolist() == start.tolist()
    largest = float(np.finfo(np.float64).max)
    start = mt.tensor([-largest], dtype="float64")
    end = mt.tensor([largest], dtype="float64")
    assert mt.lerp(start, end, 0.75).tolist() == [largest / 2]
    assert mt.lerp(start, end, 1.0).tolist() == [largest]


def test_value_takes_the_width_of_the_result():
    # A literal `value` is read at the result's width, as `0.1` beside a
    # float32 tensor is.
    x = np.array([1.0, 2.0], np.float32)
    got = mt.addcmul(mt.Tensor(x), mt.Tensor(x), mt.Tensor(x), value=0.1).tolist()
    tenth = Fraction(float(np.float32(0.1)))
    assert got == [_float32(Fraction(1) + tenth), _float32(Fraction(2) + tenth * 4)]


def test_integers_beside_a_float32_are_read_exactly():
    # 2^24 + 1 is not a float32, and read as one it would lose the 1.
    big = 2**24 + 1
    ints = mt.tensor([big], dtype="int32")
    half = mt.tensor([0.5], dtype="float32")
    got = mt.addcmul(ints, half, half, value=2.0)
    assert str(got.dtype) == "float32"
    assert got.tolist() == [_float32(Fraction(big) + Fraction(1, 2))]
    got = mt.lerp(ints, mt.tensor([0.0], dtype="float32"), half)
    assert got.tolist() == [_float32(Fraction(big, 2))]
    # Two integers divide into a float, rounded once.
    got = mt.addcdiv(mt.tensor([1, 2]), mt.tensor([1, 1]), mt.tensor([3, 3]), value=3)
    assert str(got.dtype) == "float64"
    assert got.tolist() == [2.0, 3.0]
    # Integers alone keep the formula's integer result.
    got = mt.lerp(mt.tensor([1, 2]), mt.tensor([3, 4]), mt.tensor([2, 2]))
    assert got.tolist() == [5, 6]
    assert "int" in str(got.dtype)


def test_broadcasting():
    rng = np.random.default_rng(463)
    column = rng.normal(size=(5, 1))
    row = rng.normal(size=(1, 4))
    weight = rng.normal(size=4)
    got = mt.lerp(
        mt.Tensor(column, dtype="float64"),
        mt.Tensor(row, dtype="float64"),
        mt.Tensor(weight, dtype="float64"),
    ).tolist()
    want = [
        [
            float(Fraction(s) + Fraction(w) * (Fraction(e) - Fraction(s)))
            for e, w in zip(row[0], weight)
        ]
        for s in column[:, 0]
    ]
    assert got == want
    got = mt.addcdiv(
        mt.Tensor(column, dtype="float64"), 1.0, mt.Tensor(row, dtype="float64")
    )
    assert got.tolist() == [
        [float(Fraction(s) + 1 / Fraction(r)) for r in row[0]] for s in column[:, 0]
    ]


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_specials(dtype):
    def lerp(s, e, w):
        return mt.lerp(
            mt.tensor([s], dtype=dtype), mt.tensor([e], dtype=dtype), w
        ).item()

    assert lerp(0.0, math.inf, 0.5) == math.inf
    assert math.isnan(lerp(math.inf, math.inf, 0.5))
    assert math.isnan(lerp(math.nan, 1.0, 0.0))
    one = mt.tensor([1.0], dtype=dtype)
    zero = mt.tensor([0.0], dtype=dtype)
    assert mt.addcdiv(one, one, zero).item() == math.inf
    assert math.isnan(mt.addcdiv(one, zero, zero).item())
    # An exact zero keeps the formula's sign.
    negative = mt.tensor([-0.0], dtype=dtype)
    got = mt.addcmul(negative, negative, one).item()
    assert got == 0.0 and math.copysign(1.0, got) == -1.0
    got = mt.addcmul(-one, one, one).item()
    assert got == 0.0 and math.copysign(1.0, got) == 1.0


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_gradients(dtype):
    start = mt.tensor([1.0, 2.0], dtype=dtype, requires_grad=True)
    end = mt.tensor([3.0, 5.0], dtype=dtype, requires_grad=True)
    weight = mt.tensor([0.25, 0.5], dtype=dtype, requires_grad=True)
    mt.lerp(start, end, weight).sum().backward()
    assert start.grad.tolist() == [0.75, 0.5]
    assert end.grad.tolist() == [0.25, 0.5]
    assert weight.grad.tolist() == [2.0, 3.0]

    x = mt.tensor([1.0, 2.0], dtype=dtype, requires_grad=True)
    a = mt.tensor([3.0, 4.0], dtype=dtype, requires_grad=True)
    b = mt.tensor([2.0, 8.0], dtype=dtype, requires_grad=True)
    mt.addcmul(x, a, b, value=0.5).sum().backward()
    assert x.grad.tolist() == [1.0, 1.0]
    assert a.grad.tolist() == [1.0, 4.0]
    assert b.grad.tolist() == [1.5, 2.0]
    for t in (x, a, b):
        t.grad = None
    mt.addcdiv(x, a, b, value=2.0).sum().backward()
    assert x.grad.tolist() == [1.0, 1.0]
    assert a.grad.tolist() == [1.0, 0.25]
    assert b.grad.tolist() == [-1.5, -0.125]
    # A broadcast operand sums its gradient back.
    shared = mt.tensor(0.5, dtype=dtype, requires_grad=True)
    mt.lerp(start.detach(), end.detach(), shared).sum().backward()
    assert shared.grad.item() == 5.0
