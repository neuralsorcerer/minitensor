# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""`hypot` is the float nearest the exact `sqrt(x ** 2 + y ** 2)`.

The platform's float64 `hypot` was not rounded once: about one root in 170
of ordinary data was an ulp out. A float32
`hypot` of an integer beside a float32 was taken in float64 and rounded again
on the way back, so a root just past a float32 halfway point fell on it and
went to the even side. Each root is now settled from the exact squares by a
bound, or taken exactly where the bound cannot say -- which a root lying on a
halfway point always needs.
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


def _root(x, y, dtype: str) -> float:
    """The float nearest the exact root of `x ** 2 + y ** 2`."""

    total = Fraction(float(x)) ** 2 + Fraction(float(y)) ** 2
    if total == 0:
        return 0.0
    # Far more bits than either float keeps, with whatever is left over
    # marked strictly inside the last place.
    shift = 1200
    whole = total.numerator * total.denominator * 4**shift
    root = math.isqrt(whole)
    scale = total.denominator * 2**shift
    value = Fraction(root, scale)
    if root * root != whole:
        value += Fraction(1, 2 * scale)
    if dtype == "float32":
        return _float32(value)
    try:
        return float(value)
    except OverflowError:
        return math.inf


def _halfway_triple(low: int) -> tuple[int, int, int]:
    """Legs below `low` whose hypotenuse is an odd whole number past it."""

    # Euclid's triples near equal legs, where the hypotenuse is `sqrt(2)`
    # times a leg.
    n = math.isqrt(int(low / (4 + 2 * math.sqrt(2)))) + 1
    while True:
        for m in range(
            int(n * (1 + math.sqrt(2))) - 3, int(n * (1 + math.sqrt(2))) + 4
        ):
            if m <= n or (m - n) % 2 == 0 or math.gcd(m, n) != 1:
                continue
            a, b, c = m * m - n * n, 2 * m * n, m * m + n * n
            if c > low and max(a, b) < low:
                return a, b, c
        n += 1


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_roots_round_once(dtype):
    rng = np.random.default_rng(443)
    for _ in range(200):
        n = int(rng.integers(1, 60))
        x = rng.normal(loc=float(rng.choice([0.0, 5.0, 1e3])), size=n)
        y = rng.normal(size=n) * 2.0 ** rng.integers(-30, 30, size=n)
        x, y = x.astype(dtype), y.astype(dtype)
        got = mt.hypot(mt.Tensor(x, dtype=dtype), mt.Tensor(y, dtype=dtype))
        assert got.tolist() == [_root(a, b, dtype) for a, b in zip(x, y)]


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_roots_on_a_halfway_point(dtype):
    # Integer legs whose hypotenuse is odd and one place too long for the
    # dtype: it lies halfway between two floats, and rounds to the even one.
    bits = 24 if dtype == "float32" else 53
    a, b, c = _halfway_triple(2**bits)
    got = mt.hypot(
        mt.tensor([float(a)], dtype=dtype), mt.tensor([float(b)], dtype=dtype)
    )
    assert got.tolist() == [_root(a, b, dtype)]
    assert got.tolist()[0] in (c - 1, c + 1)
    assert got.tolist()[0] % 4 == 0


def test_an_integer_beside_a_float32():
    # 2^30 + 64 lies halfway between two float32s, and the exact root of
    # its square plus one lies just past it, so it rounds up.
    big = 2**30 + 64
    got = mt.hypot(mt.tensor([big], dtype="int32"), mt.tensor([1.0], dtype="float32"))
    assert str(got.dtype) == "float32"
    assert got.tolist() == [float(2**30 + 128)]
    rng = np.random.default_rng(449)
    ints = rng.integers(-(2**31), 2**31, size=300).astype(np.int32)
    floats = rng.normal(size=300).astype(np.float32) * 1e3
    got = mt.hypot(mt.Tensor(ints, dtype="int32"), mt.Tensor(floats, dtype="float32"))
    assert got.tolist() == [_root(a, b, "float32") for a, b in zip(ints, floats)]


def test_the_ends_of_the_range():
    largest = float(np.finfo(np.float64).max)
    cases = [
        (largest, largest),
        (largest, 1.0),
        (largest / 2, largest / 2),
        (1e300, 1e300),
        (3e-310, 4e-310),
        (5e-324, 5e-324),
        (1e-320, 1e-300),
        (2.0**-1020, 2.0**-1021),
        (1e200, 1e-200),
        (3.0, 4.0),
    ]
    for x, y in cases:
        got = mt.hypot(mt.tensor([x], dtype="float64"), mt.tensor([y], dtype="float64"))
        assert got.tolist() == [_root(x, y, "float64")], (x, y)
    assert mt.hypot(
        mt.tensor([3e38], dtype="float32"), mt.tensor([3e38], dtype="float32")
    ).tolist() == [math.inf]
    tiny = float(np.float32(1e-45))
    got = mt.hypot(
        mt.tensor([tiny], dtype="float32"), mt.tensor([tiny], dtype="float32")
    )
    assert got.tolist() == [_root(tiny, tiny, "float32")]


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_specials_and_zeros(dtype):
    def hypot(x, y):
        return mt.hypot(mt.tensor([x], dtype=dtype), mt.tensor([y], dtype=dtype)).item()

    assert hypot(math.inf, math.nan) == math.inf
    assert hypot(math.nan, -math.inf) == math.inf
    assert math.isnan(hypot(math.nan, 1.0))
    assert math.copysign(1.0, hypot(-0.0, -0.0)) == 1.0
    assert hypot(-2.5, 0.0) == 2.5


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_broadcasting_and_the_gradient(dtype):
    column = mt.tensor([[3.0], [5.0]], dtype=dtype, requires_grad=True)
    row = mt.tensor([4.0, 12.0], dtype=dtype, requires_grad=True)
    got = mt.hypot(column, row)
    assert got.tolist() == [[5.0, _root(3, 12, dtype)], [_root(5, 4, dtype), 13.0]]
    got.sum().backward()
    want = [[3 / 5 + 3 / 153**0.5], [5 / 41**0.5 + 5 / 13]]
    assert np.allclose(column.grad.tolist(), want)
