# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""A whole exponent gives the float nearest the exact power.

The transcendental power missed it by an ulp one time in forty in float64,
and the float32 one carried in float64 rounded twice. Each is that float now,
for an exponent shared by the slice, one per element or one base to many,
past either end of the range, and for an integer beside a float32, whose
power was rounded to float64 first.
"""

from __future__ import annotations

import math
import struct
from decimal import Decimal, localcontext
from fractions import Fraction

import numpy as np
import pytest

import minitensor as mt


def _float32(value: Fraction) -> float:
    """The float32 nearest `value`, ties to even, a zero keeping its sign."""

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


def _float64(value: Fraction) -> float:
    try:
        return value.numerator / value.denominator
    except OverflowError:
        return math.inf if value > 0 else -math.inf


def _power(x: float, n: int, dtype: str) -> float:
    """The float nearest `x ** n`, and IEEE's power for a zero, an infinity
    or a NaN `x`."""

    if n == 0:
        return 1.0
    if math.isnan(x):
        return math.nan
    negative = math.copysign(1.0, x) < 0 and n % 2 == 1
    if x == 0 or math.isinf(x):
        large = (x != 0) == (n > 0)
        return math.copysign(math.inf if large else 0.0, -1.0 if negative else 1.0)
    exact = Fraction(x) ** n
    return _float32(exact) if dtype == "float32" else _float64(exact)


def _bits(values) -> list[int]:
    """Each value's float64 bits, every NaN alike."""

    return [
        -1 if math.isnan(v) else struct.unpack("<q", struct.pack("<d", v))[0]
        for v in values
    ]


def _draw(rng, size, dtype: str):
    return (rng.normal(size=size) * 10.0 ** rng.uniform(-3, 3, size)).astype(dtype)


_EXPONENTS = [0, 1, -1, 2, -2, 3, -3, 4, -4, 5, -5, 7, 10, -10, 33, -33, 100, -100]

_SPECIALS = [0.0, -0.0, math.inf, -math.inf, math.nan, 1.0, -1.0, 2.0, -0.5]


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("n", _EXPONENTS)
def test_one_exponent_for_the_slice(dtype, n):
    rng = np.random.default_rng(881 + n)
    x = np.concatenate([_draw(rng, 400, dtype), np.array(_SPECIALS, dtype=dtype)])
    want = _bits([_power(float(v), n, dtype) for v in x])
    tensor = mt.Tensor(x, dtype=dtype)
    for got in (tensor**n, tensor ** float(n), tensor.pow(n)):
        assert str(got.dtype) == dtype
        assert _bits(got.tolist()) == want


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_one_exponent_per_element(dtype):
    rng = np.random.default_rng(883)
    x = _draw(rng, 3000, dtype)
    n = rng.integers(-40, 41, 3000)
    got = mt.Tensor(x, dtype=dtype) ** mt.Tensor(n.astype(dtype), dtype=dtype)
    want = [_power(float(v), int(k), dtype) for v, k in zip(x, n)]
    assert _bits(got.tolist()) == _bits(want)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_whole_exponents_beside_others(dtype):
    # A whole exponent's power is the nearest float; any other's is what the
    # same pair gives on its own, whatever the exponents beside it.
    rng = np.random.default_rng(887)
    x = np.abs(_draw(rng, 2000, dtype))
    y = rng.integers(-12, 13, 2000).astype(np.float64)
    fractional = rng.random(2000) < 0.5
    y[fractional] += rng.uniform(0.05, 0.95, int(fractional.sum()))
    y = y.astype(dtype)
    got = (mt.Tensor(x, dtype=dtype) ** mt.Tensor(y, dtype=dtype)).tolist()
    alone = (
        mt.Tensor(x[fractional], dtype=dtype) ** mt.Tensor(y[fractional], dtype=dtype)
    ).tolist()
    assert _bits([g for g, f in zip(got, fractional) if f]) == _bits(alone)
    want = [
        _power(float(v), int(k), dtype) for v, k, f in zip(x, y, fractional) if not f
    ]
    assert _bits([g for g, f in zip(got, fractional) if not f]) == _bits(want)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_one_base_to_many_exponents(dtype):
    exponents = np.array([-60, -7, -1, 0, 1, 2, 3, 9, 31, 64], dtype=dtype)
    for base in (0.1, -2.7, 3.0, 1e-3, -0.0, math.inf):
        typed = float(np.dtype(dtype).type(base))
        got = base ** mt.Tensor(exponents, dtype=dtype)
        want = [_power(typed, int(k), dtype) for k in exponents]
        assert _bits(got.tolist()) == _bits(want), base
    mixed = np.array([2.5, 3.0, -0.5, -4.0], dtype=dtype)
    got = (1.1 ** mt.Tensor(mixed, dtype=dtype)).tolist()
    typed = float(np.dtype(dtype).type(1.1))
    assert _bits([got[1], got[3]]) == _bits(
        [_power(typed, 3, dtype), _power(typed, -4, dtype)]
    )
    alone = (1.1 ** mt.Tensor(mixed[[0, 2]], dtype=dtype)).tolist()
    assert _bits([got[0], got[2]]) == _bits(alone)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_powers_past_the_range(dtype):
    info = np.finfo(dtype)
    tiny = float(info.smallest_subnormal)
    big = float(info.max)
    x = [big, -big, tiny, -tiny, float(np.sqrt(info.tiny)), 1.5 * math.sqrt(tiny)]
    for n in (2, 3, -1, -2, -3, 1001, -1001):
        typed = [float(np.dtype(dtype).type(v)) for v in x]
        got = mt.Tensor(np.array(typed, dtype=dtype), dtype=dtype) ** n
        assert _bits(got.tolist()) == _bits([_power(v, n, dtype) for v in typed]), n
    # A power so far past the range it is never formed, an odd exponent
    # keeping a negative base's sign.
    huge = mt.Tensor(np.array([1.5, -1.5, 0.75, -1.0], dtype=dtype), dtype=dtype)
    assert _bits((huge ** (2**52 + 1)).tolist()) == _bits(
        [math.inf, -math.inf, 0.0, -1.0]
    )
    assert _bits((huge ** (-(2**52) - 1)).tolist()) == _bits(
        [0.0, -0.0, math.inf, -1.0]
    )


@pytest.mark.parametrize(
    "dtype, x, n",
    [
        ("float64", 1.0 + 2.0**-52, 2**60),
        ("float64", 1.0 + 2.0**-52, -(2**60)),
        ("float64", 1.0 - 2.0**-53, 2**62),
        ("float64", -(1.0 - 2.0**-53), 2**61 + 2**40),
        ("float64", 1.0 + 2.0**-50, 2**56 + 12345 * 2**4),
        ("float32", 1.0 + 2.0**-23, 2**29),
        ("float32", 1.0 - 2.0**-24, -(2**30)),
        ("float32", -(1.0 + 2.0**-20), 2**25 + 2**3),
    ],
)
def test_exponents_far_past_a_float_mantissa(dtype, x, n):
    # Powers of a value just off one stay in range to exponents far past
    # 2^53, and are formed exactly there too.
    with localcontext() as context:
        context.prec = 90
        exact = Fraction((Decimal(abs(x)).ln() * n).exp())
    if x < 0 and n % 2 == 1:
        exact = -exact
    want = _float32(exact) if dtype == "float32" else _float64(exact)
    assert math.isfinite(want) and want != 0
    got = mt.Tensor(np.array([x], dtype=dtype), dtype=dtype) ** n
    assert got.tolist() == [want]


def test_an_integer_beside_a_float32():
    # Rounded to float64 first, each of these powers lands on a float32
    # halfway point and goes to the even side.
    cases = [(884702309, 2), (1613624363, 2), (637950001, 4)]
    bases = mt.tensor([b for b, _ in cases], dtype="int32")
    exponents = mt.tensor([float(n) for _, n in cases], dtype="float32")
    want = [_float32(Fraction(b) ** n) for b, n in cases]
    got = bases**exponents
    assert str(got.dtype) == "float32" and got.tolist() == want
    for (b, n), w in zip(cases, want):
        one = mt.tensor([b], dtype="int32")
        assert (one ** mt.tensor([float(n)], dtype="float32")).tolist() == [w]
        assert (one ** float(n)).tolist() == [w]
    # A float32 base to integer exponents.
    rng = np.random.default_rng(907)
    x = _draw(rng, 500, "float32")
    n = rng.integers(-30, 31, 500)
    got = mt.Tensor(x, dtype="float32") ** mt.tensor(n.tolist(), dtype="int64")
    assert str(got.dtype) == "float32"
    want = [_power(float(v), int(k), "float32") for v, k in zip(x, n)]
    assert _bits(got.tolist()) == _bits(want)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_vander(dtype):
    rng = np.random.default_rng(911)
    x = _draw(rng, 60, dtype)
    got = mt.vander(mt.Tensor(x, dtype=dtype), 12).tolist()
    want = [[_power(float(v), 11 - j, dtype) for j in range(12)] for v in x]
    assert got == want
