# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""`deg2rad` and `rad2deg` are the floats nearest the exact angles.

Each was a product with `pi / 180` or `180 / pi`, the constant rounded --
to float32 for a float32 tensor -- and the product again: of random inputs
about one in eight came out other than the float nearest `x * pi / 180` or
`x * 180 / pi`, float32 and float64 alike. Each element is that float now,
an integer input taken exactly.
"""

from __future__ import annotations

import math
from fractions import Fraction

import numpy as np
import pytest

import minitensor as mt


def _arctan_inverse(k: int, bits: int) -> int:
    """`atan(1 / k) * 2^bits`, truncated, by its alternating series."""

    one = 1 << (bits + 32)
    total, term, n, sign = 0, one // k, 1, 1
    while term:
        total += sign * (term // n)
        term //= k * k
        n += 2
        sign = -sign
    return total >> 32


# Machin's formula, to far more bits than any rounding below needs: no float
# or int64 brings either product within 2^-141 of a halfway point.
_BITS = 400
_PI = Fraction(
    16 * _arctan_inverse(5, _BITS) - 4 * _arctan_inverse(239, _BITS), 1 << _BITS
)
_CONSTANTS = {"deg2rad": _PI / 180, "rad2deg": 180 / _PI}


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


def _want(values, name: str, dtype: str) -> list[float]:
    constant = _CONSTANTS[name]
    out = []
    for value in values:
        value = float(value) if not isinstance(value, (int, np.integer)) else value
        if isinstance(value, float) and not math.isfinite(value):
            out.append(value * float(constant))
            continue
        if value == 0:
            # The sign of a zero is kept.
            out.append(math.copysign(0.0, value) if isinstance(value, float) else 0.0)
            continue
        exact = Fraction(value) * constant
        if dtype == "float32":
            out.append(_float32(exact))
        else:
            try:
                out.append(float(exact))
            except OverflowError:
                out.append(math.inf if exact > 0 else -math.inf)
    return out


def _same(got: list[float], want: list[float]) -> bool:
    return len(got) == len(want) and all(
        (math.isnan(x) and math.isnan(y))
        or (x == y and math.copysign(1.0, x) == math.copysign(1.0, y))
        for x, y in zip(got, want)
    )


def _draw(rng, n: int, dtype: str):
    """Values of both signs across the exponent range, with the ends of the
    range and its subnormals among them."""

    if dtype == "float32":
        bits = rng.integers(0, 2**32, n, dtype=np.uint64).astype(np.uint32)
        values = bits.view(np.float32)
    else:
        bits = rng.integers(0, 2**63, n, dtype=np.uint64) | (
            rng.integers(0, 2, n, dtype=np.uint64) << np.uint64(63)
        )
        values = bits.view(np.float64)
    return values[np.isfinite(values)]


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("name", ["deg2rad", "rad2deg"])
def test_random_bit_patterns(dtype, name):
    rng = np.random.default_rng(673)
    values = _draw(rng, 5000, dtype)
    got = getattr(mt, name)(mt.Tensor(values, dtype=dtype))
    assert str(got.dtype) == dtype
    assert _same(got.tolist(), _want(values, name, dtype))


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("name", ["deg2rad", "rad2deg"])
def test_ordinary_angles(dtype, name):
    # Long enough to be split among threads, and the angles people write.
    rng = np.random.default_rng(677)
    values = (rng.normal(size=40_000) * 10.0 ** rng.uniform(-3, 4, 40_000)).astype(
        dtype
    )
    values[:8] = [0.0, 30.0, 45.0, 90.0, 180.0, 270.0, 360.0, -720.0]
    got = getattr(mt, name)(mt.Tensor(values, dtype=dtype)).tolist()
    assert _same(got, _want(values, name, dtype))
    assert mt.deg2rad(mt.tensor([180.0], dtype=dtype)).item() == (
        float(np.float32(math.pi)) if dtype == "float32" else math.pi
    )


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_specials_and_the_ends_of_the_range(dtype):
    info = np.finfo(dtype)
    values = np.array(
        [
            0.0,
            -0.0,
            math.inf,
            -math.inf,
            math.nan,
            float(info.max),
            -float(info.max),
            float(info.tiny),
            float(info.smallest_subnormal),
            -float(info.smallest_subnormal),
            float(info.max) / 57.29577951308232,
            float(info.max) / 57.2957795,
        ],
        dtype=dtype,
    )
    for name in ("deg2rad", "rad2deg"):
        got = getattr(mt, name)(mt.Tensor(values, dtype=dtype)).tolist()
        assert _same(got, _want(values, name, dtype)), name


def test_integers_are_taken_exactly():
    rng = np.random.default_rng(683)
    small = rng.integers(-(2**31), 2**31, 2000, dtype=np.int64).astype(np.int32)
    small[:2] = [np.iinfo(np.int32).min, np.iinfo(np.int32).max]
    big = rng.integers(-(2**63), 2**63 - 1, 2000, dtype=np.int64)
    big[:3] = [np.iinfo(np.int64).min, np.iinfo(np.int64).max, 2**53 + 1]
    for name in ("deg2rad", "rad2deg"):
        got = getattr(mt, name)(mt.Tensor(small, dtype="int32"))
        assert str(got.dtype) == "float32"
        assert _same(got.tolist(), _want(small.tolist(), name, "float32"))
        got = getattr(mt, name)(mt.Tensor(big, dtype="int64"))
        assert str(got.dtype) == "float64"
        assert _same(got.tolist(), _want(big.tolist(), name, "float64"))
        got = getattr(mt, name)(mt.tensor([True, False]))
        assert got.tolist() == _want([1, 0], name, "float32")


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_gradients_and_methods(dtype):
    x = mt.tensor([1.0, -2.5, 90.0], dtype=dtype, requires_grad=True)
    (mt.deg2rad(x) * mt.tensor([1.0, 2.0, -1.0], dtype=dtype)).sum().backward()
    unit = _want([1.0], "deg2rad", dtype)[0]
    assert x.grad.tolist() == _want([1.0, 2.0, -1.0], "deg2rad", dtype)
    assert x.grad.tolist()[0] == unit
    x = mt.tensor([1.0, -2.5], dtype=dtype, requires_grad=True)
    x.rad2deg().sum().backward()
    assert x.grad.tolist() == _want([1.0, 1.0], "rad2deg", dtype)
    # A Python list, as any operand.
    assert mt.deg2rad([180.0]).tolist() == [float(np.float32(math.pi))]
