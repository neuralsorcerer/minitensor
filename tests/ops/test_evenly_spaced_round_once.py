# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""Each point of `linspace` is the exact one rounded once.

Point `i` of `n` from `a` to `b` is `(a * (n - 1 - i) + b * i) / (n - 1)`.
Spaced as `a + i * step` it rounded three times, so `linspace(0, 1, 11)` had
`0.30000000000000004` where `0.3` belongs, and a third of the points from
`1e-3` to `1e3` were an ulp or more off. Rounded once, the points are the
nearest floats to the exact ones -- to the nearest integer, a half away from
zero, at an integer dtype -- which also keeps them in order, symmetric where
the exact points are, and finite between finite ends.
"""

from __future__ import annotations

import random
from fractions import Fraction

import pytest

import minitensor as mt


def _exact(start, end, steps):
    if steps == 1:
        return [Fraction(start)]
    intervals = steps - 1
    return [
        (Fraction(start) * (intervals - i) + Fraction(end) * i) / intervals
        for i in range(steps)
    ]


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
    if rounded >= Fraction(2) ** 128:
        return sign * float("inf")
    return sign * float(rounded)


def _nearest_integer(value: Fraction, low: int, high: int) -> int:
    """`value` to the nearest integer, a half away from zero, saturated."""

    whole = abs(value).numerator // abs(value).denominator
    if abs(value) - whole >= Fraction(1, 2):
        whole += 1
    return max(low, min(high, whole if value >= 0 else -whole))


def _spacings():
    rng = random.Random(1103)
    cases = [
        (0.0, 1.0, 11),
        (1e-3, 1e3, 1001),
        (-1.0, 1.0, 1001),
        (0.1, 0.7, 77),
        (-3.3, 1e10, 999),
        (1.0, 1.0 + 2**-52, 3),
        (5e-324, 1e-320, 9),
        (-1e308, 1e308, 101),
        (2.0**62, 2.0**62 + 4096, 7),
        (0.0, 10.0, 5),
    ]
    for _ in range(120):
        start = rng.choice(
            [rng.uniform(-1e3, 1e3), rng.uniform(-1, 1) * 10 ** rng.randint(-300, 300)]
        )
        end = rng.choice(
            [rng.uniform(-1e3, 1e3), start * (1 + rng.uniform(-1, 1) * 1e-12), -start]
        )
        cases.append((start, end, rng.randint(2, 300)))
    return cases


def test_tenths_are_the_nearest_floats():
    assert mt.linspace(0.0, 1.0, 11, dtype="float64").tolist() == [
        0.0,
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
        1.0,
    ]


@pytest.mark.parametrize("dtype", ["float64", "float32", "int64", "int32"])
def test_every_point_is_the_exact_one_rounded_once(dtype):
    for start, end, steps in _spacings():
        exact = _exact(start, end, steps)
        if dtype == "float64":
            expected = [float(value) for value in exact]
        elif dtype == "float32":
            expected = [_float32(value) for value in exact]
        else:
            bits = 64 if dtype == "int64" else 32
            low, high = -(2 ** (bits - 1)), 2 ** (bits - 1) - 1
            expected = [_nearest_integer(value, low, high) for value in exact]
        got = mt.linspace(start, end, steps, dtype=dtype).tolist()
        assert got == expected, (start, end, steps)


def test_points_are_symmetric_ordered_and_finite():
    points = mt.linspace(-1.0, 1.0, 1001, dtype="float64").tolist()
    assert points[500] == 0.0
    assert all(points[i] == -points[1000 - i] for i in range(1001))
    assert all(a < b for a, b in zip(points, points[1:]))
    # Ends whose difference overflows still space finitely.
    wide = mt.linspace(-1.7e308, 1.7e308, 5, dtype="float64").tolist()
    assert wide == [-1.7e308, -8.5e307, 0.0, 8.5e307, 1.7e308]


def test_halfway_points():
    # The float64 point is halfway between two float32s; the exact one is
    # just past it.
    one_up = float(Fraction(1) + Fraction(1, 2**23))
    past = float(Fraction(1) + Fraction(1, 2**23) + Fraction(1, 2**52))
    assert mt.linspace(1.0, past, 3, dtype="float32").tolist()[1] == one_up
    # Exactly halfway, it rounds to even.
    assert mt.linspace(1.0, one_up, 3, dtype="float32").tolist()[1] == 1.0
    # An integer point exactly halfway goes away from zero.
    assert mt.linspace(0, 10, 5, dtype="int64").tolist() == [0, 3, 5, 8, 10]
    assert mt.linspace(-10, 0, 5, dtype="int64").tolist() == [-10, -8, -5, -3, 0]
    # Past 2^53 a float64 point is no longer the integer.
    end = 2.0**62 + 2048
    assert mt.linspace(0.0, end, 4, dtype="int64").tolist()[1] == (2**62 + 2048) // 3


def test_histogram_edges_and_logspace_use_the_exact_points():
    edges = mt.histogram_bin_edges(mt.tensor([0.0, 1.0]), bins=10).tolist()
    assert edges == [float(value) for value in _exact(0.0, 1.0, 11)]
    exponents = mt.linspace(0.0, 1.0, 11, dtype="float64").tolist()
    assert mt.logspace(0.0, 1.0, 11, dtype="float64").tolist() == [
        10.0**exponent for exponent in exponents
    ]
