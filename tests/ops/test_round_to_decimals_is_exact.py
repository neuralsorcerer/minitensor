# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""`round(x, decimals)` rounds the value `x` holds, not its rounded scaling.

It was `round(x * 10^d) / 10^d` in floating point, which rounds the product
before deciding which integer is nearest: `2.675` is
2.67499999999999982236431605997495353221893310546875 as a double, but its
product with 100 rounds to exactly 267.5, and the tie went to the even
neighbour, 2.68. The answer is the float nearest `N / 10^d` for `N` the
integer nearest the exact `x * 10^d`, halves to even -- Python's own `round`,
and what these check against in exact rational arithmetic.
"""

from __future__ import annotations

import math
from fractions import Fraction

import numpy as np
import pytest

import minitensor as mt


def _exact(x: float, decimals: int) -> Fraction | None:
    if not math.isfinite(x) or x == 0:
        return None
    scale = Fraction(10) ** decimals
    return Fraction(round(Fraction(x) * scale)) / scale


def _float64(x: float, decimals: int) -> float:
    value = _exact(x, decimals)
    if value is None:
        return x
    return float(value) if value else math.copysign(0.0, x)


def _float32(x: np.float32, decimals: int) -> np.float32:
    value = _exact(float(x), decimals)
    if value is None:
        return x
    if not value:
        return np.float32(math.copysign(0.0, float(x)))
    near = np.float32(float(value))
    candidates = [
        near,
        np.nextafter(near, np.float32(np.inf)),
        np.nextafter(near, np.float32(-np.inf)),
    ]
    return min(candidates, key=lambda c: abs(Fraction(float(c)) - value))


def _same(got, want) -> bool:
    return got == want and math.copysign(1, got) == math.copysign(1, want)


def _values(decimals: int) -> np.ndarray:
    rng = np.random.default_rng(193 + decimals)
    halves = (np.round(rng.normal(0, 1000, 200)) + 0.5) / 10.0 ** max(
        0, min(decimals, 15)
    )
    return np.concatenate(
        [
            rng.normal(0, 3, 200),
            np.exp(rng.uniform(-50, 50, 200)) * rng.choice([-1, 1], 200),
            halves,
            [2.675, 1.005, 0.125, 0.375, -2.5, 2.5, 0.5, -0.5, 1e300, -1e-300, 5e-324],
        ]
    )


@pytest.mark.parametrize("decimals", [-23, -10, -3, -1, 1, 2, 3, 7, 15, 22, 23, 30])
def test_float64_is_the_float_nearest_the_exact_rounding(decimals):
    values = _values(decimals)
    got = mt.round(mt.Tensor(values, dtype="float64"), decimals=decimals).numpy()
    for value, have in zip(values, got):
        assert _same(float(have), _float64(float(value), decimals)), (value, decimals)


@pytest.mark.parametrize("decimals", [-23, -3, -1, 1, 2, 4, 9, 22, 30])
def test_float32_is_the_float32_nearest_the_exact_rounding(decimals):
    with np.errstate(over="ignore"):
        values = _values(decimals).astype(np.float32)
    got = mt.round(mt.Tensor(values), decimals=decimals).numpy()
    for value, have in zip(values, got):
        assert _same(float(have), float(_float32(value, decimals))), (value, decimals)


def test_the_values_that_scaling_rounded_the_wrong_way():
    got = mt.round(mt.Tensor([2.675, 1.005, -2.675, 0.285], dtype="float64"), 2)
    assert got.tolist() == [round(v, 2) for v in (2.675, 1.005, -2.675, 0.285)]
    assert got.tolist()[0] == 2.67


def test_signs_and_non_finite_values_survive():
    got = mt.round(
        mt.Tensor([-0.004, -0.0, math.inf, -math.inf, math.nan], dtype="float64"), 2
    ).tolist()
    assert math.copysign(1, got[0]) == -1 and math.copysign(1, got[1]) == -1
    assert got[2:4] == [math.inf, -math.inf] and math.isnan(got[4])
