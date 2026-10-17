# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""`x ** 3` and `x ** 0.5` are each the float nearest the exact answer.

`x ** 3` was `x * x * x`, which rounds twice: a quarter of all float32 and
float64 inputs came back an ulp from the float nearest their cube. Float32 now
cubes in float64, and float64 recovers each product's rounding error and adds
it back before the last rounding. `x ** 0.5` went through `exp` and `log` at
seven times the cost of the square root, which is rounded once; it is now the
square root, at the two places IEEE says the power differs from it.
"""

from __future__ import annotations

import math
from fractions import Fraction

import numpy as np
import pytest

import minitensor as mt


def _nearest_float32(exact):
    near = np.float32(float(exact))
    candidates = [
        near,
        np.nextafter(near, np.float32(np.inf)),
        np.nextafter(near, np.float32(-np.inf)),
    ]
    return min(candidates, key=lambda c: abs(Fraction(float(c)) - exact))


def test_a_float32_cube_is_the_float32_nearest_it():
    rng = np.random.default_rng(79)
    values = np.concatenate(
        [
            rng.uniform(0.5, 2, 3000) * rng.choice([1, -1], 3000),
            np.exp(rng.uniform(-29, 29, 1000)) * rng.choice([1, -1], 1000),
        ]
    ).astype(np.float32)
    got = (mt.Tensor(values) ** 3).numpy()
    for x, have in zip(values, got):
        assert have == _nearest_float32(Fraction(float(x)) ** 3), x


def _nearest_float64(exact):
    try:
        return float(exact)
    except OverflowError:  # Past the largest float, which rounds to infinity.
        return math.inf if exact > 0 else -math.inf


@pytest.mark.parametrize(
    "low,high",
    [(-1, 1), (-500, 500), (-341, -290), (-358.5, -340), (330, 341.5)],
    ids=["unit", "wide", "scaled", "subnormal-cube", "near-overflow"],
)
def test_a_float64_cube_is_the_float_nearest_it(low, high):
    rng = np.random.default_rng(83)
    values = np.exp2(rng.uniform(low, high, 2000)) * rng.choice([1, -1], 2000)
    got = (mt.Tensor(values, dtype="float64") ** 3).numpy()
    for x, have in zip(values, got):
        assert have == _nearest_float64(Fraction(float(x)) ** 3), x.hex()


def test_the_cubes_that_two_roundings_missed():
    # Each of these is an ulp off as `x * x * x`.
    for x, cube in [
        ("0x1.3b4a07ee8237dp-1", "0x1.de3dd4c2284c0p-3"),
        ("0x1.eb715f40b2ccdp+0", "0x1.c4c586c99dc47p+2"),
        ("0x1.81b6ea1ef884ep-342", "0x0.36b9febb61f55p-1022"),
    ]:
        x = float.fromhex(x)
        assert x * x * x != float.fromhex(cube)
        got = (mt.Tensor([x], dtype="float64") ** 3).item()
        assert got == float.fromhex(cube)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_a_cube_keeps_its_special_values(dtype):
    inf, nan = math.inf, math.nan
    got = (mt.Tensor([-0.0, 0.0, inf, -inf, nan, 1e300], dtype=dtype) ** 3).tolist()
    assert [math.copysign(1, v) for v in got[:2]] == [-1.0, 1.0]
    assert got[2:4] == [inf, -inf] and math.isnan(got[4]) and got[5] == inf


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_a_half_power_is_the_square_root(dtype):
    values = np.abs(np.random.default_rng(89).normal(0, 100, 10_000)).astype(dtype)
    np.testing.assert_array_equal((mt.Tensor(values) ** 0.5).numpy(), np.sqrt(values))


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_a_half_power_differs_from_the_root_where_the_power_does(dtype):
    inf = math.inf
    got = (mt.Tensor([-0.0, -inf, -4.0, inf, 0.0], dtype=dtype) ** 0.5).tolist()
    # The power of -0 is +0, and of -inf is +inf; the root of either is not.
    assert got[0] == 0.0 and math.copysign(1, got[0]) == 1.0
    assert got[1] == inf and math.isnan(got[2]) and got[3:] == [inf, 0.0]


def test_the_gradients_are_unchanged():
    base = mt.Tensor([1.0, 4.0, -2.0], requires_grad=True)
    (base**3).sum().backward()
    assert base.grad.tolist() == [3.0, 48.0, 12.0]
    root = mt.Tensor([1.0, 4.0], requires_grad=True)
    (root**0.5).sum().backward()
    assert root.grad.tolist() == [0.5, 0.25]
