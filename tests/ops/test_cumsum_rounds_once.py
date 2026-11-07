# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""Every running sum of a float `cumsum` is its exact prefix sum rounded once.

A float64 running total rounded at every step and drifted from the exact
one: of two hundred short runs of ordinary data, every one held a wrong
prefix. A float32 total carried in float64 was rounded again on the way back,
and lost what fell below float64's last bit where the values cancelled. Each
running sum is now the float nearest the exact sum of the values so far, and
so is each entry of the gradient, which runs the sum the other way.
"""

from __future__ import annotations

import math
from fractions import Fraction

import numpy as np
import pytest

import minitensor as mt

# Every float64 is a whole multiple of 2^-1074.
_SCALE = 2**1074


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


def _running(values, dtype: str) -> list[float]:
    """Each exact prefix sum, rounded once."""

    total, out = 0, []
    for value in np.asarray(values, np.float64).tolist():
        numerator, denominator = value.as_integer_ratio()
        total += numerator * (_SCALE // denominator)
        exact = Fraction(total, _SCALE)
        out.append(_float32(exact) if dtype == "float32" else float(exact))
    return out


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_short_runs_round_once(dtype):
    rng = np.random.default_rng(419)
    for _ in range(60):
        n = int(rng.integers(1, 400))
        values = rng.normal(loc=float(rng.choice([0.0, 3.0, 1e3])), size=n)
        values = values.astype(dtype)
        got = mt.Tensor(values, dtype=dtype).cumsum(0).tolist()
        assert got == _running(values, dtype)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_values_that_cancel_across_the_range(dtype):
    rng = np.random.default_rng(421)
    values = rng.normal(size=500) * 2.0 ** rng.integers(-30, 30, size=500)
    values[1::2] = -values[::2][: len(values[1::2])] * (1 + 2.0**-20)
    values = values.astype(dtype)
    assert mt.Tensor(values, dtype=dtype).cumsum(0).tolist() == _running(values, dtype)
    # A float32 halfway point in float64, and a hair past it.
    halfway = np.array([1.0, 2**-24, 2**-60], np.float32)
    assert mt.Tensor(halfway).cumsum(0).tolist() == [1.0, 1.0, 1 + 2**-23]


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_every_axis_and_layout(dtype):
    rng = np.random.default_rng(431)
    values = rng.normal(loc=1.0, size=(6, 70, 90)).astype(dtype)
    tensor = mt.Tensor(values, dtype=dtype)
    for dim in range(3):
        got = np.asarray(tensor.cumsum(dim).tolist())
        moved = np.moveaxis(values, dim, -1)
        want = np.array(
            [_running(run, dtype) for run in moved.reshape(-1, moved.shape[-1])]
        )
        assert np.array_equal(
            np.moveaxis(got, dim, -1).reshape(-1, moved.shape[-1]), want
        ), dim


def test_a_long_run_split_across_the_pool():
    rng = np.random.default_rng(433)
    values = rng.normal(size=200_000)
    # A stretch that cancels everything before it.
    values[100_000] = -values[:100_000].sum()
    got = mt.Tensor(values, dtype="float64").cumsum(0).numpy()
    want = _running(values, "float64")
    for index in list(range(0, len(values), 997)) + list(range(99_995, 100_005)):
        assert got[index] == want[index], index
    narrow = values.astype(np.float32)
    got = mt.Tensor(narrow).cumsum(0).numpy()
    want = _running(narrow, "float32")
    for index in range(0, len(narrow), 997):
        assert got[index] == want[index], index


def test_specials_and_signed_zeros():
    def cumsum(values, dtype="float64"):
        return mt.Tensor(values, dtype=dtype).cumsum(0).tolist()

    largest = float(np.finfo(np.float64).max)
    assert cumsum([largest, largest, -largest]) == [largest, math.inf, largest]
    got = cumsum([1.0, math.inf, 2.0, -math.inf, 3.0])
    assert got[:3] == [1.0, math.inf, math.inf]
    assert math.isnan(got[3]) and math.isnan(got[4])
    assert math.isnan(cumsum([math.nan, 1.0])[1])
    signs = [math.copysign(1.0, v) for v in cumsum([-0.0, -0.0, 1.0, -1.0])]
    assert signs == [-1.0, -1.0, 1.0, 1.0]
    assert mt.nancumsum(mt.Tensor([1.0, math.nan, 2.0], dtype="float64")).tolist() == [
        1.0,
        1.0,
        3.0,
    ]


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_the_gradient_runs_the_sum_backwards(dtype):
    rng = np.random.default_rng(439)
    weights = rng.normal(loc=2.0, size=300).astype(dtype)
    values = mt.Tensor(np.zeros(300, dtype), dtype=dtype, requires_grad=True)
    (values.cumsum(0) * mt.Tensor(weights, dtype=dtype)).sum().backward()
    want = _running(weights[::-1], dtype)[::-1]
    assert values.grad.tolist() == want
