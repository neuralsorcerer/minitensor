# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""`rsqrt` is the float nearest `1 / sqrt(x)`.

A float64 `rsqrt` was the root and then its reciprocal, two roundings, and a
quarter of the answers across the range were not the float64 nearest the
exact one. Each is now that float: a Newton step on the two, carried as a
pair and settled by a bound, and placed exactly beside a halfway point where
the bound cannot say. Float32 already was, for every positive float32.
"""

from __future__ import annotations

import math
from fractions import Fraction

import numpy as np
import pytest

import minitensor as mt


def _nearest(x: float, r: float) -> bool:
    """Whether the float64 `r` is the one nearest `1 / sqrt(x)`: the exact
    root lies between the halfway points either side of `r`."""

    low = (Fraction(r) + Fraction(math.nextafter(r, 0.0))) / 2
    high = (Fraction(r) + Fraction(math.nextafter(r, math.inf))) / 2
    exact = Fraction(x)
    return low * low * exact <= 1 <= high * high * exact


@pytest.mark.parametrize("kind", ["near one", "across the range", "subnormal"])
def test_float64_answers_are_the_nearest(kind):
    rng = np.random.default_rng(641)
    if kind == "near one":
        values = rng.uniform(0.25, 4.0, 4000)
    elif kind == "across the range":
        values = np.exp(rng.uniform(-705.0, 709.0, 4000))
    else:
        values = rng.integers(1, 2**52, 2000).astype(np.float64) * 2.0**-1074
    got = mt.Tensor(values, dtype="float64").rsqrt().tolist()
    assert all(_nearest(float(x), r) for x, r in zip(values, got))


def test_ends_of_the_range_and_powers_of_four():
    values = [5e-324, 2.2250738585072014e-308, 1.7976931348623157e308, 3.0, 0.5]
    for power in (-537, -100, -1, 0, 1, 100, 511):
        four = 4.0**power
        values += [four, math.nextafter(four, 0.0), math.nextafter(four, math.inf)]
    values = [value for value in values if value > 0.0]
    got = mt.tensor(values, dtype="float64").rsqrt().tolist()
    assert all(_nearest(x, r) for x, r in zip(values, got))
    # Powers of four have exact roots.
    assert mt.tensor([4.0, 0.25, 4.0**-537], dtype="float64").rsqrt().tolist() == [
        0.5,
        2.0,
        2.0**537,
    ]


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_specials(dtype):
    nan, inf = math.nan, math.inf
    got = mt.tensor([0.0, -0.0, inf, -inf, nan, -1.0], dtype=dtype).rsqrt().tolist()
    assert got[0] == inf and got[1] == -inf and got[2] == 0.0
    assert all(math.isnan(v) for v in got[3:])
