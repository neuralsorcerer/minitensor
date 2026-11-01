# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""`isclose` and `allclose` test `|a - b| <= atol + rtol * |b|` exactly.

Each operand is the number it holds and the tolerances are the floats given.
Measured in a float dtype, an int64 past 2^53 became the float64 beside it
and was close to it with no tolerance at all; two integers ignored the
tolerances, so `5` was not within `atol=2` of `6`; float32 operands were
measured against tolerances narrowed to float32; and `allclose` did not
broadcast, so `allclose([1.0, 1.0], 1.0)` was `False` where every element of
`isclose` was `True`.
"""

from __future__ import annotations

import math
import random
from fractions import Fraction

import numpy as np
import pytest

import minitensor as mt


def _close(a, b, rtol, atol):
    return abs(Fraction(a) - Fraction(b)) <= Fraction(atol) + Fraction(rtol) * abs(
        Fraction(b)
    )


def test_an_int64_past_2_53_is_its_own_value():
    big = mt.tensor([2**60 + 1], dtype="int64")
    near = mt.tensor([float(2**60)], dtype="float64")
    assert mt.isclose(big, near, rtol=0.0, atol=0.0).tolist() == [False]
    assert not mt.allclose(big, near, rtol=0.0, atol=0.0)
    assert mt.isclose(big, near, rtol=0.0, atol=1.0).tolist() == [True]


def test_integers_are_measured_against_the_tolerances():
    ints = mt.tensor([100000, 5, -7])
    others = mt.tensor([100001, 6, -9])
    assert mt.isclose(ints, others).tolist() == [True, False, False]
    assert mt.isclose(ints, others, atol=2.0).tolist() == [True, True, True]
    assert mt.allclose(mt.tensor([5]), mt.tensor([6]), atol=2.0)


def test_float32_operands_meet_the_tolerance_given():
    one, near = mt.tensor([1.0]), mt.tensor([1.1])
    gap = float(np.float32(1.1)) - 1.0
    assert mt.isclose(one, near, rtol=0.0, atol=gap).tolist() == [True]
    below = math.nextafter(gap, 0.0)
    assert mt.isclose(one, near, rtol=0.0, atol=below).tolist() == [False]


@pytest.mark.parametrize("dtype", ["float64", "float32"])
def test_the_edge_is_decided_exactly(dtype):
    rng = random.Random(1709)
    left, right, rtols, atols, expected = [], [], [], [], []
    for _ in range(3000):
        b = rng.uniform(-10, 10)
        rtol, atol = rng.choice([(1e-5, 1e-8), (0.0, 0.1), (0.25, 0.0)])
        allowed = atol + rtol * abs(b)
        a = b + rng.choice([allowed, -allowed, allowed * (1 + 2**-52), 0.5 * allowed])
        if dtype == "float32":
            a, b = float(np.float32(a)), float(np.float32(b))
        left.append(a)
        right.append(b)
        rtols.append(rtol)
        atols.append(atol)
        expected.append(_close(a, b, rtol, atol))
    for a, b, rtol, atol, want in zip(left, right, rtols, atols, expected):
        got = mt.isclose(
            mt.tensor([a], dtype=dtype),
            mt.tensor([b], dtype=dtype),
            rtol=rtol,
            atol=atol,
        ).tolist()
        assert got == [want], (a, b, rtol, atol)


def test_allclose_broadcasts_as_isclose_does():
    values = mt.tensor([1.0, 1.0])
    assert mt.isclose(values, 1.0).tolist() == [True, True]
    assert mt.allclose(values, 1.0)
    assert values.allclose(mt.tensor(1.0))
    assert not mt.allclose(values, 2.0)
    with pytest.raises(ValueError):
        mt.allclose(values, mt.tensor([1.0, 1.0, 1.0]))


def test_specials():
    assert mt.allclose([0.0, float("inf")], [-0.0, float("inf")])
    assert not mt.allclose([float("inf")], [float("-inf")])
    assert not mt.allclose([float("nan")], [float("nan")])
    assert mt.allclose([float("nan")], [float("nan")], equal_nan=True)
    # Opposite huge values, whose difference overflows float64.
    huge = mt.tensor([1.7e308], dtype="float64")
    assert not mt.isclose(huge, -huge, rtol=1.0).tolist()[0]
    assert mt.isclose(huge, -huge, rtol=2.0).tolist()[0]
