# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""Construction refuses Python data it cannot represent, rather than guessing.

A scalar beside a sequence at the same level broke neither length check, so
`[[1, 2], 3]` became a shape [2, 2] tensor over three values. A Python int
has no width of its own and takes the dtype it is given; one that does not
fit int32 kept its low 32 bits, so 2**40 became 0, and one past int64 was
read as a float and saturated to the int64 bound. A float dtype holds either.
"""

from __future__ import annotations

import numpy as np
import pytest

import minitensor as mt

RAGGED = [
    [[1, 2], 3],
    [3, [1, 2]],
    [[1], [[2]]],
    [[[1]], [2]],
    [[], 3],
    ([1, 2], 3),
]


@pytest.mark.parametrize("data", RAGGED, ids=repr)
@pytest.mark.parametrize("dtype", [None, "int64", "float64"])
def test_a_scalar_beside_a_sequence_is_refused(data, dtype):
    with pytest.raises(ValueError, match="Inconsistent nested sequence"):
        mt.tensor(data, dtype=dtype)


def test_regular_nesting_still_builds():
    assert mt.tensor(([1, 2], (3, 4))).tolist() == [[1.0, 2.0], [3.0, 4.0]]
    assert mt.tensor([[], []]).shape == (2, 0)
    assert mt.tensor([[[1], [2]]], dtype="int64").tolist() == [[[1], [2]]]


BIG = 2**40


@pytest.mark.parametrize(
    "build",
    [
        lambda: mt.tensor(BIG, dtype="int32"),
        lambda: mt.tensor([1, BIG], dtype="int32"),
        lambda: mt.tensor([[1], [-BIG]], dtype="int32"),
        lambda: mt.full((2,), BIG, dtype="int32"),
        lambda: mt.zeros(2, dtype="int32").fill_(BIG),
        lambda: mt.zeros(2, dtype="int32") + BIG,
        lambda: mt.arange(0, BIG, 2**39, dtype="int32"),
        lambda: mt.arange(-(2**31) - 1, 0, 2**30, dtype="int32"),
    ],
    ids=["scalar", "list", "nested", "full", "fill_", "add", "arange", "arange-start"],
)
def test_a_python_int_outside_int32_is_refused(build):
    with pytest.raises(OverflowError, match="does not fit in int32"):
        build()


def test_int32_bounds_are_held_exactly():
    low, high = -(2**31), 2**31 - 1
    assert mt.tensor([low, high], dtype="int32").tolist() == [low, high]
    assert mt.full((1,), low, dtype="int32").tolist() == [low]
    assert mt.arange(high - 2, high + 1, dtype="int32").tolist() == [
        high - 2,
        high - 1,
        high,
    ]


def test_an_int64_array_converts_as_a_cast():
    # An array carries a dtype of its own, so asking for another is a cast.
    wrapped = np.array([BIG + 5]).astype(np.int32)
    assert mt.tensor(np.array([BIG + 5]), dtype="int32").tolist() == wrapped.tolist()


HUGE = 2**70


@pytest.mark.parametrize(
    "build",
    [
        lambda: mt.tensor(HUGE, dtype="int64"),
        lambda: mt.tensor([1, HUGE], dtype="int64"),
        lambda: mt.full((1,), HUGE, dtype="int64"),
        lambda: mt.full_like(mt.zeros(1, dtype="int64"), HUGE),
        lambda: mt.zeros(1, dtype="int64").fill_(-HUGE),
        lambda: mt.zeros(1, dtype="int64") - HUGE,
        lambda: mt.arange(HUGE, HUGE + 3, dtype="int64"),
    ],
    ids=["scalar", "list", "full", "full_like", "fill_", "sub", "arange"],
)
def test_a_python_int_outside_int64_is_refused(build):
    with pytest.raises(OverflowError, match="does not fit in int64"):
        build()


def test_a_float_dtype_holds_a_python_int_past_int64():
    expected = float(HUGE)
    assert mt.tensor(HUGE, dtype="float64").item() == expected
    assert mt.tensor([1, HUGE], dtype="float64").tolist() == [1.0, expected]
    assert mt.tensor([[1], [HUGE]]).dtype == mt.tensor(HUGE).dtype
    assert mt.full((1,), HUGE, dtype="float64").tolist() == [expected]
    assert (mt.zeros(1, dtype="float64") + HUGE).tolist() == [expected]
