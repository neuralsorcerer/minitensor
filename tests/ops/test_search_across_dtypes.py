# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""A search answers exactly whatever the dtypes of the sequence and the values.

A search is a comparison, and comparisons here are exact across dtypes. The
search refused any pair that differed -- float32 values against a float64
sequence, `1.5` against integers -- though every answer is plain. Two floats
or two integers now meet at the wider of the two, which holds both, as does a
bool beside anything. An integer against a float has no dtype that holds both
(int64 is not a subset of float64 past 2^53), so it is searched across the
two: a float value among integers is placed by `ceil` or `floor`, and an
integer among floats by the largest float at most it.
"""

from __future__ import annotations

import bisect
from fractions import Fraction

import numpy as np
import pytest

import minitensor as mt


def _key(value):
    # NaN sorts after every number, as `sort` places it.
    if isinstance(value, float) and value != value:
        return (1, 0)
    if isinstance(value, float) and abs(value) == float("inf"):
        return (0, value)
    return (0, Fraction(value))


def _exact(sequence, values, right):
    """The positions by exact arithmetic, which Python's numbers give."""

    keys = [_key(v) for v in sequence]
    search = bisect.bisect_right if right else bisect.bisect_left
    return [search(keys, _key(v)) for v in values]


def _typed(values, dtype):
    return mt.Tensor(np.array(values, dtype=dtype), dtype=dtype)


INF, NAN = float("inf"), float("nan")

CASES = [
    ([1, 2, 3], "int64", [1.5, 2.0, -0.5, 3.0, 3.5, INF, -INF, NAN], "float64"),
    ([1, 2, 3], "int32", [1.5, 2.0, 2.5, 0.0], "float32"),
    # Past 2^53, where the integers are not all floats.
    ([2**53, 2**53 + 1, 2**53 + 2], "int64", [2.0**53, 2.0**53 + 2], "float64"),
    (
        [-(2**63), 0, 2**63 - 1],
        "int64",
        [-(2.0**63), 9.3e18, -9.3e18, 2.0**63],
        "float64",
    ),
    (
        [0.5, 1.0, 2.0**53, 2.0**53 + 2, NAN],
        "float64",
        [0, 1, 2**53, 2**53 + 1, 2**53 + 2, 2**63 - 1, -(2**63)],
        "int64",
    ),
    (
        [0.5, 16_777_216.0, 16_777_218.0],
        "float32",
        [16_777_217, 16_777_216, 1],
        "int32",
    ),
    ([0.1, 0.5], "float64", [0.5, 0.1], "float32"),
    ([0.1, 0.5], "float32", [0.1, 0.5], "float64"),
    ([1, 5], "int32", [5, 2**40], "int64"),
    ([False, True], "bool", [0, 1, 2], "int64"),
    ([0.0, 0.5, 1.0], "float32", [False, True], "bool"),
]


@pytest.mark.parametrize("right", [False, True], ids=["left", "right"])
@pytest.mark.parametrize(
    "sequence,sequence_dtype,values,values_dtype",
    CASES,
    ids=[f"{c[1]}-by-{c[3]}" for c in CASES],
)
def test_every_pair_of_dtypes_is_searched_exactly(
    sequence, sequence_dtype, values, values_dtype, right
):
    ordered = _typed(sequence, sequence_dtype)
    probes = _typed(values, values_dtype)
    got = mt.searchsorted(ordered, probes, right=right).tolist()
    assert got == _exact(ordered.tolist(), probes.tolist(), right)


def test_bucketize_and_a_batched_sequence_across_dtypes():
    boundaries = _typed([1, 2, 3], "int64")
    assert mt.bucketize(_typed([1.5, 3.0], "float32"), boundaries).tolist() == [1, 2]
    assert mt.bucketize(
        _typed([1.5, 3.0], "float32"), boundaries, right=True
    ).tolist() == [1, 3]
    rows = _typed([[1, 2, 3], [10, 20, 30]], "int32")
    probes = _typed([[2.5], [15.0]], "float64")
    assert mt.searchsorted(rows, probes).tolist() == [[2], [1]]


def test_a_python_list_against_an_integer_sequence():
    # Read at the sequence's width, `1.5` beside int64 is a float64 and is
    # now placed rather than refused.
    assert mt.searchsorted(_typed([1, 2, 3], "int64"), [1.5, 2.5]).tolist() == [1, 2]


def test_the_comparison_methods_take_a_list():
    wide = _typed([0.1, 0.5], "float64")
    assert wide.allclose([0.1, 0.5], rtol=0, atol=0)
    assert wide.array_equal([0.1, 0.5])
    assert not wide.array_equal([0.1, 0.6])
    # An int list read exactly beside a float, as the comparisons read it.
    single = _typed([16_777_216.0], "float32")
    assert not single.array_equal([16_777_217])
    assert not single.allclose([16_777_217], rtol=0, atol=0)
