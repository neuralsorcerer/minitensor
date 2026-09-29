# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""A Python int reaches an integer tensor exactly, whichever call carries it.

An f64 holds every integer only up to 2^53, and `full`, `full_like`,
`fill_`, `clamp` and `arange` took their values as one: they wrote 2^60 for
2^60 + 1, and `arange` between two such bounds rounded both to the same value
and came back empty. Arithmetic, comparison and assignment were already exact;
these now take the same path.
"""

from __future__ import annotations

import numpy as np
import pytest

import minitensor as mt

BIG = 2**60 + 1


def test_full_and_full_like_write_the_value_exactly():
    assert mt.full((2,), BIG, dtype="int64").tolist() == [BIG, BIG]
    assert mt.full_like(mt.zeros(2, dtype="int64"), BIG).tolist() == [BIG, BIG]


def test_fill_writes_the_value_exactly():
    assert mt.zeros(3, dtype="int64").fill_(BIG).tolist() == [BIG] * 3


def test_clamp_bounds_are_exact():
    values = mt.Tensor([0, BIG + 5], dtype="int64")
    assert values.clamp(max=BIG).tolist() == [0, BIG]
    assert values.clamp(min=BIG).tolist() == [BIG, BIG + 5]
    assert values.clamp(min=BIG, max=BIG).tolist() == [BIG, BIG]
    with pytest.raises(ValueError, match="cannot be greater than maximum"):
        values.clamp(min=3, max=1)


def test_arange_counts_exactly_between_large_bounds():
    assert mt.arange(BIG, BIG + 3, dtype="int64").tolist() == [BIG, BIG + 1, BIG + 2]
    assert mt.arange(BIG, BIG - 4, -2, dtype="int64").tolist() == [BIG, BIG - 2]


@pytest.mark.parametrize(
    "args",
    [(5,), (2, 9, 3), (9, 2, -3), (3, 3), (5, 1), (-7, 7, 5), (10, 0, -4)],
)
@pytest.mark.parametrize("dtype", ["int32", "int64"])
def test_integer_arange_matches_its_definition(args, dtype):
    np.testing.assert_array_equal(
        mt.arange(*args, dtype=dtype).numpy(), np.arange(*args, dtype=dtype)
    )


def test_float_arguments_and_dtypes_keep_their_path():
    assert mt.arange(3).dtype == "float32"
    np.testing.assert_allclose(mt.arange(0, 1, 0.25).numpy(), [0.0, 0.25, 0.5, 0.75])
    assert mt.arange(0, 3, 0.5, dtype="int64").tolist() == [0, 0, 1, 1, 2, 2]
    assert mt.zeros(2, dtype="int64").fill_(2.7).tolist() == [2, 2]
    assert mt.Tensor([0, 5, 9], dtype="int64").clamp(2.5, 7.5).tolist() == [2, 5, 7]
