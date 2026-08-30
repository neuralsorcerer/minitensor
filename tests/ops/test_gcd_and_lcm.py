# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""`gcd` and `lcm`, against NumPy's.

Integer-only, like the shifts they share their machinery with, and always
non-negative: a common divisor of `-12` and `8` is a common divisor of `12` and
`8`, and the positive one is what every library reports. The cases worth
pinning are the ones a straightforward implementation gets wrong -- the zeros,
the mixed signs, the value whose magnitude no signed integer can hold, and a
multiple that fits when the product of its operands does not.
"""

from __future__ import annotations

import numpy as np
import pytest

import minitensor as mt

PAIRS = np.array(
    [
        [12, 8],
        [-12, 8],
        [12, -8],
        [-12, -8],
        [0, 5],
        [5, 0],
        [0, 0],
        [17, 5],
        [270, 192],
        [1, 1],
        [-7, -7],
    ]
)


def _i(values, dtype="int64"):
    numpy_dtype = np.int64 if dtype == "int64" else np.int32
    return mt.Tensor(
        np.ascontiguousarray(np.asarray(values, dtype=numpy_dtype)), dtype=dtype
    )


@pytest.mark.parametrize("dtype", ["int32", "int64"])
def test_gcd_and_lcm_match_numpy(dtype):
    left, right = PAIRS[:, 0], PAIRS[:, 1]
    np.testing.assert_array_equal(
        mt.gcd(_i(left, dtype), _i(right, dtype)).numpy(), np.gcd(left, right)
    )
    np.testing.assert_array_equal(
        mt.lcm(_i(left, dtype), _i(right, dtype)).numpy(), np.lcm(left, right)
    )


def test_the_answer_is_never_negative():
    left, right = PAIRS[:, 0], PAIRS[:, 1]
    assert (mt.gcd(_i(left), _i(right)).numpy() >= 0).all()
    assert (mt.lcm(_i(left), _i(right)).numpy() >= 0).all()


def test_zero_behaves_as_every_integer_dividing_it():
    # `gcd(x, 0) == |x|` because every integer divides zero, and `lcm(x, 0)`
    # is 0 because zero is the least of the multiples of zero.
    values = np.array([7, -7, 0, 1])
    zeros = np.zeros_like(values)
    np.testing.assert_array_equal(mt.gcd(_i(values), _i(zeros)).numpy(), np.abs(values))
    np.testing.assert_array_equal(mt.lcm(_i(values), _i(zeros)).numpy(), zeros)


def test_gcd_survives_the_value_with_no_positive_magnitude():
    # `abs(int64 min)` has no representation, so the magnitudes are taken
    # unsigned and the one answer that cannot come back is saturated rather
    # than wrapped to a negative divisor.
    smallest = np.iinfo(np.int64).min
    got = mt.gcd(_i([smallest, smallest]), _i([0, 2])).numpy()
    assert got[0] == np.iinfo(np.int64).max
    assert got[1] == 2
    assert (got >= 0).all()


def test_lcm_divides_before_it_multiplies():
    # The product of these leaves int64; their least common multiple does not,
    # and only the order of the arithmetic keeps it.
    left, right = 4_000_000_000, 6_000_000_000
    assert mt.lcm(_i([left]), _i([right])).numpy()[0] == 12_000_000_000
    assert mt.lcm(_i([left]), _i([right])).numpy()[0] == np.lcm(left, right)


def test_gcd_and_lcm_broadcast_and_promote():
    got = mt.gcd(_i([12, 18], "int32"), _i([8], "int64"))
    assert "int64" in str(got.dtype)
    np.testing.assert_array_equal(got.numpy(), [4, 2])


def test_gcd_and_lcm_refuse_what_has_no_divisors():
    floats = mt.Tensor(np.array([1.0, 2.0]), dtype="float64")
    booleans = mt.Tensor(np.array([True, False]), dtype="bool")
    for name in ("gcd", "lcm"):
        with pytest.raises(Exception):
            getattr(mt, name)(floats, floats)
        # Two truth values have no divisors, the same reason they have no bits
        # to shift.
        with pytest.raises(Exception):
            getattr(mt, name)(booleans, booleans)


def test_the_functional_spelling_agrees_with_the_method():
    left, right = _i(PAIRS[:, 0]), _i(PAIRS[:, 1])
    for name in ("gcd", "lcm"):
        np.testing.assert_array_equal(
            getattr(mt, name)(left, right).numpy(),
            getattr(left, name)(right).numpy(),
        )


def test_a_python_int_works_as_the_second_operand():
    np.testing.assert_array_equal(mt.gcd(_i([12, 18]), 8).numpy(), np.gcd([12, 18], 8))


def test_the_two_are_related_the_way_they_are_defined_to_be():
    # gcd(a, b) * lcm(a, b) == |a * b|, for every pair that does not overflow.
    left, right = PAIRS[:, 0], PAIRS[:, 1]
    product = mt.gcd(_i(left), _i(right)).numpy() * mt.lcm(_i(left), _i(right)).numpy()
    np.testing.assert_array_equal(product, np.abs(left * right))
