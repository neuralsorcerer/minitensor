# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""An integer compared with a float is compared exactly.

Comparisons used to promote the pair to a common float first, and the common
float could not hold both: an int32 `20_000_001` and `20_000_000.0` met in
float32 as the same number and compared equal, and so did an int64 above 2^53
and the float64 below it. A comparison answers in booleans, so there is no
result width to keep narrow, and every one of these -- the operators, the
functional forms, `isclose`, `allclose`, `array_equal` and the membership
tests -- now gives the answer the two numbers actually have.
"""

from __future__ import annotations

import operator

import numpy as np
import pytest

import minitensor as mt

BIG32 = 20_000_001  # past 2^24, so float32 cannot hold it
BIG64 = 2**53 + 1  # past 2^53, so float64 cannot hold it

OPERATORS = [
    operator.eq,
    operator.ne,
    operator.lt,
    operator.le,
    operator.gt,
    operator.ge,
]


def _int(values, dtype):
    return mt.Tensor(np.array(values, dtype=dtype), dtype=dtype)


def _float(values, dtype):
    return mt.Tensor(np.array(values, dtype=dtype), dtype=dtype)


@pytest.mark.parametrize("op", OPERATORS, ids=lambda op: op.__name__)
@pytest.mark.parametrize("int_dtype", ["int32", "int64"])
@pytest.mark.parametrize("float_dtype", ["float32", "float64"])
def test_every_operator_agrees_with_exact_arithmetic(op, int_dtype, float_dtype):
    integers = [BIG32, -BIG32, 0, 1, -1, 16_777_217, 2_000_000_000]
    floats = [
        float(np.array(v, dtype=float_dtype))
        for v in (20_000_000.0, -20_000_000.0, -0.0, 0.5, -1.0, 16_777_216.0, 2e9)
    ]
    floats += [float("nan"), float("inf"), float("-inf")]
    ints = _int(integers, int_dtype).reshape([-1, 1])
    fls = _float(floats, float_dtype).reshape([1, -1])
    # Python's own int-float comparison is exact, so it is the reference.
    expected = [[op(i, f) for f in floats] for i in integers]
    assert op(ints, fls).tolist() == expected
    flipped = [[op(f, i) for i in integers] for f in floats]
    assert op(fls.reshape([-1, 1]), ints.reshape([1, -1])).tolist() == flipped


@pytest.mark.parametrize("op", OPERATORS, ids=lambda op: op.__name__)
def test_int64_beyond_float64_precision(op):
    values = [BIG64, 2**53, -BIG64, 2**63 - 1, -(2**63)]
    tensor = _int(values, "int64")
    for scalar in (float(2**53), float(-(2**53)), 9.3e18, -9.3e18, 2.0**63, -(2.0**63)):
        assert op(tensor, scalar).tolist() == [op(v, scalar) for v in values], scalar


@pytest.mark.parametrize("op", OPERATORS, ids=lambda op: op.__name__)
def test_python_floats_beside_an_int32_tensor_keep_their_value(op):
    tensor = _int([BIG32, 3], "int32")
    for scalar in (20_000_000.0, 20_000_001.0, 20_000_000.5, 2.5, 2.0**40):
        assert op(tensor, scalar).tolist() == [op(BIG32, scalar), op(3, scalar)], scalar
        # The reflected form as well.
        assert op(scalar, tensor).tolist() == [op(scalar, BIG32), op(scalar, 3)], scalar


def test_a_python_int_beyond_int32_compares_rather_than_being_refused():
    tensor = _int([BIG32, -5], "int32")
    assert (tensor < 2**40).tolist() == [True, True]
    assert (tensor > -(2**40)).tolist() == [True, True]
    assert (tensor == 2**40).tolist() == [False, False]


def test_the_functional_forms_agree_with_the_operators():
    ints, floats = _int([BIG32], "int32"), _float([20_000_000.0], "float32")
    assert mt.eq(ints, floats).tolist() == [False]
    assert mt.ne(ints, floats).tolist() == [True]
    assert mt.gt(ints, floats).tolist() == [True]
    assert mt.le(floats, ints).tolist() == [True]


def test_closeness_and_equality_of_whole_arrays():
    ints, floats = _int([BIG32, 5], "int32"), _float([20_000_000.0, 5.0], "float32")
    assert mt.isclose(ints, floats, rtol=0, atol=0).tolist() == [False, True]
    assert mt.isclose(ints, floats, rtol=0, atol=1.0).tolist() == [True, True]
    assert not mt.allclose(ints, floats, rtol=0, atol=0)
    assert mt.allclose(ints, floats, rtol=0, atol=1.0)
    assert not mt.array_equal(ints, floats)
    assert mt.array_equal(_int([3, 5], "int32"), _float([3.0, 5.0], "float32"))


def test_membership_is_exact_too():
    ints, floats = _int([BIG32, 5], "int32"), _float([20_000_000.0, 5.0], "float32")
    assert mt.isin(ints, floats).tolist() == [False, True]
    assert mt.intersect1d(ints, floats).tolist() == [5]
    assert mt.setdiff1d(ints, floats).tolist() == [BIG32]
    assert mt.intersect1d(floats, ints).tolist() == [5.0]
    assert mt.setdiff1d(floats, ints).tolist() == [20_000_000.0]


def test_a_float32_tensor_still_compares_with_a_python_float_at_its_own_width():
    # The weak-scalar rule is unchanged where it is right: `x == 0.1` on a
    # float32 tensor asks about the float32 nearest 0.1.
    assert (_float([0.1], "float32") == 0.1).tolist() == [True]


@pytest.mark.parametrize("op", OPERATORS, ids=lambda op: op.__name__)
def test_a_python_int_beside_a_float_tensor_keeps_its_value(op):
    # Read at the float's width, 16777217 became 16777216.0, which a float32
    # holding 16777216.0 then equalled. An int literal is an exact integer.
    single = _float([16_777_216.0], "float32")
    double = _float([2.0**53], "float64")
    for scalar in (16_777_216, 16_777_217, 16_777_215):
        assert op(single, scalar).tolist() == [op(16_777_216.0, scalar)], scalar
        assert op(scalar, single).tolist() == [op(scalar, 16_777_216.0)], scalar
    for scalar in (2**53, 2**53 + 1):
        assert op(double, scalar).tolist() == [op(2.0**53, scalar)], scalar
