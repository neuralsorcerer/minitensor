# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""A Python number written against a tensor keeps every digit it was written with.

A bare `0.1` has no dtype, so one is chosen for it. Choosing the *default*
dtype and widening afterwards is not the same as choosing the tensor's: float32
cannot hold `0.1`, and widening the float32 nearest-neighbour back to float64
gives `0.10000000149011612` -- a different number from the one in the source,
wrong in the eighth digit, in an expression the reader would call exact.
"""

import numpy as np
import pytest

import minitensor as mt

# Values that survive float64 exactly and do not survive float32: each needs
# more than 24 bits of mantissa.
AWKWARD = [
    0.1,
    10.798325661547178,
    1.0 / 3.0,
    np.pi,
    2.0**-30 + 1.0,
    1e-40,
]

OPERATORS = [
    ("mul", lambda t, v: t * v, lambda a, v: a * v),
    ("rmul", lambda t, v: v * t, lambda a, v: v * a),
    ("add", lambda t, v: t + v, lambda a, v: a + v),
    ("sub", lambda t, v: t - v, lambda a, v: a - v),
    ("rsub", lambda t, v: v - t, lambda a, v: v - a),
    ("div", lambda t, v: t / v, lambda a, v: a / v),
    ("rdiv", lambda t, v: v / t, lambda a, v: v / a),
]


@pytest.mark.parametrize("value", AWKWARD, ids=repr)
@pytest.mark.parametrize("name, applied, reference", OPERATORS, ids=lambda x: x)
def test_a_python_float_against_a_float64_tensor_is_exact(
    value, name, applied, reference
):
    del name
    base = np.array([1.0, 2.0, 7.0])
    tensor = mt.Tensor(base.copy(), dtype="float64")

    got = applied(tensor, value).numpy()
    assert got.dtype == np.float64
    np.testing.assert_array_equal(got, reference(base, value))


@pytest.mark.parametrize("value", AWKWARD, ids=repr)
def test_a_python_float_in_a_list_is_exact_too(value):
    base = np.array([1.0, 2.0, 7.0])
    tensor = mt.Tensor(base.copy(), dtype="float64")

    for operand in ([value] * 3, (value,) * 3, [value]):
        got = (tensor * operand).numpy()
        assert got.dtype == np.float64
        np.testing.assert_array_equal(got, base * np.asarray(operand, dtype=np.float64))


def test_a_python_int_past_the_float32_mantissa_survives():
    # 2**24 + 1 is the first integer float32 cannot represent, and every
    # integer past 2**53 is beyond float64's reach as well -- so this is the
    # whole range where an int64 tensor is the only thing that can hold the
    # answer.
    for value in (2**24 + 1, 2**40 + 3, 2**53 - 1):
        tensor = mt.Tensor(np.array([0], dtype=np.int64), dtype="int64")
        assert (tensor + value).numpy()[0] == value
        assert (tensor + [value]).numpy()[0] == value


def test_a_float32_tensor_still_gets_a_float32_operand():
    # The rule is the tensor's width, not the widest available: a float32
    # tensor times 0.1 is the float32 product, and promoting the scalar would
    # quietly make the whole expression float64.
    tensor = mt.Tensor(np.array([1.0], dtype=np.float32), dtype="float32")
    assert "float32" in str((tensor * 0.1).dtype)
    assert "float32" in str((tensor * [0.1]).dtype)
    assert (tensor * 0.1).numpy()[0] == np.float32(1.0) * np.float32(0.1)


def test_a_numpy_array_keeps_the_dtype_its_caller_chose():
    # An array carries a declared dtype, and that outranks the context: a
    # float32 array against a float64 tensor is the float32 value promoted, not
    # a re-reading of the literal at float64.
    tensor = mt.Tensor(np.array([1.0]), dtype="float64")
    narrow = np.array([0.1], dtype=np.float32)

    got = (tensor * narrow).numpy()
    assert got.dtype == np.float64
    np.testing.assert_array_equal(got, np.array([1.0]) * narrow.astype(np.float64))
    assert got[0] != 0.1, "a float32 array cannot carry the float64 0.1"


def test_comparisons_see_the_same_number_the_arithmetic_does():
    # The comparison operators go through the same conversion, so `x == 0.1`
    # has to agree with what `x * 1.0` would have produced.
    tensor = mt.Tensor(np.array([0.1, 0.2]), dtype="float64")
    np.testing.assert_array_equal((tensor == 0.1).numpy(), [True, False])
    np.testing.assert_array_equal((tensor > 0.1).numpy(), [False, True])


def test_the_scalar_reaches_a_gradient_undamaged():
    tensor = mt.Tensor(np.array([1.0]), dtype="float64", requires_grad=True)
    (tensor * 0.1).sum().backward()
    assert tensor.grad.numpy()[0] == 0.1
