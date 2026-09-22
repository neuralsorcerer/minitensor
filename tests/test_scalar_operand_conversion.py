# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""What a non-tensor operand becomes, and at what width.

`x + 1.0` has to decide two things about the `1.0`: which of the handful of
Python and NumPy scalar kinds it is, and what dtype the tensor built from it
should have. Both answers are load-bearing and neither is obvious.

The width rule is the subtle one. A bare Python `float` has no width of its
own, so it takes the tensor's -- `x + 0.1` on a float32 tensor stays float32.
A `np.float64` *does* have one and keeps it, which matters because
`np.float64` is a subclass of `float`: any classification that asks
`isinstance` rather than for the exact type puts the two in the same branch
and silently narrows `x + np.float64(0.1)` to the float32 nearest 0.1.

The classification is ordered for speed -- the exact builtin types are tested
before anything is asked of NumPy, and before any extraction is attempted,
because a failed extraction raises and building a Python exception to discard
it is not cheap. That order is invisible when it is right and produces wrong
dtypes when it is wrong, so it is pinned here rather than left to the
operations' own tests.
"""

import numpy as np
import pytest

import minitensor as mt

BASE_DTYPES = ["float32", "float64", "int32", "int64"]


def _tensor(dtype):
    return mt.from_numpy(np.arange(1, 4).astype(dtype))


def _reference(dtype):
    return np.arange(1, 4).astype(dtype).astype(np.float64)


class FloatSubclass(float):
    pass


class IntSubclass(int):
    pass


class ConvertsToFloat:
    def __float__(self):
        return 0.25


class ConvertsToIndex:
    def __index__(self):
        return 3


@pytest.mark.parametrize("dtype", BASE_DTYPES)
@pytest.mark.parametrize(
    "operand,value",
    [
        (0.5, 0.5),
        (2, 2),
        (True, 1),
        (np.float64(0.5), 0.5),
        (np.float32(0.5), 0.5),
        (np.int64(2), 2),
        (np.int32(2), 2),
        (np.bool_(True), 1),
        (np.array(2.0), 2.0),
        (FloatSubclass(0.5), 0.5),
        (IntSubclass(2), 2),
        (ConvertsToFloat(), 0.25),
        (ConvertsToIndex(), 3),
    ],
)
def test_every_scalar_kind_reaches_the_same_answer(dtype, operand, value):
    """One value, thirteen spellings. Each takes a different branch of the
    classification and they must all add the same number.
    """
    got = (_tensor(dtype) + operand).numpy().astype(np.float64)
    np.testing.assert_allclose(got, _reference(dtype) + value, rtol=1e-6)


@pytest.mark.parametrize("dtype", BASE_DTYPES)
@pytest.mark.parametrize("build", [list, tuple, np.array])
def test_sequences_are_not_mistaken_for_scalars(dtype, build):
    operand = build([1.0, 2.0, 3.0])
    got = (_tensor(dtype) + operand).numpy().astype(np.float64)
    np.testing.assert_allclose(got, _reference(dtype) + np.array([1.0, 2.0, 3.0]))


def test_a_bare_python_float_takes_the_tensor_width():
    """No width of its own, so it takes the context's -- and the result stays
    in the tensor's dtype rather than widening it.
    """
    assert str((_tensor("float32") + 0.5).dtype) == "float32"
    assert str((_tensor("float64") + 0.5).dtype) == "float64"


def test_a_numpy_double_keeps_its_own_width():
    """`np.float64` subclasses `float`, so this is what separates asking for
    the exact type from asking `isinstance`. Getting it wrong is silent: the
    result is still a number, just the wrong one.
    """
    narrow = _tensor("float32")
    assert str((narrow + np.float64(0.5)).dtype) == "float64"
    assert str((narrow + np.float32(0.5)).dtype) == "float32"

    # And the digits are the point, not the label. 0.1 has no float32
    # representation, so a float64 tensor scaled by a Python float has to
    # stay in float64 the whole way -- narrowing and widening back cannot
    # recover what dropped.
    exact = (_tensor("float64") * 0.1).numpy()
    np.testing.assert_array_equal(exact, np.arange(1, 4, dtype=np.float64) * 0.1)


def test_a_float_subclass_that_is_not_numpys_still_works():
    """The exact-type tests are a fast path, not the whole rule: anything that
    is not exactly one of the three builtins still has to fall through to the
    general extraction.
    """
    got = (_tensor("float32") + FloatSubclass(0.5)).numpy()
    np.testing.assert_allclose(got, np.arange(1, 4, dtype=np.float32) + 0.5)
    assert str((_tensor("float32") + IntSubclass(2)).dtype) == "float32"


@pytest.mark.parametrize("operand", ["nope", None, {1: 2}, object(), [1, "a"]])
def test_operands_that_are_not_numbers_are_still_refused(operand):
    with pytest.raises(Exception):
        _tensor("float32") + operand


def test_bool_is_not_swallowed_by_the_int_branch():
    """`bool` subclasses `int` in Python, so an ordering that tests int first
    would classify `True` as an integer. Here it stays a bool operand, which
    for a bool tensor is the difference between `or` and `+`.
    """
    flags = mt.from_numpy(np.array([True, False, True]))
    np.testing.assert_array_equal(
        mt.maximum(flags, True).numpy(), np.array([True, True, True])
    )
    np.testing.assert_array_equal(
        mt.minimum(flags, True).numpy(), np.array([True, False, True])
    )


@pytest.mark.parametrize(
    "array_dtype,expected",
    [
        ("float32", "float32"),
        ("float64", "float64"),
        ("int32", "float32"),
        ("int64", "float32"),
        ("bool", "float32"),
        # Widths NumPy has and the engine does not fall back to the tensor's.
        ("int16", "float32"),
        ("uint8", "float32"),
        ("float16", "float32"),
    ],
)
def test_an_ndarray_operand_keeps_its_own_width(array_dtype, expected):
    """An array operand carries a width, unlike a bare Python float, and it
    is read off the dtype descriptor by type number rather than by parsing
    the dtype's printed name. The two have to agree on every dtype NumPy can
    hand over, including the ones the engine does not store -- those take the
    tensor's width, which is what the conversion would narrow to anyway.
    """
    narrow = mt.from_numpy(np.arange(1, 4, dtype=np.float32))
    operand = np.ones(3, dtype=array_dtype)
    result = narrow + operand
    assert str(result.dtype) == expected
    np.testing.assert_allclose(
        result.numpy().astype(np.float64), np.arange(2, 5, dtype=np.float64)
    )


def test_a_float64_array_operand_is_not_narrowed():
    """The reason the width is read at all: a float64 array added to a
    float32 tensor widens the result, so the array's digits survive. Reading
    the tensor's width instead would round them away.
    """
    narrow = mt.from_numpy(np.zeros(3, dtype=np.float32))
    operand = np.full(3, 0.1, dtype=np.float64)
    result = narrow + operand
    assert str(result.dtype) == "float64"
    np.testing.assert_array_equal(result.numpy(), operand)
