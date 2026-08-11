# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""`array_equal` compared raw bytes, which is not float equality.

For contiguous CPU tensors -- which is nearly all of them -- it returned
`bytes_a == bytes_b`, and that is wrong in *both* directions:

    array_equal(+0.0, -0.0)   was False, should be True   (bits differ, values do not)
    array_equal(NaN,  NaN)    was True,  should be False  (bits match, values do not)

Which made it disagree with the library's own `eq`, which says the opposite of
both. `allclose` next door already guarded its byte path on
`!dtype.is_float()`; this one did not.

Nothing tested it, so the whole suite passed either way. The tests here are
written as agreement checks -- against `eq` and against NumPy -- rather than
against recorded values, so they stay meaningful if the fast path is ever
reworked.
"""

from __future__ import annotations

import numpy as np
import pytest

import minitensor as mt

# Values chosen so that byte equality and float equality disagree.
TRICKY = [
    ("plus-zero-vs-minus-zero", [0.0], [-0.0]),
    ("minus-zero-vs-plus-zero", [-0.0], [0.0]),
    ("nan-vs-itself", [np.nan], [np.nan]),
    ("nan-among-equals", [1.0, np.nan, 3.0], [1.0, np.nan, 3.0]),
    ("zeros-among-equals", [1.0, 0.0, 3.0], [1.0, -0.0, 3.0]),
    ("inf-vs-itself", [np.inf], [np.inf]),
    ("inf-vs-neg-inf", [np.inf], [-np.inf]),
    ("plain-equal", [1.0, 2.0], [1.0, 2.0]),
    ("plain-different", [1.0, 2.0], [1.0, 2.5]),
]

DTYPES = ["float32", "float64"]

# The comparison splits at 1024 elements, and the byte path only applies to
# contiguous CPU tensors, so both sides of that split are exercised.
SIZES = [1, 8, 1024, 4096]


def _pair(left, right, dtype):
    return (
        mt.Tensor(np.array(left, dtype=dtype), dtype=dtype),
        mt.Tensor(np.array(right, dtype=dtype), dtype=dtype),
    )


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("name,left,right", TRICKY, ids=[case[0] for case in TRICKY])
def test_array_equal_agrees_with_numpy(name, left, right, dtype):
    a, b = _pair(left, right, dtype)
    assert mt.array_equal(a, b) == np.array_equal(a.numpy(), b.numpy())


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("name,left,right", TRICKY, ids=[case[0] for case in TRICKY])
def test_array_equal_agrees_with_elementwise_eq(name, left, right, dtype):
    """The two must not contradict each other.

    `array_equal` reading bytes while `eq` reads values is how one said NaN
    equals itself and the other said it does not.
    """
    a, b = _pair(left, right, dtype)
    assert mt.array_equal(a, b) == bool(mt.eq(a, b).numpy().all())


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("size", SIZES)
def test_a_tensor_of_nan_is_not_equal_to_itself(dtype, size):
    """Including at sizes above the parallel-reduction threshold, where a
    different code path runs."""
    values = mt.Tensor(np.full(size, np.nan, dtype=dtype), dtype=dtype)
    assert not mt.array_equal(values, values)
    assert not np.array_equal(values.numpy(), values.numpy())


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("size", SIZES)
def test_signed_zeros_compare_equal(dtype, size):
    positive = mt.Tensor(np.zeros(size, dtype=dtype), dtype=dtype)
    negative = mt.Tensor(np.full(size, -0.0, dtype=dtype), dtype=dtype)
    assert mt.array_equal(positive, negative)
    assert np.array_equal(positive.numpy(), negative.numpy())


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("size", SIZES)
def test_a_tensor_without_nan_is_equal_to_itself(dtype, size):
    """The fix must not make everything unequal."""
    values = mt.Tensor(np.arange(size, dtype=dtype), dtype=dtype)
    assert mt.array_equal(values, values)


@pytest.mark.parametrize("dtype", ["int32", "int64", "bool"])
@pytest.mark.parametrize("size", SIZES)
def test_non_float_dtypes_keep_the_byte_comparison(dtype, size):
    """Integers and booleans have no NaN and no signed zero, so bytes and
    values agree and the fast path is still correct for them."""
    base = np.arange(size) % 7
    left = mt.Tensor(base.astype(dtype), dtype=dtype)
    right = mt.Tensor(base.astype(dtype), dtype=dtype)
    assert mt.array_equal(left, right)

    if size > 1:
        changed = base.copy()
        changed[0] = changed[0] + 1
        assert not mt.array_equal(left, mt.Tensor(changed.astype(dtype), dtype=dtype))


def test_shape_and_dtype_mismatches_are_still_unequal():
    assert not mt.array_equal(mt.zeros(2, 3), mt.zeros(3, 2))
    assert not mt.array_equal(mt.zeros(4), mt.zeros(5))


def test_allclose_treats_nan_the_same_way_unless_told_otherwise():
    """`allclose` already guarded its byte path; this pins the behaviour
    `array_equal` has now been brought into line with."""
    values = mt.Tensor(np.array([1.0, np.nan], dtype=np.float32))

    assert not values.allclose(values)
    assert values.allclose(values, equal_nan=True)


def test_allclose_and_array_equal_agree_on_exact_inputs():
    """With zero tolerance and no NaN, an approximate comparison and an exact
    one must reach the same verdict."""
    for left, right in ([1.0, 2.0], [1.0, 2.0]), ([1.0, 2.0], [1.0, 2.5]):
        a, b = _pair(left, right, "float32")
        assert a.allclose(b, rtol=0.0, atol=0.0) == mt.array_equal(a, b)
