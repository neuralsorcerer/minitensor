# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""The sort-family operations NumPy answers, held to the kernel they replaced.

Float `unique` and the set operations built on it, small or integer `isin`,
and `partition`/`argpartition` go to NumPy on a view of the tensor's buffer,
because NumPy measured faster at them. Integer `unique` and the integer set
operations stay on the engine's kernel, which measured faster. These tests pin
that each route gives the same answer as the other would have, so the split is
a speed decision and nothing else.
"""

from __future__ import annotations

import numpy as np
import pytest

import minitensor as mt
from minitensor._indexing import _ISIN_NUMPY_BELOW, _unique_kernel


def _tensor(array):
    """`array` as a tensor of its own dtype; `mt.Tensor` would make it float32."""
    return mt.Tensor.from_numpy(np.ascontiguousarray(array))


def _values(tensor):
    return np.asarray(tensor)


def _same(got, expected):
    got = _values(got)
    assert got.dtype == expected.dtype
    assert got.shape == expected.shape
    np.testing.assert_array_equal(got, expected)


_AWKWARD = np.array([[3.0, -0.0, np.nan], [1.0, 0.0, 3.0], [np.nan, -2.5, 1.0]])


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize(
    "flags",
    [
        (False, False, False),
        (True, False, False),
        (False, True, False),
        (False, False, True),
        (True, True, True),
    ],
)
def test_float_unique_agrees_with_the_engine_kernel(dtype, flags):
    x = _tensor(_AWKWARD.astype(dtype))
    inverse, counts, index = flags
    got = mt.unique(x, return_inverse=inverse, return_counts=counts, return_index=index)
    kernel = _unique_kernel(x, inverse, counts, index)
    got = got if isinstance(got, tuple) else (got,)
    kernel = kernel if isinstance(kernel, tuple) else (kernel,)
    assert len(got) == len(kernel)
    for mine, theirs in zip(got, kernel):
        _same(mine, _values(theirs))


def test_unique_inverse_rebuilds_the_input():
    x = _tensor(_AWKWARD)
    values, inverse = mt.unique(x, return_inverse=True)
    np.testing.assert_array_equal(_values(values)[_values(inverse)], _AWKWARD)


@pytest.mark.parametrize("dtype", ["int32", "int64", "bool"])
def test_integer_unique_stays_on_the_kernel(dtype):
    source = np.array([4, 0, 4, 1, 0, 7]).astype(dtype)
    got = mt.unique(_tensor(source), return_counts=True)
    values, counts = np.unique(source, return_counts=True)
    _same(got[0], values)
    _same(got[1], counts.astype(np.int64))


@pytest.mark.parametrize(
    "left_dtype,right_dtype",
    [("float32", "float32"), ("float64", "float64"), ("float32", "float64")],
)
def test_float_set_operations_match_numpy(left_dtype, right_dtype):
    left_np = np.array([3.0, np.nan, -0.0, 1.5, 3.0, 2.0]).astype(left_dtype)
    right_np = np.array([2.0, 0.0, 7.0, np.nan, 1.5]).astype(right_dtype)
    left, right = _tensor(left_np), _tensor(right_np)
    promoted = np.result_type(left_np, right_np)
    a, b = left_np.astype(promoted), right_np.astype(promoted)

    _same(mt.union1d(left, right), np.union1d(a, b))
    # The one-sided operations keep the left operand's dtype.
    _same(mt.intersect1d(left, right), np.intersect1d(a, b).astype(left_dtype))
    _same(mt.setxor1d(left, right), np.setxor1d(a, b))
    _same(mt.setdiff1d(left, right), np.setdiff1d(left_np, b).astype(left_dtype))

    common, in_left, in_right = mt.intersect1d(left, right, return_indices=True)
    expected = np.intersect1d(a, b, return_indices=True)
    _same(common, expected[0].astype(left_dtype))
    _same(in_left, expected[1])
    _same(in_right, expected[2])


def test_integer_set_operations_are_unchanged():
    left_np = np.array([5, 1, 9, 1, 3], dtype=np.int64)
    right_np = np.array([3, 4, 5], dtype=np.int64)
    left, right = _tensor(left_np), _tensor(right_np)
    _same(mt.union1d(left, right), np.union1d(left_np, right_np))
    _same(mt.intersect1d(left, right), np.intersect1d(left_np, right_np))
    _same(mt.setdiff1d(left, right), np.setdiff1d(left_np, right_np))
    _same(mt.setxor1d(left, right), np.setxor1d(left_np, right_np))


@pytest.mark.parametrize("size", [8, _ISIN_NUMPY_BELOW])
@pytest.mark.parametrize("dtype", ["float32", "float64", "int32", "int64"])
@pytest.mark.parametrize("invert", [False, True])
def test_isin_on_either_side_of_the_threshold(size, dtype, invert):
    rng = np.random.default_rng(size)
    values = rng.integers(-50, 50, size=(2, size // 2)).astype(dtype)
    tests = rng.integers(-20, 20, size=size // 4).astype(dtype)
    if dtype.startswith("float"):
        values[0, 0] = np.nan
        values[0, 1] = -0.0
        tests[0] = 0.0
    got = mt.isin(_tensor(values), _tensor(tests), invert=invert)
    _same(got, np.isin(values, tests, invert=invert))


def test_isin_promotes_before_comparing():
    got = mt.isin(_tensor(np.array([1, 2, 3])), _tensor(np.array([2.5, 3.0])))
    _same(got, np.array([False, False, True]))


@pytest.mark.parametrize("dtype", ["float32", "float64", "int32", "int64"])
@pytest.mark.parametrize("dim", [None, 0, 1, -1])
def test_partition_places_the_kth_element(dtype, dim):
    rng = np.random.default_rng(7)
    source = rng.standard_normal((5, 9)).astype(dtype)
    if dtype.startswith("float"):
        source[1, 2] = np.nan
    x = _tensor(source)
    axis = -1 if dim is None else dim
    flat = source.reshape(-1) if dim is None else source
    kth = 3
    ordered = np.sort(flat, axis=axis)
    by_value = _values(mt.partition(x, kth, dim=dim))
    by_position = np.take_along_axis(
        flat, _values(mt.argpartition(x, kth, dim=dim)), axis=axis
    )
    # The two are separate selections, so only the partition property is
    # common to both: the kth element in place, nothing larger before it and
    # nothing smaller after (NaN sorting last, as in `np.sort`).
    for got in (by_value, by_position):
        np.testing.assert_array_equal(np.sort(got, axis=axis), ordered)
        np.testing.assert_array_equal(
            np.take(got, kth, axis=axis), np.take(ordered, kth, axis=axis)
        )
        head = np.take(got, range(kth), axis=axis)
        tail = np.take(got, range(kth + 1, got.shape[axis]), axis=axis)
        pivot = np.take(got, [kth], axis=axis)
        assert not np.any(head > pivot)
        assert not np.any(tail < pivot)


def test_partition_refuses_booleans():
    with pytest.raises(ValueError, match="boolean"):
        mt.partition(_tensor(np.array([True, False])), 0)


def test_delegated_results_own_their_memory():
    """A result is NumPy's array, shared rather than copied; it must outlive
    every NumPy-side name for it."""
    x = _tensor(np.array([2.0, 1.0, 2.0, np.nan]))
    got = mt.unique(x)
    del x
    np.testing.assert_array_equal(_values(got), [1.0, 2.0, np.nan])
