# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""Sets of values: which are in both, in either, in one but not the other.

Each is `unique` and a membership test, which is why they belong in Python --
and why they are worth having rather than left to the caller: the obvious
implementation compares every element against every other, which is `n * m` of
time and memory, exactly when a set operation is worth reaching for. These go
through the sorted binary search `isin` already uses, so the cost is
`(n + m) log m`.

`unique` also learned to say where each distinct value first occurred, which is
what `intersect1d(..., return_indices=True)` reports and what NumPy spells
`return_index`. The sort underneath is unstable, so "first" has to be the
smallest position in the run rather than whichever one the sort happened to
leave at its head -- that is the part a test can catch.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

import minitensor as mt


@pytest.fixture()
def pair():
    rng = np.random.default_rng(97)
    return (
        rng.integers(0, 10, 20).astype(np.int64),
        rng.integers(5, 15, 15).astype(np.int64),
    )


def test_the_four_set_operations_match_numpy(pair):
    left, right = pair
    a, b = mt.from_numpy(left), mt.from_numpy(right)

    np.testing.assert_array_equal(mt.union1d(a, b).numpy(), np.union1d(left, right))
    np.testing.assert_array_equal(
        mt.intersect1d(a, b).numpy(), np.intersect1d(left, right)
    )
    np.testing.assert_array_equal(
        mt.setdiff1d(a, b).numpy(), np.setdiff1d(left, right)
    )
    np.testing.assert_array_equal(mt.setxor1d(a, b).numpy(), np.setxor1d(left, right))


def test_intersect1d_can_say_where_the_common_values_came_from(pair):
    left, right = pair
    common, in_left, in_right = mt.intersect1d(
        mt.from_numpy(left), mt.from_numpy(right), return_indices=True
    )
    expected, expected_left, expected_right = np.intersect1d(
        left, right, return_indices=True
    )
    np.testing.assert_array_equal(common.numpy(), expected)
    np.testing.assert_array_equal(in_left.numpy(), expected_left)
    np.testing.assert_array_equal(in_right.numpy(), expected_right)
    # The positions must actually hold the values they claim to.
    np.testing.assert_array_equal(left[in_left.numpy()], common.numpy())
    np.testing.assert_array_equal(right[in_right.numpy()], common.numpy())


def test_set_operations_flatten_and_take_floats(pair):
    left, right = pair
    np.testing.assert_array_equal(
        mt.union1d(mt.from_numpy(left.reshape(4, 5)), mt.from_numpy(right)).numpy(),
        np.union1d(left.reshape(4, 5), right),
    )

    rng = np.random.default_rng(101)
    first = rng.standard_normal(12).round(1)
    second = np.concatenate([first[:3], rng.standard_normal(5).round(1)])
    np.testing.assert_allclose(
        mt.intersect1d(mt.from_numpy(first), mt.from_numpy(second)).numpy(),
        np.intersect1d(first, second),
    )
    np.testing.assert_allclose(
        mt.setxor1d(mt.from_numpy(first), mt.from_numpy(second)).numpy(),
        np.setxor1d(first, second),
    )


def test_set_operations_with_nothing_on_one_side(pair):
    left, _ = pair
    empty = np.array([], dtype=np.int64)
    a, nothing = mt.from_numpy(left), mt.from_numpy(empty)
    np.testing.assert_array_equal(
        mt.intersect1d(a, nothing).numpy(), np.intersect1d(left, empty)
    )
    np.testing.assert_array_equal(
        mt.setdiff1d(a, nothing).numpy(), np.setdiff1d(left, empty)
    )
    np.testing.assert_array_equal(
        mt.union1d(a, nothing).numpy(), np.union1d(left, empty)
    )


@pytest.mark.parametrize(
    "inverse,counts,index", list(itertools.product([False, True], repeat=3))
)
def test_unique_returns_its_extras_in_numpys_order(inverse, counts, index):
    values = np.array([3, 1, 3, 2, 1, 1], dtype=np.int64)
    got = mt.unique(
        mt.from_numpy(values),
        return_inverse=inverse,
        return_counts=counts,
        return_index=index,
    )
    expected = np.unique(
        values, return_index=index, return_inverse=inverse, return_counts=counts
    )
    got = got if isinstance(got, tuple) else (got,)
    expected = expected if isinstance(expected, tuple) else (expected,)
    assert len(got) == len(expected)
    for mine, theirs in zip(got, expected):
        np.testing.assert_array_equal(mine.numpy().reshape(theirs.shape), theirs)


def test_where_a_value_first_occurred_is_its_earliest_position():
    """The sort is unstable, so this cannot be whichever equal element landed
    at the head of its run -- it has to be the smallest position in it."""
    rng = np.random.default_rng(103)
    values = rng.integers(0, 5, 200).astype(np.int64)
    distinct, first = mt.unique(mt.from_numpy(values), return_index=True)
    np.testing.assert_array_equal(
        first.numpy(), np.unique(values, return_index=True)[1]
    )
    np.testing.assert_array_equal(values[first.numpy()], distinct.numpy())
    for position, value in zip(first.numpy(), distinct.numpy()):
        assert not (values[:position] == value).any()


def test_unique_consecutive_can_report_where_each_run_started():
    values = np.array([3, 3, 1, 1, 3, 2, 2], dtype=np.int64)
    runs, starts = mt.unique_consecutive(mt.from_numpy(values), return_index=True)
    np.testing.assert_array_equal(runs.numpy(), [3, 1, 3, 2])
    np.testing.assert_array_equal(starts.numpy(), [0, 2, 4, 5])


def test_the_array_api_spellings_agree_with_unique():
    values = np.array([3, 1, 3, 2, 1, 1], dtype=np.int64)
    tensor = mt.from_numpy(values)
    everything = mt.unique_all(tensor)
    expected = np.unique_all(values)

    np.testing.assert_array_equal(everything.values.numpy(), expected.values)
    np.testing.assert_array_equal(everything.indices.numpy(), expected.indices)
    np.testing.assert_array_equal(
        everything.inverse_indices.numpy().reshape(-1),
        expected.inverse_indices.reshape(-1),
    )
    np.testing.assert_array_equal(everything.counts.numpy(), expected.counts)

    np.testing.assert_array_equal(
        mt.unique_counts(tensor).counts.numpy(), np.unique_counts(values).counts
    )
    np.testing.assert_array_equal(
        mt.unique_inverse(tensor).inverse_indices.numpy().reshape(-1),
        np.unique_inverse(values).inverse_indices.reshape(-1),
    )
    # The standard leaves this order unspecified and NumPy returns it unsorted;
    # these come back ascending, which `unique` already promised.
    np.testing.assert_array_equal(mt.unique_values(tensor).numpy(), np.unique(values))


@pytest.mark.parametrize("trim", ["fb", "f", "b"])
def test_trim_zeros_only_touches_the_ends(trim):
    values = np.array([0, 0, 1, 2, 0, 3, 0, 0], dtype=np.int64)
    np.testing.assert_array_equal(
        mt.trim_zeros(mt.from_numpy(values), trim).numpy(),
        np.trim_zeros(values, trim),
    )


def test_trim_zeros_on_nothing_and_on_all_zeros():
    zeros = np.zeros(5, dtype=np.int64)
    assert mt.trim_zeros(mt.from_numpy(zeros)).numpy().size == 0
    empty = np.array([], dtype=np.int64)
    assert mt.trim_zeros(mt.from_numpy(empty)).numpy().size == 0
    with pytest.raises(ValueError, match="'f', 'b' or 'fb'"):
        mt.trim_zeros(mt.from_numpy(zeros), "x")
