# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""Selecting a position without ordering the rest, and sorting by several keys.

`partition` puts the `kth` element where a sort would leave it and says nothing
about the rest, which is what buys the linear time: two million floats take
20ms partitioned against 90 sorted. A library with `sort` but no `partition`
makes a caller pay the ordering for a question that did not ask for one.

Its answer is deliberately *not* unique -- any arrangement with the kth element
in place, everything before it no greater and everything after no less is
correct -- so these tests check that invariant and the multiset, never equality
against NumPy's particular arrangement. `argpartition` is pinned the only way
it can be: the positions it reports must select the values `partition` gives.

`lexsort` is the other half: one stable sort per key, least significant first,
so the last key is the primary one. That is NumPy's convention and the one that
reads correctly when the keys are a table's columns.
"""

from __future__ import annotations

import numpy as np
import pytest

import minitensor as mt


def _holds(got, expected, kth, axis):
    """The partition invariant, plus the multiset, against NumPy's answer."""
    length = got.shape[axis]
    np.testing.assert_allclose(np.sort(got, axis), np.sort(expected, axis))
    for position in np.atleast_1d(kth):
        at = position + length if position < 0 else position
        # The kth element is the one a sort would leave there.
        np.testing.assert_array_equal(
            np.take(got, [at], axis), np.take(expected, [at], axis)
        )
        pivot = np.take(got, [at], axis)
        below = np.take(got, range(0, at), axis)
        above = np.take(got, range(at + 1, length), axis)
        if below.size:
            assert (below <= pivot).all()
        if above.size:
            assert (above >= pivot).all()


@pytest.mark.parametrize("kth", [0, 2, 4, -1, [1, 3]], ids=str)
@pytest.mark.parametrize("axis", [0, 1, -1])
def test_partition_puts_the_kth_where_a_sort_would(kth, axis):
    values = np.random.default_rng(53).standard_normal((8, 6))
    got = mt.partition(mt.from_numpy(values), kth, axis).numpy()
    _holds(got, np.partition(values, kth, axis), kth, axis)


def test_partition_over_the_flattened_tensor():
    values = np.random.default_rng(59).standard_normal((6, 5))
    for kth in [0, 7, 29, -1]:
        got = mt.partition(mt.from_numpy(values), kth, None).numpy()
        _holds(got.reshape(-1), np.partition(values, kth, None), kth, 0)


@pytest.mark.parametrize("dtype", ["float32", "float64", "int32", "int64"])
def test_partition_handles_every_ordered_dtype(dtype):
    rng = np.random.default_rng(61)
    values = (rng.standard_normal((4, 5, 6)) * 50).astype(dtype)
    got = mt.partition(mt.from_numpy(values), 2, 1).numpy()
    assert got.dtype == values.dtype
    _holds(got, np.partition(values, 2, 1), 2, 1)


def test_partition_puts_nan_after_every_number():
    values = np.array([3.0, np.nan, 1.0, np.nan, 2.0])
    got = mt.partition(mt.from_numpy(values), 2).numpy()
    np.testing.assert_array_equal(got[:3], [1.0, 2.0, 3.0])
    assert np.isnan(got[3:]).all()


def test_argpartition_names_the_positions_partition_moved():
    rng = np.random.default_rng(67)
    values = rng.standard_normal((5, 9))
    for axis in [0, 1]:
        where = mt.argpartition(mt.from_numpy(values), 3, axis)
        picked = mt.take_along_axis(mt.from_numpy(values), where, axis).numpy()
        np.testing.assert_array_equal(picked, mt.partition(mt.from_numpy(values), 3, axis).numpy())
        # And it is a permutation of the axis, not a selection with repeats.
        np.testing.assert_array_equal(
            np.sort(where.numpy(), axis), np.broadcast_to(
                np.arange(values.shape[axis]).reshape(
                    [-1 if d == axis else 1 for d in range(2)]
                ),
                values.shape,
            )
        )


def test_partition_refuses_what_it_cannot_order():
    values = np.random.default_rng(71).standard_normal((4, 5))
    with pytest.raises(ValueError, match="boolean"):
        mt.partition(mt.from_numpy(np.array([True, False])), 0)
    with pytest.raises(IndexError, match="out of bounds"):
        mt.partition(mt.from_numpy(values), 99, 1)
    with pytest.raises(ValueError, match="at least one position"):
        mt.partition(mt.from_numpy(values), [], 1)


def test_lexsort_takes_the_last_key_as_the_primary_one():
    first = np.array([3, 1, 3, 1, 2])
    second = np.array([0, 5, 2, 1, 4])
    np.testing.assert_array_equal(
        mt.lexsort([mt.from_numpy(second), mt.from_numpy(first)]).numpy(),
        np.lexsort((second, first)),
    )
    np.testing.assert_array_equal(
        mt.lexsort([mt.from_numpy(first)]).numpy(), np.lexsort((first,))
    )


def test_lexsort_sorts_along_a_chosen_axis():
    rng = np.random.default_rng(73)
    # Few distinct values, so the ties the later keys break actually occur.
    first = rng.integers(0, 3, (4, 7))
    second = rng.integers(0, 3, (4, 7))
    for axis in [0, 1, -1]:
        np.testing.assert_array_equal(
            mt.lexsort([mt.from_numpy(first), mt.from_numpy(second)], axis).numpy(),
            np.lexsort((first, second), axis),
        )


def test_lexsort_rejects_keys_that_do_not_line_up():
    with pytest.raises(ValueError, match="same shape"):
        mt.lexsort([mt.from_numpy(np.arange(4)), mt.from_numpy(np.arange(5))])
    with pytest.raises(ValueError, match="at least one key"):
        mt.lexsort([])


def test_take_and_put_along_axis_are_the_two_directions():
    rng = np.random.default_rng(79)
    values = rng.standard_normal((6, 5))
    where = rng.integers(0, 5, (6, 2))

    np.testing.assert_allclose(
        mt.take_along_axis(mt.from_numpy(values), mt.from_numpy(where), 1).numpy(),
        np.take_along_axis(values, where, 1),
    )

    written = rng.standard_normal((6, 2))
    expected = values.copy()
    np.put_along_axis(expected, where, written, 1)
    np.testing.assert_allclose(
        mt.put_along_axis(
            mt.from_numpy(values), mt.from_numpy(where), mt.from_numpy(written), 1
        ).numpy(),
        expected,
    )
    # The write returns a new tensor rather than changing the old one, which is
    # how every write in this library is spelled.
    original = mt.from_numpy(values)
    mt.put_along_axis(original, mt.from_numpy(where), mt.from_numpy(written), 1)
    np.testing.assert_allclose(original.numpy(), values)


def test_compress_and_extract_select_by_flag():
    rng = np.random.default_rng(83)
    values = rng.standard_normal((4, 5))
    rows = np.array([True, False, True, True])

    np.testing.assert_allclose(
        mt.compress(mt.from_numpy(rows), mt.from_numpy(values), 0).numpy(),
        np.compress(rows, values, 0),
    )
    # A condition shorter than the axis drops what it does not reach.
    short = np.array([True, False])
    np.testing.assert_allclose(
        mt.compress(mt.from_numpy(short), mt.from_numpy(values), 1).numpy(),
        np.compress(short, values, 1),
    )
    flat = np.array([1, 0, 1, 0] + [1] * 16)
    np.testing.assert_allclose(
        mt.compress(mt.from_numpy(flat), mt.from_numpy(values)).numpy(),
        np.compress(flat, values),
    )
    np.testing.assert_allclose(
        mt.extract(mt.from_numpy(values > 0), mt.from_numpy(values)).numpy(),
        np.extract(values > 0, values),
    )

    with pytest.raises(ValueError, match="flags for an axis"):
        mt.compress(mt.from_numpy(np.ones(9, dtype=bool)), mt.from_numpy(values), 0)


def test_choose_picks_position_by_position():
    rng = np.random.default_rng(89)
    picks = rng.integers(0, 3, (4, 5))
    options = [rng.standard_normal((4, 5)) for _ in range(3)]
    np.testing.assert_allclose(
        mt.choose(mt.from_numpy(picks), [mt.from_numpy(o) for o in options]).numpy(),
        np.choose(picks, options),
    )

    with pytest.raises(IndexError, match="choices"):
        mt.choose(mt.from_numpy(np.full((2, 2), 5)), [mt.from_numpy(np.zeros((2, 2)))])
    with pytest.raises(ValueError, match="at least one choice"):
        mt.choose(mt.from_numpy(picks), [])
