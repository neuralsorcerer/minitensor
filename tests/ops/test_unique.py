# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""Asking which values are there, and which is there most.

`unique` collapses a tensor to its distinct values, `unique_consecutive`
collapses only adjacent runs, and `mode` reports the value occurring most often
along an axis. All three are the same walk over runs of equal elements: `unique`
sorts first so equal values become adjacent, `mode` keeps the longest run
instead of all of them, and `unique_consecutive` does not sort at all.

Counting distinct values needs a sort or a hash, and no arrangement of the
arithmetic and reduction operations in this library performs either -- so none
of this composes out of what was here.

The tests that matter most are the NaN ones. `NaN` is not ordered against
anything and is not equal to itself, so the two obvious implementations are both
wrong in different ways: a comparison sort over raw floating-point order has no
defined result, and a run detector over `==` emits every NaN as its own distinct
value. One comparison fixes both by putting NaN after every number and calling
it equal to itself, which is what NumPy does and what the tests below pin.
"""

from __future__ import annotations

import numpy as np
import pytest

import minitensor as mt


def _t(a):
    # Not `ascontiguousarray` unconditionally: it promotes a 0-d array to shape
    # `(1,)`, which would quietly turn the rank-rejection test below into a
    # one-element tensor that `mode` is perfectly happy to reduce.
    a = np.asarray(a)
    return mt.Tensor.from_numpy(a if a.ndim == 0 else np.ascontiguousarray(a))


# --------------------------------------------------------------------------
# unique
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "values",
    [
        [3.0, 1.0, 2.0, 1.0, 3.0, 3.0],
        [1.0],
        [2.0, 2.0, 2.0],
        [5.0, 4.0, 3.0, 2.0, 1.0],
        [-1.0, 0.0, 1.0],
    ],
)
def test_unique_matches_numpy(values):
    data = np.array(values)
    assert np.array_equal(mt.unique(_t(data)).numpy(), np.unique(data))


def test_unique_flattens():
    """A question about which values occur, not about where."""
    data = np.array([[3.0, 1.0], [1.0, 2.0]])
    assert np.array_equal(mt.unique(_t(data)).numpy(), np.unique(data))
    assert mt.unique(_t(data)).numpy().ndim == 1


def test_the_inverse_rebuilds_the_input():
    """What `return_inverse` is for, stated as the property it has to satisfy."""
    data = np.array([[3.0, 1.0, 2.0], [1.0, 3.0, 3.0]])
    values, inverse = mt.unique(_t(data), True)
    assert inverse.numpy().shape == data.shape
    assert np.array_equal(values.numpy()[inverse.numpy()], data)
    assert np.array_equal(inverse.numpy(), np.unique(data, return_inverse=True)[1])


def test_the_counts_add_up():
    data = np.array([3.0, 1.0, 2.0, 1.0, 3.0, 3.0])
    values, counts = mt.unique(_t(data), False, True)
    assert np.array_equal(counts.numpy(), np.unique(data, return_counts=True)[1])
    assert counts.numpy().sum() == data.size


def test_asking_for_everything():
    data = np.array([3.0, 1.0, 2.0, 1.0])
    values, inverse, counts = mt.unique(_t(data), True, True)
    want_v, want_i, want_c = np.unique(data, return_inverse=True, return_counts=True)
    assert np.array_equal(values.numpy(), want_v)
    assert np.array_equal(inverse.numpy(), want_i)
    assert np.array_equal(counts.numpy(), want_c)


def test_asking_for_nothing_extra_returns_a_bare_tensor():
    """NumPy and PyTorch both vary their arity with the flags; forcing a caller
    to unpack a one-tuple would be a gratuitous difference."""
    result = mt.unique(_t(np.array([1.0, 2.0])))
    assert not isinstance(result, tuple)
    assert result.numpy().shape == (2,)


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int32, np.int64])
def test_unique_of_every_numeric_dtype(dtype):
    data = np.array([3, 1, 2, 1, 3], dtype=dtype)
    got = mt.unique(_t(data))
    assert got.numpy().dtype == dtype
    assert np.array_equal(got.numpy(), np.unique(data))


def test_unique_of_booleans():
    data = np.array([True, False, True, True])
    got = mt.unique(_t(data))
    assert np.array_equal(got.numpy(), np.unique(data))
    values, counts = mt.unique(_t(data), False, True)
    assert np.array_equal(counts.numpy(), [1, 3])


def test_unique_of_nothing():
    values, inverse, counts = mt.unique(_t(np.zeros(0)), True, True)
    assert values.numpy().shape == (0,)
    assert inverse.numpy().shape == (0,)
    assert counts.numpy().shape == (0,)


def test_the_indices_are_int64():
    _, inverse, counts = mt.unique(_t(np.array([1.0, 1.0])), True, True)
    assert inverse.dtype == "int64"
    assert counts.dtype == "int64"


def test_nothing_here_carries_a_gradient():
    """`unique` returns a subset of its input and which subset changes
    discontinuously as values collide -- there is no derivative to report."""
    t = mt.Tensor.from_numpy(np.array([1.0, 2.0, 2.0]), requires_grad=True)
    assert not mt.unique(t).requires_grad
    assert not mt.mode(t)[0].requires_grad


# --------------------------------------------------------------------------
# NaN, which is where the two obvious implementations go wrong
# --------------------------------------------------------------------------


def test_nans_collapse_to_one_and_sort_last():
    """`NaN != NaN`, so a run detector over `==` would emit every one of these
    separately. They are one value here, at the end, as NumPy has them."""
    data = np.array([np.nan, 1.0, np.nan, 0.0, np.nan])
    got = mt.unique(_t(data)).numpy()
    want = np.unique(data)
    assert np.array_equal(got[:-1], want[:-1])
    assert np.isnan(got[-1]) and np.isnan(want[-1])
    assert got.size == 3
    assert np.isnan(got).sum() == 1


def test_a_tensor_of_only_nans():
    got = mt.unique(_t(np.full(5, np.nan))).numpy()
    assert got.size == 1 and np.isnan(got[0])


def test_nan_counts_as_one_value():
    values, counts = mt.unique(_t(np.array([np.nan, 1.0, np.nan])), False, True)
    assert np.array_equal(counts.numpy(), [1, 2])


def test_nan_does_not_break_the_ordering_of_the_rest():
    """A comparison sort over raw floating-point order is undefined once a NaN
    is in the slice: it can leave the numbers unsorted too."""
    data = np.array([3.0, np.nan, 1.0, 2.0, np.nan, 0.0])
    got = mt.unique(_t(data)).numpy()
    assert np.array_equal(got[:-1], [0.0, 1.0, 2.0, 3.0])
    assert np.isnan(got[-1])


def test_infinities_are_ordered_normally():
    data = np.array([np.inf, -np.inf, 0.0, np.inf])
    assert np.array_equal(mt.unique(_t(data)).numpy(), np.unique(data))


def test_nan_and_infinity_together():
    data = np.array([np.nan, np.inf, -np.inf, np.nan, 0.0])
    got = mt.unique(_t(data)).numpy()
    assert np.array_equal(got[:-1], [-np.inf, 0.0, np.inf])
    assert np.isnan(got[-1])


def test_the_inverse_still_rebuilds_an_input_holding_nans():
    data = np.array([np.nan, 1.0, np.nan, 2.0])
    values, inverse = mt.unique(_t(data), True)
    rebuilt = values.numpy()[inverse.numpy()]
    assert np.array_equal(np.isnan(rebuilt), np.isnan(data))
    assert np.array_equal(rebuilt[~np.isnan(rebuilt)], data[~np.isnan(data)])


# --------------------------------------------------------------------------
# unique_consecutive
# --------------------------------------------------------------------------


def test_only_adjacent_runs_collapse():
    """The difference from `unique`, stated by a case where they disagree: a
    value that recurs after something else appears again."""
    data = np.array([1.0, 1.0, 2.0, 2.0, 1.0, 3.0])
    assert np.array_equal(mt.unique_consecutive(_t(data)).numpy(), [1.0, 2.0, 1.0, 3.0])
    assert np.array_equal(mt.unique(_t(data)).numpy(), [1.0, 2.0, 3.0])


def test_run_lengths():
    data = np.array([1.0, 1.0, 2.0, 2.0, 2.0, 1.0])
    values, counts = mt.unique_consecutive(_t(data), False, True)
    assert np.array_equal(values.numpy(), [1.0, 2.0, 1.0])
    assert np.array_equal(counts.numpy(), [2, 3, 1])
    assert counts.numpy().sum() == data.size


def test_consecutive_inverse_rebuilds_the_input():
    data = np.array([1.0, 1.0, 2.0, 3.0, 3.0])
    values, inverse = mt.unique_consecutive(_t(data), True)
    assert np.array_equal(values.numpy()[inverse.numpy()], data)


def test_nothing_repeats_so_nothing_collapses():
    data = np.array([3.0, 1.0, 2.0])
    assert np.array_equal(mt.unique_consecutive(_t(data)).numpy(), data)


def test_consecutive_on_sorted_input_equals_unique():
    """The relationship between the two, which is why one is not the other's
    special case: they agree exactly when the input is already sorted."""
    data = np.sort(np.random.default_rng(0).integers(0, 5, 50).astype(np.float64))
    assert np.array_equal(
        mt.unique_consecutive(_t(data)).numpy(), mt.unique(_t(data)).numpy()
    )


def test_consecutive_collapses_adjacent_nans():
    data = np.array([np.nan, np.nan, 1.0, np.nan])
    got = mt.unique_consecutive(_t(data)).numpy()
    assert got.size == 3
    assert np.isnan(got[0]) and got[1] == 1.0 and np.isnan(got[2])


# --------------------------------------------------------------------------
# mode
# --------------------------------------------------------------------------


def test_mode_of_a_row():
    data = np.array([1.0, 2.0, 2.0, 3.0])
    values, indices = mt.mode(_t(data))
    assert values.item() == 2.0
    assert indices.item() == 1


def test_mode_along_the_last_axis():
    data = np.array([[1.0, 2.0, 2.0, 3.0], [5.0, 5.0, 5.0, 1.0]])
    values, indices = mt.mode(_t(data))
    assert np.array_equal(values.numpy(), [2.0, 5.0])
    assert np.array_equal(indices.numpy(), [1, 0])


def test_mode_along_another_axis():
    data = np.array([[1.0, 2.0, 2.0, 3.0], [1.0, 2.0, 3.0, 1.0]])
    values, _ = mt.mode(_t(data), 0)
    assert np.array_equal(values.numpy(), [1.0, 2.0, 2.0, 1.0])


@pytest.mark.parametrize("dim", [0, 1, 2, -1, -2, -3])
def test_mode_reduces_the_axis_it_was_given(dim):
    data = np.random.default_rng(1).integers(0, 3, (2, 3, 4)).astype(np.float64)
    values, indices = mt.mode(_t(data), dim)
    expected = list(data.shape)
    expected.pop(dim)
    assert values.numpy().shape == tuple(expected)
    assert indices.numpy().shape == tuple(expected)


def test_keepdim_leaves_the_axis_in_place():
    data = np.random.default_rng(2).integers(0, 3, (2, 5)).astype(np.float64)
    values, indices = mt.mode(_t(data), 1, True)
    assert values.numpy().shape == (2, 1)
    assert indices.numpy().shape == (2, 1)


def test_a_tie_goes_to_the_smaller_value():
    """A tie has no natural winner, so the rule is fixed here rather than left
    to whatever the sort happened to do."""
    values, _ = mt.mode(_t(np.array([2.0, 2.0, 1.0, 1.0])))
    assert values.item() == 1.0
    values, _ = mt.mode(_t(np.array([5.0, 5.0, 9.0, 9.0, 1.0, 1.0])))
    assert values.item() == 1.0


def test_the_index_is_the_first_occurrence():
    """A repeated value has no natural occurrence either."""
    _, indices = mt.mode(_t(np.array([3.0, 1.0, 1.0, 3.0, 1.0])))
    assert indices.item() == 1


def test_the_reported_index_really_holds_the_reported_value():
    """The property that ties the two outputs together, over a batch where a
    mismatch would be easy to miss."""
    data = np.random.default_rng(3).integers(0, 4, (6, 9)).astype(np.float64)
    values, indices = mt.mode(_t(data))
    for row in range(6):
        assert data[row, indices.numpy()[row]] == values.numpy()[row]


def test_every_value_distinct_gives_the_smallest():
    """With no repeats every count is one, so the tie rule decides."""
    values, _ = mt.mode(_t(np.array([4.0, 2.0, 9.0, 7.0])))
    assert values.item() == 2.0


def test_mode_of_integers():
    data = np.array([[1, 1, 2], [3, 3, 3]], dtype=np.int64)
    values, _ = mt.mode(_t(data))
    assert values.numpy().dtype == np.int64
    assert np.array_equal(values.numpy(), [1, 3])


def test_mode_indices_are_int64():
    _, indices = mt.mode(_t(np.array([1.0, 1.0])))
    assert indices.dtype == "int64"


def test_mode_rejects_an_empty_axis():
    with pytest.raises(Exception, match="empty"):
        mt.mode(_t(np.zeros((2, 0))))


def test_mode_rejects_a_scalar():
    with pytest.raises(Exception, match="at least one dimension"):
        mt.mode(_t(np.array(1.0)))


def test_mode_agrees_with_counting_by_hand():
    """Against the definition rather than another implementation."""
    rng = np.random.default_rng(4)
    for _ in range(30):
        row = rng.integers(0, 5, 11).astype(np.float64)
        values, _ = mt.mode(_t(row))
        counts = {v: int((row == v).sum()) for v in np.unique(row)}
        best = max(counts.values())
        assert values.item() == min(v for v, c in counts.items() if c == best)


# --------------------------------------------------------------------------
# What these are for
# --------------------------------------------------------------------------


def test_building_a_vocabulary():
    """`unique` plus `inverse` is exactly tokenisation: the distinct symbols
    become the vocabulary and the inverse is the encoded sequence."""
    symbols = np.array([7.0, 3.0, 7.0, 9.0, 3.0, 7.0])
    vocabulary, encoded = mt.unique(_t(symbols), True)
    assert np.array_equal(vocabulary.numpy(), [3.0, 7.0, 9.0])
    assert np.array_equal(encoded.numpy(), [1, 0, 1, 2, 0, 1])
    assert np.array_equal(vocabulary.numpy()[encoded.numpy()], symbols)


def test_run_length_encoding():
    """What `unique_consecutive` is for, and what `unique` would destroy."""
    labels = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 0.0])
    values, lengths = mt.unique_consecutive(_t(labels), False, True)
    assert np.array_equal(values.numpy(), [0.0, 1.0, 0.0])
    assert np.array_equal(lengths.numpy(), [3, 2, 1])
    assert lengths.numpy().sum() == labels.size


def test_a_majority_vote_across_models():
    """What `mode` is for: several predictions per sample, one answer."""
    predictions = np.array([[2.0, 2.0, 1.0], [0.0, 3.0, 3.0], [1.0, 1.0, 1.0]])
    voted, _ = mt.mode(_t(predictions))
    assert np.array_equal(voted.numpy(), [2.0, 3.0, 1.0])
