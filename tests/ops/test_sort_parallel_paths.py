# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""Sorting picks one of three parallel strategies, and all three must agree.

`sort` split its work by outer position, one rayon task per slice. A 1-D tensor
has exactly one slice, so sorting one ran entirely on a single core. The same
2M elements cost 134 ns each arranged as one slice and 16 ns each as 2048 --
an 8.3x spread on four cores that was pure scheduling.

A large slice is now sorted in parallel *within* itself instead. And a tensor
sorted along an axis that is not its last has its slices interleaved a stride
apart, where no cut of a contiguous buffer separates them: sorting along the
first axis has one outer position however large the tensor is, so `sort(x, 0)`
was serial for the same reason a 1-D sort was. That case sorts into a scratch
ordered slice-by-slice, which does cut apart, and lays the result back down the
axis afterwards -- 400ms to 124ms on a 2048-by-2048 sorted down its columns.

Which path runs depends on the slice count against the thread pool, on the
slice length, and on whether the axis is the last one, so the tests below
deliberately straddle all three: shapes with one, few and many slices, lengths
either side of the 16384-element threshold, and every axis of a tensor with
more than two. Every case is checked against NumPy, so the strategies cannot
drift apart.

Stability is checked on each path specifically. The sort is stable by
construction -- each element's position rides in the low bits of its sort key,
so no two entries compare equal and there is no tie for an unstable sort to
resolve -- but the three paths build and read those keys through different
gathers, so a caller whose ties come back reordered would have no other way to
notice.
"""

from __future__ import annotations

import numpy as np
import pytest

import minitensor as mt

DTYPES = ["float32", "float64", "int32", "int64"]

# (shape, dim). Slice counts of 1, 2 and 3 take the within-slice parallel sort
# when the slice is long enough; the rest keep the across-slice split.
LAYOUTS = [
    ((1,), 0),
    ((7,), 0),
    ((20000,), 0),  # one long slice: the case that was serial
    ((2, 20000), -1),  # few long slices
    ((3, 20000), -1),
    ((8, 20000), -1),  # enough slices to fill the pool
    ((64, 1024), -1),
    ((20000, 3), 0),  # long slice along dim 0, many slices along it
    ((2, 3, 9000), -1),
    ((100, 100), 0),
    ((100, 100), 1),
    # Not the last axis, and large enough to take the slice-major scratch:
    # one outer position, many interleaved slices.
    ((4000, 5), 0),
    ((5, 4000, 3), 1),
    ((300, 300), 0),
    # Not the last axis, but too small for it -- the across-slice split still.
    ((30, 30), 0),
    ((6, 7, 8), 1),
]


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("descending", [False, True])
@pytest.mark.parametrize("shape,dim", LAYOUTS, ids=[f"{s}@{d}" for s, d in LAYOUTS])
def test_sorted_values_match_numpy(shape, dim, descending, dtype):
    rng = np.random.default_rng(0)
    values = (rng.standard_normal(shape) * 100).astype(dtype)

    got, _ = mt.Tensor(values, dtype=dtype).sort(dim, descending=descending)

    expected = np.sort(values, axis=dim)
    if descending:
        expected = np.flip(expected, axis=dim)
    np.testing.assert_array_equal(got.numpy(), expected)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("descending", [False, True])
@pytest.mark.parametrize("shape,dim", LAYOUTS, ids=[f"{s}@{d}" for s, d in LAYOUTS])
def test_returned_indices_reproduce_the_values(shape, dim, descending, dtype):
    """The indices are the sort's other half and are easy to get subtly wrong
    when the sort itself is rewritten."""
    rng = np.random.default_rng(1)
    values = (rng.standard_normal(shape) * 100).astype(dtype)

    got, indices = mt.Tensor(values, dtype=dtype).sort(dim, descending=descending)

    gathered = np.take_along_axis(values, indices.numpy().astype(np.int64), axis=dim)
    np.testing.assert_array_equal(gathered, got.numpy())


@pytest.mark.parametrize("length", [8, 16383, 16384, 16385, 70000])
def test_a_single_slice_sorts_correctly_at_any_length(length):
    """Straddles the length threshold that selects the strategy."""
    rng = np.random.default_rng(2)
    values = rng.standard_normal(length).astype(np.float32)

    got, indices = mt.Tensor(values).sort(0)

    np.testing.assert_array_equal(got.numpy(), np.sort(values))
    np.testing.assert_array_equal(values[indices.numpy().astype(np.int64)], got.numpy())


@pytest.mark.parametrize("length", [1024, 20000])
def test_a_stable_sort_stays_stable_on_both_paths(length):
    """`par_sort_by` is a different algorithm from `sort_by`; asking for
    stability and quietly getting an unstable sort is invisible until ties come
    back reordered."""
    keys = (np.arange(length) % 8).astype(np.float32)

    values, indices = mt.Tensor(keys).sort(0, stable=True)

    np.testing.assert_array_equal(values.numpy(), np.sort(keys, kind="stable"))
    np.testing.assert_array_equal(
        indices.numpy().astype(np.int64), np.argsort(keys, kind="stable")
    )


@pytest.mark.parametrize(
    "shape,dim", [((4000, 5), 0), ((5, 4000, 3), 1), ((300, 300), 0)]
)
def test_a_stable_sort_stays_stable_along_a_strided_axis(shape, dim):
    """The path that rewrites the axis as the last one reaches its comparator
    through a different gather, and lays its answer back down through a
    different scatter. Ties are where a reordering would show, so this input is
    almost entirely ties."""
    size = int(np.prod(shape))
    keys = (np.arange(size) % 4).astype(np.float32).reshape(shape)

    values, indices = mt.Tensor(np.ascontiguousarray(keys)).sort(dim, stable=True)

    np.testing.assert_array_equal(
        values.numpy(), np.sort(keys, axis=dim, kind="stable")
    )
    np.testing.assert_array_equal(
        indices.numpy().astype(np.int64), np.argsort(keys, axis=dim, kind="stable")
    )


@pytest.mark.parametrize("length", [1024, 20000])
def test_nan_sorts_to_the_end_on_both_paths(length):
    """NaN compares unordered, so its placement is a property of the comparator
    rather than of the sort -- and must survive swapping the sort out."""
    rng = np.random.default_rng(3)
    values = rng.standard_normal(length).astype(np.float32)
    values[:: max(length // 8, 1)] = np.nan

    got, _ = mt.Tensor(values).sort(0)

    np.testing.assert_array_equal(got.numpy(), np.sort(values))


@pytest.mark.parametrize("length", [1024, 20000])
def test_argsort_and_topk_agree_with_numpy(length):
    """Both read the same kernel."""
    rng = np.random.default_rng(4)
    values = rng.standard_normal(length).astype(np.float32)
    tensor = mt.Tensor(values)

    np.testing.assert_array_equal(
        tensor.argsort(0).numpy().astype(np.int64), np.argsort(values, kind="stable")
    )

    top_values, _ = tensor.topk(5, 0)
    np.testing.assert_array_equal(top_values.numpy(), np.sort(values)[::-1][:5])


def test_an_already_sorted_slice_is_left_alone():
    """A degenerate input for a comparison sort, and one where an off-by-one in
    the scatter back would be invisible against random data."""
    values = np.arange(20000, dtype=np.float32)

    got, indices = mt.Tensor(values).sort(0)

    np.testing.assert_array_equal(got.numpy(), values)
    np.testing.assert_array_equal(indices.numpy().astype(np.int64), np.arange(20000))


def test_a_reversed_slice_is_fully_reordered():
    values = np.arange(20000, dtype=np.float32)[::-1].copy()

    got, indices = mt.Tensor(values).sort(0)

    np.testing.assert_array_equal(got.numpy(), np.arange(20000, dtype=np.float32))
    np.testing.assert_array_equal(
        indices.numpy().astype(np.int64), np.arange(20000)[::-1]
    )


# The ordering is now carried by an integer key rather than by a three-way
# comparison, so the places where the *bit pattern* and the numeric order
# disagree are the ones a mistake would land in. There are exactly three: the
# two zeros, which are equal but differ in a bit; NaN, whose patterns straddle
# the whole range; and the descending direction, which mirrors the values but
# must not mirror the tie-break.
LENGTHS = [7, 20000]


@pytest.mark.parametrize("length", LENGTHS)
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_the_two_zeros_are_equal_and_keep_their_order(length, dtype):
    # -0.0 sorts *below* 0.0 by bit pattern and *equal* to it by value. Equal
    # is the answer, so the two come back in the order they went in -- and
    # each keeps its own sign, which a sort that rebuilt values from the key
    # could not promise.
    pattern = np.array([-0.0, 0.0], dtype=dtype)
    values = np.resize(pattern, length)

    got, indices = mt.Tensor(values, dtype=dtype).sort(0)

    np.testing.assert_array_equal(indices.numpy().astype(np.int64), np.arange(length))
    np.testing.assert_array_equal(np.signbit(got.numpy()), np.signbit(values))


@pytest.mark.parametrize("length", LENGTHS)
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_descending_puts_nan_first_and_still_breaks_ties_by_position(length, dtype):
    values = np.resize(np.array([np.nan, 2.0, 1.0, 2.0, np.nan], dtype=dtype), length)

    got, indices = mt.Tensor(values, dtype=dtype).sort(0, descending=True)

    order = indices.numpy().astype(np.int64)
    nans = int(np.isnan(values).sum())
    assert np.isnan(got.numpy()[:nans]).all(), "NaN does not lead a descending sort"
    assert not np.isnan(got.numpy()[nans:]).any()
    # Within the NaNs, and within each run of equal numbers, the positions rise:
    # descending mirrors the values, never the tie-break.
    for start, stop in ((0, nans), (nans, length)):
        block = order[start:stop]
        equal = np.concatenate(
            [[True], np.isnan(values[block[1:]]) & np.isnan(values[block[:-1]])]
        ) | np.concatenate([[True], values[block[1:]] == values[block[:-1]]])
        for i in range(1, len(block)):
            if equal[i]:
                assert block[i] > block[i - 1]


@pytest.mark.parametrize("length", LENGTHS)
@pytest.mark.parametrize("descending", [False, True])
def test_a_negative_nan_survives_the_sort(length, descending):
    # A negative NaN keys the same as every other NaN, so the values cannot be
    # rebuilt from the keys -- they are re-read from the input, which is what
    # keeps the sign and the payload intact.
    payload = np.array([-np.nan, np.nan, 1.0, -1.0], dtype=np.float32)
    values = np.resize(payload, length)

    got, _ = mt.Tensor(values, dtype="float32").sort(0, descending=descending)

    signs = np.signbit(got.numpy()[np.isnan(got.numpy())])
    assert signs.any() and not signs.all(), "the NaNs lost their signs"


@pytest.mark.parametrize("length", LENGTHS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("descending", [False, True])
def test_both_stability_settings_give_the_same_answer(length, dtype, descending):
    # The key carries each element's position, so the ordering is total and
    # `stable=False` has no tie left to resolve. Both settings are the same
    # sort, which is what lets the faster one always run.
    values = np.resize(np.array([3, 1, 2, 1, 3, 2], dtype=dtype), length)
    tensor = mt.Tensor(values, dtype=dtype)

    loose, loose_index = tensor.sort(0, descending=descending, stable=False)
    tight, tight_index = tensor.sort(0, descending=descending, stable=True)

    np.testing.assert_array_equal(loose.numpy(), tight.numpy())
    np.testing.assert_array_equal(loose_index.numpy(), tight_index.numpy())
    # And it really is the stable answer: equal values keep their input order.
    order = tight_index.numpy().astype(np.int64)
    for i in range(1, length):
        if values[order[i]] == values[order[i - 1]]:
            assert order[i] > order[i - 1]


@pytest.mark.parametrize("dtype", ["int32", "int64"])
def test_the_extreme_integers_sort_where_they_belong(dtype):
    # The integer key shifts the range so the most negative value becomes zero;
    # an implementation that negated instead would overflow on exactly this.
    info = np.iinfo(dtype)
    values = np.array([0, info.max, -1, info.min, 1], dtype=dtype)

    got, _ = mt.Tensor(values, dtype=dtype).sort(0)

    np.testing.assert_array_equal(got.numpy(), np.sort(values))
