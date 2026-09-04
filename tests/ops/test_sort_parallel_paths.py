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

Stability is checked on each path specifically. `par_sort_by` is a different
algorithm from `sort_by`, and the transposed path reaches its comparator
through a different gather, so a caller who asks for a stable sort and silently
gets an unstable one has no way to notice until their ties come back reordered.
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
