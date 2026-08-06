# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""Sorting picks one of two parallel strategies, and both must agree.

`sort` split its work by outer position, one rayon task per slice. A 1-D tensor
has exactly one slice, so sorting one ran entirely on a single core. The same
2M elements cost 134 ns each arranged as one slice and 16 ns each as 2048 --
an 8.3x spread on four cores that was pure scheduling.

A large slice is now sorted in parallel *within* itself instead. Which path
runs depends on the slice count against the thread pool and on the slice
length, so the tests below deliberately straddle both: shapes with one, few and
many slices, and lengths either side of the 16384-element threshold. Every case
is checked against NumPy, so the two strategies cannot drift apart.

Stability is checked on the large-slice path specifically. `par_sort_by` is a
different algorithm from `sort_by`, and a caller who asks for a stable sort and
silently gets an unstable one has no way to notice until their ties come back
reordered.
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
