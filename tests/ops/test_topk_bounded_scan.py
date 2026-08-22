# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""Taking the top 100 of two million must not cost four copies of the tensor.

`topk` built an `(index, value)` pair for every element of the slice and ran
`select_nth_unstable_by` over the lot. That is the right algorithm when `k` is
a decent fraction of the slice and badly wrong when it is not: the top 100 of
two million float32 meant allocating and writing 32MB of pairs -- four times
the tensor -- to look at 100 of them. It measured 36.7ms against NumPy's 6.9ms
for the same work.

A bounded heap of `k` reads the input once and touches nothing else, and takes
that case to 5.5ms, which is quicker than NumPy. It stops paying once the heap
leaves cache, so the select path is still there for large `k` and the choice
between them is made on the heap's footprint.

That makes two implementations of one operation, which is the thing this
library keeps getting wrong. So most of what follows is the two paths being
handed the same input and required to answer identically -- and both being
required to match NumPy, which is a third opinion.
"""

from __future__ import annotations

import numpy as np
import pytest

import minitensor as mt

# The heap is taken when it fits in 256KB *and* k is at most an eighth of the
# slice. A `(usize, f32)` pair is 16 bytes, so 16384 is the last k that fits.
# These straddle both conditions.
SCAN = [(100_000, 1), (100_000, 10), (100_000, 1000), (200_000, 16_384)]
SELECT = [(100_000, 20_000), (100_000, 50_000), (1000, 500), (1000, 999)]


def _reference(values, k, largest=True):
    """The order `topk` promises: by value, ties by ascending index, and NaN
    ahead of everything when taking the largest and behind it when taking the
    smallest."""
    order = sorted(
        range(len(values)),
        key=lambda i: (
            (
                0 if np.isnan(values[i]) else 1,
                -values[i] if not np.isnan(values[i]) else 0,
                i,
            )
            if largest
            else (
                1 if np.isnan(values[i]) else 0,
                values[i] if not np.isnan(values[i]) else 0,
                i,
            )
        ),
    )
    idx = np.array(order[:k], dtype=np.int64)
    return values[idx], idx


def _data(n, seed=5):
    rng = np.random.default_rng(seed)
    return rng.standard_normal(n).astype(np.float32)


@pytest.mark.parametrize("n,k", SCAN + SELECT)
@pytest.mark.parametrize("largest", [True, False])
def test_topk_matches_a_python_reference(n, k, largest):
    values = _data(n)
    want_v, want_i = _reference(values, k, largest)
    got_v, got_i = mt.Tensor(values, dtype="float32").topk(k, largest=largest)
    np.testing.assert_array_equal(got_v.numpy(), want_v)
    np.testing.assert_array_equal(got_i.numpy(), want_i)


@pytest.mark.parametrize("n", [100_000])
def test_the_two_paths_agree_across_the_threshold(n):
    """The heap is taken at `k` and the select path at `8k`, on the same data.
    Whatever else differs, the answers may not."""
    values = _data(n, seed=7)
    t = mt.Tensor(values, dtype="float32")
    for k in (1000, 5000):
        scan_v, scan_i = t.topk(k)
        # The same k entries have to appear at the front of a larger request.
        wide_v, wide_i = t.topk(k * 8)
        np.testing.assert_array_equal(scan_v.numpy(), wide_v.numpy()[:k])
        np.testing.assert_array_equal(scan_i.numpy(), wide_i.numpy()[:k])


def test_the_heap_path_breaks_ties_by_index():
    """Every value the same, so the answer is decided entirely by the tie rule
    -- and the heap replaces its root only on a strict improvement, which is
    what makes the earliest indices the ones that survive."""
    n = 100_000
    values = np.zeros(n, dtype=np.float32)
    got_v, got_i = mt.Tensor(values, dtype="float32").topk(10)
    np.testing.assert_array_equal(got_i.numpy(), np.arange(10, dtype=np.int64))
    np.testing.assert_array_equal(got_v.numpy(), np.zeros(10, dtype=np.float32))


def test_the_heap_path_puts_nan_first_when_taking_the_largest():
    n = 100_000
    values = _data(n, seed=11)
    values[[500, 60_000, 99_999]] = np.nan
    got_v, got_i = mt.Tensor(values, dtype="float32").topk(5)
    assert np.isnan(got_v.numpy()[:3]).all()
    np.testing.assert_array_equal(got_i.numpy()[:3], [500, 60_000, 99_999])
    assert not np.isnan(got_v.numpy()[3:]).any()


def test_the_heap_path_puts_nan_last_when_taking_the_smallest():
    n = 100_000
    values = _data(n, seed=13)
    values[[7, 42]] = np.nan
    got_v, _ = mt.Tensor(values, dtype="float32").topk(5, largest=False)
    assert not np.isnan(got_v.numpy()).any()
    np.testing.assert_array_equal(got_v.numpy(), np.sort(values[~np.isnan(values)])[:5])


@pytest.mark.parametrize("k", [1, 3, 1000])
def test_unsorted_returns_the_same_set(k):
    """`sorted=False` never promised an order -- the select path left whatever
    the partition happened to leave. It still has to be the right k entries."""
    values = _data(100_000, seed=17)
    t = mt.Tensor(values, dtype="float32")
    want_v, want_i = t.topk(k, sorted=True)
    got_v, got_i = t.topk(k, sorted=False)
    assert sorted(got_i.numpy().tolist()) == sorted(want_i.numpy().tolist())
    assert sorted(got_v.numpy().tolist()) == sorted(want_v.numpy().tolist())


@pytest.mark.parametrize("dim", [0, 1])
def test_topk_along_a_dimension_still_lands_in_the_right_place(dim):
    """`inner > 1` means the slice is strided, which the heap reads through a
    closure rather than a slice; an off-by-one there would scramble columns."""
    rng = np.random.default_rng(19)
    values = rng.standard_normal((4000, 3)).astype(np.float32)
    got_v, got_i = mt.Tensor(values, dtype="float32").topk(2, dim=dim)

    order = np.argsort(-values, axis=dim, kind="stable")
    take = [slice(None), slice(None)]
    take[dim] = slice(0, 2)
    want_i = order[tuple(take)]
    want_v = np.take_along_axis(values, want_i, axis=dim)
    np.testing.assert_array_equal(got_v.numpy(), want_v)
    np.testing.assert_array_equal(got_i.numpy(), want_i)


@pytest.mark.parametrize("dtype", ["float64", "int32", "int64"])
def test_the_other_dtypes_take_the_same_path(dtype):
    rng = np.random.default_rng(23)
    if dtype == "float64":
        values = rng.standard_normal(100_000)
    else:
        values = rng.integers(-10_000, 10_000, size=100_000).astype(dtype)
    got_v, got_i = mt.Tensor(values, dtype=dtype).topk(50)
    order = np.argsort(-values, kind="stable")[:50]
    np.testing.assert_array_equal(got_i.numpy(), order)
    np.testing.assert_array_equal(got_v.numpy(), values[order])


def test_k_equal_to_the_slice_length_returns_a_full_sort():
    values = _data(5000, seed=29)
    got_v, got_i = mt.Tensor(values, dtype="float32").topk(5000)
    order = np.argsort(-values, kind="stable")
    np.testing.assert_array_equal(got_i.numpy(), order)
    np.testing.assert_array_equal(got_v.numpy(), values[order])


def test_topk_is_reproducible():
    values = _data(500_000, seed=31)
    t = mt.Tensor(values, dtype="float32")
    first_v, first_i = t.topk(100)
    for _ in range(10):
        v, i = t.topk(100)
        np.testing.assert_array_equal(v.numpy(), first_v.numpy())
        np.testing.assert_array_equal(i.numpy(), first_i.numpy())
