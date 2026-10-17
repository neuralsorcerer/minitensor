# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""A histogram puts each value in the bin of the last edge at or below it.

Each value was found by a binary search with a branch per comparison, on a
float64 copy of the whole input: 46ms for four million values into a hundred
bins. Equal-width bins are now found by arithmetic and checked against the
edges, other edges by a search without branches, and the input is read in its
own dtype. The answer is the search's, which these tests hold it to.

Two answers were wrong. An int64 past 2^53 was rounded to a float before it
was compared, so `2**60 - 1` was counted in the bin that starts at `2**60`.
And a range with finite ends but a span past the float range, such as
`(-1e308, 1e308)`, was spaced by an infinite step: its first edge was
`low + 0 * inf`, a NaN, and the counts were wrong.
"""

from __future__ import annotations

import numpy as np
import pytest

import minitensor as mt


def _searched(values, edges, weights=None):
    """The counts a search of every value against `edges` gives."""
    values = np.asarray(values, dtype=np.float64)
    keep = (values >= edges[0]) & (values <= edges[-1])
    slot = np.searchsorted(edges, values[keep], side="right") - 1
    slot = np.minimum(slot, len(edges) - 2)
    weights = None if weights is None else weights[keep]
    return np.bincount(slot, weights=weights, minlength=len(edges) - 1)


@pytest.mark.parametrize("dtype", ["float32", "float64", "int32", "int64", "bool"])
def test_every_value_lands_where_the_search_puts_it(dtype):
    rng = np.random.default_rng(61)
    for _ in range(80):
        n = int(rng.integers(0, 4000))
        if dtype == "bool":
            values = rng.random(n) < 0.3
        elif "int" in dtype:
            values = rng.integers(-50, 50, n).astype(dtype)
        else:
            values = rng.normal(0, 10, n).astype(dtype)
            if n > 3:
                values[rng.integers(0, n, 3)] = [np.nan, np.inf, -np.inf]
        bins = int(rng.choice([1, 2, 7, 100, 5000]))
        weights = rng.uniform(0, 1, n) if rng.random() < 0.3 else None
        options = {} if weights is None else {"weights": mt.Tensor(weights)}
        kind = int(rng.integers(0, 3))
        if kind == 0:
            low, high = sorted(float(v) for v in rng.normal(0, 10, 2))
            options["range"] = (low, high)
        if kind == 2:
            edges = np.unique(rng.normal(0, 10, min(bins, 300) + 1))
            if len(edges) < 2:
                continue
            bins = mt.Tensor(edges)
        counts, edges = mt.histogram(mt.Tensor(values), bins=bins, **options)
        want = _searched(values, edges.numpy(), weights)
        np.testing.assert_allclose(counts.numpy(), want, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("bins", [1, 7, 100])
@pytest.mark.parametrize("bounds", [None, (-0.5, 0.5), (10.0, 11.0)])
def test_a_long_input_counts_the_same_across_its_parallel_bands(bins, bounds):
    values = np.random.default_rng(67).normal(0, 1, 3_000_001).astype(np.float32)
    options = {} if bounds is None else {"range": bounds}
    counts, edges = mt.histogram(mt.Tensor(values), bins=bins, **options)
    np.testing.assert_array_equal(counts.numpy(), _searched(values, edges.numpy()))


def test_a_value_at_or_beside_an_edge():
    edges = np.linspace(-3, 7, 11)
    values = np.concatenate(
        [edges, np.nextafter(edges, -np.inf), np.nextafter(edges, np.inf)]
    )
    counts, got = mt.histogram(mt.Tensor(values), bins=10, range=(-3.0, 7.0))
    assert counts.tolist() == _searched(values, got.numpy()).tolist()
    assert counts.tolist() == [3.0] * 9 + [4.0]


def test_bins_narrower_than_a_rounding():
    # Eight bins across four ulps: the edges repeat, and the bin is still the
    # last edge at or below the value.
    ulp = 2.0**-52
    values = np.array([1.0 + k * ulp for k in range(5)])
    counts, edges = mt.histogram(
        mt.Tensor(values, dtype="float64"), bins=8, range=(1.0, 1.0 + 4 * ulp)
    )
    assert counts.tolist() == _searched(values, edges.numpy()).tolist()
    assert sum(counts.tolist()) == 5


def test_an_int64_past_two_to_the_53_is_placed_by_its_own_value():
    big = 2**60
    values = mt.Tensor(np.array([big - 1, big, big + 1000, -big, 2**62]))
    counts, _ = mt.histogram(values, bins=mt.Tensor([0.0, float(big), 2.0**61]))
    assert counts.tolist() == [1.0, 2.0]


def test_a_range_from_int64_data_encloses_the_largest_value():
    # 2^53 + 1 rounds down to 2^53; a range ending there would leave it out.
    counts, edges = mt.histogram(mt.Tensor(np.array([0, 2**53 + 1])), bins=2)
    assert counts.tolist() == [1.0, 1.0]
    assert edges.tolist()[-1] >= 2**53 + 1


def test_a_span_past_the_float_range_still_has_finite_edges():
    values = mt.Tensor([0.0, 1e308, -1e308], dtype="float64")
    counts, edges = mt.histogram(values, bins=4, range=(-1e308, 1e308))
    assert edges.tolist() == [-1e308, -5e307, 0.0, 5e307, 1e308]
    assert counts.tolist() == [1.0, 0.0, 1.0, 1.0]


def test_histc_spans_the_data_when_its_bounds_are_equal():
    assert mt.histc(mt.Tensor([1.0, 2.0, 3.0]), bins=4).tolist() == [
        1.0,
        0.0,
        1.0,
        1.0,
    ]
