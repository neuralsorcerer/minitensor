# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""`histogramdd`, and the `linspace` endpoint that building it exposed.

`histogram` counts along a line and there was nothing that counts in a box.
Which cell a point falls in is the same question in every dimension, so the
axes are bucketed separately and `ravel_multi_index` folds the coordinates into
the one flat position `bincount` counts. Nothing iterates over cells, which
matters because the grid is the thing that grows exponentially: a ten-bin
histogram over six dimensions is a million cells and this touches each sample
once regardless.

Writing it turned up a real bug underneath. `linspace(a, b, n)` computed every
element as `a + i * step`, so its last element sat a few ulps from `b` rather
than on it. A histogram closes its last bin on its top edge and finds that edge
by comparing against it, so the largest sample was being dropped -- one count
missing from one cell, which is exactly the kind of thing that never gets
noticed. The endpoint is now written rather than computed, and there are tests
for that here because this is where it was found.
"""

from __future__ import annotations

import numpy as np
import pytest

import minitensor as mt

RNG = np.random.default_rng(83)


def _t(values):
    return mt.Tensor(
        np.ascontiguousarray(np.asarray(values, dtype=np.float64)), dtype="float64"
    )


# --- the endpoint -----------------------------------------------------------


@pytest.mark.parametrize(
    "start,stop,steps",
    [(0.0, 1.0, 7), (-3.0, 3.0660367390488967, 6), (1e-8, 1e8, 101), (2.0, 2.0, 1), (0.0, 1.0, 2)],
)
def test_linspace_ends_exactly_where_it_was_told_to(start, stop, steps):
    """`a + (n-1) * step` is a few ulps off, and comparisons against it fail."""

    values = mt.Tensor.linspace(start, stop, steps, dtype="float64").numpy()
    assert values[0] == start
    assert values[-1] == stop
    np.testing.assert_allclose(values, np.linspace(start, stop, steps), rtol=0, atol=0)


def test_logspace_ends_exactly_where_it_was_told_to():
    values = mt.Tensor.logspace(0.0, 3.0, 7, 10.0, dtype="float64").numpy()
    assert values[-1] == 1000.0
    assert values[0] == 1.0


# --- against numpy ----------------------------------------------------------


@pytest.mark.parametrize(
    "shape,bins", [((200, 2), 5), ((300, 3), (4, 3, 2)), ((50, 1), 8), ((120, 4), 3)]
)
def test_histogramdd_matches_numpy(shape, bins):
    sample = RNG.normal(size=shape)
    counts, edges = mt.histogramdd(_t(sample), bins)
    expected, expected_edges = np.histogramdd(sample, bins=bins)

    np.testing.assert_array_equal(counts.numpy(), expected)
    for mine, theirs in zip(edges, expected_edges):
        np.testing.assert_allclose(mine.numpy(), theirs, rtol=1e-13)


@pytest.mark.parametrize(
    "bounds",
    [((-1.0, 1.0), (-2.0, 2.0)), (-1.0, 1.0, -2.0, 2.0)],
    ids=["pairs", "flat"],
)
def test_both_spellings_of_the_range_are_taken(bounds):
    """A sequence of pairs, and the flat form `torch.histogramdd` uses."""

    sample = RNG.normal(size=(150, 2))
    counts, _ = mt.histogramdd(_t(sample), 6, bounds)
    expected, _ = np.histogramdd(sample, bins=6, range=[(-1.0, 1.0), (-2.0, 2.0)])
    np.testing.assert_array_equal(counts.numpy(), expected)


def test_the_edges_may_be_given_outright_and_need_not_be_even():
    sample = RNG.normal(size=(100, 2))
    given = [np.array([-3.0, -1.0, 0.0, 1.0, 3.0]), np.array([-2.0, 0.0, 2.0])]
    counts, edges = mt.histogramdd(_t(sample), [_t(edge) for edge in given])
    np.testing.assert_array_equal(
        counts.numpy(), np.histogramdd(sample, bins=given)[0]
    )
    for mine, theirs in zip(edges, given):
        np.testing.assert_array_equal(mine.numpy(), theirs)


def test_weights_are_summed_rather_than_counted():
    sample = RNG.normal(size=(100, 2))
    weights = RNG.uniform(size=100)
    np.testing.assert_allclose(
        mt.histogramdd(_t(sample), 4, weight=_t(weights))[0].numpy(),
        np.histogramdd(sample, bins=4, weights=weights)[0],
        rtol=1e-12,
    )


def test_density_divides_by_the_total_and_by_each_cell_s_own_volume():
    sample = RNG.normal(size=(100, 2))
    given = [np.array([-3.0, -1.0, 0.0, 3.0]), np.array([-2.0, 0.0, 1.0, 2.0])]
    density, edges = mt.histogramdd(_t(sample), [_t(edge) for edge in given], density=True)
    np.testing.assert_allclose(
        density.numpy(), np.histogramdd(sample, bins=given, density=True)[0], rtol=1e-12
    )
    # Uneven cells, so this is a real test that each was divided by its own.
    volume = np.outer(np.diff(given[0]), np.diff(given[1]))
    assert float((density.numpy() * volume).sum()) == pytest.approx(1.0, rel=1e-12)


# --- the edges of the box ---------------------------------------------------


def test_the_top_edge_lands_in_the_last_bin():
    """The bug the `linspace` fix was hiding: the largest sample was dropped."""

    sample = np.array([[0.0], [0.5], [1.0]])
    counts, _ = mt.histogramdd(_t(sample), 2, ((0.0, 1.0),))
    np.testing.assert_array_equal(counts.numpy(), [1.0, 2.0])
    # And with the edges computed from the data, where the top edge *is* the
    # maximum, every sample has to be counted.
    counts, _ = mt.histogramdd(_t(RNG.normal(size=(500, 3))), 4)
    assert float(counts.numpy().sum()) == 500.0


def test_a_point_outside_any_dimension_is_dropped():
    sample = np.array([[0.5, 0.5], [0.5, 9.0], [-9.0, 0.5]])
    counts, _ = mt.histogramdd(_t(sample), 2, ((0.0, 1.0), (0.0, 1.0)))
    assert float(counts.numpy().sum()) == 1.0


def test_a_dimension_that_never_varies_is_widened_rather_than_divided_by_zero():
    counts, edges = mt.histogramdd(_t(np.full((5, 1), 2.0)), 3)
    assert float(counts.numpy().sum()) == 5.0
    np.testing.assert_allclose(edges[0].numpy()[[0, -1]], [1.5, 2.5])


def test_a_one_dimensional_sample_is_read_as_a_single_column():
    counts, _ = mt.histogramdd(_t([0.5, 1.5, 2.5]), 3)
    np.testing.assert_array_equal(counts.numpy(), [1.0, 1.0, 1.0])


# --- what it refuses --------------------------------------------------------


def test_a_sample_of_the_wrong_rank_is_refused():
    with pytest.raises(ValueError, match="points, dimensions"):
        mt.histogramdd(_t(np.zeros((2, 3, 4))), 3)


def test_the_wrong_number_of_bin_counts_is_refused():
    with pytest.raises(ValueError, match="one bin count per dimension"):
        mt.histogramdd(_t(np.zeros((5, 2))), (3, 4, 5))


def test_a_range_of_the_wrong_length_is_refused():
    with pytest.raises(ValueError, match="flat range of 4 values or 2 pairs"):
        mt.histogramdd(_t(np.zeros((5, 2))), 3, (0.0, 1.0, 2.0))


def test_a_bin_count_below_one_is_refused():
    with pytest.raises(ValueError, match="at least one bin"):
        mt.histogramdd(_t(np.zeros((5, 2))), 0)


def test_an_empty_edge_vector_is_refused():
    with pytest.raises(ValueError, match="at least two edges"):
        mt.histogramdd(_t(np.zeros((5, 1))), [_t([1.0])])


def test_an_integer_sample_is_refused():
    with pytest.raises(ValueError, match="floating point"):
        mt.histogramdd(mt.Tensor.zeros([4, 2], dtype="int64"), 3)
