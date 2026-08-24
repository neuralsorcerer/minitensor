# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""Asking a sorted sequence where a value belongs.

Four operations and one binary search. `searchsorted` is the search;
`bucketize` is the same call with the arguments the other way round;
`histogram` is the search followed by a count; `histc` is `histogram` with the
edges chosen for you.

None of it composes out of what the library had. Comparing every value against
every boundary is `O(values * boundaries)` and still leaves the counting to do,
and nothing else here knows how to exploit the fact that a sequence is sorted.

The comparisons are against NumPy because NumPy defines these -- particularly
the two conventions that are easy to get backwards and impossible to guess:
which side of a run of equal elements an insertion lands on, and the fact that a
histogram's last bin is closed on the right while every other one is half-open.
Both have a test that would fail if the convention flipped.
"""

from __future__ import annotations

import numpy as np
import pytest

import minitensor as mt


def _t(a):
    return mt.Tensor.from_numpy(np.ascontiguousarray(a))


SORTED = np.array([1.0, 3.0, 5.0, 7.0, 9.0])
PROBES = np.array([0.0, 1.0, 2.0, 5.0, 9.0, 10.0])


# --------------------------------------------------------------------------
# searchsorted
# --------------------------------------------------------------------------


@pytest.mark.parametrize("right", [False, True])
def test_searchsorted_matches_numpy(right):
    side = "right" if right else "left"
    got = mt.searchsorted(_t(SORTED), _t(PROBES), right).numpy()
    assert np.array_equal(got, np.searchsorted(SORTED, PROBES, side=side))


def test_the_two_sides_differ_exactly_on_an_element():
    """The whole content of `right`: where an insertion lands among equals.

    Away from the sequence's own values the two agree, so a test on random
    probes would pass with the flag ignored entirely.
    """
    on = np.array([1.0, 5.0, 9.0])
    left = mt.searchsorted(_t(SORTED), _t(on), False).numpy()
    right = mt.searchsorted(_t(SORTED), _t(on), True).numpy()
    assert np.array_equal(left, [0, 2, 4])
    assert np.array_equal(right, [1, 3, 5])

    between = np.array([0.0, 2.0, 10.0])
    assert np.array_equal(
        mt.searchsorted(_t(SORTED), _t(between), False).numpy(),
        mt.searchsorted(_t(SORTED), _t(between), True).numpy(),
    )


def test_repeated_elements():
    """A run of equals is where the two sides are furthest apart."""
    repeated = np.array([1.0, 2.0, 2.0, 2.0, 3.0])
    for right in (False, True):
        side = "right" if right else "left"
        got = mt.searchsorted(_t(repeated), _t(np.array([2.0])), right).numpy()
        assert np.array_equal(got, np.searchsorted(repeated, [2.0], side=side))


def test_a_value_past_the_end_answers_with_the_length():
    """Not clamped into the sequence: the answer is an insertion point, and
    past the end is a real insertion point."""
    assert mt.searchsorted(_t(SORTED), _t(np.array([100.0])), False).item() == 5
    assert mt.searchsorted(_t(SORTED), _t(np.array([-100.0])), False).item() == 0


def test_the_result_is_int64():
    result = mt.searchsorted(_t(SORTED), _t(PROBES), False)
    assert result.dtype == "int64"
    assert not result.requires_grad


def test_the_result_keeps_the_values_shape():
    for shape in [(6,), (2, 3), (1, 2, 3)]:
        got = mt.searchsorted(_t(SORTED), _t(PROBES.reshape(shape)), False).numpy()
        assert got.shape == shape
        assert np.array_equal(got.reshape(-1), np.searchsorted(SORTED, PROBES))


@pytest.mark.parametrize(
    "dtype", [np.float32, np.float64, np.int32, np.int64]
)
def test_every_numeric_dtype(dtype):
    sequence = np.array([1, 3, 5, 7], dtype=dtype)
    probes = np.array([0, 3, 6, 9], dtype=dtype)
    got = mt.searchsorted(_t(sequence), _t(probes), False).numpy()
    assert np.array_equal(got, np.searchsorted(sequence, probes))


def test_batched_sequences_are_matched_row_for_row():
    sequences = np.array([[1.0, 3.0, 5.0], [2.0, 4.0, 6.0]])
    probes = np.array([[2.0, 4.0], [1.0, 7.0]])
    got = mt.searchsorted(_t(sequences), _t(probes), False).numpy()
    expected = np.stack(
        [np.searchsorted(sequences[row], probes[row]) for row in range(2)]
    )
    assert np.array_equal(got, expected)


def test_a_three_dimensional_batch():
    rng = np.random.default_rng(0)
    sequences = np.sort(rng.standard_normal((2, 3, 5)), axis=-1)
    probes = rng.standard_normal((2, 3, 4))
    got = mt.searchsorted(_t(sequences), _t(probes), False).numpy()
    for i in range(2):
        for j in range(3):
            assert np.array_equal(
                got[i, j], np.searchsorted(sequences[i, j], probes[i, j])
            )


def test_an_empty_sequence_puts_everything_at_zero():
    got = mt.searchsorted(_t(np.zeros(0)), _t(PROBES), False).numpy()
    assert np.array_equal(got, np.zeros(6, dtype=np.int64))


def test_no_values_gives_no_answers():
    assert mt.searchsorted(_t(SORTED), _t(np.zeros(0)), False).numpy().shape == (0,)


def test_rejects_mismatched_dtypes():
    with pytest.raises(Exception, match="they must match"):
        mt.searchsorted(_t(SORTED), _t(np.array([1, 2], dtype=np.int64)), False)


def test_rejects_a_batched_sequence_that_does_not_line_up():
    with pytest.raises(Exception, match="must match its values"):
        mt.searchsorted(
            _t(np.zeros((3, 4))), _t(np.zeros((2, 4))), False
        )


# --------------------------------------------------------------------------
# bucketize
# --------------------------------------------------------------------------


@pytest.mark.parametrize("right", [False, True])
def test_bucketize_is_searchsorted_with_the_arguments_swapped(right):
    boundaries = np.array([1.0, 3.0, 5.0])
    got = mt.bucketize(_t(PROBES), _t(boundaries), right).numpy()
    assert np.array_equal(
        got, mt.searchsorted(_t(boundaries), _t(PROBES), right).numpy()
    )
    side = "right" if right else "left"
    assert np.array_equal(got, np.searchsorted(boundaries, PROBES, side=side))


@pytest.mark.parametrize(
    "boundaries,values",
    [
        ([0.0, 1.0, 2.0], [-1.0, 0.5, 1.5, 3.0]),
        ([10.0], [5.0, 10.0, 15.0]),
        ([-2.0, -1.0, 0.0, 1.0, 2.0], [-3.0, -1.5, 0.0, 1.7, 9.0]),
    ],
)
@pytest.mark.parametrize("right", [False, True])
def test_bucketize_puts_the_arguments_in_the_right_order(boundaries, values, right):
    """The two arguments are both one-dimensional sequences of numbers, so
    passing them the wrong way round is a mistake that still runs and still
    returns something shaped plausibly. Only the answer differs."""
    boundaries, values = np.array(boundaries), np.array(values)
    got = mt.bucketize(_t(values), _t(boundaries), right).numpy()
    side = "right" if right else "left"
    assert np.array_equal(got, np.searchsorted(boundaries, values, side=side))
    assert got.shape == values.shape


def test_bucketize_keeps_the_input_shape():
    boundaries = np.array([0.0, 1.0])
    got = mt.bucketize(_t(np.zeros((2, 3))), _t(boundaries), False).numpy()
    assert got.shape == (2, 3)


def test_bucketize_rejects_multidimensional_boundaries():
    with pytest.raises(Exception, match="one-dimensional"):
        mt.bucketize(_t(np.zeros(3)), _t(np.zeros((2, 2))), False)


# --------------------------------------------------------------------------
# histogram
# --------------------------------------------------------------------------


@pytest.mark.parametrize("bins", [1, 3, 7, 10, 64])
def test_histogram_matches_numpy(bins):
    data = np.random.default_rng(1).standard_normal(500)
    counts, edges = mt.histogram(_t(data), bins)
    want_counts, want_edges = np.histogram(data, bins=bins)
    assert np.allclose(counts.numpy(), want_counts)
    assert np.allclose(edges.numpy(), want_edges)


def test_histogram_over_an_explicit_range():
    data = np.random.default_rng(2).standard_normal(400)
    counts, edges = mt.histogram(_t(data), 10, (-2.0, 2.0))
    want_counts, want_edges = np.histogram(data, bins=10, range=(-2.0, 2.0))
    assert np.allclose(counts.numpy(), want_counts)
    assert np.allclose(edges.numpy(), want_edges)


def test_values_outside_the_range_are_dropped_not_clamped():
    """What makes a histogram over an explicit range mean what it says.

    Clamping would pile everything below the range into the first bin, which is
    a different and usually wrong answer.
    """
    data = np.array([-10.0, 0.5, 1.5, 10.0])
    counts, _ = mt.histogram(_t(data), 2, (0.0, 2.0))
    assert np.allclose(counts.numpy(), [1.0, 1.0])
    assert counts.numpy().sum() == 2


def test_the_last_bin_is_closed_on_the_right():
    """Every other bin is half-open. The asymmetry is not an oversight: without
    it the largest value in the data would fall out of the histogram."""
    data = np.array([0.0, 0.5, 1.0])
    counts, _ = mt.histogram(_t(data), 2, (0.0, 1.0))
    assert np.allclose(counts.numpy(), [1.0, 2.0])
    assert np.allclose(counts.numpy(), np.histogram(data, bins=2, range=(0.0, 1.0))[0])


def test_histogram_with_explicit_edges():
    data = np.random.default_rng(3).standard_normal(300)
    edges = np.array([-3.0, -1.0, 0.0, 0.5, 3.0])
    counts, back = mt.histogram(_t(data), _t(edges))
    want, _ = np.histogram(data, bins=edges)
    assert np.allclose(counts.numpy(), want)
    assert np.allclose(back.numpy(), edges)


def test_histogram_with_weights():
    data = np.random.default_rng(4).standard_normal(200)
    weights = np.random.default_rng(5).standard_normal(200)
    counts, _ = mt.histogram(_t(data), 8, None, _t(weights))
    want, _ = np.histogram(data, bins=8, weights=weights)
    assert np.allclose(counts.numpy(), want)


def test_density_integrates_to_one():
    data = np.random.default_rng(6).standard_normal(1000)
    counts, edges = mt.histogram(_t(data), 20, None, None, True)
    want, _ = np.histogram(data, bins=20, density=True)
    assert np.allclose(counts.numpy(), want)
    widths = np.diff(edges.numpy())
    assert np.isclose((counts.numpy() * widths).sum(), 1.0)


def test_density_is_comparable_across_binnings():
    """The property `density` exists for: the same distribution binned two ways
    gives two curves on the same scale."""
    data = np.random.default_rng(7).standard_normal(4000)
    coarse, coarse_edges = mt.histogram(_t(data), 10, (-3.0, 3.0), None, True)
    fine, fine_edges = mt.histogram(_t(data), 40, (-3.0, 3.0), None, True)
    assert np.isclose((coarse.numpy() * np.diff(coarse_edges.numpy())).sum(), 1.0, atol=0.05)
    assert np.isclose((fine.numpy() * np.diff(fine_edges.numpy())).sum(), 1.0, atol=0.05)
    assert abs(coarse.numpy().max() - fine.numpy().max()) < 0.2


@pytest.mark.parametrize("bins", [3, 7, 9, 11, 13, 97])
@pytest.mark.parametrize("bounds", [(0.0, 1.0), (-1.0, 1.0), (0.1, 0.7), (-2.5, 3.5)])
def test_the_top_edge_is_exactly_what_was_asked_for(bins, bounds):
    """Walking `low + k * width` up to the top does not always land on it.

    The top edge is the one the closed last bin depends on: a value sitting
    exactly at the requested maximum has to fall inside the histogram, and if
    the accumulated arithmetic stops a rounding short of it the value is
    silently dropped instead. So the last edge is written rather than computed.
    """
    top = np.array([bounds[1]])
    counts, edges = mt.histogram(_t(top), bins, bounds)
    assert edges.numpy()[-1] == bounds[1]
    assert edges.numpy()[0] == bounds[0]
    assert counts.numpy()[-1] == 1.0
    assert counts.numpy().sum() == 1.0


@pytest.mark.parametrize(
    "low,high,bins",
    [
        (-0.00063319409019222674, 0.37693031114261599, 147),
        (-0.29758403894333035, -0.061429409090388326, 102),
        (0.0017637590695593738, 94.155919956116051, 85),
    ],
)
def test_a_range_whose_last_edge_the_arithmetic_lands_short_of(low, high, bins):
    """The cases the previous test could not find, searched for on purpose.

    `low + bins * ((high - low) / bins)` usually lands exactly on `high`, which
    is why bounds picked by hand do not exercise this at all -- twenty-four of
    them did not. These three are the result of hunting for triples where it
    falls short instead, by as little as `6e-17`. A value sitting at `high` is
    then *above* the computed last edge and is dropped from its own histogram,
    so the last edge is written rather than accumulated.
    """
    assert low + bins * ((high - low) / bins) < high, "this triple no longer misses"

    counts, edges = mt.histogram(_t(np.array([high])), bins, (low, high))
    assert edges.numpy()[-1] == high
    assert counts.numpy().sum() == 1.0
    assert counts.numpy()[-1] == 1.0


@pytest.mark.parametrize("value", [-1.0, -0.5, 0.0, 0.5, 1.0])
def test_a_value_sitting_exactly_on_an_edge(value):
    """Interior edges belong to the bin above them and the top edge to the bin
    below it, which is the whole content of the half-open convention."""
    counts, edges = mt.histogram(_t(np.array([value])), 4, (-1.0, 1.0))
    want, _ = np.histogram([value], bins=4, range=(-1.0, 1.0))
    assert np.allclose(counts.numpy(), want)
    assert counts.numpy().sum() == 1.0


def test_density_over_unequal_bins():
    """Where dividing by the width stops being a scale factor and starts being
    the point: with edges of different widths, a density that only divided by
    the total would be wrong bin by bin rather than uniformly."""
    data = np.random.default_rng(20).standard_normal(2000)
    edges = np.array([-4.0, -1.0, -0.5, 0.0, 0.5, 1.0, 4.0])
    counts, back = mt.histogram(_t(data), _t(edges), None, None, True)
    want, _ = np.histogram(data, bins=edges, density=True)
    assert np.allclose(counts.numpy(), want)
    assert np.isclose((counts.numpy() * np.diff(back.numpy())).sum(), 1.0)


@pytest.mark.parametrize("value", [0.0, 2.0, -3.5, 1e6])
@pytest.mark.parametrize("bins", [1, 3, 8])
def test_constant_data_is_opened_out_around_its_value(value, bins):
    """A range of no width has to become one somehow, and the rule has to put
    the sample inside the result rather than on its boundary."""
    data = np.full(5, value)
    counts, edges = mt.histogram(_t(data), bins)
    want_counts, want_edges = np.histogram(data, bins=bins)
    assert np.allclose(counts.numpy(), want_counts)
    assert np.allclose(edges.numpy(), want_edges)
    assert counts.numpy().sum() == 5


def test_constant_data_still_produces_bins():
    """A range of no width has to be opened somehow, and NumPy's rule is half a
    unit either side -- the only one that puts the sample somewhere sensible."""
    data = np.array([2.0, 2.0, 2.0])
    counts, edges = mt.histogram(_t(data), 3)
    want_counts, want_edges = np.histogram(data, bins=3)
    assert np.allclose(counts.numpy(), want_counts)
    assert np.allclose(edges.numpy(), want_edges)


def test_histogram_flattens_its_input():
    """A histogram is a question about a collection of numbers, not about their
    arrangement."""
    data = np.random.default_rng(8).standard_normal((4, 25))
    flat, _ = mt.histogram(_t(data.reshape(-1)), 5, (-3.0, 3.0))
    shaped, _ = mt.histogram(_t(data), 5, (-3.0, 3.0))
    assert np.allclose(flat.numpy(), shaped.numpy())


def test_histogram_of_integers():
    data = np.array([1, 2, 2, 3, 3, 3], dtype=np.int64)
    counts, _ = mt.histogram(_t(data), 3, (1.0, 4.0))
    assert np.allclose(counts.numpy(), [1.0, 2.0, 3.0])


def test_non_finite_values_are_skipped():
    data = np.array([0.5, np.nan, np.inf, -np.inf, 1.5])
    counts, _ = mt.histogram(_t(data), 2, (0.0, 2.0))
    assert np.allclose(counts.numpy(), [1.0, 1.0])


def test_histogram_of_nothing():
    counts, edges = mt.histogram(_t(np.zeros(0)), 4)
    assert counts.numpy().shape == (4,)
    assert np.allclose(counts.numpy(), 0.0)
    assert edges.numpy().shape == (5,)


def test_histogram_rejects_bad_bins():
    data = _t(np.zeros(4))
    with pytest.raises(Exception, match="at least one bin"):
        mt.histogram(data, 0)
    with pytest.raises(Exception, match="at least two bin edges"):
        mt.histogram(data, _t(np.array([1.0])))
    with pytest.raises(Exception, match="must increase"):
        mt.histogram(data, _t(np.array([1.0, 0.0, 2.0])))
    with pytest.raises(Exception, match="one-dimensional"):
        mt.histogram(data, _t(np.zeros((2, 2))))


def test_histogram_rejects_mismatched_weights():
    with pytest.raises(Exception, match="weights for"):
        mt.histogram(_t(np.zeros(4)), 2, None, _t(np.zeros(3)))


def test_the_counts_add_up():
    """Nothing is counted twice and nothing inside the range is lost."""
    data = np.random.default_rng(9).standard_normal(777)
    counts, _ = mt.histogram(_t(data), 13)
    assert counts.numpy().sum() == 777


# --------------------------------------------------------------------------
# histc
# --------------------------------------------------------------------------


def test_histc_matches_a_histogram_over_the_same_range():
    data = np.random.default_rng(10).standard_normal(300)
    got = mt.histc(_t(data), 10, -2.0, 2.0).numpy()
    assert np.allclose(got, np.histogram(data, bins=10, range=(-2.0, 2.0))[0])


@pytest.mark.parametrize("bins", [1, 5, 20])
def test_histc_over_the_data_range_counts_everything(bins):
    """Equal bounds mean the data's own range, so nothing can fall outside it
    and every element has to be counted."""
    data = np.random.default_rng(21).standard_normal(400)
    counts = mt.histc(_t(data), bins).numpy()
    assert counts.sum() == 400
    assert np.allclose(counts, np.histogram(data, bins=bins)[0])


def test_histc_with_explicit_bounds_drops_what_is_outside():
    """The other half of the same rule: unequal bounds are a real range, and a
    real range excludes."""
    data = np.array([-5.0, 0.5, 1.5, 5.0])
    assert mt.histc(_t(data), 2, 0.0, 2.0).numpy().sum() == 2


def test_histc_spans_the_data_when_the_bounds_are_equal():
    """PyTorch's rule, and the reason `histc` is not simply `histogram`: equal
    bounds mean "use the data's own range" rather than "an empty range"."""
    data = np.random.default_rng(11).standard_normal(300)
    got = mt.histc(_t(data), 10).numpy()
    assert np.allclose(got, np.histogram(data, bins=10)[0])


def test_histc_returns_counts_alone():
    result = mt.histc(_t(np.zeros(5)), 4)
    assert result.numpy().shape == (4,)


# --------------------------------------------------------------------------
# What these are for
# --------------------------------------------------------------------------


def test_quantile_lookup_by_search():
    """The use `searchsorted` exists for: mapping values onto a precomputed
    table without a comparison against every entry."""
    table = np.linspace(0.0, 1.0, 101)
    values = np.array([0.0, 0.155, 0.5, 0.999, 1.0])
    got = mt.searchsorted(_t(table), _t(values), False).numpy()
    assert np.array_equal(got, np.searchsorted(table, values))


def test_binning_a_feature_for_a_lookup_table():
    """What `bucketize` is for: turning a continuous feature into an integer
    index, which is then an embedding lookup."""
    boundaries = np.array([-1.0, 0.0, 1.0])
    feature = np.array([-2.0, -0.5, 0.5, 2.0])
    indices = mt.bucketize(_t(feature), _t(boundaries), False).numpy()
    assert np.array_equal(indices, [0, 1, 2, 3])
    assert indices.max() < len(boundaries) + 1


def test_a_histogram_recovers_a_known_distribution():
    """Uniform data should fill its bins evenly, which is a property of the
    binning rather than a comparison against another implementation."""
    data = np.random.default_rng(12).uniform(0.0, 1.0, 100_000)
    counts, _ = mt.histogram(_t(data), 10, (0.0, 1.0))
    assert np.allclose(counts.numpy(), 10_000, rtol=0.05)
