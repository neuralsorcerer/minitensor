# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""The statistics NumPy has that this library did not.

Each of these is a short arrangement of kernels that were already here --
`quantile` in percent, two reductions for a span, a binary search for an
interpolation -- which is why they belong in Python rather than in the
extension. What they are not is optional: a library that has `quantile` but no
`percentile`, or `cumsum` but no `nancumsum`, makes its user write the
arrangement themselves and get the edges wrong.

The edges are where these differ from a naive version, so that is what these
tests pin, against NumPy in every case:

* `interp` outside the sample range, exactly on a sample, and wrapped by a
  period.
* `digitize` on decreasing edges, where counting back from the total flips
  which side of an edge is inside it -- the reason it is not just
  `searchsorted`.
* `average` with weights that are one-dimensional along the reduced axis
  rather than shaped like the input.
* `nancumsum` where a NaN must contribute nothing rather than poison every
  total after it.
"""

from __future__ import annotations

import numpy as np
import pytest

import minitensor as mt


@pytest.fixture()
def values():
    return np.random.default_rng(31).standard_normal((6, 5))


def test_ptp_is_the_span_of_the_values(values):
    np.testing.assert_allclose(mt.ptp(mt.from_numpy(values)).numpy(), np.ptp(values))
    np.testing.assert_allclose(
        mt.ptp(mt.from_numpy(values), 0).numpy(), np.ptp(values, 0)
    )
    np.testing.assert_allclose(
        mt.ptp(mt.from_numpy(values), 1, True).numpy(), np.ptp(values, 1, keepdims=True)
    )
    # Integers have a span too, and it is not a float.
    counts = np.array([3, 1, 7], dtype=np.int64)
    assert int(mt.ptp(mt.from_numpy(counts)).item()) == 6


def test_average_weights_the_mean(values):
    rng = np.random.default_rng(37)
    along = np.abs(rng.standard_normal(6)) + 0.1
    everywhere = np.abs(rng.standard_normal((6, 5))) + 0.1

    np.testing.assert_allclose(
        mt.average(mt.from_numpy(values)).numpy(), np.average(values)
    )
    np.testing.assert_allclose(
        mt.average(mt.from_numpy(values), 0).numpy(), np.average(values, 0)
    )
    # One weight per position along the reduced axis, broadcast over the rest.
    np.testing.assert_allclose(
        mt.average(mt.from_numpy(values), 0, mt.from_numpy(along)).numpy(),
        np.average(values, 0, along),
    )
    np.testing.assert_allclose(
        mt.average(mt.from_numpy(values), 1, mt.from_numpy(everywhere)).numpy(),
        np.average(values, 1, everywhere),
    )
    np.testing.assert_allclose(
        mt.average(mt.from_numpy(values), None, mt.from_numpy(everywhere)).numpy(),
        np.average(values, None, everywhere),
    )

    mean, total = mt.average(
        mt.from_numpy(values), 0, mt.from_numpy(along), returned=True
    )
    expected, expected_total = np.average(values, 0, along, returned=True)
    np.testing.assert_allclose(mean.numpy(), expected)
    np.testing.assert_allclose(total.numpy(), expected_total)


def test_average_rejects_weights_that_do_not_line_up(values):
    with pytest.raises(ValueError, match="one weight per position"):
        mt.average(mt.from_numpy(values), 0, mt.from_numpy(np.ones(5)))
    with pytest.raises(ValueError, match="shaped like the input"):
        mt.average(mt.from_numpy(values), None, mt.from_numpy(np.ones(3)))


def test_percentile_is_quantile_in_percent(values):
    for q in [0, 25, 50, 99.5, 100]:
        np.testing.assert_allclose(
            mt.percentile(mt.from_numpy(values), q).numpy(), np.percentile(values, q)
        )
        np.testing.assert_allclose(
            mt.percentile(mt.from_numpy(values), q, 1).numpy(),
            np.percentile(values, q, 1),
        )
    np.testing.assert_allclose(
        mt.percentile(mt.from_numpy(values), [25, 50, 75]).numpy(),
        np.percentile(values, [25, 50, 75]),
    )

    poisoned = values.copy()
    poisoned[0, 0] = np.nan
    np.testing.assert_allclose(
        mt.nanpercentile(mt.from_numpy(poisoned), 40).numpy(),
        np.nanpercentile(poisoned, 40),
    )

    with pytest.raises(ValueError, match=r"\[0, 100\]"):
        mt.percentile(mt.from_numpy(values), 101)


SAMPLES = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
CURVE = np.array([0.0, 1.0, 4.0, 9.0, 16.0])


def test_interp_matches_numpy_inside_outside_and_on_the_samples():
    queries = np.array([-5.0, -1e-9, 0.0, 0.5, 1.0, 2.0, 3.7, 4.0, 4.0 + 1e-9, 9.0])
    np.testing.assert_allclose(
        mt.interp(mt.from_numpy(queries), SAMPLES, CURVE).numpy(),
        np.interp(queries, SAMPLES, CURVE),
    )
    # A query landing exactly on a sample returns that sample's value exactly,
    # not to within a rounding of the slope.
    exact = mt.interp(mt.from_numpy(SAMPLES), SAMPLES, CURVE).numpy()
    np.testing.assert_array_equal(exact, CURVE)


def test_interp_holds_or_substitutes_outside_the_range():
    queries = np.array([-1.0, 5.0])
    np.testing.assert_allclose(
        mt.interp(mt.from_numpy(queries), SAMPLES, CURVE).numpy(),
        np.interp(queries, SAMPLES, CURVE),
    )
    np.testing.assert_allclose(
        mt.interp(
            mt.from_numpy(queries), SAMPLES, CURVE, left=-99.0, right=99.0
        ).numpy(),
        np.interp(queries, SAMPLES, CURVE, left=-99.0, right=99.0),
    )
    # `left` applies below the first sample, not on it.
    on_the_edge = mt.interp(mt.from_numpy(np.array([0.0])), SAMPLES, CURVE, left=-99.0)
    assert float(on_the_edge.numpy()[0]) == 0.0


def test_interp_wraps_when_given_a_period():
    queries = np.array([-3.0, 0.5, 3.2, 7.0, 12.5])
    np.testing.assert_allclose(
        mt.interp(mt.from_numpy(queries), SAMPLES, CURVE, period=5.0).numpy(),
        np.interp(queries, SAMPLES, CURVE, period=5.0),
    )


def test_interp_keeps_the_shape_of_its_queries():
    queries = np.array([[0.5, 1.5], [2.5, 3.5]])
    got = mt.interp(mt.from_numpy(queries), SAMPLES, CURVE)
    assert tuple(got.shape) == (2, 2)
    np.testing.assert_allclose(got.numpy(), np.interp(queries, SAMPLES, CURVE))


def test_interp_with_one_sample_is_that_sample():
    np.testing.assert_allclose(
        mt.interp([0.0, 5.0], [2.0], [7.0]).numpy(), np.interp([0.0, 5.0], [2.0], [7.0])
    )


@pytest.mark.parametrize(
    "edges",
    [
        np.array([0.0, 1.0, 2.0, 3.0]),
        np.array([3.0, 2.0, 1.0, 0.0]),
        np.array([-1.0, 5.0]),
        np.array([5.0, -1.0]),
        np.array([0.0]),
    ],
    ids=["up", "down", "pair", "pair-down", "single"],
)
@pytest.mark.parametrize("right", [False, True])
def test_digitize_matches_numpy_in_both_directions(edges, right):
    rng = np.random.default_rng(41)
    # The edges themselves, and either side of them, are where the rule bites.
    queries = np.concatenate(
        [rng.standard_normal(20) * 3, edges, edges - 1e-12, edges + 1e-12]
    )
    np.testing.assert_array_equal(
        mt.digitize(mt.from_numpy(queries), mt.from_numpy(edges), right).numpy(),
        np.digitize(queries, edges, right),
    )


def test_histogram_bin_edges_are_the_ones_histogram_uses(values):
    flat = values.reshape(-1)
    edges = mt.histogram_bin_edges(mt.from_numpy(flat), 7)
    np.testing.assert_allclose(edges.numpy(), np.histogram_bin_edges(flat, 7))
    # And reusing them gives the same counts as asking for that bin count.
    counts, _ = mt.histogram(mt.from_numpy(flat), edges)
    expected, _ = np.histogram(flat, 7)
    np.testing.assert_array_equal(counts.numpy().astype(np.int64), expected)


@pytest.mark.parametrize("dtype", ["float32", "float64", "int64"])
@pytest.mark.parametrize(
    "bins,span",
    [(10, None), (3, None), (7, (-1.5, 2.5)), (1, None)],
)
def test_the_edges_are_the_ones_the_histogram_would_have_counted_into(
    dtype, bins, span
):
    """`histogram_bin_edges` used to run the whole histogram and drop the
    counts -- 12.4ms for a million values where the edges alone take 0.8 --
    so the two could not disagree. Now that the edges have their own path,
    that they agree is a claim rather than a tautology, and it is the claim
    the name makes.
    """
    rng = np.random.default_rng(5)
    data = (rng.standard_normal(500) * 3).astype(
        np.float32
        if dtype == "float32"
        else np.float64 if dtype == "float64" else np.int64
    )
    tensor = mt.from_numpy(data.copy())

    alone = mt.histogram_bin_edges(tensor, bins, span).numpy()
    _, alongside = mt.histogram(tensor, bins, span)
    np.testing.assert_array_equal(alone, alongside.numpy())

    # The edges are computed and returned in float64 whatever the input is --
    # which is what `histogram` always did -- where NumPy narrows them to the
    # input's own dtype. So a float32 input agrees only to float32's precision,
    # and the difference is on NumPy's side of the last bit.
    np.testing.assert_allclose(
        alone,
        np.histogram_bin_edges(data, bins, span),
        rtol=1e-6 if dtype == "float32" else 1e-12,
    )


def test_the_edges_of_the_awkward_inputs():
    """A constant column, an empty one, and one with a NaN in it: the three
    places the rule that picks the outer edges has to say something."""
    for data, expected in (
        (np.full(20, 3.0), np.histogram_bin_edges(np.full(20, 3.0), 4)),
        (np.zeros(0), np.histogram_bin_edges(np.zeros(0), 4)),
        # NumPy propagates the NaN into the edges and then raises; the finite
        # values are binned here instead, which is the behaviour `histogram`
        # already had and this now shares rather than re-deriving.
        (np.array([1.0, np.nan, 5.0]), np.array([1.0, 2.0, 3.0, 4.0, 5.0])),
    ):
        got = mt.histogram_bin_edges(mt.from_numpy(data.copy()), 4).numpy()
        np.testing.assert_allclose(got, expected, rtol=1e-12)
        _, alongside = mt.histogram(mt.from_numpy(data.copy()), 4)
        np.testing.assert_array_equal(got, alongside.numpy())


def test_histogram2d_matches_numpy(values):
    first, second = values[:, 0], values[:, 1]
    counts, x_edges, y_edges = mt.histogram2d(
        mt.from_numpy(first), mt.from_numpy(second), 4
    )
    expected, expected_x, expected_y = np.histogram2d(first, second, 4)
    np.testing.assert_allclose(counts.numpy(), expected)
    np.testing.assert_allclose(x_edges.numpy(), expected_x)
    np.testing.assert_allclose(y_edges.numpy(), expected_y)

    with pytest.raises(ValueError, match="same number of values"):
        mt.histogram2d(mt.from_numpy(first), mt.from_numpy(second[:-1]), 4)


def test_the_nan_aware_scans_skip_rather_than_poison(values):
    poisoned = values.copy()
    poisoned[1, 2] = np.nan
    poisoned[4, 0] = np.nan

    np.testing.assert_allclose(
        mt.nancumsum(mt.from_numpy(poisoned)).numpy(), np.nancumsum(poisoned)
    )
    np.testing.assert_allclose(
        mt.nancumsum(mt.from_numpy(poisoned), 1).numpy(), np.nancumsum(poisoned, 1)
    )
    np.testing.assert_allclose(
        mt.nancumprod(mt.from_numpy(poisoned), 0).numpy(), np.nancumprod(poisoned, 0)
    )
    # An integer tensor has no NaN to skip and comes back unchanged.
    counts = np.array([1, 2, 3], dtype=np.int64)
    np.testing.assert_array_equal(
        mt.nancumsum(mt.from_numpy(counts)).numpy(), np.cumsum(counts)
    )


def test_ediff1d_flattens_and_can_be_bracketed(values):
    np.testing.assert_allclose(
        mt.ediff1d(mt.from_numpy(values)).numpy(), np.ediff1d(values)
    )
    np.testing.assert_allclose(
        mt.ediff1d(mt.from_numpy(values), to_end=[9.0], to_begin=[-9.0]).numpy(),
        np.ediff1d(values, to_end=[9.0], to_begin=[-9.0]),
    )
    # One element has no differences, and none is not an error.
    assert mt.ediff1d(mt.from_numpy(np.array([5.0]))).numpy().size == 0
    assert mt.ediff1d(mt.from_numpy(np.array([], dtype=np.float64))).numpy().size == 0


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("length", [1, 7, 8, 9, 16, 17, 8191, 8192, 8193, 20_000])
def test_bin_edges_skip_the_non_finite_at_every_lane_and_chunk_seam(dtype, length):
    """The scan that finds the range folds several lanes at once and hands
    rayon fixed-size chunks, so a value's fate depends on where in a block and
    in a chunk it lands. These lengths straddle both seams.

    Skipping them is a divergence from NumPy, which raises on a non-finite
    range instead, so each case also pins that NumPy still refuses -- if a
    later NumPy starts skipping, the reason this test computes its own
    reference has gone away and should be revisited.

    It also takes two shapes: a cheap pass that asks nothing about the values,
    and a skipping pass taken only when the cheap one returns something
    non-finite. That is safe because a comparison against a NaN is false
    either way, so the cheap pass discards NaN without testing for it -- but
    only if the comparison is written with the candidate on the left. Written
    the other way it adopts the NaN and every edge becomes NaN, which is what
    this pins.
    """
    rng = np.random.default_rng(11)
    base = rng.standard_normal(length) * 3
    places = {
        "clean": [],
        "nan first": [(0, np.nan)],
        "nan last": [(length - 1, np.nan)],
        "inf first": [(0, np.inf)],
        "-inf last": [(length - 1, -np.inf)],
        "both ends": [(0, np.inf), (length - 1, -np.inf)],
        "inf and nan": [(0, np.inf), (length // 2, np.nan), (length - 1, -np.inf)],
    }
    for name, spec in places.items():
        data = base.copy()
        for index, value in spec:
            data[index] = value
        data = data.astype(dtype)
        got = mt.histogram_bin_edges(mt.from_numpy(data), 8).numpy()
        # There is no NumPy answer to compare against: it takes a plain min
        # and max, so one NaN makes its range [nan, nan] and it raises rather
        # than binning anything. The reference is the range of what is left
        # once the non-finite are dropped, which is what skipping them means.
        finite = data[np.isfinite(data)]
        low, high = (
            (0.0, 1.0)
            if finite.size == 0
            else (float(finite.min()), float(finite.max()))
        )
        # A range of no width -- one finite value, or all of them equal --
        # is opened half a unit either side, the rule NumPy uses too.
        if low == high:
            low, high = low - 0.5, high + 0.5
        want = np.linspace(low, high, 9)
        want[-1] = high
        np.testing.assert_allclose(
            got, want, rtol=1e-6 if dtype == "float32" else 1e-12, err_msg=name
        )
        assert np.all(np.isfinite(got)), name
        with pytest.raises(ValueError, match="not finite"):
            np.histogram_bin_edges(np.append(data, np.nan), 8)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_bin_edges_substitute_a_range_when_nothing_is_finite(dtype):
    """With no finite value there is no range to take, and the skipping pass
    returns its identities -- an empty range the caller replaces with the unit
    one rather than propagating an infinity into every edge.
    """
    for fill in (np.nan, np.inf, -np.inf):
        data = np.full(1000, fill, dtype=dtype)
        edges = mt.histogram_bin_edges(mt.from_numpy(data), 4).numpy()
        np.testing.assert_allclose(edges, np.linspace(0.0, 1.0, 5))
    mixed = np.array([np.nan, np.inf, -np.inf] * 400, dtype=dtype)
    np.testing.assert_allclose(
        mt.histogram_bin_edges(mt.from_numpy(mixed), 4).numpy(),
        np.linspace(0.0, 1.0, 5),
    )


@pytest.mark.parametrize("length", [1, 8, 9, 8193])
@pytest.mark.parametrize("content", ["all false", "all true", "mixed"])
def test_bin_edges_over_booleans(length, content):
    """A boolean's extremes are answered by two short-circuiting scans rather
    than by the fold, so they are a separate implementation and get their own
    check -- including the constant cases, where the range has no width and is
    opened half a unit either side.
    """
    if content == "all false":
        data = np.zeros(length, dtype=bool)
    elif content == "all true":
        data = np.ones(length, dtype=bool)
    else:
        data = (np.arange(length) % 2).astype(bool)
    low, high = float(data.min()), float(data.max())
    if low == high:
        low, high = low - 0.5, high + 0.5
    want = np.linspace(low, high, 5)
    want[-1] = high
    np.testing.assert_allclose(
        mt.histogram_bin_edges(mt.from_numpy(data), 4).numpy(), want
    )


def test_bin_edges_over_integers_wider_than_float64_can_hold():
    """The fold compares in the values' own width and widens only the answer,
    so two int64 that round to the same float64 are still ordered correctly.
    Folding in float64 -- which is what this used to do, one conversion per
    element -- compared them after rounding and could return either.
    """
    data = np.array([2**62 + 1, 2**62 + 3, -(2**62)], dtype=np.int64)
    edges = mt.histogram_bin_edges(mt.from_numpy(data), 2).numpy()
    assert edges[0] == float(-(2**62))
    assert edges[-1] == float(2**62 + 3)
