# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""The last of the derived helpers: shape, numerical calculus, and draws.

`unflatten`, `msort`, the three splits, `kthvalue` and `combinations` are
rearrangements of `reshape`, `sort` and `tensor_split`. `gradient` is the
numerical derivative of data, pinned to `numpy.gradient` in every mode it has.
The three draws are transformations of `rand` and `randn`, so what is checked
of them is the distribution they produce, over enough samples for the answer to
mean something.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

import minitensor as mt

RNG = np.random.default_rng(31)
GRID = np.arange(24.0).reshape(2, 3, 4)


def _t(values, dtype="float64"):
    return mt.Tensor(
        np.ascontiguousarray(np.asarray(values, dtype=np.float64)), dtype=dtype
    )


# --- unflatten --------------------------------------------------------------


def test_unflatten_is_the_inverse_of_flatten():
    tensor = _t(GRID)
    flat = mt.flatten(tensor, 1, 2)
    assert tuple(flat.shape) == (2, 12)
    np.testing.assert_array_equal(mt.unflatten(flat, 1, (3, 4)).numpy(), GRID)


def test_unflatten_infers_a_single_minus_one():
    tensor = _t(GRID)
    assert tuple(mt.unflatten(tensor, 2, (-1, 2)).shape) == (2, 3, 2, 2)
    assert tuple(mt.unflatten(tensor, 2, (2, -1)).shape) == (2, 3, 2, 2)


def test_unflatten_reports_sizes_that_do_not_fit():
    tensor = _t(GRID)
    with pytest.raises(ValueError, match="multiply to"):
        mt.unflatten(tensor, 2, (3, 3))
    with pytest.raises(ValueError, match="at most one dimension"):
        mt.unflatten(tensor, 2, (-1, -1))
    with pytest.raises(ValueError, match="cannot split"):
        mt.unflatten(tensor, 2, (-1, 3))


def test_unflatten_takes_a_negative_dim():
    tensor = _t(GRID)
    np.testing.assert_array_equal(
        mt.unflatten(tensor, -1, (2, 2)).numpy(), GRID.reshape(2, 3, 2, 2)
    )


# --- msort and the splits ---------------------------------------------------


def test_msort_sorts_the_rows_into_order():
    values = RNG.standard_normal((5, 3))
    np.testing.assert_allclose(mt.msort(_t(values)).numpy(), np.sort(values, axis=0))


@pytest.mark.parametrize(
    "name,reference,axis",
    [("hsplit", np.hsplit, 1), ("vsplit", np.vsplit, 0), ("dsplit", np.dsplit, 2)],
)
def test_the_splits_match_numpy(name, reference, axis):
    values = np.arange(24.0).reshape(2, 4, 3)
    got = getattr(mt, name)(_t(values), values.shape[axis])
    want = reference(values, values.shape[axis])
    assert len(got) == len(want)
    for piece, expected in zip(got, want):
        np.testing.assert_array_equal(piece.numpy(), expected)


def test_hsplit_takes_the_only_axis_a_vector_has():
    values = np.arange(6.0)
    got = mt.hsplit(_t(values), 3)
    for piece, expected in zip(got, np.hsplit(values, 3)):
        np.testing.assert_array_equal(piece.numpy(), expected)


def test_the_splits_refuse_a_rank_that_has_no_such_axis():
    with pytest.raises(ValueError, match="at least 2 dimensions"):
        mt.vsplit(_t(np.arange(4.0)), 2)
    with pytest.raises(ValueError, match="at least 3 dimensions"):
        mt.dsplit(_t(np.arange(4.0).reshape(2, 2)), 2)


# --- kthvalue ---------------------------------------------------------------


@pytest.mark.parametrize("k", [1, 2, 3, 4])
def test_kthvalue_matches_a_sort(k):
    values = RNG.standard_normal((3, 4))
    got, where = mt.kthvalue(_t(values), k, 1)
    want = np.sort(values, axis=1)[:, k - 1]
    np.testing.assert_allclose(got.numpy(), want)
    # And the index points back at the value it reported.
    np.testing.assert_allclose(
        np.take_along_axis(values, where.numpy()[:, None], 1).squeeze(1), want
    )


def test_kthvalue_at_one_is_the_minimum_and_at_n_the_maximum():
    values = RNG.standard_normal((3, 5))
    np.testing.assert_allclose(mt.kthvalue(_t(values), 1, 1)[0].numpy(), values.min(1))
    np.testing.assert_allclose(mt.kthvalue(_t(values), 5, 1)[0].numpy(), values.max(1))


def test_kthvalue_keeps_the_axis_when_asked():
    values = RNG.standard_normal((3, 4))
    got, where = mt.kthvalue(_t(values), 2, 1, True)
    assert tuple(got.shape) == (3, 1) and tuple(where.shape) == (3, 1)


def test_kthvalue_counts_from_one():
    with pytest.raises(ValueError, match="1 <= k"):
        mt.kthvalue(_t(RNG.standard_normal((3, 4))), 0, 1)
    with pytest.raises(ValueError, match="1 <= k"):
        mt.kthvalue(_t(RNG.standard_normal((3, 4))), 5, 1)


# --- combinations -----------------------------------------------------------


@pytest.mark.parametrize("r", [0, 1, 2, 3])
@pytest.mark.parametrize("with_replacement", [False, True])
def test_combinations_matches_itertools(r, with_replacement):
    values = np.array([1.0, 2.0, 3.0, 4.0])
    got = mt.combinations(_t(values), r, with_replacement).numpy()
    choose = (
        itertools.combinations_with_replacement
        if with_replacement
        else itertools.combinations
    )
    rows = list(choose(values, r))
    # `np.array([()])` cannot be reshaped to `(1, 0)`, so the degenerate widths
    # are built from their shape rather than from their (empty) contents.
    want = (
        np.array(rows, dtype=np.float64)
        if rows and r
        else np.zeros((len(rows), r), dtype=np.float64)
    )
    np.testing.assert_array_equal(got, want)
    assert got.shape == want.shape


def test_combinations_of_more_than_there_are_is_empty():
    got = mt.combinations(_t([1.0, 2.0]), 3)
    assert tuple(got.shape) == (0, 3)


def test_combinations_requires_a_vector():
    with pytest.raises(ValueError, match="1-D tensor"):
        mt.combinations(_t(GRID))


# --- gradient ---------------------------------------------------------------

SAMPLES = np.array([1.0, 2.0, 4.0, 7.0, 11.0, 16.0])
UNEVEN = np.array([0.0, 1.0, 1.5, 3.5, 4.0, 6.0])


@pytest.mark.parametrize("edge_order", [1, 2])
def test_gradient_matches_numpy_with_a_uniform_step(edge_order):
    for step in (1.0, 2.0, 0.25):
        np.testing.assert_allclose(
            mt.gradient(_t(SAMPLES), step, edge_order=edge_order).numpy(),
            np.gradient(SAMPLES, step, edge_order=edge_order),
            rtol=1e-13,
        )


@pytest.mark.parametrize("edge_order", [1, 2])
def test_gradient_matches_numpy_with_uneven_coordinates(edge_order):
    np.testing.assert_allclose(
        mt.gradient(_t(SAMPLES), _t(UNEVEN), edge_order=edge_order).numpy(),
        np.gradient(SAMPLES, UNEVEN, edge_order=edge_order),
        rtol=1e-13,
    )


def test_gradient_over_every_axis_returns_one_tensor_each():
    values = RNG.standard_normal((4, 5, 3))
    got = mt.gradient(_t(values))
    want = np.gradient(values)
    assert len(got) == 3
    for a, b in zip(got, want):
        np.testing.assert_allclose(a.numpy(), b, rtol=1e-13)


def test_gradient_over_one_axis_returns_one_tensor():
    values = RNG.standard_normal((4, 5))
    got = mt.gradient(_t(values), dim=1)
    assert isinstance(got, mt.Tensor)
    np.testing.assert_allclose(got.numpy(), np.gradient(values, axis=1), rtol=1e-13)


def test_gradient_takes_a_step_per_axis():
    values = RNG.standard_normal((4, 5))
    got = mt.gradient(_t(values), (2.0, 0.5))
    want = np.gradient(values, 2.0, 0.5)
    for a, b in zip(got, want):
        np.testing.assert_allclose(a.numpy(), b, rtol=1e-13)


def test_gradient_is_exact_for_a_straight_line():
    # A second-order method reproduces a linear function exactly, at every
    # point including the ends, which is the cheapest check that the edge
    # stencils are right.
    x = np.array([0.0, 0.5, 1.7, 4.0, 4.25])
    got = mt.gradient(_t(3.0 * x - 1.0), _t(x), edge_order=2).numpy()
    np.testing.assert_allclose(got, np.full(5, 3.0), rtol=1e-12)


def test_gradient_reports_what_it_cannot_do():
    with pytest.raises(ValueError, match="edge_order"):
        mt.gradient(_t(SAMPLES), edge_order=3)
    with pytest.raises(ValueError, match="at least 3 samples"):
        mt.gradient(_t([1.0, 2.0]), edge_order=2)
    with pytest.raises(ValueError, match="non-zero spacing"):
        mt.gradient(_t(SAMPLES), 0.0)
    with pytest.raises(ValueError, match="at least 2 samples"):
        mt.gradient(_t([1.0]))


# --- the draws --------------------------------------------------------------


def test_bernoulli_draws_at_the_probability_it_is_given():
    mt.manual_seed(7)
    probabilities = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    draws = np.stack([mt.bernoulli(_t(probabilities)).numpy() for _ in range(4000)])

    assert set(np.unique(draws)) <= {0.0, 1.0}
    np.testing.assert_allclose(draws.mean(axis=0), probabilities, atol=0.03)
    # The two certainties have to be certain, not merely likely.
    assert (draws[:, 0] == 0.0).all()
    assert (draws[:, 4] == 1.0).all()


def test_bernoulli_keeps_the_shape_and_dtype_of_its_probabilities():
    probabilities = _t(np.full((2, 3), 0.5), dtype="float32")
    drawn = mt.bernoulli(probabilities)
    assert tuple(drawn.shape) == (2, 3)
    assert "float32" in str(drawn.dtype)


def test_bernoulli_requires_probabilities_not_counts():
    integers = mt.Tensor(np.array([0, 1], dtype=np.int64), dtype="int64")
    with pytest.raises(ValueError, match="floating point"):
        mt.bernoulli(integers)


def test_normal_has_the_mean_and_spread_it_is_asked_for():
    mt.manual_seed(11)
    drawn = mt.normal(3.0, 2.0, (20000,)).numpy()
    assert drawn.mean() == pytest.approx(3.0, abs=0.05)
    assert drawn.std() == pytest.approx(2.0, abs=0.05)


def test_normal_takes_its_shape_from_a_tensor_mean_or_std():
    means = _t(np.zeros((3, 4)))
    assert tuple(mt.normal(means, 1.0).shape) == (3, 4)
    assert tuple(mt.normal(0.0, _t(np.ones((2, 5)))).shape) == (2, 5)


def test_normal_needs_a_shape_from_somewhere():
    with pytest.raises(ValueError, match="needs a size"):
        mt.normal(0.0, 1.0)


def test_normal_refuses_a_negative_spread():
    with pytest.raises(ValueError, match="non-negative standard deviation"):
        mt.normal(0.0, -1.0, (4,))


def test_a_zero_spread_gives_the_mean_exactly():
    np.testing.assert_array_equal(mt.normal(2.5, 0.0, (6,)).numpy(), np.full(6, 2.5))


def test_multinomial_with_replacement_draws_at_the_stated_proportions():
    mt.manual_seed(13)
    weights = np.array([1.0, 0.0, 3.0, 6.0])
    drawn = np.concatenate(
        [mt.multinomial(_t(weights), 8, True).numpy() for _ in range(600)]
    )
    proportions = np.bincount(drawn, minlength=4) / drawn.size
    np.testing.assert_allclose(proportions, weights / weights.sum(), atol=0.02)
    assert (drawn != 1).all(), "a zero weight is never drawn"


def test_multinomial_without_replacement_never_repeats():
    mt.manual_seed(17)
    weights = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    for _ in range(200):
        drawn = mt.multinomial(_t(weights), 4, False).numpy()
        assert len(set(drawn.tolist())) == 4


def test_multinomial_without_replacement_still_prefers_the_heavier_weights():
    mt.manual_seed(19)
    # One category is a thousand times likelier than the rest; over enough
    # draws of one it should win nearly always.
    weights = np.array([1.0, 1.0, 1000.0, 1.0])
    drawn = np.concatenate(
        [mt.multinomial(_t(weights), 1, False).numpy() for _ in range(400)]
    )
    assert (drawn == 2).mean() > 0.95


def test_multinomial_handles_a_batch_of_rows():
    mt.manual_seed(23)
    rows = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    drawn = mt.multinomial(_t(rows), 2, True).numpy()
    assert tuple(drawn.shape) == (2, 2)
    assert (drawn[0] == 0).all() and (drawn[1] == 2).all()


def test_multinomial_normalizes_counts_as_readily_as_probabilities():
    mt.manual_seed(29)
    counts = np.array([10.0, 30.0])
    drawn = np.concatenate(
        [mt.multinomial(_t(counts), 4, True).numpy() for _ in range(500)]
    )
    assert (drawn == 1).mean() == pytest.approx(0.75, abs=0.03)


def test_multinomial_reports_what_it_cannot_do():
    with pytest.raises(ValueError, match="without replacement"):
        mt.multinomial(_t([1.0, 2.0]), 3, False)
    with pytest.raises(ValueError, match="non-negative weights"):
        mt.multinomial(_t([1.0, -1.0]), 1, True)
    with pytest.raises(ValueError, match="sum above zero"):
        mt.multinomial(_t([0.0, 0.0]), 1, True)
    with pytest.raises(ValueError, match="1-D or 2-D"):
        mt.multinomial(_t(GRID), 1, True)
