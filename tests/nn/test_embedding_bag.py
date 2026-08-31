# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""`embedding_bag`: a lookup and a reduction, with nothing named in between.

A bag-of-words encoder and a recommender's sparse features both look rows up
only to reduce them at once. Doing that in two steps is correct and costs a
`(total, dim)` intermediate that is immediately thrown away; this is the same
two steps with the intermediate never named, which is the whole reason the
fused operation exists elsewhere.

Every case here is checked against the definition -- the bags sliced out and
reduced in NumPy -- rather than against another implementation. The parts worth
naming are the ones with no obvious answer: an empty bag, which reduces to zero
in all three modes because there is nothing else for it to reduce to; a bag of
negatives under `"max"`, which must reduce to its own maximum rather than being
clipped at the zeros it is scattered into; and the offsets, where a repeated
value means an empty bag and the final value may be a start or an end depending
on `include_last_offset`.
"""

from __future__ import annotations

import numpy as np
import pytest

import minitensor as mt
from minitensor import functional as F

RNG = np.random.default_rng(101)
TABLE = RNG.normal(size=(10, 4))


def _t(values, requires_grad=False):
    return mt.Tensor(
        np.ascontiguousarray(np.asarray(values, dtype=np.float64)),
        dtype="float64",
        requires_grad=requires_grad,
    )


def _i(values):
    return mt.Tensor.from_numpy(np.asarray(values, dtype=np.int64))


def _reference(indices, table, offsets, mode, weights=None):
    """The bags sliced out and reduced, one at a time."""

    flat = np.asarray(indices).reshape(-1)
    bounds = list(offsets) + [len(flat)]
    out = np.zeros((len(offsets), table.shape[1]))
    for bag in range(len(offsets)):
        low, high = bounds[bag], bounds[bag + 1]
        if high <= low:
            continue  # an empty bag stays at zero
        rows = table[flat[low:high]]
        if weights is not None:
            rows = rows * np.asarray(weights)[low:high, None]
        out[bag] = {
            "sum": rows.sum(0),
            "mean": rows.mean(0),
            "max": rows.max(0),
        }[mode]
    return out


MODES = ["sum", "mean", "max"]


# --- the two ways a bag arrives ---------------------------------------------


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("shape", [(5, 3), (1, 1), (4, 7)])
def test_a_two_dimensional_input_is_one_bag_per_row(mode, shape):
    indices = RNG.integers(0, 10, size=shape)
    np.testing.assert_allclose(
        F.embedding_bag(_i(indices), _t(TABLE), None, mode).numpy(),
        _reference(indices, TABLE, np.arange(shape[0]) * shape[1], mode),
        rtol=1e-13,
    )


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize(
    "offsets", [[0, 3, 6], [0, 2, 2, 5], [0], [0, 2, 2, 5, 9], [0, 9]]
)
def test_offsets_divide_a_flat_input_into_bags_that_may_differ(mode, offsets):
    indices = RNG.integers(0, 10, size=9)
    np.testing.assert_allclose(
        F.embedding_bag(_i(indices), _t(TABLE), _i(offsets), mode).numpy(),
        _reference(indices, TABLE, offsets, mode),
        rtol=1e-13,
    )


@pytest.mark.parametrize("mode", MODES)
def test_include_last_offset_reads_the_final_entry_as_an_end(mode):
    """Which is how a caller passes the cumulative counts they already have."""

    indices = RNG.integers(0, 10, size=9)
    starts = [0, 2, 5]
    np.testing.assert_allclose(
        F.embedding_bag(
            _i(indices), _t(TABLE), _i(starts + [9]), mode, include_last_offset=True
        ).numpy(),
        F.embedding_bag(_i(indices), _t(TABLE), _i(starts), mode).numpy(),
        rtol=0,
    )


# --- the cases with no obvious answer ---------------------------------------


@pytest.mark.parametrize("mode", MODES)
def test_an_empty_bag_reduces_to_zero(mode):
    indices = RNG.integers(0, 10, size=6)
    result = F.embedding_bag(_i(indices), _t(TABLE), _i([0, 3, 3, 6]), mode).numpy()
    np.testing.assert_array_equal(result[1], np.zeros(4))
    assert (result[[0, 2, 3]] != 0).any()


def test_a_bag_of_negatives_keeps_its_own_maximum():
    """It is scattered into zeros, and must not be clipped at them."""

    negative = -np.abs(RNG.normal(size=(6, 3))) - 1.0
    result = F.embedding_bag(_i([0, 1, 2, 3]), _t(negative), _i([0, 2]), "max").numpy()
    assert (result < 0).all()
    np.testing.assert_allclose(
        result, np.stack([negative[[0, 1]].max(0), negative[[2, 3]].max(0)]), rtol=1e-14
    )


def test_a_repeated_offset_is_an_empty_bag_and_not_a_repeated_one():
    indices = [1, 2, 3, 4]
    doubled = F.embedding_bag(_i(indices), _t(TABLE), _i([0, 2, 2, 4]), "sum").numpy()
    np.testing.assert_allclose(doubled[0], TABLE[[1, 2]].sum(0), rtol=1e-14)
    np.testing.assert_array_equal(doubled[1], np.zeros(4))
    np.testing.assert_allclose(doubled[2], TABLE[[3, 4]].sum(0), rtol=1e-14)


def test_per_sample_weights_scale_each_row_before_the_sum():
    indices = RNG.integers(0, 10, size=9)
    weights = RNG.uniform(size=9)
    np.testing.assert_allclose(
        F.embedding_bag(
            _i(indices), _t(TABLE), _i([0, 2, 5]), "sum", _t(weights)
        ).numpy(),
        _reference(indices, TABLE, [0, 2, 5], "sum", weights),
        rtol=1e-13,
    )


def test_padding_idx_is_passed_through_to_the_lookup():
    weight = _t(TABLE, requires_grad=True)
    F.embedding_bag(
        _i([0, 1, 0, 2]), weight, _i([0, 2]), "sum", padding_idx=0
    ).sum().backward()
    np.testing.assert_array_equal(weight.grad.numpy()[0], np.zeros(4))
    np.testing.assert_array_equal(weight.grad.numpy()[1], np.ones(4))
    mt.clear_autograd_graph()


# --- gradients --------------------------------------------------------------


@pytest.mark.parametrize("mode", MODES)
def test_the_gradient_reaches_the_rows_that_were_used(mode):
    indices = [1, 2, 3, 4, 5]
    weight = _t(TABLE, requires_grad=True)
    F.embedding_bag(_i(indices), weight, _i([0, 2]), mode).sum().backward()
    gradient = weight.grad.numpy()

    assert np.isfinite(gradient).all()
    np.testing.assert_array_equal(gradient[[0, 6, 7, 8, 9]], np.zeros((5, 4)))
    mt.clear_autograd_graph()


def test_the_sum_sends_one_to_every_row_and_the_mean_a_share():
    weight = _t(TABLE, requires_grad=True)
    F.embedding_bag(_i([1, 2, 3, 4]), weight, _i([0, 2]), "sum").sum().backward()
    np.testing.assert_allclose(weight.grad.numpy()[1:5], np.ones((4, 4)), rtol=0)
    mt.clear_autograd_graph()

    weight = _t(TABLE, requires_grad=True)
    F.embedding_bag(_i([1, 2, 3, 4]), weight, _i([0, 2]), "mean").sum().backward()
    np.testing.assert_allclose(
        weight.grad.numpy()[1:5], np.full((4, 4), 0.5), rtol=1e-14
    )
    mt.clear_autograd_graph()


def test_the_maximum_sends_its_gradient_only_to_the_row_that_won():
    table = np.array([[1.0, 0.0], [5.0, 0.0], [3.0, 0.0]])
    weight = _t(table, requires_grad=True)
    F.embedding_bag(_i([0, 1, 2]), weight, _i([0]), "max").sum().backward()
    # The first column's maximum is row 1; the second column is a tie the
    # reduction breaks one way, so only the first column is asserted.
    np.testing.assert_array_equal(weight.grad.numpy()[:, 0], [0.0, 1.0, 0.0])
    mt.clear_autograd_graph()


# --- what it refuses --------------------------------------------------------


def test_an_unknown_mode_is_refused():
    with pytest.raises(ValueError, match='"sum", "mean" or "max"'):
        F.embedding_bag(_i([[0, 1]]), _t(TABLE), None, "prod")


def test_offsets_with_a_two_dimensional_input_are_refused():
    with pytest.raises(ValueError, match="already says where each bag is"):
        F.embedding_bag(_i([[0, 1]]), _t(TABLE), _i([0]))


def test_a_flat_input_without_offsets_is_refused():
    with pytest.raises(ValueError, match="needs offsets"):
        F.embedding_bag(_i([0, 1]), _t(TABLE))


def test_per_sample_weights_outside_sum_are_refused():
    with pytest.raises(ValueError, match='only with mode="sum"'):
        F.embedding_bag(_i([0, 1]), _t(TABLE), _i([0]), "mean", _t([1.0, 1.0]))


def test_offsets_that_do_not_start_at_zero_are_refused():
    with pytest.raises(IndexError, match="first offset at zero"):
        F.embedding_bag(_i([0, 1, 2]), _t(TABLE), _i([1, 2]), "sum")


def test_an_input_of_the_wrong_rank_is_refused():
    with pytest.raises(ValueError, match="one- or two-dimensional input"):
        F.embedding_bag(_i(np.zeros((2, 2, 2))), _t(TABLE))


def test_offsets_of_the_wrong_rank_are_refused():
    with pytest.raises(ValueError, match="one-dimensional offsets"):
        F.embedding_bag(_i([0, 1]), _t(TABLE), _i([[0], [1]]))
