# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""The neural-network pieces that are arrangements, not kernels.

`nll_loss` is a gather and a weighted mean, `prelu` is two rectifiers,
`gumbel_softmax` is a softmax of perturbed logits, and the pixel shuffles are a
reshape and a permute. Each is checked against the definition written out in
NumPy, and -- where the arrangement is the whole claim -- against the operation
it is supposed to agree with: `nll_loss` of `log_softmax` has to be
`cross_entropy`, and `pixel_unshuffle` has to undo `pixel_shuffle`.
"""

from __future__ import annotations

import numpy as np
import pytest

import minitensor as mt
from minitensor import functional as F

RNG = np.random.default_rng(17)


def _t(values, dtype="float64", requires_grad=False):
    return mt.Tensor(
        np.ascontiguousarray(np.asarray(values, dtype=np.float64)),
        dtype=dtype,
        requires_grad=requires_grad,
    )


def _i(values):
    return mt.Tensor(
        np.ascontiguousarray(np.asarray(values, dtype=np.int64)), dtype="int64"
    )


def _log_softmax(values, axis=1):
    shifted = values - values.max(axis=axis, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=axis, keepdims=True))


# --- nll_loss ---------------------------------------------------------------

LOGITS = RNG.standard_normal((5, 4))
TARGETS = np.array([0, 3, 1, 2, 3])


def test_nll_loss_is_the_negative_log_probability_of_the_target():
    log_probs = _log_softmax(LOGITS)
    got = F.nll_loss(_t(log_probs), _i(TARGETS), reduction="none").numpy()
    np.testing.assert_allclose(got, -log_probs[np.arange(5), TARGETS], rtol=1e-14)


@pytest.mark.parametrize("reduction", ["none", "mean", "sum"])
def test_nll_loss_reduces_the_way_it_says(reduction):
    log_probs = _log_softmax(LOGITS)
    per_sample = -log_probs[np.arange(5), TARGETS]
    want = {"none": per_sample, "mean": per_sample.mean(), "sum": per_sample.sum()}[
        reduction
    ]
    got = F.nll_loss(_t(log_probs), _i(TARGETS), reduction=reduction).numpy()
    np.testing.assert_allclose(got, want, rtol=1e-14)


def test_log_softmax_then_nll_loss_is_cross_entropy():
    # The claim that makes them two functions rather than one written twice.
    logits = _t(LOGITS)
    np.testing.assert_allclose(
        F.nll_loss(F.log_softmax(logits, 1), _i(TARGETS)).item(),
        F.cross_entropy(logits, _i(TARGETS)).item(),
        rtol=1e-13,
    )


def test_a_class_weight_divides_by_the_total_weight_not_the_count():
    log_probs = _log_softmax(LOGITS)
    weight = np.array([0.5, 1.0, 2.0, 4.0])

    per_sample = -log_probs[np.arange(5), TARGETS] * weight[TARGETS]
    want = per_sample.sum() / weight[TARGETS].sum()
    got = F.nll_loss(_t(log_probs), _i(TARGETS), _t(weight)).item()
    assert got == pytest.approx(want, rel=1e-13)

    # Dividing by the count instead would be a different number, so the test
    # is actually distinguishing the two.
    assert got != pytest.approx(per_sample.mean(), rel=1e-6)


def test_an_ignored_index_leaves_both_the_sum_and_the_divisor():
    log_probs = _log_softmax(LOGITS)
    targets = TARGETS.copy()
    targets[1] = -100
    kept = targets != -100

    want = (-log_probs[np.arange(5)[kept], targets[kept]]).mean()
    got = F.nll_loss(_t(log_probs), _i(targets)).item()
    assert got == pytest.approx(want, rel=1e-13)


def test_every_position_ignored_gives_a_nan_rather_than_a_zero():
    # Nothing was measured, so there is no average; `0 / 0` says so, where a
    # zero would read as a perfect fit.
    log_probs = _log_softmax(LOGITS)
    got = F.nll_loss(_t(log_probs), _i(np.full(5, -100))).item()
    assert np.isnan(got)


def test_nll_loss_handles_a_trailing_spatial_axis():
    logits = RNG.standard_normal((2, 3, 4))
    targets = RNG.integers(0, 3, size=(2, 4))
    log_probs = _log_softmax(logits, axis=1)

    got = F.nll_loss(_t(log_probs), _i(targets), reduction="none").numpy()
    want = -np.take_along_axis(log_probs, targets[:, None, :], 1).squeeze(1)
    np.testing.assert_allclose(got, want, rtol=1e-14)


def test_nll_loss_reports_a_mismatched_rank_and_a_wrong_weight_length():
    log_probs = _t(_log_softmax(LOGITS))
    with pytest.raises(ValueError, match="target of rank"):
        F.nll_loss(log_probs, _i(np.zeros((5, 2))))
    with pytest.raises(ValueError, match="one weight per class"):
        F.nll_loss(log_probs, _i(TARGETS), _t([1.0, 2.0]))
    with pytest.raises(ValueError, match="reduction must be"):
        F.nll_loss(log_probs, _i(TARGETS), reduction="average")


def test_nll_loss_carries_a_gradient_to_the_positions_it_read():
    log_probs = _t(_log_softmax(LOGITS), requires_grad=True)
    F.nll_loss(log_probs, _i(TARGETS), reduction="sum").backward()

    want = np.zeros_like(LOGITS)
    want[np.arange(5), TARGETS] = -1.0
    np.testing.assert_allclose(log_probs.grad.numpy(), want, rtol=1e-14)


# --- prelu ------------------------------------------------------------------

MIXED = np.array([[-2.0, -0.5, 0.0, 1.5], [3.0, -1.0, 0.25, -4.0]])


def test_prelu_is_the_leaky_rectifier_with_a_shared_slope():
    got = F.prelu(_t(MIXED), _t([0.25])).numpy()
    np.testing.assert_allclose(got, np.where(MIXED > 0, MIXED, 0.25 * MIXED))


def test_prelu_lines_a_per_channel_slope_up_with_dimension_one():
    slopes = np.array([0.1, 0.2, 0.3, 0.4])
    got = F.prelu(_t(MIXED), _t(slopes)).numpy()
    np.testing.assert_allclose(got, np.where(MIXED > 0, MIXED, slopes * MIXED))


def test_prelu_reaches_a_four_dimensional_input_on_its_channel_axis():
    values = RNG.standard_normal((2, 3, 4, 5))
    slopes = np.array([0.1, 0.5, 0.9])
    got = F.prelu(_t(values), _t(slopes)).numpy()
    want = np.where(values > 0, values, slopes[None, :, None, None] * values)
    np.testing.assert_allclose(got, want)


def test_the_prelu_gradient_reaches_the_slope():
    # The entire point of the op: a fixed leak needs no gradient here.
    values = _t(MIXED, requires_grad=True)
    slope = _t([0.25], requires_grad=True)
    F.prelu(values, slope).sum().backward()

    np.testing.assert_allclose(values.grad.numpy(), np.where(MIXED > 0, 1.0, 0.25))
    assert slope.grad.numpy()[0] == pytest.approx(MIXED[MIXED < 0].sum())


def test_prelu_agrees_with_leaky_relu_at_a_fixed_slope():
    np.testing.assert_array_equal(
        F.prelu(_t(MIXED), _t([0.01])).numpy(), F.leaky_relu(_t(MIXED), 0.01).numpy()
    )


def test_prelu_and_leaky_relu_take_the_same_side_at_exactly_zero():
    # A rectifier has a choice only at the origin, and the two spellings of one
    # have to make the same one.
    zero = _t([0.0], requires_grad=True)
    F.prelu(zero, _t([0.3])).sum().backward()
    learned = zero.grad.numpy()[0]

    fixed_zero = _t([0.0], requires_grad=True)
    F.leaky_relu(fixed_zero, 0.3).sum().backward()
    assert learned == fixed_zero.grad.numpy()[0] == 0.3


def test_prelu_leaves_the_positive_branch_bit_exact():
    # `relu(x) + w (x - relu(x))` is written that way rather than as
    # `w x + (1 - w) relu(x)` precisely so this holds: the second form pays a
    # rounding error on every positive element for a `w` that is not a dyadic
    # rational.
    values = np.array([1e-300, 0.1, 1.0, 3.0, 1e300])
    np.testing.assert_array_equal(F.prelu(_t(values), _t([0.1])).numpy(), values)


def test_prelu_reports_a_slope_that_does_not_fit():
    with pytest.raises(ValueError, match="one weight per channel"):
        F.prelu(_t(MIXED), _t([0.1, 0.2]))
    with pytest.raises(ValueError, match="scalar or 1-D weight"):
        F.prelu(_t(MIXED), _t(np.zeros((2, 2))))


# --- gumbel_softmax ---------------------------------------------------------


def test_gumbel_softmax_returns_a_distribution():
    logits = _t(RNG.standard_normal((6, 5)))
    got = F.gumbel_softmax(logits).numpy()
    np.testing.assert_allclose(got.sum(axis=1), 1.0, rtol=1e-12)
    assert (got >= 0.0).all()


def test_a_low_temperature_concentrates_and_a_high_one_spreads():
    logits = _t(np.tile([1.0, 2.0, 3.0], (200, 1)))
    cold = F.gumbel_softmax(logits, 0.1).numpy().max(axis=1).mean()
    warm = F.gumbel_softmax(logits, 10.0).numpy().max(axis=1).mean()
    assert cold > 0.9 > warm, f"{cold} then {warm}"


def test_the_hard_draw_is_one_hot_and_still_differentiable():
    logits = _t(RNG.standard_normal((8, 4)), requires_grad=True)
    drawn = F.gumbel_softmax(logits, 1.0, hard=True)
    values = drawn.numpy()

    np.testing.assert_array_equal(np.sort(np.unique(values)), [0.0, 1.0])
    np.testing.assert_allclose(values.sum(axis=1), 1.0)

    # Straight through: a hard draw has a zero gradient of its own, and this
    # one has the soft draw's, which is not zero for a non-constant readout.
    (drawn * mt.Tensor(np.arange(4.0), dtype="float64")).sum().backward()
    assert np.abs(logits.grad.numpy()).sum() > 0.0


def test_the_draw_prefers_the_larger_logit_over_many_samples():
    logits = _t(np.tile([0.0, 3.0], (2000, 1)))
    drawn = F.gumbel_softmax(logits, 0.5, hard=True).numpy()
    assert drawn[:, 1].mean() > 0.8, "the 3 should win far more often than the 0"


def test_gumbel_softmax_samples_along_the_dim_it_is_given():
    logits = _t(RNG.standard_normal((4, 6)))
    got = F.gumbel_softmax(logits, dim=0).numpy()
    np.testing.assert_allclose(got.sum(axis=0), 1.0, rtol=1e-12)


def test_gumbel_softmax_refuses_a_non_positive_temperature():
    with pytest.raises(ValueError, match="positive tau"):
        F.gumbel_softmax(_t([[1.0, 2.0]]), 0.0)


# --- pixel_shuffle ----------------------------------------------------------


def _pixel_shuffle_reference(values, factor):
    n, c, h, w = values.shape
    out_c = c // (factor * factor)
    out = np.zeros((n, out_c, h * factor, w * factor))
    for b in range(n):
        for channel in range(out_c):
            for i in range(factor):
                for j in range(factor):
                    source = channel * factor * factor + i * factor + j
                    out[b, channel, i::factor, j::factor] = values[b, source]
    return out


@pytest.mark.parametrize("factor", [1, 2, 3])
def test_pixel_shuffle_matches_the_definition(factor):
    values = RNG.standard_normal((2, 2 * factor * factor, 3, 4))
    got = F.pixel_shuffle(_t(values), factor).numpy()
    np.testing.assert_allclose(got, _pixel_shuffle_reference(values, factor))


@pytest.mark.parametrize("factor", [1, 2, 4])
def test_the_two_shuffles_undo_each_other(factor):
    values = RNG.standard_normal((2, 3 * factor * factor, 5, 6))
    shuffled = F.pixel_shuffle(_t(values), factor)
    np.testing.assert_allclose(F.pixel_unshuffle(shuffled, factor).numpy(), values)


def test_pixel_shuffle_moves_every_element_and_invents_none():
    values = RNG.standard_normal((1, 8, 2, 3))
    got = F.pixel_shuffle(_t(values), 2).numpy()
    assert tuple(got.shape) == (1, 2, 4, 6)
    np.testing.assert_allclose(np.sort(got.reshape(-1)), np.sort(values.reshape(-1)))


def test_pixel_shuffle_works_on_a_batchless_input():
    values = RNG.standard_normal((4, 2, 2))
    got = F.pixel_shuffle(_t(values), 2).numpy()
    assert tuple(got.shape) == (1, 4, 4)


def test_the_shuffles_report_a_size_that_does_not_divide():
    with pytest.raises(ValueError, match="divide by"):
        F.pixel_shuffle(_t(RNG.standard_normal((1, 5, 2, 2))), 2)
    with pytest.raises(ValueError, match="divide by"):
        F.pixel_unshuffle(_t(RNG.standard_normal((1, 1, 3, 4))), 2)
    with pytest.raises(ValueError, match="positive factor"):
        F.pixel_shuffle(_t(RNG.standard_normal((1, 4, 2, 2))), 0)


def test_pixel_shuffle_carries_a_gradient_through_the_rearrangement():
    values = _t(RNG.standard_normal((1, 4, 2, 2)), requires_grad=True)
    F.pixel_shuffle(values, 2).sum().backward()
    # Every element lands somewhere exactly once.
    np.testing.assert_allclose(values.grad.numpy(), np.ones((1, 4, 2, 2)))


# --- shared -----------------------------------------------------------------


@pytest.mark.parametrize(
    "name", ["nll_loss", "prelu", "gumbel_softmax", "pixel_shuffle", "pixel_unshuffle"]
)
def test_the_nn_and_functional_names_are_the_same_object(name):
    assert getattr(mt.nn, name) is getattr(F, name)
