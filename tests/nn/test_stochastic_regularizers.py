# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""Dropout at the ranks and in the forms the library did not have.

`dropout` and `dropout2d` existed, and `dropout2d` takes a four-dimensional
input only -- so a signal and a volume had no channel-wise dropout at all.
Those two are the same operation with the positions flattened, and they are
written that way.

`alpha_dropout` is the one with something to prove. Ordinary dropout zeroes an
element and rescales the rest, which keeps the mean and moves the variance;
that suits a rectifier, whose negative side is zero anyway, and defeats `selu`,
whose entire premise is that activations arrive with a mean of zero and a
variance of one. So it drops to `selu`'s own saturation value and applies the
affine correction that restores both moments. The test for it is the property
rather than the formula: a standard normal in, a standard normal out, at every
`p`, measured over enough samples for the answer to mean something.

`rrelu` draws its negative slope instead of fixing it, and is spelled
`relu(x) + slope * (x - relu(x))` so that it is bit-exact on the positive side
and agrees with `leaky_relu` and `prelu` at the origin, where the two sides of
a rectifier disagree about the derivative.
"""

from __future__ import annotations

import numpy as np
import pytest

import minitensor as mt
from minitensor import functional as F
from minitensor._nn_extras import _SELU_SATURATION

RNG = np.random.default_rng(71)


def _t(values, requires_grad=False):
    return mt.Tensor(
        np.ascontiguousarray(np.asarray(values, dtype=np.float64)),
        dtype="float64",
        requires_grad=requires_grad,
    )


# --- dropout1d and dropout3d ------------------------------------------------


@pytest.mark.parametrize(
    "name,shape", [("dropout1d", (4, 6, 5)), ("dropout3d", (2, 6, 2, 3, 3))]
)
def test_a_channel_is_dropped_whole_or_not_at_all(name, shape):
    mt.manual_seed(0)
    dropped = getattr(F, name)(_t(np.ones(shape)), 0.5).numpy()
    for sample in range(shape[0]):
        for channel in range(shape[1]):
            plane = dropped[sample, channel]
            assert (plane == 0).all() or (plane != 0).all(), (sample, channel)


@pytest.mark.parametrize(
    "name,shape", [("dropout1d", (4, 6, 5)), ("dropout3d", (2, 6, 2, 3, 3))]
)
def test_the_kept_channels_are_rescaled_to_keep_the_mean(name, shape):
    mt.manual_seed(1)
    dropped = getattr(F, name)(_t(np.ones(shape)), 0.5).numpy()
    kept = dropped[dropped != 0]
    assert kept.size, "the seed dropped everything; pick another"
    np.testing.assert_allclose(kept, np.full(kept.shape, 2.0), rtol=1e-14)


@pytest.mark.parametrize(
    "name,shape", [("dropout1d", (4, 6, 5)), ("dropout3d", (2, 6, 2, 3, 3))]
)
def test_evaluation_leaves_everything_alone(name, shape):
    values = _t(RNG.normal(size=shape))
    np.testing.assert_array_equal(
        getattr(F, name)(values, 0.9, False).numpy(), values.numpy()
    )


@pytest.mark.parametrize("name,rank", [("dropout1d", 3), ("dropout3d", 5)])
def test_the_channelwise_dropouts_state_the_rank_they_take(name, rank):
    with pytest.raises(ValueError, match=f"{rank}-dimensional input"):
        getattr(F, name)(_t(np.ones((2, 3, 4, 5, 6, 7))), 0.5)


# --- alpha_dropout ----------------------------------------------------------


@pytest.mark.parametrize("p", [0.1, 0.3, 0.5, 0.8])
def test_a_standard_normal_stays_standard(p):
    """The property the whole construction exists for."""

    mt.manual_seed(7)
    values = _t(RNG.standard_normal(400_000))
    out = F.alpha_dropout(values, p).numpy()
    assert out.mean() == pytest.approx(0.0, abs=0.01)
    assert out.var() == pytest.approx(1.0, abs=0.02)


@pytest.mark.parametrize("p", [0.25, 0.5])
def test_the_dropped_elements_all_take_one_value(p):
    """`selu`'s saturation, carried through the same affine correction."""

    mt.manual_seed(3)
    out = F.alpha_dropout(_t(RNG.standard_normal(2000)), p).numpy()
    keep = 1.0 - p
    scale = (keep * (1.0 + p * _SELU_SATURATION**2)) ** -0.5
    expected = scale * _SELU_SATURATION - scale * _SELU_SATURATION * p

    dropped = out[np.isclose(out, expected, rtol=1e-12)]
    assert dropped.size > 0
    # Roughly `p` of them, and every one exactly that value.
    assert abs(dropped.size / out.size - p) < 0.05
    np.testing.assert_allclose(dropped, np.full(dropped.shape, expected), rtol=1e-13)


def test_dropping_nothing_and_evaluating_are_both_the_identity():
    values = _t(RNG.standard_normal(100))
    np.testing.assert_array_equal(F.alpha_dropout(values, 0.0).numpy(), values.numpy())
    np.testing.assert_array_equal(
        F.alpha_dropout(values, 0.5, False).numpy(), values.numpy()
    )


def test_dropping_everything_gives_the_mean_and_not_a_nan():
    """The formula's limit, which computing it would reach as `inf - inf`."""

    out = F.alpha_dropout(_t(RNG.standard_normal(50)), 1.0).numpy()
    np.testing.assert_array_equal(np.abs(out), np.zeros(50))


@pytest.mark.parametrize("p", [-0.1, 1.5])
def test_a_probability_outside_the_unit_interval_is_refused(p):
    with pytest.raises(ValueError, match="between 0 and 1"):
        F.alpha_dropout(_t([1.0]), p)


def test_alpha_dropout_carries_a_gradient_to_what_it_kept():
    mt.manual_seed(5)
    values = _t(RNG.standard_normal(200), requires_grad=True)
    F.alpha_dropout(values, 0.5).sum().backward()
    gradient = values.grad.numpy()
    assert np.isfinite(gradient).all()
    # Kept elements pass the rescaling; dropped ones are a constant.
    assert set(np.round(gradient, 12).tolist()) <= {0.0, round(gradient.max(), 12)}


# --- feature_alpha_dropout --------------------------------------------------


def test_a_dropped_channel_is_constant_all_the_way_through():
    """The only difference from `alpha_dropout`: one draw per channel."""

    mt.manual_seed(13)
    out = F.feature_alpha_dropout(_t(RNG.standard_normal((4, 8, 3, 3))), 0.5).numpy()
    keep = 0.5
    scale = (keep * (1.0 + 0.5 * _SELU_SATURATION**2)) ** -0.5
    expected = scale * _SELU_SATURATION - scale * _SELU_SATURATION * 0.5

    constant = [
        (sample, channel)
        for sample in range(4)
        for channel in range(8)
        if np.allclose(out[sample, channel], expected, rtol=1e-12)
    ]
    assert constant, "no channel was dropped; pick another seed"
    varied = [
        (sample, channel)
        for sample in range(4)
        for channel in range(8)
        if not np.allclose(out[sample, channel], out[sample, channel].ravel()[0])
    ]
    assert varied, "every channel was dropped; pick another seed"


def test_feature_alpha_dropout_needs_a_channel_axis():
    with pytest.raises(ValueError, match="batch and a channel axis"):
        F.feature_alpha_dropout(_t([1.0, 2.0]), 0.5)


# --- rrelu ------------------------------------------------------------------


def test_the_positive_side_passes_through_bit_for_bit():
    mt.manual_seed(17)
    values = _t([0.5, 1.0, 2.0, 1e-30])
    np.testing.assert_array_equal(F.rrelu(values).numpy(), values.numpy())


def test_each_element_gets_its_own_slope_while_training():
    mt.manual_seed(19)
    out = F.rrelu(_t(np.full(1000, -1.0))).numpy()
    slopes = -out
    assert (slopes >= 1 / 8 - 1e-12).all() and (slopes <= 1 / 3 + 1e-12).all()
    assert slopes.std() > 0.01, "every element took the same slope"


def test_evaluation_uses_the_middle_of_the_range():
    midpoint = (1 / 8 + 1 / 3) / 2
    np.testing.assert_allclose(
        F.rrelu(_t([-2.0, -1.0, 0.0, 1.0]), training=False).numpy(),
        [-2 * midpoint, -midpoint, 0.0, 1.0],
        rtol=1e-14,
    )


def test_a_range_with_nothing_in_it_is_a_leaky_rectifier():
    values = _t([-2.0, -0.5, 1.0])
    np.testing.assert_allclose(
        F.rrelu(values, 0.25, 0.25).numpy(),
        F.leaky_relu(values, 0.25).numpy(),
        rtol=1e-15,
    )


def test_the_derivative_at_the_origin_is_the_negative_side():
    """Which is what `prelu` and `leaky_relu` do, on the one disputed point."""

    mt.manual_seed(23)
    origin = _t([0.0], requires_grad=True)
    F.rrelu(origin, 0.25, 0.25).sum().backward()
    assert float(origin.grad.item()) == pytest.approx(0.25)
    mt.clear_autograd_graph()


def test_a_range_the_wrong_way_round_is_refused():
    with pytest.raises(ValueError, match="lower bound first"):
        F.rrelu(_t([1.0]), 0.5, 0.1)


def test_a_negative_slope_bound_is_refused():
    with pytest.raises(ValueError, match="non-negative slopes"):
        F.rrelu(_t([1.0]), -0.1, 0.5)
