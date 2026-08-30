# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""The normalizations that do not take statistics across the batch.

`batch_norm` and `layer_norm` were the two the library had, and they are the
ends of a range: one takes every sample together and one takes every channel
together. `group_norm` is the middle, and `instance_norm` is the far end -- one
channel of one sample at a time.

That relationship is not a comment here, it is the implementation.
`instance_norm` is `group_norm` with one group per channel, and the test that
says so compares them with `rtol=0`: they have to be the same numbers, not
close ones, because they are the same code. What `instance_norm` adds on top is
the running statistics, and those are checked against a hand-computed average
-- including the detail that the buffer takes the *unbiased* variance while the
normalization uses the biased one, which is what `batch_norm` does here too.

`local_response_norm` is the odd one: it normalizes along the channel axis
rather than over it, by averaging a window of neighbours. There is no kernel
for a window over channels, and none is needed -- it is `avg_pool3d` with the
channel axis moved into the depth slot and a window of one in the other two.
"""

from __future__ import annotations

import numpy as np
import pytest

import minitensor as mt
from minitensor import functional as F

RNG = np.random.default_rng(59)


def _t(values, requires_grad=False):
    return mt.Tensor(
        np.ascontiguousarray(np.asarray(values, dtype=np.float64)),
        dtype="float64",
        requires_grad=requires_grad,
    )


def _reference_group_norm(values, groups, weight=None, bias=None, eps=1e-5):
    batch, channels = values.shape[:2]
    grouped = values.reshape(batch, groups, -1)
    centred = (grouped - grouped.mean(-1, keepdims=True)) / np.sqrt(
        grouped.var(-1, keepdims=True) + eps
    )
    out = centred.reshape(values.shape)
    shape = (1, channels) + (1,) * (values.ndim - 2)
    if weight is not None:
        out = out * weight.reshape(shape)
    if bias is not None:
        out = out + bias.reshape(shape)
    return out


def _reference_lrn(values, size, alpha=1e-4, beta=0.75, k=1.0):
    batch, channels = values.shape[:2]
    squared = (values**2).reshape(batch, channels, -1)
    padded = np.pad(squared, ((0, 0), (size // 2, (size - 1) // 2), (0, 0)))
    windows = np.stack(
        [padded[:, start : start + size].mean(1) for start in range(channels)], axis=1
    )
    return values / (k + alpha * windows.reshape(values.shape)) ** beta


# --- group_norm -------------------------------------------------------------


@pytest.mark.parametrize(
    "shape,groups",
    [
        ((2, 6, 4, 4), 3),
        ((3, 4, 5), 2),
        ((2, 8, 3, 3, 3), 8),
        ((1, 4, 2, 2), 1),
        ((2, 6, 4, 4), 6),
    ],
)
def test_group_norm_matches_the_definition(shape, groups):
    values = RNG.normal(size=shape)
    weight, bias = RNG.normal(size=shape[1]), RNG.normal(size=shape[1])
    np.testing.assert_allclose(
        F.group_norm(_t(values), groups, _t(weight), _t(bias)).numpy(),
        _reference_group_norm(values, groups, weight, bias),
        rtol=1e-11,
    )


def test_one_group_normalizes_over_everything_but_the_batch():
    """Which is `layer_norm` over the whole of each sample."""

    values = RNG.normal(size=(3, 4, 5))
    normalized = F.group_norm(_t(values), 1).numpy()
    for sample in normalized:
        assert sample.mean() == pytest.approx(0.0, abs=1e-12)
        assert sample.var() == pytest.approx(1.0, rel=1e-4)


def test_the_statistics_never_cross_the_batch():
    """What makes it work at a batch size of one, unlike `batch_norm`."""

    values = RNG.normal(size=(4, 6, 3, 3))
    together = F.group_norm(_t(values), 3).numpy()
    for index in range(4):
        alone = F.group_norm(_t(values[index : index + 1]), 3).numpy()
        np.testing.assert_allclose(alone[0], together[index], rtol=1e-12)


def test_group_norm_carries_a_gradient_to_every_operand():
    values = _t(RNG.normal(size=(2, 4, 3, 3)), requires_grad=True)
    weight = _t(RNG.normal(size=4), requires_grad=True)
    bias = _t(RNG.normal(size=4), requires_grad=True)
    F.group_norm(values, 2, weight, bias).sum().backward()
    for operand in (values, weight, bias):
        assert np.isfinite(operand.grad.numpy()).all()
    # Every position takes the bias once, so its gradient counts them.
    np.testing.assert_allclose(bias.grad.numpy(), np.full(4, 2 * 9.0))
    mt.clear_autograd_graph()


def test_channels_that_do_not_divide_into_groups_are_refused():
    with pytest.raises(ValueError, match="divide by the number of groups"):
        F.group_norm(_t(RNG.normal(size=(2, 5, 3))), 2)


def test_a_group_count_below_one_is_refused():
    with pytest.raises(ValueError, match="at least one group"):
        F.group_norm(_t(RNG.normal(size=(2, 4, 3))), 0)


def test_group_norm_needs_a_batch_and_a_channel_axis():
    with pytest.raises(ValueError, match="batch and a channel axis"):
        F.group_norm(_t([1.0, 2.0]), 1)


def test_a_weight_of_the_wrong_length_is_refused():
    with pytest.raises(ValueError, match="one weight per channel"):
        F.group_norm(_t(RNG.normal(size=(2, 4, 3))), 2, _t(np.ones(3)))


# --- instance_norm ----------------------------------------------------------


def test_instance_norm_is_group_norm_with_one_group_per_channel():
    """Not close: the same code, so the same numbers."""

    values = _t(RNG.normal(size=(2, 4, 5, 5)))
    np.testing.assert_array_equal(
        F.instance_norm(values).numpy(), F.group_norm(values, 4).numpy()
    )


def test_instance_norm_normalizes_each_channel_of_each_sample_alone():
    values = RNG.normal(size=(3, 4, 6, 6))
    normalized = F.instance_norm(_t(values)).numpy()
    for sample in range(3):
        for channel in range(4):
            plane = normalized[sample, channel]
            assert plane.mean() == pytest.approx(0.0, abs=1e-12)
            assert plane.var() == pytest.approx(1.0, rel=1e-3)


def test_the_running_buffers_move_towards_this_batch():
    values = RNG.normal(size=(2, 4, 5, 5))
    running_mean = _t(np.zeros(4))
    running_var = _t(np.ones(4))
    F.instance_norm(_t(values), running_mean, running_var, None, None, True, 0.1)

    flat = values.reshape(2, 4, -1)
    np.testing.assert_allclose(
        running_mean.numpy(), 0.1 * flat.mean(-1).mean(0), rtol=1e-12
    )
    # The buffer takes the unbiased variance; the normalization takes the
    # biased one. `batch_norm` splits them the same way.
    np.testing.assert_allclose(
        running_var.numpy(), 0.9 + 0.1 * flat.var(-1, ddof=1).mean(0), rtol=1e-12
    )


def test_the_buffers_stay_out_of_the_graph():
    """A running average of past batches is not something a loss differentiates."""

    values = _t(RNG.normal(size=(2, 3, 4, 4)), requires_grad=True)
    running_mean, running_var = _t(np.zeros(3)), _t(np.ones(3))
    F.instance_norm(values, running_mean, running_var).sum().backward()
    assert not running_mean.requires_grad
    assert not running_var.requires_grad
    mt.clear_autograd_graph()


def test_without_input_statistics_the_buffers_are_what_is_used():
    values = RNG.normal(size=(2, 3, 4, 4))
    running_mean = _t([1.0, -2.0, 0.5])
    running_var = _t([4.0, 9.0, 1.0])
    np.testing.assert_allclose(
        F.instance_norm(
            _t(values), running_mean, running_var, None, None, False
        ).numpy(),
        (values - running_mean.numpy().reshape(1, 3, 1, 1))
        / np.sqrt(running_var.numpy().reshape(1, 3, 1, 1) + 1e-5),
        rtol=1e-12,
    )


def test_evaluation_without_buffers_is_refused():
    with pytest.raises(ValueError, match="needs both running_mean and running_var"):
        F.instance_norm(_t(RNG.normal(size=(2, 3, 4))), use_input_stats=False)


def test_instance_norm_needs_positions_to_normalize_over():
    with pytest.raises(ValueError, match="at least one position axis"):
        F.instance_norm(_t(RNG.normal(size=(2, 3))))


# --- local_response_norm ----------------------------------------------------


@pytest.mark.parametrize(
    "shape,size",
    [((2, 6, 4, 4), 3), ((1, 5, 7), 2), ((2, 4, 3, 3, 3), 5), ((1, 3, 2, 2), 1)],
)
def test_local_response_norm_matches_the_definition(shape, size):
    values = RNG.normal(size=shape)
    np.testing.assert_allclose(
        F.local_response_norm(_t(values), size).numpy(),
        _reference_lrn(values, size),
        rtol=1e-11,
    )


def test_an_even_window_reaches_one_further_below_than_above():
    """An even window has no centre, and which side gets the extra channel is
    the only place this could differ from `torch`. It takes it from below.

    A single spike in channel 1 is therefore seen by channels 1 and 2, and not
    by 0 or 3: with `alpha` and `beta` at one and `k` at one, the divisor is
    `1 + mean(squares in the window)`, so the answer says exactly which
    channels the window covered.
    """

    values = np.zeros((1, 4, 1))
    values[0, 1, 0] = 10.0
    # Squares are [0, 100, 0, 0]; window means are [0, 50, 50, 0].
    np.testing.assert_allclose(
        F.local_response_norm(_t(values), 2, alpha=1.0, beta=1.0, k=1.0).numpy()[
            0, :, 0
        ],
        [0.0, 10.0 / 51.0, 0.0, 0.0],
        rtol=1e-14,
    )


def test_a_window_of_one_divides_each_element_by_its_own_energy():
    values = RNG.normal(size=(1, 3, 4))
    np.testing.assert_allclose(
        F.local_response_norm(_t(values), 1, alpha=2.0, beta=1.0, k=1.0).numpy(),
        values / (1.0 + 2.0 * values**2),
        rtol=1e-13,
    )


def test_local_response_norm_carries_a_gradient():
    values = _t(RNG.normal(size=(2, 5, 3, 3)), requires_grad=True)
    F.local_response_norm(values, 3).sum().backward()
    assert np.isfinite(values.grad.numpy()).all()
    mt.clear_autograd_graph()


def test_a_window_below_one_is_refused():
    with pytest.raises(ValueError, match="window of at least one"):
        F.local_response_norm(_t(RNG.normal(size=(1, 3, 4))), 0)


def test_local_response_norm_needs_a_position_axis():
    with pytest.raises(ValueError, match="at least one position axis"):
        F.local_response_norm(_t(RNG.normal(size=(2, 3))), 3)
