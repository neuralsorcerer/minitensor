# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""Degenerate-shape behaviour for the operations added on this branch.

Empty axes, single elements and windows wider than their input are where an
indexing kernel panics rather than returning an error, and a Rust panic crossing
the binding is far worse for a caller than an exception: it carries no useful
message and can poison the interpreter state. Each case below must therefore
either produce a sensible result or raise -- never abort.

This file exists to keep an audit from becoming a one-off. It found nothing when
written; its value is that a regression would now show up as a failure rather
than as a crash in someone's training run.
"""

import numpy as np
import pytest

import minitensor as mt
from minitensor import functional as F
from minitensor import nn


def f32(*shape):
    return mt.Tensor(np.zeros(shape, dtype=np.float32))


def f64(*shape):
    return mt.Tensor(np.zeros(shape), dtype="float64")


@pytest.mark.parametrize(
    "name, call",
    [
        ("max_pool1d", lambda: F.max_pool1d(f64(1, 1, 3), 5)),
        ("avg_pool1d", lambda: F.avg_pool1d(f64(1, 1, 3), 5)),
        ("max_pool2d", lambda: F.max_pool2d(f64(1, 1, 2, 2), 5)),
        ("avg_pool2d", lambda: F.avg_pool2d(f64(1, 1, 2, 2), 5)),
        ("conv1d", lambda: F.conv1d(f32(1, 1, 3), f32(1, 1, 5))),
        ("conv2d", lambda: F.conv2d(f32(1, 1, 2, 2), f32(1, 1, 5, 5))),
    ],
)
def test_a_window_wider_than_its_input_raises(name, call):
    # There is no valid output position, so this must be rejected rather than
    # producing a zero-length axis or reading out of bounds.
    with pytest.raises(Exception) as excinfo:
        call()
    assert "larger than" in str(excinfo.value) or "cannot be larger" in str(
        excinfo.value
    )


def test_conv1d_with_a_kernel_exactly_the_input_length_gives_one_position():
    result = F.conv1d(f32(1, 1, 5), f32(1, 1, 5))
    assert result.shape == (1, 1, 1)


def test_convolution_over_an_empty_batch_keeps_the_empty_axis():
    # No rows to convolve, but the channel and length arithmetic still applies.
    assert F.conv1d(f32(0, 1, 4), f32(1, 1, 2)).shape == (0, 1, 3)


def test_norm_over_an_empty_axis_is_zero():
    # An empty sum accumulates nothing, so every order gives zero.
    np.testing.assert_array_equal(f64(0, 3).norm(2.0, 0).numpy(), np.zeros(3))


def test_norm_of_a_single_element_is_its_magnitude():
    assert mt.Tensor([-3.0], dtype="float64").norm(2.0).item() == pytest.approx(3.0)


def test_scatter_with_an_empty_index_leaves_the_input_alone():
    base = mt.Tensor(np.arange(6, dtype=np.float64).reshape(2, 3), dtype="float64")
    empty_index = mt.Tensor(np.zeros((2, 0), dtype=np.int64), dtype="int64")
    result = base.scatter(1, empty_index, f64(2, 0))
    np.testing.assert_array_equal(result.numpy(), base.numpy())


@pytest.mark.parametrize("reduction, expected", [("sum", 0.0), ("none", None)])
def test_bce_with_logits_over_an_empty_input(reduction, expected):
    empty = f32(0, 3)
    result = F.binary_cross_entropy_with_logits(
        empty, empty, reduction=reduction
    ).numpy()
    if expected is None:
        assert result.shape == (0, 3)
    else:
        assert float(result) == pytest.approx(expected)


def test_bce_with_logits_mean_over_an_empty_input_is_nan_like_every_other_mean():
    # 0/0. This matches `mse_loss`, `binary_cross_entropy`, a plain `mean()` and
    # NumPy, so it is the consistent answer rather than a special case.
    empty = f32(0, 3)
    assert np.isnan(F.binary_cross_entropy_with_logits(empty, empty).numpy())
    assert np.isnan(F.mse_loss(empty, empty).numpy())
    assert np.isnan(empty.mean().numpy())


@pytest.mark.parametrize("kind", ["LSTM", "GRU"])
@pytest.mark.parametrize("bidirectional", [False, True])
def test_recurrent_layers_accept_a_single_timestep(kind, bidirectional):
    layer = getattr(nn, kind)(2, 3, bidirectional=bidirectional, dtype="float64")
    width = 3 * (2 if bidirectional else 1)
    assert layer(f64(1, 1, 2)).shape == (1, 1, width)


@pytest.mark.parametrize("kind", ["LSTM", "GRU"])
def test_recurrent_layers_accept_width_one_everywhere(kind):
    layer = getattr(nn, kind)(1, 1, dtype="float64")
    assert layer(f64(3, 1, 1)).shape == (3, 1, 1)


@pytest.mark.parametrize("kind", ["LSTM", "GRU"])
def test_recurrent_layers_reject_an_empty_sequence(kind):
    # Zero timesteps has no meaningful final state, so it is rejected rather
    # than silently returning the initial one.
    layer = getattr(nn, kind)(2, 3, dtype="float64")
    with pytest.raises(Exception):
        layer(f64(0, 1, 2))
