# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""Layers and functions refuse arguments they cannot compute with.

A convolution kernel with no taps ran: its span, `dilation * (k - 1) + 1`,
wrapped around, and `conv2d` handed back a map of zeros larger than its
input while `conv3d` summed no terms and failed on `None`. The conv layers
also took a stride or dilation of zero and failed only on their first call.
A NaN or infinite slope in `leaky_relu`, `elu`, `celu` or `softplus` made
every negative input NaN.
"""

from __future__ import annotations

import math

import pytest

import minitensor as mt

F = mt.functional
nn = mt.nn


@pytest.mark.parametrize(
    "call",
    [
        lambda: F.conv1d(mt.ones(1, 1, 5), mt.ones(1, 1, 0)),
        lambda: F.conv2d(mt.ones(1, 1, 5, 5), mt.ones(1, 1, 0, 3)),
        lambda: F.conv_transpose1d(mt.ones(1, 1, 5), mt.ones(1, 1, 0)),
        lambda: F.conv_transpose2d(mt.ones(1, 1, 5, 5), mt.ones(1, 1, 0, 0)),
        lambda: F.conv3d(mt.ones(1, 1, 3, 3, 3), mt.ones(1, 1, 0, 2, 2)),
    ],
    ids=["conv1d", "conv2d", "conv_transpose1d", "conv_transpose2d", "conv3d"],
)
def test_a_kernel_with_no_taps_is_refused(call):
    with pytest.raises(ValueError, match="kernel size must be greater than zero"):
        call()


@pytest.mark.parametrize(
    ("build", "name"),
    [
        (lambda: nn.Conv2d(1, 1, 0), "kernel_size"),
        (lambda: nn.Conv2d(1, 1, 3, stride=0), "stride"),
        (lambda: nn.Conv2d(1, 1, 3, dilation=(1, 0)), "dilation"),
        (lambda: nn.Conv1d(1, 1, 3, stride=0), "stride"),
        (lambda: nn.ConvTranspose2d(1, 1, 0), "kernel_size"),
        (lambda: nn.ConvTranspose1d(1, 1, 3, dilation=0), "dilation"),
    ],
    ids=["conv2d-k", "conv2d-s", "conv2d-d", "conv1d-s", "convT2d-k", "convT1d-d"],
)
def test_conv_layers_refuse_a_window_that_cannot_slide(build, name):
    with pytest.raises(ValueError, match=f"{name} must be greater than zero"):
        build()


@pytest.mark.parametrize("value", [math.nan, math.inf], ids=["nan", "inf"])
@pytest.mark.parametrize(
    "call",
    [
        lambda v: F.leaky_relu(mt.tensor([-1.0, 2.0]), v),
        lambda v: F.elu(mt.tensor([-1.0, 2.0]), v),
        lambda v: F.celu(mt.tensor([-1.0, 2.0]), v),
        lambda v: F.softplus(mt.tensor([-1.0, 2.0]), beta=v),
        lambda v: nn.LeakyReLU(v),
        lambda v: nn.ELU(v),
    ],
    ids=["leaky_relu", "elu", "celu", "softplus", "LeakyReLU", "ELU"],
)
def test_a_non_finite_activation_parameter_is_refused(call, value):
    with pytest.raises(ValueError, match="finite"):
        call(value)


@pytest.mark.parametrize(
    "call",
    [
        lambda: F.gumbel_softmax(mt.tensor([[1.0, 2.0]]), math.nan),
        lambda: F.gumbel_softmax(mt.tensor([[1.0, 2.0]]), math.inf),
        lambda: F.rrelu(mt.tensor([-1.0, 2.0]), math.nan, 0.3),
        lambda: F.rrelu(mt.tensor([-1.0, 2.0]), 0.1, math.nan),
        lambda: F.softplus(mt.tensor([-1.0, 2.0]), 1.0, math.nan),
        lambda: F.threshold(mt.tensor([-1.0, 2.0]), math.nan, 0.0),
    ],
    ids=[
        "gumbel tau nan",
        "gumbel tau inf",
        "rrelu lower",
        "rrelu upper",
        "softplus threshold",
        "threshold",
    ],
)
def test_a_nan_parameter_that_compared_false_is_refused(call):
    # Each was checked with `<=` or `>`, which a NaN fails in both directions,
    # so it passed: `gumbel_softmax` returned all zeros, `rrelu` NaN slopes,
    # and `threshold` replaced every element.
    with pytest.raises(ValueError):
        call()


def test_an_infinite_threshold_and_a_nan_fill_are_still_settings():
    x = mt.tensor([-1.0, 2.0])
    assert F.threshold(x, math.inf, -9.0).tolist() == [-9.0, -9.0]
    assert math.isnan(F.threshold(x, 0.0, math.nan).tolist()[0])
    assert F.softplus(x, 1.0, math.inf).tolist() == pytest.approx(
        [math.log1p(math.exp(-1.0)), math.log1p(math.exp(2.0))]
    )
