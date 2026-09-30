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
