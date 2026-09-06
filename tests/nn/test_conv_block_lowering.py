# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""Convolution lowers its input a block of output rows at a time.

im2col turns a convolution into one matrix multiply, and the lowered matrix is
`C_in * kH * kW` by `N * H_out * W_out` -- for a 32x3x224x224 stem with a 7x7
kernel that is 944MB of scratch, written once and read straight back out of
main memory by the GEMM. Lowering a block of output rows, multiplying it and
scattering it while it is still in cache bounds that scratch at a few megabytes
whatever the image is, and lands the multiply near the column count it runs
fastest at.

What the block can get wrong is its seams. A block is a run of output rows in
the flattened `(image, row)` space, so it can begin part-way down one image and
end part-way down the next; the buffer is reused between blocks, so anything a
previous block left in the padding has to be cleared rather than assumed zero;
and the last block is short. These shapes put a seam in each of those places --
they are large enough to need several blocks, which is what makes them slower
than the rest of the file.
"""

from __future__ import annotations

import numpy as np
import pytest

import minitensor as mt
from minitensor import nn


def _reference(image, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    """Direct im2col in float64, built with stride tricks rather than loops."""
    batch, channels, height, width = image.shape
    out_channels, group_in, kernel_h, kernel_w = weight.shape
    padded = np.pad(image, ((0, 0), (0, 0), (padding, padding), (padding, padding)))
    out_h = (height + 2 * padding - dilation * (kernel_h - 1) - 1) // stride + 1
    out_w = (width + 2 * padding - dilation * (kernel_w - 1) - 1) // stride + 1

    steps = padded.strides
    patches = np.lib.stride_tricks.as_strided(
        padded,
        (batch, channels, out_h, out_w, kernel_h, kernel_w),
        (
            steps[0],
            steps[1],
            steps[2] * stride,
            steps[3] * stride,
            steps[2] * dilation,
            steps[3] * dilation,
        ),
    )
    per_group = channels // groups
    out_per_group = out_channels // groups
    out = np.empty((batch, out_channels, out_h, out_w), np.float64)
    for g in range(groups):
        lowered = (
            patches[:, g * per_group : (g + 1) * per_group]
            .transpose(0, 2, 3, 1, 4, 5)
            .reshape(batch, out_h * out_w, per_group * kernel_h * kernel_w)
        )
        kernels = weight[g * out_per_group : (g + 1) * out_per_group].reshape(
            out_per_group, -1
        )
        out[:, g * out_per_group : (g + 1) * out_per_group] = (
            (lowered @ kernels.T)
            .transpose(0, 2, 1)
            .reshape(batch, out_per_group, out_h, out_w)
        )
    if bias is not None:
        out += bias.reshape(1, -1, 1, 1)
    return out


# `(batch, channels, size)`, kernel, out channels, stride, padding, groups.
# The lowered rows are `C_in * kH * kW` wide by `W_out` long, so 64 channels of
# a 3x3 over a 64-wide image is 144KB a row -- a few dozen rows to the block
# against 64 rows to the image, which is what puts a seam mid-image.
SEAMS = [
    ((3, 64, 64), 3, 8, 1, 1, 1),
    ((2, 64, 65), 3, 8, 1, 1, 1),
    ((3, 64, 64), 3, 8, 2, 1, 1),
    ((2, 64, 64), 3, 16, 1, 1, 4),
    ((2, 96, 48), 5, 8, 1, 2, 1),
    ((5, 64, 40), 3, 8, 1, 0, 1),
]


@pytest.mark.parametrize("shape,kernel,out_channels,stride,padding,groups", SEAMS)
def test_a_lowering_that_spans_several_blocks(
    shape, kernel, out_channels, stride, padding, groups
):
    batch, channels, size = shape
    rng = np.random.default_rng(89)
    image = rng.standard_normal((batch, channels, size, size)).astype(np.float32)
    weight = rng.standard_normal(
        (out_channels, channels // groups, kernel, kernel)
    ).astype(np.float32)
    bias = rng.standard_normal(out_channels).astype(np.float32)

    got = nn.conv2d(
        mt.from_numpy(image),
        mt.from_numpy(weight),
        mt.from_numpy(bias),
        stride=stride,
        padding=padding,
        groups=groups,
    ).numpy()
    expected = _reference(
        image.astype(np.float64),
        weight.astype(np.float64),
        bias.astype(np.float64),
        stride=stride,
        padding=padding,
        groups=groups,
    )

    assert got.shape == expected.shape
    relative = np.abs(got - expected).max() / np.abs(expected).max()
    assert relative < 1e-5, f"{relative:.3e}"


def test_the_padding_of_one_block_does_not_leak_into_the_next():
    """The block buffer is reused, and only the in-bounds part of each row is
    written -- so a row whose padding a previous block filled must be cleared.

    A tall image with padding on every side puts padded columns in every block
    but the values behind them differ block to block, which is what a missing
    clear would show up as.
    """
    rng = np.random.default_rng(97)
    image = rng.standard_normal((4, 64, 64, 64)).astype(np.float32)
    weight = rng.standard_normal((8, 64, 3, 3)).astype(np.float32)

    got = nn.conv2d(
        mt.from_numpy(image), mt.from_numpy(weight), None, padding=1
    ).numpy()
    expected = _reference(
        image.astype(np.float64), weight.astype(np.float64), padding=1
    )

    # The border is where padding lands, so check it on its own as well.
    relative = np.abs(got - expected).max() / np.abs(expected).max()
    assert relative < 1e-5, f"{relative:.3e}"
    np.testing.assert_allclose(
        got[:, :, 0, :], expected[:, :, 0, :], rtol=2e-4, atol=2e-5
    )
    np.testing.assert_allclose(
        got[:, :, -1, :], expected[:, :, -1, :], rtol=2e-4, atol=2e-5
    )
    np.testing.assert_allclose(
        got[:, :, :, 0], expected[:, :, :, 0], rtol=2e-4, atol=2e-5
    )
    np.testing.assert_allclose(
        got[:, :, :, -1], expected[:, :, :, -1], rtol=2e-4, atol=2e-5
    )


def _numerical_gradient(loss, values, eps=1e-4):
    gradient = np.zeros_like(values)
    walk = np.nditer(values, flags=["multi_index"])
    while not walk.finished:
        at = walk.multi_index
        held = values[at]
        values[at] = held + eps
        high = loss(values)
        values[at] = held - eps
        low = loss(values)
        values[at] = held
        gradient[at] = (high - low) / (2 * eps)
        walk.iternext()
    return gradient


# `(batch, channels, size)`, kernel, out channels, stride, padding, groups.
BACKWARD = [
    ((2, 4, 7), 3, 5, 1, 1, 1),
    ((1, 3, 6), 3, 4, 2, 1, 1),
    ((3, 6, 5), 3, 6, 1, 1, 3),
    ((2, 2, 9), 5, 3, 2, 2, 1),
]


@pytest.mark.parametrize("shape,kernel,out_channels,stride,padding,groups", BACKWARD)
def test_both_gradients_match_a_numerical_one(
    shape, kernel, out_channels, stride, padding, groups
):
    """The backward pass takes one image at a time now.

    It used to rearrange the whole batch's signal channel-major and lower every
    image at once -- 944MB for a stem -- because a single image's slice of
    `[C, N*OH*OW]` is not contiguous. Per image it already is, so the
    rearrangement went away with the buffer. What that could get wrong is which
    image's signal each gradient contracts against, and the sum over images;
    both show up here as a gradient that no longer matches a difference
    quotient. Float64 throughout, so the quotient is worth comparing against.
    """
    batch, channels, size = shape
    rng = np.random.default_rng(103)
    image = rng.standard_normal((batch, channels, size, size))
    weight = rng.standard_normal((out_channels, channels // groups, kernel, kernel))
    bias = rng.standard_normal(out_channels)

    def run(image_values, weight_values, bias_values):
        return nn.conv2d(
            mt.from_numpy(image_values),
            mt.from_numpy(weight_values),
            mt.from_numpy(bias_values),
            stride=stride,
            padding=padding,
            groups=groups,
        )

    shaped = run(image, weight, bias)
    # A weighted sum, so the seed is not all ones and a mis-taken image shows.
    seed = rng.standard_normal(tuple(shaped.shape))

    def loss(image_values, weight_values, bias_values):
        out = run(image_values, weight_values, bias_values)
        return float((out * mt.from_numpy(seed)).sum().numpy())

    inputs = mt.from_numpy(image.copy())
    inputs.requires_grad_(True)
    kernels = mt.from_numpy(weight.copy())
    kernels.requires_grad_(True)
    shifts = mt.from_numpy(bias.copy())
    shifts.requires_grad_(True)
    (
        nn.conv2d(
            inputs, kernels, shifts, stride=stride, padding=padding, groups=groups
        )
        * mt.from_numpy(seed)
    ).sum().backward()

    for got, expected in [
        (
            mt.get_gradient(inputs).numpy(),
            _numerical_gradient(lambda v: loss(v, weight, bias), image.copy()),
        ),
        (
            mt.get_gradient(kernels).numpy(),
            _numerical_gradient(lambda v: loss(image, v, bias), weight.copy()),
        ),
        (
            mt.get_gradient(shifts).numpy(),
            _numerical_gradient(lambda v: loss(image, weight, v), bias.copy()),
        ),
    ]:
        relative = np.abs(got - expected).max() / np.abs(expected).max()
        assert relative < 1e-6, f"{relative:.3e}"
    mt.clear_autograd_graph()


def test_a_single_image_backward_still_uses_the_pool():
    """The scatter used to be parallel over the batch, so a batch of one ran on
    one core. It is parallel over channel planes now, which are disjoint within
    an image -- this checks the answer, not the timing."""
    rng = np.random.default_rng(107)
    image = rng.standard_normal((1, 8, 12, 12))
    weight = rng.standard_normal((4, 8, 3, 3))

    inputs = mt.from_numpy(image.copy())
    inputs.requires_grad_(True)
    out = nn.conv2d(inputs, mt.from_numpy(weight), None, padding=1)
    seed = rng.standard_normal(tuple(out.shape))
    (out * mt.from_numpy(seed)).sum().backward()

    def loss(values):
        return float(
            (
                nn.conv2d(mt.from_numpy(values), mt.from_numpy(weight), None, padding=1)
                * mt.from_numpy(seed)
            )
            .sum()
            .numpy()
        )

    expected = _numerical_gradient(loss, image.copy())
    relative = (
        np.abs(mt.get_gradient(inputs).numpy() - expected).max()
        / np.abs(expected).max()
    )
    assert relative < 1e-6, f"{relative:.3e}"
    mt.clear_autograd_graph()


def test_the_gradient_still_matches_a_numerical_one():
    """The forward's shape changed; the backward reads the same geometry."""
    rng = np.random.default_rng(101)
    image = rng.standard_normal((2, 64, 64, 64)).astype(np.float32)
    weight = rng.standard_normal((4, 64, 3, 3)).astype(np.float32)

    t = mt.from_numpy(image.copy())
    t.requires_grad_(True)
    w = mt.from_numpy(weight.copy())
    w.requires_grad_(True)
    out = nn.conv2d(t, w, None, padding=1)
    seed = rng.standard_normal(tuple(out.shape)).astype(np.float32)
    (out * mt.from_numpy(seed)).sum().backward()

    # The weight gradient is the correlation of the seed with the image, which
    # the reference computes as a convolution with the axes swapped.
    expected = np.zeros_like(weight, dtype=np.float64)
    padded = np.pad(image.astype(np.float64), ((0, 0), (0, 0), (1, 1), (1, 1)))
    for ky in range(3):
        for kx in range(3):
            patch = padded[:, :, ky : ky + 64, kx : kx + 64]
            expected[:, :, ky, kx] = np.einsum(
                "nohw,nchw->oc", seed.astype(np.float64), patch
            )

    np.testing.assert_allclose(
        mt.get_gradient(w).numpy(), expected, rtol=2e-3, atol=2e-3
    )
    mt.clear_autograd_graph()
