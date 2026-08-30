# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""Three spatial axes, out of the two the kernels have.

A 3-D window is a stack of 2-D windows, one per depth tap. So a 3-D convolution
is `kD` two-dimensional convolutions of the depth slices each tap reads, summed;
a 3-D maximum is the maximum of the 2-D maxima; and a 3-D mean is the mean of
the 2-D means. None of the three needs a kernel, and each leaves the arithmetic
with the `conv2d` and `max_pool2d` kernels that already exist rather than
laying the volume out as columns, which for a 3x3x3 kernel over eight channels
would cost twenty-seven times the volume in memory.

Every case here is checked against the definition, written out as explicit
loops over the windows in NumPy -- not against another library and not against
the same decomposition spelled twice. Stride, padding, dilation and groups are
covered because they are passed through to `conv2d` and passing the wrong one
through is the failure this arrangement invites.

Two details get their own tests. The depth padding is negative infinity for the
maximum and zero for the mean, so a padded position can neither win a maximum
nor be counted as a real zero. And `count_include_pad=False` divides by the
same pipeline run over a volume of ones, which is what makes the divisor
exactly the positions the numerator summed.
"""

from __future__ import annotations

import numpy as np
import pytest

import minitensor as mt
from minitensor import functional as F

RNG = np.random.default_rng(41)


def _t(values, requires_grad=False):
    return mt.Tensor(
        np.ascontiguousarray(np.asarray(values, dtype=np.float64)),
        dtype="float64",
        requires_grad=requires_grad,
    )


def _reference_conv3d(volume, weight, bias, stride, padding, dilation, groups):
    """The definition: one dot product per output position."""

    batch = volume.shape[0]
    out_channels, in_per_group = weight.shape[:2]
    kernel = weight.shape[2:]
    padded = np.pad(
        volume, ((0, 0), (0, 0)) + tuple((pad, pad) for pad in padding)
    )
    sizes = [
        (padded.shape[2 + axis] - dilation[axis] * (kernel[axis] - 1) - 1)
        // stride[axis]
        + 1
        for axis in range(3)
    ]
    out = np.zeros((batch, out_channels, *sizes))
    per_group = out_channels // groups
    for channel in range(out_channels):
        group = channel // per_group
        channels = slice(group * in_per_group, (group + 1) * in_per_group)
        for i, j, k in np.ndindex(*sizes):
            window = padded[
                :,
                channels,
                i * stride[0] : i * stride[0] + dilation[0] * (kernel[0] - 1) + 1 : dilation[0],
                j * stride[1] : j * stride[1] + dilation[1] * (kernel[1] - 1) + 1 : dilation[1],
                k * stride[2] : k * stride[2] + dilation[2] * (kernel[2] - 1) + 1 : dilation[2],
            ]
            out[:, channel, i, j, k] = (window * weight[channel]).sum(
                axis=(1, 2, 3, 4)
            )
    if bias is not None:
        out += bias.reshape(1, out_channels, 1, 1, 1)
    return out


def _reference_pool3d(volume, kernel, stride, padding, how, count_include_pad=True):
    fill = -np.inf if how == "max" else 0.0
    edges = ((0, 0), (0, 0)) + tuple((pad, pad) for pad in padding)
    padded = np.pad(volume, edges, constant_values=fill)
    present = np.pad(np.ones_like(volume), edges)
    sizes = [
        (padded.shape[2 + axis] - kernel[axis]) // stride[axis] + 1 for axis in range(3)
    ]
    out = np.zeros((*volume.shape[:2], *sizes))
    for i, j, k in np.ndindex(*sizes):
        window = (
            slice(None),
            slice(None),
            slice(i * stride[0], i * stride[0] + kernel[0]),
            slice(j * stride[1], j * stride[1] + kernel[1]),
            slice(k * stride[2], k * stride[2] + kernel[2]),
        )
        if how == "max":
            out[:, :, i, j, k] = padded[window].max(axis=(2, 3, 4))
        else:
            divisor = (
                np.prod(kernel)
                if count_include_pad
                else present[window].sum(axis=(2, 3, 4))
            )
            out[:, :, i, j, k] = padded[window].sum(axis=(2, 3, 4)) / divisor
    return out


# --- conv3d -----------------------------------------------------------------


@pytest.mark.parametrize(
    "shape,weight_shape,stride,padding,dilation,groups",
    [
        ((1, 2, 4, 5, 6), (3, 2, 2, 2, 2), (1, 1, 1), (0, 0, 0), (1, 1, 1), 1),
        ((2, 3, 6, 7, 5), (4, 3, 3, 3, 3), (2, 1, 2), (1, 1, 1), (1, 1, 1), 1),
        ((1, 2, 7, 7, 7), (2, 2, 2, 2, 2), (1, 1, 1), (0, 0, 0), (2, 2, 2), 1),
        ((2, 4, 5, 5, 5), (4, 2, 3, 3, 3), (1, 2, 1), (1, 0, 1), (1, 1, 1), 2),
        ((1, 4, 4, 4, 4), (4, 1, 2, 2, 2), (1, 1, 1), (0, 0, 0), (1, 1, 1), 4),
    ],
)
def test_conv3d_matches_the_definition(
    shape, weight_shape, stride, padding, dilation, groups
):
    volume = RNG.normal(size=shape)
    weight = RNG.normal(size=weight_shape)
    bias = RNG.normal(size=weight_shape[0])
    np.testing.assert_allclose(
        F.conv3d(_t(volume), _t(weight), _t(bias), stride, padding, dilation, groups).numpy(),
        _reference_conv3d(volume, weight, bias, stride, padding, dilation, groups),
        rtol=1e-11,
        atol=1e-12,
    )


def test_conv3d_without_a_bias_adds_nothing():
    volume, weight = RNG.normal(size=(1, 2, 4, 4, 4)), RNG.normal(size=(3, 2, 2, 2, 2))
    bias = np.zeros(3)
    np.testing.assert_allclose(
        F.conv3d(_t(volume), _t(weight)).numpy(),
        F.conv3d(_t(volume), _t(weight), _t(bias)).numpy(),
        rtol=0,
    )


def test_a_single_depth_tap_is_conv2d_on_every_slice():
    """`kD == 1` reduces to the 2-D kernel, which is a useful thing to know."""

    volume = RNG.normal(size=(2, 3, 4, 6, 6))
    weight = RNG.normal(size=(5, 3, 1, 3, 3))
    built = F.conv3d(_t(volume), _t(weight), None, 1, (0, 1, 1)).numpy()
    for depth in range(4):
        np.testing.assert_allclose(
            built[:, :, depth],
            F.conv2d(
                _t(volume[:, :, depth]), _t(weight[:, :, 0]), None, (1, 1), (1, 1)
            ).numpy(),
            rtol=1e-12,
        )


def test_a_single_integer_means_the_same_along_every_axis():
    volume, weight = RNG.normal(size=(1, 2, 5, 5, 5)), RNG.normal(size=(2, 2, 3, 3, 3))
    np.testing.assert_allclose(
        F.conv3d(_t(volume), _t(weight), None, 2, 1).numpy(),
        F.conv3d(_t(volume), _t(weight), None, (2, 2, 2), (1, 1, 1)).numpy(),
        rtol=0,
    )


def test_conv3d_carries_a_gradient_to_every_operand():
    volume = _t(RNG.normal(size=(2, 2, 5, 5, 5)), requires_grad=True)
    weight = _t(RNG.normal(size=(3, 2, 3, 3, 3)), requires_grad=True)
    bias = _t(RNG.normal(size=3), requires_grad=True)
    F.conv3d(volume, weight, bias, 1, 1).sum().backward()
    for operand in (volume, weight, bias):
        assert operand.grad is not None
        assert np.isfinite(operand.grad.numpy()).all()
    # Every output takes the bias once, so its gradient counts the outputs.
    np.testing.assert_allclose(bias.grad.numpy(), np.full(3, 2 * 5 * 5 * 5.0))
    mt.clear_autograd_graph()


# --- max_pool3d and avg_pool3d ----------------------------------------------


POOLINGS = [
    ((2, 3, 6, 6, 6), (2, 2, 2), None, (0, 0, 0)),
    ((1, 2, 5, 5, 5), (3, 3, 3), (2, 2, 2), (1, 1, 1)),
    ((2, 2, 7, 5, 6), (3, 2, 2), (1, 2, 3), (1, 1, 0)),
    ((1, 1, 4, 4, 4), (4, 4, 4), None, (0, 0, 0)),
]


@pytest.mark.parametrize("shape,kernel,stride,padding", POOLINGS)
def test_max_pool3d_matches_the_definition(shape, kernel, stride, padding):
    volume = RNG.normal(size=shape)
    np.testing.assert_array_equal(
        F.max_pool3d(_t(volume), kernel, stride, padding).numpy(),
        _reference_pool3d(
            volume, kernel, kernel if stride is None else stride, padding, "max"
        ),
    )


@pytest.mark.parametrize("shape,kernel,stride,padding", POOLINGS)
@pytest.mark.parametrize("count_include_pad", [True, False])
def test_avg_pool3d_matches_the_definition(
    shape, kernel, stride, padding, count_include_pad
):
    volume = RNG.normal(size=shape)
    np.testing.assert_allclose(
        F.avg_pool3d(
            _t(volume), kernel, stride, padding, count_include_pad
        ).numpy(),
        _reference_pool3d(
            volume,
            kernel,
            kernel if stride is None else stride,
            padding,
            "avg",
            count_include_pad,
        ),
        rtol=1e-12,
    )


def test_the_stride_of_a_pooling_defaults_to_its_window():
    volume = RNG.normal(size=(1, 2, 6, 6, 6))
    np.testing.assert_array_equal(
        F.max_pool3d(_t(volume), 2).numpy(),
        F.max_pool3d(_t(volume), 2, 2).numpy(),
    )


def test_depth_padding_never_wins_a_maximum():
    """Padded with zero it would, on a volume that is entirely negative."""

    volume = -np.arange(1.0, 28.0).reshape(1, 1, 3, 3, 3)
    pooled = F.max_pool3d(_t(volume), 3, 1, 1).numpy()
    assert (pooled < 0).all(), pooled


def test_depth_padding_is_a_real_zero_for_a_mean():
    """With the padding counted, a corner window of ones averages below one."""

    ones = np.ones((1, 1, 3, 3, 3))
    corner = F.avg_pool3d(_t(ones), 3, 1, 1).numpy()[0, 0, 0, 0, 0]
    assert corner == pytest.approx(8 / 27)
    # And with it not counted, only the real positions divide.
    assert F.avg_pool3d(_t(ones), 3, 1, 1, False).numpy()[0, 0, 0, 0, 0] == pytest.approx(1.0)


def test_the_two_divisors_agree_where_there_is_no_padding():
    volume = RNG.normal(size=(1, 2, 4, 4, 4))
    np.testing.assert_allclose(
        F.avg_pool3d(_t(volume), 2).numpy(),
        F.avg_pool3d(_t(volume), 2, None, 0, False).numpy(),
        rtol=1e-14,
    )


@pytest.mark.parametrize("name", ["max_pool3d", "avg_pool3d"])
def test_a_pooling_carries_a_gradient(name):
    volume = _t(RNG.normal(size=(1, 2, 4, 4, 4)), requires_grad=True)
    getattr(F, name)(volume, 2).sum().backward()
    # Sixteen outputs; a maximum sends one to its winner and a mean an eighth
    # to each of eight, so either way the total is the number of outputs.
    assert float(volume.grad.numpy().sum()) == pytest.approx(16.0)
    mt.clear_autograd_graph()


# --- what they refuse -------------------------------------------------------


@pytest.mark.parametrize("name", ["max_pool3d", "avg_pool3d"])
def test_a_pooling_needs_five_dimensions(name):
    with pytest.raises(ValueError, match="five-dimensional"):
        getattr(F, name)(_t(np.zeros((1, 2, 4, 4))), 2)


def test_conv3d_needs_a_five_dimensional_weight():
    with pytest.raises(ValueError, match="five-dimensional weight"):
        F.conv3d(_t(np.zeros((1, 2, 4, 4, 4))), _t(np.zeros((3, 2, 3, 3))))


def test_a_kernel_too_deep_for_the_volume_is_refused():
    with pytest.raises(ValueError, match="does not fit along the depth"):
        F.conv3d(_t(np.zeros((1, 1, 2, 5, 5))), _t(np.zeros((1, 1, 4, 2, 2))))


@pytest.mark.parametrize("name", ["max_pool3d", "avg_pool3d"])
def test_padding_beyond_half_the_window_is_refused(name):
    with pytest.raises(ValueError, match="at most half the window"):
        getattr(F, name)(_t(np.zeros((1, 1, 6, 6, 6))), 2, None, 2)


def test_a_geometry_argument_needs_one_value_per_spatial_axis():
    with pytest.raises(ValueError, match="one stride per spatial axis"):
        F.max_pool3d(_t(np.zeros((1, 1, 6, 6, 6))), 2, (2, 2))
