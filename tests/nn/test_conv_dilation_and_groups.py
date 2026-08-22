# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""Convolution could not be grouped or dilated, so a whole class of model was
inexpressible.

`conv1d`/`conv2d` took stride and padding and nothing else. Without `groups`
there is no depthwise convolution, and without depthwise convolution there is
no depthwise-separable one -- which is the block MobileNet, EfficientNet and
ConvNeXt are built out of, and which no amount of reshaping recovers from a
dense convolution. Without `dilation` there is no atrous convolution, so no
dilated temporal network and no segmentation head that widens its receptive
field without downsampling.

Neither is a variation on the existing kernel that a caller could work around.
A grouped convolution over `g` groups is `g` independent convolutions sharing
nothing; a dilated one reads a different set of input positions entirely.

The lowering already had the right shape for both. `k` runs channel-major, so
a group owns a contiguous row-block of the im2col matrix and of the weight,
which makes the groups `g` GEMMs by pointer offset rather than `g` repacked
copies. And `in_bounds_range` was already written in terms of a tap's offset
into the input, so passing `ky * dilation` instead of `ky` is the whole of it.

These tests check the forward against an explicit loop -- not against a
rearranged call of the same kernel, which would not catch a shared mistake --
and the three gradients against numerical differentiation.
"""

from __future__ import annotations

import numpy as np
import pytest

import minitensor as mt


def _conv2d_reference(x, w, b, stride, padding, dilation, groups):
    """Written as the definition, one output element at a time."""
    n_, c_, h_, w_ = x.shape
    o_, cg, kh, kw = w.shape
    sh, sw = stride
    ph, pw = padding
    dh, dw = dilation
    padded = np.pad(x, ((0, 0), (0, 0), (ph, ph), (pw, pw)))
    out_h = (h_ + 2 * ph - dh * (kh - 1) - 1) // sh + 1
    out_w = (w_ + 2 * pw - dw * (kw - 1) - 1) // sw + 1
    out = np.zeros((n_, o_, out_h, out_w))
    per_group = o_ // groups
    for n in range(n_):
        for o in range(o_):
            g = o // per_group
            for i in range(out_h):
                for j in range(out_w):
                    acc = 0.0
                    for c in range(cg):
                        for ki in range(kh):
                            for kj in range(kw):
                                acc += (
                                    padded[
                                        n,
                                        g * cg + c,
                                        i * sh + ki * dh,
                                        j * sw + kj * dw,
                                    ]
                                    * w[o, c, ki, kj]
                                )
                    out[n, o, i, j] = acc + (b[o] if b is not None else 0.0)
    return out


def _conv1d_reference(x, w, b, stride, padding, dilation, groups):
    x4 = x[:, :, None, :]
    w4 = w[:, :, None, :]
    out = _conv2d_reference(x4, w4, b, (1, stride), (0, padding), (1, dilation), groups)
    return out[:, :, 0, :]


def _numeric_grad(f, arr, eps=1e-5):
    grad = np.zeros_like(arr)
    flat, gflat = arr.reshape(-1), grad.reshape(-1)
    for i in range(flat.size):
        old = flat[i]
        flat[i] = old + eps
        high = f(arr)
        flat[i] = old - eps
        low = f(arr)
        flat[i] = old
        gflat[i] = (high - low) / (2 * eps)
    return grad


GEOMETRIES = [
    (1, 0, 1, 1),
    (2, 1, 1, 1),
    (1, 2, 2, 1),
    (2, 1, 3, 1),
    (1, 1, 1, 2),
    (1, 0, 1, 3),
    (1, 1, 2, 2),
    (2, 2, 2, 3),
]


@pytest.mark.parametrize("stride,padding,dilation,groups", GEOMETRIES)
def test_conv2d_matches_the_definition(stride, padding, dilation, groups):
    rng = np.random.default_rng(0)
    channels, out_channels = 6, 6
    x = rng.standard_normal((2, channels, 9, 11)).astype(np.float32)
    w = rng.standard_normal((out_channels, channels // groups, 3, 3)).astype(np.float32)
    b = rng.standard_normal(out_channels).astype(np.float32)

    want = _conv2d_reference(
        x.astype(np.float64),
        w.astype(np.float64),
        b.astype(np.float64),
        (stride, stride),
        (padding, padding),
        (dilation, dilation),
        groups,
    )
    got = mt.nn.conv2d(
        mt.Tensor(x, dtype="float32"),
        mt.Tensor(w, dtype="float32"),
        mt.Tensor(b, dtype="float32"),
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
    ).numpy()
    np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("stride,padding,dilation,groups", GEOMETRIES[:6])
def test_conv1d_matches_the_definition(stride, padding, dilation, groups):
    rng = np.random.default_rng(2)
    channels, out_channels = 6, 6
    x = rng.standard_normal((2, channels, 19)).astype(np.float32)
    w = rng.standard_normal((out_channels, channels // groups, 3)).astype(np.float32)
    b = rng.standard_normal(out_channels).astype(np.float32)

    want = _conv1d_reference(
        x.astype(np.float64),
        w.astype(np.float64),
        b.astype(np.float64),
        stride,
        padding,
        dilation,
        groups,
    )
    got = mt.nn.conv1d(
        mt.Tensor(x, dtype="float32"),
        mt.Tensor(w, dtype="float32"),
        mt.Tensor(b, dtype="float32"),
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
    ).numpy()
    np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-5)


def test_a_depthwise_convolution_is_expressible():
    """`groups == in_channels`: every channel gets its own kernel and nothing
    is mixed. This is the half of a depthwise-separable block that could not be
    written at all before."""
    rng = np.random.default_rng(5)
    channels = 8
    x = rng.standard_normal((2, channels, 7, 7)).astype(np.float32)
    w = rng.standard_normal((channels, 1, 3, 3)).astype(np.float32)

    got = mt.nn.conv2d(
        mt.Tensor(x, dtype="float32"),
        mt.Tensor(w, dtype="float32"),
        None,
        padding=1,
        groups=channels,
    ).numpy()

    # Each output channel must equal a plain single-channel convolution of the
    # matching input channel -- computed here through the ungrouped kernel, so
    # the two paths have to agree.
    for c in range(channels):
        alone = mt.nn.conv2d(
            mt.Tensor(x[:, c : c + 1], dtype="float32"),
            mt.Tensor(w[c : c + 1], dtype="float32"),
            None,
            padding=1,
        ).numpy()
        np.testing.assert_allclose(got[:, c : c + 1], alone, rtol=1e-6, atol=1e-6)


def test_groups_of_one_is_the_old_behaviour():
    """The defaults must not have moved: every existing caller passes neither."""
    rng = np.random.default_rng(7)
    x = rng.standard_normal((2, 4, 8, 8)).astype(np.float32)
    w = rng.standard_normal((5, 4, 3, 3)).astype(np.float32)
    tx, tw = mt.Tensor(x, dtype="float32"), mt.Tensor(w, dtype="float32")
    np.testing.assert_array_equal(
        mt.nn.conv2d(tx, tw, None, stride=2, padding=1).numpy(),
        mt.nn.conv2d(tx, tw, None, stride=2, padding=1, dilation=1, groups=1).numpy(),
    )


def test_dilation_of_one_is_the_old_behaviour():
    rng = np.random.default_rng(11)
    x = rng.standard_normal((2, 3, 10)).astype(np.float32)
    w = rng.standard_normal((4, 3, 3)).astype(np.float32)
    tx, tw = mt.Tensor(x, dtype="float32"), mt.Tensor(w, dtype="float32")
    np.testing.assert_array_equal(
        mt.nn.conv1d(tx, tw, None, stride=1, padding=2).numpy(),
        mt.nn.conv1d(tx, tw, None, stride=1, padding=2, dilation=1).numpy(),
    )


@pytest.mark.parametrize(
    "stride,padding,dilation,groups",
    [
        (1, 0, 1, 1),
        (2, 1, 2, 1),
        (1, 1, 1, 2),
        (1, 1, 2, 2),
        (1, 1, 1, 4),
        (2, 2, 3, 2),
    ],
)
def test_all_three_gradients_match_numerical_differentiation(
    stride, padding, dilation, groups
):
    rng = np.random.default_rng(1)
    channels, out_channels = 4, 4
    x0 = rng.standard_normal((2, channels, 6, 7))
    w0 = rng.standard_normal((out_channels, channels // groups, 3, 3))
    b0 = rng.standard_normal(out_channels)

    def run(xv, wv, bv):
        return mt.nn.conv2d(
            mt.Tensor(xv, dtype="float64"),
            mt.Tensor(wv, dtype="float64"),
            mt.Tensor(bv, dtype="float64"),
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
        )

    # Random weighting: a structured one can sum to zero per channel and make
    # the bias gradient trivially zero, which compares noise against noise.
    coefficients = rng.standard_normal(run(x0, w0, b0).numpy().shape)

    def loss(xv, wv, bv):
        return float((run(xv, wv, bv).numpy() * coefficients).sum())

    tx = mt.Tensor(x0.copy(), dtype="float64", requires_grad=True)
    tw = mt.Tensor(w0.copy(), dtype="float64", requires_grad=True)
    tb = mt.Tensor(b0.copy(), dtype="float64", requires_grad=True)
    out = mt.nn.conv2d(
        tx,
        tw,
        tb,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
    )
    (out * mt.Tensor(coefficients, dtype="float64")).sum().backward()

    for got, want, name in (
        (tx.grad.numpy(), _numeric_grad(lambda a: loss(a, w0, b0), x0.copy()), "input"),
        (
            tw.grad.numpy(),
            _numeric_grad(lambda a: loss(x0, a, b0), w0.copy()),
            "weight",
        ),
        (tb.grad.numpy(), _numeric_grad(lambda a: loss(x0, w0, a), b0.copy()), "bias"),
    ):
        np.testing.assert_allclose(got, want, rtol=1e-6, atol=1e-7, err_msg=name)


def test_a_grouped_gradient_does_not_leak_between_groups():
    """Each group is an independent convolution, so an input channel in one
    group must have no gradient from an output channel in another -- the
    failure a single ungrouped GEMM would produce silently."""
    rng = np.random.default_rng(13)
    channels, out_channels, groups = 4, 4, 2
    x = mt.Tensor(
        rng.standard_normal((1, channels, 5, 5)), dtype="float64", requires_grad=True
    )
    w = mt.Tensor(
        rng.standard_normal((out_channels, channels // groups, 3, 3)), dtype="float64"
    )
    out = mt.nn.conv2d(x, w, None, padding=1, groups=groups)

    # Backpropagate through the first output channel only, which belongs to
    # group 0 and so may only reach input channels 0 and 1.
    mask = np.zeros(out.numpy().shape)
    mask[0, 0] = 1.0
    (out * mt.Tensor(mask, dtype="float64")).sum().backward()

    grad = x.grad.numpy()
    assert np.abs(grad[0, :2]).max() > 0
    np.testing.assert_array_equal(grad[0, 2:], np.zeros_like(grad[0, 2:]))


@pytest.mark.parametrize("groups", [0, 3, 5])
def test_a_grouping_the_channels_cannot_be_split_into_is_refused(groups):
    rng = np.random.default_rng(17)
    x = mt.Tensor(rng.standard_normal((1, 4, 5, 5)).astype(np.float32), dtype="float32")
    w = mt.Tensor(rng.standard_normal((4, 4, 3, 3)).astype(np.float32), dtype="float32")
    with pytest.raises(Exception):
        mt.nn.conv2d(x, w, None, groups=groups)


def test_zero_dilation_is_refused():
    rng = np.random.default_rng(19)
    x = mt.Tensor(rng.standard_normal((1, 2, 5, 5)).astype(np.float32), dtype="float32")
    w = mt.Tensor(rng.standard_normal((2, 2, 3, 3)).astype(np.float32), dtype="float32")
    with pytest.raises(Exception):
        mt.nn.conv2d(x, w, None, dilation=0)


def test_a_dilated_kernel_too_large_for_the_padded_input_is_refused():
    """A 3-tap kernel at dilation 4 spans 9 columns, which a 5-wide input
    cannot hold -- the bound is on the span, not the kernel size."""
    rng = np.random.default_rng(23)
    x = mt.Tensor(rng.standard_normal((1, 2, 5, 5)).astype(np.float32), dtype="float32")
    w = mt.Tensor(rng.standard_normal((2, 2, 3, 3)).astype(np.float32), dtype="float32")
    with pytest.raises(Exception):
        mt.nn.conv2d(x, w, None, dilation=4)


class TestTheLayers:
    def test_conv2d_layer_shapes_its_weight_for_the_grouping(self):
        layer = mt.nn.Conv2d(8, 16, 3, padding=1, dilation=2, groups=4)
        weight = layer.parameters()[0]
        assert tuple(weight.shape) == (16, 2, 3, 3)
        x = mt.Tensor(
            np.random.default_rng(29)
            .standard_normal((2, 8, 16, 16))
            .astype(np.float32),
            dtype="float32",
        )
        assert tuple(layer(x).shape) == (2, 16, 14, 14)

    def test_conv1d_layer_can_be_depthwise(self):
        layer = mt.nn.Conv1d(6, 6, 3, padding=2, dilation=2, groups=6)
        assert tuple(layer.parameters()[0].shape) == (6, 1, 3)
        x = mt.Tensor(
            np.random.default_rng(31).standard_normal((2, 6, 32)).astype(np.float32),
            dtype="float32",
        )
        assert tuple(layer(x).shape) == (2, 6, 32)

    def test_a_layer_with_an_impossible_grouping_is_refused_at_construction(self):
        """Not at the first forward pass: the weight shape depends on `groups`,
        so a bad one would otherwise allocate parameters nothing can match."""
        with pytest.raises(Exception):
            mt.nn.Conv2d(8, 12, 3, groups=5)
