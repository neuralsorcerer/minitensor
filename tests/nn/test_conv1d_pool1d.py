# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the 1-D convolution and pooling family.

These delegate to the 2-D kernels with a singleton height, so they are checked
against a direct 1-D sliding-window reference rather than against the 2-D path
they are built on -- otherwise a mistake in the reshaping would be invisible.
"""

import numpy as np
import pytest

import minitensor as mt
from minitensor import functional as F
from minitensor import nn


def conv1d_reference(x, w, b, stride, padding):
    """Explicit sliding window, no reshape trick."""
    batch, _, length = x.shape
    out_channels, _, kernel = w.shape
    padded = np.pad(x, ((0, 0), (0, 0), (padding, padding)))
    out_length = (length + 2 * padding - kernel) // stride + 1
    out = np.zeros((batch, out_channels, out_length), dtype=np.float64)
    for n in range(batch):
        for co in range(out_channels):
            for i in range(out_length):
                window = padded[n, :, i * stride : i * stride + kernel]
                out[n, co, i] = (window * w[co]).sum()
                if b is not None:
                    out[n, co, i] += b[co]
    return out


def pool1d_reference(x, kernel, stride, padding, mode, count_include_pad=True):
    batch, channels, length = x.shape
    out_length = (length + 2 * padding - kernel) // stride + 1
    out = np.zeros((batch, channels, out_length))
    for n in range(batch):
        for c in range(channels):
            for i in range(out_length):
                start = i * stride - padding
                values = [
                    x[n, c, j] for j in range(start, start + kernel) if 0 <= j < length
                ]
                if mode == "max":
                    out[n, c, i] = max(values)
                else:
                    divisor = kernel if count_include_pad else len(values)
                    out[n, c, i] = sum(values) / divisor
    return out


CONV_CASES = [
    (2, 3, 10, 4, 3, 1, 0),
    (1, 2, 8, 3, 4, 2, 1),
    (3, 1, 7, 2, 2, 3, 2),
    (2, 4, 12, 5, 5, 1, 2),
]


@pytest.mark.parametrize("case", CONV_CASES)
def test_conv1d_matches_reference(case):
    batch, in_ch, length, out_ch, kernel, stride, padding = case
    rng = np.random.default_rng(0)
    x = rng.standard_normal((batch, in_ch, length)).astype(np.float32)
    w = rng.standard_normal((out_ch, in_ch, kernel)).astype(np.float32)
    b = rng.standard_normal(out_ch).astype(np.float32)

    got = F.conv1d(mt.Tensor(x), mt.Tensor(w), mt.Tensor(b), stride, padding).numpy()
    expected = conv1d_reference(
        x.astype(np.float64),
        w.astype(np.float64),
        b.astype(np.float64),
        stride,
        padding,
    )
    assert got.shape == expected.shape
    np.testing.assert_allclose(got, expected, atol=1e-4)


def test_conv1d_without_bias():
    rng = np.random.default_rng(9)
    x = rng.standard_normal((1, 2, 6)).astype(np.float32)
    w = rng.standard_normal((3, 2, 3)).astype(np.float32)
    got = F.conv1d(mt.Tensor(x), mt.Tensor(w)).numpy()
    expected = conv1d_reference(x.astype(np.float64), w.astype(np.float64), None, 1, 0)
    np.testing.assert_allclose(got, expected, atol=1e-4)


@pytest.mark.parametrize(
    "kernel, stride, padding", [(2, 2, 0), (3, 1, 0), (2, 1, 1), (3, 2, 1)]
)
@pytest.mark.parametrize("mode", ["max", "avg"])
def test_pool1d_matches_reference(kernel, stride, padding, mode):
    x = np.random.default_rng(1).standard_normal((2, 3, 9))
    tensor = mt.Tensor(x, dtype="float64")
    got = (
        F.max_pool1d(tensor, kernel, stride, padding)
        if mode == "max"
        else F.avg_pool1d(tensor, kernel, stride, padding)
    ).numpy()
    expected = pool1d_reference(x, kernel, stride, padding, mode)
    assert got.shape == expected.shape
    np.testing.assert_allclose(got, expected, atol=1e-12)


def test_avg_pool1d_count_include_pad_changes_the_divisor():
    # A window overlapping the end covers 2 real cells; including the pad makes
    # the divisor 3 instead, which is the documented default.
    x = mt.Tensor(np.array([[[1.0, 2.0, 3.0, 4.0]]]), dtype="float64")
    included = F.avg_pool1d(x, 3, 1, 1, True).numpy()
    excluded = F.avg_pool1d(x, 3, 1, 1, False).numpy()
    np.testing.assert_allclose(included[0, 0, -1], (3.0 + 4.0) / 3.0)
    np.testing.assert_allclose(excluded[0, 0, -1], (3.0 + 4.0) / 2.0)


def test_pooling_defaults_stride_to_kernel_but_convolution_defaults_to_one():
    """The two conventions differ, and getting it wrong changes output length."""
    ramp = np.arange(8, dtype=np.float64).reshape(1, 1, 8)
    pooled = F.max_pool1d(mt.Tensor(ramp, dtype="float64"), 2).numpy()
    assert pooled.shape == (1, 1, 4)
    np.testing.assert_allclose(pooled[0, 0], [1.0, 3.0, 5.0, 7.0])

    signal = mt.Tensor(np.arange(8, dtype=np.float32).reshape(1, 1, 8))
    weight = mt.Tensor(np.ones((1, 1, 2), dtype=np.float32))
    convolved = F.conv1d(signal, weight).numpy()
    assert convolved.shape == (1, 1, 7)


def test_conv1d_gradient_matches_central_differences():
    rng = np.random.default_rng(3)
    x_np = rng.standard_normal((1, 2, 6)).astype(np.float32)
    w_np = rng.standard_normal((2, 2, 3)).astype(np.float32)
    weights = rng.standard_normal((1, 2, 4)).astype(np.float32)

    x = mt.Tensor(x_np, requires_grad=True)
    (F.conv1d(x, mt.Tensor(w_np)) * mt.Tensor(weights)).sum().backward()
    analytic = x.grad.numpy()
    mt.clear_autograd_graph()

    def loss_at(values):
        return float(
            (F.conv1d(mt.Tensor(values), mt.Tensor(w_np)).numpy() * weights).sum()
        )

    # conv1d inherits conv2d's float32-only restriction, so the step has to be
    # large enough to survive f32 cancellation; 1e-3 is the resulting floor.
    h = 1e-2
    for idx in np.ndindex(*x_np.shape):
        plus, minus = x_np.copy(), x_np.copy()
        plus[idx] += h
        minus[idx] -= h
        central = (loss_at(plus) - loss_at(minus)) / (2 * h)
        np.testing.assert_allclose(analytic[idx], central, atol=1e-3)


def test_conv1d_layer_trains_and_exposes_its_configuration():
    layer = nn.Conv1d(3, 4, 5, stride=2, padding=1)
    assert layer.in_channels == 3
    assert layer.out_channels == 4
    assert layer.kernel_size == 5
    assert layer.stride == 2
    assert layer.padding == 1
    assert "Conv1d" in repr(layer)

    x = mt.Tensor(
        np.random.default_rng(0).standard_normal((2, 3, 10)).astype(np.float32),
        requires_grad=True,
    )
    output = nn.Conv1d(3, 4, 3, padding=1)(x)
    assert output.shape == (2, 4, 10)
    output.sum().backward()
    assert x.grad is not None
    mt.clear_autograd_graph()


@pytest.mark.parametrize("name", ["MaxPool1d", "AvgPool1d"])
def test_pool1d_layers_pass_gradient_through(name):
    layer = getattr(nn, name)(2)
    assert layer.kernel_size == 2
    # Pooling's stride convention: defaults to the window.
    assert layer.stride == 2
    assert name in repr(layer)

    x = mt.Tensor(
        np.random.default_rng(1).standard_normal((2, 3, 8)),
        dtype="float64",
        requires_grad=True,
    )
    output = layer(x)
    assert output.shape == (2, 3, 4)
    output.sum().backward()
    assert x.grad is not None
    # Non-overlapping windows: every output takes exactly one unit of gradient,
    # so the total equals the number of outputs however it was distributed.
    assert float(x.grad.numpy().sum()) == pytest.approx(2 * 3 * 4)
    mt.clear_autograd_graph()


def test_conv1d_reports_its_own_name_on_a_dtype_error():
    """Delegating to conv2d must not leak into the message.

    conv1d is implemented on top of conv2d, but a caller who never mentioned
    conv2d should not be told about it.
    """
    x = mt.Tensor(np.zeros((1, 1, 4)), dtype="float64")
    w = mt.Tensor(np.zeros((1, 1, 2)), dtype="float64")
    with pytest.raises(Exception, match="conv1d"):
        F.conv1d(x, w)


def test_conv1d_rejects_wrong_rank():
    good_w = mt.Tensor(np.zeros((1, 1, 2), dtype=np.float32))
    with pytest.raises(Exception):  # 2-D input
        F.conv1d(mt.Tensor(np.zeros((1, 4), dtype=np.float32)), good_w)
    with pytest.raises(Exception):  # 4-D weight
        F.conv1d(
            mt.Tensor(np.zeros((1, 1, 4), dtype=np.float32)),
            mt.Tensor(np.zeros((1, 1, 1, 2), dtype=np.float32)),
        )


@pytest.mark.parametrize("name", ["max_pool1d", "avg_pool1d"])
def test_pool1d_rejects_wrong_rank(name):
    with pytest.raises(Exception):
        getattr(F, name)(mt.Tensor(np.zeros((2, 4)), dtype="float64"), 2)


def test_conv1d_rejects_degenerate_configuration():
    for kwargs in ({"in_channels": 0}, {"out_channels": 0}, {"kernel_size": 0}):
        args = {"in_channels": 2, "out_channels": 3, "kernel_size": 3, **kwargs}
        with pytest.raises(Exception):
            nn.Conv1d(**args)
