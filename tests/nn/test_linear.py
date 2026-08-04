# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""`input @ weight^T + bias`, and its gradients.

The weight is stored `[out_features, in_features]` but the product needs
`[in_features, out_features]`. Composing this out of `transpose` and `matmul`
copies the whole weight matrix on every forward -- and, because the backward
differentiates that transpose, copies it again for `grad_input` and once more
to carry the weight gradient back through it. A GEMM addresses its operands by
row and column stride, so it can read the weight transposed in place instead.

These tests pin the two things that has to preserve: the result, bit for bit
against the composition it replaces, and the three gradients against their
closed forms.
"""

from __future__ import annotations

import numpy as np
import pytest

import minitensor as mt
from minitensor import nn


@pytest.fixture(autouse=True)
def _clean_graph():
    mt.clear_autograd_graph()
    yield
    mt.clear_autograd_graph()


_SHAPES = [
    ((7, 5), 5, 4),
    ((1, 1), 1, 1),
    ((3, 4, 5), 5, 6),
    ((2, 3, 4, 8), 8, 3),
    ((64, 128), 128, 256),
    ((16, 1024), 1024, 1024),
    # more output features than the batch has rows, and the reverse
    ((2, 96), 96, 7),
    ((96, 2), 2, 96),
]


@pytest.mark.parametrize("shape,in_features,out_features", _SHAPES)
@pytest.mark.parametrize("with_bias", [True, False])
def test_linear_matches_the_transpose_and_matmul_it_replaces(
    shape, in_features, out_features, with_bias
):
    # Bit-for-bit, not merely close: the GEMM sees the same operands in the
    # same order, only addressed differently.
    rng = np.random.default_rng(1)
    x = mt.Tensor(rng.standard_normal(shape).astype(np.float32))
    w = mt.Tensor(rng.standard_normal((out_features, in_features)).astype(np.float32))
    b = mt.Tensor(rng.standard_normal(out_features).astype(np.float32)) if with_bias else None

    got = nn.dense_layer(x, w, b) if with_bias else nn.dense_layer(x, w)
    reference = mt.matmul(x, w.transpose(0, 1))
    if with_bias:
        reference = reference + b

    assert tuple(got.shape) == tuple(reference.shape)
    assert np.array_equal(got.numpy(), reference.numpy())


@pytest.mark.parametrize("shape,in_features,out_features", _SHAPES)
@pytest.mark.parametrize("with_bias", [True, False])
def test_linear_gradients_match_their_closed_forms(shape, in_features, out_features, with_bias):
    rng = np.random.default_rng(2)
    x = rng.standard_normal(shape)
    w = rng.standard_normal((out_features, in_features))
    b = rng.standard_normal(out_features)

    tx = mt.Tensor(x, dtype="float64").requires_grad_(True)
    tw = mt.Tensor(w, dtype="float64").requires_grad_(True)
    tb = mt.Tensor(b, dtype="float64").requires_grad_(True) if with_bias else None

    out = nn.dense_layer(tx, tw, tb) if with_bias else nn.dense_layer(tx, tw)
    np.testing.assert_allclose(out.numpy(), x @ w.T + (b if with_bias else 0), atol=1e-12)

    # A non-uniform upstream gradient: `sum(out)` would hide a transposed or
    # mis-strided weight gradient behind its symmetry.
    upstream = rng.standard_normal(out.shape)
    mt.sum(out * mt.Tensor(upstream, dtype="float64")).backward()

    batch_axes = tuple(range(upstream.ndim - 1))
    np.testing.assert_allclose(mt.get_gradient(tx).numpy(), upstream @ w, atol=1e-11)
    np.testing.assert_allclose(
        mt.get_gradient(tw).numpy(),
        np.tensordot(upstream, x, axes=(batch_axes, batch_axes)),
        atol=1e-11,
    )
    if with_bias:
        np.testing.assert_allclose(
            mt.get_gradient(tb).numpy(),
            upstream.reshape(-1, out_features).sum(0),
            atol=1e-11,
        )


@pytest.mark.parametrize("shape,in_features,out_features", _SHAPES)
def test_linear_gradients_are_right_in_float32_too(shape, in_features, out_features):
    # The dtype arms are separate code paths -- separate GEMM entry points, one
    # per dtype -- so a float64-only gradient check leaves half of them
    # unexercised. Tolerances are float32's, not a weaker claim.
    rng = np.random.default_rng(7)
    x = rng.standard_normal(shape).astype(np.float32)
    w = rng.standard_normal((out_features, in_features)).astype(np.float32)
    b = rng.standard_normal(out_features).astype(np.float32)

    tx = mt.Tensor(x).requires_grad_(True)
    tw = mt.Tensor(w).requires_grad_(True)
    tb = mt.Tensor(b).requires_grad_(True)

    out = nn.dense_layer(tx, tw, tb)
    upstream = rng.standard_normal(out.shape).astype(np.float32)
    mt.sum(out * mt.Tensor(upstream)).backward()

    batch_axes = tuple(range(upstream.ndim - 1))
    scale = max(in_features, out_features)
    np.testing.assert_allclose(
        mt.get_gradient(tx).numpy(), upstream @ w, rtol=1e-4, atol=1e-4 * scale
    )
    np.testing.assert_allclose(
        mt.get_gradient(tw).numpy(),
        np.tensordot(upstream, x, axes=(batch_axes, batch_axes)),
        rtol=1e-4,
        atol=1e-4 * scale,
    )
    np.testing.assert_allclose(
        mt.get_gradient(tb).numpy(),
        upstream.reshape(-1, out_features).sum(0),
        rtol=1e-4,
        atol=1e-4 * scale,
    )


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_linear_keeps_its_dtype(dtype):
    rng = np.random.default_rng(3)
    x = mt.Tensor(rng.standard_normal((4, 6)), dtype=dtype)
    w = mt.Tensor(rng.standard_normal((3, 6)), dtype=dtype)
    out = nn.dense_layer(x, w)
    assert str(out.dtype) == dtype


def test_dense_layer_output_is_unchanged():
    rng = np.random.default_rng(4)
    layer = nn.DenseLayer(64, 32)
    weight, bias = layer.parameters()[0], layer.parameters()[1]
    x = mt.Tensor(rng.standard_normal((8, 64)).astype(np.float32))

    assert np.array_equal(
        layer(x).numpy(), (mt.matmul(x, weight.transpose(0, 1)) + bias).numpy()
    )


def test_a_frozen_weight_gets_no_gradient_but_the_input_still_does():
    rng = np.random.default_rng(5)
    x = mt.Tensor(rng.standard_normal((5, 4)), dtype="float64").requires_grad_(True)
    w = mt.Tensor(rng.standard_normal((3, 4)), dtype="float64")  # requires_grad False

    mt.sum(nn.dense_layer(x, w)).backward()

    assert mt.get_gradient(w) is None
    assert mt.get_gradient(x) is not None


def test_linear_under_no_grad_produces_no_gradient():
    rng = np.random.default_rng(6)
    x = mt.Tensor(rng.standard_normal((5, 4)), dtype="float64").requires_grad_(True)
    w = mt.Tensor(rng.standard_normal((3, 4)), dtype="float64").requires_grad_(True)

    with mt.no_grad():
        out = nn.dense_layer(x, w)

    assert not out.requires_grad


def test_linear_rejects_mismatched_shapes():
    x = mt.Tensor(np.zeros((4, 6), dtype=np.float32))
    with pytest.raises(Exception):
        nn.dense_layer(x, mt.Tensor(np.zeros((3, 5), dtype=np.float32)))
    with pytest.raises(Exception):
        nn.dense_layer(x, mt.Tensor(np.zeros((2, 3, 6), dtype=np.float32)))


def test_linear_handles_a_zero_length_batch():
    x = mt.Tensor(np.zeros((0, 6), dtype=np.float32))
    w = mt.Tensor(np.zeros((3, 6), dtype=np.float32))
    assert tuple(nn.dense_layer(x, w).shape) == (0, 3)


# --- conv2d's weight, read transposed --------------------------------------
#
# `grad_input = weight^T @ grad_cols` needs the weight as `[K, C_out]` while it
# is stored `[C_out, K]`. Materialising that transpose wrote with a stride of
# `C_out` floats across the whole output, which for a 512-channel layer is
# 9 MB touched a cache line at a time -- 17 ms of a 17 ms backward. The GEMM
# reads it transposed by stride instead.
#
# This path only runs when the *input* requires a gradient, i.e. for every conv
# but the first in a network, which is why these tests set that explicitly.


def _conv2d_reference(x, w, b, padding, stride):
    n, _, height, width = x.shape
    out_channels, _, kh, kw = w.shape
    padded = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)))
    out_h = (height + 2 * padding - kh) // stride + 1
    out_w = (width + 2 * padding - kw) // stride + 1
    out = np.zeros((n, out_channels, out_h, out_w))
    for i in range(n):
        for o in range(out_channels):
            for r in range(out_h):
                for c in range(out_w):
                    patch = padded[i, :, r * stride : r * stride + kh, c * stride : c * stride + kw]
                    out[i, o, r, c] = (patch * w[o]).sum() + b[o]
    return out


@pytest.mark.parametrize(
    "n,in_ch,size,out_ch,kernel,padding,stride",
    [
        (2, 3, 8, 4, 3, 1, 1),
        (1, 2, 6, 3, 2, 0, 1),
        (2, 4, 7, 5, 3, 1, 2),
        (1, 5, 5, 2, 5, 2, 1),
        # more output channels than input, and the reverse: the transposed
        # operand's two extents differ, so a swapped pair would show here
        (1, 9, 5, 2, 3, 1, 1),
        (1, 2, 5, 9, 3, 1, 1),
    ],
)
def test_conv2d_input_gradient_matches_finite_differences(
    n, in_ch, size, out_ch, kernel, padding, stride
):
    rng = np.random.default_rng(11)
    x = rng.standard_normal((n, in_ch, size, size))
    w = rng.standard_normal((out_ch, in_ch, kernel, kernel))
    b = rng.standard_normal(out_ch)

    tx = mt.Tensor(x, dtype="float64").requires_grad_(True)
    tw = mt.Tensor(w, dtype="float64").requires_grad_(True)
    tb = mt.Tensor(b, dtype="float64").requires_grad_(True)

    out = nn.conv2d(tx, tw, tb, stride=stride, padding=padding)
    expected = _conv2d_reference(x, w, b, padding, stride)
    np.testing.assert_allclose(out.numpy(), expected, atol=1e-11)

    upstream = rng.standard_normal(expected.shape)
    mt.sum(out * mt.Tensor(upstream, dtype="float64")).backward()
    grad_x = mt.get_gradient(tx).numpy()

    eps = 1e-6
    for _ in range(8):
        idx = tuple(int(rng.integers(0, dim)) for dim in x.shape)
        plus, minus = x.copy(), x.copy()
        plus[idx] += eps
        minus[idx] -= eps
        numeric = (
            (_conv2d_reference(plus, w, b, padding, stride) * upstream).sum()
            - (_conv2d_reference(minus, w, b, padding, stride) * upstream).sum()
        ) / (2 * eps)
        assert abs(numeric - grad_x[idx]) < 1e-5 * max(1.0, abs(numeric)), idx
