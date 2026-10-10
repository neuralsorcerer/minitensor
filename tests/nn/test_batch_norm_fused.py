# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""Batch normalization is two passes and a written-out gradient, held here to
the definition.

It was a chain of tensor operations whose backward was composed from theirs,
so its gradient was right by construction. A single kernel with its own
backward has to be shown to be, so every case below is checked against float64
arithmetic written from the formula: the output, all three gradients, and the
running statistics a training step leaves behind, in training and evaluation
mode, over the layouts a batch norm meets -- `[N, C]`, `[N, C, L]` and
`[N, C, H, W]` -- with and without the affine parameters.
"""

from __future__ import annotations

import numpy as np
import pytest

import minitensor as mt
from minitensor import functional as F


def _reference(x, w, b, go, mean, var, eps, from_batch):
    """Output and the gradients of `sum(output * go)` from the definition."""
    axes = tuple(d for d in range(x.ndim) if d != 1)
    shape = [1, -1] + [1] * (x.ndim - 2)
    m = x.size // x.shape[1]
    inv = 1.0 / np.sqrt(var + eps)
    xhat = (x - mean.reshape(shape)) * inv.reshape(shape)
    gain = np.ones(x.shape[1]) if w is None else w
    shift = np.zeros(x.shape[1]) if b is None else b
    out = xhat * gain.reshape(shape) + shift.reshape(shape)
    sum_g = go.sum(axis=axes)
    sum_gx = (go * xhat).sum(axis=axes)
    scale = (gain * inv).reshape(shape)
    if from_batch:
        dx = scale * (
            go - (sum_g / m).reshape(shape) - xhat * (sum_gx / m).reshape(shape)
        )
    else:
        dx = scale * go
    return out, dx, sum_gx, sum_g


# The last two have planes past one task's worth (32768 values), which are
# summed and written in pieces, one of them partial.
@pytest.mark.parametrize(
    "shape",
    [
        (37, 5),
        (6, 4, 9),
        (5, 3, 7, 6),
        (1, 2, 4, 4),
        (1, 2, 200, 200),
        (2, 3, 190, 190),
    ],
)
@pytest.mark.parametrize("affine", [True, False])
@pytest.mark.parametrize("training", [True, False])
def test_batch_norm_matches_the_definition(shape, affine, training):
    rng = np.random.default_rng(sum(shape) + 7 * affine + 3 * training)
    c = shape[1]
    x0 = rng.standard_normal(shape) * 3 + 2
    w0 = rng.standard_normal(c) if affine else None
    b0 = rng.standard_normal(c) if affine else None
    rm0 = rng.standard_normal(c)
    rv0 = rng.random(c) + 0.5
    eps, momentum = 1e-5, 0.3

    x = mt.Tensor(x0, dtype="float64", requires_grad=True)
    w = None if w0 is None else mt.Tensor(w0, dtype="float64", requires_grad=True)
    b = None if b0 is None else mt.Tensor(b0, dtype="float64", requires_grad=True)
    rm = mt.Tensor(rm0.copy(), dtype="float64")
    rv = mt.Tensor(rv0.copy(), dtype="float64")
    out = F.batch_norm(x, rm, rv, w, b, training=training, momentum=momentum, eps=eps)
    go = rng.standard_normal(shape)
    (out * mt.Tensor(go, dtype="float64")).sum().backward()

    axes = tuple(d for d in range(len(shape)) if d != 1)
    if training:
        mean, var = x0.mean(axis=axes), x0.var(axis=axes)
    else:
        mean, var = rm0, rv0
    want = _reference(x0, w0, b0, go, mean, var, eps, from_batch=training)

    np.testing.assert_allclose(out.numpy(), want[0], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(x.grad.numpy(), want[1], rtol=1e-10, atol=1e-11)
    if affine:
        np.testing.assert_allclose(w.grad.numpy(), want[2], rtol=1e-11, atol=1e-11)
        np.testing.assert_allclose(b.grad.numpy(), want[3], rtol=1e-12, atol=1e-12)

    if training:
        m = x0.size // c
        np.testing.assert_allclose(
            rm.numpy(), (1 - momentum) * rm0 + momentum * mean, rtol=1e-13
        )
        np.testing.assert_allclose(
            rv.numpy(),
            (1 - momentum) * rv0 + momentum * var * m / (m - 1),
            rtol=1e-13,
        )
    else:
        np.testing.assert_array_equal(rm.numpy(), rm0)
        np.testing.assert_array_equal(rv.numpy(), rv0)


def test_float32_agrees_with_float64():
    rng = np.random.default_rng(2)
    x0 = (rng.standard_normal((16, 8, 12, 12)) * 4 + 10).astype(np.float32)
    layer = mt.nn.BatchNorm2d(8)
    got = layer(mt.Tensor(x0, dtype="float32")).numpy()
    axes = (0, 2, 3)
    x = x0.astype(np.float64)
    want = (x - x.mean(axis=axes, keepdims=True)) / np.sqrt(
        x.var(axis=axes, keepdims=True) + 1e-5
    )
    np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-5)


def test_a_nan_reaches_only_its_own_channel():
    x0 = np.random.default_rng(3).standard_normal((4, 3, 5))
    x0[1, 1, 2] = np.nan
    out = F.batch_norm(mt.Tensor(x0, dtype="float64"), training=True).numpy()
    assert np.isnan(out[:, 1]).all()
    assert np.isfinite(out[:, [0, 2]]).all()


def test_a_frozen_layer_still_passes_its_input_gradient():
    """Weight and bias frozen, so only the input asks for a gradient."""
    rng = np.random.default_rng(4)
    x0 = rng.standard_normal((6, 3, 4))
    layer = mt.nn.BatchNorm1d(3)
    layer.requires_grad_(False)
    x = mt.Tensor(x0, dtype="float64", requires_grad=True)
    layer.astype("float64")
    layer(x).sum().backward()
    # A plain sum's gradient through a batch-statistics normalization is zero:
    # shifting every value of a channel moves its mean by the same amount.
    np.testing.assert_allclose(x.grad.numpy(), 0.0, atol=1e-12)


@pytest.mark.parametrize("shape", [(5, 3), (4, 3, 6)])
def test_in_evaluation_a_nan_input_leaves_the_gradient_finite(shape):
    """With running statistics the output is affine in the input, so its
    gradient does not involve the input at all -- a NaN value makes a NaN
    output, not a NaN gradient."""
    rng = np.random.default_rng(5)
    x0 = rng.standard_normal(shape)
    x0.flat[4] = np.nan
    rv0 = rng.random(shape[1]) + 0.5
    w0 = rng.standard_normal(shape[1])
    x = mt.Tensor(x0, dtype="float64", requires_grad=True)
    out = F.batch_norm(
        x,
        mt.Tensor(np.zeros(shape[1]), dtype="float64"),
        mt.Tensor(rv0, dtype="float64"),
        mt.Tensor(w0, dtype="float64"),
        None,
        training=False,
    )
    out.sum().backward()
    expected = np.broadcast_to(
        (w0 / np.sqrt(rv0 + 1e-5)).reshape([1, -1] + [1] * (len(shape) - 2)), shape
    )
    np.testing.assert_allclose(x.grad.numpy(), expected, rtol=1e-14)
