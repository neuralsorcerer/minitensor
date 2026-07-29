# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""2-D pooling: values against an explicit reference, and gradient flow."""

from __future__ import annotations

import numpy as np
import pytest

import minitensor as mt


def _reference_pool(a, kernel, stride, padding, mode, count_include_pad=True):
    """Walk every window explicitly, independent of the implementation."""
    n, c, h, w = a.shape
    out_h = (h + 2 * padding[0] - kernel[0]) // stride[0] + 1
    out_w = (w + 2 * padding[1] - kernel[1]) // stride[1] + 1
    out = np.zeros((n, c, out_h, out_w), dtype=np.float64)
    for ni in range(n):
        for ci in range(c):
            for i in range(out_h):
                for j in range(out_w):
                    values = []
                    for ky in range(kernel[0]):
                        ih = i * stride[0] + ky
                        if ih < padding[0] or ih >= h + padding[0]:
                            continue
                        for kx in range(kernel[1]):
                            iw = j * stride[1] + kx
                            if iw < padding[1] or iw >= w + padding[1]:
                                continue
                            values.append(a[ni, ci, ih - padding[0], iw - padding[1]])
                    if mode == "max":
                        out[ni, ci, i, j] = max(values)
                    else:
                        divisor = (
                            kernel[0] * kernel[1] if count_include_pad else len(values)
                        )
                        out[ni, ci, i, j] = sum(values) / divisor
    return out


GEOMETRY = [
    ((2, 2), None, None),
    ((3, 3), (2, 2), (1, 1)),
    ((2, 2), (1, 1), (0, 0)),
    ((3, 2), (2, 1), (1, 0)),
]


@pytest.mark.parametrize("kernel,stride,padding", GEOMETRY)
def test_max_pool2d_matches_reference(kernel, stride, padding):
    a = np.random.default_rng(0).standard_normal((2, 3, 7, 6)).astype(np.float32)
    got = np.array(
        mt.functional.max_pool2d(mt.Tensor(a), kernel, stride, padding).tolist()
    )
    want = _reference_pool(
        a.astype(np.float64), kernel, stride or kernel, padding or (0, 0), "max"
    )
    assert got.shape == want.shape
    np.testing.assert_allclose(got, want, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("kernel,stride,padding", GEOMETRY)
@pytest.mark.parametrize("count_include_pad", [True, False])
def test_avg_pool2d_matches_reference(kernel, stride, padding, count_include_pad):
    a = np.random.default_rng(1).standard_normal((2, 3, 7, 6)).astype(np.float32)
    got = np.array(
        mt.functional.avg_pool2d(
            mt.Tensor(a), kernel, stride, padding, count_include_pad
        ).tolist()
    )
    want = _reference_pool(
        a.astype(np.float64),
        kernel,
        stride or kernel,
        padding or (0, 0),
        "avg",
        count_include_pad,
    )
    assert got.shape == want.shape
    np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-6)


def test_pooling_stride_defaults_to_the_window():
    # Pooling defaults its stride to the kernel, unlike convolution which
    # defaults to 1. Getting this wrong silently changes the output shape.
    layer = mt.nn.MaxPool2d((2, 2))
    assert layer.stride == (2, 2)
    a = np.zeros((1, 1, 4, 4), dtype=np.float32)
    assert tuple(mt.functional.max_pool2d(mt.Tensor(a), (2, 2)).shape) == (1, 1, 2, 2)


def test_max_pool_gradient_reaches_only_the_winners():
    a = np.array([[[[1.0, 2.0], [3.0, 4.0]]]], dtype=np.float32)
    t = mt.Tensor(a, requires_grad=True)
    mt.functional.max_pool2d(t, (2, 2)).sum().backward()
    # Only the 4.0 wins its single window.
    np.testing.assert_allclose(
        np.array(t.grad.tolist()), np.array([[[[0.0, 0.0], [0.0, 1.0]]]])
    )


def test_avg_pool_gradient_is_spread_evenly():
    a = np.arange(16, dtype=np.float32).reshape(1, 1, 4, 4)
    t = mt.Tensor(a, requires_grad=True)
    mt.functional.avg_pool2d(t, (2, 2)).sum().backward()
    # Every element belongs to exactly one 2x2 window.
    np.testing.assert_allclose(np.array(t.grad.tolist()), np.full((1, 1, 4, 4), 0.25))


def test_pooling_layers_compose_in_sequential_and_pass_gradients():
    a = np.random.default_rng(2).standard_normal((2, 3, 8, 8)).astype(np.float32)
    model = mt.nn.Sequential(
        [
            mt.nn.Conv2d(3, 4, (3, 3), padding=(1, 1)),
            mt.nn.ReLU(),
            mt.nn.MaxPool2d((2, 2)),
            mt.nn.AvgPool2d((2, 2)),
        ]
    )
    out = model(mt.Tensor(a))
    assert tuple(out.shape) == (2, 4, 2, 2)
    out.sum().backward()
    grads = [p.grad for p in model.parameters()]
    assert grads and all(g is not None for g in grads)


def test_pooling_rejects_invalid_geometry():
    t = mt.Tensor(np.zeros((1, 1, 4, 4), dtype=np.float32))
    with pytest.raises(Exception):
        mt.functional.max_pool2d(t, (5, 5))
    with pytest.raises(Exception):
        mt.functional.max_pool2d(t, (2, 2), (1, 1), (2, 2))
    with pytest.raises(Exception):
        mt.functional.max_pool2d(mt.Tensor(np.zeros(16, dtype=np.float32)), (2, 2))


def test_pooling_repr_uses_python_booleans():
    assert "count_include_pad=False" in repr(
        mt.nn.AvgPool2d((2, 2), count_include_pad=False)
    )
    assert repr(mt.nn.MaxPool2d((2, 2))).startswith("MaxPool2d(kernel_size=(2, 2)")
