# Copyright (c) 2025 Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import pytest

import minitensor as mt


def _layer_norm_reference(x, normalized_shape, weight=None, bias=None, eps=1e-5):
    normalized_shape = tuple(normalized_shape)
    dims = len(normalized_shape)
    axis = tuple(range(x.ndim - dims, x.ndim))
    mean = x.mean(axis=axis, keepdims=True)
    var = ((x - mean) ** 2).mean(axis=axis, keepdims=True)
    inv_std = 1.0 / np.sqrt(var + eps)
    normalized = (x - mean) * inv_std

    output = normalized
    if weight is not None:
        reshape = (1,) * (x.ndim - dims) + weight.shape
        output = output * weight.reshape(reshape)
    if bias is not None:
        reshape = (1,) * (x.ndim - dims) + bias.shape
        output = output + bias.reshape(reshape)

    return output, normalized, inv_std


def test_layer_norm_forward_matches_numpy():
    x = np.array([[1.2, -0.5, 2.0], [0.7, -1.3, 0.25]], dtype=np.float32)
    weight = np.array([1.5, 0.75, -0.25], dtype=np.float32)
    bias = np.array([0.1, -0.2, 0.05], dtype=np.float32)
    eps = 1e-5

    ref, _, _ = _layer_norm_reference(x, (3,), weight=weight, bias=bias, eps=eps)

    tensor = mt.Tensor(x, requires_grad=True)
    weight_tensor = mt.Tensor(weight, requires_grad=True)
    bias_tensor = mt.Tensor(bias, requires_grad=True)

    out = tensor.layer_norm((3,), weight=weight_tensor, bias=bias_tensor, eps=eps)
    np.testing.assert_allclose(out.numpy(), ref, rtol=1e-6, atol=1e-6)


def test_layer_norm_gradients_match_manual_formula():
    x = np.array([[1.2, -0.5, 2.0], [0.7, -1.3, 0.25]], dtype=np.float32)
    weight = np.array([1.5, 0.75, -0.25], dtype=np.float32)
    bias = np.array([0.1, -0.2, 0.05], dtype=np.float32)
    eps = 1e-5

    tensor = mt.Tensor(x, requires_grad=True)
    weight_tensor = mt.Tensor(weight, requires_grad=True)
    bias_tensor = mt.Tensor(bias, requires_grad=True)

    out = tensor.layer_norm((3,), weight=weight_tensor, bias=bias_tensor, eps=eps)
    out.sum().backward()

    grad_output = np.ones_like(x, dtype=np.float32)
    _, normalized, inv_std = _layer_norm_reference(
        x, (3,), weight=weight, bias=bias, eps=eps
    )

    grad_output_hat = grad_output * weight.reshape(1, -1)
    m = float(np.prod((3,), dtype=np.int64))
    sum_grad = grad_output_hat.sum(axis=-1, keepdims=True)
    sum_grad_norm = (grad_output_hat * normalized).sum(axis=-1, keepdims=True)
    grad_input_expected = (
        (grad_output_hat * m - sum_grad - normalized * sum_grad_norm) * inv_std / m
    )
    grad_weight_expected = (grad_output * normalized).sum(axis=0)
    grad_bias_expected = grad_output.sum(axis=0)

    np.testing.assert_allclose(
        tensor.grad.numpy(), grad_input_expected, rtol=1e-6, atol=1e-6
    )
    np.testing.assert_allclose(
        weight_tensor.grad.numpy(), grad_weight_expected, rtol=1e-6, atol=1e-6
    )
    np.testing.assert_allclose(
        bias_tensor.grad.numpy(), grad_bias_expected, rtol=1e-6, atol=1e-6
    )


def test_layer_norm_requires_normalized_shape():
    t = mt.Tensor([1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        _ = t.layer_norm(())
