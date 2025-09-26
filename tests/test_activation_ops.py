# Copyright (c) 2025 Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

import math

import numpy as np
import pytest

import minitensor as mt


def test_relu_negative_and_nan():
    t = mt.Tensor([-1.0, float("nan"), 2.0])
    out = t.relu()
    vals = out.numpy()
    np.testing.assert_allclose(vals, np.array([0.0, np.nan, 2.0]), equal_nan=True)


def test_hardshrink_matches_numpy_and_grad():
    data = np.array([-1.2, -0.25, 0.0, 0.35, 0.8], dtype=np.float32)
    lambd = 0.3
    tensor = mt.Tensor(data, requires_grad=True)

    out = tensor.hardshrink(lambd=lambd)
    expected = np.where((data > lambd) | (data < -lambd), data, 0.0)
    np.testing.assert_allclose(out.numpy(), expected, rtol=1e-6)

    out.sum().backward()
    expected_grad = np.where((data > lambd) | (data < -lambd), 1.0, 0.0)
    np.testing.assert_allclose(tensor.grad.numpy(), expected_grad, rtol=1e-6)


def test_hardshrink_invalid_lambda():
    t = mt.Tensor([1.0])
    with pytest.raises(ValueError):
        _ = t.hardshrink(lambd=-1.0)


def test_sigmoid_tanh_extreme_inputs():
    t = mt.Tensor([1000.0, -1000.0])
    sig = t.sigmoid()
    tanh = t.tanh()
    np.testing.assert_allclose(sig.numpy(), np.array([1.0, 0.0]), atol=1e-6)
    np.testing.assert_allclose(tanh.numpy(), np.array([1.0, -1.0]), atol=1e-6)


def test_softsign_matches_numpy_and_grad():
    data = np.array([-5.0, -0.5, 0.0, 0.25, 4.0], dtype=np.float32)
    tensor = mt.Tensor(data, requires_grad=True)

    out = tensor.softsign()
    expected = data / (1.0 + np.abs(data))
    np.testing.assert_allclose(out.numpy(), expected, rtol=1e-6)

    out.sum().backward()
    expected_grad = 1.0 / (1.0 + np.abs(data)) ** 2
    np.testing.assert_allclose(tensor.grad.numpy(), expected_grad, rtol=1e-6)


def test_log_softmax_stability_large_range():
    t = mt.Tensor([[1000.0, -1000.0, 0.0]])
    log_sm = t.log_softmax(dim=1)
    sm_log = t.softmax(dim=1).log()
    np.testing.assert_allclose(log_sm.numpy(), sm_log.numpy(), atol=1e-6)
    np.testing.assert_allclose(
        np.exp(log_sm.numpy()).sum(axis=1), np.array([1.0]), atol=1e-6
    )


def test_log_softmax_backward_matches_manual():
    data = np.array([[1.0, -2.0, 0.5], [-0.5, 1.5, -1.0]], dtype=np.float32)
    grad = np.array([[0.2, -0.1, 0.3], [-0.4, 0.25, -0.15]], dtype=np.float32)

    tensor = mt.Tensor(data, requires_grad=True)
    out = tensor.log_softmax(dim=1)
    out.backward(mt.Tensor(grad))

    expected = grad - np.exp(out.numpy()) * grad.sum(axis=1, keepdims=True)
    np.testing.assert_allclose(tensor.grad.numpy(), expected, rtol=1e-6)


def test_log1p_expm1_matches_numpy():
    values = np.array([-0.5, 0.0, 1.5], dtype=np.float32)
    t = mt.Tensor(values)

    np.testing.assert_allclose(t.log1p().numpy(), np.log1p(values), rtol=1e-6)
    np.testing.assert_allclose(t.expm1().numpy(), np.expm1(values), rtol=1e-6)

    edge = mt.Tensor([-1.0, -2.0], dtype="float32")
    res = edge.log1p().numpy()
    assert np.isneginf(res[0])
    assert np.isnan(res[1])


def test_rsqrt_matches_numpy_and_grad():
    data = np.array([0.25, 1.0, 4.0], dtype=np.float32)
    tensor = mt.Tensor(data, requires_grad=True)

    out = tensor.rsqrt()
    expected = 1.0 / np.sqrt(data)
    np.testing.assert_allclose(out.numpy(), expected, rtol=1e-6)

    out.sum().backward()
    expected_grad = -0.5 * np.power(data, -1.5)
    np.testing.assert_allclose(tensor.grad.numpy(), expected_grad, rtol=1e-5)


def test_logsumexp_matches_numpy_and_grad():
    data = np.array([[1.0, 1.0, -1.0], [2.0, -2.0, 3.0]], dtype=np.float32)
    tensor = mt.Tensor(data, requires_grad=True)

    out = tensor.logsumexp(dim=1)
    expected = np.log(np.exp(data).sum(axis=1))
    np.testing.assert_allclose(out.numpy(), expected, rtol=1e-6)

    out.sum().backward()
    expected_grad = np.exp(data - expected[:, None])
    np.testing.assert_allclose(tensor.grad.numpy(), expected_grad, rtol=1e-6)


def test_logsumexp_keepdim_and_all_dims():
    data = np.array([[[1.0, -2.0], [3.0, 0.5]]], dtype=np.float32)
    tensor = mt.Tensor(data)

    keepdim = tensor.logsumexp(dim=(1, 2), keepdim=True)
    expected_keepdim = np.log(np.exp(data).sum(axis=(1, 2), keepdims=True))
    np.testing.assert_allclose(keepdim.numpy(), expected_keepdim, rtol=1e-6)

    collapsed = tensor.logsumexp(dim=(1, 2), keepdim=False)
    expected_collapsed = expected_keepdim.reshape(collapsed.shape)
    np.testing.assert_allclose(collapsed.numpy(), expected_collapsed, rtol=1e-6)

    with pytest.raises(RuntimeError):
        _ = tensor.astype("int32").logsumexp(dim=1)


def test_softplus_matches_numpy_and_grad():
    data = np.array([-5.0, 0.0, 5.0, 50.0], dtype=np.float32)
    tensor = mt.Tensor(data, requires_grad=True)

    out = tensor.softplus(beta=1.5, threshold=10.0)
    expected = np.where(
        1.5 * data > 10.0,
        data,
        np.log1p(np.exp(1.5 * data)) / 1.5,
    )
    np.testing.assert_allclose(out.numpy(), expected, rtol=1e-5)

    out.sum().backward()
    expected_grad = np.where(
        1.5 * data > 10.0,
        1.0,
        1.0 / (1.0 + np.exp(-1.5 * data)),
    )
    np.testing.assert_allclose(tensor.grad.numpy(), expected_grad, rtol=1e-5)


def test_logaddexp_matches_numpy_and_grad():
    a_vals = np.array([[1.0, 2.0, -100.0], [0.5, -0.75, 3.0]], dtype=np.float32)
    b_vals = np.array([[0.0, -2.0, -100.0], [1.5, 0.25, -4.0]], dtype=np.float32)

    a = mt.Tensor(a_vals, requires_grad=True)
    b = mt.Tensor(b_vals, requires_grad=True)
    result = a.logaddexp(b)
    expected = np.logaddexp(a_vals, b_vals)
    np.testing.assert_allclose(result.numpy(), expected, rtol=1e-6)

    result.sum().backward()
    expected_grad_a = np.exp(a_vals - expected)
    expected_grad_b = np.exp(b_vals - expected)
    np.testing.assert_allclose(a.grad.numpy(), expected_grad_a, rtol=1e-6)
    np.testing.assert_allclose(b.grad.numpy(), expected_grad_b, rtol=1e-6)


def test_gelu_exact_matches_reference_and_grad():
    data = np.array([-2.5, -0.5, 0.0, 0.75, 3.25], dtype=np.float32)
    tensor = mt.Tensor(data, requires_grad=True)

    out = tensor.gelu(approximate="none")
    sqrt_two = np.sqrt(np.array(2.0, dtype=data.dtype))
    erf_vals = np.array(
        [math.erf(float(v)) for v in (data / sqrt_two)], dtype=data.dtype
    )
    expected = 0.5 * data * (1.0 + erf_vals)
    np.testing.assert_allclose(out.numpy(), expected, rtol=1e-6, atol=1e-7)

    out.sum().backward()
    sqrt_two_pi = np.sqrt(np.array(2.0 * np.pi, dtype=data.dtype))
    expected_grad = 0.5 * (1.0 + erf_vals)
    expected_grad += data * np.exp(-0.5 * data**2) / sqrt_two_pi
    np.testing.assert_allclose(tensor.grad.numpy(), expected_grad, rtol=1e-6, atol=1e-7)


def test_gelu_tanh_approximation_grad():
    data = np.array([-1.5, -0.25, 0.5, 1.25], dtype=np.float32)
    tensor = mt.Tensor(data, requires_grad=True)

    out = tensor.gelu(approximate="tanh")
    k = np.sqrt(np.array(2.0 / np.pi, dtype=data.dtype))
    inner = k * (data + 0.044715 * data**3)
    expected = 0.5 * data * (1.0 + np.tanh(inner))
    np.testing.assert_allclose(out.numpy(), expected, rtol=1e-6)

    out.sum().backward()
    tanh_inner = np.tanh(inner)
    sech2 = 1.0 - tanh_inner**2
    expected_grad = 0.5 * (1.0 + tanh_inner)
    expected_grad += 0.5 * data * sech2 * k * (1.0 + 3.0 * 0.044715 * data**2)
    np.testing.assert_allclose(tensor.grad.numpy(), expected_grad, rtol=1e-6)


def test_elu_forward_and_grad():
    data = np.array([-2.0, -0.1, 0.0, 1.5], dtype=np.float32)
    alpha = 1.3
    tensor = mt.Tensor(data, requires_grad=True)

    out = tensor.elu(alpha=alpha)
    expected = np.where(data > 0.0, data, alpha * (np.exp(data) - 1.0))
    np.testing.assert_allclose(out.numpy(), expected, rtol=1e-6)

    out.sum().backward()
    expected_grad = np.where(data > 0.0, 1.0, expected + alpha)
    np.testing.assert_allclose(tensor.grad.numpy(), expected_grad, rtol=1e-6)


def test_selu_forward_and_grad():
    data = np.array([-1.0, -0.2, 0.3, 2.0], dtype=np.float32)
    tensor = mt.Tensor(data, requires_grad=True)

    out = tensor.selu()
    scale = np.float32(1.0507009873554805)
    alpha = np.float32(1.6732632423543772)
    expected = np.where(
        data > 0.0,
        scale * data,
        scale * alpha * (np.exp(data) - 1.0),
    )
    np.testing.assert_allclose(out.numpy(), expected, rtol=1e-6)

    out.sum().backward()
    expected_grad = np.where(data > 0.0, scale, expected + scale * alpha)
    np.testing.assert_allclose(tensor.grad.numpy(), expected_grad, rtol=1e-6)


def test_silu_forward_and_grad():
    data = np.array([-3.0, -0.5, 0.5, 2.5], dtype=np.float32)
    tensor = mt.Tensor(data, requires_grad=True)

    out = tensor.silu()
    sigmoid = 1.0 / (1.0 + np.exp(-data))
    expected = data * sigmoid
    np.testing.assert_allclose(out.numpy(), expected, rtol=1e-6)

    out.sum().backward()
    expected_grad = sigmoid * (1.0 + data * (1.0 - sigmoid))
    np.testing.assert_allclose(tensor.grad.numpy(), expected_grad, rtol=1e-6)
