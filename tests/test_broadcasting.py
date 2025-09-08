# Copyright (c) 2025 Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import pytest

import minitensor as mt


def test_scalar_broadcasting_addition():
    a = mt.Tensor([[1.0, 2.0], [3.0, 4.0]])
    b = mt.Tensor(1.0)
    c = a + b
    expected = np.array([[2.0, 3.0], [4.0, 5.0]])
    np.testing.assert_allclose(c.numpy(), expected)


def test_scalar_broadcasting_subtraction():
    a = mt.Tensor([[1.0, 2.0], [3.0, 4.0]])
    b = mt.Tensor(1.0)
    c = a - b
    expected = np.array([[0.0, 1.0], [2.0, 3.0]])
    np.testing.assert_allclose(c.numpy(), expected)


def test_scalar_broadcasting_division():
    a = mt.Tensor([[2.0, 4.0], [6.0, 8.0]])
    b = mt.Tensor(2.0)
    c = a / b
    expected = np.array([[1.0, 2.0], [3.0, 4.0]])
    np.testing.assert_allclose(c.numpy(), expected)


def test_broadcast_incompatible_shapes_error():
    a = mt.Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    b = mt.Tensor([1.0, 2.0])
    with pytest.raises(ValueError):
        _ = a + b


def test_clone_creates_independent_tensor():
    a = mt.Tensor([1.0, 2.0, 3.0])
    b = a.clone()
    b = b + mt.Tensor([1.0, 1.0, 1.0])
    np.testing.assert_allclose(a.numpy(), np.array([1.0, 2.0, 3.0]))
    np.testing.assert_allclose(b.numpy(), np.array([2.0, 3.0, 4.0]))


def test_multi_dimensional_broadcasting():
    a = mt.Tensor(np.ones((2, 1, 3), dtype=np.float32))
    b = mt.Tensor(np.ones((1, 4, 1), dtype=np.float32) * 2)
    c = a + b
    expected = np.ones((2, 4, 3), dtype=np.float32) * 3
    np.testing.assert_allclose(c.numpy(), expected)


def test_high_dim_broadcast_mismatch_error():
    a = mt.Tensor(np.ones((2, 3, 1), dtype=np.float32))
    b = mt.Tensor(np.ones((4, 3, 1), dtype=np.float32))
    with pytest.raises(ValueError):
        _ = a + b


def test_broadcast_zero_dimension():
    a = mt.Tensor(np.empty((0, 3), dtype=np.float32))
    b = mt.Tensor(np.ones((3,), dtype=np.float32))
    result = a + b
    assert result.shape == (0, 3)


def test_conv_bias_broadcasting():
    inp = mt.Tensor(np.ones((2, 3, 4, 4), dtype=np.float32))
    bias = mt.Tensor(np.arange(3, dtype=np.float32).reshape(3, 1, 1))
    out = inp + bias
    expected = np.ones((2, 3, 4, 4), dtype=np.float32) + np.arange(
        3, dtype=np.float32
    ).reshape(1, 3, 1, 1)
    np.testing.assert_allclose(out.numpy(), expected)


def test_conv_bias_broadcasting():
    inp = mt.Tensor(np.ones((2, 3, 4, 4), dtype=np.float32))
    bias = mt.Tensor(np.arange(3, dtype=np.float32).reshape(3, 1, 1))
    out = inp + bias
    expected = np.ones((2, 3, 4, 4), dtype=np.float32) + np.arange(
        3, dtype=np.float32
    ).reshape(1, 3, 1, 1)
    np.testing.assert_allclose(out.numpy(), expected)


def test_broadcast_backward_add():
    dev = mt.device("cpu")
    a = mt.Tensor(np.arange(6, dtype=np.float32).reshape(2, 3), requires_grad=True, device=dev)
    b = mt.Tensor(np.arange(3, dtype=np.float32), requires_grad=True, device=dev)
    c = a + b
    grad = mt.Tensor(1.0, device=dev)
    c.sum().backward(grad)
    np.testing.assert_allclose(a.grad.numpy(), np.ones((2, 3), dtype=np.float32))
    np.testing.assert_allclose(b.grad.numpy(), np.array([2.0, 2.0, 2.0], dtype=np.float32))


def test_broadcast_backward_mul():
    dev = mt.device("cpu")
    a = mt.Tensor(np.arange(6, dtype=np.float32).reshape(2, 3), requires_grad=True, device=dev)
    b = mt.Tensor(np.arange(3, dtype=np.float32) + 1, requires_grad=True, device=dev)
    c = a * b
    grad = mt.Tensor(1.0, device=dev)
    c.sum().backward(grad)
    np.testing.assert_allclose(a.grad.numpy(), np.tile(np.arange(1, 4, dtype=np.float32), (2, 1)))
    np.testing.assert_allclose(b.grad.numpy(), np.array([3.0, 7.0, 11.0], dtype=np.float32))
