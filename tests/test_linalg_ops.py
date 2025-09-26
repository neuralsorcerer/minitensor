# Copyright (c) 2025 Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import pytest

import minitensor as mt


def test_matmul_basic():
    a = mt.Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    b = mt.Tensor([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]])
    result = a.matmul(b)
    np.testing.assert_allclose(result.numpy(), np.array([[58.0, 64.0], [139.0, 154.0]]))


def test_matmul_shape_mismatch():
    a = mt.Tensor([[1.0, 2.0]])
    b = mt.Tensor([[3.0, 4.0, 5.0]])
    with pytest.raises(ValueError):
        a.matmul(b)


def test_matmul_dtype_mismatch():
    a = mt.Tensor([[1.0, 2.0]], dtype="float32")
    b = mt.Tensor([[3.0], [4.0]], dtype="float64")
    with pytest.raises(TypeError):
        a.matmul(b)


def test_matmul_bool_error():
    a = mt.Tensor([[True, False], [False, True]], dtype="bool")
    b = mt.Tensor([[True, True], [False, False]], dtype="bool")
    with pytest.raises(ValueError):
        a.matmul(b)


def test_matmul_insufficient_dims():
    a = mt.Tensor([1.0, 2.0])
    b = mt.Tensor([3.0, 4.0])
    with pytest.raises(ValueError):
        a.matmul(b)


def test_matmul_batch_dimensions():
    a = mt.Tensor(np.arange(12, dtype=np.float32).reshape(2, 2, 3))
    b = mt.Tensor(np.arange(12, dtype=np.float32).reshape(2, 3, 2))
    result = a.matmul(b)
    expected = np.matmul(
        np.arange(12, dtype=np.float32).reshape(2, 2, 3),
        np.arange(12, dtype=np.float32).reshape(2, 3, 2),
    )
    np.testing.assert_allclose(result.numpy(), expected)


def test_matmul_batch_mismatch_error():
    a = mt.Tensor(np.arange(12, dtype=np.float32).reshape(2, 2, 3))
    b = mt.Tensor(np.arange(18, dtype=np.float32).reshape(3, 3, 2))
    with pytest.raises(ValueError):
        a.matmul(b)


def test_matmul_zero_dimension():
    a = mt.Tensor(np.empty((2, 0), dtype=np.float32))
    b = mt.Tensor(np.empty((0, 3), dtype=np.float32))
    result = a.matmul(b)
    np.testing.assert_allclose(result.numpy(), np.zeros((2, 3), dtype=np.float32))


def test_matmul_inf_nan_propagation():
    a = mt.Tensor([[np.inf, np.nan], [1.0, 2.0]], dtype="float32")
    b = mt.Tensor([[1.0, 2.0], [3.0, 4.0]], dtype="float32")
    result = a.matmul(b).numpy()
    assert np.isnan(result[0]).all()


def test_matmul_backward_batched():
    dev = mt.device("cpu")
    a = mt.Tensor(
        np.arange(24, dtype=np.float32).reshape(2, 3, 4), requires_grad=True, device=dev
    )
    b = mt.Tensor(
        np.arange(32, dtype=np.float32).reshape(2, 4, 4), requires_grad=True, device=dev
    )
    c = a.matmul(b)
    c.sum().backward()
    expected_a = np.matmul(np.ones_like(c.numpy()), np.swapaxes(b.numpy(), -1, -2))
    expected_b = np.matmul(np.swapaxes(a.numpy(), -1, -2), np.ones_like(c.numpy()))
    np.testing.assert_allclose(a.grad.numpy(), expected_a)
    np.testing.assert_allclose(b.grad.numpy(), expected_b)


def test_matmul_backward_requires_grad_flags():
    dev = mt.device("cpu")
    a = mt.Tensor(
        np.arange(4, dtype=np.float32).reshape(2, 2), requires_grad=True, device=dev
    )
    b = mt.Tensor(
        np.arange(4, dtype=np.float32).reshape(2, 2), requires_grad=False, device=dev
    )
    c = a.matmul(b)
    c.sum().backward()
    assert a.grad is not None
    assert b.grad is None


def test_transpose_backward_permutation():
    dev = mt.device("cpu")
    x = mt.Tensor(
        np.arange(24, dtype=np.float32).reshape(2, 3, 4), requires_grad=True, device=dev
    )
    y = x.permute(1, 2, 0)
    grad = mt.Tensor(np.ones((3, 4, 2), dtype=np.float32), device=dev)
    y.backward(grad)
    expected = np.transpose(grad.numpy(), (2, 0, 1))
    np.testing.assert_allclose(x.grad.numpy(), expected)


def test_triu_matches_numpy():
    data = np.arange(9, dtype=np.float32).reshape(3, 3)
    tensor = mt.Tensor(data)
    result = tensor.triu()
    np.testing.assert_allclose(result.numpy(), np.triu(data))


def test_triu_diagonal_offset():
    data = np.arange(9, dtype=np.float32).reshape(3, 3)
    tensor = mt.Tensor(data)
    result = tensor.triu(diagonal=1)
    np.testing.assert_allclose(result.numpy(), np.triu(data, k=1))


def test_triu_gradient_mask():
    data = np.arange(9, dtype=np.float32).reshape(3, 3)
    tensor = mt.Tensor(data, requires_grad=True)
    tensor.triu(diagonal=-1).sum().backward()
    expected_grad = np.triu(np.ones_like(data), k=-1)
    np.testing.assert_allclose(tensor.grad.numpy(), expected_grad)


def test_triu_requires_minimum_rank():
    tensor = mt.Tensor([1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        tensor.triu()


def test_tril_matches_numpy_batched():
    data = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    tensor = mt.Tensor(data)
    result = tensor.tril(diagonal=-1)
    np.testing.assert_allclose(result.numpy(), np.tril(data, k=-1))


def test_tril_gradient_mask():
    data = np.arange(9, dtype=np.float32).reshape(3, 3)
    tensor = mt.Tensor(data, requires_grad=True)
    tensor.tril(diagonal=0).sum().backward()
    expected_grad = np.tril(np.ones_like(data), k=0)
    np.testing.assert_allclose(tensor.grad.numpy(), expected_grad)


def test_tril_requires_minimum_rank():
    tensor = mt.Tensor([1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        tensor.tril()
