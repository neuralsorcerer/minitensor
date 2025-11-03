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


def test_solve_vector_matches_numpy():
    a = mt.Tensor([[3.0, 1.0], [1.0, 2.0]])
    b = mt.Tensor([9.0, 8.0])
    result = a.solve(b)
    expected = np.linalg.solve(a.numpy(), b.numpy())
    np.testing.assert_allclose(result.numpy(), expected)


def test_functional_solve_matches_tensor():
    a = mt.Tensor([[4.0, 2.0], [1.0, 3.0]])
    b = mt.Tensor([10.0, 7.0])
    tensor_result = a.solve(b)
    functional_result = mt.solve(a, b)
    np.testing.assert_allclose(functional_result.numpy(), tensor_result.numpy())


def test_solve_multiple_rhs():
    a = mt.Tensor(np.array([[2.0, 1.0], [5.0, 7.0]], dtype=np.float32))
    b = mt.Tensor(np.array([[11.0, 5.0], [13.0, 6.0]], dtype=np.float32))
    result = a.solve(b)
    expected = np.linalg.solve(a.numpy(), b.numpy())
    np.testing.assert_allclose(result.numpy(), expected)


def test_solve_batched_systems():
    a = mt.Tensor(
        np.array(
            [
                [[3.0, 1.0], [1.0, 2.0]],
                [[4.0, 2.0], [2.0, 5.0]],
            ],
            dtype=np.float32,
        )
    )
    b = mt.Tensor(
        np.array(
            [
                [9.0, 8.0],
                [13.0, 17.0],
            ],
            dtype=np.float32,
        )
    )
    result = a.solve(b)
    a_np = a.numpy()
    b_np = b.numpy()
    expected = np.stack([np.linalg.solve(a_np[i], b_np[i]) for i in range(a_np.shape[0])])
    np.testing.assert_allclose(result.numpy(), expected)


def test_solve_handles_empty_rhs_columns():
    a = mt.Tensor(np.eye(3, dtype=np.float32))
    b = mt.Tensor(np.empty((3, 0), dtype=np.float32))

    result = a.solve(b)

    assert result.shape == (3, 0)
    assert result.numpy().size == 0


def test_solve_backward_gradients():
    a = mt.Tensor([[3.0, 1.0], [1.0, 2.0]], requires_grad=True)
    b = mt.Tensor([9.0, 8.0], requires_grad=True)
    solution = a.solve(b)
    solution.sum().backward()

    a_np = a.numpy()
    x_np = solution.detach().numpy()
    grad_out = np.ones_like(x_np)
    expected_grad_b = np.linalg.solve(a_np.T, grad_out)
    grad_matrix = np.outer(grad_out, x_np)
    expected_grad_a = -np.linalg.solve(a_np.T, grad_matrix)

    np.testing.assert_allclose(b.grad.numpy(), expected_grad_b, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(a.grad.numpy(), expected_grad_a, rtol=1e-5, atol=1e-6)


def test_solve_requires_square_matrix():
    a = mt.Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    b = mt.Tensor([1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        a.solve(b)


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


def test_bmm_matches_numpy():
    a_data = np.arange(12, dtype=np.float32).reshape(2, 2, 3)
    b_data = np.linspace(0.5, 6.0, 12, dtype=np.float32).reshape(2, 3, 2)
    a = mt.Tensor(a_data)
    b = mt.Tensor(b_data)
    result = a.bmm(b)
    expected = np.matmul(a_data, b_data)
    np.testing.assert_allclose(result.numpy(), expected)


def test_top_level_bmm_matches_method():
    a = mt.Tensor(np.arange(12, dtype=np.float32).reshape(2, 2, 3))
    b = mt.Tensor(np.arange(12, dtype=np.float32).reshape(2, 3, 2))
    method_result = a.bmm(b)
    functional_result = mt.bmm(a, b)
    np.testing.assert_allclose(functional_result.numpy(), method_result.numpy())


def test_bmm_requires_three_dimensions():
    a = mt.Tensor(np.ones((2, 3), dtype=np.float32))
    b = mt.Tensor(np.ones((2, 3, 3), dtype=np.float32))
    with pytest.raises(ValueError):
        a.bmm(b)


def test_bmm_batch_mismatch_error():
    a = mt.Tensor(np.arange(12, dtype=np.float32).reshape(2, 2, 3))
    b = mt.Tensor(np.arange(18, dtype=np.float32).reshape(3, 3, 2))
    with pytest.raises(ValueError):
        a.bmm(b)


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


def test_dot_matches_numpy_float():
    a = mt.Tensor([1.0, 2.0, 3.0], dtype="float32")
    b = mt.Tensor([4.0, 5.0, 6.0], dtype="float32")
    result = a.dot(b)
    assert result.dtype == "float32"
    np.testing.assert_allclose(result.numpy(), np.dot(a.numpy(), b.numpy()))


def test_top_level_dot_matches_tensor_method():
    a = mt.Tensor([1.0, 2.0, 3.0], dtype="float32")
    b = mt.Tensor([4.0, 5.0, 6.0], dtype="float32")
    tensor_result = a.dot(b)
    functional_result = mt.dot(a, b)
    assert tensor_result.dtype == "float32"
    assert functional_result.dtype == "float32"
    np.testing.assert_allclose(functional_result.numpy(), tensor_result.numpy())


def test_dot_matches_numpy_int():
    a = mt.Tensor([2, 3, 4], dtype="int64")
    b = mt.Tensor([5, 6, 7], dtype="int64")
    result = a.dot(b)
    assert result.dtype == "int64"
    assert result.item() == int(np.dot(a.numpy(), b.numpy()))


def test_dot_requires_1d_inputs():
    a = mt.Tensor([[1.0, 2.0], [3.0, 4.0]])
    b = mt.Tensor([1.0, 2.0])
    with pytest.raises(ValueError):
        a.dot(b)


def test_dot_mismatched_lengths_raise():
    a = mt.Tensor([1.0, 2.0, 3.0])
    b = mt.Tensor([4.0, 5.0])
    with pytest.raises(ValueError):
        a.dot(b)


def test_dot_backward_gradients():
    a = mt.Tensor([1.0, 2.0, 3.0], requires_grad=True)
    b = mt.Tensor([4.0, 5.0, 6.0], requires_grad=True)
    result = a.dot(b)
    result.backward()
    np.testing.assert_allclose(a.grad.numpy(), b.numpy())
    np.testing.assert_allclose(b.grad.numpy(), a.numpy())


def test_dot_bool_not_supported():
    a = mt.Tensor([True, False, True], dtype="bool")
    b = mt.Tensor([True, True, False], dtype="bool")
    with pytest.raises(ValueError):
        a.dot(b)
