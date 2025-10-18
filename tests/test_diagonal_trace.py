# Copyright (c) 2025 Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np

import minitensor as mt


def test_tensor_diagonal_default():
    tensor = mt.Tensor([[1.0, 2.0], [3.0, 4.0]])
    diag = tensor.diagonal()
    np.testing.assert_allclose(diag.numpy(), np.array([1.0, 4.0], dtype=np.float32))


def test_tensor_diagonal_offsets():
    tensor = mt.Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    upper = tensor.diagonal(offset=1)
    lower = tensor.diagonal(offset=-1)
    np.testing.assert_allclose(upper.numpy(), np.array([2.0, 6.0], dtype=np.float32))
    np.testing.assert_allclose(lower.numpy(), np.array([4.0], dtype=np.float32))


def test_tensor_diagonal_high_dimension_shape():
    tensor = mt.arange(24, dtype="float32").reshape(2, 3, 4)
    diag = tensor.diagonal(dim1=1, dim2=2)
    assert diag.shape == (2, 3)


def test_tensor_diagonal_empty_for_large_offset():
    tensor = mt.Tensor([[1.0, 2.0], [3.0, 4.0]])
    diag = tensor.diagonal(offset=5)
    assert diag.shape == (0,)
    assert diag.numel() == 0


def test_tensor_trace_matches_numpy():
    tensor = mt.Tensor([[1.0, 2.0], [3.0, 4.0]])
    traced = tensor.trace()
    np.testing.assert_allclose(traced.numpy(), np.array(5.0, dtype=np.float32))


def test_functional_diagonal_and_trace():
    tensor = mt.Tensor([[1.0, 2.0], [3.0, 4.0]])
    diag_fn = mt.diagonal(tensor)
    trace_fn = mt.trace(tensor)
    np.testing.assert_allclose(diag_fn.numpy(), np.array([1.0, 4.0], dtype=np.float32))
    np.testing.assert_allclose(trace_fn.numpy(), np.array(5.0, dtype=np.float32))


def test_diagonal_backward_gradients():
    tensor = mt.Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    diag = tensor.diagonal()
    loss = diag.sum()
    loss.backward()
    grad = tensor.grad
    assert grad is not None
    np.testing.assert_allclose(
        grad.numpy(), np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    )
    mt.clear_autograd_graph()
