# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np

import minitensor as mt
from minitensor import functional as F
from minitensor.tensor import Tensor


def test_argmax_dim():
    x = Tensor([[1.0, 5.0, 3.0], [4.0, 2.0, 6.0]], dtype="float32")
    result = x.argmax(dim=1)
    assert np.array_equal(result.numpy(), np.array([1, 2], dtype=np.int64))


def test_argmax_negative_dim():
    x = Tensor([[1.0, 5.0, 3.0], [4.0, 2.0, 6.0]], dtype="float32")
    result = x.argmax(dim=-1)
    assert np.array_equal(result.numpy(), np.array([1, 2], dtype=np.int64))


def test_argmin_dim_keepdim():
    x = Tensor([[1.0, 5.0, 3.0], [4.0, 2.0, 6.0]], dtype="float32")
    result = x.argmin(dim=1, keepdim=True)
    assert result.shape == (2, 1)
    assert np.array_equal(result.numpy(), np.array([[0], [1]], dtype=np.int64))


def test_argmax_no_dim_first_index():
    x = Tensor([1.0, 5.0, 5.0, -1.0], dtype="float32")
    result = x.argmax()
    assert result.shape == ()
    assert result.numpy().item() == 1


def test_argmin_all_equal_returns_zero():
    x = Tensor([2.0, 2.0, 2.0], dtype="float32")
    result = x.argmin()
    assert result.numpy().item() == 0


def test_argmin_negative_dim():
    x = Tensor([[1.0, 5.0, 3.0], [4.0, 2.0, 6.0]], dtype="float32")
    result = x.argmin(dim=-2)
    assert np.array_equal(result.numpy(), np.array([0, 1, 0], dtype=np.int64))


def test_functional_extrema_return_values_and_indices_like_tensor_methods():
    x_np = np.array([[3.0, 1.0, 2.0], [4.0, 6.0, 5.0]], dtype=np.float32)
    x = Tensor(x_np.tolist())

    max_values, max_indices = F.max(x, dim=1)
    min_values, min_indices = mt.min(x, dim=0)

    np.testing.assert_allclose(max_values.numpy(), np.max(x_np, axis=1))
    assert max_indices.numpy().tolist() == np.argmax(x_np, axis=1).tolist()
    np.testing.assert_allclose(min_values.numpy(), np.min(x_np, axis=0))
    assert min_indices.numpy().tolist() == np.argmin(x_np, axis=0).tolist()
    np.testing.assert_allclose(F.max(x).numpy(), x_np.max())
    np.testing.assert_allclose(mt.min(x).numpy(), x_np.min())


def test_functional_arg_reductions_support_keepdim_and_top_level_exports():
    x_np = np.array([[3.0, 1.0, 2.0], [4.0, 6.0, 5.0]], dtype=np.float32)
    x = Tensor(x_np.tolist())

    assert F.argmax(x, dim=1, keepdim=True).shape == (2, 1)
    assert F.argmax(x, dim=1, keepdim=True).numpy().tolist() == [[0], [1]]
    assert mt.argmin(x, dim=-1).numpy().tolist() == np.argmin(x_np, axis=-1).tolist()
