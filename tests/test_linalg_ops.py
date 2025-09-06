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
