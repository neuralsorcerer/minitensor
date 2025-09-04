# Copyright (c) 2025 Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import pytest

import minitensor as mt


def test_cross_product_matches_numpy():
    a = mt.Tensor([1.0, 2.0, 3.0])
    b = mt.Tensor([4.0, 5.0, 6.0])
    c = mt.numpy_compat.cross(a, b)
    expected = np.cross(np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0]))
    np.testing.assert_allclose(c.numpy(), expected)


def test_cross_product_batch_axis():
    a = mt.Tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    b = mt.Tensor([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    c = mt.numpy_compat.cross(a, b, axis=-1)
    expected = np.cross(
        np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
        axis=-1,
    )
    np.testing.assert_allclose(c.numpy(), expected)


def test_cross_product_non_last_axis():
    a_np = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    b_np = np.array([[6.0, 5.0], [4.0, 3.0], [2.0, 1.0]])
    a = mt.Tensor(a_np.tolist())
    b = mt.Tensor(b_np.tolist())
    c = mt.numpy_compat.cross(a, b, axis=0)
    expected = np.cross(a_np, b_np, axis=0)
    np.testing.assert_allclose(c.numpy(), expected)


def test_cross_product_broadcasting():
    a_np = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    b_np = np.array([0.0, 0.0, 1.0])
    a = mt.Tensor(a_np.tolist())
    b = mt.Tensor(b_np.tolist())
    c = mt.numpy_compat.cross(a, b)
    expected = np.cross(a_np, b_np)
    np.testing.assert_allclose(c.numpy(), expected)


def test_cross_product_invalid_axis():
    a = mt.Tensor([1.0, 2.0, 3.0])
    b = mt.Tensor([4.0, 5.0, 6.0])
    with pytest.raises(ValueError):
        mt.numpy_compat.cross(a, b, axis=1)


def test_cross_product_invalid_dimension():
    a = mt.Tensor([1.0, 2.0, 3.0, 4.0])
    b = mt.Tensor([5.0, 6.0, 7.0, 8.0])
    with pytest.raises(ValueError):
        mt.numpy_compat.cross(a, b)


def test_cross_product_broadcast_mismatch():
    a_np = np.ones((2, 3))
    b_np = np.ones((3, 3))
    a = mt.Tensor(a_np.tolist())
    b = mt.Tensor(b_np.tolist())
    with pytest.raises(ValueError):
        mt.numpy_compat.cross(a, b)


def test_cross_product_negative_axis_equivalent():
    a_np = np.array(
        [
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            [[1.0, 1.0, 0.0], [0.0, 1.0, 1.0], [1.0, 0.0, 1.0]],
        ]
    )
    b_np = np.array(
        [
            [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]],
            [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        ]
    )
    a = mt.Tensor(a_np.tolist())
    b = mt.Tensor(b_np.tolist())
    c = mt.numpy_compat.cross(a, b, axis=-2)
    expected = np.cross(a_np, b_np, axis=-2)
    np.testing.assert_allclose(c.numpy(), expected)


def test_cross_product_high_dimensional_broadcasting():
    a_np = np.arange(24.0).reshape(2, 1, 4, 3)
    b_np = np.array([1.0, 0.0, 0.0])
    a = mt.Tensor(a_np.tolist())
    b = mt.Tensor(b_np.tolist())
    c = mt.numpy_compat.cross(a, b, axis=-1)
    expected = np.cross(a_np, b_np, axis=-1)
    np.testing.assert_allclose(c.numpy(), expected)


def test_cross_product_dtype_mismatch():
    a = mt.Tensor([1.0, 2.0, 3.0], dtype="float32")
    b = mt.Tensor([4.0, 5.0, 6.0], dtype="float64")
    with pytest.raises(ValueError):
        mt.numpy_compat.cross(a, b)


def test_cross_product_negative_axis_out_of_range():
    a = mt.Tensor([[1.0, 0.0, 0.0]])
    b = mt.Tensor([[0.0, 1.0, 0.0]])
    with pytest.raises(ValueError):
        mt.numpy_compat.cross(a, b, axis=-3)


def test_cross_product_anti_commutativity():
    a_np = np.array([1.0, 2.0, 3.0])
    b_np = np.array([0.5, -1.0, 2.0])
    a = mt.Tensor(a_np.tolist())
    b = mt.Tensor(b_np.tolist())
    ab = mt.numpy_compat.cross(a, b)
    ba = mt.numpy_compat.cross(b, a)
    np.testing.assert_allclose(ab.numpy(), -ba.numpy())
