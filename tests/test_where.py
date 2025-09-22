# Copyright (c) 2025 Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import pytest

import minitensor as mt


def test_where_basic_selection():
    condition = mt.Tensor([[True, False], [False, True]], dtype="bool")
    input_tensor = mt.Tensor([[1.0, 2.0], [3.0, 4.0]])
    other_tensor = mt.Tensor([[10.0, 20.0], [30.0, 40.0]])

    result = input_tensor.where(condition, other_tensor)
    expected = np.array([[1.0, 20.0], [30.0, 4.0]], dtype=np.float32)
    np.testing.assert_allclose(result.numpy(), expected)


def test_where_broadcasting():
    condition = mt.Tensor([[True], [False]], dtype="bool")
    input_tensor = mt.Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    other_tensor = mt.Tensor([10.0, 20.0, 30.0])

    result = input_tensor.where(condition, other_tensor)
    expected = np.where(
        np.array([[True], [False]]),
        np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        np.array([[10.0, 20.0, 30.0], [10.0, 20.0, 30.0]]),
    )
    np.testing.assert_allclose(result.numpy(), expected)


def test_where_requires_bool_condition():
    condition = mt.Tensor([0, 1])
    input_tensor = mt.Tensor([1.0, 2.0])
    other_tensor = mt.Tensor([3.0, 4.0])

    with pytest.raises(TypeError):
        input_tensor.where(condition, other_tensor)


def test_where_autograd_masks_gradients():
    condition = mt.Tensor([[True, False], [False, True]], dtype="bool")
    input_tensor = mt.Tensor(
        [[1.0, 2.0], [3.0, 4.0]], requires_grad=True, dtype="float32"
    )
    other_tensor = mt.Tensor(
        [[10.0, 20.0], [30.0, 40.0]], requires_grad=True, dtype="float32"
    )

    result = input_tensor.where(condition, other_tensor)
    loss = result.sum()
    loss.backward()

    np.testing.assert_allclose(
        input_tensor.grad.numpy(),
        np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
    )
    np.testing.assert_allclose(
        other_tensor.grad.numpy(),
        np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32),
    )


def test_where_functional_and_top_level_match():
    condition = [[True, False], [False, True]]
    input_data = [[1.0, 2.0], [3.0, 4.0]]
    other_data = [[10.0, 20.0], [30.0, 40.0]]

    via_method = mt.Tensor(input_data).where(
        mt.Tensor(condition, dtype="bool"), mt.Tensor(other_data)
    )
    via_functional = mt.functional.where(condition, input_data, other_data)
    via_top_level = mt.where(condition, input_data, other_data)
    core_where = mt.numpy_compat.where(
        mt.Tensor(condition, dtype="bool")._tensor,
        mt.Tensor(input_data)._tensor,
        mt.Tensor(other_data)._tensor,
    )
    via_numpy_compat = mt.Tensor.__new__(mt.Tensor)
    via_numpy_compat._tensor = core_where

    expected = np.array([[1.0, 20.0], [30.0, 4.0]], dtype=np.float32)
    np.testing.assert_allclose(via_method.numpy(), expected)
    np.testing.assert_allclose(via_functional.numpy(), expected)
    np.testing.assert_allclose(via_top_level.numpy(), expected)
    np.testing.assert_allclose(via_numpy_compat.numpy(), expected)
