# Copyright (c) 2025 Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import pytest

import minitensor as mt


def test_tensor_sign_float_dtype():
    values = np.array([-2.5, 0.0, 3.25, -0.0], dtype=np.float32)
    tensor = mt.Tensor(values.tolist(), dtype="float32")

    result = tensor.sign()

    np.testing.assert_allclose(result.numpy(), np.sign(values))
    assert result.dtype == tensor.dtype


def test_tensor_sign_integer_dtype():
    tensor = mt.Tensor([-3, 0, 4, -7], dtype="int32")

    result = tensor.sign()

    np.testing.assert_array_equal(result.numpy(), np.array([-1, 0, 1, -1], dtype=np.int32))
    assert result.dtype == tensor.dtype


def test_tensor_sign_rejects_boolean():
    tensor = mt.Tensor([True, False], dtype="bool")

    with pytest.raises(ValueError):
        tensor.sign()


def test_tensor_reciprocal_matches_numpy():
    values = np.array([2.0, -4.0, 0.25], dtype=np.float32)
    tensor = mt.Tensor(values.tolist(), dtype="float32")

    result = tensor.reciprocal()

    np.testing.assert_allclose(result.numpy(), np.reciprocal(values))
    assert result.dtype == tensor.dtype


def test_reciprocal_backward_propagates_gradients():
    tensor = mt.Tensor([2.0, -4.0], dtype="float32", requires_grad=True)

    reciprocal = tensor.reciprocal()
    loss = reciprocal.sum()
    loss.backward()

    expected_grad = np.array([-0.25, -0.0625], dtype=np.float32)
    np.testing.assert_allclose(tensor.grad.numpy(), expected_grad, rtol=1e-6, atol=1e-7)


def test_reciprocal_rejects_integers():
    tensor = mt.Tensor.arange(1, 4, dtype="int32")

    with pytest.raises(ValueError):
        tensor.reciprocal()


def test_functional_and_top_level_forwarders():
    tensor = mt.Tensor([-3.0, -1.0, 0.5], dtype="float32")

    np.testing.assert_allclose(mt.functional.sign(tensor).numpy(), tensor.sign().numpy())
    np.testing.assert_allclose(mt.functional.reciprocal(tensor).numpy(), tensor.reciprocal().numpy())
    np.testing.assert_allclose(mt.sign(tensor).numpy(), tensor.sign().numpy())
    np.testing.assert_allclose(mt.reciprocal(tensor).numpy(), tensor.reciprocal().numpy())
