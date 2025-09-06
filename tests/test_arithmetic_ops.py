# Copyright (c) 2025 Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import pytest

import minitensor as mt


def test_subtraction_broadcasting():
    a = mt.Tensor([[5.0, 6.0], [7.0, 8.0]])
    b = mt.Tensor([1.0, 2.0])
    c = a - b
    expected = np.array([[4.0, 4.0], [6.0, 6.0]])
    np.testing.assert_allclose(c.numpy(), expected)


def test_multiplication_broadcasting():
    a = mt.Tensor([[1.0, 2.0], [3.0, 4.0]])
    b = mt.Tensor(2.0)
    c = a * b
    expected = np.array([[2.0, 4.0], [6.0, 8.0]])
    np.testing.assert_allclose(c.numpy(), expected)


def test_division_broadcasting_and_zero():
    a = mt.Tensor([[1.0, 2.0], [3.0, 4.0]])
    b = mt.Tensor([0.0, 2.0])
    c = a / b
    result = c.numpy()
    assert np.isinf(result[0, 0])
    np.testing.assert_allclose(result[0, 1], 1.0)


def test_boolean_arithmetic_error():
    a = mt.Tensor([True, False], dtype="bool")
    b = mt.Tensor([False, True], dtype="bool")
    for op in [
        lambda x, y: x + y,
        lambda x, y: x - y,
        lambda x, y: x * y,
        lambda x, y: x / y,
    ]:
        with pytest.raises(ValueError):
            op(a, b)


def test_shape_mismatch_error():
    a = mt.Tensor([1.0, 2.0, 3.0])
    b = mt.Tensor([[1.0, 2.0]])
    with pytest.raises(ValueError):
        _ = a * b


def test_dtype_mismatch_error():
    a = mt.Tensor([1.0, 2.0], dtype="float32")
    b = mt.Tensor([1, 2], dtype="int32")
    with pytest.raises(TypeError):
        _ = a + b

def test_empty_tensor_arithmetic():
    a = mt.Tensor([]).reshape([0])
    b = mt.Tensor([]).reshape([0])
    c = a + b
    m = a * b
    assert c.tolist() == []
    assert m.tolist() == []


def test_nan_propagation():
    a = mt.Tensor([np.nan, 1.0])
    b = mt.Tensor([1.0, 2.0])
    c = a + b
    result = c.numpy()
    assert np.isnan(result[0])
    np.testing.assert_allclose(result[1], 3.0)
