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


def test_boolean_arithmetic_matches_pytorch():
    a = mt.Tensor([True, False], dtype="bool")
    b = mt.Tensor([False, True], dtype="bool")

    added = a + b
    assert added.dtype == "bool"
    np.testing.assert_array_equal(added.numpy(), np.array([True, True]))

    with pytest.raises(ValueError):
        _ = a - b

    multiplied = a * b
    assert multiplied.dtype == "bool"
    np.testing.assert_array_equal(multiplied.numpy(), np.array([False, False]))

    divided = a / b
    assert divided.dtype == "float32"
    np.testing.assert_allclose(
        divided.numpy(), np.array([np.inf, 0.0], dtype=np.float32)
    )


def test_shape_mismatch_error():
    a = mt.Tensor([1.0, 2.0, 3.0])
    b = mt.Tensor([[1.0, 2.0]])
    with pytest.raises(ValueError):
        _ = a * b


def test_tensor_tensor_dtype_promotion():
    a = mt.Tensor([1.0, 2.0], dtype="float32")
    b = mt.Tensor([1, 2], dtype="int32")
    result = a + b
    assert result.dtype == "float32"
    np.testing.assert_allclose(result.numpy(), np.array([2.0, 4.0], dtype=np.float32))

    c = mt.Tensor([1, 2], dtype="int32")
    d = mt.Tensor([1, 2], dtype="int64")
    promoted = c + d
    assert promoted.dtype == "int64"
    np.testing.assert_array_equal(promoted.numpy(), np.array([2, 4], dtype=np.int64))

    e = mt.Tensor([1, 2], dtype="int32")
    f = mt.Tensor([1, 2], dtype="int32")
    quotient = e / f
    assert quotient.dtype == "float32"
    np.testing.assert_allclose(quotient.numpy(), np.array([1.0, 1.0], dtype=np.float32))


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


def test_inf_minus_inf_nan():
    a = mt.Tensor([np.inf])
    b = mt.Tensor([np.inf])
    c = a - b
    assert np.isnan(c.numpy()).all()


def test_python_float_promotes_int_tensor():
    t = mt.Tensor([1, 2, 3], dtype="int32")
    result = t + 1.5
    assert result.dtype == "float32"
    np.testing.assert_allclose(
        result.numpy(), np.array([2.5, 3.5, 4.5], dtype=np.float32)
    )


def test_python_float_promotes_reverse_add():
    t = mt.Tensor([1, 2, 3], dtype="int32")
    result = 1.5 + t
    assert result.dtype == "float32"
    np.testing.assert_allclose(
        result.numpy(), np.array([2.5, 3.5, 4.5], dtype=np.float32)
    )


def test_python_int_preserves_int_dtype():
    t = mt.Tensor([1, 2, 3], dtype="int32")
    result = t + 1
    assert result.dtype == "int32"
    np.testing.assert_array_equal(result.numpy(), np.array([2, 3, 4], dtype=np.int32))


def test_float64_tensor_with_python_float():
    t = mt.Tensor([1.0, 2.0, 3.0], dtype="float64")
    result = t + 1.5
    assert result.dtype == "float64"
    np.testing.assert_allclose(
        result.numpy(), np.array([2.5, 3.5, 4.5], dtype=np.float64)
    )


def test_boolean_numeric_interactions():
    a = mt.Tensor([True, False], dtype="bool")
    b = mt.Tensor([1, 2], dtype="int32")
    summed = a + b
    assert summed.dtype == "int32"
    np.testing.assert_array_equal(summed.numpy(), np.array([2, 2], dtype=np.int32))

    divided = a / b
    assert divided.dtype == "float32"
    np.testing.assert_allclose(divided.numpy(), np.array([1.0, 0.0], dtype=np.float32))


def test_int64_tensor_with_python_float_promotes_to_float32():
    t = mt.Tensor([1, 2, 3], dtype="int64")
    result = t + 1.5
    assert result.dtype == "float32"
    np.testing.assert_allclose(
        result.numpy(), np.array([2.5, 3.5, 4.5], dtype=np.float32)
    )


def test_reverse_int64_tensor_with_python_float():
    t = mt.Tensor([1, 2, 3], dtype="int64")
    result = 1.5 + t
    assert result.dtype == "float32"
    np.testing.assert_allclose(
        result.numpy(), np.array([2.5, 3.5, 4.5], dtype=np.float32)
    )


def test_maximum_dtype_promotion():
    bools = mt.Tensor([True, False], dtype="bool")
    ints = mt.Tensor([0, 1], dtype="int32")
    floats = mt.Tensor([0.5, -1.5], dtype="float32")

    promoted = bools.maximum(ints)
    assert promoted.dtype == "int32"
    assert np.array_equal(promoted.numpy(), np.array([1, 1], dtype=np.int32))

    promoted_float = bools.maximum(floats)
    assert promoted_float.dtype == "float32"
    assert np.allclose(promoted_float.numpy(), np.array([1.0, 0.0], dtype=np.float32))

    mixed = ints.maximum(mt.Tensor([0.25, 2.5], dtype="float64"))
    assert mixed.dtype == "float64"
    assert np.allclose(mixed.numpy(), np.array([0.25, 2.5], dtype=np.float64))


def test_minimum_dtype_promotion():
    bools = mt.Tensor([True, False], dtype="bool")
    ints = mt.Tensor([0, 1], dtype="int32")
    floats = mt.Tensor([0.5, -1.5], dtype="float32")

    promoted = bools.minimum(ints)
    assert promoted.dtype == "int32"
    assert np.array_equal(promoted.numpy(), np.array([0, 0], dtype=np.int32))

    promoted_float = bools.minimum(floats)
    assert promoted_float.dtype == "float32"
    assert np.allclose(promoted_float.numpy(), np.array([0.5, -1.5], dtype=np.float32))

    mixed = ints.minimum(mt.Tensor([0.25, 2.5], dtype="float64"))
    assert mixed.dtype == "float64"
    assert np.allclose(mixed.numpy(), np.array([0.0, 1.0], dtype=np.float64))


def test_maximum_minimum_nan_behavior():
    a = mt.Tensor([np.nan, 1.0], dtype="float32")
    b = mt.Tensor([0.0, np.nan], dtype="float32")

    max_res = a.maximum(b).numpy()
    min_res = a.minimum(b).numpy()

    assert np.isnan(max_res[0]) and np.isnan(max_res[1])
    assert np.isnan(min_res[0]) and np.isnan(min_res[1])


def test_maximum_backward_flow():
    a = mt.Tensor([-1.0, 2.0, 3.0], requires_grad=True)
    b = mt.Tensor([0.0, 1.5, 3.0], requires_grad=True)

    out = a.maximum(b)
    out.sum().backward()

    np.testing.assert_allclose(
        a.grad.numpy(), np.array([0.0, 1.0, 1.0], dtype=np.float32)
    )
    np.testing.assert_allclose(
        b.grad.numpy(), np.array([1.0, 0.0, 0.0], dtype=np.float32)
    )


def test_minimum_backward_flow():
    a = mt.Tensor([-1.0, 2.0, 3.0], requires_grad=True)
    b = mt.Tensor([0.0, 1.5, 3.0], requires_grad=True)

    out = a.minimum(b)
    out.sum().backward()

    np.testing.assert_allclose(
        a.grad.numpy(), np.array([1.0, 0.0, 1.0], dtype=np.float32)
    )
    np.testing.assert_allclose(
        b.grad.numpy(), np.array([0.0, 1.0, 0.0], dtype=np.float32)
    )
