# Copyright (c) 2025 Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np

from minitensor.tensor import Tensor


def test_np_asarray_returns_numpy_array():
    t = Tensor([[1, 2], [3, 4]], dtype="float32")
    arr = np.asarray(t)
    assert isinstance(arr, np.ndarray)
    assert arr.dtype == np.float32
    assert np.array_equal(arr, np.array([[1, 2], [3, 4]], dtype=np.float32))


def test_np_asarray_with_dtype():
    t = Tensor([1, 2, 3], dtype="float32")
    arr = np.asarray(t, dtype=np.float64)
    assert arr.dtype == np.float64
    assert np.array_equal(arr, np.array([1.0, 2.0, 3.0], dtype=np.float64))


def test_np_add_dispatches_to_tensor():
    a = Tensor([1, 2, 3])
    b = Tensor([4, 5, 6])
    result = np.add(a, b)
    assert isinstance(result, Tensor)
    assert np.array_equal(result.numpy(), np.array([5, 7, 9]))


def test_np_multiply_with_numpy_array():
    t = Tensor([1, 2, 3])
    arr = np.array([2, 2, 2])
    result = np.multiply(t, arr)
    assert isinstance(result, Tensor)
    assert np.array_equal(result.numpy(), np.array([2, 4, 6]))


def test_np_multiply_int32_array():
    t = Tensor([1, 2, 3], dtype="int32")
    arr = np.array([2, 2, 2], dtype=np.int32)
    result = np.multiply(t, arr)
    assert isinstance(result, Tensor)
    assert result.dtype == "int32"
    assert np.array_equal(result.numpy(), np.array([2, 4, 6], dtype=np.int32))


def test_np_negative_returns_tensor():
    t = Tensor([1, -2, 3])
    result = np.negative(t)
    assert isinstance(result, Tensor)
    assert np.array_equal(result.numpy(), np.array([-1, 2, -3]))


def test_np_trig_dispatches_to_tensor():
    t = Tensor([0.0, np.pi / 2, np.pi])
    sin_result = np.sin(t)
    cos_result = np.cos(t)
    assert isinstance(sin_result, Tensor)
    assert isinstance(cos_result, Tensor)
    np.testing.assert_allclose(
        sin_result.numpy(), np.sin([0.0, np.pi / 2, np.pi]), rtol=1e-6, atol=1e-6
    )
    np.testing.assert_allclose(
        cos_result.numpy(), np.cos([0.0, np.pi / 2, np.pi]), rtol=1e-6, atol=1e-6
    )


def test_np_add_dtype_promotion():
    t = Tensor([1, 2, 3]).astype("float64")
    arr = np.array([1, 2, 3], dtype=np.int32)
    result = np.add(t, arr)
    assert isinstance(result, Tensor)
    assert result.dtype == "float64"
    np.testing.assert_allclose(result.numpy(), np.array([2, 4, 6], dtype=np.float64))


def test_from_numpy_int_and_bool():
    int_arr = np.array([1, 2, 3], dtype=np.int32)
    t_int = Tensor.from_numpy(int_arr)
    assert t_int.dtype == "int32"
    assert np.array_equal(t_int.numpy(), int_arr)

    bool_arr = np.array([True, False], dtype=np.bool_)
    t_bool = Tensor.from_numpy(bool_arr)
    assert t_bool.dtype == "bool"
    assert np.array_equal(t_bool.numpy(), bool_arr)


def test_np_maximum_minimum_dispatch():
    a = Tensor([1, 3, 2], dtype="float32")
    b = np.array([2, 1, 4], dtype=np.float32)
    max_res = np.maximum(a, b)
    min_res = np.minimum(a, b)
    assert isinstance(max_res, Tensor)
    assert isinstance(min_res, Tensor)
    assert np.array_equal(max_res.numpy(), np.array([2, 3, 4], dtype=np.float32))
    assert np.array_equal(min_res.numpy(), np.array([1, 1, 2], dtype=np.float32))


def test_np_maximum_minimum_bool():
    a = Tensor([True, False], dtype="bool")
    b = np.array([False, True], dtype=np.bool_)
    max_res = np.maximum(a, b)
    min_res = np.minimum(a, b)
    assert max_res.dtype == "bool"
    assert min_res.dtype == "bool"
    assert np.array_equal(max_res.numpy(), np.array([True, True], dtype=np.bool_))
    assert np.array_equal(min_res.numpy(), np.array([False, False], dtype=np.bool_))
