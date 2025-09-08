# Copyright (c) 2025 Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import pytest

from minitensor.tensor import Tensor


def test_to_float64():
    x = Tensor([1.5, -2.3], dtype="float32")
    y = x.to("float64")
    assert y.dtype == "float64"
    assert np.allclose(y.numpy(), np.array([1.5, -2.3], dtype=np.float64))


def test_astype_int():
    x = Tensor([1.5, -2.3], dtype="float32")
    y = x.astype("int32")
    assert y.dtype == "int32"
    assert np.array_equal(y.numpy(), np.array([1, -2], dtype=np.int32))


def test_astype_nan_and_overflow():
    x = Tensor([float("nan"), 1e40, -1e40], dtype="float32")
    y = x.astype("int32")
    assert np.array_equal(
        y.numpy(),
        np.array([0, np.iinfo(np.int32).max, np.iinfo(np.int32).min], dtype=np.int32),
    )


def test_astype_bool_from_float():
    x = Tensor([-0.1, 0.0, 2.0], dtype="float32")
    y = x.astype("bool")
    assert y.dtype == "bool"
    assert np.array_equal(y.numpy(), np.array([True, False, True], dtype=bool))


def test_int64_to_float32():
    x = Tensor([1, -2, 3], dtype="int64")
    y = x.astype("float32")
    assert y.dtype == "float32"
    assert np.array_equal(y.numpy(), np.array([1.0, -2.0, 3.0], dtype=np.float32))


def test_bool_to_int64_and_float():
    x = Tensor([True, False, True], dtype="bool")
    y = x.astype("int64")
    z = x.astype("float64")
    assert y.dtype == "int64"
    assert z.dtype == "float64"
    assert np.array_equal(y.numpy(), np.array([1, 0, 1], dtype=np.int64))
    assert np.array_equal(z.numpy(), np.array([1.0, 0.0, 1.0], dtype=np.float64))


def test_empty_tensor_conversion():
    x = Tensor([], dtype="float32")
    y = x.astype("int32")
    assert y.dtype == "int32"
    assert y.numpy().size == 0


@pytest.mark.parametrize("src_dtype", ["float32", "float64", "int32", "int64", "bool"])
@pytest.mark.parametrize("target_dtype", ["float32", "float64", "int32", "int64", "bool"])
def test_empty_tensor_all_dtype_conversions(src_dtype, target_dtype):
    if src_dtype == target_dtype:
        pytest.skip("identity conversion")
    x = Tensor([], dtype=src_dtype)
    y = x.astype(target_dtype)
    assert y.dtype == target_dtype
    assert y.numpy().size == 0


def test_large_astype_parallel():
    data = np.arange(2048, dtype=np.float32)
    x = Tensor(data, dtype="float32")
    y = x.astype("int32")
    assert y.dtype == "int32"
    assert np.array_equal(y.numpy(), data.astype(np.int32))
