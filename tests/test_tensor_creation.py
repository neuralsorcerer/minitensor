# Copyright (c) 2025 Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import pytest

from minitensor.numpy_compat import empty_like
from minitensor.tensor import Tensor


def test_eye_int32():
    x = Tensor.eye(3, dtype="int32")
    assert x.dtype == "int32"
    assert np.array_equal(x.numpy(), np.eye(3, dtype=np.int32))


def test_full_bool():
    x = Tensor.full([2, 2], 1, dtype="bool")
    assert x.dtype == "bool"
    assert np.array_equal(x.numpy(), np.ones((2, 2), dtype=bool))


def test_arange_int64():
    x = Tensor.arange(0, 5, dtype="int64")
    assert x.dtype == "int64"
    assert np.array_equal(x.numpy(), np.arange(0, 5, dtype=np.int64))


def test_rand_bool_shape_and_dtype():
    x = Tensor.rand(2, 2, dtype="bool")
    arr = x.numpy()
    assert x.dtype == "bool"
    assert arr.dtype == np.bool_
    assert arr.shape == (2, 2)


def test_randn_int32_dtype():
    x = Tensor.randn(3, dtype="int32")
    arr = x.numpy()
    assert x.dtype == "int32"
    assert arr.dtype == np.int32
    assert arr.shape == (3,)


def test_empty_like_shape_and_dtype():
    a = Tensor.ones([2, 3], dtype="float32")
    b = empty_like(a)
    assert b.shape == a.shape
    assert b.dtype == a.dtype


def test_mixed_type_creation_error():
    with pytest.raises(TypeError):
        Tensor([1, "a"])
