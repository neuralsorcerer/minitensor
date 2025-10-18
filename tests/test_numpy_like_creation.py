# Copyright (c) 2025 Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np

import minitensor as mt
from minitensor.tensor import Tensor


def test_zeros_like_preserves_metadata():
    base = Tensor.ones((2, 3), dtype="float32", requires_grad=True)

    result = mt.numpy_compat.zeros_like(base)

    assert isinstance(result, Tensor)
    assert result.device == base.device
    assert result.dtype == base.dtype
    assert result.requires_grad is True
    np.testing.assert_array_equal(result.numpy(), np.zeros((2, 3), dtype=np.float32))


def test_ones_like_accepts_array_like_and_dtype_override():
    base = np.arange(4, dtype=np.int32).reshape(2, 2)

    result = mt.numpy_compat.ones_like(base, dtype="float64")

    assert isinstance(result, Tensor)
    assert result.device == "cpu"
    assert result.dtype == "float64"
    assert result.requires_grad is False
    np.testing.assert_array_equal(result.numpy(), np.ones((2, 2), dtype=np.float64))


def test_empty_like_matches_shape_and_requires_grad():
    base = Tensor.arange(0, 6, dtype="float32", requires_grad=True).reshape(2, 3)

    result = mt.numpy_compat.empty_like(base)

    assert isinstance(result, Tensor)
    assert result.shape == base.shape
    assert result.dtype == base.dtype
    assert result.device == base.device
    assert result.requires_grad is True


def test_full_like_uses_source_metadata():
    base = Tensor.ones((3,), dtype="float32", requires_grad=True)

    result = mt.numpy_compat.full_like(base, 7.5)

    assert isinstance(result, Tensor)
    assert result.device == base.device
    assert result.dtype == base.dtype
    assert result.requires_grad is True
    np.testing.assert_allclose(result.numpy(), np.full((3,), 7.5, dtype=np.float32))
