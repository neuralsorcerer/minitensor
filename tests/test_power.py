# Copyright (c) 2025 Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import pytest

from minitensor.tensor import Tensor


def test_tensor_pow_scalar():
    x = Tensor([1.0, 2.0, 3.0], dtype="float32")
    y = x**2
    assert np.allclose(y.numpy(), np.array([1.0, 4.0, 9.0], dtype=np.float32))


def test_tensor_pow_tensor():
    base = Tensor([2.0, 3.0, 4.0], dtype="float32")
    exp = Tensor([1.0, 2.0, 0.5], dtype="float32")
    y = base**exp
    expected = np.array([2.0, 9.0, np.sqrt(4.0)], dtype=np.float32)
    assert np.allclose(y.numpy(), expected)


def test_tensor_pow_shape_mismatch_error():
    base = Tensor([1.0, 2.0], dtype="float32")
    exp = Tensor([3.0, 4.0, 5.0], dtype="float32")
    with pytest.raises(ValueError):
        _ = base**exp


def test_tensor_pow_dtype_mismatch_error():
    base = Tensor([1.0, 2.0], dtype="float32")
    exp = Tensor([1.0, 2.0], dtype="float64")
    with pytest.raises(TypeError):
        _ = base**exp


def test_negative_base_fractional_power_clamps_zero():
    base = Tensor([-1.0], dtype="float32")
    exp = Tensor([0.5], dtype="float32")
    y = base ** exp
    assert y.numpy()[0] == 0.0
