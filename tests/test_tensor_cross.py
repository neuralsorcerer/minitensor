# Copyright (c) 2025 Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import pytest

import minitensor as mt
from minitensor.tensor import Tensor


def test_tensor_cross_matches_numpy():
    a = Tensor([1.0, 2.0, 3.0])
    b = Tensor([4.0, 5.0, 6.0])
    c = a.cross(b)
    expected = np.cross(np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0]))
    np.testing.assert_allclose(c.numpy(), expected)


def test_tensor_cross_axis_parameter():
    a_np = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    b_np = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    a = Tensor(a_np.tolist())
    b = Tensor(b_np.tolist())
    c = a.cross(b, axis=-1)
    expected = np.cross(a_np, b_np, axis=-1)
    np.testing.assert_allclose(c.numpy(), expected)


def test_tensor_cross_invalid_axis():
    a = Tensor([1.0, 2.0, 3.0])
    b = Tensor([4.0, 5.0, 6.0])
    with pytest.raises(ValueError):
        a.cross(b, axis=1)
