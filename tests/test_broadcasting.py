# Copyright (c) 2025 Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import pytest

import minitensor as mt


def test_scalar_broadcasting_addition():
    a = mt.Tensor([[1.0, 2.0], [3.0, 4.0]])
    b = mt.Tensor(1.0)
    c = a + b
    expected = np.array([[2.0, 3.0], [4.0, 5.0]])
    np.testing.assert_allclose(c.numpy(), expected)


def test_broadcast_incompatible_shapes_error():
    a = mt.Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    b = mt.Tensor([1.0, 2.0])
    with pytest.raises(ValueError):
        _ = a + b


def test_clone_creates_independent_tensor():
    a = mt.Tensor([1.0, 2.0, 3.0])
    b = a.clone()
    b = b + mt.Tensor([1.0, 1.0, 1.0])
    np.testing.assert_allclose(a.numpy(), np.array([1.0, 2.0, 3.0]))
    np.testing.assert_allclose(b.numpy(), np.array([2.0, 3.0, 4.0]))
