# Copyright (c) 2025 Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import pytest
from minitensor import Tensor, functional as F

def test_conv2d_basic():
    x = Tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
    w = Tensor([[[[1.0]]]])
    b = Tensor([1.0])
    y = F.conv2d(x, w, b)
    np.testing.assert_allclose(y.numpy(), np.array([[[[2.0, 3.0], [4.0, 5.0]]]]))


def test_conv2d_padding_stride():
    x = Tensor(np.arange(1, 17, dtype=np.float32).reshape(1, 1, 4, 4))
    w = Tensor(np.array([1, 0, 0, 1], dtype=np.float32).reshape(1, 1, 2, 2))
    y = F.conv2d(x, w, stride=2, padding=1)
    np.testing.assert_allclose(
        y.numpy(),
        np.array([[[[1.0, 3.0, 0.0], [9.0, 17.0, 8.0], [0.0, 14.0, 16.0]]]])
    )


def test_conv2d_kernel_too_large_raises():
    x = Tensor(np.zeros((1, 1, 2, 2), dtype=np.float32))
    w = Tensor(np.zeros((1, 1, 5, 5), dtype=np.float32))
    with pytest.raises(Exception):
        F.conv2d(x, w)
