# Copyright (c) 2025 Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np

from minitensor.tensor import Tensor


def test_argmax_dim():
    x = Tensor([[1.0, 5.0, 3.0], [4.0, 2.0, 6.0]], dtype="float32")
    result = x.argmax(dim=1)
    assert np.array_equal(result.numpy(), np.array([1, 2], dtype=np.int64))


def test_argmin_dim_keepdim():
    x = Tensor([[1.0, 5.0, 3.0], [4.0, 2.0, 6.0]], dtype="float32")
    result = x.argmin(dim=1, keepdim=True)
    assert result.shape == (2, 1)
    assert np.array_equal(result.numpy(), np.array([[0], [1]], dtype=np.int64))
