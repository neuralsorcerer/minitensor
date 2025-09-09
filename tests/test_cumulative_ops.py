# Copyright (c) 2025 Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import pytest

from minitensor.tensor import Tensor


def test_cumsum_cumprod():
    t = Tensor.arange(1, 7, dtype="float32").reshape([2, 3])
    c0 = t.cumsum(0)
    np.testing.assert_array_equal(
        c0.numpy(), np.array([[1, 2, 3], [5, 7, 9]], dtype=np.float32)
    )
    c1 = t.cumsum(1)
    np.testing.assert_array_equal(
        c1.numpy(), np.array([[1, 3, 6], [4, 9, 15]], dtype=np.float32)
    )
    p0 = t.cumprod(0)
    np.testing.assert_array_equal(
        p0.numpy(), np.array([[1, 2, 3], [4, 10, 18]], dtype=np.float32)
    )
    p1 = t.cumprod(1)
    np.testing.assert_array_equal(
        p1.numpy(), np.array([[1, 2, 6], [4, 20, 120]], dtype=np.float32)
    )


def test_cumulative_invalid_axis():
    t = Tensor.arange(1, 7, dtype="float32").reshape([2, 3])
    with pytest.raises(IndexError):
        t.cumsum(2)
    with pytest.raises(IndexError):
        t.cumprod(2)
