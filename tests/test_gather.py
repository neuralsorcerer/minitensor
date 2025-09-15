# Copyright (c) 2025 Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import pytest

import minitensor.functional as F
from minitensor import Tensor, gather


def test_gather_basic():
    t = Tensor([[1, 2, 3], [4, 5, 6]])
    idx = Tensor([[0, 1, 1], [2, 0, 0]], dtype="int64")
    g = t.gather(1, idx)
    expected = np.array([[1, 2, 2], [6, 4, 4]], dtype=np.float32)
    assert np.array_equal(g.numpy(), expected)


def test_gather_error():
    t = Tensor([1, 2, 3])
    idx = Tensor([3], dtype="int64")
    with pytest.raises(Exception):
        t.gather(0, idx)


def test_functional_top_level():
    t = Tensor([10, 20, 30])
    idx = Tensor([2, 0, 1], dtype="int64")
    g1 = F.gather(t, 0, idx)
    g2 = gather(t, 0, idx)
    expected = np.array([30.0, 10.0, 20.0], dtype=np.float32)
    assert np.array_equal(g1.numpy(), expected)
    assert np.array_equal(g2.numpy(), expected)
