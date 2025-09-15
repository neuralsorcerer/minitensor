# Copyright (c) 2025 Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import pytest

from minitensor import Tensor
from minitensor import repeat as top_repeat
from minitensor.functional import repeat as F_repeat


def test_tensor_repeat_basic():
    t = Tensor([1, 2])
    r = t.repeat(2)
    assert r.numpy().tolist() == [1, 2, 1, 2]


def test_tensor_repeat_multi_dim():
    t = Tensor([[1, 2], [3, 4]])
    r = t.repeat(2, 3)
    expected = np.tile(np.array([[1, 2], [3, 4]]), (2, 3))
    assert r.shape == expected.shape
    assert np.array_equal(r.numpy(), expected)


def test_repeat_functional_and_top_level():
    t = Tensor([1, 2])
    r_func = F_repeat(t, 2, 3, 1)
    r_top = top_repeat(t, 2, 3, 1)
    expected = np.tile(np.array([1, 2]), (2, 3, 1))
    assert np.array_equal(r_func.numpy(), expected)
    assert np.array_equal(r_top.numpy(), expected)


def test_repeat_errors():
    t = Tensor([[1, 2], [3, 4]])
    with pytest.raises(ValueError):
        t.repeat(2)
    with pytest.raises(ValueError):
        t.repeat(2, -1)
