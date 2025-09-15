# Copyright (c) 2025 Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import pytest

import minitensor as mt
import minitensor.functional as F


def test_narrow_1d():
    t = mt.arange(0, 10)
    r = t.narrow(0, 2, 5)
    assert np.array_equal(r.numpy(), np.arange(2, 7))


def test_narrow_negative_dim():
    t = mt.arange(0, 12).reshape((3, 4))
    r = t.narrow(-1, 1, 2)
    expected = np.array([[1, 2], [5, 6], [9, 10]], dtype=np.float32)
    assert np.array_equal(r.numpy(), expected)


def test_narrow_out_of_bounds():
    t = mt.arange(0, 5)
    with pytest.raises((ValueError, IndexError)):
        t.narrow(0, 3, 3)


def test_functional_and_top_level_narrow():
    t = mt.arange(0, 6)
    r_method = t.narrow(0, 2, 2)
    r_func = F.narrow(t, 0, 2, 2)
    r_top = mt.narrow(t, 0, 2, 2)
    expected = np.arange(2, 4)
    assert np.array_equal(r_method.numpy(), expected)
    assert np.array_equal(r_func.numpy(), expected)
    assert np.array_equal(r_top.numpy(), expected)
