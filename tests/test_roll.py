# Copyright (c) 2025 Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for tensor roll operation."""

import numpy as np
import pytest

import minitensor as mt
from minitensor import functional as F


def test_roll_1d():
    t = mt.arange(0, 5)
    r = t.roll(2)
    np_r = np.roll(np.arange(0, 5), 2)
    np.testing.assert_array_equal(r.numpy(), np_r)


def test_roll_multi_dim():
    t = mt.arange(0, 12).reshape((3, 4))
    r = t.roll((1, 2), dims=(0, 1))
    np_r = np.roll(t.numpy(), shift=(1, 2), axis=(0, 1))
    np.testing.assert_array_equal(r.numpy(), np_r)


def test_roll_negative_shift():
    t = mt.arange(0, 5)
    r = t.roll(-1)
    np_r = np.roll(np.arange(0, 5), -1)
    np.testing.assert_array_equal(r.numpy(), np_r)


def test_roll_mismatch_raises():
    t = mt.arange(0, 5)
    with pytest.raises(ValueError):
        t.roll((1, 2), dims=(0,))


def test_functional_and_top_level_roll():
    t = mt.arange(0, 5)
    r_func = F.roll(t, 1)
    r_top = mt.roll(t, 1)
    np.testing.assert_array_equal(r_func.numpy(), r_top.numpy())
