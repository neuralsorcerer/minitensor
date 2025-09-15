# Copyright (c) 2025 Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for tensor flip operation."""

import numpy as np
import pytest

import minitensor as mt
from minitensor import functional as F


def test_flip_1d():
    t = mt.arange(0, 5)
    r = t.flip(0)
    np_r = np.flip(np.arange(0, 5), 0)
    np.testing.assert_array_equal(r.numpy(), np_r)


def test_flip_multi_dim():
    t = mt.arange(0, 12).reshape((3, 4))
    r = t.flip((0, 1))
    np_r = np.flip(t.numpy(), axis=(0, 1))
    np.testing.assert_array_equal(r.numpy(), np_r)


def test_flip_negative_dims():
    t = mt.arange(0, 6).reshape((2, 3))
    r = t.flip(-1)
    np_r = np.flip(t.numpy(), axis=-1)
    np.testing.assert_array_equal(r.numpy(), np_r)


def test_flip_duplicate_dims_error():
    t = mt.arange(0, 5)
    with pytest.raises(ValueError):
        t.flip((0, 0))


def test_functional_and_top_level_flip():
    t = mt.arange(0, 5)
    r_func = F.flip(t, 0)
    r_top = mt.flip(t, 0)
    np.testing.assert_array_equal(r_func.numpy(), r_top.numpy())
