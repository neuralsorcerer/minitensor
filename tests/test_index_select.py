# Copyright (c) 2025 Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import pytest

import minitensor as mt
from minitensor import functional as F


def test_tensor_index_select_dim0():
    t = mt.arange(0, 6).reshape((3, 2))
    out = t.index_select(0, [0, 2])
    expected = np.arange(0, 6, dtype=np.float32).reshape(3, 2)[[0, 2], :]
    np.testing.assert_allclose(out.numpy(), expected)


def test_tensor_index_select_neg_dim():
    t = mt.arange(0, 6).reshape((2, 3))
    out = t.index_select(-1, [2, 0])
    expected = np.arange(0, 6, dtype=np.float32).reshape(2, 3)[:, [2, 0]]
    np.testing.assert_allclose(out.numpy(), expected)


def test_functional_and_top_level_index_select():
    t = mt.arange(0, 6).reshape((3, 2))
    expected = np.arange(0, 6, dtype=np.float32).reshape(3, 2)[:, [1]]
    out_func = F.index_select(t, 1, [1])
    out_top = mt.index_select(t, 1, [1])
    np.testing.assert_allclose(out_func.numpy(), expected)
    np.testing.assert_allclose(out_top.numpy(), expected)


def test_index_select_out_of_range():
    t = mt.arange(0, 6).reshape((3, 2))
    with pytest.raises(IndexError):
        t.index_select(0, [3])
