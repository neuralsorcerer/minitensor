# Copyright (c) 2025 Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import pytest
import minitensor as mt


def test_max_min_with_indices():
    x = mt.Tensor([[1.0, 5.0, 3.0], [4.0, 2.0, 6.0]], dtype="float32")
    max_vals, max_idx = x.max(dim=1)
    assert np.array_equal(max_vals.numpy(), np.array([5.0, 6.0], dtype=np.float32))
    assert np.array_equal(max_idx.numpy(), np.array([1, 2], dtype=np.int64))

    min_vals, min_idx = x.min(dim=1)
    assert np.array_equal(min_vals.numpy(), np.array([1.0, 2.0], dtype=np.float32))
    assert np.array_equal(min_idx.numpy(), np.array([0, 1], dtype=np.int64))


def test_max_min_with_nan_inf():
    t = mt.Tensor([np.nan, 1.0, np.inf, -np.inf], dtype="float32")
    max_val = t.max()
    min_val = t.min()
    assert np.isinf(max_val.numpy()) and max_val.numpy() > 0
    assert np.isinf(min_val.numpy()) and min_val.numpy() < 0


def test_max_min_all_equal():
    t = mt.Tensor([3.0, 3.0, 3.0], dtype="float32")
    assert t.max().numpy() == 3.0
    assert t.min().numpy() == 3.0


def test_max_min_empty_tensor_values():
    t = mt.Tensor(np.array([], dtype=np.float32))
    assert np.isneginf(t.max().numpy())
    assert np.isinf(t.min().numpy())


def test_max_min_all_nan_returns_extremes():
    t = mt.Tensor([np.nan, np.nan], dtype="float32")
    assert np.isneginf(t.max().numpy())
    assert np.isinf(t.min().numpy())
