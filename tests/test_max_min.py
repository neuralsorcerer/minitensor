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


def test_max_min_empty_tensor_with_dim():
    t = mt.Tensor(np.empty((0, 3), dtype=np.float32))
    max_vals, max_idx = t.max(dim=0)
    assert np.isneginf(max_vals.numpy()).all()
    assert np.array_equal(max_idx.numpy(), np.zeros(3, dtype=np.int64))

    min_vals, min_idx = t.min(dim=0)
    assert np.isposinf(min_vals.numpy()).all()
    assert np.array_equal(min_idx.numpy(), np.zeros(3, dtype=np.int64))


def test_max_min_empty_tensor_with_dim_keepdim():
    t = mt.Tensor(np.empty((0, 3), dtype=np.float32))
    max_vals, max_idx = t.max(dim=0, keepdim=True)
    assert max_vals.shape == (1, 3)
    assert np.isneginf(max_vals.numpy()).all()
    assert np.array_equal(max_idx.numpy(), np.zeros((1, 3), dtype=np.int64))

    min_vals, min_idx = t.min(dim=0, keepdim=True)
    assert min_vals.shape == (1, 3)
    assert np.isposinf(min_vals.numpy()).all()
    assert np.array_equal(min_idx.numpy(), np.zeros((1, 3), dtype=np.int64))


def test_max_min_all_nan_with_dim_returns_extremes():
    t = mt.Tensor(np.array([[np.nan, np.nan], [np.nan, np.nan]], dtype=np.float32))
    max_vals, max_idx = t.max(dim=1)
    assert np.isneginf(max_vals.numpy()).all()
    assert np.array_equal(max_idx.numpy(), np.zeros(2, dtype=np.int64))

    min_vals, min_idx = t.min(dim=1)
    assert np.isposinf(min_vals.numpy()).all()
    assert np.array_equal(min_idx.numpy(), np.zeros(2, dtype=np.int64))


def test_max_min_all_nan_with_dim_keepdim():
    t = mt.Tensor(np.array([[np.nan, np.nan], [np.nan, np.nan]], dtype=np.float32))
    max_vals, max_idx = t.max(dim=1, keepdim=True)
    assert max_vals.shape == (2, 1)
    assert np.isneginf(max_vals.numpy()).all()
    assert np.array_equal(max_idx.numpy(), np.zeros((2, 1), dtype=np.int64))

    min_vals, min_idx = t.min(dim=1, keepdim=True)
    assert min_vals.shape == (2, 1)
    assert np.isposinf(min_vals.numpy()).all()
    assert np.array_equal(min_idx.numpy(), np.zeros((2, 1), dtype=np.int64))


def test_max_min_empty_int_tensor_with_dim():
    t = mt.Tensor(np.empty((0, 2), dtype=np.int32), dtype="int32")
    max_vals, max_idx = t.max(dim=0)
    assert max_vals.numpy().tolist() == [np.iinfo(np.int32).min] * 2
    assert np.array_equal(max_idx.numpy(), np.zeros(2, dtype=np.int64))

    min_vals, min_idx = t.min(dim=0)
    assert min_vals.numpy().tolist() == [np.iinfo(np.int32).max] * 2
    assert np.array_equal(min_idx.numpy(), np.zeros(2, dtype=np.int64))
