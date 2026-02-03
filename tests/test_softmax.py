# Copyright (c) 2025 Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import pytest

import minitensor as mt
from minitensor import functional as F


def test_functional_softmax_matches_tensor():
    x_np = np.array([[1.0, 2.0], [3.0, 4.0]])
    x = mt.Tensor(x_np.tolist())
    result = F.softmax(x)
    expected = np.exp(x_np - x_np.max(axis=1, keepdims=True))
    expected = expected / expected.sum(axis=1, keepdims=True)
    assert np.allclose(result.numpy(), expected)


def test_functional_softmax_dim():
    x_np = np.array([[1.0, 2.0], [3.0, 4.0]])
    x = mt.Tensor(x_np.tolist())
    result = F.softmax(x, dim=0)
    expected = np.exp(x_np - x_np.max(axis=0, keepdims=True))
    expected = expected / expected.sum(axis=0, keepdims=True)
    assert np.allclose(result.numpy(), expected)


def test_softmax_extreme_values():
    x_np = np.array([[1e9, -1e9], [-1e9, 1e9]], dtype=np.float32)
    x = mt.Tensor(x_np.tolist())
    result = F.softmax(x)
    shifted = x_np - x_np.max(axis=1, keepdims=True)
    expected = np.exp(shifted)
    expected = expected / expected.sum(axis=1, keepdims=True)
    assert np.allclose(result.numpy(), expected)
    assert np.allclose(result.numpy().sum(axis=1), np.array([1.0, 1.0]))


def test_softmax_scalar_returns_one():
    x = mt.Tensor(3.5)
    result = F.softmax(x)
    assert result.shape == ()
    assert result.item() == pytest.approx(1.0)


def test_softmax_empty_dim_returns_empty():
    x = mt.Tensor.zeros((2, 0, 3), dtype="float32")
    result = F.softmax(x, dim=1)
    assert result.shape == (2, 0, 3)
    assert result.numel() == 0


def test_masked_softmax_matches_expected():
    x_np = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    mask_np = np.array([[True, False], [False, True]])
    x = mt.Tensor(x_np.tolist())
    mask = mt.Tensor(mask_np.tolist(), dtype="bool")
    result = F.masked_softmax(x, mask, dim=1)
    expected = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
    assert np.allclose(result.numpy(), expected)


def test_masked_softmax_broadcasts_mask():
    x_np = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    mask_np = np.array([[True], [False]])
    x = mt.Tensor(x_np.tolist())
    mask = mt.Tensor(mask_np.tolist(), dtype="bool")
    result = F.masked_softmax(x, mask, dim=1)
    expected = np.array([[0.0, 0.0], [0.26894143, 0.7310586]], dtype=np.float32)
    assert np.allclose(result.numpy(), expected, atol=1e-6)


def test_masked_log_softmax_all_masked_is_neg_inf():
    x_np = np.array([1.0, 2.0], dtype=np.float32)
    mask_np = np.array([True, True])
    x = mt.Tensor(x_np.tolist())
    mask = mt.Tensor(mask_np.tolist(), dtype="bool")
    result = F.masked_log_softmax(x, mask, dim=0)
    out = result.numpy()
    assert np.all(np.isneginf(out))


def test_masked_softmax_all_neg_inf_unmasked_is_zero():
    x_np = np.array([-np.inf, -np.inf], dtype=np.float32)
    mask_np = np.array([False, False])
    x = mt.Tensor(x_np.tolist())
    mask = mt.Tensor(mask_np.tolist(), dtype="bool")
    result = F.masked_softmax(x, mask, dim=0)
    assert np.allclose(result.numpy(), np.array([0.0, 0.0], dtype=np.float32))
