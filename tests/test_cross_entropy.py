# Copyright (c) 2025 Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np

import minitensor as mt
from minitensor import functional as F


def test_cross_entropy_matches_numpy():
    x_np = np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]], dtype=np.float32)
    target_np = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]], dtype=np.float32)
    x = mt.Tensor(x_np.tolist())
    target = mt.Tensor(target_np.tolist())

    loss = F.cross_entropy(x, target, reduction="mean")

    # Manual cross entropy averaged over batch
    shifted = x_np - x_np.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    softmax = exp / exp.sum(axis=1, keepdims=True)
    expected = -(target_np * np.log(softmax)).sum(axis=1).mean()

    assert np.allclose(loss.numpy(), expected)


def test_cross_entropy_dim_argument():
    x_np = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]], dtype=np.float32)
    target_np = np.array([2, 0], dtype=np.int64)
    x = mt.Tensor(x_np.tolist())
    target = mt.Tensor(target_np.tolist())

    loss = F.cross_entropy(x, target, reduction="mean", dim=0)

    shifted = x_np - x_np.max(axis=0, keepdims=True)
    exp = np.exp(shifted)
    softmax = exp / exp.sum(axis=0, keepdims=True)
    expected = -np.log(softmax[target_np, np.arange(softmax.shape[1])]).mean()

    assert np.allclose(loss.numpy(), expected)


def test_cross_entropy_negative_dim():
    x_np = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]], dtype=np.float32)
    target_np = np.array([2, 0], dtype=np.int64)
    x = mt.Tensor(x_np.tolist())
    target = mt.Tensor(target_np.tolist())

    loss = F.cross_entropy(x, target, reduction="mean", dim=-2)

    shifted = x_np - x_np.max(axis=0, keepdims=True)
    exp = np.exp(shifted)
    softmax = exp / exp.sum(axis=0, keepdims=True)
    expected = -np.log(softmax[target_np, np.arange(softmax.shape[1])]).mean()

    assert np.allclose(loss.numpy(), expected)


def test_cross_entropy_invalid_dim_raises():
    x_np = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]], dtype=np.float32)
    target_np = np.array([2, 0], dtype=np.int64)
    x = mt.Tensor(x_np.tolist())
    target = mt.Tensor(target_np.tolist())

    try:
        F.cross_entropy(x, target, dim=2)
    except IndexError:
        pass
    else:
        assert False, "Expected IndexError for invalid dim"


def test_cross_entropy_no_reduction_shape_and_values():
    x_np = np.array(
        [
            [[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]],
            [[2.0, 1.0, 0.0, -1.0], [2.0, 1.0, 0.0, -1.0], [2.0, 1.0, 0.0, -1.0]],
        ],
        dtype=np.float32,
    )
    target_np = np.array([[0, 1, 2, 1], [2, 0, 1, 2]], dtype=np.int64)
    x = mt.Tensor(x_np.tolist())
    target = mt.Tensor(target_np.tolist())

    loss = F.cross_entropy(x, target, reduction="none", dim=1)

    shifted = x_np - x_np.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    softmax = exp / exp.sum(axis=1, keepdims=True)
    gathered = np.take_along_axis(softmax, target_np[:, None, :], axis=1).squeeze(1)
    expected = -np.log(gathered)

    assert loss.shape == expected.shape
    assert np.allclose(loss.numpy(), expected)
