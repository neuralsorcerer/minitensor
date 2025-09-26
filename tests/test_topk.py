# Copyright (c) 2025 Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest

np = pytest.importorskip("numpy")

import minitensor as mt


def test_topk_default_last_dim():
    x = mt.tensor([[1.0, 3.0, 2.0], [4.0, 0.0, 5.0]], requires_grad=True)
    values, indices = x.topk(2)

    np.testing.assert_allclose(
        values.numpy(),
        np.array([[3.0, 2.0], [5.0, 4.0]], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        indices.numpy(),
        np.array([[1, 2], [2, 0]], dtype=np.int64),
    )
    assert values.requires_grad is True
    assert indices.requires_grad is False
    assert indices.dtype == "int64"


def test_topk_smallest_unsorted():
    x = mt.tensor([1.0, -2.0, 3.5, 0.0], dtype="float32")
    values, indices = x.topk(2, largest=False, sorted=False)

    pairs = sorted(zip(indices.numpy().tolist(), values.numpy().tolist()))
    assert pairs == [(1, -2.0), (3, 0.0)]


def test_topk_with_dim_argument():
    x = mt.tensor([[1, 4, 2], [3, -1, 0]], dtype="float32")
    values, indices = x.topk(1, dim=1, largest=False)

    np.testing.assert_array_equal(
        values.numpy(), np.array([[1.0], [-1.0]], dtype=np.float32)
    )
    np.testing.assert_array_equal(indices.numpy(), np.array([[0], [1]], dtype=np.int64))


def test_topk_zero_k():
    x = mt.tensor([1.0, 2.0, 3.0])
    values, indices = x.topk(0)

    assert values.shape == (0,)
    assert indices.shape == (0,)
    assert values.numpy().size == 0
    assert indices.numpy().size == 0


def test_topk_out_of_range():
    x = mt.tensor([[1.0, 2.0, 3.0]])
    with pytest.raises(RuntimeError, match="selected index k out of range"):
        x.topk(4, dim=1)
