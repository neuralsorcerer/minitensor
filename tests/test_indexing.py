# Copyright (c) 2025 Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import pytest
import minitensor as mt


def test_all_any():
    t = mt.Tensor([[1.0, 0.0], [2.0, 3.0]])
    assert t.any().tolist() == [True]
    assert t.all().tolist() == [False]
    b = mt.Tensor([[True, False], [True, True]], dtype="bool")
    res = b.all(dim=1)
    assert res.tolist() == [False, True]


def test_indexing_and_assignment():
    t = mt.Tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]])
    assert t[0, 1].tolist() == [1.0]
    t[0, 1] = 10.0
    assert t[0, 1].tolist() == [10.0]
    col = t[:, 1]
    assert col.tolist() == [10.0, 4.0]


def test_negative_indexing_and_bounds():
    t = mt.Tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]])
    assert t[1, -1].tolist() == [5.0]
    with pytest.raises(IndexError):
        _ = t[2, 0]
    with pytest.raises(IndexError):
        _ = t[0, -4]


def test_slice_with_start():
    t = mt.arange(10)
    sliced = t[5:]
    assert sliced.tolist() == [5.0, 6.0, 7.0, 8.0, 9.0]


def test_slice_out_of_range_empty():
    t = mt.arange(5)
    assert t[10:].tolist() == []


def test_reverse_slice_error():
    t = mt.arange(10)
    with pytest.raises(IndexError):
        _ = t[::-1]


def test_multi_dim_slice():
    t = mt.Tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
    sub = t[1:, :2]
    np.testing.assert_allclose(
        sub.numpy(), np.array([[3.0, 4.0], [6.0, 7.0]], dtype=np.float32)
    )
