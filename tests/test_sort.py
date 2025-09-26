# Copyright (c) 2025 Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

import math

import pytest

import minitensor as mt


def test_sort_default_last_dim():
    x = mt.tensor([[3.0, 1.0, 2.0], [0.5, -1.0, 0.0]], dtype="float32")
    values, indices = x.sort()

    assert values.numpy().tolist() == [[1.0, 2.0, 3.0], [-1.0, 0.0, 0.5]]
    assert indices.numpy().tolist() == [[1, 2, 0], [1, 2, 0]]


def test_sort_along_dim_zero_int_tensor():
    x = mt.tensor([[3, 1], [2, 0], [5, -2]], dtype="int32")
    values, indices = x.sort(dim=0)

    assert values.numpy().tolist() == [[2, -2], [3, 0], [5, 1]]
    assert indices.numpy().tolist() == [[1, 2], [0, 1], [2, 0]]


def test_sort_descending_with_nan():
    x = mt.tensor([float("nan"), 3.0, 1.0], dtype="float32")
    values, indices = x.sort(descending=True)

    # NaNs should be placed first to align with PyTorch semantics
    assert math.isnan(values.numpy()[0])
    assert indices.numpy().tolist()[1:] == [1, 2]


def test_sort_stable_keeps_duplicate_order():
    x = mt.tensor([1.0, 2.0, 1.0, 1.0], dtype="float64")
    values, indices = x.sort(stable=True)

    assert values.numpy().tolist() == [1.0, 1.0, 1.0, 2.0]
    assert indices.numpy().tolist() == [0, 2, 3, 1]


def test_argsort_matches_sort_indices():
    x = mt.tensor([[3.0, 1.0], [2.5, -4.0]], dtype="float32")
    values, indices = x.sort(dim=1, descending=True)
    argsorted = x.argsort(dim=1, descending=True)

    assert values.numpy().tolist() == [[3.0, 1.0], [2.5, -4.0]]
    assert indices.numpy().tolist() == argsorted.numpy().tolist()


def test_sort_scalar_returns_zero_index():
    x = mt.tensor(5.0)
    values, indices = x.sort()

    assert pytest.approx(values.item()) == 5.0
    assert indices.item() == 0


def test_sort_scalar_invalid_dim_raises():
    x = mt.tensor(1.0)
    with pytest.raises(IndexError):
        x.sort(dim=1)


def test_top_level_sort_and_argsort_dispatch():
    x = mt.tensor([True, False, True], dtype="bool")
    values, indices = mt.sort(x, descending=True)
    args = mt.argsort(x, descending=True)

    assert values.numpy().tolist() == [True, True, False]
    assert indices.numpy().tolist() == args.numpy().tolist()
