# Copyright (c) 2025 Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import pytest

from minitensor import Tensor
from minitensor import repeat as top_repeat
from minitensor.functional import repeat as F_repeat


class IndexLike:
    def __init__(self, value: int):
        self._value = value

    def __index__(self) -> int:  # pragma: no cover - simple accessor
        return self._value


def test_tensor_repeat_basic():
    t = Tensor([1, 2])
    r = t.repeat(2)
    assert r.numpy().tolist() == [1, 2, 1, 2]


def test_tensor_repeat_multi_dim():
    t = Tensor([[1, 2], [3, 4]])
    r = t.repeat(2, 3)
    expected = np.tile(np.array([[1, 2], [3, 4]]), (2, 3))
    assert r.shape == expected.shape
    assert np.array_equal(r.numpy(), expected)


def test_repeat_functional_and_top_level():
    t = Tensor([1, 2])
    r_func = F_repeat(t, 2, 3, 1)
    r_top = top_repeat(t, 2, 3, 1)
    expected = np.tile(np.array([1, 2]), (2, 3, 1))
    assert np.array_equal(r_func.numpy(), expected)
    assert np.array_equal(r_top.numpy(), expected)


def test_repeat_errors():
    t = Tensor([[1, 2], [3, 4]])
    with pytest.raises(ValueError):
        t.repeat(2)
    with pytest.raises(ValueError):
        t.repeat(2, -1)


def test_repeat_zero():
    t = Tensor([[1, 2], [3, 4]])
    r = t.repeat(0, 2)
    assert r.shape == (0, 4)
    assert r.numel() == 0


def test_repeat_accepts_index_like_scalars():
    t = Tensor([[1, 2], [3, 4]])
    repeated = t.repeat(IndexLike(1), IndexLike(2))
    assert repeated.shape == (2, 4)

    via_sequence = t.repeat([IndexLike(1), IndexLike(2)])
    assert via_sequence.shape == (2, 4)

    numpy_repeats = (np.int64(2), np.int64(1))
    repeated_numpy = t.repeat(*numpy_repeats)
    assert repeated_numpy.shape == (4, 2)


def test_repeat_rejects_non_integer_values():
    t = Tensor([1, 2])
    with pytest.raises(TypeError):
        t.repeat(2.5)

    with pytest.raises(TypeError):
        t.repeat([1, 2.2])
