# Copyright (c) 2025 Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import pytest

import minitensor as mt
from minitensor import functional as F


def test_split_method_functional_top_level_int():
    t = mt.arange(0, 10)
    method_parts = t.split(4)
    func_parts = F.split(t, 4)
    top_parts = mt.split(t, 4)
    expected = [np.arange(0, 4), np.arange(4, 8), np.arange(8, 10)]

    assert len(method_parts) == len(func_parts) == len(top_parts) == len(expected)
    for m, f, tp, n in zip(method_parts, func_parts, top_parts, expected):
        np.testing.assert_array_equal(m.numpy(), n)
        np.testing.assert_array_equal(f.numpy(), n)
        np.testing.assert_array_equal(tp.numpy(), n)


def test_split_negative_dim():
    t = mt.arange(0, 9).reshape((3, 3))
    parts = mt.split(t, 2, dim=-1)
    np_parts = np.array_split(t.numpy(), 2, axis=-1)

    assert len(parts) == len(np_parts)
    for p, n in zip(parts, np_parts):
        np.testing.assert_array_equal(p.numpy(), n)


def test_split_with_explicit_sections_and_dim():
    t = mt.arange(0, 10).reshape((2, 5))
    method_parts = t.split([2, 3], dim=1)
    func_parts = F.split(t, [2, 3], dim=1)
    top_parts = mt.split(t, [2, 3], dim=1)
    np_parts = np.array_split(t.numpy(), [2], axis=1)

    assert len(method_parts) == len(func_parts) == len(top_parts) == 2
    for m, f, tp, n in zip(method_parts, func_parts, top_parts, np_parts):
        np.testing.assert_array_equal(m.numpy(), n)
        np.testing.assert_array_equal(f.numpy(), n)
        np.testing.assert_array_equal(tp.numpy(), n)


def test_split_size_mismatch_raises():
    t = mt.arange(0, 6)
    with pytest.raises(ValueError):
        t.split([2, 5])
