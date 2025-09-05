# Copyright (c) 2025 Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import pytest

import minitensor as mt


def test_eq_broadcasting():
    a = mt.Tensor([[1.0, 2.0], [3.0, 4.0]])
    b = mt.Tensor([[1.0, 4.0]])
    result = a.eq(b)
    expected = np.array([[True, False], [False, True]])
    np.testing.assert_array_equal(result.numpy(), expected)


def test_lt_bool_error():
    a = mt.Tensor([True, False], dtype="bool")
    b = mt.Tensor([False, True], dtype="bool")
    with pytest.raises(ValueError):
        a.lt(b)


def test_gt_incompatible_shapes_error():
    a = mt.Tensor([1.0, 2.0, 3.0])
    b = mt.Tensor([1.0, 2.0])
    with pytest.raises(ValueError):
        a.gt(b)


def test_eq_type_mismatch_error():
    a = mt.Tensor([1.0, 2.0], dtype="float32")
    b = mt.Tensor([1, 2], dtype="int32")
    with pytest.raises(TypeError):
        a.eq(b)
