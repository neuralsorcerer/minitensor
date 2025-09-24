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


def test_eq_promotes_mixed_dtypes():
    a = mt.Tensor([1.0, 2.0], dtype="float32")
    b = mt.Tensor([1, 3], dtype="int32")
    result = a.eq(b)
    expected = np.array([True, False])
    np.testing.assert_array_equal(result.numpy(), expected)


def test_bool_numeric_comparisons():
    bools = mt.Tensor([True, False], dtype="bool")
    ints = mt.Tensor([1, 0], dtype="int32")
    floats = mt.Tensor([1.0, 0.5], dtype="float32")

    eq_res = bools.eq(ints)
    np.testing.assert_array_equal(eq_res.numpy(), np.array([True, True]))

    lt_res = bools.lt(floats)
    np.testing.assert_array_equal(lt_res.numpy(), np.array([False, True]))


def test_comparison_invalid_operand_type():
    a = mt.Tensor([1.0, 2.0])
    with pytest.raises(TypeError):
        a.eq("foo")


def test_nan_and_inf_comparisons():
    a = mt.Tensor([float("nan"), float("inf")])
    b = mt.Tensor([0.0, 1.0])
    eq_res = a.eq(a)
    lt_res = a.lt(b)
    gt_res = a.gt(b)
    assert not eq_res.numpy()[0]
    assert not lt_res.numpy()[0]
    assert gt_res.numpy()[1] and not lt_res.numpy()[1]
