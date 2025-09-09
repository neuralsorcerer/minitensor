# Copyright (c) 2025 Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import minitensor as mt

def test_eq_with_scalar():
    a = mt.Tensor([1.0, 2.0, 3.0])
    res = (a == 2.0).numpy()
    expected = np.array([False, True, False])
    np.testing.assert_array_equal(res, expected)

def test_lt_with_list():
    a = mt.Tensor([1, 2, 3])
    res = a < [2, 2, 4]
    expected = np.array([True, False, True])
    np.testing.assert_array_equal(res.numpy(), expected)

def test_gt_with_numpy_array():
    a = mt.Tensor([1.0, 2.0, 3.0])
    b = np.array([1.5, 1.5, 4.0])
    res = a.gt(b)
    expected = np.array([False, True, False])
    np.testing.assert_array_equal(res.numpy(), expected)

def test_negation_uses_backend():
    a = mt.Tensor([1.0, -2.0, 3.0])
    res = (-a).numpy()
    expected = np.array([-1.0, 2.0, -3.0])
    np.testing.assert_array_equal(res, expected)
