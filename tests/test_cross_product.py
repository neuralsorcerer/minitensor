# Copyright (c) 2025 Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np

import minitensor as mt


def test_cross_product_matches_numpy():
    a = mt.Tensor([1.0, 2.0, 3.0])
    b = mt.Tensor([4.0, 5.0, 6.0])
    c = mt.numpy_compat.cross(a._tensor, b._tensor)
    expected = np.cross(np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0]))
    np.testing.assert_allclose(c.numpy(), expected)
