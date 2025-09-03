# Copyright (c) 2025 Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import minitensor as mt


def test_max_min_with_indices():
    x = mt.Tensor([[1.0, 5.0, 3.0], [4.0, 2.0, 6.0]], dtype="float32")
    max_vals, max_idx = x.max(dim=1)
    assert np.array_equal(max_vals.numpy(), np.array([5.0, 6.0], dtype=np.float32))
    assert np.array_equal(max_idx.numpy(), np.array([1, 2], dtype=np.int64))

    min_vals, min_idx = x.min(dim=1)
    assert np.array_equal(min_vals.numpy(), np.array([1.0, 2.0], dtype=np.float32))
    assert np.array_equal(min_idx.numpy(), np.array([0, 1], dtype=np.int64))