# Copyright (c) 2025 Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import pytest

from minitensor import Tensor


def test_expand_basic():
    t = Tensor([1, 2, 3]).unsqueeze(0)
    e = t.expand(4, 3)
    np.testing.assert_array_equal(
        e.numpy(), np.array([[1, 2, 3]] * 4, dtype=np.float32)
    )


def test_expand_negative_one():
    t = Tensor([[1], [2]])
    e = t.expand(-1, 3)
    np.testing.assert_array_equal(
        e.numpy(), np.array([[1, 1, 1], [2, 2, 2]], dtype=np.float32)
    )


def test_expand_invalid():
    t = Tensor([1, 2, 3])
    with pytest.raises(Exception):
        t.expand(2, 3)
