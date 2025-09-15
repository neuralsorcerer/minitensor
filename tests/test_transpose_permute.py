# Copyright (c) 2025 Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np

from minitensor import functional as F
from minitensor import permute, transpose
from minitensor.tensor import Tensor


def test_functional_transpose():
    t = Tensor.arange(0, 24).reshape([2, 3, 4])
    tr = F.transpose(t, 0, 1)
    assert tr.shape == (3, 2, 4)
    assert np.array_equal(tr.numpy(), t.transpose(0, 1).numpy())


def test_top_level_transpose():
    t = Tensor.arange(0, 24).reshape([2, 3, 4])
    tr = transpose(t, 1, 2)
    assert tr.shape == (2, 4, 3)
    assert np.array_equal(tr.numpy(), t.transpose(1, 2).numpy())


def test_functional_permute():
    t = Tensor.arange(0, 24).reshape([2, 3, 4])
    p = F.permute(t, (1, 2, 0))
    assert p.shape == (3, 4, 2)
    assert np.array_equal(p.numpy(), t.permute(1, 2, 0).numpy())


def test_top_level_permute():
    t = Tensor.arange(0, 24).reshape([2, 3, 4])
    p = permute(t, (2, 0, 1))
    assert p.shape == (4, 2, 3)
    assert np.array_equal(p.numpy(), t.permute(2, 0, 1).numpy())
