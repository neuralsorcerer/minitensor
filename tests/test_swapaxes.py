# Copyright (c) 2025 Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np

from minitensor import functional as F
from minitensor import swapaxes, swapdims
from minitensor.tensor import Tensor


def test_tensor_swapaxes():
    t = Tensor.arange(0, 24).reshape([2, 3, 4])
    s = t.swapaxes(0, 2)
    assert s.shape == (4, 3, 2)
    assert np.array_equal(s.numpy(), t.transpose(0, 2).numpy())


def test_functional_swapaxes():
    t = Tensor.arange(0, 24).reshape([2, 3, 4])
    s = F.swapaxes(t, 1, 2)
    assert s.shape == (2, 4, 3)
    assert np.array_equal(s.numpy(), t.swapaxes(1, 2).numpy())


def test_top_level_swapaxes():
    t = Tensor.arange(0, 24).reshape([2, 3, 4])
    s = swapaxes(t, 0, 1)
    assert s.shape == (3, 2, 4)
    assert np.array_equal(s.numpy(), t.swapaxes(0, 1).numpy())


def test_swapdims_alias():
    t = Tensor.arange(0, 24).reshape([2, 3, 4])
    s0 = t.swapdims(0, 1)
    assert np.array_equal(s0.numpy(), t.swapaxes(0, 1).numpy())
    s1 = F.swapdims(t, 0, 2)
    assert np.array_equal(s1.numpy(), t.swapaxes(0, 2).numpy())
    s2 = swapdims(t, 1, 2)
    assert np.array_equal(s2.numpy(), t.swapaxes(1, 2).numpy())
