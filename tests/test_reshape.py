# Copyright (c) 2025 Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np

from minitensor import functional as F
from minitensor import reshape
from minitensor.tensor import Tensor


def test_functional_reshape():
    t = Tensor.arange(0, 24).reshape([2, 3, 4])
    r = F.reshape(t, (4, 6))
    assert r.shape == (4, 6)
    assert np.array_equal(r.numpy(), t.reshape(4, 6).numpy())


def test_top_level_reshape():
    t = Tensor.arange(0, 24)
    r = reshape(t, (2, 3, 4))
    assert r.shape == (2, 3, 4)
    assert np.array_equal(r.numpy(), t.reshape(2, 3, 4).numpy())
