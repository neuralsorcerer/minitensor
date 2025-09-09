# Copyright (c) 2025 Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import pytest

from minitensor.tensor import Tensor


def test_flatten_and_ravel():
    t = Tensor.ones([2, 3, 4])
    f = t.flatten()
    r = t.ravel()
    assert f.shape == (24,)
    assert r.shape == (24,)
    assert np.array_equal(f.numpy(), r.numpy())


def test_flatten_range_and_error():
    t = Tensor.arange(0, 24).reshape([2, 3, 4])
    f = t.flatten(1, -1)
    assert f.shape == (2, 12)
    with pytest.raises(ValueError):
        t.flatten(2, 0)
