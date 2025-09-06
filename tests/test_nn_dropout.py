# Copyright (c) 2025 Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from minitensor.tensor import Tensor
from minitensor.nn import Dropout, Dropout2d


def test_dropout_noop_when_p_zero():
    layer = Dropout(0.0)
    x = Tensor.ones([2, 3])
    y = layer.forward(x._tensor)
    assert np.allclose(y.numpy(), x.numpy())


def test_dropout2d_all_zero_when_p_one():
    layer = Dropout2d(1.0)
    x = Tensor.ones([1, 2, 4, 4])
    y = layer.forward(x._tensor)
    assert np.allclose(y.numpy(), 0.0)
