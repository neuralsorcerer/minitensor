# Copyright (c) 2025 Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np

from minitensor import nn
from minitensor.tensor import Tensor


def test_relu_forward():
    x = Tensor.from_numpy(np.array([-1.0, 0.0, 1.0], dtype=np.float32))
    layer = nn.ReLU()
    y = layer.forward(x._tensor)
    np.testing.assert_allclose(y.numpy(), [0.0, 0.0, 1.0])


def test_elu_forward():
    x = Tensor.from_numpy(np.array([-1.0, 0.0, 1.0], dtype=np.float32))
    layer = nn.ELU(alpha=0.5)
    y = layer.forward(x._tensor)
    expected = [0.5 * (np.exp(-1.0) - 1.0), 0.0, 1.0]
    np.testing.assert_allclose(y.numpy(), expected, rtol=1e-5)


def test_gelu_forward():
    x = Tensor.from_numpy(np.array([-1.0, 0.0, 1.0], dtype=np.float32))
    layer = nn.GELU()
    y = layer.forward(x._tensor)
    arr = x.numpy()
    expected = (
        0.5 * arr * (1.0 + np.tanh(np.sqrt(2 / np.pi) * (arr + 0.044715 * arr**3)))
    )
    np.testing.assert_allclose(y.numpy(), expected, rtol=1e-5, atol=1e-5)
