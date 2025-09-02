# Copyright (c) 2025 Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np

import minitensor as mt
from minitensor import functional as F


def test_functional_softmax_matches_tensor():
    x_np = np.array([[1.0, 2.0], [3.0, 4.0]])
    x = mt.Tensor(x_np.tolist())
    result = F.softmax(x)
    expected = np.exp(x_np - x_np.max(axis=1, keepdims=True))
    expected = expected / expected.sum(axis=1, keepdims=True)
    assert np.allclose(result.numpy(), expected)


def test_functional_softmax_dim():
    x_np = np.array([[1.0, 2.0], [3.0, 4.0]])
    x = mt.Tensor(x_np.tolist())
    result = F.softmax(x, dim=0)
    expected = np.exp(x_np - x_np.max(axis=0, keepdims=True))
    expected = expected / expected.sum(axis=0, keepdims=True)
    assert np.allclose(result.numpy(), expected)
