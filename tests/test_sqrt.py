# Copyright (c) 2025 Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np

from minitensor.tensor import Tensor


def test_sqrt_forward_backward():
    x = Tensor([4.0, 9.0], dtype="float32", requires_grad=True)
    y = x.sqrt()
    assert np.allclose(y.numpy(), np.array([2.0, 3.0], dtype=np.float32))


def test_sqrt_negative_nan():
    x = Tensor([-1.0], dtype="float32")
    y = x.sqrt()
    assert np.isnan(y.numpy()).all()
