# Copyright (c) 2025 Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from minitensor.tensor import Tensor
from minitensor import functional as F


def test_dropout_extreme_probabilities():
    x = Tensor([[1.0, 2.0], [3.0, 4.0]], dtype="float32")
    out_same = F.dropout(x, p=0.0, training=True)
    assert np.array_equal(out_same.numpy(), x.numpy())
    out_zero = F.dropout(x, p=1.0, training=True)
    assert np.allclose(out_zero.numpy(), 0.0)
    out_eval = F.dropout(x, p=0.5, training=False)
    assert np.array_equal(out_eval.numpy(), x.numpy())
