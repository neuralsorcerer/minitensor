# Copyright (c) 2025 Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np

from minitensor.nn import BatchNorm1d, BatchNorm2d
from minitensor.tensor import Tensor


def test_batchnorm1d_train_eval_behavior():
    layer = BatchNorm1d(2)
    x = Tensor([[1.0, 2.0], [3.0, 4.0]], dtype="float32")
    out = layer.forward(x._tensor)
    out_np = out.numpy()
    assert np.allclose(out_np.mean(axis=0), 0.0, atol=1e-5)
    assert np.allclose(out_np.var(axis=0), 1.0, atol=1e-5)

    layer.eval()
    x2 = Tensor([[5.0, 6.0], [7.0, 8.0]], dtype="float32")
    out2 = layer.forward(x2._tensor)
    assert out2.shape == list(x2.shape)


def test_batchnorm2d_channel_normalization():
    layer = BatchNorm2d(3)
    x = Tensor(np.random.randn(2, 3, 4, 4).astype("float32"))
    out = layer.forward(x._tensor)
    out_np = out.numpy()
    channel_var = out_np.var(axis=(0, 2, 3))
    assert np.all(channel_var > 0)
