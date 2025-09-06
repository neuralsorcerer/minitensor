# Copyright (c) 2025 Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np

from minitensor import functional as F
from minitensor.tensor import Tensor


def test_batch_norm_training_updates_stats():
    x = Tensor([[1.0, 2.0], [3.0, 4.0]], dtype="float32")
    running_mean = Tensor([0.0, 0.0], dtype="float32")
    running_var = Tensor([0.0, 0.0], dtype="float32")
    out = F.batch_norm(x, running_mean, running_var, training=True)
    out_np = out.numpy()
    assert np.allclose(out_np.mean(axis=0), 0.0, atol=1e-5)
    assert np.allclose(out_np.var(axis=0, ddof=0), 1.0, atol=1e-5)
    assert not np.allclose(running_mean.numpy(), [0.0, 0.0])
    assert not np.allclose(running_var.numpy(), [0.0, 0.0])


def test_batch_norm_eval_uses_running_stats():
    x = Tensor([[1.0, 2.0], [3.0, 4.0]], dtype="float32")
    running_mean = Tensor([2.0, 3.0], dtype="float32")
    running_var = Tensor([1.0, 4.0], dtype="float32")
    out = F.batch_norm(x, running_mean, running_var, training=False)
    expected = (x.numpy() - running_mean.numpy()) / np.sqrt(running_var.numpy() + 1e-5)
    assert np.allclose(out.numpy(), expected, atol=1e-5)


def test_batch_norm_with_weight_and_bias():
    x = Tensor([[1.0, 2.0], [3.0, 4.0]], dtype="float32")
    running_mean = Tensor([0.0, 0.0], dtype="float32")
    running_var = Tensor([1.0, 1.0], dtype="float32")
    weight = Tensor([1.5, 0.5], dtype="float32")
    bias = Tensor([0.5, -1.0], dtype="float32")
    out = F.batch_norm(x, running_mean, running_var, weight, bias, training=True)
    x_np = x.numpy()
    mean = x_np.mean(axis=0)
    var = x_np.var(axis=0)
    expected = ((x_np - mean) / np.sqrt(var + 1e-5)) * weight.numpy() + bias.numpy()
    assert np.allclose(out.numpy(), expected, atol=1e-5)


def test_batch_norm_zero_variance():
    x = Tensor([[5.0, 5.0], [5.0, 5.0]], dtype="float32")
    running_mean = Tensor([0.0, 0.0], dtype="float32")
    running_var = Tensor([1.0, 1.0], dtype="float32")
    out = F.batch_norm(x, running_mean, running_var, training=True)
    assert np.allclose(out.numpy(), 0.0, atol=1e-5)
