# Copyright (c) 2025 Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np

import minitensor as mt


def test_relu_negative_and_nan():
    t = mt.Tensor([-1.0, float("nan"), 2.0])
    out = t.relu()
    vals = out.numpy()
    np.testing.assert_allclose(vals, np.array([0.0, np.nan, 2.0]), equal_nan=True)


def test_sigmoid_tanh_extreme_inputs():
    t = mt.Tensor([1000.0, -1000.0])
    sig = t.sigmoid()
    tanh = t.tanh()
    np.testing.assert_allclose(sig.numpy(), np.array([1.0, 0.0]), atol=1e-6)
    np.testing.assert_allclose(tanh.numpy(), np.array([1.0, -1.0]), atol=1e-6)


def test_log_softmax_stability_large_range():
    t = mt.Tensor([[1000.0, -1000.0, 0.0]])
    log_sm = t.log_softmax(dim=1)
    sm_log = t.softmax(dim=1).log()
    np.testing.assert_allclose(log_sm.numpy(), sm_log.numpy(), atol=1e-6)
    np.testing.assert_allclose(
        np.exp(log_sm.numpy()).sum(axis=1), np.array([1.0]), atol=1e-6
    )
