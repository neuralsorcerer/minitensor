# Copyright (c) 2025 Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

import minitensor as mt
from minitensor import nn


def test_bce_loss_positive():
    preds = mt.Tensor([0.8, 0.2]).reshape(2, 1)
    targets = mt.Tensor([1.0, 0.0]).reshape(2, 1)
    loss = nn.BCELoss()(preds, targets)
    assert float(loss.numpy().ravel()[0]) > 0
