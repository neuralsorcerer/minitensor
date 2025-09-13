# Copyright (c) 2025 Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

import minitensor as mt
from minitensor import nn, optim


def test_sgd_updates_parameters():
    model = nn.DenseLayer(1, 1)
    params = model.parameters()
    opt = optim.SGD(params, lr=0.1)

    x = mt.randn(32, 1)
    y = 4 * x - 1
    crit = nn.MSELoss()

    # initial loss
    loss_before = float(crit(model(x), y).numpy().ravel()[0])

    for _ in range(50):
        preds = model(x)
        loss = crit(preds, y)
        opt.zero_grad()
        loss.backward()
        opt.step()

    loss_after = float(crit(model(x), y).numpy().ravel()[0])
    assert loss_after < loss_before
