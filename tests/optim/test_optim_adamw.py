# Copyright (c) 2025 Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the AdamW optimiser."""

import minitensor as mt
from minitensor import nn, optim


def test_adamw_applies_decoupled_weight_decay_with_zero_grad():
    """When the gradient is zero, AdamW should apply pure weight decay."""

    param = mt.ones(1, requires_grad=True)
    optimizer = optim.AdamW([param], lr=0.1, weight_decay=0.5)

    loss = (param * 0).sum()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    assert abs(float(param.numpy()[0]) - 0.95) < 1e-6


def test_adamw_learns_linear_function():
    """AdamW should recover a simple linear relationship with decoupled decay."""

    x = mt.arange(-1.0, 1.0, 0.02).reshape(100, 1)
    y = 2 * x + 3
    model = nn.DenseLayer(1, 1)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.05, weight_decay=0.001)

    for _ in range(200):
        pred = model(x)
        loss = criterion(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    w, b = model.parameters()
    final_loss = float(criterion(model(x), y).numpy().ravel()[0])
    assert abs(float(w.numpy().ravel()[0]) - 2.0) < 5e-3
    assert abs(float(b.numpy().ravel()[0]) - 3.0) < 5e-3
    assert final_loss < 2e-5


def test_adamw_accepts_beta_tuple():
    model = nn.DenseLayer(1, 1)
    optimizer = optim.AdamW(model.parameters(), lr=0.01, betas=(0.8, 0.888))
    assert isinstance(optimizer, optim.AdamW)
