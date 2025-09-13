# Copyright (c) 2025 Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

from pytest import raises

import minitensor as mt
from minitensor import nn, optim


def test_adam_accepts_betas_tuple():
    model = nn.DenseLayer(1, 1)
    params = model.parameters()
    optimizer = optim.Adam(params, 0.01, betas=(0.8, 0.888))

    x = mt.randn(10, 1)
    y = 2 * x + 1
    criterion = nn.MSELoss()
    preds = model(x)
    loss = criterion(preds, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    assert isinstance(optimizer, optim.Adam)


def test_adam_rejects_non_tuple_betas():
    model = nn.DenseLayer(1, 1)
    params = model.parameters()
    with raises(TypeError):
        optim.Adam(params, 0.01, betas=0.9)


def test_adam_accepts_beta1_beta2():
    model = nn.DenseLayer(1, 1)
    params = model.parameters()
    optimizer = optim.Adam(params, 0.01, beta1=0.8, beta2=0.888)

    x = mt.randn(5, 1)
    y = 3 * x - 1
    criterion = nn.MSELoss()
    loss = criterion(model(x), y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    assert isinstance(optimizer, optim.Adam)


def test_adam_rejects_mixed_beta_args():
    model = nn.DenseLayer(1, 1)
    params = model.parameters()
    with raises(TypeError):
        optim.Adam(params, 0.01, betas=(0.9, 0.999), beta1=0.9)
