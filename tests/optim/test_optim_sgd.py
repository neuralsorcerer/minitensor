# Copyright (c) 2025 Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the stochastic gradient descent optimiser."""

import minitensor as mt
from minitensor import nn, optim


def test_sgd_learns_linear_function():
    """SGD should recover parameters of a simple linear relationship."""
    x = mt.arange(-1.0, 1.0, 0.02).reshape(100, 1)
    y = 4 * x - 1
    model = nn.DenseLayer(1, 1)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    for _ in range(200):
        pred = model(x)
        loss = criterion(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    w, b = model.parameters()
    final_loss = float(criterion(model(x), y).numpy().ravel()[0])
    assert abs(float(w.numpy().ravel()[0]) - 4.0) < 1e-3
    assert abs(float(b.numpy().ravel()[0]) + 1.0) < 1e-3
    assert final_loss < 1e-8


def test_zero_grad_set_to_none():
    t = mt.ones(1, requires_grad=True)
    t.backward(mt.ones(1))
    assert t.grad is not None
    opt = optim.SGD([t], lr=0.1)
    opt.zero_grad(set_to_none=True)
    assert t.grad is None


def test_sgd_with_momentum():
    """SGD with momentum should recover parameters of a simple linear relationship."""
    x = mt.arange(-1.0, 1.0, 0.02).reshape(100, 1)
    y = 4 * x - 1
    model = nn.DenseLayer(1, 1)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

    for _ in range(200):
        pred = model(x)
        loss = criterion(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    w, b = model.parameters()
    final_loss = float(criterion(model(x), y).numpy().ravel()[0])
    assert abs(float(w.numpy().ravel()[0]) - 4.0) < 1e-3
    assert abs(float(b.numpy().ravel()[0]) + 1.0) < 1e-3
    assert final_loss < 1e-8


def test_sgd_with_weight_decay():
    """SGD with weight decay should recover parameters of a simple linear relationship."""
    x = mt.arange(-1.0, 1.0, 0.02).reshape(100, 1)
    y = 4 * x - 1
    model = nn.DenseLayer(1, 1)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, weight_decay=0.01)

    for _ in range(200):
        pred = model(x)
        loss = criterion(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    w, b = model.parameters()
    final_loss = float(criterion(model(x), y).numpy().ravel()[0])
    assert abs(float(w.numpy().ravel()[0]) - 4.0) < 1e-3
    assert abs(float(b.numpy().ravel()[0]) + 1.0) < 1e-3
    assert final_loss < 1e-8


def test_sgd_with_momentum_and_weight_decay():
    """SGD with momentum and weight decay should recover parameters of a simple linear relationship."""
    x = mt.arange(-1.0, 1.0, 0.02).reshape(100, 1)
    y = 4 * x - 1
    model = nn.DenseLayer(1, 1)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.01)

    for _ in range(200):
        pred = model(x)
        loss = criterion(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    w, b = model.parameters()
    final_loss = float(criterion(model(x), y).numpy().ravel()[0])
    assert abs(float(w.numpy().ravel()[0]) - 4.0) < 1e-3
    assert abs(float(b.numpy().ravel()[0]) + 1.0) < 1e-3
    assert final_loss < 1e-8


def test_sgd_invalid_momentum():
    """SGD should raise an error for invalid momentum values."""
    model = nn.DenseLayer(1, 1)
    try:
        optim.SGD(model.parameters(), lr=0.1, momentum=-0.1)
    except ValueError as e:
        assert str(e) == "Momentum must be between 0 and 1."
    else:
        assert False, "ValueError not raised for negative momentum."

    try:
        optim.SGD(model.parameters(), lr=0.1, momentum=1.5)
    except ValueError as e:
        assert str(e) == "Momentum must be between 0 and 1."
    else:
        assert False, "ValueError not raised for momentum greater than 1."


def test_sgd_invalid_weight_decay():
    """SGD should raise an error for negative weight decay."""
    model = nn.DenseLayer(1, 1)
    try:
        optim.SGD(model.parameters(), lr=0.1, weight_decay=-0.01)
    except ValueError as e:
        assert str(e) == "Weight decay must be non-negative."
    else:
        assert False, "ValueError not raised for negative weight decay."


def test_sgd_invalid_lr():
    """SGD should raise an error for non-positive learning rate."""
    model = nn.DenseLayer(1, 1)
    try:
        optim.SGD(model.parameters(), lr=0.0)
    except ValueError as e:
        assert str(e) == "Learning rate must be positive."
    else:
        assert False, "ValueError not raised for zero learning rate."

    try:
        optim.SGD(model.parameters(), lr=-0.1)
    except ValueError as e:
        assert str(e) == "Learning rate must be positive."
    else:
        assert False, "ValueError not raised for negative learning rate."


def test_sgd_no_parameters():
    """SGD should raise an error if no parameters are provided."""
    try:
        optim.SGD([], lr=0.1)
    except ValueError as e:
        assert str(e) == "No parameters to optimize."
    else:
        assert False, "ValueError not raised for empty parameter list."
