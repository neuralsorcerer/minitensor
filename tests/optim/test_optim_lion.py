# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the Lion optimiser (Chen et al., 2023).

Lion's update is the *sign* of an interpolated momentum, so every parameter
moves by exactly `lr` regardless of gradient magnitude. That makes the update
reproducible to the bit, and these tests pin it against a direct NumPy
transcription of the published algorithm.
"""

import numpy as np
import pytest

import minitensor as mt
from minitensor import optim


def _lion_reference_step(theta, momentum, grad, lr, beta1, beta2, weight_decay):
    update = np.sign(beta1 * momentum + (1 - beta1) * grad)
    theta = theta - lr * (update + weight_decay * theta)
    momentum = beta2 * momentum + (1 - beta2) * grad
    return theta, momentum


@pytest.mark.parametrize("weight_decay", [0.0, 0.1])
def test_lion_matches_reference_over_multiple_steps(weight_decay):
    """Eight steps of least-squares descent, compared elementwise to NumPy."""
    rng = np.random.default_rng(0)
    lr, beta1, beta2 = 0.1, 0.9, 0.99
    start = rng.standard_normal(5)
    target = rng.standard_normal(5)

    param = mt.Tensor(start.tolist(), dtype="float64", requires_grad=True)
    target_t = mt.Tensor(target.tolist(), dtype="float64")
    optimizer = optim.Lion(
        [param], lr=lr, beta1=beta1, beta2=beta2, weight_decay=weight_decay
    )

    expected = start.copy()
    momentum = np.zeros_like(start)

    for _ in range(8):
        optimizer.zero_grad()
        diff = param - target_t
        ((diff * diff).sum() * 0.5).backward()
        optimizer.step()

        # d/dtheta of 0.5*||theta - target||^2 is (theta - target).
        expected, momentum = _lion_reference_step(
            expected, momentum, expected - target, lr, beta1, beta2, weight_decay
        )

    np.testing.assert_allclose(param.numpy(), expected, rtol=1e-12, atol=1e-12)


def test_lion_first_step_moves_every_parameter_by_exactly_lr():
    """With a zero momentum buffer the update is sign(grad), so |step| == lr."""
    param = mt.Tensor([1.0, 2.0, -3.0], dtype="float64", requires_grad=True)
    grad_scale = mt.Tensor([0.5, -0.1, 4.0], dtype="float64")
    optimizer = optim.Lion([param], lr=0.1)

    optimizer.zero_grad()
    (param * grad_scale).sum().backward()
    optimizer.step()

    # Gradient magnitudes differ by ~40x but each parameter moves by 0.1.
    np.testing.assert_allclose(param.numpy(), [0.9, 2.1, -3.1], atol=1e-12)


def test_lion_leaves_parameters_untouched_on_zero_gradient():
    """sign(0) must be 0 — not +1, as a naive signum would give."""
    param = mt.Tensor([1.0, -2.0], dtype="float64", requires_grad=True)
    optimizer = optim.Lion([param], lr=0.1)

    optimizer.zero_grad()
    (param * mt.Tensor([0.0, 0.0], dtype="float64")).sum().backward()
    optimizer.step()

    np.testing.assert_allclose(param.numpy(), [1.0, -2.0], atol=1e-12)


def test_lion_applies_decoupled_weight_decay_with_zero_gradient():
    param = mt.Tensor([10.0], dtype="float64", requires_grad=True)
    optimizer = optim.Lion([param], lr=0.1, weight_decay=0.5)

    optimizer.zero_grad()
    (param * mt.Tensor([0.0], dtype="float64")).sum().backward()
    optimizer.step()

    # No sign term, so only decay remains: 10 - 0.1 * 0.5 * 10.
    np.testing.assert_allclose(param.numpy(), [9.5], atol=1e-12)


def test_lion_uses_its_own_beta_defaults():
    """Lion's published defaults are (0.9, 0.99), not Adam's (0.9, 0.999)."""
    param = mt.ones(1, requires_grad=True)
    optimizer = optim.Lion([param])
    assert optimizer.beta1 == pytest.approx(0.9)
    assert optimizer.beta2 == pytest.approx(0.99)


def test_lion_exposes_hyperparameters():
    param = mt.ones(1, requires_grad=True)
    optimizer = optim.Lion([param], lr=3e-4, weight_decay=0.5)
    assert optimizer.lr == pytest.approx(3e-4)
    assert optimizer.weight_decay == pytest.approx(0.5)
    assert "Lion(" in repr(optimizer)

    optimizer.lr = 1e-4
    assert optimizer.lr == pytest.approx(1e-4)


def test_lion_rejects_invalid_hyperparameters():
    param = mt.ones(1, requires_grad=True)
    with pytest.raises(Exception):
        optim.Lion([param], lr=-1.0)
    with pytest.raises(Exception):
        optim.Lion([param], weight_decay=-0.1)
    with pytest.raises(Exception):
        optim.Lion([param], beta1=1.5, beta2=0.99)


def test_lion_learns_a_linear_function():
    """End-to-end sanity: loss must fall substantially over training."""
    rng = np.random.default_rng(1)
    x = rng.standard_normal((64, 1))
    y = 3.0 * x + 0.5

    weight = mt.Tensor([[0.0]], dtype="float64", requires_grad=True)
    bias = mt.Tensor([0.0], dtype="float64", requires_grad=True)
    xt = mt.Tensor(x.tolist(), dtype="float64")
    yt = mt.Tensor(y.tolist(), dtype="float64")

    optimizer = optim.Lion([weight, bias], lr=0.05)

    def loss_value():
        residual = xt.matmul(weight) + bias - yt
        return (residual * residual).mean()

    first = float(loss_value().numpy())
    for _ in range(200):
        optimizer.zero_grad()
        loss = loss_value()
        loss.backward()
        optimizer.step()
    final = float(loss_value().numpy())

    assert final < first * 0.01
    assert float(weight.numpy()[0][0]) == pytest.approx(3.0, abs=0.1)
    assert float(bias.numpy()[0]) == pytest.approx(0.5, abs=0.1)
