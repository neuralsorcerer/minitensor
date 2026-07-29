# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the NAdam optimiser (Dozat, 2016).

NAdam is Adam with Nesterov momentum: the step uses the momentum the *next*
iterate will carry rather than the current one, and the momentum coefficient is
scheduled rather than fixed. These tests pin the arithmetic against a direct
NumPy transcription and then check the behaviour that separates it from Adam.
"""

import numpy as np
import pytest

import minitensor as mt
from minitensor import optim


def nadam_reference(
    start,
    grads,
    lr,
    beta1=0.9,
    beta2=0.999,
    eps=1e-8,
    weight_decay=0.0,
    momentum_decay=0.004,
):
    param = np.array(start, dtype=np.float64)
    m = np.zeros_like(param)
    v = np.zeros_like(param)
    mu_product = 1.0
    trajectory = []
    for t, grad in enumerate(grads, start=1):
        grad = np.array(grad, dtype=np.float64) + weight_decay * param
        mu = beta1 * (1 - 0.5 * 0.96 ** (t * momentum_decay))
        mu_next = beta1 * (1 - 0.5 * 0.96 ** ((t + 1) * momentum_decay))
        mu_product *= mu
        mu_product_next = mu_product * mu_next

        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * grad * grad
        denom = np.sqrt(v / (1 - beta2**t)) + eps
        param = (
            param
            - lr
            * ((1 - mu) / (1 - mu_product) * grad + mu_next / (1 - mu_product_next) * m)
            / denom
        )
        trajectory.append(param.copy())
    return trajectory


GRADS = [[0.5, -2.0], [1.0, 0.25], [-0.75, 1.5], [0.1, -0.6], [0.3, 0.9]]


@pytest.mark.parametrize(
    "options",
    [
        {"lr": 0.01},
        {"lr": 0.01, "weight_decay": 0.05},
        {"lr": 0.002, "beta1": 0.8, "beta2": 0.99},
        {"lr": 0.01, "momentum_decay": 0.02},
        {
            "lr": 0.005,
            "beta1": 0.95,
            "beta2": 0.9995,
            "weight_decay": 0.01,
            "momentum_decay": 0.008,
        },
    ],
)
def test_nadam_matches_reference_over_multiple_steps(options):
    start = [1.0, -1.0]
    param = mt.Tensor(start, dtype="float64", requires_grad=True)
    optimizer = optim.NAdam([param], **options)
    expected = nadam_reference(start, GRADS, **options)

    for grad, want in zip(GRADS, expected):
        (param * mt.Tensor(grad, dtype="float64")).sum().backward()
        optimizer.step()
        optimizer.zero_grad()
        mt.clear_autograd_graph()
        np.testing.assert_allclose(param.numpy(), want, rtol=1e-12, atol=1e-15)


def test_momentum_schedule_advances_once_per_step_not_per_parameter():
    """The running momentum product is shared, so it must advance once a step.

    Two parameters with identical state and identical gradients must move
    identically. Advancing the schedule inside the per-parameter loop instead
    would leave every parameter after the first on a further-advanced schedule —
    a bug that a single-parameter test cannot see.
    """
    first = mt.Tensor([1.0], dtype="float64", requires_grad=True)
    second = mt.Tensor([1.0], dtype="float64", requires_grad=True)
    optimizer = optim.NAdam([first, second], lr=0.1)
    ones = mt.Tensor([1.0], dtype="float64")

    for _ in range(5):
        (first * ones).sum().backward()
        (second * ones).sum().backward()
        optimizer.step()
        optimizer.zero_grad()
        mt.clear_autograd_graph()

    assert first.item() == second.item()


def test_nadam_differs_from_adam():
    """Nesterov look-ahead is not a no-op.

    If the momentum terms had collapsed to Adam's bias correction the two would
    coincide; they must not.
    """
    grads = [[0.5], [1.0], [-0.75], [0.1], [0.3], [0.8]]

    def run(name):
        param = mt.Tensor([1.0], dtype="float64", requires_grad=True)
        optimizer = getattr(optim, name)([param], lr=0.05)
        for grad in grads:
            (param * mt.Tensor(grad, dtype="float64")).sum().backward()
            optimizer.step()
            optimizer.zero_grad()
            mt.clear_autograd_graph()
        return param.item()

    assert abs(run("Adam") - run("NAdam")) > 1e-3


def test_momentum_schedule_rises_from_half_beta1_toward_beta1():
    beta1, decay = 0.9, 0.004

    def mu(t):
        return beta1 * (1 - 0.5 * 0.96 ** (t * decay))

    # Starts just above beta1/2, damping the early steps, and climbs from there.
    assert mu(1) == pytest.approx(beta1 / 2, abs=1e-3)
    assert mu(1) < mu(100) < mu(10_000) < beta1


def test_nadam_minimises_a_quadratic():
    x = mt.Tensor([3.0, -4.0], dtype="float64", requires_grad=True)
    optimizer = optim.NAdam([x], lr=0.1)
    for _ in range(400):
        (x * x).sum().backward()
        optimizer.step()
        optimizer.zero_grad()
        mt.clear_autograd_graph()
    np.testing.assert_allclose(x.numpy(), [0.0, 0.0], atol=1e-4)


def test_nadam_exposes_its_hyperparameters():
    param = mt.Tensor([1.0], requires_grad=True)
    optimizer = optim.NAdam(
        [param],
        lr=0.003,
        beta1=0.85,
        beta2=0.995,
        epsilon=1e-7,
        weight_decay=0.2,
        momentum_decay=0.01,
    )
    assert optimizer.lr == pytest.approx(0.003)
    assert optimizer.beta1 == pytest.approx(0.85)
    assert optimizer.beta2 == pytest.approx(0.995)
    assert optimizer.epsilon == pytest.approx(1e-7)
    assert optimizer.weight_decay == pytest.approx(0.2)
    assert optimizer.momentum_decay == pytest.approx(0.01)
    assert "NAdam" in repr(optimizer)


def test_nadam_defaults_match_the_paper():
    optimizer = optim.NAdam([mt.Tensor([1.0], requires_grad=True)])
    assert optimizer.lr == pytest.approx(0.002)
    assert optimizer.beta1 == pytest.approx(0.9)
    assert optimizer.beta2 == pytest.approx(0.999)
    assert optimizer.momentum_decay == pytest.approx(0.004)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"lr": 0.0},
        {"lr": -1.0},
        {"lr": 0.01, "beta1": 1.0},
        {"lr": 0.01, "beta2": 1.0},
        {"lr": 0.01, "beta1": -0.1},
        {"lr": 0.01, "epsilon": 0.0},
        {"lr": 0.01, "weight_decay": -0.1},
        {"lr": 0.01, "momentum_decay": -0.1},
    ],
)
def test_nadam_rejects_invalid_hyperparameters(kwargs):
    param = mt.Tensor([1.0], requires_grad=True)
    with pytest.raises(ValueError):
        optim.NAdam([param], **kwargs)
