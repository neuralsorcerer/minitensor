# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the Adagrad optimiser.

Adagrad accumulates a running *sum* of squared gradients rather than the moving
average RMSprop keeps, so the denominator never shrinks and each parameter's
effective step decays monotonically. These tests pin the arithmetic against a
direct NumPy transcription and then check the behaviour that distinguishes the
method from its moving-average relatives.
"""

import math

import numpy as np
import pytest

import minitensor as mt
from minitensor import optim


def _adagrad_reference(
    start,
    grads,
    lr,
    lr_decay=0.0,
    weight_decay=0.0,
    initial_accumulator_value=0.0,
    eps=1e-10,
):
    """Adagrad written straight from the update rule."""
    param = np.array(start, dtype=np.float64)
    state_sum = np.full_like(param, initial_accumulator_value)
    trajectory = []
    for step, grad in enumerate(grads, start=1):
        grad = np.array(grad, dtype=np.float64) + weight_decay * param
        state_sum += grad * grad
        decayed_lr = lr / (1.0 + (step - 1) * lr_decay)
        param = param - decayed_lr * grad / (np.sqrt(state_sum) + eps)
        trajectory.append(param.copy())
    return trajectory


GRADS = [[0.5, -2.0], [1.0, 0.25], [-0.75, 1.5], [0.1, -0.6]]


@pytest.mark.parametrize(
    "options",
    [
        {"lr": 0.1},
        {"lr": 0.1, "lr_decay": 0.5},
        {"lr": 0.1, "weight_decay": 0.05},
        {"lr": 0.1, "initial_accumulator_value": 0.3},
        {
            "lr": 0.05,
            "lr_decay": 0.2,
            "weight_decay": 0.01,
            "initial_accumulator_value": 0.1,
        },
    ],
)
def test_adagrad_matches_reference_over_multiple_steps(options):
    start = [1.0, -1.0]
    param = mt.Tensor(start, dtype="float64", requires_grad=True)
    optimizer = optim.Adagrad([param], **options)
    expected = _adagrad_reference(start, GRADS, **options)

    for grad, want in zip(GRADS, expected):
        (param * mt.Tensor(grad, dtype="float64")).sum().backward()
        optimizer.step()
        optimizer.zero_grad()
        mt.clear_autograd_graph()
        np.testing.assert_allclose(param.numpy(), want, rtol=1e-12, atol=1e-15)


def test_adagrad_step_size_decays_like_one_over_sqrt_t():
    """Under a constant gradient the accumulator is `t`, so the step is `lr/sqrt(t)`.

    This is the property that separates Adagrad from RMSprop, whose moving
    average would settle to a constant step instead.
    """
    param = mt.Tensor([1.0], dtype="float64", requires_grad=True)
    optimizer = optim.Adagrad([param], lr=0.1)
    ones = mt.Tensor([1.0], dtype="float64")

    steps = []
    for _ in range(12):
        before = param.item()
        (param * ones).sum().backward()
        optimizer.step()
        optimizer.zero_grad()
        mt.clear_autograd_graph()
        steps.append(before - param.item())

    for earlier, later in zip(steps, steps[1:]):
        assert later <= earlier + 1e-15, "effective step increased"

    # eps is 1e-10 but shifts the answer by a relative 1e-10, so it belongs in
    # the expected value rather than being absorbed by a loose tolerance.
    for t, step in enumerate(steps, start=1):
        assert step == pytest.approx(0.1 / (math.sqrt(t) + 1e-10), rel=1e-14)


def test_adagrad_keeps_a_large_step_for_rarely_updated_parameters():
    """The reason Adagrad suits sparse features.

    A coordinate that only occasionally receives a gradient accumulates little,
    so its denominator stays small and it keeps moving at close to the full
    learning rate while a densely-updated coordinate has already slowed down.
    """
    param = mt.Tensor([0.0, 0.0], dtype="float64", requires_grad=True)
    optimizer = optim.Adagrad([param], lr=0.1)

    # Coordinate 0 gets a gradient every step; coordinate 1 only on the last.
    for step in range(10):
        grad = [1.0, 1.0 if step == 9 else 0.0]
        (param * mt.Tensor(grad, dtype="float64")).sum().backward()
        optimizer.step()
        optimizer.zero_grad()
        mt.clear_autograd_graph()

    dense, sparse = -param.numpy()
    # The sparse coordinate saw one unit gradient, so it moved a full lr step.
    assert sparse == pytest.approx(0.1 / (1.0 + 1e-10), rel=1e-14)
    # The dense one accumulated ten, so it is well short of ten full steps.
    assert dense < 10 * 0.1
    assert dense == pytest.approx(
        sum(0.1 / (math.sqrt(t) + 1e-10) for t in range(1, 11)), rel=1e-14
    )


def test_adagrad_lr_decay_slows_progress_further():
    def run(lr_decay):
        param = mt.Tensor([1.0], dtype="float64", requires_grad=True)
        optimizer = optim.Adagrad([param], lr=0.1, lr_decay=lr_decay)
        ones = mt.Tensor([1.0], dtype="float64")
        for _ in range(5):
            (param * ones).sum().backward()
            optimizer.step()
            optimizer.zero_grad()
            mt.clear_autograd_graph()
        return param.item()

    # The first step is unaffected -- the decay divisor is 1 there -- so the
    # decayed run must land strictly closer to where it started.
    assert run(0.5) > run(0.0)


def test_adagrad_minimises_a_quadratic():
    x = mt.Tensor([3.0, -4.0], dtype="float64", requires_grad=True)
    optimizer = optim.Adagrad([x], lr=0.5)
    for _ in range(300):
        (x * x).sum().backward()
        optimizer.step()
        optimizer.zero_grad()
        mt.clear_autograd_graph()
    np.testing.assert_allclose(x.numpy(), [0.0, 0.0], atol=1e-4)


def test_adagrad_initial_accumulator_value_damps_the_first_step():
    def first_step(initial):
        param = mt.Tensor([0.0], dtype="float64", requires_grad=True)
        optimizer = optim.Adagrad([param], lr=0.1, initial_accumulator_value=initial)
        (param * mt.Tensor([1.0], dtype="float64")).sum().backward()
        optimizer.step()
        optimizer.zero_grad()
        mt.clear_autograd_graph()
        return -param.item()

    assert first_step(0.0) == pytest.approx(0.1 / (1.0 + 1e-10), rel=1e-14)
    # Starting the accumulator at 3 makes the first denominator sqrt(1 + 3) = 2.
    assert first_step(3.0) == pytest.approx(0.1 / (2.0 + 1e-10), rel=1e-14)


def test_adagrad_exposes_its_hyperparameters():
    param = mt.Tensor([1.0], requires_grad=True)
    optimizer = optim.Adagrad(
        [param], lr=0.02, lr_decay=0.1, weight_decay=0.3, initial_accumulator_value=0.4
    )
    assert optimizer.lr == pytest.approx(0.02)
    assert optimizer.lr_decay == pytest.approx(0.1)
    assert optimizer.weight_decay == pytest.approx(0.3)
    assert optimizer.initial_accumulator_value == pytest.approx(0.4)
    # Adagrad's epsilon floors a sum that only grows, so it is smaller than the
    # 1e-8 the moving-average optimisers use.
    assert optimizer.epsilon == pytest.approx(1e-10)
    assert "Adagrad" in repr(optimizer)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"lr": 0.0},
        {"lr": -1.0},
        {"lr": 0.1, "lr_decay": -0.1},
        {"lr": 0.1, "weight_decay": -0.1},
        {"lr": 0.1, "initial_accumulator_value": -0.1},
        {"lr": 0.1, "epsilon": 0.0},
    ],
)
def test_adagrad_rejects_invalid_hyperparameters(kwargs):
    param = mt.Tensor([1.0], requires_grad=True)
    with pytest.raises(ValueError):
        optim.Adagrad([param], **kwargs)
