# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""Learning-rate schedulers.

The engine implemented seven of these behind `LearningRateScheduler` and none
had a binding, so a Python user could train but could not decay a learning
rate without writing the schedule by hand. Each test states the schedule's
closed form and checks the sequence against it, rather than against a recorded
trace.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import minitensor as mt


@pytest.fixture
def optimizer():
    parameter = mt.zeros((2,), requires_grad=True)
    return mt.optim.SGD([parameter], 1.0)


def _trace(optimizer, scheduler, steps):
    """Learning rates observed on the optimizer, starting before any step."""
    seen = [optimizer.lr]
    for _ in range(steps):
        scheduler.step()
        seen.append(optimizer.lr)
    return seen


def test_constant_holds_the_rate(optimizer):
    seen = _trace(optimizer, mt.optim.ConstantLR(optimizer), 5)
    assert seen == [1.0] * 6


def test_step_lr_decays_every_step_size(optimizer):
    seen = _trace(optimizer, mt.optim.StepLR(optimizer, 3, 0.5), 9)
    expected = [0.5 ** (t // 3) for t in range(10)]
    np.testing.assert_allclose(seen, expected)


def test_exponential_lr_is_gamma_to_the_step(optimizer):
    seen = _trace(optimizer, mt.optim.ExponentialLR(optimizer, 0.9), 8)
    np.testing.assert_allclose(seen, [0.9**t for t in range(9)])


def test_cosine_annealing_follows_a_half_cosine_then_holds(optimizer):
    t_max = 8
    seen = _trace(optimizer, mt.optim.CosineAnnealingLR(optimizer, t_max, 0.0), 12)
    expected = [0.5 * (1.0 + math.cos(math.pi * min(t, t_max) / t_max)) for t in range(13)]
    np.testing.assert_allclose(seen, expected, atol=1e-12)
    # Past t_max it holds at eta_min rather than turning back up.
    assert seen[-1] == pytest.approx(0.0, abs=1e-12)


def test_cosine_annealing_respects_eta_min(optimizer):
    seen = _trace(optimizer, mt.optim.CosineAnnealingLR(optimizer, 4, 0.25), 6)
    expected = [0.25 + 0.75 * 0.5 * (1 + math.cos(math.pi * min(t, 4) / 4)) for t in range(7)]
    np.testing.assert_allclose(seen, expected, atol=1e-12)


def test_linear_warmup_ramps_from_zero(optimizer):
    seen = _trace(optimizer, mt.optim.LinearWarmupLR(optimizer, 4), 7)
    np.testing.assert_allclose(seen, [0.0, 0.25, 0.5, 0.75, 1.0, 1.0, 1.0, 1.0])
    # The rate is 0 before the first step: a warmup starts at zero by
    # definition, so the very first optimizer step does nothing.
    assert seen[0] == 0.0


def test_polynomial_decay_matches_its_closed_form(optimizer):
    decay_steps, end_lr, power = 8, 0.1, 2.0
    seen = _trace(
        optimizer, mt.optim.PolynomialDecayLR(optimizer, decay_steps, end_lr, power), 10
    )
    expected = [
        end_lr
        if t >= decay_steps
        else (1.0 - end_lr) * (1.0 - t / decay_steps) ** power + end_lr
        for t in range(11)
    ]
    np.testing.assert_allclose(seen, expected, atol=1e-12)


def test_multi_step_decays_once_per_milestone(optimizer):
    seen = _trace(optimizer, mt.optim.MultiStepLR(optimizer, [3, 7], 0.1), 9)
    expected = [0.1 ** sum(t >= m for m in (3, 7)) for t in range(10)]
    np.testing.assert_allclose(seen, expected)


def test_milestones_need_not_be_sorted(optimizer):
    a = _trace(optimizer, mt.optim.MultiStepLR(optimizer, [7, 3], 0.1), 9)
    optimizer.lr = 1.0
    b = _trace(optimizer, mt.optim.MultiStepLR(optimizer, [3, 7], 0.1), 9)
    np.testing.assert_allclose(a, b)


def test_get_lr_inspects_without_applying(optimizer):
    scheduler = mt.optim.StepLR(optimizer, 2, 0.5)
    before = optimizer.lr
    assert scheduler.get_lr(4) == pytest.approx(0.25)
    assert scheduler.get_lr(10) == pytest.approx(0.5**5)
    assert optimizer.lr == before


def test_base_lr_and_last_epoch_are_reported(optimizer):
    optimizer.lr = 0.4
    scheduler = mt.optim.StepLR(optimizer, 2, 0.5)
    assert scheduler.base_lr == pytest.approx(0.4)
    assert scheduler.last_epoch == 0
    for expected_epoch in (1, 2, 3):
        scheduler.step()
        assert scheduler.last_epoch == expected_epoch
    assert scheduler.get_last_lr() == pytest.approx(optimizer.lr)
    assert "last_epoch=3" in repr(scheduler)


@pytest.mark.parametrize(
    "factory,kwargs",
    [
        (lambda o, **k: mt.optim.StepLR(o, **k), dict(step_size=0)),
        (lambda o, **k: mt.optim.StepLR(o, **k), dict(step_size=2, gamma=-1.0)),
        (lambda o, **k: mt.optim.CosineAnnealingLR(o, **k), dict(t_max=0)),
        (lambda o, **k: mt.optim.CosineAnnealingLR(o, **k), dict(t_max=4, eta_min=-1.0)),
        (lambda o, **k: mt.optim.LinearWarmupLR(o, **k), dict(warmup_steps=0)),
        (lambda o, **k: mt.optim.PolynomialDecayLR(o, **k), dict(decay_steps=0)),
        (lambda o, **k: mt.optim.PolynomialDecayLR(o, **k), dict(decay_steps=4, power=0.0)),
        (lambda o, **k: mt.optim.MultiStepLR(o, **k), dict(milestones=[])),
        (lambda o, **k: mt.optim.ExponentialLR(o, **k), dict(gamma=float("nan"))),
    ],
)
def test_degenerate_schedule_parameters_are_rejected(optimizer, factory, kwargs):
    with pytest.raises(ValueError):
        factory(optimizer, **kwargs)


def test_scheduler_actually_drives_training():
    # End to end: the parameter update must shrink as the schedule decays.
    parameter = mt.Tensor(np.array([1.0]), dtype="float64").requires_grad_(True)
    optimizer = mt.optim.SGD([parameter], 1.0)
    scheduler = mt.optim.ExponentialLR(optimizer, 0.5)

    deltas = []
    for _ in range(4):
        before = float(parameter.numpy()[0])
        optimizer.zero_grad(True)
        (parameter * 1.0).sum().backward()
        optimizer.step()
        deltas.append(before - float(parameter.numpy()[0]))
        scheduler.step()

    # Gradient is 1 every step, so each delta is exactly that step's lr.
    np.testing.assert_allclose(deltas, [1.0, 0.5, 0.25, 0.125], rtol=1e-12)
