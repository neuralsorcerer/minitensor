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
    expected = [
        0.5 * (1.0 + math.cos(math.pi * min(t, t_max) / t_max)) for t in range(13)
    ]
    np.testing.assert_allclose(seen, expected, atol=1e-12)
    # Past t_max it holds at eta_min rather than turning back up.
    assert seen[-1] == pytest.approx(0.0, abs=1e-12)


def test_cosine_annealing_respects_eta_min(optimizer):
    seen = _trace(optimizer, mt.optim.CosineAnnealingLR(optimizer, 4, 0.25), 6)
    expected = [
        0.25 + 0.75 * 0.5 * (1 + math.cos(math.pi * min(t, 4) / 4)) for t in range(7)
    ]
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
        (
            end_lr
            if t >= decay_steps
            else (1.0 - end_lr) * (1.0 - t / decay_steps) ** power + end_lr
        )
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
        (
            lambda o, **k: mt.optim.CosineAnnealingLR(o, **k),
            dict(t_max=4, eta_min=-1.0),
        ),
        (lambda o, **k: mt.optim.LinearWarmupLR(o, **k), dict(warmup_steps=0)),
        (lambda o, **k: mt.optim.PolynomialDecayLR(o, **k), dict(decay_steps=0)),
        (
            lambda o, **k: mt.optim.PolynomialDecayLR(o, **k),
            dict(decay_steps=4, power=0.0),
        ),
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


# --- resuming a schedule --------------------------------------------------
#
# A checkpoint restores the model and, since optimizer state landed, the
# optimizer. The schedule was the remaining hole: reconstructing a scheduler
# put `last_epoch` back to 0, so a run that had decayed to a quarter of its
# base rate resumed at the base rate and stayed a whole schedule ahead of
# where it should have been for the rest of training.

SCHEDULES = {
    "ConstantLR": lambda opt: mt.optim.ConstantLR(opt),
    "StepLR": lambda opt: mt.optim.StepLR(opt, step_size=2, gamma=0.5),
    "ExponentialLR": lambda opt: mt.optim.ExponentialLR(opt, gamma=0.9),
    "CosineAnnealingLR": lambda opt: mt.optim.CosineAnnealingLR(opt, t_max=12),
    "LinearWarmupLR": lambda opt: mt.optim.LinearWarmupLR(opt, warmup_steps=4),
    "PolynomialDecayLR": lambda opt: mt.optim.PolynomialDecayLR(
        opt, decay_steps=10, end_lr=0.01, power=2.0
    ),
    "MultiStepLR": lambda opt: mt.optim.MultiStepLR(opt, [3, 6], gamma=0.5),
}


def _fresh_optimizer():
    return mt.optim.SGD([mt.zeros((2,), requires_grad=True)], 1.0)


@pytest.mark.parametrize("name", sorted(SCHEDULES))
def test_restored_schedule_continues_where_it_stopped(name):
    build = SCHEDULES[name]

    straight_opt = _fresh_optimizer()
    straight = build(straight_opt)
    for _ in range(5):
        straight.step()
    state = straight.state_dict()

    resumed_opt = _fresh_optimizer()
    resumed = build(resumed_opt)
    resumed.load_state_dict(state)

    assert resumed.last_epoch == straight.last_epoch
    assert resumed_opt.lr == straight_opt.lr

    # ... and the rest of the schedule agrees, step for step.
    assert _trace(straight_opt, straight, 8) == _trace(resumed_opt, resumed, 8)


@pytest.mark.parametrize("name", sorted(SCHEDULES))
def test_without_restoring_the_schedule_restarts(name):
    """Guards the test above against passing vacuously.

    A schedule that happens to be flat over the compared range would satisfy
    it either way, so each entry must be one where dropping the state is
    observable -- except ConstantLR, which is flat by definition.
    """
    build = SCHEDULES[name]

    opt = _fresh_optimizer()
    scheduler = build(opt)
    for _ in range(5):
        scheduler.step()

    fresh_opt = _fresh_optimizer()
    build(fresh_opt)

    if name == "ConstantLR":
        assert fresh_opt.lr == opt.lr
    else:
        assert fresh_opt.lr != opt.lr, f"{name} is flat here and proves nothing"


def test_load_applies_the_rate_immediately():
    """Not on the next `step()`.

    The optimizer carries whatever rate it was constructed with until
    something writes to it, so deferring would train one step at the wrong
    rate -- exactly the step after a resume.
    """
    opt = _fresh_optimizer()
    scheduler = mt.optim.ExponentialLR(opt, gamma=0.5)
    for _ in range(4):
        scheduler.step()
    state = scheduler.state_dict()

    target = _fresh_optimizer()
    restored = mt.optim.ExponentialLR(target, gamma=0.5)
    assert target.lr == 1.0
    restored.load_state_dict(state)
    assert target.lr == pytest.approx(0.5**4)


def test_state_dict_is_a_plain_dict():
    opt = _fresh_optimizer()
    state = mt.optim.StepLR(opt, step_size=2).state_dict()
    assert isinstance(state, dict)
    assert set(state) == {"base_lr", "last_epoch"}
    assert state == {"base_lr": 1.0, "last_epoch": 0}


def test_base_lr_travels_with_the_state():
    """The schedule is relative to the rate the optimizer had at construction,
    so restoring `last_epoch` alone into a differently-configured optimizer
    would continue the wrong curve."""
    opt = _fresh_optimizer()
    opt.lr = 0.25
    scheduler = mt.optim.ExponentialLR(opt, gamma=0.5)
    for _ in range(3):
        scheduler.step()

    target = _fresh_optimizer()  # base lr 1.0, not 0.25
    restored = mt.optim.ExponentialLR(target, gamma=0.5)
    restored.load_state_dict(scheduler.state_dict())

    assert restored.base_lr == 0.25
    assert target.lr == pytest.approx(0.25 * 0.5**3)


@pytest.mark.parametrize(
    "state,error,message",
    [
        ({"last_epoch": 3}, KeyError, "base_lr"),
        ({"base_lr": 1.0}, KeyError, "last_epoch"),
        ({"base_lr": float("nan"), "last_epoch": 3}, ValueError, "finite"),
        ({"base_lr": float("inf"), "last_epoch": 3}, ValueError, "finite"),
        ({"base_lr": 1.0, "last_epoch": -1}, ValueError, "negative"),
    ],
)
def test_malformed_state_is_refused(state, error, message):
    scheduler = mt.optim.StepLR(_fresh_optimizer(), step_size=2)
    with pytest.raises(error) as excinfo:
        scheduler.load_state_dict(state)
    assert message in str(excinfo.value)


def test_a_refused_load_leaves_the_schedule_alone():
    opt = _fresh_optimizer()
    scheduler = mt.optim.StepLR(opt, step_size=2, gamma=0.5)
    for _ in range(3):
        scheduler.step()
    before = (scheduler.base_lr, scheduler.last_epoch, opt.lr)

    with pytest.raises(ValueError):
        scheduler.load_state_dict({"base_lr": 1.0, "last_epoch": -5})

    assert (scheduler.base_lr, scheduler.last_epoch, opt.lr) == before


def test_round_trip_through_json(tmp_path):
    """The state is JSON-serialisable, which is the point of returning a dict
    rather than an opaque object: it goes wherever the rest of the checkpoint
    goes."""
    import json

    opt = _fresh_optimizer()
    scheduler = mt.optim.CosineAnnealingLR(opt, t_max=10)
    for _ in range(6):
        scheduler.step()

    path = tmp_path / "sched.json"
    path.write_text(json.dumps(scheduler.state_dict()))

    target = _fresh_optimizer()
    restored = mt.optim.CosineAnnealingLR(target, t_max=10)
    restored.load_state_dict(json.loads(path.read_text()))
    assert target.lr == pytest.approx(opt.lr)
