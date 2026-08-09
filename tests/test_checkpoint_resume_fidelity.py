# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""Resuming from a checkpoint continues the run it interrupted, exactly.

The parts of a checkpoint are tested individually elsewhere -- the model state
dict round-trips, the optimizer state carries its buffers and step count, the
schedulers follow their formulas. None of that says the pieces compose. The
question a checkpoint exists to answer is whether stopping at step 4, writing
everything out, rebuilding from scratch and running to step 10 lands where an
uninterrupted run of 10 steps lands, and only an end-to-end comparison answers
it.

Bit-for-bit is the right bar rather than a tolerance. Every quantity involved is
restored rather than recomputed, so any difference at all means something was
dropped: a momentum buffer, Adam's step count (its bias correction reads it, so
losing it makes the first step after a resume an outsized one), or the
scheduler's position in its decay.

The models are rebuilt under a *different* seed before loading, so a value that
fails to load shows up as a difference rather than coinciding with the original
initialisation.

`test_forgetting_the_scheduler_diverges` is what keeps the rest honest: it
performs the same resume while leaving one piece behind, and requires the
trajectories to part. Without it, a comparison that passed vacuously -- because
nothing depended on the restored state -- would look like a success.
"""

from __future__ import annotations

import numpy as np
import pytest

import minitensor as mt

nn = mt.nn
optim = mt.optim

TOTAL_STEPS = 10
INTERRUPT_AT = 4

OPTIMIZERS = {
    "SGD": lambda p: optim.SGD(p, lr=0.05),
    "SGD_momentum": lambda p: optim.SGD(p, lr=0.05, momentum=0.9),
    "Adam": lambda p: optim.Adam(p, lr=0.05),
    "AdamW": lambda p: optim.AdamW(p, lr=0.05),
    "NAdam": lambda p: optim.NAdam(p, lr=0.05),
    "RMSprop": lambda p: optim.RMSprop(p, lr=0.05),
    "Adagrad": lambda p: optim.Adagrad(p, lr=0.05),
    "Lion": lambda p: optim.Lion(p, lr=0.05),
}


def _data():
    rng = np.random.default_rng(0)
    return (
        mt.Tensor(rng.standard_normal((16, 4)).astype(np.float32)),
        mt.Tensor(rng.standard_normal((16, 3)).astype(np.float32)),
    )


def _train(model, opt, steps, scheduler=None):
    inputs, targets = _data()
    for _ in range(steps):
        opt.zero_grad()
        nn.MSELoss()(model(inputs), targets).backward()
        opt.step()
        if scheduler is not None:
            scheduler.step()


def _weights(model):
    return {name: tensor.numpy().copy() for name, tensor in model.state_dict().items()}


def _uninterrupted(build_optimizer, build_scheduler=None):
    mt.manual_seed(0)
    model = nn.DenseLayer(4, 3)
    opt = build_optimizer(model.parameters())
    scheduler = None if build_scheduler is None else build_scheduler(opt)
    _train(model, opt, TOTAL_STEPS, scheduler)
    return model, opt


def _assert_same(got, want):
    assert sorted(got) == sorted(want)
    for name, values in want.items():
        np.testing.assert_array_equal(got[name], values, err_msg=name)


# --- every optimizer, in memory ---------------------------------------------


@pytest.mark.parametrize("name", list(OPTIMIZERS), ids=list(OPTIMIZERS))
def test_resuming_reproduces_the_uninterrupted_run(name):
    build = OPTIMIZERS[name]
    reference, _ = _uninterrupted(build)

    mt.manual_seed(0)
    model = nn.DenseLayer(4, 3)
    opt = build(model.parameters())
    _train(model, opt, INTERRUPT_AT)
    weights, optimizer_state = model.state_dict(), opt.state_dict()

    mt.manual_seed(321)  # a different initialisation, so a failed load shows
    resumed = nn.DenseLayer(4, 3)
    resumed.load_state_dict(weights)
    resumed_opt = build(resumed.parameters())
    resumed_opt.load_state_dict(optimizer_state)
    _train(resumed, resumed_opt, TOTAL_STEPS - INTERRUPT_AT)

    _assert_same(_weights(resumed), _weights(reference))


@pytest.mark.parametrize("name", list(OPTIMIZERS), ids=list(OPTIMIZERS))
def test_the_step_count_survives(name):
    """Adam's and NAdam's bias correction read it, so losing it makes the first
    step after a resume an outsized one."""
    build = OPTIMIZERS[name]
    _, reference_opt = _uninterrupted(build)

    mt.manual_seed(0)
    model = nn.DenseLayer(4, 3)
    opt = build(model.parameters())
    _train(model, opt, INTERRUPT_AT)

    mt.manual_seed(321)
    resumed = nn.DenseLayer(4, 3)
    resumed.load_state_dict(model.state_dict())
    resumed_opt = build(resumed.parameters())
    resumed_opt.load_state_dict(opt.state_dict())
    _train(resumed, resumed_opt, TOTAL_STEPS - INTERRUPT_AT)

    assert (
        resumed_opt.state_dict().step_count
        == reference_opt.state_dict().step_count
        == TOTAL_STEPS
    )


# --- through files, which is how it is actually done ------------------------


def test_resuming_through_files_reproduces_the_run(tmp_path):
    build = OPTIMIZERS["Adam"]
    reference, _ = _uninterrupted(build)

    mt.manual_seed(0)
    model = nn.DenseLayer(4, 3)
    opt = build(model.parameters())
    _train(model, opt, INTERRUPT_AT)

    model_path = str(tmp_path / "model.bin")
    optimizer_path = str(tmp_path / "optimizer.bin")
    model.save(model_path)
    opt.state_dict().save(optimizer_path)

    mt.manual_seed(321)
    resumed = nn.DenseLayer(4, 3)
    resumed.load_state_dict(type(resumed).load_state_from(model_path))
    resumed_opt = build(resumed.parameters())
    resumed_opt.load_state_dict(optim.OptimizerState.load(optimizer_path))
    _train(resumed, resumed_opt, TOTAL_STEPS - INTERRUPT_AT)

    _assert_same(_weights(resumed), _weights(reference))


# --- with a scheduler, whose position is state too ---------------------------


def _scheduler(opt):
    return optim.StepLR(opt, 3, 0.5)


def _resume_with_scheduler(restore_scheduler):
    build = OPTIMIZERS["Adam"]

    mt.manual_seed(0)
    model = nn.DenseLayer(4, 3)
    opt = build(model.parameters())
    scheduler = _scheduler(opt)
    _train(model, opt, INTERRUPT_AT, scheduler)

    mt.manual_seed(321)
    resumed = nn.DenseLayer(4, 3)
    resumed.load_state_dict(model.state_dict())
    resumed_opt = build(resumed.parameters())
    resumed_opt.load_state_dict(opt.state_dict())
    resumed_scheduler = _scheduler(resumed_opt)
    if restore_scheduler:
        resumed_scheduler.load_state_dict(scheduler.state_dict())
    _train(resumed, resumed_opt, TOTAL_STEPS - INTERRUPT_AT, resumed_scheduler)

    return resumed, resumed_opt


def test_resuming_a_scheduled_run_reproduces_it():
    reference, reference_opt = _uninterrupted(OPTIMIZERS["Adam"], _scheduler)
    resumed, resumed_opt = _resume_with_scheduler(restore_scheduler=True)

    _assert_same(_weights(resumed), _weights(reference))
    assert resumed_opt.lr == reference_opt.lr


def test_forgetting_the_scheduler_diverges():
    """The control. A resume that leaves the scheduler behind restarts its
    decay, so the learning rates differ and the weights must too -- otherwise
    the comparisons above would be passing on nothing."""
    reference, reference_opt = _uninterrupted(OPTIMIZERS["Adam"], _scheduler)
    resumed, resumed_opt = _resume_with_scheduler(restore_scheduler=False)

    assert resumed_opt.lr != reference_opt.lr
    got, want = _weights(resumed), _weights(reference)
    assert any(not np.array_equal(got[name], want[name]) for name in want)


# --- a model with buffers, which travel separately --------------------------


def test_a_model_with_running_statistics_resumes():
    """BatchNorm's running mean and variance are buffers, not parameters, and
    they keep updating during training -- so they are part of the run's state."""

    def build_model():
        return nn.Sequential(
            [nn.DenseLayer(4, 4), nn.BatchNorm1d(4), nn.DenseLayer(4, 3)]
        )

    mt.manual_seed(0)
    reference = build_model()
    reference_opt = optim.Adam(reference.parameters(), lr=0.05)
    _train(reference, reference_opt, TOTAL_STEPS)

    mt.manual_seed(0)
    model = build_model()
    opt = optim.Adam(model.parameters(), lr=0.05)
    _train(model, opt, INTERRUPT_AT)

    mt.manual_seed(321)
    resumed = build_model()
    resumed.load_state_dict(model.state_dict())
    resumed_opt = optim.Adam(resumed.parameters(), lr=0.05)
    resumed_opt.load_state_dict(opt.state_dict())
    _train(resumed, resumed_opt, TOTAL_STEPS - INTERRUPT_AT)

    _assert_same(_weights(resumed), _weights(reference))
