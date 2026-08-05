# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""Saving a checkpoint has to save the optimizer too, or it is half a run.

Restoring only the weights leaves a fresh optimizer: zeroed moment estimates
and `step_count = 0`, so Adam's bias correction restarts from t=0 and the first
step after the resume is an outsized one. Measured on the regression below,
that step moved the parameters 2.05x as far as the step it was supposed to be
continuing, and was still 1.47x off five steps later. The resumed run was a
different run.

`test_resume_is_an_exact_continuation` is the load-bearing test here: it runs
a model straight through, runs it again with a save/reload in the middle, and
requires the two loss curves to be **bit-identical**. Anything the optimizer
forgets shows up as a difference, so it does not need to know what the state
is made of.
"""

from __future__ import annotations

import numpy as np
import pytest

import minitensor as mt
from minitensor import nn, optim

# Every optimizer, each configured so it actually carries state -- plain SGD
# without momentum has none, and would pass this file trivially.
OPTIMIZERS = {
    "Adam": lambda params: optim.Adam(params, lr=1e-2),
    "AdamW": lambda params: optim.AdamW(params, lr=1e-2, weight_decay=1e-2),
    "SGD-momentum": lambda params: optim.SGD(params, lr=1e-2, momentum=0.9),
    "SGD-nesterov": lambda params: optim.SGD(
        params, lr=1e-2, momentum=0.9, nesterov=True
    ),
    "RMSprop": lambda params: optim.RMSprop(params, lr=1e-2, momentum=0.9),
    "RMSprop-centered": lambda params: optim.RMSprop(params, lr=1e-2, centered=True),
    "Adagrad": lambda params: optim.Adagrad(params, lr=1e-1),
    "NAdam": lambda params: optim.NAdam(params, lr=1e-2),
    "Lion": lambda params: optim.Lion(params, lr=1e-3),
}


def _model():
    return nn.Sequential([nn.DenseLayer(16, 32), nn.ReLU(), nn.DenseLayer(32, 4)])


def _train(make_optimizer, steps=40, interrupt_at=None, paths=None):
    """Run `steps` optimizer steps, optionally checkpointing and reloading."""
    mt.manual_seed(7)
    np.random.seed(7)
    features = mt.Tensor(np.random.randn(64, 16).astype(np.float32))
    targets = mt.Tensor(np.random.randn(64, 4).astype(np.float32))

    model = _model()
    optimizer = make_optimizer(model.parameters())

    losses = []
    for step in range(steps):
        if interrupt_at is not None and step == interrupt_at:
            model_path, optimizer_path = paths
            model.save(model_path)
            optimizer.save(optimizer_path)

            model = _model()
            model.load_state_dict(type(model).load_state_from(model_path))
            optimizer = make_optimizer(model.parameters())
            optimizer.load(optimizer_path)

        optimizer.zero_grad()
        loss = nn.mse_loss(model(features), targets)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.numpy()))

    return losses, model


@pytest.mark.parametrize("name", sorted(OPTIMIZERS))
def test_resume_is_an_exact_continuation(name, tmp_path):
    make = OPTIMIZERS[name]
    paths = (str(tmp_path / "model.bin"), str(tmp_path / "optimizer.bin"))

    straight, straight_model = _train(make)
    resumed, resumed_model = _train(make, interrupt_at=20, paths=paths)

    np.testing.assert_array_equal(straight, resumed)
    for left, right in zip(straight_model.parameters(), resumed_model.parameters()):
        np.testing.assert_array_equal(left.numpy(), right.numpy())


@pytest.mark.parametrize("name", sorted(OPTIMIZERS))
def test_without_the_optimizer_state_the_run_diverges(name, tmp_path):
    """The test above would pass vacuously if the state did not matter.

    Reloading the weights but *not* the optimizer must produce a different
    trajectory -- which is the bug this file exists for.
    """
    make = OPTIMIZERS[name]
    model_path = str(tmp_path / "model.bin")

    straight, _ = _train(make)

    mt.manual_seed(7)
    np.random.seed(7)
    features = mt.Tensor(np.random.randn(64, 16).astype(np.float32))
    targets = mt.Tensor(np.random.randn(64, 4).astype(np.float32))
    model = _model()
    optimizer = make(model.parameters())
    weights_only = []
    for step in range(40):
        if step == 20:
            model.save(model_path)
            model = _model()
            model.load_state_dict(type(model).load_state_from(model_path))
            optimizer = make(model.parameters())  # state discarded
        optimizer.zero_grad()
        loss = nn.mse_loss(model(features), targets)
        loss.backward()
        optimizer.step()
        weights_only.append(float(loss.numpy()))

    assert weights_only[:20] == straight[:20], "divergence must start at the resume"
    assert weights_only[20:] != straight[20:], (
        f"{name} carries no state across a step, so this file proves nothing for it"
    )


@pytest.mark.parametrize("name", sorted(OPTIMIZERS))
def test_state_dict_round_trips_in_memory(name, tmp_path):
    """No file involved -- `state_dict()` into `load_state_dict()` directly."""
    make = OPTIMIZERS[name]
    mt.manual_seed(3)
    features, targets = mt.randn(8, 16), mt.randn(8, 4)

    model = _model()
    optimizer = make(model.parameters())
    for _ in range(5):
        optimizer.zero_grad()
        nn.mse_loss(model(features), targets).backward()
        optimizer.step()

    saved = optimizer.state_dict()
    restored = make(model.parameters())
    restored.load_state_dict(saved)

    assert restored.step_count == optimizer.step_count
    assert restored.state_dict().buffer_names() == saved.buffer_names()


@pytest.mark.parametrize("name", sorted(OPTIMIZERS))
def test_step_count_survives(name):
    make = OPTIMIZERS[name]
    model = _model()
    optimizer = make(model.parameters())
    features, targets = mt.randn(8, 16), mt.randn(8, 4)
    for _ in range(7):
        optimizer.zero_grad()
        nn.mse_loss(model(features), targets).backward()
        optimizer.step()

    assert optimizer.step_count == 7
    assert optimizer.state_dict().step_count == 7

    fresh = make(model.parameters())
    assert fresh.step_count == 0
    fresh.load_state_dict(optimizer.state_dict())
    assert fresh.step_count == 7


def test_state_names_the_algorithm_that_wrote_it():
    model = _model()
    for name, make in OPTIMIZERS.items():
        expected = name.split("-")[0]
        assert make(model.parameters()).state_dict().algorithm == expected


def test_loading_into_a_different_algorithm_is_refused():
    """Silently dropping unrecognised buffers is the failure this guards.

    Adam and AdamW are the sharp case: identical buffer layout, different
    update rule, so nothing about the *data* would object.
    """
    model = _model()
    adam = optim.Adam(model.parameters(), lr=1e-2)

    with pytest.raises(Exception) as excinfo:
        optim.SGD(model.parameters(), lr=1e-2).load_state_dict(adam.state_dict())
    assert "saved by Adam" in str(excinfo.value)

    with pytest.raises(Exception) as excinfo:
        optim.AdamW(model.parameters(), lr=1e-2).load_state_dict(adam.state_dict())
    assert "saved by Adam but is being loaded into AdamW" in str(excinfo.value)


def test_loading_into_a_different_parameter_count_is_refused():
    model = _model()
    saved = optim.Adam(model.parameters(), lr=1e-2).state_dict()
    fewer = optim.Adam(list(model.parameters())[:2], lr=1e-2)

    with pytest.raises(Exception) as excinfo:
        fewer.load_state_dict(saved)
    message = str(excinfo.value)
    assert "saved for 4 parameters" in message
    assert "tracking 2" in message


def test_loading_into_differently_shaped_parameters_is_refused():
    """State is matched by position, so a same-length but differently-shaped
    model would otherwise install a buffer of the wrong size."""
    model = _model()
    optimizer = optim.Adam(model.parameters(), lr=1e-2)
    features, targets = mt.randn(8, 16), mt.randn(8, 4)
    optimizer.zero_grad()
    nn.mse_loss(model(features), targets).backward()
    optimizer.step()

    wider = nn.Sequential([nn.DenseLayer(16, 64), nn.ReLU(), nn.DenseLayer(64, 4)])
    with pytest.raises(Exception) as excinfo:
        optim.Adam(wider.parameters(), lr=1e-2).load_state_dict(optimizer.state_dict())
    assert "has shape" in str(excinfo.value)


def test_a_stepless_optimizer_saves_no_buffers():
    """Buffers are allocated on a parameter's first step. Writing zeros for a
    parameter that has never been stepped would resume it from a position it
    was never in."""
    model = _model()
    state = optim.Adam(model.parameters(), lr=1e-2).state_dict()
    assert state.buffer_names() == []
    assert state.step_count == 0
    assert state.num_parameters == 4


def test_file_round_trip_preserves_everything(tmp_path):
    path = str(tmp_path / "opt.bin")
    model = _model()
    optimizer = optim.NAdam(model.parameters(), lr=1e-2)
    features, targets = mt.randn(8, 16), mt.randn(8, 4)
    for _ in range(4):
        optimizer.zero_grad()
        nn.mse_loss(model(features), targets).backward()
        optimizer.step()

    optimizer.save(path)
    loaded = optim.OptimizerState.load(path)
    original = optimizer.state_dict()

    assert loaded.algorithm == original.algorithm
    assert loaded.step_count == original.step_count
    assert loaded.num_parameters == original.num_parameters
    assert loaded.buffer_names() == original.buffer_names()


def test_nadam_restores_its_momentum_product(tmp_path):
    """NAdam carries a scalar beyond `step_count` and the per-parameter buffers.

    `mu_product` is a running product of the momentum schedule, and the
    schedule depends on `momentum_decay` -- so it is not recoverable from
    `step_count`. Two models are stepped in parallel from identical weights,
    one by the original optimizer and one by an optimizer restored from disk;
    with `mu_product` reset they take different steps.
    """
    path = str(tmp_path / "nadam.bin")
    mt.manual_seed(11)
    np.random.seed(11)
    features = mt.Tensor(np.random.randn(8, 16).astype(np.float32))
    targets = mt.Tensor(np.random.randn(8, 4).astype(np.float32))

    def step(model, optimizer):
        optimizer.zero_grad()
        nn.mse_loss(model(features), targets).backward()
        optimizer.step()

    weights = str(tmp_path / "model.bin")
    original = _model()
    optimizer = optim.NAdam(original.parameters(), lr=1e-2, momentum_decay=4e-3)
    for _ in range(6):
        step(original, optimizer)
    original.save(weights)
    optimizer.save(path)

    # A second model at the same weights, driven by a restored optimizer.
    twin = _model()
    twin.load_state_dict(type(twin).load_state_from(weights))
    restored = optim.NAdam(twin.parameters(), lr=1e-2, momentum_decay=4e-3)
    restored.load(path)

    step(original, optimizer)
    step(twin, restored)

    assert restored.step_count == 7
    for left, right in zip(original.parameters(), twin.parameters()):
        np.testing.assert_array_equal(left.numpy(), right.numpy())
