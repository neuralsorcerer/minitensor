# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""`module.requires_grad_(False)` freezes a module, and a step leaves it alone.

There was no way to freeze a module. The handles `parameters()` returns carry
`requires_grad` flags of their own, so setting one changed that handle and
nothing else: the module went on recording its forward pass and receiving
gradients.

Freezing also exposed a separate fault. `zero_grad()` made a zero gradient
for every parameter that had none, and an optimizer applies any gradient it
finds, so a parameter the backward pass never reached -- a frozen layer, a
branch the loss did not use -- was stepped as if it had a zero gradient.
Weight decay then shrank it on a step it took no part in.
"""

from __future__ import annotations

import numpy as np
import pytest

import minitensor as mt

nn = mt.nn
optim = mt.optim

OPTIMIZERS = {
    "SGD+decay": lambda params: optim.SGD(params, lr=0.1, weight_decay=0.1),
    "SGD+momentum": lambda params: optim.SGD(
        params, lr=0.1, momentum=0.9, weight_decay=0.1
    ),
    "Adam+decay": lambda params: optim.Adam(params, lr=0.1, weight_decay=0.1),
    "AdamW": lambda params: optim.AdamW(params, lr=0.1),
    "RMSprop+decay": lambda params: optim.RMSprop(params, lr=0.1, weight_decay=0.1),
}


def _snapshot(module):
    return {name: t.numpy().copy() for name, t in module.state_dict().items()}


def _changed(before, after):
    return [name for name in before if not np.array_equal(before[name], after[name])]


def _inputs(shape=(4, 4), seed=0):
    rng = np.random.default_rng(seed)
    return mt.from_numpy(rng.standard_normal(shape).astype(np.float32))


def _encoder_and_head():
    mt.manual_seed(0)
    return nn.DenseLayer(4, 5), nn.DenseLayer(5, 2)


def _train_step(optimizer, loss_of):
    optimizer.zero_grad()
    loss_of().sum().backward()
    optimizer.step()


# --- the switch --------------------------------------------------------------


def test_requires_grad_returns_the_module():
    layer = nn.DenseLayer(4, 3)
    assert layer.requires_grad_(False) is layer
    assert layer.requires_grad_() is layer


def test_it_sets_every_parameter_and_can_be_undone():
    model = nn.Sequential([nn.DenseLayer(4, 5), nn.ReLU(), nn.DenseLayer(5, 2)])
    model.requires_grad_(False)
    assert [p.requires_grad for p in model.parameters()] == [False] * 4
    assert model.parameter_stats()["trainable_parameters"] == 0

    model.requires_grad_()
    assert [p.requires_grad for p in model.parameters()] == [True] * 4


def test_a_handles_flag_does_not_reach_the_module():
    """What `requires_grad_` on the module is for."""
    layer = nn.DenseLayer(4, 3)
    for handle in layer.parameters():
        handle.requires_grad_(False)
    assert all(p.requires_grad for p in layer.parameters())


# --- a frozen module in training ---------------------------------------------


@pytest.mark.parametrize("name", list(OPTIMIZERS))
def test_a_frozen_encoder_is_left_alone_by_an_optimizer_over_everything(name):
    encoder, head = _encoder_and_head()
    optimizer = OPTIMIZERS[name](encoder.parameters() + head.parameters())
    encoder.requires_grad_(False)
    encoder_before, head_before = _snapshot(encoder), _snapshot(head)

    for _ in range(3):
        _train_step(optimizer, lambda: head(encoder(_inputs())) ** 2)

    assert _changed(encoder_before, _snapshot(encoder)) == []
    assert sorted(_changed(head_before, _snapshot(head))) == ["bias", "weight"]


@pytest.mark.parametrize("name", list(OPTIMIZERS))
def test_unfreezing_lets_the_same_optimizer_train_it_again(name):
    encoder, head = _encoder_and_head()
    optimizer = OPTIMIZERS[name](encoder.parameters() + head.parameters())
    encoder.requires_grad_(False)
    _train_step(optimizer, lambda: head(encoder(_inputs())) ** 2)

    encoder.requires_grad_()
    before = _snapshot(encoder)
    _train_step(optimizer, lambda: head(encoder(_inputs())) ** 2)
    assert sorted(_changed(before, _snapshot(encoder))) == ["bias", "weight"]


def test_a_frozen_module_records_nothing_for_its_parameters():
    encoder, _ = _encoder_and_head()
    encoder.requires_grad_(False)
    assert not encoder(_inputs()).requires_grad


def test_gradients_still_flow_through_a_frozen_module_to_its_input():
    encoder, head = _encoder_and_head()
    encoder.requires_grad_(False)
    features = _inputs().requires_grad_()

    head(encoder(features)).sum().backward()

    assert features.grad is not None
    assert all(p.grad is None for p in encoder.parameters())
    assert all(p.grad is not None for p in head.parameters())


def test_a_frozen_layers_running_statistics_still_update():
    """Freezing is about gradients; buffers are updated by the forward pass
    in training mode, and that is what `eval()` controls."""
    norm = nn.BatchNorm1d(4)
    norm.requires_grad_(False)
    before = norm.state_dict()["running_mean"].numpy().copy()
    norm(_inputs() + 3.0)
    assert not np.array_equal(norm.state_dict()["running_mean"].numpy(), before)


def test_a_frozen_module_saves_and_loads():
    encoder, _ = _encoder_and_head()
    encoder.requires_grad_(False)
    mt.manual_seed(7)
    source = nn.DenseLayer(4, 5)

    encoder.load_state_dict(source.state_dict())

    assert all(not p.requires_grad for p in encoder.parameters())
    for name, values in _snapshot(source).items():
        np.testing.assert_array_equal(_snapshot(encoder)[name], values)


@pytest.mark.parametrize("name", list(OPTIMIZERS))
def test_freeze_load_unfreeze_keeps_the_optimizer_attached(name):
    """Fine-tuning's order of events: build the model and its optimizer,
    freeze the backbone, load pretrained weights into it, unfreeze it later.
    A load gave a frozen parameter storage of its own, so the optimizer built
    first went on stepping the old storage once it was unfrozen."""
    encoder, head = _encoder_and_head()
    optimizer = OPTIMIZERS[name](encoder.parameters() + head.parameters())
    encoder.requires_grad_(False)
    mt.manual_seed(7)
    encoder.load_state_dict(nn.DenseLayer(4, 5).state_dict())
    encoder.requires_grad_()

    before = _snapshot(encoder)
    _train_step(optimizer, lambda: head(encoder(_inputs())) ** 2)
    assert sorted(_changed(before, _snapshot(encoder))) == ["bias", "weight"]


def test_a_frozen_layer_a_pending_backward_pass_reads_is_not_loaded_into():
    """The backward pass of `head(encoder(x))` reads the frozen encoder's
    weight to reach `x`, so writing it first would change `x`'s gradient."""
    encoder, head = _encoder_and_head()
    encoder.requires_grad_(False)
    features = _inputs().requires_grad_()
    loss = head(encoder(features)).sum()

    with pytest.raises(Exception, match="pending backward pass"):
        encoder.load_state_dict(nn.DenseLayer(4, 5).state_dict())

    loss.backward()
    encoder.load_state_dict(nn.DenseLayer(4, 5).state_dict())


# --- `zero_grad` does not make gradients up ----------------------------------


def test_zero_grad_leaves_a_parameter_without_a_gradient_without_one():
    parameters = nn.DenseLayer(4, 3).parameters()
    optimizer = optim.SGD(parameters, lr=0.1)
    optimizer.zero_grad()
    assert all(p.grad is None for p in parameters)


@pytest.mark.parametrize("name", list(OPTIMIZERS))
def test_a_branch_the_loss_did_not_use_is_not_stepped(name):
    encoder, head = _encoder_and_head()
    unused = nn.DenseLayer(4, 5)
    optimizer = OPTIMIZERS[name](
        encoder.parameters() + head.parameters() + unused.parameters()
    )
    before = _snapshot(unused)

    for _ in range(3):
        _train_step(optimizer, lambda: head(encoder(_inputs())) ** 2)

    assert _changed(before, _snapshot(unused)) == []


def test_a_step_with_no_backward_pass_changes_nothing():
    encoder, _ = _encoder_and_head()
    optimizer = optim.SGD(encoder.parameters(), lr=0.1, weight_decay=0.5)
    before = _snapshot(encoder)
    optimizer.zero_grad()
    optimizer.step()
    assert _changed(before, _snapshot(encoder)) == []


def test_zero_grad_still_zeroes_a_gradient_that_exists():
    weight = mt.Tensor([1.0, 2.0], requires_grad=True)
    (weight * 3.0).sum().backward()
    assert weight.grad is not None
    weight.zero_grad()
    assert weight.grad is None or not weight.grad.numpy().any()
