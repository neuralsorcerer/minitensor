# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""`copy.deepcopy(layer)` gives an independent layer of the same class.

It raised `TypeError`, so a frozen target network, a weight average kept beside
the model it averages, or a snapshot taken before an experiment all had to be
built by hand and loaded with `load_state_dict`.

Independent is the property that matters, and it is not what cloning the layer
gives. A parameter is updated in place and an optimizer keys its state by
tensor identity, so a copy that shared either would be trained by an optimizer
built over the original. Every parameter and buffer of the copy has storage and
an identity of its own; the tests below train each side and check the other.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest

import minitensor as mt

nn = mt.nn
optim = mt.optim

CATALOGUE = {
    "DenseLayer": lambda: nn.DenseLayer(4, 3),
    "ReLU": lambda: nn.ReLU(),
    "Sigmoid": lambda: nn.Sigmoid(),
    "Tanh": lambda: nn.Tanh(),
    "Softmax": lambda: nn.Softmax(),
    "LeakyReLU": lambda: nn.LeakyReLU(),
    "ELU": lambda: nn.ELU(),
    "GELU": lambda: nn.GELU(),
    "Dropout": lambda: nn.Dropout(0.5),
    "Dropout2d": lambda: nn.Dropout2d(0.5),
    "Conv1d": lambda: nn.Conv1d(3, 4, 3),
    "Conv2d": lambda: nn.Conv2d(3, 4, 3),
    "ConvTranspose1d": lambda: nn.ConvTranspose1d(3, 4, 3),
    "ConvTranspose2d": lambda: nn.ConvTranspose2d(3, 4, 3),
    "MaxPool1d": lambda: nn.MaxPool1d(2),
    "AvgPool1d": lambda: nn.AvgPool1d(2),
    "MaxPool2d": lambda: nn.MaxPool2d(2),
    "AvgPool2d": lambda: nn.AvgPool2d(2),
    "Upsample": lambda: nn.Upsample(scale_factor=2.0),
    "AdaptiveAvgPool1d": lambda: nn.AdaptiveAvgPool1d(2),
    "AdaptiveAvgPool2d": lambda: nn.AdaptiveAvgPool2d(2),
    "AdaptiveMaxPool1d": lambda: nn.AdaptiveMaxPool1d(2),
    "AdaptiveMaxPool2d": lambda: nn.AdaptiveMaxPool2d(2),
    "BatchNorm1d": lambda: nn.BatchNorm1d(4),
    "BatchNorm2d": lambda: nn.BatchNorm2d(3),
    "Embedding": lambda: nn.Embedding(10, 4),
    "LayerNorm": lambda: nn.LayerNorm([4]),
    "RMSNorm": lambda: nn.RMSNorm([4]),
    "MultiheadAttention": lambda: nn.MultiheadAttention(4, 2),
    "Sequential": lambda: nn.Sequential(
        [nn.DenseLayer(4, 6), nn.BatchNorm1d(6), nn.ReLU(), nn.DenseLayer(6, 3)]
    ),
    "LSTM": lambda: nn.LSTM(4, 3),
    "GRU": lambda: nn.GRU(4, 3),
}

# A parameterised layer for each shape of input the training tests need.
TRAINABLE = {
    "DenseLayer": (lambda: nn.DenseLayer(4, 3), (8, 4)),
    "Sequential": (CATALOGUE["Sequential"], (8, 4)),
    "Conv2d": (lambda: nn.Conv2d(3, 4, 3), (2, 3, 6, 6)),
    "BatchNorm1d": (lambda: nn.BatchNorm1d(4), (8, 4)),
    "LayerNorm": (lambda: nn.LayerNorm([4]), (8, 4)),
}


def _snapshot(module):
    return {name: t.numpy().copy() for name, t in module.state_dict().items()}


def _assert_same(a, b):
    assert sorted(a) == sorted(b)
    for name in a:
        np.testing.assert_array_equal(a[name], b[name], err_msg=name)


def _changed(a, b):
    return any(not np.array_equal(a[name], b[name]) for name in a)


def _train_step(module, shape, seed=0):
    rng = np.random.default_rng(seed)
    features = mt.from_numpy(rng.standard_normal(shape).astype(np.float32))
    optimizer = optim.SGD(module.parameters(), lr=0.5)
    optimizer.zero_grad()
    module(features).sum().backward()
    optimizer.step()


def test_the_catalogue_is_every_built_in_layer():
    """A layer added to `nn` without being taught to `copy.deepcopy` would
    raise; this is where that shows up."""
    built_in = {
        name
        for name in dir(nn)
        if isinstance(getattr(nn, name), type)
        and issubclass(getattr(nn, name), nn.Module)
        and name != "Module"
    }
    assert built_in == set(CATALOGUE)


@pytest.mark.parametrize("name", list(CATALOGUE))
def test_the_copy_is_the_same_class_with_the_same_state(name):
    mt.manual_seed(0)
    original = CATALOGUE[name]()
    copied = copy.deepcopy(original)

    assert type(copied) is type(original)
    _assert_same(_snapshot(original), _snapshot(copied))
    assert [p.requires_grad for p in copied.parameters()] == [
        p.requires_grad for p in original.parameters()
    ]


@pytest.mark.parametrize("name", list(TRAINABLE))
def test_training_the_copy_leaves_the_original_alone(name):
    build, shape = TRAINABLE[name]
    mt.manual_seed(0)
    original = build()
    copied = copy.deepcopy(original)
    before = _snapshot(original)

    _train_step(copied, shape)

    assert _changed(before, _snapshot(copied))
    _assert_same(before, _snapshot(original))
    assert all(p.grad is None for p in original.parameters())


@pytest.mark.parametrize("name", list(TRAINABLE))
def test_training_the_original_leaves_the_copy_alone(name):
    build, shape = TRAINABLE[name]
    mt.manual_seed(0)
    original = build()
    copied = copy.deepcopy(original)
    before = _snapshot(copied)

    _train_step(original, shape)

    _assert_same(before, _snapshot(copied))
    assert all(p.grad is None for p in copied.parameters())


def test_running_statistics_are_the_copys_own():
    """Buffers are written in place by a training-mode forward, not by an
    optimizer, so they need their own storage as much as the parameters do."""
    original = nn.BatchNorm1d(3)
    copied = copy.deepcopy(original)
    before = original.state_dict()["running_mean"].numpy().copy()

    copied(mt.from_numpy(np.full((4, 3), 5.0, np.float32)))

    np.testing.assert_array_equal(original.state_dict()["running_mean"].numpy(), before)
    assert not np.array_equal(copied.state_dict()["running_mean"].numpy(), before)


def test_a_copy_made_under_no_grad_stays_trainable():
    """`no_grad` is where a snapshot is usually taken, and a tensor built there
    does not require a gradient unless told to after the fact."""
    original = nn.DenseLayer(4, 3)
    with mt.no_grad():
        copied = copy.deepcopy(original)
    assert all(p.requires_grad for p in copied.parameters())
    before = _snapshot(copied)
    _train_step(copied, (2, 4))
    assert _changed(before, _snapshot(copied))


def test_the_copy_sits_inside_a_deep_copied_structure():
    state = {"model": nn.DenseLayer(4, 3), "step": 7}
    copied = copy.deepcopy(state)
    assert copied["step"] == 7
    _assert_same(_snapshot(state["model"]), _snapshot(copied["model"]))
