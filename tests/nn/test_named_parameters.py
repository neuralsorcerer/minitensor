# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""`module.named_parameters()` pairs each parameter with its `state_dict` name.

The names existed -- `state_dict()` saves under them -- but a parameter could
only be reached as a position in `parameters()`, so finding "the second
block's bias" meant counting.
"""

from __future__ import annotations

import numpy as np
import pytest

import minitensor as mt

nn = mt.nn

CATALOGUE = {
    "DenseLayer": lambda: nn.DenseLayer(4, 3),
    "Conv1d": lambda: nn.Conv1d(3, 4, 3),
    "Conv2d": lambda: nn.Conv2d(3, 4, 3),
    "ConvTranspose1d": lambda: nn.ConvTranspose1d(3, 4, 3),
    "ConvTranspose2d": lambda: nn.ConvTranspose2d(3, 4, 3),
    "BatchNorm1d": lambda: nn.BatchNorm1d(4),
    "BatchNorm2d": lambda: nn.BatchNorm2d(3),
    "LayerNorm": lambda: nn.LayerNorm([4]),
    "RMSNorm": lambda: nn.RMSNorm([4]),
    "Embedding": lambda: nn.Embedding(10, 4),
    "MultiheadAttention": lambda: nn.MultiheadAttention(4, 2),
    "LSTM": lambda: nn.LSTM(4, 3, num_layers=2, bidirectional=True),
    "GRU": lambda: nn.GRU(4, 3),
    "ReLU": lambda: nn.ReLU(),
    "Sequential": lambda: nn.Sequential(
        [nn.DenseLayer(4, 3), nn.Sequential([nn.BatchNorm1d(3), nn.ReLU()])]
    ),
}


@pytest.mark.parametrize("name", list(CATALOGUE))
def test_the_names_are_the_state_dicts_and_the_order_is_parameters(name):
    module = CATALOGUE[name]()
    named = module.named_parameters()
    state = module.state_dict()

    names = [entry for entry, _ in named]
    assert len(set(names)) == len(names)
    for entry, tensor in named:
        np.testing.assert_array_equal(tensor.numpy(), state[entry].numpy())
    assert [t.numpy().tobytes() for _, t in named] == [
        t.numpy().tobytes() for t in module.parameters()
    ]


def test_a_nested_part_is_found_by_its_path():
    model = nn.Sequential([nn.DenseLayer(4, 3), nn.Sequential([nn.DenseLayer(3, 2)])])
    names = [entry for entry, _ in model.named_parameters()]
    assert names == ["0.weight", "0.bias", "1.0.weight", "1.0.bias"]
    bias = dict(model.named_parameters())["1.0.bias"]
    np.testing.assert_array_equal(bias.numpy(), model[1][0].bias.numpy())


def test_a_named_handle_is_the_parameter():
    """Like `parameters()`, a handle an optimizer can step."""
    layer = nn.DenseLayer(4, 3)
    weight = dict(layer.named_parameters())["weight"]
    optimizer = mt.optim.SGD([weight], lr=0.5)
    before = layer.weight.numpy().copy()
    optimizer.zero_grad()
    layer(mt.ones(2, 4)).sum().backward()
    optimizer.step()
    assert not np.array_equal(layer.weight.numpy(), before)
