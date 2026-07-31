# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""State dicts are keyed by name, and those names round-trip.

Every layer used to serialize as `param_0`, `param_1`, ... because the naming
hook lived on `Module`, which is supplied by a blanket impl and so cannot be
overridden. A positional checkpoint loads only into a layer whose parameter
*order* is identical, cannot be inspected, and -- for `MultiheadAttention`'s
eight same-shaped tensors or a bidirectional LSTM's sixteen -- cannot be
checked by eye at all.
"""

import os
import tempfile

import numpy as np
import pytest

import minitensor as mt
from minitensor import nn


def _perturb(layer, seed):
    """Move every parameter off its initial value so a load is observable."""
    rng = np.random.default_rng(seed)
    for p in layer.parameters():
        p.copy_(mt.Tensor(rng.standard_normal(tuple(p.shape)).astype(np.float32)))


def _f32(shape, seed):
    return mt.Tensor(
        np.random.default_rng(seed).standard_normal(shape).astype(np.float32)
    )


LAYERS = [
    (
        "DenseLayer",
        lambda: nn.DenseLayer(4, 3),
        lambda: _f32((2, 4), 1),
        {"weight", "bias"},
    ),
    (
        "Conv1d",
        lambda: nn.Conv1d(2, 3, 3),
        lambda: _f32((1, 2, 7), 2),
        {"weight", "bias"},
    ),
    (
        "Conv2d",
        lambda: nn.Conv2d(2, 3, 3),
        lambda: _f32((1, 2, 5, 5), 3),
        {"weight", "bias"},
    ),
    ("LayerNorm", lambda: nn.LayerNorm(4), lambda: _f32((2, 4), 4), {"weight", "bias"}),
    ("RMSNorm", lambda: nn.RMSNorm(4), lambda: _f32((2, 4), 5), {"weight"}),
    (
        "Embedding",
        lambda: nn.Embedding(6, 3),
        lambda: mt.Tensor(np.array([[0, 2], [3, 5]]), dtype="int64"),
        {"weight"},
    ),
    (
        "LSTM",
        lambda: nn.LSTM(3, 4),
        lambda: _f32((4, 2, 3), 6),
        {"weight_ih_l0", "weight_hh_l0", "bias_ih_l0", "bias_hh_l0"},
    ),
    (
        "GRU",
        lambda: nn.GRU(3, 4),
        lambda: _f32((4, 2, 3), 7),
        {"weight_ih_l0", "weight_hh_l0", "bias_ih_l0", "bias_hh_l0"},
    ),
    (
        "MultiheadAttention",
        lambda: nn.MultiheadAttention(4, 2),
        lambda: _f32((2, 3, 4), 8),
        {
            "q_proj",
            "k_proj",
            "v_proj",
            "out_proj",
            "q_bias",
            "k_bias",
            "v_bias",
            "out_bias",
        },
    ),
    (
        "Sequential",
        lambda: nn.Sequential([nn.DenseLayer(4, 3), nn.ReLU(), nn.DenseLayer(3, 2)]),
        lambda: _f32((2, 4), 9),
        # Prefixed by child index; ReLU (index 1) contributes nothing.
        {"0.weight", "0.bias", "2.weight", "2.bias"},
    ),
]


@pytest.mark.parametrize(
    "name,make,make_input,expected", LAYERS, ids=[c[0] for c in LAYERS]
)
def test_state_dict_uses_meaningful_names(name, make, make_input, expected):
    assert set(make().state_dict().parameter_names()) == expected


@pytest.mark.parametrize(
    "name,make,make_input,expected", LAYERS, ids=[c[0] for c in LAYERS]
)
def test_state_dict_round_trips_between_instances_and_through_a_file(
    name, make, make_input, expected
):
    x = make_input()

    source = make()
    _perturb(source, 101)
    want = np.asarray(source.forward(x).numpy())

    target = make()
    _perturb(target, 202)
    # Guard against a vacuous test: the two must genuinely differ first.
    assert not np.allclose(np.asarray(target.forward(x).numpy()), want)

    target.load_state_dict(source.state_dict())
    np.testing.assert_allclose(np.asarray(target.forward(x).numpy()), want, atol=1e-6)

    with tempfile.TemporaryDirectory() as folder:
        path = os.path.join(folder, "model.json")
        source.save(path, format="json")
        restored = make()
        _perturb(restored, 303)
        restored.load_state_dict(nn.Module.load_state_from(path, format="json"))
        np.testing.assert_allclose(
            np.asarray(restored.forward(x).numpy()), want, atol=1e-6
        )


def test_bidirectional_stack_names_every_layer_and_direction():
    # 2 layers x 2 directions x 4 tensors, several sharing a shape: the case a
    # positional scheme makes impossible to verify.
    names = set(
        nn.LSTM(3, 4, num_layers=2, bidirectional=True).state_dict().parameter_names()
    )
    expected = {
        f"{kind}_{gate}_l{layer}{suffix}"
        for kind in ("weight", "bias")
        for gate in ("ih", "hh")
        for layer in (0, 1)
        for suffix in ("", "_reverse")
    }
    assert names == expected


def test_batchnorm_buffers_are_named_and_survive_a_round_trip():
    layer = nn.BatchNorm1d(4)
    x = _f32((8, 4), 11)
    layer.forward(x)
    layer.forward(x)  # move the running stats off their defaults
    layer.eval()
    want = np.asarray(layer.forward(x).numpy())

    state = layer.state_dict()
    assert set(state.buffer_names()) == {"running_mean", "running_var"}

    restored = nn.BatchNorm1d(4)
    restored.load_state_dict(state)
    restored.eval()
    np.testing.assert_allclose(np.asarray(restored.forward(x).numpy()), want, atol=1e-6)
