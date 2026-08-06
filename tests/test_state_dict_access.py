# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""A state dict you cannot read is only half a state dict.

`module.state_dict()` returned an object exposing `parameter_names()`,
`buffer_names()`, `len()` and `in` -- and no way to get a tensor out of it or
put one in. So a checkpoint could be handed straight back to
`load_state_dict()` and never inspected: no reading a saved weight, no copying
one model's weights into another, no building a state dict by hand.

The engine's `StateDict` has had `load_parameter`, `load_buffer`,
`add_parameter` and `add_buffer` since the start; none had a binding.

The tests here check the two namespaces separately, because parameters and
buffers are distinct in the file format and it would be easy to wire
subscripting to only one of them -- which for a BatchNorm model would silently
lose the running statistics.
"""

from __future__ import annotations

import numpy as np
import pytest

import minitensor as mt
from minitensor import nn


def _model():
    return nn.Sequential(
        [nn.DenseLayer(4, 8), nn.BatchNorm1d(8), nn.ReLU(), nn.DenseLayer(8, 2)]
    )


@pytest.fixture
def trained():
    """A model whose BatchNorm buffers have moved away from their defaults."""
    mt.manual_seed(0)
    np.random.seed(0)
    model = _model()
    model.train()
    for _ in range(5):
        model(mt.Tensor((np.random.randn(16, 4) * 2 + 1).astype(np.float32)))
    return model


def test_every_listed_name_can_be_read(trained):
    state = trained.state_dict()
    assert state.parameter_names(), "the fixture should have parameters"
    assert state.buffer_names(), "the fixture should have BatchNorm buffers"

    for name in state.parameter_names():
        assert state.get_parameter(name).numpy().size > 0
    for name in state.buffer_names():
        assert state.get_buffer(name).numpy().size > 0


def test_subscripting_spans_both_namespaces(trained):
    """`__contains__` already covered parameters and buffers, so `__getitem__`
    matches it. Wiring it to parameters alone would silently lose a BatchNorm
    model's running statistics."""
    state = trained.state_dict()

    for name in state.parameter_names():
        assert name in state
        np.testing.assert_array_equal(
            state[name].numpy(), state.get_parameter(name).numpy()
        )
    for name in state.buffer_names():
        assert name in state
        np.testing.assert_array_equal(
            state[name].numpy(), state.get_buffer(name).numpy()
        )


def test_bulk_accessors_agree_with_the_names(trained):
    state = trained.state_dict()

    assert sorted(state.parameters()) == sorted(state.parameter_names())
    assert sorted(state.buffers()) == sorted(state.buffer_names())
    assert len(state) == len(state.parameters()) + len(state.buffers())

    for name, tensor in state.parameters().items():
        np.testing.assert_array_equal(tensor.numpy(), state[name].numpy())


def test_read_values_match_the_live_module(trained):
    """The point of reading a state dict is that it holds the actual weights."""
    state = trained.state_dict()
    live = {id(p): p for p in trained.parameters()}

    stored = [state[name].numpy() for name in state.parameter_names()]
    for parameter in live.values():
        assert any(
            candidate.shape == parameter.numpy().shape
            and np.array_equal(candidate, parameter.numpy())
            for candidate in stored
        ), "a live parameter has no matching entry in the state dict"


def test_buffers_carry_the_running_statistics(trained):
    """BatchNorm's running mean and variance are buffers, not parameters, and
    they must have moved from their initial 0/1."""
    state = trained.state_dict()

    mean = state["1.running_mean"].numpy()
    var = state["1.running_var"].numpy()
    assert not np.allclose(mean, 0.0), "running_mean never updated"
    assert not np.allclose(var, 1.0), "running_var never updated"


def test_a_state_dict_can_be_built_by_hand():
    state = mt.serialization.StateDict()
    assert len(state) == 0

    weight = mt.randn(2, 3)
    state.add_parameter("weight", weight)
    state.add_buffer("running_mean", mt.zeros(3))

    assert len(state) == 2
    assert state.parameter_names() == ["weight"]
    assert state.buffer_names() == ["running_mean"]
    np.testing.assert_array_equal(state["weight"].numpy(), weight.numpy())


def test_adding_the_same_name_twice_replaces_it():
    state = mt.serialization.StateDict()
    state.add_parameter("w", mt.zeros(3))
    state.add_parameter("w", mt.ones(3))

    assert len(state) == 1
    np.testing.assert_array_equal(state["w"].numpy(), np.ones(3, dtype=np.float32))


@pytest.mark.parametrize("dtype", ["float32", "float64", "int32", "int64", "bool"])
def test_every_dtype_survives_a_state_dict(dtype):
    values = (
        np.array([True, False, True]) if dtype == "bool" else np.arange(3).astype(dtype)
    )
    tensor = mt.Tensor(values, dtype=dtype)

    state = mt.serialization.StateDict()
    state.add_parameter("v", tensor)

    read = state["v"]
    assert read.dtype == dtype
    np.testing.assert_array_equal(read.numpy(), values)


def test_a_missing_name_says_what_is_there():
    state = mt.serialization.StateDict()
    state.add_parameter("weight", mt.zeros(2))

    with pytest.raises(KeyError) as excinfo:
        state["bias"]
    message = str(excinfo.value)
    assert "bias" in message
    assert "weight" in message, "the message should list what the dict does hold"


def test_get_parameter_does_not_reach_buffers_and_vice_versa(trained):
    """Keeping the two namespaces distinct is what lets a model with a
    parameter and a buffer of the same name round-trip."""
    state = trained.state_dict()
    parameter = state.parameter_names()[0]
    buffer = state.buffer_names()[0]

    with pytest.raises(Exception):
        state.get_buffer(parameter)
    with pytest.raises(Exception):
        state.get_parameter(buffer)


def test_weights_can_be_copied_between_models(trained, tmp_path):
    """The use case this existed to enable: read a checkpoint's tensors and do
    something with them other than hand them back."""
    path = str(tmp_path / "model.bin")
    trained.save(path)

    loaded = type(trained).load_state_from(path)
    names = loaded.parameter_names() + loaded.buffer_names()

    rebuilt = mt.serialization.StateDict()
    for name in loaded.parameter_names():
        rebuilt.add_parameter(name, loaded.get_parameter(name))
    for name in loaded.buffer_names():
        rebuilt.add_buffer(name, loaded.get_buffer(name))

    assert sorted(rebuilt.parameter_names() + rebuilt.buffer_names()) == sorted(names)

    target = _model()
    target.load_state_dict(rebuilt)
    target.eval()
    trained.eval()

    probe = mt.Tensor(np.random.randn(3, 4).astype(np.float32))
    np.testing.assert_array_equal(target(probe).numpy(), trained(probe).numpy())
