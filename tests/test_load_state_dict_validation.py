# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""`load_state_dict` reports what it could not load instead of accepting it.

It used to accept everything. Both lookups were written `if let Ok(..)`, which
throws the error away, and each failure was silent in a different way:

- **A name the state dict does not carry** -- a renamed parameter, a truncated
  checkpoint, an empty state dict -- left that slot holding whatever it already
  had, and the call reported success. Resuming training from such a checkpoint
  quietly continued from the initialisation. Nothing distinguishes that from a
  run that simply is not converging.

- **A name it does carry at the wrong shape** replaced the slot with that
  tensor. The layer came out structurally inconsistent -- a `DenseLayer(4, 3)`
  whose weight is `(7, 9)` -- and the load still reported success. The failure
  surfaced at the next forward pass, as a shape error that never mentions
  loading:

      Shape mismatch: expected [7, 4], got [2, 4]

Both are checked now, before anything is written, so a rejected load leaves the
layer exactly as it was. That matters for the caller who catches the error and
falls back: they get the model they had, not one holding half a checkpoint.

Every problem is reported at once, with the qualified name a nested module gives
it (`1.bias`), rather than surfacing one per attempt.
"""

from __future__ import annotations

import numpy as np
import pytest

import minitensor as mt

nn = mt.nn
S = mt.serialization


def _source():
    mt.manual_seed(0)
    layer = nn.DenseLayer(4, 3)
    marked = S.StateDict()
    marked.add_parameter("weight", mt.Tensor(np.full((3, 4), 7.0, np.float32)))
    marked.add_parameter("bias", mt.Tensor(np.full(3, 9.0, np.float32)))
    layer.load_state_dict(marked)
    return layer


def _target():
    mt.manual_seed(1)
    return nn.DenseLayer(4, 3)


def _snapshot(module):
    return {name: tensor.numpy().copy() for name, tensor in module.state_dict().items()}


def _state(**tensors):
    state = S.StateDict()
    for name, array in tensors.items():
        state.add_parameter(name, mt.Tensor(np.asarray(array, np.float32)))
    return state


# --- what must still work ---------------------------------------------------


def test_a_matching_state_dict_loads():
    source, target = _source(), _target()
    target.load_state_dict(source.state_dict())
    for name, values in _snapshot(source).items():
        np.testing.assert_array_equal(_snapshot(target)[name], values)


def test_a_round_trip_through_a_nested_module_loads():
    mt.manual_seed(0)
    model = nn.Sequential([nn.DenseLayer(4, 3), nn.BatchNorm1d(3)])
    mt.manual_seed(5)
    restored = nn.Sequential([nn.DenseLayer(4, 3), nn.BatchNorm1d(3)])

    restored.load_state_dict(model.state_dict())
    for name, values in _snapshot(model).items():
        np.testing.assert_array_equal(_snapshot(restored)[name], values)


def test_buffers_round_trip_too():
    """BatchNorm's running statistics travel as buffers, on the indexed path."""
    mt.manual_seed(0)
    layer = nn.BatchNorm1d(4)
    layer(mt.Tensor(np.random.default_rng(0).standard_normal((8, 4)), dtype="float32"))

    restored = nn.BatchNorm1d(4)
    restored.load_state_dict(layer.state_dict())
    np.testing.assert_array_equal(
        _snapshot(restored)["running_mean"], _snapshot(layer)["running_mean"]
    )


# --- a name the state dict does not have ------------------------------------


@pytest.mark.parametrize(
    "label,build",
    [
        ("misspelled", lambda src: _state(wieght=np.zeros((3, 4)), bias=np.zeros(3))),
        ("bias absent", lambda src: _state(weight=np.zeros((3, 4)))),
        ("weight absent", lambda src: _state(bias=np.zeros(3))),
        ("empty", lambda src: S.StateDict()),
    ],
)
def test_a_missing_entry_is_reported(label, build):
    target = _target()
    with pytest.raises(Exception) as excinfo:
        target.load_state_dict(build(_source()))
    assert "missing" in str(excinfo.value), str(excinfo.value)


def test_the_message_names_every_missing_entry():
    with pytest.raises(Exception) as excinfo:
        _target().load_state_dict(S.StateDict())
    message = str(excinfo.value)
    assert "weight" in message and "bias" in message


# --- a name it has, at the wrong shape --------------------------------------


def test_a_wrong_shape_is_reported_with_both_shapes():
    with pytest.raises(Exception) as excinfo:
        _target().load_state_dict(
            _state(weight=np.zeros((7, 9)), bias=np.zeros(9)),
        )
    message = str(excinfo.value)
    assert "wrong shape" in message
    assert "[3, 4]" in message and "[7, 9]" in message


def test_a_wrong_shape_no_longer_reaches_the_forward_pass():
    """This is the failure it used to become: the load succeeded and the layer
    broke later somewhere that says nothing about checkpoints."""
    target = _target()
    with pytest.raises(Exception):
        target.load_state_dict(_state(weight=np.zeros((7, 9)), bias=np.zeros(9)))

    out = target(mt.Tensor(np.ones((2, 4), np.float32)))
    assert tuple(out.shape_vec()) == (2, 3)


def test_both_kinds_of_problem_are_reported_together():
    with pytest.raises(Exception) as excinfo:
        _target().load_state_dict(_state(weight=np.zeros((7, 9))))
    message = str(excinfo.value)
    assert "missing" in message and "wrong shape" in message


def test_a_nested_mismatch_is_named_by_its_path():
    mt.manual_seed(0)
    model = nn.Sequential([nn.DenseLayer(4, 3), nn.BatchNorm1d(3)])
    wider = nn.Sequential([nn.DenseLayer(4, 3), nn.BatchNorm1d(5)])

    with pytest.raises(Exception) as excinfo:
        wider.load_state_dict(model.state_dict())
    assert "1." in str(excinfo.value), str(excinfo.value)


# --- a rejected load changes nothing ----------------------------------------


@pytest.mark.parametrize(
    "build",
    [
        lambda: S.StateDict(),
        lambda: _state(weight=np.zeros((3, 4))),
        lambda: _state(weight=np.zeros((7, 9)), bias=np.zeros(9)),
        lambda: _state(wieght=np.zeros((3, 4)), bias=np.zeros(3)),
    ],
    ids=["empty", "half", "wrong_shape", "misspelled"],
)
def test_a_rejected_load_leaves_the_module_alone(build):
    """`weight` sorts before `bias` in neither order reliably, so a load that
    writes as it goes would leave one of them changed. Nothing may be."""
    target = _target()
    before = _snapshot(target)

    with pytest.raises(Exception):
        target.load_state_dict(build())

    after = _snapshot(target)
    assert sorted(before) == sorted(after)
    for name, values in before.items():
        np.testing.assert_array_equal(after[name], values, err_msg=name)


def test_the_shapes_survive_a_rejected_load():
    target = _target()
    with pytest.raises(Exception):
        target.load_state_dict(_state(weight=np.zeros((7, 9)), bias=np.zeros(9)))

    state = target.state_dict()
    assert tuple(state["weight"].shape_vec()) == (3, 4)
    assert tuple(state["bias"].shape_vec()) == (3,)


def test_the_module_still_trains_after_a_rejected_load():
    target = _target()
    with pytest.raises(Exception):
        target.load_state_dict(S.StateDict())

    out = target(mt.Tensor(np.ones((2, 4), np.float32)))
    out.sum().backward()
    assert all(p.grad is not None for p in target.parameters())
