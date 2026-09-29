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

- **The right shape in another dtype** -- a float64 checkpoint loaded into a
  float32 layer -- replaced the slot too, turning the layer float64 in place.
  The next forward pass refused its float32 input with a dtype error that
  again says nothing about loading. Converting on the way in would choose the
  precision for the caller, so this is reported like the other two.

- **An entry the layer has no slot for** was ignored. A checkpoint of a
  deeper model loaded into a shallower one: the layers the two shared were
  filled, the rest of the checkpoint was dropped, and the load reported success
  on weights that were never the ones trained together.

All four are checked now, before anything is written, so a rejected load leaves the
layer exactly as it was. That matters for the caller who catches the error and
falls back: they get the model they had, not one holding half a checkpoint.

Every problem is reported at once, with the qualified name a nested module gives
it (`1.bias`), rather than surfacing one per attempt.

A load that passes writes the values into the parameters the layer already
has. It used to swap new tensors in, which left an optimizer built before the
load stepping tensors the layer no longer held.
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


def _state(dtype="float32", **tensors):
    state = S.StateDict()
    for name, array in tensors.items():
        state.add_parameter(name, mt.from_numpy(np.asarray(array, dtype)))
    return state


def _differently_initialised(build):
    mt.manual_seed(0)
    source = build()
    mt.manual_seed(99)
    target = build()
    return source, target


def _changed(before, after):
    return any(not np.array_equal(before[name], after[name]) for name in before)


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


def test_a_wrong_dtype_is_reported_with_both_dtypes():
    with pytest.raises(Exception) as excinfo:
        _target().load_state_dict(
            _state("float64", weight=np.zeros((3, 4)), bias=np.zeros(3)),
        )
    message = str(excinfo.value)
    assert "wrong dtype" in message
    assert "bias (expected float32, got float64)" in message
    assert "weight (expected float32, got float64)" in message


def test_a_float64_checkpoint_no_longer_reaches_the_forward_pass(tmp_path):
    """Through a file, the way it happens: a float64 model saved and loaded
    into a float32 one. The load used to succeed and turn the layer float64."""
    path = str(tmp_path / "wide.json")
    nn.DenseLayer(4, 3, dtype="float64").save(path)
    target = _target()

    with pytest.raises(Exception, match="wrong dtype"):
        target.load_state_dict(nn.DenseLayer.load_state_from(path))

    out = target(mt.Tensor(np.ones((2, 4), np.float32)))
    assert out.dtype == "float32"


def test_integer_running_statistics_are_refused():
    norm = nn.BatchNorm1d(3)
    state = norm.state_dict()
    integral = S.StateDict()
    for name in state.parameter_names():
        integral.add_parameter(name, state.get_parameter(name))
    for name in state.buffer_names():
        integral.add_buffer(name, mt.from_numpy(np.zeros(3, np.int64)))

    with pytest.raises(Exception) as excinfo:
        norm.load_state_dict(integral)
    message = str(excinfo.value)
    assert "running_mean (expected float32, got int64)" in message
    assert "running_var (expected float32, got int64)" in message


def test_every_kind_of_problem_is_reported_together():
    target = nn.Sequential([nn.DenseLayer(4, 3), nn.DenseLayer(3, 2)])
    renamed = S.StateDict()
    renamed.add_parameter("0.weight", mt.from_numpy(np.zeros((7, 9), np.float32)))
    renamed.add_parameter("0.bias", mt.from_numpy(np.zeros(3, np.float64)))
    renamed.add_parameter("1.weight", mt.from_numpy(np.zeros((2, 3), np.float32)))

    with pytest.raises(Exception) as excinfo:
        target.load_state_dict(renamed)
    message = str(excinfo.value)
    assert "missing from the state dict: 1.bias" in message
    assert "wrong shape: 0.weight" in message
    assert "wrong dtype: 0.bias" in message


def test_loading_plain_tensors_keeps_the_layer_trainable():
    """A load sets values, not whether a parameter trains. A state dict built
    from tensors that do not require a gradient -- weights read from a file of
    another format, say -- used to freeze every parameter it reached, and the
    layer stopped training without a word."""
    target = _target()
    target.load_state_dict(_state(weight=np.ones((3, 4)), bias=np.zeros(3)))
    assert all(p.requires_grad for p in target.parameters())

    before = _snapshot(target)
    optimizer = mt.optim.SGD(target.parameters(), lr=0.1)
    optimizer.zero_grad()
    target(mt.Tensor(np.ones((2, 4), np.float32))).sum().backward()
    optimizer.step()
    after = _snapshot(target)
    assert not np.array_equal(before["weight"], after["weight"])
    assert not np.array_equal(before["bias"], after["bias"])


# --- an entry the module has no slot for -----------------------------------


def test_an_unexpected_entry_is_reported():
    with pytest.raises(Exception, match="not in this module: scale"):
        _target().load_state_dict(
            _state(weight=np.zeros((3, 4)), bias=np.zeros(3), scale=np.zeros(3))
        )


def test_a_deeper_checkpoint_does_not_load_into_a_shallower_model():
    """Every layer the two share matches, so nothing else would catch it."""
    deeper = nn.Sequential(
        [nn.DenseLayer(4, 3), nn.DenseLayer(3, 3), nn.DenseLayer(3, 2)]
    )
    shallower = nn.Sequential([nn.DenseLayer(4, 3), nn.DenseLayer(3, 3)])
    before = _snapshot(shallower)

    with pytest.raises(Exception, match=r"not in this module: 2\.bias, 2\.weight"):
        shallower.load_state_dict(deeper.state_dict())
    for name, values in before.items():
        np.testing.assert_array_equal(_snapshot(shallower)[name], values, err_msg=name)


def test_an_entry_under_the_wrong_namespace_says_which_one_it_belongs_to():
    norm = nn.BatchNorm1d(3)
    state = norm.state_dict()
    swapped = S.StateDict()
    for name in state.parameter_names():
        swapped.add_parameter(name, state.get_parameter(name))
    swapped.add_parameter("running_mean", state.get_buffer("running_mean"))
    swapped.add_buffer("running_var", state.get_buffer("running_var"))

    with pytest.raises(Exception) as excinfo:
        norm.load_state_dict(swapped)
    message = str(excinfo.value)
    assert "missing from the state dict: running_mean" in message
    assert "running_mean (given as a parameter; it is a buffer)" in message


# --- a load writes into the parameters the layer already has ----------------

# An optimizer holds its own handles to the parameters and keys them by
# identity. A load that swapped new tensors into the layer left those handles
# on storage the layer no longer read: every step succeeded and the model never
# moved. Building the optimizer first and restoring a checkpoint second is the
# ordinary way to resume, so this was the common case, not a corner.

TRAINED = {
    "DenseLayer": (lambda: nn.DenseLayer(4, 3), (2, 4)),
    "Sequential": (
        lambda: nn.Sequential([nn.DenseLayer(4, 3), nn.BatchNorm1d(3), nn.ReLU()]),
        (4, 4),
    ),
    "Conv2d": (lambda: nn.Conv2d(3, 4, 3), (2, 3, 6, 6)),
    "LayerNorm": (lambda: nn.LayerNorm([4]), (2, 4)),
    "LSTM": (lambda: nn.LSTM(4, 3), (5, 2, 4)),
}


def _inputs(shape):
    rng = np.random.default_rng(3)
    return mt.from_numpy(rng.standard_normal(shape).astype(np.float32))


def _output(value):
    return value[0] if isinstance(value, tuple) else value


@pytest.mark.parametrize("name", list(TRAINED))
@pytest.mark.parametrize("make", [mt.optim.SGD, mt.optim.Adam], ids=["SGD", "Adam"])
def test_an_optimizer_built_before_the_load_trains_the_loaded_layer(name, make):
    build, shape = TRAINED[name]
    source, target = _differently_initialised(build)
    optimizer = make(target.parameters(), lr=0.1)

    target.load_state_dict(source.state_dict())
    loaded = _snapshot(target)
    optimizer.zero_grad()
    (_output(target(_inputs(shape))) ** 2).sum().backward()
    optimizer.step()

    assert _changed(loaded, _snapshot(target)), "the step did not reach the layer"


@pytest.mark.parametrize("name", list(TRAINED))
def test_handles_taken_before_the_load_see_the_loaded_values(name):
    source, target = _differently_initialised(TRAINED[name][0])
    handles = target.parameters()

    target.load_state_dict(source.state_dict())

    for handle, param in zip(handles, source.parameters(), strict=True):
        np.testing.assert_array_equal(handle.numpy(), param.numpy())


def test_a_load_while_a_backward_pass_is_pending_is_refused():
    """Writing a parameter that a recorded forward pass read would change the
    gradients its backward pass produces. A load is refused for that the way
    every in-place write is -- before anything is written."""
    source, target = _source(), _target()
    before = _snapshot(target)
    loss = target(mt.Tensor(np.ones((2, 4), np.float32))).sum()

    with pytest.raises(Exception, match="pending backward pass"):
        target.load_state_dict(source.state_dict())
    for name, values in before.items():
        np.testing.assert_array_equal(_snapshot(target)[name], values, err_msg=name)

    loss.backward()
    target.load_state_dict(source.state_dict())
    for name, values in _snapshot(source).items():
        np.testing.assert_array_equal(_snapshot(target)[name], values, err_msg=name)


def test_a_load_after_the_forward_pass_is_dropped_goes_through():
    source, target = _source(), _target()
    target(mt.Tensor(np.ones((2, 4), np.float32))).sum()

    target.load_state_dict(source.state_dict())
    for name, values in _snapshot(source).items():
        np.testing.assert_array_equal(_snapshot(target)[name], values, err_msg=name)


# --- a rejected load changes nothing ----------------------------------------


@pytest.mark.parametrize(
    "build",
    [
        lambda: S.StateDict(),
        lambda: _state(weight=np.zeros((3, 4))),
        lambda: _state(weight=np.zeros((7, 9)), bias=np.zeros(9)),
        lambda: _state(wieght=np.zeros((3, 4)), bias=np.zeros(3)),
        lambda: _state("float64", weight=np.zeros((3, 4)), bias=np.zeros(3)),
        lambda: _state(weight=np.zeros((3, 4)), bias=np.zeros(3), scale=np.zeros(3)),
    ],
    ids=["empty", "half", "wrong_shape", "misspelled", "wrong_dtype", "unexpected"],
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


# --- the whole layer catalogue, now that a bad load is detectable ------------

# `state_dict` names parameters and buffers from `named_parameters` /
# `named_buffers` when a layer provides them and falls back to `param_{i}` /
# `buffer_{i}` when it does not -- and the two halves of a layer can disagree,
# as BatchNorm does, naming its parameters and indexing its buffers. While a
# load accepted anything, a layer whose save and load disagreed on names
# round-tripped to silence. It now raises, which turns this into a real check
# that the fallback is symmetric for every layer rather than a formality.

CATALOGUE = {
    "DenseLayer": lambda: nn.DenseLayer(4, 3),
    "Conv1d": lambda: nn.Conv1d(3, 4, 3),
    "Conv2d": lambda: nn.Conv2d(3, 4, 3),
    "BatchNorm1d": lambda: nn.BatchNorm1d(4),
    "BatchNorm2d": lambda: nn.BatchNorm2d(3),
    "LayerNorm": lambda: nn.LayerNorm([4]),
    "RMSNorm": lambda: nn.RMSNorm([4]),
    "Embedding": lambda: nn.Embedding(10, 4),
    "MultiheadAttention": lambda: nn.MultiheadAttention(4, 2),
    "LSTM": lambda: nn.LSTM(4, 3),
    "GRU": lambda: nn.GRU(4, 3),
    "Dropout": lambda: nn.Dropout(0.5),
    "ReLU": lambda: nn.ReLU(),
    "Sequential": lambda: nn.Sequential(
        [nn.DenseLayer(4, 3), nn.BatchNorm1d(3), nn.ReLU()]
    ),
}


@pytest.mark.parametrize("name", list(CATALOGUE), ids=list(CATALOGUE))
def test_every_layer_round_trips_in_memory(name):
    source, target = _differently_initialised(CATALOGUE[name])
    expected = _snapshot(source)

    target.load_state_dict(source.state_dict())

    assert sorted(_snapshot(target)) == sorted(expected)
    for entry, values in expected.items():
        np.testing.assert_array_equal(_snapshot(target)[entry], values, err_msg=entry)


@pytest.mark.parametrize("name", list(CATALOGUE), ids=list(CATALOGUE))
def test_every_layer_round_trips_through_a_file(name, tmp_path):
    source, target = _differently_initialised(CATALOGUE[name])
    expected = _snapshot(source)

    path = str(tmp_path / "model.bin")
    source.save(path)
    target.load_state_dict(type(target).load_state_from(path))

    for entry, values in expected.items():
        np.testing.assert_array_equal(_snapshot(target)[entry], values, err_msg=entry)


# --- a plain mapping --------------------------------------------------------

# A state dict reads as a mapping, so `dict(state)` or a comprehension over
# `state.items()` is how one gets changed -- and what that makes has to load
# back. A mapping does not say which entries are buffers; the module does.


@pytest.mark.parametrize("name", list(CATALOGUE), ids=list(CATALOGUE))
def test_every_layer_round_trips_through_a_plain_dict(name):
    source, target = _differently_initialised(CATALOGUE[name])
    expected = _snapshot(source)

    target.load_state_dict(dict(source.state_dict()))

    for entry, values in expected.items():
        np.testing.assert_array_equal(_snapshot(target)[entry], values, err_msg=entry)


def test_running_statistics_in_a_dict_are_loaded_as_buffers():
    mt.manual_seed(0)
    trained = nn.BatchNorm1d(3)
    trained(
        mt.from_numpy(
            np.random.default_rng(0).standard_normal((8, 3)).astype(np.float32)
        )
    )
    fresh = nn.BatchNorm1d(3)

    fresh.load_state_dict(
        {name: tensor for name, tensor in trained.state_dict().items()}
    )
    np.testing.assert_array_equal(
        fresh.state_dict().get_buffer("running_mean").numpy(),
        trained.state_dict().get_buffer("running_mean").numpy(),
    )


def test_a_checkpoint_cast_through_a_dict_loads_into_the_wider_model():
    """The documented way to load a float32 checkpoint into a float64 model."""
    mt.manual_seed(0)
    narrow = nn.Sequential([nn.DenseLayer(4, 3), nn.BatchNorm1d(3)])
    wide = nn.Sequential(
        [nn.DenseLayer(4, 3, dtype="float64"), nn.BatchNorm1d(3, dtype="float64")]
    )

    wide.load_state_dict(
        {name: tensor.astype("float64") for name, tensor in narrow.state_dict().items()}
    )
    for name, values in _snapshot(narrow).items():
        assert _snapshot(wide)[name].dtype == np.float64
        np.testing.assert_array_equal(_snapshot(wide)[name], values.astype(np.float64))


def test_a_dict_is_checked_like_a_state_dict():
    target = _target()
    before = _snapshot(target)
    checkpoint = dict(_source().state_dict())

    with pytest.raises(Exception, match="missing from the state dict: bias"):
        target.load_state_dict({"weight": checkpoint["weight"]})
    with pytest.raises(Exception, match="not in this module: scale"):
        target.load_state_dict({**checkpoint, "scale": mt.zeros(3)})
    for name, values in before.items():
        np.testing.assert_array_equal(_snapshot(target)[name], values, err_msg=name)


@pytest.mark.parametrize(
    "state,message",
    [
        (
            [("weight", mt.zeros(3, 4))],
            "a StateDict or a mapping from name to tensor, not list",
        ),
        (
            {"weight": np.zeros((3, 4)), "bias": mt.zeros(3)},
            '"weight" holds a ndarray, not a Tensor',
        ),
        ({0: mt.zeros(3)}, "every name must be a str, not int"),
    ],
    ids=["not_a_mapping", "not_a_tensor", "not_a_name"],
)
def test_what_is_not_a_mapping_of_tensors_is_refused_by_name(state, message):
    with pytest.raises(TypeError) as excinfo:
        _target().load_state_dict(state)
    assert message in str(excinfo.value)
