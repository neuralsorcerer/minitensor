# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""`state_dict()` is a mapping, and its order does not change between runs.

Two things were wrong with it, and they compound.

`StateDict` defined `__getitem__`, `__len__` and `__contains__` -- the three
that make an object look like a mapping -- but no `__iter__`, `keys`, `values`
or `items`. So `len(state)`, `state["weight"]` and `"weight" in state` worked
while `for name in state`, `list(state)` and `dict(state)` failed with

    TypeError: 'int' object is not an instance of 'str'

which says nothing about what went wrong. It comes from Python falling back to
the legacy sequence protocol and calling `__getitem__(0)` on a subscript that
wants a name. Since a `state_dict()` is an ordered mapping, `.items()` and
`dict(...)` are the first things anyone reaches for.

Underneath, the parameters and buffers were `HashMap`s. Rust seeds its hasher
randomly per process, so the iteration order changed from run to run -- and it
reached the file, because serde writes a map in iteration order. Saving one
model twice produced two different byte streams, so a checkpoint could not be
content-hashed, compared against another by digest, or diffed without spurious
reordering. `BTreeMap` makes the order sorted and the bytes a function of the
weights.

The two fixes meet at `keys()`: it spans parameters then buffers, matching what
`__len__` and `__contains__` already covered, and each half is sorted.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import textwrap

import numpy as np
import pytest

import minitensor as mt

S = mt.serialization


def _layer_state():
    return mt.nn.BatchNorm1d(4).state_dict()


def _built_state():
    state = S.StateDict()
    for name in ["weight", "bias", "gamma", "beta"]:
        state.add_parameter(name, mt.Tensor(np.arange(3, dtype=np.float32)))
    for name in ["running_mean", "running_var"]:
        state.add_buffer(name, mt.Tensor(np.arange(3, dtype=np.float32)))
    return state


# --- the mapping protocol ---------------------------------------------------


def test_iterating_yields_names():
    state = _layer_state()
    assert list(state) == state.keys()
    assert [name for name in state] == state.keys()


def test_dict_and_unpacking_work():
    """Both go through `keys()` plus subscripting, which is why defining
    `keys()` is what makes them work."""
    state = _layer_state()
    as_dict = dict(state)
    unpacked = {**state}

    assert sorted(as_dict) == sorted(state.keys())
    assert sorted(unpacked) == sorted(state.keys())
    for name in state.keys():
        np.testing.assert_array_equal(as_dict[name].numpy(), state[name].numpy())


def test_items_pairs_each_name_with_its_tensor():
    state = _layer_state()
    for name, tensor in state.items():
        np.testing.assert_array_equal(tensor.numpy(), state[name].numpy())
    assert [name for name, _ in state.items()] == state.keys()


def test_values_are_in_keys_order():
    state = _layer_state()
    for tensor, name in zip(state.values(), state.keys()):
        np.testing.assert_array_equal(tensor.numpy(), state[name].numpy())


def test_the_four_views_agree_with_each_other():
    """`keys`, `len`, `__contains__` and `__getitem__` describe one collection,
    so they have to span the same names."""
    state = _built_state()
    names = state.keys()

    assert len(names) == len(state)
    assert len(set(names)) == len(names)
    assert all(name in state for name in names)
    assert all(state[name] is not None for name in names)


def test_keys_spans_parameters_and_buffers():
    state = _built_state()
    assert set(state.keys()) == set(state.parameter_names()) | set(state.buffer_names())
    # parameters come first, as the subscript lookup checks them first
    assert state.keys()[: len(state.parameter_names())] == state.parameter_names()


def test_an_empty_state_dict_iterates_empty():
    state = S.StateDict()
    assert state.keys() == [] and list(state) == [] and dict(state) == {}
    assert len(state) == 0


def test_an_unknown_name_still_raises_keyerror():
    """Adding iteration must not soften the subscript."""
    with pytest.raises(KeyError) as excinfo:
        _layer_state()["not_a_parameter"]
    assert "not_a_parameter" in str(excinfo.value)


def test_the_old_confusing_message_is_gone():
    try:
        list(_layer_state())
    except TypeError as exc:  # pragma: no cover - the point is that it does not
        pytest.fail(f"iteration still fails: {exc}")


# --- order stability --------------------------------------------------------


def test_names_are_sorted_within_each_half():
    state = _built_state()
    assert state.parameter_names() == sorted(state.parameter_names())
    assert state.buffer_names() == sorted(state.buffer_names())


_ORDER_PROBE = textwrap.dedent("""
    import numpy as np, minitensor as mt
    state = mt.serialization.StateDict()
    for name in ["weight", "bias", "gamma", "beta", "delta", "epsilon"]:
        state.add_parameter(name, mt.Tensor(np.zeros(2, dtype=np.float32)))
    print(",".join(state.keys()))
    """)


def test_the_order_is_the_same_in_a_fresh_process():
    """Rust seeds its hash maps per process, so a single run cannot show this.
    Four separate interpreters must agree."""
    seen = {
        subprocess.run(
            [sys.executable, "-c", _ORDER_PROBE],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        for _ in range(4)
    }
    assert len(seen) == 1, f"order varies between runs: {seen}"


# --- the same model saves to the same bytes ---------------------------------


def _checkpoint(tmp_path, fmt):
    state = _built_state()
    metadata = S.ModelMetadata("m", "arch")
    for key in ["zeta", "alpha", "mu"]:
        metadata.add_custom(key, "v")

    path = str(tmp_path / f"model.{fmt}")
    S.ModelSerializer.save(
        S.SerializedModel(metadata, state), path, S.SerializationFormat(fmt)
    )
    return path


def _read_json(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _read_bytes(path):
    with open(path, "rb") as handle:
        return handle.read()


def test_the_key_order_in_the_file_is_sorted(tmp_path):
    document = _read_json(_checkpoint(tmp_path, "json"))
    for section in [
        document["state_dict"]["parameters"],
        document["state_dict"]["buffers"],
        document["metadata"]["custom"],
    ]:
        assert list(section) == sorted(section)


def _bytes_without_the_timestamp(path, fmt):
    """The file's bytes with `created_at` cut out of them.

    It is the one field meant to differ between two saves, so it has to come
    out before they can be compared. It cannot be masked in place, because it
    is not a fixed width: chrono prints the fractional second to 0, 3, 6 or 9
    digits depending on how many trailing zeros it has, so two saves a
    microsecond apart differ in length as well as in content roughly one time
    in a thousand. All three formats store the field as its RFC 3339 string
    preceded by the byte giving that string's length -- an opening quote, in
    JSON -- and both have to go, since the length byte varies with the width
    too.
    """
    raw = _read_bytes(path)
    stamp = S.ModelSerializer.load(
        path, S.SerializationFormat(fmt)
    ).metadata.created_at.encode()
    start = raw.index(stamp)
    return raw[: start - 1] + raw[start + len(stamp) :]


@pytest.mark.parametrize("fmt", ["json", "binary", "messagepack"])
def test_saving_the_same_model_twice_produces_the_same_bytes(tmp_path, fmt):
    """Every byte but the timestamp is a function of the model.

    This is what the `BTreeMap`s buy. `HashMap` seeds its hasher per instance,
    so the parameter and buffer sections came out in a different order on each
    save and no two checkpoints of the same weights were ever the same file.
    """
    (tmp_path / "one").mkdir()
    (tmp_path / "two").mkdir()

    one = _bytes_without_the_timestamp(_checkpoint(tmp_path / "one", fmt), fmt)
    two = _bytes_without_the_timestamp(_checkpoint(tmp_path / "two", fmt), fmt)

    assert one == two


def test_the_state_dict_half_is_byte_identical(tmp_path):
    """Isolating the part of the file with no timestamp in it."""
    (tmp_path / "one").mkdir()
    (tmp_path / "two").mkdir()

    digests = {
        hashlib.sha256(
            json.dumps(
                _read_json(_checkpoint(tmp_path / name, "json"))["state_dict"],
                sort_keys=False,
            ).encode()
        ).hexdigest()
        for name in ["one", "two"]
    }
    assert len(digests) == 1


# --- reading what an earlier version wrote ----------------------------------


def test_a_checkpoint_stored_out_of_order_still_loads(tmp_path):
    """Files written before this change have their keys in whatever order that
    process's hasher produced. Sorting on write must not make them unreadable."""
    document = _read_json(_checkpoint(tmp_path, "json"))
    parameters = document["state_dict"]["parameters"]
    document["state_dict"]["parameters"] = dict(reversed(list(parameters.items())))

    shuffled = str(tmp_path / "shuffled.json")
    with open(shuffled, "w", encoding="utf-8") as handle:
        json.dump(document, handle)
    assert list(_read_json(shuffled)["state_dict"]["parameters"]) != sorted(parameters)

    loaded = S.ModelSerializer.load(shuffled, S.SerializationFormat("json")).state_dict
    reference = _built_state()
    assert loaded.keys() == reference.keys()
    for name in reference.keys():
        np.testing.assert_array_equal(loaded[name].numpy(), reference[name].numpy())
