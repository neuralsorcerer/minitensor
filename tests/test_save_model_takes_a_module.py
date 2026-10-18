# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""`save_model` writes what it is handed in the format it is asked for.

It took only a `SerializedModel`, so `save_model(model, path)` -- the call its
own signature suggests -- refused a module. With no `format` it wrote binary
whatever the path said, so `model.json` held no JSON, though both functions
are documented to read the format from the extension. And `Module.save` read
an unknown `format` as "use the extension": `format="xml"` wrote binary and
said nothing.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

import minitensor as mt

nn, ser = mt.nn, mt.serialization


def _model():
    return nn.Sequential([nn.DenseLayer(4, 8), nn.ReLU(), nn.DenseLayer(8, 3)])


@pytest.mark.parametrize(
    "name,format,is_json",
    [
        ("model.bin", None, False),
        ("model.json", None, True),
        ("MODEL.JSON", None, True),
        ("model.msgpack", None, False),
        ("model.ckpt", None, False),
        ("model.data", "json", True),
        ("model.data", "messagepack", False),
    ],
)
def test_a_module_round_trips_in_the_format_named(tmp_path, name, format, is_json):
    model = _model()
    x = mt.Tensor(np.random.default_rng(0).normal(0, 1, (5, 4)).astype(np.float32))
    path = os.fspath(tmp_path / name)
    ser.save_model(model, path, format)
    with open(path, "rb") as handle:
        assert (handle.read(1) == b"{") == is_json
    loaded = ser.load_model(path, format)
    assert loaded.metadata.architecture == "Sequential"
    fresh = _model()
    fresh.load_state_dict(loaded.state_dict)
    np.testing.assert_array_equal(fresh(x).numpy(), model(x).numpy())


def test_a_serialized_model_is_still_written_as_it_is(tmp_path):
    state = nn.DenseLayer(2, 2).state_dict()
    model = ser.SerializedModel(ser.ModelMetadata("dense", "DenseLayer"), state)
    path = os.fspath(tmp_path / "dense.json")
    ser.save_model(model, path)
    assert ser.load_model(path).metadata.name == "dense"


def test_an_unknown_format_is_refused_everywhere(tmp_path):
    path = os.fspath(tmp_path / "model.bin")
    model = _model()
    with pytest.raises(ValueError, match="xml"):
        ser.save_model(model, path, "xml")
    with pytest.raises(ValueError, match="xml"):
        model.save(path, format="xml")
    model.save(path)
    with pytest.raises(ValueError, match="xml"):
        nn.Module.load_state_from(path, format="xml")
    with pytest.raises(ValueError, match="xml"):
        ser.load_model(path, "xml")


def test_something_that_is_neither_is_named_in_the_error(tmp_path):
    with pytest.raises(TypeError, match="int"):
        ser.save_model(42, os.fspath(tmp_path / "x.bin"))
