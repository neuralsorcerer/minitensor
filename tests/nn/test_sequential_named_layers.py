# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""`Sequential.add_module(name, module)` names the layer it adds.

The name used to be accepted and thrown away: the layer's keys were its
position, so `add_module("head", layer)` saved `2.weight`, and nothing said the
name had gone.
"""

from __future__ import annotations

import copy

import pytest

import minitensor as mt

nn = mt.nn


def _model():
    model = nn.Sequential([nn.DenseLayer(3, 4), nn.ReLU()])
    model.add_module("head", nn.DenseLayer(4, 2))
    return model


def test_the_name_prefixes_the_layers_keys():
    assert sorted(_model().state_dict().keys()) == [
        "0.bias",
        "0.weight",
        "head.bias",
        "head.weight",
    ]


def test_the_repr_shows_the_name():
    assert "(head): DenseLayer(in_features=4, out_features=2)" in repr(_model())


def test_a_named_nested_sequential_prefixes_its_childrens_keys():
    outer = nn.Sequential([])
    outer.add_module("encoder", nn.Sequential([nn.DenseLayer(2, 2)]))
    assert sorted(outer.state_dict().keys()) == ["encoder.0.bias", "encoder.0.weight"]


def test_a_deep_copy_keeps_the_names_and_takes_the_originals_weights():
    model = _model()
    duplicate = copy.deepcopy(model)
    assert sorted(duplicate.state_dict().keys()) == sorted(model.state_dict().keys())
    duplicate.load_state_dict(model.state_dict())


@pytest.mark.parametrize(
    ("name", "problem"),
    [
        ("head", "already taken"),
        ("3", "cannot be an integer"),
        ("", "cannot be empty"),
        ("a.b", "cannot contain '.'"),
    ],
)
def test_a_name_that_cannot_key_a_parameter_is_refused(name, problem):
    model = _model()
    with pytest.raises(ValueError, match=problem):
        model.add_module(name, nn.ReLU())
    assert len(model) == 3
