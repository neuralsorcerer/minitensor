# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""A layer refuses an input of another dtype than its parameters, by name.

The refusal used to come from the first operation inside the layer, which
reported its own operands the wrong way round -- a float64 layer given a
float32 input read "expected Float32, got Float64" -- and never said which
layer. BatchNorm did not refuse at all.
"""

from __future__ import annotations

import pytest

import minitensor as mt

nn = mt.nn


@pytest.mark.parametrize(
    ("build", "name"),
    [
        (lambda: nn.DenseLayer(3, 2, dtype="float64"), "DenseLayer"),
        (lambda: nn.BatchNorm1d(3, dtype="float64"), "BatchNorm1d"),
        (lambda: nn.LayerNorm([3], dtype="float64"), "LayerNorm"),
        (
            lambda: nn.Sequential([nn.ReLU(), nn.DenseLayer(3, 2, dtype="float64")]),
            "DenseLayer",
        ),
    ],
    ids=["dense", "batch_norm", "layer_norm", "inside-sequential"],
)
def test_a_float32_input_to_a_float64_layer_is_refused_by_name(build, name):
    with pytest.raises(TypeError) as caught:
        build()(mt.randn(2, 3))
    message = str(caught.value)
    assert "expected float64, got float32" in message
    assert f"{name} has float64 parameters" in message
    assert ".astype('float64')" in message
    assert "convert the model with .astype('float32')" in message


def test_layers_without_parameters_and_embeddings_take_their_own_inputs():
    nn.ReLU()(mt.randn(2, 3, dtype="float64"))
    nn.Embedding(5, 2)(mt.tensor([1, 2], dtype="int64"))


def test_astype_converts_a_model_in_place_and_keeps_it_trainable():
    model = nn.Sequential([nn.DenseLayer(3, 4), nn.BatchNorm1d(4), nn.DenseLayer(4, 2)])
    before = {
        name: tensor.numpy().copy() for name, tensor in model.state_dict().items()
    }

    assert model.astype("float64") is model
    assert {p.dtype for p in model.parameters()} == {"float64"}
    assert {b.dtype for b in model.buffers()} == {"float64"}
    assert [name for name, _ in model.named_buffers()] == [
        "1.running_mean",
        "1.running_var",
    ]
    after = model.state_dict()
    for name, values in before.items():
        assert (after[name].numpy() == values).all()

    output = model(mt.randn(5, 3, dtype="float64"))
    optimizer = mt.optim.SGD(model.parameters(), lr=0.1)
    output.sum().backward()
    optimizer.step()


def test_astype_keeps_a_frozen_model_frozen_and_refuses_integers():
    frozen = nn.DenseLayer(3, 2).requires_grad_(False).astype("float64")
    assert not any(p.requires_grad for p in frozen.parameters())
    with pytest.raises(ValueError, match="must be floating point"):
        nn.DenseLayer(3, 2).astype("int64")


@pytest.mark.parametrize("which", ["input", "hx", "cx"])
def test_a_recurrent_layer_names_the_state_of_the_wrong_dtype(which):
    layer = nn.LSTM(3, 4)
    tensors = {
        "input": mt.randn(5, 2, 3),
        "hx": mt.zeros(1, 2, 4),
        "cx": mt.zeros(1, 2, 4),
    }
    tensors[which] = tensors[which].astype("float64")
    with pytest.raises(TypeError, match=f"was given a float64 {which};"):
        layer.forward_with_state(tensors["input"], tensors["hx"], tensors["cx"])


def test_loading_a_state_of_another_dtype_names_the_conversion():
    saved = nn.DenseLayer(3, 2).astype("float64").state_dict()
    model = nn.DenseLayer(3, 2)
    with pytest.raises(
        ValueError, match=r"convert the model with \.astype\('float64'\)"
    ):
        model.load_state_dict(saved)
    model.astype("float64").load_state_dict(saved)
