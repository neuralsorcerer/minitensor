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


def test_layers_without_parameters_and_embeddings_take_their_own_inputs():
    nn.ReLU()(mt.randn(2, 3, dtype="float64"))
    nn.Embedding(5, 2)(mt.tensor([1, 2], dtype="int64"))
