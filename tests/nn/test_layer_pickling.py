# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""Built-in layers pickle, and `copy.copy` them.

Every built-in layer could be deep-copied but none could be pickled, which is
how a whole model is checkpointed and how one reaches a worker process. A
layer now reduces to its constructor keywords -- each of which reads back
under its own name -- plus its state dict, its training mode and whether it is
frozen, so the rebuilt layer is the same in everything that can be observed.
"""

import copy
import pickle

import numpy as np
import pytest

import minitensor as mt
import minitensor.nn as nn

LAYERS = {
    "dense": lambda: nn.DenseLayer(3, 4, bias=False, dtype="float64"),
    "conv1d same": lambda: nn.Conv1d(2, 3, 3, padding="same", dilation=2, bias=False),
    "conv2d": lambda: nn.Conv2d(
        2, 4, (3, 2), stride=(2, 1), padding=(1, 0), dilation=(1, 2), groups=2
    ),
    "conv2d same": lambda: nn.Conv2d(2, 3, 3, padding="same"),
    "conv transpose 1d": lambda: nn.ConvTranspose1d(
        2, 3, 3, stride=2, padding=1, output_padding=1, bias=False
    ),
    "conv transpose 2d": lambda: nn.ConvTranspose2d(
        4, 2, 3, stride=2, padding=1, groups=2
    ),
    "batch norm": lambda: nn.BatchNorm1d(4, eps=1e-3, momentum=0.3, affine=False),
    "batch norm 2d": lambda: nn.BatchNorm2d(3, momentum=0.05),
    "layer norm": lambda: nn.LayerNorm([2, 4], eps=1e-4, elementwise_affine=False),
    "rms norm": lambda: nn.RMSNorm(4, eps=1e-6),
    "embedding": lambda: nn.Embedding(10, 3, padding_idx=2),
    "lstm": lambda: nn.LSTM(
        3, 4, num_layers=2, bias=False, batch_first=True, bidirectional=True
    ),
    "gru": lambda: nn.GRU(3, 2),
    "attention": lambda: nn.MultiheadAttention(4, 2, bias=False, is_causal=True),
    "gelu": lambda: nn.GELU(approximate="none"),
    "leaky": lambda: nn.LeakyReLU(0.3),
    "dropout": lambda: nn.Dropout(0.25),
    "upsample": lambda: nn.Upsample(scale_factor=2, mode="bilinear"),
    "upsample size": lambda: nn.Upsample(size=[5, 6]),
    "avg pool": lambda: nn.AvgPool2d(3, stride=2, padding=1, count_include_pad=False),
    "adaptive pool": lambda: nn.AdaptiveMaxPool2d((2, 3)),
}

INPUTS = {
    "dense": (4, 3),
    "conv1d same": (2, 2, 7),
    "conv2d": (1, 2, 6, 5),
    "conv2d same": (1, 2, 5, 5),
    "conv transpose 1d": (1, 2, 5),
    "conv transpose 2d": (1, 4, 3, 3),
    "batch norm": (6, 4),
    "batch norm 2d": (2, 3, 4, 4),
    "layer norm": (3, 2, 4),
    "rms norm": (3, 4),
    "lstm": (2, 5, 3),
    "gru": (5, 2, 3),
    "attention": (2, 3, 4),
    "gelu": (5,),
    "leaky": (5,),
    "dropout": (5,),
    "upsample": (1, 1, 2, 3),
    "upsample size": (1, 1, 2, 3),
    "avg pool": (1, 1, 5, 5),
    "adaptive pool": (1, 1, 5, 7),
}

COPIES = {
    "pickle": lambda layer: pickle.loads(pickle.dumps(layer)),
    "copy": copy.copy,
    "deepcopy": copy.deepcopy,
}


def _input(name, layer):
    if name == "embedding":
        return mt.tensor([[1, 2, 9]], dtype="int64")
    dtype = "float64" if name == "dense" else "float32"
    return mt.tensor(
        np.random.default_rng(0).standard_normal(INPUTS[name]), dtype=dtype
    )


@pytest.mark.parametrize("how", list(COPIES))
@pytest.mark.parametrize("name", list(LAYERS))
def test_a_copy_is_the_same_layer(name, how):
    layer = LAYERS[name]()
    # Trained away from the initial values, so the state is what is compared.
    with mt.no_grad():
        for parameter in layer.parameters():
            parameter[...] = parameter * 0.5 + 0.25
    copied = COPIES[how](layer)
    assert type(copied) is type(layer)
    assert repr(copied) == repr(layer)
    original, rebuilt = layer.state_dict(), copied.state_dict()
    assert list(rebuilt.keys()) == list(original.keys())
    for key in original.keys():
        assert rebuilt[key].dtype == original[key].dtype
        assert rebuilt[key].tolist() == original[key].tolist()
    layer.eval()
    copied.eval()
    x = _input(name, layer)
    assert copied(x).tolist() == layer(x).tolist()


@pytest.mark.parametrize("how", list(COPIES))
def test_mode_and_freezing_come_along(how):
    layer = nn.Dropout(0.5)
    layer.eval()
    assert COPIES[how](layer).training is False
    dense = nn.DenseLayer(2, 3)
    dense.requires_grad_(False)
    copied = COPIES[how](dense)
    assert [p.requires_grad for p in copied.parameters()] == [False, False]


@pytest.mark.parametrize("how", list(COPIES))
def test_a_model_with_named_layers_and_mixed_modes(how):
    model = nn.Sequential([nn.DenseLayer(3, 4), nn.ReLU()])
    model.add_module("drop", nn.Dropout(0.5))
    model.add_module("head", nn.DenseLayer(4, 2))
    # An unnamed layer after named ones is named by its position, as ever.
    model.append(nn.Tanh())
    model[2].eval()
    model[3].requires_grad_(False)
    copied = COPIES[how](model)
    assert [name for name, _ in copied.named_children()] == [
        "0",
        "1",
        "drop",
        "head",
        "4",
    ]
    assert list(copied.state_dict().keys()) == list(model.state_dict().keys())
    assert [child.training for child in copied] == [True, True, False, True, True]
    assert [p.requires_grad for p in copied[3].parameters()] == [False, False]
    model.eval()
    copied.eval()
    x = mt.randn(5, 3)
    assert copied(x).tolist() == model(x).tolist()


def test_a_copy_is_independent_of_the_original():
    layer = nn.DenseLayer(2, 2)
    copied = copy.copy(layer)
    with mt.no_grad():
        layer.weight[...] = 0.0
    assert copied.weight.tolist() != layer.weight.tolist()


def test_configuration_reads_back_under_the_constructor_names():
    conv = nn.Conv2d(2, 4, 3, stride=2, padding=1, dilation=2, groups=2, bias=False)
    assert (conv.stride, conv.padding, conv.dilation, conv.groups, conv.bias) == (
        (2, 2),
        (1, 1),
        (2, 2),
        2,
        None,
    )
    assert nn.Conv2d(2, 3, 3, padding="same").padding == "same"
    norm = nn.BatchNorm1d(4, eps=1e-3, momentum=0.2, affine=False)
    assert (norm.eps, norm.momentum, norm.affine) == (1e-3, 0.2, False)
    assert nn.GELU(approximate="none").approximate == "none"
    assert nn.LayerNorm(4, elementwise_affine=False).elementwise_affine is False
    assert nn.MultiheadAttention(4, 2, bias=False).bias is False
    up = nn.Upsample(scale_factor=2, mode="bilinear")
    assert (up.size, up.scale_factor, up.mode) == (None, [2.0], "linear")
