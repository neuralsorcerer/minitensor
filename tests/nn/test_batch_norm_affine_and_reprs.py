# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""BatchNorm honours `affine`, and every layer prints the call that builds it.

`BatchNorm1d(affine=False)` and `BatchNorm2d(affine=False)` read the flag and
dropped it, so the layer still had a scale and shift for an optimizer to
train. `BatchNorm2d` also named its parameters `param_0` and `param_1`, so its
state dict could not be read against `BatchNorm1d`'s.

Several reprs printed Rust (`Softmax(dim=Some(0))`, `padding_idx=Some(1)`,
`mode=Nearest`) and several left out arguments that change what the layer
computes -- a strided, biasless convolution printed as a plain one.
"""

from __future__ import annotations

import numpy as np
import pytest

import minitensor as mt
from minitensor.nn import (
    ELU,
    AdaptiveAvgPool2d,
    BatchNorm1d,
    BatchNorm2d,
    Conv1d,
    Conv2d,
    ConvTranspose1d,
    ConvTranspose2d,
    DenseLayer,
    Dropout,
    Embedding,
    LayerNorm,
    LeakyReLU,
    MultiheadAttention,
    RMSNorm,
    Softmax,
    Upsample,
)

F = mt.functional


@pytest.mark.parametrize("cls", [BatchNorm1d, BatchNorm2d])
def test_affine_false_leaves_nothing_to_train(cls):
    layer = cls(3, affine=False)
    assert list(layer.named_parameters()) == []
    assert sorted(layer.state_dict().keys()) == ["running_mean", "running_var"]
    with pytest.raises(ValueError, match="No parameters"):
        mt.optim.SGD(layer.parameters(), lr=0.1)


@pytest.mark.parametrize("cls", [BatchNorm1d, BatchNorm2d])
def test_affine_layers_name_their_scale_and_shift(cls):
    layer = cls(3)
    assert sorted(name for name, _ in layer.named_parameters()) == ["bias", "weight"]


def test_a_non_affine_layer_normalizes_like_the_functional_form():
    x = mt.randn(6, 3, dtype="float64", requires_grad=True)
    layer = BatchNorm1d(3, affine=False, dtype="float64")
    y = layer(x)
    expected = F.batch_norm(x.detach(), None, None, training=True)
    np.testing.assert_allclose(y.detach().numpy(), expected.numpy(), rtol=1e-12)
    y.sum().backward()
    assert x.grad is not None


LAYERS = [
    lambda: DenseLayer(3, 2),
    lambda: DenseLayer(3, 2, bias=False),
    lambda: Softmax(),
    lambda: Softmax(dim=0),
    lambda: LeakyReLU(0.2),
    lambda: ELU(1.0),
    lambda: Conv2d(2, 4, (3, 5), stride=2, padding=1, dilation=2, groups=2, bias=False),
    lambda: Conv1d(2, 4, 3, stride=2, padding=1, groups=2),
    lambda: ConvTranspose2d(1, 2, 3, stride=2, output_padding=1),
    lambda: ConvTranspose1d(1, 2, 3, stride=2, bias=False),
    lambda: BatchNorm1d(4, eps=1e-3, momentum=0.2, affine=False),
    lambda: BatchNorm2d(3),
    lambda: Dropout(0.3),
    lambda: Embedding(5, 2, padding_idx=1),
    lambda: Embedding(5, 2),
    lambda: LayerNorm([4], eps=1e-3, elementwise_affine=False),
    lambda: RMSNorm([4]),
    lambda: MultiheadAttention(8, 2, bias=False, is_causal=True),
    lambda: Upsample(scale_factor=2),
    lambda: Upsample(size=(4, 6), mode="bilinear", align_corners=True),
    lambda: AdaptiveAvgPool2d((2, 3)),
]


@pytest.mark.parametrize("build", LAYERS, ids=lambda f: repr(f()))
def test_the_repr_rebuilds_the_same_layer(build):
    text = repr(build())
    assert "Some(" not in text and "None" not in text
    assert repr(eval(text)) == text


def test_settings_that_change_the_computation_are_shown():
    assert repr(Conv2d(1, 2, 3)) != repr(Conv2d(1, 2, 3, stride=2))
    assert repr(DenseLayer(3, 2)) != repr(DenseLayer(3, 2, bias=False))
    assert repr(BatchNorm1d(4)) != repr(BatchNorm1d(4, affine=False))
