# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""Giving a layer the wrong width is the first mistake most people make.

It used to be the least informative message in the library. Every layer that
validates a feature count built it as `shape_mismatch(vec![expected],
vec![got])`, which renders as two one-element shapes:

    DenseLayer(10, 5) on a (2, 7) input
      -> Shape mismatch: expected [10], got [7]
         💡 ... Use .view() or .reshape() to change the tensor shape

Neither `[10]` nor `[7]` is a shape anyone has. Nothing says that `[10]` is the
`in_features` argument, that `7` was read off the last axis, or that the input
was `[2, 7]` -- and reshaping is rarely the fix. Meanwhile the recurrent layers
next door already said "LSTM expects input feature size 8, got 3", so the
library disagreed with itself about how to report the same mistake.

The tests below assert the three facts a reader needs: which constructor
argument, which axis, and what the input's shape actually was.
"""

from __future__ import annotations

import numpy as np
import pytest

import minitensor as mt
from minitensor import nn


def _indices(*shape):
    return mt.Tensor(np.zeros(shape, dtype=np.int64), dtype="int64")


# (description, construct-and-call, constructor argument, offending value,
#  configured value, input shape)
FEATURE_LAYERS = [
    (
        "DenseLayer",
        lambda: nn.DenseLayer(10, 5)(mt.randn(2, 7)),
        "in_features",
        7,
        10,
        [2, 7],
    ),
    (
        "BatchNorm1d",
        lambda: nn.BatchNorm1d(8)(mt.randn(4, 3)),
        "num_features",
        3,
        8,
        [4, 3],
    ),
    (
        "BatchNorm1d-3d",
        lambda: nn.BatchNorm1d(8)(mt.randn(4, 3, 6)),
        "num_features",
        3,
        8,
        [4, 3, 6],
    ),
    (
        "BatchNorm2d",
        lambda: nn.BatchNorm2d(8)(mt.randn(2, 3, 4, 4)),
        "num_features",
        3,
        8,
        [2, 3, 4, 4],
    ),
    (
        "MultiheadAttention",
        lambda: nn.MultiheadAttention(256, 8)(mt.randn(8, 128, 64)),
        "embed_dim",
        64,
        256,
        [8, 128, 64],
    ),
]


@pytest.mark.parametrize(
    "name,call,argument,actual,configured,shape",
    FEATURE_LAYERS,
    ids=[case[0] for case in FEATURE_LAYERS],
)
def test_message_names_argument_axis_and_input_shape(
    name, call, argument, actual, configured, shape
):
    with pytest.raises(Exception) as excinfo:
        call()
    message = str(excinfo.value)

    assert f"{argument}={configured}" in message, message
    assert str(actual) in message, message
    assert f"input shape {shape}" in message, message


@pytest.mark.parametrize(
    "name,call,argument,actual,configured,shape",
    FEATURE_LAYERS,
    ids=[case[0] for case in FEATURE_LAYERS],
)
def test_suggestion_offers_both_directions(
    name, call, argument, actual, configured, shape
):
    """Either widen the layer or reshape the input -- the reader picks."""
    with pytest.raises(Exception) as excinfo:
        call()
    message = str(excinfo.value)
    assert f"{argument}={actual}" in message, message


def test_dense_layer_says_which_axis_it_read():
    with pytest.raises(Exception) as excinfo:
        nn.DenseLayer(10, 5)(mt.randn(2, 3, 7))
    assert "last dimension" in str(excinfo.value)


def test_batch_norm_says_which_axis_it_read():
    """BatchNorm's channel axis is 1, not the last -- a distinction the old
    message could not make, since it printed neither."""
    with pytest.raises(Exception) as excinfo:
        nn.BatchNorm2d(8)(mt.randn(2, 3, 4, 4))
    assert "dimension 1" in str(excinfo.value)


@pytest.mark.parametrize("layer", ["LayerNorm", "RMSNorm"])
def test_normalization_compares_suffix_not_whole_shape(layer):
    """`expected [512], got [4, 256]` implied the input should have been 1-D."""
    with pytest.raises(Exception) as excinfo:
        getattr(nn, layer)([512])(mt.randn(4, 256))
    message = str(excinfo.value)

    assert "last 1 dimension is [256]" in message, message
    assert "[512]" in message
    assert "input shape [4, 256]" in message


@pytest.mark.parametrize("layer", ["LayerNorm", "RMSNorm"])
def test_normalization_pluralises_multi_axis_shapes(layer):
    with pytest.raises(Exception) as excinfo:
        getattr(nn, layer)([8, 16])(mt.randn(4, 8, 32))
    message = str(excinfo.value)
    assert "last 2 dimensions [8, 16]" in message, message
    assert "last 2 dimensions are [8, 32]" in message, message


@pytest.mark.parametrize("layer", ["LayerNorm", "RMSNorm"])
def test_normalization_accepts_a_matching_suffix(layer):
    out = getattr(nn, layer)([8, 16])(mt.randn(4, 8, 16))
    assert tuple(out.shape) == (4, 8, 16)


def test_cross_entropy_index_target_count():
    """Was "Shape mismatch: expected [5], got [4]" -- the target count named as
    the expectation, and neither tensor's shape mentioned."""
    with pytest.raises(Exception) as excinfo:
        nn.cross_entropy(mt.randn(4, 10), _indices(5))
    message = str(excinfo.value)

    assert "[4, 10]" in message
    assert "10 classes" in message
    assert "must have shape [4]" in message
    assert "the target is [5]" in message


def test_cross_entropy_reports_the_class_axis_it_used():
    """With `dim=1` on a 3-D input the classes are dimension 1, so the target
    shape is the *other* two axes. The flattened message could not say this:
    `(2,3,10)` against `(2,4)` came out as "expected [8], got [20]"."""
    with pytest.raises(Exception) as excinfo:
        nn.cross_entropy(mt.randn(2, 3, 10), _indices(2, 4))
    message = str(excinfo.value)

    assert "3 classes on dim 1" in message
    assert "must have shape [2, 10]" in message
    assert "20" not in message


def test_cross_entropy_dense_target_must_match_exactly():
    with pytest.raises(Exception) as excinfo:
        nn.cross_entropy(mt.randn(4, 10), mt.randn(4, 7))
    message = str(excinfo.value)
    assert "one score per class" in message
    assert "input is [4, 10] and the target is [4, 7]" in message


def test_cross_entropy_bad_dim_uses_the_dimension_message():
    with pytest.raises(IndexError) as excinfo:
        nn.cross_entropy(mt.randn(4, 10), _indices(4), dim=5)
    message = str(excinfo.value)
    assert "Dimension out of range" in message
    assert "class axis" in message


@pytest.mark.parametrize(
    "predictions,target,dim",
    [
        ((4, 10), (4,), 1),
        ((4, 10), (4, 10), 1),
        ((2, 3, 10), (2, 10), 1),
        ((2, 3, 10), (2, 3), -1),
        ((4, 10), (4,), -1),
    ],
)
def test_valid_cross_entropy_calls_still_work(predictions, target, dim):
    """The check runs before the forward flattens anything, so it has to accept
    every layout the forward handles."""
    logits = mt.randn(*predictions)
    targets = (
        _indices(*target)
        if len(target) < len(predictions)
        else mt.randn(*target).softmax(-1)
    )
    loss = nn.cross_entropy(logits, targets, dim=dim)
    assert tuple(loss.shape) == ()
    assert np.isfinite(loss.numpy()).all()


@pytest.mark.parametrize("loss", ["CrossEntropyLoss", "FocalLoss"])
def test_classification_loss_layers_check_the_target_count(loss):
    """The layers reach the kernel directly rather than through the functional
    `cross_entropy`, so they need their own check -- they had none, and the
    mismatch surfaced from a broadcast several calls later."""
    with pytest.raises(Exception) as excinfo:
        getattr(nn, loss)()(mt.randn(4, 10), _indices(5))
    message = str(excinfo.value)

    assert "[4, 10]" in message
    assert "10 classes" in message
    assert "[5]" in message


@pytest.mark.parametrize("loss", ["CrossEntropyLoss", "FocalLoss"])
def test_classification_loss_layers_accept_matching_targets(loss):
    value = getattr(nn, loss)()(mt.randn(4, 10), _indices(4))
    assert np.isfinite(value.numpy()).all()


def test_correct_widths_are_unaffected():
    """Nothing above should have narrowed what the layers accept."""
    assert tuple(nn.DenseLayer(10, 5)(mt.randn(2, 10)).shape) == (2, 5)
    assert tuple(nn.BatchNorm1d(3)(mt.randn(4, 3)).shape) == (4, 3)
    assert tuple(nn.BatchNorm1d(3)(mt.randn(4, 3, 6)).shape) == (4, 3, 6)
    assert tuple(nn.BatchNorm2d(3)(mt.randn(2, 3, 4, 4)).shape) == (2, 3, 4, 4)
    assert tuple(nn.MultiheadAttention(256, 8)(mt.randn(8, 12, 256)).shape) == (
        8,
        12,
        256,
    )
