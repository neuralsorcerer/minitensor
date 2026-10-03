# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""`cross_entropy(..., label_smoothing=e)` against its definition.

The smoothed target puts `1 - e` on the true class and `e / C` on every class.
With class weights, each class's share of that spread is weighted, and a mean
still divides by the total weight of the targets kept -- the unsmoothed
divisor -- so smoothing changes what the loss measures without rescaling it.
An ignored position contributes nothing to either part.

The reference below is written from that definition in float64, independently
of the implementation.
"""

import numpy as np
import pytest

import minitensor as mt

F = mt.functional
RNG = np.random.default_rng(23)
CLASSES = 4


def _reference(scores, labels, smoothing, weight, ignore_index, reduction):
    shifted = scores - scores.max(1, keepdims=True)
    log_probs = shifted - np.log(np.exp(shifted).sum(1, keepdims=True))
    log_probs = np.moveaxis(log_probs, 1, -1).reshape(-1, CLASSES)
    flat = labels.reshape(-1)
    weight = np.ones(CLASSES) if weight is None else weight
    kept = flat != ignore_index
    safe = np.where(kept, flat, 0)
    on_target = -log_probs[np.arange(flat.size), safe] * weight[safe] * kept
    uniform = -(log_probs * weight).sum(1) / CLASSES * kept
    losses = (1 - smoothing) * on_target + smoothing * uniform
    if reduction == "none":
        return losses.reshape(labels.shape)
    if reduction == "sum":
        return losses.sum()
    return losses.sum() / (weight[safe] * kept).sum()


@pytest.mark.parametrize("shape", [(6, CLASSES), (3, CLASSES, 5)])
@pytest.mark.parametrize("smoothing", [0.0, 0.1, 0.5, 1.0])
@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
def test_matches_the_definition(shape, smoothing, weighted, reduction):
    scores = RNG.standard_normal(shape)
    labels = RNG.integers(0, CLASSES, (shape[0],) + shape[2:])
    labels.flat[1] = -100
    weight = RNG.random(CLASSES) + 0.5 if weighted else None
    got = F.cross_entropy(
        mt.from_numpy(scores),
        mt.from_numpy(labels),
        reduction,
        weight=None if weight is None else mt.from_numpy(weight),
        label_smoothing=smoothing,
    )
    expected = _reference(scores, labels, smoothing, weight, -100, reduction)
    np.testing.assert_allclose(got.numpy(), expected, rtol=1e-10, atol=1e-12)


def test_zero_smoothing_is_the_plain_loss():
    scores = mt.from_numpy(RNG.standard_normal((5, CLASSES)))
    labels = mt.from_numpy(RNG.integers(0, CLASSES, 5))
    assert (
        F.cross_entropy(scores, labels, label_smoothing=0.0).item()
        == F.cross_entropy(scores, labels).item()
    )


def test_a_one_hot_target_smooths_the_same_as_its_index():
    scores = mt.from_numpy(RNG.standard_normal((5, CLASSES)))
    labels = RNG.integers(0, CLASSES, 5)
    one_hot = mt.from_numpy(np.eye(CLASSES)[labels])
    np.testing.assert_allclose(
        F.cross_entropy(scores, one_hot, label_smoothing=0.3).item(),
        F.cross_entropy(scores, mt.from_numpy(labels), label_smoothing=0.3).item(),
        rtol=1e-12,
    )


def test_a_class_axis_other_than_one():
    scores = RNG.standard_normal((2, 5, CLASSES))
    labels = mt.from_numpy(RNG.integers(0, CLASSES, (2, 5)))
    last = F.cross_entropy(mt.from_numpy(scores), labels, dim=-1, label_smoothing=0.2)
    moved = F.cross_entropy(
        mt.from_numpy(np.moveaxis(scores, -1, 1).copy()), labels, label_smoothing=0.2
    )
    np.testing.assert_allclose(last.item(), moved.item(), rtol=1e-12)


@pytest.mark.parametrize("reduction", ["mean", "sum"])
def test_the_gradient_matches_central_differences(reduction):
    labels = mt.from_numpy(np.array([0, -100, 1, 3]))
    weight = mt.from_numpy(np.array([1.0, 2.0, 0.5, 1.5]))
    scores = mt.from_numpy(RNG.standard_normal((4, CLASSES))).requires_grad_(True)
    assert mt.gradcheck(
        lambda t: F.cross_entropy(
            t, labels, reduction, weight=weight, label_smoothing=0.25
        ),
        [scores],
    )


def test_the_layer_passes_it_through_and_shows_it():
    layer = mt.nn.CrossEntropyLoss(label_smoothing=0.2)
    assert layer.label_smoothing == 0.2
    assert repr(layer) == "CrossEntropyLoss(reduction='mean', label_smoothing=0.2)"
    assert repr(mt.nn.CrossEntropyLoss()) == "CrossEntropyLoss(reduction='mean')"
    scores = mt.from_numpy(RNG.standard_normal((3, CLASSES)))
    labels = mt.from_numpy(np.array([0, 1, 2]))
    assert layer(scores, labels).item() == pytest.approx(
        F.cross_entropy(scores, labels, label_smoothing=0.2).item(), rel=1e-12
    )


@pytest.mark.parametrize("bad", [-0.1, 1.5, float("nan")])
def test_a_smoothing_outside_zero_to_one_is_refused(bad):
    scores = mt.zeros((2, CLASSES))
    labels = mt.from_numpy(np.array([0, 1]))
    with pytest.raises(ValueError, match="label_smoothing must be in"):
        F.cross_entropy(scores, labels, label_smoothing=bad)
    with pytest.raises(ValueError, match="label_smoothing must be in"):
        mt.nn.CrossEntropyLoss(label_smoothing=bad)
