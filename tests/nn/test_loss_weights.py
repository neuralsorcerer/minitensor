# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""Class weights, ignored targets and element weights in the classification
losses, checked against their definitions.

`nll_loss` took `weight` and `ignore_index` but `cross_entropy` -- the same loss
with the log-softmax folded in -- took neither, and nor did `CrossEntropyLoss`.
The binary cross entropies had no per-element `weight`.
"""

from __future__ import annotations

import numpy as np
import pytest

import minitensor as mt

F = mt.functional

RNG = np.random.default_rng(0)
LOGITS = RNG.normal(size=(6, 4))
TARGET = np.array([0, 3, 1, 1, 2, 3])
CLASS_WEIGHT = np.array([0.5, 2.0, 1.0, 3.0])


def _t(values):
    return mt.tensor(values, dtype="float64")


def _i(values):
    return mt.tensor(values, dtype="int64")


def _reference(weight=None, ignore=-100, reduction="mean", target=TARGET):
    log_p = LOGITS - np.log(np.exp(LOGITS).sum(1, keepdims=True))
    keep = target != ignore
    safe = np.where(keep, target, 0)
    scale = (weight[safe] if weight is not None else np.ones(len(target))) * keep
    losses = -log_p[np.arange(len(target)), safe] * scale
    return losses.sum() / scale.sum() if reduction == "mean" else losses.sum()


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({}, _reference()),
        ({"weight": CLASS_WEIGHT}, _reference(CLASS_WEIGHT)),
        ({"ignore_index": 1}, _reference(ignore=1)),
        ({"weight": CLASS_WEIGHT, "ignore_index": 3}, _reference(CLASS_WEIGHT, 3)),
        (
            {"weight": CLASS_WEIGHT, "reduction": "sum"},
            _reference(CLASS_WEIGHT, reduction="sum"),
        ),
    ],
    ids=["plain", "weight", "ignore", "weight+ignore", "weight-sum"],
)
def test_cross_entropy_and_its_layer_follow_the_definition(kwargs, expected):
    call = {k: (_t(v) if k == "weight" else v) for k, v in kwargs.items()}
    assert np.isclose(F.cross_entropy(_t(LOGITS), _i(TARGET), **call).item(), expected)
    layer = mt.nn.CrossEntropyLoss(**call)
    assert np.isclose(layer(_t(LOGITS), _i(TARGET)).item(), expected)


def test_the_default_ignore_index_drops_its_positions():
    target = TARGET.copy()
    target[2] = -100
    got = F.cross_entropy(_t(LOGITS), _i(target)).item()
    assert np.isclose(got, _reference(target=target))


def test_a_weighted_cross_entropy_reads_the_class_axis_it_is_given():
    scores = RNG.normal(size=(2, 5, 4))
    labels = _i(RNG.integers(0, 4, (2, 5)))
    last = F.cross_entropy(_t(scores), labels, weight=_t(CLASS_WEIGHT), dim=-1)
    first = F.cross_entropy(
        _t(np.moveaxis(scores, -1, 1)), labels, weight=_t(CLASS_WEIGHT)
    )
    assert np.isclose(last.item(), first.item())


def test_the_binary_cross_entropies_weight_each_element():
    logits = RNG.normal(size=(4, 3))
    labels = RNG.integers(0, 2, (4, 3)).astype(float)
    weight = RNG.uniform(0.5, 2.0, (4, 3))
    p = 1 / (1 + np.exp(-logits))
    losses = -(labels * np.log(p) + (1 - labels) * np.log(1 - p))

    got = F.binary_cross_entropy_with_logits(_t(logits), _t(labels), weight=_t(weight))
    assert np.isclose(got.item(), (losses * weight).mean())
    got = F.binary_cross_entropy(_t(p), _t(labels), reduction="sum", weight=_t(weight))
    assert np.isclose(got.item(), (losses * weight).sum())
