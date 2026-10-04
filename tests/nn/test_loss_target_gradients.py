# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""A target that requires a gradient gets one, from every loss that takes scores.

`mse_loss`, `l1_loss`, `kl_div` and the rest always differentiated their
target. `cross_entropy` and `focal_loss` with per-class scores,
`binary_cross_entropy` and `binary_cross_entropy_with_logits` did not: the
target was left with no gradient and no error, which an optimizer reads as
"this tensor was never used". A soft label produced by another model is
exactly such a target. Each loss is linear in its target, so the gradient is
cheap, and it is checked here against central differences.

The same file covers the logits loss at an infinite logit, where its value was
NaN: `0 * inf` against a target of 1, `inf - inf` at `-inf`.
"""

import math

import pytest

import minitensor as mt
import minitensor.functional as F

PRED = [[0.2, 0.7, 0.4], [0.9, 0.35, 0.05]]
# Strictly inside (0, 1), so a central difference stays a probability.
TARGET = [[0.1, 0.6, 0.3], [0.95, 0.25, 0.05]]
LOGITS = [[1.5, -0.25, 3.0], [-2.0, 0.5, 0.0]]


def _pair(a, b):
    return (
        mt.tensor(a, dtype="float64", requires_grad=True),
        mt.tensor(b, dtype="float64", requires_grad=True),
    )


CASES = {
    "binary_cross_entropy": (lambda p, t, r: F.binary_cross_entropy(p, t, r), PRED),
    "binary_cross_entropy_with_logits": (
        lambda x, t, r: F.binary_cross_entropy_with_logits(x, t, reduction=r),
        LOGITS,
    ),
    "with pos_weight": (
        lambda x, t, r: F.binary_cross_entropy_with_logits(
            x,
            t,
            pos_weight=mt.tensor([2.0, 0.5, 1.0], dtype="float64"),
            reduction=r,
        ),
        LOGITS,
    ),
    "cross_entropy": (lambda x, t, r: F.cross_entropy(x, t, r), LOGITS),
    "cross_entropy label_smoothing": (
        lambda x, t, r: F.cross_entropy(x, t, r, label_smoothing=0.2),
        LOGITS,
    ),
    "focal_loss": (
        lambda x, t, r: F.focal_loss(x, t, alpha=0.5, gamma=1.5, reduction=r),
        LOGITS,
    ),
}


@pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
@pytest.mark.parametrize("name", list(CASES))
def test_both_inputs_match_central_differences(name, reduction):
    loss, first = CASES[name]
    a, b = _pair(first, TARGET)
    weights = mt.tensor([[0.5, -1.0, 2.0], [1.5, 0.25, -0.75]], dtype="float64")

    def objective(a, b):
        out = loss(a, b, reduction)
        if reduction == "none":
            # A weight per output, so the per-sample scaling is exercised too.
            w = weights if out.ndim() == 2 else weights[:, 0]
            out = out * w
        return out.sum()

    assert mt.gradcheck(objective, (a, b), atol=1e-7, rtol=1e-5)


@pytest.mark.parametrize("name", list(CASES))
def test_a_frozen_input_still_gets_its_gradient_alone(name):
    loss, first = CASES[name]
    for frozen in (0, 1):
        a, b = _pair(first, TARGET)
        (a, b)[frozen].requires_grad_(False)
        loss(a, b, "sum").backward()
        assert (a, b)[frozen].grad is None
        assert (a, b)[1 - frozen].grad is not None


def test_a_target_in_another_float_dtype_gets_a_gradient_in_its_own():
    for loss in (F.cross_entropy, F.focal_loss):
        x = mt.tensor(LOGITS, dtype="float64", requires_grad=True)
        t = mt.tensor(TARGET, dtype="float32", requires_grad=True)
        loss(x, t).backward()
        assert t.grad is not None and t.grad.dtype == "float32"


def test_cross_entropy_target_gradient_is_the_negative_log_softmax():
    x, t = _pair(LOGITS, TARGET)
    F.cross_entropy(x, t, "sum").backward()
    expected = (-F.log_softmax(x.detach(), dim=1)).tolist()
    for got, want in zip(t.grad.tolist(), expected):
        assert got == pytest.approx(want, rel=1e-12)


def test_class_index_targets_are_unaffected():
    x = mt.tensor(LOGITS, dtype="float64", requires_grad=True)
    F.cross_entropy(x, mt.tensor([2, 0], dtype="int64")).backward()
    probs = F.softmax(x.detach(), dim=1).tolist()
    probs[0][2] -= 1.0
    probs[1][0] -= 1.0
    for got, want in zip(x.grad.tolist(), probs):
        assert got == pytest.approx([v / 2 for v in want], rel=1e-12)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_logits_loss_takes_its_limit_at_infinite_logits(dtype):
    inf = math.inf
    x = mt.tensor([inf, -inf, inf, -inf], dtype=dtype, requires_grad=True)
    t = mt.tensor([1.0, 0.0, 0.25, 0.75], dtype=dtype)
    out = F.binary_cross_entropy_with_logits(x, t, reduction="none")
    # A perfect prediction costs nothing; a confident wrong one costs inf.
    assert out.tolist() == [0.0, 0.0, inf, inf]
    out[:2].sum().backward()
    assert x.grad.tolist() == [0.0, 0.0, 0.0, 0.0]


def test_logits_loss_with_pos_weight_at_infinite_logits():
    inf = math.inf
    x = mt.tensor([-inf, -inf], dtype="float64")
    t = mt.tensor([0.0, 1.0], dtype="float64")
    w = mt.tensor([3.0, 0.0], dtype="float64")
    out = F.binary_cross_entropy_with_logits(x, t, pos_weight=w, reduction="none")
    # The positive term is weighted by w * t, which is 0 for both.
    assert out.tolist() == [0.0, 0.0]
