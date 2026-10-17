# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""A loss of two dtypes is computed at the float they promote to.

The predictions are a model's activations and must be floats, as they must
for every layer-shaped op. The targets may be of any dtype, and meet the
predictions as arithmetic meets them: the wider float decides, and an integer
or bool target takes the predictions' width. Most of the losses required the
two dtypes to match, which refused a float32 prediction against float64
targets and every integer target -- `binary_cross_entropy(p, [1, 0])` among
them -- and the margin losses failed on an integer label. The casts are
differentiable, so a gradient still reaches each operand at its own dtype.
"""

from __future__ import annotations

import math

import pytest

import minitensor as mt

F = mt.functional

PAIRWISE = [
    "mse_loss",
    "l1_loss",
    "smooth_l1_loss",
    "huber_loss",
    "binary_cross_entropy",
    "binary_cross_entropy_with_logits",
    "kl_div",
    "poisson_nll_loss",
]
LABELLED = ["soft_margin_loss", "hinge_embedding_loss"]


def _t(values, dtype):
    return mt.Tensor(values, dtype=dtype)


def _close(got, want):
    if math.isinf(want):
        return got == want
    return abs(got - want) <= 1e-6 * max(1.0, abs(want))


@pytest.mark.parametrize("name", PAIRWISE)
@pytest.mark.parametrize(
    "first,second,common",
    [
        ("float32", "int64", "float32"),
        ("float64", "int32", "float64"),
        ("float32", "float64", "float64"),
        ("float64", "float32", "float64"),
        ("float32", "bool", "float32"),
    ],
)
def test_a_pair_of_dtypes_meets_at_the_float_it_promotes_to(
    name, first, second, common
):
    fn = getattr(F, name)
    predictions = _t([0.75, 0.25], first)
    if second == "bool":
        targets = mt.Tensor([True, False])
    elif "int" in second:
        targets = _t([1, 0], second)
    else:
        targets = _t([0.5, 0.25], second)
    got = fn(predictions, targets)
    assert str(got.dtype) == common
    want = fn(predictions.astype("float64"), targets.astype("float64"))
    assert _close(got.item(), want.item())


@pytest.mark.parametrize("name", PAIRWISE)
def test_an_integer_prediction_is_still_refused(name):
    # The other side of the line: a prediction is an activation.
    with pytest.raises(ValueError, match="floating point"):
        getattr(F, name)(_t([1, 0], "int64"), _t([0.5, 0.25], "float32"))


@pytest.mark.parametrize("name", LABELLED)
def test_an_integer_label_is_read_as_a_number(name):
    fn = getattr(F, name)
    scores = _t([0.75, 2.0], "float32")
    got = fn(scores, _t([1, -1], "int64"))
    want = fn(scores.astype("float64"), _t([1.0, -1.0], "float64"))
    assert str(got.dtype) == "float32"
    assert _close(got.item(), want.item())
    with pytest.raises(ValueError, match="floating point"):
        fn(_t([1, 2], "int32"), _t([1, -1], "int64"))


def test_margin_ranking_and_cosine_embedding_take_integer_labels():
    got = F.margin_ranking_loss(
        _t([1.0, 3.0], "float32"), _t([2.0, 1.0], "float64"), [1, -1]
    )
    want = F.margin_ranking_loss(
        _t([1.0, 3.0], "float64"),
        _t([2.0, 1.0], "float64"),
        _t([1.0, -1.0], "float64"),
    )
    assert str(got.dtype) == "float64" and _close(got.item(), want.item())
    pairs = F.cosine_embedding_loss(
        _t([[1.0, 0.0]], "float32"), _t([[1.0, 1.0]], "float32"), [1]
    )
    assert str(pairs.dtype) == "float32"
    assert _close(pairs.item(), 1 - 1 / math.sqrt(2))


def test_the_gradient_reaches_each_operand_at_its_own_dtype():
    prediction = _t([0.5, 1.0], "float32").requires_grad_(True)
    loss = F.mse_loss(prediction, _t([1.0, 0.0], "float64"))
    assert str(loss.dtype) == "float64"
    loss.backward()
    assert str(prediction.grad.dtype) == "float32"
    assert prediction.grad.tolist() == [-0.5, 1.0]


def test_a_positive_weight_of_another_dtype_is_read_at_the_loss_dtype():
    logits, targets = _t([0.9, -0.2], "float32"), _t([1, 0], "int64")
    got = F.binary_cross_entropy_with_logits(
        logits, targets, pos_weight=_t([2.0], "float64")
    )
    want = F.binary_cross_entropy_with_logits(
        logits, targets.astype("float32"), pos_weight=_t([2.0], "float32")
    )
    assert str(got.dtype) == "float32" and got.item() == want.item()
    listed = F.binary_cross_entropy_with_logits(logits, [1, 0], pos_weight=[2.0])
    assert listed.item() == want.item()


def test_a_negative_integer_count_is_refused_as_a_negative_float_is():
    # The check read float targets only, so an integer -1 passed as a count.
    with pytest.raises(ValueError, match="count"):
        F.poisson_nll_loss(_t([0.5, 1.0], "float32"), _t([-1, 2], "int64"))
    with pytest.raises(ValueError, match="count"):
        F.poisson_nll_loss(_t([0.5, 1.0], "float32"), _t([-1.0, 2.0], "float32"))


def test_a_python_list_target_is_read_beside_the_predictions():
    predictions = _t([0.75, 0.25], "float64")
    for name in PAIRWISE:
        fn = getattr(F, name)
        assert (
            fn(predictions, [1, 0]).item()
            == fn(predictions, _t([1.0, 0.0], "float64")).item()
        ), name


@pytest.mark.parametrize(
    "module", ["MSELoss", "HuberLoss", "SmoothL1Loss", "BCELoss", "BCEWithLogitsLoss"]
)
def test_the_loss_modules_take_integer_targets(module):
    loss = getattr(mt.nn, module)()
    prediction = _t([0.75, 0.25], "float32")
    got = loss(prediction, mt.Tensor([1, 0]))
    assert got.item() == loss(prediction, _t([1.0, 0.0], "float32")).item()
