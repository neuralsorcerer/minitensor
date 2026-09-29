# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""The loss functions that had no functional form, plus the parameters that
had no way in.

The engine implements eleven losses; six had Python bindings. `MAELoss`,
`HuberLoss` and `FocalLoss` were reachable only as `mt.nn` classes, `kl_div`
not at all, and `smooth_l1_loss`'s `beta` was pinned at 1.0 because the class
it was routed through has no field for it.
"""

from __future__ import annotations

import numpy as np
import pytest

import minitensor as mt
from minitensor import functional as F


def _smooth_l1(a, b, beta):
    d = np.abs(a - b)
    return np.where(d < beta, 0.5 * d * d / beta, d - 0.5 * beta)


def _huber(a, b, delta):
    d = np.abs(a - b)
    return np.where(d < delta, 0.5 * d * d, delta * (d - 0.5 * delta))


def _numeric_grad(fn, src, eps=1e-6):
    flat = src.reshape(-1).astype(np.float64).copy()
    out = np.zeros_like(flat)
    for i in range(flat.size):
        plus, minus = flat.copy(), flat.copy()
        plus[i] += eps
        minus[i] -= eps
        out[i] = (fn(plus.reshape(src.shape)) - fn(minus.reshape(src.shape))) / (
            2 * eps
        )
    return out.reshape(src.shape)


@pytest.fixture
def pair():
    rng = np.random.default_rng(4)
    return rng.standard_normal(64) * 2.0, rng.standard_normal(64) * 2.0


def test_smooth_l1_default_is_unchanged(pair):
    x, y = pair
    got = F.smooth_l1_loss(mt.as_tensor(x), mt.as_tensor(y)).item()
    assert np.isclose(got, _smooth_l1(x, y, 1.0).mean(), rtol=1e-12)


@pytest.mark.parametrize("beta", [0.1, 0.5, 1.0, 2.0, 5.0])
@pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
def test_smooth_l1_honours_beta(pair, beta, reduction):
    x, y = pair
    got = F.smooth_l1_loss(mt.as_tensor(x), mt.as_tensor(y), reduction, beta).numpy()
    want = _smooth_l1(x, y, beta)
    want = {"mean": want.mean(), "sum": want.sum(), "none": want}[reduction]
    np.testing.assert_allclose(got, want, rtol=1e-12)


@pytest.mark.parametrize("delta", [0.1, 0.5, 1.0, 2.0, 5.0])
def test_huber_matches_its_definition_and_scales_smooth_l1(pair, delta):
    x, y = pair
    tx, ty = mt.as_tensor(x), mt.as_tensor(y)

    got = F.huber_loss(tx, ty, "mean", delta).item()
    assert np.isclose(got, _huber(x, y, delta).mean(), rtol=1e-12)
    # huber(x, d) == d * smooth_l1(x, beta=d); they coincide only at 1.0, which
    # is why routing smooth-l1 straight to huber was right for the default and
    # wrong for every other beta.
    assert np.isclose(
        got, delta * F.smooth_l1_loss(tx, ty, "mean", delta).item(), rtol=1e-12
    )


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
@pytest.mark.parametrize("fn", [F.smooth_l1_loss, F.huber_loss])
def test_non_positive_or_non_finite_thresholds_are_rejected(pair, fn, bad):
    x, y = pair
    with pytest.raises(ValueError):
        fn(mt.as_tensor(x), mt.as_tensor(y), "mean", bad)


@pytest.mark.parametrize("reduction,agg", [("mean", np.mean), ("sum", np.sum)])
def test_l1_loss(pair, reduction, agg):
    x, y = pair
    got = F.l1_loss(mt.as_tensor(x), mt.as_tensor(y), reduction).item()
    assert np.isclose(got, agg(np.abs(x - y)), rtol=1e-12)


def test_kl_div_reductions_and_gradient_agree():
    # `mean` used to divide the forward by the batch dimension while the
    # backward divided by the element count, so the gradient came out
    # numel/batch times too small -- 4x for this shape.
    rng = np.random.default_rng(9)
    p = np.abs(rng.random((3, 4))) + 0.2
    q = np.abs(rng.random((3, 4))) + 0.2
    tp, tq = mt.as_tensor(p), mt.as_tensor(q)
    elementwise = q * (np.log(q) - np.log(p))

    assert np.isclose(F.kl_div(tp, tq, "sum").item(), elementwise.sum(), rtol=1e-12)
    assert np.isclose(F.kl_div(tp, tq, "mean").item(), elementwise.mean(), rtol=1e-12)
    assert np.isclose(
        F.kl_div(tp, tq, "batchmean").item(), elementwise.sum() / 3, rtol=1e-12
    )
    np.testing.assert_allclose(
        F.kl_div(tp, tq, "none").numpy(), elementwise, rtol=1e-12
    )

    for reduction in ("mean", "batchmean", "sum"):
        tensor = mt.Tensor(p.copy(), dtype="float64").requires_grad_(True)
        F.kl_div(tensor, tq, reduction).backward()
        numeric = _numeric_grad(
            lambda arr: F.kl_div(mt.Tensor(arr, dtype="float64"), tq, reduction).item(),
            p,
        )
        np.testing.assert_allclose(
            tensor.grad.numpy(), numeric, rtol=1e-5, atol=1e-8, err_msg=reduction
        )


def test_focal_loss_matches_its_definition():
    rng = np.random.default_rng(13)
    logits = rng.standard_normal((8, 4))
    labels = rng.integers(0, 4, size=8)
    onehot = np.eye(4)[labels]
    alpha, gamma = 0.25, 2.0

    shifted = logits - logits.max(-1, keepdims=True)
    log_p = shifted - np.log(np.exp(shifted).sum(-1, keepdims=True))
    probs = np.exp(log_p)
    expected = (alpha * ((1 - probs) ** gamma * (-log_p) * onehot).sum(-1)).mean()

    got = F.focal_loss(mt.as_tensor(logits), mt.as_tensor(onehot), alpha, gamma).item()
    assert np.isclose(got, expected, rtol=1e-10)


@pytest.mark.parametrize(
    "name,call",
    [
        ("smooth_l1", lambda t, y: F.smooth_l1_loss(t, y, "mean", 0.5)),
        ("huber", lambda t, y: F.huber_loss(t, y, "mean", 2.0)),
        ("l1", lambda t, y: F.l1_loss(t, y)),
    ],
)
def test_new_losses_are_differentiable(pair, name, call):
    x, y = pair
    x, y = x[:8], y[:8]
    ty = mt.as_tensor(y)

    tensor = mt.Tensor(x.copy(), dtype="float64").requires_grad_(True)
    call(tensor, ty).backward()
    numeric = _numeric_grad(
        lambda arr: call(mt.Tensor(arr, dtype="float64"), ty).item(), x
    )
    np.testing.assert_allclose(tensor.grad.numpy(), numeric, rtol=1e-5, atol=1e-8)


# Every loss class in the library, with the arguments it needs besides
# `reduction`. Derived by name so a new one has to be added here or noticed.
_LOSS_CLASSES = sorted(n for n in dir(mt.nn) if n.endswith("Loss"))


def test_the_loss_classes_are_the_ones_this_file_knows_about():
    assert _LOSS_CLASSES == [
        "BCELoss",
        "BCEWithLogitsLoss",
        "CrossEntropyLoss",
        "FocalLoss",
        "HuberLoss",
        "LogCoshLoss",
        "MAELoss",
        "MSELoss",
        "SmoothL1Loss",
    ], "a loss class was added or removed; give it a case below"


@pytest.mark.parametrize("name", _LOSS_CLASSES)
@pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
def test_a_loss_class_takes_the_three_reduction_modes(name, reduction):
    getattr(mt.nn, name)(reduction=reduction)


@pytest.mark.parametrize("name", _LOSS_CLASSES)
@pytest.mark.parametrize("reduction", ["average", "MEAN", "", "batchmean"])
def test_a_loss_class_refuses_a_mode_it_cannot_use_at_construction(name, reduction):
    """The function refused at the call; the class used to wait for a forward.

    `MSELoss("men")` built happily and failed at the first training step,
    where `mse_loss(..., "men")` refused immediately -- so which spelling you
    reached for decided how far a typo travelled. `batchmean` is in the list
    because only `kl_div` has a meaning for it, and these classes do not.
    """

    with pytest.raises(ValueError, match="[Rr]eduction"):
        getattr(mt.nn, name)(reduction=reduction)


@pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
def test_a_loss_class_and_its_function_agree(reduction):
    predictions = mt.Tensor(np.array([[0.2, 0.8], [0.4, 0.6]]), dtype="float64")
    targets = mt.Tensor(np.array([[0.0, 1.0], [1.0, 0.0]]), dtype="float64")
    for cls, function in (
        ("MSELoss", F.mse_loss),
        ("MAELoss", F.l1_loss),
        ("HuberLoss", F.huber_loss),
        ("SmoothL1Loss", F.smooth_l1_loss),
        ("LogCoshLoss", F.log_cosh_loss),
    ):
        from_class = getattr(mt.nn, cls)(reduction=reduction)(predictions, targets)
        from_function = function(predictions, targets, reduction)
        assert tuple(from_class.shape) == tuple(from_function.shape)
        np.testing.assert_allclose(from_class.numpy(), from_function.numpy())


# --- focal loss: one value per sample, and gradients for any targets --------

_RNG = np.random.default_rng(23)
_LOGITS = _RNG.standard_normal((3, 4))
_LABELS = mt.from_numpy(np.array([0, 1, 2], np.int64))
_SOFT = mt.Tensor(_RNG.dirichlet(np.ones(4), size=3), dtype="float64")
_WEIGHTS = _RNG.standard_normal(3)


def _central_difference(value, x, eps=1e-6):
    grad = np.zeros_like(x)
    for index in np.ndindex(x.shape):
        up, down = x.copy(), x.copy()
        up[index] += eps
        down[index] -= eps
        grad[index] = (value(up) - value(down)) / (2 * eps)
    return grad


@pytest.mark.parametrize("gamma", [0.0, 0.5, 2.0])
@pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
@pytest.mark.parametrize("targets", ["labels", "soft"])
def test_focal_loss_gradient_matches_central_differences(targets, reduction, gamma):
    """The backward was the one-hot special case -- `(p - t)` times a factor
    of the true class's probability -- so soft targets got a gradient off by
    8e-3 while their loss was right. And `"none"` returned per-class terms
    whose gradient was only right when every class of a sample was weighted
    alike; weighting samples differently put it off by 0.13."""
    target = _LABELS if targets == "labels" else _SOFT

    def loss(x, track=False):
        logits = mt.Tensor(x, dtype="float64", requires_grad=track)
        out = F.focal_loss(logits, target, 0.25, gamma, reduction)
        if reduction == "none":
            out = (out * mt.Tensor(_WEIGHTS, dtype="float64")).sum()
        return logits, out

    logits, out = loss(_LOGITS, track=True)
    out.backward()
    expected = _central_difference(lambda x: loss(x)[1].item(), _LOGITS)
    np.testing.assert_allclose(logits.grad.numpy(), expected, atol=1e-8)


# --- every loss: "none" is what "mean" and "sum" reduce ----------------------

_T = lambda values: mt.Tensor(np.asarray(values), dtype="float64")  # noqa: E731
_A, _B = _T(_RNG.standard_normal(5)), _T(_RNG.standard_normal(5))
_P = _T(_RNG.uniform(0.05, 0.95, 5))
_BITS = _T(_RNG.integers(0, 2, 5).astype(float))
_SIGNS = _T(_RNG.choice([-1.0, 1.0], 5))
_COUNTS = _T(_RNG.uniform(0.0, 3.0, 5))
_SCORES = _T(_RNG.standard_normal((5, 4)))
_CLASSES = mt.from_numpy(_RNG.integers(0, 4, 5).astype(np.int64))
_DIST = _T(_RNG.dirichlet(np.ones(4), 5))
_U, _V, _W = (_T(_RNG.standard_normal((5, 3))) for _ in range(3))

REDUCIBLE = {
    "mse_loss": lambda r: F.mse_loss(_A, _B, reduction=r),
    "l1_loss": lambda r: F.l1_loss(_A, _B, reduction=r),
    "huber_loss": lambda r: F.huber_loss(_A, _B, reduction=r),
    "smooth_l1_loss": lambda r: F.smooth_l1_loss(_A, _B, reduction=r),
    "log_cosh_loss": lambda r: F.log_cosh_loss(_A, _B, reduction=r),
    "binary_cross_entropy": lambda r: F.binary_cross_entropy(_P, _BITS, reduction=r),
    "cross_entropy": lambda r: F.cross_entropy(_SCORES, _CLASSES, reduction=r),
    "cross_entropy soft": lambda r: F.cross_entropy(_SCORES, _DIST, reduction=r),
    "nll_loss": lambda r: F.nll_loss(_SCORES, _CLASSES, reduction=r),
    "kl_div": lambda r: F.kl_div(_SCORES, _DIST, reduction=r),
    "focal_loss": lambda r: F.focal_loss(_SCORES, _CLASSES, reduction=r),
    "focal_loss soft": lambda r: F.focal_loss(_SCORES, _DIST, reduction=r),
    "soft_margin_loss": lambda r: F.soft_margin_loss(_A, _SIGNS, reduction=r),
    "poisson_nll_loss": lambda r: F.poisson_nll_loss(_A, _COUNTS, reduction=r),
    "hinge_embedding_loss": lambda r: F.hinge_embedding_loss(
        _COUNTS, _SIGNS, reduction=r
    ),
    "margin_ranking_loss": lambda r: F.margin_ranking_loss(_A, _B, _SIGNS, reduction=r),
    "cosine_embedding_loss": lambda r: F.cosine_embedding_loss(
        _U, _V, _SIGNS, reduction=r
    ),
    "triplet_margin_loss": lambda r: F.triplet_margin_loss(_U, _V, _W, reduction=r),
}


@pytest.mark.parametrize("name", list(REDUCIBLE))
def test_none_is_what_mean_and_sum_reduce(name):
    """`focal_loss(.., "none")` returned one term per class, so its mean was
    `1 / num_classes` of `focal_loss(.., "mean")`. Whatever `"none"` returns,
    `"mean"` and `"sum"` have to be its mean and its sum."""
    loss = REDUCIBLE[name]
    per_item = loss("none")
    assert np.isclose(per_item.mean().item(), loss("mean").item(), rtol=1e-12)
    assert np.isclose(per_item.sum().item(), loss("sum").item(), rtol=1e-12)


def test_focal_loss_none_has_one_value_per_sample():
    assert tuple(F.focal_loss(_SCORES, _CLASSES, reduction="none").shape) == (5,)
