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
