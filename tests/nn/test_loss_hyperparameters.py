# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""Loss hyperparameters must be values a loss can use.

A NaN or infinite margin, eps or gamma is not a setting. Several were
accepted, and each turned every loss NaN, infinite or -- for focal loss's
`gamma` -- silently zero: a run that went wrong at its first step with nothing
to say why. The range checks that did exist were written so a NaN compared
false against both bounds and passed. Optimizer hyperparameters already follow
this rule; the losses now do too, and so do the loss layers' constructors,
which validated nothing.

Two values that were refused have a meaning and are now accepted:
`smooth_l1_loss(beta=0)` is the L1 loss, the limit as the quadratic region
shrinks away, and focal loss's `alpha=1` -- `alpha` scales every class alike,
so `1` is simply no weighting.
"""

import math

import numpy as np
import pytest

import minitensor as mt

F = mt.functional
nn = mt.nn
RNG = np.random.default_rng(37)


def _t(values):
    return mt.from_numpy(np.asarray(values, dtype=np.float64))


X = _t(RNG.standard_normal(4))
Y = _t(RNG.standard_normal(4))
SIGNS = _t([1.0, -1.0, 1.0, -1.0])
ROWS = _t(RNG.standard_normal((4, 3)))
CLASSES = mt.from_numpy(np.array([0, 1, 2, 0]))

CALLS = {
    "huber_loss delta": lambda v: F.huber_loss(X, Y, "mean", v),
    "smooth_l1_loss beta": lambda v: F.smooth_l1_loss(X, Y, "mean", v),
    "margin_ranking_loss margin": lambda v: F.margin_ranking_loss(X, Y, SIGNS, v),
    "hinge_embedding_loss margin": lambda v: F.hinge_embedding_loss(X.abs(), SIGNS, v),
    "triplet_margin_loss margin": lambda v: F.triplet_margin_loss(
        ROWS, ROWS * 2.0, ROWS * 3.0, v
    ),
    "triplet_margin_loss eps": lambda v: F.triplet_margin_loss(
        ROWS, ROWS * 2.0, ROWS * 3.0, 1.0, 2.0, v
    ),
    "poisson_nll_loss eps": lambda v: F.poisson_nll_loss(
        X.abs(), Y.abs(), False, False, v
    ),
    "cosine_similarity eps": lambda v: F.cosine_similarity(ROWS, ROWS, 1, v),
    "focal_loss alpha": lambda v: F.focal_loss(ROWS, CLASSES, v, 2.0),
    "focal_loss gamma": lambda v: F.focal_loss(ROWS, CLASSES, 0.25, v),
    "HuberLoss delta": lambda v: nn.HuberLoss(delta=v),
    "SmoothL1Loss beta": lambda v: nn.SmoothL1Loss(beta=v),
    "FocalLoss alpha": lambda v: nn.FocalLoss(alpha=v),
    "FocalLoss gamma": lambda v: nn.FocalLoss(gamma=v),
}


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
@pytest.mark.parametrize("name", sorted(CALLS))
def test_a_nan_or_infinite_hyperparameter_is_refused(name, value):
    with pytest.raises(ValueError):
        CALLS[name](value)


@pytest.mark.parametrize(
    "name",
    [
        "huber_loss delta",
        "smooth_l1_loss beta",
        "triplet_margin_loss eps",
        "poisson_nll_loss eps",
        "cosine_similarity eps",
        "focal_loss alpha",
        "focal_loss gamma",
        "HuberLoss delta",
        "SmoothL1Loss beta",
        "FocalLoss alpha",
        "FocalLoss gamma",
    ],
)
def test_a_negative_where_the_value_is_a_size_is_refused(name):
    with pytest.raises(ValueError):
        CALLS[name](-1.0)


@pytest.mark.parametrize(
    "name",
    [
        "margin_ranking_loss margin",
        "hinge_embedding_loss margin",
        "triplet_margin_loss margin",
    ],
)
def test_a_negative_margin_is_still_a_margin(name):
    assert math.isfinite(CALLS[name](-0.5).item())


def test_smooth_l1_at_zero_beta_is_the_l1_loss_and_its_gradient():
    for reduction in ("none", "mean", "sum"):
        np.testing.assert_array_equal(
            F.smooth_l1_loss(X, Y, reduction, 0.0).numpy(),
            F.l1_loss(X, Y, reduction).numpy(),
        )
    x = X.detach().requires_grad_(True)
    F.smooth_l1_loss(x, Y, "sum", 0.0).backward()
    np.testing.assert_array_equal(x.grad.numpy(), np.sign(X.numpy() - Y.numpy()))
    layer = nn.SmoothL1Loss(beta=0.0)
    assert layer.beta == 0.0
    assert layer(X, Y).item() == F.l1_loss(X, Y).item()


def test_the_smooth_l1_layer_takes_beta_and_shows_it():
    layer = nn.SmoothL1Loss("sum", beta=0.5)
    assert layer.beta == 0.5
    assert repr(layer) == "SmoothL1Loss(reduction='sum', beta=0.5)"
    assert repr(nn.SmoothL1Loss()) == "SmoothL1Loss(reduction='mean')"
    assert layer(X, Y).item() == pytest.approx(
        F.smooth_l1_loss(X, Y, "sum", 0.5).item()
    )


def test_focal_alpha_of_one_is_no_weighting():
    unit = F.focal_loss(ROWS, CLASSES, 1.0, 0.0).item()
    assert unit == pytest.approx(F.cross_entropy(ROWS, CLASSES).item(), rel=1e-12)
    assert F.focal_loss(ROWS, CLASSES, 0.5, 0.0).item() == pytest.approx(unit / 2)
    assert (
        repr(nn.FocalLoss(1.0, 2.0))
        == "FocalLoss(alpha=1.0, gamma=2.0, reduction='mean')"
    )


def test_an_infinite_norm_order_is_the_max_norm():
    anchor, positive, negative = (RNG.standard_normal((4, 3)) for _ in range(3))

    def distance(u, v):
        return np.abs(u - v + 1e-6).max(1)

    expected = np.maximum(
        0, distance(anchor, positive) - distance(anchor, negative) + 1
    )
    got = F.triplet_margin_loss(
        _t(anchor), _t(positive), _t(negative), 1.0, math.inf, 1e-6, False, "none"
    )
    np.testing.assert_allclose(got.numpy(), expected, rtol=1e-12)
