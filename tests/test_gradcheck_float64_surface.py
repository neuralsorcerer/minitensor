# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""Central-difference gradient check over the differentiable op surface.

`test_gradcheck_differential.py` does this too, but in float32, where the
subtraction in a central difference cancels most of the significant digits and
the comparison has to be run at `rtol=3e-2`. A wrong gradient has to be wrong by
3% before that notices, and a factor like a missing `0.5` on one branch of a
piecewise activation can easily hide under it.

In float64 the same difference resolves to about 1e-10, so this runs at 1e-6 --
four orders of magnitude sharper -- and covers the ops that one does not: the
activations with analytical backwards (gelu, silu, elu, softplus, erf), the
cancellation-avoiding pairs (expm1, log1p), the normalising reductions
(softmax, log_softmax, logsumexp, var, std, norm) and the sequential ones
(prod, cumsum).

Inputs are deliberately kept away from the kinks of the piecewise ops, where a
central difference straddles two branches and measures neither.
"""

from __future__ import annotations

import numpy as np
import pytest

import minitensor as mt

SHAPE = (3, 4)
EPS = 1e-6
TOL = 1e-6


@pytest.fixture(autouse=True)
def _clean_graph():
    mt.clear_autograd_graph()
    yield
    mt.clear_autograd_graph()


def _sample(domain: str) -> np.ndarray:
    """A deterministic input for `domain`, chosen to avoid non-differentiable points."""
    rng = np.random.default_rng(20240611)
    if domain == "positive":
        return rng.random(SHAPE) * 2.0 + 0.3
    if domain == "unit":
        return rng.random(SHAPE) * 0.8 + 0.1
    if domain == "small":
        return rng.standard_normal(SHAPE) * 0.5
    if domain == "away_from_zero":
        # `relu`/`abs`/`elu`/`leaky_relu` are not differentiable at 0, and a
        # central difference across it returns the average of the two slopes.
        base = rng.standard_normal(SHAPE)
        return np.where(np.abs(base) < 0.2, np.sign(base + 1e-12) * 0.5, base)
    return rng.standard_normal(SHAPE)


# (name, function, input domain)
_OPS = [
    ("relu", mt.relu, "away_from_zero"),
    ("leaky_relu", mt.leaky_relu, "away_from_zero"),
    ("elu", mt.elu, "away_from_zero"),
    ("abs", mt.abs, "away_from_zero"),
    ("sigmoid", mt.sigmoid, "real"),
    ("tanh", mt.tanh, "real"),
    ("gelu", mt.gelu, "real"),
    ("silu", mt.silu, "real"),
    ("softplus", mt.softplus, "real"),
    ("erf", mt.erf, "real"),
    ("exp", mt.exp, "small"),
    ("expm1", mt.expm1, "small"),
    ("log", mt.log, "positive"),
    ("log1p", mt.log1p, "positive"),
    ("sqrt", mt.sqrt, "positive"),
    ("reciprocal", lambda t: 1.0 / t, "positive"),
    ("square", lambda t: t * t, "real"),
    ("cube", lambda t: t**3, "real"),
    ("sin", mt.sin, "real"),
    ("cos", mt.cos, "real"),
    ("tan", mt.tan, "small"),
    ("sinh", mt.sinh, "small"),
    ("cosh", mt.cosh, "small"),
    ("negate", lambda t: -t, "real"),
    ("clamp", lambda t: t.clamp(-0.5, 0.5), "small"),
    ("softmax", lambda t: mt.softmax(t, -1), "real"),
    ("log_softmax", lambda t: mt.log_softmax(t, -1), "real"),
    ("sum_all", mt.sum, "real"),
    ("mean_all", mt.mean, "real"),
    ("sum_axis", lambda t: mt.sum(t, 1), "real"),
    ("mean_axis", lambda t: mt.mean(t, 1), "real"),
    ("prod_axis", lambda t: mt.prod(t, 1), "positive"),
    ("logsumexp", lambda t: mt.logsumexp(t, 1), "real"),
    ("cumsum", lambda t: mt.cumsum(t, 1), "real"),
    ("var", lambda t: mt.var(t, 1), "real"),
    ("std", lambda t: mt.std(t, 1), "real"),
    ("norm", mt.norm, "real"),
    ("transpose", lambda t: t.transpose(0, 1), "real"),
    ("reshape", lambda t: t.reshape([4, 3]), "real"),
    ("slice", lambda t: t[1:3, 1:], "real"),
]


def _weights_for(shape) -> np.ndarray:
    """A fixed non-uniform upstream gradient.

    A plain `sum()` weights every output equally, which lets a backward that
    permutes or misroutes its outputs still produce the right total.
    """
    return np.random.default_rng(99).standard_normal(shape)


def _objective(fn, values: np.ndarray, weights: np.ndarray) -> float:
    with mt.no_grad():
        out = fn(mt.Tensor(values, dtype="float64")).numpy()
    return float((out * weights).sum())


@pytest.mark.parametrize("name,fn,domain", _OPS, ids=[case[0] for case in _OPS])
def test_gradient_matches_central_differences_in_float64(name, fn, domain):
    values = _sample(domain)

    tensor = mt.Tensor(values, dtype="float64").requires_grad_(True)
    out = fn(tensor)
    weights = _weights_for(tuple(out.shape))
    mt.sum(out * mt.Tensor(weights, dtype="float64")).backward()

    analytic = mt.get_gradient(tensor)
    assert analytic is not None, f"{name} produced no gradient"
    analytic = analytic.numpy()

    # Every element, not a sample: a backward that is wrong on one branch of a
    # piecewise op may be right everywhere else.
    numeric = np.zeros_like(values)
    for index in np.ndindex(*SHAPE):
        plus, minus = values.copy(), values.copy()
        plus[index] += EPS
        minus[index] -= EPS
        numeric[index] = (
            _objective(fn, plus, weights) - _objective(fn, minus, weights)
        ) / (2 * EPS)

    scale = np.maximum(1.0, np.abs(numeric))
    worst = float(np.max(np.abs(analytic - numeric) / scale))
    assert worst < TOL, f"{name}: worst relative error {worst:.3e}"
