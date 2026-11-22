# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""`poisson_nll_loss`'s terms and gradients against exact values.

A log-rate `x` moves the term `exp(x) - t x` by `exp(x) - t`, and a rate by
`1 - t / (x + eps)`: two numbers that all but cancel where the model fits
its counts, and as tensor operations each was formed from two rounded terms
-- 1.7e7 ulps out in float32 and 4e8 in float64 near the fit. They are now
formed where they cancel without taking either term rounded first, and the
terms themselves rounded once.
"""

from __future__ import annotations

import decimal
from decimal import Decimal

import numpy as np
import pytest

import minitensor as mt
from minitensor import functional as F

_BOUND = {"float32": 1, "float64": 3}
_EPS = 1e-8


def _close(got: float, want: Decimal, dtype: str) -> bool:
    expected = float(want)
    unit = float(np.spacing(np.array(abs(expected), dtype=dtype)))
    if expected == 0.0:
        unit = float(np.finfo(dtype).smallest_subnormal)
    return abs(Decimal(float(got)) - want) <= _BOUND[dtype] * Decimal(unit)


def _exact(x: float, t: float, log_input: bool):
    """The term, and its slopes in `x` and in `t`, exactly."""

    X, T = Decimal(x), Decimal(t)
    if log_input:
        return X.exp() - T * X, X.exp() - T, -X
    shifted = X + Decimal(_EPS)
    return X - T * shifted.ln(), 1 - T / shifted, -shifted.ln()


def _cases(log_input: bool, dtype: str):
    """Inputs near the fit of each count, and some away from it."""

    rng = np.random.default_rng(107)
    counts = np.array([1.0, 2.0, 3.0, 10.0, 1000.0, 0.5, 7.0, 40.0])
    xs, ts = [], []
    for t in counts:
        # A rate stays positive.
        for delta in (1e-2, 1e-5, 1e-9, -1e-7, 0.3, -2.0 if log_input else -0.9):
            xs.append(np.log(t) + delta if log_input else t * (1 + delta))
            ts.append(t)
    # Counts of zero, and draws about the fit.
    xs += list(rng.uniform(-2.0, 2.0, 6) if log_input else rng.uniform(0.1, 3.0, 6))
    ts += [0.0] * 6
    drawn = rng.poisson(4.0, 40).astype(float) + 1.0
    noise = 0.05 * rng.standard_normal(40)
    xs += list(np.log(drawn) + noise if log_input else drawn * (1 + noise))
    ts += list(drawn)
    return np.array(xs).astype(dtype), np.array(ts).astype(dtype)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("log_input", [True, False])
def test_terms_and_gradients(log_input, dtype):
    x, t = _cases(log_input, dtype)
    xi = mt.Tensor(x, dtype=dtype, requires_grad=True)
    ti = mt.Tensor(t, dtype=dtype, requires_grad=True)
    terms = F.poisson_nll_loss(xi, ti, log_input, False, _EPS, "none")
    upstream = np.random.default_rng(109).uniform(0.5, 2.0, x.shape).astype(dtype)
    terms.backward(mt.Tensor(upstream, dtype=dtype))
    got_terms = terms.numpy().astype(np.float64)
    got_x, got_t = xi.grad.numpy(), ti.grad.numpy()
    with decimal.localcontext(prec=60):
        for k in range(len(x)):
            term, by_x, by_t = _exact(float(x[k]), float(t[k]), log_input)
            g = Decimal(float(upstream[k]))
            assert _close(got_terms[k], term, dtype), (k, x[k], t[k], "term")
            assert _close(got_x[k], g * by_x, dtype), (k, x[k], t[k], "input")
            assert _close(got_t[k], g * by_t, dtype), (k, x[k], t[k], "target")


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("reduction", ["mean", "sum"])
def test_reduced(reduction, dtype):
    x, t = _cases(True, dtype)
    xi = mt.Tensor(x, dtype=dtype, requires_grad=True)
    loss = F.poisson_nll_loss(
        xi, mt.Tensor(t, dtype=dtype), True, False, _EPS, reduction
    )
    loss.backward()
    scale = Decimal(1) / len(x) if reduction == "mean" else Decimal(1)
    with decimal.localcontext(prec=60):
        for k in range(len(x)):
            _, by_x, _ = _exact(float(x[k]), float(t[k]), True)
            assert _close(xi.grad.numpy()[k], scale * by_x, dtype), k


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_the_full_likelihood_keeps_its_stirling_term(dtype):
    # `full` adds Stirling's approximation to `log(t!)` for counts above
    # one: nothing to the input's gradient, and its own slope to the
    # target's.
    x = np.array([0.5, 1.2, 2.0, 0.1], dtype=dtype)
    t = np.array([0.0, 1.0, 5.0, 2.0], dtype=dtype)
    xi = mt.Tensor(x, dtype=dtype, requires_grad=True)
    ti = mt.Tensor(t, dtype=dtype, requires_grad=True)
    short = F.poisson_nll_loss(xi, ti, True, False, _EPS, "none").numpy()
    full = F.poisson_nll_loss(xi, ti, True, True, _EPS, "none")
    full.sum().backward()
    floored = np.maximum(t, 1.0)
    stirling = np.where(
        t > 1,
        floored * np.log(floored) - floored + 0.5 * np.log(2 * np.pi * floored),
        0.0,
    )
    np.testing.assert_allclose(full.numpy() - short, stirling, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(xi.grad.numpy(), np.exp(x) - t, rtol=1e-6)
    slope = np.where(t > 1, np.log(floored) + 0.5 / floored, 0.0)
    np.testing.assert_allclose(ti.grad.numpy(), slope - x, rtol=1e-5, atol=1e-6)
