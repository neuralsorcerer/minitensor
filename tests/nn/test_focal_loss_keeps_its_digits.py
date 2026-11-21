# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""`focal_loss` and its gradients against exact values.

The loss `alpha sum_c t_c (1 - p_c)^gamma (-ln p_c)` took `1 - p` as one less
`exp(log_softmax(z))`, a probability near one taken from one in a confident
row -- zero in float32 from `p = 1 - 2^-24` -- so that the loss and its
gradient there were all rounding, 1.7e7 ulps out in float32 and 3e13 in
float64, and an ordinary row's some tens of ulps. Both are now formed from the
logits in float64, with the likeliest class's `1 - p` the sum of the others'
probabilities, and rounded once: a float32 value is the float nearest the exact
one, and a float64 one within `4 + 2 gamma` ulps, its `(1 - p)^gamma` carrying
`gamma` times the error of `1 - p`.
"""

from __future__ import annotations

import decimal
import math
from decimal import Decimal
from fractions import Fraction

import numpy as np
import pytest

import minitensor as mt
from minitensor import functional as F

_DIGITS = {"float32": (24, -149, 128), "float64": (53, -1074, 1024)}
_ALPHA = 0.25
_CLASSES = 6


def _rounded(value: Decimal, dtype: str) -> float:
    """The float nearest a decimal value, ties to even, rounded once."""

    if value == 0:
        return 0.0
    exact = Fraction(abs(value))
    digits, least, top = _DIGITS[dtype]
    e = exact.numerator.bit_length() - exact.denominator.bit_length()
    if Fraction(2) ** e > exact:
        e -= 1
    unit = Fraction(2) ** max(e - (digits - 1), least)
    near = round(exact / unit) * unit
    out = math.inf if near >= Fraction(2) ** top else float(near)
    return -out if value < 0 else out


def _bound(dtype: str, gamma: float) -> float:
    return 1 if dtype == "float32" else 4 + 2 * gamma


def _ulps(got: float, want: float, dtype: str) -> float:
    if got == want:
        return 0.0
    return abs(got - want) / float(np.spacing(np.array(abs(want), dtype=dtype)))


def _exact_row(z, t, gamma):
    """A row's loss, its gradients in the logits and the targets, and the
    larger of the two terms of each logit's gradient, exactly."""

    top = max(z)
    w = [(Decimal(v) - Decimal(top)).exp() for v in z]
    total = sum(w)
    log_total = total.ln()
    n = len(z)
    p = [x / total for x in w]
    q = [sum(w[j] for j in range(n) if j != k) / total for k in range(n)]
    lp = [Decimal(v) - Decimal(top) - log_total for v in z]
    g, a = Decimal(gamma), Decimal(_ALPHA)

    def power(x):
        return Decimal(1) if gamma == 0 else (g * x.ln()).exp()

    loss = a * sum(Decimal(t[k]) * power(q[k]) * -lp[k] for k in range(n) if t[k])
    u = [
        -Decimal(t[k]) * power(q[k]) * (1 + g * p[k] * -lp[k] / q[k]) if t[k] else 0
        for k in range(n)
    ]
    total_u = sum(u)
    logits = [a * (u[k] - p[k] * total_u) for k in range(n)]
    terms = [a * max(abs(u[k] * q[k]), abs(p[k] * (total_u - u[k]))) for k in range(n)]
    targets = [a * power(q[k]) * -lp[k] for k in range(n)]
    return loss, logits, targets, terms


def _rows(rows, confident, rng):
    z = rng.standard_normal((rows, _CLASSES)) * 3.0
    k = rng.integers(0, _CLASSES, rows)
    if confident:
        z[np.arange(rows), k] = z.max(axis=1) + rng.uniform(3.0, 30.0, rows)
    return z, k


def _run(z, target, gamma, dtype, weights, target_grad=False):
    x = mt.Tensor(z, dtype=dtype, requires_grad=True)
    loss = F.focal_loss(x, target, _ALPHA, gamma, "none")
    loss.backward(mt.Tensor(weights, dtype=dtype))
    out = loss.numpy().astype(np.float64), x.grad.numpy().astype(np.float64)
    if target_grad:
        out += (target.grad.numpy().astype(np.float64),)
    mt.clear_autograd_graph()
    return out


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("gamma", [0.0, 0.5, 2.0])
@pytest.mark.parametrize("confident", [True, False], ids=["confident", "ordinary"])
def test_class_index_targets(confident, gamma, dtype):
    rng = np.random.default_rng(41)
    z, k = _rows(60, confident, rng)
    z = z.astype(dtype)
    weights = rng.uniform(0.5, 2.0, 60).astype(dtype)
    target = mt.Tensor(k.astype(np.int64), dtype="int64")
    loss, grad = _run(z, target, gamma, dtype, weights)
    bound = _bound(dtype, gamma)
    with decimal.localcontext(prec=60, Emin=-(10**6), Emax=10**6):
        for r in range(60):
            t = [0.0] * _CLASSES
            t[k[r]] = 1.0
            want, slopes, _, _ = _exact_row(z[r].astype(np.float64).tolist(), t, gamma)
            expected = _rounded(want, dtype)
            assert _ulps(loss[r], expected, dtype) <= bound, (r, loss[r], expected)
            w = Decimal(float(weights[r]))
            for c in range(_CLASSES):
                expected = _rounded(w * slopes[c], dtype)
                assert _ulps(grad[r, c], expected, dtype) <= bound, (
                    r,
                    c,
                    grad[r, c],
                    expected,
                )


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("gamma", [0.5, 2.0])
def test_soft_targets(gamma, dtype):
    # A soft target's gradient in a logit is two terms of opposite signs,
    # which cancel where they meet: to a few ulps of the larger. Its loss and
    # its gradient in the targets cancel nowhere.
    rng = np.random.default_rng(43)
    z, _ = _rows(40, False, rng)
    z = z.astype(dtype)
    t = rng.dirichlet(np.ones(_CLASSES), 40).astype(dtype)
    weights = rng.uniform(0.5, 2.0, 40).astype(dtype)
    target = mt.Tensor(t, dtype=dtype, requires_grad=True)
    loss, grad, target_grad = _run(z, target, gamma, dtype, weights, True)
    bound = _bound(dtype, gamma)
    with decimal.localcontext(prec=60, Emin=-(10**6), Emax=10**6):
        for r in range(40):
            row_t = t[r].astype(np.float64).tolist()
            want, slopes, target_slopes, terms = _exact_row(
                z[r].astype(np.float64).tolist(), row_t, gamma
            )
            assert _ulps(loss[r], _rounded(want, dtype), dtype) <= bound, r
            w = Decimal(float(weights[r]))
            for c in range(_CLASSES):
                expected = _rounded(w * target_slopes[c], dtype)
                assert _ulps(target_grad[r, c], expected, dtype) <= bound, (r, c)
                scale = float(w * terms[c])
                unit = float(np.spacing(np.array(scale, dtype=dtype)))
                assert abs(grad[r, c] - float(w * slopes[c])) <= bound * unit, (r, c)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("reduction", ["mean", "sum"])
def test_reduced(reduction, dtype):
    rng = np.random.default_rng(47)
    z, k = _rows(20, True, rng)
    z = z.astype(dtype)
    x = mt.Tensor(z, dtype=dtype, requires_grad=True)
    target = mt.Tensor(k.astype(np.int64), dtype="int64")
    F.focal_loss(x, target, _ALPHA, 2.0, reduction).backward()
    got = x.grad.numpy().astype(np.float64)
    scale = Decimal(1) / 20 if reduction == "mean" else Decimal(1)
    bound = _bound(dtype, 2.0)
    with decimal.localcontext(prec=60, Emin=-(10**6), Emax=10**6):
        for r in range(20):
            t = [0.0] * _CLASSES
            t[k[r]] = 1.0
            _, slopes, _, _ = _exact_row(z[r].astype(np.float64).tolist(), t, 2.0)
            for c in range(_CLASSES):
                expected = _rounded(scale * slopes[c], dtype)
                assert _ulps(got[r, c], expected, dtype) <= bound, (r, c)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_the_ends(dtype):
    inf, nan = math.inf, math.nan
    target = mt.Tensor(np.array([0, 1, 2]), dtype="int64")
    # A logit of `+inf` takes all of the mass, and its class's loss is zero; a
    # NaN makes its row NaN; and a row whose target class is certain to the
    # last bit -- every other probability below float64 -- has no loss.
    z = np.array([[inf, 0.0, 1.0], [nan, 0.0, 1.0], [0.0, -800.0, 800.0]])
    loss = F.focal_loss(mt.Tensor(z, dtype=dtype), target, _ALPHA, 2.0, "none")
    got = loss.tolist()
    assert got[0] == 0.0 and math.isnan(got[1]) and got[2] == 0.0
