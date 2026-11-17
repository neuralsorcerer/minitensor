# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""`kl_div`, `binary_cross_entropy` and `log_cosh_loss` keep their digits.

Each formed its terms from pieces that cancel exactly where a model does well:

* `kl_div` took `t (log t - log p)`, the difference of two rounded logarithms,
  which cancelled where the prediction is near the target, tens of ulps out;
* `binary_cross_entropy` took `log(1 - p)` of a rounded `1 - p`, so a small
  prediction against a target of 0 -- a loss near `p` -- was mostly that
  rounding;
* `log_cosh_loss` took `|d| + log1p(exp(-2|d|)) - log 2`, a number near
  `log 2` less `log 2` for a small difference, tens of thousands of ulps out.

Each term is now formed without cancelling, in float64, and a float32 one is
narrowed once.
"""

from __future__ import annotations

import math
from decimal import Decimal, localcontext
from fractions import Fraction

import numpy as np
import pytest

import minitensor as mt
from minitensor import functional as F


def _float32(value: Fraction) -> float:
    """The float32 nearest `value`, ties to even."""

    if value == 0:
        return 0.0
    sign = -1.0 if value < 0 else 1.0
    magnitude = abs(value)
    exponent = magnitude.numerator.bit_length() - magnitude.denominator.bit_length()
    if Fraction(2) ** exponent > magnitude:
        exponent -= 1
    unit = Fraction(2) ** max(exponent - 23, -149)
    rounded = round(magnitude / unit) * unit
    return sign * (math.inf if rounded >= Fraction(2) ** 128 else float(rounded))


def _nearest(value: Decimal, dtype: str) -> float:
    exact = Fraction(value)
    return _float32(exact) if dtype == "float32" else float(exact)


def _kl(t: float, p: float) -> Decimal:
    if t == 0:
        return Decimal(0)
    return Decimal(t) * (Decimal(t) / Decimal(p)).ln()


def _bce(p: float, t: float) -> Decimal:
    def floor(log):
        return max(log, Decimal(-100))

    log_p = floor(Decimal(p).ln()) if p > 0 else Decimal(-100)
    log_q = floor((1 - Decimal(p)).ln()) if p < 1 else Decimal(-100)
    return -(Decimal(t) * log_p + (1 - Decimal(t)) * log_q)


def _log_cosh(p: float, t: float) -> Decimal:
    d = Decimal(p) - Decimal(t)
    return ((d.exp() + (-d).exp()) / 2).ln()


def _check(got, want, dtype, ulps):
    """Float32 terms rounded once; float64 ones within `ulps`."""

    for g, w in zip(got, want):
        if dtype == "float32":
            assert g == w
        else:
            assert abs(g - w) <= ulps * math.ulp(w), (g, w)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_kl_div_near_the_target(dtype):
    rng = np.random.default_rng(1153)
    t = rng.uniform(0.01, 1.0, 400)
    # Half the predictions within a hair of the target, the rest anywhere.
    p = np.concatenate(
        [t[:200] * (1 + rng.standard_normal(200) * 10.0 ** rng.uniform(-12, -2, 200))]
        + [rng.uniform(1e-6, 1.0, 200)]
    )
    t, p = t.astype(dtype), np.abs(p).astype(dtype)
    got = F.kl_div(
        mt.Tensor(p, dtype=dtype), mt.Tensor(t, dtype=dtype), reduction="none"
    ).tolist()
    with localcontext() as context:
        context.prec = 60
        want = [_nearest(_kl(float(a), float(b)), dtype) for a, b in zip(t, p)]
    _check(got, want, dtype, 2)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_binary_cross_entropy_of_a_small_prediction(dtype):
    rng = np.random.default_rng(1163)
    p = np.concatenate(
        [10.0 ** rng.uniform(-30, -1, 200), 1 - 10.0 ** rng.uniform(-7, -1, 100)]
    )
    p = np.clip(p, 0, 1).astype(dtype)
    t = np.concatenate([np.zeros(200), rng.uniform(0, 1, 100)]).astype(dtype)
    got = F.binary_cross_entropy(
        mt.Tensor(p, dtype=dtype), mt.Tensor(t, dtype=dtype), reduction="none"
    ).tolist()
    with localcontext() as context:
        context.prec = 60
        want = [_nearest(_bce(float(a), float(b)), dtype) for a, b in zip(p, t)]
    _check(got, want, dtype, 2)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_log_cosh_of_a_small_difference(dtype):
    rng = np.random.default_rng(1171)
    p = rng.standard_normal(400) * 3
    t = p + rng.standard_normal(400) * 10.0 ** rng.uniform(-9, 1.5, 400)
    p, t = p.astype(dtype), t.astype(dtype)
    got = F.log_cosh_loss(
        mt.Tensor(p, dtype=dtype), mt.Tensor(t, dtype=dtype), reduction="none"
    ).tolist()
    with localcontext() as context:
        context.prec = 80
        want = [_nearest(_log_cosh(float(a), float(b)), dtype) for a, b in zip(p, t)]
    _check(got, want, dtype, 3)


def test_log_cosh_ends_and_reductions():
    inf, nan = math.inf, math.nan
    p = mt.Tensor([inf, -inf, nan, 0.0, 1e308], dtype="float64")
    t = mt.Tensor([0.0, 0.0, 0.0, 0.0, -1e308], dtype="float64")
    got = F.log_cosh_loss(p, t, reduction="none").tolist()
    assert got[0] == inf and got[1] == inf and math.isnan(got[2])
    assert got[3] == 0.0 and got[4] == inf
    rng = np.random.default_rng(1181)
    a, b = rng.standard_normal(50), rng.standard_normal(50)
    terms = F.log_cosh_loss(mt.Tensor(a), mt.Tensor(b), reduction="none")
    total = F.log_cosh_loss(mt.Tensor(a), mt.Tensor(b), reduction="sum").item()
    assert total == pytest.approx(math.fsum(terms.tolist()), rel=1e-15)


@pytest.mark.parametrize("reduction", ["none", "mean", "sum"])
def test_log_cosh_gradient(reduction):
    rng = np.random.default_rng(1187)
    a0, b0 = rng.standard_normal(30) * 3, rng.standard_normal(30) * 3
    a = mt.Tensor(a0, requires_grad=True)
    b = mt.Tensor(b0, requires_grad=True)
    loss = F.log_cosh_loss(a, b, reduction=reduction)
    weights = rng.standard_normal(30)
    if reduction == "none":
        loss.backward(mt.Tensor(weights))
    else:
        loss.backward()
    want = np.tanh(a0 - b0)
    if reduction == "mean":
        want = want / 30
    if reduction == "none":
        want = want * weights
    np.testing.assert_allclose(a.grad.numpy(), want, rtol=1e-14, atol=1e-300)
    np.testing.assert_allclose(b.grad.numpy(), -want, rtol=1e-14, atol=1e-300)
