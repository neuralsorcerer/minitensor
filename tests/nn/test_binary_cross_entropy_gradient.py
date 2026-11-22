# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""`binary_cross_entropy`'s gradients against the derivative of its forward.

The loss takes `-(t log p + (1 - t) log(1 - p))`, each logarithm floored at
-100. Its gradient in `p` is `(p - t) / (p (1 - p))` where neither logarithm
is floored, and the other term's alone where one is. It clamped that
denominator at `1e-12` instead, which no derivative of the forward does: a
float32 prediction of `1e-15` against a target of 0 had a slope of `0.001`
where it is 1, one of exactly 1 against a target of 0.3 a slope of `7e11`
where it is -0.3, and through a sigmoid a logit of -40 against a target of 1
was left with `-4e-6` of its `-1`.
"""

from __future__ import annotations

import decimal
import math
from decimal import Decimal

import numpy as np
import pytest

import minitensor as mt
from minitensor import functional as F

_BOUND = {"float32": 1, "float64": 2}


def _exact(p: float, t: float):
    """The slopes in `p` and in `t` of the floored forward, exactly."""

    P, T = Decimal(p), Decimal(t)
    log_p = P.ln() if P > 0 else Decimal(-1000)
    log_q = (1 - P).ln() if P < 1 else Decimal(-1000)
    if log_p >= -100 and log_q >= -100:
        by_p = (P - T) / (P * (1 - P))
    elif log_p >= -100:
        by_p = -T / P
    else:
        by_p = (1 - T) / (1 - P)
    return by_p, max(log_q, Decimal(-100)) - max(log_p, Decimal(-100))


def _close(got: float, want: Decimal, dtype: str) -> bool:
    expected = float(want)
    if expected == 0.0:
        return got == 0.0
    unit = Decimal(float(np.spacing(np.array(abs(expected), dtype=dtype))))
    return abs(Decimal(float(got)) - want) <= _BOUND[dtype] * unit


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_every_prediction(dtype):
    predictions = [1e-30, 1e-15, 1e-3, 0.0, 1.0, 1 - 2.0**-20, 0.3, 0.30001]
    predictions += [0.4999, 0.5000001, 0.7, 0.99]
    pairs = [(p, t) for p in predictions for t in (0.0, 1.0, 0.3)]
    p = np.array([pair[0] for pair in pairs], dtype=dtype)
    t = np.array([pair[1] for pair in pairs], dtype=dtype)
    pi = mt.Tensor(p, dtype=dtype, requires_grad=True)
    ti = mt.Tensor(t, dtype=dtype, requires_grad=True)
    upstream = np.random.default_rng(113).uniform(0.5, 2.0, p.shape).astype(dtype)
    F.binary_cross_entropy(pi, ti, "none").backward(mt.Tensor(upstream, dtype=dtype))
    with decimal.localcontext(prec=60):
        for k in range(len(p)):
            by_p, by_t = _exact(float(p[k]), float(t[k]))
            g = Decimal(float(upstream[k]))
            assert _close(pi.grad.numpy()[k], g * by_p, dtype), (p[k], t[k], "p")
            assert _close(ti.grad.numpy()[k], g * by_t, dtype), (p[k], t[k], "t")


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("reduction", ["mean", "sum"])
def test_reduced(reduction, dtype):
    p = np.array([1e-15, 0.2, 0.9, 1.0], dtype=dtype)
    t = np.array([0.0, 0.25, 1.0, 0.5], dtype=dtype)
    pi = mt.Tensor(p, dtype=dtype, requires_grad=True)
    F.binary_cross_entropy(pi, mt.Tensor(t, dtype=dtype), reduction).backward()
    scale = Decimal(1) / len(p) if reduction == "mean" else Decimal(1)
    with decimal.localcontext(prec=60):
        for k in range(len(p)):
            by_p, _ = _exact(float(p[k]), float(t[k]))
            assert _close(pi.grad.numpy()[k], scale * by_p, dtype), k


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("logit", [-40.0, -20.0, -1.0, 3.0, 20.0])
def test_through_a_sigmoid(logit, dtype):
    # The logit's gradient is `sigmoid(x) - t`, which the two steps give
    # whenever the sigmoid keeps a logarithm above the floor.
    x = mt.Tensor(np.array([logit]), dtype=dtype, requires_grad=True)
    target = mt.Tensor(np.array([1.0]), dtype=dtype)
    F.binary_cross_entropy(mt.sigmoid(x), target, "sum").backward()
    want = 1.0 / (1.0 + math.exp(-logit)) - 1.0
    np.testing.assert_allclose(x.grad.numpy()[0], want, rtol=1e-5)
