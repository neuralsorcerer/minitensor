# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""`binary_cross_entropy_with_logits`'s gradient against exact values.

The gradient `(1 - t) + (1 + (w - 1) t) (s(x) - 1)` took `s(x) - 1` from a
rounded `s(x)`, which is all rounding once `s(x)` is near one: a confident,
correct logit's gradient -- `-e^-x` for a target of one, `e^x` for zero -- came
out zero, or with a few digits of it, 1.7e7 ulps out in float32 and 9e15 in
float64, and a soft label's near its optimum some thousands. It is now
`(1 - t) s(x) - w t s(-x)`, each sigmoid of its own sign, in float64 and
rounded once, and as pairs where the two terms meet.
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


def _slope(x: float, t: float, w: float = 1.0) -> Decimal:
    with decimal.localcontext(prec=60, Emax=10**6, Emin=-(10**6)):
        d, t, w = Decimal(x), Decimal(t), Decimal(w)
        s = 1 / (1 + (-d).exp())
        return +((1 - t) * s - w * t * (1 - s))


def _gradients(xs, ts, dtype, pos_weight=None, reduction="sum"):
    x = mt.Tensor(np.asarray(xs), dtype=dtype, requires_grad=True)
    t = mt.Tensor(np.asarray(ts), dtype=dtype)
    kwargs = {}
    if pos_weight is not None:
        kwargs["pos_weight"] = mt.Tensor(np.asarray(pos_weight), dtype=dtype)
    F.binary_cross_entropy_with_logits(x, t, reduction=reduction, **kwargs).backward()
    got = x.grad.numpy().astype(np.float64).tolist()
    mt.clear_autograd_graph()
    return got


def _check(xs, ts, dtype, bound, ws=None):
    got = _gradients(xs, ts, dtype, pos_weight=ws)
    xv = np.asarray(xs, dtype=dtype).astype(np.float64).tolist()
    tv = np.asarray(ts, dtype=dtype).astype(np.float64).tolist()
    wv = (
        [1.0] * len(xv)
        if ws is None
        else np.asarray(ws, dtype=dtype).astype(np.float64).tolist()
    )
    for x, t, w, g in zip(xv, tv, wv, got):
        want = _rounded(_slope(x, t, w), dtype)
        if dtype == "float32":
            assert g == want, (x, t, w, g, want)
        else:
            assert abs(g - want) <= bound * math.ulp(want), (x, t, w, g, want)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_hard_labels_keep_a_confident_logits_gradient(dtype):
    rng = np.random.default_rng(18)
    xs = np.concatenate([rng.uniform(0.0, 40.0, 400), rng.uniform(-40.0, 0.0, 400)])
    ts = np.concatenate([np.ones(400), np.zeros(400)])
    # And the same logits against the other label, confidently wrong.
    _check(np.concatenate([xs, xs]), np.concatenate([ts, 1 - ts]), dtype, bound=2)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_soft_labels_about_their_optimum(dtype):
    rng = np.random.default_rng(19)
    ts = rng.choice([0.05, 0.1, 0.3, 0.9, 0.95], 600)
    optimum = np.log(ts / (1 - ts))
    xs = optimum + rng.uniform(-1e-3, 1e-3, 600) * 10.0 ** rng.uniform(-9, 0, 600)
    _check(xs, ts, dtype, bound=2)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_with_a_positive_class_weight(dtype):
    rng = np.random.default_rng(20)
    xs = rng.uniform(-30.0, 30.0, 600)
    ts = rng.choice([0.0, 1.0, 0.2, 0.7], 600)
    ws = rng.uniform(0.2, 5.0, 600)
    _check(xs, ts, dtype, bound=3, ws=ws)


def test_mean_divides_by_the_count():
    rng = np.random.default_rng(21)
    xs = rng.uniform(-10.0, 10.0, 7)
    ts = (rng.random(7) < 0.5).astype(float)
    got = _gradients(xs, ts, "float64", reduction="mean")
    for x, t, g in zip(xs.tolist(), ts.tolist(), got):
        want = _rounded(_slope(x, t) / 7, "float64")
        assert abs(g - want) <= 2 * math.ulp(want), (x, t, g, want)
