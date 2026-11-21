# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""`log_softmax`'s and `softmax`'s gradients against exact values.

`log_softmax`'s backward took its probabilities as `exp` of its rounded
output, `|log p|` ulps of `p` -- some thirty at `p = e^-40` -- and formed
`g_c - p_c S`, which for `nll_loss` took a probability near one from one: a
confident, correct row's gradient on its target was all rounding, 1.7e7 ulps
out in float32 and 9e15 in float64. It now forms the probabilities from the
input, and `g_c (1 - p_c) - p_c (S - g_c)` with the likeliest class's `1 - p`
the sum of the rest. `softmax`'s backward, `y (g - dot)`, summed the dot in the
tensor's dtype and cancelled the same way; it is now float64, likewise.
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


def _softmax(row: list[float]) -> list[Decimal]:
    top = max(row)
    weights = [(Decimal(v) - Decimal(top)).exp() for v in row]
    total = sum(weights)
    return [w / total for w in weights]


def _logits(rows: int, classes: int, confident: bool, rng):
    z = rng.standard_normal((rows, classes)) * 3.0
    k = rng.integers(0, classes, rows)
    if confident:
        z[np.arange(rows), k] = z.max(axis=1) + rng.uniform(5.0, 40.0, rows)
    return z, k


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("confident", [True, False], ids=["confident", "ordinary"])
def test_nll_of_log_softmax(confident, dtype):
    rng = np.random.default_rng(25)
    z, k = _logits(200, 10, confident, rng)
    z = z.astype(dtype)
    x = mt.Tensor(z, dtype=dtype, requires_grad=True)
    F.nll_loss(
        F.log_softmax(x, dim=1), mt.Tensor(k, dtype="int64"), reduction="sum"
    ).backward()
    got = x.grad.numpy().astype(np.float64)
    mt.clear_autograd_graph()
    bound = 1 if dtype == "float32" else 3
    with decimal.localcontext(prec=60, Emax=10**6, Emin=-(10**6)):
        for r, row in enumerate(z.astype(np.float64).tolist()):
            p = _softmax(row)
            for c in range(len(row)):
                want = _rounded(p[c] - (1 if c == k[r] else 0), dtype)
                unit = float(np.spacing(np.array(abs(want), dtype=dtype)))
                assert abs(got[r, c] - want) <= bound * unit, (r, c, got[r, c], want)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_log_softmax_down_a_leading_axis(dtype):
    # A slice whose elements are strided, the axis before the last.
    rng = np.random.default_rng(26)
    z = (rng.standard_normal((6, 5, 3)) * 8.0).astype(dtype)
    g = rng.standard_normal((6, 5, 3)).astype(dtype)
    x = mt.Tensor(z, dtype=dtype, requires_grad=True)
    F.log_softmax(x, dim=1).backward(mt.Tensor(g, dtype=dtype))
    got = x.grad.numpy().astype(np.float64)
    mt.clear_autograd_graph()
    with decimal.localcontext(prec=60, Emax=10**6, Emin=-(10**6)):
        for a in range(6):
            for b in range(3):
                p = _softmax(z[a, :, b].astype(np.float64).tolist())
                gs = [Decimal(float(v)) for v in g[a, :, b].astype(np.float64)]
                total = sum(gs)
                for c in range(5):
                    exact = gs[c] - p[c] * total
                    scale = max(abs(float(gs[c])), abs(float(p[c] * total)))
                    unit = float(np.spacing(np.array(scale, dtype=dtype)))
                    assert abs(got[a, c, b] - float(exact)) <= 2 * unit, (a, c, b)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("confident", [True, False], ids=["confident", "ordinary"])
def test_log_after_softmax(confident, dtype):
    # `-log(softmax(z)[t])` composed: the gradient reaching softmax is
    # `-1 / y_t` on the target, and its slope there `-(1 - y_t)`.
    rng = np.random.default_rng(27)
    z, k = _logits(100, 8, confident, rng)
    z = z.astype(dtype)
    onehot = np.zeros(z.shape, dtype=dtype)
    onehot[np.arange(len(k)), k] = 1
    x = mt.Tensor(z, dtype=dtype, requires_grad=True)
    y = F.softmax(x, dim=1)
    (-(mt.log(y) * mt.Tensor(onehot, dtype=dtype)).sum()).backward()
    got = x.grad.numpy().astype(np.float64)
    mt.clear_autograd_graph()
    bound = 2 if dtype == "float32" else 4
    ys = y.numpy().astype(np.float64)
    with decimal.localcontext(prec=60, Emax=10**6, Emin=-(10**6)):
        for r in range(len(k)):
            # The slope of the rounded probabilities the forward handed on:
            # `y_c` off the target, and minus the rest of the row on it.
            row = [Decimal(float(v)) for v in ys[r]]
            for c in range(len(row)):
                if c == k[r]:
                    want = -sum(row[j] for j in range(len(row)) if j != c)
                else:
                    want = row[c]
                expected = _rounded(want, dtype)
                unit = float(np.spacing(np.array(abs(expected), dtype=dtype)))
                assert abs(got[r, c] - expected) <= bound * unit, (
                    r,
                    c,
                    got[r, c],
                    expected,
                )


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("dim", [0, 1])
def test_log_softmax_gradient_at_the_forwards_limits(dim, dtype):
    # Where a slice's maximum is infinite the forward takes its limit: the
    # `+inf`s share all of the mass and the rest none, and a slice of `-inf`
    # has none anywhere. The gradient is that of those probabilities, `g - p S`,
    # and not the NaN of `inf - inf` under the exponential.
    inf = math.inf
    rows = np.array([[inf, 1.0, inf, -2.0], [-inf, -inf, -inf, -inf]])
    upstream = np.array([[0.5, -1.0, 2.0, 0.25], [1.5, -0.5, 0.0, 3.0]])
    shares = np.array([[0.5, 0.0, 0.5, 0.0], [0.0, 0.0, 0.0, 0.0]])
    if dim == 0:
        rows, upstream, shares = rows.T, upstream.T, shares.T
    x = mt.Tensor(rows, dtype=dtype, requires_grad=True)
    F.log_softmax(x, dim).backward(mt.Tensor(upstream, dtype=dtype))
    want = upstream - shares * upstream.sum(axis=dim, keepdims=True)
    assert x.grad.numpy().astype(np.float64).tolist() == want.tolist()
