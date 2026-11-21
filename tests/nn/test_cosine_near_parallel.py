# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""`cosine_embedding_loss` and the cosine's gradient for slices that lie
almost along each other, against exact values.

A similar pair's loss, `1 - cos`, was one less the rounded cosine, and for
vectors almost along each other -- where similar pairs end up -- it was all
rounding: in float32 zero once the cosine was within `2^-25` of one, 1.7e7
ulps out, and 3e12 in float64. The cosine's gradient `b / (A B) - c a / A^2`
was two terms that all but cancel there, up to 1e9 ulps out. Both are now
formed from the part of one slice at right angles to the other, whole.
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


def _ulp(value: float, dtype: str) -> float:
    return float(np.spacing(np.array(abs(value), dtype=dtype)))


def _exact(a, b):
    """The cosine of two rows, and its gradients in each, exactly, for norms
    above the floor."""

    x = [Decimal(float(v)) for v in a]
    y = [Decimal(float(v)) for v in b]
    dot = sum(p * q for p, q in zip(x, y))
    sa, sb = sum(p * p for p in x), sum(q * q for q in y)
    na, nb = sa.sqrt(), sb.sqrt()
    c = dot / (na * nb)
    da = [q / (na * nb) - c * p / sa for p, q in zip(x, y)]
    db = [p / (na * nb) - c * q / sb for p, q in zip(x, y)]
    return c, da, db


def _close(got, want, dtype):
    """Within an ulp of the exact value in float32, and four in float64."""

    expected = _rounded(want, dtype)
    bound = 1 if dtype == "float32" else 4
    return abs(got - expected) <= bound * _ulp(expected, dtype)


def _pairs(spread, dtype, rng, rows=30, width=16):
    a = rng.standard_normal((rows, width))
    b = a * rng.uniform(0.5, 2.0, (rows, 1)) + spread * rng.standard_normal(
        (rows, width)
    )
    return a.astype(dtype), b.astype(dtype)


@pytest.mark.parametrize(
    "dtype,spread",
    [("float32", s) for s in (1e-1, 1e-3, 1e-5)]
    + [("float64", s) for s in (1e-1, 1e-4, 1e-8, 1e-12)],
)
def test_a_similar_pair(dtype, spread):
    rng = np.random.default_rng(61)
    a, b = _pairs(spread, dtype, rng)
    x1 = mt.Tensor(a, dtype=dtype, requires_grad=True)
    x2 = mt.Tensor(b, dtype=dtype, requires_grad=True)
    target = mt.Tensor(np.ones(len(a)), dtype=dtype)
    loss = F.cosine_embedding_loss(x1, x2, target, 0.0, "none")
    weights = rng.uniform(0.5, 2.0, len(a)).astype(dtype)
    loss.backward(mt.Tensor(weights, dtype=dtype))
    got = loss.numpy().astype(np.float64)
    g1, g2 = (t.grad.numpy().astype(np.float64) for t in (x1, x2))
    with decimal.localcontext(prec=80):
        for r in range(len(a)):
            c, da, db = _exact(a[r], b[r])
            want = 1 - c
            assert _close(got[r], want, dtype), (r, got[r], float(want))
            w = Decimal(float(weights[r]))
            for grad, slope in ((g1[r], da), (g2[r], db)):
                for k, s in enumerate(slope):
                    assert _close(grad[k], -w * s, dtype), (r, k, grad[k])


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("dim", [0, 1])
def test_the_cosine_similarity_gradient(dim, dtype):
    # Pairs almost along each other and pairs at random, down either axis.
    rng = np.random.default_rng(67)
    near = _pairs(1e-4, dtype, rng, rows=12, width=9)
    far = tuple(rng.standard_normal((12, 9)).astype(dtype) for _ in range(2))
    for a, b in (near, far):
        a2, b2 = (a.T.copy(), b.T.copy()) if dim == 0 else (a, b)
        x1 = mt.Tensor(a2, dtype=dtype, requires_grad=True)
        x2 = mt.Tensor(b2, dtype=dtype, requires_grad=True)
        F.cosine_similarity(x1, x2, dim, 1e-8).sum().backward()
        g1, g2 = (t.grad.numpy().astype(np.float64) for t in (x1, x2))
        if dim == 0:
            g1, g2 = g1.T, g2.T
        mt.clear_autograd_graph()
        with decimal.localcontext(prec=80):
            for r in range(12):
                _, da, db = _exact(a[r], b[r])
                for grad, slope in ((g1[r], da), (g2[r], db)):
                    for k, s in enumerate(slope):
                        assert _close(grad[k], s, dtype), (r, k, grad[k])


def test_magnitudes_at_the_ends_of_float64():
    # Sums of squares of 1e200 overflow, and of 1e-200 fall below the normal
    # range; each slice is scaled into range first, by a power of two. The
    # loss's floor, 1e-8, holds up any norm of 1e-200, so the small slices
    # meet the scaling through `cosine_similarity` with a smaller one.
    rng = np.random.default_rng(71)
    a, b = _pairs(1e-6, "float64", rng, rows=4, width=8)
    for scale_a, scale_b in ((1e200, 1e150), (1e-200, 1e-150), (1e200, 1e-200)):
        x, y = a * scale_a, b * scale_b
        x1 = mt.Tensor(x, dtype="float64", requires_grad=True)
        x2 = mt.Tensor(y, dtype="float64", requires_grad=True)
        if scale_a > 1 and scale_b > 1:
            target = mt.Tensor(np.ones(4), dtype="float64")
            out = F.cosine_embedding_loss(x1, x2, target, 0.0, "none")
            sign = -1
        else:
            out = F.cosine_similarity(x1, x2, 1, 1e-300)
            sign = 1
        out.sum().backward()
        value = out.numpy()
        g1, g2 = x1.grad.numpy(), x2.grad.numpy()
        mt.clear_autograd_graph()
        with decimal.localcontext(prec=80, Emin=-(10**6), Emax=10**6):
            for r in range(4):
                c, da, db = _exact(x[r], y[r])
                want = 1 - c if sign < 0 else c
                assert _close(value[r], want, "float64"), (r, value[r])
                for grad, slope in ((g1[r], da), (g2[r], db)):
                    for k, s in enumerate(slope):
                        assert _close(grad[k], sign * s, "float64"), (r, k)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_floors_and_opposites(dtype):
    # A zero row's cosine is zero, over its floored norm, and its loss one;
    # its gradient is the other row over the floored lengths. A row and its
    # opposite are two apart, which cancels nothing.
    a = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]])
    b = np.array([[1.0, 2.0, 2.0], [-2.0, -4.0, -6.0]])
    x1 = mt.Tensor(a, dtype=dtype, requires_grad=True)
    x2 = mt.Tensor(b, dtype=dtype, requires_grad=True)
    target = mt.Tensor([1.0, 1.0], dtype=dtype)
    loss = F.cosine_embedding_loss(x1, x2, target, 0.0, "none")
    loss.sum().backward()
    assert loss.tolist() == [1.0, 2.0]
    floor = float(np.array(1e-8, dtype=dtype))
    want = [-v / (floor * 3.0) for v in (1.0, 2.0, 2.0)]
    got = x1.grad.tolist()[0]
    assert all(abs(g - w) <= 2 * _ulp(w, dtype) for g, w in zip(got, want)), got
    assert x1.grad.tolist()[1] == [0.0, 0.0, 0.0] or max(
        abs(v) for v in x1.grad.tolist()[1]
    ) <= 4 * _ulp(1.0, dtype)
