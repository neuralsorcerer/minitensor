# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""binary_cross_entropy's gradients against exact values.

A term `-(t log p + (1 - t) log(1 - p))`, each logarithm floored at -100, has
the gradient `(p - t) g / (p (1 - p) n)` in its prediction -- `-t g / (p n)`
or `(1 - t) g / ((1 - p) n)` where a logarithm is floored -- and
`log((1 - p) / p) g / n`, the floors kept, in its target: `g` the incoming
gradient and `n` the count for a mean. Each was formed in float64 a few
roundings at a time, up to 2.1 ulps out for float64, and is now the float
nearest its exact value.
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

_FORMATS = {"float32": (24, -149, 128), "float64": (53, -1074, 1024)}
REDUCTIONS = ("mean", "sum", "none")
# Where the forward floors `log p`: below `e^-100`.
_FLOOR = Fraction(Decimal(-100).exp())


def _nearest(value, dtype: str) -> float:
    """The float of `dtype` nearest `value`, ties to even."""

    value = Fraction(value)
    if value == 0:
        return 0.0
    digits, least, top = _FORMATS[dtype]
    sign = -1.0 if value < 0 else 1.0
    magnitude = abs(value)
    exponent = magnitude.numerator.bit_length() - magnitude.denominator.bit_length()
    if Fraction(2) ** exponent > magnitude:
        exponent -= 1
    unit = Fraction(2) ** max(exponent - digits + 1, least)
    rounded = round(magnitude / unit) * unit
    return sign * (math.inf if rounded >= Fraction(2) ** top else float(rounded))


def _prediction_slope(p: Fraction, t: Fraction) -> Fraction:
    if p == 1:
        return -t / p
    if p < _FLOOR:
        return (1 - t) / (1 - p)
    return (p - t) / (p * (1 - p))


def _target_slope(p: Fraction) -> Decimal:
    with decimal.localcontext(prec=130):
        if p == 1:
            return Decimal(-100)
        q = 1 - Decimal(p.numerator) / Decimal(p.denominator)
        if p == 0:
            return Decimal(100)
        log_q = q.ln()
        if p < _FLOOR:
            return log_q + 100
        return log_q - (Decimal(p.numerator) / Decimal(p.denominator)).ln()


def _check(p, t, dtype, reduction, g=None):
    pi = mt.Tensor(p, dtype=dtype, requires_grad=True)
    ti = mt.Tensor(t, dtype=dtype, requires_grad=True)
    out = F.binary_cross_entropy(pi, ti, reduction=reduction)
    if g is None and reduction == "none":
        g = np.linspace(-1.5, 2.0, p.size).astype(dtype)
    if g is None:
        out.backward()
        flows = np.ones(p.size)
    else:
        out.backward(mt.Tensor(np.asarray(g, dtype=dtype), dtype=dtype))
        flows = np.broadcast_to(g, p.shape)
    by_p, by_t = pi.grad.numpy(), ti.grad.numpy()
    n = p.size if reduction == "mean" else 1
    for k in range(p.size):
        P, T, G = (
            Fraction(float(p[k])),
            Fraction(float(t[k])),
            Fraction(float(flows[k])),
        )
        want = _prediction_slope(P, T) * G / n
        assert by_p[k] == _nearest(want, dtype), ("prediction", k, p[k], t[k])
        with decimal.localcontext(prec=130):
            slope = _target_slope(P) * Decimal(float(flows[k])) / n
        assert by_t[k] == _nearest(slope, dtype), ("target", k, p[k], t[k])


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("reduction", REDUCTIONS)
def test_gradients_are_the_nearest_floats(dtype, reduction):
    rng = np.random.default_rng(241)
    p = rng.uniform(0.001, 0.999, 157).astype(dtype)
    t = rng.uniform(0.0, 1.0, 157).astype(dtype)
    _check(p, t, dtype, reduction)
    # Near a half, where the target's slope is near zero, and small.
    half = (0.5 + rng.uniform(-1e-6, 1e-6, 157)).astype(dtype)
    _check(half, t, dtype, reduction)
    small = (rng.uniform(0.5, 1.0, 157) * 10.0 ** rng.uniform(-30, -3, 157)).astype(
        dtype
    )
    _check(small, t, dtype, reduction)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_floored_logarithms_and_the_ends(dtype):
    # A prediction below `e^-100`, whose logarithm the forward floors, and
    # predictions of exactly zero, a half and one.
    tiny = 1e-45 if dtype == "float32" else 1e-60
    p = np.array([tiny, tiny, 0.0, 0.0, 1.0, 1.0, 0.5, 0.5, 0.25], dtype=dtype)
    t = np.array([0.0, 0.7, 0.0, 0.4, 1.0, 0.3, 0.5, 0.9, 0.25], dtype=dtype)
    for reduction in REDUCTIONS:
        _check(p, t, dtype, reduction)


def test_the_floor_is_placed_exactly():
    # The float64s just below `e^-100` have a logarithm that rounds to -100,
    # but lie below the floor: their slope in the prediction is the floored
    # term's, `(1 - t) / (1 - p)`, not `(p - t) / (p (1 - p))`, some 1e43.
    edge = 3.720075976020836e-44
    below = np.nextafter(edge, 0.0)
    assert math.log(below) == -100.0 and Fraction(below) < _FLOOR <= Fraction(edge)
    p = np.array([below, np.nextafter(below, 0.0), edge, np.nextafter(edge, 1.0)])
    t = np.array([0.25, 0.5, 0.25, 0.5])
    _check(p, t, "float64", "sum")
