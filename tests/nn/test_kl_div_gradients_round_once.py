# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""kl_div's gradients against exact values.

A term `t log(t / p)` of `kl_div` has the gradient `-t g / (p n)` in its
prediction and `(1 + log(t / p)) g / n` in its target, `g` the incoming
gradient and `n` the count the reduction divides by. The first was `-(t / p)`
times a rounded `1 / n` times `g`, up to 1.6 ulps out, and NaN for a zero
target beside a zero prediction, a term that is zero. The second was
`log t - log p + 1`, which cancelled: hundreds of ulps out for ordinary
values, and without bound near `p = e t`, where it is zero. Each is now the
float nearest its exact value.
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
REDUCTIONS = ("mean", "batchmean", "sum", "none")


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


def _gradients(p, t, dtype, reduction, g=None):
    pi = mt.Tensor(p, dtype=dtype, requires_grad=True)
    ti = mt.Tensor(t, dtype=dtype, requires_grad=True)
    out = F.kl_div(pi, ti, reduction=reduction)
    if g is None:
        out.backward()
    else:
        out.backward(mt.Tensor(np.asarray(g, dtype=dtype), dtype=dtype))
    return pi.grad.numpy(), ti.grad.numpy()


def _count(shape, reduction):
    size = int(np.prod(shape))
    return {"mean": size, "batchmean": shape[0], "sum": 1, "none": 1}[reduction]


def _target_slope(t: float, p: float) -> Decimal:
    """`1 + log(t / p)` to some hundred digits."""

    with decimal.localcontext(prec=130):
        return 1 + (Decimal(t) / Decimal(p)).ln()


def _check(p, t, dtype, reduction, g=None):
    if g is None and reduction == "none":
        g = np.linspace(-1.5, 2.0, p.size).astype(dtype).reshape(p.shape)
    by_p, by_t = _gradients(p, t, dtype, reduction, g)
    n = _count(p.shape, reduction)
    flows = np.ones(p.shape) if g is None else np.broadcast_to(g, p.shape)
    for k in np.ndindex(p.shape):
        P, T, G = float(p[k]), float(t[k]), float(flows[k])
        want = -Fraction(T) * Fraction(G) / (Fraction(P) * n)
        assert by_p[k] == _nearest(want, dtype), ("prediction", k, P, T)
        with decimal.localcontext(prec=130):
            slope = _target_slope(T, P) * Decimal(G) / n
        assert by_t[k] == _nearest(slope, dtype), ("target", k, P, T)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("reduction", REDUCTIONS)
def test_gradients_are_the_nearest_floats(dtype, reduction):
    rng = np.random.default_rng(229)
    p = rng.uniform(0.01, 1.0, (6, 23)).astype(dtype)
    t = rng.uniform(0.01, 1.0, (6, 23)).astype(dtype)
    _check(p, t, dtype, reduction)
    # Small probabilities near each other, whose logarithms are large.
    small = (rng.uniform(0.5, 1.0, (6, 23)) * 1e-30).astype(dtype)
    near = (small * rng.uniform(0.9, 1.1, (6, 23))).astype(dtype)
    _check(small, near, dtype, reduction)


def _convergents(bits):
    """Pairs of whole numbers below `2^bits`, `p / t` as near `e` as such
    pairs come."""

    e = Fraction(Decimal(1).exp())
    found = []
    for limit in (2 ** (bits - 6), 2 ** (bits - 4), 2 ** (bits - 2)):
        near = e.limit_denominator(limit)
        found.append((near.numerator, near.denominator))
    return found


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_the_target_slope_near_its_zero(dtype):
    # Near `p = e t` the slope is near zero: rounded predictions `e t`, and
    # the ratios of whole numbers nearest `e`, whose slope is within 2^-100
    # of zero for float64.
    rng = np.random.default_rng(233)
    t = rng.uniform(0.1, 0.36, 40)
    pairs = _convergents(_FORMATS[dtype][0])
    p = np.concatenate([t * np.e, [a * 2.0**-60 for a, _ in pairs]])
    t = np.concatenate([t, [b * 2.0**-60 for _, b in pairs]])
    p, t = p.astype(dtype), t.astype(dtype)
    for reduction in ("sum", "none"):
        _check(p, t, dtype, reduction)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_zero_targets(dtype):
    # A zero target's term is zero whatever the prediction: its gradient in
    # the prediction is a zero, signed as `-t g`, and in the target the
    # slope's limit, an infinity of the sign of `-g`.
    p = np.array([0.0, 0.5, 0.0, 0.5], dtype=dtype)
    t = np.array([0.0, 0.0, 0.0, 0.0], dtype=dtype)
    g = np.array([2.0, 2.0, -3.0, -3.0], dtype=dtype)
    by_p, by_t = _gradients(p, t, dtype, "none", g)
    assert np.all(by_p == 0)
    assert np.signbit(by_p).tolist() == [True, True, False, False]
    assert by_t[1] == -math.inf and by_t[3] == math.inf
    # Both zero leaves the target's slope `log 0 - log 0`, undefined.
    assert np.isnan(by_t[0]) and np.isnan(by_t[2])


def test_an_incoming_gradient_of_another_dtype():
    rng = np.random.default_rng(239)
    p = rng.uniform(0.1, 1.0, 12)
    t = rng.uniform(0.1, 1.0, 12)
    g = rng.standard_normal(12)
    pi = mt.Tensor(p, dtype="float32", requires_grad=True)
    ti = mt.Tensor(t, dtype="float32", requires_grad=True)
    F.kl_div(pi, ti, reduction="none").backward(mt.Tensor(g, dtype="float64"))
    p32, t32 = p.astype(np.float32).astype(np.float64), t.astype(np.float32)
    want_p = -t32.astype(np.float64) / p32 * g
    want_t = (1 + np.log(t32.astype(np.float64) / p32)) * g
    np.testing.assert_allclose(pi.grad.numpy(), want_p, rtol=2e-7)
    np.testing.assert_allclose(ti.grad.numpy(), want_t, rtol=2e-7, atol=1e-6)
