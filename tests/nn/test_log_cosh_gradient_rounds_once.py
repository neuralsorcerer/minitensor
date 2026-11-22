# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""log_cosh_loss's gradient against exact values.

The gradient of `log(cosh(p - t))` in a prediction is `tanh(p - t)`, times
the incoming gradient `g` and over the count `n` for a mean, and its negative
in the target. It was `tanh` of the rounded difference, times a rounded
`1 / n`, times `g`: up to 2.3 ulps out. Each is now the float nearest its
exact value.
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


def _tanh(d: Decimal) -> Decimal:
    """`tanh(d)` to some hundred digits, in the caller's context."""

    if d == 0:
        return d
    size = abs(d)
    with decimal.localcontext() as context:
        context.prec += 40
        if size < Decimal("1e-12"):
            s = size * size
            value = size * (1 - s / 3 + 2 * s * s / 15 - 17 * s**3 / 315)
        else:
            e = (-2 * size).exp()
            value = (1 - e) / (1 + e)
    return +value if d > 0 else -value


def _want(p, t, g, count, dtype):
    """The float of `dtype` nearest `tanh(p - t) g / count`."""

    with decimal.localcontext(prec=120):
        d = Decimal(float(p)) - Decimal(int(t) if isinstance(t, int) else float(t))
        return _nearest(_tanh(d) * Decimal(float(g)) / count, dtype)


def _gradients(p, t, dtype, reduction, g=None, target_dtype=None):
    pi = mt.Tensor(p, dtype=dtype, requires_grad=True)
    floating = target_dtype is None
    ti = mt.Tensor(t, dtype=target_dtype or dtype, requires_grad=floating)
    out = F.log_cosh_loss(pi, ti, reduction=reduction)
    if g is None:
        out.backward()
    else:
        out.backward(mt.Tensor(np.asarray(g, dtype=dtype), dtype=dtype))
    got = pi.grad.numpy()
    assert got.dtype == np.dtype(dtype)
    if floating:
        # The target's gradient is the prediction's, negated exactly.
        theirs = ti.grad.numpy()
        assert np.array_equal(theirs, -got, equal_nan=True)
        assert np.array_equal(np.signbit(theirs), np.signbit(-got))
    return got


def _check(p, t, dtype, reduction, g=None, target_dtype=None):
    if g is None and reduction == "none":
        g = np.ones(len(p), dtype=dtype)
    got = _gradients(p, t, dtype, reduction, g, target_dtype)
    count = len(p) if reduction == "mean" else 1
    flows = np.ones(len(p)) if g is None else np.broadcast_to(g, len(p))
    for k in range(len(p)):
        target = int(t[k]) if target_dtype is not None else t[k]
        want = _want(p[k], target, flows[k], count, dtype)
        assert got[k] == want, (k, p[k], t[k], flows[k], got[k], want)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("reduction", REDUCTIONS)
def test_gradients_are_the_nearest_floats(dtype, reduction):
    rng = np.random.default_rng(193)
    # Predictions and targets of different sizes, so that their difference
    # rounds, and a count that is not a power of two.
    p = (rng.standard_normal(301) * 10.0 ** rng.integers(-4, 3, 301)).astype(dtype)
    t = (rng.standard_normal(301) * 10.0 ** rng.integers(-4, 3, 301)).astype(dtype)
    g = rng.standard_normal(301).astype(dtype)
    _check(p, t, dtype, reduction, None if reduction != "none" else g)
    if reduction != "none":
        # One incoming gradient for the whole loss.
        _check(p, t, dtype, reduction, g[:1].reshape(()))


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_differences_large_and_small(dtype):
    # Where `tanh` is all but the difference, all but one, and between.
    sizes = [2.0**-140, 1e-30, 2.0**-60, 3e-15, 2.0**-48, 1e-9, 0.37, 1.0, 9.5]
    sizes += [19.9, 20.1, 35.9, 36.1, 37.0, 300.0, 400.0, 1e20, 1e38]
    if dtype == "float64":
        sizes += [2.0**-1000, 1e-300, 1e300]
    d = np.array([s * k for s in sizes for k in (1.0, -1.0)], dtype=dtype)
    rng = np.random.default_rng(197)
    t = (rng.standard_normal(len(d)) * 1e-3).astype(dtype)
    t[: len(t) // 2] = 0.0
    p = (d + t).astype(dtype)
    g = rng.standard_normal(len(d)).astype(dtype)
    for reduction in REDUCTIONS:
        _check(p, t, dtype, reduction, g if reduction == "none" else None)


def test_extreme_incoming_gradients():
    # Gradients near float64's ends, whose products leave the range the
    # bound is kept in.
    rng = np.random.default_rng(199)
    p = rng.standard_normal(12)
    t = rng.standard_normal(12)
    g = np.array([1e-310, -5e-324, 1e-300, 3e300, -1.7e308, 2.0**-1022] * 2)
    _check(p, t, "float64", "none", g)


def test_a_difference_past_float64():
    # The difference of two finite floats can be past the largest; its
    # `tanh` is one less a sliver, and its gradient finite.
    p = np.array([1.7e308, -1.7e308, 1e308])
    t = np.array([-1.7e308, 1.7e308, -1e308])
    got = _gradients(p, t, "float64", "mean")
    assert got.tolist() == [_nearest(Fraction(1, 3), "float64")] + [
        -_nearest(Fraction(1, 3), "float64"),
        _nearest(Fraction(1, 3), "float64"),
    ]


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_halfway_products_go_toward_zero(dtype):
    # Where `tanh(d) = d (1 - d^2 / 3 + ...)` and `d g` is exactly halfway
    # between two floats, the exact gradient lies just inside it: the float
    # toward zero is the nearest, odd as it is.
    digits = _FORMATS[dtype][0]
    half = digits // 2
    cases = []
    for i in range(1, 300):
        a = 2**half + 2 * i + 1
        for j in range(1, 60):
            b = 2 ** (digits - half) + 2 * j + 1
            product = a * b
            # Halfway, and the float toward zero is odd.
            if product.bit_length() == digits + 1 and product % 4 == 3:
                cases.append((a, b))
                break
        if len(cases) == 4:
            break
    assert cases
    for scale in (-60, -90, -200):
        if dtype == "float32" and scale < -100:
            continue
        p = np.array([a * 2.0 ** (scale - half) for a, _ in cases], dtype=dtype)
        g = np.array([float(b) for _, b in cases], dtype=dtype)
        for sign in (1.0, -1.0):
            got = _gradients(sign * p, np.zeros(len(p), dtype=dtype), dtype, "none", g)
            for k, (a, b) in enumerate(cases):
                below = float(Fraction(a * b - 1) * Fraction(2) ** (scale - half))
                assert got[k] == sign * below, (scale, k)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_zeros_and_values_that_are_not_finite(dtype):
    inf, nan = math.inf, math.nan
    p = np.array([1.0, 1.0, -0.0, 2.0, 2.0, inf, -inf, nan, 1.0, inf, 0.5], dtype=dtype)
    t = np.array([1.0, 1.0, 0.0, 1.0, 3.0, 0.0, 5.0, 0.0, 1.0, inf, 0.25], dtype=dtype)
    g = np.array(
        [1.0, -1.0, 1.0, -0.0, -0.0, 0.3, 0.3, 1.0, inf, 1.0, -inf], dtype=dtype
    )
    got = _gradients(p, t, dtype, "none", g)
    with np.errstate(invalid="ignore"):
        want = np.tanh(p.astype(np.float64) - t) * g
    for k, (value, expected) in enumerate(zip(got.tolist(), want.tolist())):
        if math.isnan(expected):
            assert math.isnan(value), k
        else:
            # The infinite differences' gradients rounded once.
            assert value == float(np.array(expected, dtype=dtype)), k
            assert math.copysign(1, value) == math.copysign(1, expected), k
    # A difference's infinity over a count that does not divide it.
    got = _gradients(
        np.array([inf, 0.0, 0.0]), np.array([0.0, 0.0, inf]), dtype, "mean"
    )
    third = _nearest(Fraction(1, 3), dtype)
    assert got.tolist() == [third, 0.0, -third]


def test_integer_targets():
    # An int64 target past 2^53 is taken exactly, beside float64 predictions,
    # and an int32 beside float32 ones.
    big = 2**60
    p = np.array([2.0**60, 2.0**60 + 2.0**8, 2.0**60, 3.0, 2.0**60 + 2.0**9])
    t = np.array([big + 3, big + 255, big - 1, big + 1, big + 511], dtype=np.int64)
    for reduction in REDUCTIONS:
        _check(p, t, "float64", reduction, target_dtype="int64")
    rng = np.random.default_rng(211)
    p = (rng.standard_normal(40) * 3).astype("float32")
    t = rng.integers(-3, 4, 40).astype(np.int32)
    for reduction in REDUCTIONS:
        _check(p, t, "float32", reduction, target_dtype="int32")
