# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""The regression losses' gradients against exact values.

A loss of the differences `d = p - t` has the gradient `s(d) g / n` in each
prediction, `s` the term's slope -- `2 d`, the sign of `d`, Huber's clamp of
`d`, or smooth-L1's `d / beta` clamped to the sign -- `g` the incoming
gradient and `n` the count for a mean; and its negative in each target. It was
formed from the rounded difference, a rounded `1 / n` and two products, up to
1.3 ulps out in float32 and 1.8 in float64. Each is now the float nearest its
exact value.
"""

from __future__ import annotations

import decimal
import math
from decimal import Decimal

import numpy as np
import pytest

import minitensor as mt
from minitensor import functional as F

LOSSES = ("mse", "l1", "huber", "smooth_l1")
REDUCTIONS = ("mean", "sum", "none")


def _ulps(got, want: Decimal, dtype: str) -> float:
    if want == 0:
        return 0.0 if got == 0 else float("inf")
    unit = Decimal(float(np.spacing(np.array(abs(float(want)), dtype=dtype))))
    return float(abs(Decimal(float(got)) - want) / unit)


def _call(kind, p, t, reduction, limit):
    if kind == "mse":
        return F.mse_loss(p, t, reduction=reduction)
    if kind == "l1":
        return F.l1_loss(p, t, reduction=reduction)
    if kind == "huber":
        return F.huber_loss(p, t, reduction=reduction, delta=limit)
    return F.smooth_l1_loss(p, t, reduction=reduction, beta=limit)


def _slope(kind, d: Decimal, limit: Decimal) -> Decimal:
    sign = Decimal((d > 0) - (d < 0))
    if kind == "mse":
        return 2 * d
    if kind == "l1":
        return sign
    if kind == "huber":
        return d if abs(d) <= limit else limit * sign
    return d / limit if abs(d) <= limit else sign


def _check(kind, p, t, reduction, limit, dtype, upstream=None, target_dtype=None):
    pi = mt.Tensor(p, dtype=dtype, requires_grad=True)
    floating = target_dtype is None
    ti = mt.Tensor(t, dtype=target_dtype or dtype, requires_grad=floating)
    out = _call(kind, pi, ti, reduction, limit)
    if upstream is None and reduction != "none":
        out.backward()
        g = np.ones(())
    else:
        g = np.ones(p.shape) if upstream is None else upstream
        g = np.asarray(g, dtype=dtype)
        out.backward(mt.Tensor(g, dtype=dtype))
    got = pi.grad.numpy()
    assert got.dtype == np.dtype(dtype)
    if floating:
        # The target's gradient is the prediction's, negated exactly.
        assert np.array_equal(ti.grad.numpy(), -got)
    # The limit as the loss reads it: at the predictions' width.
    edge = Decimal(float(np.array(limit, dtype=dtype)))
    count = Decimal(p.size if reduction == "mean" else 1)
    with decimal.localcontext(prec=80):
        for k in np.ndindex(p.shape):
            d = Decimal(float(p[k])) - Decimal(
                int(t[k]) if not floating else float(t[k])
            )
            flow = Decimal(float(g[k] if g.ndim else g))
            want = _slope(kind, d, edge) * flow / count
            assert _ulps(got[k], want, dtype) <= 0.5, (kind, reduction, k)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("kind", LOSSES)
@pytest.mark.parametrize("reduction", REDUCTIONS)
def test_gradients_are_the_nearest_floats(dtype, kind, reduction):
    rng = np.random.default_rng(101)
    # Predictions and targets of different sizes, so that their difference
    # rounds, and a count that is not a power of two.
    p = (rng.standard_normal(111) * 10.0 ** rng.integers(-3, 4, 111)).astype(dtype)
    t = (rng.standard_normal(111) * 10.0 ** rng.integers(-3, 4, 111)).astype(dtype)
    shape = () if reduction != "none" else p.shape
    g = rng.standard_normal(shape).astype(dtype)
    limit = 0.7 if kind != "huber" else 1.3
    _check(kind, p, t, reduction, limit, dtype)
    _check(kind, p, t, reduction, limit, dtype, upstream=g)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("kind", ["huber", "smooth_l1"])
def test_differences_about_the_limit(dtype, kind):
    # Differences at the limit, and a hair either side of it.
    limit = 0.75
    t = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 2.0**-60, -(2.0**-60)], dtype=dtype)
    edge = float(np.array(limit, dtype=dtype))
    p = np.array(
        [
            edge,
            -edge,
            np.nextafter(np.array(edge, dtype=dtype), 1),
            np.nextafter(np.array(edge, dtype=dtype), 0),
            np.nextafter(np.array(-edge, dtype=dtype), 0),
            edge,
            edge,
        ],
        dtype=dtype,
    )
    for reduction in REDUCTIONS:
        _check(kind, p, t, reduction, limit, dtype)


def test_integer_targets_past_two_to_the_fifty_three():
    # A target past 2^53 is no float64, and its difference from a prediction
    # is taken exactly.
    big = 2**60
    p = np.array([2.0**60, 2.0**60 + 2.0**8, 2.0**60, 3.0], dtype="float64")
    t = np.array([big + 3, big + 255, big - 1, big + 1], dtype="int64")
    for kind in LOSSES:
        for reduction in REDUCTIONS:
            _check(kind, p, t, reduction, 2.0, "float64", target_dtype="int64")


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_a_constant_slope_beside_an_infinite_difference(dtype):
    # Past the limit the slope is a constant, whatever the difference is.
    p = mt.Tensor(np.array([math.inf, -math.inf, 1.0]), dtype=dtype, requires_grad=True)
    t = mt.Tensor(np.array([0.0, 0.0, math.nan]), dtype=dtype)
    F.huber_loss(p, t, reduction="none", delta=0.3).backward(
        mt.Tensor(np.array([0.1, 0.1, 0.1]), dtype=dtype)
    )
    got = p.grad.numpy()
    with decimal.localcontext(prec=80):
        edge = Decimal(float(np.array(0.3, dtype=dtype)))
        flow = Decimal(float(np.array(0.1, dtype=dtype)))
        assert _ulps(got[0], edge * flow, dtype) <= 0.5
        assert _ulps(got[1], -edge * flow, dtype) <= 0.5
    assert np.isnan(got[2])
    p = mt.Tensor(np.array([math.inf, 2.0]), dtype=dtype, requires_grad=True)
    t = mt.Tensor(np.array([0.0, math.inf]), dtype=dtype)
    F.l1_loss(p, t).backward()
    assert p.grad.tolist() == [0.5, -0.5]
    p = mt.Tensor(np.array([math.inf, 2.0]), dtype=dtype, requires_grad=True)
    F.mse_loss(p, mt.Tensor(np.array([0.0, 1.0]), dtype=dtype)).backward()
    assert p.grad.tolist() == [math.inf, 1.0]


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_zeros_are_signed_as_the_formula_signs_them(dtype):
    # A zero difference, or a zero incoming gradient, makes a zero whose sign
    # is the slope's times the incoming gradient's; the absolute error's slope
    # is a positive zero at zero.
    p = np.array([1.0, 1.0, -0.0, 2.0, 2.0], dtype=dtype)
    t = np.array([1.0, 1.0, 0.0, 1.0, 3.0], dtype=dtype)
    g = np.array([1.0, -1.0, 1.0, -0.0, -0.0], dtype=dtype)
    signs = {
        "mse": [False, True, True, True, False],
        "l1": [False, True, False, True, False],
        "huber": [False, True, True, True, False],
        "smooth_l1": [False, True, True, True, False],
    }
    for kind, negative in signs.items():
        pi = mt.Tensor(p, dtype=dtype, requires_grad=True)
        _call(kind, pi, mt.Tensor(t, dtype=dtype), "none", 1.5).backward(
            mt.Tensor(g, dtype=dtype)
        )
        got = pi.grad.numpy()
        assert np.all(got == 0), kind
        assert np.signbit(got).tolist() == negative, kind


def test_a_float32_limit_past_its_range_is_infinite():
    # Read at float32, a limit of 1e39 is infinite: Huber's slope is the
    # difference everywhere, and smooth-L1's is zero.
    p = mt.Tensor(np.array([3.0, -2.0]), dtype="float32", requires_grad=True)
    t = mt.Tensor(np.array([1.0, 0.5]), dtype="float32")
    F.huber_loss(p, t, reduction="sum", delta=1e39).backward()
    assert p.grad.tolist() == [2.0, -2.5]
    p = mt.Tensor(np.array([3.0, -2.0]), dtype="float32", requires_grad=True)
    F.smooth_l1_loss(p, t, reduction="sum", beta=1e39).backward()
    got = p.grad.numpy()
    assert np.all(got == 0) and np.signbit(got).tolist() == [False, True]
