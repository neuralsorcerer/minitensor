# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""binary_cross_entropy_with_logits' gradients against exact values.

A term `(1 - t) x + (1 + (w - 1) t) softplus(-x)`, `w` the positive class's
weight, has the gradient `((1 - t) s(x) - w t s(-x)) g / n` in its logit and
`(-x + (w - 1) softplus(-x)) g / n` in its target: `g` the incoming gradient
and `n` the count for a mean. The first was formed in float64 and rounded
twice more, up to 3 ulps out in float64 and hundreds where its terms all but
cancel; the second in the logits' dtype, up to 37 ulps out with a weight and
nothing of it left about its zero. Each is now the float nearest its exact
value.
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


def _log1p(u: Decimal) -> Decimal:
    """`log(1 + u)`, its series below 1e-40, where `1 + u` would lose `u`."""

    if u < Decimal("1e-40"):
        return u - u * u / 2 + u * u * u / 3
    return (1 + u).ln()


def _slopes(x: float, t: float, w: float) -> tuple[Fraction, Fraction]:
    """Both slopes of a term, to some 10^-290 of themselves, or exactly where
    they are rational."""

    with decimal.localcontext(prec=300):
        X, T, W = Decimal(x), Decimal(t), Decimal(w)
        u = (-abs(X)).exp()
        if x >= 0:
            logit = ((1 - T) - W * T * u) / (1 + u)
        else:
            logit = ((1 - T) * u - W * T) / (1 + u)
        lead = 1 if x >= 0 else W
        target = -lead * X + (W - 1) * _log1p(u)
    logit, target = Fraction(logit), Fraction(target)
    a, b = 1 - Fraction(t), Fraction(w) * Fraction(t)
    if x == 0:
        logit = (a - b) / 2
    if a + b == 0:
        logit = a
    if w == 1:
        target = -Fraction(x)
    return logit, target


def _target_zero(weight: float) -> float:
    """Where `-z + (w - 1) softplus(-z)` is zero, by Newton's iteration: its
    slope `-1 - (w - 1) s(-z)` stays below -1/2 for weights from 1/2 to 3."""

    with decimal.localcontext(prec=60):
        w, z = Decimal(weight), Decimal("0.5")
        for _ in range(100):
            u = (-z).exp()
            value = -z + (w - 1) * (1 + u).ln()
            step = value / (-1 - (w - 1) * u / (1 + u))
            z -= step
            if abs(step) < Decimal("1e-50"):
                break
        return float(z)


def _check(x, t, w, dtype, reduction, g=None, which=(True, True)):
    xi = mt.Tensor(x, dtype=dtype, requires_grad=which[0])
    ti = mt.Tensor(t, dtype=dtype, requires_grad=which[1])
    kwargs = {} if w is None else {"pos_weight": mt.Tensor(w, dtype=dtype)}
    out = F.binary_cross_entropy_with_logits(xi, ti, reduction=reduction, **kwargs)
    if g is None and reduction == "none":
        g = np.linspace(-1.5, 2.0, x.size).astype(dtype)
    if g is None:
        out.backward()
        flows = np.ones(x.size)
    else:
        out.backward(mt.Tensor(np.asarray(g, dtype=dtype), dtype=dtype))
        flows = np.broadcast_to(np.asarray(g, dtype=dtype), x.shape)
    weights = np.ones(x.size) if w is None else np.broadcast_to(w, x.shape)
    n = x.size if reduction == "mean" else 1
    for k in range(x.size):
        logit, target = _slopes(float(x[k]), float(t[k]), float(weights[k]))
        G = Fraction(float(flows[k]))
        if which[0]:
            got = xi.grad.numpy()[k]
            want = _nearest(logit * G / n, dtype)
            assert got == want, ("logit", k, x[k], t[k], weights[k], got, want)
        if which[1]:
            got = ti.grad.numpy()[k]
            want = _nearest(target * G / n, dtype)
            assert got == want, ("target", k, x[k], t[k], weights[k], got, want)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("reduction", REDUCTIONS)
def test_gradients_are_the_nearest_floats(dtype, reduction):
    rng = np.random.default_rng(613)
    x = (rng.standard_normal(97) * 4).astype(dtype)
    t = rng.uniform(0.0, 1.0, 97).astype(dtype)
    w = rng.uniform(0.25, 4.0, 97).astype(dtype)
    _check(x, t, None, dtype, reduction)
    _check(x, t, w, dtype, reduction)
    hard = rng.integers(0, 2, 97).astype(dtype)
    _check(x, hard, w, dtype, reduction)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("spread", [0.0, 1e-9, 1e-6])
def test_near_the_slopes_zeros(dtype, spread):
    # The logit's slope is zero where `x = log(w t / (1 - t))`, and the
    # target's where `x = (w - 1) softplus(-x)`; about each the two terms
    # cancel, as a well-fit model's do, and at the zero rounded to the dtype
    # all but some 2^-53 of them, which a pair of floats does not hold.
    rng = np.random.default_rng(17)
    t = rng.uniform(0.05, 0.95, 41)
    w = rng.uniform(0.5, 3.0, 41)
    fit = np.log(w * t / (1 - t))
    x = (fit * (1 + rng.uniform(-spread, spread, 41))).astype(dtype)
    _check(x, t.astype(dtype), w.astype(dtype), dtype, "sum", which=(True, False))
    roots = np.array([_target_zero(weight) for weight in w])
    x = (roots * (1 + rng.uniform(-spread, spread, 41))).astype(dtype)
    _check(x, t.astype(dtype), w.astype(dtype), dtype, "sum", which=(False, True))


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_hard_labels_far_out(dtype):
    # Confident logits, right and wrong, out to where `e^-|x|` leaves the
    # float64 range; weights of zero and past it.
    sizes = [20.0, 90.0, 104.0, 650.0, 708.0, 745.0, 800.0, 2000.0]
    x = np.array([s * sign for s in sizes for sign in (1.0, -1.0)] * 2, dtype=dtype)
    t = np.array([1.0, 0.0] * len(sizes) + [0.0, 1.0] * len(sizes), dtype=dtype)
    for w in (None, np.array(0.0, dtype=dtype), np.array(3.5, dtype=dtype)):
        _check(x, t, w, dtype, "sum")
        _check(x, t, w, dtype, "none", g=np.full(x.size, 1e30, dtype=dtype))


def test_rational_slopes_are_rounded_once():
    # At a zero logit the logit's slope is `(1 - t - w t) / 2`; where
    # `1 + (w - 1) t` is zero it is `1 - t` at every logit; without a weight the
    # target's is `-x`. Each of these lands on a halfway point here, which
    # goes to the even neighbour.
    cases = [
        # (x, t, w, g, dtype): a zero logit, `1 - 2^-54` and `1 - 2^-25`.
        (0.0, 2.0**-55, 1.0, 2.0, "float64"),
        (0.0, 2.0**-54, 0.0, 2.0, "float64"),
        (0.0, 2.0**-26, 1.0, 2.0, "float32"),
        # `1 + (w - 1) t = 0`: `0.75 (1 + 2^-52)` and `0.75 (1 + 2^-23)`.
        (1.7, 0.25, -3.0, 1.0 + 2.0**-52, "float64"),
        (-30.0, 0.25, -3.0, 1.0 + 2.0**-52, "float64"),
        (700.0, 0.25, -3.0, 1.0 + 2.0**-52, "float64"),
        (2.5, 0.25, -3.0, 1.0 + 2.0**-23, "float32"),
    ]
    for x, t, w, g, dtype in cases:
        logit, _ = _slopes(x, t, w)
        exact = logit * Fraction(g)
        assert Fraction(_nearest(exact, dtype)) != exact
        _check(
            np.array([x], dtype=dtype),
            np.array([t], dtype=dtype),
            np.array([w], dtype=dtype),
            dtype,
            "none",
            g=np.array([g], dtype=dtype),
        )
    # `-x g` halfway: `(1 + 2^-52) (1 + 2^-52) = 1 + 2^-51 + 2^-104`, past
    # the halfway point, and `3 (1 + 2^-52) / 2` exactly on one.
    for x, g in [(-(1.0 + 2.0**-52), 1.0 + 2.0**-52), (-1.5, 1.0 + 2.0**-52)]:
        _check(
            np.array([x]),
            np.array([0.5]),
            None,
            "float64",
            "none",
            g=np.array([g]),
        )


def test_out_of_range_weights_and_gradients():
    # `w t` past float64's range, and an incoming gradient that brings the
    # product back into it; tiny and huge incoming gradients.
    x = np.array([-1.0, 1.0, -800.0, 3.0, 0.0])
    t = np.array([1e200, 1e200, 1e200, 0.5, 0.25])
    w = np.array([1e200, 1e200, 1e200, 1e-300, 1e300])
    g = np.array([1e-300, 1e-300, 1e-300, 1e-310, 1e-300])
    _check(x, t, w, "float64", "none", g=g)
    _check(x, t, w, "float64", "none", g=np.array([1e300] * 5))


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_infinite_logits_take_the_limit(dtype):
    x = np.array([np.inf, -np.inf, np.inf, -np.inf], dtype=dtype)
    t = np.array([0.25, 0.25, 1.0, 0.0], dtype=dtype)
    w = np.array([3.0, 3.0, 0.0, 0.0], dtype=dtype)
    xi = mt.Tensor(x, dtype=dtype, requires_grad=True)
    ti = mt.Tensor(t, dtype=dtype, requires_grad=True)
    out = F.binary_cross_entropy_with_logits(
        xi, ti, pos_weight=mt.Tensor(w, dtype=dtype), reduction="sum"
    )
    out.backward()
    # The logit's slope tends to `1 - t` and `-w t`; the target's to `-x`,
    # `-w x`, and below zero without a weight to `-0`.
    assert xi.grad.numpy().tolist() == [0.75, -0.75, 0.0, 0.0]
    assert ti.grad.numpy().tolist() == [-np.inf, np.inf, -np.inf, 0.0]
    assert math.copysign(1.0, ti.grad.numpy()[3]) == -1.0


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_zero_incoming_gradients_keep_the_slopes_sign(dtype):
    x = np.array([2.0, -2.0, 0.5, -0.5], dtype=dtype)
    t = np.array([0.0, 1.0, 0.0, 1.0], dtype=dtype)
    xi = mt.Tensor(x, dtype=dtype, requires_grad=True)
    ti = mt.Tensor(t, dtype=dtype, requires_grad=True)
    out = F.binary_cross_entropy_with_logits(xi, ti, reduction="none")
    out.backward(mt.Tensor(np.array([0.0, 0.0, -0.0, -0.0], dtype=dtype)))
    signs = [math.copysign(1.0, v) for v in xi.grad.numpy()]
    # Slopes `s(2) > 0`, `-s(2) < 0`, `s(0.5) > 0`, `-s(0.5) < 0`, times the
    # signed zeros.
    assert signs == [1.0, -1.0, -1.0, 1.0]
    signs = [math.copysign(1.0, v) for v in ti.grad.numpy()]
    # Target slopes `-2`, `2`, `-0.5`, `0.5`.
    assert signs == [-1.0, 1.0, 1.0, -1.0]


def test_zero_incoming_gradients_take_the_exact_sign():
    # Where `w t` is past float64's range the slope is finite, `-w t` over
    # one and a bit, not `inf - inf`; and at a soft label's logit rounded the
    # slope's sign is that of what is left of its two terms, past float64's
    # reach. Each times a signed zero is a zero, not NaN, of their sign.
    rng = np.random.default_rng(29)
    t = rng.uniform(0.05, 0.95, 40)
    w = rng.uniform(0.5, 3.0, 40)
    x = np.log(w * t / (1 - t))
    x = np.concatenate([x, [-1.0, 1.0]])
    t = np.concatenate([t, [1e200, 1e200]])
    w = np.concatenate([w, [1e200, 1e200]])
    g = np.where(np.arange(x.size) % 2 == 0, 0.0, -0.0)
    xi = mt.Tensor(x, requires_grad=True)
    ti = mt.Tensor(t, requires_grad=True)
    out = F.binary_cross_entropy_with_logits(
        xi, ti, pos_weight=mt.Tensor(w), reduction="none"
    )
    out.backward(mt.Tensor(g))
    for k in range(x.size):
        logit, target = _slopes(float(x[k]), float(t[k]), float(w[k]))
        flow = math.copysign(1.0, g[k])
        for got, slope in ((xi.grad.numpy()[k], logit), (ti.grad.numpy()[k], target)):
            assert got == 0.0 and not math.isnan(got), (k, got)
            want = flow if slope >= 0 else -flow
            assert math.copysign(1.0, got) == want, (k, x[k], t[k], w[k], float(slope))
