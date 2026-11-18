# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""Activation and elementwise slopes are taken of the input, without
cancellation.

Several gradients were written in the rounded output: `tanh`'s as `1 - y^2`,
`sigmoid`'s as `y (1 - y)`, `elu`'s and `selu`'s as `y + alpha`, `expm1`'s as
`y + 1`. Each is a difference of two numbers near one -- or near `-alpha` --
once the output saturates, so the slope lost its digits and then was zero
where it is still a float: `tanh` from `|x| = 19` in float64 and 9 in float32,
`elu` and `expm1` from `x = -38`. `asin`, `acos` and `atanh` formed `1 - x^2`,
millions of ulps out a millionth below one, and `acosh`, `asinh` and `atan`
squared `x` past the range and gave zero for slopes of `1e-200`.

Each is now formed from the input in float64 without either: a float32 slope
is the float32 nearest the exact one, the incoming gradient's product rounded
once, and a float64 one within a couple of ulps.
"""

from __future__ import annotations

import math
from fractions import Fraction

import mpmath
import numpy as np
import pytest

import minitensor as mt
from minitensor import functional as F

mpmath.mp.prec = 300
M = mpmath


def _rounded(value, dtype: str) -> float:
    """The float nearest an mpmath value, ties to even, rounded once."""

    value = M.mpf(value)
    if value == 0:
        return 0.0
    sign, man, exponent, _ = value._mpf_
    exact = Fraction(int(man)) * Fraction(2) ** int(exponent)
    digits, least, top = (24, -149, 128) if dtype == "float32" else (53, -1074, 1024)
    e = exact.numerator.bit_length() - exact.denominator.bit_length()
    if Fraction(2) ** e > exact:
        e -= 1
    unit = Fraction(2) ** max(e - (digits - 1), least)
    near = round(exact / unit) * unit
    out = math.inf if near >= Fraction(2) ** top else float(near)
    return -out if sign else out


def _sigmoid(x):
    return 1 / (1 + M.exp(-x))


_SELU_SCALE = M.mpf("1.0507009873554804934193349852946")
_SELU_ALPHA = M.mpf("1.6732632423543772848170429916717")

_SLOPES = {
    "tanh": (lambda t: t.tanh(), lambda x: 1 / M.cosh(x) ** 2),
    "sigmoid": (lambda t: t.sigmoid(), lambda x: _sigmoid(x) * _sigmoid(-x)),
    "elu": (lambda t: F.elu(t), lambda x: 1 if x > 0 else M.exp(x)),
    "selu": (
        lambda t: F.selu(t),
        lambda x: _SELU_SCALE if x > 0 else _SELU_SCALE * _SELU_ALPHA * M.exp(x),
    ),
    "expm1": (lambda t: t.expm1(), lambda x: M.exp(x)),
    "softplus": (lambda t: F.softplus(t), lambda x: _sigmoid(x)),
    "logsigmoid": (lambda t: F.logsigmoid(t), lambda x: _sigmoid(-x)),
    "softsign": (lambda t: F.softsign(t), lambda x: 1 / (1 + abs(x)) ** 2),
}


def _slopes(fn, x, dtype, upstream=None):
    t = mt.Tensor(np.asarray(x, dtype=dtype), dtype=dtype, requires_grad=True)
    g = np.ones(len(x)) if upstream is None else upstream
    fn(t).backward(mt.Tensor(np.asarray(g, dtype=dtype), dtype=dtype))
    got = t.grad.numpy().astype(np.float64).tolist()
    mt.clear_autograd_graph()
    return got


def _ulps(got: float, want: float) -> float:
    return 0.0 if got == want else abs(got - want) / math.ulp(want)


@pytest.mark.parametrize("name", sorted(_SLOPES))
def test_slopes_far_out_are_not_zero(name):
    fn, slope = _SLOPES[name]
    x = [-700.0, -100.0, -40.0, -20.0, -10.0, 10.0, 20.0, 40.0, 100.0]
    got = _slopes(fn, x, "float64")
    for v, g in zip(x, got):
        want = _rounded(slope(M.mpf(v)), "float64")
        assert _ulps(g, want) <= 2, (name, v, g, want)
        if want != 0:
            assert g != 0, (name, v)


@pytest.mark.parametrize("name", sorted(_SLOPES))
def test_float32_slopes_are_the_nearest_float32(name):
    fn, slope = _SLOPES[name]
    rng = np.random.default_rng(1801)
    x = np.concatenate(
        [rng.standard_normal(200) * 4, [-120.0, -60.0, -30.0, -9.5, 9.5, 30.0, 60.0]]
    ).astype(np.float32)
    upstream = rng.standard_normal(len(x)).astype(np.float32)
    got = _slopes(fn, x, "float32", upstream)
    for v, u, g in zip(x.tolist(), upstream.tolist(), got):
        want = _rounded(M.mpf(u) * slope(M.mpf(v)), "float32")
        assert g == want, (name, v, u, g, want)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_inverse_functions_near_one_and_far_out(dtype):
    near = [1 - 2.0**-k for k in (3, 10, 20, 30, 40, 50)]
    cases = {
        "atanh": (
            lambda t: t.atanh(),
            lambda x: 1 / (1 - x * x),
            near + [-v for v in near],
        ),
        "asin": (lambda t: t.asin(), lambda x: 1 / M.sqrt(1 - x * x), near),
        "acos": (lambda t: t.acos(), lambda x: -1 / M.sqrt(1 - x * x), near),
        "acosh": (
            lambda t: t.acosh(),
            lambda x: 1 / M.sqrt(x * x - 1),
            [1 + 2.0**-k for k in (3, 10, 20, 30, 40)] + [1e10, 1e30, 1e200],
        ),
        "asinh": (
            lambda t: t.asinh(),
            lambda x: 1 / M.sqrt(x * x + 1),
            [-1e200, -3.0, 1e-8, 1e10, 1e30, 1e300],
        ),
        "atan": (
            lambda t: t.atan(),
            lambda x: 1 / (1 + x * x),
            [-1e160, -1e30, 1e-8, 3.0, 1e19, 1e200],
        ),
    }
    for name, (fn, slope, xs) in cases.items():
        x = np.array(xs, dtype=np.float64)
        if dtype == "float32":
            x = x[np.abs(x) < 3e38]
            x = x[np.abs(x.astype(np.float32)) != 1.0]
        got = _slopes(fn, x.tolist(), dtype)
        for v, g in zip(np.asarray(x, dtype=dtype).tolist(), got):
            want = _rounded(slope(M.mpf(v)), dtype)
            if dtype == "float32":
                assert g == want, (name, v, g, want)
            else:
                assert _ulps(g, want) <= 2, (name, v, g, want)


def test_a_softplus_slope_keeps_the_rounding_of_beta_x():
    # `beta * x` rounded is `|beta * x|` ulps of the far tail of its logistic;
    # taken back through the logistic's own slope, the slope is within a
    # couple of ulps of `sigmoid(beta * x)` for the exact product.
    x = np.array([-600.0, -300.0, -77.0, -20.0])
    t = mt.Tensor(x, dtype="float64", requires_grad=True)
    F.softplus(t, beta=0.7).backward(mt.Tensor(np.ones(4), dtype="float64"))
    for v, g in zip(x.tolist(), t.grad.tolist()):
        want = _rounded(_sigmoid(M.mpf(0.7) * M.mpf(v)), "float64")
        assert _ulps(g, want) <= 2, (v, g, want)
    mt.clear_autograd_graph()
