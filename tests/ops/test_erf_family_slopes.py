# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""`erf`'s, `erfc`'s and `erfcx`'s slopes against exact values.

`erf`'s slope `2/sqrt(pi) e^(-x^2)`: the exponential turns the rounding of
`x^2` -- 2^-53 of it -- into `x^2` ulps of the slope, which a float64 slope
carried, some 500 ulps at `x = 23.5`, where it is still a normal float. Past
26.6 the slope falls below the normal range and kept only its last few bits,
though times a large incoming gradient it need not. The square's rounding is
now put back, and the slope rounds once with the gradient.

`erfcx`'s slope `2x erfcx(x) - 2/sqrt(pi)` is a difference of terms that
agree to `2x^2` parts in one: up to 250 ulps out below its crossover at 8. It
is now Laplace's continued fraction from one on, where nothing cancels.
"""

from __future__ import annotations

import decimal
import math
from decimal import Decimal
from fractions import Fraction

import numpy as np
import pytest

import minitensor as mt

_CONTEXT = dict(prec=40, Emax=10**6, Emin=-(10**6))
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


def _pi() -> Decimal:
    def arctan_inverse(n: int) -> Decimal:
        x = Decimal(1) / n
        total, term, k = x, x, 1
        while True:
            term = -term * x * x
            step = term / (2 * k + 1)
            if abs(step) < Decimal(10) ** -(decimal.getcontext().prec + 2):
                return total
            total, k = total + step, k + 1

    return 4 * (4 * arctan_inverse(5) - arctan_inverse(239))


def _slope(x: float, g: float) -> Decimal:
    d = Decimal(x)
    return Decimal(g) * 2 / _pi().sqrt() * (-d * d).exp()


def _gradients(name: str, xs, gs, dtype: str) -> list[float]:
    t = mt.Tensor(np.asarray(xs), dtype=dtype, requires_grad=True)
    getattr(t, name)().backward(mt.Tensor(np.asarray(gs), dtype=dtype))
    got = t.grad.numpy().astype(np.float64).tolist()
    mt.clear_autograd_graph()
    return got


@pytest.mark.parametrize("name,sign", [("erf", 1), ("erfc", -1)])
def test_float64_slope_keeps_the_square_it_is_the_exponential_of(name, sign):
    rng = np.random.default_rng(11)
    xs = np.concatenate([rng.uniform(-26.5, 26.5, 1500), rng.uniform(15.0, 26.5, 500)])
    got = _gradients(name, xs, np.ones(len(xs)), "float64")
    with decimal.localcontext(**_CONTEXT):
        for x, g in zip(xs.tolist(), got):
            want = _rounded(sign * _slope(x, 1.0), "float64")
            assert abs(g - want) <= 2 * math.ulp(want), (name, x, g, want)


def test_float64_slope_times_a_large_gradient_below_the_normal_range():
    # `e^(-x^2)` is subnormal or below the smallest float from 26.6 on; times
    # 1e200 it is an ordinary number, to the last bit.
    rng = np.random.default_rng(12)
    xs = rng.uniform(26.7, 34.0, 400)
    xs = np.concatenate([xs, -xs])
    gs = np.full(len(xs), 1e200)
    got = _gradients("erf", xs, gs, "float64")
    with decimal.localcontext(**_CONTEXT):
        for x, g in zip(xs.tolist(), got):
            want = _rounded(_slope(x, 1e200), "float64")
            assert abs(g - want) <= 2 * math.ulp(want), (x, g, want)


def test_float32_slope_is_the_nearest_float():
    rng = np.random.default_rng(13)
    xs = rng.uniform(-10.0, 10.0, 2000).astype(np.float32)
    got = _gradients("erf", xs, np.ones(len(xs)), "float32")
    with decimal.localcontext(**_CONTEXT):
        for x, g in zip(xs.astype(np.float64).tolist(), got):
            assert g == _rounded(_slope(x, 1.0), "float32"), (x, g)


def test_slope_at_the_ends_and_at_nan():
    got = _gradients(
        "erf", [0.0, 40.0, -math.inf, math.inf, math.nan], [1.0] * 5, "float64"
    )
    assert got[0] == pytest.approx(2 / math.sqrt(math.pi), rel=1e-16)
    assert got[1:4] == [0.0, 0.0, 0.0] and math.isnan(got[4])


def _erfcx_slope(x: float) -> Decimal:
    """`2x e^(x^2) erfc(x) - 2/sqrt(pi)`, with `erf` by its power series and
    digits enough for both the series' and the slope's cancellation."""

    with decimal.localcontext() as context:
        context.prec = 40 + int(x * x / 2.3) + 10
        d = Decimal(x)
        total, term, n = Decimal(0), d, 0
        while True:
            step = term / (2 * n + 1)
            total += step
            if abs(step) < Decimal(10) ** -(context.prec + 2):
                break
            n += 1
            term = -term * d * d / n
        root = _pi().sqrt()
        erfc = 1 - 2 * total / root
        slope = 2 * d * (d * d).exp() * erfc - 2 / root
    return +slope


@pytest.mark.parametrize(
    "low,high,bound", [(1.0, 8.0, 3), (-5.0, 1.0, 6)], ids=["fraction", "below_one"]
)
def test_float64_erfcx_slope(low, high, bound):
    rng = np.random.default_rng(14)
    xs = rng.uniform(low, high, 600)
    got = _gradients("erfcx", xs, np.ones(len(xs)), "float64")
    with decimal.localcontext(**_CONTEXT):
        for x, g in zip(xs.tolist(), got):
            want = _rounded(_erfcx_slope(x), "float64")
            assert abs(g - want) <= bound * math.ulp(want), (x, g, want)


def test_float32_erfcx_slope_is_the_nearest_float():
    rng = np.random.default_rng(15)
    xs = rng.uniform(-4.0, 8.0, 800).astype(np.float32)
    got = _gradients("erfcx", xs, np.ones(len(xs)), "float32")
    with decimal.localcontext(**_CONTEXT):
        for x, g in zip(xs.astype(np.float64).tolist(), got):
            assert g == _rounded(_erfcx_slope(x), "float32"), (x, g)
