# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""`i0e`'s and `i1e`'s slopes below their crossover, against exact values.

`i0e' = sign(x) e^-t (I_1(t) - I_0(t))` and `i1e' = e^-t (I_0(t) - I_1(t)/t -
I_1(t))` for `t = |x|` subtract terms that agree to `2t` parts in one as `t`
nears thirty, where the asymptotic series takes over: formed from `i0e` and
`i1e` each rounded, a float64 slope was a hundred ulps out there. `i1e'` also
crosses zero near 1.5451, where it had no digit right. Both series are now
pairs, and `i1e'` its Taylor series about the zero.
"""

from __future__ import annotations

import decimal
import math
from decimal import Decimal
from fractions import Fraction

import numpy as np
import pytest

import minitensor as mt

_DIGITS = {"float32": (24, -149, 128), "float64": (53, -1074, 1024)}
# The float64 nearest the zero of `i1e'` above one.
_ZERO = 1.5451272579925426


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


def _bessel(t: Decimal) -> tuple[Decimal, Decimal]:
    """`I_0(t)` and `I_1(t)` by their power series, every term positive."""

    quarter = t * t / 4
    limit = Decimal(10) ** -(decimal.getcontext().prec + 4)
    zero, one = Decimal(0), Decimal(0)
    term, k = Decimal(1), 0
    while True:
        zero += term
        one += term / (k + 1)
        k += 1
        term = term * quarter / (k * k)
        if term < limit * zero:
            return zero, one * t / 2


def _slope(name: str, x: float) -> Decimal:
    with decimal.localcontext(prec=70):
        t = abs(Decimal(x))
        zero, one = _bessel(t)
        scale = (-t).exp()
        if name == "i0e":
            value = scale * (one - zero)
            return value if x > 0 else -value
        return scale * (zero - one / t - one)


def _gradients(name: str, xs, dtype: str) -> list[float]:
    t = mt.Tensor(np.asarray(xs), dtype=dtype, requires_grad=True)
    getattr(mt, name)(t).backward(mt.Tensor(np.ones(len(xs)), dtype=dtype))
    got = t.grad.numpy().astype(np.float64).tolist()
    mt.clear_autograd_graph()
    return got


@pytest.mark.parametrize("name", ["i0e", "i1e"])
def test_float64_slopes_below_the_crossover(name):
    rng = np.random.default_rng(16)
    xs = np.concatenate([rng.uniform(0.05, 30.0, 500), rng.uniform(20.0, 30.0, 200)])
    xs = np.concatenate([xs, -xs])
    got = _gradients(name, xs, "float64")
    for x, g in zip(xs.tolist(), got):
        want = _rounded(_slope(name, x), "float64")
        assert abs(g - want) <= 1.5 * math.ulp(want), (name, x, g, want)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_i1e_slope_at_the_inputs_nearest_its_zero(sign, dtype):
    centre = np.array(sign * _ZERO, dtype=dtype)
    below, above, xs = centre, centre, [centre]
    for _ in range(40):
        below = np.nextafter(below, np.array(-np.inf, dtype=dtype))
        above = np.nextafter(above, np.array(np.inf, dtype=dtype))
        xs += [below, above]
    xs = np.array(xs, dtype=dtype)
    got = _gradients("i1e", xs, dtype)
    for x, g in zip(xs.astype(np.float64).tolist(), got):
        want = _rounded(_slope("i1e", x), dtype)
        if dtype == "float32":
            assert g == want, (x, g, want)
        else:
            assert abs(g - want) <= 1.5 * math.ulp(want), (x, g, want)


@pytest.mark.parametrize("name", ["i0e", "i1e"])
def test_float32_slopes_are_the_nearest_float(name):
    rng = np.random.default_rng(17)
    xs = rng.uniform(-40.0, 40.0, 1000).astype(np.float32)
    got = _gradients(name, xs, "float32")
    for x, g in zip(xs.astype(np.float64).tolist(), got):
        if x == 0:
            continue
        assert g == _rounded(_slope(name, x), "float32"), (name, x, g)


def test_slopes_at_the_origin():
    got_zero = _gradients("i0e", [0.0], "float64")
    got_one = _gradients("i1e", [0.0], "float64")
    assert got_zero == [0.0] and got_one == [0.5]
