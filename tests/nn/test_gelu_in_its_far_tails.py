# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""float64 `gelu` and its slope in their far negative tails.

Below `x = -37.6` the normal density `phi(x)`, and `Phi(x)` below it, fall
below float64's normal range, where a float holds only its last few bits,
while the slope `Phi(x) + x phi(x)` -- and for a while `gelu` itself -- does
not: `x` made those bits some thirty ulps of either. The tanh form does the
same below `x = -21`, where `e^(2u)` leaves the range and `x u'`, some two
thousand there, multiplies it: its slope was out by over a thousand ulps.
Each is now formed with the small factor's exponent kept apart and rounded
once.
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

_CONTEXT = dict(prec=50, Emax=10**6, Emin=-(10**6))


def _rounded(value: Decimal) -> float:
    """The float64 nearest a decimal value, ties to even, rounded once."""

    if value == 0:
        return 0.0
    exact = Fraction(abs(value))
    e = exact.numerator.bit_length() - exact.denominator.bit_length()
    if Fraction(2) ** e > exact:
        e -= 1
    unit = Fraction(2) ** max(e - 52, -1074)
    out = float(round(exact / unit) * unit)
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


def _erfc(z: Decimal) -> Decimal:
    """`erfc(z)` for `z` well above zero, from its continued fraction
    `e^-z^2 / sqrt(pi) / (z + (1/2) / (z + 1 / (z + (3/2) / (z + ...))))`,
    which converges in a few dozen terms this far out."""

    tail = Decimal(0)
    for n in range(80, 0, -1):
        tail = Decimal(n) / 2 / (z + tail)
    return (-z * z).exp() / _pi().sqrt() / (z + tail)


def _erf_form(x: Decimal):
    cdf = _erfc(-x / Decimal(2).sqrt()) / 2
    pdf = (-x * x / 2).exp() / (2 * _pi()).sqrt()
    return x * cdf, cdf + x * pdf


def _tanh_form(x: Decimal):
    k, a = (2 / _pi()).sqrt(), Decimal("0.044715")
    u = k * (x + a * x**3)
    s = 1 / (1 + (-2 * u).exp())
    sech2 = 4 / (u.exp() + (-u).exp()) ** 2
    return x * s, s + x / 2 * sech2 * k * (1 + 3 * a * x * x)


@pytest.mark.parametrize(
    "approximate,exact,low",
    [("none", _erf_form, -40.0), ("tanh", _tanh_form, -26.0)],
    ids=["erf", "tanh"],
)
def test_gelu_and_its_slope_in_the_far_tail(approximate, exact, low):
    xs = np.linspace(low, -15.0, 1201)
    t = mt.Tensor(xs, dtype="float64", requires_grad=True)
    y = F.gelu(t, approximate=approximate)
    values = y.numpy().tolist()
    y.backward(mt.Tensor(np.ones(len(xs)), dtype="float64"))
    slopes = t.grad.numpy().tolist()
    mt.clear_autograd_graph()
    with decimal.localcontext(**_CONTEXT):
        for x, value, slope in zip(xs.tolist(), values, slopes):
            want_value, want_slope = (_rounded(w) for w in exact(Decimal(x)))
            for got, want in ((value, want_value), (slope, want_slope)):
                spacing = math.ulp(want) if want != 0 else 5e-324
                assert abs(got - want) <= 4 * spacing, (approximate, x, got, want)
