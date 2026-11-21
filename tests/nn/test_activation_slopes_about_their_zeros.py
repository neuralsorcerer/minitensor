# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""`silu`, `gelu` (both forms) and `mish` slopes about their zeros.

Each slope is the sum of two terms that cancel where it crosses zero, a little
below the origin: `s (1 + x s(-x))` near -1.2785 for `silu`, `Phi(x) + x phi(x)`
near -0.7518 for `gelu`. Formed as float64 there a slope was its terms'
roundings -- at the inputs nearest the zero it had no correct digit, and a few
hundredths away it was still twenty ulps out -- and a float32 one was now and
then an ulp out. About each zero the slope is now its series in `x - x0`.
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
# The tanh form's cubic constant, exact; `sqrt(2/pi)` is formed with `_pi`.
_A = Decimal("0.044715")


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
    """Machin's formula, to the context's precision."""

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


def _erf(x: Decimal) -> Decimal:
    """The power series, carried twenty places past the context for the terms
    it adds and takes away again at a few units of `x`."""

    with decimal.localcontext() as context:
        context.prec += 20
        total, term, n = Decimal(0), x, 0
        while True:
            step = term / (2 * n + 1)
            total += step
            if abs(step) < Decimal(10) ** -context.prec:
                break
            n += 1
            term = -term * x * x / n
        result = 2 * total / _pi().sqrt()
    return +result


def _sigmoid(x: Decimal) -> Decimal:
    return 1 / (1 + (-x).exp())


def _silu(x: Decimal) -> Decimal:
    return _sigmoid(x) * (1 + x * _sigmoid(-x))


def _gelu(x: Decimal) -> Decimal:
    cdf = (1 + _erf(x / Decimal(2).sqrt())) / 2
    pdf = (-x * x / 2).exp() / (2 * _pi()).sqrt()
    return cdf + x * pdf


def _gelu_tanh(x: Decimal) -> Decimal:
    k = (2 / _pi()).sqrt()
    v = k * (x + _A * x**3)
    sech2 = 4 / (v.exp() + (-v).exp()) ** 2
    return 1 / (1 + (-2 * v).exp()) + x / 2 * sech2 * k * (1 + 3 * _A * x * x)


def _mish(x: Decimal) -> Decimal:
    # tanh(softplus(x)) = n / (n + 2) and sech^2 = 4 (n + 1) / (n + 2)^2 for
    # n = e (e + 2), e = e^x: each term without a difference.
    e = x.exp()
    n = e * (e + 2)
    return n / (n + 2) + x * 4 * (n + 1) / (n + 2) ** 2 * e / (e + 1)


_CASES = {
    "silu": (lambda t: F.silu(t), _silu, -1.2784645427610737),
    "gelu": (lambda t: F.gelu(t), _gelu, -0.7517915246935645),
    "gelu_tanh": (
        lambda t: F.gelu(t, approximate="tanh"),
        _gelu_tanh,
        -0.7524614220710163,
    ),
    "mish": (lambda t: F.mish(t), _mish, -1.1924312145154952),
}


def _check(name: str, xs: np.ndarray, dtype: str, bound: float):
    fn, slope, _ = _CASES[name]
    t = mt.Tensor(xs, dtype=dtype, requires_grad=True)
    fn(t).backward(mt.Tensor(np.ones(len(xs)), dtype=dtype))
    got = t.grad.numpy().astype(np.float64).tolist()
    mt.clear_autograd_graph()
    with decimal.localcontext(prec=60):
        for x, g in zip(xs.astype(np.float64).tolist(), got):
            want = _rounded(slope(Decimal(x)), dtype)
            if dtype == "float32":
                assert g == want, (name, x, g, want)
            else:
                spacing = float(np.spacing(np.array(abs(want), dtype=dtype)))
                assert abs(g - want) <= bound * spacing, (name, x, g, want)


def _nearest(root: float, dtype: str, count: int) -> np.ndarray:
    """The `count` floats either side of `root`."""

    centre = np.array(root, dtype=dtype)
    below, above, xs = centre, centre, [centre]
    for _ in range(count):
        below = np.nextafter(below, np.array(-np.inf, dtype=dtype))
        above = np.nextafter(above, np.array(np.inf, dtype=dtype))
        xs += [below, above]
    return np.array(xs, dtype=dtype)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("name", sorted(_CASES))
def test_slopes_at_the_inputs_nearest_the_zero(name, dtype):
    # Where the old float64 slopes had no digit right.
    _check(name, _nearest(_CASES[name][2], dtype, 40), dtype, bound=2)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("name", sorted(_CASES))
def test_slopes_across_the_span_and_past_it(name, dtype):
    rng = np.random.default_rng(2203)
    xs = np.concatenate([np.linspace(-3.0, 0.5, 1501), rng.uniform(-3.0, 0.5, 1500)])
    # Past the zero's span a float64 slope keeps the few ulps its own
    # roundings put in it.
    _check(name, xs.astype(dtype), dtype, bound=8)
