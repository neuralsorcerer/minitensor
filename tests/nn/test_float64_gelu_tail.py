# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""Float64 GELU keeps its digits in the negative tail.

`x * Phi(x)` was `0.5 * x * erfc(-x / sqrt(2))`, and the rounding of
`-x / sqrt(2)` -- 2^-53 of it -- came out of `erfc` as `x^2` ulps of the
answer: 32 at `x = -6`, 1590 near `-36`. The tanh form lost its exponent's
rounding the same way, and returned zero near `x = -21`, where `e^(-2u)`
overflows but the answer is a normal float. Each argument's rounding is now
recovered and taken back out, and the tanh form's logistic is written so that
it cannot overflow. The references below are computed to sixty digits.
"""

from __future__ import annotations

from decimal import Decimal, localcontext

import numpy as np
import pytest

import minitensor as mt

PI = Decimal("3.14159265358979323846264338327950288419716939937510582097494")
TINY = 2.2250738585072014e-308  # The smallest normal float64.


def _erfc(t: Decimal) -> Decimal:
    if t < 0:
        return 2 - _erfc(-t)
    if t < 3:
        # The series for erf, which loses about four of the sixty digits here.
        total = term = t
        n = 0
        while True:
            n += 1
            term *= -t * t / n
            step = term / (2 * n + 1)
            total += step
            if abs(step) < Decimal(10) ** -70:
                return 1 - 2 / PI.sqrt() * total
    # The continued fraction, evaluated from the bottom.
    fraction = t
    for k in range(400, 0, -1):
        fraction = t + Decimal(k) / 2 / fraction
    return (-t * t).exp() / PI.sqrt() / fraction


def _gelu(x: float) -> float:
    with localcontext() as context:
        context.prec = 60
        d = Decimal(x)
        return float(d * _erfc(-d / Decimal(2).sqrt()) / 2)


def _gelu_slope(x: float) -> float:
    with localcontext() as context:
        context.prec = 60
        d = Decimal(x)
        density = (-d * d / 2).exp() / (2 * PI).sqrt()
        return float(_erfc(-d / Decimal(2).sqrt()) / 2 + d * density)


def _tanh_gelu(x: float) -> float:
    with localcontext() as context:
        context.prec = 60
        d = Decimal(x)
        u = (2 / PI).sqrt() * (d + Decimal("0.044715") * d**3)
        return float(d / (1 + (-2 * u).exp()))


def _ulps(got: float, want: float) -> float:
    return abs(got - want) / float(np.spacing(abs(want)))


def _tail():
    rng = np.random.default_rng(97)
    return np.concatenate([rng.uniform(-37.5, -3, 300), rng.uniform(-3, 6, 100)])


def test_the_exact_form_is_within_a_few_ulps_of_the_true_value():
    x = _tail()
    got = mt.functional.gelu(mt.Tensor(x, dtype="float64")).numpy()
    for value, have in zip(x, got):
        want = _gelu(float(value))
        if abs(want) >= TINY:
            assert _ulps(float(have), want) <= 6, value


def test_the_tanh_form_is_within_a_few_ulps_of_its_own_formula():
    x = np.concatenate([_tail()[:250] / 1.75, [-21.1605, -21.0, -20.5]])
    got = mt.functional.gelu(mt.Tensor(x, dtype="float64"), approximate="tanh")
    for value, have in zip(x, got.numpy()):
        want = _tanh_gelu(float(value))
        if abs(want) >= TINY:
            assert _ulps(float(have), want) <= 6, value


def test_the_tanh_form_does_not_overflow_to_zero():
    # e^(-2u) is past the float range here; the answer is not.
    got = mt.functional.gelu(
        mt.Tensor([-21.1605], dtype="float64"), approximate="tanh"
    ).item()
    want = _tanh_gelu(-21.1605)
    assert TINY < abs(want) and got != 0.0 and _ulps(got, want) <= 6


def test_the_exact_slope_keeps_its_digits_in_the_tail():
    x = np.random.default_rng(101).uniform(-37, -3, 200)
    tensor = mt.Tensor(x, dtype="float64", requires_grad=True)
    mt.functional.gelu(tensor).sum().backward()
    for value, have in zip(x, tensor.grad.numpy()):
        want = _gelu_slope(float(value))
        if abs(want) >= TINY:
            assert _ulps(float(have), want) <= 24, value


@pytest.mark.parametrize("approximate", ["none", "tanh"])
def test_the_limits_are_unchanged(approximate):
    values = mt.Tensor([-np.inf, np.inf, np.nan, 0.0], dtype="float64")
    got = mt.functional.gelu(values, approximate=approximate).tolist()
    assert got[0] == 0.0 and got[1] == np.inf and np.isnan(got[2]) and got[3] == 0.0
