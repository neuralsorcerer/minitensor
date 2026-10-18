# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""`erfcx` past eight, and the slope of `erfinv`, keep their last digits.

`erfcx` there is `(1 + tail) / (x sqrt(pi))`, which was three plain roundings
on a root that was itself an ulp off -- the square root of the rounded `pi`,
not the rounded root of `pi` -- and came out two ulps out; past about 1e308
`x sqrt(pi)` overflowed and the answer, a subnormal, was zero. Each rounding is
carried now, and `x` scaled by a power of two where it has to be.

The slope of `erfinv` is `sqrt(pi)/2 exp(y^2)` for `y = erfinv(x)`, and the
exponential multiplies the relative error of `y` by `2 y^2`: near the ends,
where `y` is five or six, `erfinv`'s own ulp came out as thirty. From `|x| = 1/2`
the factor is `erfcx(y) / (1 - |x|)` instead, the same value, through which an
error in `y` passes unamplified.
"""

from __future__ import annotations

import math
from decimal import Decimal, localcontext

import numpy as np
import pytest

import minitensor as mt

PI = Decimal("3.14159265358979323846264338327950288419716939937510582097494")


def _ulps(got: float, want: float) -> float:
    return abs(got - want) / float(np.spacing(abs(want)))


def _erfcx_far(x: float) -> float:
    """The asymptotic series to its smallest term, which past eight is below
    1e-27 of the sum."""

    with localcontext() as context:
        context.prec = 50
        d = Decimal(x)
        total = term = Decimal(1)
        for k in range(1, 200):
            following = -term * (2 * k - 1) / (2 * d * d)
            if abs(following) >= abs(term):
                break
            term = following
            total += term
        return float(total / (d * PI.sqrt()))


def _erfc(t: Decimal) -> Decimal:
    if t < 3:
        total = term = t
        n = 0
        while True:
            n += 1
            term *= -t * t / n
            step = term / (2 * n + 1)
            total += step
            if abs(step) < Decimal(10) ** -70:
                return 1 - 2 / PI.sqrt() * total
    fraction = t
    for k in range(400, 0, -1):
        fraction = t + Decimal(k) / 2 / fraction
    return (-t * t).exp() / PI.sqrt() / fraction


def _erfinv_slope(x: float) -> float:
    """`sqrt(pi)/2 exp(y^2)` at the `y` whose `erfc` is `1 - |x|`, found by
    Newton's method to sixty digits."""

    with localcontext() as context:
        context.prec = 60
        target = 1 - abs(Decimal(x))
        y = Decimal(math.sqrt(-math.log(float(target))))
        for _ in range(40):
            step = (_erfc(y) - target) * PI.sqrt() / 2 * (y * y).exp()
            y += step
            if abs(step) < Decimal(10) ** -50:
                break
        return float(PI.sqrt() / 2 * (y * y).exp())


def test_erfcx_past_eight_is_within_half_an_ulp():
    rng = np.random.default_rng(131)
    x = np.concatenate(
        [rng.uniform(8, 30, 60), np.exp(rng.uniform(np.log(30), np.log(1e300), 60))]
    )
    got = mt.erfcx(mt.Tensor(x, dtype="float64")).numpy()
    for value, have in zip(x, got):
        assert _ulps(float(have), _erfcx_far(float(value))) <= 0.51, value


def test_erfcx_is_subnormal_rather_than_zero_at_the_top():
    x = [2.6e307, 1e308, 1.5e308, 1.7976931348623157e308]
    got = mt.erfcx(mt.Tensor(x, dtype="float64")).tolist()
    assert got == [_erfcx_far(v) for v in x]
    assert all(0 < v < 2.2250738585072014e-308 for v in got[1:])


@pytest.mark.parametrize("side", [1, -1])
def test_the_slope_of_erfinv_keeps_its_digits_near_the_ends(side):
    rng = np.random.default_rng(137)
    x = side * np.concatenate(
        [rng.uniform(0.5, 0.99, 30), 1 - np.exp(-rng.uniform(5, 36, 50))]
    )
    tensor = mt.Tensor(x, dtype="float64", requires_grad=True)
    mt.erfinv(tensor).sum().backward()
    for value, have in zip(x, tensor.grad.numpy()):
        # `erfcx` there is its platform's `erfc` times `exp(y^2)`.
        assert _ulps(float(have), _erfinv_slope(float(value))) <= 6, value


def test_the_slope_of_erfinv_at_and_past_the_ends():
    tensor = mt.Tensor([1.0, -1.0, 0.0], dtype="float64", requires_grad=True)
    mt.erfinv(tensor).sum().backward()
    assert tensor.grad.tolist() == [math.inf, math.inf, float(PI.sqrt() / 2)]
    outside = mt.Tensor([1.5], dtype="float64", requires_grad=True)
    mt.erfinv(outside).sum().backward()
    assert math.isnan(outside.grad.item())
