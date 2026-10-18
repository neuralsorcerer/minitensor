# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""Above thirty, the Bessel functions keep the digits their series has.

The asymptotic series there is one plus terms that, for orders zero and one,
all have one sign, and from the third on are each less than an ulp of the sum.
Added to it one at a time they were each rounded away, the same way every time,
so `i0e` came out up to five ulps low and `i1e` three. And `i0` and `i1`, formed
as `exp(x)` times the scaled forms, overflowed with `exp(x)` at 709.78 rather
than with themselves at 713.99: `i0(710)` was infinite, not 3.3e306.

The references are the defining power series, summed to fifty digits.
"""

from __future__ import annotations

import math
from decimal import Decimal, localcontext

import numpy as np
import pytest

import minitensor as mt

F = mt.functional
PI = Decimal("3.14159265358979323846264338327950288419716939937510582097494")


def _bessel(order: int, x: float, scaled: bool) -> float:
    with localcontext() as context:
        context.prec = 50
        t = Decimal(abs(x))
        square = t * t / 4
        term = (t / 2) ** order
        total = term
        k = 0
        while True:
            k += 1
            term = term * square / (k * (k + order))
            total += term
            if term < total * Decimal(10) ** -45:
                break
        if scaled:
            total *= (-t).exp()
        return float(total) if order == 0 or x >= 0 else -float(total)


def _ulps(got: float, want: float) -> float:
    return abs(got - want) / float(np.spacing(abs(want)))


def _arguments():
    rng = np.random.default_rng(127)
    magnitudes = np.concatenate(
        [rng.uniform(30, 60, 40), np.exp(rng.uniform(np.log(60), np.log(700), 40))]
    )
    return magnitudes * rng.choice([1, -1], magnitudes.size)


@pytest.mark.parametrize("name,order", [("i0e", 0), ("i1e", 1)])
def test_the_scaled_forms_are_within_half_an_ulp(name, order):
    x = _arguments()
    got = getattr(F, name)(mt.Tensor(x, dtype="float64")).numpy()
    for value, have in zip(x, got):
        assert _ulps(float(have), _bessel(order, float(value), True)) <= 0.55, value


@pytest.mark.parametrize("name,order", [("i0", 0), ("i1", 1)])
def test_the_plain_forms_are_within_the_exponential_s_rounding(name, order):
    # The scaled form's half an ulp, the exponential's, and the product's.
    x = _arguments()
    got = getattr(F, name)(mt.Tensor(x, dtype="float64")).numpy()
    for value, have in zip(x, got):
        assert _ulps(float(have), _bessel(order, float(value), False)) <= 2, value


@pytest.mark.parametrize("name,order", [("i0", 0), ("i1", 1)])
def test_the_plain_forms_overflow_where_they_do_not_where_the_exponential_does(
    name, order
):
    x = [709.5, 709.8, 710.0, 711.25, 712.5, 713.5, 713.9, -710.0]
    got = getattr(F, name)(mt.Tensor(x, dtype="float64")).tolist()
    for value, have in zip(x, got):
        assert math.isfinite(have), value
        assert _ulps(have, _bessel(order, value, False)) <= 3, value
    past = getattr(F, name)(mt.Tensor([714.0, -714.0], dtype="float64")).tolist()
    assert past == [math.inf, math.inf if order == 0 else -math.inf]


def test_their_gradients_are_finite_there_too():
    x = mt.Tensor([710.0, 713.0], dtype="float64", requires_grad=True)
    F.i0(x).sum().backward()
    np.testing.assert_array_equal(x.grad.numpy(), F.i1(x.detach()).numpy())
    y = mt.Tensor([710.0, 713.0], dtype="float64", requires_grad=True)
    F.i1(y).sum().backward()
    for value, slope in zip([710.0, 713.0], y.grad.tolist()):
        want = _bessel(0, value, False) - _bessel(1, value, False) / value
        assert math.isfinite(slope) and _ulps(slope, want) <= 4, value


@pytest.mark.parametrize("name", ["i0e", "i1e"])
def test_far_out_the_scaled_forms_are_the_leading_term(name):
    # Past 1e17 every term after the first is below the last bit.
    for x in (1e17, 1e200, 1e300, 1e308):
        got = getattr(F, name)(mt.Tensor([x], dtype="float64")).item()
        with localcontext() as context:
            context.prec = 50
            want = float(1 / (2 * PI * Decimal(x)).sqrt())
        assert _ulps(got, want) <= 1, x
    limits = getattr(F, name)(mt.Tensor([math.inf, -math.inf], dtype="float64"))
    assert [abs(v) for v in limits.tolist()] == [0.0, 0.0]
