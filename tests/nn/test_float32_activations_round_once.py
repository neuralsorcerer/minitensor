# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""Float32 activations whose formulas are short round once, in float64.

`logit`, `softsign` and `hardswish` evaluated their formulas in float32,
where every operation rounded: the quotient and the logarithm in `logit` left
it 2.7 ulps out, the sum and quotient in `softsign` 1.3, and the three steps of
`hardswish` 1.8. Each is now taken in float64 -- where `hardswish`'s product
is exact, and `logit`'s direct form needs no help near a half -- and rounded
to float32 once. `selu` and `elu` multiplied float32 constants in float32 and
now carry the scales in float64, leaving only the exponential's own rounding.
"""

from __future__ import annotations

import math
from decimal import Decimal, localcontext
from fractions import Fraction

import numpy as np
import pytest

import minitensor as mt

F = mt.functional


def _nearest_float32(exact):
    near = np.float32(float(exact))
    candidates = [
        near,
        np.nextafter(near, np.float32(np.inf)),
        np.nextafter(near, np.float32(-np.inf)),
    ]
    return min(candidates, key=lambda c: abs(Fraction(float(c)) - exact))


def _ulps32(got, want):
    nearest = np.float32(want)
    return abs(float(got) - want) / float(np.spacing(np.abs(nearest)))


def _samples(low, high, count, seed):
    return np.random.default_rng(seed).uniform(low, high, count).astype(np.float32)


def test_softsign_is_the_float32_nearest_it():
    x = np.concatenate([_samples(-20, 20, 1500, 139), [1e30, -3e-30, 0.0]])
    x = x.astype(np.float32)
    got = F.softsign(mt.Tensor(x)).numpy()
    for value, have in zip(x, got):
        exact = Fraction(float(value)) / (1 + abs(Fraction(float(value))))
        assert have == _nearest_float32(exact), value


def test_hardswish_is_the_float32_nearest_it():
    x = _samples(-4, 4, 3000, 149)
    got = F.hardswish(mt.Tensor(x)).numpy()
    for value, have in zip(x, got):
        v = Fraction(float(value))
        exact = 0 if v <= -3 else v if v >= 3 else v * (v + 3) / 6
        assert have == _nearest_float32(exact), value


def test_logit_is_within_half_an_ulp():
    rng = np.random.default_rng(151)
    x = np.concatenate(
        [
            np.exp(rng.uniform(math.log(1e-40), math.log(0.99), 800)),
            rng.uniform(0.25, 0.75, 400),
            1 - np.exp(rng.uniform(math.log(1e-7), math.log(0.5), 300)),
            # The float32s next to a half, where the ratio is next to one.
            0.5 + np.arange(-40, 41) * 2.0**-25,
        ]
    ).astype(np.float32)
    x = x[(x > 0) & (x < 1) & (x != 0.5)]
    got = mt.logit(mt.Tensor(x)).numpy()
    with localcontext() as context:
        context.prec = 40
        for value, have in zip(x, got):
            d = Decimal(float(value))
            assert _ulps32(have, float((d / (1 - d)).ln())) <= 0.5001, value
    assert mt.logit(mt.Tensor([0.5, 0.0, 1.0])).tolist() == [0.0, -math.inf, math.inf]


@pytest.mark.parametrize(
    "name,scale,alpha",
    [
        (
            "selu",
            Decimal("1.0507009873554804934193349852946"),
            Decimal("1.6732632423543772848170429916717"),
        ),
        ("elu", Decimal(1), Decimal("1.3")),
    ],
)
def test_the_exponential_units_round_their_scales_once(name, scale, alpha):
    x = np.concatenate([_samples(0, 30, 1000, 157), _samples(-12, 0, 1000, 163)])
    if name == "selu":
        got = F.selu(mt.Tensor(x)).numpy()
    else:
        got = F.elu(mt.Tensor(x), alpha=1.3).numpy()
    with localcontext() as context:
        context.prec = 40
        for value, have in zip(x, got):
            d = Decimal(float(value))
            if value > 0:
                # A rounding of the exact product, and nothing else.
                assert have == _nearest_float32(Fraction(scale * d)), value
            else:
                exact = float(scale * alpha * (d.exp() - 1))
                # The exponential's own rounding is left.
                assert _ulps32(have, exact) <= 1.5, value
