# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""The rational activations are the floats nearest their exact values.

`hardsigmoid` was `x / 6 + 1/2`, whose two terms are near `-1/2` and `1/2`
where `x` is near -3: there the answer is small and the sum cancelled every
digit it had, a float64 answer some 10^6 ulps out and a float32 one a quarter
of itself. It is now `(x + 3) / 6` rounded once. `hardswish`'s
`x (x + 3) / 6` and `softsign`'s `x / (1 + |x|)` rounded at each step in
float64; they are rounded once too.

Float32 `selu` and `elu` scaled an `expm1` already rounded to float32, and
float32 `sinc` took its sine and quotient in float32: each was an ulp out for
an eighth to a third of the inputs. Each is now taken in float64 and narrowed
once.
"""

from __future__ import annotations

import math
from fractions import Fraction

import numpy as np
import pytest

import minitensor as mt
from minitensor import functional as F


def _float32(value: Fraction) -> float:
    """The float32 nearest `value`, ties to even."""

    if value == 0:
        return 0.0
    sign = -1.0 if value < 0 else 1.0
    magnitude = abs(value)
    exponent = magnitude.numerator.bit_length() - magnitude.denominator.bit_length()
    if Fraction(2) ** exponent > magnitude:
        exponent -= 1
    unit = Fraction(2) ** max(exponent - 23, -149)
    rounded = round(magnitude / unit) * unit
    return sign * (math.inf if rounded >= Fraction(2) ** 128 else float(rounded))


def _rounded(value: Fraction, dtype: str) -> float:
    return _float32(value) if dtype == "float32" else float(value)


def _inputs(dtype: str) -> np.ndarray:
    """Values across the bends, with many just inside -3 and 3 and near 0."""

    rng = np.random.default_rng(1103)
    spread = rng.uniform(-4.0, 4.0, 300)
    eps = np.finfo(dtype).eps
    near = np.concatenate(
        [
            -3.0 + eps * rng.integers(1, 1 << 20, 60) * 4,
            3.0 - eps * rng.integers(1, 1 << 20, 30) * 4,
            -3.0 + 10.0 ** rng.uniform(-6, -1, 30),
            rng.standard_normal(30) * 10.0 ** rng.uniform(-30, -1, 30),
        ]
    )
    values = np.concatenate([spread, near]).astype(dtype)
    return values[np.isfinite(values)]


def _hardsigmoid(x: Fraction) -> Fraction:
    return min(max((x + 3) / 6, Fraction(0)), Fraction(1))


def _hardswish(x: Fraction) -> Fraction:
    if x <= -3:
        return Fraction(0)
    if x >= 3:
        return x
    return x * (x + 3) / 6


def _softsign(x: Fraction) -> Fraction:
    return x / (1 + abs(x))


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize(
    "name, exact",
    [
        ("hardsigmoid", _hardsigmoid),
        ("hardswish", _hardswish),
        ("softsign", _softsign),
    ],
)
def test_every_value_is_rounded_once(dtype, name, exact):
    x = _inputs(dtype)
    got = getattr(F, name)(mt.Tensor(x, dtype=dtype)).tolist()
    want = [_rounded(exact(Fraction(float(v))), dtype) for v in x]
    assert got == want


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_hardsigmoid_just_inside_minus_three(dtype):
    # One ulp of 3 inside the bend: `(x + 3) / 6` is that ulp over six, which
    # the old sum lost to cancellation entirely or in part.
    ulp = float(np.spacing(np.array(3.0, dtype=dtype)))
    x = -3.0 + ulp
    got = F.hardsigmoid(mt.Tensor([x], dtype=dtype)).item()
    assert got == _rounded(Fraction(x) / 6 + Fraction(1, 2), dtype)
    assert got != 0.0


def test_the_ends_and_specials():
    values = [math.inf, -math.inf, -0.0, 0.0, math.nan, 5e-324, -5e-324]
    for dtype in ("float32", "float64"):
        t = mt.Tensor(values, dtype=dtype)
        soft = F.softsign(t).tolist()
        assert soft[:2] == [1.0, -1.0]
        assert soft[2] == 0.0 and math.copysign(1.0, soft[2]) < 0
        assert math.isnan(soft[4])
        hard = F.hardswish(t).tolist()
        assert hard[:2] == [math.inf, 0.0]
        assert hard[2] == 0.0 and math.copysign(1.0, hard[2]) < 0
        sigmoid = F.hardsigmoid(t).tolist()
        assert sigmoid[:4] == [1.0, 0.0, 0.5, 0.5]
    # The least subnormal times about a half rounds up to itself.
    assert F.hardswish(mt.Tensor([5e-324], dtype="float64")).item() == 5e-324


_SELU_ALPHA = 1.6732632423543772
_SELU_SCALE = 1.0507009873554805


@pytest.mark.parametrize(
    "name, positive, negative, call",
    [
        ("selu", _SELU_SCALE, _SELU_SCALE * _SELU_ALPHA, lambda t: F.selu(t)),
        ("elu", 1.0, 0.37, lambda t: F.elu(t, 0.37)),
    ],
)
def test_float32_selu_and_elu_scale_before_they_round(name, positive, negative, call):
    rng = np.random.default_rng(1109)
    x = (rng.standard_normal(2000) * 4).astype(np.float32)
    got = call(mt.Tensor(x, dtype="float32")).tolist()
    want = [
        float(
            np.float32(
                positive * float(v) if v > 0 else negative * math.expm1(float(v))
            )
        )
        for v in x
    ]
    assert got == want, name


def test_float32_sinc_is_the_float64_one_narrowed():
    rng = np.random.default_rng(1117)
    x = (rng.standard_normal(2000) * 10.0 ** rng.uniform(-3, 2, 2000)).astype(
        np.float32
    )
    narrow = mt.sinc(mt.Tensor(x, dtype="float32")).tolist()
    wide = mt.sinc(mt.Tensor(x.astype(np.float64), dtype="float64")).numpy()
    assert narrow == wide.astype(np.float32).tolist()
    assert mt.sinc(mt.Tensor([0.0, 3.0, -2.0], dtype="float32")).tolist() == [
        1.0,
        0.0,
        0.0,
    ]
