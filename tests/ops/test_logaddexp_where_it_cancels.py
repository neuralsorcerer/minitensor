# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""`logaddexp` and `logaddexp2` keep their digits where the sum cancels.

Each is the larger operand plus `log_b(1 + b^(low - high))`. Where `b^a + b^b`
is near one, the two terms are near each other's negatives and the sum is a
number many times smaller than either, which kept the error of the larger --
an ulp of it: `logaddexp(-0.7, -0.69)` was some 270 ulps out, the sum of equal
operands near `-log 2` thousands, and near the zero of the result millions.
Those are now formed with `exp` and `log1p` carried as pairs of floats and the
sum rounded once; the exponent everywhere is the exact difference. A float32
result is the float32 nearest its value, and a float64 one within an ulp or
two.
"""

from __future__ import annotations

import math
from decimal import Decimal, localcontext
from fractions import Fraction

import numpy as np
import pytest

import minitensor as mt


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


def _exact(a: float, b: float, base2: bool) -> Fraction:
    with localcontext() as context:
        context.prec = 90
        x, y = Decimal(a), Decimal(b)
        if base2:
            ln2 = Decimal(2).ln()
            value = ((x * ln2).exp() + (y * ln2).exp()).ln() / ln2
        else:
            value = (x.exp() + y.exp()).ln()
        return Fraction(value)


def _cases(rng, n):
    """Pairs everywhere, and pairs where the sum cancels."""

    a = -rng.uniform(0.0, 0.69, n)
    near_zero = (a, np.log(-np.expm1(a)) + rng.uniform(-1e-4, 1e-4, n))
    equal = rng.uniform(-0.8, -0.6, n)
    small_high = (rng.uniform(-1e-3, 1e-3, n), -rng.uniform(5.0, 40.0, n))
    anywhere = (rng.standard_normal(n) * 10, rng.standard_normal(n) * 10)
    return {
        "anywhere": anywhere,
        "near zero": near_zero,
        "equal": (equal, equal.copy()),
        "small high": small_high,
    }


@pytest.mark.parametrize("base2", [False, True])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_every_pair_rounds_as_its_exact_value(dtype, base2):
    rng = np.random.default_rng(1301 + base2)
    fn = mt.logaddexp2 if base2 else mt.logaddexp
    for name, (a, b) in _cases(rng, 300).items():
        if base2 and name == "near zero":
            # The base-2 curve where 2^a + 2^b is one.
            b = np.log2(-np.expm1(a * math.log(2))) + rng.uniform(-1e-4, 1e-4, len(a))
        a, b = a.astype(dtype), b.astype(dtype)
        got = fn(mt.Tensor(a, dtype=dtype), mt.Tensor(b, dtype=dtype)).tolist()
        for x, y, g in zip(a, b, got):
            exact = _exact(float(x), float(y), base2)
            if dtype == "float32":
                assert g == _float32(exact), (name, x, y, g)
            else:
                want = float(exact)
                ulps = 0 if g == want else abs(g - want) / math.ulp(want)
                assert ulps <= 2, (name, x, y, g, want, ulps)


def _close(got: float, want: float, operand: float) -> bool:
    """Within an ulp, or within 2^-100 of the operands: where the sum is
    below 2^-50 of its terms, the pairs' own precision is what is left."""

    return got == want or abs(got - want) <= max(
        math.ulp(want), abs(operand) * 2.0**-100
    )


@pytest.mark.parametrize("base2", [False, True])
def test_a_float64_sum_near_zero_keeps_its_digits(base2):
    # log(p) beside log(1 - p): a total of one, less what rounding p and
    # 1 - p left of it -- a sum some 2^55 times smaller than its terms.
    rng = np.random.default_rng(1307 + base2)
    p = rng.uniform(0.05, 0.95, 200)
    if base2:
        a, b = np.log2(p), np.log2(-np.expm1(np.log(p)))
        got = mt.logaddexp2(mt.Tensor(a), mt.Tensor(b)).tolist()
    else:
        a, b = np.log(p), np.log1p(-p)
        got = mt.logaddexp(mt.Tensor(a), mt.Tensor(b)).tolist()
    for x, y, g in zip(a, b, got):
        want = float(_exact(float(x), float(y), base2))
        assert _close(g, want, float(x)), (x, y, g, want)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_the_ends(dtype):
    inf, nan = math.inf, math.nan
    a = mt.Tensor([inf, inf, -inf, -inf, 1.0, nan, -inf], dtype=dtype)
    b = mt.Tensor([inf, -inf, -inf, 2.0, nan, 1.0, 0.0], dtype=dtype)
    for fn in (mt.logaddexp, mt.logaddexp2):
        got = fn(a, b).tolist()
        assert got[:4] == [inf, inf, -inf, 2.0]
        assert math.isnan(got[4]) and math.isnan(got[5])
        assert got[6] == 0.0
    # Equal operands are the operand plus log 2, or plus one in base 2.
    x = mt.Tensor([-0.6931471805599453, 0.5], dtype=dtype)
    assert mt.logaddexp2(x, x).tolist() == (x + 1.0).tolist()


def test_logcumsumexp_adds_the_same_way():
    # The second step's sum is near zero.
    p = 0.3
    x = mt.Tensor([math.log(p), math.log1p(-p)], dtype="float64")
    got = mt.logcumsumexp(x, 0).tolist()
    want = float(_exact(math.log(p), math.log1p(-p), False))
    assert got[0] == math.log(p)
    assert _close(got[1], want, math.log(p))
