# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""`huber_loss` and `smooth_l1_loss` are the floats nearest the exact losses.

A Huber term is `d^2 / 2` within `delta` of the target and
`delta * (|d| - delta / 2)` past it, for the exact difference `d`; a
smooth-L1 term with `beta` is that, with `delta = beta`, over `beta`. Each was
taken a rounding at a time -- the difference, its magnitude, the square or
the product, and for smooth-L1 the division by `beta` -- and a fifth of the
Huber terms of random operands, and nearly half of the smooth-L1 ones, were
an ulp or more out, in float32 as in float64, as were some sums and means. Each term is now the float nearest its exact
value, and so are the sum and the mean.
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


def _float64(value: Fraction) -> float:
    """The float64 nearest `value`: an infinity past the largest."""

    try:
        return float(value)
    except OverflowError:
        return math.inf if value > 0 else -math.inf


def _rounded(value: Fraction, dtype: str) -> float:
    return _float32(value) if dtype == "float32" else _float64(value)


def _width(value: float, dtype: str) -> Fraction:
    """A setting as the operands' width reads it."""

    return Fraction(float(np.float32(value)) if dtype == "float32" else value)


def _term(d: Fraction, delta: Fraction, divisor: Fraction) -> Fraction:
    if abs(d) <= delta:
        return d * d / 2 / divisor
    return delta * (abs(d) - delta / 2) / divisor


def _exact_terms(p, t, delta, divisor):
    return [
        _term(
            Fraction(float(a)) - Fraction(int(b) if isinstance(b, int) else float(b)),
            delta,
            divisor,
        )
        for a, b in zip(p, t)
    ]


def _losses(p, t, dtype, setting, smooth):
    """Each reduction of the loss, from the minitensor functions."""

    pt, tt = mt.Tensor(p, dtype=dtype), mt.Tensor(t, dtype=dtype)
    if smooth:
        return {
            r: F.smooth_l1_loss(pt, tt, r, setting) for r in ("none", "sum", "mean")
        }
    return {r: F.huber_loss(pt, tt, r, setting) for r in ("none", "sum", "mean")}


def _operands(rng, n, kind):
    p = rng.standard_normal(n) * 10.0 ** rng.uniform(-1, 1, n)
    t = rng.standard_normal(n) * 10.0 ** rng.uniform(-1, 1, n)
    if kind == "close":
        t = p + rng.standard_normal(n) * 1e-3
    elif kind == "spread":
        p *= 2.0 ** rng.integers(-30, 30, n)
        t *= 2.0 ** rng.integers(-30, 30, n)
    elif kind == "whole":
        p, t = np.round(p * 4) / 4, np.round(t * 4) / 4
    return p, t


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("kind", ["plain", "close", "spread", "whole"])
@pytest.mark.parametrize("smooth", [False, True])
@pytest.mark.parametrize("setting", [1.0, 0.37, 3.0])
def test_terms_sums_and_means(dtype, kind, smooth, setting):
    rng = np.random.default_rng(len(kind) * 97 + int(setting * 13) + smooth)
    p, t = _operands(rng, 200, kind)
    p = p.astype(dtype)
    t = t.astype(dtype)
    got = _losses(p, t, dtype, setting, smooth)
    delta = _width(setting, dtype)
    divisor = delta if smooth else Fraction(1)
    exact = _exact_terms(p, t, delta, divisor)
    assert got["none"].tolist() == [_rounded(term, dtype) for term in exact]
    assert got["sum"].item() == _rounded(sum(exact), dtype)
    assert got["mean"].item() == _rounded(sum(exact) / len(exact), dtype)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_long_operands(dtype):
    # Long enough to be added a block at a time across the pool.
    rng = np.random.default_rng(1031)
    p, t = _operands(rng, 50_000, "plain")
    p, t = p.astype(dtype), t.astype(dtype)
    for smooth, setting in ((False, 0.8), (True, 1.7)):
        got = _losses(p, t, dtype, setting, smooth)
        delta = _width(setting, dtype)
        divisor = delta if smooth else Fraction(1)
        total = sum(_exact_terms(p, t, delta, divisor))
        assert got["sum"].item() == _rounded(total, dtype)
        assert got["mean"].item() == _rounded(total / len(p), dtype)


def test_halfway_points_and_zeros():
    # Whole numbers under `delta = 1`: halves exactly, a term of `2^24 + 1`
    # that float32 rounds to the even neighbour, and zero differences that
    # are positive zeros.
    p = np.array([3.0, -2.0, 0.5, 7.0, -0.0, 16_777_217.5], dtype=np.float64)
    t = np.array([3.0, 0.0, 0.0, -1.0, 0.0, 0.0], dtype=np.float64)
    got = F.huber_loss(mt.Tensor(p), mt.Tensor(t), "none").tolist()
    assert got == [0.0, 1.5, 0.125, 7.5, 0.0, 16_777_217.0]
    assert not any(math.copysign(1.0, v) < 0 for v in got)
    # Float32 operands a difference of `2^24 + 1.5` apart, whose term
    # `2^24 + 1` lies halfway between two float32s.
    narrow = F.huber_loss(
        mt.Tensor([16_777_216.0], dtype="float32"),
        mt.Tensor([-1.5], dtype="float32"),
        "none",
    )
    assert narrow.tolist() == [16_777_216.0]
    zero = F.smooth_l1_loss(mt.Tensor([1.5, -2.0]), mt.Tensor([1.5, -2.0]), "sum", 0.5)
    assert zero.item() == 0.0 and math.copysign(1.0, zero.item()) > 0


def test_values_past_the_cut_and_a_difference_past_the_range():
    # Operands too large or too small for the float64 products to split,
    # which are summed in whole numbers; and a difference past the largest
    # float64 whose term is in range.
    p = np.array([2.0**450, 3.0 * 2.0**-460, 1e300, -1e-300, 5.0])
    t = np.array([2.0**450 - 2.0**400, 2.0**-460, -1e300, 1e-300, 2.0**-430])
    for smooth, setting in ((False, 2.0**-200), (True, 1e-310), (False, 1e250)):
        delta = Fraction(setting)
        divisor = delta if smooth else Fraction(1)
        got = (F.smooth_l1_loss if smooth else F.huber_loss)(
            mt.Tensor(p), mt.Tensor(t), "none", setting
        )
        exact = _exact_terms(p, t, delta, divisor)
        assert got.tolist() == [_float64(term) for term in exact], setting
        total = (F.smooth_l1_loss if smooth else F.huber_loss)(
            mt.Tensor(p), mt.Tensor(t), "sum", setting
        )
        assert total.item() == _float64(sum(exact)), setting


def test_specials():
    p = mt.Tensor([math.nan, math.inf, math.inf, 1.0])
    t = mt.Tensor([0.0, 1.0, math.inf, 2.0])
    got = F.huber_loss(p, t, "none").tolist()
    assert math.isnan(got[0]) and got[1] == math.inf and math.isnan(got[2])
    assert got[3] == 0.5
    assert math.isnan(F.huber_loss(p, t, "sum").item())
    infinite = F.smooth_l1_loss(
        mt.Tensor([math.inf, 1.0]), mt.Tensor([0.0, 0.0]), "mean", 2.0
    )
    assert infinite.item() == math.inf


def test_integer_targets_past_float_precision():
    # An int64 target past 2^53 is the integer it is.
    high = 2**60
    p = mt.Tensor([float(high), float(high), 3.5], dtype="float64")
    t = mt.Tensor(np.array([high - 3, high + 1, 2], dtype=np.int64))
    delta = Fraction(2)
    exact = [
        _term(Fraction(float(high)) - (high - 3), delta, Fraction(1)),
        _term(Fraction(float(high)) - (high + 1), delta, Fraction(1)),
        _term(Fraction(3.5) - 2, delta, Fraction(1)),
    ]
    assert F.huber_loss(p, t, "none", 2.0).tolist() == [float(v) for v in exact]
    assert F.huber_loss(p, t, "mean", 2.0).item() == float(sum(exact) / 3)
    smooth = F.smooth_l1_loss(p, t, "sum", 3.0).item()
    beta = Fraction(3)
    want = sum(
        _term(Fraction(float(high)) - b, beta, beta) for b in (high - 3, high + 1)
    )
    want += _term(Fraction(3.5) - 2, beta, beta)
    assert smooth == float(want)


def test_a_beta_float32_holds_only_as_zero_is_the_l1_loss():
    p = mt.Tensor([1.0, -2.5, 4.0], dtype="float32")
    t = mt.Tensor([0.5, 0.5, 0.5], dtype="float32")
    for reduction in ("none", "mean", "sum"):
        assert (
            F.smooth_l1_loss(p, t, reduction, 1e-60).tolist()
            == F.l1_loss(p, t, reduction).tolist()
        )


@pytest.mark.parametrize("smooth", [False, True])
@pytest.mark.parametrize("reduction", ["none", "mean", "sum"])
def test_gradients(smooth, reduction):
    rng = np.random.default_rng(1033)
    p0, t0 = _operands(rng, 30, "plain")
    setting = 0.7
    x = mt.Tensor(p0, requires_grad=True)
    y = mt.Tensor(t0, requires_grad=True)
    loss = (F.smooth_l1_loss if smooth else F.huber_loss)(x, y, reduction, setting)
    weights = rng.standard_normal(30) if reduction == "none" else None
    if weights is None:
        loss.backward()
    else:
        loss.backward(mt.Tensor(weights))
    d = p0 - t0
    want = np.clip(d, -setting, setting) / (setting if smooth else 1.0)
    if reduction == "mean":
        want = want / len(d)
    if weights is not None:
        want = want * weights
    np.testing.assert_allclose(x.grad.numpy(), want, rtol=1e-14, atol=1e-300)
    np.testing.assert_allclose(y.grad.numpy(), -want, rtol=1e-14, atol=1e-300)
