# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""`mse_loss` and `l1_loss` are the floats nearest the exact losses.

Each difference was rounded, squared and rounded again, and the squares or
magnitudes summed and divided, a rounding at every step: about one loss in
three of random operands, and one square in three, was an ulp or more from
the exact value, in float32 as in float64. The mean and the sum are now the
exact `Σ (p - t)²` or `Σ |p - t|`, over the count for the mean, rounded once,
and each element of the unreduced squared error the exact square rounded
once.
"""

from __future__ import annotations

import math
from fractions import Fraction

import numpy as np
import pytest

import minitensor as mt
from minitensor import nn


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
    if dtype == "float32":
        return _float32(value)
    try:
        return float(value)
    except OverflowError:
        return math.inf if value > 0 else -math.inf


def _terms(p, t, squared: bool) -> list[Fraction]:
    differences = [Fraction(float(a)) - Fraction(float(b)) for a, b in zip(p, t)]
    return [d * d if squared else abs(d) for d in differences]


def _want(p, t, squared: bool, reduction: str, dtype: str):
    terms = _terms(np.ravel(p), np.ravel(t), squared)
    if reduction == "none":
        return [_rounded(term, dtype) for term in terms]
    if reduction == "sum":
        return _rounded(sum(terms, Fraction(0)), dtype)
    return _rounded(sum(terms, Fraction(0)) / len(terms), dtype)


def _got(p, t, squared: bool, reduction: str, dtype: str):
    loss = nn.mse_loss if squared else nn.l1_loss
    result = loss(mt.Tensor(p, dtype=dtype), mt.Tensor(t, dtype=dtype), reduction)
    if reduction == "none":
        assert result.shape == tuple(np.shape(p))
        return np.ravel(result.tolist()).tolist()
    assert result.shape == ()
    return result.item()


def _same(got, want) -> bool:
    if isinstance(want, list):
        return len(got) == len(want) and all(map(_same, got, want))
    return (math.isnan(got) and math.isnan(want)) or got == want


def _draw(rng, shape, kind: str, dtype: str):
    p, t = rng.normal(size=shape), rng.normal(size=shape)
    if kind == "offset":
        # Targets far from the predictions, the differences a few units
        # each in a large magnitude.
        t = t + 1e3
    elif kind == "close":
        # Predictions within a few ulps of their targets.
        p = t * (1 + rng.normal(size=shape) * 1e-6)
    elif kind == "wide":
        p = p * 2.0 ** rng.integers(-30, 31, shape)
        t = t * 2.0 ** rng.integers(-30, 31, shape)
    elif kind == "past the range":
        # Squares, or sums of them, that pass an end of the range.
        ends = [-560, 520] if dtype == "float64" else [-80, 62]
        scale = 2.0 ** rng.choice(ends, size=shape)
        p, t = p * scale, t * scale
    return p.astype(dtype), t.astype(dtype)


KINDS = ["normal", "offset", "close", "wide", "past the range"]
REDUCTIONS = ["mean", "sum", "none"]


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("squared", [True, False], ids=["mse", "l1"])
def test_random_operands(dtype, kind, squared):
    rng = np.random.default_rng(641)
    for trial in range(20):
        shape = (int(rng.integers(1, 6)), int(rng.integers(1, 40)))
        p, t = _draw(rng, shape, kind, dtype)
        for reduction in REDUCTIONS:
            got = _got(p, t, squared, reduction, dtype)
            assert _same(got, _want(p, t, squared, reduction, dtype)), (
                trial,
                reduction,
            )


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("squared", [True, False], ids=["mse", "l1"])
def test_long_operands(dtype, squared):
    # Long enough to be summed in blocks across threads.
    rng = np.random.default_rng(643)
    p = (rng.normal(size=30_000) + 10).astype(dtype)
    t = rng.normal(size=30_000).astype(dtype)
    # Every so often a square the plain one cannot settle: on a float32
    # halfway point, or for float64 in the subnormal range.
    if dtype == "float32":
        p[::997], t[::997] = 1 + 2.0**-12, -(2.0**-60)
    else:
        p[::997], t[::997] = 3.0 * 2.0**-540, 2.0**-541
    for reduction in REDUCTIONS:
        got = _got(p, t, squared, reduction, dtype)
        assert _same(got, _want(p, t, squared, reduction, dtype)), reduction


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_specials_and_empty(dtype):
    nan, inf = math.nan, math.inf
    cases = [
        ([1.0, nan], [0.0, 1.0], nan, nan),
        ([1.0, inf], [0.0, 1.0], inf, inf),
        ([inf, 1.0], [inf, 1.0], nan, nan),
        ([-inf, 2.0], [inf, 0.0], inf, inf),
    ]
    for p, t, squared, magnitude in cases:
        tensors = mt.tensor(p, dtype=dtype), mt.tensor(t, dtype=dtype)
        for reduction in ("mean", "sum"):
            assert _same(nn.mse_loss(*tensors, reduction).item(), squared)
            assert _same(nn.l1_loss(*tensors, reduction).item(), magnitude)
    empty = mt.Tensor(np.zeros(0, dtype=dtype), dtype=dtype)
    assert math.isnan(nn.mse_loss(empty, empty).item())
    assert math.isnan(nn.l1_loss(empty, empty).item())
    assert nn.mse_loss(empty, empty, "sum").item() == 0.0
    assert nn.l1_loss(empty, empty, "sum").item() == 0.0
    # Equal operands lose nothing.
    same = mt.tensor([0.1, -3.0, 1e-30], dtype=dtype)
    assert nn.mse_loss(same, same).item() == 0.0
    assert nn.l1_loss(same, same, "sum").item() == 0.0


def test_differences_past_the_range():
    # Finite operands a whole range apart, whose mean magnitude is a float.
    big = float(np.finfo(np.float64).max) * 0.9
    p, t = [big, 0.0], [-big, 0.0]
    got = nn.l1_loss(mt.tensor(p, dtype="float64"), mt.tensor(t, dtype="float64"))
    assert got.item() == big
    got = nn.mse_loss(mt.tensor(p, dtype="float64"), mt.tensor(t, dtype="float64"))
    assert got.item() == math.inf
    # A difference rounded up to `2^512`, whose square would round past the
    # range, though the exact one is the largest float.
    p, t = [2.0**512], [2.0**458]
    want = _want(p, t, True, "none", "float64")
    assert want == [float(np.finfo(np.float64).max)]
    for reduction in REDUCTIONS:
        got = _got(np.array(p), np.array(t), True, reduction, "float64")
        assert _same(got, want if reduction == "none" else want[0]), reduction


def test_squares_on_a_halfway_point():
    # `1 + 2^-12` less `-2^-60` rounds to `1 + 2^-12` in float64, whose
    # square is the float32 halfway point `1 + 2^-11 + 2^-24`; the exact
    # square lies above it.
    p, t = np.array([1 + 2.0**-12], np.float32), np.array([-(2.0**-60)], np.float32)
    want = [1 + 2.0**-11 + 2.0**-23]
    assert _want(p, t, True, "none", "float32") == want
    for reduction in REDUCTIONS:
        got = _got(p, t, True, reduction, "float32")
        assert _same(got, want if reduction == "none" else want[0]), reduction


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("reduction", ["mean", "sum"])
def test_gradients(dtype, reduction):
    rng = np.random.default_rng(647)
    p, t = rng.normal(size=(3, 4)), rng.normal(size=(3, 4))
    for squared in (True, False):
        loss = nn.mse_loss if squared else nn.l1_loss
        x = mt.Tensor(p, dtype=dtype, requires_grad=True)
        y = mt.Tensor(t, dtype=dtype, requires_grad=True)
        loss(x, y, reduction).backward()
        a, b = p.astype(dtype).astype(np.float64), t.astype(dtype).astype(np.float64)
        want = 2 * (a - b) if squared else np.sign(a - b)
        if reduction == "mean":
            want = want / a.size
        tolerance = 1e-6 if dtype == "float32" else 1e-14
        np.testing.assert_allclose(x.grad.tolist(), want, rtol=tolerance)
        np.testing.assert_allclose(y.grad.tolist(), -want, rtol=tolerance)
