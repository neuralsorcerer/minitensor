# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""`logcumsumexp` and its gradient are rounded once.

Each prefix folded the next value into the rounded running one, so every
step's rounding stayed in every later prefix: of a thousand float64 prefixes
of standard normals most were an ulp or more out, and of a hundred thousand
some a hundred. The gradient was formed from the rounded outputs as
`exp(x + log(Σ g exp(-y)))`, its sums run as logarithms that drifted the same
way, positive and negative `g` apart and subtracted at the end: float64
slopes were tens to thousands of ulps out, and where the two parts cancelled
float32 slopes thousands.

A prefix is now carried as its largest value and the excess of
`Σ exp(x - max)` over one as a pair of floats, and a slope as the softmax of
each prefix summed from the end of the run back, scaled pairs with the parts
their sums could not hold kept beside them: each is the float nearest a value
within about `2^-100` of the exact one.
"""

from __future__ import annotations

import math
from fractions import Fraction

import mpmath
import numpy as np
import pytest

import minitensor as mt
from minitensor import functional as F

_DIGITS = {"float32": (24, -149, 128), "float64": (53, -1074, 1024)}


def _rounded(value, dtype: str) -> float:
    """The float nearest an mpmath value, ties to even, rounded once."""

    sign, man, exponent, _ = value._mpf_
    if man == 0:
        return 0.0
    exact = Fraction(int(man)) * Fraction(2) ** int(exponent)
    digits, least, top = _DIGITS[dtype]
    e = exact.numerator.bit_length() - exact.denominator.bit_length()
    if Fraction(2) ** e > exact:
        e -= 1
    unit = Fraction(2) ** max(e - (digits - 1), least)
    near = round(exact / unit) * unit
    out = math.inf if near >= Fraction(2) ** top else float(near)
    return -out if sign else out


def _prefixes(x):
    total, out = mpmath.mpf(0), []
    for v in x:
        total += mpmath.exp(mpmath.mpf(float(v)))
        out.append(mpmath.log(total))
    return out


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_every_prefix_is_the_nearest_float(dtype):
    rng = np.random.default_rng(1701)
    with mpmath.workprec(160):
        for scale in (0.01, 3.0, 300.0):
            x = (rng.standard_normal(1500) * scale).astype(dtype)
            got = F.logcumsumexp(mt.Tensor(x, dtype=dtype), 0).tolist()
            assert got == [_rounded(y, dtype) for y in _prefixes(x)], scale


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_a_long_run_does_not_drift(dtype):
    # A rounding at every step put a hundred float64 ulps into the prefixes
    # of a hundred thousand values.
    rng = np.random.default_rng(1709)
    x = (rng.standard_normal(100_000) * 3).astype(dtype)
    got = F.logcumsumexp(mt.Tensor(x, dtype=dtype), 0).numpy()
    with mpmath.workprec(120):
        want = _prefixes(x)
    for at in range(0, len(x), 997):
        assert float(got[at]) == _rounded(want[at], dtype), at


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_along_every_axis(dtype):
    rng = np.random.default_rng(1721)
    x = (rng.standard_normal((4, 5, 6)) * 4).astype(dtype)
    with mpmath.workprec(160):
        for dim in range(3):
            got = np.moveaxis(
                F.logcumsumexp(mt.Tensor(x, dtype=dtype), dim).numpy(), dim, -1
            )
            runs = np.moveaxis(x, dim, -1).reshape(-1, x.shape[dim])
            for row, run in zip(got.reshape(-1, x.shape[dim]), runs):
                assert row.tolist() == [_rounded(y, dtype) for y in _prefixes(run)]


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_prefixes_at_the_ends(dtype):
    inf, nan = math.inf, math.nan
    got = F.logcumsumexp(mt.Tensor([-inf, -inf, 0.0, -inf], dtype=dtype), 0).tolist()
    assert got == [-inf, -inf, 0.0, 0.0]
    got = F.logcumsumexp(mt.Tensor([1.0, inf, 2.0, -inf], dtype=dtype), 0).tolist()
    assert got == [1.0, inf, inf, inf]
    got = F.logcumsumexp(mt.Tensor([1.0, nan, inf], dtype=dtype), 0).tolist()
    assert got[0] == 1.0 and math.isnan(got[1]) and math.isnan(got[2])
    # A value far below the largest is far below anything the sum holds.
    got = F.logcumsumexp(mt.Tensor([5.0, -1000.0], dtype=dtype), 0).tolist()
    assert got == [5.0, 5.0]


def _slopes(x, g, dtype, dim=0):
    t = mt.Tensor(np.asarray(x, dtype=dtype), dtype=dtype, requires_grad=True)
    F.logcumsumexp(t, dim).backward(mt.Tensor(np.asarray(g, dtype=dtype), dtype=dtype))
    got = t.grad.tolist()
    mt.clear_autograd_graph()
    return got


def _softmax(prefix):
    """The slope `logsumexp` gives each entry of a slice, as (kind, value)."""

    inf = math.inf
    if any(math.isnan(v) for v in prefix):
        return [("nan",)] * len(prefix)
    count = sum(v == inf for v in prefix)
    if count:
        return [
            ("pos", mpmath.mpf(1) / count) if v == inf else ("zero",) for v in prefix
        ]
    if all(v == -inf for v in prefix):
        return [("zero",)] * len(prefix)
    total = mpmath.fsum(mpmath.exp(mpmath.mpf(v)) for v in prefix if v != -inf)
    return [
        ("zero",) if v == -inf else ("pos", mpmath.exp(mpmath.mpf(v)) / total)
        for v in prefix
    ]


def _exact_slopes(x, g, dtype, cancelled=None):
    """`Σ_{k ≥ i} g[k] s_k(i)` with each term multiplied out as IEEE does:
    NaN for a NaN or an infinite `g` against a zero share, a zero signed as
    the product, and the sum NaN for both infinities, a zero negative only
    where every term is, and otherwise the exact sum rounded once. Into
    `cancelled`, where given, whether each sum is under `2^-100` of its
    terms."""

    x, g = [float(v) for v in x], [float(v) for v in g]
    shares = [_softmax(x[: k + 1]) for k in range(len(x))]
    out = []
    for i in range(len(x)):
        terms = []
        for k in range(i, len(x)):
            share, weight = shares[k][i], g[k]
            if share[0] == "nan" or math.isnan(weight):
                terms.append(("nan",))
            elif math.isinf(weight):
                terms.append(("nan",) if share[0] == "zero" else ("inf", weight < 0))
            elif share[0] == "zero" or weight == 0:
                terms.append(("zero", math.copysign(1.0, weight) < 0))
            else:
                terms.append(("pos", mpmath.mpf(weight) * share[1]))
        infinities = {t[1] for t in terms if t[0] == "inf"}
        if any(t[0] == "nan" for t in terms) or len(infinities) == 2:
            out.append(math.nan)
        elif infinities:
            out.append(-math.inf if True in infinities else math.inf)
        elif all(t[0] == "zero" for t in terms):
            out.append(-0.0 if all(t[1] for t in terms) else 0.0)
        else:
            finite = [t[1] for t in terms if t[0] == "pos"]
            total = mpmath.fsum(finite)
            if cancelled is not None:
                size = mpmath.fsum(abs(t) for t in finite)
                cancelled.append(abs(total) < size * mpmath.mpf(2) ** -100)
            out.append(_rounded(total, dtype))
            continue
        if cancelled is not None:
            cancelled.append(False)
    return out


def _same(got, want):
    for a, b in zip(got, want):
        if math.isnan(b):
            assert math.isnan(a), (got, want)
        else:
            assert a == b and math.copysign(1.0, a) == math.copysign(1.0, b), (
                got,
                want,
            )


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_every_slope_is_the_nearest_float(dtype):
    rng = np.random.default_rng(1723)
    with mpmath.workprec(200):
        for scale, mixed in ((3.0, False), (3.0, True), (30.0, False), (300.0, True)):
            x = (rng.standard_normal(60) * scale).astype(dtype)
            g = rng.standard_normal(60) if mixed else np.abs(rng.standard_normal(60))
            g = g.astype(dtype)
            _same(_slopes(x, g, dtype), _exact_slopes(x, g, dtype))


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_slopes_along_every_axis(dtype):
    rng = np.random.default_rng(1733)
    x = (rng.standard_normal((3, 4, 5)) * 5).astype(dtype)
    g = rng.standard_normal((3, 4, 5)).astype(dtype)
    with mpmath.workprec(200):
        for dim in range(3):
            got = np.moveaxis(np.array(_slopes(x, g, dtype, dim)), dim, -1)
            runs = np.moveaxis(x, dim, -1).reshape(-1, x.shape[dim])
            grads = np.moveaxis(g, dim, -1).reshape(-1, x.shape[dim])
            for row, run, grad in zip(got.reshape(-1, x.shape[dim]), runs, grads):
                _same(row.tolist(), _exact_slopes(run, grad, dtype))


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_slopes_at_the_ends(dtype):
    # Each prefix's slopes are softmax's: shared among its infinities, none
    # to its `-inf` entries, and NaN throughout one that holds a NaN.
    rng = np.random.default_rng(1741)
    inf, nan = math.inf, math.nan
    values = [0.0, 1.0, -2.0, 3.5, inf, -inf, nan, 700.0, -700.0]
    weights = [0.0, -0.0, 1.0, -1.0, 2.5, inf, -inf, nan]
    with mpmath.workprec(4000):
        for _ in range(250):
            n = int(rng.integers(1, 6))
            x = [values[j] for j in rng.integers(0, len(values), n)]
            g = (
                [weights[j] for j in rng.integers(0, len(weights), n)]
                if rng.uniform() < 0.5
                else rng.standard_normal(n).tolist()
            )
            x = np.array(x, dtype=dtype).tolist()
            g = np.array(g, dtype=dtype).tolist()
            cancelled = []
            got = _slopes(x, g, dtype)
            want = _exact_slopes(x, g, dtype, cancelled)
            # Where the terms cancel past what pairs hold -- to 2^-1000 of
            # themselves, as entries 700 above and below their neighbours
            # make them -- the slope is within a few of its last places.
            for a, b, far in zip(got, want, cancelled):
                if far and a != b:
                    assert abs(a - b) <= 4 * float(
                        np.spacing(np.array(abs(b), dtype=dtype))
                    )
                else:
                    _same([a], [b])


def test_a_halfway_point_broken_by_entries_far_below():
    # With `x = [0, 0, -80]` the largest entry's slope is
    # `g[0] + g[1] / 2 + g[2] / (2 + e^-80)`: here `1 + 1.5 * 2^-52` less a part
    # 2^-117 of it -- halfway between two floats but for that part, which a
    # pair's sum drops and a crumb beside it keeps. Ties to even would round
    # it up.
    for dtype, place in (("float64", 2.0**-52), ("float32", 2.0**-23)):
        got = _slopes([0.0, 0.0, -80.0], [1.0 + place, 0.0, place], dtype)
        assert got[0] == 1.0 + place, (dtype, got)
