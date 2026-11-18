# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""`cumprod`'s slope with respect to each factor is rounded once.

The slope of `x[i]` is `Σ_{k ≥ i} g[k] · Π_{j ≤ k, j ≠ i} x[j]`, for `g` the
gradient of the running products. It was the running sum of `g · cumprod`,
rounded at every step, divided by `x[i]`, and where a running product left
the range the slopes went with it: `cumprod([1e200, 1e200, 1e-200])` under a
gradient of `[0, 0, 1]` had slopes of NaN where two of them are one, and
`[1e300, 1e300]` had slopes of infinity where each is the other factor.

Each slope is now the float nearest that exact sum -- the product of the
factors before `i` times the sum from the end of the run back, each carried
with its exponent apart -- and a zero, an infinity or a NaN among the values
makes it what multiplying out its terms and adding them makes it.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import minitensor as mt


def _dyadic(value: float) -> tuple[int, int]:
    """A finite float as `m · 2^e`, exactly."""

    numerator, denominator = float(value).as_integer_ratio()
    return numerator, 1 - denominator.bit_length()


def _plus(a: tuple[int, int], b: tuple[int, int]) -> tuple[int, int]:
    (m, e), (n, f) = a, b
    low = min(e, f)
    return (m << (e - low)) + (n << (f - low)), low


def _rounded(value: tuple[int, int], dtype: str) -> float:
    """The float nearest `m · 2^e`, ties to even; zero unsigned."""

    m, e = value
    if m == 0:
        return 0.0
    digits, least, top = (24, -149, 128) if dtype == "float32" else (53, -1074, 1024)
    sign, m = (-1.0 if m < 0 else 1.0), abs(m)
    place = max(m.bit_length() - 1 + e - (digits - 1), least)
    cut = place - e
    if cut > 0:
        kept, rest = m >> cut, m & ((1 << cut) - 1)
        half = 1 << (cut - 1)
        m = kept + (rest > half or (rest == half and kept & 1))
    else:
        place = e
    if m.bit_length() + place > top:
        return sign * math.inf
    return sign * math.ldexp(float(m), place)


def _product(values) -> tuple[int, int]:
    m, e = 1, 0
    for value in values:
        n, f = _dyadic(value)
        m, e = m * n, e + f
    return m, e


def _by_parts(x, g, dtype):
    """Each slope exactly, as the product before it times the sum after it;
    for finite values, a zero slope unsigned."""

    xs = [_dyadic(v) for v in x]
    gs = [_dyadic(v) for v in g]
    before = [(1, 0)]
    for m, e in xs[:-1]:
        before.append((before[-1][0] * m, before[-1][1] + e))
    out = [0.0] * len(xs)
    after = gs[-1]
    for i in range(len(xs) - 1, -1, -1):
        if i < len(xs) - 1:
            after = _plus(gs[i], (xs[i + 1][0] * after[0], xs[i + 1][1] + after[1]))
        out[i] = _rounded((before[i][0] * after[0], before[i][1] + after[1]), dtype)
    return out


def _term(factors):
    """A product multiplied out: NaN, a signed infinity or zero, or exact."""

    if any(math.isnan(f) for f in factors):
        return ("nan",)
    negative = sum(math.copysign(1.0, f) < 0 for f in factors) % 2 == 1
    zero = any(f == 0 for f in factors)
    infinite = any(math.isinf(f) for f in factors)
    if zero and infinite:
        return ("nan",)
    if infinite:
        return ("inf", negative)
    if zero:
        return ("zero", negative)
    return ("finite", _product(factors))


def _by_terms(x, g, dtype):
    """Each slope as its terms multiplied out and added: NaN for a NaN term
    or infinities of both signs, a zero negative only where every term is,
    and otherwise the exact sum rounded once."""

    x, g = [float(v) for v in x], [float(v) for v in g]
    out = []
    for i in range(len(x)):
        terms = [
            _term([g[k]] + [x[j] for j in range(k + 1) if j != i])
            for k in range(i, len(x))
        ]
        infinities = {t[1] for t in terms if t[0] == "inf"}
        if any(t[0] == "nan" for t in terms) or len(infinities) == 2:
            out.append(math.nan)
        elif infinities:
            out.append(-math.inf if True in infinities else math.inf)
        elif all(t[0] == "zero" for t in terms):
            out.append(-0.0 if all(t[1] for t in terms) else 0.0)
        else:
            total = (0, 0)
            for t in terms:
                if t[0] == "finite":
                    total = _plus(total, t[1])
            out.append(_rounded(total, dtype))
    return out


def _slopes(x, g, dtype, dim=0):
    t = mt.Tensor(np.asarray(x, dtype=dtype), dtype=dtype, requires_grad=True)
    t.cumprod(dim).backward(mt.Tensor(np.asarray(g, dtype=dtype), dtype=dtype))
    got = t.grad.numpy()
    mt.clear_autograd_graph()
    return got


def _same(got, want):
    """Equal, NaN for NaN, and a zero's sign where `want` gives one."""

    for a, b in zip(got, want):
        if math.isnan(b):
            assert math.isnan(a), (got, want)
        else:
            assert a == b and math.copysign(1.0, a) == math.copysign(1.0, b), (
                got,
                want,
            )


@pytest.mark.parametrize(
    "x,g,want",
    [
        ([1e200, 1e200, 1e-200], [0.0, 0.0, 1.0], None),
        ([1e300, 1e300], [0.0, 1.0], [1e300, 1e300]),
        ([1e-300, 1e-300], [0.0, 1.0], [1e-300, 1e-300]),
        ([1e-200, 1e-200, 1e200, 1e200], [0.0, 0.0, 0.0, 1.0], None),
        ([1e300, 1e300, 1e-300, 1e-300], [1.0, 1.0, 1.0, 1.0], None),
    ],
)
def test_slopes_where_the_running_products_leave_the_range(x, g, want):
    got = _slopes(x, g, "float64").tolist()
    assert got == _by_parts(x, g, "float64")
    if want is not None:
        assert got == want
    assert all(not math.isnan(v) for v in got)


def test_the_first_case_in_full():
    # The slopes of the first two are 1e200 · 1e-200, the third's 1e400.
    got = _slopes([1e200, 1e200, 1e-200], [0.0, 0.0, 1.0], "float64").tolist()
    one = _rounded(_product([1e200, 1e-200]), "float64")
    assert got == [one, one, math.inf]


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_every_slope_is_the_nearest_float_to_its_exact_sum(dtype):
    rng = np.random.default_rng(1601)
    top = 100 if dtype == "float32" else 900
    for n in (1, 2, 3, 5, 9):
        for _ in range(40):
            spread = rng.uniform(-top, top, n) / max(n, 1) ** 0.5
            x = rng.choice([-1, 1], n) * rng.uniform(1, 2, n) * 2.0**spread
            g = rng.standard_normal(n) * 2.0 ** rng.uniform(-top / 4, top / 4, n)
            x, g = x.astype(dtype), g.astype(dtype)
            got = _slopes(x, g, dtype).tolist()
            assert got == _by_parts(x, g, dtype), (x.tolist(), g.tolist())


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_long_runs_and_every_axis(dtype):
    # Past 256 factors the products before each index are kept a span at a
    # time; down a column or along a middle axis the runs are gathered.
    rng = np.random.default_rng(1607)
    for n in (255, 256, 257, 600):
        x = (rng.choice([-1, 1], n) * 2.0 ** rng.uniform(-1, 1, n)).astype(dtype)
        g = rng.standard_normal(n).astype(dtype)
        assert _slopes(x, g, dtype).tolist() == _by_parts(x, g, dtype)
    x = (
        rng.choice([-1, 1], (5, 6, 7)) * 2.0 ** rng.uniform(-30, 30, (5, 6, 7))
    ).astype(dtype)
    g = rng.standard_normal((5, 6, 7)).astype(dtype)
    for dim in range(3):
        got = np.moveaxis(_slopes(x, g, dtype, dim), dim, -1).reshape(-1, x.shape[dim])
        runs = np.moveaxis(x, dim, -1).reshape(-1, x.shape[dim])
        grads = np.moveaxis(g, dim, -1).reshape(-1, x.shape[dim])
        for row, xr, gr in zip(got, runs, grads):
            assert row.tolist() == _by_parts(xr, gr, dtype)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_zeros_infinities_and_nans(dtype):
    rng = np.random.default_rng(1609)
    inf, nan = math.inf, math.nan
    pool = np.array(
        [0.0, -0.0, 1.0, -1.0, 2.0, -3.0, 0.5, inf, -inf, nan, 1e30, -1e-30]
    )

    def pick(n):
        if rng.uniform() < 0.5:
            return pool[rng.integers(0, len(pool), n)]
        return rng.integers(-3, 4, n).astype(float)

    for _ in range(300):
        n = int(rng.integers(1, 6))
        x = pick(n).astype(dtype).astype(float)
        g = pick(n).astype(dtype).astype(float)
        _same(_slopes(x, g, dtype).tolist(), _by_terms(x, g, dtype))


def test_zeros_in_the_run():
    # No division by a factor, so a zero needs no care of its own: the zero's
    # own slope is the product before it times the sum after it, and every
    # slope after it carries it.
    got = _slopes([2.0, 0.0, 3.0, 0.0], [1.0, 1.0, 1.0, 1.0], "float64").tolist()
    assert got == [1.0, 8.0, 0.0, 0.0]
    got = _slopes([0.0, 2.0, 4.0], [5.0, -5.0, -5.0], "float64").tolist()
    # Each later slope's terms are all negative zeros.
    assert got == [-45.0, 0.0, 0.0]
    assert [math.copysign(1.0, v) for v in got[1:]] == [-1.0, -1.0]


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_terms_that_cancel(dtype):
    # Terms that cancel exactly make a positive zero, found without forming
    # the slope exactly: the roundings' bound is zero where none rounded.
    n = 2001
    x = np.ones(n)
    g = np.array([(-1.0) ** k for k in range(n)])
    got = _slopes(x, g, dtype).tolist()
    want = [0.0 if (n - i) % 2 == 0 else 1.0 for i in range(n)]
    _same(got, want)
    # Terms that cancel to nearly nothing, which the carried sum cannot
    # place, are formed exactly.
    rng = np.random.default_rng(1613)
    eps = 2.0**-52 if dtype == "float64" else 2.0**-23
    for _ in range(100):
        k = int(rng.integers(2, 6))
        x = (rng.uniform(1, 2, k) * rng.choice([-1, 1], k)).astype(dtype).astype(float)
        g = [0.0] * k
        g[-1] = float(np.array(rng.uniform(1, 2), dtype=dtype))
        g[-2] = float(
            np.array(-x[-1] * g[-1] * (1 + int(rng.integers(-2, 3)) * eps), dtype=dtype)
        )
        assert _slopes(x, g, dtype).tolist() == _by_parts(x, g, dtype)
    for _ in range(100):
        k = int(rng.integers(2, 7))
        x = rng.integers(-4, 5, k).astype(float)
        g = rng.integers(-6, 7, k).astype(float)
        _same(_slopes(x, g, dtype).tolist(), _by_terms(x, g, dtype))


def test_values_just_above_the_subnormals():
    # Running products that fall through the last binades of the normal
    # range and among the subnormals: their slopes, the running products
    # themselves and products landing there, each the nearest float64.
    rng = np.random.default_rng(1619)
    n = 1100
    x = 0.5 * rng.uniform(0.9, 1.1, n)
    g = rng.standard_normal(n)
    t = mt.Tensor(x, dtype="float64", requires_grad=True)
    running = t.cumprod(0)
    running.backward(mt.Tensor(g, dtype="float64"))
    got = t.grad.numpy().tolist()
    mt.clear_autograd_graph()
    want = _by_parts(x, g, "float64")
    assert sum(1 for v in want if 0 < abs(v) < 2.0**-969) > 80
    assert got == want
    product, exact = (1, 0), []
    for value in x:
        m, e = _dyadic(value)
        product = (product[0] * m, product[1] + e)
        exact.append(_rounded(product, "float64"))
    assert running.detach().numpy().tolist() == exact
    rows = rng.uniform(1, 2, (500, 4)) * 2.0 ** rng.integers(-260, -245, (500, 4))
    want = [_rounded(_product(row), "float64") for row in rows]
    assert mt.Tensor(rows, dtype="float64").prod(1).numpy().tolist() == want


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_a_long_run_through_the_bottom_of_the_range(dtype):
    # Two hundred thousand factors whose running products fall below the
    # range on the way: the slopes past that point are zero, those before
    # finite, and together they keep the identity that the slopes weighted
    # by their factors are the running products weighted by their gradients
    # and counts, `Σ x[i] s[i] = Σ (k + 1) g[k] y[k]`.
    rng = np.random.default_rng(1621)
    n = 200_000
    step = 2.0 ** (-0.0008 if dtype == "float32" else -0.0074)
    x = (step * rng.uniform(0.9, 1.1, n)).astype(dtype)
    g = rng.standard_normal(n).astype(dtype)
    t = mt.Tensor(x, dtype=dtype, requires_grad=True)
    running = t.cumprod(0)
    running.backward(mt.Tensor(g, dtype=dtype))
    slopes = t.grad.numpy().astype(np.float64)
    y = running.detach().numpy().astype(np.float64)
    mt.clear_autograd_graph()
    assert np.isfinite(slopes).all()
    assert slopes[-1000:].tolist() == [0.0] * 1000
    weighted = x.astype(np.float64) * slopes
    counted = (np.arange(n) + 1.0) * g.astype(np.float64) * y
    # Each slope and running product is within half an ulp of its own.
    ulp = 2.0**-24 if dtype == "float32" else 2.0**-52
    scale = max(np.abs(weighted).sum(), np.abs(counted).sum())
    assert (
        abs(math.fsum(weighted.tolist()) - math.fsum(counted.tolist())) <= ulp * scale
    )
