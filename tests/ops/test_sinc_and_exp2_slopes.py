# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""`sinc`'s and `exp2`'s slopes against exact references.

`sinc`'s slope `(cos(pi x) - sinc(x)) / x` is the difference of two terms
that agree near the origin and again near the slope's zeros away from it,
where `tan(pi x) = pi x`. As float64 the difference there was what the terms'
roundings left of it, up to 7e14 ulps out at the inputs nearest the zeros,
and a float32 slope took `pi` rounded to float32, which near the zeros was
all of it. The series by the origin also stopped a term short, sixty ulps at its
cutoff. The slope is now formed as pairs and rounded once.

`exp2`'s float32 slope likewise took `ln 2` rounded to float32, an ulp out
for one input in ten.
"""

from __future__ import annotations

import decimal
import functools
import math
from decimal import Decimal
from fractions import Fraction

import numpy as np
import pytest

import minitensor as mt

_DIGITS = {"float32": (24, -149, 128), "float64": (53, -1074, 1024)}

# The first zeros of the slope, the float64 nearest each root of
# `tan(pi x) = pi x` above one.
_ZEROS = [1.4302966531242027, 2.459024032956761, 3.47088972434702, 4.477408583432515]


def _rounded(value: Decimal, dtype: str) -> float:
    """The float nearest a decimal value, ties to even, rounded once."""

    if value == 0:
        return 0.0
    exact = Fraction(abs(value))
    digits, least, top = _DIGITS[dtype]
    e = exact.numerator.bit_length() - exact.denominator.bit_length()
    if Fraction(2) ** e > exact:
        e -= 1
    unit = Fraction(2) ** max(e - (digits - 1), least)
    near = round(exact / unit) * unit
    out = math.inf if near >= Fraction(2) ** top else float(near)
    return -out if value < 0 else out


@functools.lru_cache(maxsize=None)
def _pi_to(prec: int) -> Decimal:
    with decimal.localcontext(prec=prec):
        return _machin()


def _pi() -> Decimal:
    return _pi_to(decimal.getcontext().prec)


def _machin() -> Decimal:
    """Machin's formula, to the context's precision."""

    def arctan_inverse(n: int) -> Decimal:
        x = Decimal(1) / n
        total, term, k = x, x, 1
        while True:
            term = -term * x * x
            step = term / (2 * k + 1)
            if abs(step) < Decimal(10) ** -(decimal.getcontext().prec + 2):
                return total
            total, k = total + step, k + 1

    return 4 * (4 * arctan_inverse(5) - arctan_inverse(239))


def _sin_cos(theta: Decimal) -> tuple[Decimal, Decimal]:
    """Both series, for `|theta|` at most a little over `pi`, each to the
    context's precision of `theta^3`: by the origin the slope is `theta^3 / 3`
    of terms near `theta`, and the sine's `theta^3 / 6` is a part of it."""

    if theta == 0:
        return Decimal(0), Decimal(1)
    scale = min(Decimal(1), abs(theta)) ** 3
    limit = Decimal(10) ** -(decimal.getcontext().prec + 2) * scale
    sine, cosine = Decimal(0), Decimal(0)
    term, n = Decimal(1), 0
    while True:
        if n % 2 == 0:
            cosine += term if n % 4 == 0 else -term
        else:
            sine += term if n % 4 == 1 else -term
        n += 1
        term = term * theta / n
        if abs(term) < limit:
            return sine, cosine


def _sinc_slope(x: float) -> Decimal:
    """`(pi x cos(pi x) - sin(pi x)) / (pi x^2)`, with `x` reduced mod 2
    exactly -- the remainder of a float by 2 is one -- and as many digits
    again as the terms cancel by near the origin."""

    if x == 0:
        return Decimal(0)
    extra = max(0, -2 * math.floor(math.log10(abs(x))))
    with decimal.localcontext(prec=80 + extra):
        d = Decimal(x)
        r = Decimal(math.remainder(x, 2.0))
        pi = _pi()
        sine, cosine = _sin_cos(pi * r)
        u = pi * d
        slope = (u * cosine - sine) / (u * d)
    return +slope


def _exp2_slope(x: float) -> Decimal:
    with decimal.localcontext(prec=60):
        ln2 = Decimal(2).ln()
        return ln2 * (Decimal(x) * ln2).exp()


def _gradients(fn, xs: np.ndarray, dtype: str, upstream=None) -> list[float]:
    t = mt.Tensor(xs, dtype=dtype, requires_grad=True)
    if upstream is None:
        upstream = np.ones(len(xs))
    fn(t).backward(mt.Tensor(upstream, dtype=dtype))
    got = t.grad.numpy().astype(np.float64).tolist()
    mt.clear_autograd_graph()
    return got


def _spread(dtype: str) -> np.ndarray:
    rng = np.random.default_rng(2207)
    pieces = [
        np.exp(rng.uniform(np.log(1e-300), np.log(1e-3), 60)),
        # Either side of where the series by the origin hands over.
        rng.uniform(0.02, 0.04, 60),
        rng.uniform(0.04, 0.6, 60),
        rng.uniform(0.6, 6.0, 120),
        np.exp(rng.uniform(np.log(6.0), np.log(1e4), 60)),
        np.exp(rng.uniform(np.log(1e4), np.log(1e15), 40)),
        np.arange(1, 13) * 0.5,
        [2.0**51 + 0.5, 2.0**52 + 1.0, 2.0**60, 1e300],
    ]
    xs = np.concatenate([np.asarray(p, dtype=np.float64) for p in pieces])
    if dtype == "float32":
        # Those float32 holds, and not as zero.
        xs = xs[(xs < 3e38) & (xs > 1e-45)]
    return np.concatenate([xs, -xs]).astype(dtype)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_sinc_slope_is_the_nearest_float_across_its_range(dtype):
    xs = _spread(dtype)
    got = _gradients(lambda t: t.sinc(), xs, dtype)
    for x, g in zip(xs.astype(np.float64).tolist(), got):
        want = _rounded(_sinc_slope(x), dtype)
        assert g == want, (dtype, x, g, want)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_sinc_slope_rounds_its_product_with_the_incoming_gradient_once(dtype):
    rng = np.random.default_rng(31)
    xs = rng.uniform(-6.0, 6.0, 200).astype(dtype)
    upstream = (rng.standard_normal(200) * 10.0 ** rng.uniform(-3, 3, 200)).astype(
        dtype
    )
    got = _gradients(lambda t: t.sinc(), xs, dtype, upstream)
    for x, w, g in zip(xs.astype(np.float64).tolist(), upstream.tolist(), got):
        want = _rounded(Decimal(float(w)) * _sinc_slope(x), dtype)
        assert g == want, (dtype, x, w, g, want)


def _nearest(root: float, dtype: str, count: int) -> np.ndarray:
    """The `count` floats either side of `root`, and it."""

    centre = np.array(root, dtype=dtype)
    below, above, xs = centre, centre, [centre]
    for _ in range(count):
        below = np.nextafter(below, np.array(-np.inf, dtype=dtype))
        above = np.nextafter(above, np.array(np.inf, dtype=dtype))
        xs += [below, above]
    return np.array(xs, dtype=dtype)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("zero", _ZEROS + [-z for z in _ZEROS[:2]])
def test_sinc_slope_at_the_inputs_nearest_its_zeros(zero, dtype):
    # Where the float64 slope had no digit right, and the float32 one took
    # all of itself from `pi`'s rounding.
    xs = _nearest(zero, dtype, 30)
    got = _gradients(lambda t: t.sinc(), xs, dtype)
    for x, g in zip(xs.astype(np.float64).tolist(), got):
        want = _rounded(_sinc_slope(x), dtype)
        if dtype == "float32":
            assert g == want, (x, g, want)
        else:
            # The pairs hold 2^-104 of the terms, which this near a zero
            # can be a part of an ulp of the slope.
            assert abs(g - want) <= math.ulp(want), (x, g, want)


def test_sinc_slope_by_the_origin_and_at_the_whole_numbers():
    # Zero at the origin, which the symmetry of an even function requires;
    # +-1/x at the whole numbers, where `cos(pi x)` is +-1 and `sinc` is 0.
    xs = np.array([0.0, 1.0, -1.0, 2.0, 3.0, -7.0, 2.0**53, math.inf])
    got = _gradients(lambda t: t.sinc(), xs, "float64")
    assert got == [0.0, -1.0, 1.0, 0.5, -1.0 / 3.0, 1.0 / 7.0, 2.0**-53, 0.0]


def test_float32_exp2_slope_is_the_nearest_float():
    rng = np.random.default_rng(5)
    xs = np.concatenate(
        [rng.uniform(-30.0, 30.0, 1500), rng.uniform(-149.0, 128.0, 500)]
    ).astype(np.float32)
    got = _gradients(lambda t: t.exp2(), xs, "float32")
    bad = [
        (x, g)
        for x, g in zip(xs.astype(np.float64).tolist(), got)
        if g != _rounded(_exp2_slope(x), "float32")
    ]
    assert not bad, (len(bad), bad[:5])


def test_exp2_slope_past_where_two_to_the_x_overflows():
    # `2^x ln 2` is below the top of float64 up to `x = 1024.53`, though `2^x`
    # itself overflows from 1024; and below float32's top up to 128.53.
    for dtype, xs in (
        ("float64", [1023.5, 1024.0, 1024.25, 1024.5]),
        ("float32", [128.0, 128.5]),
    ):
        got = _gradients(lambda t: t.exp2(), np.array(xs), dtype)
        for x, g in zip(xs, got):
            want = _rounded(_exp2_slope(x), dtype)
            assert math.isfinite(g) and abs(g - want) <= 2 * math.ulp(want), (
                dtype,
                x,
                g,
                want,
            )
