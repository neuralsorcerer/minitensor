# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""`normalize`'s and `renorm`'s gradients against exact values, for incoming
gradients almost along the slice's direction.

A slice taken over its norm moves by `(g - d (x . g) / |x|) / |x|` -- `d` its
direction for the 2-norm and its signs for the 1-norm -- times the cap for
`renorm`. Formed as tensor operations, `g - d (x . g) / |x|` was two terms
that all but cancel for a gradient almost along `d`: up to 1.8e8 ulps out in
float32 and 2.6e8 in float64. It is now formed whole, `g`'s part at right
angles to `x` for the 2-norm and `g` less the signs times a pair for the
1-norm, and rounded once.
"""

from __future__ import annotations

import decimal
import math
from decimal import Decimal
from fractions import Fraction

import numpy as np
import pytest

import minitensor as mt

_DIGITS = {"float32": (24, -149, 128), "float64": (53, -1074, 1024)}


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


def _close(got: float, want: Decimal, dtype: str) -> bool:
    """Within an ulp of the exact value in float32, and four in float64."""

    expected = _rounded(want, dtype)
    unit = float(np.spacing(np.array(abs(expected), dtype=dtype)))
    return abs(got - expected) <= (1 if dtype == "float32" else 4) * unit


def _moving(x, g, p: float, scale: Decimal):
    """`scale (g - d (x . g) / |x|) / |x|` of one slice, exactly, and its norm."""

    xs = [Decimal(float(v)) for v in x]
    gs = [Decimal(float(v)) for v in g]
    norm = sum(v * v for v in xs).sqrt() if p == 2 else sum(abs(v) for v in xs)
    unit = [v / norm for v in xs]
    if p == 2:
        direction = unit
    else:
        direction = [Decimal((v > 0) - (v < 0)) for v in xs]
    along = sum(a * b for a, b in zip(unit, gs))
    return [scale * (w - d * along) / norm for w, d in zip(gs, direction)], norm


def _along(x: np.ndarray, p: float, spread: float, rng) -> np.ndarray:
    """An incoming gradient for each row of `x` almost along its direction."""

    if p == 2:
        direction = x / np.linalg.norm(x, axis=1, keepdims=True)
    else:
        direction = np.sign(x)
    size = rng.uniform(0.5, 2.0, (x.shape[0], 1))
    return direction * size + spread * rng.standard_normal(x.shape)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("p", [2.0, 1.0])
@pytest.mark.parametrize("dim", [0, 1])
@pytest.mark.parametrize("spread", [1e-2, 1e-5])
def test_normalize(spread, dim, p, dtype):
    rng = np.random.default_rng(73)
    rows = rng.standard_normal((20, 12))
    upstream = _along(rows, p, spread, rng)
    rows, upstream = rows.astype(dtype), upstream.astype(dtype)
    x, g = (rows, upstream) if dim == 1 else (rows.T.copy(), upstream.T.copy())
    t = mt.Tensor(x, dtype=dtype, requires_grad=True)
    mt.normalize(t, p, dim).backward(mt.Tensor(g, dtype=dtype))
    got = t.grad.numpy().astype(np.float64)
    got = got if dim == 1 else got.T
    with decimal.localcontext(prec=80):
        for r in range(rows.shape[0]):
            want, _ = _moving(rows[r], upstream[r], p, Decimal(1))
            for k, w in enumerate(want):
                assert _close(got[r, k], w, dtype), (r, k, got[r, k], float(w))


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_normalize_under_its_floor(dtype):
    # A slice whose norm is under the floor is divided by the floor, and so
    # is its gradient.
    eps = 1e-3
    x = np.array([[0.0, 0.0, 0.0], [1e-4, -2e-4, 2e-4], [1.0, 2.0, 2.0]])
    g = np.array([[1.0, -2.0, 3.0], [0.5, 0.5, 0.5], [1.0, 2.0, 2.0]])
    t = mt.Tensor(x, dtype=dtype, requires_grad=True)
    mt.normalize(t, 2.0, 1, eps).backward(mt.Tensor(g, dtype=dtype))
    got = t.grad.numpy().astype(np.float64)
    floor = float(np.array(eps, dtype=dtype))
    for r in (0, 1):
        for k in range(3):
            assert _close(got[r, k], Decimal(g[r, k]) / Decimal(floor), dtype)
    # A gradient exactly along a slice's direction moves it nowhere.
    assert np.all(np.abs(got[2]) <= 4 * np.spacing(np.array(1.0, dtype=dtype)))


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("p", [2.0, 1.0])
def test_a_zero_slice_with_no_floor(p, dtype):
    # Its quotients are zero over zero, and so is its gradient.
    t = mt.Tensor(np.zeros((2, 3)), dtype=dtype, requires_grad=True)
    out = mt.normalize(t, p, 1, 0.0)
    out.backward(mt.Tensor(np.array([[1.0, -2.0, 0.0]] * 2), dtype=dtype))
    assert np.isnan(out.numpy()).all() and np.isnan(t.grad.numpy()).all()


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("p", [2.0, 1.0])
def test_an_infinite_incoming_gradient(p, dtype):
    # Each entry is its row of the slope times the incoming gradient, summed
    # as floats sum: off the diagonal the slope is -sign(x_j) sign(x_k) times
    # a positive number, or zero beside a zero of x, and on it positive. An
    # infinity there is its terms' sign, and NaN where they disagree or one
    # meets a zero.
    inf, nan = math.inf, math.nan
    x = [[1.0, 2.0, 2.0], [1.0, 0.0, 2.0], [1.0, 2.0, 2.0], [1.0, -2.0, 2.0]]
    g = [[1.0, inf, 1.0], [inf, 1.0, 1.0], [inf, inf, 1.0], [-inf, 1.0, 1.0]]
    want = [
        [-inf, inf, -inf],
        [inf, nan, -inf],
        [nan, nan, -inf],
        [-inf, -inf, inf],
    ]
    t = mt.Tensor(np.array(x), dtype=dtype, requires_grad=True)
    mt.normalize(t, p, 1).backward(mt.Tensor(np.array(g), dtype=dtype))
    np.testing.assert_array_equal(t.grad.numpy(), np.array(want, dtype=dtype))
    # `renorm` the same, times its cap, where it scales a sub-tensor, and the
    # incoming gradient as it is where it does not.
    t = mt.Tensor(np.array(x + [[0.1, 0.0, 0.1]]), dtype=dtype, requires_grad=True)
    g = g + [[inf, 1.0, -1.0]]
    mt.renorm(t, p, 0, 1.0).backward(mt.Tensor(np.array(g), dtype=dtype))
    want = want + [[inf, 1.0, -1.0]]
    np.testing.assert_array_equal(t.grad.numpy(), np.array(want, dtype=dtype))


@pytest.mark.parametrize("p", [2.0, 1.0])
def test_renorm_with_a_cap_past_the_middle_of_float64(p):
    # The cap over a scaled norm left the range on the way, where the
    # gradient does not.
    rng = np.random.default_rng(89)
    subs = rng.standard_normal((4, 6)) * 1e300
    upstream = rng.standard_normal((4, 6)) * 1e100
    t = mt.Tensor(subs, dtype="float64", requires_grad=True)
    mt.renorm(t, p, 0, 1e300).backward(mt.Tensor(upstream, dtype="float64"))
    got = t.grad.numpy()
    with decimal.localcontext(prec=80):
        for r in range(4):
            want, norm = _moving(subs[r], upstream[r], p, Decimal(1e300))
            assert norm > Decimal(1e300)
            for k, w in enumerate(want):
                assert _close(got[r, k], w, "float64"), (r, k, got[r, k], float(w))


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("p", [2.0, 1.0])
@pytest.mark.parametrize("axis", [0, 1])
def test_renorm(axis, p, dtype):
    rng = np.random.default_rng(79)
    subs = rng.standard_normal((12, 10))
    # Every other sub-tensor shrunk under the cap, which passes its gradient
    # on as it is.
    subs[::2] *= 0.01
    upstream = _along(subs, p, 1e-5, rng)
    subs, upstream = subs.astype(dtype), upstream.astype(dtype)
    maxnorm = 0.5 if p == 2 else 2.0
    x, g = (subs, upstream) if axis == 0 else (subs.T.copy(), upstream.T.copy())
    t = mt.Tensor(x, dtype=dtype, requires_grad=True)
    mt.renorm(t, p, axis, maxnorm).backward(mt.Tensor(g, dtype=dtype))
    got = t.grad.numpy().astype(np.float64)
    got = got if axis == 0 else got.T
    cap = Decimal(float(np.array(maxnorm, dtype=dtype)))
    with decimal.localcontext(prec=80):
        for r in range(subs.shape[0]):
            want, norm = _moving(subs[r], upstream[r], p, cap)
            if norm <= cap:
                want = [Decimal(float(v)) for v in upstream[r]]
            for k, w in enumerate(want):
                assert _close(got[r, k], w, dtype), (r, k, got[r, k], float(w))
