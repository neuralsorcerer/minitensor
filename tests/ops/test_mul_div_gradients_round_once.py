# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""The gradients of `*` and `/` against exact values.

A factor broadcast against the other gathers, at each of its elements, the
dot of the incoming gradient with the other factor over the positions it was
read at; a divisor, minus that dot with the numerator over its square, and
elementwise `-g x / y^2`. Each term was rounded and then added -- a broadcast
weight's gradient was up to 8.4 ulps out in float32 and 3.7 in float64 over
64 rows, and the divisor's own up to 1.5 -- and each is now the float nearest
its exact value.
"""

from __future__ import annotations

import math
from fractions import Fraction

import numpy as np
import pytest

import minitensor as mt

_FORMATS = {"float32": (24, -149, 128), "float64": (53, -1074, 1024)}


def _nearest(value: Fraction, dtype: str) -> float:
    """The float of `dtype` nearest `value`, ties to even."""

    if value == 0:
        return 0.0
    digits, least, top = _FORMATS[dtype]
    sign = -1.0 if value < 0 else 1.0
    magnitude = abs(value)
    exponent = magnitude.numerator.bit_length() - magnitude.denominator.bit_length()
    if Fraction(2) ** exponent > magnitude:
        exponent -= 1
    unit = Fraction(2) ** max(exponent - digits + 1, least)
    rounded = round(magnitude / unit) * unit
    return sign * (math.inf if rounded >= Fraction(2) ** top else float(rounded))


def _exact(values) -> list:
    return [Fraction(float(v)) for v in np.ravel(values)]


def _grads(op, x, y, g, dtype, need=(True, True)):
    xi = mt.Tensor(x, dtype=dtype, requires_grad=need[0])
    yi = mt.Tensor(y, dtype=dtype, requires_grad=need[1])
    op(xi, yi).backward(mt.Tensor(g, dtype=dtype))
    return [t.grad.numpy() if t.grad is not None else None for t in (xi, yi)]


def _gathered(x_shape, w_shape):
    """For each element of an operand of `w_shape` broadcast against one of
    `x_shape`, the flat positions of the result it was read at."""

    out = np.broadcast_shapes(x_shape, w_shape)
    index = np.broadcast_to(np.arange(int(np.prod(w_shape))).reshape(w_shape), out)
    return [np.flatnonzero(index == k) for k in range(int(np.prod(w_shape)))]


SHAPES = [(5,), (1, 5), (64, 1), (), (1, 1)]


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("w_shape", SHAPES)
def test_a_broadcast_factor_gathers_its_dot(dtype, w_shape):
    rng = np.random.default_rng(7)
    x = rng.standard_normal((64, 5)).astype(dtype)
    w = rng.uniform(0.5, 2.0, w_shape).astype(dtype)
    g = rng.standard_normal((64, 5)).astype(dtype)
    gx, gw = _grads(lambda a, b: a * b, x, w, g, dtype)
    gf, xf, wf = _exact(g), _exact(x), _exact(np.broadcast_to(w, x.shape))
    for k, positions in enumerate(_gathered(x.shape, w_shape)):
        want = sum(gf[i] * xf[i] for i in positions)
        assert np.ravel(gw)[k] == _nearest(want, dtype), k
    for i in range(x.size):
        assert np.ravel(gx)[i] == _nearest(gf[i] * wf[i], dtype), i


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("w_shape", SHAPES)
def test_a_broadcast_divisor_gathers_its_dot_over_its_square(dtype, w_shape):
    rng = np.random.default_rng(11)
    x = rng.standard_normal((64, 5)).astype(dtype)
    w = (rng.uniform(0.5, 2.0, w_shape) * rng.choice([-1.0, 1.0], w_shape)).astype(
        dtype
    )
    g = rng.standard_normal((64, 5)).astype(dtype)
    gx, gw = _grads(lambda a, b: a / b, x, w, g, dtype)
    gf, xf = _exact(g), _exact(x)
    wf = _exact(np.broadcast_to(w, x.shape))
    for k, positions in enumerate(_gathered(x.shape, w_shape)):
        want = -sum(gf[i] * xf[i] for i in positions) / wf[positions[0]] ** 2
        assert np.ravel(gw)[k] == _nearest(want, dtype), k
    for i in range(x.size):
        assert np.ravel(gx)[i] == _nearest(gf[i] / wf[i], dtype), i


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_a_divisor_elementwise(dtype):
    rng = np.random.default_rng(13)
    n = 400
    # Magnitudes across the range, so that some quotients leave it.
    spread = 30 if dtype == "float32" else 150
    x = (rng.standard_normal(n) * 10.0 ** rng.uniform(-spread, spread, n)).astype(dtype)
    y = (rng.standard_normal(n) * 10.0 ** rng.uniform(-spread, spread, n)).astype(dtype)
    y[y == 0] = 1.0
    g = (rng.standard_normal(n) * 10.0 ** rng.uniform(-spread, spread, n)).astype(dtype)
    with np.errstate(over="ignore", under="ignore"):
        _, gy = _grads(lambda a, b: a / b, x, y, g, dtype, (False, True))
    for i in range(n):
        want = (
            -Fraction(float(g[i])) * Fraction(float(x[i])) / Fraction(float(y[i])) ** 2
        )
        assert gy[i] == _nearest(want, dtype), i


def _ties(dtype):
    """Products of two floats of `dtype` that lie halfway between two of its
    floats, as `(a, b)`: their last digits make one past the format's."""

    digits = _FORMATS[dtype][0]
    half = digits // 2
    found = []
    for i in range(1, 400):
        a = 1 + Fraction(2 * i + 1, 2**half)
        for j in range(1, 40):
            b = 1 + Fraction(2 * j + 1, 2 ** (digits - half))
            product = a * b
            if Fraction(_nearest(product, dtype)) != product and (
                (product * 2 ** (digits - 1)).denominator == 2
            ):
                found.append((float(a), float(b)))
                break
        if len(found) >= 6:
            break
    return found


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_halfway_quotients_go_to_the_even_float(dtype):
    pairs = _ties(dtype)
    assert pairs
    g = np.array([a for a, _ in pairs], dtype=dtype)
    x = np.array([b for _, b in pairs], dtype=dtype)
    # Over a power of two the quotient is the product moved, a halfway point
    # still; over its square, the divisor's gradient is minus it.
    y = np.array([1.0, 2.0, 0.5, 4.0, 1.0, 0.25][: len(pairs)], dtype=dtype)
    _, gy = _grads(lambda a, b: a / b, x, y, g, dtype, (False, True))
    for i in range(len(pairs)):
        want = (
            -Fraction(float(g[i])) * Fraction(float(x[i])) / Fraction(float(y[i])) ** 2
        )
        assert gy[i] == _nearest(want, dtype), i


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_zeros_and_values_that_are_not_finite(dtype):
    inf, nan = math.inf, math.nan
    x = np.array([2.0, 2.0, -3.0, 1.0, inf, 1.0, 0.0], dtype=dtype)
    y = np.array([1.0, -1.0, 0.0, 0.0, 2.0, nan, 0.0], dtype=dtype)
    g = np.array([0.0, -0.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=dtype)
    with np.errstate(divide="ignore", invalid="ignore"):
        _, gy = _grads(lambda a, b: a / b, x, y, g, dtype, (False, True))
        want = -(g.astype(np.float64) * x) / (y.astype(np.float64) * y)
    for got, expected in zip(gy.tolist(), want.tolist()):
        if math.isnan(expected):
            assert math.isnan(got)
        else:
            assert got == expected and math.copysign(1, got) == math.copysign(
                1, expected
            )


def test_operands_of_two_dtypes_still_differentiate():
    rng = np.random.default_rng(17)
    x = mt.Tensor(rng.standard_normal((8, 3)), dtype="float32", requires_grad=True)
    w = mt.Tensor(rng.uniform(0.5, 2.0, 3), dtype="float64", requires_grad=True)
    (x / w).sum().backward()
    want = -(x.numpy().astype(np.float64) / w.numpy() ** 2).sum(axis=0)
    np.testing.assert_allclose(w.grad.numpy(), want, rtol=1e-6)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_a_factor_read_at_no_positions_has_a_zero_gradient(dtype):
    x = mt.Tensor(np.zeros((0, 4)), dtype=dtype)
    for op in (lambda a, b: a * b, lambda a, b: a / b):
        w = mt.Tensor(np.full(4, 2.0), dtype=dtype, requires_grad=True)
        op(x, w).sum().backward()
        assert w.grad.tolist() == [0.0, 0.0, 0.0, 0.0]
