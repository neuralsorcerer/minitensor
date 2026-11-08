# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""`cosine_similarity` is the float nearest the exact cosine.

The cosine was formed from ordinary operations -- each operand over its own
norm, and their products summed -- every step rounded on its own, and nearly
two in three of the cosines of pairs of random normal vectors were an ulp or
more out, in float32 as in float64. Each is now the exact dot over the
exact norms, each floored at `eps` as the dtype holds it, rounded once.
"""

from __future__ import annotations

import math
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


def _rounded(value: Fraction, dtype: str) -> float:
    return _float32(value) if dtype == "float32" else float(value)


def _root(square: Fraction, dtype: str) -> float:
    """The float nearest the root of `square`.

    Exactly where `square` is the square of a fraction; otherwise the root is
    irrational, never halfway between two floats, and bracketed a little
    finer each time until both ends of the bracket round alike."""

    top, bottom = square.numerator, square.denominator
    if math.isqrt(top) ** 2 == top and math.isqrt(bottom) ** 2 == bottom:
        return _rounded(Fraction(math.isqrt(top), math.isqrt(bottom)), dtype)
    scale = (top.bit_length() - bottom.bit_length()) // 2
    extra = 64
    while True:
        bits = extra - scale
        root = math.isqrt(top * 4**bits // bottom)
        low = _rounded(Fraction(root, 2**bits), dtype)
        if low == _rounded(Fraction(root + 1, 2**bits), dtype):
            return low
        extra *= 2


def _cosine(a, b, eps: float, dtype: str) -> float:
    a = [float(value) for value in np.ravel(a)]
    b = [float(value) for value in np.ravel(b)]
    if not all(math.isfinite(value) for value in a + b):
        return math.nan
    floor = Fraction(float(np.float32(eps)) if dtype == "float32" else eps)
    dot = sum(Fraction(x) * Fraction(y) for x, y in zip(a, b))
    squares = [sum(Fraction(x) ** 2 for x in vector) for vector in (a, b)]
    lengths = [max(total, floor * floor) for total in squares]
    if lengths[0] * lengths[1] == 0:
        return math.nan
    if dot == 0:
        return 0.0
    return math.copysign(_root(dot * dot / (lengths[0] * lengths[1]), dtype), dot)


def _same(got: list[float], want: list[float]) -> bool:
    return len(got) == len(want) and all(
        (math.isnan(x) and math.isnan(y))
        or (x == y and math.copysign(1.0, x) == math.copysign(1.0, y))
        for x, y in zip(got, want)
    )


def _draw(rng, shape, dtype: str, kind: str):
    """Two operands of one kind, and the `eps` to compare them with."""

    a, b, eps = rng.normal(size=shape), rng.normal(size=shape), 1e-8
    if kind == "wide":
        a = a * 2.0 ** rng.integers(-20, 21, shape)
        b = b * 2.0 ** rng.integers(-20, 21, shape)
    elif kind == "nearly parallel":
        b = -a + rng.normal(size=shape) * 1e-5
    elif kind == "under the floor":
        a = a * 1e-10
    elif kind == "past the range":
        # Squares that overflow or underflow, and subnormal elements, with a
        # floor below them all.
        if dtype == "float64":
            a, b, eps = a * 2.0**-1060, b * 2.0**600, 5e-324
        else:
            a, b, eps = a * 2.0**-140, b * 2.0**100, 1e-45
    return a.astype(dtype), b.astype(dtype), eps


KINDS = ["spread", "wide", "nearly parallel", "under the floor", "past the range"]


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("kind", KINDS)
def test_rows_columns_and_middle_axes(dtype, kind):
    rng = np.random.default_rng(601)
    a, b, eps = _draw(rng, (40, 23), dtype, kind)
    tensors = mt.Tensor(a, dtype=dtype), mt.Tensor(b, dtype=dtype)
    got = mt.cosine_similarity(*tensors, 1, eps).tolist()
    assert _same(got, [_cosine(x, y, eps, dtype) for x, y in zip(a, b)])
    got = mt.cosine_similarity(*tensors, 0, eps).tolist()
    assert _same(got, [_cosine(a[:, j], b[:, j], eps, dtype) for j in range(23)])
    a, b, eps = _draw(rng, (4, 9, 5), dtype, kind)
    tensors = mt.Tensor(a, dtype=dtype), mt.Tensor(b, dtype=dtype)
    got = mt.cosine_similarity(*tensors, -2, eps).tolist()
    want = [
        [_cosine(a[i, :, k], b[i, :, k], eps, dtype) for k in range(5)]
        for i in range(4)
    ]
    assert all(_same(row, expected) for row, expected in zip(got, want))


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_broadcast_long_and_many(dtype):
    rng = np.random.default_rng(607)
    batch = rng.normal(size=(30, 17)).astype(dtype)
    query = rng.normal(size=(1, 17)).astype(dtype)
    got = mt.cosine_similarity(
        mt.Tensor(batch, dtype=dtype), mt.Tensor(query, dtype=dtype)
    ).tolist()
    assert _same(got, [_cosine(row, query, 1e-8, dtype) for row in batch])
    # One long pair, and enough pairs to be split among threads.
    a, b = (rng.normal(size=6000).astype(dtype) for _ in range(2))
    got = mt.cosine_similarity(mt.Tensor(a, dtype=dtype), mt.Tensor(b, dtype=dtype), 0)
    assert _same([got.item()], [_cosine(a, b, 1e-8, dtype)])
    a, b = (rng.normal(size=(3000, 40)).astype(dtype) for _ in range(2))
    whole = mt.cosine_similarity(mt.Tensor(a, dtype=dtype), mt.Tensor(b, dtype=dtype))
    whole = whole.tolist()
    for i in (0, 1777, 2999):
        assert _same([whole[i]], [_cosine(a[i], b[i], 1e-8, dtype)])


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_exact_answers_zeros_and_specials(dtype):
    def cosine(a, b, eps=1e-8):
        tensors = mt.tensor([a], dtype=dtype), mt.tensor([b], dtype=dtype)
        return mt.cosine_similarity(*tensors, 1, eps).item()

    nan, inf = math.nan, math.inf
    # A vector with itself, or turned round, and two at right angles.
    values = [0.1, -0.7, 3.3, 1e-3]
    assert cosine(values, values) == 1.0
    assert cosine(values, [-v for v in values]) == -1.0
    assert cosine([3.0, 4.0], [4.0, 3.0]) == _rounded(Fraction(24, 25), dtype)
    got = cosine([1.0, 0.0], [0.0, 1.0])
    assert got == 0.0 and math.copysign(1.0, got) == 1.0
    got = cosine([1.0, 1.0], [1.0, -1.0])
    assert got == 0.0 and math.copysign(1.0, got) == 1.0
    # A zero vector against anything, under its floor.
    assert cosine([0.0, 0.0], [1.0, 2.0]) == 0.0
    assert cosine([0.0, -0.0], [0.0, 0.0]) == 0.0
    # Two floored vectors: the dot over the floor squared.
    floor = Fraction(float(np.float32(0.5)) if dtype == "float32" else 0.5)
    got = cosine([0.25, 0.125], [0.125, -0.25], 0.5)
    assert got == _rounded(Fraction(0) / floor, dtype)
    got = cosine([0.25, 0.125], [0.25, 0.25], 0.5)
    assert got == _rounded(Fraction(3, 32) / (floor * floor), dtype)
    # A NaN or an infinity anywhere has no angle.
    assert math.isnan(cosine([nan, 1.0], [1.0, 1.0]))
    assert math.isnan(cosine([inf, 1.0], [1.0, 1.0]))
    assert math.isnan(cosine([inf, 0.0], [0.0, 1.0]))


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_gradients(dtype):
    rows = np.array([[1.5, -2.0, 4.0], [3e-10, -4e-10, 1e-10], [0.5, 0.25, -2.0]])
    other = np.array([[0.3, -1.0, 2.0]])
    weights = np.array([0.7, -1.3, 2.0])
    x = mt.tensor(rows.tolist(), dtype=dtype, requires_grad=True)
    y = mt.tensor(other.tolist(), dtype=dtype, requires_grad=True)
    cosine = mt.cosine_similarity(x, y)
    (cosine * mt.tensor(weights.tolist(), dtype=dtype)).sum().backward()
    assert str(x.grad.dtype) == dtype and x.grad.shape == (3, 3)
    assert y.grad.shape == (1, 3)

    a = rows.astype(dtype).astype(np.float64)
    b = np.broadcast_to(other.astype(dtype).astype(np.float64), a.shape)
    eps = float(np.float32(1e-8)) if dtype == "float32" else 1e-8
    norm_a = np.linalg.norm(a, axis=1, keepdims=True)
    norm_b = np.linalg.norm(b, axis=1, keepdims=True)
    length_a, length_b = np.maximum(norm_a, eps), np.maximum(norm_b, eps)
    c = (a * b).sum(1, keepdims=True) / (length_a * length_b)
    g = weights[:, None]
    moving_a = np.where(norm_a > eps, c * a / length_a**2, 0.0)
    moving_b = np.where(norm_b > eps, c * b / length_b**2, 0.0)
    want_x = g * (b / (length_a * length_b) - moving_a)
    want_y = (g * (a / (length_a * length_b) - moving_b)).sum(0, keepdims=True)
    tolerance = 1e-5 if dtype == "float32" else 1e-12
    assert np.allclose(x.grad.tolist(), want_x, rtol=tolerance, atol=tolerance)
    assert np.allclose(y.grad.tolist(), want_y, rtol=tolerance, atol=tolerance)
