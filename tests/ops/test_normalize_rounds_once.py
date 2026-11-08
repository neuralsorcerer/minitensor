# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""`normalize` with `p` 1 or 2 is the float nearest the exact quotient.

Each slice was divided by its norm as computed -- a sum of rounded squares or
magnitudes, a rounded root, a rounded division -- and a fifth to a third of
the elements came out an ulp or more from the exact quotient. Where the
squares or the sum left the float range the norm was infinite or zero although
the slice was not: `[1e308, 1e308]` normalized to `[0, 0]` under the 1-norm,
and two of the smallest subnormals to `[1, 1]` under the 2-norm. Each element
is now the exact element over the exact norm -- or over `eps` as the dtype
holds it, where the norm is smaller, decided exactly -- rounded once.
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


def _root_quotient(magnitude: Fraction, squares: Fraction, dtype: str) -> float:
    """The float nearest `magnitude / sqrt(squares)`, at most one.

    The root is bracketed between two neighbouring multiples of a power of two
    a little finer each time, until both ends round alike. They always come to:
    a quotient of a float by the root of a sum of squares of floats is never
    halfway between two floats."""

    square = magnitude * magnitude / squares
    scale = (square.numerator.bit_length() - square.denominator.bit_length()) // 2
    extra = 64
    while True:
        bits = extra - scale
        root = math.isqrt(square.numerator * 4**bits // square.denominator)
        low = _rounded(Fraction(root, 2**bits), dtype)
        if low == _rounded(Fraction(root + 1, 2**bits), dtype):
            return low
        extra *= 2


def _expected(slice_, p: int, eps: float, dtype: str) -> list[float]:
    values = [float(value) for value in np.ravel(slice_)]
    exact = [Fraction(value) for value in values]
    floor = Fraction(float(np.float32(eps)) if dtype == "float32" else eps)
    if p == 1:
        norm = sum(abs(value) for value in exact)
        under = norm < floor
    else:
        norm = sum(value * value for value in exact)
        under = floor > 0 and norm < floor * floor
    out = []
    for value, x in zip(values, exact):
        if (norm if not under else floor) == 0:
            out.append(math.nan)
        elif x == 0:
            out.append(math.copysign(0.0, value))
        elif under:
            out.append(_rounded(x / floor, dtype))
        elif p == 1:
            out.append(_rounded(x / norm, dtype))
        else:
            out.append(math.copysign(_root_quotient(abs(x), norm, dtype), value))
    return out


def _same(got: list[float], want: list[float]) -> bool:
    """Equal value for value, NaN to NaN and zero to zero of the same sign."""

    return len(got) == len(want) and all(
        (math.isnan(a) and math.isnan(b))
        or (a == b and math.copysign(1.0, a) == math.copysign(1.0, b))
        for a, b in zip(got, want)
    )


def _draw(rng, shape, dtype: str, kind: str, p: int):
    """Values of one kind, and the `eps` to normalize them with."""

    eps = 1e-12
    if kind == "spread":
        values = rng.normal(size=shape)
    elif kind == "wide":
        values = rng.normal(size=shape) * 2.0 ** rng.integers(-20, 21, shape)
    elif kind == "near the floor":
        # Every slice along the last axis a few ulps of its norm either side
        # of `eps`, so that whether the floor applies takes the exact norm.
        values = rng.normal(size=shape)
        eps = 0.5
        norms = np.linalg.norm(values, p, axis=-1, keepdims=True)
        nudge = 1 + rng.integers(-6, 7, size=norms.shape) * 2.0**-50
        values = values * (eps / norms) * nudge
    else:
        # Past the range: squares that overflow or underflow, and
        # subnormal elements, with no floor in the way.
        eps = 0.0
        if dtype == "float64":
            scale = 2.0 ** rng.choice([-1060, -1000, 600, 1000], size=shape[:-1] + (1,))
        else:
            scale = 2.0 ** rng.choice([-140, -80, 70, 100], size=shape[:-1] + (1,))
        values = rng.normal(size=shape) * scale
    return values.astype(dtype), eps


KINDS = ["spread", "wide", "near the floor", "past the range"]


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("p", [1, 2])
@pytest.mark.parametrize("kind", KINDS)
def test_rows_columns_and_middle_axes(dtype, p, kind):
    rng = np.random.default_rng(577 + p)
    values, eps = _draw(rng, (23, 37), dtype, kind, p)
    tensor = mt.Tensor(values, dtype=dtype)
    got = mt.normalize(tensor, p, 1, eps).tolist()
    for row, out in zip(values, got):
        assert _same(out, _expected(row, p, eps, dtype))
    values, eps = _draw(rng, (37, 23), dtype, kind, p)
    got = mt.normalize(mt.Tensor(values.T.copy(), dtype=dtype), p, 0, eps).tolist()
    for j in range(37):
        column = [got[i][j] for i in range(23)]
        assert _same(column, _expected(values[j], p, eps, dtype))
    values, eps = _draw(rng, (5, 7, 6), dtype, kind, p)
    cube = values.transpose(0, 2, 1).copy()
    got = mt.normalize(mt.Tensor(cube, dtype=dtype), p, -2, eps).tolist()
    for i in range(5):
        for k in range(7):
            along = [got[i][j][k] for j in range(6)]
            assert _same(along, _expected(cube[i, :, k], p, eps, dtype))


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("p", [1, 2])
def test_long_slices_and_many_of_them(dtype, p):
    rng = np.random.default_rng(587 + p)
    # A long row, and long columns gathered from a wide layout.
    values = rng.normal(size=5000).astype(dtype)
    got = mt.normalize(mt.Tensor(values, dtype=dtype), p, 0).tolist()
    assert _same(got, _expected(values, p, 1e-12, dtype))
    columns = rng.normal(size=(3000, 3)).astype(dtype)
    got = mt.normalize(mt.Tensor(columns, dtype=dtype), p, 0).tolist()
    for j in range(3):
        assert _same([row[j] for row in got], _expected(columns[:, j], p, 1e-12, dtype))
    # Enough slices to be split among threads: each one is what it is alone.
    values = rng.normal(size=(400, 400)).astype(dtype)
    whole = mt.normalize(mt.Tensor(values, dtype=dtype), p, 1).tolist()
    for i in (0, 133, 399):
        alone = mt.normalize(mt.Tensor(values[i : i + 1], dtype=dtype), p, 1)
        assert alone.tolist()[0] == whole[i]
        assert _same(whole[i], _expected(values[i], p, 1e-12, dtype))


def test_norms_past_the_range():
    def normalize(values, p, dtype="float64", eps=1e-12):
        return mt.normalize(mt.tensor([values], dtype=dtype), p, 1, eps).tolist()[0]

    half = math.sqrt(0.5)
    # A sum of magnitudes, and of squares, past the largest float.
    assert normalize([1e308, 1e308], 1) == [0.5, 0.5]
    assert normalize([1e308, -1e308], 2) == [half, -half]
    assert normalize([1e154, 1e154], 2) == [half, half]
    assert normalize([3e38, 3e38], 1, "float32") == [0.5, 0.5]
    assert normalize([1e30, 1e30], 2, "float32") == [_float32(Fraction(half))] * 2
    # Squares below the smallest float, with no floor to hide them.
    assert normalize([5e-324, 5e-324], 2, eps=0.0) == [half, half]
    assert normalize([1e-200, -1e-200], 2, eps=0.0) == [half, -half]
    assert normalize([1e-40, 0.0], 2, "float32", eps=0.0) == [1.0, 0.0]


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("p", [1, 2])
def test_the_floor_zeros_and_specials(dtype, p):
    def normalize(values, eps=1e-12):
        tensor = mt.tensor([values], dtype=dtype)
        return mt.normalize(tensor, p, 1, eps).tolist()[0]

    nan, inf = math.nan, math.inf
    # Below the floor a slice is divided by `eps` as the dtype holds it.
    floor = Fraction(float(np.float32(0.1)) if dtype == "float32" else 0.1)
    small = [float(np.array(v, dtype=dtype)) for v in (0.03, -0.04)]
    want = [_rounded(Fraction(v) / floor, dtype) for v in small]
    assert normalize(small, 0.1) == want
    # A zero slice keeps its zeros and their signs; with no floor it is NaN.
    got = normalize([0.0, -0.0])
    assert got == [0.0, 0.0] and math.copysign(1.0, got[1]) == -1.0
    got = normalize([-0.0, 1e-20])
    assert math.copysign(1.0, got[0]) == -1.0
    for eps in (0.0, -1.0, nan):
        assert all(math.isnan(v) for v in normalize([0.0, -0.0], eps))
    # A NaN anywhere makes the slice NaN; an infinite norm divides by itself.
    assert all(math.isnan(v) for v in normalize([nan, 1.0]))
    got = normalize([inf, 1.0])
    assert math.isnan(got[0]) and got[1] == 0.0
    assert all(math.isnan(v) for v in normalize([inf, -inf]))
    assert normalize([1.0, 1.0], inf) == [0.0, 0.0]
    # A NaN floor is no floor.
    assert _same(normalize([3.0, 4.0], nan), _expected([3.0, 4.0], p, 0.0, dtype))


def test_shapes_and_layouts():
    values = mt.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype="float64")
    got = mt.normalize(values.transpose(0, 1), 2, 1).tolist()
    for row, out in zip([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]], got):
        assert out == _expected(row, 2, 1e-12, "float64")
    assert mt.normalize(mt.zeros((0, 3), dtype="float64"), 2, 1).shape == (0, 3)
    assert mt.normalize(mt.zeros((3, 0), dtype="float64"), 1, 0).shape == (3, 0)
    assert mt.normalize(mt.tensor([3.0, 4.0], dtype="float64"), 2, 0).tolist() == [
        0.6,
        0.8,
    ]
    with pytest.raises(ValueError):
        mt.normalize(mt.tensor(3.0, dtype="float64"), 2, 0)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("p", [1, 2])
def test_gradients(dtype, p):
    rows = np.array([[1.5, -2.0, 4.0], [3e-13, -4e-13, 0.0], [0.5, 0.25, -2.0]])
    weights = np.array([[0.3, -1.0, 2.0], [1.0, 2.0, -0.5], [-1.5, 0.5, 1.0]])
    x = mt.tensor(rows.tolist(), dtype=dtype, requires_grad=True)
    y = mt.normalize(x, p, 1)
    (y * mt.tensor(weights.tolist(), dtype=dtype)).sum().backward()
    assert str(x.grad.dtype) == dtype
    rows = rows.astype(dtype).astype(np.float64)
    norms = np.linalg.norm(rows, p, axis=1, keepdims=True)
    length = np.maximum(norms, float(np.array(1e-12, dtype=dtype)))
    result = rows / length
    direction = result if p == 2 else np.sign(rows)
    along = (result * weights).sum(1, keepdims=True)
    want = np.where(
        norms > 1e-12, (weights - direction * along) / length, weights / length
    )
    tolerance = 1e-5 if dtype == "float32" else 1e-12
    assert np.allclose(x.grad.tolist(), want, rtol=tolerance, atol=tolerance)
    # With no floor, a NaN `eps` among them, the length moves with the slice.
    for eps in (0.0, math.nan):
        x = mt.tensor([[3.0, -4.0]], dtype=dtype, requires_grad=True)
        mt.normalize(x, p, 1, eps).sum().backward()
        if p == 2:
            want = [[(1 - 0.6 * -0.2) / 5, (1 + 0.8 * -0.2) / 5]]
        else:
            want = [[(1 - -1 / 7) / 7, (1 + -1 / 7) / 7]]
        assert np.allclose(x.grad.tolist(), want, rtol=tolerance, atol=tolerance)
