# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""`renorm` scales a sub-tensor over `maxnorm` to `x * maxnorm / ||x||`, rounded once.

It took the norm, rounded, divided `maxnorm` by it, rounding again, and
multiplied each element by that, a third rounding: a third of the scaled
elements of random rows were an ulp or more out, and the rounded norm decided
which rows were over the limit. Its floor of `1e-30` under the norm also made
a limit below `1e-30` scale a tiny row by `maxnorm / 1e-30` rather than by
`maxnorm` over its norm, and an infinite row under an infinite limit came back
NaN rather than as it was. For `p` 1 and 2 each scaled element is now the float
nearest its exact value, a row is scaled exactly when its exact norm exceeds
the limit, and the gradient follows that decision.
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
    """The float nearest the square root of `square`."""

    if square == 0:
        return 0.0
    # The root's leading eighty bits or more, and whether anything is left.
    bits = (square.numerator.bit_length() - square.denominator.bit_length()) // 2
    shift = 80 - bits
    scaled = square * Fraction(4) ** shift
    whole = math.isqrt(scaled.numerator // scaled.denominator)
    if Fraction(whole) ** 2 == scaled:
        root = Fraction(whole)
    else:
        # Strictly between `whole` and the next, it rounds as the point
        # halfway between them does: no halfway point of the dtype lies
        # between two whole numbers this far down.
        root = Fraction(2 * whole + 1, 2)
    return _rounded(root / Fraction(2) ** shift, dtype)


def _limit(maxnorm: float, dtype: str) -> Fraction:
    """`maxnorm` as the dtype holds it."""

    return Fraction(float(np.float32(maxnorm)) if dtype == "float32" else maxnorm)


def _renormed_row(row, p, limit: Fraction, dtype: str):
    """The row as renorm leaves it, and whether it was scaled."""

    values = [Fraction(float(v)) for v in row]
    if p == 1:
        norm = sum(abs(v) for v in values)
        if norm <= limit:
            return [float(v) for v in row], False
        return [_rounded(v * limit / norm, dtype) for v in values], True
    squares = sum(v * v for v in values)
    if squares <= limit * limit:
        return [float(v) for v in row], False
    scaled = []
    for v in values:
        magnitude = _root(v * v * limit * limit / squares, dtype)
        scaled.append(-magnitude if v < 0 else magnitude)
    return scaled, True


def _renormed(array, p, dim, maxnorm, dtype):
    """What renorm leaves, from the definition, each sub-tensor along `dim`."""

    limit = _limit(maxnorm, dtype)
    moved = np.moveaxis(np.asarray(array, dtype=np.float64), dim, 0)
    rows = moved.reshape(moved.shape[0], -1)
    out = np.array(
        [_renormed_row(row, p, limit, dtype)[0] for row in rows], dtype=np.float64
    )
    return np.moveaxis(out.reshape(moved.shape), 0, dim)


def _draw(rng, shape, dtype):
    values = rng.standard_normal(shape) * 10.0 ** rng.uniform(-2, 2, shape)
    return values.astype(dtype)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("p", [1, 2])
@pytest.mark.parametrize(
    "shape, dim",
    [((40, 7), 0), ((6, 30), 1), ((3, 8, 5), 1), ((3, 4, 6), 2), ((9,), 0)],
)
def test_every_scaled_element_is_rounded_once(dtype, p, shape, dim):
    rng = np.random.default_rng(1051 + 7 * p + len(shape) + dim)
    array = _draw(rng, shape, dtype)
    # A limit half the rows' norms are over.
    norms = np.linalg.norm(
        np.moveaxis(array.astype(np.float64), dim, 0).reshape(shape[dim], -1),
        ord=p,
        axis=1,
    )
    maxnorm = float(np.median(norms))
    got = mt.renorm(mt.Tensor(array, dtype=dtype), float(p), dim, maxnorm)
    assert str(got.dtype) == dtype
    want = _renormed(array, p, dim, maxnorm, dtype)
    assert got.tolist() == want.tolist()


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("p", [1, 2])
def test_long_rows(dtype, p):
    rng = np.random.default_rng(1061 + p)
    array = _draw(rng, (3, 3000), dtype)
    got = mt.renorm(mt.Tensor(array, dtype=dtype), float(p), 0, 2.5)
    assert got.tolist() == _renormed(array, p, 0, 2.5, dtype).tolist()


def test_a_row_at_the_limit_is_left_as_it_is():
    rows = mt.Tensor([[3.0, 4.0], [1.0, -2.0]], dtype="float64")
    assert mt.renorm(rows, 2.0, 0, 5.0).tolist() == [[3.0, 4.0], [1.0, -2.0]]
    assert mt.renorm(rows, 1.0, 0, 3.0).tolist()[1] == [1.0, -2.0]
    # Just under, each element the float nearest its exact value.
    limit = 5.0 * (1 - 2.0**-52)
    got = mt.renorm(rows, 2.0, 0, limit).tolist()[0]
    assert got == _renormed_row([3.0, 4.0], 2, Fraction(limit), "float64")[0]
    assert got != [3.0, 4.0]


def test_integer_rows_widen():
    got = mt.renorm(mt.Tensor(np.array([[3, 4], [0, 1]], dtype=np.int64)), 2.0, 0, 1.0)
    assert str(got.dtype) == "float64"
    assert got.tolist() == [[0.6, 0.8], [0.0, 1.0]]
    narrow = mt.renorm(mt.Tensor(np.array([[3, 4]], dtype=np.int32)), 2.0, 0, 1.0)
    assert str(narrow.dtype) == "float32"
    assert narrow.tolist() == [[_float32(Fraction(3, 5)), _float32(Fraction(4, 5))]]


def test_limits_far_from_one():
    # A limit below the old floor under the norm, and rows past it.
    tiny = mt.renorm(
        mt.Tensor([[1e-35, 0.0], [3e-41, 4e-41]], dtype="float64"), 2.0, 0, 1e-40
    )
    assert tiny.tolist() == [
        [1e-40, 0.0],
        _renormed_row([3e-41, 4e-41], 2, Fraction(1e-40), "float64")[0],
    ]
    huge = mt.renorm(mt.Tensor([[1e308, -1e308]], dtype="float64"), 2.0, 0, 1e300)
    assert huge.tolist() == [
        _renormed_row([1e308, -1e308], 2, Fraction(1e300), "float64")[0]
    ]
    taxicab = mt.renorm(
        mt.Tensor([[1e308, 1e308, 1e308]], dtype="float64"), 1.0, 0, 1e-300
    )
    want = _rounded(Fraction(1e-300) / 3, "float64")
    assert taxicab.tolist() == [[want, want, want]]


def test_special_values():
    nan, inf = math.nan, math.inf
    rows = mt.Tensor(
        [[nan, 1e-10], [inf, 1.0], [3.0, 4.0], [0.0, -0.0]], dtype="float64"
    )
    got = mt.renorm(rows, 2.0, 0, 1.0).tolist()
    assert all(math.isnan(v) for v in got[0])
    assert math.isnan(got[1][0]) and got[1][1] == 0.0
    assert got[2] == [_root(Fraction(9, 25), "float64"), 0.8]
    assert got[3] == [0.0, -0.0] and math.copysign(1.0, got[3][1]) < 0
    # Nothing is over an infinite limit, an infinite row included.
    unlimited = mt.renorm(rows, 2.0, 0, inf).tolist()
    assert all(math.isnan(v) for v in unlimited[0])
    assert unlimited[1:] == [[inf, 1.0], [3.0, 4.0], [0.0, -0.0]]
    # A limit of zero leaves zero of every row that is not already.
    nothing = mt.renorm(
        mt.Tensor([[-3.0, 4.0], [0.0, -0.0]], dtype="float64"), 1.0, 0, 0.0
    ).tolist()
    assert nothing == [[-0.0, 0.0], [0.0, -0.0]]
    assert [math.copysign(1.0, v) for v in nothing[0] + nothing[1]] == [-1, 1, 1, -1]


def test_settings_it_refuses():
    for limit in (-1.0, math.nan):
        with pytest.raises(ValueError, match="non-negative maxnorm"):
            mt.renorm(mt.Tensor([[1.0]], dtype="float64"), 2.0, 0, limit)
    with pytest.raises(ValueError, match="floating point"):
        mt.renorm(mt.Tensor([[True, False]], dtype="bool"), 2.0, 0, 1.0)


def test_other_norms_compare_the_norm_as_computed():
    rows = mt.Tensor([[3.0, 4.0], [0.1, 0.2], [math.inf, 1.0]], dtype="float64")
    got = mt.renorm(rows, math.inf, 0, 1.0).tolist()
    assert got[0] == [0.75, 1.0] and got[1] == [0.1, 0.2]
    assert math.isnan(got[2][0]) and got[2][1] == 0.0
    assert mt.renorm(rows, 3.0, 0, math.inf).tolist() == rows.tolist()


def test_empty_tensors():
    for shape in ((0, 3), (3, 0)):
        got = mt.renorm(mt.Tensor(np.zeros(shape)), 2.0, 0, 1.0)
        assert tuple(got.shape) == shape


def _projected(x, w, p, maxnorm):
    """The gradient of `sum(renorm(x) * w)` over a row that was scaled."""

    norm = np.sum(np.abs(x) ** p) ** (1 / p)
    unit = x / norm
    direction = unit if p == 2 else np.sign(x)
    return (w - direction * np.sum(unit * w)) * (maxnorm / norm)


@pytest.mark.parametrize("p", [1, 2])
@pytest.mark.parametrize("dim", [0, 1])
def test_gradients(p, dim):
    rng = np.random.default_rng(1069 + p + dim)
    array = rng.standard_normal((5, 6))
    weights = rng.standard_normal((5, 6))
    x = mt.Tensor(array, requires_grad=True)
    (mt.renorm(x, float(p), dim, 1.5) * mt.Tensor(weights)).sum().backward()
    moved_x, moved_w = np.moveaxis(array, dim, 0), np.moveaxis(weights, dim, 0)
    want = np.empty_like(moved_x)
    for k, (row, w) in enumerate(zip(moved_x, moved_w)):
        scaled = _renormed_row(row, p, Fraction(1.5), "float64")[1]
        want[k] = _projected(row, w, p, 1.5) if scaled else w
    np.testing.assert_allclose(
        x.grad.numpy(), np.moveaxis(want, 0, dim), rtol=1e-13, atol=1e-15
    )


def test_the_gradient_follows_the_exact_decision():
    # Norms of `5 + 2^-62` and of `5` exactly: the first rounds to the limit
    # but is over it, and is scaled; the second is not.
    x = mt.Tensor(
        [[3.0, 4.0, 2.0**-30], [3.0, 4.0, 0.0]], dtype="float64", requires_grad=True
    )
    w = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    (mt.renorm(x, 2.0, 0, 5.0) * mt.Tensor(w)).sum().backward()
    grad = x.grad.numpy()
    np.testing.assert_allclose(
        grad[0], _projected(np.array([3.0, 4.0, 2.0**-30]), w[0], 2, 5.0), rtol=1e-14
    )
    assert grad[1].tolist() == [1.0, 0.0, 0.0]


def test_normalize_gradient_follows_the_exact_floor():
    # A norm of `1 - 0.72 * 2^-54`, which rounds to the floor of 1 but is
    # under it: each element was taken over the floor, and the gradient is
    # the floor's.
    x = mt.Tensor([1 - 2.0**-53, 1.6 * 2.0**-27], dtype="float64", requires_grad=True)
    y = mt.normalize(x, 2.0, 0, 1.0)
    assert y.tolist() == [1 - 2.0**-53, 1.6 * 2.0**-27]
    (y * mt.Tensor([1.0, 0.0], dtype="float64")).sum().backward()
    assert x.grad.tolist() == [1.0, 0.0]


def test_normalize_over_an_empty_axis():
    # Three slices of nothing: three flags, and nothing to divide.
    x = mt.Tensor(np.zeros((3, 0)), requires_grad=True)
    y = mt.normalize(x, 2.0, 1)
    assert tuple(y.shape) == (3, 0)
    y.sum().backward()
    assert tuple(x.grad.shape) == (3, 0)
