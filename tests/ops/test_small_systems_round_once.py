# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""A determinant of order two or three, and each entry of an inverse or a
solution of that order, is the float nearest its exact value.

Taken by elimination -- each pivot, product and update rounded -- the
determinant of a matrix of whole numbers came out -0.9999999999999998 where it
is -1, and nearly half the entries of the inverses of small integer matrices
were not the float nearest the exact fraction. By Cramer's rule each is a
determinant or a quotient of two, sums of a few products of the entries, and
those are taken exactly and rounded once, in float32 as in float64. A matrix
whose exact determinant is zero is singular however elimination would have
rounded its last pivot, and `slogdet` agrees with `det` about it.
"""

from __future__ import annotations

import math
from decimal import Decimal, localcontext
from fractions import Fraction

import numpy as np
import pytest

import minitensor as mt

DTYPES = ["float32", "float64"]


def _exact_det(rows):
    n = len(rows)
    if n == 1:
        return rows[0][0]
    return sum(
        (-1) ** j
        * rows[0][j]
        * _exact_det([row[:j] + row[j + 1 :] for row in rows[1:]])
        for j in range(n)
    )


def _fractions(matrix):
    return [[Fraction(float(v)) for v in row] for row in np.asarray(matrix)]


def _nearest(value: Fraction, dtype: str):
    """The float of `dtype` nearest `value`, ties to even."""

    if value == 0:
        return 0.0
    digits = 24 if dtype == "float32" else 53
    tiny = -149 if dtype == "float32" else -1074
    magnitude = abs(value)
    exponent = magnitude.numerator.bit_length() - magnitude.denominator.bit_length()
    if Fraction(2) ** exponent > magnitude:
        exponent -= 1
    unit = Fraction(2) ** max(exponent - digits + 1, tiny)
    steps = magnitude / unit
    whole = math.floor(steps)
    if steps - whole > Fraction(1, 2) or (
        steps - whole == Fraction(1, 2) and whole % 2
    ):
        whole += 1
    rounded = float(whole * unit)
    return math.copysign(rounded, value)


def _log_magnitude(value: Fraction) -> float:
    """`log |value|` to well past float64's precision."""

    with localcontext() as context:
        context.prec = 60
        magnitude = abs(value)
        log = Decimal(magnitude.numerator).ln() - Decimal(magnitude.denominator).ln()
        return float(log)


def _same(got, want, dtype):
    got = np.asarray(got, dtype=dtype).ravel()
    want = np.asarray(want, dtype=dtype).ravel()
    view = np.int32 if dtype == "float32" else np.int64
    assert np.array_equal(got.view(view), want.view(view)), (got, want)


def _matrices(rng, n, count, dtype):
    out = []
    while len(out) < count:
        kind = len(out) % 3
        if kind == 0:
            matrix = rng.integers(-9, 10, (n, n)).astype(float)
        elif kind == 1:
            matrix = rng.standard_normal((n, n))
        else:
            matrix = rng.standard_normal((n, n)) * 2.0 ** rng.integers(-20, 20, (n, n))
        matrix = matrix.astype(dtype)
        if _exact_det(_fractions(matrix)) != 0:
            out.append(matrix)
    return np.stack(out)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("n", [2, 3])
def test_det_is_the_nearest_float(dtype, n):
    stack = _matrices(np.random.default_rng(n), n, 30, dtype)
    got = mt.det(mt.tensor(stack)).numpy()
    want = [_nearest(_exact_det(_fractions(m)), dtype) for m in stack]
    _same(got, want, dtype)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("n", [2, 3])
def test_inverse_and_solution_are_the_nearest_floats(dtype, n):
    rng = np.random.default_rng(10 + n)
    stack = _matrices(rng, n, 24, dtype)
    rhs = rng.integers(-9, 10, (24, n, 2)).astype(dtype)
    inverse = mt.inv(mt.tensor(stack)).numpy()
    solution = mt.solve(mt.tensor(stack), mt.tensor(rhs)).numpy()
    for k, matrix in enumerate(stack):
        rows = _fractions(matrix)
        det = _exact_det(rows)
        for i in range(n):
            for j in range(n):
                # entry (i, j) of the inverse: the cofactor of (j, i) over det
                minor = [row[:i] + row[i + 1 :] for r, row in enumerate(rows) if r != j]
                cofactor = (-1) ** (i + j) * _exact_det(minor)
                _same(inverse[k, i, j], _nearest(cofactor / det, dtype), dtype)
            for c in range(2):
                replaced = [
                    row[:i] + [Fraction(float(rhs[k, r, c]))] + row[i + 1 :]
                    for r, row in enumerate(rows)
                ]
                _same(
                    solution[k, i, c],
                    _nearest(_exact_det(replaced) / det, dtype),
                    dtype,
                )


def test_whole_numbers_give_whole_numbers():
    a = mt.tensor(np.array([[2.0, 1.0, 1.0], [1.0, 3.0, 2.0], [1.0, 0.0, 0.0]]))
    assert mt.det(a).item() == -1.0
    np.testing.assert_array_equal(
        mt.inv(a).numpy(), [[0.0, 0.0, 1.0], [-2.0, 1.0, 3.0], [3.0, -1.0, -5.0]]
    )
    np.testing.assert_array_equal(
        mt.inv(mt.tensor(np.array([[1.0, 2.0], [3.0, 4.0]]))).numpy(),
        [[-2.0, 1.0], [1.5, -0.5]],
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_an_exactly_singular_matrix_is_singular(dtype):
    # Elimination leaves this one a last pivot of 6.7e-16 in float64.
    singular = mt.tensor(np.arange(1.0, 10.0).reshape(3, 3).astype(dtype))
    assert mt.det(singular).item() == 0.0
    sign, log = mt.slogdet(singular)
    assert sign.item() == 0.0 and log.item() == -math.inf
    with pytest.raises(Exception, match="singular"):
        mt.inv(singular)
    with pytest.raises(Exception, match="singular"):
        mt.solve(singular, mt.tensor(np.ones(3, dtype=dtype)))


def test_slogdet_takes_the_exact_determinant():
    rng = np.random.default_rng(4)
    for n in (2, 3):
        stack = _matrices(rng, n, 30, "float64")
        sign, log = mt.slogdet(mt.tensor(stack))
        for k, matrix in enumerate(stack):
            det = _exact_det(_fractions(matrix))
            assert sign.numpy()[k] == (1.0 if det > 0 else -1.0)
            want = _log_magnitude(det)
            assert abs(log.numpy()[k] - want) <= 4e-16 * abs(want), (log, want)
    # A determinant near one keeps its logarithm's digits: 1 + 2^-40.
    near_one = mt.tensor(np.array([[1.0 + 2.0**-40, 0.0], [0.0, 1.0]]))
    want = math.log1p(2.0**-40)
    assert abs(mt.slogdet(near_one)[1].item() - want) <= 2.5e-16 * want
    # One past float64's range, from the entries scaled.
    huge = mt.tensor(np.array([[1e200, 3e199], [2e199, 1e200]]))
    assert math.isinf(mt.det(huge).item())
    assert (
        abs(mt.slogdet(huge)[1].item() - (math.log(0.94) + 400 * math.log(10))) < 1e-12
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_the_det_gradient_is_the_scaled_cofactor_matrix(dtype):
    rng = np.random.default_rng(6)
    for n in (2, 3):
        matrix = rng.standard_normal((n, n)).astype(dtype)
        x = mt.tensor(matrix, requires_grad=True)
        (mt.det(x) * 3.0).backward()
        rows = _fractions(matrix)
        for i in range(n):
            for j in range(n):
                minor = [row[:j] + row[j + 1 :] for r, row in enumerate(rows) if r != i]
                cofactor = (-1) ** (i + j) * _exact_det(minor)
                _same(x.grad.numpy()[i, j], _nearest(3 * cofactor, dtype), dtype)


def test_values_that_are_not_finite_take_elimination():
    a = mt.tensor(np.array([[math.inf, 0.0], [0.0, 1.0]]))
    x = mt.solve(a, mt.tensor(np.array([1.0, 1.0]))).numpy()
    np.testing.assert_array_equal(x, [0.0, 1.0])
    assert math.isnan(mt.det(mt.tensor(np.array([[math.nan, 1.0], [1.0, 1.0]]))).item())
