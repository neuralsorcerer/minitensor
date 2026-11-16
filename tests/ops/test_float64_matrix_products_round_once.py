# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""A float64 matrix product gives each entry as the float64 nearest its exact
dot, and an affine one as the float64 nearest its exact value.

Accumulated in float64, as a matrix product conventionally is, one entry in
four of a product of ordinary random matrices thirty deep came out the
float64 nearest its exact dot, and one in twelve of one a thousand deep; and
where a product cancels -- a matrix by its inverse -- the entries that should
be nothing came out a fifth of themselves off, at the median. Each entry is
that float now, for `matmul`, `mm`, `bmm` and the `@` operator, batched or
broadcast, for operands read transposed where they lie, for the gradients a
product hands back, and -- the product scaled and an addend added before the
one rounding -- for a dense layer with a bias, `addmm` and `baddbmm`.
"""

from __future__ import annotations

import math
from fractions import Fraction

import numpy as np
import pytest

import minitensor as mt


def _float64(value: Fraction) -> float:
    """The float64 nearest `value`, ties to even; an infinity past the
    largest."""

    try:
        return float(value)
    except OverflowError:
        return math.copysign(math.inf, value)


def _whole(values) -> tuple[np.ndarray, int]:
    """Each float64 of `values` as a whole number over one power of two,
    exactly, and that power."""

    array = np.asarray(values, dtype=np.float64)
    ratios = [float(v).as_integer_ratio() for v in array.ravel()]
    scale = max(denominator for _, denominator in ratios)
    flat = [numerator * (scale // denominator) for numerator, denominator in ratios]
    return np.array(flat, dtype=object).reshape(array.shape), scale


def _exact_product(a, b) -> list:
    """The float64 nearest each entry of `a @ b`, batched as `matmul` is."""

    (left, left_scale), (right, right_scale) = _whole(a), _whole(b)
    total = np.matmul(left, right)
    rounded = np.vectorize(
        lambda v: _float64(Fraction(v, left_scale * right_scale)), otypes=[float]
    )
    return rounded(total).tolist()


def _exact_affine(base, left, right, beta, alpha) -> list:
    """The float64 nearest each entry of `beta * base + alpha * left @ right`."""

    left, right = np.asarray(left, np.float64), np.asarray(right, np.float64)
    out = np.broadcast_to(np.asarray(base), left.shape[:-1] + right.shape[-1:])
    (lefts, left_scale), (rights, right_scale) = _whole(left), _whole(right)
    total = np.matmul(lefts, rights)
    scale = Fraction(alpha) / (left_scale * right_scale)
    exact = np.vectorize(lambda v: Fraction(float(v)), otypes=[object])
    scaled = total * scale + exact(out) * Fraction(beta)
    return np.vectorize(_float64, otypes=[float])(scaled).tolist()


def _draw(rng, shape):
    return rng.normal(size=shape) * 10.0 ** rng.uniform(-2, 2, shape)


def _tensor(values, **options):
    return mt.Tensor(np.asarray(values, dtype=np.float64), dtype="float64", **options)


# A few entries, which are their dots; a product cut in one run, in several,
# in two bands of rows and in two of columns; and dots of two terms and one.
@pytest.mark.parametrize(
    "m, k, n",
    [
        (3, 5, 4),
        (7, 300, 5),
        (31, 64, 33),
        (20, 700, 24),
        (140, 70, 30),
        (9, 40, 300),
        (40, 2, 50),
        (48, 1, 48),
    ],
)
def test_matrix_products(m, k, n):
    rng = np.random.default_rng(1009 + k)
    a, b = _draw(rng, (m, k)), _draw(rng, (k, n))
    want = _exact_product(a, b)
    left, right = _tensor(a), _tensor(b)
    for got in (mt.matmul(left, right), mt.mm(left, right), left @ right):
        assert str(got.dtype) == "float64"
        assert got.tolist() == want


def test_batched_and_broadcast_products():
    rng = np.random.default_rng(1013)
    a, b = _draw(rng, (3, 14, 60)), _draw(rng, (3, 60, 15))
    want = _exact_product(a, b)
    assert mt.bmm(_tensor(a), _tensor(b)).tolist() == want
    assert mt.matmul(_tensor(a), _tensor(b)).tolist() == want
    # One matrix against a stack, and a stack broadcast against another.
    c = _draw(rng, (60, 16))
    assert mt.matmul(_tensor(a), _tensor(c)).tolist() == _exact_product(a, c)
    d = _draw(rng, (1, 14, 60))
    assert mt.matmul(_tensor(d), _tensor(b)).tolist() == _exact_product(d, b)


def test_operands_read_transposed():
    rng = np.random.default_rng(1019)
    a, b = _draw(rng, (90, 26)), _draw(rng, (27, 90))
    want = _exact_product(a.T, b.T)
    got = mt.matmul(_tensor(a).T, _tensor(b).T)
    assert got.tolist() == want


def test_gradients_of_a_product():
    rng = np.random.default_rng(1021)
    a, b, weights = _draw(rng, (26, 80)), _draw(rng, (80, 25)), _draw(rng, (26, 25))
    left = _tensor(a, requires_grad=True)
    right = _tensor(b, requires_grad=True)
    (mt.matmul(left, right) * _tensor(weights)).sum().backward()
    # Each reads a forward operand across its storage.
    assert left.grad.tolist() == _exact_product(weights, b.T)
    assert right.grad.tolist() == _exact_product(a.T, weights)


def test_a_dense_layer_and_its_gradients():
    rng = np.random.default_rng(1031)
    x, w = _draw(rng, (2, 15, 70)), _draw(rng, (18, 70))
    got = mt.nn.dense_layer(_tensor(x), _tensor(w))
    assert got.tolist() == _exact_product(x, w.T)

    x, w = _draw(rng, (29, 40)), _draw(rng, (23, 40))
    inputs = _tensor(x, requires_grad=True)
    weight = _tensor(w, requires_grad=True)
    upstream = _draw(rng, (29, 23))
    (mt.nn.dense_layer(inputs, weight) * _tensor(upstream)).sum().backward()
    assert inputs.grad.tolist() == _exact_product(upstream, w)
    assert weight.grad.tolist() == _exact_product(upstream.T, x)


def test_a_dense_layer_with_a_bias():
    rng = np.random.default_rng(1033)
    x, w, b = _draw(rng, (3, 14, 50)), _draw(rng, (17, 50)), _draw(rng, 17)
    got = mt.nn.dense_layer(_tensor(x), _tensor(w), _tensor(b))
    assert got.tolist() == _exact_affine(b, x, w.T, 1, 1)
    # A single output feature is a column of dots, the bias added to each.
    one = mt.nn.dense_layer(_tensor(x), _tensor(w[:1]), _tensor(b[:1]))
    assert one.tolist() == _exact_affine(b[:1], x, w[:1].T, 1, 1)
    # The gradient of the bias is each column of the incoming one summed.
    upstream = _draw(rng, (3, 14, 17))
    bias = _tensor(b, requires_grad=True)
    inputs = _tensor(x, requires_grad=True)
    (mt.nn.dense_layer(inputs, _tensor(w), bias) * _tensor(upstream)).sum().backward()
    flat = upstream.reshape(-1, 17)
    assert bias.grad.tolist() == _exact_product(np.ones((1, 42)), flat)[0]
    assert inputs.grad.tolist() == _exact_product(upstream, w)


def test_whole_numbers_and_zeros_are_exact():
    # Whole numbers have no part below their first pieces, so nothing of
    # their products is rounded before the entry is: sums past 2^53 land on
    # halfway points, which go to the even float64.
    rng = np.random.default_rng(1039)
    a = rng.integers(-(2**20), 2**20, (40, 50)).astype(np.float64)
    b = rng.integers(-(2**31), 2**31, (50, 45)).astype(np.float64)
    a[3] = 0.0
    a[5] = -0.0
    b[:, 7] = 0.0
    got = mt.matmul(_tensor(a), _tensor(b)).tolist()
    assert got == _exact_product(a, b)
    # A zero entry is a positive zero, as a dot's is.
    assert all(math.copysign(1.0, v) > 0 for v in got[3] + got[5])
    assert all(math.copysign(1.0, row[7]) > 0 for row in got)

    ones = np.ones((36, 36))
    signs = np.where(np.arange(36) % 2 == 0, 3.0, -3.0) * np.ones((36, 1))
    cancelled = mt.matmul(_tensor(signs), _tensor(ones)).tolist()
    assert all(v == 0.0 and math.copysign(1.0, v) > 0 for row in cancelled for v in row)


def test_entries_whose_terms_are_all_zeros():
    # The product of two lower triangles: above the diagonal every term of
    # an entry is a zero, and the entry a positive zero with nothing to
    # round. The triangle itself is rounded once as any product is.
    rng = np.random.default_rng(1043)
    lower = np.tril(_draw(rng, (60, 60)))
    got = mt.matmul(_tensor(lower), _tensor(lower)).tolist()
    assert got == _exact_product(lower, lower)
    assert all(
        got[i][j] == 0.0 and math.copysign(1.0, got[i][j]) > 0
        for i in range(60)
        for j in range(i + 1, 60)
    )


def test_cancellation_far_ranges_and_special_values():
    big, tiny = 2.0**600, 2.0**-600
    a = np.array(
        [
            [big, 1.0, -big, 2.0**-30],
            [3.0, -3.0, 2.0**-100, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [7.0, tiny, 1.0, 5e-324],
        ]
    )
    b = np.array(
        [[1.0, 2.0, 1.0], [1.0, 2.0, 3.0], [1.0, 2.0, 1.0], [1.0, -4.0, 2.0**40]]
    )
    got = mt.matmul(_tensor(a), _tensor(b)).tolist()
    assert got == _exact_product(a, b)
    assert all(math.copysign(1.0, v) > 0 for v in got[2])

    special = np.array([[math.inf, 1.0], [math.nan, 1.0], [math.inf, -1.0]])
    columns = np.array([[1.0, 0.0, -1.0], [2.0, 1.0, 0.0]])
    got = mt.matmul(_tensor(special), _tensor(columns)).tolist()
    assert got[0][0] == math.inf and math.isnan(got[0][1]) and got[0][2] == -math.inf
    assert all(math.isnan(v) for v in got[1])
    assert got[2][0] == math.inf and math.isnan(got[2][1]) and got[2][2] == -math.inf


def test_rows_and_columns_out_of_range_in_a_product_that_is_cut():
    # Rows and columns past where a product's pieces are cut, a row that
    # overflows, and rows holding an infinity and a NaN, among ordinary ones:
    # every entry is `dot` of its row and column.
    rng = np.random.default_rng(1049)
    a, b = _draw(rng, (40, 64)), _draw(rng, (64, 44))
    a[1] *= 2.0**700
    a[2] *= 2.0**-700
    a[3, 5] = math.inf
    a[4, 6] = math.nan
    a[5] = 1e300
    a[6, :] = 0.0
    a[6, 9] = 5e-324
    b[:, 1] *= 2.0**-650
    b[:, 2] *= 2.0**650
    b[:, 3] = 1e300
    b[9, 4] = -math.inf
    got = mt.matmul(_tensor(a), _tensor(b)).tolist()
    for i in range(40):
        row = _tensor(a[i])
        for j in range(44):
            want = mt.dot(row, _tensor(np.ascontiguousarray(b[:, j]))).item()
            assert got[i][j] == want or (math.isnan(got[i][j]) and math.isnan(want))
    assert got[5][3] == math.inf
    assert all(math.isnan(v) or abs(v) == math.inf for v in got[3])
    assert all(math.isnan(v) for v in got[4])


def test_a_product_that_cancels_throughout():
    # A matrix by its inverse: the off-diagonal entries cancel to a part in
    # 2^50 or so of their terms, far below what the pieces' bound settles,
    # and each is still the float64 nearest its exact dot.
    rng = np.random.default_rng(1051)
    a = rng.normal(size=(40, 40))
    inverse = np.linalg.inv(a)
    assert mt.matmul(_tensor(a), _tensor(inverse)).tolist() == _exact_product(
        a, inverse
    )


def test_larger_products_agree_with_their_dots():
    # Large enough to cross the pool, in several bands of rows and of
    # columns; entries checked against `dot` of their row and column.
    rng = np.random.default_rng(1061)
    a, b = _draw(rng, (270, 257)), _draw(rng, (257, 530))
    got = mt.matmul(_tensor(a), _tensor(b)).tolist()
    for i in range(0, 270, 27):
        row = _tensor(a[i])
        for j in range(0, 530, 23):
            assert (
                got[i][j] == mt.dot(row, _tensor(np.ascontiguousarray(b[:, j]))).item()
            )


@pytest.mark.parametrize(
    "beta, alpha", [(1, 1), (0.5, -1.5), (3.0, 0.0), (0.0, 2.0), (0.1, 1e-3)]
)
def test_addmm_and_baddbmm(beta, alpha):
    rng = np.random.default_rng(1063)
    a, b = _draw(rng, (26, 45)), _draw(rng, (45, 25))
    for base in (_draw(rng, (26, 25)), _draw(rng, 25), _draw(rng, (26, 1))):
        got = mt.addmm(_tensor(base), _tensor(a), _tensor(b), beta=beta, alpha=alpha)
        assert str(got.dtype) == "float64"
        assert got.tolist() == _exact_affine(base, a, b, beta, alpha), base.shape
    left, right = _draw(rng, (3, 4, 30)), _draw(rng, (3, 30, 2))
    for base in (_draw(rng, (3, 4, 2)), _draw(rng, (3, 1, 2)), _draw(rng, (4, 2))):
        got = mt.baddbmm(
            _tensor(base), _tensor(left), _tensor(right), beta=beta, alpha=alpha
        )
        assert got.tolist() == _exact_affine(base, left, right, beta, alpha), base.shape


def test_gradients_of_addmm():
    rng = np.random.default_rng(1069)
    a, b, base = _draw(rng, (26, 35)), _draw(rng, (35, 24)), _draw(rng, 24)
    upstream = _draw(rng, (26, 24))
    left = _tensor(a, requires_grad=True)
    right = _tensor(b, requires_grad=True)
    bias = _tensor(base, requires_grad=True)
    (
        mt.addmm(bias, left, right, beta=0.75, alpha=-2.5) * _tensor(upstream)
    ).sum().backward()
    assert left.grad.tolist() == _exact_affine(
        np.zeros((26, 35)), upstream, b.T, 0, -2.5
    )
    assert right.grad.tolist() == _exact_affine(
        np.zeros((35, 24)), a.T, upstream, 0, -2.5
    )
    ones = np.ones((1, 26))
    assert (
        bias.grad.tolist()
        == _exact_affine(np.zeros((1, 24)), ones, upstream, 0, 0.75)[0]
    )


def test_a_scale_float64_does_not_hold_takes_the_composed_form():
    # A whole number past 2^53 is not a float64, so the product is scaled as
    # the composed form scales it.
    rng = np.random.default_rng(1087)
    a, b, base = _draw(rng, (4, 6)), _draw(rng, (6, 5)), _draw(rng, (4, 5))
    got = mt.addmm(_tensor(base), _tensor(a), _tensor(b), alpha=2**53 + 1)
    want = _tensor(base) + mt.mm(_tensor(a), _tensor(b)) * (2**53 + 1)
    assert got.tolist() == want.tolist()
