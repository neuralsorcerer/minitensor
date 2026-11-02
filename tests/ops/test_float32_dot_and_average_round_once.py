# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""A float32 `dot` and weighted `average` round once.

The product of two float32s is exact in float64. Accumulated in float32, `dot`
landed up to 90 ulps from the exact inner product of offset data, and the
weighted average several. Summed in float64 and narrowed, a dot was nearly
always right, but not where the float64 sum lost what fell below its last bit
as the products cancelled, nor where it landed on a float32 halfway point and
was rounded a second time: `1 + 2^-24 + 2^-60` came out `1`. A `dot`, a
`vecdot`, and a product with one row or one column are now the float32
nearest the exact sum of the products.
"""

from __future__ import annotations

import math
from fractions import Fraction

import numpy as np
import pytest

import minitensor as mt


def _samples(seed):
    rng = np.random.default_rng(seed)
    for _ in range(25):
        n = int(rng.integers(2, 2500))
        offset = float(rng.choice([0.0, 100.0, 1e4]))
        x = (offset + rng.normal(0, 1, n)).astype(np.float32)
        y = rng.normal(0, 1, n).astype(np.float32)
        w = rng.uniform(0.1, 3.0, n).astype(np.float32)
        yield x, y, w


# Every product of two float32s is a whole multiple of 2^-298.
_SCALE = 2**300


def _exact_dot(a, b) -> Fraction:
    total = 0
    for p, q in zip(
        np.asarray(a, np.float32).tolist(), np.asarray(b, np.float32).tolist()
    ):
        numerator, denominator = (p * q).as_integer_ratio()
        total += numerator * (_SCALE // denominator)
    return Fraction(total, _SCALE)


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


def test_dot_is_correctly_rounded():
    for x, y, _ in _samples(307):
        want = _float32(_exact_dot(x, y))
        assert mt.dot(mt.Tensor(x), mt.Tensor(y)).item() == want


def test_weighted_average_is_correctly_rounded():
    for x, _, w in _samples(311):
        exact = _exact_dot(x, w) / sum(Fraction(float(v)) for v in w)
        want = _float32(exact)
        assert mt.average(mt.Tensor(x), weights=mt.Tensor(w)).item() == want
        rows = mt.Tensor(np.stack([x, x]))
        got = mt.average(rows, dim=1, weights=mt.Tensor(w)).tolist()
        assert got == [want, want]


def test_returned_totals_and_gradients_stay_float32():
    values = mt.Tensor(np.array([1.0, 2.0, 3.0], np.float32), requires_grad=True)
    weights = mt.Tensor(np.array([1.0, 1.0, 2.0], np.float32))
    mean, total = mt.average(values, weights=weights, returned=True)
    assert str(mean.dtype) == "float32" and str(total.dtype) == "float32"
    assert total.item() == 4.0
    mean.backward()
    assert values.grad.tolist() == [0.25, 0.25, 0.5]
    a = mt.Tensor(np.array([1.0, 2.0], np.float32), requires_grad=True)
    b = mt.Tensor(np.array([3.0, 4.0], np.float32), requires_grad=True)
    mt.dot(a, b).backward()
    assert a.grad.tolist() == [3.0, 4.0] and b.grad.tolist() == [1.0, 2.0]


def test_products_with_one_row_or_column_round_once():
    # A matrix times a vector is a row of dot products, and `a @ b` of two
    # vectors is `dot(a, b)`: each is the exact dot rounded once.
    rng = np.random.default_rng(331)
    for _ in range(6):
        m, k = int(rng.integers(1, 40)), int(rng.integers(1, 3000))
        offset = float(rng.choice([0.0, 3.0, 100.0]))
        matrix = (offset + rng.normal(0, 1, (m, k))).astype(np.float32)
        vector = rng.normal(0, 1, k).astype(np.float32)
        want = [_float32(_exact_dot(row, vector)) for row in matrix]
        assert (mt.Tensor(matrix) @ mt.Tensor(vector)).tolist() == want
        assert (mt.Tensor(vector) @ mt.Tensor(matrix.T.copy())).tolist() == want
        assert (mt.Tensor(vector) @ mt.Tensor(matrix).t()).tolist() == want
        batched = mt.bmm(mt.Tensor(matrix[None]), mt.Tensor(vector[None, :, None]))
        assert batched.reshape(-1).tolist() == want
        for row, value in zip(matrix[:3], want):
            assert (mt.Tensor(row) @ mt.Tensor(vector)).item() == value
            assert mt.dot(mt.Tensor(row), mt.Tensor(vector)).item() == value


def test_matrix_vector_gradients_stay_float32():
    matrix = mt.Tensor(np.ones((3, 4), np.float32), requires_grad=True)
    vector = mt.Tensor(np.arange(4, dtype=np.float32), requires_grad=True)
    (matrix @ vector).sum().backward()
    assert matrix.grad.tolist() == [[0.0, 1.0, 2.0, 3.0]] * 3
    assert vector.grad.tolist() == [3.0, 3.0, 3.0, 3.0]


def test_vecdot_rounds_once_shared_vector_or_not():
    rng = np.random.default_rng(337)
    rows = (3.0 + rng.normal(0, 1, (20, 700))).astype(np.float32)
    others = rng.normal(0, 1, (20, 700)).astype(np.float32)
    vector = others[0].copy()
    by_vector = [_float32(_exact_dot(row, vector)) for row in rows]
    paired = [_float32(_exact_dot(r, o)) for r, o in zip(rows, others)]
    assert mt.vecdot(mt.Tensor(rows), mt.Tensor(vector)).tolist() == by_vector
    assert mt.vecdot(mt.Tensor(vector), mt.Tensor(rows)).tolist() == by_vector
    assert mt.vecdot(mt.Tensor(rows), mt.Tensor(others)).tolist() == paired
    shared = mt.Tensor(vector[:4].copy(), requires_grad=True)
    matrix = mt.Tensor(rows[:3, :4].copy(), requires_grad=True)
    mt.vecdot(matrix, shared).sum().backward()
    assert np.allclose(shared.grad.tolist(), rows[:3, :4].sum(0))
    assert np.allclose(matrix.grad.tolist(), np.tile(vector[:4], (3, 1)))


def _products(a, b):
    """`dot`, `vecdot` and `a @ b` of two vectors, then the matrix with rows
    `a` and `-a` times `b`, and `a` times the matrix with columns `b` and `-b`,
    each upright and transposed."""

    a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
    left, right = mt.Tensor(a), mt.Tensor(b)
    rows, columns = np.stack([a, -a]), np.stack([b, -b], axis=1)
    got = [mt.dot(left, right).item(), mt.vecdot(left, right).item()]
    got.append((left @ right).item())
    got += (mt.Tensor(rows) @ right).tolist()
    got += (mt.Tensor(rows.T.copy()).t() @ right).tolist()
    got += (left @ mt.Tensor(columns)).tolist()
    got += (left @ mt.Tensor(columns.T.copy()).t()).tolist()
    return got


def _expect(value):
    return [value] * 3 + [value, -value] * 4


def test_halfway_points_and_cancellation_round_once():
    # Each a float32 halfway point in float64, and past it exactly.
    assert _products([1.0, 2**-24, 2**-60], [1.0] * 3) == _expect(1 + 2**-23)
    assert _products([1.0, -(2**-25), -(2**-60)], [1.0] * 3) == _expect(1 - 2**-24)
    # Lost below float64's last bit as the products cancel.
    assert _products([2.0**60, 1.0, -(2.0**60)], [1.0] * 3) == _expect(1.0)
    assert _products(
        [2.0**35, 3.0, 2.0**35], [2.0**35, 1.0, -(2.0**35)]
    ) == _expect(3.0)
    # Two halves of the least subnormal make it.
    tiny = 2.0**-149
    assert _products([tiny, tiny], [0.5, 0.5]) == _expect(tiny)


def test_specials():
    largest = float(np.finfo(np.float32).max)
    assert _products([largest, largest], [1.0, 1.0]) == _expect(math.inf)
    assert _products([largest, largest, -largest], [1.0] * 3) == _expect(largest)
    assert _products([math.inf, 1.0], [1.0, 1.0]) == _expect(math.inf)
    for got in _products([math.inf, 1.0], [0.0, 1.0]):
        assert math.isnan(got)
    for got in _products([math.nan, 1.0], [1.0, 1.0]):
        assert math.isnan(got)


def _cancelling(rng, n):
    """Products spread over eighty binades that nearly cancel in pairs."""

    a = (rng.normal(size=n) * 2.0 ** rng.integers(-40, 40, size=n)).astype(np.float32)
    b = rng.normal(size=n).astype(np.float32)
    a, b = np.append(a, -a[1:]), np.append(b, b[1:] * np.float32(1 + 2**-20))
    order = rng.permutation(len(a))
    return a[order], b[order]


@pytest.mark.parametrize("n", [3, 900, 20_000, 200_000])
def test_long_dots_round_once(n):
    rng = np.random.default_rng(n)
    a, b = _cancelling(rng, n // 2 + 1)
    want = _float32(_exact_dot(a, b))
    assert mt.dot(mt.Tensor(a), mt.Tensor(b)).item() == want
    assert mt.vecdot(mt.Tensor(a), mt.Tensor(b)).item() == want


def test_long_rows_and_many_columns_round_once():
    # Several blocks of a column and several tiles of columns, spread across
    # the pool.
    rng = np.random.default_rng(347)
    pairs = [_cancelling(rng, 10_001) for _ in range(3)]
    vector = pairs[0][0]
    others = np.stack([pair[1] for pair in pairs])
    want = [_float32(_exact_dot(vector, other)) for other in others]
    wide = np.tile(others, (100, 1))
    assert (mt.Tensor(wide) @ mt.Tensor(vector)).tolist() == want * 100
    assert (mt.Tensor(vector) @ mt.Tensor(wide.T.copy())).tolist() == want * 100
    assert (mt.Tensor(vector) @ mt.Tensor(wide).t()).tolist() == want * 100
