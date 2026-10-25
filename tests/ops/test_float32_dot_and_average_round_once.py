# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""A float32 `dot` and weighted `average` round once, from float64.

The product of two float32s is exact in float64, so carried there the only
rounding is the last one. Accumulated in float32, `dot` landed up to 90 ulps
from the exact inner product of offset data, and the weighted average several.
"""

from __future__ import annotations

from fractions import Fraction

import numpy as np

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


def _exact_dot(a, b) -> Fraction:
    return sum(Fraction(float(p)) * Fraction(float(q)) for p, q in zip(a, b))


def test_dot_is_correctly_rounded():
    for x, y, _ in _samples(307):
        want = float(np.float32(float(_exact_dot(x, y))))
        assert mt.dot(mt.Tensor(x), mt.Tensor(y)).item() == want


def test_weighted_average_is_correctly_rounded():
    for x, _, w in _samples(311):
        exact = _exact_dot(x, w) / sum(Fraction(float(v)) for v in w)
        want = float(np.float32(float(exact)))
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
    # vectors is `dot(a, b)`: each is carried in float64 and rounded once.
    rng = np.random.default_rng(331)
    for _ in range(6):
        m, k = int(rng.integers(1, 40)), int(rng.integers(1, 3000))
        offset = float(rng.choice([0.0, 3.0, 100.0]))
        matrix = (offset + rng.normal(0, 1, (m, k))).astype(np.float32)
        vector = rng.normal(0, 1, k).astype(np.float32)
        want = [float(np.float32(float(_exact_dot(row, vector)))) for row in matrix]
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
    by_vector = [float(np.float32(float(_exact_dot(row, vector)))) for row in rows]
    paired = [float(np.float32(float(_exact_dot(r, o)))) for r, o in zip(rows, others)]
    assert mt.vecdot(mt.Tensor(rows), mt.Tensor(vector)).tolist() == by_vector
    assert mt.vecdot(mt.Tensor(vector), mt.Tensor(rows)).tolist() == by_vector
    assert mt.vecdot(mt.Tensor(rows), mt.Tensor(others)).tolist() == paired
    shared = mt.Tensor(vector[:4].copy(), requires_grad=True)
    matrix = mt.Tensor(rows[:3, :4].copy(), requires_grad=True)
    mt.vecdot(matrix, shared).sum().backward()
    assert np.allclose(shared.grad.tolist(), rows[:3, :4].sum(0))
    assert np.allclose(matrix.grad.tolist(), np.tile(vector[:4], (3, 1)))
