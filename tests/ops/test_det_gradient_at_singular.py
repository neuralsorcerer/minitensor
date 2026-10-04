# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""`det`'s gradient is the cofactor matrix, singular matrix or not.

It was computed as `det(A) * A^-T`, which needs an inverse: on a singular
matrix the backward refused with "solve received a singular matrix" although
the forward had answered, and on a nearly singular one the product of a tiny
determinant and a huge inverse kept few of its digits -- half the answer was
wrong on a rank-deficient 4x4. The cofactors are checked here against exact
rational arithmetic on the matrix's own floating-point entries.
"""

from fractions import Fraction

import numpy as np
import pytest

import minitensor as mt


def _det(rows):
    rows = [row[:] for row in rows]
    size, result = len(rows), Fraction(1)
    for col in range(size):
        pivot = next((r for r in range(col, size) if rows[r][col] != 0), None)
        if pivot is None:
            return Fraction(0)
        if pivot != col:
            rows[col], rows[pivot] = rows[pivot], rows[col]
            result = -result
        result *= rows[col][col]
        for r in range(col + 1, size):
            factor = rows[r][col] / rows[col][col]
            for k in range(col, size):
                rows[r][k] -= factor * rows[col][k]
    return result


def _cofactors(matrix):
    size = matrix.shape[0]
    exact = [[Fraction(float(v)) for v in row] for row in matrix]
    out = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            minor = [
                [exact[r][c] for c in range(size) if c != j]
                for r in range(size)
                if r != i
            ]
            out[i, j] = float((-1) ** (i + j) * _det(minor)) if size > 1 else 1.0
    return out


_RNG = np.random.default_rng(0)
MATRICES = {
    "rank-one": np.array([[1.0, 2.0], [3.0, 6.0]]),
    "zero": np.zeros((3, 3)),
    "one-by-one-zero": np.zeros((1, 1)),
    "rank-deficient-4x4": _RNG.standard_normal((4, 4))
    @ np.diag([2.0, 1.0, 0.5, 0.0])
    @ _RNG.standard_normal((4, 4)),
    "ill-conditioned": np.array([[1.0, 1.0], [1.0, 1.0 + 1e-10]]),
    "regular": _RNG.standard_normal((5, 5)),
}


@pytest.mark.parametrize("dtype,tolerance", [("float64", 1e-12), ("float32", 1e-4)])
@pytest.mark.parametrize("name", list(MATRICES))
def test_det_gradient_is_the_cofactor_matrix(name, dtype, tolerance):
    matrix = MATRICES[name].astype(dtype)
    a = mt.tensor(matrix, dtype=dtype, requires_grad=True)
    mt.det(a).backward()
    expected = _cofactors(matrix.astype(np.float64))
    error = np.abs(a.grad.numpy() - expected).max() / max(np.abs(expected).max(), 1.0)
    assert error < tolerance


def test_a_batch_with_one_singular_matrix_differentiates_both():
    batch = np.stack(
        [np.array([[1.0, 2.0], [2.0, 4.0]]), np.array([[2.0, 1.0], [0.0, 3.0]])]
    )
    a = mt.tensor(batch, dtype="float64", requires_grad=True)
    (mt.det(a) * mt.tensor([1.0, 2.0], dtype="float64")).sum().backward()
    expected = np.stack([_cofactors(batch[0]), 2 * _cofactors(batch[1])])
    assert np.allclose(a.grad.numpy(), expected)
