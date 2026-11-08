# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""A float32 matrix product gives each entry as the float32 nearest its exact
dot, and an affine one as the float32 nearest its exact value.

Accumulated in float32, nine entries in ten of a product of ordinary random
matrices three hundred deep came out other than that float. Each is that float
now, for `matmul`, `mm`, `bmm` and the `@` operator, batched or broadcast,
for operands read transposed where they lie, for the gradients a product
hands back, and -- the product scaled and an addend added before the one
rounding -- for a dense layer with a bias, `addmm` and `baddbmm`.
"""

from __future__ import annotations

import math
from fractions import Fraction

import numpy as np
import pytest

import minitensor as mt

# Every float32 is a whole multiple of 2^-149, so every product of two is one
# of 2^-298: the exact dot is a sum of whole numbers over that.
_SCALE = 2**149


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


def _whole(values) -> np.ndarray:
    """Each float32 of `values` as a whole number of 2^-149, exactly."""

    array = np.asarray(values, dtype=np.float32)
    flat = [int(Fraction(float(v)) * _SCALE) for v in array.ravel()]
    return np.array(flat, dtype=object).reshape(array.shape)


def _exact_product(a, b) -> list:
    """The float32 nearest each entry of `a @ b`, batched as `matmul` is."""

    total = np.matmul(_whole(a), _whole(b))
    rounded = np.vectorize(
        lambda v: _float32(Fraction(v, _SCALE * _SCALE)), otypes=[float]
    )
    return rounded(total).tolist()


def _draw(rng, shape):
    values = rng.normal(size=shape) * 10.0 ** rng.uniform(-2, 2, shape)
    return values.astype(np.float32)


@pytest.mark.parametrize(
    "m, k, n", [(7, 300, 5), (31, 64, 33), (3, 1000, 4), (40, 2, 50)]
)
def test_matrix_products(m, k, n):
    rng = np.random.default_rng(941 + k)
    a, b = _draw(rng, (m, k)), _draw(rng, (k, n))
    want = _exact_product(a, b)
    left, right = mt.Tensor(a), mt.Tensor(b)
    for got in (mt.matmul(left, right), mt.mm(left, right), left @ right):
        assert str(got.dtype) == "float32"
        assert got.tolist() == want


def test_batched_and_broadcast_products():
    rng = np.random.default_rng(947)
    a, b = _draw(rng, (3, 4, 60)), _draw(rng, (3, 60, 5))
    want = _exact_product(a, b)
    assert mt.bmm(mt.Tensor(a), mt.Tensor(b)).tolist() == want
    assert mt.matmul(mt.Tensor(a), mt.Tensor(b)).tolist() == want
    # One matrix against a stack, and a stack broadcast against another.
    c = _draw(rng, (60, 6))
    assert mt.matmul(mt.Tensor(a), mt.Tensor(c)).tolist() == _exact_product(a, c)
    d = _draw(rng, (1, 4, 60))
    assert mt.matmul(mt.Tensor(d), mt.Tensor(b)).tolist() == _exact_product(d, b)


def test_operands_read_transposed():
    rng = np.random.default_rng(953)
    a, b = _draw(rng, (90, 6)), _draw(rng, (7, 90))
    want = _exact_product(a.T, b.T)
    got = mt.matmul(mt.Tensor(a).T, mt.Tensor(b).T)
    assert got.tolist() == want


def test_a_dense_layer_without_a_bias():
    rng = np.random.default_rng(967)
    x, w = _draw(rng, (2, 5, 70)), _draw(rng, (8, 70))
    got = mt.nn.dense_layer(mt.Tensor(x), mt.Tensor(w))
    assert got.tolist() == _exact_product(x, w.T)


def test_gradients_of_a_product():
    rng = np.random.default_rng(971)
    a, b, weights = _draw(rng, (6, 80)), _draw(rng, (80, 5)), _draw(rng, (6, 5))
    left = mt.Tensor(a, requires_grad=True)
    right = mt.Tensor(b, requires_grad=True)
    (mt.matmul(left, right) * mt.Tensor(weights)).sum().backward()
    assert left.grad.tolist() == _exact_product(weights, b.T)
    assert right.grad.tolist() == _exact_product(a.T, weights)

    x, w = _draw(rng, (9, 40)), _draw(rng, (3, 40))
    inputs = mt.Tensor(x, requires_grad=True)
    weight = mt.Tensor(w, requires_grad=True)
    upstream = _draw(rng, (9, 3))
    (mt.nn.dense_layer(inputs, weight) * mt.Tensor(upstream)).sum().backward()
    assert inputs.grad.tolist() == _exact_product(upstream, w)
    assert weight.grad.tolist() == _exact_product(upstream.T, x)


def test_cancellation_and_special_values():
    big = np.float32(2.0**60)
    a = np.array(
        [
            [big, 1.0, -big, 2.0**-30],
            [3.0, -3.0, 2.0**-100, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [7.0, 1.0, 1.0, 1.0],
        ],
        dtype=np.float32,
    )
    b = np.array(
        [[1.0, 2.0, 1.0], [1.0, 2.0, 3.0], [1.0, 2.0, 1.0], [1.0, -4.0, 2.0**40]],
        dtype=np.float32,
    )
    got = mt.matmul(mt.Tensor(a), mt.Tensor(b)).tolist()
    assert got == _exact_product(a, b)
    # A zero entry is a positive zero, as a dot's is.
    assert all(math.copysign(1.0, v) > 0 for v in got[2])

    special = np.array(
        [[math.inf, 1.0], [math.nan, 1.0], [math.inf, -1.0]], dtype=np.float32
    )
    columns = np.array([[1.0, 0.0, -1.0], [2.0, 1.0, 0.0]], dtype=np.float32)
    got = mt.matmul(mt.Tensor(special), mt.Tensor(columns)).tolist()
    assert got[0][0] == math.inf and math.isnan(got[0][1]) and got[0][2] == -math.inf
    assert all(math.isnan(v) for v in got[1])
    assert got[2][0] == math.inf and math.isnan(got[2][1]) and got[2][2] == -math.inf


def test_larger_products_agree_with_their_dots():
    # Large enough to cross the pool and the installed provider; each entry
    # checked against `dot` of its row and column, itself rounded once.
    rng = np.random.default_rng(977)
    a, b = _draw(rng, (130, 257)), _draw(rng, (257, 140))
    got = mt.matmul(mt.Tensor(a), mt.Tensor(b)).tolist()
    for i in range(0, 130, 13):
        row = mt.Tensor(a[i])
        for j in range(0, 140, 7):
            assert (
                got[i][j]
                == mt.dot(row, mt.Tensor(np.ascontiguousarray(b[:, j]))).item()
            )


def _exact_affine(base, left, right, beta, alpha) -> list:
    """The float32 nearest each entry of `beta * base + alpha * left @ right`."""

    left, right = np.asarray(left), np.asarray(right)
    out = np.broadcast_to(np.asarray(base), left.shape[:-1] + right.shape[-1:])
    exact = lambda v: Fraction(float(v))  # noqa: E731
    total = np.matmul(
        np.vectorize(exact, otypes=[object])(left),
        np.vectorize(exact, otypes=[object])(right),
    )
    scaled = total * exact(alpha) + np.vectorize(exact, otypes=[object])(out) * exact(
        beta
    )
    return np.vectorize(_float32, otypes=[float])(scaled).tolist()


def test_a_dense_layer_with_a_bias():
    rng = np.random.default_rng(983)
    x, w, b = _draw(rng, (3, 4, 50)), _draw(rng, (7, 50)), _draw(rng, 7)
    got = mt.nn.dense_layer(mt.Tensor(x), mt.Tensor(w), mt.Tensor(b))
    assert got.tolist() == _exact_affine(b, x, w.T, 1.0, 1.0)
    # The gradient of the bias is each column of the incoming one summed.
    upstream = _draw(rng, (3, 4, 7))
    bias = mt.Tensor(b, requires_grad=True)
    inputs = mt.Tensor(x, requires_grad=True)
    (
        mt.nn.dense_layer(inputs, mt.Tensor(w), bias) * mt.Tensor(upstream)
    ).sum().backward()
    flat = upstream.reshape(-1, 7)
    assert bias.grad.tolist() == _exact_product(np.ones((1, 12), np.float32), flat)[0]
    assert inputs.grad.tolist() == _exact_product(upstream, w)


@pytest.mark.parametrize("beta, alpha", [(1, 1), (0.5, -1.5), (3.0, 0.0), (0.0, 2.0)])
def test_addmm_and_baddbmm(beta, alpha):
    rng = np.random.default_rng(991)
    a, b = _draw(rng, (6, 45)), _draw(rng, (45, 5))
    for base in (_draw(rng, (6, 5)), _draw(rng, 5), _draw(rng, (6, 1))):
        got = mt.addmm(
            mt.Tensor(base), mt.Tensor(a), mt.Tensor(b), beta=beta, alpha=alpha
        )
        assert got.tolist() == _exact_affine(base, a, b, beta, alpha), base.shape
    left, right = _draw(rng, (3, 4, 30)), _draw(rng, (3, 30, 2))
    for base in (_draw(rng, (3, 4, 2)), _draw(rng, (3, 1, 2)), _draw(rng, (4, 2))):
        got = mt.baddbmm(
            mt.Tensor(base), mt.Tensor(left), mt.Tensor(right), beta=beta, alpha=alpha
        )
        assert got.tolist() == _exact_affine(base, left, right, beta, alpha), base.shape


def test_gradients_of_addmm():
    rng = np.random.default_rng(997)
    a, b, base = _draw(rng, (6, 35)), _draw(rng, (35, 4)), _draw(rng, 4)
    upstream = _draw(rng, (6, 4))
    left = mt.Tensor(a, requires_grad=True)
    right = mt.Tensor(b, requires_grad=True)
    bias = mt.Tensor(base, requires_grad=True)
    (
        mt.addmm(bias, left, right, beta=0.75, alpha=-2.5) * mt.Tensor(upstream)
    ).sum().backward()
    assert left.grad.tolist() == _exact_affine(
        np.zeros((6, 35)), upstream, b.T, 0, -2.5
    )
    assert right.grad.tolist() == _exact_affine(
        np.zeros((35, 4)), a.T, upstream, 0, -2.5
    )
    ones = np.ones((1, 6), np.float32)
    assert (
        bias.grad.tolist()
        == _exact_affine(np.zeros((1, 4)), ones, upstream, 0, 0.75)[0]
    )
