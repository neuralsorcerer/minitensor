# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""Each component of a float cross product is `x * y - z * w` rounded once.

Each product was rounded on its own and the two subtracted, so where they
nearly cancel -- the cross product of nearly parallel vectors -- the
component was the products' rounding noise: every one of a thousand such
components came out wrong, most of them by far more than an ulp. A float32
component is now the exact difference of the exact float64 products, rounded
to odd at float64's last bit and narrowed once; a float64 one splits each
product by a fused multiply-add and settles the four parts with a bound, or
takes the products exactly where the bound cannot say.
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


def _cross(a, b, dtype: str):
    a = [Fraction(float(v)) for v in a]
    b = [Fraction(float(v)) for v in b]
    exact = [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
    return [_float32(v) if dtype == "float32" else float(v) for v in exact]


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_nearly_parallel_vectors(dtype):
    rng = np.random.default_rng(397)
    for _ in range(300):
        a = rng.normal(size=3)
        b = a * (1 + rng.normal(size=3) * 1e-7)
        a, b = a.astype(dtype), b.astype(dtype)
        want = _cross(a, b, dtype)
        left, right = mt.Tensor(a, dtype=dtype), mt.Tensor(b, dtype=dtype)
        assert left.cross(right).tolist() == want
        assert mt.cross(left, right).tolist() == want
        assert mt.numpy_compat.cross(left, right).tolist() == want


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_batches_axes_and_broadcasting(dtype):
    rng = np.random.default_rng(401)
    rows = rng.normal(size=(40, 3)).astype(dtype)
    shared = rng.normal(size=3).astype(dtype)
    got = mt.Tensor(rows, dtype=dtype).cross(mt.Tensor(shared, dtype=dtype))
    assert got.tolist() == [_cross(row, shared, dtype) for row in rows]
    columns = rng.normal(size=(3, 5)).astype(dtype)
    others = rng.normal(size=(3, 5)).astype(dtype)
    got = mt.Tensor(columns, dtype=dtype).cross(mt.Tensor(others, dtype=dtype), dim=0)
    want = np.array(
        [_cross(columns[:, j], others[:, j], dtype) for j in range(5)]
    ).T.tolist()
    assert got.tolist() == want


def test_mixed_widths_and_integers():
    rng = np.random.default_rng(409)
    a = rng.normal(size=3).astype(np.float32)
    b = rng.normal(size=3)
    got = mt.Tensor(a).cross(mt.Tensor(b, dtype="float64"))
    assert str(got.dtype) == "float64"
    assert got.tolist() == _cross(a, b, "float64")
    assert mt.tensor([1, 2, 3]).cross(mt.tensor([4, 5, 6])).tolist() == [-3, 6, -3]


def test_specials():
    # Two products of half the least subnormal: each rounds to zero, their
    # difference is the least subnormal.
    x, y = 2.0**-537, 2.0**-538
    left = mt.Tensor([x, 0.0, x], dtype="float64")
    right = mt.Tensor([y, 0.0, -y], dtype="float64")
    assert left.cross(right).tolist() == [0.0, 5e-324, 0.0]
    # Products past the largest float64 that cancel exactly.
    big = mt.Tensor([1e200, 1e200, 0.0], dtype="float64")
    assert big.cross(big).tolist() == [0.0, 0.0, 0.0]
    infinite = mt.Tensor([math.inf, 0.0, 0.0], dtype="float64")
    got = infinite.cross(mt.Tensor([0.0, 1.0, 0.0], dtype="float64")).tolist()
    assert got[0] == 0.0 and math.isnan(got[1]) and got[2] == math.inf


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_gradients(dtype):
    a = mt.tensor([1.0, 2.0, 3.0], dtype=dtype, requires_grad=True)
    b = mt.tensor([4.0, 5.0, 6.0], dtype=dtype, requires_grad=True)
    product = a.cross(b)
    assert product.tolist() == [-3.0, 6.0, -3.0]
    product.sum().backward()
    # The gradient of `sum(a x b)` is `b x 1` for `a` and `1 x a` for `b`.
    assert a.grad.tolist() == [-1.0, 2.0, -1.0]
    assert b.grad.tolist() == [1.0, -2.0, 1.0]
    rows = mt.tensor(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=dtype, requires_grad=True
    )
    shared = mt.tensor([0.0, 0.0, 1.0], dtype=dtype, requires_grad=True)
    rows.cross(shared).sum().backward()
    # `sum(r x s)` is `s . (1 x r)`, so each row adds `1 x r`.
    assert shared.grad.tolist() == [-1.0, 1.0, 0.0]
