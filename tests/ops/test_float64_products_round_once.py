# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""A float64 `dot`, `vecdot`, and product with one row or one column are
rounded once.

Added in lanes and folded, a float64 dot rounded every product and every
addition: nine dots in ten of ordinary data were not the float64 nearest the
exact one, products that cancelled lost everything below float64's last bit,
and `0.1 * 0.1 + 0.2 * 0.2` rounded three times. Each product is now split
exactly into the float64 nearest it and that rounding's error, the halves are
carried as a sum's values are, and the exact dot decides where the bound on
that cannot.
"""

from __future__ import annotations

import math
from fractions import Fraction

import numpy as np
import pytest

import minitensor as mt

# Every product of two float64s is a whole multiple of 2^-2148.
_SCALE = 2**2148


def _exact_dot(a, b) -> Fraction:
    total = 0
    for p, q in zip(
        np.asarray(a, np.float64).tolist(), np.asarray(b, np.float64).tolist()
    ):
        (pn, pd), (qn, qd) = p.as_integer_ratio(), q.as_integer_ratio()
        total += pn * qn * (_SCALE // (pd * qd))
    return Fraction(total, _SCALE)


def _float64(value: Fraction) -> float:
    """The float64 nearest `value`, ties to even; an infinity past the
    largest."""

    try:
        return float(value)
    except OverflowError:
        return math.copysign(math.inf, value)


def _tensor(values):
    return mt.Tensor(np.asarray(values, dtype=np.float64), dtype="float64")


def _products(a, b):
    """`dot`, `vecdot` and `a @ b` of two vectors, then the matrix with rows
    `a` and `-a` times `b`, and `a` times the matrix with columns `b` and `-b`,
    each upright and transposed."""

    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    left, right = _tensor(a), _tensor(b)
    rows, columns = np.stack([a, -a]), np.stack([b, -b], axis=1)
    got = [mt.dot(left, right).item(), mt.vecdot(left, right).item()]
    got.append((left @ right).item())
    got += (_tensor(rows) @ right).tolist()
    got += (_tensor(rows.T.copy()).t() @ right).tolist()
    got += (left @ _tensor(columns)).tolist()
    got += (left @ _tensor(columns.T.copy()).t()).tolist()
    return got


def _expect(value):
    return [value] * 3 + [value, -value] * 4


def test_halfway_points_and_cancellation_round_once():
    half = 2.0**-53
    # On a float64 halfway point, and a hair past it.
    assert _products([1.0, half], [1.0, 1.0]) == _expect(1.0)
    assert _products([1.0, half, 1e-300], [1.0] * 3) == _expect(1 + 2**-52)
    # Lost below float64's last bit as the products cancel.
    assert _products([2.0**60, 1.0, -(2.0**60)], [1.0] * 3) == _expect(1.0)
    assert _products([1e100, 1.0, 1e100], [1e100, 1.0, -1e100]) == _expect(1.0)
    # Rounded once rather than at each product and the addition.
    want = _float64(_exact_dot([0.1, 0.2], [0.1, 0.2]))
    assert _products([0.1, 0.2], [0.1, 0.2]) == _expect(want)
    assert want != 0.1 * 0.1 + 0.2 * 0.2


def test_products_beyond_the_range():
    # Each product rounds to zero; together they are the least subnormal.
    tiny, small = 2.0**-600, 2.0**-475
    assert _products([tiny, tiny], [small, small]) == _expect(5e-324)
    # Products past the largest float64 that cancel, and one that does not.
    assert _products([1e200, 1e200, 3.0], [1e200, -1e200, 0.5]) == _expect(1.5)
    assert _products([1e200], [1e200]) == _expect(math.inf)
    assert _products([0.0] * 5, [1.0] * 5) == _expect(0.0)


def test_specials():
    assert _products([math.inf, 1e200], [1.0, -1e200]) == _expect(math.inf)
    for a, b in (
        ([math.inf, 1.0], [0.0, 1.0]),
        ([math.inf, math.inf], [1.0, -1.0]),
        ([math.nan, 1.0], [1.0, 1.0]),
    ):
        for got in _products(a, b):
            assert math.isnan(got), (a, b)


def _cancelling(rng, n):
    """Products spread over eighty binades that nearly cancel in pairs."""

    a = rng.normal(size=n) * 2.0 ** rng.integers(-40, 40, size=n)
    b = rng.normal(size=n)
    a, b = np.append(a, -a[1:]), np.append(b, b[1:] * (1 + 2**-30))
    order = rng.permutation(len(a))
    return a[order], b[order]


@pytest.mark.parametrize("n", [3, 900, 20_000, 200_000])
def test_long_dots_round_once(n):
    rng = np.random.default_rng(n)
    for a, b in (
        _cancelling(rng, n // 2 + 1),
        (rng.normal(size=n), rng.normal(size=n)),
    ):
        want = _float64(_exact_dot(a, b))
        assert mt.dot(_tensor(a), _tensor(b)).item() == want
        assert mt.vecdot(_tensor(a), _tensor(b)).item() == want


def test_ordinary_dots_round_once():
    rng = np.random.default_rng(353)
    rows = rng.normal(loc=1.0, size=(40, 300))
    others = rng.normal(size=(40, 300))
    want = [_float64(_exact_dot(r, o)) for r, o in zip(rows, others)]
    assert mt.vecdot(_tensor(rows), _tensor(others)).tolist() == want
    for row, other, value in zip(rows, others, want):
        assert mt.dot(_tensor(row), _tensor(other)).item() == value


def test_long_rows_and_many_columns_round_once():
    # Several blocks of a column and several tiles of columns, spread across
    # the pool.
    rng = np.random.default_rng(359)
    pairs = [_cancelling(rng, 5_001) for _ in range(3)]
    vector = pairs[0][0]
    others = np.stack([pair[1] for pair in pairs])
    want = [_float64(_exact_dot(vector, other)) for other in others]
    wide = np.tile(others, (100, 1))
    assert (_tensor(wide) @ _tensor(vector)).tolist() == want * 100
    assert (_tensor(vector) @ _tensor(wide.T.copy())).tolist() == want * 100
    assert (_tensor(vector) @ _tensor(wide).t()).tolist() == want * 100


def test_gradients_are_unchanged():
    a = mt.tensor([1.0, 2.0, 3.0], dtype="float64", requires_grad=True)
    b = mt.tensor([4.0, 5.0, 6.0], dtype="float64", requires_grad=True)
    mt.dot(a, b).backward()
    assert a.grad.tolist() == [4.0, 5.0, 6.0] and b.grad.tolist() == [1.0, 2.0, 3.0]
    matrix = mt.tensor([[1.0, 2.0], [3.0, 4.0]], dtype="float64", requires_grad=True)
    vector = mt.tensor([0.5, 0.25], dtype="float64", requires_grad=True)
    (vector @ matrix).sum().backward()
    assert matrix.grad.tolist() == [[0.5, 0.5], [0.25, 0.25]]
    assert vector.grad.tolist() == [3.0, 7.0]
