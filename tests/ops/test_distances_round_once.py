# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""The 1- and 2-distances are each the float nearest the exact value.

A distance was the norm of the difference, and the difference was rounded
before the norm saw it: between rows of a hundred normal values, one float64
2-distance in five and nearly half the 1-distances were not the float nearest
the exact one. `dist`, `cdist`, `pdist` and
`pairwise_distance` now take each difference exactly -- `pairwise_distance`'s
shift included -- and round the distance once.
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
    exponent = value.numerator.bit_length() - value.denominator.bit_length()
    if Fraction(2) ** exponent > value:
        exponent -= 1
    unit = Fraction(2) ** max(exponent - 23, -149)
    rounded = round(value / unit) * unit
    return math.inf if rounded >= Fraction(2) ** 128 else float(rounded)


def _rounded(value: Fraction, dtype: str) -> float:
    if dtype == "float32":
        return _float32(value)
    try:
        return float(value)
    except OverflowError:
        return math.inf


def _exact(a, b, p, dtype, shift=0.0) -> float:
    """The `p`-distance between two rows, from the exact differences."""

    shift = Fraction(float(np.asarray(shift, dtype=dtype)))
    differences = [
        Fraction(float(x)) - Fraction(float(y)) + shift
        for x, y in zip(np.ravel(a), np.ravel(b))
    ]
    if p == 1:
        return _rounded(sum(abs(d) for d in differences), dtype)
    total = sum(d * d for d in differences)
    if total == 0:
        return 0.0
    # Far more bits than either float keeps, with whatever is left over
    # marked strictly inside the last place.
    places = 1200
    whole = total.numerator * total.denominator * 4**places
    root = math.isqrt(whole)
    scale = total.denominator * 2**places
    value = Fraction(root, scale)
    if root * root != whole:
        value += Fraction(1, 2 * scale)
    return _rounded(value, dtype)


def _rows(rng, dtype, shape, spread):
    values = rng.normal(size=shape) * 2.0 ** rng.integers(
        -spread, spread + 1, size=shape
    )
    return values.astype(dtype)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("p", [1, 2])
def test_dist_rounds_once(dtype, p):
    rng = np.random.default_rng(467)
    for trial in range(120):
        n = int(rng.integers(1, 300))
        a = _rows(rng, dtype, n, 20)
        b = _rows(rng, dtype, n, 20)
        if trial % 3 == 0:
            # Rows that nearly coincide.
            b = (a * (1 + rng.normal(size=n) * 1e-6)).astype(dtype)
        got = mt.dist(mt.Tensor(a, dtype=dtype), mt.Tensor(b, dtype=dtype), p).item()
        assert got == _exact(a, b, p, dtype), trial


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("p", [1, 2])
def test_cdist_and_pdist_round_once(dtype, p):
    rng = np.random.default_rng(479)
    x = _rows(rng, dtype, (2, 7, 5), 10)
    y = _rows(rng, dtype, (1, 4, 5), 10)
    got = mt.cdist(mt.Tensor(x, dtype=dtype), mt.Tensor(y, dtype=dtype), p).tolist()
    want = [
        [[_exact(x[k, i], y[0, j], p, dtype) for j in range(4)] for i in range(7)]
        for k in range(2)
    ]
    assert got == want
    rows = _rows(rng, dtype, (9, 6), 10)
    got = mt.pdist(mt.Tensor(rows, dtype=dtype), p).tolist()
    want = [
        _exact(rows[i], rows[j], p, dtype) for i in range(9) for j in range(i + 1, 9)
    ]
    assert got == want


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("p", [1, 2])
def test_pairwise_distance_takes_its_shift_exactly(dtype, p):
    rng = np.random.default_rng(487)
    a = _rows(rng, dtype, (30, 8), 4)
    b = (a + rng.normal(size=(30, 8)).astype(dtype) * 1e-4).astype(dtype)
    for eps in (1e-6, 0.0, 0.3):
        tensors = mt.Tensor(a, dtype=dtype), mt.Tensor(b, dtype=dtype)
        got = mt.pairwise_distance(*tensors, p=p, eps=eps).tolist()
        assert got == [_exact(a[i], b[i], p, dtype, eps) for i in range(30)], eps
    kept = mt.pairwise_distance(*tensors, p=p, keepdim=True)
    assert kept.shape == (30, 1)
    # A single row against a batch.
    got = mt.pairwise_distance(
        mt.Tensor(a[:1], dtype=dtype), mt.Tensor(b, dtype=dtype), p=p
    ).tolist()
    assert got == [_exact(a[0], b[i], p, dtype, 1e-6) for i in range(30)]


def test_the_ends_of_the_range():
    largest = float(np.finfo(np.float64).max)
    big = mt.tensor([largest, 1.0], dtype="float64")
    zero = mt.tensor([0.0, 1.0], dtype="float64")
    assert mt.dist(big, zero).item() == largest
    assert mt.dist(big, -big, 1).item() == math.inf
    tiny = 5e-324
    a = mt.tensor([3 * tiny, 0.0], dtype="float64")
    b = mt.tensor([0.0, -4 * tiny], dtype="float64")
    assert mt.dist(a, b).item() == 5 * tiny
    values = [1e200, -3e199, 7e-200]
    others = [-1e200, 2e199, 0.0]
    got = mt.dist(
        mt.tensor(values, dtype="float64"), mt.tensor(others, dtype="float64")
    )
    assert got.item() == _exact(values, others, 2, "float64")


def test_specials_empty_rows_and_other_p():
    nan = mt.tensor([math.nan, 1.0], dtype="float64")
    one = mt.tensor([1.0, 1.0], dtype="float64")
    assert math.isnan(mt.dist(nan, one).item())
    inf = mt.tensor([math.inf, 1.0], dtype="float64")
    assert mt.dist(inf, one, 1).item() == math.inf
    empty = mt.Tensor(np.zeros((3, 0)), dtype="float64")
    assert mt.pairwise_distance(empty, empty, eps=0.0).tolist() == [0.0, 0.0, 0.0]
    assert mt.cdist(empty, empty).tolist() == [[0.0] * 3] * 3
    # Other orders still take the norm of the difference.
    x = mt.tensor([1.0, -2.0], dtype="float64")
    assert math.isclose(mt.dist(x, -x, 3).item(), (2**3 + 4**3) ** (1 / 3))
    assert mt.dist(x, -x, math.inf).item() == 4.0


@pytest.mark.parametrize("p", [1, 2])
def test_gradients(p):
    x = mt.tensor([[0.0, 3.0], [1.0, 1.0]], dtype="float64", requires_grad=True)
    y = mt.tensor([[4.0, 0.0], [1.0, 1.0]], dtype="float64", requires_grad=True)
    mt.pairwise_distance(x, y, p=p, eps=0.0).sum().backward()
    if p == 2:
        assert x.grad.tolist() == [[-0.8, 0.6], [0.0, 0.0]]
    else:
        assert x.grad.tolist() == [[-1.0, 1.0], [0.0, 0.0]]
    assert y.grad.tolist() == [[-v for v in row] for row in x.grad.tolist()]
    x.grad = None
    y.grad = None
    mt.cdist(x, y, p).sum().backward()
    difference = np.array(x.tolist())[:, None, :] - np.array(y.tolist())[None, :, :]
    lengths = np.linalg.norm(difference, ord=p, axis=-1, keepdims=True)
    if p == 2:
        direction = np.divide(
            difference, lengths, out=np.zeros_like(difference), where=lengths != 0
        )
    else:
        direction = np.sign(difference)
    assert np.allclose(x.grad.tolist(), direction.sum(axis=1))
    assert np.allclose(y.grad.tolist(), -direction.sum(axis=0))
    z = mt.tensor([1.0, 2.0, 2.0], dtype="float64", requires_grad=True)
    mt.dist(z, mt.tensor([0.0], dtype="float64"), p).backward()
    want = [1 / 3, 2 / 3, 2 / 3] if p == 2 else [1.0, 1.0, 1.0]
    assert np.allclose(z.grad.tolist(), want)
