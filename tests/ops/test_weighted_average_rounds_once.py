# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""A weighted `average` is the exact `sum(a * w) / sum(w)` rounded once.

Formed in floats, each product, the two sums and the quotient were rounded
on their own: two weighted means in three of ordinary float64 data were not
the float64 nearest the exact one, and a float32 pair, carried in float64,
was rounded again on the way to float32. A weight total that nearly cancels
magnified every one of those roundings, and values and weights whose
products passed the largest float64 gave an infinity where the mean was in
range. The weighted sum and the weight total are now carried with their
exact errors, their quotient settled by a bound, and both taken exactly and
divided in whole limbs where the bound cannot say.
"""

from __future__ import annotations

import math
from fractions import Fraction

import numpy as np
import pytest

import minitensor as mt


def _exact(values, weights) -> Fraction:
    total = sum(Fraction(float(w)) for w in weights)
    return (
        sum(Fraction(float(v)) * Fraction(float(w)) for v, w in zip(values, weights))
        / total
    )


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


def _samples(seed, n_max=500):
    rng = np.random.default_rng(seed)
    for round_ in range(30):
        n = int(rng.integers(1, n_max))
        values = rng.normal(loc=float(rng.choice([0.0, 3.0, 1e4])), size=n)
        weights = rng.uniform(0.1, 3.0, size=n)
        if round_ % 3 == 1:
            # A weight total that nearly cancels.
            weights = np.append(weights, -weights.sum() * (1 - 2**-30))
            values = np.append(values, 1.0)
        yield values, weights


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_weighted_means_round_once(dtype):
    for values, weights in _samples(367):
        values, weights = values.astype(dtype), weights.astype(dtype)
        want = _rounded(_exact(values, weights), dtype)
        got = mt.average(
            mt.Tensor(values, dtype=dtype), weights=mt.Tensor(weights, dtype=dtype)
        )
        assert str(got.dtype) == dtype
        assert got.item() == want


def test_mixed_operands_are_read_exactly():
    rng = np.random.default_rng(373)
    values = rng.integers(-(2**30), 2**30, size=300).astype(np.int32)
    weights = rng.uniform(0.1, 3.0, size=300).astype(np.float32)
    got = mt.average(mt.Tensor(values, dtype="int32"), weights=mt.Tensor(weights))
    assert str(got.dtype) == "float32"
    assert got.item() == _float32(_exact(values, weights))
    wide = rng.uniform(0.1, 3.0, size=300)
    got = mt.average(mt.Tensor(values.astype(np.float32)), weights=mt.Tensor(wide))
    assert str(got.dtype) == "float64"
    assert got.item() == float(_exact(values.astype(np.float32), wide))


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_axes_and_one_dimensional_weights(dtype):
    rng = np.random.default_rng(379)
    values = rng.normal(loc=5.0, size=(20, 30, 7)).astype(dtype)
    along = rng.uniform(0.1, 2.0, size=30).astype(dtype)
    got = mt.average(mt.Tensor(values, dtype=dtype), dim=1, weights=mt.Tensor(along))
    want = [
        [_rounded(_exact(values[i, :, k], along), dtype) for k in range(7)]
        for i in range(20)
    ]
    assert got.tolist() == want
    full = rng.uniform(0.1, 2.0, size=values.shape).astype(dtype)
    got = mt.average(
        mt.Tensor(values, dtype=dtype),
        dim=[0, 2],
        weights=mt.Tensor(full, dtype=dtype),
        keepdim=True,
    )
    assert got.shape == (1, 30, 1)
    want = [
        _rounded(_exact(values[:, j, :].ravel(), full[:, j, :].ravel()), dtype)
        for j in range(30)
    ]
    assert got.reshape(-1).tolist() == want
    mean, total = mt.average(
        mt.Tensor(values, dtype=dtype),
        weights=mt.Tensor(full, dtype=dtype),
        returned=True,
    )
    assert mean.item() == _rounded(_exact(values.ravel(), full.ravel()), dtype)
    assert total.item() == _rounded(
        sum(Fraction(float(w)) for w in full.ravel()), dtype
    )


def test_totals_past_the_range_and_specials():
    big = mt.Tensor([1e300, 1e300], dtype="float64")
    assert mt.average(big, weights=big).item() == 1e300
    # A weight total that is exactly zero.
    ones = mt.Tensor([1.0, 2.0], dtype="float64")
    assert (
        mt.average(ones, weights=mt.Tensor([1.0, -1.0], dtype="float64")).item()
        == -math.inf
    )
    zeros = mt.Tensor([0.0, 0.0], dtype="float64")
    assert math.isnan(
        mt.average(zeros, weights=mt.Tensor([1.0, -1.0], dtype="float64")).item()
    )
    infinite = mt.Tensor([math.inf, 1.0], dtype="float64")
    assert mt.average(infinite, weights=ones).item() == math.inf
    assert math.isnan(mt.average(ones, weights=infinite).item())
    assert math.isnan(
        mt.average(mt.Tensor([math.nan, 1.0]), weights=mt.Tensor([1.0, 1.0])).item()
    )


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_gradients_reach_values_and_weights(dtype):
    values = mt.tensor([1.0, 2.0, 4.0], dtype=dtype, requires_grad=True)
    weights = mt.tensor([1.0, 1.0, 2.0], dtype=dtype, requires_grad=True)
    mean = mt.average(values, weights=weights)
    assert mean.item() == 2.75
    mean.backward()
    assert values.grad.tolist() == [0.25, 0.25, 0.5]
    assert weights.grad.tolist() == [-0.4375, -0.1875, 0.3125]
    assert str(values.grad.dtype) == dtype and str(weights.grad.dtype) == dtype
    # One weight per column, shared by every row.
    rows = mt.tensor([[1.0, 3.0], [2.0, 6.0]], dtype=dtype, requires_grad=True)
    along = mt.tensor([1.0, 3.0], dtype=dtype, requires_grad=True)
    mt.average(rows, dim=1, weights=along).sum().backward()
    assert rows.grad.tolist() == [[0.25, 0.75], [0.25, 0.75]]
    # Means 2.5 and 5: each row's (value - mean) / 4, summed over the rows.
    assert along.grad.tolist() == [(-1.5 - 3.0) / 4, (0.5 + 1.0) / 4]


def test_integer_operands_are_multiplied_exactly():
    # Products past int32, which wrapped to nothing.
    values = mt.tensor([2**20, 2**20], dtype="int32")
    weights = mt.tensor([2**12, 2**12], dtype="int32")
    got = mt.average(values, weights=weights)
    assert str(got.dtype) == "float64" and got.item() == 2.0**20
    # Products and totals past int64, and a quotient that is not whole.
    big = [2**62, 2**62 - 1, -(2**62), 7]
    weights = [2**62, 3, 2**62 - 5, 2**40 + 1]
    want = float(Fraction(sum(v * w for v, w in zip(big, weights)), sum(weights)))
    got = mt.average(
        mt.tensor(big, dtype="int64"), weights=mt.tensor(weights, dtype="int64")
    )
    assert got.item() == want
    rows = mt.tensor([[1, 2], [3, 4]], dtype="int32")
    assert mt.average(rows, dim=0, weights=mt.tensor([1, 3])).tolist() == [2.5, 3.5]
    flags = mt.tensor([True, False, True])
    assert mt.average(flags, weights=mt.tensor([1, 2, 3])).item() == 4 / 6
