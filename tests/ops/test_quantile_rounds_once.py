# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""`quantile` is the float nearest the exact interpolated value.

The linear interpolation `lower + (upper - lower) * weight` rounded the gap,
the product and the sum -- and a float32 result once more, narrowed from
float64 -- and of random float64 samples about one quantile in four came
out other than the float nearest the exact value. The position is the
float64 nearest `q * (n - 1)`, as it was, and the interpolation at it is
exact and rounded once.
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


def _rounded(value: Fraction, dtype: str) -> float:
    if dtype == "float32":
        return _float32(value)
    try:
        return float(value)
    except OverflowError:
        return math.inf if value > 0 else -math.inf


def _want(values, q: float, mode: str, dtype: str) -> float:
    """The exact quantile of finite `values`, rounded once."""

    ordered = sorted(Fraction(float(value)) for value in values)
    # The float64 nearest `q * (n - 1)`, which Python's own product is.
    position = Fraction(min(max(float(q) * (len(ordered) - 1), 0.0), len(ordered) - 1))
    lower = math.floor(position)
    fraction = position - lower
    upper = min(lower + 1, len(ordered) - 1) if fraction else lower
    low, high = ordered[lower], ordered[upper]
    if mode == "linear":
        exact = low + fraction * (high - low)
    elif mode == "lower":
        exact = low
    elif mode == "higher":
        exact = high
    elif mode == "midpoint":
        exact = (low + high) / 2
    elif fraction == Fraction(1, 2):
        exact = ordered[lower + (lower & 1)]
    else:
        exact = low if fraction < Fraction(1, 2) else high
    return _rounded(exact, dtype)


MODES = ["linear", "lower", "higher", "midpoint", "nearest"]


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("mode", MODES)
def test_random_samples(dtype, mode):
    rng = np.random.default_rng(719)
    for trial in range(200):
        n = int(rng.integers(2, 40))
        values = (rng.normal(size=n) * 10.0 ** rng.uniform(-6, 6, n)).astype(dtype)
        q = float(rng.uniform())
        got = mt.quantile(mt.Tensor(values, dtype=dtype), q, interpolation=mode)
        assert got.item() == _want(values, q, mode, dtype), (trial, n, q)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_wide_gaps(dtype):
    # Neighbours far apart, where the formula's three roundings show most.
    rng = np.random.default_rng(733)
    for trial in range(100):
        n = int(rng.integers(2, 12))
        values = rng.normal(size=n) * 10.0 ** rng.integers(-20, 21, n)
        values = values.astype(dtype)
        q = float(rng.uniform())
        got = mt.quantile(mt.Tensor(values, dtype=dtype), q).item()
        assert got == _want(values, q, "linear", dtype), (trial, list(values), q)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_whole_and_half_positions(dtype):
    values = np.arange(11, dtype=dtype) * 1.5
    tensor = mt.Tensor(values, dtype=dtype)
    # `0.3 * 10` is the float64 3: rank 3 every way.
    for mode in MODES:
        assert mt.quantile(tensor, 0.3, interpolation=mode).item() == 4.5, mode
    # A half goes to the even rank.
    four = mt.Tensor(np.array([1.0, 2.0, 3.0, 4.0], dtype=dtype), dtype=dtype)
    assert mt.quantile(four, 0.5, interpolation="nearest").item() == 3.0
    nine = mt.Tensor(values[:10], dtype=dtype)
    assert mt.quantile(nine, 1 / 6, interpolation="nearest").item() == 3.0


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_axes_several_probabilities_and_nan(dtype):
    rng = np.random.default_rng(727)
    matrix = (rng.normal(size=(12, 31)) * 10.0 ** rng.uniform(-3, 3, (12, 31))).astype(
        dtype
    )
    tensor = mt.Tensor(matrix, dtype=dtype)
    probabilities = [0.1, 1 / 3, 0.5, 0.9, 0.999]
    for mode in MODES:
        got = mt.quantile(tensor, 0.3, dim=1, interpolation=mode).tolist()
        assert got == [_want(row, 0.3, mode, dtype) for row in matrix], mode
        got = mt.quantile(tensor, probabilities, dim=1, interpolation=mode).tolist()
        want = [[_want(row, q, mode, dtype) for row in matrix] for q in probabilities]
        assert got == want, mode
        got = mt.quantile(tensor, probabilities, interpolation=mode).tolist()
        assert got == [_want(matrix.ravel(), q, mode, dtype) for q in probabilities]
    # NaNs left out, a different count in each row.
    holes = matrix.copy()
    holes[rng.uniform(size=holes.shape) < 0.3] = np.nan
    holes[:, 0] = 1.0
    got = mt.nanquantile(mt.Tensor(holes, dtype=dtype), 0.3, dim=1).tolist()
    want = [_want(row[~np.isnan(row)], 0.3, "linear", dtype) for row in holes]
    assert got == want


def test_ends_and_specials():
    big = float(np.finfo(np.float64).max)
    pair = mt.tensor([-big, big], dtype="float64")
    assert mt.quantile(pair, 0.75).item() == _want(
        [-big, big], 0.75, "linear", "float64"
    )
    assert mt.quantile(pair, 0.5).item() == 0.0
    inf = math.inf
    values = mt.tensor([-inf, -inf, 1.0, 2.0, inf, inf], dtype="float64")
    assert mt.quantile(values, 0.3).item() == -inf
    assert mt.quantile(values, 0.5).item() == 1.5
    assert mt.quantile(mt.tensor([5.0], dtype="float64"), 0.3).item() == 5.0
    assert mt.quantile(mt.tensor([1.0, 2.0], dtype="float64"), 1.0).item() == 2.0
    assert mt.quantile(mt.tensor([1.0, 2.0], dtype="float64"), 0.0).item() == 1.0
    # A probability so small the position's error is below the subnormals.
    tiny = 5e-324
    pair = mt.tensor([0.0, big], dtype="float64")
    assert mt.quantile(pair, tiny).item() == _want(
        [0.0, big], tiny, "linear", "float64"
    )


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_gradients(dtype):
    x = mt.tensor([5.0, 1.0, 4.0, 2.0, 3.0], dtype=dtype, requires_grad=True)
    mt.quantile(x, 0.375).backward()
    # The position 1.5 lies between the values 2 and 3, half to each.
    assert x.grad.tolist() == [0.0, 0.0, 0.0, 0.5, 0.5]
    x = mt.tensor([5.0, 1.0, 4.0, 2.0, 3.0], dtype=dtype, requires_grad=True)
    mt.quantile(x, 0.5, interpolation="nearest").backward()
    assert x.grad.tolist() == [0.0, 0.0, 0.0, 0.0, 1.0]
