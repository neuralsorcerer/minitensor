# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""The 2-norm is the float nearest the exact `sqrt(sum(x ** 2))`.

A float64 norm summed rounded squares and rounded their root: one norm in
four of ordinary data was not the float64 nearest the exact one. A float32
norm rooted a float64 sum of the exact squares, right but for where that sum
fell beside a halfway point. Squares that left the float64 range had the
whole reduction redone scaled by the largest magnitude, rounding at the
divide and again at the multiply. The squares are now summed exactly where a
bound cannot settle the root, and rooted in whole numbers, so no scaling is
needed at either end of the range.
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


def _root(values, dtype: str) -> float:
    """The float nearest the exact root of the sum of squares."""

    total = sum(Fraction(float(v)) ** 2 for v in values)
    if total == 0:
        return 0.0
    # Far more bits than either float keeps, with whatever is left over
    # marked strictly inside the last place.
    shift = 1200
    whole = total.numerator * total.denominator * 4**shift
    root = math.isqrt(whole)
    scale = total.denominator * 2**shift
    value = Fraction(root, scale)
    if root * root != whole:
        value += Fraction(1, 2 * scale)
    if dtype == "float32":
        return _float32(value)
    try:
        return float(value)
    except OverflowError:
        return math.inf


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_norms_round_once(dtype):
    rng = np.random.default_rng(383)
    for _ in range(40):
        n = int(rng.integers(1, 3000))
        values = (rng.normal(size=n) * 2.0 ** rng.integers(-20, 20, size=n)).astype(
            dtype
        )
        assert mt.Tensor(values, dtype=dtype).norm().item() == _root(values, dtype)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_every_set_of_axes(dtype):
    rng = np.random.default_rng(389)
    values = rng.normal(loc=1.0, size=(12, 30, 9)).astype(dtype)
    tensor = mt.Tensor(values, dtype=dtype)
    for dims in ([0], [1], [2], [0, 1], [1, 2], [0, 2], [0, 1, 2]):
        got = tensor.norm(dim=dims).numpy().ravel().tolist()
        moved = np.moveaxis(values, dims, list(range(-len(dims), 0)))
        runs = moved.reshape(-1, int(np.prod([values.shape[d] for d in dims])))
        assert got == [_root(run, dtype) for run in runs], dims
    kept = tensor.norm(dim=[0, 2], keepdim=True)
    assert kept.shape == (1, 30, 1)


def test_the_ends_of_the_range_need_no_scaling():
    big, small = 2.0**600, 2.0**-600
    assert mt.Tensor([3 * big, 4 * big], dtype="float64").norm().item() == 5 * big
    assert mt.Tensor([3 * small, 4 * small], dtype="float64").norm().item() == 5 * small
    values = [1e200, 1e200, -3e199]
    assert mt.Tensor(values, dtype="float64").norm().item() == _root(values, "float64")
    tiny = [5e-324, 5e-324, 1e-320]
    assert mt.Tensor(tiny, dtype="float64").norm().item() == _root(tiny, "float64")
    largest = float(np.finfo(np.float64).max)
    assert mt.Tensor([largest, largest], dtype="float64").norm().item() == math.inf
    assert mt.Tensor([3e30, 4e30], dtype="float32").norm().item() == _root(
        np.array([3e30, 4e30], np.float32), "float32"
    )


def test_specials_and_zeros():
    assert mt.Tensor([0.0, -0.0], dtype="float64").norm().item() == 0.0
    assert mt.Tensor([-math.inf, 1.0], dtype="float64").norm().item() == math.inf
    assert math.isnan(mt.Tensor([math.nan, 1.0], dtype="float32").norm().item())


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_the_gradient_is_unchanged(dtype):
    values = mt.tensor([3.0, 4.0], dtype=dtype, requires_grad=True)
    values.norm().backward()
    assert np.allclose(values.grad.tolist(), [0.6, 0.8])
    rows = mt.tensor([[3.0, 4.0], [6.0, 8.0]], dtype=dtype, requires_grad=True)
    rows.norm(dim=0).sum().backward()
    assert np.allclose(
        rows.grad.tolist(),
        [[3 / 45**0.5, 4 / 80**0.5], [6 / 45**0.5, 8 / 80**0.5]],
    )
