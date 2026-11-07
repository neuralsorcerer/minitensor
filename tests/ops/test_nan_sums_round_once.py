# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""`nansum` and `nanmean` are the float nearest the exact sum and mean of the
values that are not NaN.

A float64 one added the values plainly: of sixty small tensors of ordinary data
with a fifth of their entries NaN, two thirds of the sums over every element
and half of those along an axis were an ulp or more out, and over several axes
each axis rounded its partial sums before the next added them. Each is now
the exact total of the values kept, rounded once; a slice of nothing but NaN
sums to zero and has a mean of NaN, and a slice holding infinities of both
signs sums to NaN.
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


def _expected(values, mean: bool, dtype: str) -> float:
    kept = [Fraction(float(v)) for v in np.ravel(values) if not math.isnan(v)]
    if not kept:
        return math.nan if mean else 0.0
    total = sum(kept)
    value = total / len(kept) if mean else total
    return _float32(value) if dtype == "float32" else float(value)


def _same(got, want):
    return got == want or (math.isnan(got) and math.isnan(want))


def _data(rng, shape, dtype, trial):
    values = rng.normal(size=shape) * 2.0 ** rng.integers(-20, 21, size=shape)
    if trial % 3 == 0:
        values = 1e4 + rng.normal(size=shape)
    values = values.astype(dtype)
    values[rng.random(shape) < 0.2] = np.nan
    return values


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_every_layout(dtype):
    rng = np.random.default_rng(523)
    for trial in range(30):
        shape = (int(rng.integers(2, 30)), int(rng.integers(2, 30)))
        values = _data(rng, shape, dtype, trial)
        tensor = mt.Tensor(values, dtype=dtype)
        for mean, reduce in ((False, tensor.nansum), (True, tensor.nanmean)):
            assert _same(reduce().item(), _expected(values, mean, dtype)), trial
            got = reduce(dim=1).tolist()
            want = [_expected(values[i], mean, dtype) for i in range(shape[0])]
            assert all(_same(g, w) for g, w in zip(got, want)), (trial, got, want)
            got = reduce(dim=0, keepdim=True).tolist()[0]
            want = [_expected(values[:, j], mean, dtype) for j in range(shape[1])]
            assert all(_same(g, w) for g, w in zip(got, want)), (trial, got, want)
        # Over two axes, each output's values summed as one run.
        cube = np.stack([values, values[::-1], -values], axis=-1)
        tensor = mt.Tensor(cube, dtype=dtype)
        for mean, reduce in ((False, tensor.nansum), (True, tensor.nanmean)):
            got = reduce(dim=[0, 1]).tolist()
            want = [_expected(cube[:, :, k], mean, dtype) for k in range(3)]
            assert all(_same(g, w) for g, w in zip(got, want)), (trial, got, want)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_edges(dtype):
    nan, inf = math.nan, math.inf

    def tensor(values):
        return mt.tensor(values, dtype=dtype)

    assert tensor([nan, nan]).nansum().item() == 0.0
    assert math.isnan(tensor([nan, nan]).nanmean().item())
    # A NaN made of infinities of both signs is the answer, not one to skip.
    assert math.isnan(tensor([inf, nan, -inf]).nansum().item())
    assert tensor([inf, nan, 1.0]).nanmean().item() == inf
    # Values that cancel exactly, around a NaN.
    big = 1e300 if dtype == "float64" else 1e30
    assert tensor([big, nan, 1.0, -big]).nansum().item() == 1.0
    assert tensor([big, nan, 1.0, -big]).nanmean().item() == _expected(
        [big, 1.0, -big], True, dtype
    )
    # Reducing over no axes keeps each entry.
    got = tensor([1.5, nan]).nanmean(dim=[]).tolist()
    assert got[0] == 1.5 and math.isnan(got[1])


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_gradients(dtype):
    x = mt.tensor(
        [[1.0, math.nan, 3.0], [math.nan, math.nan, math.nan]],
        dtype=dtype,
        requires_grad=True,
    )
    x.nansum(dim=1).sum().backward()
    assert x.grad.tolist() == [[1.0, 0.0, 1.0], [0.0, 0.0, 0.0]]
    assert str(x.grad.dtype) == dtype
    x.grad = None
    x.nanmean(dim=1).sum().backward()
    assert x.grad.tolist() == [[0.5, 0.0, 0.5], [0.0, 0.0, 0.0]]
    assert str(x.grad.dtype) == dtype
