# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""A float32 sum is carried in float64 and rounded once.

Summed in float32 it took a rounding per addition against a growing total, and
landed up to 45 ulps from the exact sum of offset data; a sum over several axes
rounded once an axis on top. And `nansum` over several axes skipped NaN on all
of them, including the NaN an earlier axis made of `inf + -inf`, so it could
lose a NaN that was the answer.
"""

from __future__ import annotations

import math
from fractions import Fraction

import numpy as np
import pytest

import minitensor as mt


def _exact(values) -> Fraction:
    return sum(Fraction(float(v)) for v in values if not math.isnan(v))


def _samples(seed, nan=False):
    rng = np.random.default_rng(seed)
    for _ in range(15):
        n = int(rng.integers(2, 3000))
        offset = float(rng.choice([0.0, 1.0, 100.0, 1e4]))
        values = (offset + rng.normal(0, 1, n)).astype(np.float32)
        if nan:
            values[rng.random(n) < 0.2] = np.nan
        yield values


@pytest.mark.parametrize("layout", ["flat", "row", "column", "gradient"])
@pytest.mark.parametrize("name", ["sum", "nansum"])
def test_sums_are_correctly_rounded(layout, name):
    for values in _samples(271, nan=name == "nansum"):
        want = _float32(_exact(values))
        if layout == "flat":
            tensor, dim = mt.Tensor(values), None
        elif layout == "row":
            tensor, dim = mt.Tensor(values.reshape(1, -1)), 1
        elif layout == "column":
            tensor, dim = mt.Tensor(values.reshape(-1, 1)), 0
        else:
            tensor, dim = mt.Tensor(values, requires_grad=True), None
        got = getattr(tensor, name)(dim=dim).detach().numpy().item()
        assert got == want


@pytest.mark.parametrize("dims", [[0, 1], [1, 2], [0, 2]])
@pytest.mark.parametrize("keepdim", [False, True])
def test_several_axes_round_once(dims, keepdim):
    rng = np.random.default_rng(277)
    values = (1e4 + rng.normal(0, 1, (37, 53, 11))).astype(np.float32)
    moved = np.moveaxis(values, dims, list(range(-len(dims), 0)))
    runs = moved.reshape(-1, int(np.prod([values.shape[d] for d in dims])))
    want = [_float32(_exact(run.tolist())) for run in runs]
    tensor = mt.Tensor(values, requires_grad=True)
    total = tensor.sum(dim=dims, keepdim=keepdim)
    assert total.detach().numpy().ravel().tolist() == want
    assert total.shape == tuple(
        1 if d in dims else n
        for d, n in enumerate(values.shape)
        if keepdim or d not in dims
    )
    total.sum().backward()
    np.testing.assert_array_equal(tensor.grad.numpy(), np.ones_like(values))
    with_nan = values.copy()
    with_nan[::3, ::2] = np.nan
    want = np.nansum(with_nan.astype(np.float64), axis=tuple(dims), keepdims=keepdim)
    got = mt.Tensor(with_nan).nansum(dim=dims, keepdim=keepdim).numpy()
    np.testing.assert_array_equal(got, want.astype(np.float32))


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_nansum_keeps_the_nan_an_infinity_makes(dtype):
    values = [[[math.inf], [-math.inf]], [[1.0], [2.0]]]
    tensor = mt.Tensor(values, dtype=dtype)
    assert math.isnan(tensor.nansum(dim=[0, 1]).item())
    assert math.isnan(tensor.nansum(dim=[1, 0], keepdim=True).reshape([]).item())
    skipped = mt.Tensor([[[1.0], [math.nan]], [[1.0], [2.0]]], dtype=dtype)
    assert skipped.nansum(dim=[0, 1]).item() == 4.0


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_nanmean_keeps_the_nan_an_infinity_makes(dtype):
    values = [[[math.inf, 1.0], [2.0, math.nan]], [[-math.inf, 3.0], [1.0, 1.0]]]
    got = mt.Tensor(values, dtype=dtype).nanmean(dim=[0, 1]).tolist()
    assert math.isnan(got[0]) and math.isclose(got[1], 5.0 / 3.0, rel_tol=1e-7)
    kept = mt.Tensor(values, dtype=dtype).nanmean(dim=[1, 0], keepdim=True)
    assert math.isnan(kept.reshape([2]).tolist()[0])
    traced = mt.Tensor(values, dtype=dtype, requires_grad=True).nanmean(dim=[0, 1])
    assert math.isnan(traced.tolist()[0])


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


def test_cancellation_and_halfway_points_are_rounded_once():
    """Narrowed from float64, a sum lost what fell below float64's last bit
    where the values cancelled, and rounded twice where the float64 sum fell
    on a float32 halfway point: `1 + 2^-24 + 2^-60` came out `1`."""

    def tensor(values):
        return mt.Tensor(np.asarray(values, dtype=np.float32), dtype="float32")

    assert tensor([1.0, 2.0**-24, 2.0**-60]).sum().item() == 1.0 + 2.0**-23
    assert tensor([1.0, 2.0**-24]).sum().item() == 1.0
    assert tensor([1e30, 1e-10, -1e30]).sum().item() == float(np.float32(1e-10))
    rng = np.random.default_rng(1601)
    for round_ in range(120):
        n = int(rng.integers(2, 2000))
        if round_ % 2:
            scales = 2.0 ** rng.integers(-60, 60, size=n)
            values = (rng.normal(size=n) * scales).astype(np.float32)
            values = np.append(values, -values[:-1]).astype(np.float32)
        else:
            big = (rng.normal(size=n) * 1e30).astype(np.float32)
            small = (rng.normal(size=3) * 1e-10).astype(np.float32)
            values = np.concatenate([big, small, -big]).astype(np.float32)
        total = sum(Fraction(float(v)) for v in values)
        t = tensor(values)
        assert t.sum().item() == _float32(total)
        assert t.mean().item() == _float32(total / len(values))
        rows = tensor(np.stack([values, values[::-1]]))
        assert rows.sum(dim=1).tolist() == [_float32(total)] * 2
        assert rows.T.contiguous().sum(dim=0).tolist() == [_float32(total)] * 2
