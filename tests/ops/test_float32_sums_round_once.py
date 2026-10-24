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


def _exact(values) -> float:
    return float(sum(Fraction(float(v)) for v in values if not math.isnan(v)))


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
        want = float(np.float32(_exact(values)))
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
    want = values.astype(np.float64).sum(axis=tuple(dims), keepdims=keepdim)
    tensor = mt.Tensor(values, requires_grad=True)
    total = tensor.sum(dim=dims, keepdim=keepdim)
    np.testing.assert_array_equal(total.detach().numpy(), want.astype(np.float32))
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
