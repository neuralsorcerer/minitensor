# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""A variance does not carry the rounding of the mean it is centred on.

A mean rounded to its dtype is off by up to half an ulp, and every squared
deviation from it carries the offset: `n * offset^2` too much in all, which is
most of the answer for data that varies over a few ulps of its mean. Float32
readings near 1 that vary by 1e-7 had a variance 4e-4 high, and the float32s
`999 x 2.0` and one `2.0 + 2^-22` a variance a thousandth high. The deviations'
own sum measures the offset, and its square over `n` is taken back out.
"""

from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest

import minitensor as mt


def _exact_var(values, unbiased=True) -> float:
    exact = [Fraction(float(v)) for v in values]
    mean = sum(exact) / len(exact)
    return float(sum((v - mean) ** 2 for v in exact) / (len(exact) - int(unbiased)))


def _offset_data(dtype, offset, spread, n, seed):
    rng = np.random.default_rng(seed)
    return (offset + rng.normal(0, spread, n)).astype(dtype)


@pytest.mark.parametrize(
    "offset,spread", [(1.0, 1e-7), (1000.0, 1e-4), (1e4, 1e-3), (0.0, 1.0)]
)
@pytest.mark.parametrize("path", ["fused", "gradient", "nan", "nan_gradient"])
def test_float32_offset_data(offset, spread, path):
    values = _offset_data(np.float32, offset, spread, 4000, 211)
    want = _exact_var(values)
    grad = path.endswith("gradient")
    tensor = mt.Tensor(values, requires_grad=grad)
    got = (tensor.nanvar() if path.startswith("nan") else tensor.var()).item()
    # Within a few roundings of float32, where the uncorrected variance was
    # thousands of ulps away.
    assert abs(got - want) / want < 4e-7, (got, want)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_a_spread_of_one_ulp(dtype):
    np_dtype = np.dtype(dtype).type
    a = np_dtype(2.0)
    b = np.nextafter(a, np_dtype(3.0))
    values = np.array([a] * 999 + [b], dtype=dtype)
    want = _exact_var(values)
    tolerance = 1e-6 if dtype == "float32" else 1e-14
    for got in (
        mt.Tensor(values, dtype=dtype).var().item(),
        mt.Tensor(values, dtype=dtype, requires_grad=True).var().item(),
        mt.Tensor(values, dtype=dtype).nanvar().item(),
        mt.Tensor(values.reshape(-1, 1), dtype=dtype).var(dim=0).item(),
    ):
        assert abs(got - want) / want < tolerance, (got, want)


@pytest.mark.parametrize("shape,dim", [((6, 500), 1), ((500, 6), 0), ((4, 500, 3), 1)])
def test_every_layout_is_corrected(shape, dim):
    rng = np.random.default_rng(223)
    values = (1000.0 + rng.normal(0, 1e-4, shape)).astype(np.float32)
    got = mt.Tensor(values).var(dim=dim).numpy()
    moved = np.moveaxis(values, dim, -1)
    want = np.array([_exact_var(row) for row in moved.reshape(-1, shape[dim])])
    np.testing.assert_allclose(got.reshape(-1), want, rtol=4e-7)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_constant_data_has_zero_variance(dtype):
    for value in (0.1, 1e10, -3.7):
        values = np.full(1000, value, dtype=dtype)
        assert mt.Tensor(values, dtype=dtype).var().item() == 0.0
        assert mt.Tensor(values, dtype=dtype, requires_grad=True).var().item() == 0.0
        assert mt.Tensor(values, dtype=dtype).nanvar().item() == 0.0


def test_the_gradient_is_unchanged():
    rng = np.random.default_rng(227)
    values = rng.normal(5, 2, 50)
    x = mt.Tensor(values, dtype="float64", requires_grad=True)
    x.var().backward()
    want = 2 * (values - values.mean()) / (len(values) - 1)
    np.testing.assert_allclose(x.grad.numpy(), want, rtol=1e-12, atol=1e-15)
    y = mt.Tensor(np.append(values, np.nan), dtype="float64", requires_grad=True)
    y.nanvar().backward()
    np.testing.assert_allclose(y.grad.numpy()[:-1], want, rtol=1e-12, atol=1e-15)
    assert y.grad.numpy()[-1] == 0.0


def test_undefined_variances_stay_undefined():
    nan = float("nan")
    assert np.isnan(mt.Tensor([1.0]).var().item())
    assert np.isnan(mt.Tensor([nan, nan]).nanvar().item())
    assert np.isnan(mt.Tensor([float("inf"), 1.0]).var().item())
    assert np.isnan(mt.Tensor([float("inf"), 1.0]).nanvar().item())
    assert np.isnan(mt.Tensor([nan, 1.0]).var().item())
