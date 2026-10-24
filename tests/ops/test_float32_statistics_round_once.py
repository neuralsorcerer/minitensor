# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""Float32 variances and standard deviations are carried in float64.

Accumulated in float32, a variance was within a couple of ulps of the exact
one; carried in float64 and rounded once, the root of a standard deviation
included, it is the float32 nearest the exact answer, in every layout, with or
without a gradient, and over NaN-skipping slices too. These check that against
exact rational arithmetic.
"""

from __future__ import annotations

import math
from fractions import Fraction

import numpy as np
import pytest

import minitensor as mt


def _exact(values, unbiased=True) -> tuple[Fraction, Fraction]:
    exact = [Fraction(float(v)) for v in values if not math.isnan(v)]
    mean = sum(exact) / len(exact)
    return mean, sum((v - mean) ** 2 for v in exact) / (len(exact) - int(unbiased))


def _float32(value: float) -> float:
    return float(np.float32(value))


def _samples(seed, nan=False):
    rng = np.random.default_rng(seed)
    for _ in range(12):
        n = int(rng.integers(2, 1500))
        offset = float(rng.choice([0.0, 1.0, 100.0, 1e4]))
        spread = float(rng.choice([1.0, 1e-3]))
        values = (offset + rng.normal(0, spread, n)).astype(np.float32)
        if nan:
            values[rng.random(n) < 0.3] = np.nan
            values[0] = offset
        yield values


@pytest.mark.parametrize(
    "layout", ["flat", "row", "column", "gradient", "columns_of_two"]
)
def test_var_and_std_are_correctly_rounded(layout):
    for values in _samples(251):
        _, variance = _exact(values)
        if layout == "flat":
            tensor, dim = mt.Tensor(values), None
        elif layout == "row":
            tensor, dim = mt.Tensor(values.reshape(1, -1)), 1
        elif layout == "column":
            tensor, dim = mt.Tensor(values.reshape(-1, 1)), 0
        elif layout == "gradient":
            tensor, dim = mt.Tensor(values, requires_grad=True), None
        else:
            doubled = np.stack([values, values], axis=1)
            tensor, dim = mt.Tensor(doubled), 0
        var = tensor.var(dim=dim).detach().numpy().reshape(-1)
        std = tensor.std(dim=dim).detach().numpy().reshape(-1)
        assert set(var.tolist()) == {_float32(float(variance))}
        assert set(std.tolist()) == {_float32(math.sqrt(float(variance)))}


@pytest.mark.parametrize("layout", ["flat", "row", "column", "gradient"])
def test_the_nan_statistics_are_correctly_rounded(layout):
    for values in _samples(257, nan=True):
        mean, variance = _exact(values)
        if layout == "flat":
            tensor, dim = mt.Tensor(values), None
        elif layout == "row":
            tensor, dim = mt.Tensor(values.reshape(1, -1)), 1
        elif layout == "column":
            tensor, dim = mt.Tensor(values.reshape(-1, 1)), 0
        else:
            tensor, dim = mt.Tensor(values, requires_grad=True), None
        assert tensor.nanmean(dim=dim).detach().numpy().item() == _float32(float(mean))
        assert tensor.nanvar(dim=dim).detach().numpy().item() == _float32(
            float(variance)
        )
        assert tensor.nanstd(dim=dim).detach().numpy().item() == _float32(
            math.sqrt(float(variance))
        )


def test_int32_statistics_round_once():
    rng = np.random.default_rng(263)
    values = rng.integers(-(2**31), 2**31, 1001).astype(np.int32)
    _, variance = _exact(values.astype(np.float64))
    tensor = mt.Tensor(values, dtype="int32")
    assert tensor.var().item() == _float32(float(variance))
    assert tensor.std().item() == _float32(math.sqrt(float(variance)))


def test_a_float32_standard_deviation_past_its_variance():
    values = np.array([1e20, -1e20, 3e20], dtype=np.float32)
    _, variance = _exact(values)
    tensor = mt.Tensor(values)
    assert tensor.var().item() == math.inf
    assert tensor.std().item() == _float32(math.sqrt(float(variance)))
    assert tensor.nanstd().item() == _float32(math.sqrt(float(variance)))
    tiny = np.array([1e-45, 0.0, 3e-45], dtype=np.float32)
    _, small = _exact(tiny)
    assert mt.Tensor(tiny).std().item() == _float32(math.sqrt(float(small)))


def test_a_variance_over_no_axes():
    tensor = mt.Tensor([1.0, 2.0])
    assert all(math.isnan(v) for v in tensor.var(dim=[]).tolist())
    assert tensor.var(dim=[], unbiased=False).tolist() == [0.0, 0.0]
    assert tensor.std(dim=[], unbiased=False).tolist() == [0.0, 0.0]


def test_the_gradient_is_the_float32_one():
    rng = np.random.default_rng(269)
    values = rng.normal(3, 1, 64).astype(np.float32)
    x = mt.Tensor(values, requires_grad=True)
    x.std().backward()
    wide = values.astype(np.float64)
    want = (wide - wide.mean()) / ((len(wide) - 1) * wide.std(ddof=1))
    assert str(x.grad.dtype) == "float32"
    np.testing.assert_allclose(x.grad.numpy(), want, rtol=2e-6, atol=1e-7)
