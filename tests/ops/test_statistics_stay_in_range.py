# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""A statistic in range is computed in range, whatever its intermediates.

A mean sums before it divides and a variance squares before it averages, so
the mean of `[1e308, 1e308]` overflowed to infinity and the standard deviation
of the float32s `[1e20, -1e20]` did too, while the squares of `1e-25` fell to
zero. Such a slice is now taken again at a power of two near its largest
magnitude, which scales exactly. And int64s above 2^53 converted before they
were subtracted: `var([2^53, 2^53 + 1, 2^53 + 2])` was 2, not 1.
"""

from __future__ import annotations

import math
from fractions import Fraction

import numpy as np
import pytest

import minitensor as mt


def _exact_var(values, unbiased=True) -> Fraction:
    exact = [Fraction(float(v)) for v in values]
    mean = sum(exact) / len(exact)
    return sum((v - mean) ** 2 for v in exact) / (len(exact) - int(unbiased))


def _float(value: Fraction) -> float:
    # A Fraction past the float64 range, or below it, as the nearest float.
    if value == 0:
        return 0.0
    shift = value.numerator.bit_length() - value.denominator.bit_length()
    try:
        return math.ldexp(float(value / Fraction(2) ** shift), shift)
    except OverflowError:
        return math.inf


def _sqrt(value: Fraction) -> float:
    shift = (value.numerator.bit_length() - value.denominator.bit_length()) // 2
    return math.ldexp(math.sqrt(float(value / Fraction(4) ** shift)), shift)


@pytest.mark.parametrize(
    "dtype,scale",
    [
        ("float32", 1e20),
        ("float32", 1e-25),
        ("float32", 1e35),
        ("float64", 1e200),
        ("float64", 1e-200),
        ("float64", 1e300),
    ],
)
def test_std_and_var_of_values_far_from_one(dtype, scale):
    base = np.array([1.0, -1.0, 0.5, 2.0], dtype=np.float64)
    values = (base * scale).astype(dtype)
    tensor = mt.Tensor(values, dtype=dtype)
    tolerance = 1e-6 if dtype == "float32" else 1e-14
    variance = _exact_var(values)
    got_std = tensor.std().item()
    assert math.isfinite(got_std) and got_std > 0
    assert math.isclose(got_std, _sqrt(variance), rel_tol=tolerance)
    # The variance itself is in range for these, and so is computed.
    want_var = _float(variance)
    limits = np.finfo(dtype)
    if float(limits.tiny) <= want_var <= float(limits.max):
        assert math.isclose(tensor.var().item(), want_var, rel_tol=tolerance)


def test_a_variance_past_the_range_is_infinite():
    # The answer itself overflows; nothing rescues it.
    got = mt.Tensor([1e20, -1e20], dtype="float32").var().item()
    assert got == math.inf


@pytest.mark.parametrize("dtype,exponent", [("float64", 1023), ("float32", 127)])
def test_the_mean_of_large_values(dtype, exponent):
    # Values whose sums are exact once scaled, so the mean is exact too.
    big = math.ldexp(1.5, exponent)
    got = mt.Tensor([big, big, big, big], dtype=dtype).mean().item()
    assert got == big
    got = mt.Tensor([big, big / 2], dtype=dtype).mean().item()
    assert got == big * 0.75


def test_rescuing_works_per_slice():
    x = mt.Tensor([[1e300, -1e300, 1e300], [1.0, 2.0, 3.0]], dtype="float64")
    got = x.std(dim=1).tolist()
    assert math.isclose(got[0], math.sqrt(4 / 3) * 1e300, rel_tol=1e-14)
    assert got[1] == 1.0
    mean = mt.Tensor([[1e308, 1e308], [1.0, 3.0]], dtype="float64").mean(dim=1)
    assert mean.tolist() == [1e308, 2.0]


def test_non_finite_inputs_still_give_non_finite_statistics():
    assert math.isnan(mt.Tensor([math.inf, 1.0], dtype="float64").std().item())
    assert mt.Tensor([math.inf, 1.0], dtype="float64").mean().item() == math.inf
    assert math.isnan(mt.Tensor([math.nan, 1.0], dtype="float64").var().item())


def test_constant_slices_have_zero_variance():
    for dtype in ("float32", "float64"):
        assert mt.Tensor([1e-30] * 5, dtype=dtype).var().item() == 0.0
        assert mt.Tensor([7.0] * 5, dtype=dtype).std().item() == 0.0


def test_large_int64_variance_is_exact():
    base = 2**53
    values = [base, base + 1, base + 2]
    tensor = mt.Tensor(np.array(values, dtype=np.int64), dtype="int64")
    assert tensor.var().item() == 1.0
    assert tensor.std().item() == 1.0
    rows = np.array([[base, base + 2], [-(2**62), -(2**62) + 4]], dtype=np.int64)
    got = mt.Tensor(rows, dtype="int64").var(dim=1, unbiased=False).tolist()
    assert got == [1.0, 4.0]


def test_int32_variance_of_values_past_float32_precision():
    base = 2**25
    values = np.array([base, base + 2, base + 4], dtype=np.int32)
    assert mt.Tensor(values, dtype="int32").var().item() == 4.0


def test_rescued_statistics_carry_gradients():
    values = np.array([1e200, -1e200, 3e200], dtype=np.float64)
    x = mt.Tensor(values, dtype="float64", requires_grad=True)
    x.std().backward()
    mean = values.mean()
    std = _sqrt(_exact_var(values))
    want = (values - mean) / (len(values) - 1) / std
    np.testing.assert_allclose(x.grad.numpy(), want, rtol=1e-12)


def test_cosine_similarity_of_large_vectors():
    F = mt.functional
    a = mt.Tensor([[1e30, 2e30, 3e30]], dtype="float32")
    b = mt.Tensor([[2e30, 4e30, 6e30]], dtype="float32")
    got = F.cosine_similarity(a, b, dim=1).item()
    assert math.isclose(got, 1.0, rel_tol=1e-6)
    # Small vectors, whose norms are above the floor but whose squared norms
    # multiplied together are below the float32 range.
    small = mt.Tensor([[1e-7, 0.0]], dtype="float32")
    other = mt.Tensor([[1e-7, 1e-7]], dtype="float32")
    got = F.cosine_similarity(small, other, dim=1).item()
    assert math.isclose(got, 1 / math.sqrt(2), rel_tol=1e-6)
