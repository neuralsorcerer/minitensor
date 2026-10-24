# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""`cov` and `corrcoef` keep the digits and the range their answers have.

An integer input was converted before it was centred, so int64s past `2^53` a
step apart became one float and `cov([2^53, 2^53 + 1, 2^53 + 2])` was 2. A
float covariance entry whose products left the range though it did not came out
infinite or zero, and `corrcoef`, which does not depend on the scale of a row
at all, was NaN for the float32 rows `[1e20, 2e20, 4e20]`.
"""

from __future__ import annotations

import math
from fractions import Fraction

import numpy as np
import pytest

import minitensor as mt


def _exact_cov(rows) -> list[list[Fraction]]:
    exact = [
        [
            Fraction(int(v)) if isinstance(v, (int, np.integer)) else Fraction(float(v))
            for v in row
        ]
        for row in rows
    ]
    n = len(exact[0])
    means = [sum(row) / n for row in exact]
    return [
        [
            sum((a - ma) * (b - mb) for a, b in zip(ra, rb)) / (n - 1)
            for rb, mb in zip(exact, means)
        ]
        for ra, ma in zip(exact, means)
    ]


def test_int64_past_two_to_the_53():
    big = 2**53
    assert mt.cov(mt.Tensor([big, big + 1, big + 2], dtype="int64")).item() == 1.0
    rows = [[big, big + 1, big + 2], [-(2**62), -(2**62) + 2, 2**62]]
    got = mt.cov(mt.Tensor(np.array(rows, dtype=np.int64), dtype="int64")).tolist()
    want = _exact_cov(rows)
    # The deviations are exact; what is left is the float64 arithmetic of the
    # covariance itself, a rounding or two.
    for got_row, want_row in zip(got, want):
        for value, exact in zip(got_row, want_row):
            assert math.isclose(value, float(exact), rel_tol=4e-16)


def test_int32_past_float32_precision():
    base = 2**25
    values = np.array([base, base + 2, base + 4], dtype=np.int32)
    assert mt.cov(mt.Tensor(values, dtype="int32")).item() == 4.0


def test_integer_correlation_is_exact():
    big = 2**53
    rows = np.array(
        [[big, big + 1, big + 2], [big + 5, big + 3, big + 1]], dtype=np.int64
    )
    got = mt.corrcoef(mt.Tensor(rows, dtype="int64")).tolist()
    assert got == [[1.0, -1.0], [-1.0, 1.0]]


@pytest.mark.parametrize(
    "dtype,scale",
    [("float32", 1e20), ("float32", 1e-20), ("float64", 1e200), ("float64", 1e-200)],
)
def test_correlation_does_not_depend_on_scale(dtype, scale):
    base = np.array([[1.0, 2.0, 4.0, 3.0], [1.0, 3.0, 2.0, 5.0], [4.0, 1.0, 1.0, 2.0]])
    want = np.corrcoef(base)
    got = mt.corrcoef(mt.Tensor((base * scale).astype(dtype), dtype=dtype)).numpy()
    tolerance = 1e-6 if dtype == "float32" else 1e-14
    np.testing.assert_allclose(got, want, rtol=tolerance, atol=tolerance)
    assert np.all(np.diag(got) == 1.0)


def test_rows_of_different_scales():
    rows = np.array([[1e20, 2e20, 4e20], [1e-20, 3e-20, 2e-20]], dtype=np.float32)
    got = mt.cov(mt.Tensor(rows)).tolist()
    want = _exact_cov(rows)
    # The first variance is past float32 and is infinite, honestly; the rest
    # are in range and come out as such.
    assert got[0][0] == math.inf
    assert math.isclose(got[0][1], float(want[0][1]), rel_tol=1e-6)
    assert got[1][0] == got[0][1]
    assert got[1][1] == float(np.float32(float(want[1][1])))


def test_the_diagonal_is_one_and_a_constant_row_nan():
    rows = np.array(
        [[0.1, 0.7, 0.3], [0.2, 0.25, 0.9], [1.0, 1.0, 1.0]], dtype=np.float32
    )
    got = mt.corrcoef(mt.Tensor(rows)).numpy()
    assert got[0, 0] == 1.0 and got[1, 1] == 1.0
    assert np.isnan(got[2]).all() and np.isnan(got[:, 2]).all()
    assert np.isnan(mt.corrcoef(mt.Tensor([2.0, 2.0])).item())
    assert (
        mt.corrcoef(mt.Tensor(np.array([1e20, 2e20], dtype=np.float32))).item() == 1.0
    )


def test_in_range_covariances_are_unchanged():
    rng = np.random.default_rng(229)
    rows = rng.normal(0, 1, (4, 50))
    got = mt.cov(mt.Tensor(rows, dtype="float64")).numpy()
    np.testing.assert_allclose(got, np.cov(rows), rtol=1e-12)
    weights = rng.integers(1, 4, 50)
    reliability = rng.uniform(0.5, 1.5, 50)
    got = mt.cov(
        mt.Tensor(rows * 1e200, dtype="float64"),
        fweights=mt.Tensor(weights.astype(np.int64), dtype="int64"),
        aweights=mt.Tensor(reliability, dtype="float64"),
    ).numpy()
    reference = np.cov(rows, fweights=weights, aweights=reliability)
    # Scaled by 1e200 the covariance is past float64, and is infinite.
    assert np.isinf(got).all()
    got = mt.cov(
        mt.Tensor(rows * 1e150, dtype="float64"),
        fweights=mt.Tensor(weights.astype(np.int64), dtype="int64"),
        aweights=mt.Tensor(reliability, dtype="float64"),
    ).numpy()
    np.testing.assert_allclose(got, reference * 1e300, rtol=1e-12)


def test_gradients_flow_through_the_rescue():
    rows = np.array([[1.0, 2.0, 4.0], [1.0, 3.0, 2.0]])
    small = mt.Tensor(rows, dtype="float64", requires_grad=True)
    mt.corrcoef(small).sum().backward()
    large = mt.Tensor(rows * 1e200, dtype="float64", requires_grad=True)
    mt.corrcoef(large).sum().backward()
    np.testing.assert_allclose(
        large.grad.numpy() * 1e200, small.grad.numpy(), rtol=1e-12
    )


@pytest.mark.parametrize("offset", [0.0, 100.0, 1e4])
def test_float32_is_carried_in_float64(offset):
    rng = np.random.default_rng(281)
    rows = (offset + rng.normal(0, 1, (3, 4000))).astype(np.float32)
    wide = rows.astype(np.float64)
    got = mt.cov(mt.Tensor(rows))
    assert str(got.dtype) == "float32"
    np.testing.assert_array_equal(got.numpy(), np.cov(wide).astype(np.float32))
    correlation = mt.corrcoef(mt.Tensor(rows)).numpy()
    want = np.corrcoef(wide)
    np.testing.assert_allclose(correlation, want, rtol=6e-8, atol=6e-8)
    assert np.all(np.diag(correlation) == 1.0)


def test_float64_centring_is_corrected():
    rng = np.random.default_rng(283)
    rows = 1e8 + rng.normal(0, 1e-6, (2, 400))
    got = mt.cov(mt.Tensor(rows, dtype="float64")).numpy()
    want = np.array([[float(v) for v in row] for row in _exact_cov(rows)])
    np.testing.assert_allclose(got, want, rtol=1e-14)


def test_float32_gradients_come_back_float32():
    rng = np.random.default_rng(293)
    rows = rng.normal(0, 1, (3, 7))
    narrow = mt.Tensor(rows.astype(np.float32), requires_grad=True)
    mt.corrcoef(narrow).sum().backward()
    wide = mt.Tensor(rows, dtype="float64", requires_grad=True)
    mt.corrcoef(wide).sum().backward()
    assert str(narrow.grad.dtype) == "float32"
    np.testing.assert_allclose(narrow.grad.numpy(), wide.grad.numpy(), atol=1e-6)
