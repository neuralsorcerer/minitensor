# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""A float32 scan is each running value rounded once.

Carried in float32 the running value rounded at every step, and the error grew
with the length. Over a million values the last element of `cumsum` was 7e-6
from the exact total -- eighty times the error `sum` makes -- so
`cumsum(x)[-1]` and `x.sum()` disagreed in the fifth digit; `logcumsumexp`
ended 0.0125 off, a sum wrong by more than one percent; and over 100,000
factors near one `cumprod` drifted 9e-6 from the exact product. The running
value is now carried in float64, and each element is the float32 nearest it.
"""

from __future__ import annotations

import numpy as np
import pytest

import minitensor as mt


def test_every_partial_sum_is_the_nearest_float32():
    values = np.random.default_rng(41).uniform(0, 1, 1_000_000).astype(np.float32)
    exact = np.cumsum(values.astype(np.float64)).astype(np.float32)
    got = mt.cumsum(mt.Tensor(values), 0).numpy()
    np.testing.assert_array_equal(got, exact)
    # And so within an ulp of `sum`, whose pairwise float32 total is itself
    # within one of the exact one.
    total = np.float32(mt.Tensor(values).sum().item())
    assert abs(float(got[-1]) - float(total)) <= float(np.spacing(total))


@pytest.mark.parametrize("dim", [0, 1])
def test_along_either_axis_of_a_matrix(dim):
    values = np.random.default_rng(43).standard_normal((800, 300)).astype(np.float32)
    exact = np.cumsum(values.astype(np.float64), dim).astype(np.float32)
    np.testing.assert_array_equal(mt.cumsum(mt.Tensor(values), dim).numpy(), exact)


def test_signed_zero_overflow_and_nan():
    assert np.signbit(mt.cumsum(mt.Tensor([-0.0]), 0).numpy()[0])
    # Past float32's range a partial sum is infinite where it is, and the
    # next one comes back with the total.
    assert mt.cumsum(mt.Tensor([3e38, 3e38, -3e38]), 0).tolist()[1:] == [
        float("inf"),
        float(np.float32(3e38)),
    ]
    assert np.isnan(mt.cumsum(mt.Tensor([1.0, float("nan"), 1.0]), 0).numpy()[1:]).all()


def test_the_gradient_is_the_reversed_running_sum():
    weights = mt.Tensor(np.ones((2, 5), dtype=np.float32), requires_grad=True)
    mt.cumsum(weights, 1).sum().backward()
    assert weights.grad.tolist() == [[5.0, 4.0, 3.0, 2.0, 1.0]] * 2


def test_every_running_product_is_the_nearest_float32():
    rng = np.random.default_rng(47)
    factors = (1 + rng.normal(0, 1e-3, 100_000)).astype(np.float32)
    exact = np.cumprod(factors.astype(np.float64)).astype(np.float32)
    np.testing.assert_array_equal(mt.cumprod(mt.Tensor(factors), 0).numpy(), exact)
    matrix = rng.standard_normal((400, 30)).astype(np.float32)
    for dim in (0, 1):
        exact = np.cumprod(matrix.astype(np.float64), dim).astype(np.float32)
        np.testing.assert_array_equal(mt.cumprod(mt.Tensor(matrix), dim).numpy(), exact)


def test_a_running_product_comes_back_from_past_the_range():
    got = mt.cumprod(mt.Tensor([1e30, 1e30, 1e-30, 0.0]), 0).tolist()
    assert got == [float(np.float32(1e30)), float("inf"), float(np.float32(1e30)), 0.0]


def test_every_running_log_sum_is_the_nearest_float32():
    rng = np.random.default_rng(53)
    values = rng.standard_normal(1_000_000).astype(np.float32)
    exact = np.logaddexp.accumulate(values.astype(np.float64))
    got = mt.logcumsumexp(mt.Tensor(values), 0).numpy().astype(np.float64)
    # The float64 kernel's own rounding can land a tie on the other float32.
    assert np.all(np.abs(got - exact) <= np.spacing(np.abs(exact).astype(np.float32)))
    matrix = rng.standard_normal((2000, 7)).astype(np.float32)
    exact = np.logaddexp.accumulate(matrix.astype(np.float64), 0)
    got = mt.logcumsumexp(mt.Tensor(matrix), 0).numpy().astype(np.float64)
    assert np.all(np.abs(got - exact) <= np.spacing(np.abs(exact).astype(np.float32)))


def test_a_log_sum_of_nothing_stays_minus_infinity():
    inf = float("inf")
    got = mt.logcumsumexp(mt.Tensor([-inf, -inf, 0.0, -inf]), 0).tolist()
    assert got == [-inf, -inf, 0.0, 0.0]
