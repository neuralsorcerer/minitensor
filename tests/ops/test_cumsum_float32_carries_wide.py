# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""A float32 running sum is each partial sum rounded once.

Carried in float32 the running total rounded at every step, and over a
million values the last element was 7e-6 from the exact total -- eighty times
the error `sum` makes -- so `cumsum(x)[-1]` and `x.sum()` disagreed in the
fifth digit. The total is now carried in float64, and each element is the
float32 nearest its partial sum.
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
