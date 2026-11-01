# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""A float64 `sum` and `mean` over every element are rounded once.

Added in lanes and folded pairwise, a float64 sum was accurate but not the
float64 nearest the exact sum: about half the sums of ordinary data were off,
by up to tens of ulps where the values cancel, and `mean` divided that and
rounded again. A sum over every axis named one at a time rounded once per
axis. An int64 mean summed in float64 rounded each value past 2^53 first.
Each is now the exact value rounded once, whatever order the additions ran
in.
"""

from __future__ import annotations

import math
from fractions import Fraction

import numpy as np
import pytest

import minitensor as mt


def _tensor(values):
    return mt.Tensor(np.asarray(values, dtype=np.float64), dtype="float64")


@pytest.mark.parametrize("n", [1, 7, 100, 10_000, 300_000])
def test_sum_and_mean_are_the_exact_ones_rounded(n):
    rng = np.random.default_rng(n)
    for location in (0.0, 1.0, 1e3):
        values = rng.normal(loc=location, size=n)
        tensor = _tensor(values)
        assert tensor.sum().item() == math.fsum(values.tolist())
        exact_mean = sum(Fraction(v) for v in values.tolist()) / n
        assert tensor.mean().item() == float(exact_mean)


def test_cancellation_and_extremes():
    assert _tensor([1.0, 1e100, 1.0, -1e100]).sum().item() == 2.0
    assert _tensor([0.1] * 10).sum().item() == 1.0
    # The exact sum is in range though the running one is not.
    largest = np.finfo(np.float64).max
    assert _tensor([largest, largest, -largest]).sum().item() == largest
    assert _tensor([largest, largest]).mean().item() == largest
    assert _tensor([largest, largest]).sum().item() == math.inf
    assert math.isnan(_tensor([math.inf, -math.inf]).sum().item())
    assert _tensor([0.1, 0.2, 0.3]).mean().item() == 0.2


def test_every_axis_is_one_sum():
    rng = np.random.default_rng(5)
    values = rng.normal(size=(300, 400))
    tensor = _tensor(values)
    exact = math.fsum(values.ravel().tolist())
    assert tensor.sum(dim=[0, 1]).item() == exact
    assert tensor.sum().item() == exact
    assert tensor.sum(dim=[0, 1], keepdim=True).shape == (1, 1)
    assert tensor.mean(dim=[0, 1]).item() == float(
        sum(Fraction(v) for v in values.ravel().tolist()) / values.size
    )


def test_int64_mean_divides_the_exact_sum():
    values = [2**60 + 1, 2**60 + 3, 2**60 + 8, -(2**62) + 5]
    tensor = mt.tensor(values, dtype="int64")
    assert tensor.mean().item() == float(Fraction(sum(values), len(values)))
    assert mt.tensor([2**62, 2**62], dtype="int64").mean().item() == 2.0**62


def test_mean_keeps_its_gradient():
    weights = mt.tensor([1.0, 2.0, 4.0], dtype="float64", requires_grad=True)
    weights.mean().backward()
    assert weights.grad.tolist() == [1 / 3, 1 / 3, 1 / 3]
