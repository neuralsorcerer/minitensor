# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""`softplus` takes its linear tail only where it is asked to.

The default threshold was 20, which in float64 took the tail where the formula
and `x` still differ: `softplus(20.06)` was 2e-9 off, 544,512 ulps, and its
gradient reported 1 where it is `1 - 2e-9`. Without that threshold the formula
overflowed instead -- as `log1p(exp(x))` it was infinite from 709 in float64,
and the float32 kernel returned -0 for 800 -- so an infinite threshold, which
was accepted as a setting, gave wrong answers. The formula is now evaluated as
`max(x, 0) + log1p(exp(-|x|))`, which cannot overflow and equals `x` where it
should, and the default is no threshold at all. A threshold that is given is
honoured as before.
"""

from __future__ import annotations

from decimal import Decimal, localcontext

import numpy as np
import pytest

import minitensor as mt

F = mt.functional


def _softplus(x: float, beta: float = 1.0) -> float:
    # `ln(1 + e^v)` keeps only the digits of `e^v` that fit beside the one,
    # so the precision has to cover `e^v` down to about 1e-200.
    with localcontext() as context:
        context.prec = 250
        b = Decimal(beta)
        return float((1 + (b * Decimal(x)).exp()).ln() / b)


def _ulps(got: float, want: float) -> float:
    return abs(got - want) / float(np.spacing(abs(want)))


@pytest.mark.parametrize("beta", [1.0, 0.7, 3.3])
def test_float64_is_within_two_ulps_everywhere(beta):
    x = np.random.default_rng(103).uniform(-60, 60, 400)
    got = F.softplus(mt.Tensor(x, dtype="float64"), beta=beta).numpy()
    for value, have in zip(x, got):
        assert _ulps(float(have), _softplus(float(value), beta)) <= 2, value


def test_the_old_tail_is_the_formula_now():
    got = F.softplus(mt.Tensor([20.06], dtype="float64")).item()
    assert got != 20.06 and _ulps(got, _softplus(20.06)) <= 1
    weights = mt.Tensor([25.0], dtype="float64", requires_grad=True)
    F.softplus(weights).sum().backward()
    assert weights.grad.item() == pytest.approx(1 - np.exp(-25.0), rel=1e-15)
    assert weights.grad.item() < 1.0


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("threshold", [None, float("inf")])
def test_large_inputs_do_not_overflow(dtype, threshold):
    values = [800.0, 1e30, -800.0, -1e30, float("inf"), float("-inf")]
    options = {} if threshold is None else {"threshold": threshold}
    got = F.softplus(mt.Tensor(values, dtype=dtype), **options).tolist()
    want = [float(np.array(v, dtype=dtype)) for v in (800.0, 1e30)]
    assert got == want + [0.0, 0.0, float("inf"), 0.0]
    assert np.isnan(F.softplus(mt.Tensor([float("nan")], dtype=dtype)).item())


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_a_given_threshold_is_still_the_linear_tail(dtype):
    weights = mt.Tensor([20.5, 19.5], dtype=dtype, requires_grad=True)
    got = F.softplus(weights, threshold=20.0)
    assert got.tolist()[0] == 20.5
    got.sum().backward()
    assert weights.grad.tolist()[0] == 1.0
    assert weights.softplus(threshold=20.0).tolist()[0] == 20.5


def test_float32_is_the_nearest_float32():
    x = np.random.default_rng(107).uniform(-60, 60, 400).astype(np.float32)
    got = F.softplus(mt.Tensor(x)).numpy()
    want = np.array([_softplus(float(v)) for v in x]).astype(np.float32)
    np.testing.assert_array_equal(got, want)
