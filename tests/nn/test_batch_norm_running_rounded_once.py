# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""`batch_norm`'s running buffers after a training step against exact values.

A step moves each buffer to `(1 - m) old + m new`, `new` the batch's mean, or
its variance over `n - 1`. They were summed in float64 and blended in three
roundings, and a float64 buffer was up to 2 ulps out. Each is now the value
nearest its exact one.
"""

from __future__ import annotations

import decimal
from decimal import Decimal

import numpy as np
import pytest

import minitensor as mt
from minitensor import functional as F


def _ulps(got, want: Decimal, dtype: str) -> float:
    if want == 0:
        return 0.0 if got == 0 else float("inf")
    unit = Decimal(float(np.spacing(np.array(abs(float(want)), dtype=dtype))))
    return float(abs(Decimal(float(got)) - want) / unit)


def _exact(values, old_mean, old_var, momentum):
    data = [Decimal(float(v)) for v in values]
    n = len(data)
    mean = sum(data) / n
    var = sum((v - mean) ** 2 for v in data) / (n - 1)
    rate = Decimal(momentum)
    return (
        (1 - rate) * Decimal(float(old_mean)) + rate * mean,
        (1 - rate) * Decimal(float(old_var)) + rate * var,
    )


def _step(x, old_mean, old_var, momentum, dtype):
    rm = mt.Tensor(old_mean.copy(), dtype=dtype)
    rv = mt.Tensor(old_var.copy(), dtype=dtype)
    F.batch_norm(mt.Tensor(x, dtype=dtype), rm, rv, None, None, True, momentum, 1e-5)
    return rm.numpy(), rv.numpy()


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("shape", [(64, 3), (8, 3, 40)])
@pytest.mark.parametrize("offset", [0.0, 1e3])
@pytest.mark.parametrize("momentum", [0.1, 0.3])
def test_running_buffers_are_the_nearest_floats(dtype, shape, offset, momentum):
    rng = np.random.default_rng(81 + len(shape))
    x = (offset + rng.standard_normal(shape)).astype(dtype)
    old_mean = rng.uniform(-1.0, 1.0, 3).astype(dtype)
    old_var = rng.uniform(0.5, 2.0, 3).astype(dtype)
    got_mean, got_var = _step(x, old_mean, old_var, momentum, dtype)
    with decimal.localcontext(prec=80):
        for c in range(3):
            values = np.moveaxis(x, 1, 0)[c].ravel()
            mean, var = _exact(values, old_mean[c], old_var[c], momentum)
            assert _ulps(got_mean[c], mean, dtype) <= 0.5, ("mean", c)
            assert _ulps(got_var[c], var, dtype) <= 0.5, ("var", c)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_a_momentum_of_zero_keeps_and_of_one_takes_the_batch(dtype):
    rng = np.random.default_rng(89)
    x = (1e3 + rng.standard_normal((16, 2, 5))).astype(dtype)
    old_mean = np.array([0.25, -3.5], dtype=dtype)
    old_var = np.array([1.5, 0.75], dtype=dtype)
    kept_mean, kept_var = _step(x, old_mean, old_var, 0.0, dtype)
    assert np.array_equal(kept_mean, old_mean) and np.array_equal(kept_var, old_var)
    taken_mean, taken_var = _step(x, old_mean, old_var, 1.0, dtype)
    with decimal.localcontext(prec=80):
        for c in range(2):
            values = x[:, c, :].ravel()
            mean, var = _exact(values, old_mean[c], old_var[c], 1.0)
            assert _ulps(taken_mean[c], mean, dtype) <= 0.5, ("mean", c)
            assert _ulps(taken_var[c], var, dtype) <= 0.5, ("var", c)
