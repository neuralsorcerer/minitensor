# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""`instance_norm`'s running buffers after a training step against exact
values.

A step moves each buffer to `(1 - m) old + m new`, `new` the channel's mean
for the mean and, for the variance, the mean over the samples of each one's
variance over its positions less one. Composed from means, a variance and a
blend, a buffer was up to an ulp out. Each is now the value nearest its exact
one.
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


def _exact(samples, old_mean, old_var, momentum):
    means, variances = [], []
    for sample in samples:
        data = [Decimal(float(v)) for v in sample.ravel()]
        mean = sum(data) / len(data)
        means.append(mean)
        variances.append(sum((v - mean) ** 2 for v in data) / (len(data) - 1))
    rate = Decimal(momentum)
    return (
        (1 - rate) * Decimal(float(old_mean)) + rate * sum(means) / len(means),
        (1 - rate) * Decimal(float(old_var)) + rate * sum(variances) / len(variances),
    )


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("shape", [(8, 3, 50), (4, 3, 5, 6)])
@pytest.mark.parametrize("offset", [0.0, 1e3])
def test_running_buffers_are_the_nearest_floats(dtype, shape, offset):
    rng = np.random.default_rng(91 + len(shape))
    x = (offset + rng.standard_normal(shape)).astype(dtype)
    old_mean = rng.uniform(-1.0, 1.0, 3).astype(dtype)
    old_var = rng.uniform(0.5, 2.0, 3).astype(dtype)
    rm = mt.Tensor(old_mean.copy(), dtype=dtype)
    rv = mt.Tensor(old_var.copy(), dtype=dtype)
    F.instance_norm(mt.Tensor(x, dtype=dtype), rm, rv, momentum=0.1, eps=1e-5)
    got_mean, got_var = rm.numpy(), rv.numpy()
    with decimal.localcontext(prec=80):
        for c in range(3):
            mean, var = _exact(x[:, c], old_mean[c], old_var[c], 0.1)
            assert _ulps(got_mean[c], mean, dtype) <= 0.5, ("mean", c)
            assert _ulps(got_var[c], var, dtype) <= 0.5, ("var", c)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_one_buffer_alone_is_moved_and_the_other_left(dtype):
    rng = np.random.default_rng(97)
    x = rng.standard_normal((3, 2, 7)).astype(dtype)
    old_var = np.array([1.25, 0.5], dtype=dtype)
    rv = mt.Tensor(old_var.copy(), dtype=dtype)
    F.instance_norm(mt.Tensor(x, dtype=dtype), None, rv, momentum=0.25, eps=1e-5)
    with decimal.localcontext(prec=80):
        for c in range(2):
            _, var = _exact(x[:, c], 0.0, old_var[c], 0.25)
            assert _ulps(rv.numpy()[c], var, dtype) <= 0.5, c
