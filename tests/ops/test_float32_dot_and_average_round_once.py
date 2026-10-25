# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""A float32 `dot` and weighted `average` round once, from float64.

The product of two float32s is exact in float64, so carried there the only
rounding is the last one. Accumulated in float32, `dot` landed up to 90 ulps
from the exact inner product of offset data, and the weighted average several.
"""

from __future__ import annotations

from fractions import Fraction

import numpy as np

import minitensor as mt


def _samples(seed):
    rng = np.random.default_rng(seed)
    for _ in range(25):
        n = int(rng.integers(2, 2500))
        offset = float(rng.choice([0.0, 100.0, 1e4]))
        x = (offset + rng.normal(0, 1, n)).astype(np.float32)
        y = rng.normal(0, 1, n).astype(np.float32)
        w = rng.uniform(0.1, 3.0, n).astype(np.float32)
        yield x, y, w


def _exact_dot(a, b) -> Fraction:
    return sum(Fraction(float(p)) * Fraction(float(q)) for p, q in zip(a, b))


def test_dot_is_correctly_rounded():
    for x, y, _ in _samples(307):
        want = float(np.float32(float(_exact_dot(x, y))))
        assert mt.dot(mt.Tensor(x), mt.Tensor(y)).item() == want


def test_weighted_average_is_correctly_rounded():
    for x, _, w in _samples(311):
        exact = _exact_dot(x, w) / sum(Fraction(float(v)) for v in w)
        want = float(np.float32(float(exact)))
        assert mt.average(mt.Tensor(x), weights=mt.Tensor(w)).item() == want
        rows = mt.Tensor(np.stack([x, x]))
        got = mt.average(rows, dim=1, weights=mt.Tensor(w)).tolist()
        assert got == [want, want]


def test_returned_totals_and_gradients_stay_float32():
    values = mt.Tensor(np.array([1.0, 2.0, 3.0], np.float32), requires_grad=True)
    weights = mt.Tensor(np.array([1.0, 1.0, 2.0], np.float32))
    mean, total = mt.average(values, weights=weights, returned=True)
    assert str(mean.dtype) == "float32" and str(total.dtype) == "float32"
    assert total.item() == 4.0
    mean.backward()
    assert values.grad.tolist() == [0.25, 0.25, 0.5]
    a = mt.Tensor(np.array([1.0, 2.0], np.float32), requires_grad=True)
    b = mt.Tensor(np.array([3.0, 4.0], np.float32), requires_grad=True)
    mt.dot(a, b).backward()
    assert a.grad.tolist() == [3.0, 4.0] and b.grad.tolist() == [1.0, 2.0]
