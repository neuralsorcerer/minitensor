# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""A weighted `bincount` or `histogram` bin is the float nearest the exact
total of its weights.

Each bin's weights were added plainly, a float32 bin in float32: of a hundred
small `bincount`s of ordinary weights spread over eight bins, nearly half the
bins were an ulp or more from the exact total, and so were half a weighted
histogram's. Each bin is now carried exactly and rounded once.
"""

from __future__ import annotations

import math
from fractions import Fraction

import numpy as np
import pytest

import minitensor as mt


def _float32(value: Fraction) -> float:
    """The float32 nearest `value`, ties to even."""

    if value == 0:
        return 0.0
    sign = -1.0 if value < 0 else 1.0
    magnitude = abs(value)
    exponent = magnitude.numerator.bit_length() - magnitude.denominator.bit_length()
    if Fraction(2) ** exponent > magnitude:
        exponent -= 1
    unit = Fraction(2) ** max(exponent - 23, -149)
    rounded = round(magnitude / unit) * unit
    return sign * (math.inf if rounded >= Fraction(2) ** 128 else float(rounded))


def _total(weights, dtype):
    exact = sum((Fraction(float(w)) for w in weights), Fraction(0))
    return _float32(exact) if dtype == "float32" else float(exact)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_bincount(dtype):
    rng = np.random.default_rng(541)
    for trial in range(60):
        n = int(rng.integers(1, 300))
        labels = rng.integers(0, 8, n)
        weights = rng.normal(size=n) * 2.0 ** rng.integers(-20, 21, n)
        if trial % 3 == 0:
            weights = 1e4 + rng.normal(size=n)
        weights = weights.astype(dtype)
        got = mt.bincount(
            mt.tensor(labels.tolist(), dtype="int64"),
            weights=mt.Tensor(weights, dtype=dtype),
            minlength=10,
        ).tolist()
        want = [_total(weights[labels == b], dtype) for b in range(len(got))]
        assert got == want, trial


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_bincount_edges(dtype):
    big = 1e300 if dtype == "float64" else 1e30
    labels = mt.tensor([0, 1, 0, 1, 0, 2, 2], dtype="int64")
    weights = mt.tensor([big, 1.0, 1.0, math.nan, -big, math.inf, 2.0], dtype=dtype)
    got = mt.bincount(labels, weights=weights, minlength=4).tolist()
    # Cancelling exactly, a NaN kept to its own bin, an infinity, and a bin
    # with nothing in it.
    assert got[0] == 1.0
    assert math.isnan(got[1])
    assert got[2] == math.inf
    assert got[3] == 0.0


def test_weighted_histogram():
    rng = np.random.default_rng(547)
    for trial in range(40):
        n = int(rng.integers(1, 300))
        values = rng.normal(size=n)
        weights = rng.normal(size=n) * 2.0 ** rng.integers(-20, 21, n)
        counts, edges = mt.histogram(
            mt.Tensor(values, dtype="float64"),
            bins=6,
            weights=mt.Tensor(weights, dtype="float64"),
        )
        counts, edges = counts.tolist(), edges.tolist()
        for b in range(6):
            low, high = edges[b], edges[b + 1]
            inside = (values >= low) & ((values <= high) if b == 5 else (values < high))
            assert counts[b] == _total(weights[inside], "float64"), (trial, b)
