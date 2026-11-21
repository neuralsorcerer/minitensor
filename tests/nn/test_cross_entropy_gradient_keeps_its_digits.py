# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""`cross_entropy`'s gradient in its logits against exact values.

The gradient is `softmax(z) T - t`. Its target class subtracted one from a
rounded probability near it, so a confident, correct row's gradient there --
`-(1 - p)`, small -- was all rounding: 1.7e7 ulps out in float32 and 9e15 in
float64. The probabilities themselves were `exp(log_softmax(z))`, which turned
`log_softmax`'s rounding into `|log p|` ulps of `p`, some thirty at
`p = e^-40`. They are now `softmax`'s own, and the target class's `1 - p` the
sum of the row's other probabilities.
"""

from __future__ import annotations

import decimal
import math
from decimal import Decimal
from fractions import Fraction

import numpy as np
import pytest

import minitensor as mt
from minitensor import functional as F

_DIGITS = {"float32": (24, -149, 128), "float64": (53, -1074, 1024)}


def _rounded(value: Decimal, dtype: str) -> float:
    """The float nearest a decimal value, ties to even, rounded once."""

    if value == 0:
        return 0.0
    exact = Fraction(abs(value))
    digits, least, top = _DIGITS[dtype]
    e = exact.numerator.bit_length() - exact.denominator.bit_length()
    if Fraction(2) ** e > exact:
        e -= 1
    unit = Fraction(2) ** max(e - (digits - 1), least)
    near = round(exact / unit) * unit
    out = math.inf if near >= Fraction(2) ** top else float(near)
    return -out if value < 0 else out


def _rows(z: np.ndarray, targets: np.ndarray):
    """Each row's exact probabilities and `p_c T - t_c`, with `targets` dense."""

    with decimal.localcontext(prec=60, Emax=10**6, Emin=-(10**6)):
        for row, t in zip(z.tolist(), targets.tolist()):
            top = max(row)
            weights = [(Decimal(v) - Decimal(top)).exp() for v in row]
            total = sum(weights)
            mass = sum(Decimal(v) for v in t)
            p = [w / total for w in weights]
            yield p, [p[c] * mass - Decimal(t[c]) for c in range(len(row))], t


def _logits(
    rows: int, classes: int, confident: bool, rng
) -> tuple[np.ndarray, np.ndarray]:
    z = rng.standard_normal((rows, classes)) * 3.0
    k = rng.integers(0, classes, rows)
    if confident:
        z[np.arange(rows), k] = z.max(axis=1) + rng.uniform(5.0, 40.0, rows)
    return z, k


def _grad(z, target, dtype):
    x = mt.Tensor(z, dtype=dtype, requires_grad=True)
    F.cross_entropy(x, target, reduction="sum").backward()
    got = x.grad.numpy().astype(np.float64)
    mt.clear_autograd_graph()
    return got


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("confident", [True, False], ids=["confident", "ordinary"])
@pytest.mark.parametrize("dense", [False, True], ids=["indices", "one_hot"])
def test_hard_targets(dense, confident, dtype):
    rng = np.random.default_rng(23)
    z, k = _logits(200, 10, confident, rng)
    z = z.astype(dtype)
    one_hot = np.zeros(z.shape)
    one_hot[np.arange(len(k)), k] = 1.0
    target = mt.Tensor(one_hot, dtype=dtype) if dense else mt.Tensor(k, dtype="int64")
    got = _grad(z, target, dtype)
    bound = 1 if dtype == "float32" else 3
    for r, (_, want, _) in enumerate(_rows(z.astype(np.float64), one_hot)):
        for c, w in enumerate(want):
            expected = _rounded(w, dtype)
            unit = float(np.spacing(np.array(abs(expected), dtype=dtype)))
            assert abs(got[r, c] - expected) <= bound * unit, (
                r,
                c,
                got[r, c],
                expected,
            )


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_soft_targets_to_an_ulp_of_their_terms(dtype):
    # Where a soft target is met its two terms cancel, and what is left is
    # their roundings: a few ulps of the larger term.
    rng = np.random.default_rng(24)
    z, _ = _logits(200, 10, False, rng)
    z = z.astype(dtype)
    t = rng.dirichlet(np.ones(10), 200).astype(dtype)
    got = _grad(z, mt.Tensor(t, dtype=dtype), dtype)
    for r, (p, want, row_t) in enumerate(
        _rows(z.astype(np.float64), t.astype(np.float64))
    ):
        mass = sum(row_t)
        for c, w in enumerate(want):
            scale = max(float(p[c]) * mass, row_t[c])
            unit = float(np.spacing(np.array(scale, dtype=dtype)))
            assert abs(got[r, c] - float(w)) <= 4 * unit, (r, c, got[r, c], float(w))
