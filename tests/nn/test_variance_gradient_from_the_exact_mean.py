# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""The gradients of `var`, `std`, `nanvar` and `nanstd` against exact values.

A variance moves with each value by `2 g (x - mean) / (N - ddof)`. Taken with
the mean rounded to the dtype, `x - mean` was all rounding for values that
sit far from zero against their spread: float32 values about `3e7` two apart
came back a third off, and a thousand about `1e4` a part in a thousand; in
float64 a thousand about `1e12` were a part in ten thousand off. Each
deviation is now the difference with the slice's exact mean, carried as a
pair of floats, and rounded once.
"""

from __future__ import annotations

import decimal
import math
from decimal import Decimal
from fractions import Fraction

import numpy as np
import pytest

import minitensor as mt

_BOUND = {"float32": 1, "float64": 2}


def _ulp(value: float, dtype: str) -> float:
    return float(np.spacing(np.array(abs(value), dtype=dtype)))


def _close(got: float, want: Fraction, dtype: str, bound: float) -> bool:
    """Within `bound` ulps of the float nearest the exact value."""

    expected = float(np.array(float(want), dtype=dtype))
    return abs(Fraction(got) - want) <= bound * Fraction(_ulp(expected, dtype))


def _slopes(values, g: float, ddof: int, root: bool):
    """Each value's gradient, exactly: of the variance, or of its root."""

    xs = [Fraction(float(v)) for v in values]
    n = len(xs)
    mean = sum(xs) / n
    divisor = n - ddof
    if not root:
        return [Fraction(g) * 2 * (x - mean) / divisor for x in xs]
    variance = sum((x - mean) ** 2 for x in xs) / divisor
    with decimal.localcontext(prec=60):
        spread = Decimal(variance.numerator) / Decimal(variance.denominator)
        spread = Fraction(spread.sqrt())
    return [Fraction(g) * (x - mean) / (divisor * spread) for x in xs]


def _offset(shape, dtype, rng):
    """Values far from zero against their spread: the dtype's worst case."""

    size = 3e4 if dtype == "float32" else 3e12
    return (size + rng.standard_normal(shape)).astype(dtype)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("fn", ["var", "std"])
@pytest.mark.parametrize("unbiased", [True, False])
def test_every_value(unbiased, fn, dtype):
    rng = np.random.default_rng(97)
    x = _offset(200, dtype, rng)
    t = mt.Tensor(x, dtype=dtype, requires_grad=True)
    g = 0.75
    (getattr(mt, fn)(t, None, unbiased) * g).backward()
    got = t.grad.numpy().astype(np.float64)
    want = _slopes(x, g, int(unbiased), fn == "std")
    bound = _BOUND[dtype] + (1 if fn == "std" else 0)
    for k, w in enumerate(want):
        assert _close(got[k], w, dtype, bound), (k, got[k], float(w))


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("dims, keepdim", [(1, False), (0, True), ((0, 2), False)])
def test_slices_along_axes(dims, keepdim, dtype):
    rng = np.random.default_rng(101)
    x = _offset((4, 5, 6), dtype, rng)
    t = mt.Tensor(x, dtype=dtype, requires_grad=True)
    out = mt.var(t, dims, True, keepdim)
    upstream = rng.uniform(-2.0, 2.0, out.shape).astype(dtype)
    out.backward(mt.Tensor(upstream, dtype=dtype))
    got = t.grad.numpy().astype(np.float64)
    axes = (dims,) if isinstance(dims, int) else dims
    kept = [axis for axis in range(3) if axis not in axes]
    moved = np.transpose(x, kept + list(axes)).reshape(
        math.prod(x.shape[a] for a in kept), -1
    )
    slopes = np.transpose(got, kept + list(axes)).reshape(moved.shape)
    flowing = upstream.reshape(-1)
    for r in range(moved.shape[0]):
        want = _slopes(moved[r], float(flowing[r]), 1, False)
        for k, w in enumerate(want):
            assert _close(slopes[r, k], w, dtype, _BOUND[dtype]), (r, k)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("fn", ["nanvar", "nanstd"])
def test_skipping_nan(fn, dtype):
    rng = np.random.default_rng(103)
    x = _offset((3, 40), dtype, rng)
    x[0, ::7] = np.nan
    x[2, :] = np.nan
    t = mt.Tensor(x, dtype=dtype, requires_grad=True)
    getattr(mt, fn)(t, 1).sum().backward()
    got = t.grad.numpy().astype(np.float64)
    bound = _BOUND[dtype] + (1 if fn == "nanstd" else 0)
    for r in (0, 1):
        taken = ~np.isnan(x[r])
        want = _slopes(x[r][taken], 1.0, 1, fn == "nanstd")
        for k, w in zip(np.flatnonzero(taken), want):
            assert _close(got[r, k], w, dtype, bound), (r, k)
        assert np.all(got[r][~taken] == 0.0)
    # An all-NaN slice skips every value, and each takes zero.
    assert np.all(got[2] == 0.0)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_undefined_slices(dtype):
    inf, nan = math.inf, math.nan
    # A single value has no unbiased variance, and a slice holding an
    # infinity or a NaN none at all: NaN throughout, as the variance is.
    for values in ([5.0], [1.0, inf, 2.0], [1.0, nan, 3.0]):
        t = mt.Tensor(np.array(values), dtype=dtype, requires_grad=True)
        out = mt.var(t)
        out.backward()
        assert np.isnan(out.numpy()).all()
        assert np.isnan(t.grad.numpy()).all(), values
    # A constant slice's standard deviation is zero, and so is its gradient.
    t = mt.Tensor(np.full(4, 3.0), dtype=dtype, requires_grad=True)
    mt.std(t).backward()
    assert np.all(t.grad.numpy() == 0.0)
