# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""A float32 product is accumulated in float64 and rounded once.

Rounding a product of two factors near one to float32 is biased -- the exact
product has structured low bits, and nearest rounding lands low on average,
about 0.05 ulp a multiplication -- so a running float32 product drifts in one
direction rather than wandering: four million factors within 5e-4 of one came
out 15000 ulps low. No regrouping fixes a drift (pairwise was thirty times
worse); a wider accumulator does. NumPy multiplies in float32 and drifts too.

Lanes and blocks multiply separately, so on factors spanning a huge range one
partial product could overflow while another underflowed, and joining them
would give NaN from finite data. An answer that is not a normal float64 is
therefore recomputed exactly, which these tests pin with a rational oracle.
"""

from __future__ import annotations

import math
from fractions import Fraction

import numpy as np
import pytest

import minitensor as mt


def _ulps(got, exact):
    return int(
        np.abs(
            np.asarray(got).view(np.int32).astype(np.int64)
            - np.asarray(exact).view(np.int32).astype(np.int64)
        ).max()
    )


@pytest.mark.parametrize(
    "shape,dim",
    [((1 << 22,), None), ((1 << 22,), 0), ((4, 1_000_000, 1), 1), ((1024, 4096), 1)]
    + [((16, 100_000, 2), 1), ((1_000_000, 2), 0), ((4096, 1024), 0)],
)
def test_a_long_float32_product_is_rounded_once(shape, dim):
    rng = np.random.default_rng(2)
    values = (1 + (rng.random(shape) - 0.5) * 1e-3).astype(np.float32)
    exact = values.astype(np.float64).prod(axis=dim).astype(np.float32)
    t = mt.from_numpy(values)
    got = (t.prod() if dim is None else t.prod(dim)).numpy()
    assert _ulps(got, exact) <= 1


def _exact(values):
    values = [float(v) for v in values]
    if any(math.isnan(v) for v in values):
        return math.nan
    sign = -1.0 if sum(math.copysign(1, v) < 0 for v in values) % 2 else 1.0
    zero = any(v == 0 for v in values)
    infinite = any(math.isinf(v) for v in values)
    if zero and infinite:
        return math.nan
    if zero:
        return sign * 0.0
    if infinite:
        return sign * math.inf
    product = Fraction(1)
    for v in values:
        product *= Fraction(v)
    try:
        f = float(product)
    except OverflowError:
        return sign * math.inf
    return f if f != 0 else sign * 0.0


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize(
    "case",
    ["zeros", "infinities", "nan", "zero-and-inf", "wide-range", "extreme-range"],
)
@pytest.mark.parametrize("shape,dim", [((300, 2), 0), ((3, 900), 1), ((16, 40, 3), 1)])
def test_special_values_and_range_follow_the_exact_product(shape, dim, case, dtype):
    rng = np.random.default_rng(len(case) + sum(shape))
    values = (1 + (rng.random(shape) - 0.5) * 0.2).astype(dtype)
    flat = values.reshape(-1)
    pick = lambda k: rng.integers(0, flat.size, k)  # noqa: E731
    if case == "zeros":
        flat[pick(2)] = 0
        flat[pick(2)] *= -1
    elif case == "infinities":
        flat[pick(2)] = np.inf
        flat[pick(1)] = -np.inf
    elif case == "nan":
        flat[pick(1)] = np.nan
    elif case == "zero-and-inf":
        flat[pick(1)] = 0
        flat[pick(1)] = np.inf
    elif case == "wide-range":
        flat[:] = rng.choice([1e20, 1e-20, -3.0], flat.size)
    else:
        flat[:] = rng.choice([1e30, 1e-30, 0.7], flat.size)
    got = mt.from_numpy(values).prod(dim).numpy()
    moved = np.moveaxis(values, dim, -1)
    want = np.array(
        [_exact(row) for row in moved.reshape(-1, moved.shape[-1])], dtype=np.float64
    )
    with np.errstate(over="ignore"):
        want = want.astype(dtype).reshape(got.shape)
    for kind in (np.isnan, np.isposinf, np.isneginf):
        np.testing.assert_array_equal(kind(got), kind(want))
    np.testing.assert_array_equal(np.signbit(got), np.signbit(want))
    finite = np.isfinite(want)
    if not finite.any():
        return
    if dtype == "float32":
        # Rounded once from float64, so within an ulp of the exact product.
        assert _ulps(got[finite], want[finite]) <= 1
    else:
        # A float64 product rounds at every step; it cannot be exact.
        np.testing.assert_allclose(got[finite], want[finite], rtol=1e-12, atol=0)


def test_an_empty_axis_multiplies_to_one():
    for shape, dim in [((0, 4), 0), ((4, 0), 1), ((2, 0, 3), 1)]:
        for dtype in ("float32", "float64"):
            got = mt.from_numpy(np.zeros(shape, dtype)).prod(dim).numpy()
            np.testing.assert_array_equal(got, np.ones(got.shape, dtype))
