# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""`prod`'s slope with respect to each factor is the product of the others.

It was the whole product, rounded, over the factor: two roundings, and where
the whole product left the range the slopes went with it -- `prod([1e300,
1e300])` had slopes of infinity and `prod([1e-300, 1e-300])` of zero, where
each is the other factor. Each slope is now the float nearest the exact
product of the others, whatever the whole product does.
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


def _float64(value: Fraction) -> float:
    try:
        return float(value)
    except OverflowError:
        return math.inf if value > 0 else -math.inf


def _others(values, dtype):
    """The nearest float to the exact product of all but each one."""

    out = []
    for i in range(len(values)):
        product = Fraction(1)
        for j, v in enumerate(values):
            if j != i:
                product *= Fraction(float(v))
        out.append(_float32(product) if dtype == "float32" else _float64(product))
    return out


def _slopes(x, dtype, dims=None, keepdim=False):
    t = mt.Tensor(x, dtype=dtype, requires_grad=True)
    out = t.prod() if dims is None else t.prod(dims, keepdim)
    out.backward(mt.Tensor(np.ones(out.shape), dtype=dtype))
    grad = t.grad.numpy()
    mt.clear_autograd_graph()
    return grad


@pytest.mark.parametrize(
    "values,want",
    [
        ([1e300, 1e300], [1e300, 1e300]),
        ([1e-300, 1e-300], [1e-300, 1e-300]),
        (
            [1e-200, 1e-200, 1e10],
            [_float64(Fraction(1e-200) * Fraction(1e10))] * 2 + [0.0],
        ),
        ([1e300, -1e300, 0.0], [-0.0, 0.0, -math.inf]),
    ],
)
def test_slopes_where_the_whole_product_leaves_the_range(values, want):
    got = _slopes(np.array(values), "float64").tolist()
    assert got == want
    assert [math.copysign(1, g) for g in got] == [math.copysign(1, w) for w in want]


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_every_slope_is_the_nearest_float_to_the_product_of_the_others(dtype):
    rng = np.random.default_rng(1433)
    top = 120 if dtype == "float32" else 1000
    for n in (1, 2, 3, 7, 25):
        for _ in range(12):
            exponents = rng.uniform(-top, top, n) / max(n - 1, 1) ** 0.5
            x = (
                rng.choice([-1, 1], n) * rng.uniform(1, 2, n) * 2.0**exponents
            ).astype(dtype)
            got = _slopes(x, dtype).tolist()
            assert got == _others(x, dtype), (n, x)


@pytest.mark.parametrize(
    "dims,keepdim", [(0, False), (1, True), ([0, 2], False), ([1, 2], True)]
)
def test_over_some_axes(dims, keepdim):
    rng = np.random.default_rng(1439)
    x = rng.uniform(0.5, 2.0, (3, 4, 5)) * 2.0 ** rng.integers(-300, 300, (3, 4, 5))
    got = _slopes(x, "float64", dims, keepdim)
    axes = [dims] if isinstance(dims, int) else dims
    moved = np.moveaxis(x, axes, list(range(-len(axes), 0)))
    flat = moved.reshape(-1, int(np.prod([x.shape[a] for a in axes])))
    want = np.array([_others(row, "float64") for row in flat]).reshape(moved.shape)
    want = np.moveaxis(want, list(range(-len(axes), 0)), axes)
    np.testing.assert_array_equal(got, want)


def test_zeros_infinities_and_nans_among_the_factors():
    inf, nan = math.inf, math.nan
    # One zero: its slope is the product of the rest; the rest's are zero.
    assert _slopes(np.array([2.0, 0.0, -3.0]), "float64").tolist() == [-0.0, -6.0, 0.0]
    # Two zeros: every product of the others holds one.
    assert _slopes(np.array([0.0, 0.0, 5.0]), "float64").tolist() == [0.0, 0.0, 0.0]
    got = _slopes(np.array([inf, 2.0, 3.0]), "float64").tolist()
    assert got == [6.0, inf, inf]
    # The zero's others are the infinity and 3; only the 3's hold both.
    got = _slopes(np.array([inf, 0.0, 3.0]), "float64").tolist()
    assert got[0] == 0.0 and got[1] == inf and math.isnan(got[2])
    got = _slopes(np.array([nan, 2.0, 3.0]), "float64").tolist()
    assert got[0] == 6.0 and math.isnan(got[1]) and math.isnan(got[2])
    assert _slopes(np.array([7.0]), "float64").tolist() == [1.0]


def test_the_upstream_gradient_scales_each_slope():
    rng = np.random.default_rng(1447)
    x = rng.uniform(0.5, 2.0, (4, 6))
    upstream = rng.standard_normal(4)
    t = mt.Tensor(x, dtype="float64", requires_grad=True)
    t.prod(1).backward(mt.Tensor(upstream, dtype="float64"))
    want = np.array([_others(row, "float64") for row in x]) * upstream[:, None]
    np.testing.assert_allclose(t.grad.numpy(), want, rtol=4e-16, atol=0)
    mt.clear_autograd_graph()


@pytest.mark.parametrize("include_self", [False, True])
def test_scatter_reduce_prod_slopes_are_the_products_of_the_others(include_self):
    rng = np.random.default_rng(1451)
    base = rng.uniform(0.5, 2.0, 4) * 2.0 ** rng.integers(-500, 500, 4)
    src = rng.uniform(0.5, 2.0, 12) * 2.0 ** rng.integers(-500, 500, 12)
    index = rng.integers(0, 4, 12)
    x = mt.Tensor(base, dtype="float64", requires_grad=True)
    s = mt.Tensor(src, dtype="float64", requires_grad=True)
    out = mt.scatter_reduce(
        x, 0, mt.Tensor(index, dtype="int64"), s, "prod", include_self=include_self
    )
    out.backward(mt.Tensor(np.ones(4), dtype="float64"))
    want_src = np.zeros(12)
    want_base = np.ones(4) if not include_self else np.zeros(4)
    for d in range(4):
        members = [i for i in range(12) if index[i] == d]
        if not members:
            want_base[d] = 1.0
            continue
        factors = ([base[d]] if include_self else []) + [src[i] for i in members]
        slopes = _others(factors, "float64")
        if include_self:
            want_base[d] = slopes[0]
        else:
            want_base[d] = 0.0
        for i, slope in zip(members, slopes[int(include_self) :]):
            want_src[i] = slope
    np.testing.assert_array_equal(s.grad.numpy(), want_src)
    np.testing.assert_array_equal(x.grad.numpy(), want_base)
    mt.clear_autograd_graph()


def test_scatter_reduce_prod_at_the_ends_of_the_range():
    inf = math.inf
    for include_self, base, src, want_base, want_src in (
        (False, [1.0], [1e300, 1e300], [0.0], [1e300, 1e300]),
        (True, [1e-300], [1e-300], [1e-300], [1e-300]),
        (False, [1.0], [inf, 2.0, 3.0], [0.0], [6.0, inf, inf]),
    ):
        x = mt.Tensor(base, dtype="float64", requires_grad=True)
        s = mt.Tensor(src, dtype="float64", requires_grad=True)
        index = mt.Tensor([0] * len(src), dtype="int64")
        out = mt.scatter_reduce(x, 0, index, s, "prod", include_self=include_self)
        out.backward(mt.Tensor([1.0], dtype="float64"))
        assert x.grad.tolist() == want_base
        assert s.grad.tolist() == want_src
        mt.clear_autograd_graph()
