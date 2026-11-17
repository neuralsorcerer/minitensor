# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""A destination several values are multiplied into holds their exact product, rounded once.

`scatter_reduce` with `"prod"` multiplied each arriving value into the
destination in the element type, a rounding for each, in the order they
arrived: a third of the destinations of random operands were an ulp or more
out, and a running product that left the range on the way stayed out of it --
`1e200 * 1e200 * 1e-300` was infinite where the product is `1e100`. Each float
destination is now the float nearest the exact product of what arrives there,
and of its own value where it counts, as `prod` is; the gradient's product of
the other factors is taken from the exact product too.
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


def _product(values, dtype: str) -> float:
    """The float of `dtype` nearest the exact product of finite, nonzero `values`."""

    exact = Fraction(1)
    for value in values:
        exact *= Fraction(float(value))
    return _float32(exact) if dtype == "float32" else _float64(exact)


def _scattered(base, index, source, dim, dtype, own):
    lists: dict = {}
    for position in np.ndindex(*index.shape):
        destination = list(position)
        destination[dim] = int(index[position])
        lists.setdefault(tuple(destination), []).append(source[position])
    want = np.array(base, dtype=np.float64)
    for destination, values in lists.items():
        factors = list(values) + ([base[destination]] if own else [])
        want[destination] = _product(factors, dtype)
    return want.tolist()


def _call(base, index, source, dtype, dim=0, own=True):
    return mt.scatter_reduce(
        mt.Tensor(base, dtype=dtype),
        dim,
        mt.Tensor(index, dtype="int64"),
        mt.Tensor(source, dtype=dtype),
        "prod",
        include_self=own,
    )


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("own", [True, False])
@pytest.mark.parametrize("shape, dim, rows", [((7, 5), 0, 40), ((6, 9), 1, 30)])
def test_every_destination_is_rounded_once(dtype, own, shape, dim, rows):
    rng = np.random.default_rng(1087 + rows + own)
    base = (rng.uniform(0.5, 2.0, shape) * rng.choice([-1, 1], shape)).astype(dtype)
    source_shape = shape[:dim] + (rows,) + shape[dim + 1 :]
    source = rng.uniform(0.5, 2.0, source_shape) * rng.choice([-1, 1], source_shape)
    source = source.astype(dtype)
    # Destination 0 along the axis takes nothing, and keeps what it holds.
    index = rng.integers(1, shape[dim], source_shape)
    got = _call(base, index, source, dtype, dim, own)
    assert str(got.dtype) == dtype
    assert got.tolist() == _scattered(base, index, source, dim, dtype, own)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_a_running_product_past_the_range_on_the_way(dtype):
    # Two large factors overflow and two small ones underflow on the way;
    # neither product does.
    large, small = (1e30, 1e-30) if dtype == "float32" else (1e200, 1e-250)
    base = np.array([1.0, 1.0], dtype=dtype)
    source = np.array([large, large, small, small, small, large], dtype=dtype)
    index = np.array([0, 0, 0, 1, 1, 1])
    got = _call(base, index, source, dtype).tolist()
    assert got == [
        _product(source[:3], dtype),
        _product(source[3:], dtype),
    ]
    assert all(math.isfinite(v) and v != 0.0 for v in got)


def test_zeros_infinities_and_nan():
    base = np.array([2.0, -3.0, 1.0, 5.0])
    source = np.array([-0.0, 4.0, math.inf, -0.0, math.nan, 7.0])
    index = np.array([0, 1, 1, 2, 3, 3])
    got = _call(base, index, source, "float64").tolist()
    assert got[0] == 0.0 and math.copysign(1.0, got[0]) < 0
    assert got[1] == -math.inf
    assert got[2] == 0.0 and math.copysign(1.0, got[2]) < 0
    assert math.isnan(got[3])
    # A zero and an infinity together make NaN.
    both = _call(np.array([0.0]), np.array([0]), np.array([math.inf]), "float64")
    assert math.isnan(both.item())


def test_integers_still_multiply():
    base = mt.Tensor(np.array([2, 3], dtype=np.int64))
    got = mt.scatter_reduce(
        base,
        0,
        mt.Tensor(np.array([0, 0, 1]), dtype="int64"),
        mt.Tensor(np.array([5, -7, 11], dtype=np.int64)),
        "prod",
    )
    assert got.tolist() == [-70, 33]


@pytest.mark.parametrize("own", [True, False])
def test_the_gradient_of_a_product_past_the_range_on_the_way(own):
    # The factors other than each one multiply to `1e-100`, `1e-100` and
    # `1e400`: the last overflows, the others do not.
    base = mt.Tensor([1.0, 1.0], dtype="float64", requires_grad=True)
    source = mt.Tensor([1e200, 1e200, 1e-300], dtype="float64", requires_grad=True)
    index = mt.Tensor(np.array([0, 0, 0]), dtype="int64")
    out = mt.scatter_reduce(base, 0, index, source, "prod", include_self=own)
    product = _product([1e200, 1e200, 1e-300], "float64")
    assert out.tolist() == [product, 1.0]
    out.sum().backward()
    # The product of everything, rounded once, over each factor.
    others = _product([1e200, 1e-300], "float64")
    np.testing.assert_allclose(source.grad.tolist()[:2], [others] * 2, rtol=3e-16)
    assert source.grad.tolist()[2] == math.inf
    if own:
        assert base.grad.tolist() == [product, 1.0]
