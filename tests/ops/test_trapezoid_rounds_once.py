# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""`trapezoid` is the float nearest the exact trapezoidal integral.

The heights, the widths, their products and the sum each rounded, and of
random float32 and float64 samples about two integrals in five came out
other than the float nearest the exact value. Each integral is that float
now, evenly spaced or at given positions, along any axis; `dx` is read at the
samples' width, as a number beside them is.
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


def _rounded(value: Fraction, dtype: str) -> float:
    if dtype == "float32":
        return _float32(value)
    try:
        return float(value)
    except OverflowError:
        return math.inf if value > 0 else -math.inf


def _exact(y, x=None, dx=1.0, dtype="float64") -> Fraction:
    """The exact trapezoidal integral of one slice of finite samples."""

    samples = [Fraction(float(value)) for value in y]
    if x is None:
        step = Fraction(float(np.dtype(dtype).type(dx)))
        places = [step * index for index in range(len(samples))]
    else:
        places = [Fraction(float(value)) for value in x]
    return sum(
        (
            (places[i + 1] - places[i]) * (samples[i] + samples[i + 1]) / 2
            for i in range(len(samples) - 1)
        ),
        Fraction(0),
    )


def _want(y, x=None, dx=1.0, dtype="float64", axis=-1):
    """Each slice's exact integral along `axis`, rounded once."""

    y = np.moveaxis(np.asarray(y), axis, -1)
    if x is not None:
        x = np.broadcast_to(np.moveaxis(np.asarray(x), axis, -1), y.shape)
    out = np.empty(y.shape[:-1], dtype=np.float64)
    for index in np.ndindex(*y.shape[:-1]):
        places = None if x is None else x[index]
        out[index] = _rounded(_exact(y[index], places, dx, dtype), dtype)
    return out.tolist()


def _draw(rng, shape, dtype: str, spread: float = 3.0):
    values = rng.normal(size=shape) * 10.0 ** rng.uniform(-spread, spread, shape)
    return values.astype(dtype)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_evenly_spaced(dtype):
    rng = np.random.default_rng(739)
    for trial in range(300):
        n = int(rng.integers(2, 60))
        y = _draw(rng, n, dtype)
        dx = float(rng.uniform(0.01, 3.0))
        got = mt.trapezoid(mt.Tensor(y, dtype=dtype), dx=dx)
        assert str(got.dtype) == dtype
        assert got.item() == _want(y, None, dx, dtype), (trial, n, dx)
    # The default step, and a step of a power of two held exactly.
    y = _draw(rng, 41, dtype)
    assert mt.trapezoid(mt.Tensor(y, dtype=dtype)).item() == _want(y, None, 1.0, dtype)
    got = mt.trapezoid(mt.Tensor(y, dtype=dtype), dx=0.25).item()
    assert got == _want(y, None, 0.25, dtype)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_at_positions(dtype):
    rng = np.random.default_rng(743)
    for trial in range(300):
        n = int(rng.integers(2, 60))
        y = _draw(rng, n, dtype)
        x = np.sort(_draw(rng, n, dtype, 2.0))
        if trial % 3 == 0:
            # Positions out of order make intervals of negative width.
            rng.shuffle(x)
        got = mt.trapezoid(mt.Tensor(y, dtype=dtype), mt.Tensor(x, dtype=dtype))
        assert str(got.dtype) == dtype
        assert got.item() == _want(y, x, dtype=dtype), (trial, n)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_axes_and_shared_or_broadcast_positions(dtype):
    rng = np.random.default_rng(751)
    y = _draw(rng, (4, 7, 5), dtype)
    tensor = mt.Tensor(y, dtype=dtype)
    for axis in range(3):
        got = mt.trapezoid(tensor, dx=0.3, dim=axis).tolist()
        assert got == _want(y, None, 0.3, dtype, axis), axis
        got = mt.trapezoid(tensor, dx=0.3, dim=axis - 3).tolist()
        assert got == _want(y, None, 0.3, dtype, axis), axis
        # A coordinate shared by every slice, along the axis it measures.
        grid = np.sort(_draw(rng, y.shape[axis], dtype))
        shape = [1, 1, 1]
        shape[axis] = y.shape[axis]
        got = mt.trapezoid(tensor, mt.Tensor(grid, dtype=dtype), dim=axis).tolist()
        assert got == _want(y, grid.reshape(shape), dtype=dtype, axis=axis), axis
        # Positions for every sample.
        places = _draw(rng, y.shape, dtype)
        got = mt.trapezoid(tensor, mt.Tensor(places, dtype=dtype), dim=axis).tolist()
        assert got == _want(y, places, dtype=dtype, axis=axis), axis
    # Positions that broadcast against the samples.
    places = _draw(rng, (7, 1), dtype)
    got = mt.trapezoid(tensor, mt.Tensor(places, dtype=dtype), dim=1).tolist()
    assert got == _want(y, np.broadcast_to(places, y.shape), dtype=dtype, axis=1)
    # Positions with more axes than the samples broadcast the samples, and
    # `dim` still names an axis of `y`.
    row = _draw(rng, 6, dtype)
    places = _draw(rng, (3, 6), dtype)
    got = mt.trapezoid(
        mt.Tensor(row, dtype=dtype), mt.Tensor(places, dtype=dtype), dim=0
    )
    assert got.shape == (3,)
    assert got.tolist() == _want(np.broadcast_to(row, (3, 6)), places, dtype=dtype)


def test_mixed_dtypes_and_integers():
    rng = np.random.default_rng(757)
    y = _draw(rng, 30, "float32")
    x = np.sort(_draw(rng, 30, "float64"))
    got = mt.trapezoid(mt.Tensor(y, dtype="float32"), mt.Tensor(x, dtype="float64"))
    assert str(got.dtype) == "float64"
    assert got.item() == _want(y.astype(np.float64), x)
    # Integer samples are widened, exactly in reach of the float.
    got = mt.trapezoid(mt.tensor([1, 4, 9, 16], dtype="int32"), dx=0.5)
    assert str(got.dtype) == "float32" and got.item() == 10.75
    got = mt.trapezoid(mt.tensor([3, 1, 4, 1, 5], dtype="int64"))
    assert str(got.dtype) == "float64" and got.item() == 10.0


def test_cancellation_and_range():
    # Large samples beside small ones, where the plain sum loses the small.
    y = [1e17, 3.0, -1e17, 5.0, 2.0**-30]
    tensor = mt.tensor(y, dtype="float64")
    assert mt.trapezoid(tensor).item() == _rounded(_exact(y), "float64")
    x = [0.0, 1.0, 1.0 + 2.0**-40, 3.0, 7.5]
    got = mt.trapezoid(tensor, mt.tensor(x, dtype="float64")).item()
    assert got == _rounded(_exact(y, x), "float64")
    # Positions so large their products with the samples pass the largest
    # float, though the integral does not.
    big = float(np.finfo(np.float64).max) / 4
    x = [big, big * (1 + 2.0**-50), big * (1 + 2.0**-49)]
    y = [1e10, -3e10, 7.0]
    got = mt.trapezoid(mt.tensor(y, dtype="float64"), mt.tensor(x, dtype="float64"))
    assert got.item() == _rounded(_exact(y, x), "float64")
    # An integral past the largest float is infinite.
    huge = [big * 3, big * 3, big * 3]
    assert mt.trapezoid(mt.tensor(huge, dtype="float64"), dx=2.0).item() == math.inf
    # Steps and integrals among the subnormals.
    for dtype, tiny in (("float64", 5e-324), ("float32", 2.0**-149)):
        for step in (tiny, 3 * tiny, 2.0**-1000 if dtype == "float64" else 2.0**-120):
            y = [3.0, 1.0, 0.5, 7.0, 2.0]
            got = mt.trapezoid(mt.tensor(y, dtype=dtype), dx=step).item()
            assert got == _want(y, None, step, dtype), (dtype, step)
        y = [1.0, 3.0, 1.0]
        x = [0.0, tiny, 2 * tiny]
        got = mt.trapezoid(mt.tensor(y, dtype=dtype), mt.tensor(x, dtype=dtype))
        assert got.item() == _want(y, x, dtype=dtype), dtype


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_specials(dtype):
    inf, nan = math.inf, math.nan
    one = lambda y, **kw: mt.trapezoid(mt.tensor(y, dtype=dtype), **kw).item()
    assert one([1.0, inf, 2.0]) == inf
    assert one([1.0, -inf, 2.0]) == -inf
    assert math.isnan(one([1.0, inf, -inf]))
    assert math.isnan(one([1.0, nan, 2.0]))
    # A zero-width interval at an infinite sample is undefined.
    places = mt.tensor([0.0, 0.0, 1.0], dtype=dtype)
    assert math.isnan(
        mt.trapezoid(mt.tensor([inf, 1.0, 2.0], dtype=dtype), places).item()
    )
    places = mt.tensor([0.0, inf, inf], dtype=dtype)
    assert math.isnan(
        mt.trapezoid(mt.tensor([1.0, 1.0, 1.0], dtype=dtype), places).item()
    )
    # One sample, or none, has no interval.
    assert one([5.0]) == 0.0
    assert mt.trapezoid(mt.zeros([3, 0], dtype=dtype)).tolist() == [0.0, 0.0, 0.0]
    assert mt.trapezoid(mt.zeros([0, 4], dtype=dtype)).tolist() == []


def test_positions_must_match_the_samples():
    y = mt.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype="float64")
    for x in ([0.0, 1.0], [0.0], [[0.0, 1.0, 2.0, 3.0]], [[0.0], [1.0]]):
        with pytest.raises(ValueError):
            mt.trapezoid(y, mt.tensor(x, dtype="float64"))
    with pytest.raises(ValueError):
        mt.trapezoid(y, mt.tensor([0.0, 1.0, 2.0], dtype="float64"), dim=0)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_gradients(dtype):
    rows = [[1.0, 4.0, 9.0, 16.0], [2.0, -1.0, 0.5, 3.0]]
    y = mt.tensor(rows, dtype=dtype, requires_grad=True)
    (mt.trapezoid(y, dx=0.5) * mt.tensor([1.0, 2.0], dtype=dtype)).sum().backward()
    assert y.grad.tolist() == [[0.25, 0.5, 0.5, 0.25], [0.5, 1.0, 1.0, 0.5]]

    # Each sample weighs half the intervals either side of it, and each
    # position moves the interval on its left one way and the one on its
    # right the other, by their mean heights.
    grid = [0.0, 1.0, 3.0, 4.0]
    y = mt.tensor(rows, dtype=dtype, requires_grad=True)
    x = mt.tensor(grid, dtype=dtype, requires_grad=True)
    mt.trapezoid(y, x).sum().backward()
    assert y.grad.tolist() == [[0.5, 1.5, 1.5, 0.5]] * 2
    heights = [[(r[i] + r[i + 1]) / 2 for i in range(3)] for r in rows]
    moving = [sum(([0.0] + h)[i] - (h + [0.0])[i] for h in heights) for i in range(4)]
    assert x.grad.tolist() == moving

    # Positions for every sample, along the first axis.
    y = mt.tensor(rows, dtype=dtype, requires_grad=True)
    x = mt.tensor([grid, [g * 2 for g in grid]], dtype=dtype, requires_grad=True)
    mt.trapezoid(y, x, dim=0).sum().backward()
    widths = [2 * g - g for g in grid]
    assert y.grad.tolist() == [[w / 2 for w in widths]] * 2
    assert x.grad.tolist() == [
        [-(a + b) / 2 for a, b in zip(*rows)],
        [(a + b) / 2 for a, b in zip(*rows)],
    ]


def test_gradients_match_central_differences():
    rng = np.random.default_rng(761)
    base_y = rng.normal(size=(3, 6))
    base_x = np.sort(rng.normal(size=(3, 6)), axis=1)

    def integral(y, x):
        return mt.trapezoid(
            mt.Tensor(y, dtype="float64"), mt.Tensor(x, dtype="float64")
        )

    y = mt.Tensor(base_y.copy(), dtype="float64", requires_grad=True)
    x = mt.Tensor(base_x.copy(), dtype="float64", requires_grad=True)
    weights = mt.Tensor(np.array([1.0, -2.0, 0.5]), dtype="float64")
    (mt.trapezoid(y, x) * weights).sum().backward()
    eps = 1e-6
    for grad, which in ((y.grad.numpy(), 0), (x.grad.numpy(), 1)):
        for index in np.ndindex(*base_y.shape):
            up = [base_y.copy(), base_x.copy()]
            down = [base_y.copy(), base_x.copy()]
            up[which][index] += eps
            down[which][index] -= eps
            numeric = (
                (integral(*up).numpy() - integral(*down).numpy())
                @ np.array([1.0, -2.0, 0.5])
            ) / (2 * eps)
            assert grad[index] == pytest.approx(numeric, rel=1e-6, abs=1e-8)


def test_long_rows_in_parallel():
    rng = np.random.default_rng(769)
    for dtype in ("float32", "float64"):
        y = _draw(rng, (3, 70_000), dtype)
        x = np.sort(_draw(rng, 70_000, dtype, 1.0))
        tensor = mt.Tensor(y, dtype=dtype)
        got = mt.trapezoid(tensor, mt.Tensor(x, dtype=dtype)).tolist()
        assert got == _want(y, np.broadcast_to(x, y.shape), dtype=dtype), dtype
        got = mt.trapezoid(tensor, dx=0.7).tolist()
        assert got == _want(y, None, 0.7, dtype), dtype
        columns = _draw(rng, (20_000, 9), dtype)
        got = mt.trapezoid(mt.Tensor(columns, dtype=dtype), dx=1.5, dim=0).tolist()
        assert got == _want(columns, None, 1.5, dtype, 0), dtype
