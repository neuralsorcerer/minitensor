# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""An Lp pool is the `p`-norm of each window: for `p` of 1 and 2 the float
nearest the exact norm.

It was `|x| ** p` pooled to a mean, the mean times the window's size and its
`p`-th root -- a rounding at each step. A window of zeros had a NaN gradient,
the root's infinite slope at zero times the zero slope of the power, and a
norm type of infinity read the zeroth root of a sum of infinities. Each output
is now the library's norm of its window: rounded once for 1 and 2, with a
gradient of zero at a window of zeros and the largest magnitude for infinity.
"""

from __future__ import annotations

import math
from fractions import Fraction

import numpy as np
import pytest

import minitensor as mt

DTYPES = ["float32", "float64"]


def _float32(value: Fraction) -> float:
    if value == 0:
        return 0.0
    exponent = value.numerator.bit_length() - value.denominator.bit_length()
    if Fraction(2) ** exponent > value:
        exponent -= 1
    unit = Fraction(2) ** max(exponent - 23, -149)
    return float(round(value / unit) * unit)


def _root(square: Fraction, dtype: str) -> float:
    """The float nearest the square root of `square`, found from the floats
    either side of the float64 root by comparing squares exactly."""

    kind = np.float32 if dtype == "float32" else np.float64
    guess = kind(math.sqrt(float(square)))
    candidates = [
        np.nextafter(guess, kind(-1)),
        guess,
        np.nextafter(guess, kind(np.inf)),
    ]
    for value in candidates:
        below = (
            Fraction(float(np.nextafter(value, kind(-1)))) + Fraction(float(value))
        ) / 2
        above = (
            Fraction(float(value)) + Fraction(float(np.nextafter(value, kind(np.inf))))
        ) / 2
        if below * below <= square <= above * above:
            return float(value)
    raise AssertionError("no candidate brackets the root")


def _windows(x, kernel, stride):
    """Each window of the trailing axes of `x`, flattened, in output order."""

    spatial = x.shape[2:]
    out = [(size - k) // s + 1 for size, k, s in zip(spatial, kernel, stride)]
    for n in range(x.shape[0]):
        for c in range(x.shape[1]):
            for position in np.ndindex(*out):
                index = tuple(
                    slice(p * s, p * s + k) for p, s, k in zip(position, stride, kernel)
                )
                yield (n, c) + position, x[(n, c) + index].reshape(-1)


def _values(rng, shape, dtype):
    values = rng.standard_normal(shape) * 2.0 ** rng.integers(-12, 12, shape)
    values[rng.random(shape) < 0.1] = 0.0
    return values.astype(dtype)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("p", [1, 2])
@pytest.mark.parametrize(
    "shape,kernel,stride",
    [
        ((2, 3, 17), (3,), (3,)),
        ((2, 3, 17), (4,), (1,)),
        ((2, 2, 7, 9), (2, 3), (2, 3)),
        ((2, 2, 7, 9), (3, 3), (1, 2)),
    ],
)
def test_each_output_is_the_norm_of_its_window_rounded_once(
    dtype, p, shape, kernel, stride
):
    rng = np.random.default_rng(sum(shape) + 7 * p)
    x = _values(rng, shape, dtype)
    pool = mt.nn.lp_pool1d if len(kernel) == 1 else mt.nn.lp_pool2d
    got = pool(
        mt.tensor(x),
        p,
        kernel if len(kernel) > 1 else kernel[0],
        stride if len(stride) > 1 else stride[0],
    )
    got = got.numpy()
    for at, window in _windows(x, kernel, stride):
        exact = [Fraction(float(value)) for value in window]
        if p == 1:
            total = sum(abs(value) for value in exact)
            want = _float32(total) if dtype == "float32" else float(total)
        else:
            want = _root(sum(value * value for value in exact), dtype)
        assert got[at] == want, (at, list(window))


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("p", [1.5, 2, 3])
def test_a_window_of_zeros_has_a_zero_gradient(dtype, p):
    x = mt.tensor(np.array([[[0.0, 0.0, 3.0, 4.0]]], dtype=dtype), requires_grad=True)
    mt.nn.lp_pool1d(x, p, 2).sum().backward()
    grad = x.grad.numpy()
    assert (grad[0, 0, :2] == 0).all()
    assert np.isfinite(grad).all()


@pytest.mark.parametrize("dtype", DTYPES)
def test_an_infinite_norm_type_takes_the_largest_magnitude(dtype):
    x = np.array([[[0.5, 3.0, -4.0, 1.0, -0.25, 0.0]]], dtype=dtype)
    got = mt.nn.lp_pool1d(mt.tensor(x), math.inf, 2).numpy()
    np.testing.assert_array_equal(got, [[[3.0, 4.0, 0.25]]])


@pytest.mark.parametrize("dtype", DTYPES)
def test_other_norm_types_are_the_norm_of_the_window(dtype):
    rng = np.random.default_rng(3)
    x = _values(rng, (2, 3, 12), dtype)
    for p in (0.5, 3.0, 7.0):
        got = mt.nn.lp_pool1d(mt.tensor(x), p, 3).numpy()
        want = mt.tensor(x).reshape(2, 3, 4, 3).norm(p=p, dim=3).numpy()
        np.testing.assert_array_equal(got, want)


def test_integers_are_read_in_the_float_of_their_width():
    for dtype, wide in (("int32", "float32"), ("int64", "float64")):
        x = mt.tensor(np.array([[[3, -4, 1, 0]]], dtype=dtype))
        got = mt.nn.lp_pool1d(x, 2, 2)
        assert str(got.dtype) == wide
        np.testing.assert_array_equal(got.numpy(), [[[5.0, 1.0]]])


@pytest.mark.parametrize("dtype", DTYPES)
def test_overlapping_windows_add_their_gradients(dtype):
    # Each value takes `x / norm` from every window it lies in.
    rng = np.random.default_rng(5)
    x = (np.abs(rng.standard_normal((1, 1, 6))) + 0.5).astype(dtype)
    tensor = mt.tensor(x, requires_grad=True)
    mt.nn.lp_pool1d(tensor, 2, 3, 1).sum().backward()
    want = np.zeros(6)
    for start in range(4):
        window = x[0, 0, start : start + 3].astype(np.float64)
        want[start : start + 3] += window / np.sqrt((window**2).sum())
    np.testing.assert_allclose(tensor.grad.numpy()[0, 0], want, rtol=1e-6)
