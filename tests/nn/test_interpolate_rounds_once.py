# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""Linear interpolation gives each output, and each entry of its gradient, as
the float nearest its exact value.

Each axis reads a position that is a fraction of whole numbers, so each output
is an exact combination of up to four inputs with rational weights. It was two
interpolations along the width and one along the height in the element type,
the weights narrowed to it first -- a rounding at each step -- and the gradient
added each output's share into its cells one at a time. Both are rounded once
now. The references take the weights from the definition, as fractions.
"""

from __future__ import annotations

import math
from fractions import Fraction

import numpy as np
import pytest

import minitensor as mt

DTYPES = ["float32", "float64"]


def _nearest(value: Fraction, dtype: str) -> float:
    if dtype == "float64":
        return value.numerator / value.denominator
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


def _axis(size, out, mode, align_corners):
    """Each output position's two sources along one axis and the weight of
    the upper, a fraction: the definition."""

    taps = []
    for index in range(out):
        if mode == "nearest":
            cell = min(index * size // out, size - 1)
            taps.append((cell, cell, Fraction(0)))
            continue
        if align_corners:
            position = (
                Fraction(0) if out == 1 else Fraction(index * (size - 1), out - 1)
            )
        else:
            position = max(
                Fraction(2 * index + 1, 2) * Fraction(size, out) - Fraction(1, 2),
                Fraction(0),
            )
        lower = min(math.floor(position), size - 1)
        upper = min(lower + 1, size - 1)
        taps.append((lower, upper, position - math.floor(position)))
    return taps


def _exact(x, g, size, mode, align_corners, dtype):
    """The interpolation of `x` and the gradient of `sum(output * g)` for it,
    from the definition: every value exact, each rounded once."""

    rows = _axis(x.shape[2], size[0], mode, align_corners)
    cols = _axis(x.shape[3], size[1], mode, align_corners)
    out = np.zeros(x.shape[:2] + tuple(size), dtype=dtype)
    grad = np.zeros(x.shape, dtype=dtype)
    for n in range(x.shape[0]):
        for c in range(x.shape[1]):
            shares = {}
            for i, (top, bottom, a) in enumerate(rows):
                for j, (left, right, b) in enumerate(cols):
                    taps = [
                        ((top, left), (1 - a) * (1 - b)),
                        ((top, right), (1 - a) * b),
                        ((bottom, left), a * (1 - b)),
                        ((bottom, right), a * b),
                    ]
                    total = sum(
                        (w * Fraction(float(x[n, c][cell])) for cell, w in taps if w),
                        Fraction(0),
                    )
                    out[n, c, i, j] = _nearest(total, dtype)
                    share = Fraction(float(g[n, c, i, j]))
                    for cell, w in taps:
                        if w:
                            shares[cell] = shares.get(cell, Fraction(0)) + w * share
            for cell, total in shares.items():
                grad[(n, c) + cell] = _nearest(total, dtype)
    return out, grad


def _same(got, want, what):
    got = np.ascontiguousarray(got)
    want = np.ascontiguousarray(want, dtype=got.dtype)
    assert got.shape == want.shape, what
    differ = got.view(np.uint8).reshape(got.size, -1) != want.view(np.uint8).reshape(
        want.size, -1
    )
    count = int(differ.any(axis=1).sum())
    assert count == 0, f"{what}: {count} of {got.size} entries differ"


def _values(rng, shape, dtype):
    values = rng.standard_normal(shape) * 2.0 ** rng.integers(-8, 9, shape)
    values[rng.random(shape) < 0.05] = 0.0
    return values.astype(dtype)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(
    "size,out,mode,align_corners",
    [
        ((5, 6), (10, 12), "bilinear", False),
        ((5, 6), (10, 12), "bilinear", True),
        ((7, 4), (3, 9), "bilinear", False),
        ((7, 4), (3, 9), "bilinear", True),
        ((4, 4), (13, 7), "bilinear", False),
        ((6, 5), (12, 10), "nearest", False),
        ((6, 5), (4, 3), "nearest", False),
        ((1, 6), (3, 11), "bilinear", False),
    ],
)
def test_interpolation_and_its_gradient_round_once(
    dtype, size, out, mode, align_corners
):
    rng = np.random.default_rng(sum(size) + 3 * sum(out))
    x = _values(rng, (2, 2) + size, dtype)
    g = _values(rng, (2, 2) + out, dtype)
    tensor = mt.tensor(x, requires_grad=True)
    kwargs = {} if mode == "nearest" else {"align_corners": align_corners}
    result = mt.nn.interpolate(tensor, size=out, mode=mode, **kwargs)
    (result * mt.tensor(g)).sum().backward()
    want, want_grad = _exact(x, g, out, mode, align_corners, dtype)
    _same(result.numpy(), want, "output")
    _same(tensor.grad.numpy(), want_grad, "gradient")


@pytest.mark.parametrize("dtype", DTYPES)
def test_one_dimensional_linear_interpolation_rounds_once(dtype):
    rng = np.random.default_rng(5)
    x = _values(rng, (2, 3, 9), dtype)
    g = _values(rng, (2, 3, 20), dtype)
    tensor = mt.tensor(x, requires_grad=True)
    result = mt.nn.interpolate(tensor, size=20, mode="linear")
    (result * mt.tensor(g)).sum().backward()
    want, want_grad = _exact(
        x[:, :, None, :], g[:, :, None, :], (1, 20), "linear", False, dtype
    )
    _same(result.numpy(), want[:, :, 0, :], "output")
    _same(tensor.grad.numpy(), want_grad[:, :, 0, :], "gradient")


@pytest.mark.parametrize("dtype", DTYPES)
def test_doubling_lands_on_halfway_points_and_rounds_them_to_even(dtype):
    # Doubling reads quarters and three quarters, so a quarter of an output
    # times four is often a whole number of last places with half of one
    # over: exactly halfway between two floats.
    step = float(np.finfo(dtype).eps)
    row = np.array([1 + k * step for k in range(8)], dtype=dtype).reshape(1, 1, 1, 8)
    result = mt.nn.interpolate(mt.tensor(row), size=(1, 16), mode="bilinear").numpy()
    want, _ = _exact(row, np.zeros((1, 1, 1, 16)), (1, 16), "bilinear", False, dtype)
    _same(result, want, "doubled row")


def test_a_sample_beside_an_infinity_is_read_as_it_is():
    # An output on a sample takes none of its neighbour, an infinity included.
    x = np.array([[[[1.0, np.inf, 3.0]]]], dtype=np.float32)
    result = mt.nn.interpolate(
        mt.tensor(x), size=(1, 5), mode="bilinear", align_corners=True
    ).numpy()
    assert result[0, 0, 0, 0] == 1.0 and result[0, 0, 0, 4] == 3.0
    assert np.isinf(result[0, 0, 0, 1:4]).all()
