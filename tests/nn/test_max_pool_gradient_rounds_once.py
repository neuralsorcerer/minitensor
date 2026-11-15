# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""A max pool's gradient gives each cell the float nearest the exact sum of
the gradients of the windows it won.

Where windows overlap, one cell can be the largest of several of them, and
it takes each of their gradients. Added one at a time in the element type,
they took a rounding for each past the first; the sum is rounded once now.
The references here are built from the indices the pool reports, so they do
not depend on how a tie within a window is broken.
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


def _routed(indices, g, cells, dtype):
    """Each cell's gradient from the definition: the exact sum of the
    gradients of the windows that chose it, rounded once."""

    planes = indices.reshape(indices.shape[0] * indices.shape[1], -1)
    flowing = g.reshape(planes.shape)
    out = np.zeros((planes.shape[0], cells), dtype=dtype)
    for plane in range(planes.shape[0]):
        totals = {}
        for window, cell in enumerate(planes[plane]):
            share = Fraction(float(flowing[plane, window]))
            totals[int(cell)] = totals.get(int(cell), Fraction(0)) + share
        for cell, total in totals.items():
            out[plane, cell] = _nearest(total, dtype)
    return out


def _same(got, want, what):
    got = np.ascontiguousarray(got).reshape(-1)
    want = np.ascontiguousarray(want, dtype=got.dtype).reshape(-1)
    differ = int(
        (
            got.view(np.uint8).reshape(got.size, -1)
            != want.view(np.uint8).reshape(want.size, -1)
        )
        .any(axis=1)
        .sum()
    )
    assert differ == 0, f"{what}: {differ} of {got.size} entries differ"


def _peaks(rng, shape, dtype):
    """Values whose largest few stand far above the rest, so that each wins
    every window around it, with gradients of mixed magnitudes."""

    x = rng.standard_normal(shape)
    x[rng.random(shape) < 0.06] += 50.0
    return x.astype(dtype)


def _flowing(rng, shape, dtype):
    g = rng.standard_normal(shape) * 2.0 ** rng.integers(-24, 2, shape)
    return g.astype(dtype)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(
    "size,kernel,stride,padding",
    [
        ((9, 11), 3, 1, 1),
        ((12, 12), 3, 2, 1),
        ((10, 13), (4, 3), (2, 1), (2, 1)),
        ((8, 8), 2, 2, 0),
    ],
)
def test_max_pool2d_gradients_round_once(dtype, size, kernel, stride, padding):
    rng = np.random.default_rng(sum(size))
    x = _peaks(rng, (2, 3) + size, dtype)
    tensor = mt.tensor(x, requires_grad=True)
    values, indices = mt.nn.max_pool2d(
        tensor, kernel, stride, padding, return_indices=True
    )
    g = _flowing(rng, tuple(values.shape), dtype)
    (values * mt.tensor(g)).sum().backward()
    want = _routed(indices.numpy(), g, size[0] * size[1], dtype)
    _same(tensor.grad.numpy(), want, "gradient")


@pytest.mark.parametrize("dtype", DTYPES)
def test_max_pool1d_and_adaptive_gradients_round_once(dtype):
    rng = np.random.default_rng(7)
    x = _peaks(rng, (2, 3, 40), dtype)
    tensor = mt.tensor(x, requires_grad=True)
    values, indices = mt.nn.max_pool1d(tensor, 5, 1, 2, return_indices=True)
    g = _flowing(rng, tuple(values.shape), dtype)
    (values * mt.tensor(g)).sum().backward()
    _same(tensor.grad.numpy(), _routed(indices.numpy(), g, 40, dtype), "1-D gradient")

    # Adaptive windows overlap where the extents do not divide. The cell a
    # window chose is the one its gradient reaches, which the reference
    # finds from the gradient of each window alone.
    x = _peaks(rng, (1, 2, 7, 10), dtype)
    tensor = mt.tensor(x, requires_grad=True)
    pooled = mt.nn.adaptive_max_pool2d(tensor, (4, 6))
    g = _flowing(rng, tuple(pooled.shape), dtype)
    (pooled * mt.tensor(g)).sum().backward()
    winners = np.zeros(pooled.shape, dtype=np.int64)
    for window in np.ndindex(*pooled.shape):
        probe = mt.tensor(x, requires_grad=True)
        chosen = np.zeros(pooled.shape, dtype=dtype)
        chosen[window] = 1.0
        (mt.nn.adaptive_max_pool2d(probe, (4, 6)) * mt.tensor(chosen)).sum().backward()
        flat = probe.grad.numpy()[window[:2]].reshape(-1)
        winners[window] = int(np.flatnonzero(flat)[0])
    _same(tensor.grad.numpy(), _routed(winners, g, 70, dtype), "adaptive gradient")


@pytest.mark.parametrize("dtype", DTYPES)
def test_a_cell_winning_four_windows_takes_their_sum_rounded_once(dtype):
    # One peak in the middle of a 3x3 plane wins all four 2x2 windows of
    # stride one. Their gradients, one and three halves of a last place,
    # come to one and a halfway point, which goes to the even float past it;
    # added one at a time they stayed at one.
    step = float(np.finfo(dtype).eps)
    x = np.zeros((1, 1, 3, 3), dtype=dtype)
    x[0, 0, 1, 1] = 5.0
    tensor = mt.tensor(x, requires_grad=True)
    pooled = mt.nn.max_pool2d(tensor, 2, 1)
    g = np.array([1.0, step / 2, step / 2, step / 2], dtype=dtype).reshape(1, 1, 2, 2)
    (pooled * mt.tensor(g)).sum().backward()
    grad = tensor.grad.numpy()
    assert grad[0, 0, 1, 1] == np.array(1 + 2 * step, dtype=dtype)
    assert (np.delete(grad.reshape(-1), 4) == 0).all()
    assert not np.signbit(grad).any()
