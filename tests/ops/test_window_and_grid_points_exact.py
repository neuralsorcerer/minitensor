# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""Window samples and grid coordinates from whole numbers, rounded once.

A symmetric window has equal samples at `k` and `N - k`, but the cosine of
`k * (2 * pi / N)` and of `(N - k) * (2 * pi / N)` round apart: seven mirrored
pairs in ten differed. Bartlett's `1 - |k * (2 / N) - 1|` rounded up to four
times and half its samples were off, Hann's `0.5 - 0.5 * cos` kept a handful
of digits next to its zero ends, and Blackman's three terms summed to
-1.4e-17 at ends that are zero. The samples are now taken from the distance
to the nearer end, as an integer, with the angle measured from the nearest of
the end, the quarter and the middle. `affine_grid`'s coordinates are a whole
number over the length, divided once.
"""

from __future__ import annotations

import math
from fractions import Fraction

import numpy as np
import pytest

import minitensor as mt
from minitensor import functional as F

_WINDOWS = ["hann_window", "hamming_window", "blackman_window", "bartlett_window"]


@pytest.mark.parametrize("name", _WINDOWS + ["kaiser_window"])
@pytest.mark.parametrize("periodic", [False, True])
def test_mirrored_samples_are_equal(name, periodic):
    for length in range(2, 200):
        samples = getattr(mt, name)(length, periodic=periodic).tolist()
        divisor = length if periodic else length - 1
        for k in range(1, length):
            mirror = divisor - k
            if 0 <= mirror < length:
                assert samples[k] == samples[mirror], (length, k)


@pytest.mark.parametrize("periodic", [False, True])
def test_bartlett_samples_are_exact(periodic):
    for length in range(2, 300):
        divisor = length if periodic else length - 1
        expected = [
            float(Fraction(divisor - abs(2 * k - divisor), divisor))
            for k in range(length)
        ]
        assert mt.bartlett_window(length, periodic=periodic).tolist() == expected


def test_the_ends_quarters_and_middle_are_exact():
    assert mt.hann_window(8).tolist()[::2] == [0.0, 0.5, 1.0, 0.5]
    assert mt.hann_window(9, periodic=False).tolist()[::2] == [0.0, 0.5, 1.0, 0.5, 0.0]
    blackman = mt.blackman_window(9, periodic=False).tolist()
    assert blackman[0] == blackman[-1] == 0.0
    assert blackman[4] == 1.0
    assert min(mt.blackman_window(1001, periodic=False).tolist()) == 0.0
    assert mt.hamming_window(8).tolist()[2] == 0.54


def test_samples_beside_a_zero_end_keep_their_digits():
    # `sin(pi / N) ** 2`, where `0.5 - 0.5 * cos(2 * pi / N)` was off in the
    # sixth digit at this length.
    length = 1 << 16
    expected = math.sin(math.pi / length) ** 2
    got = mt.hann_window(length).tolist()[1]
    assert abs(got - expected) <= 4e-16 * expected
    got = mt.blackman_window(length).tolist()[1]
    expected = expected * (0.36 + 0.64 * expected)
    assert abs(got - expected) <= 8e-16 * expected


@pytest.mark.parametrize("align_corners", [False, True])
def test_grid_coordinates_are_rounded_once(align_corners):
    for width in range(1, 120):
        theta = mt.Tensor(
            np.array([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]]), dtype="float64"
        )
        grid = F.affine_grid(theta, (1, 1, 1, width), align_corners).numpy()
        across = grid[0, 0, :, 0].tolist()
        if align_corners:
            divisor = max(width - 1, 1)
            exact = [Fraction(2 * i - (width - 1), divisor) for i in range(width)]
            if width == 1:
                exact = [Fraction(0)]
        else:
            exact = [Fraction(2 * i + 1 - width, width) for i in range(width)]
        assert across == [float(value) for value in exact], width
