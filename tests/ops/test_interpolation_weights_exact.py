# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""Linear interpolation weights are exact fractions, rounded once.

The source position of output `i` is `i * (in - 1) / (out - 1)` with the
corners aligned and `(i + 0.5) * in / out - 0.5` without. Computed in floats
and split into a floor and a remainder, the weight carried the rounding of
the whole position -- an ulp of the index, not of the weight -- and four in
five weights were off, some by a quarter of a million ulps. Both parts now
come from whole numbers: the floor is an exact quotient, the weight its
remainder over the denominator.
"""

from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest

import minitensor as mt
from minitensor import functional as F


def _positions(size_in, size_out, align_corners):
    """Each output's lower sample and exact weight; `None` past the last."""

    out = []
    for i in range(size_out):
        if align_corners:
            if size_out == 1:
                numerator, denominator = 0, 1
            else:
                numerator, denominator = i * (size_in - 1), size_out - 1
        else:
            numerator = max((2 * i + 1) * size_in - size_out, 0)
            denominator = 2 * size_out
        lower = numerator // denominator
        weight = Fraction(numerator % denominator, denominator)
        out.append((lower, weight if lower < size_in - 1 else None))
    return out


@pytest.mark.parametrize("align_corners", [False, True])
@pytest.mark.parametrize(
    "size_in,size_out", [(3, 7), (5, 11), (100, 333), (4000, 4097), (17, 1000)]
)
def test_linear_weights_are_exact(size_in, size_out, align_corners):
    # Samples alternating 0 and 1: an output whose lower sample is a 0 reads
    # `0 + (1 - 0) * weight`, the weight itself.
    samples = (np.arange(size_in) % 2).astype(np.float64).reshape(1, 1, size_in)
    got = F.interpolate(
        mt.Tensor(samples, dtype="float64"),
        size=[size_out],
        mode="linear",
        align_corners=align_corners,
    ).numpy()
    got = got.ravel().tolist()
    for (lower, weight), value in zip(
        _positions(size_in, size_out, align_corners), got
    ):
        if weight is not None and lower % 2 == 0:
            assert value == float(weight), (lower, weight)


def test_bilinear_weights_are_exact_on_both_axes():
    rows, cols = 7, 9
    samples = np.zeros((1, 1, rows, cols))
    samples[:, :, ::2, :] = (np.arange(cols) % 2)[None, :]
    got = F.interpolate(
        mt.Tensor(samples, dtype="float64"),
        size=[rows, 23],
        mode="bilinear",
        align_corners=False,
    ).numpy()[0, 0]
    # On a row that is a sample row of alternating 0 and 1, the column weight.
    for (lower, weight), value in zip(_positions(cols, 23, False), got[0].tolist()):
        if weight is not None and lower % 2 == 0:
            assert value == float(weight)
