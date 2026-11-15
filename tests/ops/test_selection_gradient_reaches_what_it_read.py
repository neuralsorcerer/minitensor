# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""A maximum, a minimum or a median sends its gradient to the elements it was
read from, split evenly over ties, and nothing to the others.

The gradient was a mask of the ties times the share, so an infinite gradient
made every other element `0 * inf`, NaN; and a NaN value, equal to no element,
made every element of its slice NaN. A NaN is read from the slice's NaNs now,
unless the reduction leaves NaNs out, when it was read from nothing.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import minitensor as mt

DTYPES = ["float32", "float64"]
INF, NAN = math.inf, math.nan


def _grad(reduce, values, g, dtype):
    x = mt.tensor(np.array(values, dtype=dtype), requires_grad=True)
    (reduce(x) * mt.tensor(np.array(g, dtype=dtype))).sum().backward()
    return x.grad.numpy()


def _same(got, want):
    want = np.array(want, dtype=got.dtype)
    assert got.shape == want.shape
    assert (np.isnan(got) == np.isnan(want)).all(), (got, want)
    assert (got[~np.isnan(got)] == want[~np.isnan(want)]).all(), (got, want)


WHOLE = {
    "max": lambda x: x.max(),
    "min": lambda x: x.min(),
    "amax": lambda x: mt.amax(x),
    "amin": lambda x: mt.amin(x),
    "median": lambda x: x.median(),
    "nanmax": lambda x: mt.nanmax(x),
    "nanmin": lambda x: mt.nanmin(x),
    "nanmedian": lambda x: mt.nanmedian(x),
}


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("name", sorted(WHOLE))
def test_an_infinite_gradient_reaches_only_the_elements_read(dtype, name):
    values = [1.0, 7.0, 4.0, 7.0, -2.0]
    read = {
        "max": [1, 3],
        "amax": [1, 3],
        "nanmax": [1, 3],
        "min": [4],
        "amin": [4],
        "nanmin": [4],
        "median": [2],
        "nanmedian": [2],
    }[name]
    for g in (INF, -INF):
        got = _grad(WHOLE[name], values, g, dtype)
        _same(got, [g if index in read else 0.0 for index in range(5)])


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("name", ["max", "min", "amax", "amin", "median"])
def test_a_nan_value_is_read_from_the_nans(dtype, name):
    got = _grad(WHOLE[name], [1.0, NAN, 4.0, NAN, -2.0], 1.0, dtype)
    _same(got, [0.0, 0.5, 0.0, 0.5, 0.0])


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("name", ["nanmax", "nanmin", "nanmedian"])
def test_a_reduction_that_leaves_nans_out_reads_none(dtype, name):
    # A slice of NaNs alone has nothing to read: its value is NaN, and no
    # element takes a gradient. Beside a number, a NaN takes none either.
    _same(_grad(WHOLE[name], [NAN, NAN, NAN], 1.0, dtype), [0.0, 0.0, 0.0])
    _same(_grad(WHOLE[name], [NAN, 3.0, NAN], 1.0, dtype), [0.0, 1.0, 0.0])


@pytest.mark.parametrize("dtype", DTYPES)
def test_ties_share_the_gradient_each_rounded_once(dtype):
    got = _grad(lambda x: x.max(), [5.0, 5.0, 1.0, 5.0], 1.0, dtype)
    third = np.array(1.0, dtype=dtype) / np.array(3.0, dtype=dtype)
    _same(got, [third, third, 0.0, third])
    # Three times the smallest subnormal over three ties is one of it each.
    tiny = float(np.finfo(dtype).smallest_subnormal)
    got = _grad(lambda x: x.max(), [5.0, 5.0, 5.0], 3 * tiny, dtype)
    _same(got, [tiny, tiny, tiny])


@pytest.mark.parametrize("dtype", DTYPES)
def test_along_an_axis(dtype):
    x = np.array(
        [[1.0, 9.0, 9.0], [NAN, 2.0, 3.0], [4.0, 4.0, 4.0], [0.0, -1.0, 5.0]],
        dtype=dtype,
    )
    g = np.array([INF, 2.0, 3.0, -INF], dtype=dtype)
    want_max = [
        [0.0, INF, INF],
        [2.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],
        [0.0, 0.0, -INF],
    ]
    _same(_grad(lambda t: mt.amax(t, 1), x, g, dtype), want_max)
    # Leaving NaNs out, the second row's maximum is its 3.
    want_nan = [row[:] for row in want_max]
    want_nan[1] = [0.0, 0.0, 2.0]
    _same(_grad(lambda t: mt.nanamax(t, 1), x, g, dtype), want_nan)
    # Down the columns: the first one's minimum is its NaN.
    columns = np.array([INF, 1.0, -INF], dtype=dtype)
    want = [[0.0, 0.0, 0.0], [INF, 0.0, -INF], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    _same(_grad(lambda t: mt.amin(t, 0), x, columns, dtype), want)
