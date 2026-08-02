# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""`max`/`min` fold over several accumulators, so the lane seams need testing.

A single running `best` makes the compare-and-select a serial dependency chain
that cannot vectorize; `max` was the one f32 reduction slower than NumPy while
`sum`, which already split its accumulator, was four times quicker. Splitting
the fold the same way made f32 `max` 6.4x faster (0.807ms -> 0.126ms over 2M
elements) and turned a 1.96x deficit against NumPy into a 3.3x lead.

The risk that buys is a class of off-by-one bug the old loop could not have:
lengths that do not divide the lane count leave a remainder handled by separate
code, and a value landing in that tail -- or in only one lane -- must still win.
Hence the sizes below straddle the 8- and 4-wide seams, and the NaN and extreme
values are placed at the front, middle and end rather than at a fixed spot.
"""

import numpy as np
import pytest

import minitensor as mt

# Around the lane widths (8 for 32-bit, 4 for 64-bit) and the 8192-element
# parallel chunk boundary, where the remainder paths live.
SIZES = [1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 33, 8191, 8192, 8193, 100_000]

FLOAT_DTYPES = ["float32", "float64"]
INT_DTYPES = ["int32", "int64"]


def _sample(size, dtype, rng):
    if dtype.startswith("int"):
        info = np.iinfo(dtype)
        return rng.integers(info.min // 2, info.max // 2, size).astype(dtype)
    return rng.standard_normal(size).astype(dtype)


@pytest.mark.parametrize("dtype", FLOAT_DTYPES + INT_DTYPES)
@pytest.mark.parametrize("size", SIZES)
def test_matches_numpy_across_the_lane_seams(dtype, size):
    values = _sample(size, dtype, np.random.default_rng(size))
    tensor = mt.from_numpy(values)
    assert tensor.max().numpy() == np.max(values)
    assert tensor.min().numpy() == np.min(values)


@pytest.mark.parametrize("dtype", FLOAT_DTYPES + INT_DTYPES)
@pytest.mark.parametrize("size", [8, 9, 17, 8193])
@pytest.mark.parametrize("position", ["first", "middle", "last"])
def test_a_lone_extreme_value_wins_from_any_lane(dtype, size, position):
    # One winner among otherwise middling values: if the lane it lands in were
    # dropped -- or the remainder tail skipped -- the result would be wrong
    # without any shape or dtype change to notice.
    values = _sample(size, dtype, np.random.default_rng(0))
    index = {"first": 0, "middle": size // 2, "last": size - 1}[position]

    values[index] = np.iinfo(dtype).max if dtype.startswith("int") else 1e30
    assert mt.from_numpy(values).max().numpy() == np.max(values)

    values[index] = np.iinfo(dtype).min if dtype.startswith("int") else -1e30
    assert mt.from_numpy(values).min().numpy() == np.min(values)


# Sizes start at 2: a single element leaves no non-NaN value for the nanmax
# comparison below, and NumPy warns on an all-NaN slice. That case is covered
# by `test_all_nan_and_infinities` instead.
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("size", [2, 7, 8, 9, 8193])
@pytest.mark.parametrize("position", ["first", "middle", "last"])
def test_nan_propagates_from_any_lane(dtype, size, position):
    # NaN is tracked by a per-lane flag rather than by the value comparison,
    # since `v > best` is false for NaN and would silently drop it.
    values = np.random.default_rng(1).standard_normal(size).astype(dtype)
    values[{"first": 0, "middle": size // 2, "last": size - 1}[position]] = np.nan
    tensor = mt.from_numpy(values)

    assert np.isnan(tensor.max().numpy())
    assert np.isnan(tensor.min().numpy())
    # nanmax/nanmin still skip it, as NumPy does.
    assert tensor.nanmax().numpy() == np.nanmax(values)
    assert tensor.nanmin().numpy() == np.nanmin(values)


@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_all_nan_and_infinities(dtype):
    all_nan = np.full(1000, np.nan, dtype=dtype)
    assert np.isnan(mt.from_numpy(all_nan).max().numpy())

    with_inf = np.random.default_rng(2).standard_normal(1000).astype(dtype)
    with_inf[500] = np.inf
    with_inf[600] = -np.inf
    tensor = mt.from_numpy(with_inf)
    assert tensor.max().numpy() == np.inf
    assert tensor.min().numpy() == -np.inf


@pytest.mark.parametrize("dtype", INT_DTYPES)
def test_integer_extremes_are_representable_winners(dtype):
    # The identity the fold starts from is `iinfo.min`/`iinfo.max`, so a real
    # element equal to it must still be reported rather than mistaken for
    # "nothing seen yet".
    info = np.iinfo(dtype)
    values = np.array([info.min, 0, info.max], dtype=dtype)
    tensor = mt.from_numpy(values)
    assert tensor.max().numpy() == info.max
    assert tensor.min().numpy() == info.min
