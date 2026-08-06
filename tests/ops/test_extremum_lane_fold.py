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


# `max(dim=...)` used to walk one output at a time, striding the input by the
# row width, so `max(dim=0)` on a 2048x1024 f32 matrix cost 4.3ms against 0.23ms
# for `sum` over the same axis. Above a threshold the loops are now swapped to
# stream memory in order. That path computes a whole band of outputs at once, so
# the index bookkeeping is what needs pinning: ties must still resolve to the
# first winner, and a NaN must still take the first NaN's position.
BLOCKED_SHAPES = [
    (2048, 1024),  # wide: takes the memory-order path
    (5, 257),  # just over the threshold, with a remainder band
    (2048, 64),  # narrow: stays on the strided path
    (131072, 16),  # narrow and tall
    (64, 32, 128),  # rank 3, so `inner` differs per dim
]


@pytest.mark.parametrize("shape", BLOCKED_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES + INT_DTYPES)
def test_dim_reduction_matches_numpy_values_and_indices(shape, dtype):
    values = _sample(int(np.prod(shape)), dtype, np.random.default_rng(4)).reshape(
        shape
    )
    tensor = mt.from_numpy(values)
    for dim in range(len(shape)):
        got_values, got_indices = tensor.max(dim, False)
        np.testing.assert_array_equal(got_values.numpy(), np.max(values, axis=dim))
        np.testing.assert_array_equal(got_indices.numpy(), np.argmax(values, axis=dim))
        np.testing.assert_array_equal(
            tensor.argmax(dim).numpy(), np.argmax(values, axis=dim)
        )


@pytest.mark.parametrize("shape", [(2048, 1024), (5, 257), (131072, 16)])
def test_ties_resolve_to_the_first_index(shape):
    # Every element equal, so the index is decided purely by the scan order.
    values = np.full(shape, 7.0, dtype=np.float32)
    tensor = mt.from_numpy(values)
    for dim in range(len(shape)):
        _, indices = tensor.max(dim, False)
        np.testing.assert_array_equal(indices.numpy(), np.argmax(values, axis=dim))


@pytest.mark.parametrize("shape", [(512, 1024), (5, 257)])
@pytest.mark.parametrize("row", ["first", "middle", "last"])
def test_a_nan_takes_the_first_nan_position(shape, row):
    # The memory-order path has no early exit, so NaN is folded into the
    # comparison instead. A later NaN must not displace an earlier one.
    values = np.random.default_rng(5).standard_normal(shape).astype(np.float32)
    index = {"first": 0, "middle": shape[0] // 2, "last": shape[0] - 1}[row]
    values[index, :] = np.nan
    tensor = mt.from_numpy(values)

    got_values, got_indices = tensor.max(0, False)
    assert np.all(np.isnan(got_values.numpy()))
    np.testing.assert_array_equal(got_indices.numpy(), np.argmax(values, axis=0))


def test_indices_do_not_depend_on_the_thread_count():
    # The band split follows `rayon::current_num_threads`, so a result that
    # varied with it would be a reproducibility bug rather than a wrong answer.
    import os
    import subprocess
    import sys
    import zlib

    script = (
        "import numpy as np, minitensor as mt, zlib\n"
        "a = np.random.default_rng(3).standard_normal((512, 1024)).astype(np.float32)\n"
        "a[100, 200] = a[300, 200]\n"
        "v, i = mt.from_numpy(a).max(0, False)\n"
        "print(zlib.crc32(v.numpy().tobytes()), zlib.crc32(i.numpy().tobytes()))\n"
    )
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    def run(threads):
        env = dict(os.environ, RAYON_NUM_THREADS=threads, PYTHONPATH=root)
        return subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            env=env,
            check=True,
        ).stdout.strip()

    assert run("1") == run("2") == run("8")
