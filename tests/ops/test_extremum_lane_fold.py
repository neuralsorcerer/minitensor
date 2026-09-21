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
# comparison below, and an all-NaN slice warns. That case is covered
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
    # nanmax/nanmin still skip it.
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


# The whole-tensor `argmax`/`argmin` fold is lane-blocked the same way, but it
# carries a position beside each lane's running best, and that makes ties a
# correctness question rather than a free choice. Lane `l` walks positions
# `l, l + LANES, l + 2 * LANES, ...`, so a later lane holds earlier positions
# than an earlier lane's second block: folding the lanes together in lane order
# would report index 9 over index 2 unless the tie is broken on the index.
ARG_SIZES = [1, 2, 3, 7, 8, 9, 15, 16, 17, 31, 33, 8191, 8192, 8193, 100_000]


@pytest.mark.parametrize("dtype", FLOAT_DTYPES + INT_DTYPES)
@pytest.mark.parametrize("size", ARG_SIZES)
def test_argmax_and_argmin_match_numpy_across_the_lane_seams(dtype, size):
    values = _sample(size, dtype, np.random.default_rng(size))
    tensor = mt.from_numpy(values)
    assert tensor.argmax().numpy() == np.argmax(values)
    assert tensor.argmin().numpy() == np.argmin(values)


@pytest.mark.parametrize("dtype", FLOAT_DTYPES + INT_DTYPES)
@pytest.mark.parametrize("size", ARG_SIZES)
def test_an_all_equal_tensor_answers_index_zero(dtype, size):
    # Nothing ever beats the running best, so the answer is decided entirely by
    # the tie-break -- across lanes, across the remainder, and across chunks.
    values = np.full(size, 3, dtype=dtype)
    tensor = mt.from_numpy(values)
    assert tensor.argmax().numpy() == 0
    assert tensor.argmin().numpy() == 0


@pytest.mark.parametrize("dtype", FLOAT_DTYPES + INT_DTYPES)
@pytest.mark.parametrize(
    "first,second",
    [
        (2, 9),  # a later lane's first block against an earlier lane's second
        (9, 2),  # the same pair, met in the other order
        (0, 7),  # the first and last lane of one block
        (3, 8195),  # either side of the 8192-element parallel chunk
        (8190, 8192),  # the chunk seam itself
        (5, 99_999),  # a lane block against the remainder tail
    ],
)
def test_a_repeated_extremum_answers_the_lower_index(dtype, first, second):
    size = 100_000
    values = _sample(size, dtype, np.random.default_rng(6))
    winner = np.iinfo(dtype).max if dtype.startswith("int") else 1e30
    loser = np.iinfo(dtype).min if dtype.startswith("int") else -1e30

    values[first] = values[second] = winner
    assert mt.from_numpy(values).argmax().numpy() == min(first, second)

    values[first] = values[second] = loser
    assert mt.from_numpy(values).argmin().numpy() == min(first, second)


@pytest.mark.parametrize("dtype", FLOAT_DTYPES + INT_DTYPES)
def test_the_seed_value_is_a_real_answer(dtype):
    # The fold seeds each lane with the type's extreme, which an input can
    # equal. Every element equal to the seed means nothing ever beats it, and
    # the answer has to be index 0 rather than "nothing seen".
    if dtype.startswith("int"):
        low, high = np.iinfo(dtype).min, np.iinfo(dtype).max
    else:
        low, high = -np.inf, np.inf

    assert mt.from_numpy(np.full(100_000, low, dtype=dtype)).argmax().numpy() == 0
    assert mt.from_numpy(np.full(100_000, high, dtype=dtype)).argmin().numpy() == 0


@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("size", [1, 8, 9, 17, 8193, 100_000])
@pytest.mark.parametrize("position", ["first", "middle", "last"])
def test_a_nan_wins_the_argument_from_any_lane(dtype, size, position):
    # A NaN satisfies no comparison, so it can never become a lane's best; its
    # position is tracked separately and overrides the comparison's answer.
    values = np.random.default_rng(7).standard_normal(size).astype(dtype)
    index = {"first": 0, "middle": size // 2, "last": size - 1}[position]
    values[index] = np.nan
    tensor = mt.from_numpy(values)

    assert tensor.argmax().numpy() == np.argmax(values) == index
    assert tensor.argmin().numpy() == np.argmin(values) == index


@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("first,second", [(2, 9), (9, 2), (3, 8195), (5, 99_999)])
def test_ties_among_nans_also_go_to_the_lower_index(dtype, first, second):
    values = np.random.default_rng(8).standard_normal(100_000).astype(dtype)
    values[first] = values[second] = np.nan
    tensor = mt.from_numpy(values)
    assert tensor.argmax().numpy() == min(first, second)
    assert tensor.argmin().numpy() == min(first, second)


@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("size", [1, 9, 100_000])
def test_all_nan_answers_the_first_position(dtype, size):
    values = np.full(size, np.nan, dtype=dtype)
    tensor = mt.from_numpy(values)
    assert tensor.argmax().numpy() == 0
    assert tensor.argmin().numpy() == 0


@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_signed_zeros_tie_on_the_index(dtype):
    # `-0.0 == 0.0`, and neither is greater, so the pair is a tie and the lower
    # index wins -- which is what the `==` in the lane tie-break spells out.
    for values in ([-0.0, 0.0], [0.0, -0.0]):
        tensor = mt.from_numpy(np.array(values * 50_000, dtype=dtype))
        assert tensor.argmax().numpy() == 0
        assert tensor.argmin().numpy() == 0


def test_bool_argument_finds_the_first_of_its_kind():
    values = np.zeros(100_000, dtype=bool)
    values[12_345] = True
    values[54_321] = True
    tensor = mt.from_numpy(values)
    assert tensor.argmax().numpy() == np.argmax(values) == 12_345
    assert tensor.argmin().numpy() == np.argmin(values) == 0

    assert mt.from_numpy(np.ones(1000, dtype=bool)).argmax().numpy() == 0
    assert mt.from_numpy(np.zeros(1000, dtype=bool)).argmax().numpy() == 0


def test_the_answer_does_not_depend_on_the_thread_count():
    # The chunks are folded by rayon, so a tie resolved by combine order rather
    # than by index would move with the worker count.
    import os
    import subprocess
    import sys

    script = (
        "import numpy as np, minitensor as mt\n"
        "a = np.random.default_rng(9).standard_normal(200_000).astype(np.float32)\n"
        "a[7] = a[80_000] = a[199_999] = 1e30\n"
        "a[11] = a[90_000] = -1e30\n"
        "t = mt.from_numpy(a)\n"
        "print(t.argmax().numpy(), t.argmin().numpy())\n"
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

    assert run("1") == run("2") == run("8") == "7 11"
