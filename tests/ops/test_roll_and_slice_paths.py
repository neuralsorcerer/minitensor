# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""`roll` and `slice` each pick between two copy strategies; all of them agree.

Both operations used to copy far more than they had to.

`slice` copied the selected run as `count` separate runs of `inner` elements.
With a unit step that whole region is contiguous, so it is one copy -- and when
`inner` was 1, which is every slice along the last dimension and every slice of
a 1-D tensor, the old shape meant copying one element at a time.

`roll` was built out of slice + slice + concatenate, once per rolled dimension:
three allocations and three passes each, on top of the elementwise slice above.
A roll is an index remapping, so every dimension is now resolved in a single
pass. Measured on this machine, 4096x1024 float32, with NumPy timed in the same
interleaved rounds:

                     before     after    NumPy
    flat roll       21.37 ms   2.34 ms   1.57 ms
    roll dim=0       5.29 ms   1.26 ms   1.91 ms
    roll dim=1       6.96 ms   1.26 ms   2.76 ms
    roll both dims  22.78 ms   1.27 ms   2.96 ms
    narrow, 1-D 2M  12.72 ms   0.95 ms   0.62 ms

Everything but the 1-D cases now runs at or under NumPy. Those two stay a little
over because a single row is one task: the copy is bandwidth-bound and NumPy is
single-threaded there too.

Which path runs depends on the element count against `PAR_THRESHOLD`, so the
sizes below deliberately straddle 131072. The multi-dimensional cases matter
most: shifts are now accumulated per dimension and applied at once rather than
one whole tensor at a time, so a repeated dimension has to still compose the way
rolling it twice did, and the source-row arithmetic has to agree with NumPy for
every combination of rolled and unrolled axes.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

import minitensor as mt

DTYPES = ["float32", "float64", "int32", "int64", "bool"]

# Straddles PAR_THRESHOLD (131072 elements) in both directions, and covers
# rank 1 through 4, unit axes, and axes shorter than the shift.
SHAPES = [
    (1,),
    (7,),
    (2, 3),
    (1, 6),
    (6, 1),
    (3, 4, 5),
    (2, 1, 3),
    (2, 3, 4, 5),
    (400, 400),  # 160000 elements: above the threshold
    (70000,),  # a single long row, below it
    (200000,),  # a single long row, above it
]

EMPTY_SHAPES = [(0,), (0, 3), (3, 0), (2, 0, 4)]


def _values(shape, dtype, seed=0):
    rng = np.random.default_rng(seed)
    if dtype == "bool":
        return rng.integers(0, 2, size=shape).astype(dtype)
    return (rng.standard_normal(shape) * 10).astype(dtype)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", SHAPES, ids=[str(s) for s in SHAPES])
@pytest.mark.parametrize("shift", [-9, -1, 0, 1, 3, 8])
def test_flat_roll_matches_numpy(shape, dtype, shift):
    values = _values(shape, dtype)
    got = mt.Tensor(values, dtype=dtype).roll(shift).numpy()
    np.testing.assert_array_equal(got, np.roll(values, shift))


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", SHAPES, ids=[str(s) for s in SHAPES])
def test_roll_along_each_dim_matches_numpy(shape, dtype):
    values = _values(shape, dtype)
    tensor = mt.Tensor(values, dtype=dtype)

    for dim in range(len(shape)):
        size = shape[dim]
        # Straddle the axis length so the wrap is exercised in both directions,
        # exhaustively while that is cheap and at the boundaries when it is not.
        if size <= 8:
            shifts = range(-2 * size - 1, 2 * size + 2)
        else:
            shifts = (
                -2 * size - 1,
                -size,
                -size + 1,
                -1,
                0,
                1,
                size - 1,
                size,
                2 * size + 1,
            )
        for shift in shifts:
            np.testing.assert_array_equal(
                tensor.roll([shift], [dim]).numpy(),
                np.roll(values, shift, axis=dim),
                err_msg=f"shape={shape} dim={dim} shift={shift}",
            )


@pytest.mark.parametrize("shape", SHAPES, ids=[str(s) for s in SHAPES])
def test_negative_dims_name_the_same_axis(shape):
    values = _values(shape, "float32")
    tensor = mt.Tensor(values)

    for dim in range(len(shape)):
        np.testing.assert_array_equal(
            tensor.roll([2], [dim - len(shape)]).numpy(),
            tensor.roll([2], [dim]).numpy(),
        )


@pytest.mark.parametrize("shape", [(3, 4, 5), (2, 3, 4, 5), (400, 400)])
def test_rolling_several_dims_at_once_equals_rolling_them_in_turn(shape):
    """The dimensions are now applied in one pass instead of one tensor each,
    so the combined result has to match the sequence it replaced."""
    values = _values(shape, "float32")
    tensor = mt.Tensor(values)

    for dims in itertools.combinations(range(len(shape)), 2):
        for shifts in [(1, 1), (-2, 3), (0, 4), (5, -7), (0, 0)]:
            expected = values
            for shift, dim in zip(shifts, dims):
                expected = np.roll(expected, shift, axis=dim)
            np.testing.assert_array_equal(
                tensor.roll(list(shifts), list(dims)).numpy(),
                expected,
                err_msg=f"shape={shape} dims={dims} shifts={shifts}",
            )


def test_every_dim_rolled_at_once():
    values = _values((3, 4, 5), "float32")
    got = mt.Tensor(values).roll([1, 2, 3], [0, 1, 2]).numpy()
    np.testing.assert_array_equal(got, np.roll(values, (1, 2, 3), axis=(0, 1, 2)))


@pytest.mark.parametrize("shifts", [(1, 1), (2, -2), (3, 5), (-1, 1)])
def test_a_repeated_dim_accumulates(shifts):
    """Shifts are reduced per dimension before the copy. Naming a dimension
    twice used to roll the tensor twice, so the two shifts have to add."""
    values = _values((7, 4), "float32")

    got = mt.Tensor(values).roll(list(shifts), [0, 0]).numpy()

    expected = np.roll(np.roll(values, shifts[0], axis=0), shifts[1], axis=0)
    np.testing.assert_array_equal(got, expected)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", EMPTY_SHAPES, ids=[str(s) for s in EMPTY_SHAPES])
def test_rolling_an_empty_tensor_keeps_its_shape(shape, dtype):
    """A zero-length axis makes a block zero elements long, which the copy
    cannot be handed."""
    values = _values(shape, dtype)
    tensor = mt.Tensor(values, dtype=dtype)

    assert tensor.roll(3).shape == list(shape)
    for dim in range(len(shape)):
        rolled = tensor.roll([3], [dim])
        assert rolled.shape == list(shape)
        np.testing.assert_array_equal(rolled.numpy(), np.roll(values, 3, axis=dim))


def test_a_shift_that_reduces_to_zero_is_the_identity():
    values = _values((6, 8), "float32")
    tensor = mt.Tensor(values)

    for shifts, dims in [([6], [0]), ([-8], [1]), ([12], [0]), ([0], [0])]:
        np.testing.assert_array_equal(tensor.roll(shifts, dims).numpy(), values)
    np.testing.assert_array_equal(tensor.roll(48).numpy(), values)


def test_roll_is_undone_by_the_opposite_shift():
    values = _values((5, 6, 7), "float32")
    tensor = mt.Tensor(values)

    there = tensor.roll([2, -3, 4], [0, 1, 2])
    back = there.roll([-2, 3, -4], [0, 1, 2])
    np.testing.assert_array_equal(back.numpy(), values)


@pytest.mark.parametrize("shape", [(5,), (4, 6), (3, 4, 5)])
def test_roll_gradient_is_the_opposite_roll(shape):
    """The forward is a permutation, so the backward is its inverse -- and the
    internal copy must not leave gradient edges of its own behind."""
    values = _values(shape, "float32")
    weights = _values(shape, "float32", seed=1)

    x = mt.Tensor(values, requires_grad=True)
    dims = list(range(len(shape)))
    shifts = [2, -1, 3][: len(shape)]
    (x.roll(shifts, dims) * mt.Tensor(weights)).sum().backward()

    expected = weights
    for shift, dim in zip(reversed(shifts), reversed(dims)):
        expected = np.roll(expected, -shift, axis=dim)
    np.testing.assert_allclose(mt.get_gradient(x).numpy(), expected, rtol=1e-6)


def test_flat_roll_gradient_is_the_opposite_roll():
    values = _values((4, 6), "float32")
    weights = _values((4, 6), "float32", seed=2)

    x = mt.Tensor(values, requires_grad=True)
    (x.roll(5) * mt.Tensor(weights)).sum().backward()

    np.testing.assert_allclose(
        mt.get_gradient(x).numpy(), np.roll(weights, -5), rtol=1e-6
    )


def test_shifts_and_dims_must_agree_in_length():
    with pytest.raises(Exception):
        mt.randn(3, 4).roll([1, 2], [0])


@pytest.mark.parametrize("dim", [-3, 2, 9])
def test_roll_rejects_an_out_of_range_dim(dim):
    with pytest.raises(IndexError):
        mt.randn(3, 4).roll([1], [dim])


# --- slice, which roll used to be built out of -------------------------------


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", [(9,), (4, 6), (3, 4, 5), (400, 400)])
def test_narrow_matches_numpy_over_every_range(shape, dtype):
    """`narrow` is `slice` with a unit step -- the path that now copies each
    block whole. The last dimension is the case that used to go one element at
    a time."""
    values = _values(shape, dtype)
    tensor = mt.Tensor(values, dtype=dtype)

    for dim in range(len(shape)):
        size = shape[dim]
        starts = range(size + 1) if size <= 9 else (0, 1, size // 2, size)
        for start in starts:
            room = size - start
            lengths = (
                range(room + 1) if size <= 9 else {0, min(1, room), room // 2, room}
            )
            for length in sorted(lengths):
                got = tensor.narrow(dim, start, length).numpy()
                expected = np.take(values, range(start, start + length), axis=dim)
                assert got.shape == expected.shape
                np.testing.assert_array_equal(
                    got, expected, err_msg=f"{shape} dim={dim} {start}+{length}"
                )


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", [(12,), (4, 6), (3, 4, 6), (400, 400)])
def test_chunk_matches_numpy_split(shape, dtype):
    values = _values(shape, dtype)
    tensor = mt.Tensor(values, dtype=dtype)

    for dim in range(len(shape)):
        for parts in (1, 2, 3):
            if shape[dim] % parts:
                continue
            got = [piece.numpy() for piece in tensor.chunk(parts, dim)]
            expected = np.split(values, parts, axis=dim)
            assert len(got) == len(expected)
            for a, b in zip(got, expected):
                np.testing.assert_array_equal(a, b)


def test_narrow_gradient_reaches_only_the_selected_range():
    x = mt.Tensor(_values((4, 5), "float32"), requires_grad=True)

    x.narrow(1, 1, 3).sum().backward()

    expected = np.zeros((4, 5), dtype=np.float32)
    expected[:, 1:4] = 1.0
    np.testing.assert_array_equal(mt.get_gradient(x).numpy(), expected)


def test_a_narrow_of_the_whole_axis_is_an_exact_copy():
    """A degenerate range, and the one where an off-by-one in the block offset
    would be invisible against a partial selection."""
    values = _values((400, 400), "float32")
    tensor = mt.Tensor(values)

    for dim in (0, 1):
        np.testing.assert_array_equal(tensor.narrow(dim, 0, 400).numpy(), values)
