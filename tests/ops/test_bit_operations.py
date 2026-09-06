# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""The bit-level operations NumPy has and this library did not.

Three of them are second spellings -- `left_shift`, `right_shift` and `invert`
are NumPy's names for shifts and complement that already existed -- and the
tests for those check the *identity* of the objects, because that is the whole
claim: one implementation under two names cannot drift apart, and a copied one
can.

`bitwise_count` is the operation with a decision in it. NumPy counts the bits
of the *absolute value*, so `-3` answers 2 at every width; counting the two's
complement representation instead would answer 31 at int32 and 63 at int64 for
the same number. The most negative value of a width is the case that separates
a correct implementation from one that used `abs`: its magnitude is one past
what the signed type holds.

`packbits` and `unpackbits` are checked against NumPy exhaustively over shapes,
axes, bit orders and counts, with one documented divergence: NumPy pads an
*empty* input by reading uninitialised memory, and this library answers zeros.
"""

from __future__ import annotations

import numpy as np
import pytest

import minitensor as mt

INTEGER_DTYPES = ["int32", "int64"]


@pytest.mark.parametrize(
    "alias,target",
    [
        ("left_shift", "bitwise_left_shift"),
        ("right_shift", "bitwise_right_shift"),
        ("invert", "bitwise_not"),
        ("popcount", "bitwise_count"),
    ],
)
def test_the_numpy_spellings_are_the_same_object(alias, target):
    # Not "give the same answer": the same object, so there is nothing to
    # diverge. Both the free function and the method.
    assert getattr(mt, alias) is getattr(mt, target)
    assert getattr(mt.functional, alias) is getattr(mt.functional, target)
    assert getattr(mt.Tensor, alias) is getattr(mt.Tensor, target)


@pytest.mark.parametrize("dtype", INTEGER_DTYPES)
def test_the_numpy_spellings_answer_what_numpy_answers(dtype):
    values = np.array([-8, -3, -1, 0, 1, 2, 7, 255], dtype=dtype)
    counts = np.array([0, 1, 2, 3, 4, 5, 6, 7], dtype=dtype)
    tensor, by = mt.from_numpy(values), mt.from_numpy(counts)

    np.testing.assert_array_equal(
        mt.left_shift(tensor, by).numpy(), np.left_shift(values, counts)
    )
    np.testing.assert_array_equal(
        mt.right_shift(tensor, by).numpy(), np.right_shift(values, counts)
    )
    np.testing.assert_array_equal(mt.invert(tensor).numpy(), np.invert(values))


@pytest.mark.parametrize("dtype", INTEGER_DTYPES)
def test_bitwise_count_matches_numpy_over_the_whole_byte_range(dtype):
    values = np.arange(-256, 257, dtype=dtype)
    counted = mt.bitwise_count(mt.from_numpy(values)).numpy()
    np.testing.assert_array_equal(counted, np.bitwise_count(values))


def test_bitwise_count_describes_the_number_not_its_width():
    # The same value at two widths: the count is of the magnitude, so the
    # storage does not show through. Counting the representation would give
    # 31 and 63 here.
    values = [-8, -3, -1, 0, 1, 2, 7, 255]
    narrow = mt.bitwise_count(mt.from_numpy(np.array(values, dtype="int32"))).numpy()
    wide = mt.bitwise_count(mt.from_numpy(np.array(values, dtype="int64"))).numpy()
    np.testing.assert_array_equal(narrow, wide)
    np.testing.assert_array_equal(narrow, [1, 2, 1, 0, 1, 1, 3, 8])


@pytest.mark.parametrize(
    "dtype,most_negative", [("int32", -(2**31)), ("int64", -(2**63))]
)
def test_bitwise_count_answers_the_value_abs_cannot_reach(dtype, most_negative):
    # The magnitude of the most negative value is one past what the signed type
    # holds, so an implementation built on `abs` overflows here and nowhere
    # else. It is a single set bit.
    counted = mt.bitwise_count(mt.from_numpy(np.array([most_negative], dtype=dtype)))
    np.testing.assert_array_equal(counted.numpy(), [1])


def test_bitwise_count_takes_booleans_and_refuses_floats():
    flags = mt.from_numpy(np.array([True, False, True]))
    np.testing.assert_array_equal(mt.bitwise_count(flags).numpy(), [1, 0, 1])
    with pytest.raises(Exception, match="boolean and integer"):
        mt.bitwise_count(mt.from_numpy(np.array([1.0, 2.0])))


def test_bitwise_count_is_int32_whatever_went_in():
    # At most 64, so carrying it at the input's width would cost eight bytes
    # an element to say nothing more.
    wide = mt.from_numpy(np.arange(8, dtype="int64"))
    assert str(mt.bitwise_count(wide).dtype) == "int32"
    assert str(mt.bitwise_count(wide.astype("bool")).dtype) == "int32"


SHAPES = [(8,), (9,), (16,), (2, 9), (3, 5, 7), (0,), (2, 0)]
ORDERS = ["big", "little"]


def _bits(shape, seed=0):
    return (np.random.default_rng(seed).random(shape) > 0.5).astype(np.uint8)


@pytest.mark.parametrize("shape", SHAPES, ids=[str(s) for s in SHAPES])
@pytest.mark.parametrize("bitorder", ORDERS)
def test_packbits_matches_numpy_on_every_axis(shape, bitorder):
    flags = _bits(shape)
    tensor = mt.from_numpy(flags.astype(np.int32))
    for dim in [None, *range(len(shape)), -1]:
        packed = mt.packbits(tensor, dim, bitorder).numpy()
        expected = np.packbits(flags, axis=dim, bitorder=bitorder)
        assert packed.shape == expected.shape
        np.testing.assert_array_equal(packed, expected)


@pytest.mark.parametrize("shape", SHAPES, ids=[str(s) for s in SHAPES])
@pytest.mark.parametrize("bitorder", ORDERS)
def test_unpackbits_matches_numpy_including_the_counts(shape, bitorder):
    flags = _bits(shape, seed=1)
    for dim in [None, *range(len(shape)), -1]:
        packed = np.packbits(flags, axis=dim, bitorder=bitorder)
        tensor = mt.from_numpy(packed.astype(np.int32))
        for count in (None, 0, 3, 8, 9, 17, -1, -7, -100):
            if packed.size == 0 and count is not None and count > 0:
                # NumPy pads an empty input by reading uninitialised memory:
                # `np.unpackbits(np.array([], dtype=np.uint8), count=3)` gives
                # a different answer than zeros, and the same one every time
                # only because the page happens not to move. This library
                # answers zeros, which is the documented meaning of padding.
                assert not mt.unpackbits(tensor, dim, count, bitorder).numpy().any()
                continue
            try:
                expected = np.unpackbits(
                    packed, axis=dim, count=count, bitorder=bitorder
                )
            except Exception:
                with pytest.raises(ValueError):
                    mt.unpackbits(tensor, dim, count, bitorder)
                continue
            unpacked = mt.unpackbits(tensor, dim, count, bitorder).numpy()
            assert unpacked.shape == expected.shape
            np.testing.assert_array_equal(unpacked, expected)


@pytest.mark.parametrize("length", [1, 7, 8, 9, 13, 64])
@pytest.mark.parametrize("bitorder", ORDERS)
def test_unpackbits_undoes_packbits_when_told_the_length(length, bitorder):
    # The round trip is exact only with `count`: packing pads the axis up to a
    # multiple of eight, and nothing in the packed tensor records how much.
    flags = _bits((4, length), seed=2)
    packed = mt.packbits(mt.from_numpy(flags.astype(np.int32)), 1, bitorder)
    restored = mt.unpackbits(packed, 1, length, bitorder).numpy()
    np.testing.assert_array_equal(restored, flags)


def test_packbits_takes_any_non_zero_as_a_set_bit():
    values = np.array([0, 1, 5, -3, 0, 0, 0, 0], dtype=np.int32)
    np.testing.assert_array_equal(
        mt.packbits(mt.from_numpy(values)).numpy(),
        np.packbits((values != 0).astype(np.uint8)),
    )


def test_packbits_and_unpackbits_refuse_what_has_no_bits():
    with pytest.raises(TypeError, match="boolean or integer"):
        mt.packbits(mt.from_numpy(np.array([1.0, 0.0])))
    with pytest.raises(TypeError, match="boolean or integer"):
        mt.unpackbits(mt.from_numpy(np.array([1.0, 0.0])))
    with pytest.raises(ValueError, match="0\\.\\.255"):
        mt.unpackbits(mt.from_numpy(np.array([256], dtype=np.int32)))
    with pytest.raises(ValueError, match="0\\.\\.255"):
        mt.unpackbits(mt.from_numpy(np.array([-1], dtype=np.int32)))
    byte = mt.from_numpy(np.array([1], dtype=np.int32))
    with pytest.raises(ValueError, match="bitorder"):
        mt.packbits(byte, bitorder="mid")
    with pytest.raises(ValueError, match="bitorder"):
        mt.unpackbits(byte, bitorder="mid")


def test_packbits_and_unpackbits_check_the_axis():
    tensor = mt.from_numpy(np.array([[1, 0], [0, 1]], dtype=np.int32))
    with pytest.raises(IndexError, match="rank 2"):
        mt.packbits(tensor, 2)
    with pytest.raises(IndexError, match="rank 2"):
        mt.unpackbits(tensor, -3)


def test_packbits_answers_int32_where_numpy_answers_uint8():
    # The one place the missing unsigned byte shows. The values are NumPy's;
    # only the box is wider, and the cast recovers NumPy's array exactly.
    flags = _bits((3, 16), seed=3)
    packed = mt.packbits(mt.from_numpy(flags.astype(np.int32)), 1)
    assert str(packed.dtype) == "int32"
    np.testing.assert_array_equal(
        packed.numpy().astype(np.uint8), np.packbits(flags, axis=1)
    )
