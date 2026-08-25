# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""Bitwise (`&`, `|`, `^`, `<<`, `>>`) and logical (`and`/`or`/`xor`/`not`) ops.

NumPy is the reference for every value here except the two places where it has
no answer to give: shift counts at or past the dtype's width, and negative shift
counts, both of which are undefined in C and so in NumPy.
"""

import numpy as np
import pytest

import minitensor as mt

INT_A = np.array([[0b1100, 5, 0], [-1, 2**40 + 3, -9]], dtype=np.int64)
INT_B = np.array([[0b1010, 3, 7], [0, 2**33, 4]], dtype=np.int64)
BOOL_A = np.array([False, False, True, True])
BOOL_B = np.array([False, True, False, True])
# Every kind of value a truth test has to rule on: signed zeros, a negative, a
# subnormal that is not zero, and both non-finite kinds.
MIXED_FLOAT = np.array([0.0, -0.0, -2.5, 5e-324, np.nan, np.inf], dtype=np.float64)


@pytest.mark.parametrize(
    "op, reference",
    [
        (lambda a, b: a & b, np.bitwise_and),
        (lambda a, b: a | b, np.bitwise_or),
        (lambda a, b: a ^ b, np.bitwise_xor),
    ],
)
def test_bitwise_operators_match_numpy_on_integers(op, reference):
    result = op(mt.from_numpy(INT_A.copy()), mt.from_numpy(INT_B.copy()))
    assert "int64" in str(result.dtype)
    np.testing.assert_array_equal(result.numpy(), reference(INT_A, INT_B))


@pytest.mark.parametrize(
    "op, reference",
    [
        (lambda a, b: a & b, np.bitwise_and),
        (lambda a, b: a | b, np.bitwise_or),
        (lambda a, b: a ^ b, np.bitwise_xor),
    ],
)
def test_bitwise_operators_are_logic_on_booleans(op, reference):
    result = op(mt.from_numpy(BOOL_A.copy()), mt.from_numpy(BOOL_B.copy()))
    assert "bool" in str(result.dtype)
    np.testing.assert_array_equal(result.numpy(), reference(BOOL_A, BOOL_B))


def test_bitwise_not_matches_numpy():
    np.testing.assert_array_equal(
        (~mt.from_numpy(INT_A.copy())).numpy(), np.bitwise_not(INT_A)
    )
    np.testing.assert_array_equal(
        (~mt.from_numpy(BOOL_A.copy())).numpy(), np.bitwise_not(BOOL_A)
    )


def test_methods_functions_and_operators_agree():
    a, b = mt.from_numpy(INT_A.copy()), mt.from_numpy(INT_B.copy())
    for name, operator in [
        ("bitwise_and", lambda x, y: x & y),
        ("bitwise_or", lambda x, y: x | y),
        ("bitwise_xor", lambda x, y: x ^ y),
        ("bitwise_left_shift", lambda x, y: x << y),
        ("bitwise_right_shift", lambda x, y: x >> y),
    ]:
        shift = "shift" in name
        rhs = mt.from_numpy(np.array([1, 2, 3], dtype=np.int64)) if shift else b
        expected = operator(a, rhs).numpy()
        np.testing.assert_array_equal(getattr(a, name)(rhs).numpy(), expected)
        np.testing.assert_array_equal(getattr(mt, name)(a, rhs).numpy(), expected)
        np.testing.assert_array_equal(
            getattr(mt.functional, name)(a, rhs).numpy(), expected
        )

    expected = (~a).numpy()
    np.testing.assert_array_equal(a.bitwise_not().numpy(), expected)
    np.testing.assert_array_equal(mt.bitwise_not(a).numpy(), expected)


def test_python_scalars_work_on_either_side():
    a = mt.from_numpy(INT_A.copy())
    np.testing.assert_array_equal((a & 3).numpy(), INT_A & 3)
    np.testing.assert_array_equal((3 & a).numpy(), 3 & INT_A)
    np.testing.assert_array_equal((a | 3).numpy(), INT_A | 3)
    np.testing.assert_array_equal((3 | a).numpy(), 3 | INT_A)
    np.testing.assert_array_equal((a ^ 3).numpy(), INT_A ^ 3)
    np.testing.assert_array_equal((3 ^ a).numpy(), 3 ^ INT_A)
    np.testing.assert_array_equal((a << 2).numpy(), INT_A << 2)
    np.testing.assert_array_equal((a >> 2).numpy(), INT_A >> 2)
    # The reversed shifts put the scalar in the value position, not the count.
    counts = mt.from_numpy(np.array([0, 1, 2], dtype=np.int64))
    np.testing.assert_array_equal((5 << counts).numpy(), 5 << np.array([0, 1, 2]))
    np.testing.assert_array_equal((-9 >> counts).numpy(), -9 >> np.array([0, 1, 2]))


def test_bitwise_broadcasts_and_promotes():
    mask = mt.from_numpy(np.array([[True], [False]]))
    values = mt.from_numpy(np.array([7, 8], dtype=np.int32))
    result = mask & values
    assert "int32" in str(result.dtype)
    np.testing.assert_array_equal(
        result.numpy(), np.array([[True], [False]]) & np.array([7, 8], dtype=np.int32)
    )

    wide = mt.from_numpy(np.array([1, 2], dtype=np.int64))
    narrow = mt.from_numpy(np.array([3, 3], dtype=np.int32))
    assert "int64" in str((narrow | wide).dtype)


def test_shifts_match_numpy_within_the_defined_range():
    values = np.array([1, 3, -3, 0, -1], dtype=np.int64)
    counts = np.array([0, 1, 2, 5, 62], dtype=np.int64)
    a, b = mt.from_numpy(values.copy()), mt.from_numpy(counts.copy())
    np.testing.assert_array_equal((a << b).numpy(), values << counts)
    np.testing.assert_array_equal((a >> b).numpy(), values >> counts)

    values32 = np.array([1, 3, -3, 0], dtype=np.int32)
    counts32 = np.array([0, 1, 2, 30], dtype=np.int32)
    a32, b32 = mt.from_numpy(values32.copy()), mt.from_numpy(counts32.copy())
    assert "int32" in str((a32 << b32).dtype)
    np.testing.assert_array_equal((a32 << b32).numpy(), values32 << counts32)
    np.testing.assert_array_equal((a32 >> b32).numpy(), values32 >> counts32)


@pytest.mark.parametrize("dtype, width", [(np.int32, 32), (np.int64, 64)])
def test_shift_counts_past_the_width_converge(dtype, width):
    # Undefined in C and so in NumPy; taken here as the limit of the operation.
    # Everything is shifted out, leaving zero -- except an arithmetic right
    # shift of a negative value, which smears the sign bit to -1.
    values = mt.from_numpy(np.array([1, -1, 6, 0], dtype=dtype))
    for count in (width, width + 1, 4 * width):
        counts = mt.from_numpy(np.full(4, count, dtype=dtype))
        np.testing.assert_array_equal((values << counts).numpy(), [0, 0, 0, 0])
        np.testing.assert_array_equal((values >> counts).numpy(), [0, -1, 0, 0])

    # The last count below the width still moves a bit rather than clearing it.
    one = mt.from_numpy(np.array([1], dtype=dtype))
    edge = mt.from_numpy(np.array([width - 1], dtype=dtype))
    assert (one << edge).numpy()[0] == np.iinfo(dtype).min


def test_negative_shift_counts_are_rejected():
    values = mt.from_numpy(np.array([1, 2], dtype=np.int64))
    counts = mt.from_numpy(np.array([1, -1], dtype=np.int64))
    for shift in (lambda: values << counts, lambda: values >> counts):
        with pytest.raises(ValueError, match="non-negative shift counts"):
            shift()


def test_bitwise_rejects_floats():
    a = mt.from_numpy(np.array([1.0, 2.0], dtype=np.float32))
    b = mt.from_numpy(np.array([1, 2], dtype=np.int64))
    for call in (
        lambda: a & b,
        lambda: a | b,
        lambda: b ^ a,
        lambda: a << b,
        lambda: b >> a,
        lambda: ~a,
    ):
        with pytest.raises(ValueError):
            call()


def test_shifting_two_boolean_tensors_is_rejected():
    mask = mt.from_numpy(BOOL_A.copy())
    with pytest.raises(ValueError, match="boolean"):
        mask << mask

    # A boolean paired with an integer count promotes and shifts.
    counts = mt.from_numpy(np.full(4, 3, dtype=np.int64))
    np.testing.assert_array_equal(
        (mask << counts).numpy(), BOOL_A.astype(np.int64) << 3
    )


@pytest.mark.parametrize(
    "name, reference",
    [
        ("logical_and", np.logical_and),
        ("logical_or", np.logical_or),
        ("logical_xor", np.logical_xor),
    ],
)
def test_logical_ops_match_numpy_over_every_dtype_pair(name, reference):
    left = MIXED_FLOAT
    right = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float64)
    expected = reference(left, right)

    a, b = mt.from_numpy(left.copy()), mt.from_numpy(right.copy())
    for call in (
        getattr(a, name),
        lambda other: getattr(mt, name)(a, other),
        lambda other: getattr(mt.functional, name)(a, other),
    ):
        result = call(b)
        assert "bool" in str(result.dtype)
        np.testing.assert_array_equal(result.numpy(), expected)

    # Mixed dtypes need no common numeric type: each side is reduced to truth
    # values on its own.
    ints = np.array([0, 1, 0, 2, 0, 3], dtype=np.int32)
    np.testing.assert_array_equal(
        getattr(mt, name)(a, mt.from_numpy(ints.copy())).numpy(),
        reference(left, ints),
    )


def test_logical_not_matches_numpy():
    a = mt.from_numpy(MIXED_FLOAT.copy())
    expected = np.logical_not(MIXED_FLOAT)
    for result in (a.logical_not(), mt.logical_not(a), mt.functional.logical_not(a)):
        assert "bool" in str(result.dtype)
        np.testing.assert_array_equal(result.numpy(), expected)

    # Unlike `~`, this accepts floats, because it asks about zero rather than
    # about bits.
    with pytest.raises(ValueError):
        ~a


def test_logical_ops_broadcast():
    a = mt.from_numpy(np.array([[1.0], [0.0]], dtype=np.float32))
    b = mt.from_numpy(np.array([True, False]))
    result = mt.logical_or(a, b)
    assert result.shape == (2, 2)
    np.testing.assert_array_equal(
        result.numpy(),
        np.logical_or(np.array([[1.0], [0.0]]), np.array([True, False])),
    )


def test_logical_ops_agree_with_bitwise_on_booleans():
    a, b = mt.from_numpy(BOOL_A.copy()), mt.from_numpy(BOOL_B.copy())
    np.testing.assert_array_equal(mt.logical_and(a, b).numpy(), (a & b).numpy())
    np.testing.assert_array_equal(mt.logical_or(a, b).numpy(), (a | b).numpy())
    np.testing.assert_array_equal(mt.logical_xor(a, b).numpy(), (a ^ b).numpy())
    np.testing.assert_array_equal(mt.logical_not(a).numpy(), (~a).numpy())


def test_results_do_not_track_gradients():
    a = mt.from_numpy(np.array([1.0, 0.0], dtype=np.float32), requires_grad=True)
    b = mt.from_numpy(np.array([0, 1], dtype=np.int64))
    assert not mt.logical_and(a, b).requires_grad
    assert not mt.logical_not(a).requires_grad
    assert not (b & b).requires_grad
    assert not (b << b).requires_grad


def test_empty_and_mismatched_shapes():
    empty = mt.from_numpy(np.array([], dtype=np.int64))
    assert (empty & empty).shape == (0,)
    assert mt.logical_and(empty, empty).shape == (0,)

    a = mt.from_numpy(np.array([1, 2, 3], dtype=np.int64))
    b = mt.from_numpy(np.array([1, 2], dtype=np.int64))
    with pytest.raises(ValueError):
        a | b
    with pytest.raises(ValueError):
        mt.logical_or(a, b)
