# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""Boolean operands are judged by the result dtype, not by the operand dtypes.

`-`, `//` and `%` have no meaning on two booleans -- there is no boolean
difference or quotient -- so they are rejected, which is what NumPy and PyTorch
do too. But the guards tested the *operands*, so they also rejected every mixed
pair, where the boolean promotes to the other operand's dtype and the operation
is ordinary arithmetic from there:

    counts - mask        # rejected, though counts + mask, counts * mask
                         # and counts / mask were all accepted

Nine pairs per operator went that way. `+`, `*` and `/` accepted exactly the
same operands, so the inconsistency was internal as much as it was a
divergence from NumPy.

The ordered comparisons had the mirror-image problem. `lt`, `le`, `gt` and `ge`
were rejected when *both* operands were boolean, and accepted when only one was
-- so `mask < counts` worked and `mask < other_mask` did not, even though
`false < true` is the same ordering `minimum` and `maximum` were already
applying to boolean tensors and `eq`/`ne` were already accepting.

Both are now decided by the promoted dtype: bool-with-bool for the three
arithmetic operators is the only rejected case, and ordering works everywhere.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

import minitensor as mt

NUMERIC = ["int32", "int64", "float32", "float64"]

MASK = np.array([True, False, True, True])
# No zeros: `//` and `%` reject a zero divisor for integers whatever the
# operand dtypes are, which is a separate rule and not what is under test here.
NUMBERS = {dtype: np.array([2, 3, 1, 4], dtype=dtype) for dtype in NUMERIC}

ARITHMETIC = {
    "sub": (lambda a, b: a - b, lambda a, b: a - b),
    "floor_divide": (lambda a, b: a // b, lambda a, b: a.floor_divide(b)),
    "remainder": (np.remainder, lambda a, b: a.remainder(b)),
}

ORDERED = {
    "lt": (lambda a, b: a < b, lambda a, b: a.lt(b)),
    "le": (lambda a, b: a <= b, lambda a, b: a.le(b)),
    "gt": (lambda a, b: a > b, lambda a, b: a.gt(b)),
    "ge": (lambda a, b: a >= b, lambda a, b: a.ge(b)),
}


def _t(array, dtype):
    return mt.Tensor(array, dtype=dtype)


# --- the pairs that were rejected and should not have been ------------------


@pytest.mark.parametrize("op", list(ARITHMETIC), ids=list(ARITHMETIC))
@pytest.mark.parametrize("dtype", NUMERIC)
@pytest.mark.parametrize("bool_side", ["left", "right"], ids=["mask_op_n", "n_op_mask"])
def test_a_bool_with_a_number_promotes_and_computes(op, dtype, bool_side):
    npf, mtf = ARITHMETIC[op]
    number = NUMBERS[dtype]
    # A boolean divisor would be dividing by `False` half the time, so when the
    # mask is on the right of `//` or `%` it is all-True.
    mask = MASK if bool_side == "left" else np.ones(4, bool)
    a, b = (mask, number) if bool_side == "left" else (number, mask)

    result = mtf(
        _t(a, "bool" if bool_side == "left" else dtype),
        _t(b, dtype if bool_side == "left" else "bool"),
    )
    expected = npf(a, b)

    np.testing.assert_array_equal(result.numpy(), expected)
    assert str(result.dtype) == expected.dtype.name


@pytest.mark.parametrize("op", list(ARITHMETIC), ids=list(ARITHMETIC))
@pytest.mark.parametrize("dtype", NUMERIC)
def test_it_agrees_with_the_operators_that_already_accepted_those_operands(op, dtype):
    """`+`, `*` and `/` never rejected a boolean operand. The three that did
    now land on the same dtype for the same inputs."""
    _, mtf = ARITHMETIC[op]
    mask, number = MASK, NUMBERS[dtype]

    got = mtf(_t(mask, "bool"), _t(number, dtype)).dtype
    assert str(got) == str((_t(mask, "bool") + _t(number, dtype)).dtype)


# --- the pair that stays rejected -------------------------------------------


@pytest.mark.parametrize("op", list(ARITHMETIC), ids=list(ARITHMETIC))
def test_two_booleans_have_no_result_dtype(op):
    _, mtf = ARITHMETIC[op]
    with pytest.raises(Exception) as excinfo:
        mtf(_t(MASK, "bool"), _t(np.ones(4, bool), "bool"))
    assert "boolean" in str(excinfo.value)


def test_the_subtraction_message_offers_the_alternative():
    """`a != b` is the boolean difference, so the refusal names it rather than
    only stating the rule."""
    with pytest.raises(Exception) as excinfo:
        _t(MASK, "bool") - _t(np.ones(4, bool), "bool")
    message = str(excinfo.value)
    assert "ne" in message and "xor" in message

    # and the suggestion is the operation the caller wanted
    difference = _t(MASK, "bool").ne(_t(np.ones(4, bool), "bool"))
    np.testing.assert_array_equal(difference.numpy(), MASK != np.ones(4, bool))


# --- ordering ---------------------------------------------------------------


@pytest.mark.parametrize("op", list(ORDERED), ids=list(ORDERED))
def test_ordered_comparisons_accept_two_booleans(op):
    npf, mtf = ORDERED[op]
    left = np.array([True, False, True, False])
    right = np.array([False, True, True, False])

    result = mtf(_t(left, "bool"), _t(right, "bool"))
    np.testing.assert_array_equal(result.numpy(), npf(left, right))
    assert str(result.dtype) == "bool"


@pytest.mark.parametrize("op", list(ORDERED), ids=list(ORDERED))
def test_ordering_is_the_one_minimum_and_maximum_already_used(op):
    """`minimum`/`maximum` accepted boolean tensors the whole time, which is the
    same `false < true` ordering. Rejecting `lt` on them was the outlier."""
    left = np.array([True, False, True, False])
    right = np.array([False, True, True, False])
    a, b = _t(left, "bool"), _t(right, "bool")

    np.testing.assert_array_equal(
        a.minimum(b).numpy(), np.where(a.lt(b).numpy(), left, right)
    )
    np.testing.assert_array_equal(
        a.maximum(b).numpy(), np.where(a.gt(b).numpy(), left, right)
    )


@pytest.mark.parametrize("op", list(ORDERED) + ["eq", "ne"])
def test_every_dtype_pairing_matches_numpy(op):
    """Including the mixed pairs, which already worked -- the bool/bool case
    now joins them instead of being the one hole in the table."""
    npf = {
        "lt": lambda a, b: a < b,
        "le": lambda a, b: a <= b,
        "gt": lambda a, b: a > b,
        "ge": lambda a, b: a >= b,
        "eq": lambda a, b: a == b,
        "ne": lambda a, b: a != b,
    }[op]

    values = {"bool": np.array([True, False, True, False])}
    values.update({d: np.array([1, 0, 1, 2], dtype=d) for d in NUMERIC})

    for d1, d2 in itertools.product(values, repeat=2):
        a, b = values[d1], values[d2]
        result = getattr(_t(a, d1), op)(_t(b, d2))
        np.testing.assert_array_equal(
            result.numpy(), npf(a, b), err_msg=f"{d1} {op} {d2}"
        )


def test_broadcasting_still_applies_to_boolean_comparisons():
    left = np.array([[True, False], [False, True]])
    right = np.array([True, False])
    np.testing.assert_array_equal(
        _t(left, "bool").lt(_t(right, "bool")).numpy(), left < right
    )
