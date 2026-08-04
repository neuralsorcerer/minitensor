# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""A matmul that cannot be done should name both operands as they were passed.

The old message synthesised an "expected" shape out of one dimension of each
operand, so `(3, 4) @ (5, 6)` reported

    Shape mismatch: expected [4, 6], got [5, 6]
    💡 Suggestion: ... Use .view() or .reshape() to change the tensor shape

`[4, 6]` was never a shape either operand had, the `(3, 4)` operand went
unmentioned, and reshaping to `[4, 6]` is not how anyone fixes this. Worse,
1-D operands were promoted to matrices *before* anything checked them, so
`(3, 4) @ (7,)` was reported as "expected [4, 1], got [7, 1]" -- three numbers
the caller never wrote.
"""

from __future__ import annotations

import numpy as np
import pytest

import minitensor as mt


def _t(*shape):
    return mt.randn(*shape)


BAD_PRODUCTS = [
    ((3, 4), (5, 6)),
    ((3, 4), (5, 4)),
    ((4, 3), (4, 5)),
    ((2, 3, 4), (2, 5, 6)),
    ((8, 3, 4), (5, 4, 6)),
    ((3, 4), (7,)),
    ((7,), (3, 4)),
    ((16, 128), (256, 64)),
]


@pytest.mark.parametrize("lhs_shape,rhs_shape", BAD_PRODUCTS)
def test_message_quotes_both_shapes_as_passed(lhs_shape, rhs_shape):
    with pytest.raises(Exception) as excinfo:
        mt.matmul(_t(*lhs_shape), _t(*rhs_shape))

    message = str(excinfo.value)
    assert f"{list(lhs_shape)} and {list(rhs_shape)}" in message, message


@pytest.mark.parametrize("lhs_shape,rhs_shape", BAD_PRODUCTS)
def test_message_invents_no_shape(lhs_shape, rhs_shape):
    """Every bracketed shape in the message must be one of the two operands'.

    The batch-mismatch case also prints the batch prefixes, which are slices of
    the real shapes rather than inventions.
    """
    with pytest.raises(Exception) as excinfo:
        mt.matmul(_t(*lhs_shape), _t(*rhs_shape))

    import re

    quoted = re.findall(r"\[[\d, ]*\]", str(excinfo.value))
    allowed = {
        str(list(lhs_shape)),
        str(list(rhs_shape)),
        str(list(lhs_shape[:-2])),
        str(list(rhs_shape[:-2])),
    }
    assert quoted, str(excinfo.value)
    for shape in quoted:
        assert shape in allowed, f"{shape} is not either operand: {excinfo.value}"


def test_one_dimensional_operand_is_described_as_a_length():
    """Not as the [n, 1] it gets promoted to internally."""
    with pytest.raises(Exception) as excinfo:
        mt.matmul(_t(3, 4), _t(7))
    message = str(excinfo.value)
    assert "the length of the second (7)" in message
    assert "[7, 1]" not in message
    assert "[4, 1]" not in message

    with pytest.raises(Exception) as excinfo:
        mt.matmul(_t(7), _t(3, 4))
    assert "the length of the first operand (7)" in str(excinfo.value)


def test_batch_mismatch_says_the_matrices_were_fine():
    """`(8,3,4) @ (5,4,6)` has agreeing inner dimensions; only the batch is bad."""
    with pytest.raises(Exception) as excinfo:
        mt.matmul(_t(8, 3, 4), _t(5, 4, 6))
    message = str(excinfo.value)
    assert "batch dimensions [8] and [5]" in message
    assert "matrix dimensions agree" in message


def test_transpose_hint_points_at_the_second_operand():
    with pytest.raises(Exception) as excinfo:
        mt.matmul(_t(3, 4), _t(5, 4))
    assert "b.transpose(-1, -2)" in str(excinfo.value)


def test_transpose_hint_points_at_the_first_operand():
    with pytest.raises(Exception) as excinfo:
        mt.matmul(_t(4, 3), _t(4, 5))
    assert "a.transpose(-1, -2)" in str(excinfo.value)


def test_no_transpose_hint_when_neither_would_help():
    """`(3,4) @ (5,6)`: transposing either operand still leaves 4 against 5."""
    with pytest.raises(Exception) as excinfo:
        mt.matmul(_t(3, 4), _t(5, 6))
    message = str(excinfo.value)
    assert "likely stored transposed" not in message
    assert "right way round" in message


def test_the_suggested_transpose_actually_fixes_it():
    """A hint that does not work is worse than no hint."""
    a, b = _t(3, 4), _t(5, 4)
    with pytest.raises(Exception):
        mt.matmul(a, b)
    result = mt.matmul(a, b.transpose(-1, -2))
    assert result.shape == [3, 5]

    a, b = _t(4, 3), _t(4, 5)
    with pytest.raises(Exception):
        mt.matmul(a, b)
    result = mt.matmul(a.transpose(-1, -2), b)
    assert result.shape == [3, 5]


VALID_PRODUCTS = [
    ((3, 4), (4, 6), (3, 6)),
    ((2, 3, 4), (4, 5), (2, 3, 5)),
    ((1, 3, 4), (7, 4, 5), (7, 3, 5)),
    ((3, 4), (4,), (3,)),
    ((4,), (4, 5), (5,)),
    ((3,), (3,), ()),
    ((2, 1, 3, 4), (5, 4, 6), (2, 5, 3, 6)),
]


@pytest.mark.parametrize("lhs_shape,rhs_shape,expected", VALID_PRODUCTS)
def test_valid_products_still_go_through(lhs_shape, rhs_shape, expected):
    """The check runs before the promotion and folding paths, so it has to
    accept everything those paths handle."""
    lhs, rhs = _t(*lhs_shape), _t(*rhs_shape)
    result = mt.matmul(lhs, rhs)
    assert tuple(result.shape) == expected
    np.testing.assert_allclose(
        result.numpy(), lhs.numpy() @ rhs.numpy(), rtol=1e-4, atol=1e-5
    )


@pytest.mark.parametrize("lhs_shape,rhs_shape", BAD_PRODUCTS)
def test_numpy_rejects_the_same_products(lhs_shape, rhs_shape):
    """Keeps the validation from drifting into rejecting legal products."""
    with pytest.raises(ValueError):
        np.zeros(lhs_shape, dtype=np.float32) @ np.zeros(rhs_shape, dtype=np.float32)
