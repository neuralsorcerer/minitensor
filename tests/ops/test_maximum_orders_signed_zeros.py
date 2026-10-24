# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""`maximum` and `minimum` put `+0.0` above `-0.0`, whichever side each is on.

They took the first operand whenever the two compared equal, and the two zeros
compare equal: `maximum(-0.0, 0.0)` was `-0.0` and `maximum(0.0, -0.0)` was
`0.0`, so the operation was not commutative, and it disagreed with `fmax`,
which already ordered them. IEEE 754-2019 defines both operations with `+0`
above `-0`; the reductions keep the first of their tied elements, which is the
one their indices name.
"""

from __future__ import annotations

import numpy as np
import pytest

import minitensor as mt


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("size", [1, 3, 8, 17, 64, 1001])
def test_the_zeros_are_ordered_on_both_sides_at_every_length(dtype, size):
    minus = mt.from_numpy(np.full(size, -0.0, dtype=dtype))
    plus = mt.from_numpy(np.full(size, 0.0, dtype=dtype))
    for left, right in ((minus, plus), (plus, minus)):
        assert not np.signbit(mt.maximum(left, right).numpy()).any()
        assert np.signbit(mt.minimum(left, right).numpy()).all()
        assert not np.signbit(left.maximum(right).numpy()).any()
        assert np.signbit(mt.functional.minimum(left, right).numpy()).all()
    # Two of a kind keep their sign.
    assert np.signbit(mt.maximum(minus, minus).numpy()).all()
    assert not np.signbit(mt.minimum(plus, plus).numpy()).any()


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_broadcast_operands_follow_the_same_rule(dtype):
    column = mt.Tensor(np.array([[-0.0], [0.0]], dtype=dtype))
    row = mt.Tensor(np.array([[0.0, -0.0, 1.0, -1.0]], dtype=dtype))
    for got in (mt.maximum(column, row), mt.maximum(row, column)):
        np.testing.assert_array_equal(
            np.signbit(got.numpy()), [[False, True, False, True], [False] * 4]
        )
    for got in (mt.minimum(column, row), mt.minimum(row, column)):
        np.testing.assert_array_equal(
            np.signbit(got.numpy()),
            [[True, True, True, True], [False, True, False, True]],
        )


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_it_is_fmax_wherever_there_is_no_nan(dtype):
    rng = np.random.default_rng(191)
    values = np.concatenate(
        [rng.normal(0, 1, 500), [0.0, -0.0] * 50, [np.inf, -np.inf]]
    )
    left = rng.permutation(values).astype(dtype)
    right = rng.permutation(values).astype(dtype)
    a, b = mt.from_numpy(left), mt.from_numpy(right)
    for got, want in (
        (mt.maximum(a, b), mt.fmax(a, b)),
        (mt.minimum(a, b), mt.fmin(a, b)),
    ):
        np.testing.assert_array_equal(
            got.numpy().view(np.uint8), want.numpy().view(np.uint8)
        )
    nan = np.array([np.nan, 1.0, np.nan], dtype=dtype)
    other = np.array([1.0, np.nan, np.nan], dtype=dtype)
    assert np.isnan(mt.maximum(mt.from_numpy(nan), mt.from_numpy(other)).numpy()).all()


def test_a_tie_sends_the_gradient_to_the_first_operand():
    first = mt.Tensor([-0.0, 0.0, 2.0], requires_grad=True)
    second = mt.Tensor([0.0, -0.0, 2.0], requires_grad=True)
    mt.maximum(first, second).sum().backward()
    assert first.grad.tolist() == [1.0, 1.0, 1.0]
    assert second.grad.tolist() == [0.0, 0.0, 0.0]


def test_a_reduction_returns_the_tied_element_its_index_names():
    values = mt.Tensor([[-0.0, 0.0], [0.0, -0.0]], dtype="float64")
    largest, where = mt.max(values, dim=1)
    assert where.tolist() == [0, 0]
    np.testing.assert_array_equal(np.signbit(largest.numpy()), [True, False])
