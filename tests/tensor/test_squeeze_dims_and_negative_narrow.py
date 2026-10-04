# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""`squeeze` over several axes, and `narrow` from a negative start.

`squeeze((1, 3))` raised `TypeError: 'tuple' object cannot be interpreted as an
integer`, though every reduction here takes a sequence of axes. And `narrow`
took its start as unsigned, so `narrow(dim, -2, 2)` surfaced as the
conversion's `OverflowError` rather than counting back from the end the way an
index does.
"""

import numpy as np
import pytest

import minitensor as mt

X = np.arange(6.0).reshape(2, 1, 3, 1)


@pytest.mark.parametrize(
    "dims,expected",
    [
        ((1, 3), (2, 3)),
        ([1, 1], (2, 3, 1)),
        ((1, 2), (2, 3, 1)),
        ((-1,), (2, 1, 3)),
        ((), (2, 1, 3, 1)),
    ],
    ids=["two-units", "repeated", "unit-and-not", "negative", "none-named"],
)
def test_squeeze_takes_a_sequence_of_axes(dims, expected):
    t = mt.from_numpy(X.copy())
    for result in (t.squeeze(dims), mt.squeeze(t, dims)):
        assert tuple(result.shape) == expected
        np.testing.assert_array_equal(result.numpy().ravel(), X.ravel())


def test_squeezing_several_axes_carries_the_gradient():
    x = mt.from_numpy(np.arange(6.0).reshape(1, 6, 1)).requires_grad_(True)
    weights = mt.arange(6.0).astype("float64")
    (x.squeeze((0, 2)) * weights).sum().backward()
    assert tuple(x.grad.shape) == (1, 6, 1)
    np.testing.assert_array_equal(x.grad.numpy().ravel(), np.arange(6.0))


def test_squeezing_an_axis_out_of_range_names_the_range():
    with pytest.raises(IndexError, match="Dimension out of range"):
        mt.from_numpy(X.copy()).squeeze((1, 7))


@pytest.mark.parametrize(
    "start,length,expected",
    [(-1, 1, [2.0]), (-3, 3, [0.0, 1.0, 2.0]), (-2, 2, [1.0, 2.0]), (3, 0, [])],
)
def test_narrow_counts_a_negative_start_back_from_the_end(start, length, expected):
    t = mt.from_numpy(np.arange(3.0))
    assert t.narrow(0, start, length).tolist() == expected
    assert mt.narrow(t, 0, start, length).tolist() == expected


@pytest.mark.parametrize("start", [-4, 4])
def test_a_start_outside_the_axis_is_an_index_error(start):
    with pytest.raises(IndexError, match="out of bounds for dimension 0 with size 3"):
        mt.from_numpy(np.arange(3.0)).narrow(0, start, 0)
