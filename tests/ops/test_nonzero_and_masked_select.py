# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""You could select by a mask but never find out where the selection came from.

`tensor[mask]` worked and was differentiable, so pulling the values out was
covered. What had no answer was *which* elements those were: no `nonzero`, no
`argwhere`, nothing. That is the half you need to write a result back, to index
a second tensor by the same positions, or to report which sample in a batch
failed -- and it is the half that cannot be assembled from anything else,
because the length of the answer depends on the data rather than on the shape.

`nonzero` returns `[found, ndim]` of int64 in row-major order, so `result[i]`
is a multi-index that reads straight back into the input; the round-trip is
tested below rather than assumed. `count_nonzero` is written as a sum of the
mask rather than as its own reduction, so the dimension handling and the
int64 widening are `sum`'s and cannot drift from it. `masked_select` is the
name everyone reaches for, on the call `tensor[mask]` already made.

Truthiness is the predicate `any` and `all` use -- nonzero for the numeric
dtypes, the value itself for bool -- because all four come from one macro. NaN
is nonzero, so it counts, which is NumPy's answer too and is worth pinning
because the alternative is defensible and would be a silent difference.
"""

from __future__ import annotations

import numpy as np
import pytest

import minitensor as mt

SHAPES = [(7,), (3, 4), (2, 3, 4), (2, 2, 2, 2), (1, 9)]


def _sparse(shape, seed=0, density=0.4):
    """Mostly zeros, so the answer is a small fraction of the input -- which is
    the shape these are actually used at."""
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(shape) * (rng.random(shape) < density)).astype(
        np.float64
    )


@pytest.mark.parametrize("shape", SHAPES)
def test_nonzero_matches_numpy(shape):
    values = _sparse(shape)
    got = mt.Tensor(values, dtype="float64").nonzero().numpy()
    np.testing.assert_array_equal(got, np.argwhere(values != 0))


@pytest.mark.parametrize("shape", SHAPES)
def test_the_indices_read_back_to_the_nonzero_values(shape):
    """The property that makes the result useful, checked without reference to
    NumPy's ordering: indexing the input by the returned rows must give exactly
    the non-zero elements, in the order they appear."""
    values = _sparse(shape, seed=3)
    indices = mt.Tensor(values, dtype="float64").nonzero().numpy()
    np.testing.assert_array_equal(values[tuple(indices.T)], values[values != 0])


def test_the_rows_come_back_in_row_major_order():
    """Not merely the right set of indices -- the order is part of the contract,
    since callers zip it against other row-major results."""
    values = np.array([[0.0, 2.0, 0.0], [3.0, 0.0, 4.0]])
    got = mt.Tensor(values, dtype="float64").nonzero().numpy()
    np.testing.assert_array_equal(got, [[0, 1], [1, 0], [1, 2]])


def test_nonzero_reports_int64_indices():
    values = _sparse((4, 4), seed=5)
    assert mt.Tensor(values, dtype="float64").nonzero().dtype == "int64"


@pytest.mark.parametrize(
    "dtype,values",
    [
        ("int32", np.array([0, 3, 0, -2], dtype=np.int32)),
        ("int64", np.array([[0, 1], [5, 0]], dtype=np.int64)),
        ("float32", np.array([0.0, -0.0, 1.5], dtype=np.float32)),
        ("bool", np.array([True, False, True])),
    ],
)
def test_every_dtype_uses_the_same_truthiness(dtype, values):
    want = np.argwhere(values if dtype == "bool" else values != 0)
    got = mt.Tensor(values, dtype=dtype).nonzero().numpy()
    np.testing.assert_array_equal(got, want)


def test_nan_counts_as_nonzero():
    """It is not zero, so it counts. The other choice is defensible, which is
    exactly why this is pinned."""
    values = np.array([np.nan, 0.0, 1.0])
    np.testing.assert_array_equal(
        mt.Tensor(values, dtype="float64").nonzero().numpy(), [[0], [2]]
    )
    assert mt.Tensor(values, dtype="float64").count_nonzero().item() == 2


def test_negative_zero_is_zero():
    values = np.array([-0.0, 0.0, 1.0])
    np.testing.assert_array_equal(
        mt.Tensor(values, dtype="float64").nonzero().numpy(), [[2]]
    )


def test_an_all_zero_tensor_gives_no_rows():
    got = mt.Tensor(np.zeros((3, 5)), dtype="float64").nonzero().numpy()
    assert got.shape == (0, 2)


def test_an_empty_tensor_gives_no_rows():
    """No rows, but still one column per dimension -- the width comes from the
    rank, not from how many elements happened to be there."""
    got = mt.Tensor(np.zeros((0, 3)), dtype="float64").nonzero().numpy()
    assert got.shape == (0, 2)
    assert got.shape == np.argwhere(np.zeros((0, 3)) != 0).shape


@pytest.mark.parametrize("value,rows", [(5.0, 1), (0.0, 0)])
def test_a_scalar_has_a_row_but_no_columns(value, rows):
    """A 0-d tensor has no index to report -- only whether there is a row at
    all. The width is its rank, which is zero."""
    got = mt.Tensor(np.array(value), dtype="float64").nonzero().numpy()
    assert got.shape == (rows, 0)


# --- count_nonzero ---------------------------------------------------------


@pytest.mark.parametrize("shape", SHAPES)
def test_count_nonzero_matches_numpy(shape):
    values = _sparse(shape, seed=7)
    got = mt.Tensor(values, dtype="float64").count_nonzero().item()
    assert got == int(np.count_nonzero(values))


@pytest.mark.parametrize("dim", [0, 1, -1])
def test_count_nonzero_along_a_dimension(dim):
    values = _sparse((4, 5), seed=11)
    got = mt.Tensor(values, dtype="float64").count_nonzero(dim).numpy()
    np.testing.assert_array_equal(got, np.count_nonzero(values, axis=dim))


def test_count_nonzero_keepdim():
    values = _sparse((4, 5), seed=13)
    got = mt.Tensor(values, dtype="float64").count_nonzero(1, True)
    assert tuple(got.shape) == (4, 1)
    np.testing.assert_array_equal(got.numpy().ravel(), np.count_nonzero(values, axis=1))


def test_count_nonzero_reports_int64():
    """A count leaves the range of what it is counting -- a bool tensor has no
    room for its own count at all."""
    mask = np.array([[True, False], [True, True]])
    got = mt.Tensor(mask, dtype="bool").count_nonzero()
    assert got.dtype == "int64"
    assert got.item() == 3


def test_the_count_agrees_with_the_number_of_indices():
    """Two spellings of one question: how many, and which ones."""
    values = _sparse((6, 7), seed=17)
    t = mt.Tensor(values, dtype="float64")
    assert t.count_nonzero().item() == t.nonzero().numpy().shape[0]


# --- masked_select ---------------------------------------------------------


@pytest.mark.parametrize("shape", SHAPES)
def test_masked_select_matches_numpy(shape):
    values = _sparse(shape, seed=19)
    mask = values > 0
    got = (
        mt.Tensor(values, dtype="float64")
        .masked_select(mt.Tensor(mask, dtype="bool"))
        .numpy()
    )
    np.testing.assert_array_equal(got, values[mask])


def test_masked_select_is_the_same_call_as_indexing():
    """It is a name, not a second implementation -- so it must not be able to
    disagree with the indexing form."""
    values = _sparse((5, 6), seed=23)
    t = mt.Tensor(values, dtype="float64")
    mask = mt.Tensor(values > 0, dtype="bool")
    np.testing.assert_array_equal(t.masked_select(mask).numpy(), t[mask].numpy())


def test_masked_select_carries_a_gradient_back_to_the_selected_positions():
    values = np.array([[1.0, 2.0], [3.0, 4.0]])
    t = mt.Tensor(values, dtype="float64", requires_grad=True)
    mask = mt.Tensor(np.array([[True, False], [False, True]]), dtype="bool")
    t.masked_select(mask).sum().backward()
    np.testing.assert_array_equal(t.grad.numpy(), [[1.0, 0.0], [0.0, 1.0]])


def test_masked_select_with_nothing_selected():
    values = _sparse((3, 3), seed=29)
    mask = mt.Tensor(np.zeros((3, 3), dtype=bool), dtype="bool")
    assert mt.Tensor(values, dtype="float64").masked_select(mask).numpy().size == 0


def test_a_mask_of_the_wrong_shape_is_refused():
    values = mt.Tensor(np.zeros((3, 4)), dtype="float64")
    mask = mt.Tensor(np.zeros((3, 5), dtype=bool), dtype="bool")
    with pytest.raises(Exception):
        values.masked_select(mask)


def test_a_non_bool_mask_is_refused():
    values = mt.Tensor(np.zeros((3, 4)), dtype="float64")
    with pytest.raises(Exception):
        values.masked_select(mt.Tensor(np.zeros((3, 4)), dtype="float64"))


# --- the three together ----------------------------------------------------


def test_the_module_level_functions_agree_with_the_methods():
    values = _sparse((4, 5), seed=31)
    t = mt.Tensor(values, dtype="float64")
    mask = mt.Tensor(values > 0, dtype="bool")
    np.testing.assert_array_equal(mt.nonzero(t).numpy(), t.nonzero().numpy())
    np.testing.assert_array_equal(
        mt.count_nonzero(t).numpy(), t.count_nonzero().numpy()
    )
    np.testing.assert_array_equal(
        mt.masked_select(t, mask).numpy(), t.masked_select(mask).numpy()
    )


def test_selecting_by_the_indices_and_by_the_mask_agree():
    """`nonzero` and `masked_select` answer the same question two ways, and the
    point of having both is that you can move between them."""
    values = _sparse((5, 6), seed=37)
    t = mt.Tensor(values, dtype="float64")
    by_mask = t.masked_select(mt.Tensor(values != 0, dtype="bool")).numpy()
    by_index = values[tuple(t.nonzero().numpy().T)]
    np.testing.assert_array_equal(by_mask, by_index)
