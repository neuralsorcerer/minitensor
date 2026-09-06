# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""Rearranging a tensor: joining, removing, inserting, repeating, assembling.

These are the NumPy manipulations this library had no spelling for. Each is a
few existing calls -- a narrow and a concatenate, a select of what to keep --
but the arrangement is where the mistakes are, so each is written once here and
checked against NumPy rather than left to every caller.

The positions are the interesting part. `insert` places values *before* the
positions named, and those positions refer to the original tensor, so
`insert(x, [1, 1], [a, b])` puts both before the element that was at 1. `-1`
wraps against the axis in both `insert` and `delete`, but only `insert` may
name the position one past the end -- inserting before the end is a real place
to insert, and deleting the element after the last one is not a real place to
delete.

`resize` is NumPy's and not PyTorch's: it returns a new tensor and *repeats*
the elements to fill a larger shape rather than zero-filling.
"""

from __future__ import annotations

import numpy as np
import pytest

import minitensor as mt


@pytest.fixture()
def values():
    return np.random.default_rng(107).standard_normal((4, 5))


def test_the_array_api_spellings_of_existing_shapes(values):
    tensor = mt.from_numpy(values)
    np.testing.assert_allclose(
        mt.expand_dims(tensor, 0).numpy(), np.expand_dims(values, 0)
    )
    np.testing.assert_allclose(
        mt.expand_dims(tensor, (0, 2)).numpy(), np.expand_dims(values, (0, 2))
    )
    np.testing.assert_allclose(
        mt.permute_dims(tensor, (1, 0)).numpy(), np.permute_dims(values, (1, 0))
    )
    np.testing.assert_allclose(
        mt.matrix_transpose(tensor).numpy(), np.matrix_transpose(values)
    )
    np.testing.assert_allclose(mt.identity(4).numpy(), np.identity(4))
    assert len(mt.unstack(tensor, 0)) == 4
    assert len(mt.array_split(tensor, 3, 0)) == 3
    assert len(mt.broadcast_arrays(tensor, mt.from_numpy(np.ones(5)))) == 2

    with pytest.raises(ValueError, match="at least two dimensions"):
        mt.matrix_transpose(mt.from_numpy(np.arange(3.0)))
    with pytest.raises(ValueError, match="same axis twice"):
        mt.expand_dims(tensor, (0, 0))


def test_append_joins_along_an_axis_or_flattens(values):
    tensor = mt.from_numpy(values)
    np.testing.assert_allclose(
        mt.append(tensor, mt.from_numpy(np.array([1.0]))).numpy(),
        np.append(values, [1.0]),
    )
    row = np.random.default_rng(109).standard_normal((1, 5))
    np.testing.assert_allclose(
        mt.append(tensor, mt.from_numpy(row), 0).numpy(), np.append(values, row, 0)
    )


@pytest.mark.parametrize(
    "obj,dim",
    [(1, 0), ([0, 2], 1), (slice(0, 3, 2), 1), ([0, 5, 19], None), (-1, 0)],
    ids=["one", "several", "slice", "flat", "negative"],
)
def test_delete_removes_what_it_names(values, obj, dim):
    got = mt.delete(mt.from_numpy(values), obj, dim).numpy()
    np.testing.assert_allclose(got, np.delete(values, obj, dim))


def test_delete_takes_a_mask_and_refuses_a_position_that_is_not_there(values):
    mask = np.array([True, False, True, False])
    np.testing.assert_allclose(
        mt.delete(mt.from_numpy(values), mt.from_numpy(mask), 0).numpy(),
        np.delete(values, mask, 0),
    )
    with pytest.raises(IndexError, match="out of bounds"):
        mt.delete(mt.from_numpy(values), 4, 0)


@pytest.mark.parametrize(
    "obj,fill,dim",
    [(1, 9.0, 0), ([1, 3], 9.0, 1), (2, 7.0, None), (-1, 9.0, 0), (slice(0, 3), 9.0, 0)],
    ids=["one", "several", "flat", "negative", "slice"],
)
def test_insert_places_values_before_the_positions_named(values, obj, fill, dim):
    got = mt.insert(mt.from_numpy(values), obj, fill, dim).numpy()
    np.testing.assert_allclose(got, np.insert(values, obj, fill, dim))


def test_insert_at_the_end_and_at_a_repeated_position():
    line = np.arange(5.0)
    tensor = mt.from_numpy(line)
    # One past the last element is a place to insert, and only for `insert`.
    np.testing.assert_allclose(
        mt.insert(tensor, 5, 9.0).numpy(), np.insert(line, 5, 9.0)
    )
    np.testing.assert_allclose(
        mt.insert(tensor, [5], 9.0).numpy(), np.insert(line, [5], 9.0)
    )
    # Two values at the same position keep the order they were given in.
    np.testing.assert_allclose(
        mt.insert(tensor, [1, 1], mt.from_numpy(np.array([10.0, 20.0]))).numpy(),
        np.insert(line, [1, 1], [10.0, 20.0]),
    )
    with pytest.raises(IndexError, match="out of bounds"):
        mt.insert(tensor, 6, 1.0)


def test_insert_spreads_a_row_across_an_axis(values):
    row = np.arange(5.0)
    np.testing.assert_allclose(
        mt.insert(mt.from_numpy(values), 2, mt.from_numpy(row), 0).numpy(),
        np.insert(values, 2, row, 0),
    )


def test_resize_repeats_rather_than_zero_filling(values):
    np.testing.assert_allclose(
        mt.resize(mt.from_numpy(values), (5, 6)).numpy(), np.resize(values, (5, 6))
    )
    np.testing.assert_allclose(
        mt.resize(mt.from_numpy(values), (2, 3)).numpy(), np.resize(values, (2, 3))
    )
    # Nothing to repeat, so it fills with zeros -- as NumPy does.
    empty = np.array([], dtype=np.float64)
    np.testing.assert_allclose(
        mt.resize(mt.from_numpy(empty), (2, 2)).numpy(), np.resize(empty, (2, 2))
    )


def test_block_assembles_the_way_it_is_written():
    corners = [
        [np.eye(2), np.zeros((2, 3))],
        [np.ones((1, 2)), np.full((1, 3), 7.0)],
    ]
    np.testing.assert_allclose(
        mt.block([[mt.from_numpy(part) for part in row] for row in corners]).numpy(),
        np.block(corners),
    )
    # A flat list joins along the last axis.
    np.testing.assert_allclose(
        mt.block([mt.from_numpy(np.array([1.0, 2.0])), mt.from_numpy(np.array([3.0]))]).numpy(),
        np.block([np.array([1.0, 2.0]), np.array([3.0])]),
    )
    with pytest.raises(ValueError, match="empty list"):
        mt.block([])
    with pytest.raises(ValueError, match="nest equally"):
        mt.block([[mt.from_numpy(np.ones((1, 1)))], mt.from_numpy(np.ones((1, 1)))])


def test_cumulative_sum_can_start_from_the_empty_total(values):
    np.testing.assert_allclose(
        mt.cumulative_sum(mt.from_numpy(values), 1).numpy(),
        np.cumulative_sum(values, axis=1),
    )
    got = mt.cumulative_sum(mt.from_numpy(values), 1, include_initial=True)
    np.testing.assert_allclose(
        got.numpy(), np.cumulative_sum(values, axis=1, include_initial=True)
    )
    # One longer than the axis, and each entry is the total of what came before.
    assert got.shape[1] == values.shape[1] + 1
    np.testing.assert_allclose(got.numpy()[:, 0], 0.0)

    with pytest.raises(ValueError, match="needs a dim"):
        mt.cumulative_sum(mt.from_numpy(values))
