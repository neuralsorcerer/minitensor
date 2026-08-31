# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""Flat positions and the coordinates they stand for.

`take` reads a tensor in row-major order whatever its shape, and `nonzero`,
`tril_indices` and `argmax` all hand back positions in one of those two
currencies. Converting between them was left to the caller, who has to write
the stride arithmetic themselves and get the row-major order right.

`unravel_index` and `ravel_multi_index` are that conversion, checked against
NumPy's functions of the same names and against each other -- a round trip is
the identity, which is the property that actually matters and the one a wrong
stride breaks. `put` is the write direction of `take`, and `diag_indices`
completes the set of index builders that `tril_indices` and `triu_indices`
started.

Both converters check their inputs, which costs a pass over the index tensor.
That is deliberate: an out-of-range flat position does not fail on its own, it
quietly names a different element, and finding that later is much more
expensive than the check.
"""

from __future__ import annotations

import numpy as np
import pytest

import minitensor as mt

SHAPES = [(6,), (2, 3), (2, 3, 4), (5, 1, 2), (1, 1, 1, 7)]


def _i(values):
    return mt.Tensor.from_numpy(np.asarray(values, dtype=np.int64))


def _f(values):
    return mt.Tensor(
        np.ascontiguousarray(np.asarray(values, dtype=np.float64)), dtype="float64"
    )


# --- unravel_index / ravel_multi_index --------------------------------------


@pytest.mark.parametrize("shape", SHAPES)
def test_unravel_index_matches_numpy(shape):
    positions = np.arange(int(np.prod(shape)))
    got = mt.unravel_index(_i(positions), shape)
    expected = np.unravel_index(positions, shape)

    assert len(got) == len(shape)
    for axis, (mine, theirs) in enumerate(zip(got, expected)):
        np.testing.assert_array_equal(mine.numpy(), theirs, err_msg=f"axis {axis}")


@pytest.mark.parametrize("shape", SHAPES)
def test_ravel_multi_index_matches_numpy(shape):
    positions = np.arange(int(np.prod(shape)))
    coordinates = np.unravel_index(positions, shape)
    np.testing.assert_array_equal(
        mt.ravel_multi_index([_i(part) for part in coordinates], shape).numpy(),
        np.ravel_multi_index(coordinates, shape),
    )


@pytest.mark.parametrize("shape", SHAPES)
def test_the_two_are_inverses(shape):
    positions = _i(np.arange(int(np.prod(shape))))
    np.testing.assert_array_equal(
        mt.ravel_multi_index(list(mt.unravel_index(positions, shape)), shape).numpy(),
        positions.numpy(),
    )


def test_unravel_index_keeps_the_shape_of_its_positions():
    coordinates = mt.unravel_index(_i([[0, 5], [11, 23]]), (2, 3, 4))
    assert all(tuple(part.shape) == (2, 2) for part in coordinates)


def test_ravel_multi_index_takes_the_layout_the_index_builders_produce():
    """`tril_indices` hands back one row per axis, which is what this reads."""

    pair = mt.tril_indices(4, 4)
    np.testing.assert_array_equal(
        mt.ravel_multi_index(pair, (4, 4)).numpy(),
        np.ravel_multi_index(tuple(pair.numpy()), (4, 4)),
    )


def test_the_positions_a_flat_read_would_use_are_the_ones_it_names():
    """The claim tying the two currencies together, checked on real data."""

    values = np.arange(24.0).reshape(2, 3, 4)
    positions = _i([0, 7, 13, 23])
    coordinates = mt.unravel_index(positions, values.shape)
    np.testing.assert_array_equal(
        mt.take(_f(values), positions).numpy(),
        values[tuple(part.numpy() for part in coordinates)],
    )


def test_a_single_axis_shape_is_the_identity():
    positions = _i([0, 3, 5])
    (only,) = mt.unravel_index(positions, 6)
    np.testing.assert_array_equal(only.numpy(), positions.numpy())


def test_an_empty_index_stays_empty():
    coordinates = mt.unravel_index(_i(np.zeros(0, dtype=np.int64)), (2, 3))
    assert all(tuple(part.shape) == (0,) for part in coordinates)
    assert tuple(mt.ravel_multi_index(list(coordinates), (2, 3)).shape) == (0,)


# --- put --------------------------------------------------------------------


def test_put_is_the_write_direction_of_take():
    values = np.arange(6.0).reshape(2, 3)
    written = mt.put(_f(values), _i([0, 4]), _f([-1.0, -2.0])).numpy()
    expected = values.copy()
    expected.reshape(-1)[[0, 4]] = [-1.0, -2.0]
    np.testing.assert_array_equal(written, expected)


def test_put_counts_negative_positions_from_the_end():
    values = np.arange(6.0).reshape(2, 3)
    np.testing.assert_array_equal(
        mt.put(_f(values), _i([-1]), _f([9.0])).numpy(),
        mt.put(_f(values), _i([5]), _f([9.0])).numpy(),
    )


def test_accumulate_decides_what_a_repeated_position_means():
    values = np.zeros((2, 3))
    repeated = _i([1, 1, 1])
    source = _f([1.0, 2.0, 4.0])
    assert float(mt.put(_f(values), repeated, source, True).numpy().sum()) == 7.0
    # Without it the last write stands, so the total is one of the three.
    assert float(mt.put(_f(values), repeated, source).numpy().sum()) in (1.0, 2.0, 4.0)


def test_put_broadcasts_a_scalar_source():
    values = np.arange(6.0).reshape(2, 3)
    expected = values.copy()
    expected.reshape(-1)[[0, 2, 4]] = 7.0
    np.testing.assert_array_equal(
        mt.put(_f(values), _i([0, 2, 4]), _f(7.0)).numpy(), expected
    )


def test_put_leaves_its_input_alone():
    values = _f(np.arange(6.0))
    mt.put(values, _i([0]), _f([99.0]))
    assert float(values.numpy()[0]) == 0.0


def test_the_gradient_of_put_splits_between_its_operands():
    values = mt.Tensor(np.arange(6.0), dtype="float64", requires_grad=True)
    source = mt.Tensor(np.zeros(2), dtype="float64", requires_grad=True)
    weights = _f(np.arange(1.0, 7.0))
    (mt.put(values, _i([1, 4]), source) * weights).sum().backward()

    np.testing.assert_array_equal(values.grad.numpy(), [1.0, 0.0, 3.0, 4.0, 0.0, 6.0])
    np.testing.assert_array_equal(source.grad.numpy(), [2.0, 5.0])
    mt.clear_autograd_graph()


def test_accumulating_leaves_the_input_a_full_gradient():
    """An addition keeps what was already there, so nothing is displaced."""

    values = mt.Tensor(np.arange(6.0), dtype="float64", requires_grad=True)
    weights = _f(np.arange(1.0, 7.0))
    (mt.put(values, _i([1, 4]), _f([0.0, 0.0]), True) * weights).sum().backward()
    np.testing.assert_array_equal(values.grad.numpy(), weights.numpy())
    mt.clear_autograd_graph()


# --- diag_indices -----------------------------------------------------------


@pytest.mark.parametrize("size,rank", [(4, 2), (3, 3), (1, 2), (0, 2), (5, 1)])
def test_diag_indices_matches_numpy(size, rank):
    np.testing.assert_array_equal(
        mt.diag_indices(size, rank).numpy(), np.array(np.diag_indices(size, rank))
    )


def test_diag_indices_selects_the_main_diagonal():
    matrix = np.arange(16.0).reshape(4, 4)
    rows, columns = mt.diag_indices(4).numpy()
    np.testing.assert_array_equal(matrix[rows, columns], np.diag(matrix))


def test_the_index_builders_all_shape_their_answers_the_same_way():
    """`diag_indices` joins `tril_indices` and `triu_indices`, not `nonzero`."""

    for built in (mt.diag_indices(4), mt.tril_indices(4, 4), mt.triu_indices(4, 4)):
        assert tuple(built.shape)[0] == 2
        assert "int64" in str(built.dtype)


# --- what they refuse -------------------------------------------------------


def test_a_position_past_the_end_is_refused():
    with pytest.raises(IndexError, match="which holds 6"):
        mt.unravel_index(_i([0, 6]), (2, 3))


def test_a_negative_position_is_refused():
    with pytest.raises(IndexError, match=r"\[-1, 2\]"):
        mt.unravel_index(_i([-1, 2]), (2, 3))


def test_a_coordinate_past_the_end_of_its_axis_is_refused():
    with pytest.raises(IndexError, match="axis 1 of size 3"):
        mt.ravel_multi_index([_i([0]), _i([3])], (2, 3))


def test_the_wrong_number_of_coordinates_is_refused():
    with pytest.raises(ValueError, match="1 coordinate"):
        mt.ravel_multi_index([_i([0])], (2, 3))


def test_a_float_index_is_refused():
    with pytest.raises(TypeError, match="integer indices"):
        mt.unravel_index(_f([0.0]), (2, 3))


def test_a_shape_that_is_not_integers_is_refused():
    with pytest.raises(TypeError, match="dimensions must be integers"):
        mt.unravel_index(_i([0]), "nope")


def test_a_shape_argument_is_normalized_the_way_every_other_one_is():
    """One rejection message for `shape`, not one per function that takes it."""

    for reject in ("nope", 1.5, [1, "two"]):
        with pytest.raises(TypeError) as mine:
            mt.unravel_index(_i([0]), reject)
        with pytest.raises(TypeError) as theirs:
            mt.broadcast_shapes(reject)
        assert str(mine.value).split(" ", 1)[1] == str(theirs.value).split(" ", 1)[1]


def test_a_shape_with_no_axes_has_no_coordinates_to_convert():
    """The one thing a shape argument may be elsewhere and not here."""

    assert mt.broadcast_shapes(()) == ()
    with pytest.raises(ValueError, match="at least one axis"):
        mt.unravel_index(_i([0]), ())


def test_diag_indices_refuses_a_negative_size():
    with pytest.raises(ValueError, match="non-negative"):
        mt.diag_indices(-1)


def test_diag_indices_refuses_a_rank_below_one():
    with pytest.raises(ValueError, match="at least one dimension"):
        mt.diag_indices(3, 0)


@pytest.mark.parametrize("shape", SHAPES)
def test_coordinates_are_never_the_index_tensor_they_came_from(shape):
    """Both converters skip the arithmetic that cannot change an answer -- the
    divide by a stride of one, the modulus the bounds check already settled.
    On a one-axis shape every skip applies at once, and the result would be
    the caller's own tensor unless something copies it. NumPy's converters
    hand back storage of their own, and a coordinate written to afterwards
    must not reach back into the positions it was derived from."""

    positions = _i(np.arange(int(np.prod(shape))))
    coordinates = mt.unravel_index(positions, shape)
    assert all(axis is not positions for axis in coordinates)
    assert mt.ravel_multi_index(coordinates, shape) is not coordinates[0]

    before = positions.numpy().copy()
    coordinates[0].fill_(99)
    np.testing.assert_array_equal(positions.numpy(), before)
