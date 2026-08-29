# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""The joining, splitting and reorienting helpers.

`hstack`, `vstack`, `dstack`, `column_stack`, `tile`, `unbind`, `tensor_split`,
`fliplr`, `flipud` and `rot90` -- all of them NumPy spellings, so NumPy is the
reference for every value, including the rank-promotion rules that are the only
reason several of them exist alongside `cat`.
"""

import numpy as np
import pytest

import minitensor as mt

MATRIX = np.arange(6.0).reshape(2, 3)
OTHER = np.arange(6.0, 12.0).reshape(2, 3)
VECTOR = np.arange(3.0)
SCALAR = np.float64(7.0)


def _t(array):
    # `ascontiguousarray` promotes a 0-D input to shape (1,), which is exactly
    # the rank these helpers are being tested about; `asarray` leaves it alone.
    values = np.asarray(array, dtype=np.float64)
    return mt.Tensor(
        values if values.ndim == 0 else np.ascontiguousarray(values), dtype="float64"
    )


STACKERS = ["hstack", "vstack", "dstack", "column_stack"]


@pytest.mark.parametrize("name", STACKERS)
@pytest.mark.parametrize(
    "arrays",
    [
        (MATRIX, OTHER),
        (VECTOR, VECTOR),
        (MATRIX, MATRIX, MATRIX),
    ],
    ids=["matrices", "vectors", "three"],
)
def test_stackers_match_numpy(name, arrays):
    got = getattr(mt, name)([_t(a) for a in arrays]).numpy()
    want = getattr(np, name)(list(arrays))
    assert got.shape == want.shape
    np.testing.assert_array_equal(got, want)


@pytest.mark.parametrize("name", STACKERS)
def test_stackers_promote_rank_the_way_numpy_does(name):
    # The promotion is the whole point: `vstack` of two vectors is a matrix,
    # `hstack` of them is a longer vector, and `cat` alone cannot tell those
    # apart because it never changes rank.
    got = getattr(mt, name)([_t(VECTOR), _t(VECTOR)]).numpy()
    assert got.shape == getattr(np, name)([VECTOR, VECTOR]).shape


@pytest.mark.parametrize("name", STACKERS)
def test_stackers_reject_a_bare_tensor_and_an_empty_sequence(name):
    op = getattr(mt, name)
    with pytest.raises(TypeError, match="sequence of tensors"):
        op(_t(MATRIX))
    with pytest.raises(ValueError, match="at least one tensor"):
        op([])


@pytest.mark.parametrize("reps", [2, (2,), (2, 3), (1, 2, 2), (3, 1)], ids=str)
def test_tile_matches_numpy_including_short_reps(reps):
    # Short `reps` is what separates `tile` from `repeat`: NumPy pads the
    # missing leading entries with 1, and `repeat` insists on one per axis.
    np.testing.assert_array_equal(
        mt.tile(_t(MATRIX), reps).numpy(), np.tile(MATRIX, reps)
    )


def test_tile_of_a_vector_and_a_scalar():
    np.testing.assert_array_equal(mt.tile(_t(VECTOR), 2).numpy(), np.tile(VECTOR, 2))
    np.testing.assert_array_equal(
        mt.tile(_t(SCALAR), (2, 2)).numpy(), np.tile(SCALAR, (2, 2))
    )


@pytest.mark.parametrize("dim", [0, 1, -1, -2])
def test_unbind_drops_the_dimension_it_walks(dim):
    pieces = mt.unbind(_t(MATRIX), dim)
    axis = dim if dim >= 0 else dim + 2

    assert len(pieces) == MATRIX.shape[axis]
    for index, piece in enumerate(pieces):
        assert piece.ndim() == MATRIX.ndim - 1
        np.testing.assert_array_equal(piece.numpy(), np.take(MATRIX, index, axis=axis))

    # It is the inverse of `stack`, which is what makes it not `split`.
    np.testing.assert_array_equal(mt.stack(list(pieces), axis).numpy(), MATRIX)


def test_unbind_rejects_a_scalar():
    with pytest.raises(ValueError, match="at least one dimension"):
        mt.unbind(_t(SCALAR))


@pytest.mark.parametrize("sections", [1, 2, 3, 4, 7, 10, 11])
def test_tensor_split_by_count_matches_array_split(sections):
    values = np.arange(10.0)
    got = mt.tensor_split(_t(values), sections)
    want = np.array_split(values, sections)

    assert len(got) == len(want)
    for piece, expected in zip(got, want):
        np.testing.assert_array_equal(piece.numpy(), expected)


def test_tensor_split_takes_a_count_where_split_takes_a_size():
    # The two read alike and mean different things. `split(10, 3)` cuts pieces
    # of three and leaves a short tail; `tensor_split(10, 3)` cuts three pieces
    # and balances them.
    values = np.arange(10.0)
    assert [piece.shape[0] for piece in mt.tensor_split(_t(values), 3)] == [4, 3, 3]
    assert [piece.shape[0] for piece in mt.split(_t(values), 3, 0)] == [3, 3, 3, 1]


@pytest.mark.parametrize(
    "indices", [[2, 4], [0, 6], [3], [], [1, 1, 5], [-100, 100]], ids=str
)
def test_tensor_split_by_indices_matches_numpy(indices):
    values = np.arange(6.0)
    got = mt.tensor_split(_t(values), indices)
    want = np.array_split(values, [min(max(i, 0), 6) for i in indices])

    assert len(got) == len(want)
    for piece, expected in zip(got, want):
        np.testing.assert_array_equal(piece.numpy(), expected)


def test_tensor_split_along_another_dim_and_its_errors():
    got = mt.tensor_split(_t(MATRIX), 2, dim=1)
    want = np.array_split(MATRIX, 2, axis=1)
    for piece, expected in zip(got, want):
        np.testing.assert_array_equal(piece.numpy(), expected)

    with pytest.raises(ValueError, match="positive number of sections"):
        mt.tensor_split(_t(MATRIX), 0)
    with pytest.raises(ValueError, match="out of range"):
        mt.tensor_split(_t(MATRIX), 2, dim=5)


def test_fliplr_and_flipud_match_numpy():
    np.testing.assert_array_equal(mt.fliplr(_t(MATRIX)).numpy(), np.fliplr(MATRIX))
    np.testing.assert_array_equal(mt.flipud(_t(MATRIX)).numpy(), np.flipud(MATRIX))
    np.testing.assert_array_equal(mt.flipud(_t(VECTOR)).numpy(), np.flipud(VECTOR))

    with pytest.raises(ValueError, match="at least two dimensions"):
        mt.fliplr(_t(VECTOR))


@pytest.mark.parametrize("k", [-5, -2, -1, 0, 1, 2, 3, 4, 9])
def test_rot90_matches_numpy_for_every_quarter_turn(k):
    np.testing.assert_array_equal(mt.rot90(_t(MATRIX), k).numpy(), np.rot90(MATRIX, k))


def test_rot90_in_a_chosen_plane_and_its_errors():
    volume = np.arange(24.0).reshape(2, 3, 4)
    for dims in [(0, 1), (1, 2), (0, 2), (2, 0), (-1, -2)]:
        np.testing.assert_array_equal(
            mt.rot90(_t(volume), 1, dims).numpy(), np.rot90(volume, 1, dims)
        )

    with pytest.raises(ValueError, match="two different axes"):
        mt.rot90(_t(MATRIX), 1, (0, 0))
    with pytest.raises(ValueError, match="at least two dimensions"):
        mt.rot90(_t(VECTOR))
    with pytest.raises(TypeError, match="pair of integers"):
        mt.rot90(_t(MATRIX), 1, 0)


def test_four_quarter_turns_are_the_identity():
    once = mt.rot90(_t(MATRIX))
    twice = mt.rot90(once)
    thrice = mt.rot90(twice)
    np.testing.assert_array_equal(mt.rot90(thrice).numpy(), MATRIX)


def test_gradients_flow_through_the_helpers():
    values = MATRIX.copy()
    for build in (
        lambda t: mt.hstack([t, t]),
        lambda t: mt.vstack([t, t]),
        lambda t: mt.tile(t, 2),
        lambda t: mt.unbind(t)[1],
        lambda t: mt.tensor_split(t, 2)[0],
        lambda t: mt.fliplr(t),
        lambda t: mt.rot90(t),
    ):
        tensor = mt.Tensor(values.copy(), dtype="float64", requires_grad=True)
        build(tensor).sum().backward()
        assert tensor.grad is not None
        assert np.all(np.isfinite(tensor.grad.numpy()))
        mt.clear_autograd_graph()

    # The gradient counts how many times each element was used: `hstack` of a
    # tensor with itself uses every element twice.
    tensor = mt.Tensor(values.copy(), dtype="float64", requires_grad=True)
    mt.hstack([tensor, tensor]).sum().backward()
    np.testing.assert_array_equal(tensor.grad.numpy(), np.full_like(values, 2.0))
