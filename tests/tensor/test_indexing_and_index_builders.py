# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""The indexing helpers that are arrangements of the indexing kernels.

`take`, `take_along_dim`, `index_add`, `index_copy`, `index_fill`,
`masked_scatter`, `select`, `flatnonzero`, `argwhere`, `isin`,
`tril_indices`, `triu_indices`, `diagflat`, `block_diag` and `cartesian_prod`
add no kernel: each is `index_select`, `gather`, `scatter`, `nonzero`,
`searchsorted` or `pad` pointed at a rearranged view. So what these tests
establish is that the rearrangement is the right one -- checked against NumPy
wherever NumPy has the same function -- and that the gradients each inherits
land where the arrangement says they should.
"""

from __future__ import annotations

import numpy as np
import pytest

import minitensor as mt

BASE = np.arange(12.0).reshape(3, 4)


def _t(values, dtype="float64", requires_grad=False):
    array = np.ascontiguousarray(np.asarray(values))
    return mt.Tensor(array, dtype=dtype, requires_grad=requires_grad)


def _i(values):
    return mt.Tensor(np.ascontiguousarray(np.asarray(values, dtype=np.int64)), dtype="int64")


# --- take and take_along_dim ------------------------------------------------


def test_take_matches_numpy():
    for index in ([0, 5, 11], [[0, 1], [10, 11]], [7]):
        np.testing.assert_array_equal(
            mt.take(_t(BASE), _i(index)).numpy(), np.take(BASE, index)
        )


def test_take_counts_negative_positions_from_the_end():
    np.testing.assert_array_equal(
        mt.take(_t(BASE), _i([-1, -12])).numpy(), np.take(BASE, [-1, -12])
    )


def test_take_keeps_the_index_s_shape_not_the_input_s():
    got = mt.take(_t(BASE), _i([[0, 1, 2]]))
    assert tuple(got.shape) == (1, 3)


def test_take_rejects_a_float_index():
    with pytest.raises(TypeError, match="integer indices"):
        mt.take(_t(BASE), _t([0.0, 1.0]))


@pytest.mark.parametrize("dim", [0, 1, -1])
def test_take_along_dim_matches_numpy(dim):
    axis = dim % BASE.ndim
    index = np.argsort(BASE[::-1].copy(), axis=axis)
    np.testing.assert_array_equal(
        mt.take_along_dim(_t(BASE), _i(index), dim).numpy(),
        np.take_along_axis(BASE, index, axis),
    )


def test_take_along_dim_broadcasts_a_single_column_of_indices():
    index = np.array([[3], [0], [2]])
    np.testing.assert_array_equal(
        mt.take_along_dim(_t(BASE), _i(index), 1).numpy(),
        np.take_along_axis(BASE, index, 1),
    )


def test_take_along_dim_without_a_dim_flattens_both():
    index = np.argsort(BASE.reshape(-1))
    np.testing.assert_array_equal(
        mt.take_along_dim(_t(BASE), _i(index)).numpy(),
        np.take_along_axis(BASE.reshape(-1), index, 0),
    )


def test_take_along_dim_needs_a_matching_rank():
    with pytest.raises(ValueError, match="same rank"):
        mt.take_along_dim(_t(BASE), _i([0, 1]), 1)


# --- index_add, index_copy, index_fill --------------------------------------


def test_index_add_accumulates_repeated_indices():
    source = np.ones((3, 2))
    got = mt.index_add(_t(BASE), 1, _i([0, 0]), _t(source)).numpy()

    want = BASE.copy()
    np.add.at(want, (slice(None), np.array([0, 0])), source)
    np.testing.assert_array_equal(got, want)
    assert got[0, 0] == BASE[0, 0] + 2, "both writes have to land"


def test_index_add_scales_by_alpha():
    source = np.full((3, 1), 2.0)
    got = mt.index_add(_t(BASE), 1, _i([2]), _t(source), alpha=-0.5).numpy()
    want = BASE.copy()
    want[:, 2] += -0.5 * 2.0
    np.testing.assert_allclose(got, want)


def test_index_copy_overwrites_rather_than_adding():
    got = mt.index_copy(_t(BASE), 0, _i([1]), _t(np.full((1, 4), 9.0))).numpy()
    want = BASE.copy()
    want[1] = 9.0
    np.testing.assert_array_equal(got, want)


def test_index_fill_sets_whole_slices():
    got = mt.index_fill(_t(BASE), 1, _i([0, 3]), -1.0).numpy()
    want = BASE.copy()
    want[:, [0, 3]] = -1.0
    np.testing.assert_array_equal(got, want)


def test_the_index_helpers_leave_the_input_alone():
    # They return a copy, as their `scatter` does; an in-place read of the
    # original afterwards has to see what it always saw.
    tensor = _t(BASE)
    mt.index_fill(tensor, 0, _i([0]), 99.0)
    mt.index_add(tensor, 0, _i([0]), _t(np.ones((1, 4))))
    np.testing.assert_array_equal(tensor.numpy(), BASE)


@pytest.mark.parametrize("name", ["index_add", "index_copy", "index_fill"])
def test_the_index_helpers_take_a_negative_dimension_and_index(name):
    args = {
        "index_add": (_t(np.ones((3, 1))),),
        "index_copy": (_t(np.full((3, 1), 5.0)),),
        "index_fill": (7.0,),
    }[name]
    positive = getattr(mt, name)(_t(BASE), 1, _i([3]), *args).numpy()
    negative = getattr(mt, name)(_t(BASE), -1, _i([-1]), *args).numpy()
    np.testing.assert_array_equal(positive, negative)


def test_index_add_reaches_the_source_with_a_gradient():
    tensor = _t(BASE, requires_grad=True)
    source = _t(np.ones((3, 2)), requires_grad=True)
    mt.index_add(tensor, 1, _i([0, 2]), source).sum().backward()

    # Every element of the target survives untouched, so each carries a 1.
    np.testing.assert_array_equal(tensor.grad.numpy(), np.ones_like(BASE))
    np.testing.assert_array_equal(source.grad.numpy(), np.ones((3, 2)))


# --- masked_scatter ---------------------------------------------------------


def test_masked_scatter_writes_the_source_in_order():
    mask = np.array([[True, False, True, False]] * 3)
    source = np.arange(6.0) * 10
    got = mt.masked_scatter(_t(BASE), _t(mask, "bool"), _t(source)).numpy()

    want = BASE.copy()
    want[mask] = source[: mask.sum()]
    np.testing.assert_array_equal(got, want)


def test_masked_scatter_broadcasts_the_mask():
    mask = np.array([True, False, False, True])
    source = np.arange(6.0)
    got = mt.masked_scatter(_t(BASE), _t(mask, "bool"), _t(source)).numpy()

    want = BASE.copy()
    want[np.broadcast_to(mask, BASE.shape)] = source
    np.testing.assert_array_equal(got, want)


def test_masked_scatter_needs_enough_source_to_go_round():
    mask = np.ones_like(BASE, dtype=bool)
    with pytest.raises(ValueError, match="at least 12 source elements"):
        mt.masked_scatter(_t(BASE), _t(mask, "bool"), _t(np.zeros(5)))


def test_an_empty_mask_leaves_the_input_unchanged():
    mask = np.zeros_like(BASE, dtype=bool)
    np.testing.assert_array_equal(
        mt.masked_scatter(_t(BASE), _t(mask, "bool"), _t(np.zeros(0))).numpy(), BASE
    )


def test_masked_scatter_differs_from_masked_fill_by_being_positional():
    mask = np.array([[True, True], [False, True]])
    values = np.zeros((2, 2))
    scattered = mt.masked_scatter(
        _t(values), _t(mask, "bool"), _t([1.0, 2.0, 3.0])
    ).numpy()
    filled = mt.masked_fill(_t(values), _t(mask, "bool"), 1.0).numpy()

    np.testing.assert_array_equal(scattered, [[1.0, 2.0], [0.0, 3.0]])
    np.testing.assert_array_equal(filled, [[1.0, 1.0], [0.0, 1.0]])


# --- select -----------------------------------------------------------------


@pytest.mark.parametrize("dim,index", [(0, 1), (1, 3), (-1, -1), (-2, 0)])
def test_select_is_plain_indexing(dim, index):
    np.testing.assert_array_equal(
        mt.select(_t(BASE), dim, index).numpy(),
        np.take(BASE, index, axis=dim),
    )


def test_select_drops_the_axis_that_narrow_keeps():
    assert tuple(mt.select(_t(BASE), 0, 1).shape) == (4,)
    assert tuple(mt.narrow(_t(BASE), 0, 1, 1).shape) == (1, 4)


def test_select_reports_an_index_past_the_end():
    with pytest.raises(IndexError, match="out of range"):
        mt.select(_t(BASE), 0, 3)


# --- flatnonzero and argwhere -----------------------------------------------


def test_flatnonzero_matches_numpy():
    values = np.array([[0.0, 3.0], [0.0, 0.0], [-1.0, 0.0]])
    np.testing.assert_array_equal(
        mt.flatnonzero(_t(values)).numpy(), np.flatnonzero(values)
    )
    assert mt.flatnonzero(_t(values)).ndim() == 1


def test_argwhere_matches_numpy():
    values = np.array([[0.0, 3.0], [0.0, 0.0], [-1.0, 0.0]])
    np.testing.assert_array_equal(
        mt.argwhere(_t(values)).numpy(), np.argwhere(values)
    )


def test_nothing_non_zero_gives_an_empty_answer_of_the_right_rank():
    zeros = np.zeros((2, 3))
    assert tuple(mt.flatnonzero(_t(zeros)).shape) == (0,)
    assert tuple(mt.argwhere(_t(zeros)).shape) == (0, 2)


# --- isin -------------------------------------------------------------------


@pytest.mark.parametrize(
    "tests",
    [[1.0, 4.0, 11.0], [], [-5.0], list(np.arange(12.0)), [3.0, 3.0, 3.0]],
    ids=["some", "none", "absent", "all", "repeated"],
)
def test_isin_matches_numpy(tests):
    np.testing.assert_array_equal(
        mt.isin(_t(BASE), _t(np.asarray(tests, dtype=np.float64))).numpy(),
        np.isin(BASE, tests),
    )


def test_isin_inverts_on_request():
    got = mt.isin(_t(BASE), _t([1.0, 4.0]), invert=True).numpy()
    np.testing.assert_array_equal(got, np.isin(BASE, [1.0, 4.0], invert=True))


def test_isin_promotes_the_two_operands_to_a_common_dtype():
    integers = mt.Tensor(np.array([1, 2, 3], dtype=np.int64), dtype="int64")
    np.testing.assert_array_equal(
        mt.isin(integers, _t([2.0, 5.0])).numpy(), [False, True, False]
    )


def test_a_nan_is_a_member_of_nothing_including_itself():
    # `nan != nan`, so a sorted lookup answers False, and so does NumPy.
    values = np.array([np.nan, 1.0])
    np.testing.assert_array_equal(
        mt.isin(_t(values), _t([np.nan, 1.0])).numpy(), np.isin(values, [np.nan, 1.0])
    )


def test_isin_keeps_the_shape_of_its_first_operand():
    assert tuple(mt.isin(_t(BASE), _t([1.0])).shape) == BASE.shape
    assert "bool" in str(mt.isin(_t(BASE), _t([1.0])).dtype)


# --- the index builders -----------------------------------------------------


@pytest.mark.parametrize("offset", [-2, -1, 0, 1, 3])
@pytest.mark.parametrize("row,col", [(3, 3), (4, 2), (2, 5), (1, 1)])
def test_tril_and_triu_indices_match_numpy(offset, row, col):
    np.testing.assert_array_equal(
        mt.tril_indices(row, col, offset).numpy(),
        np.array(np.tril_indices(row, offset, col)),
    )
    np.testing.assert_array_equal(
        mt.triu_indices(row, col, offset).numpy(),
        np.array(np.triu_indices(row, offset, col)),
    )


def test_the_index_builders_select_exactly_the_triangle_they_name():
    matrix = np.arange(9.0).reshape(3, 3)
    rows, cols = mt.tril_indices(3, 3).numpy()
    np.testing.assert_array_equal(matrix[rows, cols], matrix[np.tril_indices(3)])
    assert (rows >= cols).all(), "every selected position is on or below"


def test_an_empty_triangle_is_an_empty_pair_of_rows():
    got = mt.triu_indices(3, 3, 5)
    assert tuple(got.shape) == (2, 0)


def test_the_index_builders_refuse_a_negative_size():
    with pytest.raises(ValueError, match="non-negative"):
        mt.tril_indices(-1, 3)


# --- diagflat, block_diag, cartesian_prod -----------------------------------


@pytest.mark.parametrize("offset", [-2, 0, 1])
def test_diagflat_matches_numpy(offset):
    for values in (np.array([1.0, 2.0, 3.0]), np.arange(4.0).reshape(2, 2)):
        np.testing.assert_array_equal(
            mt.diagflat(_t(values), offset).numpy(), np.diagflat(values, offset)
        )


def test_block_diag_places_each_input_on_the_diagonal():
    a = np.ones((1, 2))
    b = np.full((2, 1), 2.0)
    c = np.full((1, 1), 3.0)
    got = mt.block_diag(_t(a), _t(b), _t(c)).numpy()

    want = np.zeros((4, 4))
    want[0, 0:2] = a
    want[1:3, 2:3] = b
    want[3:4, 3:4] = c
    np.testing.assert_array_equal(got, want)


def test_block_diag_reads_a_vector_as_a_row_and_a_scalar_as_a_block():
    got = mt.block_diag(_t([1.0, 2.0]), _t(3.0)).numpy()
    np.testing.assert_array_equal(got, [[1.0, 2.0, 0.0], [0.0, 0.0, 3.0]])


def test_block_diag_refuses_a_rank_three_block():
    with pytest.raises(ValueError, match="at most two dimensions"):
        mt.block_diag(_t(np.zeros((2, 2, 2))))


def test_cartesian_prod_enumerates_every_combination():
    a = np.array([1.0, 2.0])
    b = np.array([3.0, 4.0, 5.0])
    got = mt.cartesian_prod(_t(a), _t(b)).numpy()

    want = np.array([[x, y] for x in a for y in b])
    np.testing.assert_array_equal(got, want)


def test_cartesian_prod_of_three_inputs_varies_the_last_fastest():
    got = mt.cartesian_prod(_t([0.0, 1.0]), _t([0.0, 1.0]), _t([0.0, 1.0])).numpy()
    want = np.array([[a, b, c] for a in (0, 1) for b in (0, 1) for c in (0, 1)])
    np.testing.assert_array_equal(got, want)


def test_cartesian_prod_of_one_input_is_that_input():
    values = _t([1.0, 2.0, 3.0])
    np.testing.assert_array_equal(mt.cartesian_prod(values).numpy(), [1.0, 2.0, 3.0])


def test_cartesian_prod_refuses_a_matrix():
    with pytest.raises(ValueError, match="1-D tensors"):
        mt.cartesian_prod(_t(BASE))


# --- gradients --------------------------------------------------------------


def test_take_routes_the_gradient_back_to_the_positions_it_read():
    tensor = _t(BASE, requires_grad=True)
    mt.take(tensor, _i([0, 0, 5])).sum().backward()

    want = np.zeros_like(BASE)
    want.reshape(-1)[0] = 2  # read twice
    want.reshape(-1)[5] = 1
    np.testing.assert_array_equal(tensor.grad.numpy(), want)


def test_masked_scatter_splits_the_gradient_between_target_and_source():
    mask = np.array([[True, False], [False, True]])
    tensor = _t(np.zeros((2, 2)), requires_grad=True)
    source = _t([1.0, 2.0], requires_grad=True)
    mt.masked_scatter(tensor, _t(mask, "bool"), source).sum().backward()

    # The target keeps the positions the mask skipped, the source the rest.
    np.testing.assert_array_equal(tensor.grad.numpy(), (~mask).astype(float))
    np.testing.assert_array_equal(source.grad.numpy(), [1.0, 1.0])


def test_select_and_take_along_dim_carry_gradients_too():
    tensor = _t(BASE, requires_grad=True)
    mt.select(tensor, 0, 1).sum().backward()
    want = np.zeros_like(BASE)
    want[1] = 1.0
    np.testing.assert_array_equal(tensor.grad.numpy(), want)

    other = _t(BASE, requires_grad=True)
    mt.take_along_dim(other, _i([[0], [1], [2]]), 1).sum().backward()
    want = np.zeros_like(BASE)
    want[0, 0] = want[1, 1] = want[2, 2] = 1.0
    np.testing.assert_array_equal(other.grad.numpy(), want)
