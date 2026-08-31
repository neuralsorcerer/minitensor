# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""The write direction of `narrow`, `select` and `diagonal`.

Each reader had no counterpart: there was a way to look at a slice, a row or a
diagonal, and no way to produce a tensor with that region replaced -- short of
an in-place write, which a tensor in the graph cannot take. These three are
that expression, and the tests hold them to the two things that makes them
useful.

First, they mean what the assignment means. `slice_scatter` is checked against
`x[..., a:b:c] = src` run in NumPy, negative steps and out-of-range bounds
included, because the bounds are computed by a Python slice rather than by
arithmetic written here.

Second, the gradient splits. `src` gets it at the positions it landed on and
`input` gets it everywhere else, which is the property an in-place write cannot
offer and the reason to have the functions at all.

The index arithmetic gets its own check. The positions are built with `ix_` and
`ravel_multi_index` so only the written region is materialised, and that is
tested against the obvious form -- slicing a full index template -- which is
correct by inspection and too expensive to ship.
"""

from __future__ import annotations

import numpy as np
import pytest

import minitensor as mt
from minitensor._indexing import _axis_positions, _scatter_into

BASE = np.arange(24.0).reshape(2, 3, 4)
WEIGHTS = np.arange(1.0, 25.0).reshape(2, 3, 4)

# (dim, slice) pairs, spanning the cases a Python slice has to resolve.
SLICES = [
    (0, slice(1, 2)),
    (0, slice(None)),
    (0, slice(5, 9)),  # entirely past the end: an empty write
    (1, slice(0, 3, 2)),
    (1, slice(None, None, -1)),  # reversed, which numpy allows and torch does not
    (1, slice(-2, None)),
    (2, slice(1, 3)),
    (2, slice(None, None, 3)),
]


def _t(values, requires_grad=False):
    return mt.Tensor(
        np.ascontiguousarray(np.asarray(values, dtype=np.float64)),
        dtype="float64",
        requires_grad=requires_grad,
    )


# --- slice_scatter ----------------------------------------------------------


@pytest.mark.parametrize("dim,window", SLICES)
def test_slice_scatter_is_the_assignment_it_spells(dim, window):
    key = (slice(None),) * dim + (window,)
    expected = BASE.copy()
    source = -np.arange(1.0, expected[key].size + 1).reshape(expected[key].shape)
    expected[key] = source

    np.testing.assert_array_equal(
        mt.slice_scatter(
            _t(BASE), _t(source), dim, window.start, window.stop, window.step or 1
        ).numpy(),
        expected,
    )


def test_slice_scatter_leaves_the_input_alone():
    """An expression, not an in-place write."""

    values = _t(BASE)
    mt.slice_scatter(values, _t(np.zeros((1, 3, 4))), 0, 0, 1)
    np.testing.assert_array_equal(values.numpy(), BASE)


def test_slice_scatter_broadcasts_its_source_the_way_an_assignment_does():
    expected = BASE.copy()
    expected[:, 1:2, :] = -5.0
    np.testing.assert_array_equal(
        mt.slice_scatter(_t(BASE), _t(-5.0), 1, 1, 2).numpy(), expected
    )


def test_slice_scatter_takes_a_negative_dim():
    np.testing.assert_array_equal(
        mt.slice_scatter(_t(BASE), _t(np.zeros((2, 3, 1))), -1, 0, 1).numpy(),
        mt.slice_scatter(_t(BASE), _t(np.zeros((2, 3, 1))), 2, 0, 1).numpy(),
    )


def test_the_gradient_of_slice_scatter_splits_between_its_operands():
    values = _t(BASE, requires_grad=True)
    source = _t(np.zeros((1, 3, 4)), requires_grad=True)
    (mt.slice_scatter(values, source, 0, 1, 2) * _t(WEIGHTS)).sum().backward()

    written = np.zeros_like(BASE, dtype=bool)
    written[1:2] = True
    np.testing.assert_array_equal(values.grad.numpy(), np.where(written, 0.0, WEIGHTS))
    np.testing.assert_array_equal(source.grad.numpy(), WEIGHTS[1:2])
    mt.clear_autograd_graph()


def test_a_broadcast_source_gathers_the_whole_regions_gradient():
    source = _t(0.0, requires_grad=True)
    (mt.slice_scatter(_t(BASE), source, 1, 1, 2) * _t(WEIGHTS)).sum().backward()
    np.testing.assert_allclose(
        float(source.grad.item()), WEIGHTS[:, 1:2, :].sum(), rtol=0
    )
    mt.clear_autograd_graph()


# --- select_scatter ---------------------------------------------------------


@pytest.mark.parametrize("dim", [0, 1, 2, -1])
def test_select_scatter_is_the_assignment_that_drops_the_axis(dim):
    axis = dim % BASE.ndim
    key = (slice(None),) * axis + (1,)
    expected = BASE.copy()
    source = -np.arange(1.0, expected[key].size + 1).reshape(expected[key].shape)
    expected[key] = source

    got = mt.select_scatter(_t(BASE), _t(source), dim, 1)
    np.testing.assert_array_equal(got.numpy(), expected)


def test_select_scatter_wants_one_axis_fewer_than_slice_scatter():
    """The difference between the two, stated as a shape."""

    row = _t(np.full((2, 4), -1.0))
    np.testing.assert_array_equal(
        mt.select_scatter(_t(BASE), row, 1, 1).numpy(),
        mt.slice_scatter(_t(BASE), row.reshape(2, 1, 4), 1, 1, 2).numpy(),
    )


def test_select_scatter_counts_a_negative_index_from_the_end():
    np.testing.assert_array_equal(
        mt.select_scatter(_t(BASE), _t(np.zeros((2, 4))), 1, -1).numpy(),
        mt.select_scatter(_t(BASE), _t(np.zeros((2, 4))), 1, 2).numpy(),
    )


def test_the_gradient_of_select_scatter_splits_between_its_operands():
    values = _t(BASE, requires_grad=True)
    source = _t(np.zeros((2, 4)), requires_grad=True)
    (mt.select_scatter(values, source, 1, 2) * _t(WEIGHTS)).sum().backward()

    expected = WEIGHTS.copy()
    expected[:, 2, :] = 0.0
    np.testing.assert_array_equal(values.grad.numpy(), expected)
    np.testing.assert_array_equal(source.grad.numpy(), WEIGHTS[:, 2, :])
    mt.clear_autograd_graph()


# --- diagonal_scatter -------------------------------------------------------


@pytest.mark.parametrize("offset", [-2, -1, 0, 1, 3, 9])
@pytest.mark.parametrize("shape", [(3, 4), (4, 3), (2, 3, 4), (5, 5)])
def test_writing_back_the_diagonal_it_read_changes_nothing(offset, shape):
    """`diagonal` and `diagonal_scatter` agree on which positions those are."""

    values = _t(np.arange(float(np.prod(shape))).reshape(shape))
    np.testing.assert_array_equal(
        mt.diagonal_scatter(
            values, mt.functional.diagonal(values, offset), offset
        ).numpy(),
        values.numpy(),
    )


def _diagonal_mask(shape, offset):
    """Which positions the `offset` diagonal of the last two axes covers."""

    mask = np.zeros(shape, dtype=bool)
    rows, columns = shape[-2:]
    down, across = max(0, -offset), max(0, offset)
    for step in range(max(0, min(rows - down, columns - across))):
        mask[..., down + step, across + step] = True
    return mask


@pytest.mark.parametrize("offset", [-1, 0, 2])
def test_diagonal_scatter_writes_the_diagonal_and_nothing_else(offset):
    np.testing.assert_array_equal(
        mt.diagonal_scatter(_t(BASE), _t(-1.0), offset).numpy(),
        np.where(_diagonal_mask(BASE.shape, offset), -1.0, BASE),
    )


def test_the_gradient_of_diagonal_scatter_splits_between_its_operands():
    values = _t(BASE, requires_grad=True)
    source = _t(np.zeros((2, 3)), requires_grad=True)
    (mt.diagonal_scatter(values, source) * _t(WEIGHTS)).sum().backward()

    np.testing.assert_array_equal(
        values.grad.numpy(),
        np.where(_diagonal_mask(BASE.shape, 0), 0.0, WEIGHTS),
    )
    np.testing.assert_array_equal(
        source.grad.numpy(), mt.functional.diagonal(_t(WEIGHTS)).numpy()
    )
    mt.clear_autograd_graph()


def test_a_diagonal_that_runs_off_the_matrix_writes_nothing():
    values = _t(BASE)
    np.testing.assert_array_equal(
        mt.diagonal_scatter(values, _t(np.zeros((2, 0))), 9).numpy(), BASE
    )


# --- the index arithmetic ---------------------------------------------------


@pytest.mark.parametrize("dim,window", SLICES)
def test_the_open_mesh_positions_match_slicing_a_full_index_template(dim, window):
    """The shipped form against the obvious one it replaces.

    Slicing `arange(size).reshape(shape)` is correct by inspection and costs one
    integer per element of the whole tensor; `ix_` plus `ravel_multi_index`
    costs one per element written. They have to agree.
    """

    shape = BASE.shape
    template = np.arange(int(np.prod(shape))).reshape(shape)
    key = (slice(None),) * dim + (window,)
    along = np.arange(*window.indices(shape[dim]))

    np.testing.assert_array_equal(_axis_positions(shape, dim, along), template[key])


# --- what they refuse -------------------------------------------------------


def test_a_source_that_does_not_broadcast_is_refused():
    with pytest.raises(ValueError, match="does not broadcast"):
        mt.slice_scatter(_t(BASE), _t(np.zeros((7, 7))), 0, 0, 1)


def test_a_step_of_zero_is_refused():
    with pytest.raises(ValueError, match="step cannot be zero"):
        mt.slice_scatter(_t(BASE), _t(0.0), 0, 0, 2, 0)


def test_an_axis_that_does_not_exist_is_refused():
    with pytest.raises(ValueError, match="dim 5 is out of range"):
        mt.slice_scatter(_t(BASE), _t(0.0), 5, 0, 1)


def test_select_scatter_reports_an_index_past_the_end():
    with pytest.raises(IndexError, match="out of range"):
        mt.select_scatter(_t(BASE), _t(np.zeros((2, 4))), 1, 7)


def test_diagonal_scatter_needs_a_matrix():
    with pytest.raises(ValueError, match="at least two dimensions"):
        mt.diagonal_scatter(_t(np.zeros(4)), _t(0.0))


# A step of one writes a run of neighbouring positions, which `cat` assembles
# from the pieces either side of it -- no offset is named at all. That is a
# different implementation of the same function, so it is held to the one it
# replaced: the same values and the same two gradients, over every slice a
# Python slice can resolve.
CONTIGUOUS = [
    (0, slice(None)),  # the whole axis, which has to be written as two
    (0, slice(0, 1)),
    (0, slice(1, None)),
    (1, slice(0, 2)),
    (1, slice(1, 2)),
    (1, slice(2, None)),
    (1, slice(2, 1)),  # empty: `stop` below `start`
    (1, slice(9, 12)),  # empty: entirely past the end
    (2, slice(None)),
    (2, slice(1, 3)),
    (2, slice(None, -1)),
]


def _by_positions(tensor, source, dim, window):
    """`slice_scatter` as it was: one flat offset per element written."""

    shape = tuple(int(size) for size in tensor.shape)
    bounds = window.indices(shape[dim])
    return _scatter_into(
        "slice_scatter", tensor, source, _axis_positions(shape, dim, np.arange(*bounds))
    )


@pytest.mark.parametrize("dim,window", CONTIGUOUS)
def test_a_contiguous_write_matches_the_positions_it_no_longer_builds(dim, window):
    width = len(range(*window.indices(BASE.shape[dim])))
    region = BASE.shape[:dim] + (width,) + BASE.shape[dim + 1 :]
    source = np.arange(1.0, 1 + int(np.prod(region))).reshape(region)

    np.testing.assert_array_equal(
        mt.slice_scatter(_t(BASE), _t(source), dim, window.start, window.stop).numpy(),
        _by_positions(_t(BASE), _t(source), dim, window).numpy(),
    )


@pytest.mark.parametrize("dim,window", CONTIGUOUS)
def test_both_gradients_survive_the_change_of_route(dim, window):
    """`cat`'s backward splits the gradient back into the pieces it joined,
    which has to come out where `scatter`'s did -- including the zero the
    written-into tensor keeps, and including a write that covers the whole
    axis, where nothing of that tensor reaches the answer at all."""

    width = len(range(*window.indices(BASE.shape[dim])))
    region = BASE.shape[:dim] + (width,) + BASE.shape[dim + 1 :]
    source = np.arange(1.0, 1 + int(np.prod(region))).reshape(region)

    grads = []
    for write in (
        lambda t, s: mt.slice_scatter(t, s, dim, window.start, window.stop),
        lambda t, s: _by_positions(t, s, dim, window),
    ):
        tensor, values = _t(BASE, True), _t(source, True)
        (write(tensor, values) * _t(WEIGHTS)).sum().backward()
        grads.append(
            tuple(
                None if g is None else g.numpy().copy()
                for g in (tensor.grad, values.grad)
            )
        )
        mt.clear_autograd_graph()

    for through_cat, through_positions in zip(*grads):
        assert (through_cat is None) == (through_positions is None)
        if through_cat is not None:
            np.testing.assert_array_equal(through_cat, through_positions)


def test_the_tensor_written_over_in_full_still_receives_a_zero_gradient():
    """Nothing of it reaches the answer, so it could fall out of the graph and
    leave its gradient `None`. It is written as two -- the first stopping one
    short so a piece of it survives to carry the zero -- and this is the
    assertion that says why."""

    tensor, values = _t(BASE, True), _t(BASE + 1, True)
    mt.slice_scatter(tensor, values, 0).sum().backward()
    assert tensor.grad is not None
    np.testing.assert_array_equal(tensor.grad.numpy(), np.zeros_like(BASE))
    np.testing.assert_array_equal(values.grad.numpy(), np.ones_like(BASE))
    mt.clear_autograd_graph()


@pytest.mark.parametrize("length", [1, 2, 3])
def test_an_axis_too_short_to_split_is_still_written(length):
    """The split needs a position to leave standing, so an axis of one has to
    stay on the positions. The answer is the same either way."""

    base = np.arange(float(length * 2)).reshape(length, 2)
    source = np.full((length, 2), 7.0)
    np.testing.assert_array_equal(
        mt.slice_scatter(_t(base), _t(source), 0).numpy(), source
    )

    tensor = _t(base, True)
    mt.slice_scatter(tensor, _t(source), 0).sum().backward()
    np.testing.assert_array_equal(tensor.grad.numpy(), np.zeros_like(base))
    mt.clear_autograd_graph()
