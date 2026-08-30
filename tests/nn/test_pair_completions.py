# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""Five functions that each finish something the library had half of.

`nn.Embedding` existed and its functional form did not, so a table the caller
already holds -- frozen, or shared between two models -- had no lookup.
`grid_sample` existed and `affine_grid` did not, so a spatial transformer had
its sampler and no way to describe the transform. `pixel_shuffle` rearranged
across space and nothing rearranged across channels. And `max_pool` and
`avg_pool` were the two ends of a family with no way to ask for anything
between them.

The strongest test here is the `affine_grid` one, because it pins three
conventions at once. An identity `theta` fed through `grid_sample` has to give
the image back exactly: that can only happen if the coordinate order, the
normalisation and `align_corners` all agree between the function that builds
the grid and the kernel that reads it.
"""

from __future__ import annotations

import numpy as np
import pytest

import minitensor as mt
from minitensor import functional as F

RNG = np.random.default_rng(67)


def _t(values, requires_grad=False):
    return mt.Tensor(
        np.ascontiguousarray(np.asarray(values, dtype=np.float64)),
        dtype="float64",
        requires_grad=requires_grad,
    )


def _i(values):
    return mt.Tensor.from_numpy(np.asarray(values, dtype=np.int64))


# --- embedding --------------------------------------------------------------


@pytest.mark.parametrize("shape", [(4,), (2, 3), (2, 2, 2)])
def test_embedding_looks_up_rows_and_keeps_the_shape_of_its_index(shape):
    table = RNG.normal(size=(6, 4))
    indices = RNG.integers(0, 6, size=shape)
    looked_up = F.embedding(_i(indices), _t(table))
    assert tuple(looked_up.shape) == (*shape, 4)
    np.testing.assert_array_equal(looked_up.numpy(), table[indices])


def test_embedding_agrees_with_the_module_that_owns_its_table():
    mt.manual_seed(0)
    module = mt.nn.Embedding(6, 4)
    table = module.parameters()[0]
    indices = _i([[0, 3], [5, 1]])
    np.testing.assert_array_equal(
        F.embedding(indices, table).numpy(), module(indices).numpy()
    )


def test_padding_idx_takes_no_gradient_and_changes_no_value():
    table = RNG.normal(size=(5, 3))
    weight = _t(table, requires_grad=True)

    # The row is read, and read unchanged.
    np.testing.assert_array_equal(F.embedding(_i([0]), weight, 0).numpy()[0], table[0])

    F.embedding(_i([0, 1, 0, 2]), weight, 0).sum().backward()
    gradient = weight.grad.numpy()
    np.testing.assert_array_equal(gradient[0], np.zeros(3))
    # Index 0 appeared twice; without the mask its row would have counted 2.
    np.testing.assert_array_equal(gradient[1], np.ones(3))
    np.testing.assert_array_equal(gradient[2], np.ones(3))
    mt.clear_autograd_graph()


def test_a_repeated_index_accumulates_its_gradient():
    weight = _t(RNG.normal(size=(4, 2)), requires_grad=True)
    F.embedding(_i([1, 1, 1]), weight).sum().backward()
    np.testing.assert_array_equal(weight.grad.numpy()[1], np.full(2, 3.0))
    mt.clear_autograd_graph()


def test_padding_idx_counts_from_the_end():
    weight = _t(RNG.normal(size=(5, 3)), requires_grad=True)
    F.embedding(_i([4, 0]), weight, -1).sum().backward()
    np.testing.assert_array_equal(weight.grad.numpy()[4], np.zeros(3))
    mt.clear_autograd_graph()


def test_a_padding_index_outside_the_table_is_refused():
    with pytest.raises(IndexError, match="out of range for a table"):
        F.embedding(_i([0]), _t(RNG.normal(size=(3, 2))), 7)


def test_embedding_needs_a_two_dimensional_table():
    with pytest.raises(ValueError, match="two-dimensional table"):
        F.embedding(_i([0]), _t([1.0, 2.0]))


# --- channel_shuffle --------------------------------------------------------


def test_channel_shuffle_interleaves_the_groups():
    values = np.arange(6.0).reshape(1, 6, 1, 1)
    # Three groups of two become two groups of three, taking one from each.
    np.testing.assert_array_equal(
        F.channel_shuffle(_t(values), 3).numpy()[0, :, 0, 0],
        [0.0, 2.0, 4.0, 1.0, 3.0, 5.0],
    )


@pytest.mark.parametrize(
    "shape,groups", [((2, 8, 3, 3), 4), ((1, 6, 5), 2), ((2, 9, 2, 2, 2), 3)]
)
def test_channel_shuffle_matches_the_reshape_it_names(shape, groups):
    values = RNG.normal(size=shape)
    reshaped = values.reshape(shape[0], groups, shape[1] // groups, *shape[2:])
    order = (0, 2, 1) + tuple(range(3, len(shape) + 1))
    np.testing.assert_array_equal(
        F.channel_shuffle(_t(values), groups).numpy(),
        reshaped.transpose(order).reshape(shape),
    )


def test_shuffling_by_one_group_or_by_every_channel_changes_nothing():
    values = _t(RNG.normal(size=(2, 6, 3)))
    np.testing.assert_array_equal(F.channel_shuffle(values, 1).numpy(), values.numpy())
    np.testing.assert_array_equal(F.channel_shuffle(values, 6).numpy(), values.numpy())


def test_channel_shuffle_is_a_permutation_and_loses_nothing():
    values = RNG.normal(size=(2, 8, 3, 3))
    shuffled = F.channel_shuffle(_t(values), 4).numpy()
    np.testing.assert_array_equal(np.sort(shuffled, axis=1), np.sort(values, axis=1))


def test_channels_that_do_not_divide_into_groups_are_refused():
    with pytest.raises(ValueError, match="divide by the number of groups"):
        F.channel_shuffle(_t(RNG.normal(size=(1, 5, 2))), 2)


# --- lp_pool ----------------------------------------------------------------


@pytest.mark.parametrize("p", [1.0, 2.0, 3.0, 6.0])
def test_lp_pool1d_is_the_p_norm_of_each_window(p):
    values = RNG.normal(size=(2, 3, 8))
    np.testing.assert_allclose(
        F.lp_pool1d(_t(values), p, 2).numpy(),
        (np.abs(values).reshape(2, 3, 4, 2) ** p).sum(-1) ** (1 / p),
        rtol=1e-11,
    )


@pytest.mark.parametrize("p", [1.0, 2.0, 5.0])
def test_lp_pool2d_is_the_p_norm_of_each_window(p):
    values = RNG.normal(size=(2, 3, 4, 6))
    np.testing.assert_allclose(
        F.lp_pool2d(_t(values), p, 2).numpy(),
        (np.abs(values).reshape(2, 3, 2, 2, 3, 2) ** p).sum((3, 5)) ** (1 / p),
        rtol=1e-11,
    )


def test_a_norm_type_of_one_is_the_sum_of_magnitudes():
    values = RNG.normal(size=(1, 2, 6))
    np.testing.assert_allclose(
        F.lp_pool1d(_t(values), 1.0, 3).numpy(),
        np.abs(values).reshape(1, 2, 2, 3).sum(-1),
        rtol=1e-12,
    )


def test_a_large_norm_type_approaches_the_maximum_magnitude():
    """The family `avg_pool` and `max_pool` are the ends of."""

    values = RNG.normal(size=(2, 3, 8))
    np.testing.assert_allclose(
        F.lp_pool1d(_t(values), 60.0, 2).numpy(),
        np.abs(values).reshape(2, 3, 4, 2).max(-1),
        rtol=1e-2,
    )


def test_a_negative_window_has_a_real_norm_at_an_odd_norm_type():
    """Where `torch` takes the root of a negative number, this takes `abs` first."""

    values = -np.ones((1, 1, 4))
    np.testing.assert_allclose(
        F.lp_pool1d(_t(values), 3.0, 2).numpy(),
        np.full((1, 1, 2), 2 ** (1 / 3)),
        rtol=1e-13,
    )


def test_lp_pool_carries_a_gradient_everywhere_a_maximum_would_not():
    values = _t(RNG.normal(size=(1, 2, 6)), requires_grad=True)
    F.lp_pool1d(values, 2.0, 2).sum().backward()
    # Every element contributes to its window, not just the largest.
    assert (np.abs(values.grad.numpy()) > 0).all()
    mt.clear_autograd_graph()


def test_a_norm_type_of_zero_or_less_is_refused():
    with pytest.raises(ValueError, match="positive norm type"):
        F.lp_pool1d(_t(RNG.normal(size=(1, 1, 4))), 0.0, 2)


# --- affine_grid ------------------------------------------------------------


@pytest.mark.parametrize("align_corners", [False, True])
@pytest.mark.parametrize("shape", [(1, 1, 3, 4), (2, 3, 5, 5)])
def test_an_identity_transform_samples_the_image_back(shape, align_corners):
    """Three conventions at once: coordinate order, scaling, and the corners."""

    theta = np.tile(np.array([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]]), (shape[0], 1, 1))
    grid = F.affine_grid(_t(theta), shape, align_corners)
    values = RNG.normal(size=shape)
    np.testing.assert_allclose(
        F.grid_sample(_t(values), grid, "bilinear", "zeros", align_corners).numpy(),
        values,
        rtol=1e-12,
        atol=1e-13,
    )


def test_an_identity_transform_samples_a_volume_back():
    shape = (1, 2, 2, 3, 4)
    theta = np.array([[[1.0, 0, 0, 0], [0, 1.0, 0, 0], [0, 0, 1.0, 0]]])
    grid = F.affine_grid(_t(theta), shape)
    values = RNG.normal(size=shape)
    np.testing.assert_allclose(
        F.grid_sample(_t(values), grid, "bilinear", "zeros", False).numpy(),
        values,
        rtol=1e-12,
        atol=1e-13,
    )


def test_the_grid_has_a_coordinate_per_spatial_axis_in_the_last_position():
    assert tuple(F.affine_grid(_t(np.zeros((2, 2, 3))), (2, 1, 3, 4)).shape) == (
        2,
        3,
        4,
        2,
    )
    assert tuple(F.affine_grid(_t(np.zeros((1, 3, 4))), (1, 1, 2, 3, 4)).shape) == (
        1,
        2,
        3,
        4,
        3,
    )


def test_align_corners_decides_where_minus_one_and_one_fall():
    theta = _t([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])
    corners = F.affine_grid(theta, (1, 1, 1, 4), True).numpy()[0, 0, :, 0]
    centres = F.affine_grid(theta, (1, 1, 1, 4), False).numpy()[0, 0, :, 0]
    np.testing.assert_allclose(corners, [-1.0, -1 / 3, 1 / 3, 1.0], rtol=1e-14)
    np.testing.assert_allclose(centres, [-0.75, -0.25, 0.25, 0.75], rtol=1e-14)


def test_the_coordinates_are_in_x_then_y_order():
    """The reverse of the axes they index, which is what `grid_sample` reads."""

    theta = _t([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])
    grid = F.affine_grid(theta, (1, 1, 2, 3), True).numpy()[0]
    # x varies along the width, y along the height.
    np.testing.assert_allclose(grid[0, :, 0], [-1.0, 0.0, 1.0], atol=1e-15)
    np.testing.assert_allclose(grid[:, 0, 1], [-1.0, 1.0], atol=1e-15)


def test_a_translation_moves_where_the_image_is_read_from():
    values = np.zeros((1, 1, 1, 4))
    values[0, 0, 0, 0] = 1.0
    # Shifting the sampling coordinates right by half the width reads the
    # leftmost pixel from the middle of the output.
    theta = _t([[[1.0, 0.0, -0.5], [0.0, 1.0, 0.0]]])
    grid = F.affine_grid(theta, (1, 1, 1, 4), True)
    sampled = F.grid_sample(_t(values), grid, "nearest", "zeros", True).numpy()
    assert sampled[0, 0, 0].argmax() == 1


def test_a_gradient_reaches_theta():
    """What makes a spatial transformer learn its transform rather than take one."""

    theta = _t([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]], requires_grad=True)
    values = _t(RNG.normal(size=(1, 1, 4, 4)))
    F.grid_sample(values, F.affine_grid(theta, (1, 1, 4, 4))).sum().backward()
    assert np.isfinite(theta.grad.numpy()).all()
    mt.clear_autograd_graph()


def test_a_theta_of_the_wrong_shape_is_refused():
    with pytest.raises(ValueError, match=r"theta of shape \[1, 2, 3\]"):
        F.affine_grid(_t(np.zeros((1, 3, 4))), (1, 1, 3, 3))


def test_an_output_size_of_the_wrong_rank_is_refused():
    with pytest.raises(ValueError, match="four- or five-element output size"):
        F.affine_grid(_t(np.zeros((1, 2, 3))), (1, 1, 3))
