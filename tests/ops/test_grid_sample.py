# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""Reading an image at coordinates rather than at indices.

`grid_sample` reads its input at the normalised coordinates in a grid, and --
the point of it -- differentiates with respect to those coordinates. That is
what makes a spatial transformer or an optical-flow warp trainable.

Almost all of the content is conventions that are easy to get backwards and
invisible when you do: which end of the grid's last axis is `x`, what `-1`
names, and what happens off the edge. Each has a test that fails if it is
reversed. The rest is checked against a plain reference implementation written
in the module docstring's own terms, and against central differences.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import minitensor as mt


def _t(a, requires_grad=False):
    return mt.Tensor.from_numpy(
        np.ascontiguousarray(np.asarray(a, dtype=np.float64)),
        requires_grad=requires_grad,
    )


# --------------------------------------------------------------------------
# A reference, written straight from the definition
# --------------------------------------------------------------------------


def _unnormalize(coord, size, align_corners):
    if align_corners:
        return (coord + 1) / 2 * (size - 1)
    return ((coord + 1) * size - 1) / 2


def _clip(value, size):
    return min(size - 1.0, max(value, 0.0))


def _reflect(value, low, high):
    span = high - low
    if span <= 0:
        return low
    distance = abs(value - low)
    folds = math.floor(distance / span)
    extra = distance - folds * span
    return extra + low if folds % 2 == 0 else span - extra + low


def _source(coord, size, padding, align_corners):
    value = _unnormalize(coord, size, align_corners)
    if padding == "border":
        value = _clip(value, size)
    elif padding == "reflection":
        low, high = (0.0, size - 1.0) if align_corners else (-0.5, size - 0.5)
        value = _clip(_reflect(value, low, high), size)
    return value


def _reference(image, grid, mode="bilinear", padding="zeros", align_corners=False):
    batch, channels, height, width = image.shape
    _, out_h, out_w, _ = grid.shape
    out = np.zeros((batch, channels, out_h, out_w))
    for n in range(batch):
        for i in range(out_h):
            for j in range(out_w):
                x = _source(grid[n, i, j, 0], width, padding, align_corners)
                y = _source(grid[n, i, j, 1], height, padding, align_corners)
                if mode == "nearest":
                    ix, iy = int(np.rint(x)), int(np.rint(y))
                    if 0 <= ix < width and 0 <= iy < height:
                        out[n, :, i, j] = image[n, :, iy, ix]
                    continue
                x0, y0 = math.floor(x), math.floor(y)
                fx, fy = x - x0, y - y0
                for dy in (0, 1):
                    for dx in (0, 1):
                        iy, ix = y0 + dy, x0 + dx
                        if 0 <= ix < width and 0 <= iy < height:
                            weight = (fx if dx else 1 - fx) * (fy if dy else 1 - fy)
                            out[n, :, i, j] += weight * image[n, :, iy, ix]
    return out


def _call(image, grid, **kwargs):
    return mt.nn.grid_sample(_t(image), _t(grid), **kwargs).numpy()


@pytest.mark.parametrize("mode", ["bilinear", "nearest"])
@pytest.mark.parametrize("padding", ["zeros", "border", "reflection"])
@pytest.mark.parametrize("align_corners", [False, True])
def test_against_a_reference_written_from_the_definition(mode, padding, align_corners):
    rng = np.random.default_rng(hash((mode, padding, align_corners)) % 2**32)
    image = rng.normal(size=(2, 3, 5, 4))
    # Deliberately outside [-1, 1] as well, so the padding mode is exercised.
    grid = rng.uniform(-2.0, 2.0, size=(2, 6, 7, 2))
    want = _reference(image, grid, mode, padding, align_corners)
    got = _call(
        image, grid, mode=mode, padding_mode=padding, align_corners=align_corners
    )
    assert np.allclose(got, want, rtol=0, atol=1e-12)


# --------------------------------------------------------------------------
# The conventions
# --------------------------------------------------------------------------


def _identity_grid(height, width, align_corners):
    """The grid that reads every sample exactly where it already is."""
    if align_corners:
        ys = np.linspace(-1.0, 1.0, height) if height > 1 else np.zeros(1)
        xs = np.linspace(-1.0, 1.0, width) if width > 1 else np.zeros(1)
    else:
        ys = (2 * np.arange(height) + 1) / height - 1
        xs = (2 * np.arange(width) + 1) / width - 1
    grid = np.zeros((1, height, width, 2))
    grid[0, :, :, 0] = xs[None, :]
    grid[0, :, :, 1] = ys[:, None]
    return grid


@pytest.mark.parametrize("align_corners", [False, True])
def test_the_identity_grid_returns_the_input(align_corners):
    """The first thing that must be true, and the one that pins `align_corners`
    against the normalisation rather than merely against itself."""
    rng = np.random.default_rng(1)
    image = rng.normal(size=(1, 2, 4, 5))
    grid = _identity_grid(4, 5, align_corners)
    got = _call(image, grid, align_corners=align_corners)
    assert np.allclose(got, image, atol=1e-12)


def test_the_grid_names_x_before_y():
    """The grid's last axis is `x, y` -- the reverse of the `H, W` it indexes.
    Reading it the other way round transposes the output silently, so this asks
    for a point where the two coordinates cannot be confused."""
    image = np.arange(12, dtype=np.float64).reshape(1, 1, 3, 4)
    # x = -1, y = +1 is the bottom-left sample under align_corners: row 2,
    # column 0, which holds 8. Swapping x and y would be out of range in y.
    grid = np.array([[[[-1.0, 1.0]]]])
    got = _call(image, grid, align_corners=True)
    assert got[0, 0, 0, 0] == 8.0


def test_a_midpoint_is_the_average_of_its_neighbours():
    image = np.array([[[[0.0, 10.0]]]])
    grid = np.array([[[[0.0, 0.0]]]])
    assert _call(image, grid, align_corners=True)[0, 0, 0, 0] == pytest.approx(5.0)


def test_nearest_rounds_instead_of_blending():
    image = np.array([[[[0.0, 10.0, 20.0]]]])
    # Three quarters of the way from sample 0 to sample 2, i.e. x = 1.5, which
    # rounds to even -- to sample 2.
    grid = np.array([[[[0.5, 0.0]]]])
    assert _call(image, grid, mode="bilinear", align_corners=True)[0, 0, 0, 0] == 15.0
    assert _call(image, grid, mode="nearest", align_corners=True)[0, 0, 0, 0] == 20.0


def test_align_corners_moves_the_reading_by_half_a_pixel():
    """Not a matter of taste at the call site: a model trained under one
    setting reads the wrong place under the other."""
    image = np.array([[[[0.0, 1.0, 2.0, 3.0]]]])
    grid = np.array([[[[-1.0, 0.0]]]])
    aligned = _call(image, grid, align_corners=True)[0, 0, 0, 0]
    unaligned = _call(image, grid, align_corners=False)[0, 0, 0, 0]
    # -1 is sample 0's centre when aligned, and half a pixel outside it when
    # not -- which under `zeros` reads half of sample 0 and half of nothing.
    assert aligned == pytest.approx(0.0)
    assert unaligned == pytest.approx(0.0)
    # Half a pixel apart: -0.5 lands at 0.75 of the way to sample 1 when the
    # corners are aligned, and at 0.5 of the way when they are not.
    middle = np.array([[[[-0.5, 0.0]]]])
    assert _call(image, middle, align_corners=True)[0, 0, 0, 0] == pytest.approx(0.75)
    assert _call(image, middle, align_corners=False)[0, 0, 0, 0] == pytest.approx(0.5)


# --------------------------------------------------------------------------
# Off the edge
# --------------------------------------------------------------------------


def test_zeros_reads_nothing_outside():
    image = np.array([[[[1.0, 2.0, 3.0]]]])
    grid = np.array([[[[3.0, 0.0]]]])
    assert _call(image, grid, padding_mode="zeros", align_corners=True)[0, 0, 0, 0] == 0.0


def test_border_holds_the_edge_value():
    image = np.array([[[[1.0, 2.0, 3.0]]]])
    grid = np.array([[[[3.0, 0.0]]]])
    got = _call(image, grid, padding_mode="border", align_corners=True)
    assert got[0, 0, 0, 0] == pytest.approx(3.0)


def test_reflection_folds_back_inside():
    """Sample 3 of a four-wide row, stepped one past the end, is sample 2."""
    image = np.array([[[[0.0, 1.0, 2.0, 3.0]]]])
    # align_corners: x in [-1, 1] maps to [0, 3]. x = 5/3 maps to 4, which
    # reflects about 3 back to 2.
    grid = np.array([[[[5.0 / 3.0, 0.0]]]])
    got = _call(image, grid, padding_mode="reflection", align_corners=True)
    assert got[0, 0, 0, 0] == pytest.approx(2.0)


def test_reflection_folds_repeatedly():
    image = np.array([[[[0.0, 1.0, 2.0, 3.0]]]])
    # x = 3 maps to 6: out to 3, back to 0, so 6 folds to 0.
    grid = np.array([[[[3.0, 0.0]]]])
    got = _call(image, grid, padding_mode="reflection", align_corners=True)
    assert got[0, 0, 0, 0] == pytest.approx(0.0)


def test_zeros_blends_with_nothing_rather_than_with_the_edge():
    """The difference between dropping a neighbour and moving the coordinate:
    just inside the edge, `zeros` still loses half the weight."""
    image = np.array([[[[4.0, 8.0]]]])
    grid = np.array([[[[-1.0, 0.0]]]])  # half a pixel outside when unaligned
    assert _call(image, grid, padding_mode="zeros", align_corners=False)[
        0, 0, 0, 0
    ] == pytest.approx(2.0)
    assert _call(image, grid, padding_mode="border", align_corners=False)[
        0, 0, 0, 0
    ] == pytest.approx(4.0)


# --------------------------------------------------------------------------
# Gradients
# --------------------------------------------------------------------------


def _grads(image, grid, want=("input", "grid"), **kwargs):
    tensors = {
        "input": _t(image, requires_grad="input" in want),
        "grid": _t(grid, requires_grad="grid" in want),
    }
    out = mt.nn.grid_sample(tensors["input"], tensors["grid"], **kwargs)
    out.sum().backward()
    return (
        tensors["input"].grad.numpy() if "input" in want else None,
        tensors["grid"].grad.numpy() if "grid" in want else None,
    )


def _numeric(fn, values, step=1e-6):
    out = np.zeros_like(values)
    flat = values.reshape(-1)
    for index in range(flat.size):
        up, down = values.copy().reshape(-1), values.copy().reshape(-1)
        up[index] += step
        down[index] -= step
        out.reshape(-1)[index] = (
            fn(up.reshape(values.shape)) - fn(down.reshape(values.shape))
        ) / (2 * step)
    return out


@pytest.mark.parametrize("padding", ["zeros", "border", "reflection"])
@pytest.mark.parametrize("align_corners", [False, True])
def test_the_input_gradient_matches_central_differences(padding, align_corners):
    rng = np.random.default_rng(2)
    image = rng.normal(size=(1, 2, 4, 3))
    grid = rng.uniform(-1.4, 1.4, size=(1, 3, 3, 2))
    options = dict(padding_mode=padding, align_corners=align_corners)
    got, _ = _grads(image, grid, want=("input",), **options)
    want = _numeric(lambda v: _call(v, grid, **options).sum(), image)
    assert np.allclose(got, want, atol=1e-6)


def _smooth_grid(rng, shape, sizes, align_corners, reach=1.0):
    """Coordinates a difference quotient can be taken at.

    The blend's derivative genuinely jumps where a coordinate crosses a sample,
    and reflection's jumps where it crosses a fold, so a random grid has to be
    nudged off those before it means anything. `reach` above one lets the grid
    leave the image, which is the only way the padding modes' own contribution
    to the derivative -- clamped to zero, or reversed by a fold -- is exercised
    at all.
    """
    grid = rng.uniform(-reach, reach, size=shape)
    for axis in range(shape[-1]):
        # Grid axis `axis` indexes spatial axis `-1 - axis`; both maps put
        # samples one apart in source units, and folds land on them too.
        size = sizes[-1 - axis]
        scale = (size - 1) / 2 if align_corners else size / 2
        if scale == 0:
            continue
        values = grid[..., axis]
        source = (values + 1) * scale - (0.0 if align_corners else 0.5)
        # Push anything within a fifth of a sample of a boundary to the middle
        # of its cell, where both one-sided derivatives agree.
        offset = source - np.floor(source)
        bad = (offset < 0.2) | (offset > 0.8)
        source = np.where(bad, np.floor(source) + 0.5, source)
        grid[..., axis] = (source + (0.0 if align_corners else 0.5)) / scale - 1
    return grid


@pytest.mark.parametrize("padding", ["zeros", "border", "reflection"])
@pytest.mark.parametrize("align_corners", [False, True])
def test_the_grid_gradient_matches_central_differences(padding, align_corners):
    """The one that matters. It is invisible in the forward pass, and a wrong
    sign here turns training into a slow drift the wrong way."""
    rng = np.random.default_rng(3)
    image = rng.normal(size=(1, 2, 4, 3))
    grid = _smooth_grid(rng, (1, 3, 3, 2), (4, 3), align_corners)
    options = dict(padding_mode=padding, align_corners=align_corners)
    _, got = _grads(image, grid, want=("grid",), **options)
    want = _numeric(lambda v: _call(image, v, **options).sum(), grid)
    assert np.allclose(got, want, atol=1e-6)


@pytest.mark.parametrize("padding", ["zeros", "border", "reflection"])
@pytest.mark.parametrize("align_corners", [False, True])
def test_the_grid_gradient_matches_central_differences_off_the_edge(
    padding, align_corners
):
    """The same check with the grid reaching well outside the image, which is
    the only way the padding mode's own contribution to the derivative is
    exercised -- `border` flattening it to zero, `reflection` turning it around.
    Inside the image every padding mode agrees, so a test that stays inside
    passes with that contribution deleted entirely."""
    rng = np.random.default_rng(11)
    image = rng.normal(size=(1, 2, 4, 3))
    grid = _smooth_grid(rng, (1, 4, 4, 2), (4, 3), align_corners, reach=2.6)
    assert np.abs(grid).max() > 1.0, "the point of this test is coordinates outside"
    options = dict(padding_mode=padding, align_corners=align_corners)
    _, got = _grads(image, grid, want=("grid",), **options)
    want = _numeric(lambda v: _call(image, v, **options).sum(), grid)
    assert np.allclose(got, want, atol=1e-6)


def test_the_border_gradient_really_reaches_zero_somewhere():
    """Guards the test above: if every drawn coordinate stayed inside, it would
    pass without ever clamping anything, which is how the first version of it
    missed a wrong rate entirely."""
    rng = np.random.default_rng(11)
    image = rng.normal(size=(1, 2, 4, 3))
    grid = _smooth_grid(rng, (1, 4, 4, 2), (4, 3), True, reach=2.6)
    _, got = _grads(image, grid, want=("grid",), padding_mode="border", align_corners=True)
    assert np.any(got == 0.0)


def test_reflection_gradients_come_back_with_both_signs():
    """The companion guard: a fold reverses the direction of travel, so over a
    grid that reaches past the edge the coordinate gradient has to take both
    signs. It cannot if the reversal is dropped."""
    rng = np.random.default_rng(11)
    image = rng.normal(size=(1, 2, 4, 3))
    grid = _smooth_grid(rng, (1, 4, 4, 2), (4, 3), True, reach=2.6)
    options = dict(padding_mode="reflection", align_corners=True)
    _, folded = _grads(image, grid, want=("grid",), **options)
    _, plain = _grads(image, grid, want=("grid",), padding_mode="zeros", align_corners=True)
    # Somewhere a fold has turned the gradient around relative to the unfolded
    # reading of the same coordinate.
    assert np.any(np.sign(folded) * np.sign(plain) < 0)


def test_nearest_has_no_gradient_in_the_coordinate():
    """Exactly zero, not merely small: rounding is flat between samples."""
    rng = np.random.default_rng(4)
    image = rng.normal(size=(1, 2, 4, 3))
    grid = rng.uniform(-0.8, 0.8, size=(1, 3, 3, 2))
    input_grad, grid_grad = _grads(image, grid, mode="nearest")
    assert np.all(grid_grad == 0.0)
    assert not np.all(input_grad == 0.0)


def test_a_coordinate_held_against_the_border_stops_moving_the_output():
    """`border` clamps, so past the edge the gradient is zero -- and a model
    that pushes a coordinate off the image gets no signal to bring it back."""
    image = np.arange(8, dtype=np.float64).reshape(1, 1, 2, 4)
    outside = np.array([[[[2.0, 0.0]]]])
    _, grid_grad = _grads(image, outside, padding_mode="border", align_corners=True)
    assert grid_grad[0, 0, 0, 0] == 0.0

    inside = np.array([[[[0.1, 0.0]]]])
    _, moving = _grads(image, inside, padding_mode="border", align_corners=True)
    assert moving[0, 0, 0, 0] != 0.0


def test_reflection_reverses_the_coordinate_gradient():
    """One fold turns the direction of travel around, so the gradient's sign
    flips with it. Reporting the unfolded sign would look right on the forward
    pass and train the coordinate away from where it should go."""
    image = np.array([[[[0.0, 1.0, 2.0, 3.0]]]])
    options = dict(padding_mode="reflection", align_corners=True)
    _, before = _grads(image, np.array([[[[0.5, 0.0]]]]), **options)
    # x = 1.4 maps to 3.6, which folds back to 2.4 and is travelling the other
    # way; 0.5 maps to 2.25 and is not folded.
    _, after = _grads(image, np.array([[[[1.4, 0.0]]]]), **options)
    assert before[0, 0, 0, 0] > 0
    assert after[0, 0, 0, 0] < 0


def test_both_gradients_arrive_together():
    rng = np.random.default_rng(5)
    image = rng.normal(size=(1, 2, 4, 3))
    grid = rng.uniform(-0.6, 0.6, size=(1, 3, 3, 2))
    both = _grads(image, grid)
    only_input = _grads(image, grid, want=("input",))[0]
    only_grid = _grads(image, grid, want=("grid",))[1]
    assert np.allclose(both[0], only_input)
    assert np.allclose(both[1], only_grid)


def test_the_input_gradient_is_the_weights_scattered_back():
    """One output position, one channel: the gradient of the sum is exactly the
    four blend weights, dropped where they belong."""
    image = np.zeros((1, 1, 2, 2))
    grid = np.array([[[[-0.5, -0.5]]]])  # a quarter of the way in, aligned
    input_grad, _ = _grads(image, grid, want=("input",), align_corners=True)
    assert np.allclose(
        input_grad[0, 0], [[0.75 * 0.75, 0.25 * 0.75], [0.75 * 0.25, 0.25 * 0.25]]
    )
    assert input_grad.sum() == pytest.approx(1.0)


# --------------------------------------------------------------------------
# Volumes
# --------------------------------------------------------------------------


def test_a_volume_is_the_same_rule_with_one_more_axis():
    rng = np.random.default_rng(6)
    volume = rng.normal(size=(1, 2, 3, 4, 5))
    grid = np.zeros((1, 1, 1, 1, 3))  # the centre of the volume
    got = mt.nn.grid_sample(_t(volume), _t(grid), align_corners=True).numpy()
    # Depth 3 and width 5 have an exact centre sample; height 4 does not, so
    # the answer is the mean of rows 1 and 2 there.
    want = 0.5 * (volume[:, :, 1, 1, 2] + volume[:, :, 1, 2, 2])
    assert np.allclose(got[:, :, 0, 0, 0], want)


def test_a_volume_grid_names_x_y_z_in_that_order():
    volume = np.arange(24, dtype=np.float64).reshape(1, 1, 2, 3, 4)
    # x = -1, y = -1, z = 1 is depth 1, row 0, column 0, which holds 12.
    grid = np.array([[[[[-1.0, -1.0, 1.0]]]]])
    got = mt.nn.grid_sample(_t(volume), _t(grid), align_corners=True).numpy()
    assert got[0, 0, 0, 0, 0] == 12.0


def test_the_volume_gradient_matches_central_differences():
    rng = np.random.default_rng(7)
    volume = rng.normal(size=(1, 1, 3, 3, 3))
    grid = rng.uniform(-0.5, 0.5, size=(1, 2, 2, 2, 3))

    def forward(g):
        return mt.nn.grid_sample(_t(volume), _t(g), align_corners=True).numpy().sum()

    tensors = _t(volume), _t(grid, requires_grad=True)
    out = mt.nn.grid_sample(tensors[0], tensors[1], align_corners=True)
    out.sum().backward()
    assert np.allclose(tensors[1].grad.numpy(), _numeric(forward, grid), atol=1e-6)


# --------------------------------------------------------------------------
# Shapes, dtypes and what it refuses
# --------------------------------------------------------------------------


def test_the_output_takes_its_shape_from_the_grid():
    rng = np.random.default_rng(8)
    image = rng.normal(size=(3, 5, 7, 9))
    grid = rng.uniform(-1, 1, size=(3, 2, 11, 2))
    assert _call(image, grid).shape == (3, 5, 2, 11)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_both_floating_dtypes(dtype):
    rng = np.random.default_rng(9)
    image = rng.normal(size=(1, 1, 4, 4)).astype(dtype)
    grid = rng.uniform(-1, 1, size=(1, 3, 3, 2)).astype(dtype)
    out = mt.nn.grid_sample(mt.Tensor.from_numpy(image), mt.Tensor.from_numpy(grid))
    assert out.numpy().dtype == dtype
    exact = _call(image.astype(np.float64), grid.astype(np.float64))
    assert np.allclose(out.numpy(), exact, atol=1e-6)


def test_the_input_and_the_grid_must_share_a_dtype():
    image = np.zeros((1, 1, 2, 2), dtype=np.float32)
    grid = np.zeros((1, 1, 1, 2), dtype=np.float64)
    with pytest.raises(Exception, match="dtype"):
        mt.nn.grid_sample(
            mt.Tensor.from_numpy(image), mt.Tensor.from_numpy(grid)
        )


def test_the_ranks_must_agree():
    with pytest.raises(Exception, match="dimensional"):
        _call(np.zeros((1, 1, 2, 2)), np.zeros((1, 1, 1, 1, 3)))


def test_only_images_and_volumes():
    with pytest.raises(Exception, match="batch, channels"):
        _call(np.zeros((1, 1, 2)), np.zeros((1, 1, 1)))


def test_the_grid_must_hold_one_coordinate_per_spatial_axis():
    with pytest.raises(Exception, match="2 coordinates"):
        _call(np.zeros((1, 1, 2, 2)), np.zeros((1, 1, 1, 3)))


def test_the_batch_sizes_must_agree():
    with pytest.raises(Exception, match="batch size"):
        _call(np.zeros((2, 1, 2, 2)), np.zeros((3, 1, 1, 2)))


def test_an_unknown_mode_is_refused():
    with pytest.raises(Exception, match="mode"):
        _call(np.zeros((1, 1, 2, 2)), np.zeros((1, 1, 1, 2)), mode="cubic")


def test_an_unknown_padding_mode_is_refused():
    with pytest.raises(Exception, match="padding"):
        _call(np.zeros((1, 1, 2, 2)), np.zeros((1, 1, 1, 2)), padding_mode="wrap")


# --------------------------------------------------------------------------
# What it is for
# --------------------------------------------------------------------------


def test_a_learned_shift_finds_its_target():
    """The whole point: gradient descent on the *coordinates* moves the window
    to where the content is. This is a one-parameter spatial transformer."""
    # A smooth bump centred on sample 6 of nine, which under align_corners
    # sits at x = 2 * 6 / 8 - 1 = 0.5.
    image = np.zeros((1, 1, 1, 9))
    image[0, 0, 0, :] = np.exp(-0.5 * ((np.arange(9) - 6.0) / 2.0) ** 2)

    grid = np.zeros((1, 1, 1, 2))
    shift = -0.75  # start looking near the left edge
    for _ in range(400):
        grid[0, 0, 0, 0] = shift
        coords = _t(grid, requires_grad=True)
        read = mt.nn.grid_sample(_t(image), coords, align_corners=True)
        # Climb: descending on the negative of what is read.
        (-read.sum()).backward()
        shift -= 0.05 * float(coords.grad.numpy()[0, 0, 0, 0])
        shift = float(np.clip(shift, -1.0, 1.0))
    assert abs(shift - 0.5) < 0.12
