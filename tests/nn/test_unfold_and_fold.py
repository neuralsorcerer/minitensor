# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""`unfold` lays the sliding blocks out as columns; `fold` sums them back.

The point of the pair is that a convolution is a matrix product once the
windows are columns, so `unfold` plus `matmul` has to give exactly what
`conv2d` gives -- that is the test that says the layout is right, and it is
also the reason to have the function at all: a caller who wants a convolution
variant the library does not ship can write it in two lines instead of a Rust
kernel.

`fold` is checked against `unfold` from the other side. It is the adjoint, so
the two inner products agree to within summation order, and -- the exact claim
-- `unfold`'s gradient *is* `fold`, bit for bit, because the backward of a
gather is a scatter-add over the positions the gather read. Neither function
knows about the other; they agree because the kernels underneath do.
"""

from __future__ import annotations

import numpy as np
import pytest

import minitensor as mt
from minitensor import functional as F

RNG = np.random.default_rng(23)

# (input shape, kernel, dilation, padding, stride)
GEOMETRIES = [
    ((2, 3, 5, 6), (2, 2), (1, 1), (0, 0), (1, 1)),
    ((1, 1, 5, 5), (3, 3), (1, 1), (1, 1), (1, 1)),
    ((2, 2, 7, 8), (3, 2), (2, 1), (2, 1), (2, 3)),
    ((1, 4, 4, 4), (4, 4), (1, 1), (0, 0), (1, 1)),
    ((3, 2, 9, 9), (3, 3), (2, 2), (2, 2), (3, 3)),
]


def _t(values, requires_grad=False):
    return mt.Tensor(
        np.ascontiguousarray(np.asarray(values, dtype=np.float64)),
        dtype="float64",
        requires_grad=requires_grad,
    )


def _im2col(array, kernel, dilation, padding, stride):
    """The definition, written out as slices rather than as an index map."""

    batch, channels = array.shape[:2]
    padded = np.pad(
        array,
        ((0, 0), (0, 0), (padding[0],) * 2, (padding[1],) * 2),
    )
    (kh, kw), (dh, dw), (sh, sw) = kernel, dilation, stride
    high = (padded.shape[2] - dh * (kh - 1) - 1) // sh + 1
    wide = (padded.shape[3] - dw * (kw - 1) - 1) // sw + 1
    columns = np.zeros((batch, channels, kh, kw, high * wide))
    for row in range(kh):
        for col in range(kw):
            window = padded[
                :,
                :,
                row * dh : row * dh + (high - 1) * sh + 1 : sh,
                col * dw : col * dw + (wide - 1) * sw + 1 : sw,
            ]
            columns[:, :, row, col, :] = window.reshape(batch, channels, -1)
    return columns.reshape(batch, channels * kh * kw, high * wide)


# --- unfold ----------------------------------------------------------------


@pytest.mark.parametrize("shape,kernel,dilation,padding,stride", GEOMETRIES)
def test_unfold_is_im2col(shape, kernel, dilation, padding, stride):
    array = RNG.normal(size=shape)
    np.testing.assert_array_equal(
        F.unfold(_t(array), kernel, dilation, padding, stride).numpy(),
        _im2col(array, kernel, dilation, padding, stride),
    )


def test_a_single_integer_means_the_same_along_every_axis():
    array = _t(RNG.normal(size=(2, 3, 6, 6)))
    np.testing.assert_array_equal(
        F.unfold(array, 3, 1, 1, 2).numpy(),
        F.unfold(array, (3, 3), (1, 1), (1, 1), (2, 2)).numpy(),
    )


def test_unfold_keeps_the_dtype_it_was_given():
    array = mt.Tensor(RNG.normal(size=(1, 2, 4, 4)), dtype="float32")
    assert "float32" in str(F.unfold(array, 2).dtype)


def test_a_convolution_is_unfold_and_a_matrix_product():
    """The claim the function exists for, checked against the kernel."""

    batch, in_channels, out_channels = 2, 3, 5
    image = _t(RNG.normal(size=(batch, in_channels, 9, 8)))
    weight = _t(RNG.normal(size=(out_channels, in_channels, 3, 3)))
    bias = _t(RNG.normal(size=out_channels))

    columns = F.unfold(image, 3, padding=1, stride=2)
    flat = F.matmul(weight.reshape(out_channels, -1), columns)
    built = (flat + bias.reshape(1, out_channels, 1)).reshape(batch, out_channels, 5, 4)

    np.testing.assert_allclose(
        built.numpy(),
        F.conv2d(image, weight, bias, stride=(2, 2), padding=(1, 1)).numpy(),
        rtol=1e-12,
        atol=1e-13,
    )


def test_a_gradient_reaches_the_weights_through_the_matrix_product():
    """An unfolded convolution has to train, not merely evaluate."""

    image = _t(RNG.normal(size=(2, 3, 6, 6)))
    weight = _t(RNG.normal(size=(4, 3, 3, 3)), requires_grad=True)
    columns = F.unfold(image, 3, padding=1)
    F.matmul(weight.reshape(4, -1), columns).sum().backward()
    assert weight.grad is not None
    assert np.isfinite(weight.grad.numpy()).all()
    mt.clear_autograd_graph()


# --- fold ------------------------------------------------------------------


@pytest.mark.parametrize("shape,kernel,dilation,padding,stride", GEOMETRIES)
def test_fold_is_the_adjoint_of_unfold(shape, kernel, dilation, padding, stride):
    values = _t(RNG.normal(size=shape))
    columns = F.unfold(values, kernel, dilation, padding, stride)
    cotangent = _t(RNG.normal(size=tuple(int(size) for size in columns.shape)))
    folded = F.fold(cotangent, shape[2:], kernel, dilation, padding, stride)

    # <unfold(x), y> == <x, fold(y)>, up to the order the two sums are taken in.
    np.testing.assert_allclose(
        float((columns * cotangent).sum().item()),
        float((values * folded).sum().item()),
        rtol=1e-12,
    )


@pytest.mark.parametrize("shape,kernel,dilation,padding,stride", GEOMETRIES)
def test_the_gradient_of_unfold_is_exactly_fold(
    shape, kernel, dilation, padding, stride
):
    values = _t(RNG.normal(size=shape), requires_grad=True)
    columns = F.unfold(values, kernel, dilation, padding, stride)
    cotangent = _t(RNG.normal(size=tuple(int(size) for size in columns.shape)))
    (columns * cotangent).sum().backward()

    # Bit-identical, not merely close: both are the same scatter-add.
    np.testing.assert_array_equal(
        values.grad.numpy(),
        F.fold(cotangent, shape[2:], kernel, dilation, padding, stride).numpy(),
    )
    mt.clear_autograd_graph()


def test_the_gradient_of_fold_is_exactly_unfold():
    # 5x5 padded to 7x7, a 2-tap kernel at stride 2: three blocks each way.
    columns = _t(RNG.normal(size=(2, 12, 9)), requires_grad=True)
    cotangent = _t(RNG.normal(size=(2, 3, 5, 5)))
    (F.fold(columns, (5, 5), 2, padding=1, stride=2) * cotangent).sum().backward()

    np.testing.assert_array_equal(
        columns.grad.numpy(),
        F.unfold(cotangent, 2, padding=1, stride=2).numpy(),
    )
    mt.clear_autograd_graph()


def test_overlapping_blocks_are_summed_and_not_overwritten():
    """Two blocks reading one position put both their values back into it."""

    ones = _t(np.ones((1, 4, 4)))  # 2x2 kernel, stride 1, over a 3x3 plane
    counts = F.fold(ones, (3, 3), 2).numpy()[0, 0]
    np.testing.assert_array_equal(
        counts,
        [[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]],
    )


def test_folding_ones_gives_the_divisor_that_turns_the_sum_into_a_mean():
    """The recipe the docstring gives for averaging instead of summing."""

    values = _t(RNG.normal(size=(1, 2, 5, 5)))
    columns = F.unfold(values, 3, padding=1)
    coverage = F.fold(mt.Tensor.ones_like(columns), (5, 5), 3, padding=1)
    averaged = F.fold(columns, (5, 5), 3, padding=1) / coverage

    # Every position is read by at least one block, so the mean of the copies
    # of a value is that value.
    np.testing.assert_allclose(averaged.numpy(), values.numpy(), rtol=1e-13)


def test_fold_reads_its_rank_from_whichever_argument_is_a_sequence():
    columns = _t(np.ones((1, 4, 4)))
    np.testing.assert_array_equal(
        F.fold(columns, (3, 3), 2).numpy(),
        F.fold(columns, [3, 3], (2, 2)).numpy(),
    )


# --- ranks beyond the two a 2-D convolution needs ---------------------------


def test_a_single_spatial_axis_works():
    values = _t(RNG.normal(size=(2, 3, 10)))
    columns = F.unfold(values, 3, stride=2)
    assert tuple(columns.shape) == (2, 9, 4)
    np.testing.assert_allclose(
        F.fold(F.unfold(values, 1), (10,), 1).numpy(), values.numpy(), rtol=0
    )


def test_three_spatial_axes_work_where_torch_stops_at_two():
    values = _t(RNG.normal(size=(1, 2, 4, 5, 6)))
    columns = F.unfold(values, (2, 2, 2))
    assert tuple(columns.shape) == (1, 2 * 8, 3 * 4 * 5)
    assert tuple(F.fold(columns, (4, 5, 6), (2, 2, 2)).shape) == (1, 2, 4, 5, 6)


def test_a_three_dimensional_convolution_is_the_same_matrix_product():
    """What the extra ranks buy: `conv3d` without a `conv3d`."""

    volume = _t(RNG.normal(size=(2, 2, 4, 5, 6)))
    weight = _t(RNG.normal(size=(3, 2, 2, 2, 2)))
    columns = F.unfold(volume, (2, 2, 2))
    built = F.matmul(weight.reshape(3, -1), columns).reshape(2, 3, 3, 4, 5)

    reference = np.zeros((2, 3, 3, 4, 5))
    array, taps = volume.numpy(), weight.numpy().reshape(3, -1).T
    for i in range(3):
        for j in range(4):
            for k in range(5):
                window = array[:, :, i : i + 2, j : j + 2, k : k + 2]
                reference[:, :, i, j, k] = window.reshape(2, -1) @ taps
    np.testing.assert_allclose(built.numpy(), reference, rtol=1e-12)


# --- what they refuse -------------------------------------------------------


@pytest.mark.parametrize("shape", [(0, 3, 4, 4), (2, 0, 4, 4)])
def test_an_empty_batch_or_channel_axis_gives_an_empty_answer(shape):
    """Not a reshape error: there is nothing to unfold, and that is a shape."""

    columns = F.unfold(_t(np.zeros(shape)), 2)
    assert tuple(columns.shape) == (shape[0], shape[1] * 4, 9)
    assert tuple(F.fold(columns, (4, 4), 2).shape) == shape


def test_unfold_needs_a_batch_a_channel_and_a_spatial_axis():
    with pytest.raises(ValueError, match="at least one spatial axis"):
        F.unfold(_t(np.zeros((3, 3))), 2)


def test_a_kernel_larger_than_the_padded_input_is_refused():
    with pytest.raises(ValueError, match="does not fit"):
        F.unfold(_t(np.zeros((1, 1, 3, 3))), 5)


def test_dilation_counts_towards_whether_the_kernel_fits():
    """A 2-tap kernel at dilation 4 reaches 5 positions, not 2."""

    with pytest.raises(ValueError, match=r"dilated extent of \(5, 5\)"):
        F.unfold(_t(np.zeros((1, 1, 4, 4))), 2, dilation=4)


@pytest.mark.parametrize(
    "argument,value", [("kernel_size", 0), ("stride", 0), ("dilation", 0)]
)
def test_the_geometry_arguments_that_must_be_positive(argument, value):
    with pytest.raises(ValueError, match=f"{argument} of at least 1"):
        F.unfold(_t(np.zeros((1, 1, 4, 4))), **{"kernel_size": 2, argument: value})


def test_padding_may_be_zero_but_not_negative():
    with pytest.raises(ValueError, match="padding of at least 0"):
        F.unfold(_t(np.zeros((1, 1, 4, 4))), 2, padding=-1)


def test_one_geometry_value_per_spatial_axis():
    with pytest.raises(ValueError, match="one kernel_size per spatial axis"):
        F.unfold(_t(np.zeros((1, 1, 4, 4))), (2, 2, 2))


def test_a_geometry_argument_has_to_be_integers():
    with pytest.raises(TypeError, match="integer or a sequence"):
        F.unfold(_t(np.zeros((1, 1, 4, 4))), 2.5)


def test_fold_needs_the_packed_channels_to_divide_by_the_taps():
    with pytest.raises(ValueError, match="divide by the number of kernel taps"):
        F.fold(_t(np.zeros((1, 5, 4))), (4, 4), 3)


def test_fold_checks_the_block_count_against_the_geometry():
    with pytest.raises(ValueError, match=r"expects 4 block\(s\)"):
        F.fold(_t(np.zeros((1, 9, 5))), (4, 4), 3)


def test_fold_wants_a_three_dimensional_input():
    with pytest.raises(ValueError, match="three-dimensional"):
        F.fold(_t(np.zeros((1, 9))), (4, 4), 3)
