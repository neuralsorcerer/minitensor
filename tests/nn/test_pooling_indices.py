# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""Where each maximum came from, and putting it back there.

`max_pool` had no way to report which element won its window, so `max_unpool`
could not exist -- there was nothing to feed it. The backward pass needs the
same position, to send the gradient to the element that won, so the two share
one kernel. What they do not share is a forward that wants neither: carrying an
index through a window turns a comparison the compiler can put in a vector into
a loop-carried pair it cannot, and it is most of what the operation costs, so
the position is found only when `return_indices` asks for it or a gradient is
going to scatter through it. The three paths have to agree exactly, NaN and
ties included, and that is what `test_every_path_finds_the_same_maxima` is for.
The plain call still returns a bare tensor, which is checked here, because
adding a second return value is the kind of change that breaks every existing
caller if it is done carelessly.

`max_unpool` is then an arrangement and needs no kernel: the planes are laid
end to end, each one's offsets shifted by where it starts -- arithmetic on
sizes, so NumPy computes the shifts -- and the values are scattered into zeros.

The inverse is partial in two ways, and both get a test. It cannot restore what
pooling discarded, which is the point rather than a limitation: an unpooling
decoder wants the shape and the locations back, not the values. And it cannot
know the input's size, because pooling loses the remainder -- a seven-wide input
and a six-wide one pool to the same three columns, which is what `output_size`
is for.
"""

from __future__ import annotations

import numpy as np
import pytest

import minitensor as mt
from minitensor import functional as F

RNG = np.random.default_rng(97)


def _t(values, requires_grad=False):
    return mt.Tensor(
        np.ascontiguousarray(np.asarray(values, dtype=np.float64)),
        dtype="float64",
        requires_grad=requires_grad,
    )


def _reference_indices(values, kernel, stride, padding):
    """Where each window's maximum sits, as a flat offset into the plane."""

    batch, channels, high, wide = values.shape
    filled = np.pad(
        values,
        ((0, 0), (0, 0), (padding[0],) * 2, (padding[1],) * 2),
        constant_values=-np.inf,
    )
    rows = (filled.shape[2] - kernel[0]) // stride[0] + 1
    columns = (filled.shape[3] - kernel[1]) // stride[1] + 1
    out = np.zeros((batch, channels, rows, columns), dtype=np.int64)
    for n, c, i, j in np.ndindex(batch, channels, rows, columns):
        window = filled[
            n,
            c,
            i * stride[0] : i * stride[0] + kernel[0],
            j * stride[1] : j * stride[1] + kernel[1],
        ]
        offset = np.unravel_index(np.argmax(window), window.shape)
        row = i * stride[0] + offset[0] - padding[0]
        column = j * stride[1] + offset[1] - padding[1]
        out[n, c, i, j] = row * wide + column
    return out


# --- return_indices ---------------------------------------------------------


def test_the_plain_call_still_returns_one_tensor():
    """Adding a second return value must not change what everyone already gets."""

    result = F.max_pool2d(_t(RNG.normal(size=(2, 3, 4, 4))), 2)
    assert isinstance(result, mt.Tensor)
    assert isinstance(F.max_pool1d(_t(RNG.normal(size=(2, 3, 8))), 2), mt.Tensor)


@pytest.mark.parametrize(
    "shape,kernel,stride,padding",
    [
        ((1, 1, 4, 4), (2, 2), (2, 2), (0, 0)),
        ((2, 3, 6, 6), (3, 3), (1, 1), (1, 1)),
        ((2, 2, 5, 7), (2, 3), (2, 1), (1, 1)),
    ],
)
def test_the_indices_are_where_the_maxima_are(shape, kernel, stride, padding):
    values = RNG.normal(size=shape)
    pooled, indices = F.max_pool2d(
        _t(values), kernel, stride, padding, return_indices=True
    )

    assert "int64" in str(indices.dtype)
    assert tuple(indices.shape) == tuple(pooled.shape)
    np.testing.assert_array_equal(
        indices.numpy(), _reference_indices(values, kernel, stride, padding)
    )

    # And reading the input at those positions gives the pooled values back.
    flat = values.reshape(shape[0], shape[1], -1)
    picked = np.take_along_axis(
        flat, indices.numpy().reshape(shape[0], shape[1], -1), axis=2
    )
    np.testing.assert_array_equal(picked.reshape(pooled.numpy().shape), pooled.numpy())


@pytest.mark.parametrize(
    "shape,kernel,stride,padding",
    [
        ((4, 3, 16, 16), (2, 2), (2, 2), (0, 0)),
        ((2, 2, 15, 15), (3, 3), (2, 2), (1, 1)),
        ((3, 4, 9, 11), (2, 2), (1, 1), (1, 1)),
        ((1, 1, 8, 8), (4, 4), (4, 4), (2, 2)),
        ((2, 3, 7, 7), (3, 3), (3, 3), (1, 1)),
    ],
)
def test_every_path_finds_the_same_maxima(shape, kernel, stride, padding):
    """Three ways in, and they must not be three answers.

    Only one of them tracks an index: a plain forward on a tensor that wants no
    gradient takes a loop that finds the maximum and nothing else, which is
    several times quicker and is the whole reason the split exists. It would be
    a poor trade if the quick path disagreed anywhere, and the place two
    maximum-finding loops disagree is never the ordinary case -- it is a NaN, or
    a tie, where which element "wins" is a convention rather than a fact. So the
    input is seeded with NaNs and with repeated values, and the comparison is
    for equality rather than closeness.
    """
    values = RNG.standard_normal(shape)
    values.flat[::11] = np.nan
    # Ties, so that "the first maximum wins" is actually under test.
    values.flat[3::23] = 2.5
    values.flat[4::23] = 2.5

    plain = F.max_pool2d(_t(values), kernel, stride, padding).numpy()
    reported, _ = F.max_pool2d(_t(values), kernel, stride, padding, return_indices=True)
    differentiable = F.max_pool2d(
        _t(values, requires_grad=True), kernel, stride, padding
    ).numpy()

    # NaN is not equal to itself, so compare a NaN-free stand-in alongside the
    # pattern of NaNs -- both have to match, which `allclose` alone would not
    # say.
    def shape_of(a):
        return np.isnan(a), np.where(np.isnan(a), 0.0, a)

    for other in (reported.numpy(), differentiable):
        assert np.array_equal(shape_of(plain)[0], shape_of(other)[0])
        assert np.array_equal(shape_of(plain)[1], shape_of(other)[1])


def test_the_gradient_still_reaches_the_element_that_won():
    # The quick path is only reachable when no gradient is wanted, so the
    # backward has to keep working exactly as it did -- one unit of gradient to
    # each window's maximum and nothing anywhere else.
    values = RNG.standard_normal((2, 3, 8, 8))
    tensor = _t(values, requires_grad=True)
    F.max_pool2d(tensor, (2, 2), (2, 2), (0, 0)).sum().backward()

    expected = np.zeros_like(values)
    for n, c, i, j in np.ndindex(2, 3, 4, 4):
        window = values[n, c, 2 * i : 2 * i + 2, 2 * j : 2 * j + 2]
        row, col = np.unravel_index(np.argmax(window), window.shape)
        expected[n, c, 2 * i + row, 2 * j + col] += 1.0
    assert np.array_equal(tensor.grad.numpy(), expected)


def test_the_one_dimensional_indices_are_positions_along_the_axis():
    values = RNG.normal(size=(2, 3, 9))
    pooled, indices = F.max_pool1d(_t(values), 3, return_indices=True)
    np.testing.assert_array_equal(
        indices.numpy(), values.reshape(2, 3, 3, 3).argmax(-1) + np.arange(0, 9, 3)
    )
    np.testing.assert_array_equal(
        np.take_along_axis(values, indices.numpy(), axis=2), pooled.numpy()
    )


def test_indices_never_name_a_padded_position():
    """A window with no real element would report -1; the padding rule forbids one."""

    for padding in (0, 1):
        _, indices = F.max_pool2d(
            _t(RNG.normal(size=(2, 2, 5, 5))), 2, 2, padding, return_indices=True
        )
        assert (indices.numpy() >= 0).all()
        assert (indices.numpy() < 25).all()
    with pytest.raises(ValueError, match="half the window"):
        F.max_pool2d(_t(np.zeros((1, 1, 4, 4))), 2, 2, 2, return_indices=True)


# --- max_unpool -------------------------------------------------------------


def test_unpooling_puts_the_maxima_back_and_nothing_else():
    values = np.arange(16.0).reshape(1, 1, 4, 4)
    pooled, indices = F.max_pool2d(_t(values), 2, return_indices=True)
    np.testing.assert_array_equal(
        F.max_unpool2d(pooled, indices, 2).numpy()[0, 0],
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 5.0, 0.0, 7.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 13.0, 0.0, 15.0],
        ],
    )


@pytest.mark.parametrize("shape,kernel", [((2, 3, 8, 8), 2), ((1, 2, 9, 6), 3)])
def test_pooling_an_unpooled_tensor_returns_it_where_it_was_not_negative(shape, kernel):
    """The half of the inverse that holds, stated exactly.

    Unpooling writes the maximum back and zeroes the rest of its window, so
    pooling again finds `max(value, 0)` -- the identity wherever the value was
    already non-negative, which after a rectifier is everywhere, and zero where
    it was not. Asserting the identity outright would be asserting that the
    activations happened to be positive.
    """

    pooled, indices = F.max_pool2d(
        _t(RNG.normal(size=shape)), kernel, return_indices=True
    )
    spread = F.max_unpool2d(pooled, indices, kernel)
    again, _ = F.max_pool2d(spread, kernel, return_indices=True)
    np.testing.assert_array_equal(again.numpy(), np.maximum(pooled.numpy(), 0.0))

    # After a rectifier there is nothing negative left, and it is the identity.
    rectified = F.relu(_t(RNG.normal(size=shape)))
    pooled, indices = F.max_pool2d(rectified, kernel, return_indices=True)
    restored, _ = F.max_pool2d(
        F.max_unpool2d(pooled, indices, kernel), kernel, return_indices=True
    )
    np.testing.assert_array_equal(restored.numpy(), pooled.numpy())


def test_everything_pooling_discarded_stays_discarded():
    values = RNG.normal(size=(2, 2, 6, 6)) + 10.0
    pooled, indices = F.max_pool2d(_t(values), 2, return_indices=True)
    spread = F.max_unpool2d(pooled, indices, 2).numpy()
    assert (spread != 0.0).sum() == pooled.numpy().size
    assert float(spread.sum()) == pytest.approx(float(pooled.numpy().sum()))


def test_output_size_says_which_input_it_came_from():
    """Seven columns and six pool to the same three, so the caller must say."""

    seven = RNG.normal(size=(1, 1, 1, 7))
    pooled, indices = F.max_pool2d(_t(seven), (1, 2), return_indices=True)
    assert tuple(pooled.shape) == (1, 1, 1, 3)

    assert tuple(F.max_unpool2d(pooled, indices, (1, 2)).shape) == (1, 1, 1, 6)
    assert tuple(F.max_unpool2d(pooled, indices, (1, 2), output_size=(1, 7)).shape) == (
        1,
        1,
        1,
        7,
    )


def test_unpooling_a_signal_works_the_same_way():
    signal = np.arange(8.0).reshape(1, 1, 8)
    pooled, indices = F.max_pool1d(_t(signal), 2, return_indices=True)
    np.testing.assert_array_equal(
        F.max_unpool1d(pooled, indices, 2).numpy()[0, 0],
        [0.0, 1.0, 0.0, 3.0, 0.0, 5.0, 0.0, 7.0],
    )


def test_the_gradient_gathers_back_to_the_values_it_scattered():
    pooled = _t(RNG.normal(size=(1, 2, 2, 2)), requires_grad=True)
    indices = mt.Tensor.from_numpy(
        np.array([[[[0, 3], [12, 15]], [[5, 6], [9, 10]]]], dtype=np.int64)
    )
    weights = RNG.normal(size=(1, 2, 4, 4))
    (F.max_unpool2d(pooled, indices, 2) * _t(weights)).sum().backward()

    flat = weights.reshape(1, 2, -1)
    expected = np.take_along_axis(flat, indices.numpy().reshape(1, 2, -1), axis=2)
    np.testing.assert_allclose(
        pooled.grad.numpy(), expected.reshape(1, 2, 2, 2), rtol=0
    )
    mt.clear_autograd_graph()


def test_an_unpooled_decoder_trains_end_to_end():
    """What the pair is for: a gradient reaching the encoder through both."""

    image = _t(RNG.normal(size=(1, 1, 6, 6)), requires_grad=True)
    pooled, indices = F.max_pool2d(image, 2, return_indices=True)
    F.max_unpool2d(pooled * 2.0, indices, 2).sum().backward()
    # Only the nine winners take a gradient, and each takes two.
    gradient = image.grad.numpy()
    assert (gradient != 0).sum() == 9
    assert set(np.unique(gradient).tolist()) == {0.0, 2.0}
    mt.clear_autograd_graph()


# --- what it refuses --------------------------------------------------------


def test_one_index_per_value_is_required():
    with pytest.raises(ValueError, match="one index per value"):
        F.max_unpool2d(
            _t(np.zeros((1, 1, 2, 2))),
            mt.Tensor.from_numpy(np.zeros((1, 1, 3, 3), dtype=np.int64)),
            2,
        )


def test_float_indices_are_refused():
    with pytest.raises(TypeError, match="integer indices"):
        F.max_unpool2d(_t(np.zeros((1, 1, 2, 2))), _t(np.zeros((1, 1, 2, 2))), 2)


def test_an_index_outside_the_plane_is_refused():
    with pytest.raises(IndexError, match="which holds 4"):
        F.max_unpool2d(
            _t(np.zeros((1, 1, 1, 1))),
            mt.Tensor.from_numpy(np.array([[[[99]]]], dtype=np.int64)),
            2,
        )


@pytest.mark.parametrize("name,rank", [("max_unpool1d", 3), ("max_unpool2d", 4)])
def test_the_rank_each_one_takes_is_stated(name, rank):
    with pytest.raises(ValueError, match=f"{rank}-dimensional input"):
        getattr(F, name)(
            _t(np.zeros((2, 3, 4, 5, 6))),
            mt.Tensor.from_numpy(np.zeros((2, 3, 4, 5, 6), dtype=np.int64)),
            2,
        )


def test_an_output_size_of_the_wrong_rank_is_refused():
    with pytest.raises(ValueError, match="one entry per spatial axis"):
        F.max_unpool2d(
            _t(np.zeros((1, 1, 2, 2))),
            mt.Tensor.from_numpy(np.zeros((1, 1, 2, 2), dtype=np.int64)),
            2,
            output_size=(4, 4, 4),
        )
