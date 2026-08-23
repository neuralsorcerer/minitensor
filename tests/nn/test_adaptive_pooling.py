# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""A pooling layer could not be told what size to produce, only what window to use.

`max_pool2d` and `avg_pool2d` take a kernel and a stride, so hitting a chosen
output extent means the caller computing the kernel from the input extent -- and
when the extent does not divide evenly there is no kernel that works at all,
because a fixed window cannot produce unequal groups. That is the whole reason
every modern convolutional network ends in an adaptive pool: it is what lets the
classifier take any input size and still hand the linear layer the shape it
expects. Without it, a network is pinned to one input resolution.

The window comes from the ratio of the two extents:

    start(i) = floor(i * in / out)      end(i) = ceil((i + 1) * in / out)

which is exactly PyTorch's rule, and the tests pin it rather than accepting any
plausible neighbour. Two properties follow and are checked below. When `out`
divides `in` it degenerates to a regular pool with kernel and stride both
`in / out` -- checked against `avg_pool2d` for bit equality, not approximation.
And when `out` exceeds `in` the windows shrink to single elements and to
overlapping pairs, so it upsamples by averaging neighbours rather than by
repeating them -- that is not a special case in the code and should not be one
in the tests.

Windows overlap whenever the ratio is not an integer, which is where the
gradient gets interesting: an input position inside two windows collects from
both, with each window's own divisor. A backward that assumed a uniform window
would be wrong by a little almost everywhere and by a lot at the edges, so the
overlap is spelled out on a case small enough to check by hand.
"""

from __future__ import annotations

import numpy as np
import pytest

import minitensor as mt


def _reference(x, output_size, reduce):
    """The window rule, written out."""
    batch, channels, height, width = x.shape
    out_h, out_w = output_size
    out = np.zeros((batch, channels, out_h, out_w))
    for i in range(out_h):
        row_start, row_end = (i * height) // out_h, -(-((i + 1) * height) // out_h)
        for j in range(out_w):
            col_start, col_end = (j * width) // out_w, -(-((j + 1) * width) // out_w)
            out[:, :, i, j] = reduce(
                x[:, :, row_start:row_end, col_start:col_end], axis=(2, 3)
            )
    return out


def _numeric_grad(f, arr, eps=1e-6):
    grad = np.zeros_like(arr)
    flat, gflat = arr.reshape(-1), grad.reshape(-1)
    for i in range(flat.size):
        old = flat[i]
        flat[i] = old + eps
        high = f()
        flat[i] = old - eps
        low = f()
        flat[i] = old
        gflat[i] = (high - low) / (2 * eps)
    return grad


# Divisible, indivisible, one axis of each, down to a point, and up.
CASES = [
    ((2, 3, 8, 8), (4, 4)),
    ((2, 3, 7, 7), (3, 3)),
    ((1, 2, 5, 9), (2, 4)),
    ((2, 4, 6, 6), (1, 1)),
    ((2, 2, 10, 7), (3, 3)),
    ((1, 1, 3, 3), (5, 5)),
    ((1, 2, 9, 4), (9, 4)),
]

OPS = [
    (mt.nn.adaptive_avg_pool2d, np.mean, "avg"),
    (mt.nn.adaptive_max_pool2d, np.max, "max"),
]


@pytest.mark.parametrize("shape,output_size", CASES)
@pytest.mark.parametrize("op,reduce,name", OPS)
def test_it_matches_the_window_rule(shape, output_size, op, reduce, name):
    values = np.random.default_rng(hash((shape, output_size)) % 1000).standard_normal(shape)
    got = op(mt.Tensor(values, dtype="float64"), output_size).numpy()
    assert got.shape == (shape[0], shape[1], *output_size)
    np.testing.assert_allclose(
        got, _reference(values, output_size, reduce), rtol=1e-12, atol=1e-14
    )


def test_a_divisible_ratio_is_exactly_the_fixed_window_pool():
    """Not approximately: the window rule has to reduce to kernel = stride =
    in / out, and any nearby formula would round differently somewhere."""
    values = np.random.default_rng(3).standard_normal((2, 3, 12, 8))
    tensor = mt.Tensor(values, dtype="float64")
    np.testing.assert_array_equal(
        mt.nn.adaptive_avg_pool2d(tensor, (6, 4)).numpy(),
        mt.nn.avg_pool2d(tensor, 2).numpy(),
    )
    np.testing.assert_array_equal(
        mt.nn.adaptive_max_pool2d(tensor, (3, 2)).numpy(),
        mt.nn.max_pool2d(tensor, 4).numpy(),
    )


def test_a_global_average_pool_is_the_mean_over_the_spatial_axes():
    """The `(1, 1)` case, which is what ends most convolutional networks."""
    values = np.random.default_rng(5).standard_normal((3, 5, 7, 11))
    got = mt.nn.adaptive_avg_pool2d(mt.Tensor(values, dtype="float64"), 1).numpy()
    assert got.shape == (3, 5, 1, 1)
    np.testing.assert_allclose(got[:, :, 0, 0], values.mean(axis=(2, 3)), rtol=1e-12)


def test_the_output_size_takes_an_int_or_a_pair():
    values = mt.Tensor(np.zeros((1, 2, 6, 9)), dtype="float64")
    assert mt.nn.adaptive_avg_pool2d(values, 3).numpy().shape == (1, 2, 3, 3)
    assert mt.nn.adaptive_avg_pool2d(values, (2, 3)).numpy().shape == (1, 2, 2, 3)


def test_the_window_boundaries_are_the_documented_ones():
    """Pinned directly rather than through an average, on a case where several
    plausible rules disagree: 7 into 3 gives windows [0,3), [2,5), [4,7)."""
    values = np.arange(7, dtype=np.float64).reshape(1, 1, 7, 1)
    got = mt.nn.adaptive_avg_pool2d(mt.Tensor(values, dtype="float64"), (3, 1))
    np.testing.assert_allclose(
        got.numpy().ravel(),
        [np.mean([0, 1, 2]), np.mean([2, 3, 4]), np.mean([4, 5, 6])],
    )


def test_asking_for_more_than_there_is_still_follows_the_same_rule():
    """No special case for upsampling: three into five gives windows
    [0,1) [0,2) [1,2) [1,3) [2,3), so the even positions are single elements and
    the odd ones straddle a pair. It is worth writing out because the tempting
    guess -- that it repeats each source -- would give `[1, 2, 2, 3, 3]`, and
    that is a different operation (nearest-neighbour resampling) with a
    different gradient."""
    values = np.array([1.0, 2.0, 3.0]).reshape(1, 1, 3, 1)
    got = mt.nn.adaptive_avg_pool2d(mt.Tensor(values, dtype="float64"), (5, 1))
    np.testing.assert_allclose(got.numpy().ravel(), [1.0, 1.5, 2.0, 2.5, 3.0])


def test_a_nan_in_the_window_wins_the_max():
    """As it does in `max_pool2d` and in `max` itself -- the maximum of a set
    containing one is not a number, and skipping it would let a NaN vanish."""
    values = np.array([[[[1.0, np.nan], [3.0, 4.0]]]])
    got = mt.nn.adaptive_max_pool2d(mt.Tensor(values, dtype="float64"), (1, 1))
    assert np.isnan(got.numpy()).all()


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("op,reduce,name", OPS)
def test_both_float_dtypes_are_supported(dtype, op, reduce, name):
    values = np.random.default_rng(7).standard_normal((2, 3, 7, 5)).astype(dtype)
    got = op(mt.Tensor(values, dtype=dtype), (3, 2))
    assert got.dtype == dtype
    tolerance = 1e-6 if dtype == "float32" else 1e-13
    np.testing.assert_allclose(
        got.numpy().astype(np.float64),
        _reference(values.astype(np.float64), (3, 2), reduce),
        rtol=tolerance, atol=tolerance,
    )


@pytest.mark.parametrize("op,reduce,name", OPS)
def test_an_integer_input_is_refused(op, reduce, name):
    with pytest.raises(Exception):
        op(mt.Tensor(np.zeros((1, 1, 4, 4), dtype=np.int64), dtype="int64"), (2, 2))


@pytest.mark.parametrize("op,reduce,name", OPS)
def test_a_wrongly_ranked_input_is_refused(op, reduce, name):
    with pytest.raises(Exception):
        op(mt.Tensor(np.zeros((1, 4, 4)), dtype="float64"), (2, 2))


def test_pooling_an_empty_axis_into_a_non_empty_one_is_refused():
    """There is nothing in the window, so an average would be zero over zero.
    Asking for no output along that axis is a different question and is fine."""
    empty = mt.Tensor(np.zeros((1, 2, 0, 4)), dtype="float64")
    with pytest.raises(Exception):
        mt.nn.adaptive_avg_pool2d(empty, (1, 1))
    assert mt.nn.adaptive_avg_pool2d(empty, (0, 2)).numpy().shape == (1, 2, 0, 2)


# --- gradients ---------------------------------------------------------------


@pytest.mark.parametrize("shape,output_size", CASES[:5] + [CASES[5]])
@pytest.mark.parametrize("op,reduce,name", OPS)
def test_the_gradient_matches_numerical_differentiation(shape, output_size, op, reduce, name):
    rng = np.random.default_rng(11)
    values = rng.standard_normal(shape)
    probe = rng.standard_normal((shape[0], shape[1], *output_size))

    def loss():
        return float((op(mt.Tensor(values, dtype="float64"), output_size).numpy() * probe).sum())

    t = mt.Tensor(values.copy(), dtype="float64", requires_grad=True)
    (op(t, output_size) * mt.Tensor(probe, dtype="float64")).sum().backward()
    np.testing.assert_allclose(
        t.grad.numpy(), _numeric_grad(loss, values), rtol=1e-5, atol=1e-7
    )


def test_an_overlapping_window_collects_from_every_window_it_is_in():
    """Spelled out rather than left to the numerical check. Three rows into two
    gives windows [0, 2) and [1, 3), so the middle row is in both and receives
    half from each while the ends receive half from one."""
    t = mt.Tensor(np.array([1.0, 2.0, 3.0]).reshape(1, 1, 3, 1),
                  dtype="float64", requires_grad=True)
    mt.nn.adaptive_avg_pool2d(t, (2, 1)).sum().backward()
    np.testing.assert_allclose(t.grad.numpy().ravel(), [0.5, 1.0, 0.5])


def test_an_upsampling_gradient_counts_every_window_a_row_lands_in():
    """Three rows into five. Row 0 is in windows [0,1) and [0,2), so it collects
    `1/1 + 1/2`; row 1 is in three windows and collects `1/2 + 1/1 + 1/2`. Each
    window brings its own divisor, which is what a uniform-window backward
    cannot express."""
    t = mt.Tensor(np.array([1.0, 2.0, 3.0]).reshape(1, 1, 3, 1),
                  dtype="float64", requires_grad=True)
    mt.nn.adaptive_avg_pool2d(t, (5, 1)).sum().backward()
    np.testing.assert_allclose(t.grad.numpy().ravel(), [1.5, 2.0, 1.5])


def test_the_max_gradient_reaches_only_the_winner():
    values = np.array([[[[1.0, 5.0], [3.0, 2.0]]]])
    t = mt.Tensor(values, dtype="float64", requires_grad=True)
    mt.nn.adaptive_max_pool2d(t, (1, 1)).sum().backward()
    np.testing.assert_array_equal(t.grad.numpy(), [[[[0.0, 1.0], [0.0, 0.0]]]])


# --- one dimension -----------------------------------------------------------


@pytest.mark.parametrize("length,output_size", [(10, 4), (7, 7), (9, 2), (3, 6)])
@pytest.mark.parametrize(
    "op,reduce", [(mt.nn.adaptive_avg_pool1d, np.mean), (mt.nn.adaptive_max_pool1d, np.max)]
)
def test_one_dimensional_agrees_with_the_two_dimensional(length, output_size, op, reduce):
    values = np.random.default_rng(13).standard_normal((2, 3, length))
    got = op(mt.Tensor(values, dtype="float64"), output_size).numpy()
    want = _reference(values[:, :, None, :], (1, output_size), reduce)[:, :, 0, :]
    np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-14)


def test_one_dimensional_carries_a_gradient():
    values = np.random.default_rng(17).standard_normal((1, 2, 7))
    probe = np.random.default_rng(19).standard_normal((1, 2, 3))

    def loss():
        return float(
            (mt.nn.adaptive_avg_pool1d(mt.Tensor(values, dtype="float64"), 3).numpy() * probe).sum()
        )

    t = mt.Tensor(values.copy(), dtype="float64", requires_grad=True)
    (mt.nn.adaptive_avg_pool1d(t, 3) * mt.Tensor(probe, dtype="float64")).sum().backward()
    np.testing.assert_allclose(t.grad.numpy(), _numeric_grad(loss, values), rtol=1e-5, atol=1e-7)


def test_a_wrongly_ranked_one_dimensional_input_is_refused():
    with pytest.raises(Exception):
        mt.nn.adaptive_avg_pool1d(mt.Tensor(np.zeros((2, 3, 4, 5)), dtype="float64"), 2)


# --- the layers --------------------------------------------------------------


@pytest.mark.parametrize(
    "cls,size,rank",
    [
        (mt.nn.AdaptiveAvgPool2d, (2, 3), 4),
        (mt.nn.AdaptiveMaxPool2d, (1, 1), 4),
        (mt.nn.AdaptiveAvgPool1d, 4, 3),
        (mt.nn.AdaptiveMaxPool1d, 2, 3),
    ],
)
def test_the_layers_report_themselves_and_hold_no_parameters(cls, size, rank):
    layer = cls(size)
    assert layer.output_size == size
    assert cls.__name__ in repr(layer)
    assert layer.parameters() == [], "a pooling layer has nothing to learn"
    shape = (2, 3, 6, 9) if rank == 4 else (2, 3, 9)
    expected = (2, 3, *size) if rank == 4 else (2, 3, size)
    assert layer(mt.Tensor(np.zeros(shape), dtype="float64")).numpy().shape == expected


def test_the_layer_agrees_with_the_functional_form():
    values = np.random.default_rng(23).standard_normal((2, 3, 7, 5))
    tensor = mt.Tensor(values, dtype="float64")
    np.testing.assert_array_equal(
        mt.nn.AdaptiveAvgPool2d((3, 2))(tensor).numpy(),
        mt.nn.adaptive_avg_pool2d(tensor, (3, 2)).numpy(),
    )


def test_the_layer_defaults_to_a_global_pool():
    layer = mt.nn.AdaptiveAvgPool2d()
    assert layer.output_size == (1, 1)
    assert layer(mt.Tensor(np.zeros((1, 4, 5, 6)), dtype="float64")).numpy().shape == (1, 4, 1, 1)


def test_a_classifier_head_now_takes_any_input_size():
    """The thing the gap actually blocked: the same network, several input
    resolutions, one output shape -- and a gradient that still reaches the
    convolution underneath it."""
    head = mt.nn.Sequential([
        mt.nn.Conv2d(3, 16, 3, padding=1, dtype="float64"),
        mt.nn.ReLU(),
        mt.nn.AdaptiveAvgPool2d((1, 1)),
    ])
    for height, width in [(8, 8), (17, 23), (33, 9)]:
        batch = mt.Tensor(
            np.random.default_rng(height).standard_normal((2, 3, height, width)),
            dtype="float64",
        )
        assert head(batch).numpy().shape == (2, 16, 1, 1)

    parameters = head.parameters()
    optimizer = mt.optim.Adam(parameters, lr=0.05)
    batch = mt.Tensor(np.random.default_rng(29).standard_normal((2, 3, 11, 13)), dtype="float64")
    target = mt.Tensor(np.random.default_rng(31).standard_normal((2, 16, 1, 1)), dtype="float64")
    first = None
    for _ in range(30):
        optimizer.zero_grad()
        difference = head(batch) - target
        loss = (difference * difference).mean()
        if first is None:
            first = loss.item()
        loss.backward()
        optimizer.step()
    assert loss.item() < first * 0.7
    assert all(p.grad is not None for p in parameters)
