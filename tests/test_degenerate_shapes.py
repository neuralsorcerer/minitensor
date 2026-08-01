# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""Degenerate-shape behaviour for the operations added on this branch.

Empty axes, single elements and windows wider than their input are where an
indexing kernel panics rather than returning an error, and a Rust panic crossing
the binding is far worse for a caller than an exception: it carries no useful
message and can poison the interpreter state. Each case below must therefore
either produce a sensible result or raise -- never abort.

This file exists to keep an audit from becoming a one-off. Its first pass found
nothing; a later sweep over every empty-axis position found eight operations
that panicked -- `sum`, `mean`, `nansum`, `nanmean` reducing the last axis of a
2-D tensor whose last axis is empty, and `sort`/`argsort` on any empty input,
which between them took `std`, `var`, `logsumexp`, `trace`, `layer_norm` and
`rms_norm` down as well. The cases below pin the results against NumPy.
"""

import contextlib
import warnings

import numpy as np
import pytest

import minitensor as mt
from minitensor import functional as F
from minitensor import nn


@contextlib.contextmanager
def numpy_reference():
    """Silence NumPy while it computes an expected value.

    Reducing an empty axis makes NumPy warn about an empty slice and about
    dividing by zero. That warning *is* the reference behaviour being compared
    against, but the suite runs with `filterwarnings = error`, so an unguarded
    reference call fails the test before minitensor is ever exercised.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        yield


def f32(*shape):
    return mt.Tensor(np.zeros(shape, dtype=np.float32))


def f64(*shape):
    return mt.Tensor(np.zeros(shape), dtype="float64")


@pytest.mark.parametrize(
    "name, call",
    [
        ("max_pool1d", lambda: F.max_pool1d(f64(1, 1, 3), 5)),
        ("avg_pool1d", lambda: F.avg_pool1d(f64(1, 1, 3), 5)),
        ("max_pool2d", lambda: F.max_pool2d(f64(1, 1, 2, 2), 5)),
        ("avg_pool2d", lambda: F.avg_pool2d(f64(1, 1, 2, 2), 5)),
        ("conv1d", lambda: F.conv1d(f32(1, 1, 3), f32(1, 1, 5))),
        ("conv2d", lambda: F.conv2d(f32(1, 1, 2, 2), f32(1, 1, 5, 5))),
    ],
)
def test_a_window_wider_than_its_input_raises(name, call):
    # There is no valid output position, so this must be rejected rather than
    # producing a zero-length axis or reading out of bounds.
    with pytest.raises(Exception) as excinfo:
        call()
    assert "larger than" in str(excinfo.value) or "cannot be larger" in str(
        excinfo.value
    )


def test_conv1d_with_a_kernel_exactly_the_input_length_gives_one_position():
    result = F.conv1d(f32(1, 1, 5), f32(1, 1, 5))
    assert result.shape == (1, 1, 1)


def test_convolution_over_an_empty_batch_keeps_the_empty_axis():
    # No rows to convolve, but the channel and length arithmetic still applies.
    assert F.conv1d(f32(0, 1, 4), f32(1, 1, 2)).shape == (0, 1, 3)


def test_norm_over_an_empty_axis_is_zero():
    # An empty sum accumulates nothing, so every order gives zero.
    np.testing.assert_array_equal(f64(0, 3).norm(2.0, 0).numpy(), np.zeros(3))


def test_norm_of_a_single_element_is_its_magnitude():
    assert mt.Tensor([-3.0], dtype="float64").norm(2.0).item() == pytest.approx(3.0)


def test_scatter_with_an_empty_index_leaves_the_input_alone():
    base = mt.Tensor(np.arange(6, dtype=np.float64).reshape(2, 3), dtype="float64")
    empty_index = mt.Tensor(np.zeros((2, 0), dtype=np.int64), dtype="int64")
    result = base.scatter(1, empty_index, f64(2, 0))
    np.testing.assert_array_equal(result.numpy(), base.numpy())


@pytest.mark.parametrize("reduction, expected", [("sum", 0.0), ("none", None)])
def test_bce_with_logits_over_an_empty_input(reduction, expected):
    empty = f32(0, 3)
    result = F.binary_cross_entropy_with_logits(
        empty, empty, reduction=reduction
    ).numpy()
    if expected is None:
        assert result.shape == (0, 3)
    else:
        assert float(result) == pytest.approx(expected)


def test_bce_with_logits_mean_over_an_empty_input_is_nan_like_every_other_mean():
    # 0/0. This matches `mse_loss`, `binary_cross_entropy`, a plain `mean()` and
    # NumPy, so it is the consistent answer rather than a special case.
    empty = f32(0, 3)
    assert np.isnan(F.binary_cross_entropy_with_logits(empty, empty).numpy())
    assert np.isnan(F.mse_loss(empty, empty).numpy())
    assert np.isnan(empty.mean().numpy())


@pytest.mark.parametrize("kind", ["LSTM", "GRU"])
@pytest.mark.parametrize("bidirectional", [False, True])
def test_recurrent_layers_accept_a_single_timestep(kind, bidirectional):
    layer = getattr(nn, kind)(2, 3, bidirectional=bidirectional, dtype="float64")
    width = 3 * (2 if bidirectional else 1)
    assert layer(f64(1, 1, 2)).shape == (1, 1, width)


@pytest.mark.parametrize("kind", ["LSTM", "GRU"])
def test_recurrent_layers_accept_width_one_everywhere(kind):
    layer = getattr(nn, kind)(1, 1, dtype="float64")
    assert layer(f64(3, 1, 1)).shape == (3, 1, 1)


@pytest.mark.parametrize("kind", ["LSTM", "GRU"])
def test_recurrent_layers_reject_an_empty_sequence(kind):
    # Zero timesteps has no meaningful final state, so it is rejected rather
    # than silently returning the initial one.
    layer = getattr(nn, kind)(2, 3, dtype="float64")
    with pytest.raises(Exception):
        layer(f64(0, 1, 2))


# The reduction kernels special-case rank 1 and rank 2 and fall back to a
# generic loop above that, so an empty axis has to be tried in every position:
# only the 2-D "last axis is the empty one" combination chunked the input by a
# zero-length row and panicked. Rank 3 was always fine, which is why nothing
# caught this earlier.
EMPTY_SHAPES = [(0,), (0, 0), (3, 0), (0, 3), (2, 0, 3), (2, 3, 0), (0, 2, 3)]


@pytest.mark.parametrize("shape", EMPTY_SHAPES)
@pytest.mark.parametrize("name", ["sum", "mean", "nansum", "nanmean"])
def test_reducing_an_empty_axis_matches_numpy(shape, name):
    array = np.zeros(shape, dtype=np.float32)
    tensor = mt.from_numpy(array)
    for dim in range(len(shape)):
        with numpy_reference():
            expected = getattr(np, name)(array, axis=dim)
        got = getattr(tensor, name)(dim, False).numpy()
        assert got.shape == expected.shape
        np.testing.assert_array_equal(got, expected)


@pytest.mark.parametrize("shape", EMPTY_SHAPES)
@pytest.mark.parametrize("name", ["std", "var"])
def test_dispersion_over_an_empty_axis_is_nan_like_numpy(shape, name):
    # 0/0 for the biased estimator, exactly as NumPy reports it.
    array = np.zeros(shape, dtype=np.float32)
    tensor = mt.from_numpy(array)
    for dim in range(len(shape)):
        with numpy_reference():
            expected = getattr(np, name)(array, axis=dim)
        got = getattr(tensor, name)(dim, False, False).numpy()
        assert got.shape == expected.shape
        np.testing.assert_array_equal(got, expected)


@pytest.mark.parametrize("shape", EMPTY_SHAPES)
def test_sort_and_argsort_return_the_empty_input(shape):
    # NumPy and PyTorch both hand the empty input straight back rather than
    # erroring, so an empty batch flows through a pipeline unchanged.
    tensor = mt.from_numpy(np.zeros(shape, dtype=np.float32))
    values, indices = tensor.sort()
    assert values.shape == shape
    assert indices.shape == shape
    assert tensor.argsort().shape == shape


@pytest.mark.parametrize("shape", [(3, 0), (0, 0), (0,), (0, 3)])
def test_logsumexp_over_an_empty_axis_is_negative_infinity(shape):
    # log(sum of nothing) = log(0). Finite only where the output itself is empty.
    tensor = mt.from_numpy(np.zeros(shape, dtype=np.float32))
    dim = len(shape) - 1
    got = tensor.logsumexp([dim], False).numpy()
    assert got.shape == np.sum(np.zeros(shape, np.float32), axis=dim).shape
    assert np.all(np.isneginf(got)) or got.size == 0


@pytest.mark.parametrize("shape", [(2, 0, 3), (2, 3, 0), (0, 1, 0)])
def test_trace_over_an_empty_matrix_axis(shape):
    # Summing an empty diagonal, which is zero rather than an error.
    tensor = mt.from_numpy(np.zeros(shape, dtype=np.float32))
    assert np.all(tensor.trace().numpy() == 0.0)


@pytest.mark.parametrize("shape", [(3, 0), (0, 0), (1, 0)])
@pytest.mark.parametrize("name", ["layer_norm", "rms_norm"])
def test_normalization_over_an_empty_feature_axis(shape, name):
    # Normalizing over zero features divides by an empty mean; the shape has to
    # survive regardless of what the values come out as.
    tensor = mt.from_numpy(np.zeros(shape, dtype=np.float32))
    assert getattr(mt, name)(tensor, [shape[-1]]).shape == shape
