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
    # so it is the consistent answer rather than a special case.
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
    # 0/0 for the biased estimator.
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
    # The empty input is handed straight back rather than erroring, so an
    # empty batch flows through a pipeline unchanged.
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


# `split` built its section list with `while remaining > 0`, so a zero-length
# axis produced no sections at all and the call returned an empty list. Every
# neighbouring operation disagreed: `chunk` and `split_with_sections` both
# returned the single empty piece, as does `np.split`. The visible cost was the
# round trip -- `cat(t.split(n, d), d)` is how a caller reassembles what it
# split, and on an empty batch it failed with "cannot concatenate empty list of
# tensors" rather than rebuilding the empty tensor.

EMPTY_AXIS_CASES = [
    ((0,), 0),
    ((0, 3), 0),
    ((2, 0), 1),
    ((0, 3, 4), 0),
    ((2, 0, 4), 1),
    ((2, 3, 0), 2),
]


@pytest.mark.parametrize("shape,dim", EMPTY_AXIS_CASES)
@pytest.mark.parametrize("size", [1, 2, 5])
def test_splitting_an_empty_axis_yields_one_empty_piece(shape, dim, size):
    tensor = mt.from_numpy(np.zeros(shape, dtype=np.float32))
    pieces = tensor.split(size, dim)

    assert len(pieces) == len(np.split(np.zeros(shape, np.float32), [], axis=dim))
    assert [tuple(p.shape_vec()) for p in pieces] == [shape]


@pytest.mark.parametrize("shape,dim", EMPTY_AXIS_CASES)
def test_split_and_cat_round_trip_over_an_empty_axis(shape, dim):
    """The property the pair exists for, and the one the empty list broke."""
    tensor = mt.from_numpy(np.zeros(shape, dtype=np.float32))
    assert tuple(mt.cat(list(tensor.split(2, dim)), dim).shape_vec()) == shape


@pytest.mark.parametrize("shape,dim", EMPTY_AXIS_CASES)
def test_split_agrees_with_chunk_and_explicit_sections(shape, dim):
    """All three ways of cutting an axis, on the same empty tensor."""
    tensor = mt.from_numpy(np.zeros(shape, dtype=np.float32))
    counts = {
        "split": len(tensor.split(1, dim)),
        "chunk": len(tensor.chunk(1, dim)),
        "sections": len(tensor.split_with_sections([0], dim)),
    }
    assert set(counts.values()) == {1}, counts


@pytest.mark.parametrize("dim", [0, 1])
@pytest.mark.parametrize("size", [1, 2, 3, 4, 5, 6, 7])
def test_a_non_empty_axis_is_unchanged(dim, size):
    """The remainder path is the one that was already right."""
    array = np.arange(24, dtype=np.float32).reshape(4, 6)
    tensor = mt.from_numpy(array)

    got = [piece.numpy() for piece in tensor.split(size, dim)]
    expected = np.split(array, list(range(size, array.shape[dim], size)), axis=dim)

    assert len(got) == len(expected)
    for a, b in zip(got, expected):
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("shape,dim", EMPTY_AXIS_CASES)
def test_the_functional_forms_agree(shape, dim):
    tensor = mt.from_numpy(np.zeros(shape, dtype=np.float32))
    assert (
        len(tensor.split(1, dim))
        == len(F.split(tensor, 1, dim))
        == len(mt.split(tensor, 1, dim))
    )


# `softmax` and `log_softmax` panicked -- a Rust `chunk_size must not be zero`
# crossing the binding -- whenever an axis *after* the reduced one was empty.
# Their block size is the reduced axis times everything after it, so a trailing
# zero made it zero, and both `chunks` and `par_chunks` panic on that rather
# than yielding nothing. Two sites had it: the forward's shared geometry helper,
# which already special-cased an empty reduced axis but not an empty trailing
# one, and the backward's shared block-walker.
#
# The condition is narrow, which is why an audit that swept empty axes over the
# reductions missed it: a zero *before* the reduced axis is harmless, because
# the input slice is then empty and yields no chunks at all. Only
# `softmax((3, 0), dim=0)` and its relatives reached it.

SOFTMAX_EMPTY_SHAPES = [
    (0,),
    (0, 0),
    (3, 0),
    (0, 3),
    (1, 0),
    (0, 1),
    (2, 0, 3),
    (2, 3, 0),
    (0, 2, 3),
    (3, 0, 0),
    (2, 0, 0, 3),
]


def _every_dim(shape):
    return range(-len(shape), len(shape))


@pytest.mark.parametrize("name", ["softmax", "log_softmax"])
@pytest.mark.parametrize("shape", SOFTMAX_EMPTY_SHAPES, ids=str)
def test_softmax_over_an_empty_axis_returns_an_empty_tensor(name, shape):
    tensor = mt.from_numpy(np.zeros(shape, dtype=np.float32))
    for dim in _every_dim(shape):
        result = getattr(tensor, name)(dim)
        assert tuple(result.shape_vec()) == shape, f"dim={dim}"
        assert result.numel() == 0


@pytest.mark.parametrize("name", ["softmax", "log_softmax"])
@pytest.mark.parametrize("shape", SOFTMAX_EMPTY_SHAPES, ids=str)
def test_softmax_backward_over_an_empty_axis_survives(name, shape):
    """The forward and the backward panicked at separate sites, so fixing one
    left the other reachable through `.backward()`."""
    for dim in _every_dim(shape):
        mt.clear_autograd_graph()
        tensor = mt.from_numpy(np.zeros(shape, dtype=np.float64)).requires_grad_(True)
        getattr(tensor, name)(dim).sum().backward()

        gradient = mt.get_gradient(tensor)
        if gradient is not None:
            assert tuple(gradient.shape_vec()) == shape, f"dim={dim}"


@pytest.mark.parametrize("dim", [0, 1, -1, -2])
def test_softmax_on_a_populated_tensor_is_unchanged(dim):
    """The guard must only fire on the empty case."""
    values = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    shifted = np.exp(values - values.max(axis=dim, keepdims=True))

    np.testing.assert_allclose(
        mt.from_numpy(values).softmax(dim).numpy(),
        shifted / shifted.sum(axis=dim, keepdims=True),
    )


# The other end of the degenerate-shape range: dimensions that are each valid
# but whose product overflows `usize`. `Shape::numel` computes that product with
# checked arithmetic and panics, which is deliberate -- a wrapped element count
# would under-allocate storage that indexing code still trusts, turning an
# absurd shape into out-of-bounds access rather than a clean failure. But it is
# the last line of defence, not what a caller should meet: `mt.zeros(2**32,
# 2**32)` reached it straight from Python, panicking across the binding.
#
# `reshape` already rejected its own dimensions with an ordinary error, so the
# fix is that pattern applied to the shape arguments that lacked it. The panic
# stays underneath for anything that gets past the boundary.

OVERFLOWING = [
    (2**32, 2**32),
    (2**48, 2**48),
    (2**33, 2**33),
    (2**16, 2**16, 2**16, 2**16),
]


@pytest.mark.parametrize(
    "shape", OVERFLOWING, ids=lambda s: "x".join(str(d) for d in s)
)
def test_a_shape_whose_product_overflows_is_refused(shape):
    for build in [
        lambda: mt.zeros(*shape),
        lambda: mt.ones(shape),
        lambda: mt.full(shape, 1.0),
        lambda: mt.empty(shape),
        lambda: mt.randn(*shape),
    ]:
        with pytest.raises(ValueError) as excinfo:
            build()
        assert "more elements" in str(excinfo.value)


@pytest.mark.parametrize(
    "shape", OVERFLOWING, ids=lambda s: "x".join(str(d) for d in s)
)
def test_expanding_past_the_addressable_range_is_refused(shape):
    """`expand` resolves its `-1` entries in the engine, so it cannot be checked
    where the argument is parsed and needs its own guard."""
    unit = mt.from_numpy(np.zeros((1,) * len(shape), dtype=np.float32))
    with pytest.raises(ValueError) as excinfo:
        unit.expand(list(shape))
    assert "more elements" in str(excinfo.value)


@pytest.mark.parametrize(
    "shape", OVERFLOWING, ids=lambda s: "x".join(str(d) for d in s)
)
def test_the_tensor_local_constructors_refuse_it_too(shape):
    unit = mt.from_numpy(np.zeros((1,), dtype=np.float32))
    for build in [
        lambda: unit.new_zeros(shape),
        lambda: unit.new_ones(shape),
        lambda: unit.new_full(shape, 2.0),
    ]:
        with pytest.raises(ValueError):
            build()


def test_ordinary_and_empty_shapes_are_unaffected():
    """The guard is a product check, so it must not disturb a shape containing a
    zero -- whose product is zero, not an overflow."""
    assert tuple(mt.zeros(2, 3).shape_vec()) == (2, 3)
    assert tuple(mt.zeros(0, 3).shape_vec()) == (0, 3)
    assert tuple(mt.zeros(()).shape_vec()) == ()
    assert tuple(mt.ones([4, 1, 2]).shape_vec()) == (4, 1, 2)

    unit = mt.from_numpy(np.zeros((1, 1), dtype=np.float32))
    assert tuple(unit.expand([4, 5]).shape_vec()) == (4, 5)
    assert tuple(unit.expand([-1, 3]).shape_vec()) == (1, 3)
