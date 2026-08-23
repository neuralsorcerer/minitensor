# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""A feature map could not be resampled without learning something.

`conv_transpose2d` grows a map with a kernel it has to learn and
`adaptive_avg_pool2d` shrinks one by averaging. Neither *resamples*: a U-Net's
decoder, a feature pyramid and every segmentation head need to take a map back
to an earlier resolution with no parameters at all, so that a skip connection
lines up channel for channel and pixel for pixel. The last test builds that
step -- downsample, upsample, concatenate, convolve -- and checks the gradient
still reaches the encoder.

Nearest could almost be assembled: an integer scale factor is a repeat along
each axis, and that equivalence is asserted below because it is a free check on
the index arithmetic. A non-integer factor cannot be, and bilinear cannot at
all -- it reads each output from a weighted pair of neighbours at a fractional
coordinate, which is a gather nothing else here performs.

`align_corners` is the one genuinely confusing parameter, so it is pinned from
both sides. Set, the first and last output positions sit exactly on the first
and last input samples: endpoints preserved, spacing distorted. That makes
bilinear reproduce a linear ramp *exactly* at the resampled coordinates, which
is the sharpest test in this file and the one that would catch an off-by-one in
either direction. Unset -- the default -- output positions are the centres of
equal cells covering the input, which keeps the spacing uniform and is what
makes resampling twice by two agree with resampling once by four. That property
is asserted too, and it is the reason the default is what it is.

Interpolation is linear in its input, so the gradient is the transpose of the
forward. `<interpolate(x), y> == <x, backward(y)>` therefore has to hold to the
last bit for every mode, size and convention, and it pins the weights and the
indices together in a way finite differences only approximate.
"""

from __future__ import annotations

import numpy as np
import pytest

import minitensor as mt


def _axis(in_size, out_size, mode, align_corners):
    """The documented coordinate rule, written out."""
    lower, upper, weight = [], [], []
    for index in range(out_size):
        if mode == "nearest":
            source = min(int(index * in_size / out_size), in_size - 1)
            lower.append(source)
            upper.append(source)
            weight.append(0.0)
            continue
        if align_corners:
            source = 0.0 if out_size == 1 else index * (in_size - 1) / (out_size - 1)
        else:
            source = max((index + 0.5) * in_size / out_size - 0.5, 0.0)
        base = min(int(np.floor(source)), in_size - 1)
        lower.append(base)
        upper.append(min(base + 1, in_size - 1))
        weight.append(source - np.floor(source))
    return lower, upper, weight


def _reference(x, output_size, mode, align_corners):
    batch, channels, height, width = x.shape
    out_h, out_w = output_size
    row_lo, row_hi, row_w = _axis(height, out_h, mode, align_corners)
    col_lo, col_hi, col_w = _axis(width, out_w, mode, align_corners)
    out = np.zeros((batch, channels, out_h, out_w))
    for i in range(out_h):
        for j in range(out_w):
            above = x[:, :, row_lo[i], col_lo[j]] + (
                x[:, :, row_lo[i], col_hi[j]] - x[:, :, row_lo[i], col_lo[j]]
            ) * col_w[j]
            below = x[:, :, row_hi[i], col_lo[j]] + (
                x[:, :, row_hi[i], col_hi[j]] - x[:, :, row_hi[i], col_lo[j]]
            ) * col_w[j]
            out[:, :, i, j] = above + (below - above) * row_w[i]
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


# Up, down, one axis each way, and unchanged.
CASES = [
    ((2, 3, 4, 4), (8, 8)),
    ((1, 2, 5, 7), (3, 4)),
    ((1, 1, 3, 3), (7, 5)),
    ((2, 2, 6, 6), (6, 6)),
    ((1, 3, 9, 4), (2, 11)),
    ((1, 2, 1, 1), (4, 4)),
]
MODES = [("nearest", False), ("bilinear", False), ("bilinear", True)]


@pytest.mark.parametrize("shape,output_size", CASES)
@pytest.mark.parametrize("mode,align_corners", MODES)
def test_it_matches_the_coordinate_rule(shape, output_size, mode, align_corners):
    values = np.random.default_rng(hash((shape, output_size)) % 1000).standard_normal(shape)
    got = mt.nn.interpolate(
        mt.Tensor(values, dtype="float64"),
        size=output_size, mode=mode, align_corners=align_corners,
    ).numpy()
    assert got.shape == (shape[0], shape[1], *output_size)
    np.testing.assert_allclose(
        got, _reference(values, output_size, mode, align_corners), rtol=1e-12, atol=1e-14
    )


@pytest.mark.parametrize("mode,align_corners", MODES)
def test_resampling_to_the_same_size_changes_nothing(mode, align_corners):
    """True for every mode and both conventions, and the cheapest thing to get
    wrong by half a pixel."""
    values = np.random.default_rng(3).standard_normal((2, 3, 5, 7))
    got = mt.nn.interpolate(
        mt.Tensor(values, dtype="float64"),
        size=(5, 7), mode=mode, align_corners=align_corners,
    ).numpy()
    np.testing.assert_array_equal(got, values)


def test_bilinear_reproduces_a_plane_exactly():
    """With the corners aligned, interpolating a linear function has to give
    that same function sampled at the new coordinates -- not approximately.
    An off-by-one in either the index or the weight breaks this immediately."""
    height, width = 5, 4
    rows, cols = np.meshgrid(
        np.arange(height, dtype=float), np.arange(width, dtype=float), indexing="ij"
    )
    plane = (3.0 * rows - 2.0 * cols + 1.0)[None, None]

    out_h, out_w = 9, 7
    got = mt.nn.interpolate(
        mt.Tensor(plane, dtype="float64"),
        size=(out_h, out_w), mode="bilinear", align_corners=True,
    ).numpy()

    sampled_rows = np.arange(out_h) * (height - 1) / (out_h - 1)
    sampled_cols = np.arange(out_w) * (width - 1) / (out_w - 1)
    want = (3.0 * sampled_rows[:, None] - 2.0 * sampled_cols[None, :] + 1.0)[None, None]
    np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-13)


def test_the_default_convention_composes():
    """Half-pixel centres are the default because resampling by two twice gives
    the same answer as resampling by four once. Aligned corners do not have
    this property, which is the trade the flag makes."""
    values = np.random.default_rng(5).standard_normal((1, 2, 4, 4))
    tensor = mt.Tensor(values, dtype="float64")
    once = mt.nn.interpolate(tensor, scale_factor=4, mode="nearest").numpy()
    twice = mt.nn.interpolate(
        mt.nn.interpolate(tensor, scale_factor=2, mode="nearest"),
        scale_factor=2, mode="nearest",
    ).numpy()
    np.testing.assert_array_equal(once, twice)


def test_an_integer_nearest_upsample_is_a_repeat():
    """A free check on the index arithmetic: the one case that *is* expressible
    another way has to agree with it."""
    values = np.random.default_rng(7).standard_normal((1, 2, 3, 4))
    got = mt.nn.interpolate(
        mt.Tensor(values, dtype="float64"), scale_factor=2, mode="nearest"
    ).numpy()
    np.testing.assert_array_equal(
        got, np.repeat(np.repeat(values, 2, axis=2), 2, axis=3)
    )


def test_aligned_corners_keep_the_corners():
    values = np.arange(9, dtype=np.float64).reshape(1, 1, 3, 3)
    got = mt.nn.interpolate(
        mt.Tensor(values, dtype="float64"),
        size=(5, 5), mode="bilinear", align_corners=True,
    ).numpy()
    assert got[0, 0, 0, 0] == values[0, 0, 0, 0]
    assert got[0, 0, -1, -1] == values[0, 0, -1, -1]
    assert got[0, 0, 0, -1] == values[0, 0, 0, -1]


def test_a_scale_factor_truncates():
    """Rather than rounding: a factor of 0.5 on an odd extent drops the odd
    sample instead of inventing a place for it."""
    values = mt.Tensor(np.zeros((1, 1, 7, 9)), dtype="float64")
    assert mt.nn.interpolate(values, scale_factor=0.5).numpy().shape == (1, 1, 3, 4)
    assert mt.nn.interpolate(values, scale_factor=1.5).numpy().shape == (1, 1, 10, 13)


def test_the_axes_can_be_scaled_differently():
    values = mt.Tensor(np.zeros((1, 2, 4, 6)), dtype="float64")
    assert mt.nn.interpolate(values, scale_factor=(2.0, 0.5)).numpy().shape == (1, 2, 8, 3)
    assert mt.nn.interpolate(values, size=(3, 12)).numpy().shape == (1, 2, 3, 12)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_both_float_dtypes_are_supported(dtype):
    values = np.random.default_rng(11).standard_normal((2, 3, 5, 4)).astype(dtype)
    got = mt.nn.interpolate(mt.Tensor(values, dtype=dtype), size=(7, 9), mode="bilinear")
    assert got.dtype == dtype
    tolerance = 1e-6 if dtype == "float32" else 1e-13
    np.testing.assert_allclose(
        got.numpy().astype(np.float64),
        _reference(values.astype(np.float64), (7, 9), "bilinear", False),
        rtol=tolerance, atol=tolerance,
    )


# --- what it refuses ---------------------------------------------------------


def test_giving_both_a_size_and_a_scale_factor_is_refused():
    values = mt.Tensor(np.zeros((1, 1, 4, 4)), dtype="float64")
    with pytest.raises(Exception):
        mt.nn.interpolate(values, size=(8, 8), scale_factor=2)


def test_giving_neither_is_refused():
    values = mt.Tensor(np.zeros((1, 1, 4, 4)), dtype="float64")
    with pytest.raises(Exception):
        mt.nn.interpolate(values)


def test_a_wrong_number_of_sizes_is_refused():
    values = mt.Tensor(np.zeros((1, 1, 4, 4)), dtype="float64")
    with pytest.raises(Exception):
        mt.nn.interpolate(values, size=(8, 8, 8))


def test_a_non_positive_scale_factor_is_refused():
    values = mt.Tensor(np.zeros((1, 1, 4, 4)), dtype="float64")
    for bad in (0.0, -1.0, float("inf"), float("nan")):
        with pytest.raises(Exception):
            mt.nn.interpolate(values, scale_factor=bad)


def test_an_unknown_mode_is_refused():
    values = mt.Tensor(np.zeros((1, 1, 4, 4)), dtype="float64")
    with pytest.raises(Exception):
        mt.nn.interpolate(values, size=(8, 8), mode="bicubic")


def test_a_wrongly_ranked_input_is_refused():
    with pytest.raises(Exception):
        mt.nn.interpolate(mt.Tensor(np.zeros((4, 4)), dtype="float64"), size=(8, 8))
    with pytest.raises(Exception):
        mt.nn.interpolate(mt.Tensor(np.zeros((1, 1, 2, 4, 4)), dtype="float64"), size=(8, 8))


def test_an_integer_input_is_refused():
    with pytest.raises(Exception):
        mt.nn.interpolate(
            mt.Tensor(np.zeros((1, 1, 4, 4), dtype=np.int64), dtype="int64"), size=(8, 8)
        )


def test_resampling_an_empty_axis_into_a_non_empty_one_is_refused():
    empty = mt.Tensor(np.zeros((1, 2, 0, 4)), dtype="float64")
    with pytest.raises(Exception):
        mt.nn.interpolate(empty, size=(4, 4))


# --- gradients ---------------------------------------------------------------


@pytest.mark.parametrize("shape,output_size", CASES[:5])
@pytest.mark.parametrize("mode,align_corners", MODES)
def test_the_gradient_matches_numerical_differentiation(shape, output_size, mode, align_corners):
    rng = np.random.default_rng(13)
    values = rng.standard_normal(shape)
    probe = rng.standard_normal((shape[0], shape[1], *output_size))

    def loss():
        return float(
            (
                mt.nn.interpolate(
                    mt.Tensor(values, dtype="float64"),
                    size=output_size, mode=mode, align_corners=align_corners,
                ).numpy()
                * probe
            ).sum()
        )

    t = mt.Tensor(values.copy(), dtype="float64", requires_grad=True)
    (
        mt.nn.interpolate(t, size=output_size, mode=mode, align_corners=align_corners)
        * mt.Tensor(probe, dtype="float64")
    ).sum().backward()
    np.testing.assert_allclose(
        t.grad.numpy(), _numeric_grad(loss, values), rtol=1e-5, atol=1e-7
    )


@pytest.mark.parametrize("shape,output_size", CASES)
@pytest.mark.parametrize("mode,align_corners", MODES)
def test_the_gradient_is_the_transpose_of_the_forward(shape, output_size, mode, align_corners):
    """Interpolation is linear, so `<interpolate(x), y>` and `<x, backward(y)>`
    are the same number -- to the last bit, not approximately. This pins every
    index against every weight in one equation, which finite differences can
    only approach."""
    rng = np.random.default_rng(17)
    values = rng.standard_normal(shape)
    probe = rng.standard_normal((shape[0], shape[1], *output_size))

    forward = mt.nn.interpolate(
        mt.Tensor(values, dtype="float64"),
        size=output_size, mode=mode, align_corners=align_corners,
    ).numpy()
    t = mt.Tensor(values.copy(), dtype="float64", requires_grad=True)
    (
        mt.nn.interpolate(t, size=output_size, mode=mode, align_corners=align_corners)
        * mt.Tensor(probe, dtype="float64")
    ).sum().backward()

    assert float((forward * probe).sum()) == pytest.approx(
        float((values * t.grad.numpy()).sum()), rel=1e-11
    )


def test_an_upsampled_gradient_sums_to_the_number_of_readers():
    """Nearest doubling: every input is read by exactly four outputs, so a
    gradient of ones comes back as fours."""
    t = mt.Tensor(np.zeros((1, 1, 2, 2)), dtype="float64", requires_grad=True)
    mt.nn.interpolate(t, scale_factor=2, mode="nearest").sum().backward()
    np.testing.assert_array_equal(t.grad.numpy(), np.full((1, 1, 2, 2), 4.0))


# --- one dimension -----------------------------------------------------------


@pytest.mark.parametrize("length,output_size", [(5, 9), (9, 4), (1, 6), (7, 7)])
@pytest.mark.parametrize("mode,align_corners", [("nearest", False), ("linear", False), ("linear", True)])
def test_one_dimensional_agrees_with_the_two_dimensional(length, output_size, mode, align_corners):
    """A 3-D signal is a 4-D one with a singleton height, so there is one kernel
    and one backward rather than two to keep in step."""
    values = np.random.default_rng(19).standard_normal((2, 3, length))
    got = mt.nn.interpolate(
        mt.Tensor(values, dtype="float64"),
        size=output_size, mode=mode, align_corners=align_corners,
    ).numpy()
    want = _reference(
        values[:, :, None, :], (1, output_size),
        "nearest" if mode == "nearest" else "bilinear", align_corners,
    )[:, :, 0, :]
    np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-14)


def test_linear_and_bilinear_name_the_same_mode():
    """Which word a caller writes depends only on how many spatial axes they
    have, and the rank already says that."""
    values = mt.Tensor(np.random.default_rng(23).standard_normal((1, 2, 6)), dtype="float64")
    np.testing.assert_array_equal(
        mt.nn.interpolate(values, size=11, mode="linear").numpy(),
        mt.nn.interpolate(values, size=11, mode="bilinear").numpy(),
    )


def test_one_dimensional_carries_a_gradient():
    values = np.random.default_rng(29).standard_normal((1, 2, 5))
    probe = np.random.default_rng(31).standard_normal((1, 2, 9))

    def loss():
        return float(
            (mt.nn.interpolate(mt.Tensor(values, dtype="float64"), size=9, mode="linear").numpy()
             * probe).sum()
        )

    t = mt.Tensor(values.copy(), dtype="float64", requires_grad=True)
    (mt.nn.interpolate(t, size=9, mode="linear") * mt.Tensor(probe, dtype="float64")).sum().backward()
    np.testing.assert_allclose(t.grad.numpy(), _numeric_grad(loss, values), rtol=1e-5, atol=1e-7)


# --- the layer ---------------------------------------------------------------


def test_the_layer_reports_itself_and_holds_no_parameters():
    layer = mt.nn.Upsample(scale_factor=2, mode="bilinear")
    assert "Upsample" in repr(layer)
    assert layer.align_corners is False
    assert layer.parameters() == []
    x = mt.Tensor(np.zeros((2, 3, 8, 8)), dtype="float64")
    assert layer(x).numpy().shape == (2, 3, 16, 16)


def test_the_layer_takes_a_size_too():
    layer = mt.nn.Upsample(size=(16, 20))
    assert layer(mt.Tensor(np.zeros((2, 3, 8, 8)), dtype="float64")).numpy().shape == (2, 3, 16, 20)


def test_the_layer_agrees_with_the_functional_form():
    values = np.random.default_rng(37).standard_normal((2, 3, 5, 7))
    tensor = mt.Tensor(values, dtype="float64")
    np.testing.assert_array_equal(
        mt.nn.Upsample(size=(9, 4), mode="bilinear", align_corners=True)(tensor).numpy(),
        mt.nn.interpolate(tensor, size=(9, 4), mode="bilinear", align_corners=True).numpy(),
    )


def test_a_unet_decoder_step_is_now_expressible():
    """The thing the gap actually blocked: bring a map back to an earlier
    resolution with no parameters, concatenate the skip connection, and keep a
    gradient flowing to the encoder underneath."""
    encoder = mt.nn.Conv2d(3, 8, 3, stride=2, padding=1, dtype="float64")
    decoder = mt.nn.Conv2d(8 + 3, 4, 3, padding=1, dtype="float64")
    image = mt.Tensor(
        np.random.default_rng(41).standard_normal((2, 3, 16, 16)), dtype="float64"
    )

    encoded = encoder(image)
    assert encoded.numpy().shape == (2, 8, 8, 8)
    upsampled = mt.nn.interpolate(encoded, size=(16, 16), mode="bilinear")
    merged = mt.cat([upsampled, image], dim=1)
    assert merged.numpy().shape == (2, 11, 16, 16)

    output = decoder(merged)
    assert output.numpy().shape == (2, 4, 16, 16)
    output.sum().backward()
    assert all(p.grad is not None for p in encoder.parameters()), (
        "the gradient has to reach through the resampling to the encoder"
    )
