# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""There was no way to make a feature map bigger.

`conv1d`, `conv2d` and the pooling layers all shrink a grid or leave it alone.
Nothing grew one, and that is not a gap you can work around: padding the input
and convolving is a different operation entirely. It cannot produce a stride-2
upsample at all, because the spacing between contributions comes from the stride
of the *scatter*, and no amount of padding puts gaps between input samples.

So a decoder, a GAN generator, an autoencoder and a segmentation head were all
unwritable. The test at the bottom builds one and trains it, which is the actual
thing that was blocked.

A transposed convolution is a convolution read from the other side: where a
convolution gathers a neighbourhood into each output position, this scatters
each input position across one. That makes it the adjoint of a convolution, and
the adjoint identity -- `<conv_transpose(x), y> == <x, conv(y)>` -- is the
strongest correctness test available, because it pins the entire index mapping
without any reference implementation to be wrong about. It is checked here
across strides, padding, dilation, groups and output padding.

It is not a deconvolution. It does not invert a convolution's values, only its
shape mapping, and the name is historical.

`output_padding` exists because that shape mapping is not injective: with
`stride = s`, `s` different input sizes convolve to the same output size, and it
says which one to come back to. Values from 0 to `stride - 1` name exactly
those; anything larger names no convolution at all and is refused.
"""

from __future__ import annotations

import numpy as np
import pytest

import minitensor as mt


def _reference(x, w, b=None, stride=(1, 1), padding=(0, 0), output_padding=(0, 0),
               dilation=(1, 1), groups=1):
    """The definition, written out: every input position scattered across the
    kernel footprint. Slow and obviously right, which is the point."""
    batch, in_channels, height, width = x.shape
    _, per_group_out, kernel_h, kernel_w = w.shape
    out_channels = per_group_out * groups
    out_h = ((height - 1) * stride[0] - 2 * padding[0]
             + dilation[0] * (kernel_h - 1) + output_padding[0] + 1)
    out_w = ((width - 1) * stride[1] - 2 * padding[1]
             + dilation[1] * (kernel_w - 1) + output_padding[1] + 1)
    out = np.zeros((batch, out_channels, out_h, out_w))
    group_in = in_channels // groups
    for n in range(batch):
        for g in range(groups):
            for ci in range(g * group_in, (g + 1) * group_in):
                for co_local in range(per_group_out):
                    co = g * per_group_out + co_local
                    for i in range(height):
                        for j in range(width):
                            for ky in range(kernel_h):
                                for kx in range(kernel_w):
                                    oy = i * stride[0] + ky * dilation[0] - padding[0]
                                    ox = j * stride[1] + kx * dilation[1] - padding[1]
                                    if 0 <= oy < out_h and 0 <= ox < out_w:
                                        out[n, co, oy, ox] += (
                                            x[n, ci, i, j] * w[ci, co_local, ky, kx]
                                        )
    if b is not None:
        out += b.reshape(1, -1, 1, 1)
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


CONFIGS = [
    dict(stride=(1, 1), padding=(0, 0), output_padding=(0, 0), dilation=(1, 1), groups=1),
    dict(stride=(2, 2), padding=(0, 0), output_padding=(0, 0), dilation=(1, 1), groups=1),
    dict(stride=(2, 2), padding=(1, 1), output_padding=(1, 1), dilation=(1, 1), groups=1),
    dict(stride=(2, 1), padding=(1, 0), output_padding=(1, 0), dilation=(2, 1), groups=2),
    dict(stride=(3, 2), padding=(0, 1), output_padding=(2, 1), dilation=(1, 2), groups=4),
    dict(stride=(1, 1), padding=(2, 1), output_padding=(0, 0), dilation=(1, 1), groups=1),
]


def _operands(config, seed=0, channels=(4, 4), shape=(2, 4, 5), kernel=(3, 2)):
    rng = np.random.default_rng(seed)
    in_channels, out_channels = channels
    groups = config["groups"]
    x = rng.standard_normal((shape[0], in_channels, shape[1], shape[2]))
    w = rng.standard_normal((in_channels, out_channels // groups, kernel[0], kernel[1]))
    b = rng.standard_normal(out_channels)
    return x, w, b


@pytest.mark.parametrize("config", CONFIGS)
def test_it_matches_the_definition(config):
    x, w, b = _operands(config, seed=3)
    got = mt.nn.conv_transpose2d(
        mt.Tensor(x, dtype="float64"),
        mt.Tensor(w, dtype="float64"),
        mt.Tensor(b, dtype="float64"),
        **config,
    ).numpy()
    np.testing.assert_allclose(got, _reference(x, w, b, **config), rtol=1e-12, atol=1e-13)


@pytest.mark.parametrize("config", CONFIGS)
def test_it_is_the_adjoint_of_convolution(config):
    """`<conv_transpose(x), y> == <x, conv(y)>` for every `x` and `y`. This is
    the whole index mapping in one equation, and it needs no reference
    implementation to compare against -- an error anywhere in the strides, the
    padding, the dilation or the group offsets breaks it."""
    x, w, _ = _operands(config, seed=5)
    scattered = mt.nn.conv_transpose2d(
        mt.Tensor(x, dtype="float64"), mt.Tensor(w, dtype="float64"), None, **config
    ).numpy()

    y = np.random.default_rng(7).standard_normal(scattered.shape)
    gathered = mt.nn.conv2d(
        mt.Tensor(y, dtype="float64"),
        mt.Tensor(w, dtype="float64"),
        None,
        stride=config["stride"],
        padding=config["padding"],
        dilation=config["dilation"],
        groups=config["groups"],
    ).numpy()

    assert gathered.shape == x.shape, "the adjoint has to land back on the input's shape"
    assert float((scattered * y).sum()) == pytest.approx(
        float((x * gathered).sum()), rel=1e-10
    )


@pytest.mark.parametrize(
    "shape,kernel,stride,padding,output_padding,dilation,expected",
    [
        ((4, 4), (3, 3), (1, 1), (0, 0), (0, 0), (1, 1), (6, 6)),
        ((4, 4), (2, 2), (2, 2), (0, 0), (0, 0), (1, 1), (8, 8)),
        ((8, 8), (4, 4), (2, 2), (1, 1), (0, 0), (1, 1), (16, 16)),
        ((5, 5), (3, 3), (2, 2), (1, 1), (1, 1), (1, 1), (10, 10)),
        ((4, 4), (3, 3), (1, 1), (0, 0), (0, 0), (2, 2), (8, 8)),
    ],
)
def test_the_output_size(shape, kernel, stride, padding, output_padding, dilation, expected):
    x = mt.Tensor(np.zeros((1, 2, *shape)), dtype="float64")
    w = mt.Tensor(np.zeros((2, 3, *kernel)), dtype="float64")
    got = mt.nn.conv_transpose2d(
        x, w, None, stride=stride, padding=padding,
        output_padding=output_padding, dilation=dilation,
    )
    assert got.numpy().shape == (1, 3, *expected)


def test_a_stride_two_kernel_four_layer_doubles_the_grid():
    """The upsampling block everything is built out of, and the one thing
    padding-plus-convolution cannot imitate."""
    x = mt.Tensor(np.zeros((2, 8, 7, 11)), dtype="float64")
    w = mt.Tensor(np.zeros((8, 4, 4, 4)), dtype="float64")
    got = mt.nn.conv_transpose2d(x, w, None, stride=2, padding=1)
    assert got.numpy().shape == (2, 4, 14, 22)


@pytest.mark.parametrize("length", [7, 8])
def test_output_padding_recovers_the_size_a_convolution_consumed(length):
    """The reason it exists. With stride 2 a 7-long and an 8-long signal
    convolve to the same length, so coming back needs to be told which."""
    x = np.random.default_rng(11).standard_normal((1, 2, length, length))
    weight = np.zeros((3, 2, 3, 3))
    down = mt.nn.conv2d(
        mt.Tensor(x, dtype="float64"), mt.Tensor(weight, dtype="float64"),
        None, stride=2, padding=1,
    )
    # The transposed weight layout is the convolution's, read from the far side.
    up = mt.nn.conv_transpose2d(
        down, mt.Tensor(weight, dtype="float64"), None,
        stride=2, padding=1, output_padding=(length + 1) % 2,
    )
    assert up.numpy().shape == x.shape


def test_output_padding_at_least_the_stride_is_refused():
    x = mt.Tensor(np.zeros((1, 1, 4, 4)), dtype="float64")
    w = mt.Tensor(np.zeros((1, 1, 3, 3)), dtype="float64")
    with pytest.raises(Exception):
        mt.nn.conv_transpose2d(x, w, None, stride=2, output_padding=2)
    with pytest.raises(Exception):
        mt.nn.conv_transpose2d(x, w, None, stride=1, output_padding=1)
    # One less is fine.
    assert mt.nn.conv_transpose2d(x, w, None, stride=2, output_padding=1) is not None


def test_the_weight_puts_input_channels_first():
    """The reverse of `conv2d`'s layout, and not an arbitrary choice: it is the
    same tensor a convolution would hold, so a weight can be shared between the
    two without a transpose on every pass."""
    x = mt.Tensor(np.zeros((1, 5, 4, 4)), dtype="float64")
    good = mt.Tensor(np.zeros((5, 2, 3, 3)), dtype="float64")
    assert mt.nn.conv_transpose2d(x, good, None).numpy().shape == (1, 2, 6, 6)
    swapped = mt.Tensor(np.zeros((2, 5, 3, 3)), dtype="float64")
    with pytest.raises(Exception):
        mt.nn.conv_transpose2d(x, swapped, None)


def test_bias_is_added_per_output_channel():
    x, w, _ = _operands(CONFIGS[0], seed=13)
    bias = np.arange(4, dtype=np.float64) * 10.0
    without = mt.nn.conv_transpose2d(
        mt.Tensor(x, dtype="float64"), mt.Tensor(w, dtype="float64"), None
    ).numpy()
    with_bias = mt.nn.conv_transpose2d(
        mt.Tensor(x, dtype="float64"),
        mt.Tensor(w, dtype="float64"),
        mt.Tensor(bias, dtype="float64"),
    ).numpy()
    np.testing.assert_allclose(with_bias - without, bias.reshape(1, -1, 1, 1) * np.ones_like(without))


def test_groups_keep_the_channels_apart():
    """A grouped transposed convolution is several independent ones, so zeroing
    one group's weights must leave the other group's output untouched."""
    config = dict(stride=(1, 1), padding=(0, 0), output_padding=(0, 0),
                  dilation=(1, 1), groups=2)
    x, w, _ = _operands(config, seed=17)
    full = mt.nn.conv_transpose2d(
        mt.Tensor(x, dtype="float64"), mt.Tensor(w, dtype="float64"), None, **config
    ).numpy()
    masked = w.copy()
    masked[: w.shape[0] // 2] = 0.0
    partial = mt.nn.conv_transpose2d(
        mt.Tensor(x, dtype="float64"), mt.Tensor(masked, dtype="float64"), None, **config
    ).numpy()
    half = full.shape[1] // 2
    assert np.abs(partial[:, :half]).max() == 0.0
    np.testing.assert_array_equal(partial[:, half:], full[:, half:])


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_both_float_dtypes_are_supported(dtype):
    x, w, b = _operands(CONFIGS[2], seed=19)
    got = mt.nn.conv_transpose2d(
        mt.Tensor(x.astype(dtype), dtype=dtype),
        mt.Tensor(w.astype(dtype), dtype=dtype),
        mt.Tensor(b.astype(dtype), dtype=dtype),
        **CONFIGS[2],
    )
    assert got.dtype == dtype
    tolerance = 1e-4 if dtype == "float32" else 1e-12
    np.testing.assert_allclose(
        got.numpy().astype(np.float64),
        _reference(x.astype(dtype).astype(np.float64), w.astype(dtype).astype(np.float64),
                   b.astype(dtype).astype(np.float64), **CONFIGS[2]),
        rtol=tolerance, atol=tolerance,
    )


def test_a_non_float_input_is_refused():
    with pytest.raises(Exception):
        mt.nn.conv_transpose2d(
            mt.Tensor(np.zeros((1, 1, 3, 3), dtype=np.int64), dtype="int64"),
            mt.Tensor(np.zeros((1, 1, 2, 2), dtype=np.int64), dtype="int64"),
        )


def test_a_wrongly_ranked_input_is_refused():
    with pytest.raises(Exception):
        mt.nn.conv_transpose2d(
            mt.Tensor(np.zeros((1, 3, 3)), dtype="float64"),
            mt.Tensor(np.zeros((1, 1, 2, 2)), dtype="float64"),
        )


def test_a_zero_stride_is_refused():
    with pytest.raises(Exception):
        mt.nn.conv_transpose2d(
            mt.Tensor(np.zeros((1, 1, 3, 3)), dtype="float64"),
            mt.Tensor(np.zeros((1, 1, 2, 2)), dtype="float64"),
            None,
            stride=0,
        )


# --- gradients ---------------------------------------------------------------


@pytest.mark.parametrize("config", CONFIGS)
def test_every_gradient_matches_numerical_differentiation(config):
    x, w, b = _operands(config, seed=23)

    def run():
        return mt.nn.conv_transpose2d(
            mt.Tensor(x, dtype="float64"),
            mt.Tensor(w, dtype="float64"),
            mt.Tensor(b, dtype="float64"),
            **config,
        ).numpy()

    probe = np.random.default_rng(29).standard_normal(run().shape)
    loss = lambda: float((run() * probe).sum())  # noqa: E731

    tx = mt.Tensor(x.copy(), dtype="float64", requires_grad=True)
    tw = mt.Tensor(w.copy(), dtype="float64", requires_grad=True)
    tb = mt.Tensor(b.copy(), dtype="float64", requires_grad=True)
    out = mt.nn.conv_transpose2d(tx, tw, tb, **config)
    (out * mt.Tensor(probe, dtype="float64")).sum().backward()

    np.testing.assert_allclose(tx.grad.numpy(), _numeric_grad(loss, x), rtol=1e-5, atol=1e-7)
    np.testing.assert_allclose(tw.grad.numpy(), _numeric_grad(loss, w), rtol=1e-5, atol=1e-7)
    np.testing.assert_allclose(tb.grad.numpy(), _numeric_grad(loss, b), rtol=1e-5, atol=1e-7)


def test_a_gradient_reaches_only_the_operands_that_asked_for_one():
    x, w, b = _operands(CONFIGS[1], seed=31)
    tx = mt.Tensor(x, dtype="float64", requires_grad=True)
    tw = mt.Tensor(w, dtype="float64")
    out = mt.nn.conv_transpose2d(tx, tw, mt.Tensor(b, dtype="float64"), **CONFIGS[1])
    out.sum().backward()
    assert tx.grad is not None
    assert tw.grad is None


# --- one dimension -----------------------------------------------------------


@pytest.mark.parametrize(
    "stride,padding,output_padding,dilation,groups",
    [(1, 0, 0, 1, 1), (2, 1, 1, 1, 1), (2, 0, 1, 2, 2), (3, 2, 2, 1, 4)],
)
def test_one_dimensional_agrees_with_the_two_dimensional(stride, padding, output_padding,
                                                         dilation, groups):
    """`conv_transpose1d` gives the signal a singleton height and defers, so
    there is one scatter and one backward rather than two to keep in step."""
    rng = np.random.default_rng(37)
    x = rng.standard_normal((2, 4, 6))
    w = rng.standard_normal((4, 4 // groups, 3))
    b = rng.standard_normal(4)

    got = mt.nn.conv_transpose1d(
        mt.Tensor(x, dtype="float64"), mt.Tensor(w, dtype="float64"),
        mt.Tensor(b, dtype="float64"),
        stride=stride, padding=padding, output_padding=output_padding,
        dilation=dilation, groups=groups,
    ).numpy()
    want = _reference(
        x[:, :, None, :], w[:, :, None, :], b,
        stride=(1, stride), padding=(0, padding),
        output_padding=(0, output_padding), dilation=(1, dilation), groups=groups,
    )[:, :, 0, :]
    np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-13)


def test_one_dimensional_carries_a_gradient():
    rng = np.random.default_rng(41)
    x = rng.standard_normal((1, 3, 5))
    w = rng.standard_normal((3, 2, 3))
    tx = mt.Tensor(x.copy(), dtype="float64", requires_grad=True)
    tw = mt.Tensor(w.copy(), dtype="float64", requires_grad=True)
    probe = rng.standard_normal(
        mt.nn.conv_transpose1d(
            mt.Tensor(x, dtype="float64"), mt.Tensor(w, dtype="float64"), None, stride=2
        ).numpy().shape
    )

    def loss():
        return float(
            (
                mt.nn.conv_transpose1d(
                    mt.Tensor(x, dtype="float64"), mt.Tensor(w, dtype="float64"),
                    None, stride=2,
                ).numpy()
                * probe
            ).sum()
        )

    out = mt.nn.conv_transpose1d(tx, tw, None, stride=2)
    (out * mt.Tensor(probe, dtype="float64")).sum().backward()
    np.testing.assert_allclose(tx.grad.numpy(), _numeric_grad(loss, x), rtol=1e-5, atol=1e-7)
    np.testing.assert_allclose(tw.grad.numpy(), _numeric_grad(loss, w), rtol=1e-5, atol=1e-7)


# --- the layers --------------------------------------------------------------


def test_the_layer_upsamples_and_reports_itself():
    layer = mt.nn.ConvTranspose2d(3, 8, 4, stride=2, padding=1, dtype="float64")
    assert layer.in_channels == 3
    assert layer.out_channels == 8
    assert layer.kernel_size == (4, 4)
    assert layer.output_padding == (0, 0)
    assert "ConvTranspose2d" in repr(layer)
    x = mt.Tensor(np.zeros((2, 3, 8, 8)), dtype="float64")
    assert layer(x).numpy().shape == (2, 8, 16, 16)


def test_the_layer_holds_a_weight_with_input_channels_first():
    layer = mt.nn.ConvTranspose2d(6, 4, (3, 5), groups=2, bias=False, dtype="float64")
    shapes = [tuple(p.shape) for p in layer.parameters()]
    assert shapes == [(6, 2, 3, 5)], "in_channels first, out_channels // groups second"


def test_the_one_dimensional_layer():
    layer = mt.nn.ConvTranspose1d(4, 2, 3, stride=2, output_padding=1, dtype="float64")
    assert (layer.in_channels, layer.out_channels, layer.kernel_size) == (4, 2, 3)
    assert layer(mt.Tensor(np.zeros((1, 4, 5)), dtype="float64")).numpy().shape == (1, 2, 12)


def test_the_layer_agrees_with_the_functional_form():
    layer = mt.nn.ConvTranspose2d(3, 5, 3, stride=2, padding=1, output_padding=1,
                                  dtype="float64")
    weight, bias = layer.parameters()
    x = mt.Tensor(np.random.default_rng(43).standard_normal((2, 3, 4, 4)), dtype="float64")
    direct = mt.nn.conv_transpose2d(
        x, weight, bias, stride=2, padding=1, output_padding=1
    )
    np.testing.assert_array_equal(layer(x).numpy(), direct.numpy())


def test_a_decoder_can_now_be_built_and_trained():
    """The thing the gap actually blocked, end to end: a stack of upsampling
    blocks that grows a small code into an image, and a gradient that reaches
    every parameter in it."""
    decoder = mt.nn.Sequential([
        mt.nn.ConvTranspose2d(16, 8, 4, stride=2, padding=1, dtype="float64"),
        mt.nn.ReLU(),
        mt.nn.ConvTranspose2d(8, 3, 4, stride=2, padding=1, dtype="float64"),
    ])
    code = mt.Tensor(np.random.default_rng(47).standard_normal((2, 16, 4, 4)), dtype="float64")
    target = mt.Tensor(np.random.default_rng(53).standard_normal((2, 3, 16, 16)),
                       dtype="float64")
    assert decoder(code).numpy().shape == (2, 3, 16, 16)

    parameters = decoder.parameters()
    optimizer = mt.optim.Adam(parameters, lr=0.05)
    first = None
    for _ in range(40):
        optimizer.zero_grad()
        difference = decoder(code) - target
        loss = (difference * difference).mean()
        if first is None:
            first = loss.item()
        loss.backward()
        optimizer.step()
    assert loss.item() < first * 0.7, "the decoder has to actually learn"
    assert all(p.grad is not None for p in parameters), "every parameter gets a gradient"


def test_an_encoder_decoder_round_trips_the_shape():
    """Downsample then upsample with the mirrored geometry and the grid comes
    back the size it started -- which is the property a U-Net's skip connections
    are built on."""
    x = mt.Tensor(np.random.default_rng(59).standard_normal((1, 3, 16, 16)), dtype="float64")
    down = mt.nn.Conv2d(3, 8, 4, stride=2, padding=1, dtype="float64")
    up = mt.nn.ConvTranspose2d(8, 3, 4, stride=2, padding=1, dtype="float64")
    encoded = down(x)
    assert encoded.numpy().shape == (1, 8, 8, 8)
    assert up(encoded).numpy().shape == (1, 3, 16, 16)
