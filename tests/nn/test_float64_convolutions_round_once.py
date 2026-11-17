# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""A float64 convolution gives each entry as the float64 nearest its exact value.

Accumulated in float64, with no wider float to add in, most entries of a
convolution of ordinary data, and of each of its gradients, came out other
than that float. Each is that float now: the output with its bias, and the
gradients of the input, the weight and the bias -- whether a group has one
output channel or many, with a stride, a dilation, padding and groups, for a
transposed convolution, for the 1-D forms built on the two, and for a
convolution of volumes.
"""

from __future__ import annotations

import math
from fractions import Fraction

import numpy as np
import pytest

import minitensor as mt

# Every value drawn here is a whole multiple of 2^-200, so every product of
# two is one of 2^-400: each entry is a sum of whole numbers over that.
_SCALE = 2**200


def _whole(values) -> np.ndarray:
    """Each float64 of `values` as a whole number of 2^-200, exactly."""

    array = np.asarray(values, dtype=np.float64)
    scaled = [Fraction(float(v)) * _SCALE for v in array.ravel()]
    assert all(v.denominator == 1 for v in scaled)
    return np.array([int(v) for v in scaled], dtype=object).reshape(array.shape)


def _zeros(shape) -> np.ndarray:
    out = np.empty(shape, dtype=object)
    out[...] = 0
    return out


def _rounded(total) -> list:
    """The float64 nearest each whole number of 2^-400 in `total`: a
    quotient of two whole numbers is rounded once."""

    nearest = np.vectorize(lambda v: int(v) / (_SCALE * _SCALE), otypes=[float])
    return nearest(total).tolist()


def _exact(x, w, go, stride, padding, dilation, groups):
    """The convolution of `x` by `w`, and the gradients of `sum(output * go)`
    for the input and the weight, from the definition: each entry exact, a
    whole number of 2^-400."""

    xs, ws, gs = _whole(x), _whole(w), _whole(go)
    out_channels, group_in, kh, kw = w.shape
    (sh, sw), (ph, pw), (dh, dw) = stride, padding, dilation
    height, width = x.shape[2:]
    padded = _zeros((x.shape[0], x.shape[1], height + 2 * ph, width + 2 * pw))
    padded[:, :, ph : ph + height, pw : pw + width] = xs
    out_h, out_w = go.shape[2:]
    rows = (np.arange(out_h) * sh)[:, None] + np.arange(kh) * dh
    cols = (np.arange(out_w) * sw)[:, None] + np.arange(kw) * dw
    # `reads[n, c, i, j, a, b]`: the input each tap of each output reads.
    reads = padded[:, :, rows[:, None, :, None], cols[None, :, None, :]]
    per_group = out_channels // groups
    out = _zeros(go.shape)
    grad_w = _zeros(w.shape)
    grad_padded = _zeros(padded.shape)
    for g in range(groups):
        channels = slice(g * group_in, (g + 1) * group_in)
        outputs = slice(g * per_group, (g + 1) * per_group)
        out[:, outputs] = np.tensordot(
            reads[:, channels], ws[outputs], axes=([1, 4, 5], [1, 2, 3])
        ).transpose(0, 3, 1, 2)
        grad_w[outputs] = np.tensordot(
            gs[:, outputs], reads[:, channels], axes=([0, 2, 3], [0, 2, 3])
        )
        spread = np.tensordot(gs[:, outputs], ws[outputs], axes=([1], [0]))
        spread = spread.transpose(0, 3, 1, 2, 4, 5)
        for a in range(kh):
            for c in range(kw):
                np.add.at(
                    grad_padded,
                    (slice(None), channels, rows[:, a][:, None], cols[:, c][None, :]),
                    spread[..., a, c],
                )
    grad_x = grad_padded[:, :, ph : ph + height, pw : pw + width]
    return out, grad_x, grad_w


def _with_bias(total, bias) -> np.ndarray:
    """`total` with one value of `bias` added to each channel, exactly."""

    shape = (1, -1) + (1,) * (total.ndim - 2)
    return total + (_whole(bias) * _SCALE).reshape(shape)


def _channel_sums(go) -> list:
    """The float64 nearest the sum of each channel of `go`."""

    axes = (0,) + tuple(range(2, go.ndim))
    return _rounded(_whole(go).sum(axis=axes) * _SCALE)


def _draw(rng, shape):
    return rng.standard_normal(shape) * 10.0 ** rng.uniform(-1, 1, shape)


def _positive_zeros(values) -> bool:
    values = np.asarray(values)
    return not np.signbit(values[values == 0]).any()


# (input shape, weight shape, stride, padding, dilation, groups). A group of
# few output channels is taken a row of the lowered columns at a time, one of
# more as products of its operands cut in two; the last four have strides,
# which share the input's gradient out by the taps that reach each position.
CASES = {
    "depthwise": ((2, 6, 9, 11), (6, 1, 3, 3), (1, 1), (1, 1), (1, 1), 6),
    "few outputs, dilated": ((2, 3, 10, 16), (2, 3, 3, 3), (1, 1), (2, 2), (2, 2), 1),
    "eight outputs, wide rows": (
        (2, 3, 8, 17),
        (8, 3, 3, 3),
        (1, 1),
        (1, 1),
        (1, 1),
        1,
    ),
    "sixteen outputs, one channel": (
        (2, 1, 7, 9),
        (16, 1, 3, 3),
        (1, 1),
        (1, 1),
        (1, 1),
        1,
    ),
    "one output, wide kernel": (
        (1, 17, 6, 7),
        (1, 17, 3, 3),
        (1, 1),
        (0, 1),
        (1, 1),
        1,
    ),
    "grouped, many per group": (
        (2, 6, 9, 11),
        (48, 2, 3, 3),
        (1, 1),
        (1, 1),
        (1, 1),
        3,
    ),
    "many outputs, dilated": ((2, 3, 10, 16), (20, 3, 3, 2), (1, 1), (2, 2), (2, 2), 1),
    "many channels": ((2, 40, 6, 6), (24, 40, 3, 3), (1, 1), (1, 1), (1, 1), 1),
    "one position": ((1, 40, 3, 3), (24, 40, 3, 3), (1, 1), (0, 0), (1, 1), 1),
    "few outputs, strided": ((2, 3, 13, 12), (4, 3, 3, 3), (2, 2), (1, 0), (1, 1), 1),
    "many outputs, strided along one axis": (
        (1, 17, 9, 9),
        (20, 17, 3, 3),
        (1, 2),
        (1, 1),
        (1, 1),
        1,
    ),
    "stride three, dilated": (
        (2, 4, 11, 13),
        (6, 4, 4, 3),
        (3, 2),
        (2, 1),
        (2, 3),
        1,
    ),
    "padding past the kernel": (
        (1, 3, 6, 7),
        (5, 3, 2, 2),
        (2, 3),
        (3, 4),
        (1, 1),
        1,
    ),
    "stride past the kernel": (
        (2, 4, 11, 12),
        (6, 2, 2, 2),
        (3, 4),
        (0, 1),
        (1, 1),
        2,
    ),
}


@pytest.mark.parametrize("case", list(CASES))
def test_a_convolution_and_its_gradients(case):
    x_shape, w_shape, stride, padding, dilation, groups = CASES[case]
    rng = np.random.default_rng(sum(map(ord, case)))
    x0, w0, b0 = _draw(rng, x_shape), _draw(rng, w_shape), _draw(rng, w_shape[0])
    geometry = dict(stride=stride, padding=padding, dilation=dilation, groups=groups)

    x = mt.Tensor(x0, requires_grad=True)
    w = mt.Tensor(w0, requires_grad=True)
    b = mt.Tensor(b0, requires_grad=True)
    out = mt.nn.conv2d(x, w, b, **geometry)
    assert str(out.dtype) == "float64"
    go = _draw(rng, tuple(out.shape))
    (out * mt.Tensor(go)).sum().backward()

    want, grad_x, grad_w = _exact(x0, w0, go, stride, padding, dilation, groups)
    assert out.tolist() == _rounded(_with_bias(want, b0))
    assert x.grad.tolist() == _rounded(grad_x)
    assert w.grad.tolist() == _rounded(grad_w)
    assert b.grad.tolist() == _channel_sums(go)
    # Without a bias, the output is the sum of its products alone.
    plain = mt.nn.conv2d(mt.Tensor(x0), mt.Tensor(w0), None, **geometry)
    assert plain.tolist() == _rounded(want)


# (input shape, weight shape `[C_in, C_out / groups, kH, kW]`, stride, padding,
# output padding, dilation, groups); the last leaves positions of its output
# that no tap reaches, which are its bias.
TRANSPOSED = {
    "plain": ((2, 4, 5, 6), (4, 3, 3, 3), (1, 1), (0, 0), (0, 0), (1, 1), 1),
    "strided": ((2, 3, 5, 4), (3, 2, 3, 2), (2, 2), (1, 0), (1, 1), (1, 1), 1),
    "grouped, dilated": ((1, 4, 6, 6), (4, 3, 3, 3), (2, 1), (1, 1), (1, 0), (2, 1), 2),
    "depthwise": ((2, 6, 7, 7), (6, 1, 3, 3), (1, 1), (1, 1), (0, 0), (1, 1), 6),
    "many channels": ((2, 30, 5, 5), (30, 20, 3, 3), (2, 2), (1, 1), (1, 1), (1, 1), 1),
    "stride past the kernel": (
        (2, 3, 4, 5),
        (3, 4, 2, 2),
        (3, 4),
        (0, 1),
        (2, 3),
        (1, 1),
        1,
    ),
}


@pytest.mark.parametrize("case", list(TRANSPOSED))
def test_a_transposed_convolution_and_its_gradients(case):
    x_shape, w_shape, stride, padding, extra, dilation, groups = TRANSPOSED[case]
    rng = np.random.default_rng(sum(map(ord, case)))
    x0, w0 = _draw(rng, x_shape), _draw(rng, w_shape)
    b0 = _draw(rng, w_shape[1] * groups)

    x = mt.Tensor(x0, requires_grad=True)
    w = mt.Tensor(w0, requires_grad=True)
    b = mt.Tensor(b0, requires_grad=True)
    out = mt.nn.conv_transpose2d(
        x,
        w,
        b,
        stride=stride,
        padding=padding,
        output_padding=extra,
        dilation=dilation,
        groups=groups,
    )
    assert str(out.dtype) == "float64"
    go = _draw(rng, tuple(out.shape))
    (out * mt.Tensor(go)).sum().backward()

    # The transposed convolution scatters where a convolution of its output's
    # shape gathers: its output is that convolution's input gradient, and its
    # input gradient that convolution's output.
    grad_x, want, grad_w = _exact(go, w0, x0, stride, padding, dilation, groups)
    assert out.tolist() == _rounded(_with_bias(want, b0))
    assert x.grad.tolist() == _rounded(grad_x)
    assert w.grad.tolist() == _rounded(grad_w)
    assert b.grad.tolist() == _channel_sums(go)


def test_the_one_dimensional_forms():
    rng = np.random.default_rng(1009)
    x0, w0, b0 = _draw(rng, (3, 5, 40)), _draw(rng, (7, 5, 4)), _draw(rng, 7)
    x = mt.Tensor(x0, requires_grad=True)
    w = mt.Tensor(w0, requires_grad=True)
    out = mt.nn.conv1d(x, w, mt.Tensor(b0), stride=2, padding=1)
    go = _draw(rng, tuple(out.shape))
    (out * mt.Tensor(go)).sum().backward()
    want, grad_x, grad_w = _exact(
        x0[:, :, None], w0[:, :, None], go[:, :, None], (1, 2), (0, 1), (1, 1), 1
    )
    assert out.tolist() == np.array(_rounded(_with_bias(want, b0)))[:, :, 0].tolist()
    assert x.grad.tolist() == np.array(_rounded(grad_x))[:, :, 0].tolist()
    assert w.grad.tolist() == np.array(_rounded(grad_w))[:, :, 0].tolist()

    x0, w0, b0 = _draw(rng, (2, 6, 9)), _draw(rng, (6, 4, 5)), _draw(rng, 4)
    out = mt.nn.conv_transpose1d(
        mt.Tensor(x0), mt.Tensor(w0), mt.Tensor(b0), stride=3, padding=2
    )
    blank = np.zeros((2, 4, 1, out.shape[2]))
    _, want, _ = _exact(
        blank, w0[:, :, None], x0[:, :, None], (1, 3), (0, 2), (1, 1), 1
    )
    assert out.tolist() == np.array(_rounded(_with_bias(want, b0)))[:, :, 0].tolist()


# (input shape, weight shape, stride, padding, dilation, groups), along the
# depth, the height and the width.
VOLUMES = [
    ((2, 3, 5, 6, 7), (4, 3, 3, 3, 3), (1, 1, 1), (1, 1, 1), (1, 1, 1), 1),
    ((1, 4, 6, 5, 5), (4, 2, 2, 3, 2), (2, 1, 2), (1, 0, 1), (2, 1, 1), 2),
    ((1, 2, 7, 4, 4), (14, 2, 5, 2, 2), (3, 2, 1), (2, 1, 0), (1, 1, 1), 1),
]


@pytest.mark.parametrize("case", VOLUMES)
def test_a_convolution_of_volumes_and_its_gradients(case):
    x_shape, w_shape, stride, padding, dilation, groups = case
    rng = np.random.default_rng(sum(x_shape) + 7 * sum(w_shape))
    x0, w0, b0 = _draw(rng, x_shape), _draw(rng, w_shape), _draw(rng, w_shape[0])
    x = mt.Tensor(x0, requires_grad=True)
    w = mt.Tensor(w0, requires_grad=True)
    b = mt.Tensor(b0, requires_grad=True)
    out = mt.nn.conv3d(x, w, b, stride, padding, dilation, groups)
    assert str(out.dtype) == "float64"
    go = _draw(rng, tuple(out.shape))
    (out * mt.Tensor(go)).sum().backward()

    # A window of a volume is its depth taps' windows of planes: each slice
    # of the output is the sum, over the taps, of a plane's convolution of
    # the input slice that tap reads, and the gradients add up the same way.
    depth = x_shape[2]
    want, grad_x, grad_w = _zeros(go.shape), _zeros(x_shape), _zeros(w_shape)
    for o in range(go.shape[2]):
        for tap in range(w_shape[2]):
            at = o * stride[0] + tap * dilation[0] - padding[0]
            if not 0 <= at < depth:
                continue
            plane, back, taps = _exact(
                x0[:, :, at],
                w0[:, :, tap],
                go[:, :, o],
                stride[1:],
                padding[1:],
                dilation[1:],
                groups,
            )
            want[:, :, o] += plane
            grad_x[:, :, at] += back
            grad_w[:, :, tap] += taps
    assert out.tolist() == _rounded(_with_bias(want, b0))
    assert x.grad.tolist() == _rounded(grad_x)
    assert w.grad.tolist() == _rounded(grad_w)
    assert b.grad.tolist() == _channel_sums(go)


@pytest.mark.parametrize(
    "case",
    ["depthwise", "few outputs, dilated", "many channels", "stride three, dilated"],
)
def test_ties_and_zeros(case):
    # Whole numbers under a kernel of eighths: entries that cancel to zero,
    # and with a bias past 2^53 entries on halfway points, each rounded once
    # from a sum no float64 holds. A zero is a positive one.
    x_shape, w_shape, stride, padding, dilation, groups = CASES[case]
    rng = np.random.default_rng(7 + sum(map(ord, case)))
    x0 = rng.integers(-3, 4, x_shape).astype(np.float64)
    w0 = rng.integers(-8, 9, w_shape) / 8.0
    b0 = rng.integers(-5, 6, w_shape[0]) * 2.0**53
    geometry = dict(stride=stride, padding=padding, dilation=dilation, groups=groups)

    x = mt.Tensor(x0, requires_grad=True)
    w = mt.Tensor(w0, requires_grad=True)
    out = mt.nn.conv2d(x, w, mt.Tensor(b0), **geometry)
    go = np.ones(tuple(out.shape))
    out.sum().backward()

    want, grad_x, grad_w = _exact(x0, w0, go, stride, padding, dilation, groups)
    assert out.tolist() == _rounded(_with_bias(want, b0))
    plain = mt.nn.conv2d(mt.Tensor(x0), mt.Tensor(w0), None, **geometry).numpy()
    assert plain.tolist() == _rounded(want)
    assert x.grad.tolist() == _rounded(grad_x)
    assert w.grad.tolist() == _rounded(grad_w)
    for values in (plain, x.grad.numpy(), w.grad.numpy()):
        assert _positive_zeros(values)


@pytest.mark.parametrize("outputs, groups", [(4, 4), (2, 1), (24, 1)])
def test_sums_that_cancel_are_positive_zeros(outputs, groups):
    # A kernel whose taps sum to zero over a constant input, and an incoming
    # gradient that sums to zero over the batch: every entry cancels.
    x0 = np.full((2, 4, 6, 7), -3.0)
    w0 = np.ones((outputs, 4 // groups, 3, 3))
    w0[:, :, 1, :] = -2.0
    x = mt.Tensor(x0, requires_grad=True)
    w = mt.Tensor(w0, requires_grad=True)
    out = mt.nn.conv2d(x, w, None, groups=groups)
    go = np.ones(tuple(out.shape))
    go[1] = -1.0
    out.backward(mt.Tensor(go))
    for values in (out.numpy(), w.grad.numpy()):
        assert (values == 0).all() and not np.signbit(values).any()
    # Away from the border every position is read by all nine taps.
    inner = x.grad.numpy()[:, :, 2:-2, 2:-2]
    assert (inner == 0).all() and not np.signbit(inner).any()


@pytest.mark.parametrize("outputs", [3, 24])
def test_channels_of_zeros(outputs):
    # A channel of the incoming gradient that is nothing but zeros, as a unit
    # that never fired leaves it, and a channel of the input that is: their
    # slices of the kernel's gradient are positive zeros, and a negative zero
    # for a bias leaves a positive one where every product is a zero.
    rng = np.random.default_rng(1031)
    x0, w0 = _draw(rng, (3, 4, 9, 8)), _draw(rng, (outputs, 4, 3, 3))
    x0[:, 2] = 0.0
    x = mt.Tensor(x0, requires_grad=True)
    w = mt.Tensor(w0, requires_grad=True)
    out = mt.nn.conv2d(x, w, mt.Tensor(np.full(outputs, -0.0)), padding=1)
    go = _draw(rng, tuple(out.shape))
    go[:, 1] = 0.0
    out.backward(mt.Tensor(go))
    want, grad_x, grad_w = _exact(x0, w0, go, (1, 1), (1, 1), (1, 1), 1)
    assert out.tolist() == _rounded(want)
    assert x.grad.tolist() == _rounded(grad_x)
    assert w.grad.tolist() == _rounded(grad_w)
    got = w.grad.numpy()
    assert (got[1] == 0).all() and (got[:, 2] == 0).all()
    assert _positive_zeros(got) and _positive_zeros(x.grad.numpy())
    blank = mt.nn.conv2d(
        mt.Tensor(np.zeros((1, 4, 5, 5))), w, mt.Tensor(np.full(outputs, -0.0))
    ).numpy()
    assert (blank == 0).all() and not np.signbit(blank).any()


@pytest.mark.parametrize(
    "w_shape, groups", [((3, 2, 3, 3), 1), ((2, 1, 3, 3), 2), ((24, 2, 3, 3), 1)]
)
def test_special_values(w_shape, groups):
    # An infinity and a NaN reach the entries that read them, as they would
    # added up one by one, and no others; and values past the range a float64
    # product can be split in leave their entries their exact sums.
    x0 = np.ones((1, 2, 5, 5))
    x0[0, 0, 2, 2] = np.inf
    x0[0, 1, 0, 0] = np.nan
    w0 = np.ones(w_shape)
    w0[0, 0, 1, 1] = -1.0
    got = mt.nn.conv2d(mt.Tensor(x0), mt.Tensor(w0), None, padding=1, groups=groups)
    got = got.numpy()
    padded = np.pad(x0, ((0, 0), (0, 0), (1, 1), (1, 1)))
    group_in, per_group = w_shape[1], w_shape[0] // groups
    for o in range(w_shape[0]):
        first = (o // per_group) * group_in
        for i in range(5):
            for j in range(5):
                window = padded[0, first : first + group_in, i : i + 3, j : j + 3]
                with np.errstate(invalid="ignore"):
                    want = float(np.sum(window * w0[o]))
                assert got[0, o, i, j] == want or (
                    math.isnan(want) and math.isnan(got[0, o, i, j])
                ), (o, i, j)

    rng = np.random.default_rng(1033)
    x0 = _draw(rng, (1, w_shape[1] * groups, 5, 5))
    x0[0, 0, 1] *= 2.0**600
    x0[0, 0, 3] *= 2.0**-600
    w0 = _draw(rng, w_shape)
    x = mt.Tensor(x0, requires_grad=True)
    w = mt.Tensor(w0, requires_grad=True)
    out = mt.nn.conv2d(x, w, None, padding=1, groups=groups)
    go = _draw(rng, tuple(out.shape))
    out.backward(mt.Tensor(go))
    got, grad_w = out.numpy(), w.grad.numpy()

    def dot(a, b):
        a = np.ascontiguousarray(a, dtype=np.float64).ravel()
        b = np.ascontiguousarray(b, dtype=np.float64).ravel()
        return mt.dot(mt.Tensor(a), mt.Tensor(b)).item()

    padded = np.pad(x0, ((0, 0), (0, 0), (1, 1), (1, 1)))
    for o in range(w_shape[0]):
        first = (o // per_group) * group_in
        for i in range(5):
            for j in range(5):
                window = padded[0, first : first + group_in, i : i + 3, j : j + 3]
                assert got[0, o, i, j] == dot(window, w0[o]), (o, i, j)
        for c in range(group_in):
            for a in range(3):
                for b in range(3):
                    reads = padded[0, first + c, a : a + 5, b : b + 5]
                    assert grad_w[o, c, a, b] == dot(go[0, o], reads), (o, c, a, b)


@pytest.mark.parametrize("outputs", [6, 40])
def test_larger_convolutions_agree_with_their_dots(outputs):
    # Large enough to cross the pool and the blocks of the lowering, the
    # kernel's gradient carried from one block of positions to the next: as
    # dots for six output channels, and as the pairs of its cut operands'
    # products for forty. Entries are checked against `dot` of the terms
    # they sum, itself rounded once.
    rng = np.random.default_rng(1019)
    x0, w0 = _draw(rng, (4, 32, 40, 40)), _draw(rng, (outputs, 32, 3, 3))
    b0 = _draw(rng, outputs)
    x = mt.Tensor(x0, requires_grad=True)
    w = mt.Tensor(w0, requires_grad=True)
    out = mt.nn.conv2d(x, w, mt.Tensor(b0), padding=1)
    go = _draw(rng, tuple(out.shape))
    (out * mt.Tensor(go)).sum().backward()
    got, grad_x, grad_w = out.numpy(), x.grad.numpy(), w.grad.numpy()
    padded = np.pad(x0, ((0, 0), (0, 0), (1, 1), (1, 1)))
    spread = np.pad(go, ((0, 0), (0, 0), (1, 1), (1, 1)))

    def dot(a, b):
        a = np.ascontiguousarray(a, dtype=np.float64).ravel()
        b = np.ascontiguousarray(b, dtype=np.float64).ravel()
        return mt.dot(mt.Tensor(a), mt.Tensor(b)).item()

    for _ in range(60):
        n, oc, i, j = (int(rng.integers(0, size)) for size in got.shape)
        window = padded[n, :, i : i + 3, j : j + 3]
        want = dot(np.append(window, 1.0), np.append(w0[oc], b0[oc]))
        assert got[n, oc, i, j] == want
    for _ in range(24):
        oc, ic, a, c = (int(rng.integers(0, size)) for size in w0.shape)
        assert grad_w[oc, ic, a, c] == dot(
            go[:, oc], padded[:, ic, a : a + 40, c : c + 40]
        )
    for _ in range(60):
        n, ic, i, j = (int(rng.integers(0, size)) for size in grad_x.shape)
        # Tap `(a, c)` reaches the position from the output at `(i + 1 - a,
        # j + 1 - c)`: the window of the incoming gradient, turned about.
        window = spread[n, :, i : i + 3, j : j + 3][:, ::-1, ::-1]
        assert grad_x[n, ic, i, j] == dot(w0[:, ic], window)


@pytest.mark.parametrize("outputs", [1, 5, 14])
def test_a_row_wider_than_a_block_of_the_lowering(outputs):
    # One row of 19901 positions, lowered 5242 at a time: a block starts and
    # ends part way along it.
    rng = np.random.default_rng(1021)
    x0 = _draw(rng, (1, 1, 40000))
    w0, b0 = _draw(rng, (outputs, 1, 200)), _draw(rng, outputs)
    x = mt.Tensor(x0, requires_grad=True)
    w = mt.Tensor(w0, requires_grad=True)
    out = mt.nn.conv1d(x, w, mt.Tensor(b0), stride=2)
    go = _draw(rng, tuple(out.shape))
    out.backward(mt.Tensor(go))
    got, grad_x, grad_w = out.numpy(), x.grad.numpy(), w.grad.numpy()
    length = got.shape[2]

    def dot(a, b):
        a = np.ascontiguousarray(a, dtype=np.float64)
        b = np.ascontiguousarray(b, dtype=np.float64)
        return mt.dot(mt.Tensor(a), mt.Tensor(b)).item()

    places = [0, 1, 5241, 5242, 5243, length - 1]
    places += [int(v) for v in rng.integers(0, length, 12)]
    for oc in range(outputs):
        for j in places:
            window = np.append(x0[0, 0, 2 * j : 2 * j + 200], 1.0)
            assert got[0, oc, j] == dot(window, np.append(w0[oc, 0], b0[oc]))
        for tap in (0, 57, 199):
            reads = x0[0, 0, tap : tap + 2 * length - 1 : 2]
            assert grad_w[oc, 0, tap] == dot(go[0, oc], reads)
    for i in [0, 199, 200, 20969, 20970, 20971, 39999]:
        taps = [t for t in range(200) if i >= t and (i - t) % 2 == 0]
        taps = [t for t in taps if (i - t) // 2 < length]
        kernel = np.concatenate([w0[:, 0, t] for t in taps])
        signal = np.concatenate([go[0, :, (i - t) // 2] for t in taps])
        assert grad_x[0, 0, i] == dot(kernel, signal)


def test_a_transposed_convolution_leaves_its_bias_where_no_tap_reaches():
    # A stride past the kernel: between the kernel's reach of one input
    # position and the next, the output is its bias, to the bit.
    rng = np.random.default_rng(1039)
    x0, w0, b0 = _draw(rng, (1, 2, 3, 3)), _draw(rng, (2, 3, 2, 2)), _draw(rng, 3)
    out = mt.nn.conv_transpose2d(
        mt.Tensor(x0), mt.Tensor(w0), mt.Tensor(b0), stride=4
    ).numpy()
    assert out.shape == (1, 3, 10, 10)
    for channel in range(3):
        between = out[0, channel][2:4, :]
        assert (between == b0[channel]).all()
        assert (out[0, channel][:, 6:8] == b0[channel]).all()
    plain = mt.nn.conv_transpose2d(mt.Tensor(x0), mt.Tensor(w0), None, stride=4)
    plain = plain.numpy()
    assert (plain[0, :, 2:4, :] == 0).all() and _positive_zeros(plain)
