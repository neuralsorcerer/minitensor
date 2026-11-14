# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""A float32 convolution gives each entry as the float32 nearest its exact value.

Accumulated in float32, most entries of a convolution of ordinary data, and of
each of its gradients, came out other than that float. Each is that float now:
the output with its bias, and the gradients of the input, the weight and the
bias, on either path a convolution takes -- straight from the input or through
the lowered columns -- with a stride, a dilation, padding and groups, for a
transposed convolution, and for the 1-D forms built on the two.
"""

from __future__ import annotations

import math
from fractions import Fraction

import numpy as np
import pytest

import minitensor as mt

# Every float32 is a whole multiple of 2^-149, so every product of two is one
# of 2^-298: each entry here is a sum of whole numbers over that.
_SCALE = 2**149


def _float32(value: Fraction) -> float:
    """The float32 nearest `value`, ties to even."""

    if value == 0:
        return 0.0
    sign = -1.0 if value < 0 else 1.0
    magnitude = abs(value)
    exponent = magnitude.numerator.bit_length() - magnitude.denominator.bit_length()
    if Fraction(2) ** exponent > magnitude:
        exponent -= 1
    unit = Fraction(2) ** max(exponent - 23, -149)
    rounded = round(magnitude / unit) * unit
    return sign * (math.inf if rounded >= Fraction(2) ** 128 else float(rounded))


def _whole(values) -> np.ndarray:
    """Each float32 of `values` as a whole number of 2^-149, exactly."""

    array = np.asarray(values, dtype=np.float32)
    flat = [int(Fraction(float(v)) * _SCALE) for v in array.ravel()]
    return np.array(flat, dtype=object).reshape(array.shape)


def _zeros(shape) -> np.ndarray:
    out = np.empty(shape, dtype=object)
    out[...] = 0
    return out


def _rounded(total) -> list:
    """The float32 nearest each whole number of 2^-298 in `total`."""

    nearest = np.vectorize(
        lambda v: _float32(Fraction(int(v), _SCALE * _SCALE)), otypes=[float]
    )
    return nearest(total).tolist()


def _exact(x, w, go, stride, padding, dilation, groups):
    """The convolution of `x` by `w`, and the gradients of `sum(output * go)`
    for the input and the weight, from the definition: each entry exact, a
    whole number of 2^-298."""

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

    return total + (_whole(bias) * _SCALE)[None, :, None, None]


def _channel_sums(go) -> list:
    """The float32 nearest the sum of each channel of `go`."""

    return _rounded(_whole(go).sum(axis=(0, 2, 3)) * _SCALE)


def _draw(rng, shape):
    values = rng.standard_normal(shape) * 10.0 ** rng.uniform(-1, 1, shape)
    return values.astype(np.float32)


# (input shape, weight shape, stride, padding, dilation, groups). The first
# six go straight from the input and the rest through the lowered columns,
# the last two as a product of one row and of one column.
CASES = {
    "depthwise": ((2, 6, 9, 11), (6, 1, 3, 3), (1, 1), (1, 1), (1, 1), 6),
    "few outputs, strided": ((2, 3, 13, 12), (4, 3, 3, 3), (2, 2), (1, 0), (1, 1), 1),
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
    "few outputs, dilated": ((2, 3, 10, 16), (2, 3, 3, 3), (1, 1), (2, 2), (2, 2), 1),
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
    "eight outputs, strided": ((2, 3, 13, 12), (8, 3, 3, 3), (2, 2), (1, 0), (1, 1), 1),
    "many outputs, dilated": ((2, 3, 10, 16), (20, 3, 3, 2), (1, 1), (2, 2), (2, 2), 1),
    "few outputs, wide kernel strided": (
        (1, 17, 9, 9),
        (3, 17, 3, 3),
        (1, 2),
        (1, 1),
        (1, 1),
        1,
    ),
    "many channels": ((2, 40, 6, 6), (24, 40, 3, 3), (1, 1), (1, 1), (1, 1), 1),
    "one output, strided": ((2, 17, 9, 9), (1, 17, 3, 3), (1, 2), (1, 1), (1, 1), 1),
    "one position": ((1, 40, 3, 3), (24, 40, 3, 3), (1, 1), (0, 0), (1, 1), 1),
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
    assert str(out.dtype) == "float32"
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
# output padding, dilation, groups), the last through the lowered columns.
TRANSPOSED = {
    "plain": ((2, 4, 5, 6), (4, 3, 3, 3), (1, 1), (0, 0), (0, 0), (1, 1), 1),
    "strided": ((2, 3, 5, 4), (3, 2, 3, 2), (2, 2), (1, 0), (1, 1), (1, 1), 1),
    "grouped, dilated": ((1, 4, 6, 6), (4, 3, 3, 3), (2, 1), (1, 1), (1, 0), (2, 1), 2),
    "depthwise": ((2, 6, 7, 7), (6, 1, 3, 3), (1, 1), (1, 1), (0, 0), (1, 1), 6),
    "many channels": ((2, 30, 5, 5), (30, 20, 3, 3), (2, 2), (1, 1), (1, 1), (1, 1), 1),
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
    blank = np.zeros((2, 4, 1, out.shape[2]), np.float32)
    _, want, _ = _exact(
        blank, w0[:, :, None], x0[:, :, None], (1, 3), (0, 2), (1, 1), 1
    )
    assert out.tolist() == np.array(_rounded(_with_bias(want, b0)))[:, :, 0].tolist()


def test_a_convolution_is_the_same_on_either_path():
    # A depthwise convolution goes straight from the input; the same channels
    # one at a time, each kernel thirty-two times over, go through the columns.
    rng = np.random.default_rng(1013)
    x0 = _draw(rng, (3, 4, 20, 20))
    w0 = _draw(rng, (4, 1, 3, 3))
    depthwise = mt.nn.conv2d(
        mt.Tensor(x0), mt.Tensor(w0), None, padding=1, groups=4
    ).numpy()
    for c in range(4):
        wide = np.ascontiguousarray(np.repeat(w0[c : c + 1], 32, axis=0))
        lowered = mt.nn.conv2d(
            mt.Tensor(np.ascontiguousarray(x0[:, c : c + 1])),
            mt.Tensor(wide),
            None,
            padding=1,
        ).numpy()
        for copy in range(32):
            assert np.array_equal(lowered[:, copy], depthwise[:, c])


@pytest.mark.parametrize(
    "case", ["depthwise", "few outputs, dilated", "many channels", "one position"]
)
def test_ties_and_zeros(case):
    # Whole numbers under a kernel of eighths: entries that cancel to zero,
    # and with a bias past 2^24 entries on halfway points, each settled from
    # a sum that took no rounding. A zero is a positive one.
    x_shape, w_shape, stride, padding, dilation, groups = CASES[case]
    rng = np.random.default_rng(7 + sum(map(ord, case)))
    x0 = rng.integers(-3, 4, x_shape).astype(np.float32)
    w0 = (rng.integers(-8, 9, w_shape) / 8.0).astype(np.float32)
    b0 = rng.integers(-5, 6, w_shape[0]).astype(np.float32) * np.float32(2.0**24)
    geometry = dict(stride=stride, padding=padding, dilation=dilation, groups=groups)

    x = mt.Tensor(x0, requires_grad=True)
    w = mt.Tensor(w0, requires_grad=True)
    out = mt.nn.conv2d(x, w, mt.Tensor(b0), **geometry)
    go = np.ones(tuple(out.shape), np.float32)
    out.sum().backward()

    want, grad_x, grad_w = _exact(x0, w0, go, stride, padding, dilation, groups)
    assert out.tolist() == _rounded(_with_bias(want, b0))
    plain = mt.nn.conv2d(mt.Tensor(x0), mt.Tensor(w0), None, **geometry).numpy()
    assert plain.tolist() == _rounded(want)
    assert x.grad.tolist() == _rounded(grad_x)
    assert w.grad.tolist() == _rounded(grad_w)
    for values in (plain, x.grad.numpy(), w.grad.numpy()):
        assert not np.signbit(values[values == 0]).any()


@pytest.mark.parametrize("outputs, groups", [(4, 4), (2, 1), (24, 1)])
def test_sums_that_cancel_are_positive_zeros(outputs, groups):
    # A kernel whose taps sum to zero over a constant input, and an incoming
    # gradient that sums to zero over the batch: every entry cancels.
    x0 = np.full((2, 4, 6, 7), -3.0, np.float32)
    w0 = np.ones((outputs, 4 // groups, 3, 3), np.float32)
    w0[:, :, 1, :] = -2.0
    x = mt.Tensor(x0, requires_grad=True)
    w = mt.Tensor(w0, requires_grad=True)
    out = mt.nn.conv2d(x, w, None, groups=groups)
    go = np.ones(tuple(out.shape), np.float32)
    go[1] = -1.0
    out.backward(mt.Tensor(go))
    for values in (out.numpy(), w.grad.numpy()):
        assert (values == 0).all() and not np.signbit(values).any()
    # Away from the border every position is read by all nine taps.
    inner = x.grad.numpy()[:, :, 2:-2, 2:-2]
    assert (inner == 0).all() and not np.signbit(inner).any()


@pytest.mark.parametrize(
    "w_shape, groups", [((3, 2, 3, 3), 1), ((2, 1, 3, 3), 2), ((24, 2, 3, 3), 1)]
)
def test_special_values(w_shape, groups):
    # An infinity and a NaN reach the entries that read them, as they would
    # added up one by one, and no others.
    x0 = np.ones((1, 2, 5, 5), np.float32)
    x0[0, 0, 2, 2] = np.inf
    x0[0, 1, 0, 0] = np.nan
    w0 = np.ones(w_shape, np.float32)
    w0[0, 0, 1, 1] = -1.0
    got = mt.nn.conv2d(mt.Tensor(x0), mt.Tensor(w0), None, padding=1, groups=groups)
    got = got.numpy()
    padded = np.pad(x0.astype(np.float64), ((0, 0), (0, 0), (1, 1), (1, 1)))
    group_in, per_group = w_shape[1], w_shape[0] // groups
    for o in range(w_shape[0]):
        first = (o // per_group) * group_in
        for i in range(5):
            for j in range(5):
                window = padded[0, first : first + group_in, i : i + 3, j : j + 3]
                want = float(np.sum(window * w0[o].astype(np.float64)))
                assert got[0, o, i, j] == want or (
                    math.isnan(want) and math.isnan(got[0, o, i, j])
                ), (o, i, j)


def test_larger_convolutions_agree_with_their_dots():
    # Large enough to cross the pool and the blocks of the lowering; entries
    # checked against `dot` of the terms they sum, itself rounded once.
    rng = np.random.default_rng(1019)
    x0, w0, b0 = _draw(rng, (6, 32, 21, 19)), _draw(rng, (40, 32, 3, 3)), _draw(rng, 40)
    x = mt.Tensor(x0, requires_grad=True)
    w = mt.Tensor(w0, requires_grad=True)
    out = mt.nn.conv2d(x, w, mt.Tensor(b0), stride=2, padding=1)
    go = _draw(rng, tuple(out.shape))
    (out * mt.Tensor(go)).sum().backward()
    got, grad_x, grad_w = out.numpy(), x.grad.numpy(), w.grad.numpy()
    padded = np.pad(x0, ((0, 0), (0, 0), (1, 1), (1, 1)))
    out_h, out_w = got.shape[2:]

    def dot(a, b):
        a = np.ascontiguousarray(a, dtype=np.float32).ravel()
        b = np.ascontiguousarray(b, dtype=np.float32).ravel()
        return mt.dot(mt.Tensor(a), mt.Tensor(b)).item()

    for _ in range(60):
        n, oc, i, j = (int(rng.integers(0, size)) for size in got.shape)
        window = padded[n, :, 2 * i : 2 * i + 3, 2 * j : 2 * j + 3]
        want = dot(np.append(window, 1.0), np.append(w0[oc], b0[oc]))
        assert got[n, oc, i, j] == want
    for _ in range(12):
        oc, ic, a, c = (int(rng.integers(0, size)) for size in w0.shape)
        reads = padded[:, ic, a : a + 2 * out_h - 1 : 2, c : c + 2 * out_w - 1 : 2]
        assert grad_w[oc, ic, a, c] == dot(go[:, oc], reads)
    for _ in range(60):
        n, ic, i, j = (int(rng.integers(0, size)) for size in grad_x.shape)
        taps, signal = [np.zeros(0, np.float32)], [np.zeros(0, np.float32)]
        for a in range(3):
            for c in range(3):
                oh, ow = i + 1 - a, j + 1 - c
                if (
                    oh % 2
                    or ow % 2
                    or not (0 <= oh // 2 < out_h and 0 <= ow // 2 < out_w)
                ):
                    continue
                taps.append(w0[:, ic, a, c])
                signal.append(go[n, :, oh // 2, ow // 2])
        assert grad_x[n, ic, i, j] == dot(np.concatenate(taps), np.concatenate(signal))


@pytest.mark.parametrize("outputs", [1, 5])
def test_a_row_wider_than_a_block_of_the_lowering(outputs):
    # One row of 19901 positions, lowered a block of 10485 at a time: a block
    # starts and ends part way along it.
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
        a = np.ascontiguousarray(a, dtype=np.float32)
        b = np.ascontiguousarray(b, dtype=np.float32)
        return mt.dot(mt.Tensor(a), mt.Tensor(b)).item()

    places = [0, 1, 10484, 10485, 10486, length - 1]
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
