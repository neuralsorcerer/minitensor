# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""A float32 conv3d gives each entry, and each entry of each gradient, as the
float32 nearest its exact value.

It was a 2-D convolution for each depth tap of the kernel, the planes added
and the bias added after them: a rounding for each tap's plane, one for each
addition and one for the bias. It is one sum of the products of every tap of
the window now, the bias in it, rounded once; and so are the gradients of the
input, the kernel and the bias.

The values here are seven-bit whole numbers scaled by powers of two between
2^-6 and 2^6, so every product, and every sum of the few hundred taken here,
is exact in float64 while few of them fit a float32: a float64 sum is the
exact value, and narrowing it rounds once.
"""

from __future__ import annotations

import numpy as np
import pytest

import minitensor as mt


def _values(rng, shape):
    digits = rng.integers(-64, 64, size=shape)
    scale = 2.0 ** rng.integers(-6, 7, size=shape)
    values = digits * scale
    values[rng.random(shape) < 0.08] = 0.0
    return values.astype(np.float32)


def _exact(x, w, b, g, stride, padding, dilation, groups):
    """The output, and the gradients of `sum(output * g)` for the input, the
    kernel and the bias, from the definition, each exact in float64."""

    x, w, g = (np.asarray(a, dtype=np.float64) for a in (x, w, g))
    n, c, *size = x.shape
    out_channels, per_in, *kernel = w.shape
    per_out = out_channels // groups
    padded = np.zeros((n, c) + tuple(s + 2 * p for s, p in zip(size, padding)))
    inside = tuple(slice(p, p + s) for p, s in zip(padding, size))
    padded[(slice(None), slice(None)) + inside] = x
    out_size = [
        (s + 2 * p - d * (k - 1) - 1) // st + 1
        for s, p, d, k, st in zip(size, padding, dilation, kernel, stride)
    ]
    out = np.zeros((n, out_channels, *out_size))
    grad_padded = np.zeros_like(padded)
    grad_w = np.zeros_like(w)
    for group in range(groups):
        ins = slice(group * per_in, (group + 1) * per_in)
        outs = slice(group * per_out, (group + 1) * per_out)
        for taps in np.ndindex(*kernel):
            reads = tuple(
                slice(t * d, t * d + st * (o - 1) + 1, st)
                for t, d, st, o in zip(taps, dilation, stride, out_size)
            )
            window = (slice(None), ins) + reads
            tap = w[(outs, slice(None)) + taps]
            out[:, outs] += np.einsum("ncdhw,oc->nodhw", padded[window], tap)
            grad_padded[window] += np.einsum("nodhw,oc->ncdhw", g[:, outs], tap)
            grad_w[(outs, slice(None)) + taps] += np.einsum(
                "nodhw,ncdhw->oc", g[:, outs], padded[window]
            )
    if b is not None:
        out += np.asarray(b, dtype=np.float64).reshape(1, -1, 1, 1, 1)
    grad_x = grad_padded[(slice(None), slice(None)) + inside]
    return out, grad_x, grad_w, g.sum(axis=(0, 2, 3, 4))


def _same(got, want, what):
    got = np.ascontiguousarray(got)
    want = np.ascontiguousarray(want, dtype=np.float32)
    assert got.dtype == want.dtype and got.shape == want.shape, what
    differ = int((got.view(np.uint32) != want.view(np.uint32)).sum())
    assert differ == 0, f"{what}: {differ} of {got.size} entries differ"


CASES = [
    # input, weight, stride, padding, dilation, groups
    ((2, 3, 5, 6, 7), (4, 3, 3, 3, 3), (1, 1, 1), (1, 1, 1), (1, 1, 1), 1),
    ((1, 4, 6, 5, 5), (4, 2, 2, 3, 2), (2, 1, 2), (1, 0, 1), (2, 1, 1), 2),
    ((2, 4, 4, 5, 6), (4, 1, 3, 1, 3), (1, 1, 1), (1, 0, 1), (1, 1, 1), 4),
    ((1, 8, 4, 10, 12), (16, 8, 3, 3, 3), (1, 1, 1), (1, 1, 1), (1, 1, 1), 1),
    ((1, 2, 7, 4, 4), (3, 2, 5, 2, 2), (3, 2, 1), (2, 1, 0), (1, 1, 1), 1),
    ((2, 3, 3, 3, 3), (2, 3, 1, 2, 2), (1, 1, 1), (0, 1, 1), (1, 1, 1), 1),
]


@pytest.mark.parametrize("shape,weight_shape,stride,padding,dilation,groups", CASES)
def test_conv3d_and_its_gradients_round_once(
    shape, weight_shape, stride, padding, dilation, groups
):
    rng = np.random.default_rng(sum(shape) + 7 * sum(weight_shape))
    x, w = _values(rng, shape), _values(rng, weight_shape)
    b = _values(rng, (weight_shape[0],))
    volume = mt.tensor(x, requires_grad=True)
    kernel = mt.tensor(w, requires_grad=True)
    bias = mt.tensor(b, requires_grad=True)
    out = mt.nn.conv3d(volume, kernel, bias, stride, padding, dilation, groups)
    g = _values(rng, tuple(out.shape))
    (out * mt.tensor(g)).sum().backward()
    want, want_x, want_w, want_b = _exact(x, w, b, g, stride, padding, dilation, groups)
    _same(out.numpy(), want, "output")
    _same(volume.grad.numpy(), want_x, "input gradient")
    _same(kernel.grad.numpy(), want_w, "kernel gradient")
    _same(bias.grad.numpy(), want_b, "bias gradient")


def test_a_bias_of_another_dtype_is_added_after_in_the_wider_one():
    rng = np.random.default_rng(3)
    x, w = _values(rng, (1, 2, 3, 4, 4)), _values(rng, (3, 2, 2, 2, 2))
    b = rng.standard_normal(3)
    got = mt.nn.conv3d(mt.tensor(x), mt.tensor(w), b)
    assert "float64" in str(got.dtype)
    plain = mt.nn.conv3d(mt.tensor(x), mt.tensor(w)).numpy().astype(np.float64)
    np.testing.assert_array_equal(got.numpy(), plain + b.reshape(1, 3, 1, 1, 1))
    # A list carries no dtype of its own and takes the input's.
    listed = mt.nn.conv3d(mt.tensor(x), mt.tensor(w), [0.5, -1.0, 2.0])
    assert "float32" in str(listed.dtype)
    want, *_ = _exact(
        x,
        w,
        np.array([0.5, -1.0, 2.0]),
        np.zeros(listed.shape),
        (1, 1, 1),
        (0, 0, 0),
        (1, 1, 1),
        1,
    )
    _same(listed.numpy(), want, "output with a listed bias")


def test_a_zero_entry_is_a_positive_zero():
    x = np.zeros((1, 1, 3, 3, 3), dtype=np.float32)
    x[0, 0, 1, 1, 1] = 2.0
    w = np.full((1, 1, 2, 2, 2), -0.0, dtype=np.float32)
    out = mt.nn.conv3d(mt.tensor(x), mt.tensor(w), padding=1).numpy()
    assert (out == 0).all() and not np.signbit(out).any()


def test_float64_volumes_convolve_as_planes_do():
    rng = np.random.default_rng(9)
    x = rng.standard_normal((1, 2, 4, 5, 5))
    w = rng.standard_normal((3, 2, 3, 3, 3))
    want, *_ = _exact(
        x, w, None, np.zeros((1, 3, 2, 3, 3)), (1, 1, 1), (0,) * 3, (1,) * 3, 1
    )
    got = mt.nn.conv3d(mt.tensor(x), mt.tensor(w)).numpy()
    np.testing.assert_allclose(got, want, rtol=1e-13, atol=1e-13)
