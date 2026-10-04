# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""A convolution takes one of two paths, and both have to give the definition.

A group with few output channels -- a depthwise convolution has one, a first
layer reading an image few more -- is computed straight from the input, and
everything else is lowered to columns and multiplied. Which one a convolution
takes depends on its shape, so the cases below are chosen in pairs on either
side of that line, and each is held, forward and all three gradients, to a
float64 reference written from the definition rather than to the other path.
The small shapes most tests use land on the direct side; the lowered cases are
here so that path stays covered.
"""

from __future__ import annotations

import numpy as np
import pytest

import minitensor as mt


def _windows(x, kernel, stride, padding, dilation):
    """`windows[n, c, i, j, ki, kj]`: the input each tap of each output reads."""
    (kh, kw), (sh, sw), (ph, pw), (dh, dw) = kernel, stride, padding, dilation
    padded = np.pad(x, ((0, 0), (0, 0), (ph, ph), (pw, pw)))
    out_h = (padded.shape[2] - dh * (kh - 1) - 1) // sh + 1
    out_w = (padded.shape[3] - dw * (kw - 1) - 1) // sw + 1
    rows = (np.arange(out_h) * sh)[:, None] + np.arange(kh) * dh
    cols = (np.arange(out_w) * sw)[:, None] + np.arange(kw) * dw
    picked = padded[:, :, rows[:, None, :, None], cols[None, :, None, :]]
    return padded, picked, rows, cols


def _reference(x, w, b, go, stride, padding, dilation, groups):
    """Output, and the gradients of `sum(output * go)`, from the definition."""
    o_, cg, kh, kw = w.shape
    padded, picked, rows, cols = _windows(x, (kh, kw), stride, padding, dilation)
    per_group = o_ // groups
    n_, _, out_h, out_w, _, _ = picked.shape
    out = np.empty((n_, o_, out_h, out_w))
    grad_w = np.empty_like(w)
    grad_padded = np.zeros_like(padded)
    for g in range(groups):
        reads = picked[:, g * cg : (g + 1) * cg]
        kernels = w[g * per_group : (g + 1) * per_group]
        signal = go[:, g * per_group : (g + 1) * per_group]
        out[:, g * per_group : (g + 1) * per_group] = np.einsum(
            "ncijab,ocab->noij", reads, kernels
        )
        grad_w[g * per_group : (g + 1) * per_group] = np.einsum(
            "ncijab,noij->ocab", reads, signal
        )
        spread = np.einsum("noij,ocab->ncijab", signal, kernels)
        for a in range(kh):
            for c in range(kw):
                np.add.at(
                    grad_padded,
                    (
                        slice(None),
                        slice(g * cg, (g + 1) * cg),
                        rows[:, a][:, None],
                        cols[:, c][None, :],
                    ),
                    spread[..., a, c],
                )
    out += b[None, :, None, None]
    ph, pw = padding
    grad_x = grad_padded[:, :, ph : ph + x.shape[2], pw : pw + x.shape[3]]
    return out, grad_x, grad_w, go.sum(axis=(0, 2, 3))


# (input shape, weight shape, stride, padding, dilation, groups), in pairs: the
# first of each goes straight from the input, the second through the columns.
CASES = {
    "depthwise": ((2, 6, 9, 11), (6, 1, 3, 3), (1, 1), (1, 1), (1, 1), 6),
    "grouped, many per group": (
        (2, 6, 9, 11),
        (48, 2, 3, 3),
        (1, 1),
        (1, 1),
        (1, 1),
        3,
    ),
    "few outputs, strided": ((2, 3, 13, 12), (4, 3, 3, 3), (2, 2), (1, 0), (1, 1), 1),
    "eight outputs, strided": ((2, 3, 13, 12), (8, 3, 3, 3), (2, 2), (1, 0), (1, 1), 1),
    "eight outputs, wide rows": (
        (2, 3, 8, 17),
        (8, 3, 3, 3),
        (1, 1),
        (1, 1),
        (1, 1),
        1,
    ),
    "eight outputs, short rows": (
        (2, 3, 8, 9),
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
    "sixteen outputs, three channels": (
        (2, 3, 7, 9),
        (16, 3, 3, 3),
        (1, 1),
        (1, 1),
        (1, 1),
        1,
    ),
    "few outputs, dilated": ((2, 3, 10, 16), (2, 3, 3, 3), (1, 1), (2, 2), (2, 2), 1),
    "many outputs, dilated": ((2, 3, 10, 16), (20, 3, 3, 3), (1, 1), (2, 2), (2, 2), 1),
    "few outputs, wide kernel strided": (
        (1, 17, 9, 9),
        (3, 17, 3, 3),
        (1, 2),
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
}


@pytest.mark.parametrize("case", list(CASES))
def test_the_convolution_and_its_gradients_match_the_definition(case):
    x_shape, w_shape, stride, padding, dilation, groups = CASES[case]
    rng = np.random.default_rng(sum(map(ord, case)))
    x0 = rng.standard_normal(x_shape)
    w0 = rng.standard_normal(w_shape)
    b0 = rng.standard_normal(w_shape[0])

    x = mt.Tensor(x0, dtype="float64", requires_grad=True)
    w = mt.Tensor(w0, dtype="float64", requires_grad=True)
    b = mt.Tensor(b0, dtype="float64", requires_grad=True)
    out = mt.nn.conv2d(
        x, w, b, stride=stride, padding=padding, dilation=dilation, groups=groups
    )
    go = rng.standard_normal(out.shape)
    (out * mt.Tensor(go, dtype="float64")).sum().backward()

    want = _reference(x0, w0, b0, go, stride, padding, dilation, groups)
    for got, expected, name in zip(
        (out.numpy(), x.grad.numpy(), w.grad.numpy(), b.grad.numpy()),
        want,
        ("output", "input gradient", "weight gradient", "bias gradient"),
    ):
        np.testing.assert_allclose(got, expected, rtol=1e-12, atol=1e-12, err_msg=name)


def test_the_two_paths_agree_in_float32():
    """A depthwise convolution against the same channels run one at a time with
    thirty-two output channels each, which goes through the columns."""
    rng = np.random.default_rng(5)
    x0 = rng.standard_normal((3, 4, 20, 20)).astype(np.float32)
    w0 = rng.standard_normal((4, 1, 3, 3)).astype(np.float32)
    depthwise = mt.nn.conv2d(
        mt.Tensor(x0, dtype="float32"),
        mt.Tensor(w0, dtype="float32"),
        None,
        padding=1,
        groups=4,
    ).numpy()
    for c in range(4):
        # The wanted kernel thirty-two times, so the group is that wide.
        wide = np.repeat(w0[c : c + 1], 32, axis=0)
        lowered = mt.nn.conv2d(
            mt.Tensor(np.ascontiguousarray(x0[:, c : c + 1]), dtype="float32"),
            mt.Tensor(np.ascontiguousarray(wide), dtype="float32"),
            None,
            padding=1,
        ).numpy()
        np.testing.assert_allclose(lowered[:, 0], depthwise[:, c], rtol=2e-6, atol=2e-6)
