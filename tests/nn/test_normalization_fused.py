# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""`layer_norm` computes its statistics and output in one fused pass per row.

It used to be six full-size tensor operations -- mean, sub, mul, mean, sqrt,
div, then weight and bias -- each allocating and traversing a tensor the size of
the input, and three of them going through the broadcasting path because the
statistics carry a trailing 1. On a 32x128x512 float32 tensor that measured
18.4ms, against 0.47ms for a single `mean` over the same data; fused it is
3.9ms.

The normalized dimensions are trailing and contiguous, so the input is
`[rows, norm]` in memory and each row is reduced and written while it is still
in L1.
"""

import numpy as np
import pytest

import minitensor as mt


def _reference(x, weight=None, bias=None, eps=1e-5, axes=1):
    """LayerNorm in float64, over the trailing `axes` dimensions."""
    a = np.asarray(x, dtype=np.float64)
    dims = tuple(range(a.ndim - axes, a.ndim))
    mean = a.mean(dims, keepdims=True)
    var = a.var(dims, keepdims=True)
    out = (a - mean) / np.sqrt(var + eps)
    if weight is not None:
        out = out * np.asarray(weight, dtype=np.float64)
    if bias is not None:
        out = out + np.asarray(bias, dtype=np.float64)
    return out


@pytest.mark.parametrize("shape,norm", [((4, 8), [8]), ((3, 5, 16), [16]), ((2, 3, 4, 7), [7])])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_layer_norm_matches_the_float64_reference(shape, norm, dtype):
    rng = np.random.default_rng(sum(shape) + len(dtype))
    x = rng.standard_normal(shape).astype(dtype)
    w = rng.standard_normal(norm).astype(dtype)
    b = rng.standard_normal(norm).astype(dtype)

    got = mt.layer_norm(mt.from_numpy(x), norm, mt.from_numpy(w), mt.from_numpy(b)).numpy()
    want = _reference(x, w, b)
    tolerance = 3e-6 if dtype == "float32" else 1e-12
    np.testing.assert_allclose(got, want.astype(dtype), rtol=tolerance, atol=tolerance)


@pytest.mark.parametrize("axes", [1, 2])
def test_layer_norm_over_several_trailing_axes(axes):
    rng = np.random.default_rng(4)
    x = rng.standard_normal((3, 4, 6)).astype(np.float32)
    norm = list(x.shape[-axes:])
    got = mt.layer_norm(mt.from_numpy(x), norm).numpy()
    np.testing.assert_allclose(got, _reference(x, axes=axes), rtol=3e-6, atol=3e-6)


def test_layer_norm_survives_a_mean_that_dominates_the_spread():
    """The case a one-pass `E[x^2] - E[x]^2` variance would destroy.

    Values near 1e4 with unit spread: `E[x^2]` and `E[x]^2` are both about 1e8,
    where the float32 ulp is 8, so subtracting them yields 8.0 against a true
    variance of 0.942. The fused kernel takes two reduction passes over the row
    precisely to avoid that, and the row is in L1 so the second is nearly free.

    The offset has to stay where the spread is still representable: at 1e6 the
    float32 ulp is 0.0625 and a spread of 1e-2 would simply not exist.
    """
    rng = np.random.default_rng(5)
    x = (1e4 + rng.standard_normal((32, 256))).astype(np.float32)
    got = mt.layer_norm(mt.from_numpy(x), [256]).numpy()
    np.testing.assert_allclose(got, _reference(x), atol=1e-4)
    assert abs(got.mean()) < 1e-4, "rows are not centred"
    np.testing.assert_allclose(got.std(-1), 1.0, rtol=1e-3)


def test_layer_norm_gradients_still_flow():
    rng = np.random.default_rng(6)
    xv = rng.standard_normal((5, 9)).astype(np.float32)
    wv = rng.standard_normal(9).astype(np.float32)
    bv = rng.standard_normal(9).astype(np.float32)
    x = mt.Tensor(xv, dtype="float32", requires_grad=True)
    w = mt.Tensor(wv, dtype="float32", requires_grad=True)
    b = mt.Tensor(bv, dtype="float32", requires_grad=True)
    mt.layer_norm(x, [9], w, b).sum().backward()

    # d/db of sum(y) is 1 per feature, summed over rows; d/dw is sum of the
    # normalized values. Both are exact enough to pin without finite differences.
    np.testing.assert_allclose(b.grad.numpy(), np.full(9, 5.0, dtype=np.float32), rtol=1e-6)
    normalized = _reference(xv)
    np.testing.assert_allclose(w.grad.numpy(), normalized.sum(0), rtol=1e-4, atol=1e-5)
    mt.clear_autograd_graph()


def test_layer_norm_input_gradient_vanishes_without_a_weight():
    """Summing an unweighted LayerNorm output is invariant to the input.

    Every row is centred and scaled to unit variance, so `sum(y)` is 0 whatever
    the row was -- the input gradient must come back ~0. It is a real check that
    the coupling through the mean and the variance is both present and correctly
    signed; drop either term and this is nowhere near zero. With a weight the
    invariance does not hold, which is why it is tested separately.
    """
    rng = np.random.default_rng(6)
    x = mt.Tensor(rng.standard_normal((5, 9)).astype(np.float32), dtype="float32", requires_grad=True)
    mt.layer_norm(x, [9]).sum().backward()
    assert np.abs(x.grad.numpy()).max() < 1e-5, x.grad.numpy()
    mt.clear_autograd_graph()


def test_layer_norm_keeps_shape_over_an_empty_axis():
    for shape in [(0, 4), (3, 0)]:
        out = mt.layer_norm(mt.from_numpy(np.zeros(shape, dtype=np.float32)), [shape[-1]])
        assert out.shape == shape
