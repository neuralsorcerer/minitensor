# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""`layer_norm` and `rms_norm` compute statistics and output in one fused pass.

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


@pytest.mark.parametrize(
    "shape,norm", [((4, 8), [8]), ((3, 5, 16), [16]), ((2, 3, 4, 7), [7])]
)
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_layer_norm_matches_the_float64_reference(shape, norm, dtype):
    rng = np.random.default_rng(sum(shape) + len(dtype))
    x = rng.standard_normal(shape).astype(dtype)
    w = rng.standard_normal(norm).astype(dtype)
    b = rng.standard_normal(norm).astype(dtype)

    got = mt.layer_norm(
        mt.from_numpy(x), norm, mt.from_numpy(w), mt.from_numpy(b)
    ).numpy()
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
    np.testing.assert_allclose(
        b.grad.numpy(), np.full(9, 5.0, dtype=np.float32), rtol=1e-6
    )
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
    x = mt.Tensor(
        rng.standard_normal((5, 9)).astype(np.float32),
        dtype="float32",
        requires_grad=True,
    )
    mt.layer_norm(x, [9]).sum().backward()
    assert np.abs(x.grad.numpy()).max() < 1e-5, x.grad.numpy()
    mt.clear_autograd_graph()


def test_layer_norm_keeps_shape_over_an_empty_axis():
    for shape in [(0, 4), (3, 0)]:
        out = mt.layer_norm(
            mt.from_numpy(np.zeros(shape, dtype=np.float32)), [shape[-1]]
        )
        assert out.shape == shape


# `rms_norm` got the same treatment, and needed more of it: unlike `layer_norm`
# it had no explicit gradient at all -- the forward was a chain of
# autograd-tracked primitives, so gradients fell out of the graph for free.
# Fusing the forward means writing the backward out, which is:
#
#     dL/dx_j = r * (g_j w_j - x_j r^2 dot / N),  dot = sum_i g_i w_i x_i
#     dL/dw_i = sum over rows of g_i x_i r
#
# with r = 1/sqrt(mean(x^2) + eps) saved from the forward. Checked below against
# that formula written independently in numpy, and against finite differences.
def _rms_reference(x, weight=None, eps=1e-5, axes=1):
    a = np.asarray(x, dtype=np.float64)
    dims = tuple(range(a.ndim - axes, a.ndim))
    out = a / np.sqrt((a * a).mean(dims, keepdims=True) + eps)
    if weight is not None:
        out = out * np.asarray(weight, dtype=np.float64)
    return out


@pytest.mark.parametrize(
    "shape,norm", [((4, 8), [8]), ((3, 5, 16), [16]), ((2, 3, 4, 7), [7])]
)
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_rms_norm_matches_the_float64_reference(shape, norm, dtype):
    rng = np.random.default_rng(sum(shape) * 7 + len(dtype))
    x = rng.standard_normal(shape).astype(dtype)
    w = rng.standard_normal(norm).astype(dtype)
    got = mt.rms_norm(mt.from_numpy(x), norm, mt.from_numpy(w), 1e-5).numpy()
    want = _rms_reference(x, w)
    tolerance = 3e-6 if dtype == "float32" else 1e-12
    np.testing.assert_allclose(got, want.astype(dtype), rtol=tolerance, atol=tolerance)


def test_rms_norm_without_a_weight():
    rng = np.random.default_rng(21)
    x = rng.standard_normal((6, 12)).astype(np.float32)
    got = mt.rms_norm(mt.from_numpy(x), [12], None, 1e-5).numpy()
    np.testing.assert_allclose(got, _rms_reference(x), rtol=3e-6, atol=3e-6)


def test_rms_norm_gradient_matches_the_analytic_formula():
    rng = np.random.default_rng(3)
    n = 17
    xv = rng.standard_normal((6, n))
    wv = rng.standard_normal(n)
    gv = rng.standard_normal((6, n))
    eps = 1e-5

    x = mt.Tensor(xv, dtype="float64", requires_grad=True)
    w = mt.Tensor(wv, dtype="float64", requires_grad=True)
    (mt.rms_norm(x, [n], w, eps) * mt.from_numpy(gv)).sum().backward()
    grad_x, grad_w = x.grad.numpy(), w.grad.numpy()
    mt.clear_autograd_graph()

    r = 1.0 / np.sqrt((xv * xv).mean(-1, keepdims=True) + eps)
    dot = (gv * wv * xv).sum(-1, keepdims=True)
    np.testing.assert_allclose(grad_x, r * (gv * wv - xv * r * r * dot / n), rtol=1e-11)
    np.testing.assert_allclose(grad_w, (gv * xv * r).sum(0), rtol=1e-11)


def test_rms_norm_gradient_matches_finite_differences():
    """Independent of the formula above, in case both were derived wrong."""
    rng = np.random.default_rng(9)
    n = 7
    xv = rng.standard_normal((3, n))
    wv = rng.standard_normal(n)
    gv = rng.standard_normal((3, n))
    eps, h = 1e-5, 1e-6

    x = mt.Tensor(xv, dtype="float64", requires_grad=True)
    (mt.rms_norm(x, [n], mt.from_numpy(wv), eps) * mt.from_numpy(gv)).sum().backward()
    grad_x = x.grad.numpy()
    mt.clear_autograd_graph()

    def loss(a):
        return (_rms_reference(a, wv, eps) * gv).sum()

    numeric = np.zeros_like(xv)
    for i in range(xv.shape[0]):
        for j in range(n):
            up, dn = xv.copy(), xv.copy()
            up[i, j] += h
            dn[i, j] -= h
            numeric[i, j] = (loss(up) - loss(dn)) / (2 * h)
    np.testing.assert_allclose(grad_x, numeric, rtol=1e-6, atol=1e-8)


def test_rms_norm_keeps_shape_over_an_empty_axis():
    for shape in [(0, 4), (3, 0)]:
        out = mt.rms_norm(mt.from_numpy(np.zeros(shape, dtype=np.float32)), [shape[-1]])
        assert out.shape == shape


# --- what an inference forward does not have to build ------------------------
#
# The normalized values LayerNorm computes exist only to be saved for the
# backward. The forward built them unconditionally -- a second full-size buffer,
# filled and then dropped -- because `requires_grad` was not consulted until
# after the kernel had run. Under `no_grad` that is a whole extra pass over the
# tensor and, at (4096, 1024) float32, 16 MB allocated to be thrown away.


@pytest.mark.parametrize(
    "shape,normalized_shape",
    [
        ((3, 5), [5]),
        ((8, 64, 256), [256]),
        ((2, 3, 4, 6), [4, 6]),
        ((1, 1), [1]),
    ],
)
def test_layer_norm_inference_matches_training_bit_for_bit(shape, normalized_shape):
    # Skipping the saved buffer must not change a single output bit: it is the
    # same arithmetic with one store removed.
    rng = np.random.default_rng(17)
    values = rng.standard_normal(shape).astype(np.float32)
    layer = mt.nn.LayerNorm(normalized_shape)

    trained = layer(mt.Tensor(values).requires_grad_(True)).numpy()
    with mt.no_grad():
        inferred = layer(mt.Tensor(values)).numpy()

    assert np.array_equal(trained, inferred)


def test_layer_norm_still_differentiates_after_the_forward_skips_the_buffer():
    rng = np.random.default_rng(18)
    values = rng.standard_normal((6, 8))
    layer = mt.nn.LayerNorm([8], dtype="float64")
    weight, bias = layer.parameters()[0].numpy(), layer.parameters()[1].numpy()

    tensor = mt.Tensor(values, dtype="float64").requires_grad_(True)
    out = layer(tensor)
    upstream = rng.standard_normal((6, 8))
    mt.sum(out * mt.Tensor(upstream, dtype="float64")).backward()
    analytic = mt.get_gradient(tensor).numpy()

    def forward(sample):
        mean = sample.mean(-1, keepdims=True)
        var = sample.var(-1, keepdims=True)
        return (sample - mean) / np.sqrt(var + 1e-5) * weight + bias

    eps = 1e-6
    for index in np.ndindex(*values.shape):
        plus, minus = values.copy(), values.copy()
        plus[index] += eps
        minus[index] -= eps
        numeric = ((forward(plus) - forward(minus)) * upstream).sum() / (2 * eps)
        assert abs(numeric - analytic[index]) < 1e-6 * max(1.0, abs(numeric)), index


def test_layer_norm_under_no_grad_produces_no_graph():
    layer = mt.nn.LayerNorm([16])
    x = mt.Tensor(np.zeros((4, 16), dtype=np.float32)).requires_grad_(True)

    mt.clear_autograd_graph()
    with mt.no_grad():
        out = layer(x)

    assert not out.requires_grad
    assert mt.autograd_graph_size() == (0, 0)
