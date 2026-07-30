# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""Differential and finite-difference correctness tests.

These complement the repository's value-based tests with two categorically
stronger, library-agnostic checks:

* **Differential**: forward ops are compared against NumPy across dtypes,
  broadcasting shapes, reductions, and batched matmul.
* **Finite-difference gradcheck**: analytic gradients from ``backward()`` are
  compared against central-difference numerical gradients. This validates the
  autograd graph end to end without hand-derived expected values, and in
  particular guards the frozen-input gradient gating and the macro-generated
  kernels — a regression in either would show up as an analytic/numeric
  mismatch here.
"""

from __future__ import annotations

import numpy as np
import pytest

import minitensor as mt


def _np(x):
    return np.asarray(x, dtype=np.float64)


def assert_close(got, want, rtol=1e-4, atol=1e-5):
    g, w = _np(got), _np(want)
    assert g.shape == w.shape, f"shape {g.shape} != {w.shape}"
    np.testing.assert_allclose(g.ravel(), w.ravel(), rtol=rtol, atol=atol)


# --------------------------------------------------------------------------- #
# Differential forward checks vs NumPy
# --------------------------------------------------------------------------- #


@pytest.fixture(autouse=True)
def _seed():
    np.random.seed(1234)
    mt.manual_seed(1234)
    yield
    mt.clear_autograd_graph()


def test_elementwise_and_broadcasting():
    a = np.random.randn(4, 5).astype(np.float32)
    b = np.random.randn(4, 5).astype(np.float32)
    col = np.random.randn(4, 1).astype(np.float32)
    row = np.random.randn(1, 5).astype(np.float32)
    xa, xb = mt.from_numpy(a), mt.from_numpy(b)
    assert_close((xa + xb).numpy(), a + b)
    assert_close((xa - xb).numpy(), a - b)
    assert_close((xa * xb).numpy(), a * b)
    assert_close((xa / xb).numpy(), a / b)
    assert_close((xa * mt.from_numpy(col)).numpy(), a * col)
    assert_close((xa + mt.from_numpy(row)).numpy(), a + row)


@pytest.mark.parametrize(
    "name,mt_fn,np_fn,positive",
    [
        ("exp", lambda t: t.exp(), np.exp, False),
        ("log", lambda t: t.log(), np.log, True),
        ("sin", lambda t: t.sin(), np.sin, False),
        ("cos", lambda t: t.cos(), np.cos, False),
        ("tanh", lambda t: t.tanh(), np.tanh, False),
        ("sigmoid", lambda t: t.sigmoid(), lambda z: 1.0 / (1.0 + np.exp(-z)), False),
        ("abs", lambda t: t.abs(), np.abs, False),
    ],
)
def test_unary_math(name, mt_fn, np_fn, positive):
    src = np.random.randn(4, 5).astype(np.float32)
    if positive:
        src = np.abs(src) + 0.1
    assert_close(mt_fn(mt.from_numpy(src.copy())).numpy(), np_fn(src))


def test_reductions_match_numpy():
    t = np.random.randn(2, 3, 4).astype(np.float32)
    x = mt.from_numpy(t)
    assert_close(x.sum(dim=1, keepdim=True).numpy(), t.sum(1, keepdims=True))
    assert_close(x.mean().numpy(), t.mean())
    assert_close(x.max(dim=2)[0].numpy(), t.max(2))
    assert_close(x.argmax(dim=1).numpy(), t.argmax(1))
    assert_close(x.std(dim=0, unbiased=False).numpy(), t.std(0))
    assert_close(x.var(dim=1, unbiased=False).numpy(), t.var(1))
    assert_close(x.prod(dim=2).numpy(), t.prod(2))


def test_int_reductions():
    ia = np.random.randint(-4, 5, (3, 4)).astype(np.int64)
    xi = mt.from_numpy(ia)
    assert_close(xi.sum().numpy(), ia.sum())
    assert_close(xi.prod(dim=0).numpy(), ia.prod(0))
    assert_close(xi.prod(dim=1).numpy(), ia.prod(1))


def test_batched_matmul_and_transpose():
    m1 = np.random.randn(2, 3, 4).astype(np.float32)
    m2 = np.random.randn(2, 4, 5).astype(np.float32)
    assert_close(mt.from_numpy(m1).matmul(mt.from_numpy(m2)).numpy(), m1 @ m2)
    a = np.random.randn(4, 5).astype(np.float32)
    assert_close(mt.from_numpy(a).transpose(0, 1).numpy(), a.T)


# --------------------------------------------------------------------------- #
# Finite-difference gradient checks
# --------------------------------------------------------------------------- #


def _analytic_grad(fn, src):
    x = mt.from_numpy(src.copy())
    x.requires_grad_(True)
    fn(x).sum().backward()
    g = mt.get_gradient(x).numpy()
    mt.clear_autograd_graph()
    return g


def _numeric_grad(fn, src, eps=1e-3):
    grad = np.zeros_like(src, dtype=np.float64)
    flat = src.reshape(-1).astype(np.float64)
    for i in range(flat.size):
        plus = flat.copy()
        plus[i] += eps
        minus = flat.copy()
        minus[i] -= eps
        fp = float(
            fn(mt.from_numpy(plus.reshape(src.shape).astype(np.float32))).sum().numpy()
        )
        fm = float(
            fn(mt.from_numpy(minus.reshape(src.shape).astype(np.float32))).sum().numpy()
        )
        grad.reshape(-1)[i] = (fp - fm) / (2 * eps)
    return grad


@pytest.mark.parametrize(
    "name,fn",
    [
        ("exp", lambda x: x.exp()),
        ("tanh", lambda x: x.tanh()),
        ("sigmoid", lambda x: x.sigmoid()),
        ("square", lambda x: x * x),
        ("scale", lambda x: x * 3.0),
        ("affine", lambda x: x * 2.0 + 1.0),
        ("sum_of_sin", lambda x: x.sin()),
    ],
)
def test_gradcheck_unary(name, fn):
    src = (np.random.randn(3, 3).astype(np.float32)) * 0.5
    assert_close(_analytic_grad(fn, src), _numeric_grad(fn, src), rtol=3e-2, atol=3e-2)


def test_gradcheck_matmul_lhs():
    # d/dA sum(A @ W): finite-difference over A validates MatMulBackward.
    w = np.random.randn(4, 3).astype(np.float32)
    wt = mt.from_numpy(w)
    fn = lambda x: x.matmul(wt)  # noqa: E731
    src = np.random.randn(2, 4).astype(np.float32) * 0.5
    assert_close(_analytic_grad(fn, src), _numeric_grad(fn, src), rtol=3e-2, atol=3e-2)


def test_gradcheck_mse_loss_predictions_only():
    # The loss-gradient gating means only predictions accumulate a gradient;
    # finite-difference over the predictions must still match analytic.
    target = np.random.randn(3, 4).astype(np.float32)
    tgt = mt.from_numpy(target)
    from minitensor import nn

    mse = nn.MSELoss()
    fn = lambda x: mse(x, tgt)  # noqa: E731
    src = np.random.randn(3, 4).astype(np.float32) * 0.5
    assert_close(_analytic_grad(fn, src), _numeric_grad(fn, src), rtol=3e-2, atol=3e-2)


# --------------------------------------------------------------------------- #
# Gradcheck of the paths this branch changed most:
# reduce_gradient_for_broadcasting, the accumulate_grad path, reduction and
# shape backward. A regression in the frozen-input gating or broadcast
# reduction would surface here as an analytic/numeric mismatch.
# --------------------------------------------------------------------------- #


def test_gradcheck_broadcasting_backward():
    # Gradient must be reduced back to the broadcast operand's shape.
    row = mt.from_numpy(np.random.randn(1, 4).astype(np.float32))
    col = mt.from_numpy(np.random.randn(3, 1).astype(np.float32))
    base = np.random.randn(3, 4).astype(np.float32) * 0.5
    for fn in (lambda x: x + row, lambda x: x * col):
        assert_close(
            _analytic_grad(fn, base), _numeric_grad(fn, base), rtol=3e-2, atol=3e-2
        )


def test_gradcheck_grad_of_broadcast_operand():
    # The differentiated tensor is itself the broadcast (1x4) operand.
    other = mt.from_numpy(np.random.randn(3, 4).astype(np.float32))
    src = np.random.randn(1, 4).astype(np.float32) * 0.5
    fn = lambda x: x * other  # noqa: E731
    assert_close(_analytic_grad(fn, src), _numeric_grad(fn, src), rtol=3e-2, atol=3e-2)


@pytest.mark.parametrize(
    "name,fn",
    [
        ("x_times_x", lambda x: x * x),
        ("x_plus_x", lambda x: x + x),
        ("x_times_x_plus_x", lambda x: x * x + x),
    ],
)
def test_gradcheck_shared_input_accumulation(name, fn):
    # A tensor used as both operands must accumulate both contributions.
    src = np.random.randn(3, 3).astype(np.float32) * 0.5
    assert_close(_analytic_grad(fn, src), _numeric_grad(fn, src), rtol=3e-2, atol=3e-2)


@pytest.mark.parametrize(
    "name,fn",
    [
        ("sum_dim1", lambda x: x.sum(dim=1)),
        ("mean_dim2", lambda x: x.mean(dim=2)),
        ("sum_all", lambda x: x.sum()),
        ("reshape", lambda x: x.reshape(24)),
        ("transpose", lambda x: x.transpose(0, 2)),
    ],
)
def test_gradcheck_reduction_and_shape_backward(name, fn):
    src = np.random.randn(2, 3, 4).astype(np.float32) * 0.5
    assert_close(_analytic_grad(fn, src), _numeric_grad(fn, src), rtol=3e-2, atol=3e-2)


def test_gradcheck_composite_mlp_layer():
    # (x @ W).tanh() end to end — chains matmul, activation, and reduction
    # backward through one graph.
    w = mt.from_numpy(np.random.randn(4, 3).astype(np.float32))
    fn = lambda x: x.matmul(w).tanh()  # noqa: E731
    src = np.random.randn(2, 4).astype(np.float32) * 0.3
    assert_close(_analytic_grad(fn, src), _numeric_grad(fn, src), rtol=3e-2, atol=3e-2)


# --------------------------------------------------------------------------- #
# Step functions are constants, not phantom leaves
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("op", ["ceil", "floor", "round", "sign"])
def test_step_functions_produce_constants_not_phantom_leaves(op):
    # These have zero derivative wherever it exists, so no gradient is recorded
    # -- the same convention as norm(p=0). They used to copy the input's
    # requires_grad onto the output without attaching a gradient function, which
    # made the result a leaf that claimed to be differentiable: backward then
    # deposited a gradient on that intermediate, which no caller can use.
    x = mt.Tensor(np.array([1.5, -2.5, 0.3]), dtype="float64", requires_grad=True)
    y = getattr(x, op)()
    assert not y.requires_grad

    w = mt.Tensor(np.array([3.0, 3.0, 3.0]), dtype="float64", requires_grad=True)
    (y * w).sum().backward()

    assert y.grad is None, "a step-function result must not collect a gradient"
    # The other operand still trains normally.
    assert_close(w.grad.numpy(), _np(y.numpy()))
    mt.clear_autograd_graph()


def test_norm_zero_uses_the_same_convention():
    x = mt.Tensor(np.array([0.0, 1.0, 2.0]), dtype="float64", requires_grad=True)
    assert not x.norm(0.0).requires_grad


# --------------------------------------------------------------------------- #
# Finite-difference sweep over every differentiable op
# --------------------------------------------------------------------------- #

# Domains chosen so the op and its derivative are both defined and smooth over
# the sample; a finite difference across a kink or a pole is meaningless.
_ANY = np.random.default_rng(7).standard_normal(9) * 1.3
_POS = np.abs(np.random.default_rng(5).standard_normal(9)) + 0.4
_UNIT = np.random.default_rng(9).uniform(-0.85, 0.85, 9)
_GT1 = np.abs(np.random.default_rng(11).standard_normal(9)) + 1.4

_GRADCHECK_OPS = [
    ("abs", lambda t: t.abs(), _ANY),
    ("acos", lambda t: t.acos(), _UNIT),
    ("acosh", lambda t: t.acosh(), _GT1),
    ("asin", lambda t: t.asin(), _UNIT),
    ("asinh", lambda t: t.asinh(), _ANY),
    ("atan", lambda t: t.atan(), _ANY),
    ("atanh", lambda t: t.atanh(), _UNIT),
    ("cos", lambda t: t.cos(), _ANY),
    ("cosh", lambda t: t.cosh(), _ANY),
    ("cumprod", lambda t: t.cumprod(0), _POS),
    ("cumsum", lambda t: t.cumsum(0), _ANY),
    ("elu", lambda t: t.elu(), _ANY),
    ("erf", lambda t: t.erf(), _ANY),
    ("erfc", lambda t: t.erfc(), _ANY),
    ("exp", lambda t: t.exp(), _ANY),
    ("expm1", lambda t: t.expm1(), _ANY),
    ("gelu", lambda t: t.gelu(), _ANY),
    ("hardshrink", lambda t: t.hardshrink(lambd=0.3), _ANY),
    ("log", lambda t: t.log(), _POS),
    ("log10", lambda t: t.log10(), _POS),
    ("log1p", lambda t: t.log1p(), _POS),
    ("log2", lambda t: t.log2(), _POS),
    ("log_softmax", lambda t: t.log_softmax(dim=0), _ANY),
    ("logsumexp", lambda t: t.logsumexp(dim=0), _ANY),
    ("max", lambda t: t.max(), _ANY),
    ("mean", lambda t: t.mean(), _ANY),
    ("median", lambda t: t.median(), _ANY),
    ("min", lambda t: t.min(), _ANY),
    ("nanmean", lambda t: t.nanmean(), _ANY),
    ("nanmedian", lambda t: t.nanmedian(), _ANY),
    ("nansum", lambda t: t.nansum(), _ANY),
    ("nanmax", lambda t: t.nanmax(), _ANY),
    ("nanmin", lambda t: t.nanmin(), _ANY),
    ("nanquantile", lambda t: t.nanquantile(0.5), _ANY),
    ("norm1", lambda t: t.norm(1.0), _ANY),
    ("norm2", lambda t: t.norm(2.0), _ANY),
    ("norm3", lambda t: t.norm(3.0), _ANY),
    ("pow2", lambda t: t.pow(2.0), _ANY),
    ("pow3", lambda t: t.pow(3.0), _ANY),
    ("pow_half", lambda t: t.pow(0.5), _POS),
    ("prod", lambda t: t.prod(), _POS),
    ("quantile", lambda t: t.quantile(0.5), _ANY),
    ("reciprocal", lambda t: t.reciprocal(), _POS),
    ("relu", lambda t: t.relu(), _ANY),
    ("rsqrt", lambda t: t.rsqrt(), _POS),
    ("selu", lambda t: t.selu(), _ANY),
    ("sigmoid", lambda t: t.sigmoid(), _ANY),
    ("silu", lambda t: t.silu(), _ANY),
    ("sin", lambda t: t.sin(), _ANY),
    ("sinh", lambda t: t.sinh(), _ANY),
    ("softmax", lambda t: t.softmax(dim=0), _ANY),
    ("softplus", lambda t: t.softplus(), _ANY),
    ("softsign", lambda t: t.softsign(), _ANY),
    ("sqrt", lambda t: t.sqrt(), _POS),
    ("std", lambda t: t.std(), _ANY),
    ("sum", lambda t: t.sum(), _ANY),
    ("tan", lambda t: t.tan(), _UNIT),
    ("tanh", lambda t: t.tanh(), _ANY),
    ("var", lambda t: t.var(), _ANY),
    ("clamp", lambda t: t.clamp(-0.5, 0.5), _ANY),
    ("nan_to_num", lambda t: t.nan_to_num(), _ANY),
    ("flip", lambda t: t.flip([0]), _ANY),
    ("roll", lambda t: t.roll(1, 0), _ANY),
    ("sort", lambda t: t.sort()[0], _ANY),
    ("topk", lambda t: t.topk(4)[0], _ANY),
    ("trace", lambda t: t.reshape((3, 3)).trace(), _ANY),
    ("triu", lambda t: t.reshape((3, 3)).triu(0), _ANY),
    ("tril", lambda t: t.reshape((3, 3)).tril(0), _ANY),
    ("diagonal", lambda t: t.reshape((3, 3)).diagonal(), _ANY),
    ("matmul", lambda t: t.reshape((3, 3)).matmul(t.reshape((3, 3))), _ANY),
]


def _analytic_grad_f64(fn, src):
    x = mt.Tensor(src.copy(), dtype="float64", requires_grad=True)
    fn(x).sum().backward()
    g = x.grad.numpy().copy()
    mt.clear_autograd_graph()
    return g


def _numeric_grad_f64(fn, src, eps=1e-5):
    flat = src.reshape(-1).astype(np.float64)
    grad = np.zeros_like(flat)
    for i in range(flat.size):
        plus, minus = flat.copy(), flat.copy()
        plus[i] += eps
        minus[i] -= eps
        fp = fn(mt.Tensor(plus.reshape(src.shape), dtype="float64")).sum().item()
        fm = fn(mt.Tensor(minus.reshape(src.shape), dtype="float64")).sum().item()
        grad[i] = (fp - fm) / (2 * eps)
    return grad.reshape(src.shape)


@pytest.mark.parametrize(
    "name,fn,src", _GRADCHECK_OPS, ids=[o[0] for o in _GRADCHECK_OPS]
)
def test_gradcheck_every_differentiable_op(name, fn, src):
    analytic = _analytic_grad_f64(fn, src)
    numeric = _numeric_grad_f64(fn, src)
    assert analytic.shape == numeric.shape
    # A NaN on the analytic side where the numeric gradient is finite means the
    # backward invented one -- that is how the nanmedian gradient bug looked.
    assert not (
        np.isnan(analytic) & ~np.isnan(numeric)
    ).any(), f"{name}: analytic gradient has NaN where the numeric one does not"
    np.testing.assert_allclose(analytic, numeric, rtol=2e-3, atol=2e-3)


# --------------------------------------------------------------------------- #
# Convolution in double precision
# --------------------------------------------------------------------------- #


def _conv2d_reference(x, w, b, stride, padding):
    """Explicit cross-correlation, independent of the im2col + GEMM lowering."""
    n, c_in, h, ww = x.shape
    c_out, _, kh, kw = w.shape
    padded = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)))
    out_h = (h + 2 * padding - kh) // stride + 1
    out_w = (ww + 2 * padding - kw) // stride + 1
    out = np.zeros((n, c_out, out_h, out_w))
    for i in range(n):
        for co in range(c_out):
            for oh in range(out_h):
                for ow in range(out_w):
                    window = padded[
                        i,
                        :,
                        oh * stride : oh * stride + kh,
                        ow * stride : ow * stride + kw,
                    ]
                    out[i, co, oh, ow] = (window * w[co]).sum() + (
                        0 if b is None else b[co]
                    )
    return out


@pytest.mark.parametrize("stride,padding", [(1, 0), (2, 0), (1, 1), (2, 1)])
def test_conv2d_float64_matches_reference(stride, padding):
    rng = np.random.default_rng(20240726)
    x = rng.standard_normal((2, 2, 5, 5))
    w = rng.standard_normal((3, 2, 3, 3))
    b = rng.standard_normal(3)
    got = mt.functional.conv2d(
        mt.Tensor(x, dtype="float64"),
        mt.Tensor(w, dtype="float64"),
        mt.Tensor(b, dtype="float64"),
        stride,
        padding,
    )
    assert got.dtype == "float64"
    np.testing.assert_allclose(
        got.numpy(), _conv2d_reference(x, w, b, stride, padding), rtol=1e-12, atol=1e-12
    )


@pytest.mark.parametrize("which", ["input", "weight", "bias"])
def test_conv2d_float64_gradcheck(which):
    # Double precision lets the convolution gradients be checked at a step fine
    # enough to be meaningful; the float32-only implementation could only ever
    # be checked at 1e-2.
    rng = np.random.default_rng(20240727)
    src = {
        "input": rng.standard_normal((1, 2, 4, 4)),
        "weight": rng.standard_normal((2, 2, 3, 3)),
        "bias": rng.standard_normal(2),
    }
    fixed = {k: mt.Tensor(v, dtype="float64") for k, v in src.items() if k != which}

    def build(value):
        args = dict(fixed)
        args[which] = value
        return mt.functional.conv2d(args["input"], args["weight"], args["bias"])

    x = mt.Tensor(src[which], dtype="float64", requires_grad=True)
    build(x).sum().backward()
    analytic = x.grad.numpy().copy()
    mt.clear_autograd_graph()

    flat = src[which].reshape(-1)
    numeric = np.zeros_like(flat)
    eps = 1e-6
    for i in range(flat.size):
        plus, minus = flat.copy(), flat.copy()
        plus[i] += eps
        minus[i] -= eps
        fp = (
            build(mt.Tensor(plus.reshape(src[which].shape), dtype="float64"))
            .sum()
            .item()
        )
        fm = (
            build(mt.Tensor(minus.reshape(src[which].shape), dtype="float64"))
            .sum()
            .item()
        )
        numeric[i] = (fp - fm) / (2 * eps)
    np.testing.assert_allclose(
        analytic, numeric.reshape(src[which].shape), rtol=1e-5, atol=1e-6
    )
