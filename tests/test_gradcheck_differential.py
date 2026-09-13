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

import math
import pathlib
import re

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


@pytest.mark.parametrize("op", ["ceil", "floor", "round", "sign", "trunc"])
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
# `frac` jumps at every non-zero integer, so a central difference straddling one
# would compare a slope of 1 against a slope of -1e5. These sit at least 0.15
# away from any integer, and span several of them so `trunc` is not always zero.
# `logit` is only real on (0, 1), and its gradient `1/(x(1-x))` runs away at
# both ends, so a central difference needs room on either side.
_UNIT_OPEN = np.random.default_rng(13).uniform(0.15, 0.85, 9)
_NON_INTEGRAL = np.array([-3.4, -2.7, -1.25, -0.6, 0.3, 1.45, 2.8, 3.15, 4.5])

_GRADCHECK_OPS = [
    ("abs", lambda t: t.abs(), _ANY),
    ("absolute", lambda t: t.absolute(), _ANY),
    ("conj", lambda t: t.conj(), _ANY),
    ("real", lambda t: t.real(), _ANY),
    ("t", lambda t: t.t(), _ANY),
    ("vander", lambda t: t.vander(), _POS),
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
    ("erfinv", lambda t: t.erfinv(), _UNIT),
    ("erfcx", lambda t: t.erfcx(), _ANY),
    ("exp2", lambda t: t.exp2(), _ANY),
    ("sinc", lambda t: t.sinc(), _ANY),
    ("lgamma", lambda t: t.lgamma(), _POS),
    ("digamma", lambda t: t.digamma(), _POS),
    ("i0", lambda t: t.i0(), _ANY),
    ("i1", lambda t: t.i1(), _ANY),
    ("i0e", lambda t: t.i0e(), _ANY),
    ("i1e", lambda t: t.i1e(), _ANY),
    ("logit", lambda t: t.logit(), _UNIT_OPEN),
    ("exp", lambda t: t.exp(), _ANY),
    ("expm1", lambda t: t.expm1(), _ANY),
    ("gelu", lambda t: t.gelu(), _ANY),
    ("celu", lambda t: t.celu(1.5), _ANY),
    ("hardsigmoid", lambda t: t.hardsigmoid(), _ANY),
    ("hardswish", lambda t: t.hardswish(), _ANY),
    ("hardtanh", lambda t: t.hardtanh(-1.0, 1.0), _ANY),
    ("logsigmoid", lambda t: t.logsigmoid(), _ANY),
    ("mish", lambda t: t.mish(), _ANY),
    ("relu6", lambda t: t.relu6(), _ANY),
    ("softmin", lambda t: t.softmin(dim=0), _ANY),
    ("softshrink", lambda t: t.softshrink(0.25), _ANY),
    ("tanhshrink", lambda t: t.tanhshrink(), _ANY),
    ("threshold", lambda t: t.threshold(0.5, -1.0), _ANY),
    ("frac", lambda t: t.frac(), _NON_INTEGRAL),
    ("neg", lambda t: t.neg(), _ANY),
    ("negative", lambda t: t.negative(), _ANY),
    ("square", lambda t: t.square(), _ANY),
    ("deg2rad", lambda t: t.deg2rad(), _ANY),
    ("rad2deg", lambda t: t.rad2deg(), _ANY),
    ("hardshrink", lambda t: t.hardshrink(lambd=0.3), _ANY),
    ("log", lambda t: t.log(), _POS),
    ("log10", lambda t: t.log10(), _POS),
    ("log1p", lambda t: t.log1p(), _POS),
    ("log2", lambda t: t.log2(), _POS),
    ("log_softmax", lambda t: t.log_softmax(dim=0), _ANY),
    ("logsumexp", lambda t: t.logsumexp(dim=0), _ANY),
    ("amax", lambda t: t.amax(0), _ANY),
    ("amin", lambda t: t.amin(0), _ANY),
    ("nanamax", lambda t: t.nanamax(0), _ANY),
    ("nanamin", lambda t: t.nanamin(0), _ANY),
    ("max", lambda t: t.max(), _ANY),
    ("mean", lambda t: t.mean(), _ANY),
    ("median", lambda t: t.median(), _ANY),
    ("min", lambda t: t.min(), _ANY),
    ("nanmean", lambda t: t.nanmean(), _ANY),
    ("nanmedian", lambda t: t.nanmedian(), _ANY),
    ("nansum", lambda t: t.nansum(), _ANY),
    ("nanprod", lambda t: t.nanprod(), _POS),
    ("nanvar", lambda t: t.nanvar(), _ANY),
    ("nanstd", lambda t: t.nanstd(), _ANY),
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
    # Found missing by the completeness check below. `leaky_relu` is the one
    # that mattered: its gradient boundary at exactly zero was changed without
    # any finite-difference check on it. The rest are shape and identity ops
    # whose backward is a pass-through -- cheap to cover, and silent if broken.
    ("clip", lambda t: t.clip(-0.5, 0.5), _ANY),
    ("clone", lambda t: t.clone(), _ANY),
    ("contiguous", lambda t: t.contiguous(), _ANY),
    ("cpu", lambda t: t.cpu(), _ANY),
    ("flatten", lambda t: t.reshape((3, 3)).flatten(), _ANY),
    ("leaky_relu", lambda t: t.leaky_relu(), _ANY),
    ("leaky_relu_slope", lambda t: t.leaky_relu(0.1), _ANY),
    ("ravel", lambda t: t.reshape((3, 3)).ravel(), _ANY),
    ("squeeze", lambda t: t.reshape((1, 9, 1)).squeeze(), _ANY),
    ("to", lambda t: t.to("float64"), _ANY),
    ("norm_default", lambda t: t.norm(), _ANY),
    # Placing values on a diagonal and reading them back are each other's
    # derivative, so both directions are checked rather than just the one.
    ("diag_from_vector", lambda t: t.diag(), _ANY),
    ("diag_from_matrix", lambda t: t.reshape((3, 3)).diag(), _ANY),
    ("diag_offset", lambda t: t.diag(2), _ANY),
    ("diag_embed", lambda t: t.diag_embed(), _ANY),
    ("diag_embed_offset", lambda t: t.diag_embed(-1), _ANY),
    ("diag_embed_batched", lambda t: t.reshape((3, 3)).diag_embed(), _ANY),
    # Composed from differentiable pieces rather than written as primitives, so
    # a gradient exists whether or not one was intended -- which is exactly the
    # case worth pinning. `cbrt` routes the sign around the fractional power,
    # `positive` is the identity, and `frexp`'s mantissa is a scaling by a power
    # of two that `floor` holds constant.
    ("cbrt", lambda t: t.cbrt(), _POS),
    ("positive", lambda t: t.positive(), _ANY),
    ("frexp_mantissa", lambda t: t.frexp()[0], _POS),
]

# Ops a no-arg probe reaches but this list deliberately does not gradcheck.
_GRADCHECK_EXEMPT = {
    # Covered with explicit arguments by their own tests above.
    "backward",
}


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


# --------------------------------------------------------------------------- #
# Forward values against NumPy
# --------------------------------------------------------------------------- #

_FWD_ANY = np.random.default_rng(101).standard_normal((4, 5))
_FWD_POS = np.abs(np.random.default_rng(102).standard_normal((4, 5))) + 0.3
_FWD_UNIT = np.random.default_rng(103).uniform(-0.9, 0.9, (4, 5))
_FWD_GT1 = np.abs(np.random.default_rng(104).standard_normal((4, 5))) + 1.3
# Square, and materialised: the tensor constructor requires a contiguous array.
_FWD_SQUARE = np.ascontiguousarray(_FWD_ANY[:, :4])
_FWD_3D = np.random.default_rng(105).standard_normal((2, 3, 4))
_FWD_POS3D = np.abs(np.random.default_rng(106).standard_normal((2, 3, 4))) + 0.4


def _erf(x):
    return np.vectorize(math.erf)(x)


def _lgamma(x):
    return np.vectorize(math.lgamma)(x)


_FORWARD_OPS = [
    ("abs", lambda t: t.abs(), _FWD_ANY, np.abs),
    ("acos", lambda t: t.acos(), _FWD_UNIT, np.arccos),
    ("acosh", lambda t: t.acosh(), _FWD_GT1, np.arccosh),
    ("asin", lambda t: t.asin(), _FWD_UNIT, np.arcsin),
    ("asinh", lambda t: t.asinh(), _FWD_ANY, np.arcsinh),
    ("atan", lambda t: t.atan(), _FWD_ANY, np.arctan),
    ("atanh", lambda t: t.atanh(), _FWD_UNIT, np.arctanh),
    ("ceil", lambda t: t.ceil(), _FWD_ANY, np.ceil),
    ("cos", lambda t: t.cos(), _FWD_ANY, np.cos),
    ("cosh", lambda t: t.cosh(), _FWD_ANY, np.cosh),
    ("erf", lambda t: t.erf(), _FWD_ANY, _erf),
    ("erfc", lambda t: t.erfc(), _FWD_ANY, lambda a: 1.0 - _erf(a)),
    ("lgamma", lambda t: t.lgamma(), _FWD_POS, _lgamma),
    ("exp", lambda t: t.exp(), _FWD_ANY, np.exp),
    ("exp2", lambda t: t.exp2(), _FWD_ANY, np.exp2),
    ("expm1", lambda t: t.expm1(), _FWD_ANY, np.expm1),
    ("floor", lambda t: t.floor(), _FWD_ANY, np.floor),
    ("frac", lambda t: t.frac(), _FWD_ANY, lambda a: a - np.trunc(a)),
    ("log", lambda t: t.log(), _FWD_POS, np.log),
    ("log10", lambda t: t.log10(), _FWD_POS, np.log10),
    ("log1p", lambda t: t.log1p(), _FWD_POS, np.log1p),
    ("log2", lambda t: t.log2(), _FWD_POS, np.log2),
    ("reciprocal", lambda t: t.reciprocal(), _FWD_POS, lambda a: 1.0 / a),
    ("round", lambda t: t.round(), _FWD_ANY, np.round),
    ("sinc", lambda t: t.sinc(), _FWD_ANY, np.sinc),
    ("rsqrt", lambda t: t.rsqrt(), _FWD_POS, lambda a: 1.0 / np.sqrt(a)),
    ("sign", lambda t: t.sign(), _FWD_ANY, np.sign),
    ("sin", lambda t: t.sin(), _FWD_ANY, np.sin),
    ("sinh", lambda t: t.sinh(), _FWD_ANY, np.sinh),
    ("sqrt", lambda t: t.sqrt(), _FWD_POS, np.sqrt),
    ("tan", lambda t: t.tan(), _FWD_UNIT, np.tan),
    ("tanh", lambda t: t.tanh(), _FWD_ANY, np.tanh),
    ("trunc", lambda t: t.trunc(), _FWD_ANY, np.trunc),
    ("sigmoid", lambda t: t.sigmoid(), _FWD_ANY, lambda a: 1 / (1 + np.exp(-a))),
    ("relu", lambda t: t.relu(), _FWD_ANY, lambda a: np.maximum(a, 0)),
    ("relu6", lambda t: t.relu6(), _FWD_ANY, lambda a: np.clip(a, 0.0, 6.0)),
    (
        "hardtanh",
        lambda t: t.hardtanh(-1.0, 1.0),
        _FWD_ANY,
        lambda a: np.clip(a, -1.0, 1.0),
    ),
    (
        "hardsigmoid",
        lambda t: t.hardsigmoid(),
        _FWD_ANY,
        lambda a: np.clip(a / 6.0 + 0.5, 0.0, 1.0),
    ),
    (
        "hardswish",
        lambda t: t.hardswish(),
        _FWD_ANY,
        lambda a: a * np.clip(a / 6.0 + 0.5, 0.0, 1.0),
    ),
    (
        "mish",
        lambda t: t.mish(),
        _FWD_ANY,
        lambda a: a * np.tanh(np.log1p(np.exp(a))),
    ),
    (
        "celu",
        lambda t: t.celu(1.5),
        _FWD_ANY,
        lambda a: np.where(a > 0, a, 1.5 * np.expm1(a / 1.5)),
    ),
    (
        "logsigmoid",
        lambda t: t.logsigmoid(),
        _FWD_ANY,
        lambda a: -np.log1p(np.exp(-a)),
    ),
    (
        "softshrink",
        lambda t: t.softshrink(0.5),
        _FWD_ANY,
        lambda a: np.sign(a) * np.maximum(np.abs(a) - 0.5, 0.0),
    ),
    ("tanhshrink", lambda t: t.tanhshrink(), _FWD_ANY, lambda a: a - np.tanh(a)),
    (
        "threshold",
        lambda t: t.threshold(0.5, -1.0),
        _FWD_ANY,
        lambda a: np.where(a > 0.5, a, -1.0),
    ),
    ("softplus", lambda t: t.softplus(), _FWD_ANY, lambda a: np.log1p(np.exp(a))),
    ("softsign", lambda t: t.softsign(), _FWD_ANY, lambda a: a / (1 + np.abs(a))),
    ("silu", lambda t: t.silu(), _FWD_ANY, lambda a: a / (1 + np.exp(-a))),
    ("elu", lambda t: t.elu(), _FWD_ANY, lambda a: np.where(a > 0, a, np.expm1(a))),
    (
        "selu",
        lambda t: t.selu(),
        _FWD_ANY,
        lambda a: 1.0507009873554805
        * np.where(a > 0, a, 1.6732632423543772 * np.expm1(a)),
    ),
    (
        "gelu",
        lambda t: t.gelu(),
        _FWD_ANY,
        lambda a: a * 0.5 * (1 + _erf(a / np.sqrt(2))),
    ),
    (
        "hardshrink",
        lambda t: t.hardshrink(lambd=0.5),
        _FWD_ANY,
        lambda a: np.where(np.abs(a) > 0.5, a, 0.0),
    ),
    ("clamp", lambda t: t.clamp(-0.5, 0.5), _FWD_ANY, lambda a: np.clip(a, -0.5, 0.5)),
    ("sum", lambda t: t.sum(), _FWD_ANY, lambda a: np.array(a.sum())),
    ("mean", lambda t: t.mean(), _FWD_ANY, lambda a: np.array(a.mean())),
    ("prod", lambda t: t.prod(), _FWD_POS, lambda a: np.array(a.prod())),
    ("max", lambda t: t.max(), _FWD_ANY, lambda a: np.array(a.max())),
    ("min", lambda t: t.min(), _FWD_ANY, lambda a: np.array(a.min())),
    # var/std default to the unbiased (sample) estimator.
    ("var", lambda t: t.var(), _FWD_ANY, lambda a: np.array(a.var(ddof=1))),
    ("std", lambda t: t.std(), _FWD_ANY, lambda a: np.array(a.std(ddof=1))),
    ("sum_dim", lambda t: t.sum(dim=1), _FWD_3D, lambda a: a.sum(axis=1)),
    ("mean_dim", lambda t: t.mean(dim=2), _FWD_3D, lambda a: a.mean(axis=2)),
    ("prod_dim", lambda t: t.prod(dim=0), _FWD_POS3D, lambda a: a.prod(axis=0)),
    ("var_dim", lambda t: t.var(dim=1), _FWD_3D, lambda a: a.var(axis=1, ddof=1)),
    ("std_dim", lambda t: t.std(dim=1), _FWD_3D, lambda a: a.std(axis=1, ddof=1)),
    ("cumsum", lambda t: t.cumsum(1), _FWD_3D, lambda a: a.cumsum(axis=1)),
    ("cumprod", lambda t: t.cumprod(1), _FWD_POS3D, lambda a: a.cumprod(axis=1)),
    (
        "logsumexp",
        lambda t: t.logsumexp(dim=1),
        _FWD_3D,
        lambda a: np.log(np.exp(a).sum(axis=1)),
    ),
    ("norm1", lambda t: t.norm(1.0), _FWD_ANY, lambda a: np.array(np.abs(a).sum())),
    (
        "norm2",
        lambda t: t.norm(2.0),
        _FWD_ANY,
        lambda a: np.array(np.linalg.norm(a.ravel(), 2)),
    ),
    (
        "norm_inf",
        lambda t: t.norm(float("inf")),
        _FWD_ANY,
        lambda a: np.array(np.abs(a).max()),
    ),
    # median takes the lower middle for even counts, so index (n-1)//2 of the sort.
    (
        "median",
        lambda t: t.median(),
        _FWD_ANY,
        lambda a: np.array(np.sort(a.ravel())[(a.size - 1) // 2]),
    ),
    (
        "softmax",
        lambda t: t.softmax(dim=1),
        _FWD_3D,
        lambda a: np.exp(a - a.max(1, keepdims=True))
        / np.exp(a - a.max(1, keepdims=True)).sum(1, keepdims=True),
    ),
    ("argmax", lambda t: t.argmax(), _FWD_ANY, lambda a: np.array(np.argmax(a))),
    ("argmin", lambda t: t.argmin(), _FWD_ANY, lambda a: np.array(np.argmin(a))),
    ("sort", lambda t: t.sort()[0], _FWD_ANY, lambda a: np.sort(a, axis=-1)),
    (
        "argsort",
        lambda t: t.argsort(),
        _FWD_ANY,
        lambda a: np.argsort(a, axis=-1, kind="stable"),
    ),
    ("flip", lambda t: t.flip([0]), _FWD_ANY, lambda a: np.flip(a, 0)),
    ("roll", lambda t: t.roll(2, 1), _FWD_ANY, lambda a: np.roll(a, 2, 1)),
    ("transpose", lambda t: t.transpose(0, 1), _FWD_ANY, lambda a: a.T),
    (
        "permute",
        lambda t: t.permute((2, 0, 1)),
        _FWD_3D,
        lambda a: np.transpose(a, (2, 0, 1)),
    ),
    ("tril", lambda t: t.tril(0), _FWD_SQUARE, np.tril),
    ("triu", lambda t: t.triu(1), _FWD_SQUARE, lambda a: np.triu(a, 1)),
    (
        "diagonal",
        lambda t: t.diagonal(),
        _FWD_SQUARE,
        lambda a: np.diagonal(a).copy(),
    ),
    ("trace", lambda t: t.trace(), _FWD_SQUARE, lambda a: np.array(np.trace(a))),
    ("repeat", lambda t: t.repeat((2, 3)), _FWD_ANY, lambda a: np.tile(a, (2, 3))),
    (
        "repeat_interleave",
        lambda t: t.repeat_interleave(3, dim=1),
        _FWD_ANY,
        lambda a: np.repeat(a, 3, axis=1),
    ),
]


@pytest.mark.parametrize(
    "name,fn,src,reference", _FORWARD_OPS, ids=[o[0] for o in _FORWARD_OPS]
)
def test_forward_values_match_numpy(name, fn, src, reference):
    got = np.asarray(fn(mt.Tensor(src, dtype="float64")).numpy())
    want = np.asarray(reference(src.astype(np.float64)))
    assert got.shape == want.shape, f"{name}: {got.shape} != {want.shape}"
    np.testing.assert_allclose(got, want, rtol=1e-10, atol=1e-11)


def test_the_gradcheck_list_covers_every_differentiable_no_arg_op():
    """Keep `_GRADCHECK_OPS` honest as the API grows.

    The list is hand-maintained, so a new differentiable op is covered only if
    someone remembers to add it -- and a test named "every differentiable op"
    that quietly covers all but nine is worse than one that admits its scope.
    This probes the live API instead: any tensor method callable with no
    arguments whose result carries a gradient has to appear in the list.

    That is a lower bar than "every differentiable op" -- ops needing arguments
    are out of reach of a no-arg probe -- but it is checkable, and it is what
    caught `leaky_relu` sitting outside the list while its gradient boundary
    was being changed. The ops that do need arguments are swept separately, by
    `_ARG_GRADCHECK_OPS` at the bottom of this file.
    """
    # Derive coverage from the calls the lambdas actually make, rather than from
    # the parametrize ids: the ids are suffixed to stay unique ("norm1", "pow2",
    # "leaky_relu_slope"), so matching on them needs a prefix rule that quietly
    # accepts the wrong things -- "nan_to_num" would vouch for a method named
    # "nan". The source is unambiguous about which methods are exercised.
    block = pathlib.Path(__file__).read_text()
    block = block[block.index("_GRADCHECK_OPS = [") : block.index("_GRADCHECK_EXEMPT")]
    listed = set(re.findall(r"\.([a-z_0-9]+)\(", block))

    sample = np.abs(np.random.default_rng(3).standard_normal(9)) + 0.4
    missing = []
    for name in sorted(n for n in dir(mt.Tensor) if not n.startswith("_")):
        if name in _GRADCHECK_EXEMPT or name in listed:
            continue
        try:
            probe = mt.Tensor(sample.copy(), dtype="float64", requires_grad=True)
            attr = getattr(probe, name)
            if not callable(attr):
                continue
            result = attr()
        except Exception:
            continue  # needs arguments, or is not applicable to this input
        outputs = result if isinstance(result, tuple) else (result,)
        if any(getattr(o, "requires_grad", False) for o in outputs):
            missing.append(name)
    mt.clear_autograd_graph()

    assert not missing, (
        "differentiable ops with no finite-difference check: "
        + ", ".join(missing)
        + " -- add them to _GRADCHECK_OPS, or to _GRADCHECK_EXEMPT with a reason"
    )


# --------------------------------------------------------------------------- #
# Gradcheck for ops that take arguments
# --------------------------------------------------------------------------- #
#
# `test_the_gradcheck_list_covers_every_differentiable_no_arg_op` says plainly
# what it cannot reach: "ops needing arguments are out of reach of a no-arg
# probe". That is 165 of the 345 callable tensor methods, and it is where the
# interesting backward kernels live -- gather and scatter, the linalg family,
# the fused multiply-adds, the masked reductions. None of them was under a
# finite-difference check.
#
# This is the sweep for them. It does not claim to be exhaustive either: most
# of those 165 are constructors, comparisons or in-place writes with no
# gradient to check. What is listed is every argument-taking op with a backward
# kernel of its own, which is the set where an analytic gradient can be wrong.
#
# Constants are built once, at module scope, and never inside a lambda. A
# lambda that draws a fresh random operand each call makes the two arms of a
# central difference evaluate different functions -- which looks exactly like a
# broken gradient, and cost an hour before it was recognised as the harness's
# own bug rather than the library's.

_ARG_RNG = np.random.default_rng(20260913)


def _spd(n):
    """A well-conditioned symmetric positive-definite matrix."""
    a = _ARG_RNG.standard_normal((n, n))
    return a @ a.T + n * np.eye(n)


def _f64(values):
    return mt.Tensor(
        np.ascontiguousarray(np.asarray(values, dtype=np.float64)), dtype="float64"
    )


_ARG_M = _ARG_RNG.standard_normal((3, 4))
_ARG_SQ = _ARG_RNG.standard_normal((3, 3))
_ARG_POS = np.abs(_ARG_RNG.standard_normal((3, 4))) + 0.7
_ARG_SPD = _spd(3)
_ARG_VEC3 = _ARG_RNG.standard_normal(3)

_OTHER_34 = _f64(_ARG_RNG.standard_normal((3, 4)))
_OTHER_43 = _f64(_ARG_RNG.standard_normal((4, 3)))
_OTHER_3 = _f64(_ARG_RNG.standard_normal(3))
_POS_34 = _f64(np.abs(_ARG_RNG.standard_normal((3, 4))) + 0.7)
_BIAS_33 = _f64(_ARG_RNG.standard_normal((3, 3)))
_RHS_32 = _f64(_ARG_RNG.standard_normal((3, 2)))
_VEC_4 = _f64(_ARG_RNG.standard_normal(4))
_ZEROS_34 = _f64(np.zeros((3, 4)))
_GATHER_IDX = mt.Tensor(np.array([[0, 2, 1, 0], [1, 0, 2, 1]]), dtype="int64")
_ROW_IDX = mt.Tensor(np.array([0, 2]), dtype="int64")
_MASK_34 = mt.Tensor(np.array([[True, False, True, False]] * 3), dtype="bool")
# The complement. `True` in `_MASK_34` means *masked out*, so selecting with it
# picks the -inf entries of a masked log-softmax, whose gradient is a correct
# zero and whose finite difference is `-inf` minus `-inf`.
_UNMASKED_34 = mt.Tensor(np.array([[False, True, False, True]] * 3), dtype="bool")

_OTHER_33 = _f64(_ARG_RNG.standard_normal((3, 3)))
_RHS_32B = _f64(_ARG_RNG.standard_normal((3, 2)))
_TWOS_34 = _f64(np.full((3, 4), 2.0))
_ARG_SQ_DOM = _ARG_RNG.standard_normal((3, 3)) + 3 * np.eye(3)
_ARG_TRIL = np.tril(_ARG_RNG.standard_normal((3, 3))) + 3 * np.eye(3)
_ARG_RHS32 = _ARG_RNG.standard_normal((3, 2))
_ARG_VEC3B = _ARG_RNG.standard_normal(3)
_CHOL_3 = _f64(np.linalg.cholesky(_ARG_SPD))

_ARG_GRADCHECK_OPS = [
    # binary arithmetic and the fused multiply-adds
    ("add", lambda x: x.add(_OTHER_34), _ARG_M),
    ("sub", lambda x: x.sub(_OTHER_34), _ARG_M),
    ("mul", lambda x: x.mul(_OTHER_34), _ARG_M),
    ("div", lambda x: x.div(_POS_34), _ARG_M),
    ("atan2", lambda x: x.atan2(_OTHER_34), _ARG_M),
    ("hypot", lambda x: x.hypot(_OTHER_34), _ARG_M),
    ("copysign", lambda x: x.copysign(_OTHER_34), _ARG_M),
    ("logaddexp", lambda x: x.logaddexp(_OTHER_34), _ARG_M),
    ("logaddexp2", lambda x: x.logaddexp2(_OTHER_34), _ARG_M),
    ("xlogy", lambda x: x.xlogy(_POS_34), _ARG_M),
    ("lerp", lambda x: x.lerp(_OTHER_34, 0.3), _ARG_M),
    ("float_power", lambda x: x.float_power(2.5), _ARG_POS),
    ("maximum", lambda x: x.maximum(_OTHER_34), _ARG_M),
    ("minimum", lambda x: x.minimum(_OTHER_34), _ARG_M),
    ("fmax", lambda x: x.fmax(_OTHER_34), _ARG_M),
    ("fmin", lambda x: x.fmin(_OTHER_34), _ARG_M),
    ("addcmul", lambda x: x.addcmul(_OTHER_34, _POS_34, 0.7), _ARG_M),
    ("addcdiv", lambda x: x.addcdiv(_OTHER_34, _POS_34, 0.7), _ARG_M),
    ("renorm", lambda x: x.renorm(2.0, 0, 1.0), _ARG_M),
    # products
    ("matmul", lambda x: x.matmul(_OTHER_43), _ARG_M),
    ("mm", lambda x: x.mm(_OTHER_43), _ARG_M),
    ("mv", lambda x: x.mv(_VEC_4), _ARG_M),
    ("dot", lambda x: x.dot(_OTHER_3), _ARG_VEC3),
    ("inner", lambda x: x.inner(_OTHER_34), _ARG_M),
    ("vecdot", lambda x: x.vecdot(_OTHER_34), _ARG_M),
    ("tensordot", lambda x: x.tensordot(_OTHER_43, 1), _ARG_M),
    ("matrix_power", lambda x: x.matrix_power(3), _ARG_SQ),
    ("matrix_exp", lambda x: x.matrix_exp(), _ARG_SQ),
    ("addmm", lambda x: _BIAS_33.addmm(x, _OTHER_43), _ARG_M),
    # linear algebra
    ("det", lambda x: x.det(), _ARG_SQ),
    ("logdet", lambda x: x.logdet(), _ARG_SPD),
    ("inv", lambda x: x.inv(), _ARG_SQ),
    ("solve", lambda x: x.solve(_RHS_32), _ARG_SPD),
    ("eigh", lambda x: x.eigh()[0], _ARG_SPD),
    ("svd", lambda x: x.svd(False)[1], _ARG_M),
    # shape
    ("expand", lambda x: x.expand([2, 3, 4]), _ARG_M),
    ("permute", lambda x: x.permute([1, 0]), _ARG_M),
    ("moveaxis", lambda x: x.moveaxis(0, 1), _ARG_M),
    ("narrow", lambda x: x.narrow(1, 1, 2), _ARG_M),
    ("repeat", lambda x: x.repeat([2, 1]), _ARG_M),
    ("repeat_interleave", lambda x: x.repeat_interleave(2), _ARG_M),
    ("unsqueeze", lambda x: x.unsqueeze(0), _ARG_M),
    ("view", lambda x: x.view([4, 3]), _ARG_M),
    ("pad", lambda x: x.pad([1, 1]), _ARG_M),
    ("chunk", lambda x: x.chunk(2, 1)[0], _ARG_M),
    ("split", lambda x: x.split(2, 1)[1], _ARG_M),
    # indexing and masking
    ("gather", lambda x: x.gather(0, _GATHER_IDX), _ARG_M),
    ("index_select", lambda x: x.index_select(0, _ROW_IDX), _ARG_M),
    ("masked_fill", lambda x: x.masked_fill(_MASK_34, 0.5), _ARG_M),
    ("masked_select", lambda x: x.masked_select(_MASK_34), _ARG_M),
    (
        "scatter_add",
        lambda x: _ZEROS_34.scatter_add(0, _GATHER_IDX[:1].expand([3, 4]), x),
        _ARG_M,
    ),
    ("where", lambda x: x.where(_MASK_34, _POS_34), _ARG_M),
    # normalisation and masked softmaxes
    # Sum-invariant: `layer_norm` normalises, so `.sum()` is a constant and its
    # gradient is zero -- a check that would pass on any backward at all.
    # Weighting the output makes the scalar depend on the input again.
    ("layer_norm", lambda x: x.layer_norm([4]).mul(_OTHER_34), _ARG_M),
    ("rms_norm", lambda x: x.rms_norm([4]), _ARG_M),
    (
        "masked_softmax",
        lambda x: x.masked_softmax(_MASK_34, 1).mul(_OTHER_34),
        _ARG_M,
    ),  # as above
    # `masked_log_softmax` is -inf wherever `_MASK_34` is true, and a sum holding
    # -inf differences to NaN. Selecting on the complement keeps the whole
    # backward under test without asking the difference an impossible question.
    (
        "masked_log_softmax",
        lambda x: x.masked_log_softmax(_MASK_34, 1).masked_select(_UNMASKED_34),
        _ARG_M,
    ),
    # aliases of kernels already above, listed so the coverage guard below can
    # insist the list is complete rather than take its author's word for it
    ("divide", lambda x: x.divide(_POS_34), _ARG_M),
    ("multiply", lambda x: x.multiply(_OTHER_34), _ARG_M),
    ("subtract", lambda x: x.subtract(_OTHER_34), _ARG_M),
    ("true_divide", lambda x: x.true_divide(_POS_34), _ARG_M),
    ("inverse", lambda x: x.inverse(), _ARG_SQ_DOM),
    ("movedim", lambda x: x.movedim(0, 1), _ARG_M),
    ("swapaxes", lambda x: x.swapaxes(0, 1), _ARG_M),
    ("swapdims", lambda x: x.swapdims(0, 1), _ARG_M),
    ("transpose", lambda x: x.transpose(0, 1), _ARG_M),
    # more products
    ("cross", lambda x: x.cross(_OTHER_33), _ARG_SQ),
    ("matvec", lambda x: x.matvec(_VEC_4), _ARG_M),
    ("vecmat", lambda x: x.vecmat(_OTHER_34), _ARG_VEC3B),
    # branchy elementwise, with the inputs kept clear of the kinks
    ("clamp_max", lambda x: x.clamp_max(0.35), _ARG_M),
    ("clamp_min", lambda x: x.clamp_min(-0.35), _ARG_M),
    ("fmod", lambda x: x.fmod(_TWOS_34), _ARG_M),
    ("remainder", lambda x: x.remainder(_TWOS_34), _ARG_M),
    ("ldexp", lambda x: x.ldexp(_TWOS_34), _ARG_M),
    ("nextafter", lambda x: x.nextafter(_OTHER_34), _ARG_M),
    # the rest of the linear algebra surface
    ("cholesky_solve", lambda x: x.cholesky_solve(_CHOL_3), _ARG_RHS32),
    ("solve_triangular", lambda x: x.solve_triangular(_RHS_32B, True), _ARG_TRIL),
    ("eigvalsh", lambda x: x.eigvalsh(), _ARG_SPD),
    ("slogdet", lambda x: x.slogdet()[1], _ARG_SQ_DOM),
    ("svdvals", lambda x: x.svdvals(), _ARG_M),
    ("pinv", lambda x: x.pinv(), _ARG_M),
    ("pinverse", lambda x: x.pinverse(), _ARG_M),
    ("qr_q", lambda x: x.qr()[0], _ARG_M),
    ("qr_r", lambda x: x.qr()[1], _ARG_M),
    ("lstsq", lambda x: x.lstsq(_RHS_32B)[0], _ARG_SQ_DOM),
    ("tensorinv", lambda x: x.tensorinv(1), _ARG_SQ_DOM),
    ("tensorsolve", lambda x: x.tensorsolve(_OTHER_3), _ARG_SQ_DOM),
    ("matrix_norm", lambda x: x.matrix_norm(), _ARG_M),
    ("polygamma", lambda x: x.polygamma(1), _ARG_POS),
    # `divmod` returns (floor, remainder). The floor half comes back with
    # `requires_grad` false, correctly -- it is piecewise constant -- so the
    # pair is covered by checking the half that carries a gradient.
    ("divmod_remainder", lambda x: x.divmod(_TWOS_34)[1], _ARG_M),
]


@pytest.mark.parametrize(
    "name,fn,src", _ARG_GRADCHECK_OPS, ids=[o[0] for o in _ARG_GRADCHECK_OPS]
)
def test_gradcheck_ops_that_take_arguments(name, fn, src):
    src = np.ascontiguousarray(np.asarray(src, dtype=np.float64))
    analytic = _analytic_grad_f64(fn, src)
    numeric = _numeric_grad_f64(fn, src)
    assert (
        analytic.shape == numeric.shape
    ), f"{name}: {analytic.shape} != {numeric.shape}"

    assert np.isfinite(analytic).all(), f"{name}: analytic gradient is not finite"
    assert np.isfinite(numeric).all(), f"{name}: central difference is not finite"

    scale = max(np.max(np.abs(numeric)), np.max(np.abs(analytic)))
    # Every case here is scalarised so that the gradient is not identically
    # zero. A zero scale would mean the case had stopped testing anything --
    # which is what a plain `.sum()` of `layer_norm` quietly did.
    assert scale > 1e-8, f"{name}: gradient is zero, so this case checks nothing"
    err = np.max(np.abs(analytic - numeric)) / scale
    assert err < 2e-5, f"{name}: analytic and central difference differ by {err:.3e}"


def test_cholesky_gradient_is_the_symmetric_one():
    """Checked apart from the sweep, because an entrywise difference is the
    wrong question to ask it.

    `cholesky` reads one triangle, so perturbing a single upper-triangle entry
    changes nothing and an entrywise finite difference lands entirely on the
    lower triangle. The backward returns the symmetric gradient instead --
    each mirrored pair splitting the sensitivity -- which is right for the only
    perturbation a Cholesky input admits, and is what the entrywise difference
    disagrees with. So the check is the directional derivative along symmetric
    directions, where the two must agree exactly.
    """
    matrix = _spd(3)
    x = mt.Tensor(matrix.copy(), dtype="float64", requires_grad=True)
    x.cholesky().sum().backward()
    analytic = np.asarray(x.grad).copy()
    mt.clear_autograd_graph()

    assert np.allclose(
        analytic, analytic.T
    ), "a symmetric input wants a symmetric gradient"

    eps = 1e-5
    for i in range(3):
        for j in range(i, 3):
            plus, minus = matrix.copy(), matrix.copy()
            plus[i, j] += eps
            minus[i, j] -= eps
            if i != j:  # keep the perturbation symmetric
                plus[j, i] += eps
                minus[j, i] -= eps
            up = mt.Tensor(plus, dtype="float64").cholesky().sum()
            down = mt.Tensor(minus, dtype="float64").cholesky().sum()
            directional = (float(np.asarray(up)) - float(np.asarray(down))) / (2 * eps)
            want = analytic[i, j] + (analytic[j, i] if i != j else 0.0)
            assert directional == pytest.approx(want, abs=1e-6), f"direction ({i},{j})"
    mt.clear_autograd_graph()


_ARG_GRADCHECK_EXEMPT = {
    # Constructors. They take dtype, device and `requires_grad` from the tensor
    # they are called on -- deliberately, and asserted by
    # `test_tensor_new_ones_defaults_to_reference_metadata` -- so the result
    # carries the flag while its *values* do not depend on the source at all.
    # There is no gradient path to check.
    "new_empty": "constructor: inherits requires_grad, output does not depend on the input",
    "new_full": "constructor: inherits requires_grad, output does not depend on the input",
    "new_ones": "constructor: inherits requires_grad, output does not depend on the input",
    "new_tensor": "constructor: inherits requires_grad, output does not depend on the input",
    "new_zeros": "constructor: inherits requires_grad, output does not depend on the input",
    # In-place writes. The "output" is the tensor they were called on, so it
    # carries the flag it already had; none has a backward of its own.
    "copy_": "in-place: returns the tensor it was called on",
    "fill_": "in-place: returns the tensor it was called on",
    "requires_grad_": "sets the flag; there is no operation to differentiate",
    # A step. Its derivative in the first argument is zero wherever it exists --
    # confirmed against a central difference, both sides zero -- so leaving it
    # in the sweep would only trip the "this case checks nothing" assertion.
    "heaviside": "derivative in the input is identically zero",
}


def _arg_probe_recipes():
    """Argument tuples to try against each method, and the inputs to try them on.

    Deliberately small. This is a reachability probe, not a fuzzer: it needs to
    find *one* call that produces a gradient, and the shapes below are the ones
    the tensor API actually accepts.
    """
    f34 = _f64(np.random.default_rng(5).standard_normal((3, 4)))
    f33 = _f64(np.random.default_rng(6).standard_normal((3, 3)))
    f43 = _f64(np.random.default_rng(7).standard_normal((4, 3)))
    f3 = _f64(np.random.default_rng(8).standard_normal(3))
    f4 = _f64(np.random.default_rng(9).standard_normal(4))
    idx = mt.Tensor(np.array([[0, 2, 1, 0], [1, 0, 2, 1]]), dtype="int64")
    rows = mt.Tensor(np.array([0, 2]), dtype="int64")
    mask = mt.Tensor(np.array([[True, False, True, False]] * 3), dtype="bool")
    recipes = [
        (),
        (f34,),
        (f33,),
        (f43,),
        (f3,),
        (f4,),
        (0,),
        (1,),
        (-1,),
        (2,),
        (0.5,),
        (2.0,),
        (True,),
        (0, 1),
        (1, 0),
        (0, 0),
        (1, 1),
        (2, 1),
        ([4, 3],),
        ([2, 1],),
        ([0, 1],),
        ([4],),
        (f34, 0.5),
        (f34, f34),
        (f34, f34, 0.5),
        (f34, True),
        (idx,),
        (0, idx),
        (0, rows),
        (0, idx, f34),
        (mask,),
        (mask, 0.5),
        (mask, 1),
        (1, 1, 2),
    ]
    a = np.random.default_rng(10).standard_normal((3, 3))
    sources = [
        np.random.default_rng(11).standard_normal((3, 4)),
        np.random.default_rng(12).standard_normal((3, 3)) + 3 * np.eye(3),
        a @ a.T + 3 * np.eye(3),
        np.random.default_rng(13).standard_normal(3),
    ]
    return recipes, sources


def test_the_argument_gradcheck_list_covers_every_op_it_can_reach():
    """The same guard the no-argument list has, for the list that takes them.

    `_ARG_GRADCHECK_OPS` started as a hand-picked set, and the comment above it
    claimed to hold "every argument-taking op with a backward kernel of its
    own". That claim was wrong by 31 ops -- among them `qr`, `pinv`, `lstsq`,
    `slogdet` and `solve_triangular`, which is exactly the part of the surface
    where a gradient is hard to get right. Hand-picking cannot be trusted here,
    so this probes the live API instead: any method that yields a
    gradient-carrying result under some plausible argument tuple has to be in
    the list, or exempted with a reason.
    """
    recipes, sources = _arg_probe_recipes()
    # Exactly the two list literals, and nothing between them: a slice wide
    # enough to take in the test bodies would count any method those happen to
    # call, which is how a guard quietly stops guarding.
    text = pathlib.Path(__file__).read_text()
    no_arg = text[text.index("_GRADCHECK_OPS = [") : text.index("_GRADCHECK_EXEMPT")]
    with_arg = text[
        text.index("_ARG_GRADCHECK_OPS = [") : text.index("_ARG_GRADCHECK_EXEMPT")
    ]
    listed = set(re.findall(r"\.([a-z_0-9]+)\(", no_arg + with_arg))
    listed.add("cholesky")  # checked by test_cholesky_gradient_is_the_symmetric_one

    missing = {}
    for name in sorted(n for n in dir(mt.Tensor) if not n.startswith("_")):
        if name in listed or name in _ARG_GRADCHECK_EXEMPT:
            continue
        for source in sources:
            for args in recipes:
                try:
                    probe = mt.Tensor(
                        np.ascontiguousarray(source.copy()),
                        dtype="float64",
                        requires_grad=True,
                    )
                    attr = getattr(probe, name)
                    if not callable(attr):
                        break
                    result = attr(*args)
                except Exception:
                    continue
                outputs = result if isinstance(result, tuple) else (result,)
                if any(getattr(o, "requires_grad", False) for o in outputs):
                    missing[name] = f"{source.shape} {args!r:.60}"
                    break
            if name in missing:
                break
    mt.clear_autograd_graph()

    assert not missing, (
        "differentiable ops with no finite-difference check: "
        + ", ".join(sorted(missing))
        + " -- add them to _ARG_GRADCHECK_OPS, or to _ARG_GRADCHECK_EXEMPT with a reason"
    )
