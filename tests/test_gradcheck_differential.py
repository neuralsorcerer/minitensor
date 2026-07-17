# Copyright (c) 2026 Soumyadip Sarkar.
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
