# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""A float32 factorisation is carried in float64 and rounded once.

A float32 matrix is exact in float64, so the factors computed there and rounded
at the end are as near the exact ones as float32 holds. In float32 arithmetic an
`eigh` residual came out twenty to fifty times float32's own and an SVD's small
singular values three orders less accurate than that.
"""

from __future__ import annotations

import numpy as np
import pytest

import minitensor as mt


def _matrix(n, condition, seed):
    rng = np.random.default_rng(seed)
    u, _ = np.linalg.qr(rng.normal(0, 1, (n, n)))
    v, _ = np.linalg.qr(rng.normal(0, 1, (n, n)))
    values = np.logspace(0, -np.log10(condition), n)
    return ((u * values) @ v.T).astype(np.float32)


def _wide(tensor) -> np.ndarray:
    return tensor.numpy().astype(np.float64)


@pytest.mark.parametrize("n,condition", [(40, 1e2), (40, 1e6), (120, 1e4)])
def test_factorisations_are_the_rounded_float64_ones(n, condition):
    a = _matrix(n, condition, 331 + n)
    exact = a.astype(np.float64)
    tensor = mt.Tensor(a)
    eps = float(np.finfo(np.float32).eps)

    s = _wide(mt.svdvals(tensor))
    want = np.linalg.svd(exact, compute_uv=False)
    assert np.max(np.abs(s - want) / want) < 2 * eps

    sym = (a + a.T) / 2
    w, v = mt.eigh(mt.Tensor(sym))
    w, v = _wide(w), _wide(v)
    residual = np.linalg.norm(sym.astype(np.float64) @ v - v * w)
    assert residual / np.linalg.norm(sym) < 4 * eps
    assert np.linalg.norm(v.T @ v - np.eye(n)) < 2 * np.sqrt(n) * eps

    q, r = mt.qr(tensor)
    q, r = _wide(q), _wide(r)
    assert np.linalg.norm(q.T @ q - np.eye(n)) < 2 * np.sqrt(n) * eps
    assert np.linalg.norm(q @ r - exact) / np.linalg.norm(exact) < 4 * eps

    b = np.random.default_rng(337).normal(0, 1, (n, 2)).astype(np.float32)
    x = _wide(mt.solve(tensor, mt.Tensor(b)))
    backward = np.linalg.norm(exact @ x - b) / (
        np.linalg.norm(exact) * np.linalg.norm(x)
    )
    assert backward < eps

    sign, logdet = mt.slogdet(tensor)
    want_sign, want_log = np.linalg.slogdet(exact)
    assert sign.item() == want_sign
    assert abs(logdet.item() - want_log) <= 2 * eps * abs(want_log)


def test_results_and_gradients_stay_float32():
    rng = np.random.default_rng(347)
    a = mt.Tensor(rng.normal(0, 1, (6, 6)).astype(np.float32), requires_grad=True)
    u, s, vh = mt.svd(a, full_matrices=False)
    assert {str(t.dtype) for t in (u, s, vh)} == {"float32"}
    s.sum().backward()
    assert str(a.grad.dtype) == "float32"
    np.testing.assert_allclose(
        _wide(a.grad), _wide(u) @ _wide(vh), rtol=1e-5, atol=1e-6
    )
    p, low, up = mt.lu(mt.Tensor(rng.normal(0, 1, (5, 5)).astype(np.float32)))
    assert {str(t.dtype) for t in (p, low, up)} == {"float32"}
    factors, pivots = mt.lu_factor(
        mt.Tensor(rng.normal(0, 1, (5, 5)).astype(np.float32))
    )
    assert str(factors.dtype) == "float32" and str(pivots.dtype) == "int64"
