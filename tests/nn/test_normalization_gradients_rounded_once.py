# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""`layer_norm`'s and `rms_norm`'s gradients against exact values.

With `y = d r` -- `d` the deviation from the mean, or the value itself not
centred, and `r` the inverse root of the variance or mean square plus eps
-- and `h = g w`, the input's gradient is `r (u - r^2 C d)`, `u = h - mean(h)`
(or `h`) and `C = mean(u d)`, and the weight's is `g y` summed over the rows.
They were formed in the dtype from the rounded `y` and `r` the forward saved,
and the two terms all but cancel where the upstream gradient lies near the
span of one and `y`: a float32 gradient was 13 ulps out on ordinary data and
two million near that span, a float64 one 74 and fifteen million. Each input
gradient is now the float nearest the exact one.
"""

from __future__ import annotations

import decimal
from decimal import Decimal

import numpy as np
import pytest

import minitensor as mt
from minitensor import functional as F

_EPS = 1e-5


def _exact(x, g, w, eps, centred):
    """The input's gradient, row by row, and the weight's, exactly."""

    rows, n = x.shape
    weight = [Decimal(float(v)) for v in w] if w is not None else [Decimal(1)] * n
    grads, columns = [], [Decimal(0)] * n
    for row in range(rows):
        X = [Decimal(float(v)) for v in x[row]]
        G = [Decimal(float(v)) for v in g[row]]
        mean = sum(X) / n if centred else Decimal(0)
        d = [v - mean for v in X]
        r = 1 / (sum(v * v for v in d) / n + Decimal(eps)).sqrt()
        h = [a * b for a, b in zip(G, weight)]
        shift = sum(h) / n if centred else Decimal(0)
        u = [v - shift for v in h]
        c = sum(a * b for a, b in zip(u, d)) / n
        grads.append([r * (a - r * r * c * b) for a, b in zip(u, d)])
        columns = [s + a * b * r for s, a, b in zip(columns, G, d)]
    return grads, columns


def _ulps(got: float, want: Decimal, dtype: str) -> float:
    expected = float(want)
    unit = float(np.spacing(np.array(abs(expected), dtype=dtype)))
    if expected == 0.0:
        return 0.0 if got == 0.0 else float("inf")
    return float(abs(Decimal(float(got)) - want) / Decimal(unit))


def _run(kind, x, g, w, dtype):
    xi = mt.Tensor(x, dtype=dtype, requires_grad=True)
    wi = None if w is None else mt.Tensor(w, dtype=dtype, requires_grad=True)
    n = x.shape[-1]
    if kind == "layer":
        y = F.layer_norm(xi, [n], wi, None, _EPS)
    else:
        y = F.rms_norm(xi, [n], wi, _EPS)
    y.backward(mt.Tensor(g, dtype=dtype))
    return xi.grad.numpy(), None if wi is None else wi.grad.numpy()


def _rows(case, dtype, rows=6, n=37):
    rng = np.random.default_rng({"ordinary": 1, "offset": 2, "aligned": 3}[case])
    x = rng.standard_normal((rows, n))
    if case == "offset":
        # A mean that dominates the spread.
        x = 1e4 + x if dtype == "float32" else 1e9 + x
    x = x.astype(dtype)
    g = rng.standard_normal((rows, n))
    if case == "aligned":
        # An upstream gradient all but in the span of one and `y`.
        y = (x - x.mean(1, keepdims=True)) / x.std(1, keepdims=True)
        g = 0.7 + 1.3 * y + 1e-4 * g
    w = rng.uniform(0.5, 2.0, n).astype(dtype)
    return x, g.astype(dtype), w


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("kind", ["layer", "rms"])
@pytest.mark.parametrize("case", ["ordinary", "offset", "aligned"])
@pytest.mark.parametrize("weighted", [False, True])
def test_gradients_are_the_nearest_floats(case, kind, dtype, weighted):
    x, g, w = _rows(case, dtype)
    w = w if weighted else None
    got_x, got_w = _run(kind, x, g, w, dtype)
    with decimal.localcontext(prec=80):
        grads, columns = _exact(x, g, w, _EPS, kind == "layer")
        for row, values in enumerate(grads):
            for k, want in enumerate(values):
                assert _ulps(got_x[row, k], want, dtype) <= 0.5, (row, k)
        if weighted:
            for k, want in enumerate(columns):
                assert _ulps(got_w[k], want, dtype) <= 0.5, k


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_a_constant_upstream_gradient_leaves_nothing(dtype):
    # Centred, the standardized values sum to nothing whatever the input,
    # and so do their gradients under an upstream gradient that is the same
    # everywhere: exactly, not a rounding away from it.
    rng = np.random.default_rng(5)
    x = rng.standard_normal((5, 33)).astype(dtype)
    for scale in (1.0, 0.1, 1.0 / 3.0):
        w = np.full(33, scale, dtype=dtype)
        xi = mt.Tensor(x, dtype=dtype, requires_grad=True)
        F.layer_norm(xi, [33], mt.Tensor(w, dtype=dtype), None, _EPS).mean().backward()
        grad = xi.grad.numpy()
        assert np.all(grad == 0.0) and not np.any(np.signbit(grad)), scale


def test_values_too_far_out_for_the_pairs():
    # Squares past the float64 range, and below it: every gradient in whole
    # numbers, still the nearest float.
    rng = np.random.default_rng(7)
    for scale in (1e200, 1e-200):
        x = scale * (3.0 + rng.standard_normal((3, 9)))
        g = rng.standard_normal((3, 9))
        for kind in ("layer", "rms"):
            got_x, _ = _run(kind, x, g, None, "float64")
            with decimal.localcontext(prec=120):
                grads, _ = _exact(x, g, None, _EPS, kind == "layer")
                for row, values in enumerate(grads):
                    for k, want in enumerate(values):
                        assert _ulps(got_x[row, k], want, "float64") <= 0.5, (
                            scale,
                            kind,
                        )


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_a_row_holding_a_nan_is_nan_and_the_others_are_not(dtype):
    x = np.random.default_rng(9).standard_normal((3, 8)).astype(dtype)
    x[1, 4] = np.nan
    g = np.ones((3, 8), dtype=dtype)
    g[0, 0] = 2.0
    for kind in ("layer", "rms"):
        got_x, _ = _run(kind, x, g, None, dtype)
        assert np.all(np.isnan(got_x[1]))
        assert np.all(np.isfinite(got_x[[0, 2]]))


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("case", ["ordinary", "offset", "aligned"])
def test_batch_norm_gradients_are_the_nearest_floats(case, dtype):
    # Each channel is one set: its values over the batch and positions, its
    # weight the same for all of them.
    rng = np.random.default_rng({"ordinary": 11, "offset": 12, "aligned": 13}[case])
    x = rng.standard_normal((6, 3, 5))
    if case == "offset":
        x = 1e4 + x if dtype == "float32" else 1e9 + x
    x = x.astype(dtype)
    g = rng.standard_normal(x.shape)
    if case == "aligned":
        mean = x.mean((0, 2), keepdims=True)
        g = 0.7 + 1.3 * (x - mean) / x.std((0, 2), keepdims=True) + 1e-4 * g
    g = g.astype(dtype)
    w = rng.uniform(0.5, 2.0, 3).astype(dtype)
    b = rng.uniform(-1.0, 1.0, 3).astype(dtype)
    xi = mt.Tensor(x, dtype=dtype, requires_grad=True)
    wi = mt.Tensor(w, dtype=dtype, requires_grad=True)
    bi = mt.Tensor(b, dtype=dtype, requires_grad=True)
    y = F.batch_norm(xi, None, None, wi, bi, True, 0.1, _EPS)
    y.backward(mt.Tensor(g, dtype=dtype))
    got_x, got_w, got_b = xi.grad.numpy(), wi.grad.numpy(), bi.grad.numpy()
    with decimal.localcontext(prec=80):
        for c in range(3):
            values = x[:, c, :].reshape(1, -1)
            upstream = g[:, c, :].reshape(1, -1)
            weight = np.full(values.shape[1], w[c])
            grads, columns = _exact(values, upstream, weight, _EPS, True)
            for k, want in enumerate(grads[0]):
                assert _ulps(got_x[:, c, :].ravel()[k], want, dtype) <= 0.5, (c, k)
            assert _ulps(got_w[c], sum(columns), dtype) <= 0.5, c
            bias = sum(Decimal(float(v)) for v in upstream[0])
            assert _ulps(got_b[c], bias, dtype) <= 0.5, c


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_batch_norm_under_a_constant_upstream_gradient_is_exactly_zero(dtype):
    x = np.random.default_rng(15).standard_normal((6, 3, 7)).astype(dtype)
    w = mt.Tensor(np.full(3, 1.0 / 3.0, dtype=dtype), dtype=dtype)
    xi = mt.Tensor(x, dtype=dtype, requires_grad=True)
    F.batch_norm(xi, None, None, w, None, True, 0.1, _EPS).mean().backward()
    grad = xi.grad.numpy()
    assert np.all(grad == 0.0) and not np.any(np.signbit(grad))
