# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""`group_norm`'s and `instance_norm`'s outputs and gradients against exact
values.

They standardized each group with `layer_norm` and then applied the weight
and bias as tensor operations, the standardized value rounded first: where
the bias all but cancels the product that rounding was what was left, 1.2e9
ulps out in float32 and 3.7e17 in float64, and the gradients, from the
rounded `g w` and `y`, were up to 69 and 125 ulps out for the input and 67
and 60 for the weight. Each output, and each gradient of the input, is now
the value nearest its exact one.
"""

from __future__ import annotations

import decimal
from decimal import Decimal

import numpy as np
import pytest

import minitensor as mt
from minitensor import functional as F

_EPS = 1e-5


def _ulps(got, want: Decimal, dtype: str) -> float:
    if want == 0:
        return 0.0 if got == 0 else float("inf")
    unit = Decimal(float(np.spacing(np.array(abs(float(want)), dtype=dtype))))
    return float(abs(Decimal(float(got)) - want) / unit)


def _exact(x, g, w, groups):
    """Each output, the input's gradient and the weight's, exactly."""

    n, c = x.shape[:2]
    per = c // groups
    flat = x.reshape(n, c, -1)
    upstream = g.reshape(n, c, -1)
    s = flat.shape[2]
    outputs = np.empty(flat.shape, dtype=object)
    grads = np.empty(flat.shape, dtype=object)
    weight_grad = [Decimal(0)] * c
    for image in range(n):
        for group in range(groups):
            channels = range(group * per, (group + 1) * per)
            X = [Decimal(float(v)) for ch in channels for v in flat[image, ch]]
            H = [
                Decimal(float(v)) * Decimal(float(w[ch]))
                for ch in channels
                for v in upstream[image, ch]
            ]
            G = [Decimal(float(v)) for ch in channels for v in upstream[image, ch]]
            m = len(X)
            mean = sum(X) / m
            d = [v - mean for v in X]
            r = 1 / (sum(v * v for v in d) / m + Decimal(_EPS)).sqrt()
            shift = sum(H) / m
            u = [v - shift for v in H]
            cc = sum(a * b for a, b in zip(u, d)) / m
            for k, ch in enumerate(channels):
                for p in range(s):
                    i = k * s + p
                    outputs[image, ch, p] = d[i] * r
                    grads[image, ch, p] = r * (u[i] - r * r * cc * d[i])
                    weight_grad[ch] += G[i] * d[i] * r
    return outputs, grads, weight_grad


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("groups", [1, 2, 4])
@pytest.mark.parametrize("shape", [(3, 4, 5), (2, 4, 36)])
def test_outputs_and_gradients(shape, groups, dtype):
    # A channel's weight terms are summed term by term over a few positions,
    # and in one carried pass over many.
    rng = np.random.default_rng(41 + groups)
    x = rng.standard_normal(shape).astype(dtype)
    g = rng.standard_normal(x.shape).astype(dtype)
    w = rng.uniform(0.5, 2.0, 4).astype(dtype)
    with decimal.localcontext(prec=80):
        ys, grads, weight_grad = _exact(x, g, w, groups)
        # A bias that cancels the first output of each channel to its rounding.
        b = np.array(
            [-float(Decimal(float(w[c])) * ys[0, c, 0]) for c in range(4)]
        ).astype(dtype)
        xi = mt.Tensor(x, dtype=dtype, requires_grad=True)
        wi = mt.Tensor(w, dtype=dtype, requires_grad=True)
        bi = mt.Tensor(b, dtype=dtype, requires_grad=True)
        out = F.group_norm(xi, groups, wi, bi, _EPS)
        out.backward(mt.Tensor(g, dtype=dtype))
        got, got_x = out.numpy(), xi.grad.numpy()
        got_w, got_b = wi.grad.numpy(), bi.grad.numpy()
        for idx in np.ndindex(x.shape):
            c = idx[1]
            want = Decimal(float(w[c])) * ys[idx] + Decimal(float(b[c]))
            assert _ulps(got[idx], want, dtype) <= 0.5, ("output", idx)
            assert _ulps(got_x[idx], grads[idx], dtype) <= 0.5, ("input", idx)
        for c in range(4):
            assert _ulps(got_w[c], weight_grad[c], dtype) <= 0.5, ("weight", c)
            bias = sum(Decimal(float(v)) for v in g[:, c, :].ravel())
            assert _ulps(got_b[c], bias, dtype) <= 0.5, ("bias", c)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_instance_norm_is_a_group_a_channel(dtype):
    rng = np.random.default_rng(47)
    x = rng.standard_normal((2, 3, 6)).astype(dtype)
    w = rng.uniform(0.5, 2.0, 3).astype(dtype)
    b = rng.uniform(-1.0, 1.0, 3).astype(dtype)
    with decimal.localcontext(prec=80):
        ys, _, _ = _exact(x, np.zeros_like(x), w, 3)
        out = F.instance_norm(
            mt.Tensor(x, dtype=dtype),
            weight=mt.Tensor(w, dtype=dtype),
            bias=mt.Tensor(b, dtype=dtype),
            eps=_EPS,
        ).numpy()
        for idx in np.ndindex(x.shape):
            c = idx[1]
            want = Decimal(float(w[c])) * ys[idx] + Decimal(float(b[c]))
            assert _ulps(out[idx], want, dtype) <= 0.5, idx


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_a_constant_upstream_gradient_leaves_nothing(dtype):
    x = np.random.default_rng(49).standard_normal((2, 4, 5)).astype(dtype)
    xi = mt.Tensor(x, dtype=dtype, requires_grad=True)
    w = mt.Tensor(np.full(4, 1.0 / 3.0, dtype=dtype), dtype=dtype)
    F.group_norm(xi, 2, w, None, _EPS).mean().backward()
    grad = xi.grad.numpy()
    assert np.all(grad == 0.0) and not np.any(np.signbit(grad))


def test_a_weight_of_another_dtype_is_promoted_as_arithmetic_is():
    x = np.random.default_rng(51).standard_normal((2, 4, 3)).astype("float32")
    w = np.random.default_rng(52).uniform(0.5, 2.0, 4)
    out = F.group_norm(mt.Tensor(x, dtype="float32"), 2, mt.Tensor(w, dtype="float64"))
    plain = F.group_norm(mt.Tensor(x, dtype="float32"), 2).numpy()
    np.testing.assert_allclose(
        out.numpy(), plain.astype("float64") * w.reshape(1, 4, 1), rtol=1e-12
    )
