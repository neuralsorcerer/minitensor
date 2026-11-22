# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""`batch_norm` and `instance_norm` under running statistics against exact
values.

An evaluation pass takes the running mean and variance as given, and each
output is `w (x - mean) / sqrt(var + eps) + b`. `batch_norm` formed it in
float64 from the rounded inverse root, a float64 output up to 16 ulps out and,
where the bias all but cancels the product, 7.6e15; `instance_norm` composed
it from tensor operations, up to 15 ulps out and, with that bias, 5e7 in
float32 and 1.3e16 in float64. Each output is now the value nearest its exact
one, and so is each input gradient `w g / sqrt(var + eps)`.
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


def _root(var) -> Decimal:
    return (Decimal(float(var)) + Decimal(_EPS)).sqrt()


def _exact(x, mean, var, w, b) -> Decimal:
    return Decimal(float(w)) * (Decimal(float(x)) - Decimal(float(mean))) / _root(
        var
    ) + Decimal(float(b))


def _case(shape, dtype, seed):
    rng = np.random.default_rng(seed)
    channels = shape[1]
    x = rng.standard_normal(shape).astype(dtype)
    mean = rng.uniform(-0.5, 0.5, channels).astype(dtype)
    var = rng.uniform(0.5, 2.0, channels).astype(dtype)
    w = rng.uniform(0.5, 2.0, channels).astype(dtype)
    return x, mean, var, w


def _tensor(values, dtype, grad=False):
    return mt.Tensor(np.asarray(values), dtype=dtype, requires_grad=grad)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("shape", [(5, 3), (4, 3, 6)])
@pytest.mark.parametrize("cancelling", [False, True])
def test_outputs_and_gradients(shape, dtype, cancelling):
    x, mean, var, w = _case(shape, dtype, 61 + len(shape))
    channels = shape[1]
    with decimal.localcontext(prec=80):
        if cancelling:
            # A bias that cancels the first output of each channel to its
            # rounding.
            first = x.reshape(shape[0], channels, -1)[0, :, 0]
            b = np.array(
                [
                    -float(_exact(first[c], mean[c], var[c], w[c], 0.0))
                    for c in range(channels)
                ]
            ).astype(dtype)
        else:
            b = np.random.default_rng(7).uniform(-1.0, 1.0, channels).astype(dtype)
        g = np.random.default_rng(8).standard_normal(shape).astype(dtype)
        xi = _tensor(x, dtype, True)
        wi, bi = _tensor(w, dtype, True), _tensor(b, dtype, True)
        out = F.batch_norm(
            xi, _tensor(mean, dtype), _tensor(var, dtype), wi, bi, False, 0.1, _EPS
        )
        out.backward(_tensor(g, dtype))
        got, got_x = out.numpy(), xi.grad.numpy()
        got_w, got_b = wi.grad.numpy(), bi.grad.numpy()
        weight_grad = [Decimal(0)] * channels
        for idx in np.ndindex(shape):
            c = idx[1]
            want = _exact(x[idx], mean[c], var[c], w[c], b[c])
            assert _ulps(got[idx], want, dtype) <= 0.5, ("output", idx)
            slope = Decimal(float(w[c])) * Decimal(float(g[idx])) / _root(var[c])
            assert _ulps(got_x[idx], slope, dtype) <= 0.5, ("input", idx)
            y = (Decimal(float(x[idx])) - Decimal(float(mean[c]))) / _root(var[c])
            weight_grad[c] += Decimal(float(g[idx])) * y
        for c in range(channels):
            assert _ulps(got_w[c], weight_grad[c], dtype) <= 0.5, ("weight", c)
            bias = sum(Decimal(float(v)) for v in np.moveaxis(g, 1, 0)[c].ravel())
            assert _ulps(got_b[c], bias, dtype) <= 0.5, ("bias", c)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_instance_norm_takes_the_running_statistics_as_batch_norm_does(dtype):
    x, mean, var, w = _case((2, 3, 5), dtype, 67)
    with decimal.localcontext(prec=80):
        b = np.array(
            [-float(_exact(x[0, c, 0], mean[c], var[c], w[c], 0.0)) for c in range(3)]
        ).astype(dtype)
        out = F.instance_norm(
            _tensor(x, dtype),
            _tensor(mean, dtype),
            _tensor(var, dtype),
            _tensor(w, dtype),
            _tensor(b, dtype),
            use_input_stats=False,
            eps=_EPS,
        ).numpy()
        for idx in np.ndindex(x.shape):
            c = idx[1]
            want = _exact(x[idx], mean[c], var[c], w[c], b[c])
            assert _ulps(out[idx], want, dtype) <= 0.5, idx


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_a_value_that_is_not_finite_moves_no_other_output(dtype):
    x, mean, var, w = _case((3, 2, 4), dtype, 71)
    x[1, 0, 2] = np.inf
    x[2, 1, 0] = np.nan
    with decimal.localcontext(prec=80):
        out = F.batch_norm(
            _tensor(x, dtype),
            _tensor(mean, dtype),
            _tensor(var, dtype),
            _tensor(w, dtype),
            None,
            False,
            0.1,
            _EPS,
        ).numpy()
        assert out[1, 0, 2] == np.inf and np.isnan(out[2, 1, 0])
        for idx in np.ndindex(x.shape):
            if np.isfinite(x[idx]):
                want = _exact(x[idx], mean[idx[1]], var[idx[1]], w[idx[1]], 0.0)
                assert _ulps(out[idx], want, dtype) <= 0.5, idx


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_zeros_are_signed_as_the_formula_signs_them(dtype):
    # A zero input off a zero mean keeps its sign, and a value equal to a mean
    # that is not zero gives a positive zero; a missing weight and bias change
    # neither.
    x = np.array([[0.0, 0.25], [-0.0, 0.5]], dtype=dtype).reshape(2, 2, 1)
    mean = np.array([0.0, 0.25], dtype=dtype)
    var = np.array([1.0, 1.0], dtype=dtype)
    out = F.batch_norm(
        _tensor(x, dtype), _tensor(mean, dtype), _tensor(var, dtype), None, None, False
    ).numpy()
    for idx, negative in (((0, 0, 0), False), ((1, 0, 0), True), ((0, 1, 0), False)):
        assert out[idx] == 0 and bool(np.signbit(out[idx])) == negative, idx
