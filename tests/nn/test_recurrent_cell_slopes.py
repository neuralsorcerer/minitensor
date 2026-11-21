# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""LSTM and GRU cell gradients are taken of the logits, without cancellation.

The fused cells formed their slopes of the activations the forward had
rounded -- `i (1 - i)`, `1 - g^2` and `1 - tanh(c)^2` for an LSTM, `r (1 - r)`,
`1 - z` and `1 - n^2` for a GRU -- each a difference of numbers near one once
its gate saturates. The slopes lost their digits on the way there and were
zero from a logit of 17 in float32, while still floats, so that ordinary
logits left gradients up to a million ulps out and saturated ones none at all.

Every value and slope is now formed again in float64 from the logits, the cell
state from them, and each gradient rounded once: for float32 the float nearest
the exact one, and for float64 within a dozen ulps or so. Each case here runs
one step of a layer whose input weights are the identity and whose hidden
weights are zero, so that the cell sees exactly the logits given.
"""

from __future__ import annotations

import decimal
import math
from decimal import Decimal
from fractions import Fraction

import numpy as np
import pytest

import minitensor as mt

_CONTEXT = dict(prec=100, Emax=10**9, Emin=-(10**9))
_DIGITS = {"float32": (24, -149, 128), "float64": (53, -1074, 1024)}
# Products of four or five float64 factors, each a few ulps out.
_FLOAT64_ULPS = 16


def _rounded(value: Decimal, dtype: str) -> float:
    """The float nearest a decimal value, ties to even, rounded once."""

    if value == 0:
        return 0.0
    exact = Fraction(abs(value))
    digits, least, top = _DIGITS[dtype]
    e = exact.numerator.bit_length() - exact.denominator.bit_length()
    if Fraction(2) ** e > exact:
        e -= 1
    unit = Fraction(2) ** max(e - (digits - 1), least)
    near = round(exact / unit) * unit
    out = math.inf if near >= Fraction(2) ** top else float(near)
    return -out if value < 0 else out


def _sigmoid(x: Decimal) -> Decimal:
    return 1 / (1 + (-x).exp())


def _sigmoid_slope(x: Decimal) -> Decimal:
    return _sigmoid(x) * _sigmoid(-x)


def _tanh(x: Decimal) -> Decimal:
    if x == 0:
        return x
    if x < 0:
        return -_tanh(-x)
    # `1 - e` cancels the digits of a small `x` away: carry as many more.
    with decimal.localcontext() as context:
        context.prec += max(0, -x.adjusted()) + 5
        e = (-2 * x).exp()
        y = (1 - e) / (1 + e)
    return +y


def _tanh_slope(x: Decimal) -> Decimal:
    e = (-2 * abs(x)).exp()
    return 4 * e / (1 + e) ** 2


def _spacing(value: float, dtype: str) -> float:
    return float(np.spacing(np.array(abs(value), dtype=dtype)))


def _check(got, want, dtype, bound=None):
    """Each of `got` is the float nearest its exact value in `want` -- for
    float64 within `bound` ulps of it."""

    bound = _FLOAT64_ULPS if bound is None and dtype == "float64" else bound
    for index in np.ndindex(np.shape(got)):
        g, w = float(got[index]), _rounded(want[index], dtype)
        if bound is None:
            assert g == w, (index, g, w)
        else:
            assert abs(g - w) <= bound * _spacing(w, dtype), (index, g, w)


def _lstm(dtype, logits, before, grad, of_hidden=True):
    """The gradients of one LSTM step's logits and cell state before, for
    `grad` of the hidden state or else of the cell state."""

    batch, hidden = before.shape
    with mt.default_dtype(dtype):
        layer = mt.nn.LSTM(4 * hidden, hidden)
    weights = dict(layer.named_parameters())
    weights["weight_ih_l0"].copy_(mt.Tensor(np.eye(4 * hidden), dtype=dtype))
    weights["weight_hh_l0"].copy_(
        mt.Tensor(np.zeros((4 * hidden, hidden)), dtype=dtype)
    )
    for name in ("bias_ih_l0", "bias_hh_l0"):
        weights[name].copy_(mt.Tensor(np.zeros(4 * hidden), dtype=dtype))
    x = mt.Tensor(logits.reshape(1, batch, 4 * hidden), dtype=dtype, requires_grad=True)
    c0 = mt.Tensor(before.reshape(1, batch, hidden), dtype=dtype, requires_grad=True)
    h0 = mt.Tensor(np.zeros((1, batch, hidden)), dtype=dtype)
    out, (_, c_n) = layer.forward_with_state(x, h0, c0)
    probe = mt.Tensor(grad.reshape(1, batch, hidden), dtype=dtype)
    ((out if of_hidden else c_n) * probe).sum().backward()
    got = (
        x.grad.numpy().reshape(batch, 4 * hidden).astype(np.float64),
        c0.grad.numpy().reshape(batch, hidden).astype(np.float64),
    )
    mt.clear_autograd_graph()
    return got


def _lstm_exact(logits, before, grad, of_hidden=True):
    batch, hidden = before.shape
    d_logits = np.empty((batch, 4 * hidden), dtype=object)
    d_before = np.empty((batch, hidden), dtype=object)
    with decimal.localcontext(**_CONTEXT):
        for b, j in np.ndindex(batch, hidden):
            xi, xf, xg, xo = (
                Decimal(float(logits[b, k * hidden + j])) for k in range(4)
            )
            c_before, g_in = Decimal(float(before[b, j])), Decimal(float(grad[b, j]))
            i, f, g, o = _sigmoid(xi), _sigmoid(xf), _tanh(xg), _sigmoid(xo)
            c = f * c_before + i * g
            dc = g_in * o * _tanh_slope(c) if of_hidden else g_in
            d_o = g_in * _tanh(c) * _sigmoid_slope(xo) if of_hidden else Decimal(0)
            d_logits[b, j] = dc * g * _sigmoid_slope(xi)
            d_logits[b, hidden + j] = dc * c_before * _sigmoid_slope(xf)
            d_logits[b, 2 * hidden + j] = dc * i * _tanh_slope(xg)
            d_logits[b, 3 * hidden + j] = d_o
            d_before[b, j] = dc * f
    return d_logits, d_before


def _gru(dtype, from_input, from_hidden, before, grad):
    """The gradients of one GRU step's input projection, hidden projection
    (through its bias, so one row) and hidden state before."""

    batch, hidden = before.shape
    with mt.default_dtype(dtype):
        layer = mt.nn.GRU(3 * hidden, hidden)
    weights = dict(layer.named_parameters())
    weights["weight_ih_l0"].copy_(mt.Tensor(np.eye(3 * hidden), dtype=dtype))
    weights["weight_hh_l0"].copy_(
        mt.Tensor(np.zeros((3 * hidden, hidden)), dtype=dtype)
    )
    weights["bias_ih_l0"].copy_(mt.Tensor(np.zeros(3 * hidden), dtype=dtype))
    weights["bias_hh_l0"].copy_(mt.Tensor(from_hidden, dtype=dtype))
    x = mt.Tensor(
        from_input.reshape(1, batch, 3 * hidden), dtype=dtype, requires_grad=True
    )
    h0 = mt.Tensor(before.reshape(1, batch, hidden), dtype=dtype, requires_grad=True)
    out, _ = layer.forward_with_state(x, h0)
    (out * mt.Tensor(grad.reshape(1, batch, hidden), dtype=dtype)).sum().backward()
    bias = dict(layer.named_parameters())["bias_hh_l0"].grad
    got = (
        x.grad.numpy().reshape(batch, 3 * hidden).astype(np.float64),
        bias.numpy().astype(np.float64) if batch == 1 else None,
        h0.grad.numpy().reshape(batch, hidden).astype(np.float64),
    )
    mt.clear_autograd_graph()
    return got


def _gru_exact(from_input, from_hidden, before, grad):
    batch, hidden = before.shape
    d_input = np.empty((batch, 3 * hidden), dtype=object)
    d_hidden = np.empty(3 * hidden, dtype=object)
    d_before = np.empty((batch, hidden), dtype=object)
    with decimal.localcontext(**_CONTEXT):
        for row, j in np.ndindex(batch, hidden):
            ar, az, an = (
                Decimal(float(from_input[row, k * hidden + j])) for k in range(3)
            )
            br, bz, bn = (Decimal(float(from_hidden[k * hidden + j])) for k in range(3))
            h, g_in = Decimal(float(before[row, j])), Decimal(float(grad[row, j]))
            r = _sigmoid(ar + br)
            x_n = an + r * bn
            dn = g_in * _sigmoid(-(az + bz)) * _tanh_slope(x_n)
            d_r = dn * bn * _sigmoid_slope(ar + br)
            d_z = g_in * (h - _tanh(x_n)) * _sigmoid_slope(az + bz)
            d_input[row, j], d_input[row, hidden + j], d_input[row, 2 * hidden + j] = (
                d_r,
                d_z,
                dn,
            )
            d_hidden[j], d_hidden[hidden + j], d_hidden[2 * hidden + j] = (
                d_r,
                d_z,
                dn * r,
            )
            d_before[row, j] = g_in * _sigmoid(az + bz)
    return d_input, d_hidden, d_before


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("of_hidden", [True, False], ids=["hidden", "cell"])
@pytest.mark.parametrize("scale", [3.0, 25.0], ids=["ordinary", "saturated"])
def test_lstm_gradients(dtype, of_hidden, scale):
    rng = np.random.default_rng(2101 + int(scale))
    logits = (rng.standard_normal((3, 4 * 24)) * scale).astype(dtype)
    before = (rng.standard_normal((3, 24)) * scale / 2).astype(dtype)
    grad = rng.standard_normal((3, 24)).astype(dtype)
    got = _lstm(dtype, logits, before, grad, of_hidden)
    want = _lstm_exact(logits, before, grad, of_hidden)
    for g, w in zip(got, want):
        _check(g, w, dtype)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_saturated_gates_keep_their_slopes(dtype):
    # Logits past where every gate rounds to one in float32 -- and, the larger
    # ones, in float64 -- and a cell state whose tanh does too.
    span = [18.0, 30.0, 45.0] if dtype == "float32" else [40.0, 60.0, 90.0]
    logits, before = [], []
    for big in span:
        for signs in ((1, 1, -1, 1), (-1, 1, 1, -1), (1, -1, 1, 1)):
            logits.append([s * big for s in signs])
            before.append(big / 3)
    logits = np.array(logits).T.reshape(1, -1).astype(dtype)
    before = np.array(before).reshape(1, -1).astype(dtype)
    grad = np.ones_like(before)
    got = _lstm(dtype, logits, before, grad)
    want = _lstm_exact(logits, before, grad)
    for g, w in zip(got, want):
        _check(g, w, dtype)
        for index in np.ndindex(g.shape):
            if _rounded(w[index], dtype) != 0:
                assert g[index] != 0, index


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_lstm_gradients_along_every_row(dtype):
    # Enough rows to split into parallel bands.
    rng = np.random.default_rng(2111)
    logits = (rng.standard_normal((40, 4 * 32)) * 4).astype(dtype)
    before = (rng.standard_normal((40, 32)) * 3).astype(dtype)
    grad = rng.standard_normal((40, 32)).astype(dtype)
    got = _lstm(dtype, logits, before, grad)
    want = _lstm_exact(logits, before, grad)
    for g, w in zip(got, want):
        _check(g, w, dtype)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("scale", [3.0, 40.0], ids=["ordinary", "saturated"])
@pytest.mark.parametrize("batch", [1, 4])
def test_gru_gradients(dtype, scale, batch):
    rng = np.random.default_rng(2131 + int(scale) + batch)
    from_input = (rng.standard_normal((batch, 3 * 32)) * scale).astype(dtype)
    from_hidden = (rng.standard_normal(3 * 32) * scale).astype(dtype)
    before = rng.standard_normal((batch, 32)).astype(dtype)
    grad = rng.standard_normal((batch, 32)).astype(dtype)
    got = _gru(dtype, from_input, from_hidden, before, grad)
    want = _gru_exact(from_input, from_hidden, before, grad)
    for g, w in zip(got, want):
        if g is not None:
            _check(g, w, dtype)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_where_the_cell_state_cancels(dtype):
    # `c_before` set to `-i g / f` rounded, so that `c` is a last place of its
    # two terms; float64 carries it to about 2^-95 of them.
    rng = np.random.default_rng(2141)
    logits = (rng.standard_normal((2, 4 * 24)) * 3).astype(dtype)
    before = np.empty((2, 24))
    with decimal.localcontext(**_CONTEXT):
        for b, j in np.ndindex(2, 24):
            xi, xf, xg = (Decimal(float(logits[b, k * 24 + j])) for k in range(3))
            before[b, j] = float(-(_sigmoid(xi) * _tanh(xg)) / _sigmoid(xf))
    before = before.astype(dtype)
    grad = rng.standard_normal((2, 24)).astype(dtype)
    got = _lstm(dtype, logits, before, grad)
    want = _lstm_exact(logits, before, grad)
    for g, w in zip(got, want):
        if dtype == "float32":
            _check(g, w, dtype)
        else:
            for index in np.ndindex(g.shape):
                exact = float(w[index])
                assert abs(g[index] - exact) <= 2.0**-40 * abs(exact), index


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_where_the_state_before_is_the_candidate(dtype):
    # `h` set to `n` rounded, so that `h - n` is a last place of either.
    rng = np.random.default_rng(2151)
    from_input = (rng.standard_normal((1, 3 * 32)) * 2).astype(dtype)
    from_hidden = (rng.standard_normal(3 * 32) * 2).astype(dtype)
    before = np.empty((1, 32))
    with decimal.localcontext(**_CONTEXT):
        for j in range(32):
            r = _sigmoid(
                Decimal(float(from_input[0, j])) + Decimal(float(from_hidden[j]))
            )
            x_n = Decimal(float(from_input[0, 64 + j])) + r * Decimal(
                float(from_hidden[64 + j])
            )
            before[0, j] = float(_tanh(x_n))
    before = before.astype(dtype)
    grad = rng.standard_normal((1, 32)).astype(dtype)
    got = _gru(dtype, from_input, from_hidden, before, grad)
    want = _gru_exact(from_input, from_hidden, before, grad)
    for g, w in zip(got, want):
        if dtype == "float32":
            _check(g, w, dtype)
        else:
            for index in np.ndindex(g.shape):
                exact = float(w[index])
                assert abs(g[index] - exact) <= 2.0**-40 * abs(exact), index


def test_float64_slopes_below_the_range():
    # Logits past 708 and cell states past 354, whose slopes float64 holds
    # only to their last few bits, under cell states before that bring the
    # gradients back up: each gradient is still rounded once.
    logits = np.array(
        [
            [-4.67, -630.1, -370.8, 265.2],
            [255.2, 727.0, -213.7, -214.6],
            [118.4, 744.0, 120.5, 35.9],
            [-715.9, 2.0, 0.7, -1.5],
            [3.0, 1.0, -0.4, 712.0],
        ]
    )
    before = np.array([1434.4, 1599.9, -227.2, 1e300, 400.0])
    logits = logits.T.reshape(1, -1)
    before = before.reshape(1, -1)
    grad = np.array([[-0.56, -0.87, 0.28, 0.5, 1.3]])
    for of_hidden in (True, False):
        got = _lstm("float64", logits, before, grad, of_hidden)
        want = _lstm_exact(logits, before, grad, of_hidden)
        for g, w in zip(got, want):
            _check(g, w, "float64")
