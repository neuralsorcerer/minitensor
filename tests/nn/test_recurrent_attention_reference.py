# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""LSTM/GRU/attention against NumPy transcriptions of the reference equations.

Gate ordering and where the GRU's reset gate is applied are exactly the details
that are easy to get subtly wrong and hard to notice: a wrong order still
trains, just worse, and still round-trips through the library's own tests. These
check the recurrences themselves, and pin the ordering by asserting that the
alternatives do *not* match.
"""

import numpy as np
import pytest

import minitensor as mt
from minitensor import functional as F

INPUT_SIZE, HIDDEN, SEQ, BATCH = 3, 4, 5, 2


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _softmax(x, axis=-1):
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def _assign(layer, w_ih, w_hh, b_ih, b_hh):
    """Set a recurrent layer's four parameters from NumPy arrays."""
    params = layer.parameters()
    for p in params:
        if tuple(p.shape) == w_ih.shape:
            p.copy_(mt.Tensor(w_ih))
        elif tuple(p.shape) == w_hh.shape:
            p.copy_(mt.Tensor(w_hh))
    biases = [p for p in params if len(p.shape) == 1]
    assert len(biases) == 2
    biases[0].copy_(mt.Tensor(b_ih))
    biases[1].copy_(mt.Tensor(b_hh))


def _weights(gates, seed):
    rng = np.random.default_rng(seed)
    return (
        rng.standard_normal((gates * HIDDEN, INPUT_SIZE)).astype(np.float32),
        rng.standard_normal((gates * HIDDEN, HIDDEN)).astype(np.float32),
        rng.standard_normal(gates * HIDDEN).astype(np.float32),
        rng.standard_normal(gates * HIDDEN).astype(np.float32),
    )


def _reference_lstm(x, w_ih, w_hh, b_ih, b_hh, order):
    h = np.zeros((BATCH, HIDDEN))
    c = np.zeros((BATCH, HIDDEN))
    out = []
    for t in range(SEQ):
        gates = x[t] @ w_ih.T + b_ih + h @ w_hh.T + b_hh
        part = {n: gates[:, k * HIDDEN : (k + 1) * HIDDEN] for k, n in enumerate(order)}
        c = _sigmoid(part["f"]) * c + _sigmoid(part["i"]) * np.tanh(part["g"])
        h = _sigmoid(part["o"]) * np.tanh(c)
        out.append(h.copy())
    return np.stack(out)


def _reference_gru(x, w_ih, w_hh, b_ih, b_hh, order):
    h = np.zeros((BATCH, HIDDEN))
    out = []
    for t in range(SEQ):
        gi = x[t] @ w_ih.T + b_ih
        gh = h @ w_hh.T + b_hh
        pi = {n: gi[:, k * HIDDEN : (k + 1) * HIDDEN] for k, n in enumerate(order)}
        ph = {n: gh[:, k * HIDDEN : (k + 1) * HIDDEN] for k, n in enumerate(order)}
        r = _sigmoid(pi["r"] + ph["r"])
        z = _sigmoid(pi["z"] + ph["z"])
        # The reset gate multiplies only the hidden contribution, inside tanh.
        n = np.tanh(pi["n"] + r * ph["n"])
        h = (1 - z) * n + z * h
        out.append(h.copy())
    return np.stack(out)


@pytest.fixture
def sequence():
    return (
        np.random.default_rng(77)
        .standard_normal((SEQ, BATCH, INPUT_SIZE))
        .astype(np.float32)
    )


def test_lstm_matches_the_reference_recurrence_with_ifgo_gate_order(sequence):
    w_ih, w_hh, b_ih, b_hh = _weights(4, 501)
    layer = mt.nn.LSTM(INPUT_SIZE, HIDDEN)
    _assign(layer, w_ih, w_hh, b_ih, b_hh)
    got = layer.forward(mt.Tensor(sequence)).numpy()

    args = [a.astype(np.float64) for a in (sequence, w_ih, w_hh, b_ih, b_hh)]
    np.testing.assert_allclose(
        got, _reference_lstm(*args, ("i", "f", "g", "o")), rtol=1e-4, atol=1e-5
    )
    # A different order would still produce plausible-looking output.
    for wrong in (("i", "f", "o", "g"), ("f", "i", "g", "o"), ("i", "g", "f", "o")):
        assert not np.allclose(got, _reference_lstm(*args, wrong), rtol=1e-3, atol=1e-4)


def test_gru_matches_the_reference_recurrence_with_rzn_gate_order(sequence):
    w_ih, w_hh, b_ih, b_hh = _weights(3, 502)
    layer = mt.nn.GRU(INPUT_SIZE, HIDDEN)
    _assign(layer, w_ih, w_hh, b_ih, b_hh)
    got = layer.forward(mt.Tensor(sequence)).numpy()

    args = [a.astype(np.float64) for a in (sequence, w_ih, w_hh, b_ih, b_hh)]
    np.testing.assert_allclose(
        got, _reference_gru(*args, ("r", "z", "n")), rtol=1e-4, atol=1e-5
    )
    for wrong in (("z", "r", "n"), ("n", "r", "z")):
        assert not np.allclose(got, _reference_gru(*args, wrong), rtol=1e-3, atol=1e-4)


@pytest.mark.parametrize("causal", [False, True])
def test_scaled_dot_product_attention_matches_closed_form(causal):
    rng = np.random.default_rng(131)
    b, heads, length, dim = 2, 3, 4, 5
    q, k, v = (rng.standard_normal((b, heads, length, dim)) for _ in range(3))

    got = F.scaled_dot_product_attention(
        mt.Tensor(q, dtype="float64"),
        mt.Tensor(k, dtype="float64"),
        mt.Tensor(v, dtype="float64"),
        is_causal=causal,
    ).numpy()

    scores = q @ np.swapaxes(k, -1, -2) / np.sqrt(dim)
    if causal:
        scores = np.where(
            np.triu(np.ones((length, length), dtype=bool), 1), -np.inf, scores
        )
    np.testing.assert_allclose(got, _softmax(scores) @ v, rtol=1e-12, atol=1e-13)


def test_multihead_attention_matches_an_explicit_projection_reference():
    embed, heads = 6, 2
    rng = np.random.default_rng(151)
    mats = [rng.standard_normal((embed, embed)).astype(np.float32) for _ in range(4)]
    vecs = [rng.standard_normal(embed).astype(np.float32) for _ in range(4)]

    layer = mt.nn.MultiheadAttention(embed, heads)
    params = layer.parameters()
    for p, w in zip([p for p in params if len(p.shape) == 2], mats):
        p.copy_(mt.Tensor(w))
    for p, b in zip([p for p in params if len(p.shape) == 1], vecs):
        p.copy_(mt.Tensor(b))

    x = rng.standard_normal((2, 4, embed)).astype(np.float32)
    got = layer.forward(mt.Tensor(x)).numpy()

    xa = x.astype(np.float64)
    wq, wk, wv, wo = (m.astype(np.float64) for m in mats)
    bq, bk, bv, bo = (b.astype(np.float64) for b in vecs)
    n, length, _ = xa.shape
    head_dim = embed // heads

    def project(w, b):
        return (xa @ w.T + b).reshape(n, length, heads, head_dim).transpose(0, 2, 1, 3)

    q, k, v = project(wq, bq), project(wk, bk), project(wv, bv)
    attended = _softmax(q @ np.swapaxes(k, -1, -2) / np.sqrt(head_dim)) @ v
    want = attended.transpose(0, 2, 1, 3).reshape(n, length, embed) @ wo.T + bo

    np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-5)
