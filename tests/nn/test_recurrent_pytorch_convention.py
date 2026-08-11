# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""The recurrent layout around the gate order: stacking, direction, and state.

`tests/nn/test_recurrent_attention_reference.py` already pins the gate order
itself -- LSTM's `i, f, g, o` and GRU's `r, z, n` -- against reference
recurrences, and asserts that the wrong orders disagree. This file covers the
rest of what the layered/directional names promise, which nothing checked.

`weight_ih_l{k}` / `weight_hh_l{k}` / `bias_ih_l{k}` / `bias_hh_l{k}`, with
`_reverse` for the backward direction, is an invitation to drop pretrained
tensors straight in. That only works if everything around each gate matrix
agrees too, and none of it is visible in the shapes: `batch_first` has to
transpose the same run rather than compute a different one, a second layer has
to consume the first layer's output, `bidirectional` has to concatenate a
forward pass with one that genuinely runs backwards, and the names have to be
exactly right for every combination of layers and directions.

`forward_with_state` is here for the same reason -- it returns the final hidden
and cell state, and accepts an initial one, and a caller porting a loop over
`(h, c)` needs both halves to behave.

One gate detail lives here rather than in the sibling file, because it is about
placement rather than order: GRU's reset gate scales the hidden contribution
*after* its matmul, `tanh(W_in x + b_in + r*(W_hn h + b_hn))`, not the hidden
state before it. The sibling file states that in a comment without testing it;
`test_the_gru_reset_gate_applies_after_the_hidden_matmul` builds the other model
and requires it to disagree.
"""

from __future__ import annotations

import numpy as np
import pytest

import minitensor as mt

nn = mt.nn
S = mt.serialization

SEQ, BATCH, INPUT, HIDDEN = 6, 3, 4, 5
TOL = 2e-5


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _rng():
    return np.random.default_rng(0)


def _load(layer, parameters):
    state = S.StateDict()
    for name, values in parameters.items():
        state.add_parameter(name, mt.Tensor(values.astype(np.float32)))
    layer.load_state_dict(state)
    return layer


def _run(layer, array):
    return layer(mt.Tensor(array.astype(np.float32))).numpy()


def _weights(rng, gates, inp, hid):
    return (
        rng.standard_normal((gates * hid, inp)),
        rng.standard_normal((gates * hid, hid)),
        rng.standard_normal(gates * hid),
        rng.standard_normal(gates * hid),
    )


def _named(weights, suffix="l0"):
    return dict(
        zip(
            [
                f"weight_ih_{suffix}",
                f"weight_hh_{suffix}",
                f"bias_ih_{suffix}",
                f"bias_hh_{suffix}",
            ],
            weights,
        )
    )


# --- the reference recurrence -----------------------------------------------


def _lstm_reference(x, w_ih, w_hh, b_ih, b_hh, reverse=False):
    """The `i, f, g, o` recurrence, which the sibling file pins."""
    hidden = w_hh.shape[1]
    slots = {
        name: slice(k * hidden, (k + 1) * hidden)
        for k, name in enumerate(("i", "f", "g", "o"))
    }
    h = np.zeros((x.shape[1], hidden))
    c = h.copy()
    steps = range(x.shape[0] - 1, -1, -1) if reverse else range(x.shape[0])

    outputs = []
    for t in steps:
        gates = x[t] @ w_ih.T + b_ih + h @ w_hh.T + b_hh
        i = _sigmoid(gates[:, slots["i"]])
        f = _sigmoid(gates[:, slots["f"]])
        g = np.tanh(gates[:, slots["g"]])
        o = _sigmoid(gates[:, slots["o"]])
        c = f * c + i * g
        h = o * np.tanh(c)
        outputs.append(h.copy())

    if reverse:
        outputs.reverse()
    return np.stack(outputs)


# --- the one gate detail the sibling file does not test ----------------------


def test_the_gru_reset_gate_applies_after_the_hidden_matmul():
    """The sibling file's reference states this in a comment; nothing checked
    it. The other placement -- scaling `h` before `W_hn` rather than scaling
    what comes out of it -- is a different model, so it has to disagree."""
    rng = _rng()
    x = rng.standard_normal((SEQ, BATCH, INPUT))
    w_ih, w_hh, b_ih, b_hh = _weights(rng, 3, INPUT, HIDDEN)
    got = _run(_load(nn.GRU(INPUT, HIDDEN), _named((w_ih, w_hh, b_ih, b_hh))), x)

    h = np.zeros((BATCH, HIDDEN))
    outputs = []
    for t in range(SEQ):
        gi = x[t] @ w_ih.T + b_ih
        r = _sigmoid(gi[:, :HIDDEN] + (h @ w_hh.T + b_hh)[:, :HIDDEN])
        z = _sigmoid(
            gi[:, HIDDEN : 2 * HIDDEN] + (h @ w_hh.T + b_hh)[:, HIDDEN : 2 * HIDDEN]
        )
        wrong = (r * h) @ w_hh.T + b_hh  # reset applied to the state instead
        n = np.tanh(gi[:, 2 * HIDDEN :] + wrong[:, 2 * HIDDEN :])
        h = (1 - z) * n + z * h
        outputs.append(h.copy())

    assert not np.allclose(got, np.stack(outputs), atol=TOL)


# --- the layout the parameter names promise ----------------------------------


def test_batch_first_is_the_same_run_transposed():
    rng = _rng()
    x = rng.standard_normal((SEQ, BATCH, INPUT))
    weights = _named(_weights(rng, 4, INPUT, HIDDEN))

    seq_major = _run(_load(nn.LSTM(INPUT, HIDDEN), weights), x)
    batch_major = _run(
        _load(nn.LSTM(INPUT, HIDDEN, batch_first=True), weights),
        np.transpose(x, (1, 0, 2)),
    )
    np.testing.assert_allclose(
        seq_major, np.transpose(batch_major, (1, 0, 2)), atol=TOL
    )


def test_a_second_layer_consumes_the_first_layers_output():
    rng = _rng()
    x = rng.standard_normal((SEQ, BATCH, INPUT))
    first = _weights(rng, 4, INPUT, HIDDEN)
    second = _weights(rng, 4, HIDDEN, HIDDEN)

    parameters = _named(first, "l0")
    parameters.update(_named(second, "l1"))
    got = _run(_load(nn.LSTM(INPUT, HIDDEN, num_layers=2), parameters), x)

    np.testing.assert_allclose(
        got, _lstm_reference(_lstm_reference(x, *first), *second), atol=TOL
    )


def test_bidirectional_concatenates_a_forward_and_a_backward_pass():
    rng = _rng()
    x = rng.standard_normal((SEQ, BATCH, INPUT))
    forward = _weights(rng, 4, INPUT, HIDDEN)
    backward = _weights(rng, 4, INPUT, HIDDEN)

    parameters = _named(forward, "l0")
    parameters.update(_named(backward, "l0_reverse"))
    layer = _load(nn.LSTM(INPUT, HIDDEN, bidirectional=True), parameters)

    want = np.concatenate(
        [
            _lstm_reference(x, *forward),
            _lstm_reference(x, *backward, reverse=True),
        ],
        axis=-1,
    )
    got = _run(layer, x)
    assert got.shape == (SEQ, BATCH, 2 * HIDDEN)
    np.testing.assert_allclose(got, want, atol=TOL)


def test_the_parameter_names_are_the_ones_pytorch_uses():
    plain = set(nn.LSTM(INPUT, HIDDEN).state_dict().keys())
    assert plain == {"weight_ih_l0", "weight_hh_l0", "bias_ih_l0", "bias_hh_l0"}

    both = set(nn.LSTM(INPUT, HIDDEN, bidirectional=True).state_dict().keys())
    assert both == plain | {f"{name}_reverse" for name in plain}

    stacked = set(nn.GRU(INPUT, HIDDEN, num_layers=2).state_dict().keys())
    assert stacked == {
        f"{kind}_{gate}_l{layer}"
        for kind in ("weight", "bias")
        for gate in ("ih", "hh")
        for layer in (0, 1)
    }


# --- the returned state -----------------------------------------------------


def test_forward_with_state_returns_the_final_hidden_and_cell():
    rng = _rng()
    x = rng.standard_normal((SEQ, BATCH, INPUT))
    weights = _weights(rng, 4, INPUT, HIDDEN)
    layer = _load(nn.LSTM(INPUT, HIDDEN), _named(weights))

    output, (h_n, c_n) = layer.forward_with_state(mt.Tensor(x.astype(np.float32)))
    reference = _lstm_reference(x, *weights)

    np.testing.assert_allclose(output.numpy(), reference, atol=TOL)
    assert tuple(h_n.shape_vec()) == (1, BATCH, HIDDEN)
    np.testing.assert_allclose(
        h_n.numpy().reshape(BATCH, HIDDEN), reference[-1], atol=TOL
    )
    assert tuple(c_n.shape_vec()) == (1, BATCH, HIDDEN)


def test_forward_matches_the_output_half_of_forward_with_state():
    rng = _rng()
    x = mt.Tensor(rng.standard_normal((SEQ, BATCH, INPUT)).astype(np.float32))
    layer = nn.LSTM(INPUT, HIDDEN)
    np.testing.assert_array_equal(
        layer(x).numpy(), layer.forward_with_state(x)[0].numpy()
    )


def test_an_explicit_initial_state_is_used():
    rng = _rng()
    x = rng.standard_normal((SEQ, BATCH, INPUT))
    weights = _weights(rng, 4, INPUT, HIDDEN)
    layer = _load(nn.LSTM(INPUT, HIDDEN), _named(weights))

    h0 = rng.standard_normal((1, BATCH, HIDDEN))
    c0 = rng.standard_normal((1, BATCH, HIDDEN))
    output, _ = layer.forward_with_state(
        mt.Tensor(x.astype(np.float32)),
        mt.Tensor(h0.astype(np.float32)),
        mt.Tensor(c0.astype(np.float32)),
    )

    hidden = weights[1].shape[1]
    h, c = h0[0].copy(), c0[0].copy()
    expected = []
    for t in range(SEQ):
        gates = x[t] @ weights[0].T + weights[2] + h @ weights[1].T + weights[3]
        i = _sigmoid(gates[:, :hidden])
        f = _sigmoid(gates[:, hidden : 2 * hidden])
        g = np.tanh(gates[:, 2 * hidden : 3 * hidden])
        o = _sigmoid(gates[:, 3 * hidden :])
        c = f * c + i * g
        h = o * np.tanh(c)
        expected.append(h.copy())

    np.testing.assert_allclose(output.numpy(), np.stack(expected), atol=TOL)


def test_a_cell_state_without_a_hidden_state_is_rejected():
    rng = _rng()
    x = mt.Tensor(rng.standard_normal((SEQ, BATCH, INPUT)).astype(np.float32))
    c0 = mt.Tensor(rng.standard_normal((1, BATCH, HIDDEN)).astype(np.float32))

    with pytest.raises(Exception) as excinfo:
        nn.LSTM(INPUT, HIDDEN).forward_with_state(x, None, c0)
    assert "hx" in str(excinfo.value)
