# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the LSTM and GRU layers.

The cells are checked against a NumPy transcription of the published recurrences
rather than against each other, so a shared misreading cannot pass. The layers
are built from ordinary autograd-aware operations, so the interesting risks are
in the cell arithmetic, the layer stacking and the axis conventions rather than
in a hand-written backward.
"""

import numpy as np
import pytest

import minitensor as mt
from minitensor import functional as F
from minitensor import nn


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _blocks(row, count, width):
    return [row[:, k * width : (k + 1) * width] for k in range(count)]


def lstm_reference(x, layer_params, h0, c0):
    """(seq, batch, feature) -> outputs, final h, final c."""
    layer_input = x
    final_h, final_c = [], []
    for index, (w_ih, w_hh, b_ih, b_hh) in enumerate(layer_params):
        h, c = h0[index].copy(), c0[index].copy()
        width = h.shape[1]
        outputs = []
        for t in range(x.shape[0]):
            gates = layer_input[t] @ w_ih.T + b_ih + h @ w_hh.T + b_hh
            i, f, g, o = _blocks(gates, 4, width)
            i, f, g, o = _sigmoid(i), _sigmoid(f), np.tanh(g), _sigmoid(o)
            c = f * c + i * g
            h = o * np.tanh(c)
            outputs.append(h.copy())
        layer_input = np.stack(outputs)
        final_h.append(h.copy())
        final_c.append(c.copy())
    return layer_input, np.stack(final_h), np.stack(final_c)


def gru_reference(x, layer_params, h0):
    layer_input = x
    final_h = []
    for index, (w_ih, w_hh, b_ih, b_hh) in enumerate(layer_params):
        h = h0[index].copy()
        width = h.shape[1]
        outputs = []
        for t in range(x.shape[0]):
            from_input = layer_input[t] @ w_ih.T + b_ih
            from_hidden = h @ w_hh.T + b_hh
            i_r, i_z, i_n = _blocks(from_input, 3, width)
            h_r, h_z, h_n = _blocks(from_hidden, 3, width)
            reset = _sigmoid(i_r + h_r)
            update = _sigmoid(i_z + h_z)
            # The reset gate scales the *projected* hidden term.
            candidate = np.tanh(i_n + reset * h_n)
            h = (1 - update) * candidate + update * h
            outputs.append(h.copy())
        layer_input = np.stack(outputs)
        final_h.append(h.copy())
    return layer_input, np.stack(final_h)


def _layer_params(layer, num_layers):
    """Group `parameters()` into per-layer (w_ih, w_hh, b_ih, b_hh) tuples."""
    flat = [p.numpy() for p in layer.parameters()]
    return [tuple(flat[i * 4 : (i + 1) * 4]) for i in range(num_layers)]


@pytest.mark.parametrize("num_layers", [1, 2, 3])
def test_lstm_matches_reference(num_layers):
    rng = np.random.default_rng(0)
    seq, batch, features, hidden = 6, 3, 4, 5
    layer = nn.LSTM(features, hidden, num_layers=num_layers, dtype="float64")

    x = rng.standard_normal((seq, batch, features))
    h0 = rng.standard_normal((num_layers, batch, hidden))
    c0 = rng.standard_normal((num_layers, batch, hidden))

    output, (h_n, c_n) = layer.forward_with_state(
        mt.Tensor(x, dtype="float64"),
        mt.Tensor(h0, dtype="float64"),
        mt.Tensor(c0, dtype="float64"),
    )
    want_out, want_h, want_c = lstm_reference(
        x, _layer_params(layer, num_layers), h0, c0
    )

    np.testing.assert_allclose(output.numpy(), want_out, atol=1e-12)
    np.testing.assert_allclose(h_n.numpy(), want_h, atol=1e-12)
    np.testing.assert_allclose(c_n.numpy(), want_c, atol=1e-12)


@pytest.mark.parametrize("num_layers", [1, 2, 3])
def test_gru_matches_reference(num_layers):
    rng = np.random.default_rng(1)
    seq, batch, features, hidden = 6, 3, 4, 5
    layer = nn.GRU(features, hidden, num_layers=num_layers, dtype="float64")

    x = rng.standard_normal((seq, batch, features))
    h0 = rng.standard_normal((num_layers, batch, hidden))

    output, h_n = layer.forward_with_state(
        mt.Tensor(x, dtype="float64"), mt.Tensor(h0, dtype="float64")
    )
    want_out, want_h = gru_reference(x, _layer_params(layer, num_layers), h0)

    np.testing.assert_allclose(output.numpy(), want_out, atol=1e-12)
    np.testing.assert_allclose(h_n.numpy(), want_h, atol=1e-12)


def test_gru_reset_gate_scales_the_projected_hidden_term():
    """The detail GRU implementations most often get wrong.

    `n = tanh(W_in x + r * (W_hn h + b_hn))` is not the same function as
    `n = tanh(W_in x + W_hn (r * h) + b_hn)` -- among other things the first
    lets `r` scale the bias. This pins which one is implemented, and shows the
    alternative is not merely a rounding difference away.
    """
    rng = np.random.default_rng(2)
    seq, batch, features, hidden = 5, 2, 3, 4
    layer = nn.GRU(features, hidden, dtype="float64")
    w_ih, w_hh, b_ih, b_hh = (p.numpy() for p in layer.parameters())

    x = rng.standard_normal((seq, batch, features))
    h0 = rng.standard_normal((1, batch, hidden))

    def run(scale_before_projection):
        h = h0[0].copy()
        for t in range(seq):
            from_input = x[t] @ w_ih.T + b_ih
            i_r, i_z, i_n = _blocks(from_input, 3, hidden)
            from_hidden = h @ w_hh.T + b_hh
            h_r, h_z, h_n = _blocks(from_hidden, 3, hidden)
            reset = _sigmoid(i_r + h_r)
            update = _sigmoid(i_z + h_z)
            if scale_before_projection:
                gated = (reset * h) @ w_hh.T + b_hh
                candidate = np.tanh(i_n + gated[:, 2 * hidden : 3 * hidden])
            else:
                candidate = np.tanh(i_n + reset * h_n)
            h = (1 - update) * candidate + update * h
        return h

    output, _ = layer.forward_with_state(
        mt.Tensor(x, dtype="float64"), mt.Tensor(h0, dtype="float64")
    )
    final = output.numpy()[-1]

    np.testing.assert_allclose(final, run(scale_before_projection=False), atol=1e-12)
    assert np.abs(final - run(scale_before_projection=True)).max() > 1e-3


@pytest.mark.parametrize("kind", ["LSTM", "GRU"])
def test_zero_initial_state_is_the_default(kind):
    rng = np.random.default_rng(3)
    seq, batch, features, hidden = 4, 2, 3, 5
    layer = getattr(nn, kind)(features, hidden, dtype="float64")
    x = mt.Tensor(rng.standard_normal((seq, batch, features)), dtype="float64")

    implicit = layer(x).numpy()
    zeros = mt.Tensor(np.zeros((1, batch, hidden)), dtype="float64")
    if kind == "LSTM":
        explicit, _ = layer.forward_with_state(x, zeros, zeros)
    else:
        explicit, _ = layer.forward_with_state(x, zeros)
    np.testing.assert_allclose(implicit, explicit.numpy(), atol=1e-12)


@pytest.mark.parametrize("kind", ["LSTM", "GRU"])
def test_batch_first_transposes_both_ends(kind):
    rng = np.random.default_rng(4)
    seq, batch, features, hidden = 6, 2, 3, 4
    layer = getattr(nn, kind)(features, hidden, batch_first=True, dtype="float64")

    x_seq_first = rng.standard_normal((seq, batch, features))
    x_batch_first = x_seq_first.transpose(1, 0, 2).copy()

    output = layer(mt.Tensor(x_batch_first, dtype="float64")).numpy()
    assert output.shape == (batch, seq, hidden)

    params = _layer_params(layer, 1)
    zeros = np.zeros((1, batch, hidden))
    if kind == "LSTM":
        want, _, _ = lstm_reference(x_seq_first, params, zeros, zeros)
    else:
        want, _ = gru_reference(x_seq_first, params, zeros)
    np.testing.assert_allclose(output.transpose(1, 0, 2), want, atol=1e-12)


@pytest.mark.parametrize("kind", ["LSTM", "GRU"])
@pytest.mark.parametrize("num_layers", [1, 2])
def test_every_parameter_receives_gradient(kind, num_layers):
    layer = getattr(nn, kind)(3, 4, num_layers=num_layers, dtype="float64")
    x = mt.Tensor(np.random.default_rng(5).standard_normal((5, 2, 3)), dtype="float64")
    layer(x).sum().backward()

    params = layer.parameters()
    assert len(params) == 4 * num_layers
    for index, param in enumerate(params):
        assert param.grad is not None, f"parameter {index} got no gradient"
        assert (
            np.abs(param.grad.numpy()).max() > 0.0
        ), f"parameter {index} gradient is all zero"
    mt.clear_autograd_graph()


@pytest.mark.parametrize("kind", ["LSTM", "GRU"])
def test_gradient_matches_central_differences(kind):
    """Gradients accumulate across timesteps; this checks the whole unrolled sum."""
    rng = np.random.default_rng(6)
    layer = getattr(nn, kind)(2, 3, dtype="float64")
    x_np = rng.standard_normal((4, 2, 2))
    weights = rng.standard_normal((4, 2, 3))

    x = mt.Tensor(x_np, dtype="float64", requires_grad=True)
    (layer(x) * mt.Tensor(weights, dtype="float64")).sum().backward()
    analytic = x.grad.numpy()
    mt.clear_autograd_graph()

    def loss_at(values):
        return float(
            (layer(mt.Tensor(values, dtype="float64")).numpy() * weights).sum()
        )

    h = 1e-6
    for idx in np.ndindex(*x_np.shape):
        plus, minus = x_np.copy(), x_np.copy()
        plus[idx] += h
        minus[idx] -= h
        central = (loss_at(plus) - loss_at(minus)) / (2 * h)
        np.testing.assert_allclose(analytic[idx], central, atol=1e-7)


@pytest.mark.parametrize("kind", ["LSTM", "GRU"])
def test_bias_can_be_disabled(kind):
    layer = getattr(nn, kind)(3, 4, bias=False, dtype="float64")
    assert layer.bias is False
    # Only the two weight matrices remain.
    assert len(layer.parameters()) == 2
    x = mt.Tensor(np.zeros((2, 1, 3)), dtype="float64")
    # With no bias and a zero input and state, every gate input is zero, so an
    # LSTM emits zeros and a GRU emits (1 - sigmoid(0)) * tanh(0) = 0 as well.
    np.testing.assert_allclose(layer(x).numpy(), np.zeros((2, 1, 4)), atol=1e-12)


@pytest.mark.parametrize("kind", ["LSTM", "GRU"])
def test_learns_a_task_that_requires_memory(kind):
    """Predict the sum of a sequence -- impossible without carrying state.

    The last timestep is sliced with `narrow` rather than read through NumPy, so
    the graph stays intact and the recurrent weights actually train; the check
    that they moved is what distinguishes this from only fitting the head.
    """
    rng = np.random.default_rng(0)
    seq, batch, hidden = 5, 16, 8
    x = rng.standard_normal((seq, batch, 1)).astype(np.float32)
    y = x.sum(axis=0)

    mt.manual_seed(0)
    rnn = getattr(nn, kind)(1, hidden)
    head = nn.DenseLayer(hidden, 1)
    before = rnn.parameters()[0].numpy().copy()
    optimizer = mt.optim.Adam(rnn.parameters() + head.parameters(), lr=0.05)

    inputs, targets = mt.Tensor(x), mt.Tensor(y)
    first = last = None
    for epoch in range(200):
        output = rnn(inputs)
        final_step = F.narrow(output, 0, seq - 1, 1).reshape(batch, hidden)
        prediction = head(final_step)
        error = prediction - targets
        loss = (error * error).sum()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if epoch == 0:
            first = loss.item()
        last = loss.item()
        mt.clear_autograd_graph()

    assert last < first / 10.0, f"loss barely moved: {first} -> {last}"
    assert (
        np.abs(before - rnn.parameters()[0].numpy()).max() > 1e-3
    ), "recurrent weights did not train; only the head was fitted"


@pytest.mark.parametrize("kind", ["LSTM", "GRU"])
def test_reports_its_configuration(kind):
    layer = getattr(nn, kind)(3, 7, num_layers=2, batch_first=True)
    assert layer.input_size == 3
    assert layer.hidden_size == 7
    assert layer.num_layers == 2
    assert layer.batch_first is True
    assert layer.bias is True
    assert kind in repr(layer)


@pytest.mark.parametrize("kind", ["LSTM", "GRU"])
def test_rejects_malformed_inputs(kind):
    layer = getattr(nn, kind)(3, 4, dtype="float64")
    good = mt.Tensor(np.zeros((5, 2, 3)), dtype="float64")

    with pytest.raises(Exception):  # 2-D input
        layer(mt.Tensor(np.zeros((5, 3)), dtype="float64"))
    with pytest.raises(Exception):  # wrong feature width
        layer(mt.Tensor(np.zeros((5, 2, 9)), dtype="float64"))
    with pytest.raises(Exception):  # hidden state of the wrong shape
        layer.forward_with_state(good, mt.Tensor(np.zeros((1, 2, 9)), dtype="float64"))
    with pytest.raises(Exception):  # empty sequence
        layer(mt.Tensor(np.zeros((0, 2, 3)), dtype="float64"))


@pytest.mark.parametrize("kind", ["LSTM", "GRU"])
def test_rejects_invalid_construction(kind):
    for kwargs in ({"input_size": 0}, {"hidden_size": 0}, {"num_layers": 0}):
        args = {"input_size": 3, "hidden_size": 4, **kwargs}
        with pytest.raises(Exception):
            getattr(nn, kind)(**args)


def test_gru_rejects_a_cell_state():
    layer = nn.GRU(3, 4, dtype="float64")
    x = mt.Tensor(np.zeros((5, 2, 3)), dtype="float64")
    state = mt.Tensor(np.zeros((1, 2, 4)), dtype="float64")
    with pytest.raises(Exception):
        layer.forward_with_state(x, state, state)


def _cell_run(kind, x, weights, hidden, reverse):
    """One direction of one layer over `x`, returned on the input's timeline."""
    w_ih, w_hh, b_ih, b_hh = weights
    order = range(x.shape[0] - 1, -1, -1) if reverse else range(x.shape[0])
    h = np.zeros((x.shape[1], hidden))
    c = np.zeros_like(h)
    outputs = []
    for t in order:
        if kind == "LSTM":
            gates = x[t] @ w_ih.T + b_ih + h @ w_hh.T + b_hh
            i, f, g, o = _blocks(gates, 4, hidden)
            i, f, g, o = _sigmoid(i), _sigmoid(f), np.tanh(g), _sigmoid(o)
            c = f * c + i * g
            h = o * np.tanh(c)
        else:
            from_input = x[t] @ w_ih.T + b_ih
            from_hidden = h @ w_hh.T + b_hh
            i_r, i_z, i_n = _blocks(from_input, 3, hidden)
            h_r, h_z, h_n = _blocks(from_hidden, 3, hidden)
            reset = _sigmoid(i_r + h_r)
            update = _sigmoid(i_z + h_z)
            h = (1 - update) * np.tanh(i_n + reset * h_n) + update * h
        outputs.append(h.copy())
    if reverse:
        outputs.reverse()
    return np.stack(outputs), h, c


def bidirectional_reference(kind, x, flat_params, num_layers, hidden):
    layer_input = x
    final_h, final_c = [], []
    for layer in range(num_layers):
        halves = []
        for direction in range(2):
            start = layer * 8 + direction * 4
            output, h, c = _cell_run(
                kind,
                layer_input,
                flat_params[start : start + 4],
                hidden,
                direction == 1,
            )
            halves.append(output)
            final_h.append(h)
            final_c.append(c)
        # Both directions are joined along the feature axis for the next layer.
        layer_input = np.concatenate(halves, axis=2)
    return layer_input, np.stack(final_h), np.stack(final_c)


@pytest.mark.parametrize("kind", ["LSTM", "GRU"])
@pytest.mark.parametrize("num_layers", [1, 2, 3])
def test_bidirectional_matches_reference(kind, num_layers):
    rng = np.random.default_rng(7)
    seq, batch, features, hidden = 5, 3, 4, 6
    layer = getattr(nn, kind)(
        features, hidden, num_layers=num_layers, bidirectional=True, dtype="float64"
    )
    flat = [p.numpy() for p in layer.parameters()]
    x = rng.standard_normal((seq, batch, features))

    output, state = layer.forward_with_state(mt.Tensor(x, dtype="float64"))
    h_n = state[0] if kind == "LSTM" else state
    want_out, want_h, want_c = bidirectional_reference(
        kind, x, flat, num_layers, hidden
    )

    np.testing.assert_allclose(output.numpy(), want_out, atol=1e-12)
    np.testing.assert_allclose(h_n.numpy(), want_h, atol=1e-12)
    if kind == "LSTM":
        np.testing.assert_allclose(state[1].numpy(), want_c, atol=1e-12)


def test_reverse_pass_is_realigned_onto_forward_time():
    """The classic bidirectional bug.

    The backward pass produces its states last-to-first. Concatenating them in
    that order pairs each timestep's forward state with the *wrong* backward
    state; only reversing them back onto the input's timeline is correct. The
    two differ by far more than rounding, so this test can tell them apart.
    """
    rng = np.random.default_rng(1)
    seq, batch, features, hidden = 6, 2, 3, 4
    layer = nn.GRU(features, hidden, bidirectional=True, dtype="float64")
    flat = [p.numpy() for p in layer.parameters()]
    x = rng.standard_normal((seq, batch, features))

    output, _ = layer.forward_with_state(mt.Tensor(x, dtype="float64"))
    reverse_half = output.numpy()[:, :, hidden:]

    as_produced, _, _ = _cell_run("GRU", x, flat[4:8], hidden, reverse=True)
    np.testing.assert_allclose(reverse_half, as_produced, atol=1e-12)
    # `_cell_run` already re-aligns; the un-aligned order must be clearly wrong.
    assert np.abs(reverse_half - as_produced[::-1]).max() > 1e-2


@pytest.mark.parametrize("kind", ["LSTM", "GRU"])
@pytest.mark.parametrize("num_layers", [1, 2, 3])
def test_bidirectional_shapes_and_parameter_widths(kind, num_layers):
    features, hidden = 3, 4
    layer = getattr(nn, kind)(
        features, hidden, num_layers=num_layers, bidirectional=True, dtype="float64"
    )
    assert layer.bidirectional is True
    assert layer.output_size == 2 * hidden

    params = layer.parameters()
    assert len(params) == 8 * num_layers, "four tensors per direction per layer"
    for index in range(num_layers):
        expected_input = features if index == 0 else 2 * hidden
        # Directions sit next to each other, so layer k starts at 8k.
        assert params[index * 8].shape[1] == expected_input
        assert params[index * 8 + 4].shape[1] == expected_input

    x = mt.Tensor(np.zeros((5, 2, features)), dtype="float64")
    output, state = layer.forward_with_state(x)
    h_n = state[0] if kind == "LSTM" else state
    assert output.shape == (5, 2, 2 * hidden)
    assert h_n.shape == (2 * num_layers, 2, hidden)


@pytest.mark.parametrize("kind", ["LSTM", "GRU"])
def test_bidirectional_gradients_reach_both_directions(kind):
    layer = getattr(nn, kind)(3, 4, num_layers=2, bidirectional=True, dtype="float64")
    x = mt.Tensor(np.random.default_rng(5).standard_normal((5, 2, 3)), dtype="float64")
    layer(x).sum().backward()
    for index, param in enumerate(layer.parameters()):
        assert param.grad is not None, f"parameter {index} got no gradient"
        assert (
            np.abs(param.grad.numpy()).max() > 0.0
        ), f"parameter {index} gradient is zero"
    mt.clear_autograd_graph()


def test_bidirectional_solves_a_task_needing_future_context():
    """Predict the sequence's last value at *every* timestep.

    A forward-only model cannot know that at t=0, so it can do no better than
    guessing the mean; the backward pass carries it there immediately. This is
    what bidirectionality buys, and a reverse pass wired up wrongly would not
    deliver it.
    """
    rng = np.random.default_rng(0)
    seq, batch, hidden = 6, 32, 16
    x = rng.standard_normal((seq, batch, 1)).astype(np.float32)
    y = np.repeat(x[-1][None, :, :], seq, axis=0)

    def train(bidirectional):
        mt.manual_seed(0)
        rnn = nn.GRU(1, hidden, bidirectional=bidirectional)
        width = hidden * (2 if bidirectional else 1)
        head = nn.DenseLayer(width, 1)
        optimizer = mt.optim.Adam(rnn.parameters() + head.parameters(), lr=0.02)
        inputs = mt.Tensor(x)
        targets = mt.Tensor(y.reshape(seq * batch, 1))
        loss = None
        for _ in range(400):
            output = rnn(inputs).reshape(seq * batch, width)
            error = head(output) - targets
            loss = (error * error).sum() / (seq * batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            mt.clear_autograd_graph()
        return loss.item()

    forward_only = train(False)
    both_ways = train(True)
    assert both_ways < 0.01, f"bidirectional failed to solve the task: {both_ways}"
    assert both_ways < forward_only / 10.0, (
        f"bidirectional ({both_ways}) barely beat forward-only ({forward_only}); "
        "the reverse pass may not be contributing"
    )


@pytest.mark.parametrize("kind", ["LSTM", "GRU"])
def test_bidirectional_state_must_cover_every_direction(kind):
    layer = getattr(nn, kind)(3, 4, bidirectional=True, dtype="float64")
    x = mt.Tensor(np.zeros((5, 2, 3)), dtype="float64")
    # One row per direction is required, not one per layer.
    too_few = mt.Tensor(np.zeros((1, 2, 4)), dtype="float64")
    with pytest.raises(Exception):
        layer.forward_with_state(x, too_few)

    right = mt.Tensor(np.zeros((2, 2, 4)), dtype="float64")
    if kind == "LSTM":
        layer.forward_with_state(x, right, right)
    else:
        layer.forward_with_state(x, right)


def test_unidirectional_is_still_the_default():
    layer = nn.GRU(3, 4)
    assert layer.bidirectional is False
    assert layer.output_size == 4
    assert len(layer.parameters()) == 4


def test_gru_carries_state_untouched_when_the_update_gate_saturates():
    """A saturated update gate must pass the hidden state through exactly.

    This is how a GRU holds information across a long sequence, so an error here
    is injected once per timestep and accumulates. It also pins the arithmetic
    against a plausible "optimisation": `(1 - z) * n + z * h` allocates a tensor
    of ones, and the algebraically equal `n + z * (h - n)` does not -- but at
    `z == 1.0` (reachable in f32 by a logit near 17) the second form misses `h`
    in about a third of cases. The extra allocation is what buys exactness.
    """
    hidden, batch = 4, 3
    layer = nn.GRU(2, hidden, dtype="float64")

    # Drive the update-gate block of b_ih hard positive so sigmoid saturates.
    bias = layer.parameters()[2]
    updated = bias.numpy().copy()
    updated[hidden : 2 * hidden] = 60.0
    bias[...] = mt.Tensor(updated, dtype="float64")

    h0 = np.random.default_rng(0).standard_normal((1, batch, hidden))
    x = np.random.default_rng(1).standard_normal((7, batch, 2))
    output, h_n = layer.forward_with_state(
        mt.Tensor(x, dtype="float64"), mt.Tensor(h0, dtype="float64")
    )

    # Bit-for-bit, not merely close.
    for t in range(7):
        assert np.array_equal(output.numpy()[t], h0[0]), f"state drifted by step {t}"
    assert np.array_equal(h_n.numpy()[0], h0[0])
