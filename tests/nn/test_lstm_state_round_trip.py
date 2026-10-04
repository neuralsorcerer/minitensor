# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""An LSTM takes back the state it returns.

`forward_with_state` returns `(output, (h_n, c_n))` but took the initial state
as separate `hx` and `cx`, so feeding one chunk's final state into the next --
what the method is for -- was a TypeError unless the pair was unpacked by
hand. The pair is now accepted as `hx`.
"""

import numpy as np
import pytest

import minitensor as mt
import minitensor.nn as nn

X = mt.randn(6, 2, 4)


@pytest.mark.parametrize("num_layers", [1, 2])
def test_chunks_with_carried_state_match_one_pass(num_layers):
    lstm = nn.LSTM(4, 3, num_layers=num_layers)
    whole, _ = lstm.forward_with_state(X)
    first, state = lstm.forward_with_state(X[:2])
    second, _ = lstm.forward_with_state(X[2:], state)
    joined = np.concatenate([first.numpy(), second.numpy()])
    assert np.allclose(joined, whole.numpy(), atol=1e-6)


def test_the_pair_and_separate_arguments_agree():
    lstm = nn.LSTM(4, 3)
    _, (h, c) = lstm.forward_with_state(X[:3])
    paired, _ = lstm.forward_with_state(X[3:], (h, c))
    separate, _ = lstm.forward_with_state(X[3:], h, c)
    assert np.array_equal(paired.numpy(), separate.numpy())


def test_gradients_reach_a_paired_initial_state():
    h0 = mt.zeros(1, 2, 3, requires_grad=True)
    c0 = mt.zeros(1, 2, 3, requires_grad=True)
    output, _ = nn.LSTM(4, 3).forward_with_state(X, (h0, c0))
    output.sum().backward()
    assert h0.grad is not None and c0.grad is not None


def test_a_pair_and_cx_together_are_refused():
    lstm = nn.LSTM(4, 3)
    _, (h, c) = lstm.forward_with_state(X)
    with pytest.raises(ValueError, match="pass one or the other"):
        lstm.forward_with_state(X, (h, c), c)


def test_a_gru_state_is_a_single_tensor():
    gru = nn.GRU(4, 3)
    _, h = gru.forward_with_state(X)
    with pytest.raises(TypeError):
        gru.forward_with_state(X, (h, h))
    first, state = gru.forward_with_state(X[:2])
    second, _ = gru.forward_with_state(X[2:], state)
    whole, _ = gru.forward_with_state(X)
    joined = np.concatenate([first.numpy(), second.numpy()])
    assert np.allclose(joined, whole.numpy(), atol=1e-6)
