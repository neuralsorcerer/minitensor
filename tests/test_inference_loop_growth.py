# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""An inference loop grows unless it runs inside `no_grad()`.

Every forward records its intermediates, and nothing releases them until a
`backward()` walks the graph or `clear_autograd_graph()` empties it. An
inference loop does neither, so it grows for as long as it runs. Measured over
300 forwards of a two-layer `Sequential` with a 256-row batch, discarding every
output: ~42 KB per forward at width 32, ~122 KB at 128, ~496 KB at 512 -- about
145 MB over those 300 calls at the widest.

The two things that look like they should help do not, which is what makes this
worth pinning rather than merely documenting:

- **Discarding the output does not release anything.** The recording lives in a
  graph the module owns, not in the returned tensor, so a loop that keeps
  nothing still accumulates. This is where it departs from PyTorch, where the
  output tensor owns its history and dropping it frees the graph.
- **`model.eval()` does not either.** It switches Dropout off and freezes
  BatchNorm's running statistics -- what the layers compute, not whether the
  computation is recorded. PyTorch draws the same line.

`no_grad()` removes it completely, and a training loop needs no guard at all
since `backward()` releases the subgraph it walked.

These tests count graph entries rather than resident memory: the entry count is
the mechanism, and it is exact, where RSS depends on the allocator.
"""

from __future__ import annotations

import numpy as np
import pytest

import minitensor as mt

nn = mt.nn


def _model():
    mt.manual_seed(0)
    return nn.Sequential([nn.DenseLayer(16, 16), nn.ReLU(), nn.DenseLayer(16, 8)])


def _input():
    return mt.Tensor(np.ones((4, 16), np.float32))


def _entries():
    return mt.autograd_graph_size()[0]


def _run(model, x, rounds, guard=None):
    """Forward `rounds` times, discarding every output, and report the growth."""
    mt.clear_autograd_graph()
    start = _entries()
    if guard is None:
        for _ in range(rounds):
            model(x)
    else:
        with guard():
            for _ in range(rounds):
                model(x)
    return _entries() - start


# --- the growth ------------------------------------------------------------


def test_a_plain_inference_loop_grows():
    assert _run(_model(), _input(), 20) > 0


def test_it_grows_in_proportion_to_the_iterations():
    """Not a one-off cost that settles: twice the calls, twice the entries."""
    model, x = _model(), _input()
    ten = _run(model, x, 10)
    forty = _run(model, x, 40)
    assert forty == pytest.approx(4 * ten, rel=0.2), (ten, forty)


def test_discarding_the_output_does_not_release_it():
    """`_run` never keeps a result, and the graph grows anyway. This is the
    part that differs from PyTorch, where the output owns its history."""
    model, x = _model(), _input()
    mt.clear_autograd_graph()
    for _ in range(20):
        model(x)  # nothing bound, nothing kept
    assert _entries() > 0


def test_eval_mode_does_not_stop_the_recording():
    """`eval()` is about what the layers compute, not whether it is recorded."""
    model, x = _model(), _input()
    training = _run(model, x, 20)

    model.eval()
    evaluating = _run(model, x, 20)

    assert evaluating == training


# --- the remedy -------------------------------------------------------------


def test_no_grad_removes_the_growth_entirely():
    assert _run(_model(), _input(), 50, guard=mt.no_grad) == 0


@pytest.mark.parametrize("rounds", [1, 10, 100])
def test_no_grad_is_flat_at_every_length(rounds):
    assert _run(_model(), _input(), rounds, guard=mt.no_grad) == 0


def test_no_grad_still_produces_the_same_numbers():
    """The guard must change only what is recorded."""
    model, x = _model(), _input()
    recorded = model(x).numpy()
    with mt.no_grad():
        guarded = model(x).numpy()
    np.testing.assert_array_equal(guarded, recorded)


def test_clearing_the_graph_is_the_other_way_out():
    model, x = _model(), _input()
    mt.clear_autograd_graph()
    for _ in range(20):
        model(x)
        mt.clear_autograd_graph()
    assert _entries() == 0


# --- what does not need a guard ---------------------------------------------


def test_a_training_loop_stays_flat_on_its_own():
    """`backward()` releases the subgraph it walked, so forward-then-backward
    needs neither guard."""
    model, x = _model(), _input()
    mt.clear_autograd_graph()

    for _ in range(5):
        model(x).sum().backward()
    after_five = _entries()

    for _ in range(45):
        model(x).sum().backward()
    after_fifty = _entries()

    assert after_fifty == after_five


def test_gradients_still_flow_after_a_no_grad_block():
    """The guard is scoped, so training after an evaluation still records."""
    model, x = _model(), _input()
    with mt.no_grad():
        model(x)

    model(x).sum().backward()
    assert any(p.grad is not None for p in model.parameters())
