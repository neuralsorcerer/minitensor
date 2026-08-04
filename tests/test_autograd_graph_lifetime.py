# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""What `backward()` keeps, and what clears it.

Unlike PyTorch, MiniTensor exposes `.grad` on interior (non-leaf) tensors after
a backward pass. That is a deliberate feature, and it is why the gradient map
retains an entry per interior tensor rather than dropping it with the node.

It retains them for one pass. Interior tensors get a fresh id on every forward,
so nothing ever overwrote an old entry; keeping them all meant a loop that
called `backward()` without `optimizer.step()` or `clear_autograd_graph()` grew
without bound, one intermediate per iteration. That is precisely what gradient
accumulation does -- several backwards, then one step -- and it was costing
both memory and, as the map grew, time. So each backward now releases the
interior gradients the previous one left behind.

What that preserves: reading `.grad` on a non-leaf after the backward that
produced it. What it gives up: reading one several passes later. Leaf
gradients are untouched and still accumulate across passes until `zero_grad`,
which is what makes accumulation work at all.

These tests pin the mechanism deterministically rather than by watching RSS,
which is noisy.
"""

from __future__ import annotations

import gc

import numpy as np
import pytest

import minitensor as mt


@pytest.fixture(autouse=True)
def _clean_graph():
    mt.clear_autograd_graph()
    yield
    mt.clear_autograd_graph()


def _interior_and_loss():
    x = mt.as_tensor(np.random.randn(4, 4).astype(np.float32))
    weight = mt.Tensor(np.random.randn(4, 4).astype(np.float32)).requires_grad_(True)
    interior = mt.matmul(x, weight)
    return weight, interior, mt.sum(mt.tanh(interior))


def test_interior_gradients_are_available_after_backward():
    # The feature the retention exists for.
    _, interior, loss = _interior_and_loss()
    loss.backward()

    assert interior.grad is not None
    assert mt.get_gradient(interior) is not None


def test_clear_autograd_graph_releases_interior_gradients():
    _, interior, loss = _interior_and_loss()
    loss.backward()
    assert mt.get_gradient(interior) is not None

    mt.clear_autograd_graph()

    assert mt.get_gradient(interior) is None


def test_optimizer_step_releases_interior_gradients():
    # A normal training loop is bounded because `step()` clears the graph.
    weight, interior, loss = _interior_and_loss()
    optimizer = mt.optim.SGD([weight], 1e-3)
    loss.backward()
    assert mt.get_gradient(interior) is not None

    optimizer.step()

    assert mt.get_gradient(interior) is None


def test_repeated_backward_keeps_only_the_latest_interior_gradients():
    # The shape that used to grow without bound: fresh interior tensors every
    # iteration, and no optimizer step to reset the graph. Each backward now
    # releases the previous pass's interior gradients, so the map stays the
    # size of one iteration however long the loop runs.
    interiors = []
    for _ in range(8):
        _, interior, loss = _interior_and_loss()
        loss.backward()
        interiors.append(interior)

    assert mt.get_gradient(interiors[-1]) is not None, (
        "the pass that just ran must still expose its interior gradients"
    )
    assert all(mt.get_gradient(t) is None for t in interiors[:-1]), (
        "earlier passes' interior gradients must have been released"
    )

    mt.clear_autograd_graph()

    assert all(mt.get_gradient(t) is None for t in interiors)


def test_accumulating_gradients_over_many_backwards_does_not_grow_memory():
    # Gradient accumulation: backward several times, step once. The leaf
    # gradient has to keep adding up while nothing else piles up behind it.
    weight = mt.Tensor(np.ones((64, 64), dtype=np.float32)).requires_grad_(True)
    x = mt.Tensor(np.ones((32, 64), dtype=np.float32))

    for micro_batch in range(1, 17):
        mt.sum(mt.matmul(x, weight)).backward()
        grad = mt.get_gradient(weight)
        assert grad is not None
        # Every micro-batch contributes the same amount, so the running total
        # is exactly `micro_batch` times one pass's worth.
        np.testing.assert_allclose(grad.numpy(), np.full((64, 64), 32.0 * micro_batch))

    optimizer = mt.optim.SGD([weight], 1e-3)
    optimizer.step()
    assert mt.get_gradient(weight) is None


def test_forward_without_backward_retains_nothing():
    x = mt.as_tensor(np.random.randn(4, 4).astype(np.float32))
    weight = mt.Tensor(np.random.randn(4, 4).astype(np.float32)).requires_grad_(True)

    interiors = [mt.matmul(x, weight) for _ in range(8)]
    gc.collect()

    # No backward ran, so no gradient was ever stored for these.
    assert all(mt.get_gradient(t) is None for t in interiors)


def test_no_grad_forward_retains_nothing():
    x = mt.as_tensor(np.random.randn(4, 4).astype(np.float32))
    weight = mt.Tensor(np.random.randn(4, 4).astype(np.float32)).requires_grad_(True)

    with mt.no_grad():
        interior = mt.matmul(x, weight)

    assert mt.get_gradient(interior) is None
    assert not interior.requires_grad


# --- losses with hand-written backwards --------------------------------------
#
# Each of these composes its forward from ordinary tensor ops and then replaces
# the resulting `grad_fn` with a single analytical node. If the composed ops
# record their own nodes, those nodes are unreachable from the loss the moment
# the grad_fn is replaced: no backward pass ever walks them, so nothing ever
# releases them or the activations they saved. The forward therefore has to run
# with recording off. `binary_cross_entropy` and `focal_loss` were the worst,
# stranding twelve nodes per call.
#
# `log_cosh_loss` is absent on purpose: it has no analytical backward, so its
# whole graph is reachable from the loss and its own backward releases it. It
# does still grow, by a different mechanism -- it builds scalar constants per
# call, and a fresh constant becomes a leaf node that no release removes. That
# affects any op with a scalar operand (`x * 2.0` and friends), not losses, and
# costs a small map entry rather than a retained activation.


def _loss_cases():
    p = mt.Tensor(np.random.rand(8, 4).astype(np.float32) * 0.8 + 0.1).requires_grad_(True)
    t = mt.Tensor(np.random.rand(8, 4).astype(np.float32) * 0.8 + 0.1)
    onehot = mt.Tensor(np.eye(4, dtype=np.float32)[np.random.randint(0, 4, 8)])
    idx = mt.Tensor(np.random.randint(0, 4, (8,)).astype(np.int64), dtype="int64")
    logits = mt.Tensor(np.random.randn(8, 4).astype(np.float32)).requires_grad_(True)
    probs = mt.softmax(logits, -1)
    return [
        ("mse_loss", lambda: mt.nn.mse_loss(p, t)),
        ("l1_loss", lambda: mt.nn.l1_loss(p, t)),
        ("huber_loss", lambda: mt.nn.huber_loss(p, t)),
        ("smooth_l1_loss", lambda: mt.nn.smooth_l1_loss(p, t)),
        ("cross_entropy", lambda: mt.nn.cross_entropy(logits, idx)),
        ("binary_cross_entropy", lambda: mt.nn.binary_cross_entropy(p, t)),
        ("bce_with_logits", lambda: mt.nn.binary_cross_entropy_with_logits(p, t)),
        ("focal_loss", lambda: mt.nn.focal_loss(p, onehot)),
        ("kl_div", lambda: mt.nn.kl_div(probs, probs)),
    ]


@pytest.mark.parametrize("name,build", _loss_cases(), ids=lambda v: v if isinstance(v, str) else "")
def test_loss_backward_leaves_nothing_stranded_in_the_graph(name, build):
    np.random.seed(0)
    mt.clear_autograd_graph()

    sizes = []
    for _ in range(6):
        build().backward()
        sizes.append(mt.autograd_graph_size()[0])

    # From the second pass on the node count must not move: everything a pass
    # records is reachable from its loss, so its own backward releases it.
    assert sizes[1:] == sizes[1:2] * 5, f"{name} strands nodes: {sizes}"


def test_a_loss_still_produces_gradients_after_the_forward_stops_recording():
    # The forward runs with autograd off, so `requires_grad` no longer
    # propagates through it on its own and has to be set on the loss
    # explicitly. If that were missed the graph would be clean but empty.
    p = mt.Tensor(np.full((4, 3), 0.6, dtype=np.float64), dtype="float64").requires_grad_(True)
    t = mt.Tensor(np.full((4, 3), 0.25, dtype=np.float64), dtype="float64")

    loss = mt.nn.mse_loss(p, t)
    assert loss.requires_grad
    loss.backward()

    grad = mt.get_gradient(p)
    assert grad is not None
    np.testing.assert_allclose(grad.numpy(), np.full((4, 3), 2 * (0.6 - 0.25) / 12))


def test_a_loss_under_no_grad_stays_free_of_the_graph():
    p = mt.Tensor(np.full((4, 3), 0.6, dtype=np.float32)).requires_grad_(True)
    t = mt.Tensor(np.full((4, 3), 0.25, dtype=np.float32))

    mt.clear_autograd_graph()
    with mt.no_grad():
        loss = mt.nn.mse_loss(p, t)

    assert not loss.requires_grad
    assert mt.autograd_graph_size() == (0, 0)
