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
