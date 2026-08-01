# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""What `backward()` keeps, and what clears it.

Unlike PyTorch, MiniTensor exposes `.grad` on interior (non-leaf) tensors after
a backward pass. That is a deliberate feature, and it is why the gradient map
retains an entry per interior tensor rather than dropping it. The consequence
is that the map only stays bounded if something resets it between iterations --
`optimizer.step()` and `clear_autograd_graph()` both do.

A loop that calls `backward()` and neither of those therefore grows without
bound, at roughly the size of one intermediate per iteration (measured: ~65 KB
per iteration for a 256x32 intermediate, ~141 KB for 256x128). These tests pin
the mechanism deterministically rather than by watching RSS, which is noisy.
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


def test_repeated_backward_without_clearing_retains_every_interior_gradient():
    # This is the shape that grows: each iteration makes fresh interior tensors
    # whose gradients are kept until something resets the graph. A loop that
    # backwards without stepping an optimizer must call
    # `clear_autograd_graph()` itself.
    interiors = []
    for _ in range(8):
        _, interior, loss = _interior_and_loss()
        loss.backward()
        interiors.append(interior)

    assert all(mt.get_gradient(t) is not None for t in interiors)

    mt.clear_autograd_graph()

    assert all(mt.get_gradient(t) is None for t in interiors)


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
