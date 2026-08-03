# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""Stepping two optimizers after one backward pass.

Separate optimizers over disjoint parameter groups is a common arrangement --
a lower learning rate for a pretrained encoder than for a fresh head,
discriminative fine-tuning, separate optimizers for a generator and a
discriminator sharing one loss.

`PyOptimizer::step` currently ends with `autograd::clear_graph()`, and gradients
live in that graph rather than on the tensors, so the first `step()` discards
every gradient including the ones belonging to the other optimizer. The second
`step()` then finds nothing to apply and silently does nothing: no error, no
warning, parameters simply never move.

The `clear_graph()` call is not gratuitous -- it is what bounds memory per
iteration, since `backward()` marks the graph consumed but does not free it. So
the fix is a choice about where that responsibility belongs (most likely
`zero_grad`, which is where the rest of the ecosystem puts it) rather than a
line to delete, and it is left for a maintainer to make.
"""

import numpy as np
import pytest

import minitensor as mt
import minitensor.optim as optim


def test_single_optimizer_steps_correctly():
    """The ordinary case, to show the machinery itself is sound."""
    a = mt.Tensor(np.array([1.0, 2.0], dtype=np.float32), dtype="float32", requires_grad=True)
    (a * a).sum().backward()
    optim.SGD([a], lr=0.1).step()
    np.testing.assert_allclose(a.numpy(), [0.8, 1.6], rtol=1e-6)
    mt.clear_autograd_graph()


def test_gradients_do_not_survive_a_step():
    """Documents the mechanism: `step()` frees the graph, and grads live there."""
    a = mt.Tensor(np.array([1.0, 2.0], dtype=np.float32), dtype="float32", requires_grad=True)
    (a * a).sum().backward()
    assert a.grad is not None
    optim.SGD([a], lr=0.1).step()
    assert a.grad is None, "step() no longer frees the graph; update this test's sibling"
    mt.clear_autograd_graph()


@pytest.mark.xfail(
    reason="step() clears the whole autograd graph, so a second optimizer over a "
    "different parameter group finds no gradients and silently does nothing",
    strict=True,
)
def test_two_optimizers_over_disjoint_parameters_both_step():
    a = mt.Tensor(np.array([1.0, 2.0], dtype=np.float32), dtype="float32", requires_grad=True)
    b = mt.Tensor(np.array([3.0, 4.0], dtype=np.float32), dtype="float32", requires_grad=True)

    ((a * a).sum() + (b * b).sum()).backward()
    first, second = optim.SGD([a], lr=0.1), optim.SGD([b], lr=0.1)
    first.step()
    second.step()

    np.testing.assert_allclose(a.numpy(), [0.8, 1.6], rtol=1e-6)
    # b is still [3, 4]: its gradient was discarded by `first.step()`.
    np.testing.assert_allclose(b.numpy(), [2.4, 3.2], rtol=1e-6)
    mt.clear_autograd_graph()
