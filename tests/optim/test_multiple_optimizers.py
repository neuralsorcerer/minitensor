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

`step()` used to end with `autograd::clear_graph()`. Gradients live in that
graph rather than on the tensors, so the first `step()` discarded every
gradient including the ones belonging to the other optimizer; the second then
found nothing to apply and silently did nothing -- no error, no warning,
parameters simply never moving.

That wholesale clear was there to bound memory per iteration, back when
`backward()` marked the graph consumed without freeing it. `backward()` frees
the subgraph it walked now and keeps interior gradients for a single pass, so
`step()` releases only what it consumed: the gradients of its own parameters.
"""

import numpy as np
import pytest

import minitensor as mt
import minitensor.optim as optim


def test_single_optimizer_steps_correctly():
    """The ordinary case, to show the machinery itself is sound."""
    a = mt.Tensor(
        np.array([1.0, 2.0], dtype=np.float32), dtype="float32", requires_grad=True
    )
    (a * a).sum().backward()
    optim.SGD([a], lr=0.1).step()
    np.testing.assert_allclose(a.numpy(), [0.8, 1.6], rtol=1e-6)
    mt.clear_autograd_graph()


def test_gradients_do_not_survive_a_step():
    """A stepped parameter's gradient is consumed, as it always was."""
    a = mt.Tensor(
        np.array([1.0, 2.0], dtype=np.float32), dtype="float32", requires_grad=True
    )
    (a * a).sum().backward()
    assert a.grad is not None
    optim.SGD([a], lr=0.1).step()
    assert a.grad is None, "step() must still consume the gradients it applied"
    mt.clear_autograd_graph()


def test_two_optimizers_over_disjoint_parameters_both_step():
    a = mt.Tensor(
        np.array([1.0, 2.0], dtype=np.float32), dtype="float32", requires_grad=True
    )
    b = mt.Tensor(
        np.array([3.0, 4.0], dtype=np.float32), dtype="float32", requires_grad=True
    )

    ((a * a).sum() + (b * b).sum()).backward()
    first, second = optim.SGD([a], lr=0.1), optim.SGD([b], lr=0.1)
    first.step()
    second.step()

    np.testing.assert_allclose(a.numpy(), [0.8, 1.6], rtol=1e-6)
    # b moves too: `first.step()` no longer discards its gradient.
    np.testing.assert_allclose(b.numpy(), [2.4, 3.2], rtol=1e-6)
    mt.clear_autograd_graph()


def test_three_optimizers_apply_in_any_order():
    values = [
        mt.Tensor(
            np.array([2.0], dtype=np.float32), dtype="float32", requires_grad=True
        )
        for _ in range(3)
    ]
    total = values[0] * values[0]
    for value in values[1:]:
        total = total + value * value
    total.sum().backward()

    optimizers = [optim.SGD([value], lr=0.1) for value in values]
    for optimizer in reversed(optimizers):
        optimizer.step()

    for value in values:
        np.testing.assert_allclose(value.numpy(), [1.6], rtol=1e-6)
    mt.clear_autograd_graph()


def test_a_second_step_without_a_new_backward_is_a_no_op():
    # The gradient was consumed by the first step, so there is nothing left to
    # apply -- the parameter must not move twice off one backward pass.
    a = mt.Tensor(
        np.array([1.0, 2.0], dtype=np.float32), dtype="float32", requires_grad=True
    )
    (a * a).sum().backward()
    optimizer = optim.SGD([a], lr=0.1)

    optimizer.step()
    after_first = a.numpy().copy()
    optimizer.step()

    np.testing.assert_array_equal(a.numpy(), after_first)
    mt.clear_autograd_graph()
