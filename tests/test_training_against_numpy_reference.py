# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""A whole training run, checked step by step against hand-derived gradients.

The unit tests elsewhere check operations one at a time. This checks that they
compose: a two-layer MLP is trained through minitensor and again in plain NumPy
with the backward pass written out by hand, and every loss and every final
parameter is compared. A defect anywhere in the forward ops, the autograd graph,
the gradient kernels, or the optimizer shows up as divergence rather than as
"the loss went down, so it is probably fine" -- which is what a convergence-only
test actually asserts.

It runs in float64 and compares to ~1e-16, so it is a correctness test rather
than a tolerance-tuning exercise. The path it covers is deliberately wide:
matmul, broadcast addition, tanh and its gradient, log_softmax, elementwise
multiply, sum, and an optimizer step with `zero_grad`.
"""

import numpy as np
import pytest

import minitensor as mt

SAMPLES, FEATURES, HIDDEN, CLASSES = 64, 8, 16, 3
LEARNING_RATE, STEPS = 0.1, 40


@pytest.fixture(scope="module")
def problem():
    rng = np.random.default_rng(0)
    inputs = rng.standard_normal((SAMPLES, FEATURES))
    labels = rng.integers(0, CLASSES, SAMPLES)
    targets = np.eye(CLASSES)[labels]
    weights = (
        rng.standard_normal((FEATURES, HIDDEN)) * 0.3,
        np.zeros(HIDDEN),
        rng.standard_normal((HIDDEN, CLASSES)) * 0.3,
        np.zeros(CLASSES),
    )
    return inputs, targets, weights


def _numpy_training(inputs, targets, weights):
    """Cross-entropy over a tanh MLP, with the backward pass written out."""
    w1, b1, w2, b2 = (w.copy() for w in weights)
    losses = []
    for _ in range(STEPS):
        hidden = np.tanh(inputs @ w1 + b1)
        logits = hidden @ w2 + b2
        shifted = logits - logits.max(1, keepdims=True)
        log_probs = shifted - np.log(np.exp(shifted).sum(1, keepdims=True))
        losses.append(-(targets * log_probs).sum() / SAMPLES)

        d_logits = (np.exp(log_probs) - targets) / SAMPLES
        d_w2 = hidden.T @ d_logits
        d_b2 = d_logits.sum(0)
        d_hidden = (d_logits @ w2.T) * (1 - hidden**2)
        d_w1 = inputs.T @ d_hidden
        d_b1 = d_hidden.sum(0)

        w1 -= LEARNING_RATE * d_w1
        b1 -= LEARNING_RATE * d_b1
        w2 -= LEARNING_RATE * d_w2
        b2 -= LEARNING_RATE * d_b2
    return np.array(losses), (w1, b1, w2, b2)


def _minitensor_training(inputs, targets, weights):
    parameters = [
        mt.Tensor(w.copy(), dtype="float64", requires_grad=True) for w in weights
    ]
    w1, b1, w2, b2 = parameters
    x = mt.Tensor(inputs, dtype="float64")
    y = mt.Tensor(targets, dtype="float64")
    optimizer = mt.optim.SGD(parameters, lr=LEARNING_RATE)

    losses = []
    for _ in range(STEPS):
        optimizer.zero_grad()
        hidden = (x.matmul(w1) + b1).tanh()
        log_probs = (hidden.matmul(w2) + b2).log_softmax(dim=1)
        loss = (y * log_probs).sum() * (-1.0 / SAMPLES)
        losses.append(loss.item())
        loss.backward()
        optimizer.step()

    final = tuple(p.numpy().copy() for p in parameters)
    mt.clear_autograd_graph()
    return np.array(losses), final


def test_every_step_matches_a_hand_derived_reference(problem):
    inputs, targets, weights = problem
    reference_losses, reference_params = _numpy_training(inputs, targets, weights)
    losses, params = _minitensor_training(inputs, targets, weights)

    # Not just the final loss: a gradient that is wrong in one term can still
    # reach a similar place after 40 steps.
    np.testing.assert_allclose(losses, reference_losses, rtol=0, atol=1e-12)
    for name, got, want in zip(("w1", "b1", "w2", "b2"), params, reference_params):
        np.testing.assert_allclose(got, want, rtol=0, atol=1e-12, err_msg=name)


def test_the_reference_problem_actually_trains(problem):
    # Guards the test above from passing vacuously: if both implementations sat
    # still, they would agree perfectly and prove nothing.
    inputs, targets, weights = problem
    losses, _ = _minitensor_training(inputs, targets, weights)
    assert losses[-1] < losses[0] * 0.9, (losses[0], losses[-1])
    assert np.all(np.isfinite(losses))


@pytest.mark.parametrize("optimizer_name", ["Adam", "AdamW", "RMSprop", "Lion", "NAdam"])
def test_the_same_model_trains_under_every_optimizer(problem, optimizer_name):
    # Wider coverage of the optimizer surface against the same graph, checking
    # that each one reduces the loss and none produces a non-finite parameter.
    inputs, targets, weights = problem
    parameters = [
        mt.Tensor(w.copy(), dtype="float64", requires_grad=True) for w in weights
    ]
    w1, b1, w2, b2 = parameters
    x = mt.Tensor(inputs, dtype="float64")
    y = mt.Tensor(targets, dtype="float64")
    optimizer = getattr(mt.optim, optimizer_name)(parameters, lr=0.01)

    first = last = None
    for step in range(STEPS):
        optimizer.zero_grad()
        hidden = (x.matmul(w1) + b1).tanh()
        log_probs = (hidden.matmul(w2) + b2).log_softmax(dim=1)
        loss = (y * log_probs).sum() * (-1.0 / SAMPLES)
        last = loss.item()
        if step == 0:
            first = last
        loss.backward()
        optimizer.step()

    assert np.isfinite(last) and last < first, (optimizer_name, first, last)
    for parameter in parameters:
        assert np.all(np.isfinite(parameter.numpy())), optimizer_name
    mt.clear_autograd_graph()
