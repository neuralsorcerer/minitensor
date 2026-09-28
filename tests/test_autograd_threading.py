# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""The autograd graph is per-thread, and a backward pass that cannot reach one
says so.

Nothing stated that the graph is thread-local. It matters in two opposite
directions. The good direction is isolation: `clear_autograd_graph()` is a
module-level function, so if the graph were shared, one thread calling it would
wipe a graph another thread was still building. It does not.

The other direction is that a loss built in one thread cannot be
backpropagated from another -- a data pipeline or a `concurrent.futures`
worker that builds the loss where it loaded the batch. That used to fail in
silence: no gradient, no exception, and a tensor still reporting
`requires_grad=True`, so an optimizer skipped the parameter as if the loss had
not used it. The same silence followed `clear_autograd_graph()` between a
forward and its backward. Both now raise, since a result owns its history and
can tell "recorded, but not here" from "never recorded".
"""

import threading

import numpy as np
import pytest

import minitensor as mt


def _run(target):
    thread = threading.Thread(target=target)
    thread.start()
    thread.join(10)
    assert not thread.is_alive(), "worker thread did not finish"


def test_independent_threads_train_without_cross_talk():
    # Each thread's gradient is analytically known and distinct, so any shared
    # state shows up as a wrong value rather than as a crash.
    problems = []

    def train(tid, steps=100):
        try:
            weight = mt.Tensor(np.zeros(3), dtype="float64", requires_grad=True)
            target = float(tid + 1)
            for _ in range(steps):
                grads = mt.Tensor(np.full(3, target), dtype="float64")
                (weight * grads).sum().backward()
                seen = weight.grad.numpy()
                if not np.allclose(seen, target):
                    problems.append(f"thread {tid}: saw {seen} want {target}")
                    return
                mt.clear_autograd_graph()
        except Exception as exc:  # pragma: no cover - failure detail only
            problems.append(f"thread {tid}: {type(exc).__name__}: {exc}")

    threads = [threading.Thread(target=train, args=(i,)) for i in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(30)
    assert not problems, problems


def test_a_foreign_clear_does_not_disturb_a_live_graph():
    # The isolation guarantee, forced rather than hoped for: the clear happens
    # while the other thread's graph is built but not yet backpropagated.
    built, cleared = threading.Event(), threading.Event()
    result = {}

    def builder():
        weight = mt.Tensor(np.ones(4), dtype="float64", requires_grad=True)
        loss = (weight * mt.Tensor(np.full(4, 3.0), dtype="float64")).sum()
        built.set()
        cleared.wait(10)
        loss.backward()
        result["grad"] = weight.grad.numpy().copy()

    def clearer():
        built.wait(10)
        mt.clear_autograd_graph()
        cleared.set()

    threads = [threading.Thread(target=builder), threading.Thread(target=clearer)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(30)

    np.testing.assert_allclose(result["grad"], np.full(4, 3.0))


def test_backward_on_another_threads_graph_raises():
    state = {}

    def build():
        weight = mt.Tensor(np.ones(3), dtype="float64", requires_grad=True)
        state["weight"] = weight
        state["loss"] = (weight * mt.Tensor(np.full(3, 5.0), dtype="float64")).sum()

    _run(build)

    # The forward value survives the hop; only the graph is out of reach.
    assert state["loss"].item() == pytest.approx(15.0)

    with pytest.raises(RuntimeError, match="not on this thread"):
        state["loss"].backward()
    assert state["weight"].grad is None


def test_backward_after_clearing_the_graph_raises():
    weight = mt.Tensor(np.ones(3), dtype="float64", requires_grad=True)
    loss = (weight * mt.Tensor(np.full(3, 5.0), dtype="float64")).sum()
    mt.clear_autograd_graph()

    with pytest.raises(RuntimeError, match="clear_autograd_graph"):
        loss.backward()
    assert weight.grad is None


def test_backward_through_a_released_graph_raises_after_other_recording():
    # The consumed flag caught a second backward only while nothing had been
    # recorded since; any later operation reset it and the second pass walked
    # nothing in silence.
    weight = mt.Tensor(np.ones(3), dtype="float64", requires_grad=True)
    loss = (weight * 2.0).sum()
    loss.backward()
    (weight * 3.0).sum()  # records, and resets the consumed flag

    with pytest.raises(RuntimeError, match="earlier backward"):
        loss.backward()
    np.testing.assert_allclose(weight.grad.numpy(), np.full(3, 2.0))


def test_grad_mode_and_the_consumed_flag_are_thread_local_too():
    weight = mt.Tensor(np.ones(2), dtype="float64", requires_grad=True)
    (weight * 2.0).sum().backward()
    mt.mark_autograd_graph_consumed()
    assert mt.is_autograd_graph_consumed() is True

    seen = {}

    def look():
        seen["consumed"] = mt.is_autograd_graph_consumed()
        seen["grad_enabled"] = mt.is_grad_enabled()

    _run(look)
    assert seen["consumed"] is False
    assert seen["grad_enabled"] is True

    mt.clear_autograd_graph()
