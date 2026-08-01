# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""The autograd graph is per-thread, and its absence is not reported.

Nothing stated that the graph is thread-local. It matters in two opposite
directions. The good direction is isolation: `clear_autograd_graph()` is a
module-level function, so if the graph were shared, one thread calling it would
wipe a graph another thread was still building. It does not.

The bad direction is that crossing threads fails silently. A loss built in a
worker thread and backpropagated on the main thread produces no gradient and no
exception, and the tensor still reports `requires_grad=True`, so nothing marks
the mistake. That is reachable from ordinary code -- a data pipeline or a
`concurrent.futures` worker that builds the loss where it loaded the batch.

These tests pin the behaviour rather than change it: the same silence applies
within one thread after `clear_autograd_graph()`, so making the cross-thread
case raise would have to redefine what `backward()` on a released graph means.
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


def test_backward_on_another_threads_graph_is_a_silent_no_op():
    state = {}

    def build():
        weight = mt.Tensor(np.ones(3), dtype="float64", requires_grad=True)
        state["weight"] = weight
        state["loss"] = (weight * mt.Tensor(np.full(3, 5.0), dtype="float64")).sum()

    _run(build)

    # The forward value survives the hop; only the graph is out of reach.
    assert state["loss"].item() == pytest.approx(15.0)

    state["loss"].backward()  # must not raise

    assert state["weight"].grad is None
    assert mt.get_gradient(state["weight"]) is None
    # Nothing about the tensor advertises the problem.
    assert state["weight"].requires_grad is True


def test_the_same_silence_applies_after_clearing_in_one_thread():
    # Why the cross-thread case is documented rather than made to raise: an
    # absent graph behaves identically without any threads involved.
    weight = mt.Tensor(np.ones(3), dtype="float64", requires_grad=True)
    loss = (weight * mt.Tensor(np.full(3, 5.0), dtype="float64")).sum()
    mt.clear_autograd_graph()

    loss.backward()

    assert weight.grad is None


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
