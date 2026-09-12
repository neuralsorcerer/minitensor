# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""The tape holds pointers to tensors, not copies of them.

A graph node keeps whatever its backward will need -- the operands, sometimes
the output -- and keeps them by sharing the buffer, which costs a refcount. A
node that copied instead would make the memory a forward pass needs grow with
the depth of the graph rather than with the values in it, and would pay the
copy again on the way back.

That is the design, and it is already what the engine does everywhere except
one place, where a pass-through gradient deep-copied what it was handed. These
tests hold the property from outside: the cost of a chain of copies has to grow
linearly and gently rather than compounding, and sharing must not let one
branch's accumulation disturb another's gradient.

Sharing a gradient buffer is safe for one reason, and it is worth naming
because it is what these tests are really guarding: the only thing that writes
into a gradient already in the map is an in-place add, and that refuses to
write into a buffer with more than one owner -- it produces a new tensor
instead.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

import minitensor as mt


@pytest.fixture(autouse=True)
def _clean_graph():
    mt.clear_autograd_graph()
    yield
    mt.clear_autograd_graph()


def _t(values, dtype="float64", requires_grad=False):
    return mt.Tensor(
        np.ascontiguousarray(np.asarray(values, dtype=dtype)),
        dtype=dtype,
        requires_grad=requires_grad,
    )


# --- what sharing must not break ---------------------------------------------


def test_a_copy_and_its_source_get_their_own_gradients():
    """Two paths to one leaf, and the copy's own gradient in between.

    The leaf's gradient is the sum of both paths; the copy's is only its own.
    A pass-through that handed the leaf the same buffer the copy's gradient
    lives in, and then accumulated into it, would report 4 for both.
    """

    leaf = _t([1.0, 1.0, 1.0, 1.0], requires_grad=True)
    copy = leaf.clone()
    (copy.sum() * 3.0 + leaf.sum()).backward()

    np.testing.assert_array_equal(leaf.grad.numpy(), [4.0] * 4)
    np.testing.assert_array_equal(copy.grad.numpy(), [3.0] * 4)


def test_several_copies_of_one_leaf_accumulate_to_their_total():
    leaf = _t([1.0, 2.0, 3.0], requires_grad=True)
    (leaf.clone().sum() + leaf.clone().sum() * 2.0 + leaf.sum()).backward()
    np.testing.assert_array_equal(leaf.grad.numpy(), [4.0, 4.0, 4.0])


def test_a_chain_of_copies_still_carries_the_whole_gradient():
    leaf = _t([1.0, 2.0], requires_grad=True)
    chained = leaf
    for _ in range(6):
        chained = chained.clone()
    (chained * _t([10.0, 100.0])).sum().backward()
    np.testing.assert_array_equal(leaf.grad.numpy(), [10.0, 100.0])


def test_an_optimizer_step_through_a_copy_moves_only_the_parameter():
    parameter = _t([3.0], dtype="float32", requires_grad=True)
    optimizer = mt.optim.SGD([parameter], lr=0.1)
    for _ in range(20):
        optimizer.zero_grad()
        (parameter.clone() * parameter.clone()).sum().backward()
        optimizer.step()
    assert abs(float(parameter.numpy()[0])) < 0.1, parameter.numpy()


# --- what sharing is for -----------------------------------------------------


def _chain_cost(depth, side, with_backward):
    """Best of five: one pass through `depth` copies of a `side x side` tensor."""

    values = np.ascontiguousarray(
        np.random.default_rng(0).standard_normal((side, side)), dtype="float32"
    )
    leaf = mt.from_numpy(values).requires_grad_(with_backward)

    def once():
        chained = leaf
        for _ in range(depth):
            chained = chained.clone()
        if with_backward:
            chained.sum().backward()
            mt.clear_autograd_graph()
        else:
            chained.sum()

    once()  # warm the allocator
    best = float("inf")
    for _ in range(5):
        start = time.perf_counter()
        once()
        best = min(best, time.perf_counter() - start)
    return best


def _cost_per_level(side, with_backward, shallow=1, deep=9):
    span = _chain_cost(deep, side, with_backward) - _chain_cost(
        shallow, side, with_backward
    )
    return span / (deep - shallow)


@pytest.mark.parametrize("side", [900, 1500])
def test_a_backward_through_copies_costs_no_more_than_the_copies_did(side):
    """The sharpest statement of the property, and the cheapest to measure.

    Each level of the chain copies the tensor going forward -- that is what a
    copy is, and it is the baseline here rather than the thing under test. The
    gradient coming back is the same size, so a backward that copied it too
    would roughly double the marginal cost of a level.

    Measured, the two marginal costs are within a few percent of each other:
    0.30ms against 0.33ms per level at 900, 1.06ms against 1.00ms at 1500. The
    same measurement against a backward that deep-copies is 10x, not 2x -- the
    copy is an allocation as well as a memcpy, and per node it costs several
    times what copying the buffer alone costs. So the threshold sits at 3x:
    far above the noise in a ratio of two timings on a shared machine, and far
    below anything a per-node copy could come in at.
    """

    forward = _cost_per_level(side, with_backward=False)
    both = _cost_per_level(side, with_backward=True)

    assert both < forward * 3.0, (
        f"{both * 1e3:.3f}ms per level with the backward against "
        f"{forward * 1e3:.3f}ms without it"
    )
