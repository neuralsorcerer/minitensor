# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""In-place writes must not silently rewrite what a pending backward reads.

Backward nodes hold their operands by `Arc`, and `data_mut` writes through
rather than copying for one case: a leaf that requires grad. That is deliberate
-- it is what lets a `layer.weight` handle update the layer, and it is the path
optimizers use. The gap was a leaf that is *also* held as an operand by a live
backward node.

`(a * b).sum()` followed by `a.fill_(99)` used to return 99 as `b`'s gradient
instead of 2, the value of `a` during the forward. Not an error, not a missing
gradient -- a plausible number that is wrong, for a tensor the caller never
touched. `mul`, `div` and `matmul` all did it. This is the same hazard that
kept `+=` off the Python surface; `fill_`, `copy_` and `t[i] = v` reached it
anyway -- guarding the first two left the third still corrupting, which is why
the sweep at the bottom of this file checks every method rather than the ones
someone remembered.

They now refuse. Non-leaves were always safe because they copy on write, and
the ordinary orderings -- initializing a parameter, mutating before the forward,
clamping between training steps -- stay allowed, since the graph is either not
built yet or already released.
"""

import numpy as np
import pytest

import minitensor as mt
from minitensor import nn


@pytest.fixture(autouse=True)
def _clear_graph():
    yield
    mt.clear_autograd_graph()


def _leaf(value, size=1):
    return mt.Tensor(np.full(size, value), dtype="float64", requires_grad=True)


@pytest.mark.parametrize("mutate", ["fill_", "copy_"])
def test_mutating_a_consumed_leaf_raises(mutate):
    a, b = _leaf(2.0), _leaf(3.0)
    (a * b).sum()  # a is now an operand of a live backward node

    with pytest.raises(Exception, match="pending backward"):
        if mutate == "fill_":
            a.fill_(99.0)
        else:
            a.copy_(mt.Tensor(np.array([99.0]), dtype="float64"))


@pytest.mark.parametrize("op", ["mul", "div", "matmul"])
def test_the_gradient_that_used_to_be_corrupted(op):
    """The other operand's gradient was the casualty, not the mutated one."""
    if op == "matmul":
        a = mt.Tensor(np.array([[1.0, 2.0]]), dtype="float64", requires_grad=True)
        b = mt.Tensor(np.array([[3.0], [4.0]]), dtype="float64", requires_grad=True)
        out = (a @ b).sum()
        expected = np.array([[1.0], [2.0]])  # d/db of a@b is a
    else:
        a, b = _leaf(2.0), _leaf(3.0)
        out = (a * b).sum() if op == "mul" else (a / b).sum()
        expected = np.array([2.0]) if op == "mul" else np.array([-2.0 / 9.0])

    with pytest.raises(Exception, match="pending backward"):
        a.fill_(99.0)

    # With the write refused, the backward still sees the forward values.
    out.backward()
    np.testing.assert_allclose(b.grad.numpy(), expected)


def test_a_non_leaf_may_still_be_mutated_because_it_copies_on_write():
    x = _leaf(3.0)
    h = x * 2.0
    out = (h * h).sum()

    h.fill_(50.0)  # allowed: h copies on write, the graph keeps its own values

    out.backward()
    # d/dx of (2x)^2 is 8x = 24 at x = 3, unaffected by the write to h.
    np.testing.assert_allclose(x.grad.numpy(), [24.0])


def test_writing_a_parameter_before_any_forward_is_allowed():
    layer = nn.DenseLayer(2, 2, dtype="float64")
    layer.weight.fill_(0.25)
    np.testing.assert_allclose(layer.weight.numpy(), np.full((2, 2), 0.25))


def test_mutate_then_forward_gives_the_new_value_a_gradient():
    w = _leaf(1.0)
    w.copy_(mt.Tensor(np.array([3.0]), dtype="float64"))
    (w * w).sum().backward()
    np.testing.assert_allclose(w.grad.numpy(), [6.0])  # 2 * 3


def test_clamping_between_training_steps_is_allowed():
    # step() releases the graph, so the next write has nothing pending to break.
    layer = nn.DenseLayer(2, 1, dtype="float64")
    optimizer = mt.optim.SGD(layer.parameters(), lr=0.1)
    x = mt.Tensor(np.ones((1, 2)), dtype="float64")

    for _ in range(3):
        optimizer.zero_grad()
        layer(x).sum().backward()
        optimizer.step()
        layer.weight.fill_(0.5)

    np.testing.assert_allclose(layer.weight.numpy(), np.full((1, 2), 0.5))


def test_clearing_the_graph_unblocks_the_write():
    a, b = _leaf(2.0), _leaf(3.0)
    (a * b).sum()
    with pytest.raises(Exception, match="pending backward"):
        a.fill_(99.0)

    mt.clear_autograd_graph()
    a.fill_(99.0)  # nothing pending now
    np.testing.assert_allclose(a.numpy(), [99.0])


@pytest.mark.parametrize(
    "assign",
    [
        pytest.param(lambda t: t.__setitem__(0, 99.0), id="index"),
        pytest.param(lambda t: t.__setitem__(slice(None), 99.0), id="slice"),
        pytest.param(
            lambda t: t.__setitem__(
                slice(None), mt.Tensor(np.full(2, 99.0), dtype="float64")
            ),
            id="slice-tensor",
        ),
    ],
)
def test_index_assignment_is_refused_too(assign):
    # `t[i] = v` writes through the same storage as `fill_`; guarding only the
    # named mutators left this path corrupting.
    a = mt.Tensor(np.full(2, 2.0), dtype="float64", requires_grad=True)
    b = mt.Tensor(np.full(2, 3.0), dtype="float64", requires_grad=True)
    loss = (a * b).sum()

    with pytest.raises(Exception, match="pending backward"):
        assign(a)

    loss.backward()
    np.testing.assert_allclose(b.grad.numpy(), [2.0, 2.0])


def test_no_tensor_method_can_corrupt_a_pending_backward():
    """Sweep every callable, so a new mutator cannot quietly reopen this.

    The guard was added to `fill_` and `copy_` first, and `__setitem__` kept
    corrupting gradients until this sweep pointed at it. Checking the whole
    surface is the only version of this test that stays true as methods are
    added.
    """
    probe = mt.Tensor(np.ones(2), dtype="float64")
    source = mt.Tensor(np.full(2, 99.0), dtype="float64")
    corrupted = []

    def attempt(label, action):
        a = mt.Tensor(np.full(2, 2.0), dtype="float64", requires_grad=True)
        b = mt.Tensor(np.full(2, 3.0), dtype="float64", requires_grad=True)
        loss = (a * b).sum()
        try:
            action(a)
        except Exception:
            mt.clear_autograd_graph()
            return  # refused, or not applicable to this input
        try:
            loss.backward()
        except Exception:
            mt.clear_autograd_graph()
            return
        # d/db of a*b is a's forward value, 2 -- never the mutated one.
        if b.grad is not None and not np.allclose(b.grad.numpy(), [2.0, 2.0]):
            corrupted.append(f"{label} -> grad_b={b.grad.numpy()}")
        mt.clear_autograd_graph()

    attempt("t[0] = v", lambda t: t.__setitem__(0, 99.0))
    attempt("t[:] = v", lambda t: t.__setitem__(slice(None), source))

    for name in sorted(n for n in dir(probe) if not n.startswith("_")):
        if not callable(getattr(probe, name, None)):
            continue
        for args in ((), (99.0,), (source,), (0,)):
            try:
                getattr(mt.Tensor(np.ones(2), dtype="float64"), name)(*args)
            except TypeError:
                continue  # wrong arity; try the next shape
            except Exception:
                pass  # raised for another reason, but the arity is right
            attempt(f".{name}", lambda t, n=name, a=args: getattr(t, n)(*a))
            break

    assert not corrupted, (
        "these operations rewrote a tensor a pending backward still needed: "
        + "; ".join(corrupted)
    )
