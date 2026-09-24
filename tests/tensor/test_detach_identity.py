# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""`detach()` produces a tensor the tape treats as a different variable.

Clearing `requires_grad` is not enough. The graph keys every gradient by tensor
identity, so a detached tensor that kept its source's identity is the same
variable to the tape however its own fields read -- and the moment gradients are
switched back on, which is exactly what making a trainable copy does, the two
share one gradient slot.

That was the behaviour: `param.detach().requires_grad_(True)` produced an alias,
training it wrote into `param`, and `zero_grad` on either cleared both. The
in-place `detach_()` had always been right, so the two spellings of one
operation disagreed, and only the in-place one had tests.

Both spellings are checked here, side by side, so they cannot drift apart again.
"""

from __future__ import annotations

import numpy as np
import pytest

import minitensor as mt


@pytest.fixture(autouse=True)
def _clean_graph():
    yield
    mt.clear_autograd_graph()


def _leaf(values=(1.0, 2.0)):
    return mt.Tensor(list(values), dtype="float64", requires_grad=True)


def _detached_by_value(source):
    return source.detach().requires_grad_(True)


def _detached_in_place(source):
    copy = source.clone()
    copy.detach_()
    return copy.requires_grad_(True)


SPELLINGS = [("detach()", _detached_by_value), ("detach_()", _detached_in_place)]


@pytest.mark.parametrize("name,make", SPELLINGS)
def test_training_the_copy_leaves_the_source_alone(name, make):
    """The reason this matters: a frozen copy that trains its original."""
    param = _leaf()
    copy = make(param)

    param.zero_grad(True)
    (copy * copy).sum().backward()

    assert copy.grad is not None, f"{name}: the copy should get the gradient"
    assert param.grad is None, f"{name}: the source must not"


@pytest.mark.parametrize("name,make", SPELLINGS)
def test_a_gradient_on_the_source_does_not_reach_the_copy(name, make):
    param = _leaf()
    copy = make(param)

    copy.zero_grad(True)
    (param * 3.0).sum().backward()

    np.testing.assert_allclose(np.asarray(param.grad), [3.0, 3.0])
    assert copy.grad is None, f"{name}: the copy must not see the source's gradient"


@pytest.mark.parametrize("name,make", SPELLINGS)
def test_zero_grad_on_one_does_not_clear_the_other(name, make):
    param = _leaf()
    copy = make(param)

    (param * 2.0).sum().backward()
    assert param.grad is not None

    copy.zero_grad(True)
    assert param.grad is not None, f"{name}: zeroing the copy cleared the source"


@pytest.mark.parametrize("name,make", SPELLINGS)
def test_the_two_accumulate_independently(name, make):
    param = _leaf()
    copy = make(param)

    param.zero_grad(True)
    copy.zero_grad(True)
    (param * 2.0 + copy * 5.0).sum().backward()

    np.testing.assert_allclose(np.asarray(param.grad), [2.0, 2.0], err_msg=name)
    np.testing.assert_allclose(np.asarray(copy.grad), [5.0, 5.0], err_msg=name)


def test_detach_still_shares_the_buffer():
    """Identity is what changes; the memory is deliberately still shared."""
    source = _leaf()
    detached = source.detach()
    assert np.shares_memory(np.asarray(detached), np.asarray(source))
    np.testing.assert_allclose(np.asarray(detached), np.asarray(source))


def test_detach_still_clears_tracking():
    """The properties the previous behaviour did get right, kept."""
    source = _leaf()
    detached = source.detach()
    assert detached.requires_grad is False
    assert detached.grad is None
    assert detached.is_leaf is True


def test_a_detached_tensor_cannot_be_backpropagated():
    detached = _leaf().detach()
    with pytest.raises(
        RuntimeError, match="does not require grad and does not have a grad_fn"
    ):
        (detached * 3.0).sum().backward()


def test_detaching_an_intermediate_cuts_the_graph():
    """A detached non-leaf must not carry gradient back to what produced it."""
    x = _leaf()
    y = x * 3.0
    cut = y.detach().requires_grad_(True)

    x.zero_grad(True)
    (cut * 2.0).sum().backward()

    assert cut.grad is not None
    assert x.grad is None, "gradient crossed a detach() into the producing tensor"
