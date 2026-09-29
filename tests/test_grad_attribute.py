# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""`tensor.grad` can be assigned, as well as read.

It was read-only, so `p.grad = None` and `p.grad = p.grad * 0.5` -- the two
most common gradient edits in a training loop -- raised a bare "attribute is
not writable". Assigning `None` clears the gradient the way
`zero_grad(set_to_none=True)` does; assigning a tensor puts it where a backward
pass would have, so a backward accumulates onto it and an optimizer steps with
it.
"""

from __future__ import annotations

import numpy as np
import pytest

import minitensor as mt


def _with_gradient():
    weight = mt.Tensor([1.0, 2.0], requires_grad=True)
    (weight * weight).sum().backward()
    return weight


def test_assigning_none_clears_the_gradient():
    weight = _with_gradient()
    weight.grad = None
    assert weight.grad is None


def test_an_assigned_gradient_is_the_one_an_optimizer_steps_with():
    weight = mt.Tensor([1.0, 2.0], requires_grad=True)
    optimizer = mt.optim.SGD([weight], lr=0.5)
    weight.grad = mt.Tensor([1.0, -1.0])
    optimizer.step()
    np.testing.assert_allclose(weight.numpy(), [0.5, 2.5])


def test_a_backward_accumulates_onto_an_assigned_gradient():
    weight = mt.Tensor([1.0, 2.0], requires_grad=True)
    weight.grad = mt.Tensor([10.0, 10.0])
    (weight * weight).sum().backward()
    np.testing.assert_allclose(weight.grad.numpy(), [12.0, 14.0])


def test_a_gradient_can_be_rescaled_through_the_attribute():
    weight = _with_gradient()
    weight.grad = weight.grad * 0.5
    np.testing.assert_allclose(weight.grad.numpy(), [1.0, 2.0])


@pytest.mark.parametrize(
    "value,error,message",
    [
        (lambda: mt.zeros(3), ValueError, "shape"),
        (lambda: mt.zeros(2, dtype="float64"), TypeError, "dtype"),
    ],
    ids=["shape", "dtype"],
)
def test_a_gradient_for_some_other_tensor_is_refused(value, error, message):
    weight = _with_gradient()
    with pytest.raises(error, match=message):
        weight.grad = value()
    np.testing.assert_allclose(weight.grad.numpy(), [2.0, 4.0])


def test_only_a_tensor_that_requires_a_gradient_can_be_given_one():
    plain = mt.zeros(2)
    with pytest.raises(ValueError, match="requires a gradient"):
        plain.grad = mt.zeros(2)
    plain.grad = None
    assert plain.grad is None
