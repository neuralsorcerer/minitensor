# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""An optimizer is given each parameter once.

A parameter listed twice -- `a.parameters() + a.parameters()`, or two handles
to one tensor -- had its gradient applied twice: the optimizer finds a
gradient by the parameter's identity, so each handle found the same one. SGD
moved it twice as far as its learning rate said, and a stateful optimizer
advanced that parameter's state twice per step. Nothing reported it.
"""

from __future__ import annotations

import pytest

import minitensor as mt

nn = mt.nn
optim = mt.optim

OPTIMIZERS = [
    optim.SGD,
    optim.Adam,
    optim.AdamW,
    optim.Adamax,
    optim.NAdam,
    optim.RAdam,
    optim.RMSprop,
    optim.Adagrad,
    optim.Adadelta,
    optim.Rprop,
    optim.Lion,
]


@pytest.mark.parametrize("make", OPTIMIZERS, ids=lambda cls: cls.__name__)
def test_a_parameter_listed_twice_is_refused(make):
    layer = nn.DenseLayer(4, 3)
    with pytest.raises(
        ValueError, match="parameter 2 is the same tensor as parameter 0"
    ):
        make(layer.parameters() + layer.parameters(), lr=0.1)


@pytest.mark.parametrize("make", OPTIMIZERS, ids=lambda cls: cls.__name__)
def test_the_same_handle_twice_is_refused(make):
    weight = mt.Tensor([1.0, 2.0], requires_grad=True)
    with pytest.raises(
        ValueError, match="parameter 1 is the same tensor as parameter 0"
    ):
        make([weight, weight], lr=0.1)


def test_distinct_parameters_of_the_same_shape_are_fine():
    first = mt.Tensor([1.0, 2.0], requires_grad=True)
    second = mt.Tensor([1.0, 2.0], requires_grad=True)
    optim.SGD([first, second], lr=0.1)


def test_one_step_moves_a_parameter_by_its_learning_rate_once():
    weight = mt.Tensor([1.0], requires_grad=True)
    optimizer = optim.SGD([weight], lr=0.1)
    (weight * 2.0).sum().backward()
    optimizer.step()
    assert weight.numpy().tolist() == pytest.approx([0.8])


@pytest.mark.parametrize("make", OPTIMIZERS, ids=lambda cls: cls.__name__)
def test_a_single_tensor_is_one_parameter(make):
    """Iterated like a list, a tensor gave its elements, each a new tensor that
    nothing reads: the optimizer was built, stepped, and never moved it."""
    weight = mt.Tensor([1.0, 2.0], requires_grad=True)
    optimizer = make(weight, lr=0.1)
    before = weight.tolist()

    (weight * weight).sum().backward()
    optimizer.step()
    assert weight.tolist() != before
