# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""What `backward` says when it cannot run names the actual way out.

Calling it on a tensor of several elements suggested setting
`requires_grad=True` -- which the tensor already had -- rather than reducing
it or passing a gradient. Declining `create_graph=True` gave as its reason
that "all computations execute in the Rust backend", which explains nothing.
"""

import pytest

import minitensor as mt


def test_a_non_scalar_backward_says_how_many_elements_and_what_to_do():
    y = mt.tensor([1.0, 2.0, 3.0], requires_grad=True) * 2
    with pytest.raises(RuntimeError) as info:
        y.backward()
    message = str(info.value)
    assert "this one has 3 elements" in message
    assert ".sum()" in message and "gradient of the same shape" in message
    assert "requires_grad=True" not in message


def test_create_graph_is_declined_with_the_reason():
    y = mt.tensor(2.0, requires_grad=True) ** 3
    with pytest.raises(NotImplementedError, match="not itself recorded"):
        y.backward(create_graph=True)


def test_a_tensor_with_nothing_to_differentiate_says_so():
    with pytest.raises(RuntimeError, match="nothing to differentiate"):
        (mt.tensor([1.0, 2.0]) * 2).sum().backward()
