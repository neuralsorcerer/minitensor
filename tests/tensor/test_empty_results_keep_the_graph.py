# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""An empty result is still part of the graph.

Several operations took a shortcut when their result had no elements: a
zero-length `narrow` or `slice`, an empty `index_select`, `gather` or `x[[]]`,
`topk(0)`, and `sort`, `cat` and `stack` over an empty input. The shortcut built
a fresh tensor that claimed `requires_grad` with nothing behind it, so backward
stopped there and the input was left with no gradient at all -- which an
optimizer reads as "skip this parameter", not "step it by zero". Each now hands
back a gradient of zeros (or an empty one, for an empty input).

Separately, reducing over an explicitly empty list of dimensions --
`x.sum([])`, `x.nansum([])`, `x.prod([])` -- is the tensor itself, but recorded
a node whose output was its own input, and backward refused the cycle.
"""

import numpy as np
import pytest

import minitensor as mt

F = mt.functional

EMPTY_FROM_FULL = {
    "narrow": lambda x: x.narrow(0, 0, 0),
    "list index": lambda x: x[[]],
    "topk(0)": lambda x: x.topk(0, dim=-1)[0],
    "index_select": lambda x: x.index_select(0, mt.tensor([], dtype="int64")),
    "gather": lambda x: x.gather(-1, mt.zeros((2, 0), dtype="int64")),
    "diff past the length": lambda x: mt.diff(x, n=9),
}


@pytest.mark.parametrize("name", sorted(EMPTY_FROM_FULL))
def test_an_empty_selection_gives_a_zero_gradient(name):
    x = mt.ones((2, 3), requires_grad=True)
    out = EMPTY_FROM_FULL[name](x)
    assert out.numel() == 0
    assert out.requires_grad and not out.is_leaf
    out.sum().backward()
    assert x.grad is not None, name
    np.testing.assert_array_equal(x.grad.numpy(), np.zeros((2, 3), np.float32))


OVER_AN_EMPTY_INPUT = {
    "sort": lambda x: x.sort(-1)[0],
    "cat": lambda x: mt.cat([x], 0),
    "stack": lambda x: mt.stack([x], 0),
    "split": lambda x: x.split(100, 0)[0],
    "chunk": lambda x: x.chunk(1, 0)[0],
}


@pytest.mark.parametrize("shape", [(0, 3), (3, 0)])
@pytest.mark.parametrize("name", sorted(OVER_AN_EMPTY_INPUT))
def test_an_empty_input_gets_an_empty_gradient(name, shape):
    x = mt.zeros(shape, requires_grad=True)
    OVER_AN_EMPTY_INPUT[name](x).sum().backward()
    assert x.grad is not None, name
    assert tuple(x.grad.shape) == shape


def test_an_empty_piece_of_a_cat_still_gets_its_share():
    full = mt.ones((2, 3), requires_grad=True)
    empty = mt.zeros((0, 3), requires_grad=True)
    (mt.cat([empty, full], 0) * 2.0).sum().backward()
    np.testing.assert_array_equal(full.grad.numpy(), np.full((2, 3), 2.0, np.float32))
    assert tuple(empty.grad.shape) == (0, 3)


@pytest.mark.parametrize("reduce", ["sum", "nansum", "prod"])
def test_reducing_over_no_dimensions_is_the_tensor_itself(reduce):
    x = mt.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)
    y = x * 3.0
    out = getattr(y, reduce)([])
    np.testing.assert_array_equal(out.numpy(), y.detach().numpy())
    (out * 1.0).sum().backward()
    np.testing.assert_array_equal(x.grad.numpy(), np.full((2, 3), 3.0, np.float32))


def test_a_frozen_input_still_gives_an_empty_result_without_a_graph():
    x = mt.ones((2, 3))
    out = x.narrow(0, 0, 0)
    assert not out.requires_grad
    assert tuple(out.shape) == (0, 3)
