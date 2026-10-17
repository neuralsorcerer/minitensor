# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""`x op= y` writes into `x`.

With no in-place dunders Python fell back to `x = x op y`: the name moved to a
new tensor and nothing else saw it. A parameter handle shares its layer's
storage, so the manual update `p -= lr * p.grad` read as an update and left
the model as it was.
"""

from __future__ import annotations

import pytest

import minitensor as mt
from minitensor import nn


def _apply(tensor, symbol, other):
    scope = {"t": tensor, "v": other}
    exec(f"t {symbol}= v", {}, scope)
    return scope["t"]


@pytest.mark.parametrize(
    "symbol,start,other,want",
    [
        ("+", [1.0, 2.0], 1.0, [2.0, 3.0]),
        ("-", [1.0, 2.0], [0.5, 0.5], [0.5, 1.5]),
        ("*", [1.5, 2.0], mt.Tensor([2, 3]), [3.0, 6.0]),
        ("/", [1.0, 3.0], 2.0, [0.5, 1.5]),
        ("//", [7, -7], 2, [3, -4]),
        ("%", [7, -7], 3, [1, 2]),
        ("**", [2, 3], 2, [4, 9]),
        ("&", [6, 5], 3, [2, 1]),
        ("|", [4, 1], 2, [6, 3]),
        ("^", [6, 5], 3, [5, 6]),
        ("<<", [1, 2], 2, [4, 8]),
        (">>", [8, 4], 2, [2, 1]),
    ],
)
def test_every_operator_writes_into_its_tensor(symbol, start, other, want):
    tensor = mt.Tensor(start)
    alias = tensor
    result = _apply(tensor, symbol, other)
    assert result is tensor
    assert alias.tolist() == want
    assert str(alias.dtype) == str(mt.Tensor(start).dtype)


def test_matmul_in_place_keeps_a_square_shape():
    matrix = mt.Tensor([[1.0, 2.0], [3.0, 4.0]])
    alias = matrix
    matrix @= mt.Tensor([[0.0, 1.0], [1.0, 0.0]])
    assert alias.tolist() == [[2.0, 1.0], [4.0, 3.0]]


def test_a_smaller_operand_broadcasts_into_the_tensor():
    matrix = mt.Tensor([[1.0, 2.0], [3.0, 4.0]])
    matrix += mt.Tensor([10.0, 20.0])
    assert matrix.tolist() == [[11.0, 22.0], [13.0, 24.0]]


@pytest.mark.parametrize(
    "start,symbol,other",
    [
        ([1, 2], "+", 0.5),
        ([1, 2], "/", 2),
        ([1.0, 2.0], "+", mt.Tensor([0.5, 0.5], dtype="float64")),
    ],
)
def test_a_result_of_another_dtype_is_refused(start, symbol, other):
    tensor = mt.Tensor(start)
    with pytest.raises(TypeError, match="cannot hold"):
        _apply(tensor, symbol, other)
    assert tensor.tolist() == start


def test_a_result_of_another_shape_is_refused():
    row = mt.Tensor([1.0, 2.0])
    with pytest.raises(ValueError, match="shape"):
        row += mt.Tensor([[1.0, 2.0], [3.0, 4.0]])
    assert row.tolist() == [1.0, 2.0]


@pytest.mark.parametrize("untracked", [False, True], ids=["tracked", "no_grad"])
def test_a_manual_update_reaches_the_layer(untracked):
    layer = nn.DenseLayer(2, 1)
    before = layer.weight.tolist()[0]
    layer(mt.Tensor([[1.0, 2.0]])).sum().backward()
    gradient = layer.weight.grad.tolist()[0]
    if untracked:
        with mt.no_grad():
            for parameter in layer.parameters():
                parameter -= 0.1 * parameter.grad
    else:
        for parameter in layer.parameters():
            parameter -= 0.1 * parameter.grad
    after = layer.weight.tolist()[0]
    assert after == pytest.approx([b - 0.1 * g for b, g in zip(before, gradient)])
    assert layer.weight.is_leaf and layer.weight.requires_grad
    # And the layer trains on from there.
    layer(mt.Tensor([[1.0, 2.0]])).sum().backward()
    assert layer.weight.grad is not None


def test_a_tensor_a_pending_backward_needs_is_not_written():
    weight = mt.Tensor([1.0, 2.0], requires_grad=True)
    pending = (weight * weight).sum()
    with pytest.raises(ValueError, match="pending backward"):
        weight -= 1.0
    pending.backward()
    assert weight.grad.tolist() == [2.0, 4.0]


def test_a_write_into_a_computed_tensor_routes_the_gradient():
    source = mt.Tensor([1.0, 2.0], requires_grad=True)
    computed = source * 3
    computed += 1.0
    computed.sum().backward()
    assert computed.tolist() == [4.0, 7.0]
    assert source.grad.tolist() == [3.0, 3.0]


def test_a_live_array_view_keeps_the_values_it_was_taken_with():
    # The documented rule for every in-place write: a view holds the buffer,
    # so the write lands in fresh storage and the view keeps what it saw.
    import numpy as np

    tensor = mt.Tensor([1.0, 2.0])
    view = np.asarray(tensor)
    tensor += 1.0
    assert view.tolist() == [1.0, 2.0]
    assert tensor.tolist() == [2.0, 3.0]


def test_two_handles_on_one_tensor_both_see_the_write():
    first = mt.Tensor([1.0, 2.0, 3.0])
    second = first
    second *= 2.0
    second -= mt.Tensor([1.0, 1.0, 1.0])
    assert first.tolist() == [1.0, 3.0, 5.0]
