# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""Assigning into a tracked tensor is recorded, so gradients see the write.

`y[key] = value` and `y.copy_(value)` used to write into `y` without telling
the graph. `y`'s node stayed the one that computed its old values, so the
gradient reaching `y` went on to those values -- including where the write had
replaced them -- and none of it reached the value. `y = 2 * x; y[1] = 0` gave
`x` a gradient of 2 at the element the write had cut it off from.

An assignment into a tensor that is part of a recorded computation, or of a
value that is, now makes a new tensor: the gradient reaching it goes to the
value where the value was written and to the old tensor everywhere else. A leaf
that requires a gradient is still written in place and not recorded, which is
how a parameter is set, and so is anything written with gradients off.
"""

from __future__ import annotations

import numpy as np
import pytest

import minitensor as mt

RNG = np.random.default_rng(7)


def _tensor(*shape, requires_grad=True):
    return mt.tensor(
        RNG.uniform(-1.0, 1.0, size=shape), dtype="float64", requires_grad=requires_grad
    )


def _weights(shape):
    """Distinct weights per position, so a gradient sent to the wrong one shows."""
    return mt.tensor(
        np.arange(1.0, 1.0 + np.prod(shape)).reshape(shape), dtype="float64"
    )


def _check(shape, key, value_shape):
    """`y = x * 1.5; y[key] = v` checked against finite differences for both
    inputs, with a loss that is not linear in `y`."""
    weights = _weights(shape)

    def f(x, v):
        y = x * 1.5
        y[key] = v * 2.0
        return (y * y * weights).sum()

    mt.gradcheck(f, [_tensor(*shape), _tensor(*value_shape)], atol=1e-6, rtol=1e-5)


# --- every way of naming the positions --------------------------------------


@pytest.mark.parametrize(
    "shape,key,value_shape",
    [
        ((5,), 2, ()),
        ((5,), -1, ()),
        ((5,), slice(1, 4), (3,)),
        ((5,), slice(None, None, 2), (3,)),
        ((3, 4), 1, (4,)),
        ((3, 4), (slice(None), 2), (3,)),
        ((3, 4), (slice(0, 2), slice(1, 3)), (2, 2)),
        ((2, 3, 4), (Ellipsis, 1), (2, 3)),
        ((2, 3, 4), (0, Ellipsis), (3, 4)),
    ],
    ids=[
        "int",
        "negative",
        "slice",
        "strided",
        "row",
        "column",
        "block",
        "ellipsis",
        "leading",
    ],
)
def test_basic_keys(shape, key, value_shape):
    _check(shape, key, value_shape)


@pytest.mark.parametrize(
    "shape,key,value_shape",
    [
        ((5,), [0, 3], (2,)),
        ((3, 4), [2, 0], (2, 4)),
        ((3, 4), (slice(None), [3, 1]), (3, 2)),
        ((2, 3, 4), (1, slice(None), [0, 2]), (3, 2)),
    ],
    ids=["list", "rows", "columns", "middle"],
)
def test_index_array_keys(shape, key, value_shape):
    _check(shape, key, value_shape)


def test_a_boolean_mask():
    mask = mt.tensor([[True, False, True], [False, True, False]], dtype="bool")
    _check((2, 3, 4), mask, (3, 4))


# --- how the value is lined up ----------------------------------------------


def test_a_value_broadcast_to_several_positions_sums_their_gradients():
    _check((3, 4), (slice(None), slice(0, 2)), (2,))
    _check((3, 4), slice(None), ())


def test_a_position_named_twice_keeps_the_last_write_and_its_gradient():
    x = mt.tensor([1.0, 2.0, 3.0], dtype="float64", requires_grad=True)
    v = mt.tensor([4.0, 5.0], dtype="float64", requires_grad=True)
    y = x * 2.0
    y[[0, 0]] = v
    (y * y).sum().backward()

    np.testing.assert_array_equal(y.numpy(), [5.0, 4.0, 6.0])
    np.testing.assert_array_equal(x.grad.numpy(), [0.0, 16.0, 24.0])
    np.testing.assert_array_equal(v.grad.numpy(), [0.0, 10.0])


def test_a_python_value_cuts_the_positions_it_replaces():
    """The case the old write got wrong without any tracked value involved."""
    x = mt.tensor([1.0, 2.0, 3.0], dtype="float64", requires_grad=True)
    y = x * 2.0
    y[1] = 0.0
    y.sum().backward()
    np.testing.assert_array_equal(x.grad.numpy(), [2.0, 0.0, 2.0])


# --- what the assignment leaves behind --------------------------------------


def test_what_was_computed_from_the_old_values_keeps_them():
    x = mt.tensor([1.0, 2.0, 3.0], dtype="float64", requires_grad=True)
    y = x * 2.0
    squares = y * y
    y[0] = 100.0

    (squares.sum() + y.sum()).backward()
    np.testing.assert_array_equal(squares.numpy(), [4.0, 16.0, 36.0])
    np.testing.assert_array_equal(x.grad.numpy(), [8.0, 18.0, 26.0])


def test_two_assignments_in_a_row():
    def f(x, v, w):
        y = x * 1.0
        y[0] = v
        y[1:] = w * y[0]
        return (y * y).sum()

    mt.gradcheck(f, [_tensor(4), _tensor(), _tensor(3)], atol=1e-6, rtol=1e-5)


def test_a_value_computed_from_the_target_itself():
    def f(x):
        y = x * 1.0
        y[1:] = y[:-1] * 3.0
        return (y * y).sum()

    mt.gradcheck(f, [_tensor(5)], atol=1e-6, rtol=1e-5)


def test_a_plain_tensor_assigned_a_tracked_value_becomes_tracked():
    """Building a result position by position, into a buffer of zeros."""

    def f(v):
        out = mt.zeros(2, 3, dtype="float64")
        out[0] = v.sin()
        out[1, 1:] = v[:2] * v[2]
        return (out * out * _weights((2, 3))).sum()

    mt.gradcheck(f, [_tensor(3)], atol=1e-6, rtol=1e-5)


# --- copy_ ------------------------------------------------------------------


def test_copy_of_a_tracked_value_gives_it_the_gradient():
    x = mt.tensor([1.0, 2.0, 3.0], dtype="float64", requires_grad=True)
    v = mt.tensor([4.0, 5.0, 6.0], dtype="float64", requires_grad=True)
    y = x * 2.0
    y.copy_(v)
    (y * y).sum().backward()

    np.testing.assert_array_equal(v.grad.numpy(), [8.0, 10.0, 12.0])
    assert x.grad is None or not np.any(x.grad.numpy())


def test_copy_casts_and_the_gradient_comes_back_in_the_value_dtype():
    target = mt.zeros(3, dtype="float64")
    v = mt.tensor([1.0, 2.0, 3.0], dtype="float32", requires_grad=True)
    target.copy_(v)
    assert target.dtype == "float64"

    (target * target).sum().backward()
    assert v.grad.dtype == "float32"
    np.testing.assert_array_equal(v.grad.numpy(), [2.0, 4.0, 6.0])


# --- what is still written in place -----------------------------------------


def test_a_parameter_is_still_written_in_place():
    weight = mt.zeros(3, dtype="float64", requires_grad=True)
    handle = weight
    weight[0] = 5.0
    weight.copy_(mt.tensor([1.0, 2.0, 3.0], dtype="float64"))

    assert weight.is_leaf and handle is weight
    np.testing.assert_array_equal(handle.numpy(), [1.0, 2.0, 3.0])


def test_a_write_with_gradients_off_is_not_recorded():
    x = mt.tensor([1.0, 2.0, 3.0], dtype="float64", requires_grad=True)
    y = x * 2.0
    with mt.no_grad():
        y[0] = 0.0
    y.sum().backward()
    np.testing.assert_array_equal(x.grad.numpy(), [2.0, 2.0, 2.0])


def test_an_untracked_assignment_leaves_the_graph_alone():
    before = mt.autograd_graph_size()
    buffer = mt.zeros(4)
    for i in range(4):
        buffer[i] = float(i)
    assert mt.autograd_graph_size() == before
    np.testing.assert_array_equal(buffer.numpy(), [0.0, 1.0, 2.0, 3.0])


@pytest.mark.parametrize(
    "key",
    [slice(0, 2), [0, 1], "mask"],
    ids=["slice", "index_array", "mask"],
)
def test_a_value_of_another_dtype_is_refused_however_the_positions_are_named(key):
    """A value with a dtype of its own keeps it, and a write refuses one that
    disagrees with the destination. A slice or an index array refused, but a
    mask cast the value instead, so one assignment followed two rules."""
    if key == "mask":
        key = mt.tensor([True, True, False], dtype="bool")
    target = mt.zeros(3, dtype="int64")
    value = mt.tensor([1.5, 2.5], dtype="float64", requires_grad=True)

    with pytest.raises(TypeError, match="expected Int64, got Float64"):
        target[key] = value
    assert target.dtype == "int64" and not target.requires_grad
    np.testing.assert_array_equal(target.numpy(), [0, 0, 0])
