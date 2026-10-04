# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""A tensor behaves as the Python object it stands in for.

Each of these was a gap between a tensor and the number or sequence it is
used as: `int()` rounded an int64 through a float, iteration over a 0-d tensor
ran zero times instead of refusing, `t == None` raised, an integer tensor could
not subscript a list, and `round`, `divmod` and a format spec did not apply.
"""

from __future__ import annotations

import math

import pytest

import minitensor as mt


def test_int_of_an_integer_tensor_is_exact():
    """It went through an f64, so an int64 above 2^53 came back rounded."""
    big = 2**60 + 1
    assert int(mt.Tensor([big], dtype="int64")) == big
    assert int(mt.Tensor([-7], dtype="int32")) == -7
    assert int(mt.Tensor([True], dtype="bool")) == 1


def test_int_of_a_float_tensor_truncates_as_int_of_a_float_does():
    assert int(mt.Tensor(-3.7)) == -3
    assert int(mt.Tensor(1e20, dtype="float64")) == int(1e20)
    with pytest.raises(ValueError):
        int(mt.Tensor(float("nan")))


def test_an_integer_tensor_is_an_index():
    items = ["a", "b", "c"]
    assert items[mt.Tensor(2, dtype="int64")] == "c"
    assert list(range(mt.Tensor(3, dtype="int32"))) == [0, 1, 2]
    with pytest.raises(TypeError, match="only an integer tensor"):
        items[mt.Tensor(1.0)]


def test_iterating_walks_the_first_axis_and_refuses_a_scalar():
    rows = [row.tolist() for row in mt.Tensor([[1.0, 2.0], [3.0, 4.0]])]
    assert rows == [[1.0, 2.0], [3.0, 4.0]]
    assert list(mt.zeros(0)) == []
    with pytest.raises(TypeError, match="0-d"):
        list(mt.Tensor(3.5))


def test_iterated_rows_keep_their_place_in_the_graph():
    weight = mt.ones(2, 2, requires_grad=True)
    sum(row.sum() * (i + 1) for i, row in enumerate(weight)).backward()
    assert weight.grad.tolist() == [[1.0, 1.0], [2.0, 2.0]]


def test_comparing_with_something_that_is_no_tensor_is_not_equal():
    values = mt.Tensor([1.0, 2.0])
    assert (values == None) is False  # noqa: E711 - the comparison is the point
    assert (values != None) is True  # noqa: E711
    assert (values == "text") is False
    assert values in [None, values]
    assert (values == [1.0, 5.0]).tolist() == [True, False]


def test_a_shape_mismatch_in_a_comparison_still_raises():
    with pytest.raises(ValueError):
        mt.Tensor([1.0, 2.0]) == mt.ones(3)


def test_round_divmod_and_format():
    assert round(mt.Tensor([1.26, 3.5])).tolist() == [1.0, 4.0]
    assert round(mt.Tensor([1.25], dtype="float64"), 1).tolist() == [1.2]
    quotient, remainder = divmod(mt.Tensor([7.0, -7.0]), 2)
    assert quotient.tolist() == [3.0, -4.0] and remainder.tolist() == [1.0, 1.0]
    quotient, remainder = divmod(7, mt.Tensor([2.0]))
    assert quotient.tolist() == [3.0] and remainder.tolist() == [1.0]

    assert f"{mt.Tensor(3.14159):.2f}" == "3.14"
    assert f"{mt.Tensor([1.0, 2.0])}" == str(mt.Tensor([1.0, 2.0]))
    with pytest.raises(TypeError, match="one-element tensor"):
        f"{mt.Tensor([1.0, 2.0]):.2f}"


def test_the_repr_spells_booleans_as_python_does():
    assert "requires_grad=True" in repr(mt.Tensor([1.0], requires_grad=True))
    assert "requires_grad=False" in repr(mt.Tensor([1.0]))


def test_the_truth_of_an_empty_tensor_names_the_right_problem():
    with pytest.raises(ValueError, match="empty tensor"):
        bool(mt.zeros(0))
    with pytest.raises(ValueError, match="more than one element"):
        bool(mt.zeros(2))


@pytest.mark.parametrize("rounding", [math.floor, math.ceil, math.trunc])
def test_math_rounding_is_exact_for_a_large_int64(rounding):
    # `floor` and `ceil` used to go through `float()`, which rounds an int64
    # past 2**53, and `trunc` was missing outright.
    value = 2**62 + 1
    got = rounding(mt.tensor(value, dtype="int64"))
    assert type(got) is int and got == value


@pytest.mark.parametrize("value", [2.5, -2.5, -0.5, 3.0])
def test_math_rounding_of_a_float_matches_the_float(value):
    t = mt.tensor(value, dtype="float64")
    for rounding in (math.floor, math.ceil, math.trunc):
        assert rounding(t) == rounding(value)


def test_math_rounding_needs_one_element():
    with pytest.raises(TypeError, match="one element"):
        math.trunc(mt.tensor([1.0, 2.0]))


def test_reversed_walks_the_first_axis_backwards():
    t = mt.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    assert [row.tolist() for row in reversed(t)] == [[5.0, 6.0], [3.0, 4.0], [1.0, 2.0]]
    assert list(reversed(mt.zeros(0, 2))) == []
    with pytest.raises(TypeError, match="0-d"):
        reversed(mt.tensor(1.0))
