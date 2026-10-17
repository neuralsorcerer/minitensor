# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""`arange` and `full` are typed by what they are given; products take integers.

`arange` counted its values by dividing out the span, and the division rounds:
`arange(1, 1.3, 0.1)` came to four values, the last of them `1.3` -- the stop
of a range that excludes its stop. Integer bounds at a float dtype went
through an f64, where `2**62` and `2**62 + 2` are one number, and the range
came back empty. With no dtype both functions gave the default float whatever
they were handed, so `arange(2**24 + 1)` could not hold its own last value.

`matmul` refused any two dtypes, guarding against a float32 activation meeting
a float64 weight. That guard stays, but an integer or bool operand has no float
width to protect, and now takes the other's dtype as it does in arithmetic.
"""

from __future__ import annotations

import numpy as np
import pytest

import minitensor as mt


def test_the_stop_is_never_part_of_the_range():
    assert mt.arange(1, 1.3, 0.1, dtype="float64").tolist() == [1.0, 1.1, 1.2]
    rng = np.random.default_rng(7)
    for _ in range(3000):
        start = round(float(rng.uniform(-5, 5)), int(rng.integers(0, 4)))
        step = round(float(rng.uniform(0.05, 2)), int(rng.integers(1, 4)))
        step *= float(rng.choice([1, -1]))
        stop = round(start + step * int(rng.integers(0, 30)), 6)
        values = mt.arange(start, stop, step, dtype="float64").tolist()
        count = len(values)
        # Every value strictly before the stop, and the next one not.
        before = (lambda v: v < stop) if step > 0 else (lambda v: v > stop)
        assert all(before(start + i * step) for i in range(count))
        assert not before(start + count * step)


def test_integer_bounds_are_counted_exactly_at_any_dtype():
    for dtype in ("float32", "float64", "int64"):
        assert len(mt.arange(2**62, 2**62 + 2, dtype=dtype).tolist()) == 2
    assert mt.arange(2**62, 2**62 + 2).tolist() == [2**62, 2**62 + 1]
    # Each value is exact before it is rounded to the dtype.
    assert mt.arange(2**53, 2**53 + 3, dtype="float64").tolist() == [
        2.0**53,
        2.0**53,
        2.0**53 + 2,
    ]


def test_arange_is_typed_by_its_bounds():
    assert mt.arange(5).dtype == "int64"
    assert mt.arange(2, 9, 3).dtype == "int64"
    assert mt.arange(5.0).dtype == "float32"
    assert mt.arange(0, 1, 0.25).dtype == "float32"
    assert mt.arange(5, dtype="float64").dtype == "float64"
    last = 2**24 + 1
    assert mt.arange(last + 1).tolist()[-1] == last


def test_a_float_range_from_integer_bounds_can_require_a_gradient():
    ramp = mt.arange(0, 4, dtype="float32", requires_grad=True)
    assert ramp.requires_grad
    (ramp * ramp).sum().backward()
    assert ramp.grad.tolist() == [0.0, 2.0, 4.0, 6.0]
    with pytest.raises(ValueError, match="floating point"):
        mt.arange(4, requires_grad=True)


@pytest.mark.parametrize(
    "fill,dtype",
    [(7, "int64"), (True, "bool"), (0.5, "float32"), (np.float64(0.1), "float64")],
)
def test_full_is_typed_by_its_fill_value(fill, dtype):
    filled = mt.full((2, 3), fill)
    assert filled.dtype == dtype
    assert filled.tolist() == [[fill] * 3] * 2
    assert mt.full((2,), fill, dtype="float64").dtype == "float64"


@pytest.mark.parametrize(
    "product",
    [
        lambda a, b: a @ b,
        mt.matmul,
        mt.mm,
        lambda a, b: mt.bmm(a.unsqueeze(0), b.unsqueeze(0)).squeeze(0),
        lambda a, b: mt.einsum("ij,jk->ik", a, b),
        lambda a, b: mt.tensordot(a, b, 1),
    ],
    ids=["operator", "matmul", "mm", "bmm", "einsum", "tensordot"],
)
def test_an_integer_operand_takes_the_float_operands_dtype(product):
    integers = mt.arange(4).reshape(2, 2)
    weights = mt.Tensor([[1.0, 0.5], [0.25, 1.0]])
    for left, right in ((integers, weights), (weights, integers)):
        got = product(left, right)
        want = product(left.astype("float32"), right.astype("float32"))
        assert got.dtype == "float32"
        assert got.tolist() == want.tolist()


def test_two_float_widths_are_still_refused_by_the_products():
    single = mt.Tensor([[1.0, 0.0], [0.0, 1.0]])
    with pytest.raises(TypeError, match="float64"):
        single @ single.astype("float64")
    with pytest.raises(TypeError, match="float64"):
        mt.solve(single, single.astype("float64"))


def test_the_gradient_reaches_the_float_operand_of_a_mixed_product():
    weights = mt.Tensor([[1.0], [2.0]], requires_grad=True)
    (mt.Tensor([[1, 2], [3, 4]]) @ weights).sum().backward()
    assert weights.grad.dtype == "float32"
    assert weights.grad.tolist() == [[4.0], [6.0]]


def test_solve_and_cross_take_an_integer_operand():
    system = mt.Tensor([[2, 0], [0, 4]])
    assert mt.solve(system, mt.Tensor([2.0, 4.0])).tolist() == [1.0, 1.0]
    crossed = mt.cross(mt.Tensor([[1, 0, 0]]), mt.Tensor([[0.0, 1.0, 0.0]]))
    assert crossed.dtype == "float32"
    assert crossed.tolist() == [[0.0, 0.0, 1.0]]
