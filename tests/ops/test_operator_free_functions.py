# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""The operators as free functions, and the spellings built on them.

`a + b` always worked; `mt.add(a, b)` and `a.add(b)` did not, and those are
what most code that moves between array libraries writes. They are one
definition each, so the first thing these tests establish is that the three
spellings -- top level, `functional`, method -- are the same object, and the
second is that each agrees with the operator or with NumPy.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import minitensor as mt
from minitensor import functional as F

A = np.array([[1.5, -2.25], [0.0, 4.0]])
B = np.array([[3.0, 0.5], [-1.0, -8.0]])


def _t(values, dtype="float64", requires_grad=False):
    return mt.Tensor(
        np.ascontiguousarray(np.asarray(values, dtype=np.float64)),
        dtype=dtype,
        requires_grad=requires_grad,
    )


NAMES = [
    "add",
    "sub",
    "mul",
    "div",
    "neg",
    "square",
    "deg2rad",
    "rad2deg",
    "lerp",
    "addcmul",
    "addcdiv",
    "float_power",
    "logaddexp2",
    "ldexp",
    "fmax",
    "fmin",
    "isposinf",
    "isneginf",
    "isreal",
    "signbit",
    "sgn",
    "absolute",
    "concat",
    "divide",
    "true_divide",
    "multiply",
    "subtract",
    "negative",
    "greater",
    "greater_equal",
    "less",
    "less_equal",
    "not_equal",
]


@pytest.mark.parametrize("name", NAMES)
def test_every_spelling_is_the_same_object(name):
    top = getattr(mt, name)
    assert getattr(F, name) is top
    # `concat` joins a list rather than taking a tensor first, so it is the one
    # name here that is not also a method.
    if name != "concat":
        assert hasattr(mt.Tensor, name), f"{name} is missing as a method"


# --- the operators ----------------------------------------------------------


@pytest.mark.parametrize(
    "name,operator",
    [
        ("add", lambda a, b: a + b),
        ("sub", lambda a, b: a - b),
        ("mul", lambda a, b: a * b),
        ("div", lambda a, b: a / b),
    ],
)
def test_the_free_function_is_the_operator(name, operator):
    got = getattr(mt, name)(_t(A), _t(B)).numpy()
    np.testing.assert_array_equal(got, operator(_t(A), _t(B)).numpy())
    np.testing.assert_array_equal(got, operator(A, B))
    # And so is the method.
    np.testing.assert_array_equal(getattr(_t(A), name)(_t(B)).numpy(), got)


def test_alpha_scales_the_second_operand():
    np.testing.assert_allclose(mt.add(_t(A), _t(B), alpha=2.5).numpy(), A + 2.5 * B)
    np.testing.assert_allclose(mt.sub(_t(A), _t(B), alpha=-0.5).numpy(), A + 0.5 * B)


def test_alpha_leaves_a_python_number_a_python_number():
    # A bare `0.1` has no dtype and is read at the width of the tensor it
    # meets. Converting it to a tensor to scale it would pick a width first,
    # and for a float64 tensor that costs the eighth digit.
    got = mt.add(_t([0.0]), 0.1, alpha=1).numpy()[0]
    assert got == 0.1
    scaled = mt.add(_t([0.0]), 0.1, alpha=3.0).numpy()[0]
    assert scaled == 0.1 * 3.0


@pytest.mark.parametrize("mode,reference", [("floor", np.floor), ("trunc", np.trunc)])
def test_div_rounds_the_quotient_when_asked(mode, reference):
    np.testing.assert_array_equal(
        mt.div(_t(A), _t(B), rounding_mode=mode).numpy(), reference(A / B)
    )


def test_the_two_rounding_modes_part_company_on_a_mixed_sign_quotient():
    positive = np.array([7.0])
    negative = np.array([-3.0])
    assert mt.div(_t(positive), _t(negative), "floor").numpy()[0] == -3.0
    assert mt.div(_t(positive), _t(negative), "trunc").numpy()[0] == -2.0


def test_div_reports_an_unknown_rounding_mode():
    with pytest.raises(ValueError, match="rounding_mode"):
        mt.div(_t(A), _t(B), rounding_mode="round")


def test_a_python_number_works_as_either_operand():
    np.testing.assert_array_equal(mt.mul(_t(A), 3.0).numpy(), A * 3.0)
    np.testing.assert_array_equal(mt.sub(2.0, _t(A)).numpy(), 2.0 - A)


# --- the built-on names -----------------------------------------------------


def test_square_is_a_product_and_is_exact():
    np.testing.assert_array_equal(mt.square(_t(A)).numpy(), A * A)
    # `pow(x, 2)` is not required to be exact for every input; a product is.
    awkward = np.array([1e-160, 3.0000000000000004, 7.0])
    np.testing.assert_array_equal(mt.square(_t(awkward)).numpy(), awkward * awkward)


def test_deg2rad_and_rad2deg_invert_each_other():
    degrees = np.array([0.0, 30.0, 90.0, -180.0, 360.0])
    radians = mt.deg2rad(_t(degrees)).numpy()
    np.testing.assert_allclose(radians, np.deg2rad(degrees), rtol=1e-15)
    np.testing.assert_allclose(mt.rad2deg(_t(radians)).numpy(), degrees, rtol=1e-14)


def test_lerp_returns_the_endpoints_exactly():
    start, end = _t(A), _t(B)
    np.testing.assert_array_equal(mt.lerp(start, end, 0.0).numpy(), A)
    np.testing.assert_array_equal(mt.lerp(start, end, 1.0).numpy(), B)
    np.testing.assert_allclose(mt.lerp(start, end, 0.25).numpy(), A + 0.25 * (B - A))


def test_addcmul_and_addcdiv_match_the_expression_they_name():
    np.testing.assert_allclose(
        mt.addcmul(_t(A), _t(B), _t(B), value=0.5).numpy(), A + 0.5 * B * B
    )
    np.testing.assert_allclose(
        mt.addcdiv(_t(A), _t(B), _t(B), value=2.0).numpy(), A + 2.0 * (B / B)
    )


def test_float_power_promotes_where_an_integer_power_would_wrap():
    integers = mt.Tensor(np.array([2, 3, 10], dtype=np.int64), dtype="int64")
    got = mt.float_power(integers, 30.0).numpy()
    assert got.dtype == np.float64
    np.testing.assert_allclose(got, np.float_power([2, 3, 10], 30.0), rtol=1e-14)
    # 10**30 is far past int64, and the float64 answer is the honest one.
    assert got[2] == pytest.approx(1e30, rel=1e-15)


def test_logaddexp2_matches_numpy():
    np.testing.assert_allclose(
        mt.logaddexp2(_t(A), _t(B)).numpy(), np.logaddexp2(A, B), rtol=1e-14
    )


def test_logaddexp2_survives_exponents_that_would_overflow():
    big = np.array([1020.0, -1020.0])
    other = np.array([1021.0, -1021.0])
    got = mt.logaddexp2(_t(big), _t(other)).numpy()
    assert np.isfinite(got).all()
    np.testing.assert_allclose(got, np.logaddexp2(big, other), rtol=1e-14)


def test_ldexp_scales_by_an_exact_power_of_two():
    values = np.array([1.5, -0.25, 3.0])
    exponents = np.array([3.0, 10.0, -4.0])
    np.testing.assert_array_equal(
        mt.ldexp(_t(values), _t(exponents)).numpy(),
        np.ldexp(values, exponents.astype(int)),
    )


def test_fmax_and_fmin_ignore_a_nan_that_maximum_would_propagate():
    left = np.array([np.nan, 1.0, np.nan, 3.0])
    right = np.array([2.0, np.nan, np.nan, -1.0])

    np.testing.assert_array_equal(
        mt.fmax(_t(left), _t(right)).numpy(), np.fmax(left, right)
    )
    np.testing.assert_array_equal(
        mt.fmin(_t(left), _t(right)).numpy(), np.fmin(left, right)
    )

    # And the contrast that makes the pair worth having.
    propagated = mt.maximum(_t(left), _t(right)).numpy()
    assert np.isnan(propagated[0]) and not np.isnan(np.fmax(left, right)[0])


def test_fmax_leaves_nan_only_where_both_operands_are_nan():
    got = mt.fmax(_t([np.nan]), _t([np.nan])).numpy()
    assert np.isnan(got[0]), "nothing to compare, so nothing to report"


def test_isposinf_and_isneginf_split_isinf():
    values = np.array([np.inf, -np.inf, 0.0, np.nan, 1e308])
    tensor = _t(values)
    np.testing.assert_array_equal(mt.isposinf(tensor).numpy(), np.isposinf(values))
    np.testing.assert_array_equal(mt.isneginf(tensor).numpy(), np.isneginf(values))
    np.testing.assert_array_equal(
        (mt.isposinf(tensor) | mt.isneginf(tensor)).numpy(), np.isinf(values)
    )


def test_isreal_is_true_everywhere_including_at_nan():
    values = np.array([1.0, np.nan, np.inf, -0.0])
    np.testing.assert_array_equal(mt.isreal(_t(values)).numpy(), np.isreal(values))
    assert mt.isreal(_t(values)).numpy().all()


def test_signbit_tells_the_two_zeros_apart_where_a_comparison_cannot():
    values = np.array([-0.0, 0.0, -1.0, 2.0, -np.inf, np.inf])
    np.testing.assert_array_equal(mt.signbit(_t(values)).numpy(), np.signbit(values))
    # The case the name exists for.
    assert mt.signbit(_t([-0.0])).numpy()[0]
    assert not (_t([-0.0]) < 0).numpy()[0]


def test_sgn_is_sign():
    np.testing.assert_array_equal(mt.sgn(_t(A)).numpy(), mt.sign(_t(A)).numpy())


# --- the aliases ------------------------------------------------------------


@pytest.mark.parametrize(
    "alias,target",
    [
        ("absolute", "abs"),
        ("concat", "cat"),
        ("divide", "div"),
        ("true_divide", "div"),
        ("multiply", "mul"),
        ("subtract", "sub"),
        ("negative", "neg"),
        ("greater", "gt"),
        ("greater_equal", "ge"),
        ("less", "lt"),
        ("less_equal", "le"),
        ("not_equal", "ne"),
    ],
)
def test_each_alias_is_the_object_it_names(alias, target):
    assert getattr(mt, alias) is getattr(mt, target)


def test_the_comparison_aliases_answer_what_the_operators_do():
    left, right = _t(A), _t(B)
    np.testing.assert_array_equal(mt.greater(left, right).numpy(), A > B)
    np.testing.assert_array_equal(mt.greater_equal(left, right).numpy(), A >= B)
    np.testing.assert_array_equal(mt.less(left, right).numpy(), A < B)
    np.testing.assert_array_equal(mt.less_equal(left, right).numpy(), A <= B)
    np.testing.assert_array_equal(mt.not_equal(left, right).numpy(), A != B)


# --- gradients --------------------------------------------------------------


def test_the_arithmetic_free_functions_carry_the_operators_gradients():
    for build, expected in (
        (lambda x, y: mt.add(x, y, alpha=3.0), (1.0, 3.0)),
        (lambda x, y: mt.sub(x, y, alpha=2.0), (1.0, -2.0)),
        (lambda x, y: mt.mul(x, y), None),
        (lambda x, y: mt.lerp(x, y, 0.25), (0.75, 0.25)),
    ):
        left = _t([2.0], requires_grad=True)
        right = _t([5.0], requires_grad=True)
        build(left, right).sum().backward()
        if expected is None:
            # d(xy)/dx is y and d(xy)/dy is x.
            assert left.grad.numpy()[0] == 5.0
            assert right.grad.numpy()[0] == 2.0
        else:
            assert left.grad.numpy()[0] == pytest.approx(expected[0])
            assert right.grad.numpy()[0] == pytest.approx(expected[1])
        mt.clear_autograd_graph()


def test_square_differentiates_to_twice_the_input():
    values = _t([1.5, -2.0, 0.0], requires_grad=True)
    mt.square(values).sum().backward()
    np.testing.assert_allclose(values.grad.numpy(), 2.0 * np.array([1.5, -2.0, 0.0]))


def test_the_angle_conversions_scale_the_gradient_by_the_same_factor():
    degrees = _t([90.0], requires_grad=True)
    mt.deg2rad(degrees).sum().backward()
    assert degrees.grad.numpy()[0] == pytest.approx(math.pi / 180.0, rel=1e-15)
