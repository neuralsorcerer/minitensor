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


# Every one of these answers in a float whatever its operands were, so an
# integer pair promotes the way `/` does rather than being refused. `logaddexp`
# was promoting like `+` instead, landing two integers on an integer dtype that
# its own kernel then had to reject.
_FLOAT_VALUED_BINARIES = [
    ("logaddexp", np.logaddexp),
    ("logaddexp2", np.logaddexp2),
    ("hypot", np.hypot),
    ("atan2", np.arctan2),
]


@pytest.mark.parametrize("dtype", ["int32", "int64", "bool"])
@pytest.mark.parametrize(
    "name,reference", _FLOAT_VALUED_BINARIES, ids=[c[0] for c in _FLOAT_VALUED_BINARIES]
)
def test_a_float_valued_binary_takes_an_integer_pair(name, reference, dtype):
    if dtype == "bool":
        values = np.array([True, False, True])
        other = np.array([True, True, False])
    else:
        values = np.array([1, 2, 3], dtype=dtype)
        other = np.array([3, 2, 1], dtype=dtype)

    result = getattr(mt, name)(
        mt.Tensor(values, dtype=dtype), mt.Tensor(other, dtype=dtype)
    )
    # Two integers promote to float32, as they do for `/`; NumPy would widen
    # further, which is the documented difference and not this one. The
    # reference is taken in float64 for the same reason -- NumPy computes a
    # bool pair in float16, which is coarser than the answer being checked.
    assert result.dtype == "float32"
    np.testing.assert_allclose(
        result.numpy(),
        reference(values.astype(np.float64), other.astype(np.float64)),
        rtol=1e-6,
    )


@pytest.mark.parametrize(
    "name,reference", _FLOAT_VALUED_BINARIES, ids=[c[0] for c in _FLOAT_VALUED_BINARIES]
)
def test_a_float_valued_binary_lets_a_float64_operand_pull_the_result_up(
    name, reference
):
    integers = np.array([1, 2, 3], dtype=np.int64)
    doubles = np.array([3.0, 2.0, 1.0])
    result = getattr(mt, name)(
        mt.Tensor(integers, dtype="int64"), mt.Tensor(doubles, dtype="float64")
    )
    assert result.dtype == "float64"
    np.testing.assert_allclose(result.numpy(), reference(integers, doubles), rtol=1e-14)


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


@pytest.mark.parametrize("dtype", ["float32", "float64", "int32", "int64"])
def test_signbit_reads_the_bit_on_every_dtype_that_has_one(dtype):
    """The float arms read the sign bit and the integer arms compare, because
    an integer has no bit to read that is not its sign. NaN carries a sign like
    anything else, and the answer is the bit rather than any ordering, so a
    negative NaN is negative here where it is nothing at all to `<`.
    """

    if dtype.startswith("float"):
        values = np.array(
            [-0.0, 0.0, -1.0, 2.0, -np.inf, np.inf, np.nan, -np.nan], dtype=dtype
        )
    else:
        values = np.array([-2, -1, 0, 1, 3], dtype=dtype)
    np.testing.assert_array_equal(
        mt.signbit(mt.from_numpy(values)).numpy(), np.signbit(values)
    )


def test_signbit_of_a_bool_is_false_and_of_nothing_is_nothing():
    """A `bool` has no sign at all, and an empty tensor has nothing to have
    one -- both of which the kernel has to answer rather than reach past."""

    np.testing.assert_array_equal(
        mt.signbit(mt.from_numpy(np.array([True, False]))).numpy(),
        np.array([False, False]),
    )
    assert mt.signbit(mt.from_numpy(np.array([], dtype=np.float64))).numel() == 0
    scalar = mt.signbit(mt.from_numpy(np.array(-3.0)))
    assert tuple(scalar.shape) == () and bool(scalar.numpy())


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


# `fmax`/`fmin` were five passes of Python -- two `isnan`, a `maximum` and two
# `where` -- and are one kernel now. Two things had to survive that: the
# gradient, which the arrangement decided implicitly, and every NaN case. One
# thing deliberately did not.
FMAX_FMIN_EDGES = [
    (-0.0, 0.0),
    (0.0, -0.0),
    (-0.0, -0.0),
    (0.0, 0.0),
    (np.nan, 1.0),
    (1.0, np.nan),
    (np.nan, np.nan),
    (np.inf, 1.0),
    (-np.inf, np.nan),
    (np.nan, -np.inf),
    (1.0, 1.0),
    (2.0, 1.0),
    (1.0, 2.0),
]


@pytest.mark.parametrize("name", ["fmax", "fmin"])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_fmax_and_fmin_match_numpy_on_every_value(name, dtype):
    left = np.array([a for a, _ in FMAX_FMIN_EDGES], dtype=dtype)
    right = np.array([b for _, b in FMAX_FMIN_EDGES], dtype=dtype)
    np.testing.assert_array_equal(
        getattr(mt, name)(mt.from_numpy(left), mt.from_numpy(right)).numpy(),
        getattr(np, name)(left, right),
    )


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("size", [1, 2, 3, 4, 8, 17, 64, 1000])
def test_the_signed_zero_tie_is_decided_the_same_way_every_time(dtype, size):
    # `-0.0` and `+0.0` compare equal, so either is "the larger", and there is
    # no reference to copy: `np.fmax` over an array of `(-0.0, +0.0)` pairs
    # answers `+0.0` for the elements its vectorized body handles and `-0.0`
    # for the ones left to its scalar tail, so the same two operands give
    # different signs at different lengths and, at some lengths, within one
    # array. Asserting against NumPy here fails on NumPy's own inconsistency.
    #
    # The rule is IEEE 754-2019's `maximumNumber` instead: `-0.0` sits below
    # `+0.0`, which is the order the rest of this library sorts them in, and
    # the answer does not depend on where in a buffer the operands sat.
    minus = np.full(size, -0.0, dtype=dtype)
    plus = np.full(size, 0.0, dtype=dtype)

    for left, right in ((minus, plus), (plus, minus)):
        assert not np.signbit(
            mt.fmax(mt.from_numpy(left), mt.from_numpy(right)).numpy()
        ).any(), "fmax(-0.0, +0.0) must be +0.0 at every length and position"
        assert np.signbit(
            mt.fmin(mt.from_numpy(left), mt.from_numpy(right)).numpy()
        ).all(), "fmin(-0.0, +0.0) must be -0.0 at every length and position"

    # Two of a kind keep their own sign, in both directions.
    for filled, negative in ((minus, True), (plus, False)):
        for name in ("fmax", "fmin"):
            got = getattr(mt, name)(mt.from_numpy(filled), mt.from_numpy(filled))
            assert bool(np.signbit(got.numpy()).all()) is negative


@pytest.mark.parametrize("name", ["fmax", "fmin"])
def test_the_gradient_goes_to_the_first_operand_at_a_tie(name):
    # The value ties to the second operand and the derivative to the first.
    # That is not an inconsistency -- at a tie the two values are equal, so
    # which is "returned" is unobservable, while the derivative has to pick a
    # side. These are the sides the five passes picked.
    left = mt.Tensor(
        np.array([a for a, _ in FMAX_FMIN_EDGES]), dtype="float64", requires_grad=True
    )
    right = mt.Tensor(
        np.array([b for _, b in FMAX_FMIN_EDGES]), dtype="float64", requires_grad=True
    )
    getattr(mt, name)(left, right).sum().backward()

    other = "fmin" if name == "fmax" else "fmax"
    del other
    for i, (a, b) in enumerate(FMAX_FMIN_EDGES):
        if a != a:  # a NaN operand is never the answer unless both are
            want_left = 0.0
        elif b != b or (a >= b if name == "fmax" else a <= b):
            want_left = 1.0
        else:
            want_left = 0.0
        assert left.grad.numpy()[i] == want_left, f"d/dleft at {(a, b)}"
        assert right.grad.numpy()[i] == 1.0 - want_left, f"d/dright at {(a, b)}"


@pytest.mark.parametrize("name", ["fmax", "fmin"])
@pytest.mark.parametrize("size", [1, 7, 8, 1023, 100_000])
def test_fmax_and_fmin_agree_with_numpy_over_noisy_data(name, size):
    # A fifth of each operand NaN, so every combination of NaN and number
    # occurs many times and at every alignment of the block loop.
    rng = np.random.default_rng(size)
    left = rng.standard_normal(size)
    right = rng.standard_normal(size)
    left[rng.random(size) < 0.2] = np.nan
    right[rng.random(size) < 0.2] = np.nan
    got = getattr(mt, name)(_t(left), _t(right)).numpy()
    np.testing.assert_array_equal(got, getattr(np, name)(left, right))


@pytest.mark.parametrize("name", ["fmax", "fmin"])
def test_fmax_and_fmin_broadcast_and_promote(name):
    # They are declared beside `atan2` and `hypot` now, so they inherit that
    # family's broadcast.
    rows = np.array([[1.0, np.nan], [3.0, 4.0]])
    column = np.array([[2.0], [np.nan]])
    np.testing.assert_array_equal(
        getattr(mt, name)(_t(rows), _t(column)).numpy(),
        getattr(np, name)(rows, column),
    )

    # They do not inherit its promotion. An integer pair has no NaN to skip, so
    # it is a plain `maximum` and keeps its dtype, where this family turns two
    # integers into a float the way `/` does -- and NumPy's `fmax` of two
    # integers is an integer.
    left = np.array([1, 5], dtype=np.int64)
    right = np.array([3, 2], dtype=np.int64)
    result = getattr(mt, name)(mt.from_numpy(left), mt.from_numpy(right))
    assert "int64" in str(result.dtype)
    np.testing.assert_array_equal(result.numpy(), getattr(np, name)(left, right))


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_maximum_and_minimum_keep_their_rule_on_the_vectorised_path(dtype):
    """`maximum` and `minimum` reach the same-shape SIMD kernel that every
    other float binary op reaches, and the branchy NaN test that used to sit
    in the loop became two selects to get there.

    The rule that has to survive is not just "propagates NaN". It is which
    operand comes back when both are NaN, and which comes back on a tie --
    `a`, in both cases, which for signed zeros is observable. So this compares
    bit patterns rather than values, over every pair of the interesting
    floats, and at a length past the parallel threshold so the blocked path is
    the one under test.
    """
    specials = [np.nan, np.inf, -np.inf, 0.0, -0.0, 1.0, -1.0, 3.5, -2.25]
    pairs = [(a, b) for a in specials for b in specials]
    reps = -(-70_000 // len(pairs))
    left = np.array([a for a, _ in pairs] * reps, dtype=dtype)
    right = np.array([b for _, b in pairs] * reps, dtype=dtype)
    bits = np.uint32 if dtype == "float32" else np.uint64

    for name, ahead in (("maximum", True), ("minimum", False)):
        got = getattr(mt, name)(mt.from_numpy(left), mt.from_numpy(right)).numpy()
        want = np.empty_like(got)
        for i, (a, b) in enumerate(zip(left, right)):
            if np.isnan(a):
                want[i] = a
            elif np.isnan(b):
                want[i] = b
            elif (a >= b) if ahead else (a <= b):
                want[i] = a
            else:
                want[i] = b
        np.testing.assert_array_equal(
            got.view(bits), want.view(bits), err_msg=f"{name} {dtype}"
        )

        # NumPy agrees everywhere a NaN is not the answer, and agrees on
        # where the NaNs are.
        reference = getattr(np, name)(left, right)
        finite = ~(np.isnan(reference) & np.isnan(got))
        np.testing.assert_array_equal(reference[finite], got[finite])
        np.testing.assert_array_equal(np.isnan(reference), np.isnan(got))


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_maximum_and_minimum_agree_between_the_two_paths(dtype):
    """Only same-shape operands take the vectorised kernel; a broadcast falls
    through to the elementwise map. Both spell the rule with the same
    function, and this is what says so -- the broadcast result against the
    same product materialised first.
    """
    rng = np.random.default_rng(3)
    column = rng.standard_normal((257, 1)).astype(dtype)
    row = rng.standard_normal((1, 129)).astype(dtype)
    column[0, 0] = np.nan
    row[0, 1] = np.nan
    wide_column = np.repeat(column, 129, axis=1)
    wide_row = np.repeat(row, 257, axis=0)
    bits = np.uint32 if dtype == "float32" else np.uint64

    for name in ("maximum", "minimum"):
        op = getattr(mt, name)
        broadcast = op(mt.from_numpy(column), mt.from_numpy(row)).numpy()
        same_shape = op(mt.from_numpy(wide_column), mt.from_numpy(wide_row)).numpy()
        np.testing.assert_array_equal(
            broadcast.view(bits), same_shape.view(bits), err_msg=name
        )
