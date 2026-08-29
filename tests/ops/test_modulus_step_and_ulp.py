# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""`fmod`, `heaviside` and `nextafter`, against NumPy.

`fmod` and `remainder` are the same computation with one correction step
between them, so what these tests pin is the step: where the two agree, where
they part, and that each reconstructs its own quotient. `heaviside` and
`nextafter` are pinned to NumPy's, including at the values -- signed zero, the
largest finite float, infinity -- where a plausible implementation goes wrong
without ever being noticed on ordinary input.
"""

from __future__ import annotations

import numpy as np
import pytest

import minitensor as mt

# Every sign pairing, which is the only place the two conventions differ.
DIVIDENDS = np.array([7.0, -7.0, 7.0, -7.0, 0.0, 7.5, -7.5])
DIVISORS = np.array([3.0, 3.0, -3.0, -3.0, 3.0, 2.5, 2.5])


def _t(values, dtype="float64", requires_grad=False):
    array = np.asarray(values)
    if dtype.startswith("float"):
        array = array.astype(np.float64 if dtype == "float64" else np.float32)
    return mt.Tensor(
        np.ascontiguousarray(array), dtype=dtype, requires_grad=requires_grad
    )


# --- fmod -------------------------------------------------------------------


def test_fmod_matches_numpy():
    np.testing.assert_array_equal(
        mt.fmod(_t(DIVIDENDS), _t(DIVISORS)).numpy(), np.fmod(DIVIDENDS, DIVISORS)
    )


def test_remainder_still_matches_numpy_after_sharing_a_body_with_fmod():
    np.testing.assert_array_equal(
        mt.remainder(_t(DIVIDENDS), _t(DIVISORS)).numpy(),
        np.remainder(DIVIDENDS, DIVISORS),
    )


def test_the_two_agree_when_the_operands_share_a_sign_and_not_otherwise():
    got_fmod = mt.fmod(_t(DIVIDENDS), _t(DIVISORS)).numpy()
    got_remainder = mt.remainder(_t(DIVIDENDS), _t(DIVISORS)).numpy()
    same_sign = np.sign(DIVIDENDS) * np.sign(DIVISORS) >= 0

    np.testing.assert_array_equal(got_fmod[same_sign], got_remainder[same_sign])

    # Opposite signs part them -- but only when the division leaves something
    # over. `-7.5 % 2.5` is exact, and a correction of zero is no correction.
    parts = ~same_sign & (got_fmod != 0.0)
    assert parts.any(), "the fixture has to contain a case that separates them"
    assert (got_fmod[parts] != got_remainder[parts]).all()
    np.testing.assert_array_equal(
        got_remainder[~same_sign & ~parts], got_fmod[~same_sign & ~parts]
    )


def test_each_convention_reconstructs_its_own_quotient():
    # The identity that defines them: `a == q * b + r`, floored for one and
    # truncated for the other.
    for name, rounding in (("fmod", np.trunc), ("remainder", np.floor)):
        r = getattr(mt, name)(_t(DIVIDENDS), _t(DIVISORS)).numpy()
        q = rounding(DIVIDENDS / DIVISORS)
        np.testing.assert_allclose(q * DIVISORS + r, DIVIDENDS, atol=1e-15)


@pytest.mark.parametrize("name", ["fmod", "remainder"])
def test_integer_operands_stay_integral(name):
    a = mt.Tensor(np.array([7, -7, 7, -7], dtype=np.int64), dtype="int64")
    b = mt.Tensor(np.array([3, 3, -3, -3], dtype=np.int64), dtype="int64")
    got = getattr(mt, name)(a, b)
    assert "int64" in str(got.dtype)
    reference = np.fmod if name == "fmod" else np.remainder
    np.testing.assert_array_equal(
        got.numpy(), reference(np.array([7, -7, 7, -7]), np.array([3, 3, -3, -3]))
    )


@pytest.mark.parametrize("name", ["fmod", "remainder"])
def test_an_integer_zero_divisor_is_refused(name):
    a = mt.Tensor(np.array([1], dtype=np.int64), dtype="int64")
    zero = mt.Tensor(np.array([0], dtype=np.int64), dtype="int64")
    with pytest.raises(Exception):
        getattr(mt, name)(a, zero)


@pytest.mark.parametrize(
    "name,rounding", [("fmod", np.trunc), ("remainder", np.floor)], ids=["fmod", "rem"]
)
def test_the_gradient_follows_each_convention_s_own_quotient(name, rounding):
    a = _t(DIVIDENDS, requires_grad=True)
    b = _t(DIVISORS, requires_grad=True)
    getattr(mt, name)(a, b).sum().backward()

    np.testing.assert_array_equal(a.grad.numpy(), np.ones_like(DIVIDENDS))
    np.testing.assert_allclose(
        b.grad.numpy(), -rounding(DIVIDENDS / DIVISORS), atol=1e-15
    )


def test_a_float_zero_divisor_gives_nan_rather_than_raising():
    # `%` itself gives NaN there, and so does NumPy.
    got = mt.fmod(_t([1.0, -1.0]), _t([0.0, 0.0])).numpy()
    assert np.isnan(got).all()


# --- heaviside --------------------------------------------------------------


def test_heaviside_matches_numpy():
    values = np.array([-2.0, -0.0, 0.0, 1e-300, 3.5, -np.inf, np.inf])
    at_zero = np.full_like(values, 0.5)
    np.testing.assert_array_equal(
        mt.heaviside(_t(values), _t(at_zero)).numpy(), np.heaviside(values, at_zero)
    )


def test_heaviside_takes_the_second_operand_at_zero_and_only_there():
    values = np.array([-1.0, 0.0, 1.0])
    for at_zero in (0.0, 0.5, 1.0, -3.0):
        got = mt.heaviside(_t(values), _t(np.full(3, at_zero))).numpy()
        np.testing.assert_array_equal(got, [0.0, at_zero, 1.0])


def test_a_nan_input_is_on_neither_side_of_the_step():
    got = mt.heaviside(_t([np.nan]), _t([0.5])).numpy()
    assert np.isnan(got[0])


def test_the_heaviside_gradient_reaches_the_value_at_zero_alone():
    values = _t([-1.0, 0.0, 2.0], requires_grad=True)
    at_zero = _t([0.5, 0.5, 0.5], requires_grad=True)
    mt.heaviside(values, at_zero).sum().backward()

    # The step is flat wherever it is defined.
    np.testing.assert_array_equal(values.grad.numpy(), [0.0, 0.0, 0.0])
    np.testing.assert_array_equal(at_zero.grad.numpy(), [0.0, 1.0, 0.0])


def test_heaviside_broadcasts_a_single_value_at_zero():
    values = np.array([[-1.0, 0.0], [0.0, 1.0]])
    np.testing.assert_array_equal(
        mt.heaviside(_t(values), _t([0.25])).numpy(),
        np.heaviside(values, 0.25),
    )


# --- nextafter --------------------------------------------------------------


def test_nextafter_matches_numpy():
    frm = np.array([1.0, 1.0, -1.0, -1.0, 0.0, -0.0, 3.0, 1e300, -1e-320])
    to = np.array([2.0, 0.0, 0.0, -2.0, 1.0, -1.0, 3.0, np.inf, 0.0])
    np.testing.assert_array_equal(
        mt.nextafter(_t(frm), _t(to)).numpy(), np.nextafter(frm, to)
    )


def test_nextafter_moves_exactly_one_representable_value():
    got = mt.nextafter(_t([1.0]), _t([2.0])).numpy()[0]
    assert got > 1.0
    # Nothing sits between: stepping back lands exactly where it started.
    assert np.nextafter(got, 0.0) == 1.0


def test_the_values_with_no_neighbour_of_their_own():
    largest = np.finfo(np.float64).max
    smallest_subnormal = np.nextafter(0.0, 1.0)
    frm = np.array([0.0, -0.0, 0.0, largest, np.inf, -np.inf])
    to = np.array([1.0, 1.0, -1.0, np.inf, 0.0, 0.0])
    got = mt.nextafter(_t(frm), _t(to)).numpy()

    # Neither zero has a neighbour a bit away; the smallest subnormal does.
    assert got[0] == smallest_subnormal
    assert got[1] == smallest_subnormal
    assert got[2] == -smallest_subnormal
    # Past the largest finite value there is only infinity, and back from
    # infinity there is only the largest finite value.
    assert got[3] == np.inf
    assert got[4] == largest
    assert got[5] == -largest


def test_nextafter_stays_put_when_it_is_already_there_and_gives_up_on_nan():
    np.testing.assert_array_equal(mt.nextafter(_t([2.5]), _t([2.5])).numpy(), [2.5])
    assert np.isnan(mt.nextafter(_t([np.nan]), _t([1.0])).numpy()[0])
    assert np.isnan(mt.nextafter(_t([1.0]), _t([np.nan])).numpy()[0])


def test_nextafter_steps_at_the_float32_ulp_when_the_tensor_is_float32():
    frm = np.array([1.0, -1.0, 0.0], dtype=np.float32)
    to = np.array([2.0, -2.0, 1.0], dtype=np.float32)
    got = mt.nextafter(_t(frm, "float32"), _t(to, "float32")).numpy()
    assert got.dtype == np.float32
    np.testing.assert_array_equal(got, np.nextafter(frm, to))
    # A float32 step is about 1e-7, eight orders coarser than a float64 one --
    # so the width really is the tensor's, not the widest available.
    assert got[0] - np.float32(1.0) == pytest.approx(1.19e-7, rel=1e-2)


def test_the_nextafter_gradient_is_the_identity_in_its_first_operand():
    frm = _t([1.0, 2.0], requires_grad=True)
    to = _t([5.0, 5.0], requires_grad=True)
    mt.nextafter(frm, to).sum().backward()

    np.testing.assert_array_equal(frm.grad.numpy(), [1.0, 1.0])
    # Only the *direction* of the second operand reaches the answer.
    np.testing.assert_array_equal(to.grad.numpy(), [0.0, 0.0])


# --- shared -----------------------------------------------------------------


@pytest.mark.parametrize("name", ["fmod", "heaviside", "nextafter"])
def test_the_functional_spelling_agrees_with_the_method(name):
    a = _t([1.0, -2.0, 3.0])
    b = _t([2.0, 2.0, 2.0])
    np.testing.assert_array_equal(
        getattr(mt, name)(a, b).numpy(), getattr(a, name)(b).numpy()
    )


@pytest.mark.parametrize("name", ["fmod", "heaviside", "nextafter"])
def test_a_python_number_is_accepted_as_the_second_operand(name):
    a = _t([1.0, -2.0, 3.0])
    np.testing.assert_array_equal(
        getattr(mt, name)(a, 2.0).numpy(), getattr(mt, name)(a, _t([2.0])).numpy()
    )
