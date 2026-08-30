# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""`polygamma`, and the two orders that already had names.

`digamma` and `trigamma` were the zeroth and first derivatives of `lgamma`, and
there was no way to ask for the rest. `polygamma(n, x)` is all of them, and the
first thing to check is that the family has no seam in it: order 0 has to be
`digamma` and order 1 has to be `trigamma`, computed by whatever route, to the
last few bits.

Everything else is checked against a closed form or an identity rather than
against another implementation. `polygamma(n, 1)` is `(-1)^(n+1) n! zeta(n+1)`,
which for a large `n` is just `n!` -- and `n!` is exact in Python. The
duplication and recurrence formulas hold at every order and argument and are
independent of how the value was reached.

The high orders are the point of the arithmetic. `polygamma` is a factorial
times a zeta: at order 169 the factorial is `4e304` and the zeta at a large
argument is far below the smallest double, so multiplying them loses the answer
on the way to a value that fits perfectly well. Adding the two logarithms
instead is what makes order 169 work at all, and the test for it is the closed
form at 1 -- where the answer is `169!` and nothing else.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import minitensor as mt
from minitensor import functional as F

# zeta(3) and zeta(5), which have no closed form in pi.
ZETA_3 = 1.2020569031595942854
ZETA_5 = 1.0369277551433699263
ZETA = {
    2: math.pi**2 / 6,
    3: ZETA_3,
    4: math.pi**4 / 90,
    5: ZETA_5,
    6: math.pi**6 / 945,
}


def _t(values):
    return mt.Tensor(
        np.ascontiguousarray(np.asarray(values, dtype=np.float64)), dtype="float64"
    )


def _at(order, values):
    return F.polygamma(order, _t(values)).numpy()


# --- the family has no seam -------------------------------------------------


ARGUMENTS = [0.25, 0.5, 1.0, 2.5, 7.0, 40.0, 1e4]


def test_order_zero_is_digamma():
    np.testing.assert_allclose(
        _at(0, ARGUMENTS), F.digamma(_t(ARGUMENTS)).numpy(), rtol=0, atol=0
    )


def test_order_one_is_trigamma():
    """Checked against the closed forms, since `trigamma` has no public name."""

    np.testing.assert_allclose(
        _at(1, [1.0, 2.0, 0.5, -0.5, 100.0]),
        [
            ZETA[2],
            ZETA[2] - 1.0,
            math.pi**2 / 2,
            math.pi**2 / 2 + 4.0,
            0.010050166663333571,
        ],
        rtol=1e-14,
    )


# --- closed forms -----------------------------------------------------------


@pytest.mark.parametrize("order", [1, 2, 3, 4, 5])
def test_the_value_at_one_is_a_factorial_times_a_zeta(order):
    expected = (-1.0) ** (order + 1) * math.factorial(order) * ZETA[order + 1]
    assert float(_at(order, [1.0])[0]) == pytest.approx(expected, rel=1e-13)


@pytest.mark.parametrize("order", [1, 2, 3, 4, 5])
def test_the_value_at_a_half_carries_the_extra_power_of_two(order):
    """`polygamma(n, 1/2) = (-1)^(n+1) n! (2^(n+1) - 1) zeta(n+1)`."""

    expected = (
        (-1.0) ** (order + 1)
        * math.factorial(order)
        * (2.0 ** (order + 1) - 1.0)
        * ZETA[order + 1]
    )
    assert float(_at(order, [0.5])[0]) == pytest.approx(expected, rel=1e-13)


@pytest.mark.parametrize("order", [2, 3, 5, 8])
@pytest.mark.parametrize("x", [0.3, 1.7, 5.5, 30.0])
def test_the_recurrence_holds(order, x):
    """`polygamma(n, x + 1) - polygamma(n, x) = (-1)^n n! / x^(n+1)`."""

    term = math.factorial(order) / x ** (order + 1)
    stepped = float(_at(order, [x + 1.0])[0])
    expected = float(_at(order, [x])[0]) + (-1.0) ** order * term
    # The two sides cancel almost entirely for a small `x`, so what matters is
    # the error against the magnitude that cancelled.
    assert abs(stepped - expected) <= 1e-13 * max(abs(term), abs(stepped))


@pytest.mark.parametrize("order", [2, 4, 7])
@pytest.mark.parametrize("x", [0.4, 1.0, 3.25])
def test_the_duplication_formula_holds(order, x):
    """`polygamma(n, 2x) = (polygamma(n, x) + polygamma(n, x + 1/2)) / 2^(n+1)`."""

    halves = float(_at(order, [x])[0]) + float(_at(order, [x + 0.5])[0])
    assert float(_at(order, [2.0 * x])[0]) == pytest.approx(
        halves / 2.0 ** (order + 1), rel=1e-12
    )


# --- the orders the arithmetic exists for -----------------------------------


def _zeta_above_one(s, terms=400):
    """`zeta(s)` by summation, which converges at once for the `s` used here.

    The tail past `terms` is about `terms ** (1 - s) / (s - 1)`, which for the
    orders below is far under a double's precision. Summed upwards so the
    smallest contributions are not lost against the leading 1.
    """

    return 1.0 + sum(float(k) ** -float(s) for k in range(terms, 1, -1))


@pytest.mark.parametrize("order", [7, 40, 100, 150, 169])
def test_a_high_order_at_one_is_a_factorial_times_a_zeta(order):
    """`169!` is `4e304` and the answer is exactly that, to a digit or two.

    Nothing in the computation may form that factorial as a product with the
    zeta beside it; the two are combined as logarithms, which is what this
    tests by asking for an order where the product would not fit the way there.
    """

    expected = (
        (-1.0) ** (order + 1)
        * float(math.factorial(order))
        * _zeta_above_one(order + 1)
    )
    assert float(_at(order, [1.0])[0]) == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize("order,argument", [(40, 1e4), (100, 1e4), (169, 1e3)])
def test_a_high_order_at_a_large_argument_survives_the_scales_involved(
    order, argument
):
    """Where `n!` overflows one way and `x ** -(n + 1)` underflows the other.

    At order 100 and `x = 1e4` the answer is `9e-245` -- an ordinary double --
    while `100!` is `9e157` and `1e4 ** -101` is `1e-404`, which is not one.
    """

    value = float(_at(order, [argument])[0])
    assert math.isfinite(value) and value != 0.0
    # Leading term of the expansion: `(-1)^(n+1) (n-1)! x^-n`, in logs.
    expected_log = math.lgamma(order) - order * math.log(argument)
    assert math.log(abs(value)) == pytest.approx(expected_log, rel=1e-3)
    assert (value < 0) == (order % 2 == 0)


def test_an_answer_below_the_smallest_double_underflows_rather_than_failing():
    """`polygamma(169, 1e4)` is about `4e-373`, which no double holds."""

    assert float(_at(169, [1e4])[0]) == 0.0


# --- the domain -------------------------------------------------------------


@pytest.mark.parametrize("order", [2, 7])
@pytest.mark.parametrize("x", [-0.5, -1.0, -10.5, -1e6])
def test_above_order_one_the_negative_half_is_not_computed(order, x):
    """NaN rather than digits that are not there.

    Walking the recurrence up from a negative argument sums terms that are
    enormous beside the answer and alternate in sign; `scipy` stops at the same
    place, its `zeta(s, q)` being defined for positive `q` only.
    """

    assert math.isnan(float(_at(order, [x])[0]))


@pytest.mark.parametrize("order", [0, 1])
def test_the_two_named_orders_keep_the_whole_line(order):
    values = _at(order, [-0.5, -1.5, -10.5, -1e6 + 0.5])
    assert np.isfinite(values).all()


@pytest.mark.parametrize("order,sign", [(0, -1), (1, 1), (2, -1), (3, 1)])
def test_zero_is_a_pole_signed_by_the_limit_from_the_right(order, sign):
    value = float(_at(order, [0.0])[0])
    assert math.isinf(value) and (value > 0) == (sign > 0)


def test_nan_in_gives_nan_out():
    assert math.isnan(float(_at(3, [np.nan])[0]))


# --- the gradient -----------------------------------------------------------


@pytest.mark.parametrize("order", [0, 1, 2, 5])
def test_the_derivative_is_the_next_order(order):
    values = [0.4, 1.0, 2.5, 9.0]
    tracked = mt.Tensor(
        np.ascontiguousarray(values), dtype="float64", requires_grad=True
    )
    F.polygamma(order, tracked).sum().backward()
    np.testing.assert_allclose(
        tracked.grad.numpy(), _at(order + 1, values), rtol=1e-13
    )
    mt.clear_autograd_graph()


# --- dtypes and what it refuses ---------------------------------------------


def test_float32_stays_float32():
    single = mt.Tensor(np.array([1.0, 2.5], dtype=np.float32), dtype="float32")
    result = F.polygamma(2, single)
    assert "float32" in str(result.dtype)
    np.testing.assert_allclose(
        result.numpy(), _at(2, [1.0, 2.5]).astype(np.float32), rtol=1e-6
    )


@pytest.mark.parametrize("order", [-1, -5])
def test_a_negative_order_is_refused(order):
    with pytest.raises(ValueError, match="non-negative order"):
        F.polygamma(order, _t([1.0]))


def test_an_order_past_where_the_factorial_fits_is_refused():
    with pytest.raises(ValueError, match="up to order 169"):
        F.polygamma(170, _t([1.0]))
