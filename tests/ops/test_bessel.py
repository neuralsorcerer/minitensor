# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""The modified Bessel functions of the first kind, and their scaled forms.

`i0` and `i1` grow like `exp(x)`, so they overflow a double a little past 713 --
and the things they are wanted for do not. A Kaiser window is a ratio of two
`i0`s and a von Mises density divides by one, and in both the exponentials
cancel. `i0e` and `i1e` are `exp(-|x|)` times each, which is that cancellation
done before it can overflow rather than after.

Two series meet at thirty: a power series below, whose terms are all positive
so nothing cancels, and an asymptotic series above, taken to a fixed sixteen
terms. Fixed rather than "until the terms stop shrinking", which is the usual
rule and turns on the last bit of the argument -- two inputs a billionth apart
would come back differing in the tenth digit. The tests check the values
against the defining series computed independently here, and check that
crossing thirty changes nothing.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import minitensor as mt
from minitensor import functional as F


def _t(values):
    return mt.Tensor(
        np.ascontiguousarray(np.asarray(values, dtype=np.float64)), dtype="float64"
    )


def _reference(order, t, terms=4000):
    """`I_order(t)` by its defining series, in Python's own arithmetic."""

    half = t / 2.0
    term = half**order / math.factorial(order)
    total = term
    for k in range(1, terms):
        term *= half * half / (k * (k + order))
        total += term
        if term <= 1e-19 * total:
            break
    return total


ARGUMENTS = [0.0, 1e-8, 0.1, 1.0, 5.0, 29.9, 30.0, 30.1, 50.0, 100.0, -3.0, -40.0]


# --- values -----------------------------------------------------------------


@pytest.mark.parametrize("x", ARGUMENTS)
def test_the_scaled_forms_match_the_defining_series(x):
    scale = math.exp(-abs(x))
    sign = -1.0 if x < 0 else 1.0
    assert float(F.i0e(_t([x])).numpy()[0]) == pytest.approx(
        _reference(0, abs(x)) * scale, rel=1e-14
    )
    assert float(F.i1e(_t([x])).numpy()[0]) == pytest.approx(
        sign * _reference(1, abs(x)) * scale, rel=1e-14, abs=1e-300
    )


@pytest.mark.parametrize("x", [0.0, 0.1, 1.0, 5.0, 29.9, 30.1, 100.0, -3.0])
def test_the_plain_forms_are_the_scaled_ones_scaled_back(x):
    assert float(F.i0(_t([x])).numpy()[0]) == pytest.approx(
        _reference(0, abs(x)), rel=1e-13
    )
    assert float(F.i1(_t([x])).numpy()[0]) == pytest.approx(
        (-1.0 if x < 0 else 1.0) * _reference(1, abs(x)), rel=1e-13, abs=1e-300
    )


def test_the_values_at_the_origin():
    np.testing.assert_array_equal(F.i0(_t([0.0])).numpy(), [1.0])
    np.testing.assert_array_equal(F.i1(_t([0.0])).numpy(), [0.0])
    np.testing.assert_array_equal(F.i0e(_t([0.0])).numpy(), [1.0])
    np.testing.assert_array_equal(F.i1e(_t([0.0])).numpy(), [0.0])


def test_the_zeroth_is_even_and_the_first_is_odd():
    positive = [0.3, 2.0, 17.0, 45.0]
    negative = [-value for value in positive]
    np.testing.assert_array_equal(
        F.i0(_t(positive)).numpy(), F.i0(_t(negative)).numpy()
    )
    np.testing.assert_array_equal(
        F.i1(_t(positive)).numpy(), -F.i1(_t(negative)).numpy()
    )


def test_crossing_the_crossover_changes_nothing_but_the_slope():
    """Thirty is where the two series meet, and they have to meet smoothly."""

    below = float(F.i0e(_t([29.999999999])).numpy()[0])
    above = float(F.i0e(_t([30.000000001])).numpy()[0])
    slope = float(F.i1e(_t([30.0])).numpy()[0]) - float(F.i0e(_t([30.0])).numpy()[0])
    assert above - below == pytest.approx(2e-9 * slope, abs=1e-16)


# --- what the scaled forms are for ------------------------------------------


def test_the_scaled_forms_are_finite_where_the_plain_ones_are_not():
    assert math.isinf(float(F.i0(_t([750.0])).numpy()[0]))
    assert math.isinf(float(F.i1(_t([750.0])).numpy()[0]))
    scaled = float(F.i0e(_t([750.0])).numpy()[0])
    assert math.isfinite(scaled) and scaled > 0.0


@pytest.mark.parametrize("x", [100.0, 1e3, 1e6])
def test_the_scaled_forms_approach_the_same_limit_from_above(x):
    """`i0e(x)` and `i1e(x)` both tend to `1 / sqrt(2 pi x)`."""

    limit = 1.0 / math.sqrt(2.0 * math.pi * x)
    for name in ("i0e", "i1e"):
        value = float(getattr(F, name)(_t([x])).numpy()[0])
        assert value == pytest.approx(limit, rel=1e-2)


def test_a_kaiser_window_is_a_ratio_that_the_scaled_forms_make_computable():
    """The use case: at a large beta the ratio is fine and each half is not."""

    beta = 800.0
    positions = np.linspace(-1.0, 1.0, 9)
    argument = beta * np.sqrt(1.0 - positions**2)
    ratio = F.i0e(_t(argument)).numpy() / float(F.i0e(_t([beta])).numpy()[0])
    # `i0e` carries `exp(-|x|)`, so putting it back is what the ratio needs.
    window = ratio * np.exp(argument - beta)
    assert np.isfinite(window).all()
    assert window.max() == pytest.approx(1.0, rel=1e-12)
    assert window[0] < 1e-100 and window[-1] < 1e-100


# --- gradients --------------------------------------------------------------


@pytest.mark.parametrize("name", ["i0", "i1", "i0e", "i1e"])
def test_the_gradients_match_central_differences(name):
    values = np.array([0.3, 1.0, 5.0, 20.0, -2.0, -25.0])
    tracked = mt.Tensor(
        np.ascontiguousarray(values), dtype="float64", requires_grad=True
    )
    getattr(F, name)(tracked).sum().backward()
    analytic = tracked.grad.numpy()
    mt.clear_autograd_graph()

    step = 1e-6
    numeric = (
        getattr(F, name)(_t(values + step)).numpy()
        - getattr(F, name)(_t(values - step)).numpy()
    ) / (2 * step)
    np.testing.assert_allclose(analytic, numeric, rtol=2e-6)


def test_the_first_bessel_is_the_derivative_of_the_zeroth():
    values = [0.2, 1.5, 6.0, 35.0]
    tracked = mt.Tensor(
        np.ascontiguousarray(values), dtype="float64", requires_grad=True
    )
    F.i0(tracked).sum().backward()
    np.testing.assert_allclose(tracked.grad.numpy(), F.i1(_t(values)).numpy(), rtol=0)
    mt.clear_autograd_graph()


def test_the_gradient_at_the_origin_is_defined():
    """`i1(x)/x` is `0/0` there and its limit is `1/2`, which both gradients need."""

    origin = mt.Tensor(np.zeros(1), dtype="float64", requires_grad=True)
    F.i1(origin).sum().backward()
    assert float(origin.grad.item()) == pytest.approx(0.5, rel=1e-14)
    mt.clear_autograd_graph()

    origin = mt.Tensor(np.zeros(1), dtype="float64", requires_grad=True)
    F.i0(origin).sum().backward()
    assert float(origin.grad.item()) == 0.0
    mt.clear_autograd_graph()


# --- dtypes -----------------------------------------------------------------


@pytest.mark.parametrize("name", ["i0", "i1", "i0e", "i1e"])
def test_float32_stays_float32(name):
    single = mt.Tensor(np.array([0.5, 4.0], dtype=np.float32), dtype="float32")
    result = getattr(F, name)(single)
    assert "float32" in str(result.dtype)
    np.testing.assert_allclose(
        result.numpy(), getattr(F, name)(_t([0.5, 4.0])).numpy(), rtol=1e-6
    )
