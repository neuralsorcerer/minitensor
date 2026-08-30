# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""`erfcx`: `exp(x**2) erfc(x)`, where the product cannot be formed.

`erfc` underflows to zero a little past 26 and `exp(x**2)` overflows a little
past 26.6, so above there the product is `inf * 0`. The value it is reaching
for is an ordinary number -- `erfcx(30)` is `0.0188` and `erfcx(1e100)` is
`5.6e-101` -- and every Gaussian tail computation that divides one by another
needs it: a truncated normal's density, a probit likelihood, a Mills ratio.

The tests hold it to the product wherever the product exists, which is most of
the range, and to `1/(x sqrt(pi))` where it does not. The derivative has a
closed form that involves the function itself -- `2 x erfcx(x) - 2/sqrt(pi)`,
where the constant is constant because `erfc`'s own derivative cancels the
scaling exactly -- so it is checked against that and against central
differences.
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


def _at(values):
    return F.erfcx(_t(values)).numpy()


FORMABLE = [0.0, 1e-8, 0.1, 0.5, 1.0, 3.0, 7.0, 7.999, 8.0, 8.001, 15.0, 26.0]


# --- against the product ----------------------------------------------------


@pytest.mark.parametrize("x", FORMABLE + [-0.5, -2.0, -5.0, -20.0])
def test_it_is_the_product_wherever_the_product_exists(x):
    product = math.exp(x * x) * math.erfc(x)
    assert float(_at([x])[0]) == pytest.approx(product, rel=1e-13)


def test_the_value_at_the_origin_is_one():
    assert float(_at([0.0])[0]) == 1.0


def test_crossing_the_crossover_changes_nothing_but_the_slope():
    """Eight is where the product gives way to the asymptotic series."""

    below = float(_at([7.999999999])[0])
    above = float(_at([8.000000001])[0])
    slope = 2.0 * 8.0 * float(_at([8.0])[0]) - 2.0 / math.sqrt(math.pi)
    # The two branches are different mathematics; across the two-billionths
    # between these points the only thing that may change is the slope.
    assert above - below == pytest.approx(2e-9 * slope, rel=1e-6)


# --- where the product cannot be formed -------------------------------------


def test_the_product_really_cannot_be_formed_up_there():
    """The premise, stated as a test so it is not just an assertion in prose."""

    assert math.erfc(30.0) == 0.0
    # And the other half overflows, which Python's `exp` reports by refusing.
    with pytest.raises(OverflowError):
        math.exp(30.0 * 30.0)


@pytest.mark.parametrize(
    "x,expected",
    [
        (30.0, 0.018795888861416758),
        (50.0, 0.011281536265323773),
        (100.0, 0.005641613782989433),
    ],
)
def test_it_is_finite_and_right_where_the_product_is_not(x, expected):
    assert float(_at([x])[0]) == pytest.approx(expected, rel=1e-13)


@pytest.mark.parametrize("x", [1e2, 1e4, 1e100, 1e300])
def test_it_approaches_one_over_x_root_pi(x):
    assert float(_at([x])[0]) == pytest.approx(
        1.0 / (x * math.sqrt(math.pi)), rel=1e-3
    )


def test_a_mills_ratio_is_computable_at_a_depth_erfc_cannot_reach():
    """The use case: the tail's density over its mass, at 40 sigma.

    `phi(z) / (1 - Phi(z))` is `sqrt(2/pi) / erfcx(z / sqrt(2))`, and the two
    halves it is written from have both left the range of a double by then.
    """

    z = 40.0
    ratio = math.sqrt(2.0 / math.pi) / float(_at([z / math.sqrt(2.0)])[0])
    # The tail's hazard rate approaches `z` itself from above.
    assert ratio == pytest.approx(z, rel=1e-3)
    assert ratio > z


# --- the negative half ------------------------------------------------------


def test_the_negative_half_grows_like_twice_the_exponential():
    for x in (-5.0, -10.0, -20.0):
        assert float(_at([x])[0]) == pytest.approx(2.0 * math.exp(x * x), rel=1e-8)


def test_it_overflows_only_where_its_value_does():
    assert math.isfinite(float(_at([-26.0])[0]))
    assert math.isinf(float(_at([-30.0])[0]))


def test_nan_in_gives_nan_out():
    assert math.isnan(float(_at([np.nan])[0]))


# --- the gradient -----------------------------------------------------------


def test_the_gradient_is_its_own_closed_form():
    values = [0.0, 0.5, 3.0, 8.5, 20.0, -2.0]
    tracked = mt.Tensor(
        np.ascontiguousarray(values), dtype="float64", requires_grad=True
    )
    F.erfcx(tracked).sum().backward()
    expected = 2.0 * np.asarray(values) * _at(values) - 2.0 / math.sqrt(math.pi)
    np.testing.assert_allclose(tracked.grad.numpy(), expected, rtol=1e-14)
    mt.clear_autograd_graph()


def test_the_gradient_matches_central_differences():
    values = np.array([0.3, 1.0, 5.0, 12.0, -2.0])
    tracked = mt.Tensor(
        np.ascontiguousarray(values), dtype="float64", requires_grad=True
    )
    F.erfcx(tracked).sum().backward()
    analytic = tracked.grad.numpy()
    mt.clear_autograd_graph()

    step = 1e-6
    numeric = (_at(values + step) - _at(values - step)) / (2 * step)
    np.testing.assert_allclose(analytic, numeric, rtol=2e-6)


def test_the_gradient_at_the_origin_is_minus_two_over_root_pi():
    origin = mt.Tensor(np.zeros(1), dtype="float64", requires_grad=True)
    F.erfcx(origin).sum().backward()
    assert float(origin.grad.item()) == pytest.approx(-2.0 / math.sqrt(math.pi))
    mt.clear_autograd_graph()


# --- dtypes -----------------------------------------------------------------


def test_float32_stays_float32():
    single = mt.Tensor(np.array([0.5, 9.0], dtype=np.float32), dtype="float32")
    result = F.erfcx(single)
    assert "float32" in str(result.dtype)
    np.testing.assert_allclose(result.numpy(), _at([0.5, 9.0]), rtol=1e-6)
