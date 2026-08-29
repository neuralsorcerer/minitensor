# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""The special functions, against the references that define them.

`exp2`, `sinc`, `lgamma`, `digamma`, `erfinv` and `logit` are each pinned to an
independent implementation -- NumPy's or SciPy's where there is one, a closed
form where there is not -- rather than to a recorded output of this library,
which would only say that it still does what it did.

Two of them are worth more than a value check. `sinc` switches formula for its
gradient near zero, so the gradient is checked on both sides of that switch;
`digamma` is differentiated by a `trigamma` written here rather than taken from
a dependency, so its closed forms are checked directly.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import minitensor as mt


def _t(values, requires_grad=False):
    return mt.Tensor(
        np.array(values, dtype=np.float64), dtype="float64", requires_grad=requires_grad
    )


def _grad(build, values):
    """The gradient of `sum(build(x))` at `values`."""

    tensor = _t(values, requires_grad=True)
    build(tensor).sum().backward()
    return tensor.grad.numpy()


def _numeric_grad(build, values, eps=1e-6):
    base = np.asarray(values, dtype=np.float64)
    out = np.zeros_like(base)
    for i in range(base.size):
        up, down = base.copy(), base.copy()
        up[i] += eps
        down[i] -= eps
        out[i] = (build(_t(up)).sum().item() - build(_t(down)).sum().item()) / (2 * eps)
    return out


# --- exp2 -------------------------------------------------------------------


def test_exp2_matches_two_to_the_power():
    values = [-30.5, -1.0, 0.0, 0.5, 1.0, 17.25, 60.0]
    np.testing.assert_allclose(
        mt.exp2(_t(values)).numpy(), np.exp2(values), rtol=1e-15, atol=0.0
    )


def test_exp2_is_exact_on_the_integers_where_exp_of_a_product_is_not():
    # The whole reason for a dedicated kernel: `exp(n * log 2)` rounds the
    # product before exponentiating, and the answer drifts off the exact power
    # of two that `2 ** n` is.
    powers = np.arange(-60.0, 61.0)
    got = mt.exp2(_t(powers)).numpy()
    np.testing.assert_array_equal(got, np.exp2(powers))
    assert not np.array_equal(got, np.exp(powers * np.log(2.0)))


def test_exp2_gradient_is_itself_times_log_two():
    values = [-2.0, 0.0, 3.5]
    np.testing.assert_allclose(
        _grad(lambda t: t.exp2(), values),
        np.exp2(values) * np.log(2.0),
        rtol=1e-14,
    )


# --- sinc -------------------------------------------------------------------


def test_sinc_matches_numpy():
    values = np.linspace(-4.0, 4.0, 41)
    np.testing.assert_allclose(mt.sinc(_t(values)).numpy(), np.sinc(values), atol=1e-15)


def test_sinc_is_one_at_zero_and_zero_at_the_other_integers():
    got = mt.sinc(_t([0.0, 1.0, -1.0, 2.0, -5.0])).numpy()
    assert got[0] == 1.0
    np.testing.assert_allclose(got[1:], 0.0, atol=1e-15)


@pytest.mark.parametrize(
    "values",
    [
        pytest.param([-2.3, -1.1, 0.6, 1.9, 3.4], id="away_from_zero"),
        # 0.1 / pi is where the gradient switches from the quotient to the
        # series, so these straddle it.
        pytest.param([-0.05, -0.02, 0.02, 0.05], id="across_the_series_cutoff"),
    ],
)
def test_sinc_gradient_matches_central_differences(values):
    np.testing.assert_allclose(
        _grad(lambda t: t.sinc(), values),
        _numeric_grad(lambda t: t.sinc(), values),
        rtol=1e-7,
        atol=1e-9,
    )


def test_the_sinc_gradient_is_zero_at_the_origin():
    # The quotient is 0/0 there; the series is not, and it says zero, which is
    # what the symmetry of an even function requires.
    assert _grad(lambda t: t.sinc(), [0.0])[0] == 0.0


# --- lgamma and digamma -----------------------------------------------------

# gamma(n) = (n-1)!, and gamma(1/2) = sqrt(pi); both give exact references
# without a dependency on SciPy.
FACTORIAL_POINTS = [(float(n), math.lgamma(n)) for n in range(1, 15)]


@pytest.mark.parametrize("value,expected", FACTORIAL_POINTS, ids=lambda x: str(x))
def test_lgamma_matches_the_standard_library(value, expected):
    assert mt.lgamma(_t([value])).numpy()[0] == pytest.approx(expected, abs=1e-13)


def test_lgamma_stays_finite_where_gamma_overflows():
    # gamma(200) is about 1e372, past the top of float64; its logarithm is not.
    got = mt.lgamma(_t([200.0, 1000.0])).numpy()
    assert np.isfinite(got).all()
    np.testing.assert_allclose(
        got, [math.lgamma(200.0), math.lgamma(1000.0)], rtol=1e-14
    )


def test_lgamma_is_infinite_at_the_poles_and_defined_between_them():
    got = mt.lgamma(_t([0.0, -1.0, -2.0, -0.5, -1.5])).numpy()
    assert np.isposinf(got[:3]).all(), "the non-positive integers are poles"
    # gamma(-1/2) = -2 sqrt(pi), gamma(-3/2) = 4 sqrt(pi) / 3.
    np.testing.assert_allclose(
        got[3:],
        [math.log(2 * math.sqrt(math.pi)), math.log(4 * math.sqrt(math.pi) / 3)],
        rtol=1e-13,
    )


def test_digamma_matches_its_closed_forms():
    euler = 0.5772156649015328606
    got = mt.digamma(_t([1.0, 2.0, 3.0, 0.5])).numpy()
    np.testing.assert_allclose(
        got,
        [
            -euler,
            1.0 - euler,  # psi(x + 1) = psi(x) + 1/x
            1.5 - euler,
            -euler - 2 * math.log(2.0),
        ],
        rtol=1e-13,
    )


def test_digamma_satisfies_its_own_recurrence():
    # psi(x + 1) - psi(x) = 1/x, at points the closed forms do not cover.
    values = np.array([0.3, 1.7, 4.25, 11.5, 40.0])
    step = mt.digamma(_t(values + 1.0)).numpy() - mt.digamma(_t(values)).numpy()
    np.testing.assert_allclose(step, 1.0 / values, rtol=1e-12)


def test_lgamma_differentiates_to_digamma():
    values = [0.4, 1.0, 2.5, 8.0]
    np.testing.assert_allclose(
        _grad(lambda t: t.lgamma(), values),
        mt.digamma(_t(values)).numpy(),
        rtol=1e-14,
    )


def test_digamma_differentiates_to_trigamma():
    # trigamma has closed forms of its own at these points: psi'(1) = pi^2/6,
    # psi'(1/2) = pi^2/2, and the recurrence psi'(x+1) = psi'(x) - 1/x^2.
    sixth = math.pi**2 / 6
    half = math.pi**2 / 2
    values = [1.0, 2.0, 3.0, 0.5, 1.5]
    expected = [sixth, sixth - 1.0, sixth - 1.0 - 0.25, half, half - 4.0]
    np.testing.assert_allclose(
        _grad(lambda t: t.digamma(), values), expected, rtol=1e-12
    )


# --- erfinv -----------------------------------------------------------------


def test_erfinv_inverts_erf():
    values = np.linspace(-0.995, 0.995, 41)
    round_trip = mt.erf(_t(mt.erfinv(_t(values)).numpy())).numpy()
    np.testing.assert_allclose(round_trip, values, atol=1e-13)


def test_erfinv_is_infinite_at_the_endpoints_and_nan_past_them():
    got = mt.erfinv(_t([-1.0, 1.0, -1.5, 2.0])).numpy()
    assert got[0] == -np.inf and got[1] == np.inf
    assert np.isnan(got[2:]).all(), "erf never reaches past 1"


def test_erfinv_gradient_matches_central_differences():
    values = [-0.9, -0.3, 0.0, 0.45, 0.88]
    np.testing.assert_allclose(
        _grad(lambda t: t.erfinv(), values),
        _numeric_grad(lambda t: t.erfinv(), values),
        rtol=1e-6,
    )


# --- logit ------------------------------------------------------------------


def test_logit_inverts_sigmoid():
    probabilities = np.array([1e-6, 0.01, 0.25, 0.5, 0.9, 1 - 1e-6])
    logits = mt.logit(_t(probabilities)).numpy()
    np.testing.assert_allclose(logits, np.log(probabilities / (1 - probabilities)))
    np.testing.assert_allclose(
        mt.sigmoid(_t(logits)).numpy(), probabilities, rtol=1e-13
    )


def test_logit_without_an_eps_saturates_and_then_gives_up():
    got = mt.logit(_t([0.0, 1.0, -0.25, 1.25])).numpy()
    assert got[0] == -np.inf and got[1] == np.inf
    assert np.isnan(got[2:]).all()


def test_an_eps_bounds_the_answer_and_flattens_the_gradient():
    eps = 1e-4
    got = mt.logit(_t([0.0, 1.0, 0.5]), eps).numpy()
    assert np.isfinite(got[:2]).all()
    assert got[0] == pytest.approx(math.log(eps / (1 - eps)), rel=1e-12)
    assert got[1] == pytest.approx(-got[0], rel=1e-9)
    assert got[2] == 0.0

    grad = _grad(lambda t: t.logit(eps), [0.0, 0.5, 1.0])
    assert grad[0] == 0.0 and grad[2] == 0.0, "the clamp makes the output constant"
    assert grad[1] == pytest.approx(4.0, rel=1e-12), "1 / (0.5 * 0.5)"


@pytest.mark.parametrize("eps", [-1e-9, 0.5 + 1e-9, 1.0, float("nan")])
def test_an_eps_outside_zero_to_a_half_is_refused(eps):
    # A clamp to `[eps, 1 - eps]` past a half has an empty interval, and the
    # answer for it would be whatever the clamp happened to do with inverted
    # bounds.
    with pytest.raises(Exception, match=r"eps in \[0, 0\.5\]"):
        mt.logit(_t([0.5]), eps)


# --- shared -----------------------------------------------------------------

UNARY = ["exp2", "sinc", "lgamma", "digamma", "erfinv", "logit"]


@pytest.mark.parametrize("name", UNARY)
def test_the_functional_spelling_agrees_with_the_method(name):
    values = _t([0.25, 0.5, 0.75])
    np.testing.assert_array_equal(
        getattr(mt, name)(values).numpy(), getattr(values, name)().numpy()
    )


@pytest.mark.parametrize("name", UNARY)
def test_an_integer_tensor_is_refused_by_name(name):
    integers = mt.Tensor(np.array([1, 2], dtype=np.int64), dtype="int64")
    with pytest.raises(Exception, match=name):
        getattr(mt, name)(integers)


@pytest.mark.parametrize("name", UNARY)
def test_float32_agrees_with_float64(name):
    values = np.array([0.125, 0.375, 0.625], dtype=np.float32)
    single = getattr(mt, name)(mt.Tensor(values.copy(), dtype="float32")).numpy()
    double = getattr(mt, name)(
        mt.Tensor(values.astype(np.float64), dtype="float64")
    ).numpy()
    assert single.dtype == np.float32
    np.testing.assert_allclose(single, double.astype(np.float32), rtol=1e-6)
