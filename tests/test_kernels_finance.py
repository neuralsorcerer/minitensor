# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""`kernels.black_scholes`, against the closed form and against its own Greeks.

The kernel is native and fused, so three things are worth checking and one is
not. Not worth checking: that a normal CDF works. Worth checking: that the
price is the formula, that the five partial derivatives the backward writes
down really are the derivatives of that price, and that the degenerate corner
-- zero volatility or zero time, where the formula is `0 * inf` -- returns the
limit rather than a NaN.

The Greeks are checked twice over. Against central differences, which catches a
wrong expression; and delta against `Phi(d1)` written out here, which catches a
sign error that finite differences would happily agree with if the forward were
wrong in the same way.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import minitensor as mt
from minitensor.kernels import black_scholes, implied_volatility


def _t(values, requires_grad=False):
    return mt.Tensor(
        np.ascontiguousarray(np.asarray(values, dtype=np.float64)),
        dtype="float64",
        requires_grad=requires_grad,
    )


def _phi(x: float) -> float:
    """The standard normal CDF, written out."""
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


def _reference(s, k, r, v, t, kind):
    """Black-Scholes as a scalar formula: the thing the kernel fuses."""
    d1 = (math.log(s / k) + (r + 0.5 * v * v) * t) / (v * math.sqrt(t))
    d2 = d1 - v * math.sqrt(t)
    if kind == "call":
        return s * _phi(d1) - k * math.exp(-r * t) * _phi(d2)
    return k * math.exp(-r * t) * _phi(-d2) - s * _phi(-d1)


@pytest.fixture(autouse=True)
def _clean_graph():
    yield
    mt.clear_autograd_graph()


# --- the value ----------------------------------------------------------------


@pytest.mark.parametrize("kind", ["call", "put"])
@pytest.mark.parametrize(
    "case",
    [
        (100.0, 95.0, 0.04, 0.30, 0.75),
        (12.0, 20.0, 0.01, 0.60, 2.00),
        (7.0, 7.0, 0.00, 0.05, 0.25),
        (250.0, 100.0, 0.08, 0.15, 3.00),
    ],
)
def test_the_price_matches_the_closed_form(kind, case):
    s, k, r, v, t = case
    got = np.asarray(black_scholes(_t([s]), _t([k]), _t([r]), _t([v]), _t([t]), kind))
    np.testing.assert_allclose(got, [_reference(s, k, r, v, t, kind)], rtol=1e-12)


def test_the_textbook_example():
    """S=K=100, r=5%, sigma=20%, T=1: call 10.4506, put 5.5735."""
    args = (_t([100.0]), _t([100.0]), _t([0.05]), _t([0.2]), _t([1.0]))
    call = float(np.asarray(black_scholes(*args, "call"))[0])
    put = float(np.asarray(black_scholes(*args, "put"))[0])
    assert abs(call - 10.450_583_572) < 1e-8
    assert abs(put - 5.573_526_022) < 1e-8


def test_put_call_parity_holds_exactly():
    """`C - P = S - K e^{-rT}`, for any inputs, by construction."""
    s, k, r, v, t = 130.0, 90.0, 0.03, 0.45, 1.5
    args = (_t([s]), _t([k]), _t([r]), _t([v]), _t([t]))
    call = float(np.asarray(black_scholes(*args, "call"))[0])
    put = float(np.asarray(black_scholes(*args, "put"))[0])
    assert abs((call - put) - (s - k * math.exp(-r * t))) < 1e-10


def test_it_prices_a_whole_book_at_once():
    rng = np.random.default_rng(0)
    spot = rng.uniform(50, 150, 64)
    strike = rng.uniform(50, 150, 64)
    rate = np.full(64, 0.03)
    vol = rng.uniform(0.1, 0.8, 64)
    time = rng.uniform(0.1, 3.0, 64)
    got = np.asarray(
        black_scholes(_t(spot), _t(strike), _t(rate), _t(vol), _t(time), "call")
    )
    want = [_reference(*case, "call") for case in zip(spot, strike, rate, vol, time)]
    np.testing.assert_allclose(got, want, rtol=1e-12)


# --- the degenerate corner ----------------------------------------------------


def test_an_expired_option_is_worth_its_intrinsic_value():
    zero = _t([0.0])
    call = black_scholes(_t([120.0]), _t([100.0]), _t([0.05]), _t([0.2]), zero, "call")
    assert abs(float(np.asarray(call)[0]) - 20.0) < 1e-12
    worthless = black_scholes(
        _t([80.0]), _t([100.0]), _t([0.05]), _t([0.2]), zero, "call"
    )
    assert float(np.asarray(worthless)[0]) == 0.0


def test_zero_volatility_is_the_discounted_forward():
    price = black_scholes(
        _t([100.0]), _t([90.0]), _t([0.05]), _t([0.0]), _t([1.0]), "call"
    )
    assert abs(float(np.asarray(price)[0]) - (100.0 - 90.0 * math.exp(-0.05))) < 1e-12


def test_the_degenerate_corner_is_finite_rather_than_nan():
    for vol, time in [(0.0, 1.0), (0.2, 0.0), (0.0, 0.0)]:
        price = black_scholes(
            _t([100.0]), _t([100.0]), _t([0.05]), _t([vol]), _t([time]), "call"
        )
        assert np.isfinite(np.asarray(price)).all(), (vol, time)


# --- the Greeks ---------------------------------------------------------------


_BASE = (100.0, 95.0, 0.04, 0.30, 0.75)
_NAMES = ["spot", "strike", "rate", "vol", "time"]


def _numeric_partial(slot, kind, step=1e-6):
    """`dV/dx` for argument `slot`, by central difference on the price."""

    def price_at(delta):
        case = list(_BASE)
        case[slot] += delta
        return _reference(*case, kind)

    return (price_at(step) - price_at(-step)) / (2.0 * step)


@pytest.mark.parametrize("kind", ["call", "put"])
@pytest.mark.parametrize("slot", range(5))
def test_each_greek_is_the_derivative_it_claims_to_be(kind, slot):
    tensors = [
        _t([value], requires_grad=(index == slot)) for index, value in enumerate(_BASE)
    ]
    black_scholes(*tensors, kind).sum().backward()

    analytic = float(np.asarray(tensors[slot].grad)[0])
    numeric = _numeric_partial(slot, kind)
    assert abs(analytic - numeric) < 1e-5 * max(
        abs(numeric), 1.0
    ), f"{kind} d/d{_NAMES[slot]}: analytic {analytic}, numeric {numeric}"


def test_delta_is_the_named_quantity():
    """A sign error that agrees with a wrong forward would pass the check above."""
    spot = _t([100.0], requires_grad=True)
    black_scholes(
        spot, _t([100.0]), _t([0.05]), _t([0.2]), _t([1.0]), "call"
    ).sum().backward()
    d1 = (0.05 + 0.5 * 0.04) / 0.2
    np.testing.assert_allclose(np.asarray(spot.grad), [_phi(d1)], rtol=1e-9)


def test_vega_is_the_same_for_a_call_and_a_put():
    """Parity differs by a forward, which does not depend on volatility."""
    grads = []
    for kind in ("call", "put"):
        vol = _t([0.3], requires_grad=True)
        black_scholes(
            _t([100.0]), _t([95.0]), _t([0.04]), vol, _t([0.75]), kind
        ).sum().backward()
        grads.append(float(np.asarray(vol.grad)[0]))
        mt.clear_autograd_graph()
    assert abs(grads[0] - grads[1]) < 1e-9


def test_all_five_gradients_arrive_from_one_backward():
    tensors = [_t([value], requires_grad=True) for value in _BASE]
    black_scholes(*tensors, "call").sum().backward()
    for tensor, name in zip(tensors, _NAMES):
        assert tensor.grad is not None, name


def test_a_frozen_argument_receives_nothing():
    spot = _t([100.0], requires_grad=True)
    vol = _t([0.3], requires_grad=False)
    black_scholes(
        spot, _t([95.0]), _t([0.04]), vol, _t([0.75]), "call"
    ).sum().backward()
    assert spot.grad is not None
    assert vol.grad is None


# --- inverting it -------------------------------------------------------------


@pytest.mark.parametrize("vol", [0.05, 0.2, 0.75, 2.0])
def test_implied_volatility_recovers_the_volatility_that_made_the_price(vol):
    args = (_t([100.0]), _t([110.0]), _t([0.03]), _t([1.5]))
    price = black_scholes(args[0], args[1], args[2], _t([vol]), args[3], "call")
    recovered = implied_volatility(price, *args, "call")
    assert abs(float(np.asarray(recovered)[0]) - vol) < 1e-6


def test_an_unattainable_price_has_no_implied_volatility():
    recovered = implied_volatility(
        _t([150.0]), _t([100.0]), _t([100.0]), _t([0.0]), _t([1.0]), "call"
    )
    assert np.isnan(np.asarray(recovered)).all()


# --- what is refused ----------------------------------------------------------


def test_a_bad_option_kind_is_rejected():
    with pytest.raises(ValueError, match="call"):
        black_scholes(_t([1.0]), _t([1.0]), _t([0.0]), _t([0.1]), _t([1.0]), "straddle")


def test_mismatched_shapes_are_rejected():
    with pytest.raises(Exception):
        black_scholes(
            _t([1.0]), _t([1.0, 2.0]), _t([0.0]), _t([0.1]), _t([1.0]), "call"
        )
