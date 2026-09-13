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


# --- against a reference that shares no code with the kernel ------------------

# Prices from the closed form evaluated at 200 bits with `mpmath`, and the five
# partials from differentiating *that* numerically at 200 bits -- not from Greek
# formulas. The tests above check the Greeks against central differences of the
# kernel's own forward and delta against `Phi(d1)` spelled out here, both of
# which agree with the kernel if the kernel and the test share a misreading.
# This table cannot: nothing in it came from this repository.
#
# Each row is ((spot, strike, rate, vol, time), price, (d/dspot, d/dstrike,
# d/drate, d/dvol, d/dtime)). Regenerate with `mpmath` at `mp.prec = 200` if the
# cases ever change.
_REFERENCE = {
    "call": [
        (
            (100.0, 100.0, 0.05, 0.2, 1.0),
            10.450583572185568,
            (
                0.6368306511756191,
                -0.5323248154537634,
                53.23248154537634,
                37.524034691693785,
                6.414027546438196,
            ),
        ),
        (
            (100.0, 120.0, 0.05, 0.2, 1.0),
            3.2474774165608142,
            (
                0.28719163790512703,
                -0.21226405311626573,
                25.471686373951886,
                34.07384227701016,
                4.680968546398611,
            ),
        ),
        (
            (100.0, 80.0, 0.05, 0.2, 1.0),
            24.588835443927753,
            (
                0.9286374026649281,
                -0.8534363102820632,
                68.27490482256506,
                13.627194363994361,
                4.77646467752769,
            ),
        ),
        (
            (42.0, 40.0, 0.1, 0.2, 0.5),
            4.759422392871533,
            (
                0.779131290942669,
                -0.6991022956680141,
                13.982045913360281,
                8.813415059602852,
                4.559092194592627,
            ),
        ),
        (
            (100.0, 100.0, 0.0, 0.3, 2.0),
            16.79959714273635,
            (
                0.5839979857136818,
                -0.4160020142863183,
                83.20040285726365,
                55.16370633254119,
                4.13727797494059,
            ),
        ),
        (
            (100.0, 100.0, -0.01, 0.15, 0.25),
            2.8716165053154734,
            (
                0.5016622546919065,
                -0.4729460896387518,
                11.823652240968794,
                19.946940868791735,
                5.5111361709987685,
            ),
        ),
    ],
    "put": [
        (
            (100.0, 100.0, 0.05, 0.2, 1.0),
            5.573526022256968,
            (
                -0.3631693488243809,
                0.4189046090469506,
                -41.89046090469506,
                37.524034691693785,
                1.6578804239346259,
            ),
        ),
        (
            (100.0, 120.0, 0.05, 0.2, 1.0),
            17.395008356646496,
            (
                -0.712808362094873,
                0.7389653713844483,
                -88.67584456613379,
                34.07384227701016,
                -1.0264080006056737,
            ),
        ),
        (
            (100.0, 80.0, 0.05, 0.2, 1.0),
            0.6871894039848735,
            (
                -0.07136259733507189,
                0.09779311421865076,
                -7.823449137492061,
                13.627194363994361,
                0.9715469795248332,
            ),
        ),
        (
            (42.0, 40.0, 0.1, 0.2, 0.5),
            0.8085993729000936,
            (
                -0.22086870905733105,
                0.25212712883269994,
                -5.042542576653999,
                8.813415059602852,
                0.7541744965897705,
            ),
        ),
        (
            (100.0, 100.0, 0.0, 0.3, 2.0),
            16.79959714273635,
            (
                -0.4160020142863183,
                0.5839979857136818,
                -116.79959714273635,
                55.16370633254119,
                4.13727797494059,
            ),
        ),
        (
            (100.0, 100.0, -0.01, 0.15, 0.25),
            3.1219292658949818,
            (
                -0.4983377453080935,
                0.5295570379670433,
                -13.238925949176084,
                19.946940868791735,
                6.513639298604564,
            ),
        ),
    ],
}


@pytest.mark.parametrize("kind", ["call", "put"])
def test_price_and_greeks_match_a_two_hundred_bit_reference(kind):
    """Every published figure agrees to about 1e-13; the bound here is 1e-10.

    Loose on purpose. `erfc` comes from the platform's `libm` and its last bits
    differ between them, while the failure this is guarding against -- a wrong
    formula, a flipped sign, a Greek wired to the wrong input -- is off by a
    factor, not by an ulp.
    """
    for args, want_price, want_greeks in _REFERENCE[kind]:
        inputs = [_t([value], requires_grad=True) for value in args]
        priced = black_scholes(*inputs, kind)
        assert float(np.asarray(priced)[0]) == pytest.approx(want_price, rel=1e-10)

        priced.backward()
        for name, tensor, want in zip(
            ("spot", "strike", "rate", "vol", "time"), inputs, want_greeks
        ):
            got = float(np.asarray(tensor.grad)[0])
            assert got == pytest.approx(
                want, rel=1e-10, abs=1e-10
            ), f"d/d{name} at {args} ({kind}): {got} != {want}"


def test_implied_volatility_stops_at_its_guess_where_vega_underflows():
    """The solver converges in price, and deep enough in the money there is no
    price left to converge on.

    A deep in-the-money call one month out is worth its intrinsic value to the
    last bit of a double: at `S=300, K=100, r=0.05, T=0.1` the volatilities
    0.05, 0.1 and 0.2 all price to exactly 200.49875208073178, and vega at 0.2
    is 1.66e-65. Nothing can recover 0.05 from that, and what comes back is the
    0.2 the iteration starts at -- a number that looks like an answer.

    This is pinned rather than fixed because every way of detecting it needs a
    threshold on vega, and any threshold refuses quotes that are legitimately
    informative. The docstring says so; this makes the two agree. A change that
    returns NaN here is welcome and should update both.
    """
    args = (300.0, 100.0, 0.05, 0.1)  # spot, strike, rate, time
    priced_low = black_scholes(
        _t([args[0]]), _t([args[1]]), _t([args[2]]), _t([0.05]), _t([args[3]]), "call"
    )
    priced_guess = black_scholes(
        _t([args[0]]), _t([args[1]]), _t([args[2]]), _t([0.2]), _t([args[3]]), "call"
    )
    # The premise: two very different volatilities, one identical price.
    assert float(np.asarray(priced_low)[0]) == float(np.asarray(priced_guess)[0])

    recovered = implied_volatility(
        priced_low, _t([args[0]]), _t([args[1]]), _t([args[2]]), _t([args[3]]), "call"
    )
    assert float(np.asarray(recovered)[0]) == pytest.approx(0.2, abs=1e-12)


def test_implied_volatility_recovers_the_volatility_where_the_price_carries_it():
    """The other side of the same coin: where vega is not negligible, the round
    trip comes back. The bound is `tolerance / vega`, which at these points is
    far below the 1e-4 asserted here."""
    for spot, strike, rate, time in [
        (100.0, 100.0, 0.05, 1.0),
        (100.0, 120.0, 0.05, 1.0),
        (42.0, 40.0, 0.10, 0.5),
        (100.0, 100.0, -0.01, 0.25),
    ]:
        for kind in ("call", "put"):
            for vol in (0.1, 0.2, 0.5, 1.0):
                priced = black_scholes(
                    _t([spot]), _t([strike]), _t([rate]), _t([vol]), _t([time]), kind
                )
                recovered = implied_volatility(
                    priced, _t([spot]), _t([strike]), _t([rate]), _t([time]), kind
                )
                assert float(np.asarray(recovered)[0]) == pytest.approx(
                    vol, rel=1e-4
                ), f"{kind} S={spot} K={strike} r={rate} T={time} vol={vol}"
