# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""Gradients stay finite where a formula reads `0 * inf` or `0 / 0`.

Each case here is a point where the function is either flat or has a corner,
and where the textbook derivative, evaluated literally, is NaN:

* `std` of a constant slice: the square root's slope is infinite at a zero
  variance and the variance's slope there is zero.
* `x ** 0` at `x = 0`: the slope `0 * 0^-1`, of a function that is 1
  everywhere.
* `0 ** y` with respect to `y`, for `y > 0`: the slope `0 * ln(0)`, of a
  function that is 0 for every nearby `y`.
* `hypot(0, 0)`: the cone point `0 / 0`, which `norm` already reports as the
  zero subgradient.
* `xlogy(0, y)` with respect to `y`: `0 / y`, of a value defined as 0 for
  every `y`, which reads `0 / 0` at `y = 0`.

A single NaN there poisons every parameter it reaches, so each now reports 0
and the gradient away from the point is unchanged.
"""

import math

import pytest

import minitensor as mt

DTYPES = ["float32", "float64"]


def _grad(build, *values, dtype="float64"):
    leaves = [mt.tensor(v, dtype=dtype, requires_grad=True) for v in values]
    out = build(*leaves)
    out.sum().backward()
    return [leaf.grad.tolist() for leaf in leaves]


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("unbiased", [True, False])
def test_std_of_a_constant_has_a_zero_gradient(dtype, unbiased):
    (grad,) = _grad(lambda x: x.std(unbiased=unbiased), [2.0, 2.0, 2.0], dtype=dtype)
    assert grad == [0.0, 0.0, 0.0]


@pytest.mark.parametrize("dtype", DTYPES)
def test_std_along_a_dim_only_zeroes_the_constant_rows(dtype):
    (grad,) = _grad(
        lambda x: x.std(dim=1),
        [[1.0, 1.0, 1.0], [1.0, 2.0, 3.0]],
        dtype=dtype,
    )
    assert grad[0] == [0.0, 0.0, 0.0]
    # Row two: d/dx_i std = (x_i - mean) / ((n - 1) * std), std = 1.
    assert grad[1] == pytest.approx([-0.5, 0.0, 0.5], abs=1e-6)


def test_std_gradient_matches_central_differences_off_the_corner():
    x = mt.tensor([[0.5, -1.25, 2.0, 0.0], [3.0, 3.5, -2.0, 1.0]], dtype="float64")
    for dim in (None, 0, 1):
        for unbiased in (True, False):
            leaf = x.detach().requires_grad_(True)
            assert mt.gradcheck(
                lambda t, dim=dim, unbiased=unbiased: t.std(
                    dim=dim, unbiased=unbiased
                ).sum(),
                (leaf,),
                eps=1e-6,
                atol=1e-8,
                rtol=1e-6,
            )


def test_std_of_a_constant_still_carries_the_gradient_to_what_follows():
    # The zero is the std's slope, not a cut in the graph: a sum after it
    # still reaches the other operand.
    x = mt.tensor([4.0, 4.0], dtype="float64", requires_grad=True)
    y = mt.tensor([1.0], dtype="float64", requires_grad=True)
    (x.std() + 3.0 * y).sum().backward()
    assert x.grad.tolist() == [0.0, 0.0]
    assert y.grad.tolist() == [3.0]


@pytest.mark.parametrize("dtype", DTYPES)
def test_zeroth_power_has_a_zero_gradient_at_zero(dtype):
    (grad,) = _grad(lambda x: x**0, [0.0, 2.0, -3.0], dtype=dtype)
    assert grad == [0.0, 0.0, 0.0]
    # The same exponent as a tensor takes the elementwise path.
    (grad,) = _grad(
        lambda x: mt.pow(x, mt.tensor([0.0, 0.0, 0.0], dtype=dtype)),
        [0.0, 2.0, -3.0],
        dtype=dtype,
    )
    assert grad == [0.0, 0.0, 0.0]


@pytest.mark.parametrize("dtype", DTYPES)
def test_power_of_zero_has_a_zero_exponent_gradient(dtype):
    base = mt.tensor([0.0, 0.0, 2.0], dtype=dtype)
    (grad,) = _grad(lambda y: mt.pow(base, y), [0.5, 3.0, 3.0], dtype=dtype)
    assert grad[:2] == [0.0, 0.0]
    assert grad[2] == pytest.approx(8.0 * math.log(2.0), rel=1e-6)


def test_power_gradients_match_central_differences_elsewhere():
    b = mt.tensor([0.5, 1.5, 2.0, 3.0], dtype="float64", requires_grad=True)
    e = mt.tensor([2.0, -1.5, 0.5, 1.0], dtype="float64", requires_grad=True)
    assert mt.gradcheck(lambda b, e: mt.pow(b, e).sum(), (b, e), atol=1e-8, rtol=1e-6)


def test_power_keeps_its_infinite_slope_where_it_has_one():
    # `sqrt`'s slope at 0 is a genuine +inf, not a 0 * inf, and stays one.
    (grad,) = _grad(lambda x: x**0.5, [0.0])
    assert grad == [math.inf]


@pytest.mark.parametrize("dtype", DTYPES)
def test_hypot_takes_the_zero_subgradient_at_the_origin(dtype):
    gx, gy = _grad(mt.hypot, [0.0, 3.0], [0.0, 4.0], dtype=dtype)
    assert gx == pytest.approx([0.0, 0.6])
    assert gy == pytest.approx([0.0, 0.8])
    # The same point through `norm` agrees.
    pair = mt.tensor([0.0, 0.0], dtype=dtype, requires_grad=True)
    pair.norm().backward()
    assert pair.grad.tolist() == [0.0, 0.0]


@pytest.mark.parametrize("dtype", DTYPES)
def test_xlogy_is_flat_in_y_wherever_x_is_zero(dtype):
    # `xlogy(0, y)` is 0 for every y, so its y-slope is 0 -- at y = 0 too,
    # where `x / y` reads `0 / 0`.
    _, gy = _grad(mt.xlogy, [0.0, 0.0, 2.0], [0.0, 3.0, 4.0], dtype=dtype)
    assert gy == pytest.approx([0.0, 0.0, 0.5])


# --- the log-sum-exp family and atan2 at infinities -------------------------

INF = math.inf


@pytest.mark.parametrize("dtype", DTYPES)
def test_logaddexp_takes_the_limit_of_its_slopes(dtype):
    a = [-INF, INF, INF, 1.0, -INF, 2.0]
    b = [-INF, 1.0, INF, -INF, 3.0, 2.0]
    ga, gb = _grad(mt.logaddexp, a, b, dtype=dtype)
    # Two `-inf` operands take no slope, as an all `-inf` row of `logsumexp`
    # does; an operand at `+inf` takes all of it, and two share it.
    assert ga[:5] == [0.0, 1.0, 0.5, 1.0, 0.0]
    assert gb[:5] == [0.0, 0.0, 0.5, 0.0, 1.0]
    assert ga[5] == pytest.approx(0.5) and gb[5] == pytest.approx(0.5)


def test_logaddexp_matches_central_differences():
    a = mt.tensor([[0.5], [-3.0], [40.0]], dtype="float64", requires_grad=True)
    b = mt.tensor([1.0, -2.0, 39.5], dtype="float64", requires_grad=True)
    assert mt.gradcheck(lambda a, b: mt.logaddexp(a, b).sum(), (a, b), rtol=1e-6)


@pytest.mark.parametrize("dtype", DTYPES)
def test_logcumsumexp_with_leading_negative_infinities(dtype):
    (grad,) = _grad(
        lambda x: mt.logcumsumexp(x, 0), [-INF, -INF, 0.0, 1.0], dtype=dtype
    )
    e = math.e
    # Position 2 feeds out[2] (weight 1) and out[3] (weight 1 / (1 + e)).
    assert grad[:2] == [0.0, 0.0]
    assert grad[2] == pytest.approx(1.0 + 1.0 / (1.0 + e), rel=1e-6)
    assert grad[3] == pytest.approx(e / (1.0 + e), rel=1e-6)


@pytest.mark.parametrize("dtype", DTYPES)
def test_nanstd_of_a_constant_has_a_zero_gradient(dtype):
    (grad,) = _grad(lambda x: mt.nanstd(x), [2.0, 2.0, math.nan], dtype=dtype)
    assert grad == [0.0, 0.0, 0.0]


@pytest.mark.parametrize("dtype", DTYPES)
def test_atan2_slopes_vanish_at_an_infinite_point(dtype):
    gy, gx = _grad(mt.atan2, [INF, 1.0, -INF], [INF, INF, 2.0], dtype=dtype)
    assert gy == [0.0, 0.0, 0.0]
    assert gx == [0.0, 0.0, 0.0]
