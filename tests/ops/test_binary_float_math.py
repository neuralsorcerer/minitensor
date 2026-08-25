# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""`atan2`, `hypot`, `copysign` and `xlogy`.

Each exists because the obvious composition gets something wrong, so the tests
are mostly about the cases the composition would miss: the quadrant `atan(y/x)`
loses, the overflow `sqrt(x*x + y*y)` hits, the signed zero `sign(y) * abs(x)`
drops, and the `0 * -inf` that `x * log(y)` gives where entropy needs a zero.
"""

import numpy as np
import pytest

import minitensor as mt

OPS = [
    ("atan2", np.arctan2),
    ("hypot", np.hypot),
    ("copysign", np.copysign),
    ("xlogy", lambda a, b: np.where(a == 0, 0.0, a * np.log(np.where(a == 0, 1.0, b)))),
]

# Positive second operands, so `xlogy` is real and `atan2` stays off its branch
# cut; the sign cases each op cares about get their own tests.
LHS = np.array([[0.7, -1.3, 2.4], [-0.4, 3.0, 0.25]])
RHS = np.array([[1.1, 0.6, 2.2], [3.5, 0.125, 4.0]])


@pytest.mark.parametrize("name, reference", OPS)
def test_matches_numpy(name, reference):
    a, b = mt.from_numpy(LHS.copy()), mt.from_numpy(RHS.copy())
    expected = reference(LHS, RHS)

    for result in (
        getattr(a, name)(b),
        getattr(mt, name)(a, b),
        getattr(mt.functional, name)(a, b),
    ):
        np.testing.assert_allclose(result.numpy(), expected, rtol=1e-15)


@pytest.mark.parametrize("name, reference", OPS)
def test_broadcasts_and_takes_python_scalars(name, reference):
    column = LHS[:, :1]
    a, b = mt.from_numpy(np.ascontiguousarray(column)), mt.from_numpy(RHS.copy())
    result = getattr(mt, name)(a, b)
    assert result.shape == RHS.shape
    np.testing.assert_allclose(result.numpy(), reference(column, RHS), rtol=1e-15)

    np.testing.assert_allclose(
        getattr(mt, name)(mt.from_numpy(LHS.copy()), 2.0).numpy(),
        reference(LHS, 2.0),
        rtol=1e-15,
    )


def test_atan2_keeps_the_quadrant_that_atan_of_the_ratio_loses():
    y = np.array([1.0, 1.0, -1.0, -1.0, 0.0, -0.0, 1.0, -1.0, 0.0])
    x = np.array([1.0, -1.0, -1.0, 1.0, -1.0, -1.0, 0.0, 0.0, 0.0])
    got = mt.atan2(mt.from_numpy(y.copy()), mt.from_numpy(x.copy())).numpy()
    np.testing.assert_array_equal(got, np.arctan2(y, x))

    # The second and third quadrants are exactly what `atan(y / x)` folds onto
    # the first and fourth.
    assert got[1] == pytest.approx(3 * np.pi / 4)
    assert got[2] == pytest.approx(-3 * np.pi / 4)
    assert np.arctan(y[1] / x[1]) == pytest.approx(-np.pi / 4)


def test_hypot_answers_where_the_squares_would_not():
    big, tiny = 1e200, 1e-200
    values = np.array([big, tiny, 3.0])
    got = mt.hypot(mt.from_numpy(values.copy()), mt.from_numpy(values.copy())).numpy()

    assert np.isfinite(got[0]), "the large case overflowed"
    assert got[1] > 0.0, "the small case underflowed"
    np.testing.assert_allclose(got, np.hypot(values, values), rtol=1e-15)
    # What the naive form does with the same inputs.
    with np.errstate(over="ignore", under="ignore"):
        naive = np.sqrt(values * values + values * values)
    assert np.isinf(naive[0]) and naive[1] == 0.0


def test_copysign_carries_the_sign_bit_of_a_zero():
    x = np.array([3.0, 3.0, -3.0, -3.0, 0.0, np.inf])
    y = np.array([1.0, -0.0, 0.0, -1.0, -1.0, -2.0])
    got = mt.copysign(mt.from_numpy(x.copy()), mt.from_numpy(y.copy())).numpy()

    np.testing.assert_array_equal(got, np.copysign(x, y))
    # `sign(y) * abs(x)` would give +3 here, because `sign(-0.0)` is 0.
    assert got[1] == -3.0
    assert np.signbit(got[4]) and got[4] == 0.0


def test_xlogy_is_the_limit_where_the_product_is_nan():
    x = np.array([0.0, 0.0, 0.0, 2.0, 2.0])
    y = np.array([0.0, np.inf, 1.0, np.e, 0.0])
    got = mt.xlogy(mt.from_numpy(x.copy()), mt.from_numpy(y.copy())).numpy()

    np.testing.assert_allclose(got, [0.0, 0.0, 0.0, 2.0, -np.inf])
    # The plain product is NaN at the case entropy hits most.
    with np.errstate(divide="ignore", invalid="ignore"):
        assert np.isnan(x[0] * np.log(y[0]))

    # A NaN second operand has no limit to take, so it survives.
    nan_y = mt.from_numpy(np.array([np.nan]))
    assert np.isnan(mt.xlogy(mt.from_numpy(np.array([0.0])), nan_y).numpy()[0])


def test_entropy_written_with_xlogy_survives_a_zero_probability():
    p = np.array([0.0, 0.25, 0.75])
    entropy = -mt.xlogy(mt.from_numpy(p.copy()), mt.from_numpy(p.copy())).sum()
    expected = -np.sum(p[p > 0] * np.log(p[p > 0]))
    assert entropy.item() == pytest.approx(expected)


@pytest.mark.parametrize("name", ["atan2", "hypot", "copysign", "xlogy"])
def test_integer_operands_promote_the_way_division_does(name):
    ints = mt.from_numpy(np.array([3, 4], dtype=np.int64))
    assert "float32" in str(getattr(mt, name)(ints, ints).dtype)

    wide = mt.from_numpy(np.array([3.0, 4.0], dtype=np.float64))
    assert "float64" in str(getattr(mt, name)(ints, wide).dtype)


@pytest.mark.parametrize("name", ["atan2", "hypot", "xlogy"])
def test_gradients_match_central_differences(name):
    op = getattr(mt, name)
    lhs, rhs = LHS.copy(), RHS.copy()

    a = mt.Tensor(lhs, dtype="float64", requires_grad=True)
    b = mt.Tensor(rhs, dtype="float64", requires_grad=True)
    op(a, b).sum().backward()
    analytic = (a.grad.numpy().copy(), b.grad.numpy().copy())
    mt.clear_autograd_graph()

    eps = 1e-6
    for which, got in enumerate(analytic):
        operands = [lhs, rhs]
        numeric = np.zeros_like(operands[which])
        for index in np.ndindex(*operands[which].shape):
            shifted = [lhs.copy(), rhs.copy()]
            shifted[which][index] += eps
            up = op(*(mt.Tensor(s, dtype="float64") for s in shifted)).sum().item()
            shifted = [lhs.copy(), rhs.copy()]
            shifted[which][index] -= eps
            down = op(*(mt.Tensor(s, dtype="float64") for s in shifted)).sum().item()
            numeric[index] = (up - down) / (2 * eps)
        np.testing.assert_allclose(got, numeric, rtol=1e-5, atol=1e-7)


def test_copysign_gradient_is_a_sign_flip_and_nothing_flows_to_the_sign_source():
    a = mt.Tensor(LHS.copy(), dtype="float64", requires_grad=True)
    b = mt.Tensor(RHS.copy(), dtype="float64", requires_grad=True)
    mt.copysign(a, b).sum().backward()

    np.testing.assert_array_equal(a.grad.numpy(), np.sign(LHS) * np.sign(RHS))
    # The result moves with the sign bit of `b`, which no derivative can see.
    np.testing.assert_array_equal(b.grad.numpy(), np.zeros_like(RHS))


def test_a_frozen_operand_collects_no_gradient():
    a = mt.Tensor(LHS.copy(), dtype="float64", requires_grad=True)
    b = mt.Tensor(RHS.copy(), dtype="float64")
    mt.hypot(a, b).sum().backward()
    assert a.grad is not None
    assert b.grad is None


def test_empty_and_mismatched_shapes():
    empty = mt.from_numpy(np.array([]))
    assert mt.hypot(empty, empty).shape == (0,)

    a = mt.from_numpy(np.array([1.0, 2.0, 3.0]))
    b = mt.from_numpy(np.array([1.0, 2.0]))
    with pytest.raises(ValueError):
        mt.atan2(a, b)
