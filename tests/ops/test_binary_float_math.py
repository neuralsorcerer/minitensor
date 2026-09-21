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


# float32 `atan2` is a vectorized kernel now: `atan(y/x)` computed in float64
# and narrowed once, with a quadrant correction, rather than a call to scalar
# `atan2f` per element. The quotient is what the kernel cannot always form --
# `y/x` is an infinity or a NaN rather than a slope when `x` is a zero, and
# `inf/inf` when both are infinite -- so those are redone by a second pass.
# These pin that the second pass covers exactly what it must.
ATAN2_SPECIAL = [
    0.0,
    -0.0,
    1.0,
    -1.0,
    np.inf,
    -np.inf,
    np.nan,
    np.finfo("float32").tiny,
    -np.finfo("float32").tiny,
    np.finfo("float32").max,
    -np.finfo("float32").max,
    1e-30,
    -1e-30,
    0.41421356,  # tan(pi/8), the kernel's lower branch boundary
    2.4142136,  # tan(3pi/8), its upper one
]


def test_atan2_matches_numpy_on_every_pair_of_special_values():
    # Every special against every other, which is where the quotient stops
    # being a slope and the fallback has to take over. These all have answers
    # both libraries agree on exactly -- the zeros, the infinities, the NaN,
    # and the axis angles -- so this compares against NumPy directly.
    values = np.array(ATAN2_SPECIAL, dtype="float32")
    ys, xs = (a.ravel() for a in np.meshgrid(values, values))
    degenerate = (ys == 0) | (xs == 0) | ~np.isfinite(ys) | ~np.isfinite(xs)
    ys, xs = ys[degenerate], xs[degenerate]

    got = mt.atan2(mt.from_numpy(ys), mt.from_numpy(xs)).numpy()
    want = np.arctan2(ys, xs)
    np.testing.assert_array_equal(got, want)
    # The sign of a zero result is the whole point of keeping the quadrant.
    np.testing.assert_array_equal(np.signbit(got), np.signbit(want))


def test_float32_atan2_is_the_correctly_rounded_answer_where_numpy_is_not():
    # The reference is the one the rest of these kernels are held to: computed
    # in float64, narrowed once. Against it, over a wide random sample, this
    # kernel differs on nothing and `np.arctan2` on about 6% of pairs by up to
    # three ulp -- so asserting against NumPy at the same width would be
    # asserting the less accurate of the two.
    rng = np.random.default_rng(11)
    n = 200_000
    y = (np.sign(rng.standard_normal(n)) * 10.0 ** rng.uniform(-38, 38, n)).astype(
        "float32"
    )
    x = (np.sign(rng.standard_normal(n)) * 10.0 ** rng.uniform(-38, 38, n)).astype(
        "float32"
    )
    truth = np.arctan2(y.astype(np.float64), x.astype(np.float64)).astype(np.float32)

    np.testing.assert_array_equal(
        mt.atan2(mt.from_numpy(y), mt.from_numpy(x)).numpy(), truth
    )
    # And the claim about NumPy, so this test fails if that ever stops being
    # the reason the comparison is written this way.
    apart = np.abs(
        np.arctan2(y, x).view(np.int32).astype(np.int64)
        - truth.view(np.int32).astype(np.int64)
    )
    assert apart.max() >= 1, "np.arctan2 now matches; the reference can be simplified"


@pytest.mark.parametrize("size", [1, 7, 8, 1023, 100_000])
def test_float32_atan2_matches_a_float64_reference_across_the_exponent_range(size):
    # Log-uniform over the whole float32 exponent range and both signs, so the
    # quotient spans about 1e-76 to 1e76 -- far outside what a float32 ratio
    # could hold, and well inside what a float64 one can, which is why the
    # kernel forms it wide.
    rng = np.random.default_rng(size)
    y = (
        np.sign(rng.standard_normal(size)) * 10.0 ** rng.uniform(-38, 38, size)
    ).astype("float32")
    x = (
        np.sign(rng.standard_normal(size)) * 10.0 ** rng.uniform(-38, 38, size)
    ).astype("float32")
    got = mt.atan2(mt.from_numpy(y), mt.from_numpy(x)).numpy()
    want = np.arctan2(y.astype(np.float64), x.astype(np.float64)).astype(np.float32)
    np.testing.assert_array_equal(got, want)


@pytest.mark.parametrize("size", [1, 7, 8, 1023, 100_000])
def test_the_fallback_pass_finds_its_elements_at_every_alignment(size):
    # The elements needing the scalar redo are found by a second pass over the
    # block, so one landing in the vectorized body and one in the tail have to
    # come back the same. A fifth of each operand is a zero or an infinity.
    rng = np.random.default_rng(size + 1)
    y = rng.standard_normal(size).astype("float32")
    x = rng.standard_normal(size).astype("float32")
    for arr in (y, x):
        picks = rng.random(size) < 0.2
        arr[picks] = rng.choice(
            np.array([0.0, -0.0, np.inf, -np.inf], dtype="float32"), picks.sum()
        )
    got = mt.atan2(mt.from_numpy(y), mt.from_numpy(x)).numpy()
    want = np.arctan2(y.astype(np.float64), x.astype(np.float64)).astype(np.float32)
    np.testing.assert_array_equal(got, want)
    np.testing.assert_array_equal(np.signbit(got), np.signbit(want))


def test_atan2_broadcasts_off_the_kernel_path():
    # The kernel only runs when both operands already have the output shape;
    # a broadcast falls back to the element-wise path and must still agree.
    rows = np.random.default_rng(2).standard_normal((64, 129)).astype("float32")
    column = np.random.default_rng(3).standard_normal((64, 1)).astype("float32")
    np.testing.assert_allclose(
        mt.atan2(mt.from_numpy(rows), mt.from_numpy(column)).numpy(),
        np.arctan2(rows, column),
        rtol=1e-6,
    )
    scalar = np.array([2.0], dtype="float32")
    np.testing.assert_allclose(
        mt.atan2(mt.from_numpy(rows), mt.from_numpy(scalar)).numpy(),
        np.arctan2(rows, scalar),
        rtol=1e-6,
    )


def test_atan2_reads_a_transposed_operand_in_its_own_order():
    # The kernel is handed the raw buffer, which is only the logical order
    # because a tensor here is always contiguous -- `expand` materializes
    # rather than striding. If that invariant ever moves, this is where a
    # transposed operand starts answering in the wrong order.
    rows = np.random.default_rng(4).standard_normal((129, 64)).astype("float32")
    other = np.random.default_rng(5).standard_normal((64, 129)).astype("float32")
    transposed = mt.from_numpy(rows).transpose(0, 1)

    got = mt.atan2(transposed, mt.from_numpy(other)).numpy()
    want = np.arctan2(rows.T.astype(np.float64), other.astype(np.float64)).astype(
        "float32"
    )
    np.testing.assert_array_equal(got, want)
