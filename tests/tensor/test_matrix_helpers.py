# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""The named matmuls, inverses and rescalings, against NumPy.

`matmul`, `solve` and `svd` are the kernels; everything here is one of those
pointed at a rearranged operand. So the tests check the rearrangement -- that
`tensordot` contracts the axes it names, that `inverse` inverts, that `renorm`
leaves an untouched row bit-for-bit -- against NumPy wherever NumPy has the
same function, and against the definition where it does not.
"""

from __future__ import annotations

import numpy as np
import pytest

import minitensor as mt
from minitensor import functional as F

RNG = np.random.default_rng(23)
A = np.array([[2.0, 1.0], [1.0, 3.0]])
B = np.array([[1.0, 0.5], [2.0, -1.0]])


def _t(values, dtype="float64", requires_grad=False):
    return mt.Tensor(
        np.ascontiguousarray(np.asarray(values, dtype=np.float64)),
        dtype=dtype,
        requires_grad=requires_grad,
    )


NAMES = [
    "t",
    "numel",
    "mm",
    "mv",
    "inner",
    "tensordot",
    "addmm",
    "baddbmm",
    "inverse",
    "pinverse",
    "logdet",
    "renorm",
    "vander",
    "real",
    "conj",
    "imag",
    "angle",
]


@pytest.mark.parametrize("name", NAMES)
def test_every_spelling_is_the_same_object(name):
    assert getattr(F, name) is getattr(mt, name)
    assert hasattr(mt.Tensor, name), f"{name} is missing as a method"


# --- the named matmuls ------------------------------------------------------


def test_t_transposes_a_matrix_and_leaves_lower_ranks_alone():
    np.testing.assert_array_equal(mt.t(_t(A)).numpy(), A.T)
    np.testing.assert_array_equal(mt.t(_t([1.0, 2.0])).numpy(), [1.0, 2.0])
    np.testing.assert_array_equal(mt.t(_t(3.0)).numpy(), 3.0)


def test_t_declines_a_rank_it_would_have_to_guess_at():
    with pytest.raises(ValueError, match="at most two dimensions"):
        mt.t(_t(RNG.standard_normal((2, 3, 4))))


def test_numel_counts_every_element():
    assert mt.numel(_t(A)) == 4
    assert mt.numel(_t(RNG.standard_normal((2, 3, 4)))) == 24
    assert mt.numel(_t(3.0)) == 1


def test_mm_and_mv_are_matmul_with_the_ranks_pinned():
    np.testing.assert_allclose(mt.mm(_t(A), _t(B)).numpy(), A @ B)
    vector = np.array([1.5, -2.0])
    np.testing.assert_allclose(mt.mv(_t(A), _t(vector)).numpy(), A @ vector)


def test_mm_and_mv_refuse_what_matmul_would_have_guessed_at():
    with pytest.raises(ValueError, match="2-dimensional"):
        mt.mm(_t([1.0, 2.0]), _t(A))
    with pytest.raises(ValueError, match="1-dimensional"):
        mt.mv(_t(A), _t(B))


@pytest.mark.parametrize(
    "left,right",
    [
        ((3,), (3,)),
        ((2, 3), (4, 3)),
        ((2, 3, 4), (5, 4)),
    ],
    ids=["vectors", "matrices", "mixed_rank"],
)
def test_inner_matches_numpy(left, right):
    a = RNG.standard_normal(left)
    b = RNG.standard_normal(right)
    np.testing.assert_allclose(
        mt.inner(_t(a), _t(b)).numpy(), np.inner(a, b), rtol=1e-13
    )


@pytest.mark.parametrize(
    "left,right,dims",
    [
        # An integer count contracts the last `n` of the left against the first
        # `n` of the right, so each case needs shapes that line up that way.
        ((2, 3), (4, 5), 0),
        ((2, 3, 4), (4, 5), 1),
        ((2, 3, 4), (3, 4, 5), 2),
    ],
    ids=["outer", "one_axis", "two_axes"],
)
def test_tensordot_matches_numpy_for_an_integer_count(left, right, dims):
    a = RNG.standard_normal(left)
    b = RNG.standard_normal(right)
    got = mt.tensordot(_t(a), _t(b), dims).numpy()
    np.testing.assert_allclose(got, np.tensordot(a, b, dims), rtol=1e-13)


def test_tensordot_matches_numpy_for_explicit_axis_lists():
    a = RNG.standard_normal((2, 3, 4))
    b = RNG.standard_normal((4, 3, 5))
    for axes in (([1, 2], [1, 0]), ([2], [0]), ([1], [1])):
        np.testing.assert_allclose(
            mt.tensordot(_t(a), _t(b), axes).numpy(),
            np.tensordot(a, b, axes),
            rtol=1e-13,
        )


def test_tensordot_accepts_negative_axes():
    a = RNG.standard_normal((2, 3, 4))
    b = RNG.standard_normal((4, 5))
    np.testing.assert_allclose(
        mt.tensordot(_t(a), _t(b), ([-1], [0])).numpy(),
        np.tensordot(a, b, ([-1], [0])),
        rtol=1e-13,
    )


def test_tensordot_reports_axes_that_do_not_line_up():
    a = RNG.standard_normal((2, 3))
    b = RNG.standard_normal((4, 5))
    with pytest.raises(ValueError, match="contract axes of length"):
        mt.tensordot(_t(a), _t(b), 1)
    with pytest.raises(ValueError, match="same number of axes"):
        mt.tensordot(_t(a), _t(b), ([0, 1], [0]))
    with pytest.raises(ValueError, match="cannot contract 3 axes"):
        mt.tensordot(_t(a), _t(b), 3)


def test_addmm_and_baddbmm_are_the_expression_they_name():
    np.testing.assert_allclose(
        mt.addmm(_t(A), _t(A), _t(B), beta=2.0, alpha=3.0).numpy(),
        2.0 * A + 3.0 * (A @ B),
    )
    batch1 = RNG.standard_normal((3, 2, 4))
    batch2 = RNG.standard_normal((3, 4, 5))
    base = RNG.standard_normal((3, 2, 5))
    np.testing.assert_allclose(
        mt.baddbmm(_t(base), _t(batch1), _t(batch2), beta=0.5, alpha=2.0).numpy(),
        0.5 * base + 2.0 * (batch1 @ batch2),
        rtol=1e-13,
    )


def test_baddbmm_requires_three_dimensional_batches():
    with pytest.raises(ValueError, match="3-dimensional"):
        mt.baddbmm(_t(A), _t(A), _t(B))


# --- inverses ---------------------------------------------------------------


def test_inverse_matches_numpy_and_undoes_the_matrix():
    np.testing.assert_allclose(mt.inverse(_t(A)).numpy(), np.linalg.inv(A), rtol=1e-13)
    product = mt.mm(_t(A), mt.inverse(_t(A))).numpy()
    np.testing.assert_allclose(product, np.eye(2), atol=1e-14)


def test_inverse_refuses_what_it_cannot_invert():
    with pytest.raises(ValueError, match="at least two dimensions"):
        mt.inverse(_t([1.0, 2.0]))
    with pytest.raises(ValueError, match="square"):
        mt.inverse(_t(RNG.standard_normal((2, 3))))


def test_pinverse_matches_numpy_on_a_full_rank_matrix():
    tall = RNG.standard_normal((5, 3))
    np.testing.assert_allclose(
        mt.pinverse(_t(tall)).numpy(), np.linalg.pinv(tall), rtol=1e-10, atol=1e-12
    )


def test_pinverse_drops_a_direction_the_matrix_does_not_span():
    # A rank-1 matrix has one singular value and one at the noise floor;
    # inverting the second would amplify noise without bound, and dropping it
    # is what makes this a pseudo-inverse.
    outer = np.outer([1.0, 2.0, 3.0], [1.0, -1.0])
    got = mt.pinverse(_t(outer)).numpy()
    np.testing.assert_allclose(got, np.linalg.pinv(outer), rtol=1e-8, atol=1e-12)
    assert np.isfinite(got).all()

    # And the defining property: A @ A+ @ A == A.
    np.testing.assert_allclose(outer @ got @ outer, outer, atol=1e-12)


def test_pinverse_requires_a_matrix():
    with pytest.raises(ValueError, match="at least two dimensions"):
        mt.pinverse(_t([1.0, 2.0]))


def test_inverse_and_pinverse_take_a_stack():
    # Both used to be second implementations of `inv` and `pinv` written for a
    # single matrix. `pinverse` said so and refused a stack; `inverse` promised
    # one in its docstring and then built its identity flat, so every batched
    # call raised a shape mismatch instead.
    square = RNG.standard_normal((5, 4, 4))
    np.testing.assert_allclose(
        mt.inverse(_t(square)).numpy(), np.linalg.inv(square), rtol=1e-12, atol=1e-14
    )
    tall = RNG.standard_normal((2, 3, 6, 4))
    np.testing.assert_allclose(
        mt.pinverse(_t(tall)).numpy(), np.linalg.pinv(tall), rtol=1e-9, atol=1e-12
    )


def test_logdet_matches_numpy_and_stays_finite_where_det_does_not():
    np.testing.assert_allclose(
        mt.logdet(_t(A)).item(), np.log(np.linalg.det(A)), rtol=1e-13
    )

    # A determinant far past float64's range, whose logarithm is ordinary.
    big = np.eye(200) * 100.0
    got = mt.logdet(_t(big)).item()
    assert np.isfinite(got)
    assert got == pytest.approx(200 * np.log(100.0), rel=1e-12)
    # And the determinant itself genuinely cannot be represented: its logarithm
    # is past the logarithm of the largest float64 there is.
    assert got > np.log(np.finfo(np.float64).max)


def test_logdet_is_negative_infinity_where_the_determinant_is_not_positive():
    negative = np.array([[0.0, 1.0], [1.0, 0.0]])  # determinant -1
    singular = np.array([[1.0, 2.0], [2.0, 4.0]])  # determinant 0
    assert mt.logdet(_t(negative)).item() == -np.inf
    assert mt.logdet(_t(singular)).item() == -np.inf


# --- rescaling --------------------------------------------------------------


def test_renorm_scales_only_the_rows_that_are_too_long():
    rows = np.array([[3.0, 4.0], [0.3, 0.4], [0.0, 0.0]])
    got = mt.renorm(_t(rows), 2.0, 0, 1.0).numpy()

    # The long row is scaled to exactly the limit.
    assert np.linalg.norm(got[0]) == pytest.approx(1.0, rel=1e-14)
    # The short one comes back bit-for-bit, not rescaled by a factor of one.
    np.testing.assert_array_equal(got[1], rows[1])
    # And a zero row stays zero rather than becoming NaN.
    np.testing.assert_array_equal(got[2], rows[2])


@pytest.mark.parametrize("p", [1.0, 2.0, 3.0])
def test_renorm_uses_the_norm_it_is_given(p):
    rows = np.array([[3.0, 4.0], [1.0, 1.0]])
    got = mt.renorm(_t(rows), p, 0, 1.0).numpy()
    for row, original in zip(got, rows):
        norm = np.sum(np.abs(row) ** p) ** (1 / p)
        assert norm <= 1.0 + 1e-12
        if np.sum(np.abs(original) ** p) ** (1 / p) > 1.0:
            assert norm == pytest.approx(1.0, rel=1e-12)


def test_renorm_works_along_a_column_as_well_as_a_row():
    columns = np.array([[3.0, 0.3], [4.0, 0.4]])
    got = mt.renorm(_t(columns), 2.0, 1, 1.0).numpy()
    assert np.linalg.norm(got[:, 0]) == pytest.approx(1.0, rel=1e-14)
    np.testing.assert_array_equal(got[:, 1], columns[:, 1])


def test_renorm_refuses_a_negative_limit():
    with pytest.raises(ValueError, match="non-negative maxnorm"):
        mt.renorm(_t(A), 2.0, 0, -1.0)


def test_vander_matches_numpy():
    values = np.array([1.0, 2.0, 3.0, 5.0])
    for n in (None, 2, 6):
        for increasing in (False, True):
            np.testing.assert_allclose(
                mt.vander(_t(values), n, increasing).numpy(),
                np.vander(values, n, increasing),
                rtol=1e-13,
            )


def test_vander_evaluates_a_polynomial_when_multiplied_by_its_coefficients():
    # The reason the columns descend by default.
    x = np.array([0.5, 1.0, 2.0])
    coefficients = np.array([3.0, -1.0, 2.0])  # 3x^2 - x + 2
    got = mt.mv(mt.vander(_t(x)), _t(coefficients)).numpy()
    np.testing.assert_allclose(got, 3 * x**2 - x + 2, rtol=1e-14)


def test_vander_requires_a_vector():
    with pytest.raises(ValueError, match="1-D tensor"):
        mt.vander(_t(A))


# --- the real-valued answers to complex questions ---------------------------


def test_real_and_conj_are_the_input():
    values = _t(A)
    np.testing.assert_array_equal(mt.real(values).numpy(), A)
    np.testing.assert_array_equal(mt.conj(values).numpy(), A)


def test_imag_is_zero_everywhere_including_at_an_infinity():
    values = np.array([1.0, np.inf, -np.inf, np.nan, -0.0])
    np.testing.assert_array_equal(mt.imag(_t(values)).numpy(), np.imag(values))
    # `input * 0` would have answered NaN for the two infinities.
    assert not np.isnan(mt.imag(_t(values)).numpy()).any()


def test_angle_matches_numpy_including_at_negative_zero():
    values = np.array([-1.0, 0.0, -0.0, 2.0, np.nan, -np.inf, np.inf])
    np.testing.assert_array_equal(mt.angle(_t(values)).numpy(), np.angle(values))
    # The case that needs the sign bit rather than a comparison.
    assert mt.angle(_t([-0.0])).item() == pytest.approx(np.pi)
    assert (_t([-0.0]) < 0).numpy()[0] == False  # noqa: E712


def test_the_piecewise_constants_carry_no_gradient():
    # `angle` and `imag` do not vary with their input, and saying so beats
    # claiming a gradient that `backward()` would then fail to deliver.
    values = _t(A, requires_grad=True)
    assert not mt.angle(values).requires_grad
    assert not mt.imag(values).requires_grad


# --- gradients --------------------------------------------------------------


def test_the_matmuls_carry_gradients():
    left = _t(A, requires_grad=True)
    right = _t(B, requires_grad=True)
    mt.mm(left, right).sum().backward()
    # d/dA of sum(A @ B) is the row sums of B, broadcast.
    np.testing.assert_allclose(left.grad.numpy(), np.tile(B.sum(axis=1), (2, 1)))
    np.testing.assert_allclose(right.grad.numpy(), np.tile(A.sum(axis=0), (2, 1)).T)
    mt.clear_autograd_graph()


def test_tensordot_carries_a_gradient_through_the_rearrangement():
    a = RNG.standard_normal((2, 3, 4))
    b = RNG.standard_normal((4, 5))
    left = _t(a, requires_grad=True)
    mt.tensordot(left, _t(b), 1).sum().backward()
    # Each element of `a` meets one row of `b`, so its gradient is that row's
    # sum.
    np.testing.assert_allclose(
        left.grad.numpy(), np.broadcast_to(b.sum(axis=1), (2, 3, 4)), rtol=1e-13
    )
    mt.clear_autograd_graph()


def test_renorm_leaves_the_untouched_rows_with_a_gradient_of_one():
    rows = _t([[0.3, 0.4], [3.0, 4.0]], requires_grad=True)
    mt.renorm(rows, 2.0, 0, 1.0).sum().backward()
    np.testing.assert_allclose(rows.grad.numpy()[0], [1.0, 1.0])
    mt.clear_autograd_graph()


# --- linear systems over more than two axes ---------------------------------
#
# `tensorsolve` and `tensorinv` are `solve` and `inverse` with a reshape on each
# side, and NumPy has both under the same names -- so the reshape is what the
# tests check, against `np.linalg`.


@pytest.mark.parametrize(
    "coefficients,rhs",
    [((2, 3, 6), (2, 3)), ((4, 4), (4,)), ((2, 2, 2, 8), (2, 2, 2))],
)
def test_tensorsolve_matches_numpy(coefficients, rhs):
    a = RNG.standard_normal(coefficients)
    b = RNG.standard_normal(rhs)
    np.testing.assert_allclose(
        mt.tensorsolve(_t(a), _t(b)).numpy(), np.linalg.tensorsolve(a, b), rtol=1e-9
    )


def test_tensorsolve_moves_the_axes_it_is_given_to_the_end():
    a = RNG.standard_normal((3, 2, 6))
    b = RNG.standard_normal((2, 3))
    np.testing.assert_allclose(
        mt.tensorsolve(_t(a), _t(b), axes=(2,)).numpy(),
        np.linalg.tensorsolve(a, b, axes=(2,)),
        rtol=1e-9,
    )


def test_the_solution_of_tensorsolve_solves_the_system_it_names():
    """Checked against the contraction rather than against another solver."""

    a = RNG.standard_normal((2, 3, 6))
    b = RNG.standard_normal((2, 3))
    x = mt.tensorsolve(_t(a), _t(b))
    np.testing.assert_allclose(
        mt.tensordot(_t(a), x, 1).numpy(), b, rtol=1e-8, atol=1e-10
    )


def test_tensorsolve_carries_a_gradient_to_both_operands():
    a = _t(RNG.standard_normal((2, 3, 6)), requires_grad=True)
    b = _t(RNG.standard_normal((2, 3)), requires_grad=True)
    mt.tensorsolve(a, b).sum().backward()
    assert np.isfinite(a.grad.numpy()).all()
    assert np.isfinite(b.grad.numpy()).all()
    mt.clear_autograd_graph()


@pytest.mark.parametrize("shape,split", [((4, 6, 8, 3), 2), ((9, 3, 3), 1)])
def test_tensorinv_matches_numpy(shape, split):
    a = RNG.standard_normal(shape)
    np.testing.assert_allclose(
        mt.tensorinv(_t(a), split).numpy(), np.linalg.tensorinv(a, split), rtol=1e-8
    )


def test_tensorinv_contracts_back_to_the_identity():
    a = RNG.standard_normal((4, 6, 8, 3))
    contracted = mt.tensordot(mt.tensorinv(_t(a), 2), _t(a), 2).numpy()
    np.testing.assert_allclose(
        contracted.reshape(24, 24), np.eye(24), rtol=0, atol=1e-8
    )


def test_tensorsolve_reports_a_system_that_is_not_square():
    with pytest.raises(ValueError, match="square system"):
        mt.tensorsolve(
            _t(RNG.standard_normal((2, 3, 5))), _t(RNG.standard_normal((2, 3)))
        )


def test_tensorsolve_reports_a_repeated_axis():
    with pytest.raises(ValueError, match="repeated axis"):
        mt.tensorsolve(
            _t(RNG.standard_normal((2, 3, 6))),
            _t(RNG.standard_normal((2, 3))),
            axes=(1, 1),
        )


def test_tensorinv_reports_halves_that_do_not_match():
    with pytest.raises(ValueError, match="same number"):
        mt.tensorinv(_t(RNG.standard_normal((4, 6, 5))), 2)


@pytest.mark.parametrize("split", [0, 4, -1])
def test_tensorinv_refuses_a_split_outside_the_tensor(split):
    """Zero would leave nothing on the left, and the rank nothing on the right."""

    with pytest.raises(ValueError, match="strictly inside"):
        mt.tensorinv(_t(RNG.standard_normal((4, 6, 8, 3))), split)
