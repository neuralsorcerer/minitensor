# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""LU with partial pivoting, and the triangular solves it feeds.

`A = P L U` is what almost every dense question about a general square matrix
reduces to: the determinant is `U`'s diagonal with a sign, solving `A X = B` is
two substitutions, the inverse is the same against the identity. This file
tests the factorisation itself, the four spellings of the triangular solve, and
that `det` and `solve` -- which used to carry an elimination each -- still
answer what they did once both were replaced by this one.

Two things are worth testing that a residual check alone would not catch. The
pivoting: without it a matrix as ordinary as `[[0, 1], [1, 0]]` divides by zero
at the first step, and one with a merely small leading entry produces
multipliers large enough to swamp the elimination. And the triangle: a
triangular solve must read only the half it was told to, and its gradient must
be zero in the half it did not read.
"""

from __future__ import annotations

import numpy as np
import pytest

import minitensor as mt


def _t(a, requires_grad=False):
    return mt.Tensor.from_numpy(
        np.ascontiguousarray(np.asarray(a, dtype=np.float64)),
        requires_grad=requires_grad,
    )


def _triangular(rng, n, upper, unit=False, strong=True):
    """A well-conditioned triangle, so a residual means what it says."""
    m = rng.normal(size=(n, n))
    a = np.triu(m) if upper else np.tril(m)
    if unit:
        np.fill_diagonal(a, 1.0)
    elif strong:
        diag = np.diag(a)
        np.fill_diagonal(a, np.where(diag >= 0, diag + n, diag - n))
    return a


# --------------------------------------------------------------------------
# The factorisation
# --------------------------------------------------------------------------


@pytest.mark.parametrize("n", [1, 2, 3, 5, 17, 64, 65, 100])
def test_the_factors_multiply_back_to_the_input(n):
    """`P @ L @ U == A`, which is the whole claim. The sizes straddle the
    panel width, where the blocked update takes over from the unblocked one."""
    rng = np.random.default_rng(n)
    a = rng.normal(size=(n, n))
    p, l, u = (x.numpy() for x in mt.lu(_t(a)))
    assert np.allclose(p @ l @ u, a, rtol=0, atol=1e-12 * max(np.abs(a).max(), 1))


def test_the_factorisation_is_batched():
    rng = np.random.default_rng(1)
    a = rng.normal(size=(4, 9, 9))
    p, l, u = (x.numpy() for x in mt.lu(_t(a)))
    assert np.allclose(p @ l @ u, a, atol=1e-12)


def test_l_is_unit_lower_and_u_is_upper():
    """Stated separately from the reconstruction: `P @ L @ U == A` would still
    hold if the two factors traded a column, and the shapes are what make the
    factorisation useful to substitute against."""
    rng = np.random.default_rng(2)
    a = rng.normal(size=(7, 7))
    _, l, u = (x.numpy() for x in mt.lu(_t(a)))
    assert np.array_equal(np.tril(l), l)
    assert np.array_equal(np.diag(l), np.ones(7))
    assert np.array_equal(np.triu(u), u)


def test_p_is_a_permutation_matrix():
    rng = np.random.default_rng(3)
    p = mt.lu(_t(rng.normal(size=(6, 6))))[0].numpy()
    assert set(np.unique(p)) <= {0.0, 1.0}
    assert np.array_equal(p.sum(axis=0), np.ones(6))
    assert np.array_equal(p.sum(axis=1), np.ones(6))


def test_the_packed_form_holds_both_factors():
    """`lu_factor` returns the two triangles in one buffer -- `L` strictly
    below the diagonal, `U` on and above it -- which is what makes solving
    against it two substitutions and no unpacking."""
    rng = np.random.default_rng(4)
    a = rng.normal(size=(6, 6))
    packed = mt.lu_factor(_t(a))[0].numpy()
    _, l, u = (x.numpy() for x in mt.lu(_t(a)))
    assert np.array_equal(np.triu(packed), u)
    assert np.array_equal(np.tril(packed, -1), np.tril(l, -1))


def test_the_pivots_are_zero_based_int64_rows():
    rng = np.random.default_rng(5)
    _, pivots = mt.lu_factor(_t(rng.normal(size=(3, 8, 8))))
    assert pivots.dtype == "int64"
    assert pivots.numpy().shape == (3, 8)
    assert pivots.numpy().min() >= 0 and pivots.numpy().max() < 8


def test_the_identity_needs_no_exchanges():
    pivots = mt.lu_factor(_t(np.eye(5)))[1].numpy()
    assert np.array_equal(pivots, np.arange(5))


# --------------------------------------------------------------------------
# Why partial pivoting
# --------------------------------------------------------------------------


def test_a_zero_in_the_corner_is_not_a_problem():
    """The case that makes pivoting necessary rather than merely advisable:
    without an exchange the very first step divides by zero."""
    a = np.array([[0.0, 1.0], [1.0, 0.0]])
    p, l, u = (x.numpy() for x in mt.lu(_t(a)))
    assert np.allclose(p @ l @ u, a)
    x = mt.solve(_t(a), _t(np.array([[3.0], [4.0]]))).numpy()
    assert np.allclose(x, [[4.0], [3.0]])


def test_every_multiplier_is_at_most_one():
    """What pivoting buys, stated as the property it guarantees: the largest
    remaining entry goes on the diagonal, so nothing below it can exceed it."""
    rng = np.random.default_rng(6)
    packed = mt.lu_factor(_t(rng.normal(size=(40, 40))))[0].numpy()
    below = np.tril(packed, -1)
    assert np.abs(below).max() <= 1.0 + 1e-15


def test_a_tiny_leading_entry_does_not_lose_the_answer():
    """Without pivoting this is the textbook disaster: the multiplier is 1e16
    and the second equation's own coefficients are rounded away entirely."""
    a = np.array([[1e-17, 1.0], [1.0, 1.0]])
    b = np.array([[1.0], [2.0]])
    got = mt.solve(_t(a), _t(b)).numpy()
    assert np.allclose(got, np.linalg.solve(a, b), rtol=1e-12)


# --------------------------------------------------------------------------
# Singularity
# --------------------------------------------------------------------------


def test_a_singular_matrix_factors_but_does_not_solve():
    """The factorisation records the zero pivot rather than deciding what it
    means. `det` wants zero and `solve` wants an error, and both get what they
    want from the same run."""
    a = np.array([[1.0, 2.0], [2.0, 4.0]])
    packed, _ = mt.lu_factor(_t(a))
    assert np.min(np.abs(np.diag(packed.numpy()))) == 0.0
    assert mt.det(_t(a)).item() == 0.0
    with pytest.raises(Exception, match="singular"):
        mt.solve(_t(a), _t(np.array([[1.0], [1.0]])))


def test_a_zero_matrix_is_singular_rather_than_an_error_to_factor():
    packed, pivots = mt.lu_factor(_t(np.zeros((3, 3))))
    assert np.array_equal(packed.numpy(), np.zeros((3, 3)))
    assert np.array_equal(pivots.numpy(), np.arange(3))


# --------------------------------------------------------------------------
# lu_solve
# --------------------------------------------------------------------------


def test_lu_solve_agrees_with_solve():
    rng = np.random.default_rng(7)
    a = rng.normal(size=(11, 11))
    b = rng.normal(size=(11, 3))
    packed, pivots = mt.lu_factor(_t(a))
    assert np.allclose(
        mt.lu_solve(packed, pivots, _t(b)).numpy(),
        mt.solve(_t(a), _t(b)).numpy(),
        atol=1e-11,
    )


def test_one_factorisation_serves_many_right_hand_sides():
    """The reason the packed form is worth returning at all."""
    rng = np.random.default_rng(8)
    a = rng.normal(size=(9, 9))
    packed, pivots = mt.lu_factor(_t(a))
    for _ in range(5):
        b = rng.normal(size=(9, 2))
        x = mt.lu_solve(packed, pivots, _t(b)).numpy()
        assert np.allclose(a @ x, b, atol=1e-11)


def test_lu_solve_takes_a_vector_without_its_trailing_one():
    rng = np.random.default_rng(9)
    a = rng.normal(size=(5, 5))
    b = rng.normal(size=5)
    packed, pivots = mt.lu_factor(_t(a))
    x = mt.lu_solve(packed, pivots, _t(b)).numpy()
    assert x.shape == (5,)
    assert np.allclose(a @ x, b, atol=1e-12)


def test_lu_solve_is_batched():
    rng = np.random.default_rng(10)
    a = rng.normal(size=(3, 6, 6))
    b = rng.normal(size=(3, 6, 2))
    packed, pivots = mt.lu_factor(_t(a))
    x = mt.lu_solve(packed, pivots, _t(b)).numpy()
    assert np.allclose(a @ x, b, atol=1e-11)


# --------------------------------------------------------------------------
# solve_triangular
# --------------------------------------------------------------------------


@pytest.mark.parametrize("upper", [False, True])
@pytest.mark.parametrize("unit", [False, True])
def test_the_triangular_solve_residual(upper, unit):
    rng = np.random.default_rng(11)
    n = 12
    a = _triangular(rng, n, upper, unit)
    b = rng.normal(size=(n, 4))
    x = mt.solve_triangular(_t(a), _t(b), upper=upper, unitriangular=unit).numpy()
    assert np.allclose(a @ x, b, atol=1e-11)


def test_only_the_named_triangle_is_read():
    """A packed factorisation has the other factor in the half this is not
    solving against, so reading it would be a bug that shows up nowhere else."""
    rng = np.random.default_rng(12)
    a = _triangular(rng, 8, upper=False)
    b = rng.normal(size=(8, 2))
    clean = mt.solve_triangular(_t(a), _t(b)).numpy()

    littered = a.copy()
    littered[np.triu_indices(8, 1)] = rng.normal(size=8 * 7 // 2) * 1e6
    assert np.array_equal(mt.solve_triangular(_t(littered), _t(b)).numpy(), clean)


def test_unitriangular_ignores_the_diagonal_it_is_given():
    rng = np.random.default_rng(13)
    a = _triangular(rng, 6, upper=False, unit=True)
    b = rng.normal(size=(6, 2))
    want = mt.solve_triangular(_t(a), _t(b), unitriangular=True).numpy()

    lied = a.copy()
    np.fill_diagonal(lied, rng.normal(size=6) * 100)
    got = mt.solve_triangular(_t(lied), _t(b), unitriangular=True).numpy()
    assert np.array_equal(got, want)


def test_solving_from_the_right():
    """`X A = B` is `A^T X^T = B^T`, so it is the same routine on transposes --
    and getting the `upper` flag right through that flip is the whole content."""
    rng = np.random.default_rng(14)
    a = _triangular(rng, 7, upper=True)
    b = rng.normal(size=(3, 7))
    x = mt.solve_triangular(_t(a), _t(b), upper=True, left=False).numpy()
    assert np.allclose(x @ a, b, atol=1e-11)


def test_a_zero_on_the_diagonal_is_refused():
    a = np.array([[1.0, 0.0], [2.0, 0.0]])
    with pytest.raises(Exception, match="singular"):
        mt.solve_triangular(_t(a), _t(np.array([[1.0], [1.0]])))


def test_the_triangular_solve_is_batched():
    rng = np.random.default_rng(15)
    a = np.stack([_triangular(rng, 5, upper=False) for _ in range(4)])
    b = rng.normal(size=(4, 5, 2))
    x = mt.solve_triangular(_t(a), _t(b)).numpy()
    assert np.allclose(a @ x, b, atol=1e-11)


# --------------------------------------------------------------------------
# Its gradient
# --------------------------------------------------------------------------


def _numeric(fn, values, mask=None, step=1e-6):
    out = np.zeros_like(values)
    for index in np.ndindex(values.shape):
        if mask is not None and not mask[index]:
            continue
        up, down = values.copy(), values.copy()
        up[index] += step
        down[index] -= step
        out[index] = (fn(up) - fn(down)) / (2 * step)
    return out


@pytest.mark.parametrize("upper", [False, True])
@pytest.mark.parametrize("unit", [False, True])
def test_the_triangular_solve_gradient(upper, unit):
    rng = np.random.default_rng(16)
    n, k = 4, 2
    a = _triangular(rng, n, upper, unit)
    b = rng.normal(size=(n, k))
    weight = rng.normal(size=(n, k))

    def loss(matrix, rhs):
        x = mt.solve_triangular(
            _t(matrix), _t(rhs), upper=upper, unitriangular=unit
        ).numpy()
        return float((x * weight).sum())

    ta, tb = _t(a, True), _t(b, True)
    out = mt.solve_triangular(ta, tb, upper=upper, unitriangular=unit)
    (out * _t(weight)).sum().backward()

    keep = (
        np.triu(np.ones((n, n)), 1 if unit else 0)
        if upper
        else np.tril(np.ones((n, n)), -1 if unit else 0)
    )
    want_a = _numeric(lambda v: loss(v, b), a, keep)
    want_b = _numeric(lambda v: loss(a, v), b)
    scale = max(np.abs(want_a).max(), np.abs(want_b).max(), 1.0)
    assert np.allclose(ta.grad.numpy(), want_a, atol=1e-6 * scale)
    assert np.allclose(tb.grad.numpy(), want_b, atol=1e-6 * scale)


@pytest.mark.parametrize("upper", [False, True])
@pytest.mark.parametrize("unit", [False, True])
def test_the_untouched_triangle_gets_no_gradient(upper, unit):
    """The half that was never read cannot have moved the answer, so a gradient
    there would be telling a caller to change numbers that did nothing."""
    rng = np.random.default_rng(17)
    n = 5
    a = _triangular(rng, n, upper, unit)
    ta = _t(a, True)
    mt.solve_triangular(
        ta, _t(rng.normal(size=(n, 2))), upper=upper, unitriangular=unit
    ).sum().backward()
    got = ta.grad.numpy()
    dropped = (
        np.tril(np.ones((n, n)), 0 if unit else -1)
        if upper
        else np.triu(np.ones((n, n)), 0 if unit else 1)
    )
    assert np.all(got[dropped == 1] == 0.0)
    assert np.any(got[dropped == 0] != 0.0)


def test_the_factorisation_itself_carries_no_gradient():
    """A pivoted factorisation's derivative is not implemented here, and the
    docstring says where to go instead rather than leaving it to be found."""
    a = _t(np.array([[3.0, 1.0], [1.0, 2.0]]), requires_grad=True)
    packed, _ = mt.lu_factor(a)
    assert not packed.requires_grad
    assert not mt.lu(a)[1].requires_grad
    # The differentiable route through the same factorisation still works.
    assert mt.solve(a, _t(np.eye(2))).requires_grad


# --------------------------------------------------------------------------
# cholesky_solve
# --------------------------------------------------------------------------


def _spd(rng, n):
    m = rng.normal(size=(n, n))
    return m @ m.T + n * np.eye(n)


def test_cholesky_solve_residual():
    rng = np.random.default_rng(18)
    a = _spd(rng, 9)
    factor = np.linalg.cholesky(a)
    b = rng.normal(size=(9, 3))
    x = mt.cholesky_solve(_t(b), _t(factor)).numpy()
    assert np.allclose(a @ x, b, atol=1e-11)


def test_both_spellings_of_the_factor_agree():
    """`A = L L^T` inverts as `L^-T L^-1` and `A = U^T U` as `U^-1 U^-T`: the
    lower factor is applied first either way. Getting that backwards inverts
    `L^T L`, which is a different matrix and looks entirely plausible."""
    rng = np.random.default_rng(19)
    a = _spd(rng, 8)
    lower = np.linalg.cholesky(a)
    b = rng.normal(size=(8, 2))
    from_lower = mt.cholesky_solve(_t(b), _t(lower)).numpy()
    from_upper = mt.cholesky_solve(_t(b), _t(lower.T.copy()), upper=True).numpy()
    assert np.array_equal(from_lower, from_upper)
    assert np.allclose(a @ from_lower, b, atol=1e-11)


def test_cholesky_solve_is_differentiable_through_the_two_solves():
    """It has no kernel of its own and needs no gradient of its own: it is two
    `solve_triangular` calls, and the chain rule does the rest."""
    rng = np.random.default_rng(20)
    a = _spd(rng, 4)
    factor = np.linalg.cholesky(a)
    b = rng.normal(size=(4, 2))
    weight = rng.normal(size=(4, 2))

    def loss(rhs):
        return float(
            (mt.cholesky_solve(_t(rhs), _t(factor)).numpy() * weight).sum()
        )

    tb = _t(b, True)
    (mt.cholesky_solve(tb, _t(factor)) * _t(weight)).sum().backward()
    want = _numeric(loss, b)
    assert np.allclose(tb.grad.numpy(), want, atol=1e-6 * max(np.abs(want).max(), 1))


def test_cholesky_solve_is_batched():
    rng = np.random.default_rng(21)
    a = np.stack([_spd(rng, 5) for _ in range(3)])
    factor = np.linalg.cholesky(a)
    b = rng.normal(size=(3, 5, 2))
    x = mt.cholesky_solve(_t(b), _t(factor)).numpy()
    assert np.allclose(a @ x, b, atol=1e-11)


# --------------------------------------------------------------------------
# The callers that used to carry their own elimination
# --------------------------------------------------------------------------


@pytest.mark.parametrize("n", [1, 2, 7, 33])
def test_det_still_matches_numpy(n):
    rng = np.random.default_rng(100 + n)
    a = rng.normal(size=(2, n, n))
    got = mt.det(_t(a)).numpy()
    want = np.linalg.det(a)
    assert np.allclose(got, want, rtol=1e-10)


def test_slogdet_still_matches_numpy():
    rng = np.random.default_rng(22)
    a = rng.normal(size=(3, 12, 12))
    sign, logabs = (x.numpy() for x in mt.slogdet(_t(a)))
    want_sign, want_log = np.linalg.slogdet(a)
    assert np.array_equal(sign, want_sign)
    assert np.allclose(logabs, want_log, rtol=1e-11)


def test_det_and_slogdet_agree_about_singularity():
    a = np.array([[1.0, 2.0], [2.0, 4.0]])
    sign, logabs = (x.item() for x in mt.slogdet(_t(a)))
    assert mt.det(_t(a)).item() == 0.0
    assert sign == 0.0 and logabs == -np.inf


@pytest.mark.parametrize("n", [1, 4, 20])
def test_solve_still_matches_numpy(n):
    rng = np.random.default_rng(200 + n)
    a = rng.normal(size=(n, n)) + n * np.eye(n)
    b = rng.normal(size=(n, 3))
    assert np.allclose(
        mt.solve(_t(a), _t(b)).numpy(), np.linalg.solve(a, b), atol=1e-11
    )


def test_inv_still_matches_numpy():
    rng = np.random.default_rng(23)
    a = rng.normal(size=(6, 6)) + 6 * np.eye(6)
    assert np.allclose(mt.inv(_t(a)).numpy(), np.linalg.inv(a), atol=1e-12)


# --------------------------------------------------------------------------
# Shapes, dtypes and what it refuses
# --------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_both_floating_dtypes(dtype):
    rng = np.random.default_rng(24)
    a = (rng.normal(size=(6, 6)) + 6 * np.eye(6)).astype(dtype)
    packed, pivots = mt.lu_factor(mt.Tensor.from_numpy(a))
    assert packed.numpy().dtype == dtype
    assert pivots.dtype == "int64"
    b = rng.normal(size=(6, 2)).astype(dtype)
    x = mt.lu_solve(packed, pivots, mt.Tensor.from_numpy(b)).numpy()
    assert np.allclose(a.astype(np.float64) @ x, b, atol=1e-4 if dtype is np.float32 else 1e-11)


def test_the_matrices_must_be_square():
    with pytest.raises(Exception, match="square"):
        mt.lu_factor(_t(np.zeros((2, 3))))


def test_the_right_hand_side_must_line_up():
    rng = np.random.default_rng(25)
    packed, pivots = mt.lu_factor(_t(rng.normal(size=(4, 4)) + 4 * np.eye(4)))
    with pytest.raises(Exception):
        mt.lu_solve(packed, pivots, _t(np.zeros((5, 2))))


def test_the_pivots_must_be_the_ones_lu_factor_returned():
    rng = np.random.default_rng(26)
    a = rng.normal(size=(4, 4)) + 4 * np.eye(4)
    packed, _ = mt.lu_factor(_t(a))
    b = _t(np.zeros((4, 1)))
    with pytest.raises(Exception, match="int64"):
        mt.lu_solve(packed, _t(np.zeros(4)), b)
    with pytest.raises(Exception):
        mt.lu_solve(packed, mt.Tensor.from_numpy(np.zeros(3, dtype=np.int64)), b)
    with pytest.raises(Exception, match="not a row"):
        mt.lu_solve(packed, mt.Tensor.from_numpy(np.array([9, 1, 2, 3])), b)


def test_integer_matrices_are_refused():
    with pytest.raises(Exception, match="Float32 and Float64"):
        mt.lu_factor(mt.Tensor.from_numpy(np.eye(3, dtype=np.int64)))
