# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""Every other factorisation here needs something of its matrix.

`cholesky` needs positive definiteness, `eigh` symmetry, `solve` and `det`
squareness; `qr` takes anything but only describes its columns in the order they
arrived. The singular value decomposition asks for nothing and answers the
questions that do not depend on the matrix being any of those: how far it is
from a lower rank, which directions it stretches and by how much, what its rank
is when the entries are inexact, and what the least-squares solution is when the
columns are dependent. The last group of tests does pseudo-inverse, rank,
condition number, least squares and low-rank truncation end to end, because
those are what the gap actually blocked.

It is emphatically not `eigh` applied to `A.T @ A`. That identity is a proof and
not an algorithm -- forming `A.T @ A` squares the condition number, so a
singular value at `1e-9` times the largest comes back from the squared problem
with no correct digits at all, and that is exactly the singular value a rank
test is asking about. `test_small_singular_values_survive` is the measurement,
and it fails by four orders of magnitude if the work is ever routed through the
squared matrix.

`U` and `V` are determined only up to the sign of each column, and within a
repeated singular value's subspace only up to a rotation. So the tests compare
what they *do* -- `A == U @ diag(s) @ Vh`, `U.T @ U == I` -- or compare a
sign-invariant function of them. Comparing against NumPy's factors elementwise
would fail for a reason that is not a bug. The singular values themselves are
determined exactly, so those are compared directly.
"""

from __future__ import annotations

import numpy as np
import pytest

import minitensor as mt

SHAPES = [
    (1, 1),
    (2, 2),
    (3, 3),
    (5, 5),
    (8, 8),
    (5, 3),
    (3, 5),
    (9, 2),
    (2, 9),
    (2, 4, 3),
    (2, 3, 4, 4),
]

TALL_AND_WIDE = [(5, 3), (3, 5), (9, 2), (2, 9), (7, 4), (4, 7)]


def _matrix(shape, seed=0):
    return np.random.default_rng(seed).standard_normal(shape)


def _reconstruct(u, s, vt):
    """`U @ diag(s) @ Vh`, batched, for whatever shapes came back."""
    k = s.shape[-1]
    return u[..., :, :k] @ (s[..., :, None] * vt[..., :k, :])


def _call(a, full_matrices=True):
    u, s, vt = mt.svd(mt.Tensor.from_numpy(np.ascontiguousarray(a)), full_matrices)
    return u.numpy(), s.numpy(), vt.numpy()


# --------------------------------------------------------------------------
# The defining property
# --------------------------------------------------------------------------


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("full_matrices", [True, False])
def test_reconstructs_the_matrix(shape, full_matrices):
    a = _matrix(shape)
    u, s, vt = _call(a, full_matrices)
    assert np.allclose(_reconstruct(u, s, vt), a, atol=1e-12)


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("full_matrices", [True, False])
def test_factors_are_orthonormal(shape, full_matrices):
    a = _matrix(shape, seed=1)
    u, _, vt = _call(a, full_matrices)
    left = np.swapaxes(u, -1, -2) @ u
    right = vt @ np.swapaxes(vt, -1, -2)
    assert np.allclose(left, np.eye(u.shape[-1]), atol=1e-12)
    assert np.allclose(right, np.eye(vt.shape[-2]), atol=1e-12)


@pytest.mark.parametrize("shape", SHAPES)
def test_singular_values_match_numpy(shape):
    a = _matrix(shape, seed=2)
    _, s, _ = _call(a)
    assert np.allclose(s, np.linalg.svd(a, compute_uv=False), atol=1e-12)


@pytest.mark.parametrize("shape", SHAPES)
def test_singular_values_are_descending_and_non_negative(shape):
    a = _matrix(shape, seed=3)
    _, s, _ = _call(a)
    assert (s >= 0).all()
    assert (np.diff(s, axis=-1) <= 1e-12).all()


# --------------------------------------------------------------------------
# Shapes
# --------------------------------------------------------------------------


@pytest.mark.parametrize("shape", SHAPES)
def test_full_matrices_shapes(shape):
    m, n = shape[-2], shape[-1]
    batch = shape[:-2]
    u, s, vt = _call(_matrix(shape), True)
    assert u.shape == batch + (m, m)
    assert s.shape == batch + (min(m, n),)
    assert vt.shape == batch + (n, n)


@pytest.mark.parametrize("shape", SHAPES)
def test_reduced_shapes(shape):
    m, n = shape[-2], shape[-1]
    batch, k = shape[:-2], min(m, n)
    u, s, vt = _call(_matrix(shape), False)
    assert u.shape == batch + (m, k)
    assert s.shape == batch + (k,)
    assert vt.shape == batch + (k, n)


def test_full_matrices_defaults_to_true():
    a = _matrix((5, 3))
    default = mt.svd(mt.Tensor.from_numpy(a))[0].numpy()
    assert default.shape == (5, 5)


@pytest.mark.parametrize("shape", TALL_AND_WIDE)
def test_reduced_is_the_leading_block_of_full(shape):
    """The reduced factors are the full ones cut down, not a different answer.

    A wide matrix is factored by transposing it, so this is the check that the
    transposed path lands its two matrices in the right outputs the right way
    round -- getting that wrong produces factors that are individually
    orthonormal and reconstruct nothing.
    """
    a = _matrix(shape, seed=4)
    k = min(shape)
    full = _call(a, True)
    reduced = _call(a, False)
    assert np.allclose(np.abs(full[0][:, :k]), np.abs(reduced[0]), atol=1e-12)
    assert np.allclose(full[1], reduced[1], atol=1e-12)
    assert np.allclose(np.abs(full[2][:k, :]), np.abs(reduced[2]), atol=1e-12)


@pytest.mark.parametrize("shape", TALL_AND_WIDE)
def test_transposing_transposes_the_factorisation(shape):
    """`A.T = V diag(s) U.T`, which pins the wide path against the tall one."""
    a = _matrix(shape, seed=5)
    _, s, _ = _call(a, False)
    _, s_t, _ = _call(np.ascontiguousarray(a.T), False)
    assert np.allclose(s, s_t, atol=1e-12)


# --------------------------------------------------------------------------
# The hard cases
# --------------------------------------------------------------------------


def test_small_singular_values_survive():
    """The reason this is not `eigh(A.T @ A)`.

    A matrix with singular values spread over nine orders of magnitude has every
    one of them comfortably representable. Squaring spreads them over eighteen,
    which is most of what a double can hold, and the smallest does not survive
    the trip: `eigh` of the squared matrix returns it as *exactly zero*, a
    relative error of one, while the same value comes back here to eight digits.

    Eight and not sixteen, and that is not a shortfall to fix -- reducing `A` to
    bidiagonal form is accurate in absolute terms, to rounding times the largest
    singular value, so a value a billion times smaller than that one inherits a
    relative error a billion times larger. LAPACK is bound by the same argument
    and lands in the same place, which is what the comparison below actually
    measures: not a tolerance someone picked, but agreement with the best
    available answer.
    """
    n = 12
    rng = np.random.default_rng(6)
    left = np.linalg.qr(rng.standard_normal((n, n)))[0]
    right = np.linalg.qr(rng.standard_normal((n, n)))[0]
    expected = np.logspace(0, -9, n)
    a = left @ np.diag(expected) @ right.T

    _, s, _ = _call(a)
    reference = np.linalg.svd(a, compute_uv=False)
    squared = np.sqrt(np.maximum(np.linalg.eigvalsh(a.T @ a)[::-1], 0.0))

    # Every value to eight digits, and no worse than LAPACK on any of them.
    assert np.allclose(s / expected, 1.0, rtol=1e-7)
    assert np.abs(s / expected - 1).max() < 2 * np.abs(reference / expected - 1).max()
    # The squared route loses the smallest one entirely.
    assert squared[-1] == 0.0
    assert s[-1] > 0.5 * expected[-1]


def test_rank_deficient_matrix():
    """An exact zero singular value is not an error; it is the answer."""
    rng = np.random.default_rng(7)
    a = rng.standard_normal((6, 3)) @ rng.standard_normal((3, 8))
    u, s, vt = _call(a, False)
    assert np.allclose(_reconstruct(u, s, vt), a, atol=1e-12)
    assert np.allclose(s[3:], 0.0, atol=1e-13)
    assert (s[:3] > 1e-3).all()


def test_exactly_zero_matrix():
    a = np.zeros((4, 6))
    u, s, vt = _call(a)
    assert np.allclose(s, 0.0)
    assert np.allclose(np.swapaxes(u, -1, -2) @ u, np.eye(4), atol=1e-13)
    assert np.allclose(vt @ vt.T, np.eye(6), atol=1e-13)


def test_repeated_singular_values():
    """The identity has every singular value equal, so the factors are free."""
    a = np.eye(5)
    u, s, vt = _call(a)
    assert np.allclose(s, 1.0, atol=1e-13)
    assert np.allclose(_reconstruct(u, s, vt), a, atol=1e-13)


def test_diagonal_matrix_recovers_its_entries():
    a = np.diag([3.0, -1.0, 2.0, 0.5])
    _, s, _ = _call(a)
    assert np.allclose(s, [3.0, 2.0, 1.0, 0.5], atol=1e-13)


def test_zero_on_the_diagonal_of_the_band():
    """A structurally singular matrix, which is the branch the shift cannot take.

    An upper bidiagonal matrix with a zero on its diagonal is where the shifted
    sweep breaks down -- the shift is formed by dividing by that entry. It is
    handled by rotating the row out of the band instead, and this is a matrix
    that goes straight there.
    """
    a = np.array(
        [
            [0.0, 2.0, 0.0, 0.0],
            [0.0, 3.0, 4.0, 0.0],
            [0.0, 0.0, 5.0, 6.0],
            [0.0, 0.0, 0.0, 7.0],
        ]
    )
    u, s, vt = _call(a, False)
    assert np.allclose(s, np.linalg.svd(a, compute_uv=False), atol=1e-12)
    assert np.allclose(_reconstruct(u, s, vt), a, atol=1e-12)


def test_wide_range_of_magnitudes():
    """Entries near the top of the range, where the shift's squares would
    overflow if the band were not scaled first."""
    a = _matrix((6, 4), seed=8) * 1e150
    _, s, _ = _call(a)
    assert np.isfinite(s).all()
    assert np.allclose(s, np.linalg.svd(a, compute_uv=False), rtol=1e-12)


def test_tiny_entries():
    a = _matrix((6, 4), seed=9) * 1e-150
    _, s, _ = _call(a)
    assert np.allclose(s, np.linalg.svd(a, compute_uv=False), rtol=1e-12)


@pytest.mark.parametrize("n", [1, 2, 3, 4, 7, 16, 31])
def test_hilbert_matrix(n):
    """The standard badly conditioned test matrix, up to `1e18`."""
    i = np.arange(1, n + 1)
    a = 1.0 / (i[:, None] + i[None, :] - 1)
    u, s, vt = _call(a)
    assert np.allclose(_reconstruct(u, s, vt), a, atol=1e-14)
    assert np.allclose(s, np.linalg.svd(a, compute_uv=False), atol=1e-15)


@pytest.mark.parametrize("n", [10, 25, 40])
def test_larger_random_matrices(n):
    a = _matrix((n, n), seed=n)
    u, s, vt = _call(a)
    assert np.allclose(_reconstruct(u, s, vt), a, atol=1e-11)
    assert np.allclose(s, np.linalg.svd(a, compute_uv=False), atol=1e-11)


def test_blocked_reduction_path():
    """Large enough that the reduction combines its reflectors into one operator.

    The crossover is at 64 rows and 64 trailing columns, so everything above
    goes down a different code path from everything else in this file.
    """
    a = _matrix((80, 70), seed=11)
    u, s, vt = _call(a, False)
    assert np.allclose(_reconstruct(u, s, vt), a, atol=1e-11)
    assert np.allclose(s, np.linalg.svd(a, compute_uv=False), atol=1e-11)
    assert np.allclose(u.T @ u, np.eye(70), atol=1e-11)


# --------------------------------------------------------------------------
# svdvals
# --------------------------------------------------------------------------


@pytest.mark.parametrize("shape", SHAPES)
def test_svdvals_matches_svd(shape):
    a = _matrix(shape, seed=12)
    assert np.allclose(
        mt.svdvals(mt.Tensor.from_numpy(a)).numpy(), _call(a)[1], atol=1e-13
    )


def test_svdvals_is_differentiable():
    a = np.array([[3.0, 1.0], [0.0, 2.0]])
    t = mt.Tensor.from_numpy(a, requires_grad=True)
    mt.svdvals(t).sum().backward()
    assert t.grad is not None
    assert np.isfinite(t.grad.numpy()).all()


# --------------------------------------------------------------------------
# dtypes and degenerate inputs
# --------------------------------------------------------------------------


def test_float32():
    a = _matrix((6, 4), seed=13).astype(np.float32)
    u, s, vt = _call(a)
    assert u.dtype == np.float32 and s.dtype == np.float32 and vt.dtype == np.float32
    assert np.allclose(s, np.linalg.svd(a, compute_uv=False), atol=1e-5)
    assert np.allclose(_reconstruct(u, s, vt), a, atol=1e-5)


def test_empty_batch():
    a = np.zeros((0, 4, 3))
    u, s, vt = _call(a, False)
    assert u.shape == (0, 4, 3) and s.shape == (0, 3) and vt.shape == (0, 3, 3)


@pytest.mark.parametrize("shape", [(3, 0), (0, 5), (0, 0)])
@pytest.mark.parametrize("full_matrices", [True, False])
def test_zero_extent_matrix(shape, full_matrices):
    """A matrix with no rows or no columns has no singular values.

    The two factors still have a shape, and still have to be orthogonal once
    they have one: a caller who asked for `full_matrices` asked for a basis, and
    zeros are not a basis. The first version of this returned zeros, which is
    why the orthogonality is asserted here rather than assumed.
    """
    a = np.zeros(shape)
    u, s, vt = _call(a, full_matrices)
    reference = np.linalg.svd(a, full_matrices=full_matrices)
    assert (u.shape, s.shape, vt.shape) == tuple(x.shape for x in reference)
    if u.size:
        assert np.allclose(u.T @ u, np.eye(u.shape[-1]), atol=1e-13)
    if vt.size:
        assert np.allclose(vt @ vt.T, np.eye(vt.shape[-2]), atol=1e-13)


@pytest.mark.parametrize("bad", [np.zeros(4), np.zeros(())])
def test_requires_two_dimensions(bad):
    with pytest.raises(Exception):
        mt.svd(mt.Tensor.from_numpy(bad))


def test_rejects_integer_dtype():
    with pytest.raises(Exception):
        mt.svd(mt.Tensor.from_numpy(np.eye(3, dtype=np.int64)))


def test_one_by_one():
    u, s, vt = _call(np.array([[-4.0]]))
    assert np.allclose(s, [4.0])
    assert np.allclose(u @ np.diag(s) @ vt, [[-4.0]])


# --------------------------------------------------------------------------
# Gradients
# --------------------------------------------------------------------------


def _numeric_grad(f, a, eps=1e-6):
    grad = np.zeros_like(a)
    flat = a.reshape(-1)
    for index in range(flat.size):
        original = flat[index]
        flat[index] = original + eps
        high = f(a)
        flat[index] = original - eps
        low = f(a)
        flat[index] = original
        grad.reshape(-1)[index] = (high - low) / (2 * eps)
    return grad


def _grad_of(a, loss):
    t = mt.Tensor.from_numpy(np.ascontiguousarray(a), requires_grad=True)
    u, s, vt = t.svd(False)
    loss(u, s, vt).backward()
    return t.grad.numpy()


@pytest.mark.parametrize("shape", [(3, 3), (4, 4), (5, 3), (3, 5), (6, 2)])
def test_singular_value_gradient(shape):
    """`d(sum s) / dA = U @ Vh`, which is also the subgradient of the nuclear
    norm and is the one term with no division in it."""
    a = _matrix(shape, seed=14)
    got = _grad_of(a, lambda u, s, vt: s.sum())
    u, _, vt = _call(a, False)
    assert np.allclose(got, u @ vt, atol=1e-11)


@pytest.mark.parametrize("shape", [(3, 3), (4, 4), (5, 3), (3, 5)])
def test_singular_value_gradient_against_finite_differences(shape):
    a = _matrix(shape, seed=15)

    def loss(matrix):
        return float(np.linalg.svd(matrix, compute_uv=False).sum())

    expected = _numeric_grad(loss, a.copy())
    got = _grad_of(a, lambda u, s, vt: s.sum())
    assert np.allclose(got, expected, atol=1e-6)


@pytest.mark.parametrize("shape", [(3, 3), (4, 4), (5, 3), (3, 5)])
def test_weighted_singular_value_gradient(shape):
    """Distinct weights, so the terms cannot cancel each other's mistakes."""
    a = _matrix(shape, seed=16)
    k = min(shape)
    weights = np.arange(1.0, k + 1)

    def loss(matrix):
        return float(np.linalg.svd(matrix, compute_uv=False) @ weights)

    expected = _numeric_grad(loss, a.copy())
    tensor_weights = mt.Tensor.from_numpy(weights)
    got = _grad_of(a, lambda u, s, vt: (s * tensor_weights).sum())
    assert np.allclose(got, expected, atol=1e-6)


@pytest.mark.parametrize("shape", [(3, 3), (4, 4), (5, 3), (3, 5)])
def test_vector_gradient_against_finite_differences(shape):
    """A loss on the factors themselves, which is where the coupling term and
    the `1 / (s_j^2 - s_i^2)` live.

    The loss is built to be invariant to the sign of each column, because the
    factors are only determined up to that and a loss that was not invariant
    would have no finite-difference derivative to compare against.
    """
    a = _matrix(shape, seed=17)
    k = min(shape)
    weights = np.arange(1.0, k + 1)

    def loss(matrix):
        u, _, vt = np.linalg.svd(matrix, full_matrices=False)
        return float(((u**2) @ weights).sum() + ((vt.T**2) @ weights).sum())

    expected = _numeric_grad(loss, a.copy())
    tensor_weights = mt.Tensor.from_numpy(weights)

    def tensor_loss(u, s, vt):
        left = ((u * u) @ tensor_weights).sum()
        v = vt.transpose(-1, -2)
        return left + ((v * v) @ tensor_weights).sum()

    got = _grad_of(a, tensor_loss)
    assert np.allclose(got, expected, atol=1e-6)


def test_all_three_gradients_together():
    """One backward through all three outputs sums three nodes' contributions."""
    a = _matrix((4, 3), seed=18)
    weights = np.arange(1.0, 4.0)

    def loss(matrix):
        u, s, vt = np.linalg.svd(matrix, full_matrices=False)
        return float(
            ((u**2) @ weights).sum() + s @ weights + ((vt.T**2) @ weights).sum()
        )

    expected = _numeric_grad(loss, a.copy())
    tensor_weights = mt.Tensor.from_numpy(weights)

    def tensor_loss(u, s, vt):
        v = vt.transpose(-1, -2)
        return (
            ((u * u) @ tensor_weights).sum()
            + (s * tensor_weights).sum()
            + ((v * v) @ tensor_weights).sum()
        )

    got = _grad_of(a, tensor_loss)
    assert np.allclose(got, expected, atol=1e-6)


def test_batched_gradient_matches_per_matrix():
    a = _matrix((3, 4, 4), seed=19)
    batched = _grad_of(a, lambda u, s, vt: s.sum())
    each = np.stack([_grad_of(a[i], lambda u, s, vt: s.sum()) for i in range(3)])
    assert np.allclose(batched, each, atol=1e-12)


def test_batched_vector_gradient_matches_per_matrix():
    """The coupling term and the reach outside the span, over a batch.

    `test_batched_gradient_matches_per_matrix` only exercises the singular value
    term, which is the one with no matrix products in it. This is the rest.
    """
    a = _matrix((3, 5, 3), seed=28)
    weights = mt.Tensor.from_numpy(np.arange(1.0, 4.0))

    def loss(u, s, vt):
        v = vt.transpose(-1, -2)
        return ((u * u) @ weights).sum() + ((v * v) @ weights).sum()

    batched = _grad_of(a, loss)
    each = np.stack([_grad_of(a[i], loss) for i in range(3)])
    assert np.allclose(batched, each, atol=1e-12)


def test_gradient_flows_through_a_chain():
    a = _matrix((4, 4), seed=20)
    t = mt.Tensor.from_numpy(a, requires_grad=True)
    doubled = t * mt.Tensor.from_numpy(np.full_like(a, 2.0))
    doubled.svd(False)[1].sum().backward()
    u, _, vt = _call(2 * a, False)
    assert np.allclose(t.grad.numpy(), 2 * (u @ vt), atol=1e-11)


def test_full_matrices_gradient_is_refused_when_rectangular():
    """The extra columns of a square `U` are an arbitrary basis completion.

    Nothing about `A` chooses them, so there is no derivative to report and
    saying so beats inventing one.
    """
    t = mt.Tensor.from_numpy(_matrix((5, 3)), requires_grad=True)
    with pytest.raises(Exception, match="full_matrices"):
        t.svd(True)


def test_full_matrices_gradient_is_allowed_when_square():
    t = mt.Tensor.from_numpy(_matrix((4, 4), seed=21), requires_grad=True)
    u, s, vt = t.svd(True)
    s.sum().backward()
    assert np.isfinite(t.grad.numpy()).all()


def test_repeated_singular_values_have_no_vector_gradient():
    """Not a defect. A repeated singular value leaves its subspace free to
    rotate, so no particular pair of columns is determined and there is no
    derivative -- while the *singular value* gradient stays perfectly well
    defined there, which is the second half of this test.
    """
    a = np.eye(3)
    t = mt.Tensor.from_numpy(a, requires_grad=True)
    u, _, _ = t.svd(False)
    (u * u).sum().backward()
    assert not np.isfinite(t.grad.numpy()).all()

    t = mt.Tensor.from_numpy(a, requires_grad=True)
    t.svd(False)[1].sum().backward()
    assert np.isfinite(t.grad.numpy()).all()


def test_no_grad_when_not_required():
    u, s, vt = mt.Tensor.from_numpy(_matrix((4, 3))).svd(False)
    assert not u.requires_grad and not s.requires_grad and not vt.requires_grad


# --------------------------------------------------------------------------
# What the gap actually blocked
# --------------------------------------------------------------------------


def test_pseudo_inverse():
    """`pinv(A) = V diag(1/s) U.T` over the non-zero singular values.

    Solves least squares for a matrix with more rows than columns, which `solve`
    cannot take and `qr` can only manage when the columns are independent.
    """
    rng = np.random.default_rng(22)
    a = rng.standard_normal((9, 4))
    u, s, vt = _call(a, False)
    pinv = vt.T @ np.diag(1.0 / s) @ u.T

    target = rng.standard_normal(9)
    got = pinv @ target
    expected = np.linalg.lstsq(a, target, rcond=None)[0]
    assert np.allclose(got, expected, atol=1e-11)
    assert np.allclose(a @ pinv @ a, a, atol=1e-11)


def test_matrix_rank():
    """The count of singular values above the tolerance, which is the only
    numerically meaningful definition of rank for inexact entries."""
    rng = np.random.default_rng(23)
    a = rng.standard_normal((7, 5)) @ rng.standard_normal((5, 6))
    _, s, _ = _call(a)
    tolerance = max(a.shape) * np.finfo(float).eps * s[0]
    assert int((s > tolerance).sum()) == np.linalg.matrix_rank(a)


def test_condition_number():
    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    _, s, _ = _call(a)
    assert np.isclose(s[0] / s[-1], np.linalg.cond(a))


def test_low_rank_truncation_is_the_best_approximation():
    """Eckart-Young: dropping the smallest singular values is the closest
    matrix of that rank there is, and nothing but this factorisation finds it."""
    rng = np.random.default_rng(24)
    a = rng.standard_normal((8, 6))
    u, s, vt = _call(a, False)

    for rank in range(1, 6):
        approximation = u[:, :rank] @ np.diag(s[:rank]) @ vt[:rank, :]
        assert np.isclose(np.linalg.norm(a - approximation, 2), s[rank], atol=1e-11)
        for _ in range(20):
            left = rng.standard_normal((8, rank))
            right = rng.standard_normal((rank, 6))
            assert np.linalg.norm(a - left @ right, 2) >= s[rank] - 1e-11


def test_spectral_norm_matches():
    a = _matrix((6, 4), seed=25)
    _, s, _ = _call(a)
    assert np.isclose(s[0], np.linalg.norm(a, 2))
    assert np.isclose(np.sqrt((s**2).sum()), np.linalg.norm(a, "fro"))


def test_principal_components_of_a_data_matrix():
    """PCA without forming the covariance, which is the point.

    The right singular vectors of the centred data are its principal directions
    and `s**2 / (n - 1)` the variances along them -- the same answer `eigh` of
    the covariance gives, at half the condition number.
    """
    rng = np.random.default_rng(26)
    data = rng.standard_normal((200, 4)) @ np.diag([5.0, 3.0, 1.0, 0.2])
    centred = data - data.mean(axis=0)

    _, s, vt = _call(np.ascontiguousarray(centred), False)
    variances = s**2 / (centred.shape[0] - 1)

    covariance = np.cov(centred, rowvar=False)
    reference = np.linalg.eigvalsh(covariance)[::-1]
    assert np.allclose(variances, reference, atol=1e-10)

    projected = centred @ vt.T
    assert np.allclose(projected.T @ projected / (len(data) - 1), np.diag(variances), atol=1e-10)


def test_orthogonal_procrustes():
    """The nearest orthogonal matrix to `M` is `U @ Vh`, which is a one-line
    consequence of the decomposition and unreachable without it."""
    rng = np.random.default_rng(27)
    rotation = np.linalg.qr(rng.standard_normal((4, 4)))[0]
    noisy = rotation + 0.01 * rng.standard_normal((4, 4))

    u, _, vt = _call(noisy)
    nearest = u @ vt
    assert np.allclose(nearest @ nearest.T, np.eye(4), atol=1e-12)
    assert np.linalg.norm(nearest - rotation) < np.linalg.norm(noisy - rotation)
