# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""The questions a singular value decomposition is usually asked.

`pinv`, `matrix_rank`, `cond` and `lstsq` are one line of arithmetic each on top
of `svd`, and none of them is reachable without it. `inv` needs a square
non-singular matrix and `solve` needs the same; `qr` needs full column rank.
These need nothing, which is the whole point of them.

`pinv` is tested against its definition rather than against NumPy alone. The
Moore-Penrose inverse is the unique matrix satisfying four conditions, so
checking those four checks the thing itself -- an implementation that agreed
with NumPy on the cases someone thought to write down but violated a Penrose
condition on some other matrix would be wrong in a way the comparison could
not see.

`matrix_power` is here for company rather than kinship: it is `inv` and `matmul`
rather than `svd`. What it shares is being a name for something a caller would
otherwise write out and get subtly wrong -- by multiplying `n` times rather than
`log n`, or by inverting the result of a large power instead of the base.
"""

from __future__ import annotations

import numpy as np
import pytest

import minitensor as mt

SHAPES = [(1, 1), (3, 3), (5, 5), (9, 4), (4, 9), (6, 2), (2, 6)]


def _matrix(shape, seed=0):
    return np.random.default_rng(seed).standard_normal(shape)


def _low_rank(rows, cols, rank, seed=0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((rows, rank)) @ rng.standard_normal((rank, cols))


def _t(a):
    return mt.Tensor.from_numpy(np.ascontiguousarray(a))


# --------------------------------------------------------------------------
# pinv against its definition
# --------------------------------------------------------------------------


@pytest.mark.parametrize("shape", SHAPES)
def test_pinv_satisfies_the_four_penrose_conditions(shape):
    """The definition, not a comparison. These four hold for exactly one matrix."""
    a = _matrix(shape, seed=1)
    p = mt.pinv(_t(a)).numpy()

    assert np.allclose(a @ p @ a, a, atol=1e-12)
    assert np.allclose(p @ a @ p, p, atol=1e-12)
    assert np.allclose((a @ p).T, a @ p, atol=1e-12)
    assert np.allclose((p @ a).T, p @ a, atol=1e-12)


@pytest.mark.parametrize("rank", [1, 2, 3])
def test_pinv_of_a_rank_deficient_matrix(rank):
    """Where `inv` has nothing to return and this still does."""
    a = _low_rank(6, 8, rank, seed=2)
    p = mt.pinv(_t(a)).numpy()

    assert np.allclose(a @ p @ a, a, atol=1e-11)
    assert np.allclose(p @ a @ p, p, atol=1e-11)
    assert np.allclose(p, np.linalg.pinv(a), atol=1e-11)


@pytest.mark.parametrize("shape", SHAPES)
def test_pinv_matches_numpy(shape):
    a = _matrix(shape, seed=3)
    assert np.allclose(mt.pinv(_t(a)).numpy(), np.linalg.pinv(a), atol=1e-12)


def test_pinv_of_an_invertible_matrix_is_the_inverse():
    a = _matrix((5, 5), seed=4)
    assert np.allclose(mt.pinv(_t(a)).numpy(), np.linalg.inv(a), atol=1e-11)


def test_pinv_shape_is_transposed():
    assert mt.pinv(_t(_matrix((7, 3)))).numpy().shape == (3, 7)
    assert mt.pinv(_t(_matrix((3, 7)))).numpy().shape == (7, 3)


def test_pinv_batched_matches_per_matrix():
    a = _matrix((4, 5, 3), seed=5)
    batched = mt.pinv(_t(a)).numpy()
    each = np.stack([mt.pinv(_t(a[i])).numpy() for i in range(4)])
    assert np.allclose(batched, each, atol=1e-13)


def test_pinv_of_the_zero_matrix_is_zero():
    """Every singular value is below any tolerance, so every direction is
    dropped and nothing is inverted. The alternative would be infinities."""
    assert np.allclose(mt.pinv(_t(np.zeros((4, 3)))).numpy(), 0.0)


def test_rcond_controls_which_directions_survive():
    """A deliberately tiny singular value, kept or dropped by the tolerance."""
    u = np.linalg.qr(_matrix((4, 4), seed=6))[0]
    v = np.linalg.qr(_matrix((4, 4), seed=7))[0]
    a = u @ np.diag([1.0, 0.5, 0.25, 1e-10]) @ v.T

    kept = mt.pinv(_t(a), rcond=1e-14).numpy()
    dropped = mt.pinv(_t(a), rcond=1e-6).numpy()

    # Keeping it inverts a value of 1e-10, so the result is enormous.
    assert np.abs(kept).max() > 1e9
    assert np.abs(dropped).max() < 1e3
    # Dropping it leaves a rank-3 pseudo-inverse, which still obeys Penrose
    # for the rank-3 matrix it is the inverse of.
    assert np.allclose(dropped @ a @ dropped, dropped, atol=1e-9)


def test_pinv_float32():
    a = _matrix((6, 4), seed=8).astype(np.float32)
    p = mt.pinv(_t(a)).numpy()
    assert p.dtype == np.float32
    assert np.allclose(a @ p @ a, a, atol=1e-4)


# --------------------------------------------------------------------------
# matrix_rank
# --------------------------------------------------------------------------


@pytest.mark.parametrize("rank", [1, 2, 3, 4])
def test_matrix_rank_of_a_known_rank(rank):
    a = _low_rank(7, 6, rank, seed=9)
    assert mt.matrix_rank(_t(a)).item() == rank
    assert mt.matrix_rank(_t(a)).item() == np.linalg.matrix_rank(a)


def test_matrix_rank_is_full_for_a_random_matrix():
    for shape in [(5, 5), (9, 4), (4, 9)]:
        assert mt.matrix_rank(_t(_matrix(shape, seed=10))).item() == min(shape)


def test_matrix_rank_of_zero_is_zero():
    assert mt.matrix_rank(_t(np.zeros((4, 6)))).item() == 0


def test_matrix_rank_returns_int64():
    assert mt.matrix_rank(_t(np.eye(3))).dtype == "int64"
    assert mt.matrix_rank(_t(np.eye(3))).numpy().dtype == np.int64


def test_matrix_rank_tolerance_is_absolute():
    """A matrix whose values straddle the tolerance, counted both ways."""
    a = np.diag([1.0, 0.1, 1e-8])
    assert mt.matrix_rank(_t(a)).item() == 3
    assert mt.matrix_rank(_t(a), tol=1e-6).item() == 2
    assert mt.matrix_rank(_t(a), tol=0.5).item() == 1


def test_matrix_rank_batched():
    a = np.stack([np.eye(4), _low_rank(4, 4, 2, seed=11), np.zeros((4, 4))])
    assert mt.matrix_rank(_t(a)).numpy().tolist() == [4, 2, 0]


def test_matrix_rank_near_the_default_tolerance():
    """The default is relative to the largest singular value, so scaling the
    whole matrix must not change the answer."""
    a = _low_rank(6, 6, 3, seed=12)
    for scale in (1e-8, 1.0, 1e8):
        assert mt.matrix_rank(_t(a * scale)).item() == 3


# --------------------------------------------------------------------------
# cond
# --------------------------------------------------------------------------


@pytest.mark.parametrize("shape", [(2, 2), (5, 5), (7, 4)])
def test_cond_matches_numpy(shape):
    a = _matrix(shape, seed=13)
    assert np.isclose(mt.cond(_t(a)).item(), np.linalg.cond(a))


def test_cond_of_an_orthogonal_matrix_is_one():
    q = np.linalg.qr(_matrix((5, 5), seed=14))[0]
    assert np.isclose(mt.cond(_t(q)).item(), 1.0, atol=1e-12)


def test_cond_of_a_singular_matrix_is_enormous():
    """Reported rather than raised: a caller comparing against a threshold
    should not have to catch an exception to learn the matrix is singular."""
    got = mt.cond(_t(np.array([[1.0, 2.0], [2.0, 4.0]]))).item()
    assert got > 1e15


def test_cond_grows_with_a_known_spread():
    """The tolerance widens with the spread, and has to.

    The smallest singular value is only accurate to `eps` times the largest, so
    a matrix conditioned at `1e10` determines its own condition number to about
    `1e-6` relative and no better. Asking for `1e-9` there would be asking the
    factorisation for accuracy the input does not contain.

    The same tolerance covers the comparison against NumPy, and for the same
    reason rather than as a convenience: NumPy is bound by the identical limit,
    so the two answers differ from each other by as much as either differs from
    the truth. At `1e10` they differ by `4e-9` and both sit `2e-7` from exact.
    """
    u = np.linalg.qr(_matrix((4, 4), seed=15))[0]
    v = np.linalg.qr(_matrix((4, 4), seed=16))[0]
    for spread in (1e2, 1e6, 1e10):
        a = u @ np.diag([spread, 10.0, 1.0, 1.0]) @ v.T
        tolerance = 10 * spread * np.finfo(float).eps
        got = mt.cond(_t(a)).item()
        assert np.isclose(got, spread, rtol=tolerance)
        assert np.isclose(got, np.linalg.cond(a), rtol=tolerance)


def test_cond_batched():
    a = np.stack([np.eye(3), np.diag([1.0, 1.0, 0.01])])
    assert np.allclose(mt.cond(_t(a)).numpy(), [1.0, 100.0])


# --------------------------------------------------------------------------
# lstsq
# --------------------------------------------------------------------------


@pytest.mark.parametrize("shape", [(9, 4), (5, 5), (4, 9)])
def test_lstsq_matches_numpy(shape):
    a = _matrix(shape, seed=17)
    b = _matrix((shape[0],), seed=18)
    got = mt.lstsq(_t(a), _t(b)).numpy()
    assert np.allclose(got, np.linalg.lstsq(a, b, rcond=None)[0], atol=1e-11)


def test_lstsq_minimises_the_residual():
    """The property, rather than the reference: no nearby `x` does better."""
    a = _matrix((10, 3), seed=19)
    b = _matrix((10,), seed=20)
    x = mt.lstsq(_t(a), _t(b)).numpy()
    best = np.linalg.norm(a @ x - b)

    rng = np.random.default_rng(21)
    for _ in range(50):
        nudged = x + 1e-3 * rng.standard_normal(3)
        assert np.linalg.norm(a @ nudged - b) >= best - 1e-12


def test_lstsq_takes_a_matrix_of_right_hand_sides():
    a = _matrix((8, 3), seed=22)
    b = _matrix((8, 4), seed=23)
    got = mt.lstsq(_t(a), _t(b)).numpy()
    assert got.shape == (3, 4)
    assert np.allclose(got, np.linalg.lstsq(a, b, rcond=None)[0], atol=1e-11)


def test_lstsq_returns_a_vector_for_a_vector():
    a = _matrix((6, 3), seed=24)
    assert mt.lstsq(_t(a), _t(_matrix((6,), seed=25))).numpy().shape == (3,)


def test_lstsq_picks_the_smallest_solution_when_there_are_many():
    """An underdetermined system has infinitely many exact solutions. This
    returns the one of least norm, which is what makes it well defined."""
    a = _matrix((3, 7), seed=26)
    b = _matrix((3,), seed=27)
    x = mt.lstsq(_t(a), _t(b)).numpy()

    assert np.allclose(a @ x, b, atol=1e-11)
    null = np.linalg.svd(a)[2][3:]
    for direction in null:
        assert np.linalg.norm(x + 0.1 * direction) >= np.linalg.norm(x) - 1e-12


def test_lstsq_on_a_rank_deficient_matrix():
    a = _low_rank(8, 5, 3, seed=28)
    b = _matrix((8,), seed=29)
    x = mt.lstsq(_t(a), _t(b)).numpy()
    assert np.allclose(x, np.linalg.lstsq(a, b, rcond=None)[0], atol=1e-10)


def test_lstsq_rejects_a_mismatched_right_hand_side():
    with pytest.raises(Exception):
        mt.lstsq(_t(_matrix((6, 3))), _t(_matrix((5,))))


def test_lstsq_batched():
    a = _matrix((3, 6, 2), seed=30)
    b = _matrix((3, 6, 1), seed=31)
    got = mt.lstsq(_t(a), _t(b)).numpy()
    for i in range(3):
        assert np.allclose(
            got[i], np.linalg.lstsq(a[i], b[i], rcond=None)[0], atol=1e-11
        )


# --------------------------------------------------------------------------
# matrix_power
# --------------------------------------------------------------------------


@pytest.mark.parametrize("power", [0, 1, 2, 3, 5, 8, 13, -1, -2, -5])
def test_matrix_power_matches_numpy(power):
    a = _matrix((4, 4), seed=32)
    got = mt.matrix_power(_t(a), power).numpy()
    assert np.allclose(got, np.linalg.matrix_power(a, power), atol=1e-9)


def test_matrix_power_zero_is_the_identity():
    got = mt.matrix_power(_t(_matrix((5, 5), seed=33)), 0).numpy()
    assert np.allclose(got, np.eye(5))


def test_matrix_power_one_is_the_matrix():
    a = _matrix((4, 4), seed=34)
    assert np.allclose(mt.matrix_power(_t(a), 1).numpy(), a)


def test_negative_power_is_the_inverse_of_the_positive_one():
    a = _matrix((4, 4), seed=35)
    forward = mt.matrix_power(_t(a), 3).numpy()
    backward = mt.matrix_power(_t(a), -3).numpy()
    assert np.allclose(forward @ backward, np.eye(4), atol=1e-9)


def test_matrix_power_composes():
    """`A^m @ A^n == A^(m+n)`, which repeated squaring has to preserve."""
    a = _matrix((4, 4), seed=36)
    for m, n in [(2, 3), (5, 4), (1, 6), (7, 7)]:
        left = mt.matrix_power(_t(a), m).numpy() @ mt.matrix_power(_t(a), n).numpy()
        assert np.allclose(left, mt.matrix_power(_t(a), m + n).numpy(), atol=1e-8)


def test_matrix_power_of_a_large_exponent_is_not_a_loop():
    """A thousand products would be a thousand roundings as well as a thousand
    matmuls. Repeated squaring does ten."""
    a = np.diag([1.0, 0.5, 2.0, 1.0])
    got = mt.matrix_power(_t(a), 1000).numpy()
    assert np.allclose(got, np.diag([1.0, 0.5**1000, 2.0**1000, 1.0]))


def test_matrix_power_batched():
    a = _matrix((3, 4, 4), seed=37)
    got = mt.matrix_power(_t(a), 4).numpy()
    for i in range(3):
        assert np.allclose(got[i], np.linalg.matrix_power(a[i], 4), atol=1e-10)


def test_matrix_power_rejects_a_rectangular_matrix():
    with pytest.raises(Exception, match="square"):
        mt.matrix_power(_t(_matrix((3, 4))), 2)


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


@pytest.mark.parametrize("shape", [(4, 4), (6, 3), (3, 6)])
def test_pinv_gradient_against_finite_differences(shape):
    """Smooth wherever the rank does not change, which for a random matrix is
    everywhere nearby."""
    a = _matrix(shape, seed=38)
    weights = _matrix((shape[1], shape[0]), seed=39)

    def loss(matrix):
        return float((np.linalg.pinv(matrix) * weights).sum())

    expected = _numeric_grad(loss, a.copy())
    t = mt.Tensor.from_numpy(np.ascontiguousarray(a), requires_grad=True)
    (t.pinv() * mt.Tensor.from_numpy(weights)).sum().backward()
    assert np.allclose(t.grad.numpy(), expected, atol=1e-6)


def test_lstsq_gradient_against_finite_differences():
    a = _matrix((6, 3), seed=40)
    b = _matrix((6,), seed=41)
    weights = _matrix((3,), seed=42)

    def loss(matrix):
        return float(np.linalg.lstsq(matrix, b, rcond=None)[0] @ weights)

    expected = _numeric_grad(loss, a.copy())
    t = mt.Tensor.from_numpy(np.ascontiguousarray(a), requires_grad=True)
    (mt.lstsq(t, _t(b)) * mt.Tensor.from_numpy(weights)).sum().backward()
    assert np.allclose(t.grad.numpy(), expected, atol=1e-6)


@pytest.mark.parametrize("power", [2, 3, -1])
def test_matrix_power_gradient_against_finite_differences(power):
    a = _matrix((3, 3), seed=43)
    weights = _matrix((3, 3), seed=44)

    def loss(matrix):
        return float((np.linalg.matrix_power(matrix, power) * weights).sum())

    expected = _numeric_grad(loss, a.copy())
    t = mt.Tensor.from_numpy(np.ascontiguousarray(a), requires_grad=True)
    (t.matrix_power(power) * mt.Tensor.from_numpy(weights)).sum().backward()
    assert np.allclose(t.grad.numpy(), expected, atol=1e-5)


def test_matrix_rank_and_cond_do_not_pretend_to_carry_gradients():
    """`matrix_rank` counts, so it is an integer and there is nothing to
    differentiate. `cond` is a ratio of two extreme singular values, and the
    gradient of a max is a subgradient at best -- both detach rather than hand
    back something that looks differentiable and is not."""
    t = mt.Tensor.from_numpy(_matrix((4, 4), seed=45), requires_grad=True)
    assert mt.matrix_rank(t).dtype == "int64"
    assert not mt.matrix_rank(t).requires_grad
    assert not mt.cond(t).requires_grad
