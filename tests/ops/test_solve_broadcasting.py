# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""The solves broadcast their batch dimensions.

`solve`, `solve_triangular`, `cholesky_solve` and `lu_solve` required the two
batches to match exactly, so one system against a stack of right-hand sides --
or a stack of systems against one -- had to be `expand`ed by hand first, which
every other batched operation here does on its own. A right-hand side one rank
below the matrices, or a single 1-D vector, ending in `n` is read as vectors;
anything else ending in `(n, k)` as matrices.
"""

import numpy as np
import pytest

import minitensor as mt

RNG = np.random.default_rng(3)
N = 4
STACK = RNG.standard_normal((2, N, N)) + 4 * np.eye(N)


def _t(a, requires_grad=False):
    return mt.tensor(
        np.asarray(a).tolist(), dtype="float64", requires_grad=requires_grad
    )


def _reference(a, b):
    vectors = b.ndim == 1 or (b.ndim == a.ndim - 1 and b.shape[-1] == N)
    if vectors:
        batch = np.broadcast_shapes(a.shape[:-2], b.shape[:-1])
        full = np.broadcast_to(a, batch + (N, N))
        return np.linalg.solve(full, np.broadcast_to(b, batch + (N,))[..., None])[
            ..., 0
        ]
    batch = np.broadcast_shapes(a.shape[:-2], b.shape[:-2])
    return np.linalg.solve(np.broadcast_to(a, batch + (N, N)), b)


CASES = [
    (STACK, (N,)),
    (STACK, (1, N)),
    (STACK, (N, 3)),
    (STACK, (1, N, 3)),
    (STACK, (3, 1, N, 3)),
    (STACK[0], (5, N, 2)),
    (STACK[:1], (3, N)),
    (STACK, (2, N)),
    (STACK, (2, N, 3)),
]


@pytest.mark.parametrize("a, shape", CASES)
def test_solve_broadcasts(a, shape):
    b = RNG.standard_normal(shape)
    got = mt.solve(_t(a), _t(b)).numpy()
    np.testing.assert_allclose(got, _reference(a, b), rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("a, shape", CASES)
def test_the_factorisation_solves_broadcast_the_same_way(a, shape):
    b = RNG.standard_normal(shape)
    upper = np.triu(a)
    np.testing.assert_allclose(
        mt.solve_triangular(_t(upper), _t(b), upper=True).numpy(),
        _reference(upper, b),
        rtol=1e-11,
        atol=1e-12,
    )
    spd = a @ np.swapaxes(a, -1, -2) + np.eye(N)
    factor = np.linalg.cholesky(spd)
    np.testing.assert_allclose(
        mt.cholesky_solve(_t(b), _t(factor)).numpy(),
        _reference(spd, b),
        rtol=1e-11,
        atol=1e-12,
    )
    packed, pivots = mt.lu_factor(_t(a))
    np.testing.assert_allclose(
        mt.lu_solve(packed, pivots, _t(b)).numpy(),
        _reference(a, b),
        rtol=1e-11,
        atol=1e-12,
    )


def test_gradients_reach_broadcast_operands_at_their_own_shapes():
    a = _t(STACK[:1], requires_grad=True)
    b = _t(RNG.standard_normal((3, N, 2)), requires_grad=True)
    assert mt.gradcheck(lambda a, b: mt.solve(a, b).sum(), (a, b), atol=1e-8, rtol=1e-6)
    a = _t(STACK, requires_grad=True)
    b = _t(RNG.standard_normal(N), requires_grad=True)
    assert mt.gradcheck(lambda a, b: mt.solve(a, b).sum(), (a, b), atol=1e-8, rtol=1e-6)
    mt.solve(a, b).sum().backward()
    assert a.grad.shape == (2, N, N) and b.grad.shape == (N,)
    tri = _t(np.triu(STACK[0]), requires_grad=True)
    rhs = _t(RNG.standard_normal((3, N, 2)), requires_grad=True)
    assert mt.gradcheck(
        lambda t, r: mt.solve_triangular(t, r, upper=True).sum(),
        (tri, rhs),
        atol=1e-8,
        rtol=1e-6,
    )


@pytest.mark.parametrize("shape", [(3, N, 2), (N + 1,), (2, N + 1, 3)])
def test_a_right_hand_side_that_fits_neither_reading_is_refused(shape):
    with pytest.raises(ValueError, match="right-hand side|rhs|Shape mismatch"):
        mt.solve(_t(STACK), _t(RNG.standard_normal(shape)))


@pytest.mark.parametrize(
    "a_shape, b_shape",
    [((2, 6, 3), (6, 2)), ((2, 6, 3), (6,)), ((2, 6, 3), (2, 6)), ((6, 3), (4, 6, 2))],
)
def test_lstsq_reads_its_right_hand_side_as_solve_does(a_shape, b_shape):
    a = RNG.standard_normal(a_shape)
    b = RNG.standard_normal(b_shape)
    vectors = b.ndim == 1 or (b.ndim == a.ndim - 1 and b.shape[-1] == a.shape[-2])
    rhs = b[..., None] if vectors else b
    want = np.linalg.pinv(a) @ rhs
    want = want[..., 0] if vectors else want
    got = mt.lstsq(_t(a), _t(b)).numpy()
    np.testing.assert_allclose(got, want, rtol=1e-10, atol=1e-12)
