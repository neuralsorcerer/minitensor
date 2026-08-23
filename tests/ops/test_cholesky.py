# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""There was no way to get the factor of a covariance out of the library.

`solve`, `det`, `slogdet` and `inv` all run a pivoted LU, which is right for a
matrix with no structure and wrong for a covariance -- twice the arithmetic, and
none of them hand back the factorisation. The factor is what a whole class of
work is made of: `L @ z` turns standard normal noise into a sample from
`N(0, A)`, `2 * sum(log(diag(L)))` is a log-determinant with nothing in the
middle that overflows, and whitening is a triangular solve against `L`. None of
those can be assembled from `solve`, because `solve` throws its factorisation
away.

Only the lower triangle of the input is read, as LAPACK does -- the upper is
assumed to mirror it, and a matrix that does not mirror is not symmetric and had
no Cholesky factor to begin with. `upper=True` returns the transpose of the same
factorisation rather than a second implementation, so the two spellings cannot
disagree; that is asserted below rather than assumed.

A matrix that is not positive definite is an error, not a NaN, and the error
names the leading minor that failed. That order is the useful part: a
factorisation that got to row 40 of 50 says the leading 40x40 block *was*
positive definite, which is usually the difference between "this covariance
needs more jitter" and "this covariance is the wrong shape entirely".

The gradient is symmetrised. The forward reads one triangle, so the derivative
*of the routine* is triangular -- but `A` is symmetric by assumption, and a
caller who built it as `X @ X.T` needs the sensitivity of both triangles or
their gradient comes back half size. The end-to-end test below is the one that
pins this: it differentiates through the construction, where there is no
convention to choose, only a right answer.
"""

from __future__ import annotations

import numpy as np
import pytest

import minitensor as mt

SHAPES = [(1, 1), (2, 2), (3, 3), (7, 7), (4, 5, 5), (2, 3, 6, 6)]

# One panel is 64 columns wide, so these straddle the boundary in both
# directions and land on it exactly. A blocked factorisation that is off by a
# panel is correct on everything except sizes like these.
PANEL_SIZES = [1, 2, 63, 64, 65, 66, 127, 128, 129, 200]


def _spd(shape, seed=0):
    """Symmetric positive definite, and well away from singular: the diagonal
    bias means a disagreement with NumPy is a mistake rather than two
    implementations dividing by nearly nothing in different orders."""
    rng = np.random.default_rng(seed)
    root = rng.standard_normal(shape)
    return root @ np.swapaxes(root, -1, -2) + np.eye(shape[-1]) * shape[-1]


def _symmetric_numeric_grad(f, a, eps=1e-6):
    """The gradient of `f(sym(A))`, which is what the backward returns.

    Perturbing one entry alone would take `A` out of the symmetric matrices,
    where the factorisation is not defined by anything the caller meant. So the
    perturbation is applied to the pair, and the derivative it produces is
    `g[i, j] + g[j, i]` -- twice the entry, off the diagonal.
    """
    n = a.shape[-1]
    grad = np.zeros_like(a)
    flat, gflat = a.reshape(-1, n, n), grad.reshape(-1, n, n)
    for b in range(flat.shape[0]):
        for i in range(n):
            for j in range(i + 1):
                ij, ji = flat[b, i, j], flat[b, j, i]
                flat[b, i, j], flat[b, j, i] = ij + eps, ji + eps
                high = f(a)
                flat[b, i, j], flat[b, j, i] = ij - eps, ji - eps
                low = f(a)
                flat[b, i, j], flat[b, j, i] = ij, ji
                d = (high - low) / (2 * eps)
                if i == j:
                    gflat[b, i, i] = d
                else:
                    gflat[b, i, j] = gflat[b, j, i] = d / 2
    return grad


@pytest.mark.parametrize("shape", SHAPES)
def test_it_matches_numpy(shape):
    values = _spd(shape)
    got = mt.Tensor(values, dtype="float64").cholesky().numpy()
    np.testing.assert_allclose(got, np.linalg.cholesky(values), rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize("n", PANEL_SIZES)
def test_it_matches_numpy_across_the_panel_boundary(n):
    """The factorisation is panelled, and a blocked routine that is off by a
    panel is right on almost every size except the ones next to a boundary."""
    values = _spd((n, n), seed=n)
    got = mt.Tensor(values, dtype="float64").cholesky().numpy()
    np.testing.assert_allclose(got, np.linalg.cholesky(values), rtol=1e-11, atol=1e-13)


@pytest.mark.parametrize("shape", SHAPES)
def test_the_factor_reconstructs_the_matrix(shape):
    """Against the definition rather than against NumPy: `L @ L.T` is `A`,
    whatever either implementation thinks the entries of `L` are."""
    values = _spd(shape, seed=3)
    factor = mt.Tensor(values, dtype="float64").cholesky()
    product = factor.matmul(factor.transpose(-2, -1)).numpy()
    np.testing.assert_allclose(product, values, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("n", [3, 65])
def test_the_result_is_lower_triangular(n):
    got = mt.Tensor(_spd((n, n), seed=5), dtype="float64").cholesky().numpy()
    assert np.all(np.triu(got, 1) == 0), "the strict upper triangle must be zero"
    assert np.all(np.diag(got) > 0), "the diagonal of a Cholesky factor is positive"


@pytest.mark.parametrize("shape", SHAPES)
def test_upper_is_the_transpose_of_lower(shape):
    """It is a transpose, not a second implementation -- so it must not be able
    to disagree with the lower form."""
    values = _spd(shape, seed=7)
    t = mt.Tensor(values, dtype="float64")
    lower = t.cholesky().numpy()
    upper = t.cholesky(upper=True).numpy()
    np.testing.assert_array_equal(upper, np.swapaxes(lower, -1, -2))
    assert np.all(np.tril(upper, -1) == 0)


def test_upper_reconstructs_as_u_transpose_times_u():
    values = _spd((5, 5), seed=11)
    upper = mt.Tensor(values, dtype="float64").cholesky(upper=True).numpy()
    np.testing.assert_allclose(upper.T @ upper, values, rtol=1e-10, atol=1e-12)


def test_only_the_lower_triangle_is_read():
    """LAPACK's contract, and worth pinning because the alternative is
    defensible: garbage above the diagonal changes nothing, because a symmetric
    matrix has nothing up there the lower triangle does not already say."""
    values = _spd((4, 4), seed=13)
    scribbled = values.copy()
    scribbled[np.triu_indices(4, 1)] = 1e9
    np.testing.assert_array_equal(
        mt.Tensor(scribbled, dtype="float64").cholesky().numpy(),
        mt.Tensor(values, dtype="float64").cholesky().numpy(),
    )


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_both_float_dtypes_are_supported(dtype):
    values = _spd((6, 6), seed=17).astype(dtype)
    got = mt.Tensor(values, dtype=dtype).cholesky()
    assert got.dtype == dtype
    tolerance = 1e-5 if dtype == "float32" else 1e-12
    np.testing.assert_allclose(
        got.numpy().astype(np.float64),
        np.linalg.cholesky(values.astype(np.float64)),
        rtol=tolerance,
        atol=tolerance,
    )


# --- what it refuses ---------------------------------------------------------


def test_a_matrix_that_is_not_positive_definite_is_refused():
    """Not a NaN: it means the caller's assumption about the matrix was wrong,
    and a NaN would let that assumption travel."""
    for bad in (
        np.array([[1.0, 2.0], [2.0, 1.0]]),  # indefinite
        np.zeros((3, 3)),  # semidefinite, singular at the first pivot
        np.array([[1.0, 0.0], [0.0, -1.0]]),  # negative eigenvalue
    ):
        with pytest.raises(Exception):
            mt.Tensor(bad, dtype="float64").cholesky()


def test_the_error_names_the_leading_minor_that_failed():
    """The order is the useful half of the message: everything before it was
    positive definite."""
    values = _spd((5, 5), seed=19)
    values[3, :] = 0.0
    values[:, 3] = 0.0
    values[3, 3] = -1.0
    with pytest.raises(Exception, match="order 4"):
        mt.Tensor(values, dtype="float64").cholesky()


def test_a_failure_past_the_first_panel_still_names_the_right_minor():
    """The panel loop reports a row within the panel it was factoring; the
    offset has to be added back or every failure past column 64 is misreported."""
    values = _spd((100, 100), seed=23)
    values[80, :] = 0.0
    values[:, 80] = 0.0
    values[80, 80] = -1.0
    with pytest.raises(Exception, match="order 81"):
        mt.Tensor(values, dtype="float64").cholesky()


def test_a_nan_is_refused_rather_than_spread():
    values = _spd((4, 4), seed=29)
    values[2, 2] = np.nan
    with pytest.raises(Exception):
        mt.Tensor(values, dtype="float64").cholesky()


def test_a_non_square_input_is_refused():
    with pytest.raises(Exception):
        mt.Tensor(np.zeros((3, 4)), dtype="float64").cholesky()


def test_a_one_dimensional_input_is_refused():
    with pytest.raises(Exception):
        mt.Tensor(np.ones(4), dtype="float64").cholesky()


def test_an_integer_input_is_refused():
    with pytest.raises(Exception):
        mt.Tensor(np.eye(3, dtype=np.int64), dtype="int64").cholesky()


@pytest.mark.parametrize("shape", [(0, 0), (0, 3, 3), (2, 0, 0), (0, 0, 0)])
def test_an_empty_input_factors_to_an_empty_output(shape):
    """A matrix of order zero has a factorisation and it is empty -- not an
    error, and not a panic. The order-zero cases are the ones that bite: there
    is no matrix to walk, so a loop written in terms of the matrix size asks for
    chunks of nothing."""
    values = np.zeros(shape)
    got = mt.Tensor(values, dtype="float64").cholesky().numpy()
    assert got.shape == shape
    assert got.shape == np.linalg.cholesky(values).shape


def test_a_singular_matrix_inside_a_batch_fails_the_call():
    """Unlike `det`, where a zero is an answer: there is no factor to return for
    the singular one, so there is nothing to put in its slot."""
    batch = np.stack([np.eye(3), np.zeros((3, 3))])
    with pytest.raises(Exception):
        mt.Tensor(batch, dtype="float64").cholesky()


# --- gradients ---------------------------------------------------------------


@pytest.mark.parametrize("shape", [(3, 3), (5, 5), (2, 4, 4)])
@pytest.mark.parametrize("upper", [False, True])
def test_the_gradient_matches_numerical_differentiation(shape, upper):
    rng = np.random.default_rng(31)
    values = _spd(shape, seed=31)
    weights = rng.standard_normal(shape)

    def loss(v):
        return float(
            (mt.Tensor(v, dtype="float64").cholesky(upper).numpy() * weights).sum()
        )

    t = mt.Tensor(values.copy(), dtype="float64", requires_grad=True)
    (t.cholesky(upper) * mt.Tensor(weights, dtype="float64")).sum().backward()
    np.testing.assert_allclose(
        t.grad.numpy(),
        _symmetric_numeric_grad(loss, values.copy()),
        rtol=1e-6,
        atol=1e-8,
    )


def test_the_gradient_is_symmetric():
    """It is the gradient of `L(sym(A))`, so it has to be -- and an
    implementation that forgot to symmetrise would return the lower triangle
    only, which is a different tensor with the same trace."""
    values = _spd((6, 6), seed=37)
    t = mt.Tensor(values.copy(), dtype="float64", requires_grad=True)
    t.cholesky().sum().backward()
    grad = t.grad.numpy()
    np.testing.assert_array_equal(grad, grad.T)


def test_the_gradient_through_a_symmetric_construction():
    """The test that settles the convention, because there is no convention left
    to choose: `A = X @ X.T` is symmetric by construction, so the derivative
    with respect to `X` is a single well-defined thing, and only a gradient that
    accounts for both triangles of `A` produces it."""
    rng = np.random.default_rng(41)
    n = 4
    x = rng.standard_normal((n, n))
    weights = rng.standard_normal((n, n))
    bias = np.eye(n) * n

    def loss(v):
        factor = np.linalg.cholesky(v @ v.T + bias)
        return float((factor * weights).sum())

    numeric = np.zeros_like(x)
    eps = 1e-6
    for i in range(x.size):
        flat, gflat = x.reshape(-1), numeric.reshape(-1)
        old = flat[i]
        flat[i] = old + eps
        high = loss(x)
        flat[i] = old - eps
        low = loss(x)
        flat[i] = old
        gflat[i] = (high - low) / (2 * eps)

    tx = mt.Tensor(x.copy(), dtype="float64", requires_grad=True)
    covariance = tx.matmul(tx.transpose(0, 1)) + mt.Tensor(bias, dtype="float64")
    (covariance.cholesky() * mt.Tensor(weights, dtype="float64")).sum().backward()
    np.testing.assert_allclose(tx.grad.numpy(), numeric, rtol=1e-6, atol=1e-8)


def test_the_gradient_agrees_with_the_analytic_formula_past_a_panel():
    """The blocked forward changes where the arithmetic happens; the backward
    reads only the factor, so it should not notice. Checked against the closed
    form on a matrix wide enough to take more than one panel."""
    n = 70
    rng = np.random.default_rng(43)
    values = _spd((n, n), seed=43)
    weights = rng.standard_normal((n, n))

    t = mt.Tensor(values.copy(), dtype="float64", requires_grad=True)
    (t.cholesky() * mt.Tensor(weights, dtype="float64")).sum().backward()

    factor = np.linalg.cholesky(values)
    phi = np.tril(factor.T @ weights)
    np.fill_diagonal(phi, np.diag(phi) * 0.5)
    inverse = np.linalg.inv(factor)
    unsymmetrised = inverse.T @ phi @ inverse
    np.testing.assert_allclose(
        t.grad.numpy(),
        0.5 * (unsymmetrised + unsymmetrised.T),
        rtol=1e-9,
        atol=1e-11,
    )


def test_a_batched_gradient_is_the_per_matrix_gradient():
    """Batches are factored in parallel and grouped by task; the answer must not
    depend on how they were grouped."""
    values = _spd((6, 5, 5), seed=47)
    batched = mt.Tensor(values.copy(), dtype="float64", requires_grad=True)
    batched.cholesky().sum().backward()

    singles = []
    for i in range(values.shape[0]):
        one = mt.Tensor(values[i].copy(), dtype="float64", requires_grad=True)
        one.cholesky().sum().backward()
        singles.append(one.grad.numpy())
    np.testing.assert_array_equal(batched.grad.numpy(), np.stack(singles))


# --- the things it unblocks --------------------------------------------------


def test_the_log_determinant_of_a_covariance_is_now_cheap():
    """`2 * sum(log(diag(L)))`, with no intermediate the size of a determinant.
    `slogdet` answers the same question by a pivoted LU; on a covariance this
    is the routine that should be used, and the two must agree."""
    values = _spd((8, 8), seed=53)
    factor = mt.Tensor(values, dtype="float64").cholesky()
    through_factor = 2.0 * float(factor.diagonal().log().sum().item())
    _, through_lu = mt.Tensor(values, dtype="float64").slogdet()
    assert through_factor == pytest.approx(through_lu.item(), rel=1e-10)
    assert through_factor == pytest.approx(np.linalg.slogdet(values)[1], rel=1e-10)


def test_sampling_a_multivariate_normal_is_now_expressible():
    """The thing the gap actually blocked: `L @ z` has covariance `A` when `z`
    is standard normal, and there was no way to get `L`."""
    rng = np.random.default_rng(59)
    dim = 4
    covariance = _spd((dim, dim), seed=59)
    factor = mt.Tensor(covariance, dtype="float64").cholesky().numpy()

    draws = rng.standard_normal((200_000, dim)) @ factor.T
    empirical = np.cov(draws, rowvar=False)
    np.testing.assert_allclose(empirical, covariance, rtol=0.05, atol=0.05)


def test_whitening_round_trips_through_solve():
    """`L^-1 x` has identity covariance; putting `L` back recovers `x`."""
    rng = np.random.default_rng(61)
    dim = 5
    covariance = _spd((dim, dim), seed=61)
    x = rng.standard_normal((dim, 1))

    t = mt.Tensor(covariance, dtype="float64")
    factor = t.cholesky()
    whitened = factor.solve(mt.Tensor(x, dtype="float64"))
    np.testing.assert_allclose(factor.matmul(whitened).numpy(), x, rtol=1e-10)
    # And the quadratic form it exists to compute agrees with the direct one.
    quadratic = float((whitened * whitened).sum().item())
    assert quadratic == pytest.approx(
        (x.T @ np.linalg.inv(covariance) @ x).item(), rel=1e-9
    )


def test_the_module_level_function_agrees_with_the_method():
    values = _spd((4, 4), seed=67)
    t = mt.Tensor(values, dtype="float64")
    for upper in (False, True):
        np.testing.assert_array_equal(
            mt.cholesky(t, upper).numpy(), t.cholesky(upper).numpy()
        )
