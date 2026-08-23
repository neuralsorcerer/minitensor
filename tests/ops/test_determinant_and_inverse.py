# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""`det`, `slogdet` and `inv` were missing, and nothing else could stand in.

`solve` was the only square-matrix operation there was. That is enough to
answer `A x = b` and nothing else: a Gaussian log-likelihood needs the
log-determinant of its covariance, whitening and a precision matrix need the
inverse, and neither can be assembled out of `matmul` and `solve` by a caller.

`inv` is written out of `solve` -- `A @ X = I` -- so it inherits that routine's
pivoting, batching, singularity check and gradient rather than repeating any of
them. `det` and `slogdet` do need their own elimination, because the pivot sign
and the diagonal of `U` are what the answer is made of and `solve` throws both
away.

`slogdet` is not a convenience wrapper. The determinant of a large matrix
overflows long before it stops being useful: the 200x200 case below has a
determinant around 1e187, and one twice that size has none a float64 can hold,
while its logarithm is 431. Its gradient is also the better-behaved one --
`A^-T` with no determinant factor in front.
"""

from __future__ import annotations

import numpy as np
import pytest

import minitensor as mt

SHAPES = [(1, 1), (2, 2), (3, 3), (6, 6), (4, 5, 5), (2, 3, 7, 7)]


def _well_conditioned(shape, seed=0):
    """A diagonal bias keeps these away from singular, so a disagreement with
    NumPy means an actual mistake rather than two implementations dividing by
    nearly nothing in different orders."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal(shape) + np.eye(shape[-1]) * 2


def _numeric_grad(f, arr, eps=1e-6):
    grad = np.zeros_like(arr)
    flat, gflat = arr.reshape(-1), grad.reshape(-1)
    for i in range(flat.size):
        old = flat[i]
        flat[i] = old + eps
        high = f(arr)
        flat[i] = old - eps
        low = f(arr)
        flat[i] = old
        gflat[i] = (high - low) / (2 * eps)
    return grad


@pytest.mark.parametrize("shape", SHAPES)
def test_det_matches_numpy(shape):
    values = _well_conditioned(shape)
    got = mt.Tensor(values, dtype="float64").det().numpy()
    np.testing.assert_allclose(got, np.linalg.det(values), rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("shape", SHAPES)
def test_slogdet_matches_numpy(shape):
    values = _well_conditioned(shape, seed=3)
    sign, logabsdet = mt.Tensor(values, dtype="float64").slogdet()
    want_sign, want_log = np.linalg.slogdet(values)
    np.testing.assert_allclose(sign.numpy(), want_sign, rtol=0, atol=0)
    np.testing.assert_allclose(logabsdet.numpy(), want_log, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("shape", SHAPES)
def test_inv_matches_numpy(shape):
    values = _well_conditioned(shape, seed=5)
    got = mt.Tensor(values, dtype="float64").inv().numpy()
    np.testing.assert_allclose(got, np.linalg.inv(values), rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("shape", SHAPES)
def test_inv_actually_inverts(shape):
    """Against the definition rather than against NumPy: `A @ A^-1` is the
    identity, whatever either implementation thinks the entries are."""
    values = _well_conditioned(shape, seed=7)
    t = mt.Tensor(values, dtype="float64")
    product = t.matmul(t.inv()).numpy()
    identity = np.broadcast_to(np.eye(shape[-1]), values.shape)
    np.testing.assert_allclose(product, identity, rtol=1e-9, atol=1e-10)


@pytest.mark.parametrize("shape", SHAPES)
def test_the_two_determinants_agree(shape):
    """`det` and `slogdet` factorise the same matrix and must not disagree
    about it -- the reason they come out of one pass rather than two."""
    values = _well_conditioned(shape, seed=11)
    t = mt.Tensor(values, dtype="float64")
    sign, logabsdet = t.slogdet()
    rebuilt = sign.numpy() * np.exp(logabsdet.numpy())
    np.testing.assert_allclose(rebuilt, t.det().numpy(), rtol=1e-10, atol=1e-12)


def test_slogdet_survives_a_matrix_whose_determinant_does_not():
    """The whole reason `slogdet` exists next to `det`."""
    rng = np.random.default_rng(13)
    values = rng.standard_normal((200, 200))
    t = mt.Tensor(values, dtype="float64")
    _, logabsdet = t.slogdet()
    np.testing.assert_allclose(
        logabsdet.item(), np.linalg.slogdet(values)[1], rtol=1e-10
    )
    # And the determinant itself is still a real number at this size, just a
    # very large one -- it is the next doubling that loses it.
    assert np.isfinite(t.det().item())


def test_a_singular_matrix_reports_itself():
    singular = np.array([[1.0, 2.0], [2.0, 4.0]])
    t = mt.Tensor(singular, dtype="float64")
    assert t.det().item() == 0.0
    sign, logabsdet = t.slogdet()
    assert sign.item() == 0.0
    assert logabsdet.item() == -np.inf
    # `inv` has no answer to give, and says so rather than returning infinities.
    with pytest.raises(Exception):
        t.inv()


def test_a_singular_matrix_inside_a_batch_still_reports_zero():
    """The batch must not be all-or-nothing for `det`: one singular matrix
    among several is a fact about that matrix, not a failure of the call."""
    batch = np.stack([np.eye(3), np.array([[1.0, 2, 3], [2, 4, 6], [0, 0, 1]])])
    dets = mt.Tensor(batch, dtype="float64").det().numpy()
    assert dets[0] == pytest.approx(1.0)
    assert dets[1] == 0.0


def test_the_sign_is_negative_for_an_odd_number_of_swaps():
    """The pivot sign is half the answer and the easiest half to drop."""
    swapped = np.array([[0.0, 1.0], [1.0, 0.0]])
    t = mt.Tensor(swapped, dtype="float64")
    assert t.det().item() == pytest.approx(-1.0)
    sign, logabsdet = t.slogdet()
    assert sign.item() == -1.0
    assert logabsdet.item() == pytest.approx(0.0)


def test_a_triangular_determinant_is_the_product_of_the_diagonal():
    upper = np.array([[2.0, 9.0, -3.0], [0.0, -4.0, 7.0], [0.0, 0.0, 0.5]])
    assert mt.Tensor(upper, dtype="float64").det().item() == pytest.approx(-4.0)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_both_float_dtypes_are_supported(dtype):
    values = _well_conditioned((4, 4), seed=17).astype(dtype)
    t = mt.Tensor(values, dtype=dtype)
    tolerance = 1e-5 if dtype == "float32" else 1e-12
    assert t.det().item() == pytest.approx(
        float(np.linalg.det(values.astype(np.float64))), rel=tolerance
    )
    assert t.det().dtype == dtype
    assert t.inv().dtype == dtype


@pytest.mark.parametrize("op", ["det", "inv"])
def test_a_non_square_input_is_refused(op):
    values = np.zeros((3, 4))
    with pytest.raises(Exception):
        getattr(mt.Tensor(values, dtype="float64"), op)()


@pytest.mark.parametrize("op", ["det", "slogdet", "inv"])
def test_an_integer_input_is_refused(op):
    values = np.eye(3, dtype=np.int64)
    with pytest.raises(Exception):
        getattr(mt.Tensor(values, dtype="int64"), op)()


@pytest.mark.parametrize("shape", [(3, 3), (4, 4), (2, 3, 3)])
def test_the_det_gradient_matches_numerical_differentiation(shape):
    """Jacobi's formula: `d det(A) / dA = det(A) * A^-T`."""
    rng = np.random.default_rng(19)
    values = _well_conditioned(shape, seed=19)
    weights = rng.standard_normal(shape[:-2])

    def loss(v):
        out = mt.Tensor(v, dtype="float64").det().numpy()
        return float((out * weights).sum())

    t = mt.Tensor(values.copy(), dtype="float64", requires_grad=True)
    (
        t.det() * mt.Tensor(np.asarray(weights, np.float64), dtype="float64")
    ).sum().backward()
    np.testing.assert_allclose(
        t.grad.numpy(), _numeric_grad(loss, values.copy()), rtol=1e-6, atol=1e-8
    )


@pytest.mark.parametrize("shape", [(3, 3), (4, 4), (2, 3, 3)])
def test_the_slogdet_gradient_matches_numerical_differentiation(shape):
    """`d log|det(A)| / dA = A^-T`, with no determinant factor -- which is why
    this stays finite where `det`'s gradient has already overflowed."""
    rng = np.random.default_rng(23)
    values = _well_conditioned(shape, seed=23)
    weights = rng.standard_normal(shape[:-2])

    def loss(v):
        out = mt.Tensor(v, dtype="float64").slogdet()[1].numpy()
        return float((out * weights).sum())

    t = mt.Tensor(values.copy(), dtype="float64", requires_grad=True)
    _, logabsdet = t.slogdet()
    (
        logabsdet * mt.Tensor(np.asarray(weights, np.float64), dtype="float64")
    ).sum().backward()
    np.testing.assert_allclose(
        t.grad.numpy(), _numeric_grad(loss, values.copy()), rtol=1e-6, atol=1e-8
    )


@pytest.mark.parametrize("shape", [(3, 3), (4, 4), (2, 3, 3)])
def test_the_inv_gradient_matches_numerical_differentiation(shape):
    rng = np.random.default_rng(29)
    values = _well_conditioned(shape, seed=29)
    weights = rng.standard_normal(shape)

    def loss(v):
        return float((mt.Tensor(v, dtype="float64").inv().numpy() * weights).sum())

    t = mt.Tensor(values.copy(), dtype="float64", requires_grad=True)
    (t.inv() * mt.Tensor(weights, dtype="float64")).sum().backward()
    np.testing.assert_allclose(
        t.grad.numpy(), _numeric_grad(loss, values.copy()), rtol=1e-6, atol=1e-8
    )


def test_the_sign_carries_no_gradient():
    """It is locally constant wherever it is defined, and undefined exactly
    where the matrix is singular."""
    values = _well_conditioned((3, 3), seed=31)
    t = mt.Tensor(values.copy(), dtype="float64", requires_grad=True)
    sign, _ = t.slogdet()
    assert not sign.requires_grad


def test_a_gaussian_log_likelihood_is_now_expressible():
    """The thing the gap actually blocked, end to end: the log density of a
    multivariate normal needs the log-determinant of its covariance and the
    inverse to form the quadratic term."""
    rng = np.random.default_rng(37)
    dim = 5
    root = rng.standard_normal((dim, dim))
    covariance = root @ root.T + np.eye(dim) * dim
    x = rng.standard_normal(dim)

    cov = mt.Tensor(covariance, dtype="float64")
    _, logabsdet = cov.slogdet()
    centered = mt.Tensor(x, dtype="float64")
    quadratic = centered.dot(cov.inv().matmul(centered))
    got = -0.5 * (quadratic.item() + logabsdet.item() + dim * np.log(2 * np.pi))

    want = -0.5 * (
        x @ np.linalg.inv(covariance) @ x
        + np.linalg.slogdet(covariance)[1]
        + dim * np.log(2 * np.pi)
    )
    assert got == pytest.approx(want, rel=1e-10)


def test_the_module_level_functions_agree_with_the_methods():
    values = _well_conditioned((4, 4), seed=41)
    t = mt.Tensor(values, dtype="float64")
    np.testing.assert_array_equal(mt.det(t).numpy(), t.det().numpy())
    np.testing.assert_array_equal(mt.inv(t).numpy(), t.inv().numpy())
    np.testing.assert_array_equal(mt.slogdet(t)[1].numpy(), t.slogdet()[1].numpy())
