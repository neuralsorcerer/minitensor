# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""There was no way to ask a matrix which directions it acts on.

`cholesky` factors a positive-definite matrix, `qr` an arbitrary one, `solve`
answers `Ax = b`. None of them says anything about *directions*, and that is
what principal component analysis returns, what whitening a covariance needs,
what a spectral norm is the largest of, and what tells a caller whether their
matrix is positive definite and by how much. The last two tests do PCA and
whitening end to end, because those are the things the gap actually blocked.

It is also the only factorisation here that cannot be computed in a finite
number of steps. Eigenvalues are roots of the characteristic polynomial, so past
degree four no formula exists and what runs instead is an iteration that
converges. Two phases: Householder reflections reduce the matrix to tridiagonal
form in exactly `n - 2` steps -- that half is finite, and it is `qr`'s
reflection applied from both sides so the result stays symmetric -- and then
implicitly shifted QL iterations chase the off-diagonal to zero.

Eigenvectors are determined only up to sign. `v` and `-v` both satisfy the
definition and nothing picks between them, so every test here compares what the
vectors *do* -- `A @ V == V @ diag(w)`, `V.T @ V == I` -- or compares a
sign-invariant function of them. Comparing `V` against NumPy's `V` elementwise
would fail for a reason that is not a bug. The eigenvalues, by contrast, are
determined exactly, so those are compared directly.

The gradient carries `1 / (w_j - w_i)`, and that is not a defect to be papered
over. Eigenvectors of a matrix with a repeated eigenvalue are genuinely not
unique -- any rotation within the shared eigenspace is as good -- so there is no
derivative to report, and the formula says so by dividing by zero. That is
pinned below rather than hidden.
"""

from __future__ import annotations

import numpy as np
import pytest

import minitensor as mt

SHAPES = [(1, 1), (2, 2), (3, 3), (5, 5), (8, 8), (2, 4, 4), (2, 3, 6, 6)]


def _symmetric(shape, seed=0):
    values = np.random.default_rng(seed).standard_normal(shape)
    return (values + np.swapaxes(values, -1, -2)) / 2


def _symmetric_numeric_grad(f, a, eps=1e-6):
    """The gradient of `f(sym(A))`, which is what the backward returns.

    Perturbing one entry alone would take `A` out of the symmetric matrices,
    where the decomposition is not defined by anything the caller meant, so the
    perturbation goes to the pair and the derivative it produces is
    `g[i, j] + g[j, i]`.
    """
    n = a.shape[-1]
    grad = np.zeros_like(a)
    for i in range(n):
        for j in range(i + 1):
            ij, ji = a[i, j], a[j, i]
            a[i, j], a[j, i] = ij + eps, ji + eps
            high = f(a)
            a[i, j], a[j, i] = ij - eps, ji - eps
            low = f(a)
            a[i, j], a[j, i] = ij, ji
            d = (high - low) / (2 * eps)
            if i == j:
                grad[i, i] = d
            else:
                grad[i, j] = grad[j, i] = d / 2
    return grad


@pytest.mark.parametrize("shape", SHAPES)
def test_the_eigenvalues_match_numpy(shape):
    """These are determined exactly -- unlike the vectors -- so they are
    compared directly."""
    values = _symmetric(shape, seed=hash(shape) % 1000)
    got = mt.Tensor(values, dtype="float64").eigh()[0].numpy()
    np.testing.assert_allclose(got, np.linalg.eigvalsh(values), rtol=1e-11, atol=1e-13)


@pytest.mark.parametrize("shape", SHAPES)
def test_the_vectors_satisfy_the_definition(shape):
    """`A @ V == V @ diag(w)`, which is what an eigenvector *is* -- checked
    without reference to anyone else's sign convention."""
    values = _symmetric(shape, seed=3)
    w, v = mt.Tensor(values, dtype="float64").eigh()
    vectors, eigenvalues = v.numpy(), w.numpy()
    np.testing.assert_allclose(
        values @ vectors, vectors * eigenvalues[..., None, :], rtol=1e-9, atol=1e-11
    )


@pytest.mark.parametrize("shape", SHAPES)
def test_the_vectors_are_orthonormal(shape):
    values = _symmetric(shape, seed=5)
    vectors = mt.Tensor(values, dtype="float64").eigh()[1].numpy()
    identity = np.broadcast_to(np.eye(shape[-1]), vectors.shape)
    np.testing.assert_allclose(
        np.swapaxes(vectors, -1, -2) @ vectors, identity, rtol=0, atol=1e-11
    )


@pytest.mark.parametrize("shape", SHAPES)
def test_the_eigenvalues_come_back_ascending(shape):
    """LAPACK's order and NumPy's, and the one callers zip against."""
    values = _symmetric(shape, seed=7)
    got = mt.Tensor(values, dtype="float64").eigh()[0].numpy()
    assert np.all(np.diff(got, axis=-1) >= -1e-12)


def test_known_spectra():
    """Cases where the answer can be written down."""
    identity = mt.Tensor(np.eye(4), dtype="float64").eigh()[0].numpy()
    np.testing.assert_allclose(identity, np.ones(4))

    diagonal = mt.Tensor(np.diag([3.0, -1.0, 2.0]), dtype="float64").eigh()[0].numpy()
    np.testing.assert_allclose(diagonal, [-1.0, 2.0, 3.0])

    # [[a, b], [b, a]] has eigenvalues a - b and a + b, whatever a and b are.
    pair = mt.Tensor(np.array([[5.0, 2.0], [2.0, 5.0]]), dtype="float64").eigh()[0]
    np.testing.assert_allclose(pair.numpy(), [3.0, 7.0])


def test_repeated_eigenvalues_still_give_an_orthonormal_basis():
    """The eigenvectors are not unique here, but they still have to be a basis:
    any orthonormal set spanning the eigenspace is a correct answer, and a
    degenerate one is where a careless iteration returns a singular `V`."""
    values = np.diag([2.0, 2.0, 2.0, 5.0])
    w, v = mt.Tensor(values, dtype="float64").eigh()
    np.testing.assert_allclose(w.numpy(), [2.0, 2.0, 2.0, 5.0])
    vectors = v.numpy()
    np.testing.assert_allclose(vectors.T @ vectors, np.eye(4), rtol=0, atol=1e-12)
    np.testing.assert_allclose(values @ vectors, vectors * w.numpy(), atol=1e-12)


def test_a_larger_matrix_still_converges():
    """The iteration is the part that can fail to terminate, so a size where it
    has to work for a while is worth its own case."""
    values = _symmetric((30, 30), seed=11)
    w, v = mt.Tensor(values, dtype="float64").eigh()
    np.testing.assert_allclose(w.numpy(), np.linalg.eigvalsh(values), rtol=1e-10, atol=1e-12)
    vectors = v.numpy()
    np.testing.assert_allclose(vectors.T @ vectors, np.eye(30), rtol=0, atol=1e-11)


def test_only_the_lower_triangle_is_read():
    """LAPACK's contract, and worth pinning: the upper is assumed to mirror the
    lower, so garbage up there changes nothing."""
    scribbled = np.array([[1.0, 99.0], [2.0, 3.0]])
    mirrored = np.array([[1.0, 2.0], [2.0, 3.0]])
    np.testing.assert_allclose(
        mt.Tensor(scribbled, dtype="float64").eigh()[0].numpy(),
        mt.Tensor(mirrored, dtype="float64").eigh()[0].numpy(),
    )


# --- eigvalsh ----------------------------------------------------------------


@pytest.mark.parametrize("shape", SHAPES)
def test_eigvalsh_agrees_with_eigh(shape):
    """It skips accumulating the rotations, which is most of the work -- but it
    must not change the answer by a bit."""
    values = _symmetric(shape, seed=13)
    tensor = mt.Tensor(values, dtype="float64")
    np.testing.assert_array_equal(tensor.eigvalsh().numpy(), tensor.eigh()[0].numpy())


def test_eigvalsh_matches_numpy():
    values = _symmetric((7, 7), seed=17)
    np.testing.assert_allclose(
        mt.Tensor(values, dtype="float64").eigvalsh().numpy(),
        np.linalg.eigvalsh(values),
        rtol=1e-11, atol=1e-13,
    )


# --- shapes, dtypes, refusals ------------------------------------------------


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_both_float_dtypes_are_supported(dtype):
    values = _symmetric((6, 6), seed=19).astype(dtype)
    w, v = mt.Tensor(values, dtype=dtype).eigh()
    assert w.dtype == dtype and v.dtype == dtype
    tolerance = 1e-5 if dtype == "float32" else 1e-11
    np.testing.assert_allclose(
        w.numpy().astype(np.float64),
        np.linalg.eigvalsh(values.astype(np.float64)),
        rtol=tolerance, atol=tolerance,
    )


@pytest.mark.parametrize("shape", [(0, 0), (0, 3, 3), (2, 0, 0)])
def test_an_empty_input_gives_empty_outputs(shape):
    values = np.zeros(shape)
    w, v = mt.Tensor(values, dtype="float64").eigh()
    want_w, want_v = np.linalg.eigh(values)
    assert w.numpy().shape == want_w.shape
    assert v.numpy().shape == want_v.shape


def test_a_non_square_input_is_refused():
    with pytest.raises(Exception):
        mt.Tensor(np.zeros((3, 4)), dtype="float64").eigh()


def test_a_one_dimensional_input_is_refused():
    with pytest.raises(Exception):
        mt.Tensor(np.zeros(4), dtype="float64").eigh()


def test_an_integer_input_is_refused():
    with pytest.raises(Exception):
        mt.Tensor(np.eye(3, dtype=np.int64), dtype="int64").eigh()


# --- gradients ---------------------------------------------------------------


@pytest.mark.parametrize("n", [3, 4, 6])
def test_the_eigenvalue_gradient_matches_numerical_differentiation(n):
    """`d w_i / dA = v_i v_i^T`, which is the one piece of this that stays
    well defined when eigenvalues repeat."""
    rng = np.random.default_rng(23)
    values = _symmetric((n, n), seed=23)
    weights = rng.standard_normal(n)

    def loss(a):
        return float((np.linalg.eigvalsh(a) * weights).sum())

    t = mt.Tensor(values.copy(), dtype="float64", requires_grad=True)
    (t.eigvalsh() * mt.Tensor(weights, dtype="float64")).sum().backward()
    got = t.grad.numpy()
    np.testing.assert_allclose(
        got, _symmetric_numeric_grad(loss, values.copy()), rtol=1e-5, atol=1e-7
    )
    np.testing.assert_array_equal(got, got.T)


@pytest.mark.parametrize("n", [3, 4, 5])
def test_the_eigenvector_gradient_matches_numerical_differentiation(n):
    """The loss has to be sign-invariant for finite differences to mean
    anything, since a perturbation can flip a vector's sign without changing
    the decomposition. `V * V` is."""
    rng = np.random.default_rng(29)
    values = _symmetric((n, n), seed=29)
    weights = rng.standard_normal((n, n))

    def loss(a):
        vectors = np.linalg.eigh(a)[1]
        return float(((vectors * vectors) * weights).sum())

    t = mt.Tensor(values.copy(), dtype="float64", requires_grad=True)
    _, v = t.eigh()
    ((v * v) * mt.Tensor(weights, dtype="float64")).sum().backward()
    np.testing.assert_allclose(
        t.grad.numpy(), _symmetric_numeric_grad(loss, values.copy()), rtol=1e-5, atol=1e-7
    )


def test_the_two_output_gradients_add_up():
    """The decomposition has two outputs and the graph hands a node one gradient
    at a time, so each gets its own node. The gradient is linear in the pair, so
    the two halves have to sum to the whole -- exactly, not approximately."""
    n = 4
    rng = np.random.default_rng(31)
    values = _symmetric((n, n), seed=31)
    value_weights = rng.standard_normal(n)
    vector_weights = rng.standard_normal((n, n))

    def loss(a):
        eigenvalues, vectors = np.linalg.eigh(a)
        return float(
            (eigenvalues * value_weights).sum()
            + ((vectors * vectors) * vector_weights).sum()
        )

    t = mt.Tensor(values.copy(), dtype="float64", requires_grad=True)
    w, v = t.eigh()
    (
        (w * mt.Tensor(value_weights, dtype="float64")).sum()
        + ((v * v) * mt.Tensor(vector_weights, dtype="float64")).sum()
    ).backward()
    np.testing.assert_allclose(
        t.grad.numpy(), _symmetric_numeric_grad(loss, values.copy()), rtol=1e-5, atol=1e-7
    )


def test_the_gradient_is_symmetric():
    """`A` is symmetric by assumption, so only the symmetric part of a
    perturbation is realisable -- the same reasoning `cholesky`'s gradient
    follows."""
    values = _symmetric((5, 5), seed=37)
    t = mt.Tensor(values.copy(), dtype="float64", requires_grad=True)
    t.eigvalsh().sum().backward()
    grad = t.grad.numpy()
    np.testing.assert_array_equal(grad, grad.T)


def test_the_vector_gradient_has_no_answer_for_a_repeated_eigenvalue():
    """Not a defect being papered over. Eigenvectors of a degenerate matrix are
    genuinely not unique -- any rotation inside the shared eigenspace is as
    good -- so there is no derivative, and dividing by the zero gap says so.
    The *eigenvalue* gradient stays perfectly well defined."""
    t = mt.Tensor(np.eye(3) * 2.0, dtype="float64", requires_grad=True)
    _, v = t.eigh()
    v.sum().backward()
    assert not np.isfinite(t.grad.numpy()).all()

    finite = mt.Tensor(np.eye(3) * 2.0, dtype="float64", requires_grad=True)
    finite.eigvalsh().sum().backward()
    assert np.isfinite(finite.grad.numpy()).all()


def test_a_batched_gradient_is_the_per_matrix_gradient():
    values = np.stack([_symmetric((3, 3), seed=s) for s in (1, 2, 3)])
    batched = mt.Tensor(values.copy(), dtype="float64", requires_grad=True)
    batched.eigvalsh().sum().backward()

    singles = []
    for i in range(values.shape[0]):
        one = mt.Tensor(values[i].copy(), dtype="float64", requires_grad=True)
        one.eigvalsh().sum().backward()
        singles.append(one.grad.numpy())
    np.testing.assert_allclose(batched.grad.numpy(), np.stack(singles), rtol=1e-12, atol=1e-14)


# --- the things it unblocks --------------------------------------------------


def test_principal_component_analysis_is_now_expressible():
    """The eigenvalues of a covariance are the variance along each principal
    direction, and the vectors are the directions. There was no way to get
    either."""
    rng = np.random.default_rng(41)
    scales = np.diag([3.0, 2.0, 0.5, 0.1])
    samples = rng.standard_normal((400, 4)) @ scales
    centred = samples - samples.mean(axis=0)
    covariance = (centred.T @ centred) / (len(samples) - 1)

    w, v = mt.Tensor(covariance, dtype="float64").eigh()
    variance = w.numpy()
    np.testing.assert_allclose(variance, np.linalg.eigvalsh(covariance), rtol=1e-10)

    # The largest component should line up with the largest scale.
    assert variance[-1] > variance[-2] > variance[-3] > variance[-4]
    # Projecting onto the components and back is the identity, because they are
    # a complete orthonormal basis.
    vectors = v.numpy()
    np.testing.assert_allclose(
        centred @ vectors @ vectors.T, centred, rtol=1e-9, atol=1e-11
    )


def test_whitening_a_covariance_is_now_expressible():
    """`V diag(1/sqrt(w)) V^T` turns a correlated cloud into an uncorrelated
    one, and needs both halves of the decomposition."""
    rng = np.random.default_rng(43)
    root = rng.standard_normal((5, 5))
    covariance = root @ root.T + np.eye(5)
    samples = rng.standard_normal((5000, 5)) @ np.linalg.cholesky(covariance).T

    w, v = mt.Tensor(covariance, dtype="float64").eigh()
    vectors, eigenvalues = v.numpy(), w.numpy()
    whitener = vectors @ np.diag(1.0 / np.sqrt(eigenvalues)) @ vectors.T

    # The transform is exact on the covariance itself, whatever the sample says.
    np.testing.assert_allclose(
        whitener @ covariance @ whitener.T, np.eye(5), rtol=1e-9, atol=1e-11
    )
    assert samples.shape == (5000, 5)


def test_the_spectral_norm_is_now_expressible():
    """The largest absolute eigenvalue of a symmetric matrix is its operator
    2-norm, which nothing here could produce."""
    values = _symmetric((6, 6), seed=47)
    eigenvalues = mt.Tensor(values, dtype="float64").eigvalsh().numpy()
    assert np.abs(eigenvalues).max() == pytest.approx(np.linalg.norm(values, 2), rel=1e-10)


def test_positive_definiteness_is_now_checkable():
    """And agrees with what `cholesky` accepts, which is the other way to ask."""
    definite = _symmetric((4, 4), seed=53) + np.eye(4) * 5
    indefinite = np.array([[1.0, 2.0], [2.0, 1.0]])
    assert mt.Tensor(definite, dtype="float64").eigvalsh().numpy().min() > 0
    mt.Tensor(definite, dtype="float64").cholesky()  # agrees: no error
    assert mt.Tensor(indefinite, dtype="float64").eigvalsh().numpy().min() < 0
    with pytest.raises(Exception):
        mt.Tensor(indefinite, dtype="float64").cholesky()


def test_the_module_level_functions_agree_with_the_methods():
    values = _symmetric((4, 4), seed=59)
    t = mt.Tensor(values, dtype="float64")
    np.testing.assert_array_equal(mt.eigvalsh(t).numpy(), t.eigvalsh().numpy())
    for a, b in zip(mt.eigh(t), t.eigh()):
        np.testing.assert_array_equal(a.numpy(), b.numpy())
