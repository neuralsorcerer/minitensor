# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""`matrix_exp`: scaling and squaring, written as an arrangement of kernels.

The oracle is an eigendecomposition. If `A = P D P^-1` then
`exp(A) = P exp(D) P^-1` exactly, with `exp(D)` elementwise on the diagonal --
so a matrix built from eigenvalues chosen in advance has an exponential known
in closed form, at any norm, with nothing to compute it but NumPy. That is a
stronger check than comparing against another implementation of the same
algorithm, and it needs no dependency to make it.

The rest are the identities the function has to satisfy whatever route it took:
`exp(A) exp(-A) = I`, `det exp(A) = exp(tr A)`, a nilpotent matrix's series
terminating, and the rotation generator turning into a rotation.

The route itself gets two tests of its own. The degree and the number of
halvings are chosen from the matrix's norm against a table of thresholds, which
makes the function piecewise -- so it is checked immediately either side of
every threshold, where a constant transcribed wrongly leaves the lower degree
short of the precision it is supposed to reach. And the gradient is checked
against finite differences on both sides of the last threshold, because the
scaled path runs the approximant through a `solve` and a chain of squarings
that the unscaled one does not.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import minitensor as mt
from minitensor._matrix import _PADE_THRESHOLDS

RNG = np.random.default_rng(31)


def _t(values, dtype="float64", requires_grad=False):
    return mt.Tensor(
        np.ascontiguousarray(np.asarray(values, dtype=np.float64)),
        dtype=dtype,
        requires_grad=requires_grad,
    )


def _from_eigenvalues(eigenvalues, skew=None):
    """A matrix with these eigenvalues, and its exponential in closed form."""

    size = len(eigenvalues)
    basis = np.linalg.qr(RNG.normal(size=(size, size)))[0]
    if skew is not None:
        basis = basis @ np.diag(skew)
    inverse = np.linalg.inv(basis)
    diagonal = np.asarray(eigenvalues, dtype=float)
    return (
        basis @ np.diag(diagonal) @ inverse,
        basis @ np.diag(np.exp(diagonal)) @ inverse,
    )


def _one_norm(array):
    return np.abs(array).sum(axis=-2).max()


# --- against the closed form ------------------------------------------------


@pytest.mark.parametrize(
    "eigenvalues,skew",
    [
        ([0.003, -0.002, 0.001], None),  # under the first threshold
        ([0.3, -0.2, 0.05, -0.4], None),
        ([2.0, -3.0, 0.5, -1.0], None),
        ([9.0, -7.0, 3.0, -12.0], None),  # scaled and squared
        ([1.5, -2.0, 0.3], [1.0, 2.0, 0.5]),  # not normal
        ([8.0, -6.0, 2.0], [1.0, 3.0, 0.4]),
    ],
)
def test_matrix_exp_matches_the_eigendecomposition(eigenvalues, skew):
    matrix, expected = _from_eigenvalues(eigenvalues, skew)
    np.testing.assert_allclose(
        mt.matrix_exp(_t(matrix)).numpy(), expected, rtol=1e-12, atol=1e-13
    )


def test_the_exponential_of_zero_is_the_identity():
    np.testing.assert_array_equal(
        mt.matrix_exp(_t(np.zeros((4, 4)))).numpy(), np.eye(4)
    )


def test_a_diagonal_matrix_exponentiates_along_its_diagonal():
    diagonal = np.array([0.5, -1.5, 2.0, 0.0])
    np.testing.assert_allclose(
        mt.matrix_exp(_t(np.diag(diagonal))).numpy(),
        np.diag(np.exp(diagonal)),
        rtol=1e-14,
    )


def test_a_nilpotent_matrix_has_a_series_that_stops():
    """`N**4 == 0`, so `exp(N)` is `I + N + N**2/2 + N**3/6` and no more."""

    nilpotent = np.triu(np.ones((4, 4)), 1)
    expected = sum(
        np.linalg.matrix_power(nilpotent, power) / math.factorial(power)
        for power in range(4)
    )
    np.testing.assert_allclose(
        mt.matrix_exp(_t(nilpotent)).numpy(), expected, rtol=1e-14
    )


@pytest.mark.parametrize("angle", [0.0, 0.7, 1.3, 3.0])
def test_the_rotation_generator_exponentiates_to_a_rotation(angle):
    generator = np.array([[0.0, -angle], [angle, 0.0]])
    np.testing.assert_allclose(
        mt.matrix_exp(_t(generator)).numpy(),
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]],
        rtol=1e-13,
        atol=1e-14,
    )


# --- the identities it has to satisfy ---------------------------------------


@pytest.mark.parametrize("scale", [0.01, 0.5, 3.0, 9.0])
def test_the_exponential_of_the_negative_is_the_inverse(scale):
    matrix = RNG.normal(size=(4, 4)) * scale
    forward = mt.matrix_exp(_t(matrix)).numpy()
    backward = mt.matrix_exp(_t(-matrix)).numpy()

    # The identity is exact; the arithmetic is not. Both factors can be large
    # while their product is the identity, so what survives is the cancellation
    # -- and the residual is bounded by the size of what cancelled, not by 1.
    tolerance = 1e-13 * _one_norm(forward) * _one_norm(backward)
    np.testing.assert_allclose(forward @ backward, np.eye(4), rtol=0, atol=tolerance)


@pytest.mark.parametrize("scale", [0.1, 1.0, 4.0])
def test_the_determinant_is_the_exponential_of_the_trace(scale):
    matrix = RNG.normal(size=(3, 3)) * scale
    np.testing.assert_allclose(
        float(mt.det(mt.matrix_exp(_t(matrix))).item()),
        float(np.exp(np.trace(matrix))),
        rtol=1e-11,
    )


def test_matrix_exp_is_not_exp():
    """The confusion the name invites, ruled out."""

    matrix = np.array([[0.0, 1.0], [1.0, 0.0]])
    assert not np.allclose(mt.matrix_exp(_t(matrix)).numpy(), np.exp(matrix))
    np.testing.assert_allclose(
        mt.matrix_exp(_t(matrix)).numpy(),
        [[np.cosh(1.0), np.sinh(1.0)], [np.sinh(1.0), np.cosh(1.0)]],
        rtol=1e-14,
    )


# --- the route it takes -----------------------------------------------------


def test_the_thresholds_rise_with_the_degree():
    for precision, table in _PADE_THRESHOLDS.items():
        degrees = [degree for degree, _ in table]
        limits = [limit for _, limit in table]
        assert degrees == sorted(degrees), precision
        assert limits == sorted(limits), precision


@pytest.mark.parametrize("precision", sorted(_PADE_THRESHOLDS))
def test_every_threshold_is_accurate_on_both_sides_of_itself(precision):
    """Where the degree changes is where a mis-set constant would show.

    Checking the two sides against each other would mostly measure the gap
    between the two matrices, so each side is checked against the closed form
    instead. A threshold set too high leaves the lower degree short of the
    precision it is supposed to reach, and that is what fails here.
    """

    tolerance = 1e-13 if precision == "float64" else 2e-6
    direction = RNG.normal(size=(4, 4))
    direction /= _one_norm(direction)

    for _degree, threshold in _PADE_THRESHOLDS[precision]:
        for side in (1 - 1e-6, 1 + 1e-6):
            matrix = direction * threshold * side
            eigenvalues, basis = np.linalg.eig(matrix)
            reference = np.real(
                basis @ np.diag(np.exp(eigenvalues)) @ np.linalg.inv(basis)
            )
            np.testing.assert_allclose(
                mt.matrix_exp(_t(matrix, precision)).numpy(),
                reference,
                rtol=tolerance,
                atol=tolerance,
            )


def test_a_batch_is_the_same_as_the_matrices_in_it():
    matrices = [RNG.normal(size=(3, 3)) * scale for scale in (0.05, 1.0, 7.0)]
    batched = mt.matrix_exp(_t(np.stack(matrices))).numpy()
    for index, matrix in enumerate(matrices):
        # A batch shares one scaling, chosen from the largest norm in it, so the
        # smaller matrices take more squarings than they need -- which costs
        # rounding, not correctness.
        np.testing.assert_allclose(
            batched[index], mt.matrix_exp(_t(matrix)).numpy(), rtol=1e-9, atol=1e-11
        )


def test_float32_stays_float32_and_lands_within_single_precision():
    matrix, expected = _from_eigenvalues([1.5, -2.0, 0.7])
    got = mt.matrix_exp(mt.Tensor(matrix.astype(np.float32), dtype="float32"))
    assert "float32" in str(got.dtype)
    np.testing.assert_allclose(got.numpy(), expected, rtol=1e-5, atol=1e-6)


# --- the gradient -----------------------------------------------------------


def _finite_difference(matrix, weights, step=1e-6):
    gradient = np.zeros_like(matrix)
    for index in np.ndindex(matrix.shape):
        high, low = matrix.copy(), matrix.copy()
        high[index] += step
        low[index] -= step
        gradient[index] = (
            (mt.matrix_exp(_t(high)).numpy() * weights).sum()
            - (mt.matrix_exp(_t(low)).numpy() * weights).sum()
        ) / (2 * step)
    return gradient


@pytest.mark.parametrize("scale", [0.1, 4.0])
def test_the_gradient_matches_finite_differences(scale):
    """Both sides of the last threshold: without scaling, and with it."""

    matrix = RNG.normal(size=(3, 3)) * scale
    weights = RNG.normal(size=(3, 3))

    values = _t(matrix, requires_grad=True)
    (mt.matrix_exp(values) * _t(weights)).sum().backward()
    got = values.grad.numpy()
    mt.clear_autograd_graph()

    expected = _finite_difference(matrix, weights)
    np.testing.assert_allclose(got, expected, rtol=2e-6, atol=1e-7)


# --- what it refuses --------------------------------------------------------


def test_a_non_square_matrix_is_refused():
    with pytest.raises(ValueError, match="square"):
        mt.matrix_exp(_t(np.zeros((2, 3))))


def test_a_vector_is_refused():
    with pytest.raises(ValueError, match="at least two dimensions"):
        mt.matrix_exp(_t(np.zeros(4)))


def test_an_integer_matrix_is_refused():
    with pytest.raises(ValueError, match="floating-point"):
        mt.matrix_exp(mt.Tensor.zeros([3, 3], dtype="int64"))
