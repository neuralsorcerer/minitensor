# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""A rectangular matrix had no factorisation at all.

`cholesky` wants a positive-definite matrix, `solve` and `det` want a square
one. Anything else -- a design matrix, a stack of observations, a basis someone
wants orthonormalised -- had nothing. That ruled out least squares, which is the
single most common thing anyone does with a non-square matrix, and it is not a
composition of what was here: `Q` comes out of a sequence of reflections, each
built from the column the previous ones left behind, and no arrangement of
`matmul` and `solve` performs that sequence.

Householder reflections rather than Gram-Schmidt, and the difference is not
cosmetic. Gram-Schmidt loses orthogonality in proportion to the condition number
of the input; a product of reflections stays orthogonal to working precision
whatever the input was, because each reflection is orthogonal on its own and the
error has nowhere to accumulate. The ill-conditioned case below measures exactly
that, on a matrix where classical Gram-Schmidt would return a `Q` whose columns
are visibly not perpendicular.

The sign convention is LAPACK's, so `Q` and `R` can be compared to NumPy
element by element rather than only through `Q @ R`. That is a much stronger
assertion and it is used wherever it applies -- a factorisation that is correct
up to a column sign flip would pass every invariant below and still not be the
same answer as everyone else's.
"""

from __future__ import annotations

import numpy as np
import pytest

import minitensor as mt

SHAPES = [(1, 1), (3, 3), (5, 3), (3, 5), (7, 7), (2, 4, 4), (2, 3, 6, 4)]

# One panel is 32 columns and the block path only engages on a wide enough
# trailing block, so these straddle both the panel boundary and the crossover
# between the two code paths.
BOUNDARY_SIZES = [31, 32, 33, 63, 64, 65, 100, 130, 200]

# Big enough that the trailing block is wide enough to be worth blocking, in
# both the factorisation and the accumulation of Q. Anything smaller exercises
# only the direct path, and the two produce the same answer by different
# arithmetic -- so a mistake in the block reflector is invisible below this.
BLOCKED_SHAPES = [(100, 100), (200, 150), (150, 200), (256, 96), (2, 120, 120)]


def _matrix(shape, seed=0):
    return np.random.default_rng(seed).standard_normal(shape)


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
def test_it_matches_numpy(shape):
    """Element by element, not merely up to a column sign: the reflectors use
    LAPACK's convention, so there is no freedom left to differ by."""
    values = _matrix(shape, seed=hash(shape) % 1000)
    q, r = mt.Tensor(values, dtype="float64").qr()
    want_q, want_r = np.linalg.qr(values)
    np.testing.assert_allclose(q.numpy(), want_q, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(r.numpy(), want_r, rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize("n", BOUNDARY_SIZES)
def test_it_matches_numpy_across_the_panel_boundary(n):
    """The factorisation is panelled and switches between a blocked and a direct
    update depending on the size of what is left. Both boundaries live here."""
    values = _matrix((n, n), seed=n)
    q, r = mt.Tensor(values, dtype="float64").qr()
    want_q, want_r = np.linalg.qr(values)
    np.testing.assert_allclose(q.numpy(), want_q, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(r.numpy(), want_r, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("shape", BLOCKED_SHAPES)
def test_the_blocked_update_agrees_with_numpy(shape):
    """Above a size threshold a panel's reflectors are combined into one
    operator and applied as three matrix products instead of one sweep each.
    That is a second piece of arithmetic reaching the same answer, and it only
    runs on inputs this large -- everything smaller tests the other path."""
    values = _matrix(shape, seed=59)
    q, r = mt.Tensor(values, dtype="float64").qr()
    want_q, want_r = np.linalg.qr(values)
    np.testing.assert_allclose(q.numpy(), want_q, rtol=1e-9, atol=1e-11)
    np.testing.assert_allclose(r.numpy(), want_r, rtol=1e-9, atol=1e-11)


@pytest.mark.parametrize("shape", BLOCKED_SHAPES)
def test_the_blocked_update_keeps_q_orthonormal(shape):
    values = _matrix(shape, seed=61)
    q, r = mt.Tensor(values, dtype="float64").qr()
    got_q = q.numpy()
    k = min(shape[-2], shape[-1])
    gram = np.swapaxes(got_q, -1, -2) @ got_q
    np.testing.assert_allclose(
        gram, np.broadcast_to(np.eye(k), shape[:-2] + (k, k)), rtol=0, atol=1e-11
    )
    np.testing.assert_allclose(got_q @ r.numpy(), values, rtol=1e-9, atol=1e-11)


@pytest.mark.parametrize("shape", [(100, 100), (150, 80)])
def test_the_blocked_and_direct_paths_agree_in_complete_mode(shape):
    """Accumulating `Q` takes the blocked path at a different size than the
    factorisation does, because it works over a different number of columns."""
    values = _matrix(shape, seed=67)
    q, r = mt.Tensor(values, dtype="float64").qr("complete")
    want_q, want_r = np.linalg.qr(values, mode="complete")
    np.testing.assert_allclose(q.numpy(), want_q, rtol=1e-9, atol=1e-11)
    np.testing.assert_allclose(r.numpy(), want_r, rtol=1e-9, atol=1e-11)


@pytest.mark.parametrize("shape", SHAPES)
def test_the_factors_reconstruct_the_matrix(shape):
    """Against the definition rather than against NumPy."""
    values = _matrix(shape, seed=3)
    q, r = mt.Tensor(values, dtype="float64").qr()
    np.testing.assert_allclose(
        q.matmul(r).numpy(), values, rtol=1e-10, atol=1e-12
    )


@pytest.mark.parametrize("shape", SHAPES)
def test_q_has_orthonormal_columns(shape):
    values = _matrix(shape, seed=5)
    q, _ = mt.Tensor(values, dtype="float64").qr()
    gram = q.transpose(-2, -1).matmul(q).numpy()
    k = min(shape[-2], shape[-1])
    np.testing.assert_allclose(
        gram, np.broadcast_to(np.eye(k), shape[:-2] + (k, k)), rtol=0, atol=1e-12
    )


@pytest.mark.parametrize("shape", SHAPES)
def test_r_is_upper_triangular(shape):
    values = _matrix(shape, seed=7)
    _, r = mt.Tensor(values, dtype="float64").qr()
    got = r.numpy()
    np.testing.assert_array_equal(got, np.triu(got))


def test_orthogonality_survives_an_ill_conditioned_matrix():
    """The reason it is Householder and not Gram-Schmidt. This matrix has a
    condition number around 1e10; classical Gram-Schmidt would hand back columns
    whose inner products are around 1e-6 instead of 1e-16."""
    n = 12
    vandermonde = np.vander(np.linspace(1.0, 2.0, n), n, increasing=True)
    assert np.linalg.cond(vandermonde) > 1e9, "the test matrix must be nasty"
    q, r = mt.Tensor(vandermonde, dtype="float64").qr()
    gram = q.numpy().T @ q.numpy()
    assert np.abs(gram - np.eye(n)).max() < 1e-13
    np.testing.assert_allclose(
        q.numpy() @ r.numpy(), vandermonde, rtol=1e-10, atol=1e-10
    )


# --- the two modes -----------------------------------------------------------


@pytest.mark.parametrize("shape", [(5, 3), (3, 5), (4, 4), (2, 6, 2)])
def test_complete_matches_numpy(shape):
    values = _matrix(shape, seed=11)
    q, r = mt.Tensor(values, dtype="float64").qr("complete")
    want_q, want_r = np.linalg.qr(values, mode="complete")
    np.testing.assert_allclose(q.numpy(), want_q, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(r.numpy(), want_r, rtol=1e-12, atol=1e-14)


def test_complete_gives_a_square_and_fully_orthogonal_q():
    values = _matrix((6, 2), seed=13)
    q, r = mt.Tensor(values, dtype="float64").qr("complete")
    assert tuple(q.shape) == (6, 6)
    assert tuple(r.shape) == (6, 2)
    got = q.numpy()
    np.testing.assert_allclose(got.T @ got, np.eye(6), rtol=0, atol=1e-13)
    np.testing.assert_allclose(got @ r.numpy(), values, rtol=1e-10, atol=1e-12)


def test_the_two_modes_agree_on_the_columns_they_share():
    """`complete` adds columns; it does not change the ones `reduced` returns."""
    values = _matrix((6, 3), seed=17)
    t = mt.Tensor(values, dtype="float64")
    reduced_q, reduced_r = t.qr()
    complete_q, complete_r = t.qr("complete")
    np.testing.assert_array_equal(complete_q.numpy()[:, :3], reduced_q.numpy())
    np.testing.assert_array_equal(complete_r.numpy()[:3, :], reduced_r.numpy())
    assert np.abs(complete_r.numpy()[3:, :]).max() == 0.0


def test_the_modes_coincide_when_there_are_no_extra_columns():
    """A wide matrix has a square `Q` already, so there is nothing to complete."""
    values = _matrix((3, 7), seed=19)
    t = mt.Tensor(values, dtype="float64")
    for a, b in zip(t.qr(), t.qr("complete")):
        np.testing.assert_array_equal(a.numpy(), b.numpy())


def test_an_unknown_mode_is_refused():
    with pytest.raises(Exception):
        mt.Tensor(np.eye(3), dtype="float64").qr("raw")


# --- shapes and dtypes -------------------------------------------------------


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_both_float_dtypes_are_supported(dtype):
    values = _matrix((6, 4), seed=23).astype(dtype)
    q, r = mt.Tensor(values, dtype=dtype).qr()
    assert q.dtype == dtype and r.dtype == dtype
    tolerance = 1e-5 if dtype == "float32" else 1e-12
    np.testing.assert_allclose(
        q.matmul(r).numpy().astype(np.float64),
        values.astype(np.float64),
        rtol=tolerance,
        atol=tolerance,
    )


@pytest.mark.parametrize("shape", [(0, 0), (0, 3), (3, 0), (0, 4, 4), (2, 0, 0)])
def test_an_empty_input_factors_to_empty_outputs(shape):
    """A matrix with a zero extent still has shapes to report, and they are the
    ones NumPy reports. The zero *batch* is the one that bites: there is no
    matrix to walk, but the loop bound comes from the batch rather than from the
    matrix, and a count that rounds it up to one reads past the end."""
    values = np.zeros(shape)
    q, r = mt.Tensor(values, dtype="float64").qr()
    want_q, want_r = np.linalg.qr(values)
    assert q.numpy().shape == want_q.shape
    assert r.numpy().shape == want_r.shape


def test_a_one_dimensional_input_is_refused():
    with pytest.raises(Exception):
        mt.Tensor(np.ones(4), dtype="float64").qr()


def test_an_integer_input_is_refused():
    with pytest.raises(Exception):
        mt.Tensor(np.eye(3, dtype=np.int64), dtype="int64").qr()


def test_a_rank_deficient_matrix_still_factors():
    """QR does not pivot and does not need to: a dependent column simply leaves
    a zero on `R`'s diagonal, and `Q @ R` is still the input."""
    values = np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]])
    q, r = mt.Tensor(values, dtype="float64").qr()
    np.testing.assert_allclose(q.matmul(r).numpy(), values, rtol=1e-12, atol=1e-14)
    assert abs(r.numpy()[1, 1]) < 1e-14


# --- gradients ---------------------------------------------------------------


@pytest.mark.parametrize("shape", [(4, 3), (3, 3), (5, 5), (3, 5), (2, 4, 3)])
def test_the_gradient_matches_numerical_differentiation(shape):
    """Both outputs at once. The two are separate nodes in the graph -- the
    engine hands a node one gradient at a time -- and this is what checks that
    what they produce adds up to the right total."""
    rng = np.random.default_rng(29)
    values = _matrix(shape, seed=29)
    m, n = shape[-2], shape[-1]
    k = min(m, n)
    weight_q = rng.standard_normal(shape[:-2] + (m, k))
    weight_r = rng.standard_normal(shape[:-2] + (k, n))

    def loss(v):
        q, r = np.linalg.qr(v)
        return float((q * weight_q).sum() + (r * weight_r).sum())

    t = mt.Tensor(values.copy(), dtype="float64", requires_grad=True)
    q, r = t.qr()
    total = (q * mt.Tensor(weight_q, dtype="float64")).sum() + (
        r * mt.Tensor(weight_r, dtype="float64")
    ).sum()
    total.backward()
    np.testing.assert_allclose(
        t.grad.numpy(), _numeric_grad(loss, values.copy()), rtol=1e-5, atol=1e-7
    )


@pytest.mark.parametrize("shape", [(4, 3), (3, 3), (3, 5)])
@pytest.mark.parametrize("which", ["q", "r"])
def test_the_gradient_of_one_output_alone(shape, which):
    """Each output carries its own gradient path, and a caller who uses only one
    of them must get that one's derivative -- not half of a combined answer."""
    rng = np.random.default_rng(31)
    values = _matrix(shape, seed=31)
    m, n = shape[-2], shape[-1]
    k = min(m, n)
    index = 0 if which == "q" else 1
    weights = rng.standard_normal(shape[:-2] + ((m, k) if which == "q" else (k, n)))

    def loss(v):
        return float((np.linalg.qr(v)[index] * weights).sum())

    t = mt.Tensor(values.copy(), dtype="float64", requires_grad=True)
    out = t.qr()[index]
    (out * mt.Tensor(weights, dtype="float64")).sum().backward()
    np.testing.assert_allclose(
        t.grad.numpy(), _numeric_grad(loss, values.copy()), rtol=1e-5, atol=1e-7
    )


def test_the_gradient_of_a_wide_matrix_covers_both_halves():
    """A wide `A` splits at its square block: `Q` is fixed by the first `m`
    columns and the rest enter only through `R2 = Q^T A2`. Dropping either half
    leaves a gradient that is zero where it should not be."""
    rng = np.random.default_rng(37)
    values = _matrix((3, 7), seed=37)
    weights = rng.standard_normal((3, 7))

    def loss(v):
        return float((np.linalg.qr(v)[1] * weights).sum())

    t = mt.Tensor(values.copy(), dtype="float64", requires_grad=True)
    _, r = t.qr()
    (r * mt.Tensor(weights, dtype="float64")).sum().backward()
    grad = t.grad.numpy()
    assert np.abs(grad[:, :3]).max() > 1e-6, "the square half carries gradient"
    assert np.abs(grad[:, 3:]).max() > 1e-6, "so does the rest"
    np.testing.assert_allclose(
        grad, _numeric_grad(loss, values.copy()), rtol=1e-5, atol=1e-7
    )


def test_the_complete_mode_gradient_is_refused_for_a_tall_matrix():
    """`Q`'s extra columns are one arbitrary completion of the basis among many,
    so they are not a function of the input and there is nothing to
    differentiate. Better to say so than to return a gradient for a choice the
    caller never made."""
    t = mt.Tensor(_matrix((5, 3), seed=41), dtype="float64", requires_grad=True)
    with pytest.raises(Exception):
        t.qr("complete")
    # Nothing to complete, so nothing to object to.
    wide = mt.Tensor(_matrix((3, 5), seed=41), dtype="float64", requires_grad=True)
    q, _ = wide.qr("complete")
    q.sum().backward()
    assert wide.grad.numpy().shape == (3, 5)


def test_the_gradient_of_a_singular_matrix_is_refused():
    """The factorisation is fine; the gradient divides by `R`'s diagonal."""
    values = np.array([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    t = mt.Tensor(values, dtype="float64", requires_grad=True)
    q, _ = t.qr()
    with pytest.raises(Exception):
        q.sum().backward()


def test_a_batched_gradient_is_the_per_matrix_gradient():
    values = _matrix((4, 5, 3), seed=43)
    batched = mt.Tensor(values.copy(), dtype="float64", requires_grad=True)
    q, r = batched.qr()
    (q.sum() + r.sum()).backward()

    singles = []
    for i in range(values.shape[0]):
        one = mt.Tensor(values[i].copy(), dtype="float64", requires_grad=True)
        qi, ri = one.qr()
        (qi.sum() + ri.sum()).backward()
        singles.append(one.grad.numpy())
    np.testing.assert_allclose(
        batched.grad.numpy(), np.stack(singles), rtol=1e-12, atol=1e-14
    )


# --- the things it unblocks --------------------------------------------------


def test_least_squares_is_now_expressible():
    """The thing the gap actually blocked. `min ||Ax - b||` is `R x = Q^T b`,
    and there was no way to get either factor."""
    rng = np.random.default_rng(47)
    a = rng.standard_normal((20, 4))
    b = rng.standard_normal((20, 1))

    t = mt.Tensor(a, dtype="float64")
    q, r = t.qr()
    projected = q.transpose(0, 1).matmul(mt.Tensor(b, dtype="float64"))
    solution = r.solve(projected).numpy()

    want = np.linalg.lstsq(a, b, rcond=None)[0]
    np.testing.assert_allclose(solution, want, rtol=1e-9, atol=1e-11)


def test_a_basis_can_now_be_orthonormalised():
    """Nearly-parallel columns in, a perpendicular basis for the same span
    out -- and the span is checked, not just the perpendicularity."""
    basis = np.array([[1.0, 1.0, 1.0], [1.0, 1.0 + 1e-8, 1.0], [1.0, 1.0, 1.0 + 1e-8]])
    q, _ = mt.Tensor(basis, dtype="float64").qr()
    got = q.numpy()
    np.testing.assert_allclose(got.T @ got, np.eye(3), rtol=0, atol=1e-12)
    # The span is unchanged: projecting the input onto Q and back is the input.
    np.testing.assert_allclose(got @ (got.T @ basis), basis, rtol=1e-8, atol=1e-12)


def test_the_module_level_function_agrees_with_the_method():
    values = _matrix((5, 3), seed=53)
    t = mt.Tensor(values, dtype="float64")
    for mode in ("reduced", "complete"):
        for a, b in zip(mt.qr(t, mode), t.qr(mode)):
            np.testing.assert_array_equal(a.numpy(), b.numpy())


@pytest.mark.parametrize("magnitude", [1e160, 1e200, 1e250, 1e-160, 1e-200, 1e-250])
def test_extreme_magnitudes(magnitude):
    """Entries past where a sum of squares survives.

    The reflector's length is the one place the factorisation squares anything,
    and squaring overflows above about `1e154` in double precision. Before the
    reflector scaled its input, this returned `NaN` above that -- and below
    `1e-154` the sum underflowed to zero, so no reflector was built at all and
    `Q` came back as the identity: perfectly orthogonal, and a residual of order
    one, with nothing raised.
    """
    a = np.random.default_rng(3).standard_normal((6, 5)) * magnitude
    q, r = mt.qr(mt.Tensor.from_numpy(a))
    q, r = q.numpy(), r.numpy()

    assert np.isfinite(q).all() and np.isfinite(r).all()
    assert np.allclose(q.T @ q, np.eye(q.shape[1]), atol=1e-13)
    assert np.abs(q @ r - a).max() <= 1e-13 * magnitude
