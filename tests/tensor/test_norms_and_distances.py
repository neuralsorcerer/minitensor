# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""Norms, and the distances and scalings built from them.

`norm` was the kernel and nothing downstream of it existed: no way to scale a
batch of vectors to unit length, no distance between matched pairs, no
condensed distance matrix, and no norm of a matrix as a matrix rather than as a
bag of numbers. Each of those is a short arrangement, and each is checked
against NumPy where NumPy has it.

Two of them turn on a small constant, and the two use it differently, so each
gets a test rather than a comment. `normalize` uses `eps` as a *floor under the
norm*, which leaves every non-zero vector exactly unit length -- adding `eps`
to the norm instead would shrink all of them, and the length is asserted to be
1 to the last few bits.

`pairwise_distance` adds `eps` to the difference, which biases every distance
upward. That is PyTorch's behaviour and it is kept for compatibility, but the
reason PyTorch needs it does not hold here: a `p`-norm has no derivative at the
origin, and where PyTorch would produce NaN, this library's `norm` answers with
a zero gradient. The test below pins that, because it is what makes `eps=0.0`
-- the true distance -- a safe choice.
"""

from __future__ import annotations

import numpy as np
import pytest

import minitensor as mt

RNG = np.random.default_rng(53)
ORDERS = ["fro", "nuc", 1, -1, 2, -2, np.inf, -np.inf]


def _t(values, requires_grad=False):
    return mt.Tensor(
        np.ascontiguousarray(np.asarray(values, dtype=np.float64)),
        dtype="float64",
        requires_grad=requires_grad,
    )


# --- normalize --------------------------------------------------------------


@pytest.mark.parametrize("p", [1.0, 2.0, 3.0, np.inf])
@pytest.mark.parametrize("dim", [0, 1, -1])
def test_normalize_matches_numpy(p, dim):
    values = RNG.normal(size=(4, 5))
    np.testing.assert_allclose(
        mt.normalize(_t(values), p, dim).numpy(),
        values / np.linalg.norm(values, p, axis=dim, keepdims=True),
        rtol=1e-13,
    )


def test_a_normalized_vector_is_exactly_unit_length():
    """`eps` floors the norm; it is not added to it, so nothing is shrunk."""

    values = RNG.normal(size=(6, 4))
    lengths = np.linalg.norm(mt.normalize(_t(values), 2.0, 1).numpy(), 2, axis=1)
    np.testing.assert_allclose(lengths, np.ones(6), rtol=0, atol=4e-16)


def test_a_zero_vector_normalizes_to_zero_rather_than_dividing_by_zero():
    rows = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 4.0]])
    normalized = mt.normalize(_t(rows), 2.0, 1).numpy()
    np.testing.assert_array_equal(normalized[0], [0.0, 0.0, 0.0])
    np.testing.assert_allclose(normalized[1], [0.6, 0.0, 0.8], rtol=1e-15)


def test_a_tiny_vector_is_floored_rather_than_blown_up():
    """Below `eps` the direction is kept and the length is not restored."""

    tiny = mt.normalize(_t([[1e-20, 0.0]]), 2.0, 1, eps=1e-12).numpy()
    assert tiny[0, 0] == pytest.approx(1e-8)


def test_normalize_carries_a_gradient():
    values = _t(RNG.normal(size=(3, 4)), requires_grad=True)
    mt.normalize(values, 2.0, 1).sum().backward()
    assert np.isfinite(values.grad.numpy()).all()
    mt.clear_autograd_graph()


def test_normalize_refuses_an_integer_tensor():
    with pytest.raises(ValueError, match="floating point"):
        mt.normalize(mt.Tensor.zeros([2, 3], dtype="int64"))


# --- pairwise_distance ------------------------------------------------------


@pytest.mark.parametrize("p", [1.0, 2.0, 3.0])
def test_pairwise_distance_matches_numpy(p):
    left, right = RNG.normal(size=(4, 5)), RNG.normal(size=(4, 5))
    np.testing.assert_allclose(
        mt.pairwise_distance(_t(left), _t(right), p).numpy(),
        np.linalg.norm(left - right + 1e-6, p, axis=-1),
        rtol=1e-13,
    )


def test_pairwise_distance_is_the_diagonal_of_cdist():
    """The same numbers, at `n` distances instead of `n * m`."""

    left, right = RNG.normal(size=(4, 5)), RNG.normal(size=(4, 5))
    np.testing.assert_allclose(
        mt.pairwise_distance(_t(left), _t(right), eps=0.0).numpy(),
        np.diagonal(mt.cdist(_t(left), _t(right)).numpy()),
        rtol=1e-13,
    )


def test_pairwise_distance_broadcasts_a_single_row_against_a_batch():
    batch, single = RNG.normal(size=(4, 5)), RNG.normal(size=(1, 5))
    np.testing.assert_allclose(
        mt.pairwise_distance(_t(batch), _t(single)).numpy(),
        np.linalg.norm(batch - single + 1e-6, 2, axis=-1),
        rtol=1e-13,
    )


def test_pairwise_distance_keeps_the_axis_when_asked():
    left, right = RNG.normal(size=(4, 5)), RNG.normal(size=(4, 5))
    assert tuple(mt.pairwise_distance(_t(left), _t(right), keepdim=True).shape) == (4, 1)


def test_eps_biases_the_distance_upward_by_a_known_amount():
    """Two identical rows are `eps * d ** (1 / p)` apart, not zero."""

    row = _t([[1.0, 2.0, 3.0]])
    assert float(mt.pairwise_distance(row, row).item()) == pytest.approx(
        1e-6 * np.sqrt(3)
    )
    assert float(mt.pairwise_distance(row, row, eps=0.0).item()) == 0.0


@pytest.mark.parametrize("eps", [1e-6, 0.0])
@pytest.mark.parametrize("p", [1.0, 2.0, 3.0])
def test_coincident_rows_have_a_finite_gradient_with_or_without_eps(p, eps):
    """What makes `eps=0.0` safe here, where in torch it would not be.

    A `p`-norm has no derivative at the origin. PyTorch needs `eps` so that a
    loss pulling two rows together does not produce NaN at the moment it
    succeeds; this library's `norm` answers zero for that gradient instead, so
    the shift is a compatibility default and not a requirement.
    """

    same = _t([[1.0, 2.0, 3.0]], requires_grad=True)
    mt.pairwise_distance(same, _t([[1.0, 2.0, 3.0]]), p, eps).sum().backward()
    assert np.isfinite(same.grad.numpy()).all()
    mt.clear_autograd_graph()


# --- pdist ------------------------------------------------------------------


@pytest.mark.parametrize("p", [1.0, 2.0, 3.0, np.inf])
@pytest.mark.parametrize("rows", [1, 2, 5])
def test_pdist_is_the_strict_upper_triangle_of_cdist(p, rows):
    values = RNG.normal(size=(rows, 4))
    first, second = np.triu_indices(rows, 1)
    np.testing.assert_allclose(
        mt.pdist(_t(values), p).numpy(),
        np.linalg.norm(values[first] - values[second], p, axis=-1),
        rtol=1e-13,
    )


def test_pdist_agrees_with_cdist_where_they_overlap():
    values = _t(RNG.normal(size=(5, 3)))
    full = mt.cdist(values, values).numpy()
    np.testing.assert_allclose(
        mt.pdist(values).numpy(), full[np.triu_indices(5, 1)], rtol=1e-12
    )


def test_pdist_counts_the_pairs_and_orders_them_by_row_then_column():
    values = _t([[0.0, 0.0], [3.0, 0.0], [0.0, 4.0]])
    np.testing.assert_allclose(mt.pdist(values).numpy(), [3.0, 4.0, 5.0], rtol=1e-14)


def test_a_single_row_has_no_pairs():
    assert tuple(mt.pdist(_t([[1.0, 2.0]])).shape) == (0,)


def test_pdist_carries_a_gradient():
    values = _t(RNG.normal(size=(4, 3)), requires_grad=True)
    mt.pdist(values).sum().backward()
    assert np.isfinite(values.grad.numpy()).all()
    mt.clear_autograd_graph()


def test_pdist_takes_a_matrix_and_not_a_batch_of_them():
    with pytest.raises(ValueError, match="single matrix of rows"):
        mt.pdist(_t(RNG.normal(size=(2, 3, 4))))


# --- matrix_norm ------------------------------------------------------------


@pytest.mark.parametrize("order", ORDERS)
@pytest.mark.parametrize("shape", [(3, 4), (4, 3), (5, 5), (2, 3, 3)])
def test_matrix_norm_matches_numpy(order, shape):
    values = RNG.normal(size=shape)
    np.testing.assert_allclose(
        mt.matrix_norm(_t(values), order).numpy(),
        np.linalg.norm(values, order, axis=(-2, -1)),
        rtol=1e-11,
    )


def test_matrix_norm_keeps_both_axes_when_asked():
    assert tuple(mt.matrix_norm(_t(RNG.normal(size=(2, 3, 4))), "fro", True).shape) == (
        2,
        1,
        1,
    )


def test_the_frobenius_norm_is_the_elementwise_two_norm():
    values = RNG.normal(size=(3, 4))
    assert float(mt.matrix_norm(_t(values)).item()) == pytest.approx(
        float(np.sqrt((values**2).sum()))
    )


@pytest.mark.parametrize("order", ["fro", 1, np.inf])
def test_the_condition_number_recipe_in_the_docstring_works(order):
    """`matrix_norm(a, p) * matrix_norm(inverse(a), p)`, as documented."""

    values = _t(RNG.normal(size=(4, 4)))
    recipe = mt.matrix_norm(values, order) * mt.matrix_norm(mt.inverse(values), order)
    assert float(recipe.item()) == pytest.approx(
        float(np.linalg.cond(values.numpy(), order)), rel=1e-9
    )


def test_cond_stays_the_two_norm_kernel_it_already_was():
    """`matrix_norm` does not shadow it; the ratio needs no inverse."""

    values = _t(RNG.normal(size=(4, 4)))
    assert float(mt.cond(values).item()) == pytest.approx(
        float(np.linalg.cond(values.numpy())), rel=1e-11
    )


@pytest.mark.parametrize("order", ["ord", 3, 0, None])
def test_an_order_that_is_not_a_matrix_norm_is_refused(order):
    with pytest.raises(ValueError, match="as its order"):
        mt.matrix_norm(_t(RNG.normal(size=(3, 3))), order)


def test_matrix_norm_needs_a_matrix():
    with pytest.raises(ValueError, match="at least two dimensions"):
        mt.matrix_norm(_t([1.0, 2.0]))
