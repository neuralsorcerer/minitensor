# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""One notation for every product-and-sum over axes.

`matmul` contracts the last axis of one operand with the second-to-last of
another, `bmm` does it with a batch in front, `dot` does it for vectors, `trace`
sums a diagonal, and an outer product had to be spelled as a broadcast multiply.
Those are the same operation with different axes named, and the library had a
separate name for each arrangement it happened to support and none at all for
the rest. `bhqd,bhkd->bhqk` -- attention scores -- is in the "rest".

Most of these compare against NumPy, which is the specification for this
notation; there is no independent definition to check against and inventing one
would only be re-deriving NumPy's. What is *not* delegated to NumPy is the
question of whether the result was computed the way it must be: the naive
reading of `ij,jk->ik` builds an `i x j x k` intermediate and sums it back down,
which is correct and would need half a gigabyte for a 400x400 product. The
timing test below is the one that would notice.
"""

from __future__ import annotations

import numpy as np
import pytest

import minitensor as mt


def _t(a):
    return mt.Tensor.from_numpy(np.ascontiguousarray(a))


def _rand(*shapes, seed=0):
    rng = np.random.default_rng(seed)
    return [rng.standard_normal(shape) for shape in shapes]


def _check(equation, *shapes, seed=0, atol=1e-12):
    """Against NumPy, which defines the notation."""
    operands = _rand(*shapes, seed=seed)
    expected = np.einsum(equation, *operands)
    got = mt.einsum(equation, *[_t(o) for o in operands]).numpy()
    assert got.shape == expected.shape, f"{equation}: {got.shape} != {expected.shape}"
    assert np.allclose(got, expected, atol=atol), equation
    return got


# --------------------------------------------------------------------------
# The arrangements that already had names
# --------------------------------------------------------------------------


def test_matrix_product():
    _check("ij,jk->ik", (4, 5), (5, 3))


def test_matrix_vector_product():
    _check("ij,j->i", (4, 5), (5,))


def test_inner_product():
    _check("i,i->", (7,), (7,))


def test_outer_product():
    _check("i,j->ij", (4,), (5,))


def test_transpose():
    _check("ij->ji", (3, 5))


def test_trace():
    _check("ii", (5, 5))
    _check("ii->", (5, 5))


def test_diagonal():
    _check("ii->i", (5, 5))


def test_batched_matrix_product():
    _check("bij,bjk->bik", (6, 3, 4), (6, 4, 5))


def test_elementwise_product_and_sum():
    _check("ij,ij->ij", (3, 4), (3, 4))
    _check("ij,ij->", (3, 4), (3, 4))


def test_agrees_with_matmul_and_trace():
    """The same answer as the named operations, which is the point of a
    generalisation: it has to contain what it generalises."""
    a, b = _rand((6, 7), (7, 5), seed=1)
    assert np.allclose(
        mt.einsum("ij,jk->ik", _t(a), _t(b)).numpy(),
        (_t(a) @ _t(b)).numpy(),
        atol=1e-12,
    )
    square = _rand((5, 5), seed=2)[0]
    assert np.isclose(mt.einsum("ii", _t(square)).item(), mt.trace(_t(square)).item())


# --------------------------------------------------------------------------
# The arrangements that did not
# --------------------------------------------------------------------------


def test_attention_scores():
    """`bhqd,bhkd->bhqk`: two batch axes, one contracted, two free.

    Not expressible with `matmul` or `bmm` without a chain of permutes and
    reshapes, which is the reason this operation exists.
    """
    _check("bhqd,bhkd->bhqk", (2, 3, 4, 6), (2, 3, 5, 6))


def test_attention_applied_to_values():
    _check("bhqk,bhkd->bhqd", (2, 3, 4, 5), (2, 3, 5, 6))


def test_bilinear_form():
    _check("i,ij,j->", (4,), (4, 5), (5,))


def test_contract_two_axes_at_once():
    _check("ijkl,jl->ik", (2, 3, 4, 5), (3, 5))


def test_tensor_double_contraction():
    _check("ijk,jkl->il", (2, 3, 4), (3, 4, 5))


def test_three_operand_chain():
    _check("ij,jk,kl->il", (2, 3), (3, 4), (4, 5))


def test_four_operand_chain():
    _check("ij,jk,kl,lm->im", (2, 3), (3, 4), (4, 5), (5, 6))


def test_a_label_shared_by_three_operands():
    _check("ij,ik,il->jkl", (3, 2), (3, 4), (3, 5))


# --------------------------------------------------------------------------
# Implicit output
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "equation,shapes",
    [
        ("ij,jk", [(2, 3), (3, 4)]),
        ("i,i", [(5,), (5,)]),
        ("ii", [(4, 4)]),
        ("ij", [(2, 3)]),
        ("ij,ij", [(2, 3), (2, 3)]),
        ("ijk", [(2, 3, 4)]),
    ],
)
def test_implicit_output_matches_numpy(equation, shapes):
    """Without `->`, the result keeps every subscript used exactly once, in the
    order the letters sort. That is NumPy's rule and there is no better one to
    invent."""
    _check(equation, *shapes)


def test_implicit_output_sorts_by_ascii_not_by_appearance():
    """`Ba,aC` keeps `B` and `C`, and uppercase sorts before lowercase."""
    got = _check("Ba,aC", (2, 3), (3, 4))
    assert got.shape == (2, 4)


def test_implicit_output_puts_the_ellipsis_first():
    _check("...i,...i", (5, 3), (5, 3))


# --------------------------------------------------------------------------
# Ellipsis
# --------------------------------------------------------------------------


def test_ellipsis_over_one_batch_axis():
    _check("...ij,...jk->...ik", (6, 2, 3), (6, 3, 4))


def test_ellipsis_over_two_batch_axes():
    _check("...ij,...jk->...ik", (2, 6, 2, 3), (2, 6, 3, 4))


def test_ellipsis_over_none():
    _check("...ij,...jk->...ik", (2, 3), (3, 4))


@pytest.mark.parametrize(
    "left,right",
    [
        ((2, 3), (6, 3, 4)),
        ((6, 2, 3), (3, 4)),
        ((5, 1, 2, 3), (6, 3, 4)),
        ((4, 5, 2, 3), (5, 3, 4)),
        ((1, 5, 2, 3), (4, 1, 3, 4)),
        ((7, 2, 3), (1, 3, 4)),
    ],
)
def test_ellipsis_broadcasts_across_a_rank_mismatch(left, right):
    """Aligned from the right, as broadcasting is everywhere else.

    This is the one place the implementation could be subtly and invisibly
    wrong: when both operands cover the same number of axes, naming them left
    to right and right to left give the same answer, so only a mismatch -- and a
    mismatch whose extents differ -- can tell the two apart.
    """
    _check("...ij,...jk->...ik", left, right)


@pytest.mark.parametrize("shape", [(2, 3, 4, 5), (7, 2, 3, 4, 5)])
def test_ellipsis_axes_keep_their_order(shape):
    """Distinct extents, so naming the covered axes in the wrong direction
    changes the shape rather than only the values."""
    _check("...ij->...ji", shape)
    _check("...ij,...jk->...ik", shape, shape[:-2] + (shape[-1], 6))


def test_three_operands_with_three_different_ellipsis_ranks():
    _check("...i,...i,...i->...", (3,), (5, 3), (4, 5, 3))


def test_ellipsis_rank_mismatch_with_distinct_extents():
    """Every covered axis a different size, so any misalignment is a shape
    error rather than a silently different answer."""
    _check("...ij,...jk->...ik", (2, 3, 4, 5), (7, 2, 3, 5, 6))


def test_ellipsis_alone():
    _check("...,...->...", (2, 3), (2, 3))


def test_ellipsis_in_the_middle():
    _check("i...j,j->i...", (2, 4, 5, 3), (3,))


def test_ellipsis_can_be_reordered_in_the_output():
    _check("...ij->...ji", (5, 2, 3))


# --------------------------------------------------------------------------
# Reductions and diagonals
# --------------------------------------------------------------------------


def test_sum_every_axis():
    _check("ij->", (3, 4))
    _check("ijk->", (2, 3, 4))


def test_sum_one_axis():
    _check("ij->i", (3, 4))
    _check("ij->j", (3, 4))
    _check("ijk->ik", (2, 3, 4))


def test_diagonal_of_a_batch():
    _check("iij->j", (3, 3, 4))
    _check("iij->ij", (3, 3, 4))


def test_two_separate_diagonals():
    _check("iijj->ij", (3, 3, 4, 4))


def test_a_summed_axis_belonging_to_one_operand():
    """`j` is in the first operand alone and not in the result, so it is summed
    before anything is contracted -- an axis the matrix product never carries."""
    _check("ij,ik->jk", (4, 3), (4, 5))
    _check("ijk,il->jl", (2, 3, 4), (2, 5))


# --------------------------------------------------------------------------
# Shapes, dtypes and degenerate inputs
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "equation,shapes",
    [
        # elementwise, one side flat
        ("ij,ij->ij", [(1, 3), (2, 3)]),
        ("ij,ij->ij", [(2, 1), (2, 3)]),
        ("ij,ij->", [(1, 3), (2, 3)]),
        # a batch axis of one against a real batch
        ("bij,bjk->bik", [(1, 2, 3), (5, 3, 4)]),
        ("bij,bjk->bik", [(5, 2, 3), (1, 3, 4)]),
        # a free axis of one
        ("ij,jk->ik", [(1, 3), (3, 4)]),
        ("ij,jk->ik", [(2, 3), (3, 1)]),
        # a contracted axis of one on both sides at once
        ("ij,jk->ik", [(2, 1), (1, 4)]),
        # two axes of one at the same time
        ("bij,bjk->bik", [(1, 1, 3), (5, 3, 4)]),
        # three operands, each flat somewhere different
        ("ij,ij,ij->ij", [(1, 3), (2, 1), (2, 3)]),
    ],
)
def test_size_one_broadcasts_on_a_letter(equation, shapes):
    """A size-1 axis stretches against a longer one with the same subscript,
    as it does everywhere else in the library and as it does in NumPy.

    It has to happen before anything is contracted: a matrix multiply will not
    broadcast its batch axes, so an operand left flat would either be a shape
    error or, worse, contract against the wrong thing.
    """
    _check(equation, *shapes)


def test_single_operand_is_a_permutation():
    _check("ijk->kji", (2, 3, 4))
    _check("i->i", (5,))


def test_scalar_result_is_zero_dimensional():
    assert mt.einsum("ij->", _t(np.ones((2, 3)))).numpy().shape == ()
    assert mt.einsum("i,i->", _t(np.ones(3)), _t(np.ones(3))).item() == 3.0


def test_float32():
    a, b = _rand((4, 5), (5, 3), seed=3)
    got = mt.einsum("ij,jk->ik", _t(a.astype(np.float32)), _t(b.astype(np.float32)))
    assert got.numpy().dtype == np.float32
    assert np.allclose(got.numpy(), a @ b, atol=1e-4)


def test_integer_dtypes():
    a = np.arange(6, dtype=np.int64).reshape(2, 3)
    b = np.arange(12, dtype=np.int64).reshape(3, 4)
    got = mt.einsum("ij,jk->ik", _t(a), _t(b)).numpy()
    assert got.dtype == np.int64
    assert np.array_equal(got, a @ b)


def test_an_axis_of_length_zero():
    got = mt.einsum("ij,jk->ik", _t(np.zeros((2, 0))), _t(np.zeros((0, 3)))).numpy()
    assert got.shape == (2, 3)
    assert np.array_equal(got, np.zeros((2, 3)))


def test_whitespace_is_ignored():
    a, b = _rand((2, 3), (3, 4), seed=4)
    spaced = mt.einsum(" ij , jk -> ik ", _t(a), _t(b)).numpy()
    assert np.allclose(spaced, a @ b, atol=1e-12)


def test_larger_contraction_matches_numpy():
    _check("bhqd,bhkd->bhqk", (3, 4, 16, 8), (3, 4, 12, 8), seed=5, atol=1e-11)


# --------------------------------------------------------------------------
# What it refuses
# --------------------------------------------------------------------------


def test_rejects_a_wrong_operand_count():
    with pytest.raises(Exception, match="subscripts but"):
        mt.einsum("ij,jk->ik", _t(np.ones((2, 3))))


def test_rejects_a_subscript_that_does_not_match_the_rank():
    with pytest.raises(Exception, match="names"):
        mt.einsum("ij->i", _t(np.ones((2, 3, 4))))


def test_rejects_an_output_subscript_no_operand_has():
    with pytest.raises(Exception, match="which no operand has"):
        mt.einsum("ij->ik", _t(np.ones((2, 3))))


def test_rejects_a_repeated_output_subscript():
    with pytest.raises(Exception, match="twice"):
        mt.einsum("i->ii", _t(np.ones(3)))


def test_rejects_mismatched_extents():
    with pytest.raises(Exception, match="on one operand and"):
        mt.einsum("ij,jk->ik", _t(np.ones((2, 3))), _t(np.ones((4, 5))))


def test_rejects_two_arrows():
    with pytest.raises(Exception, match="more than one"):
        mt.einsum("ij->i->j", _t(np.ones((2, 3))))


def test_rejects_a_non_letter_subscript():
    with pytest.raises(Exception, match="not a valid subscript"):
        mt.einsum("i+j->i", _t(np.ones((2, 3))))


def test_rejects_no_operands():
    with pytest.raises(Exception, match="at least one operand"):
        mt.einsum("ij->i")


# --------------------------------------------------------------------------
# Gradients
# --------------------------------------------------------------------------


def _numeric_grad(f, a, eps=1e-6):
    grad = np.zeros_like(a)
    flat = a.reshape(-1)
    for index in range(flat.size):
        original = flat[index]
        flat[index] = original + eps
        high = f()
        flat[index] = original - eps
        low = f()
        flat[index] = original
        grad.reshape(-1)[index] = (high - low) / (2 * eps)
    return grad


@pytest.mark.parametrize(
    "equation,shapes",
    [
        ("ij,jk->ik", [(2, 3), (3, 4)]),
        ("ij,ij->", [(3, 4), (3, 4)]),
        ("i,j->ij", [(3,), (4,)]),
        ("bhqd,bhkd->bhqk", [(2, 2, 3, 4), (2, 2, 3, 4)]),
        ("ii->i", [(4, 4)]),
        ("ii", [(4, 4)]),
        ("ij->", [(3, 4)]),
        ("ijk,jkl->il", [(2, 3, 4), (3, 4, 5)]),
        ("...ij,...jk->...ik", [(2, 2, 3), (2, 3, 4)]),
    ],
)
def test_gradient_against_finite_differences(equation, shapes):
    """Every step of the plan is an operation that already carries a gradient,
    so this contributes no backward pass of its own -- which is a claim worth
    checking rather than asserting."""
    operands = _rand(*shapes, seed=6)
    weights = np.random.default_rng(7).standard_normal(
        np.einsum(equation, *operands).shape
    )

    def loss():
        return float((np.einsum(equation, *operands) * weights).sum())

    for position in range(len(operands)):
        expected = _numeric_grad(loss, operands[position])
        tensors = [
            mt.Tensor.from_numpy(
                np.ascontiguousarray(o), requires_grad=(index == position)
            )
            for index, o in enumerate(operands)
        ]
        (mt.einsum(equation, *tensors) * _t(weights)).sum().backward()
        assert np.allclose(
            tensors[position].grad.numpy(), expected, atol=1e-6
        ), f"{equation} operand {position}"


def test_gradient_of_a_matrix_product_is_the_matmul_rule():
    """`d(A@B)/dA` contracted with `G` is `G @ B.T`, exactly as `matmul` gives."""
    a, b = _rand((3, 4), (4, 5), seed=8)
    weights = np.random.default_rng(9).standard_normal((3, 5))

    ta = mt.Tensor.from_numpy(a, requires_grad=True)
    (mt.einsum("ij,jk->ik", ta, _t(b)) * _t(weights)).sum().backward()
    assert np.allclose(ta.grad.numpy(), weights @ b.T, atol=1e-11)


def test_gradient_reaches_every_operand_of_a_chain():
    a, b, c = _rand((2, 3), (3, 4), (4, 5), seed=10)
    tensors = [mt.Tensor.from_numpy(x, requires_grad=True) for x in (a, b, c)]
    mt.einsum("ij,jk,kl->il", *tensors).sum().backward()
    for tensor in tensors:
        assert tensor.grad is not None
        assert np.isfinite(tensor.grad.numpy()).all()


def test_no_grad_when_not_required():
    assert not mt.einsum(
        "ij,jk->ik", _t(np.ones((2, 3))), _t(np.ones((3, 4)))
    ).requires_grad


# --------------------------------------------------------------------------
# The plan, not the naive product
# --------------------------------------------------------------------------


def test_a_matrix_product_does_not_build_the_cube():
    """The test that would notice the naive implementation.

    Summing a broadcast `i x j x k` product is a correct reading of
    `ij,jk->ik` and an unusable one: at 200 it is 64 million intermediate
    elements against 40 thousand in the answer. This asserts a bound in time
    rather than in memory because time is what a test can measure, and a
    hundredfold blowup in allocation cannot come in under a few times a plain
    matrix product.
    """
    import time

    size = 200
    a, b = _rand((size, size), (size, size), seed=11)
    ta, tb = _t(a), _t(b)

    def elapsed(call):
        call()
        best = float("inf")
        for _ in range(3):
            start = time.perf_counter()
            call()
            best = min(best, time.perf_counter() - start)
        return best

    contracted = elapsed(lambda: mt.einsum("ij,jk->ik", ta, tb))
    plain = elapsed(lambda: mt.matmul(ta, tb))
    assert contracted < 10 * plain + 5e-3


def test_a_long_chain_stays_within_reach():
    """Four operands contracted pairwise never materialise more than a pair."""
    operands = _rand((30, 30), (30, 30), (30, 30), (30, 30), seed=12)
    got = mt.einsum("ij,jk,kl,lm->im", *[_t(o) for o in operands]).numpy()
    assert np.allclose(got, np.einsum("ij,jk,kl,lm->im", *operands), atol=1e-10)
