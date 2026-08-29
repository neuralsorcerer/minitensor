# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""Similarity, margin and count losses.

`cosine_similarity`, `margin_ranking_loss`, `hinge_embedding_loss`,
`cosine_embedding_loss`, `triplet_margin_loss`, `soft_margin_loss` and
`poisson_nll_loss` -- each checked against its own definition written out in
NumPy, and at the inputs where writing it out that way breaks down.
"""

import numpy as np
import pytest

import minitensor as mt

F = mt.functional


def _tensor(values, requires_grad=False):
    return mt.Tensor(np.asarray(values, dtype=np.float64), dtype="float64",
                     requires_grad=requires_grad)


def _cosine(a, b, eps=1e-8, axis=1):
    dot = (a * b).sum(axis=axis)
    na = np.maximum(np.linalg.norm(a, axis=axis), eps)
    nb = np.maximum(np.linalg.norm(b, axis=axis), eps)
    return dot / (na * nb)


LEFT = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
RIGHT = np.array([[1.0, 0.0], [1.0, 0.0], [-1.0, 1.0]])
SIGNS = np.array([1.0, -1.0, 1.0])


def test_cosine_similarity_matches_the_definition():
    got = mt.cosine_similarity(_tensor(LEFT), _tensor(RIGHT)).numpy()
    np.testing.assert_allclose(got, _cosine(LEFT, RIGHT), rtol=1e-14)
    np.testing.assert_array_equal(
        F.cosine_similarity(_tensor(LEFT), _tensor(RIGHT)).numpy(), got
    )


def test_cosine_similarity_floors_each_norm_separately():
    # A zero vector has no direction. Flooring the *product* of the norms would
    # divide a dot product of zero by eps and still give zero here, but pairing
    # a tiny vector with a long one is where the two differ: the product form
    # blows the answer far outside [-1, 1].
    tiny = np.array([[1e-30, 0.0]])
    long = np.array([[1e30, 0.0]])
    got = mt.cosine_similarity(_tensor(tiny), _tensor(long)).item()
    assert -1.0 <= got <= 1.0, got

    zero = np.array([[0.0, 0.0]])
    assert mt.cosine_similarity(_tensor(zero), _tensor(long)).item() == 0.0


def test_cosine_similarity_broadcasts_and_takes_a_dim():
    batch = _tensor(LEFT)
    query = _tensor(RIGHT[:1])
    got = mt.cosine_similarity(batch, query).numpy()
    assert got.shape == (3,)
    np.testing.assert_allclose(got, _cosine(LEFT, RIGHT[:1]), rtol=1e-14)

    # Along dim 0 instead: three-element columns rather than two-element rows.
    got = mt.cosine_similarity(_tensor(LEFT), _tensor(RIGHT), dim=0).numpy()
    np.testing.assert_allclose(got, _cosine(LEFT, RIGHT, axis=0), rtol=1e-14)


def test_margin_ranking_loss_matches_the_definition():
    x1 = np.array([3.0, 0.0, 1.0])
    x2 = np.array([1.0, 1.0, 1.0])
    for margin in (0.0, 1.5):
        expected = np.maximum(0.0, -SIGNS * (x1 - x2) + margin)
        got = F.margin_ranking_loss(
            _tensor(x1), _tensor(x2), _tensor(SIGNS), margin, "none"
        ).numpy()
        np.testing.assert_allclose(got, expected, rtol=1e-14)


def test_hinge_embedding_loss_matches_the_definition():
    distances = np.array([0.2, 0.2, 2.0])
    expected = np.where(SIGNS == 1, distances, np.maximum(0.0, 1.0 - distances))
    got = F.hinge_embedding_loss(
        _tensor(distances), _tensor(SIGNS), 1.0, "none"
    ).numpy()
    np.testing.assert_allclose(got, expected, rtol=1e-14)


def test_cosine_embedding_loss_matches_the_definition():
    for margin in (0.0, 0.5):
        cosine = _cosine(LEFT, RIGHT)
        expected = np.where(
            SIGNS == 1, 1.0 - cosine, np.maximum(0.0, cosine - margin)
        )
        got = F.cosine_embedding_loss(
            _tensor(LEFT), _tensor(RIGHT), _tensor(SIGNS), margin, "none"
        ).numpy()
        np.testing.assert_allclose(got, expected, rtol=1e-14)


def test_triplet_margin_loss_matches_the_definition_and_swaps():
    anchor = np.array([[0.0, 0.0], [1.0, 1.0]])
    positive = np.array([[1.0, 0.0], [1.0, 2.0]])
    negative = np.array([[2.0, 0.0], [4.0, 1.0]])

    def distance(u, v):
        return np.linalg.norm(u - v, axis=1)

    expected = np.maximum(
        0.0, distance(anchor, positive) - distance(anchor, negative) + 1.0
    )
    got = F.triplet_margin_loss(
        _tensor(anchor), _tensor(positive), _tensor(negative),
        margin=1.0, eps=0.0, reduction="none",
    ).numpy()
    np.testing.assert_allclose(got, expected, rtol=1e-13)

    swapped = np.maximum(
        0.0,
        distance(anchor, positive)
        - np.minimum(distance(anchor, negative), distance(positive, negative))
        + 1.0,
    )
    got = F.triplet_margin_loss(
        _tensor(anchor), _tensor(positive), _tensor(negative),
        margin=1.0, eps=0.0, swap=True, reduction="none",
    ).numpy()
    np.testing.assert_allclose(got, swapped, rtol=1e-13)
    assert np.any(swapped > expected), "the swap has to change something here"


def test_triplet_margin_loss_takes_other_norms():
    anchor = np.array([[0.0, 0.0]])
    positive = np.array([[3.0, 4.0]])
    negative = np.array([[1.0, 1.0]])
    for p in (1.0, 2.0, 3.0):
        # Along the feature axis: `np.linalg.norm` on a 2-D array without one
        # gives the *matrix* norm, which is a different function entirely.
        def row_norm(u, v):
            return np.sum(np.abs(u - v) ** p, axis=1) ** (1.0 / p)

        expected = max(
            0.0, row_norm(anchor, positive)[0] - row_norm(anchor, negative)[0] + 1.0
        )
        got = F.triplet_margin_loss(
            _tensor(anchor), _tensor(positive), _tensor(negative),
            margin=1.0, p=p, eps=0.0, reduction="none",
        ).item()
        assert got == pytest.approx(expected, rel=1e-12)


def test_soft_margin_loss_matches_the_definition_and_survives_confident_errors():
    scores = np.array([0.0, 2.0, -2.0])
    got = F.soft_margin_loss(_tensor(scores), _tensor(SIGNS), "none").numpy()
    np.testing.assert_allclose(got, np.log1p(np.exp(-SIGNS * scores)), rtol=1e-14)

    # A sample the model got confidently wrong: `exp(800)` overflows, and the
    # loss has to converge on the linear tail instead of returning inf.
    wrong = F.soft_margin_loss(
        _tensor([-800.0]), _tensor([1.0]), "none"
    ).item()
    assert wrong == 800.0
    with np.errstate(over="ignore"):
        assert np.isinf(np.log1p(np.exp(800.0)))


@pytest.mark.parametrize("log_input", [True, False])
def test_poisson_nll_loss_matches_the_definition(log_input):
    rate = np.array([1.0, 2.0, 4.0])
    counts = np.array([0.0, 1.0, 3.0])
    given = np.log(rate) if log_input else rate

    if log_input:
        expected = np.exp(given) - counts * given
    else:
        expected = given - counts * np.log(given + 1e-8)

    got = F.poisson_nll_loss(
        _tensor(given), _tensor(counts), log_input=log_input, reduction="none"
    ).numpy()
    np.testing.assert_allclose(got, expected, rtol=1e-13)


def test_the_poisson_stirling_term_matches_the_log_factorial():
    counts = np.array([0.0, 1.0, 2.0, 5.0])
    rate = np.zeros_like(counts)

    short = F.poisson_nll_loss(
        _tensor(rate), _tensor(counts), full=False, eps=0.0, reduction="none"
    ).numpy()
    full = F.poisson_nll_loss(
        _tensor(rate), _tensor(counts), full=True, eps=0.0, reduction="none"
    ).numpy()

    # Stirling's approximation to log(k!), and exactly zero for k of 0 and 1
    # where log(k!) really is zero.
    stirling = np.where(
        counts > 1,
        counts * np.log(np.maximum(counts, 1.0))
        - counts
        + 0.5 * np.log(2 * np.pi * np.maximum(counts, 1.0)),
        0.0,
    )
    np.testing.assert_allclose(full - short, stirling, atol=1e-13)
    # It is an approximation, but a close one: within 1% of log(5!) already.
    from math import lgamma

    assert stirling[3] == pytest.approx(lgamma(6.0), rel=0.01)


LOSSES = {
    "margin_ranking_loss": lambda t: F.margin_ranking_loss(
        t, _tensor(RIGHT[:, :1].ravel()), _tensor(SIGNS), 0.5, "sum"
    ),
    "hinge_embedding_loss": lambda t: F.hinge_embedding_loss(
        t, _tensor(SIGNS), 1.0, "sum"
    ),
    "soft_margin_loss": lambda t: F.soft_margin_loss(t, _tensor(SIGNS), "sum"),
    "poisson_nll_loss": lambda t: F.poisson_nll_loss(
        t, _tensor([0.0, 1.0, 3.0]), reduction="sum"
    ),
}


@pytest.mark.parametrize("name", sorted(LOSSES))
def test_gradients_match_central_differences(name):
    build = LOSSES[name]
    # Off every kink: no ties in the ranking, nothing sitting on a margin.
    base = np.array([0.3, -0.7, 1.4])

    tensor = _tensor(base, requires_grad=True)
    build(tensor).backward()
    analytic = tensor.grad.numpy().copy()
    mt.clear_autograd_graph()

    eps = 1e-6
    numeric = np.zeros_like(base)
    for i in range(base.size):
        up, down = base.copy(), base.copy()
        up[i] += eps
        down[i] -= eps
        numeric[i] = (build(_tensor(up)).item() - build(_tensor(down)).item()) / (
            2 * eps
        )
    np.testing.assert_allclose(analytic, numeric, rtol=1e-5, atol=1e-7)


def test_cosine_similarity_gradient_matches_central_differences():
    base = LEFT.copy()
    tensor = _tensor(base, requires_grad=True)
    mt.cosine_similarity(tensor, _tensor(RIGHT)).sum().backward()
    analytic = tensor.grad.numpy().copy()
    mt.clear_autograd_graph()

    eps = 1e-6
    numeric = np.zeros_like(base)
    for index in np.ndindex(*base.shape):
        up, down = base.copy(), base.copy()
        up[index] += eps
        down[index] -= eps
        numeric[index] = (
            mt.cosine_similarity(_tensor(up), _tensor(RIGHT)).sum().item()
            - mt.cosine_similarity(_tensor(down), _tensor(RIGHT)).sum().item()
        ) / (2 * eps)
    np.testing.assert_allclose(analytic, numeric, rtol=1e-5, atol=1e-7)


@pytest.mark.parametrize("name", sorted(LOSSES))
def test_reductions_agree_with_each_other(name):
    build = LOSSES[name]
    tensor = _tensor([0.3, -0.7, 1.4])

    # `build` bakes in "sum"; call the underlying op with each reduction.
    op = getattr(F, name)
    args = {
        "margin_ranking_loss": (tensor, _tensor(RIGHT[:, :1].ravel()), _tensor(SIGNS), 0.5),
        "hinge_embedding_loss": (tensor, _tensor(SIGNS), 1.0),
        "soft_margin_loss": (tensor, _tensor(SIGNS)),
        "poisson_nll_loss": (tensor, _tensor([0.0, 1.0, 3.0])),
    }[name]

    each = op(*args, reduction="none").numpy()
    assert op(*args, reduction="sum").item() == pytest.approx(each.sum())
    assert op(*args, reduction="mean").item() == pytest.approx(each.mean())
    with pytest.raises(ValueError, match="reduction"):
        op(*args, reduction="batchmean")


@pytest.mark.parametrize(
    "call, message",
    [
        (lambda: mt.cosine_similarity(_tensor(LEFT), _tensor(RIGHT), eps=0.0),
         "positive eps"),
        (lambda: F.cosine_embedding_loss(
            _tensor(LEFT), _tensor(RIGHT), _tensor(SIGNS), 2.0), "margin in"),
        (lambda: F.triplet_margin_loss(
            _tensor(LEFT), _tensor(RIGHT), _tensor(LEFT), p=0.0), "norm order"),
        (lambda: F.poisson_nll_loss(
            _tensor(LEFT), _tensor(RIGHT), eps=-1.0), "non-negative eps"),
    ],
)
def test_invalid_arguments_are_rejected(call, message):
    with pytest.raises(ValueError, match=message):
        call()


def test_integer_inputs_are_rejected():
    ints = mt.Tensor.arange(0, 4, dtype="int64")
    with pytest.raises(ValueError):
        F.soft_margin_loss(ints, ints)
    with pytest.raises(ValueError):
        mt.cosine_similarity(ints.reshape(2, 2), ints.reshape(2, 2))
