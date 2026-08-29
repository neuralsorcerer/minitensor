# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""Products, distances and statistics built from other operations.

`outer`, `vdot`, `kron`, `dist`, `cdist`, `diff`, `trapezoid`, `cov` and
`corrcoef`. Each carries a NumPy name, so NumPy is the reference for every
value; the tests that are not a NumPy comparison are the ones about gradients,
which NumPy has no opinion on.
"""

import numpy as np
import pytest

import minitensor as mt

RNG = np.random.default_rng(20240829)
VECTOR = np.arange(1.0, 4.0)
OTHER = np.arange(1.0, 5.0)
MATRIX = np.arange(6.0).reshape(2, 3)
SAMPLES = RNG.standard_normal((3, 8))


def _t(array):
    return mt.Tensor(np.ascontiguousarray(array, dtype=np.float64), dtype="float64")


def test_outer_matches_numpy():
    np.testing.assert_allclose(
        mt.outer(_t(VECTOR), _t(OTHER)).numpy(), np.outer(VECTOR, OTHER), rtol=1e-15
    )
    # It flattens, so the rank of the inputs never reaches the result.
    np.testing.assert_allclose(
        mt.outer(_t(MATRIX), _t(VECTOR)).numpy(),
        np.outer(MATRIX, VECTOR),
        rtol=1e-15,
    )


def test_vdot_flattens_where_dot_refuses():
    np.testing.assert_allclose(
        mt.vdot(_t(MATRIX), _t(MATRIX)).item(), np.vdot(MATRIX, MATRIX), rtol=1e-14
    )
    with pytest.raises(ValueError):
        mt.dot(_t(MATRIX), _t(MATRIX))

    with pytest.raises(ValueError, match="same number of elements"):
        mt.vdot(_t(VECTOR), _t(OTHER))


@pytest.mark.parametrize(
    "left, right",
    [
        (VECTOR, OTHER),
        (MATRIX, np.arange(4.0).reshape(2, 2)),
        (MATRIX, VECTOR),
        (VECTOR, MATRIX),
        (np.arange(8.0).reshape(2, 2, 2), np.arange(4.0).reshape(2, 2)),
    ],
    ids=["vectors", "matrices", "matrix-vector", "vector-matrix", "ranks-differ"],
)
def test_kron_matches_numpy_including_mismatched_ranks(left, right):
    np.testing.assert_allclose(
        mt.kron(_t(left), _t(right)).numpy(), np.kron(left, right), rtol=1e-15
    )


def test_kron_is_the_block_scaling_it_claims_to_be():
    # Every element of the first operand times a whole copy of the second.
    small = np.array([[1.0, 2.0], [3.0, 4.0]])
    block = np.array([[10.0, 20.0], [30.0, 40.0]])
    got = mt.kron(_t(small), _t(block)).numpy()
    for row in range(2):
        for column in range(2):
            np.testing.assert_allclose(
                got[2 * row : 2 * row + 2, 2 * column : 2 * column + 2],
                small[row, column] * block,
                rtol=1e-15,
            )


@pytest.mark.parametrize("p", [1.0, 2.0, 3.0, float("inf")])
def test_dist_is_the_norm_of_the_difference(p):
    left, right = RNG.standard_normal(6), RNG.standard_normal(6)
    np.testing.assert_allclose(
        mt.dist(_t(left), _t(right), p).item(),
        np.linalg.norm(left - right, ord=p),
        rtol=1e-13,
    )


@pytest.mark.parametrize("p", [1.0, 2.0, 3.0])
def test_cdist_matches_the_pairwise_definition(p):
    left, right = RNG.standard_normal((4, 3)), RNG.standard_normal((5, 3))
    expected = (np.abs(left[:, None, :] - right[None, :, :]) ** p).sum(-1) ** (1 / p)

    got = mt.cdist(_t(left), _t(right), p)
    assert got.shape == (4, 5)
    np.testing.assert_allclose(got.numpy(), expected, rtol=1e-13)


def test_cdist_batches_and_rejects_mismatched_features():
    batch_left = RNG.standard_normal((2, 4, 3))
    batch_right = RNG.standard_normal((2, 5, 3))
    got = mt.cdist(_t(batch_left), _t(batch_right))
    assert got.shape == (2, 4, 5)
    for index in range(2):
        np.testing.assert_allclose(
            got.numpy()[index],
            np.linalg.norm(
                batch_left[index][:, None, :] - batch_right[index][None, :, :], axis=-1
            ),
            rtol=1e-13,
        )

    with pytest.raises(ValueError, match="matching feature counts"):
        mt.cdist(_t(np.zeros((2, 3))), _t(np.zeros((2, 4))))
    with pytest.raises(ValueError, match="at least two dimensions"):
        mt.cdist(_t(VECTOR), _t(VECTOR))


@pytest.mark.parametrize("n", [0, 1, 2, 3, 5, 6])
def test_diff_matches_numpy_at_every_order(n):
    values = np.array([1.0, 4.0, 9.0, 16.0, 25.0])
    np.testing.assert_allclose(
        mt.diff(_t(values), n).numpy(), np.diff(values, n), rtol=1e-15
    )


@pytest.mark.parametrize("dim", [0, 1, -1, -2])
def test_diff_along_a_chosen_dim(dim):
    np.testing.assert_allclose(
        mt.diff(_t(MATRIX), 1, dim).numpy(), np.diff(MATRIX, 1, axis=dim), rtol=1e-15
    )


def test_diff_rejects_a_scalar_and_a_negative_order():
    with pytest.raises(ValueError, match="at least one dimension"):
        mt.diff(mt.Tensor(1.0, dtype="float64"))
    with pytest.raises(ValueError, match="non-negative order"):
        mt.diff(_t(VECTOR), -1)


def test_trapezoid_matches_numpy():
    values = np.array([1.0, 4.0, 9.0, 16.0, 25.0])
    positions = np.array([0.0, 1.0, 3.0, 4.0, 8.0])

    assert mt.trapezoid(_t(values)).item() == pytest.approx(np.trapezoid(values))
    assert mt.trapezoid(_t(values), dx=0.5).item() == pytest.approx(
        np.trapezoid(values, dx=0.5)
    )
    assert mt.trapezoid(_t(values), _t(positions)).item() == pytest.approx(
        np.trapezoid(values, positions)
    )
    # The alias is the same function, not a second one.
    assert mt.trapz is mt.trapezoid


def test_trapezoid_over_a_batch_shares_one_coordinate():
    grid = np.array([0.0, 1.0, 3.0, 4.0])
    np.testing.assert_allclose(
        mt.trapezoid(_t(np.arange(12.0).reshape(3, 4)), _t(grid)).numpy(),
        np.trapezoid(np.arange(12.0).reshape(3, 4), grid, axis=-1),
        rtol=1e-14,
    )
    np.testing.assert_allclose(
        mt.trapezoid(_t(np.arange(12.0).reshape(3, 4)), dim=0).numpy(),
        np.trapezoid(np.arange(12.0).reshape(3, 4), axis=0),
        rtol=1e-14,
    )


def test_trapezoid_of_a_single_sample_has_no_interval():
    assert mt.trapezoid(_t([5.0])).item() == 0.0


@pytest.mark.parametrize("correction", [0, 1, 2])
def test_cov_matches_numpy(correction):
    np.testing.assert_allclose(
        mt.cov(_t(SAMPLES), correction).numpy(),
        np.cov(SAMPLES, ddof=correction),
        rtol=1e-12,
    )


def test_cov_of_one_variable_is_its_variance():
    got = mt.cov(_t(SAMPLES[0]))
    assert got.ndim() == 0, "a single variable has a scalar variance"
    assert got.item() == pytest.approx(float(np.cov(SAMPLES[0])))


def test_cov_weights_match_numpy():
    counts = np.array([1.0, 2.0, 1.0, 3.0, 1.0, 1.0, 2.0, 1.0])
    reliability = np.abs(RNG.standard_normal(8)) + 0.5

    np.testing.assert_allclose(
        mt.cov(_t(SAMPLES), 1, _t(counts)).numpy(),
        np.cov(SAMPLES, fweights=counts.astype(int)),
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        mt.cov(_t(SAMPLES), 1, None, _t(reliability)).numpy(),
        np.cov(SAMPLES, aweights=reliability),
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        mt.cov(_t(SAMPLES), 1, _t(counts), _t(reliability)).numpy(),
        np.cov(SAMPLES, fweights=counts.astype(int), aweights=reliability),
        rtol=1e-12,
    )


def test_cov_rejects_what_it_cannot_average():
    with pytest.raises(ValueError, match="1-D or 2-D"):
        mt.cov(_t(np.zeros((2, 2, 2))))
    with pytest.raises(ValueError, match="one entry per observation"):
        mt.cov(_t(SAMPLES), 1, _t(np.ones(3)))
    with pytest.raises(ValueError, match="non-positive divisor"):
        mt.cov(_t(SAMPLES), 8)
    with pytest.raises(ValueError, match="positive value"):
        mt.cov(_t(SAMPLES), 1, _t(np.zeros(8)))


def test_corrcoef_matches_numpy_and_stays_in_range():
    got = mt.corrcoef(_t(SAMPLES)).numpy()
    np.testing.assert_allclose(got, np.corrcoef(SAMPLES), rtol=1e-12)
    assert np.all(np.abs(got) <= 1.0), "a correlation cannot exceed 1"
    np.testing.assert_allclose(np.diag(got), np.ones(3), rtol=1e-12)


def test_corrcoef_never_reports_a_correlation_above_one():
    # `c / (sqrt(c) * sqrt(c))` lands a hair either side of 1 in floating
    # point. Below is a rounding error a caller can live with; above is a value
    # outside the range a correlation is defined on, and the clamp is what
    # rules that out.
    row = RNG.standard_normal(10)
    pair = np.vstack([row, 2.0 * row + 3.0])
    got = mt.corrcoef(_t(pair)).numpy()

    assert np.all(got <= 1.0), got
    np.testing.assert_allclose(got, np.ones((2, 2)), rtol=1e-12)


def test_corrcoef_of_one_variable():
    assert mt.corrcoef(_t(SAMPLES[0])).item() == pytest.approx(1.0)


DIFFERENTIABLE = {
    "outer": lambda t: mt.outer(t, _t(OTHER)),
    "vdot": lambda t: mt.vdot(t, t),
    "kron": lambda t: mt.kron(t, _t([2.0, 3.0])),
    "dist": lambda t: mt.dist(t, _t(np.zeros(3))),
    "cdist": lambda t: mt.cdist(t.reshape(3, 1), _t(np.zeros((2, 1)))),
    "diff": lambda t: mt.diff(t),
    "trapezoid": lambda t: mt.trapezoid(t),
    "cov": lambda t: mt.cov(t),
    "corrcoef": lambda t: mt.corrcoef(mt.stack([t, t * 2.0 + 1.0], 0)),
}


@pytest.mark.parametrize("name", sorted(DIFFERENTIABLE))
def test_gradients_match_central_differences(name):
    build = DIFFERENTIABLE[name]
    # Away from zero, where `dist` and `cdist` have a kink and `cov` has none.
    base = np.array([0.7, -1.3, 2.4])

    tensor = mt.Tensor(base.copy(), dtype="float64", requires_grad=True)
    build(tensor).sum().backward()
    analytic = tensor.grad.numpy().copy()
    mt.clear_autograd_graph()

    eps = 1e-6
    numeric = np.zeros_like(base)
    for i in range(base.size):
        up, down = base.copy(), base.copy()
        up[i] += eps
        down[i] -= eps
        numeric[i] = (build(_t(up)).sum().item() - build(_t(down)).sum().item()) / (
            2 * eps
        )
    np.testing.assert_allclose(analytic, numeric, rtol=1e-5, atol=1e-6)


def test_integer_inputs_are_rejected_where_the_answer_would_be_a_float():
    ints = mt.Tensor.arange(0, 6, dtype="int64").reshape(2, 3)
    for call in (
        lambda: mt.cdist(ints, ints),
        lambda: mt.cov(ints),
        lambda: mt.trapezoid(ints),
    ):
        with pytest.raises(ValueError, match="floating point"):
            call()
