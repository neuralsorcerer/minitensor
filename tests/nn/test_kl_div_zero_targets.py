# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""`kl_div` against a one-hot target returned NaN.

The loss is `target * (log(target) - log(prediction))`. Where the target is
zero that product is `0 * -inf`, which is NaN, and one NaN term takes the whole
reduction with it. A zero-probability class is not an edge case -- a one-hot
target is nothing but zeros and a one, and that is the most common target a
classifier has -- so `kl_div` was unusable for the thing it is most often
reached for.

The term is defined as zero there. It is also its limit as the target goes to
zero, and it is what every other implementation does. So the elementwise result
is masked where the target is zero, which also covers a zero target sitting
opposite a zero prediction, where the log difference is `-inf - -inf`.

A separate defect, fixed at the same time: this takes *probabilities*, and its
docstring said log-probabilities. Passing what the docstring asked for -- which
is also what PyTorch's `kl_div` takes -- gave `inf`, silently. The engine, its
Rust tests, the Python tests and the backward all agree on probabilities, so
the documentation was the thing that was wrong; the cases below pin the
convention so it cannot drift again.
"""

from __future__ import annotations

import numpy as np
import pytest

import minitensor as mt


def _kl(prediction, target, reduction="mean", dtype="float64"):
    return mt.nn.kl_div(
        mt.Tensor(np.asarray(prediction, np.float64), dtype=dtype),
        mt.Tensor(np.asarray(target, np.float64), dtype=dtype),
        reduction,
    )


def test_a_distribution_against_itself_is_zero():
    assert _kl([[0.5, 0.5]], [[0.5, 0.5]], "sum").item() == 0.0
    assert _kl([[0.1, 0.2, 0.7]], [[0.1, 0.2, 0.7]], "sum").item() == pytest.approx(
        0.0, abs=1e-15
    )


@pytest.mark.parametrize("hot", [0, 1, 2, 3])
def test_a_one_hot_target_gives_minus_log_of_that_class(hot):
    """The whole point: this used to be NaN."""
    prediction = np.array([[0.1, 0.2, 0.3, 0.4]])
    target = np.zeros((1, 4))
    target[0, hot] = 1.0
    got = _kl(prediction, target, "sum").item()
    assert got == pytest.approx(-np.log(prediction[0, hot]))


def test_zeros_scattered_through_a_target_contribute_nothing():
    rng = np.random.default_rng(3)
    prediction = rng.dirichlet(np.ones(8), size=16)
    target = rng.dirichlet(np.ones(8), size=16)
    target[target < 0.05] = 0.0

    nonzero = target > 0
    want = np.zeros_like(target)
    want[nonzero] = target[nonzero] * (
        np.log(target[nonzero]) - np.log(prediction[nonzero])
    )

    got = _kl(prediction, target, "none").numpy()
    assert np.isfinite(got).all()
    np.testing.assert_allclose(got, want, rtol=1e-12)


def test_an_all_zero_target_is_zero_rather_than_nan():
    assert _kl([[0.25, 0.75]], [[0.0, 0.0]], "sum").item() == 0.0


def test_a_zero_target_opposite_a_zero_prediction_is_still_zero():
    """`log(0) - log(0)` is `-inf - -inf`, which is NaN before the mask rather
    than the infinity a mask on the logarithm alone would leave."""
    assert _kl([[0.0, 1.0]], [[0.0, 1.0]], "sum").item() == 0.0


def test_a_zero_prediction_under_a_live_target_is_still_infinite():
    """The mask must not swallow the case that genuinely diverges: no amount of
    evidence reconciles a target that puts mass where the prediction puts none."""
    assert np.isinf(_kl([[0.0, 1.0]], [[0.5, 0.5]], "sum").item())


def test_nan_in_the_target_still_propagates():
    assert np.isnan(_kl([[0.5, 0.5]], [[np.nan, 1.0]], "sum").item())


@pytest.mark.parametrize("reduction", ["sum", "mean", "batchmean", "none"])
def test_the_convention_is_probabilities_not_log_probabilities(reduction):
    """Both arguments are probabilities. If this ever moves to PyTorch's
    log-probability convention it has to move deliberately, not by drift."""
    rng = np.random.default_rng(5)
    prediction = rng.dirichlet(np.ones(6), size=8)
    target = rng.dirichlet(np.ones(6), size=8)
    elementwise = target * (np.log(target) - np.log(prediction))

    got = _kl(prediction, target, reduction)
    if reduction == "none":
        np.testing.assert_allclose(got.numpy(), elementwise, rtol=1e-12)
    else:
        want = {
            "sum": elementwise.sum(),
            "mean": elementwise.mean(),
            "batchmean": elementwise.sum() / 8,
        }[reduction]
        assert got.item() == pytest.approx(want, rel=1e-12)


def test_the_gradient_is_finite_where_the_target_is_zero():
    """The prediction gradient is `-target / prediction`, which is zero at a
    zero target -- so masking the forward must not have left a NaN behind it."""
    prediction = mt.Tensor(
        np.array([[0.1, 0.2, 0.3, 0.4]]), dtype="float64", requires_grad=True
    )
    target = mt.Tensor(np.array([[0.0, 0.0, 1.0, 0.0]]), dtype="float64")
    mt.nn.kl_div(prediction, target, "sum").backward()
    grad = prediction.grad.numpy()
    assert np.isfinite(grad).all()
    np.testing.assert_allclose(grad, [[0.0, 0.0, -1.0 / 0.3, 0.0]], rtol=1e-12)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_both_float_dtypes_behave_the_same(dtype):
    got = _kl([[0.1, 0.9]], [[0.0, 1.0]], "sum", dtype=dtype).item()
    assert got == pytest.approx(-np.log(0.9), rel=1e-6)
