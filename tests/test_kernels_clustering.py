# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""`kernels.soft_assignment`, against a reference and against its own limits.

The kernel is one `autograd.Function`, so what is worth checking is not that a
softmax works but the three things that make it a kernel rather than a
composition: that the value is the softmax of the negative squared distances
scaled by the temperature, that the hand-written backward -- written over the
`[n, k]` responsibilities rather than the `[n, k, d]` differences -- is the
derivative of that value, and that the temperature does what a temperature is
supposed to do at both ends of its range.
"""

from __future__ import annotations

import numpy as np
import pytest

import minitensor as mt
from minitensor.kernels import soft_assignment


def _t(values, requires_grad=False):
    return mt.Tensor(
        np.ascontiguousarray(np.asarray(values, dtype=np.float64)),
        dtype="float64",
        requires_grad=requires_grad,
    )


def _reference(points, centroids, temperature):
    """The definition, written the expensive way on purpose.

    `[n, k, d]` differences and an explicit softmax: the thing the kernel
    exists to avoid, which makes it the right thing to check it against.
    """

    differences = points[:, None, :] - centroids[None, :, :]
    distances = (differences**2).sum(-1)
    logits = -distances / (2.0 * temperature)
    logits = logits - logits.max(axis=1, keepdims=True)
    weights = np.exp(logits)
    return weights / weights.sum(axis=1, keepdims=True)


@pytest.fixture
def data():
    generator = np.random.default_rng(0)
    return (
        generator.standard_normal((7, 3)),
        generator.standard_normal((4, 3)),
    )


# --- the value ---------------------------------------------------------------


def test_the_answer_has_a_row_per_point_and_a_column_per_centroid(data):
    points, centroids = data
    out = soft_assignment(_t(points), _t(centroids))
    assert tuple(out.shape) == (7, 4)


def test_every_row_is_a_distribution(data):
    points, centroids = data
    out = soft_assignment(_t(points), _t(centroids)).numpy()
    np.testing.assert_allclose(out.sum(axis=1), np.ones(7), rtol=0, atol=1e-12)
    assert (out > 0.0).all()


@pytest.mark.parametrize("temperature", [0.05, 0.5, 1.0, 4.0])
def test_it_matches_the_definition_written_out(data, temperature):
    points, centroids = data
    out = soft_assignment(_t(points), _t(centroids), temperature).numpy()
    np.testing.assert_allclose(
        out, _reference(points, centroids, temperature), rtol=1e-12
    )


def test_a_point_sitting_on_a_centroid_still_answers_a_distribution():
    points = np.array([[0.0, 0.0]])
    centroids = np.array([[0.0, 0.0], [3.0, 4.0]])
    out = soft_assignment(_t(points), _t(centroids)).numpy()
    np.testing.assert_allclose(out, _reference(points, centroids, 1.0), rtol=1e-12)
    assert out[0, 0] > out[0, 1]


def test_one_centroid_takes_every_point_entirely(data):
    points, _ = data
    out = soft_assignment(_t(points), _t(points[:1])).numpy()
    np.testing.assert_allclose(out, np.ones((7, 1)))


# --- what the temperature does ----------------------------------------------


def test_a_low_temperature_reproduces_the_hard_assignment(data):
    points, centroids = data
    out = soft_assignment(_t(points), _t(centroids), 1e-4).numpy()

    distances = ((points[:, None, :] - centroids[None, :, :]) ** 2).sum(-1)
    hard = np.zeros_like(out)
    hard[np.arange(len(points)), distances.argmin(axis=1)] = 1.0
    np.testing.assert_allclose(out, hard, atol=1e-9)


def test_a_high_temperature_approaches_the_uniform_assignment(data):
    points, centroids = data
    out = soft_assignment(_t(points), _t(centroids), 1e6).numpy()
    np.testing.assert_allclose(out, np.full((7, 4), 0.25), atol=1e-5)


def test_falling_temperature_sharpens_every_row(data):
    points, centroids = data
    sharpness = [
        float((soft_assignment(_t(points), _t(centroids), t).numpy() ** 2).sum())
        for t in (4.0, 1.0, 0.25, 0.05)
    ]
    assert sharpness == sorted(sharpness)


@pytest.mark.parametrize("temperature", [0.0, -1.0, -0.5])
def test_a_temperature_that_is_not_positive_is_refused(temperature):
    with pytest.raises(ValueError, match="positive temperature"):
        soft_assignment(_t([[0.0]]), _t([[1.0]]), temperature)


def test_the_refusal_happens_before_anything_is_computed():
    """The caller's own traceback, not one wrapped in a forward's report."""

    with pytest.raises(ValueError) as raised:
        soft_assignment(_t([[0.0]]), _t([[1.0]]), 0.0)
    assert "custom op" not in str(raised.value)


# --- the gradient ------------------------------------------------------------


def _numeric_jacobian_vector(function, base, cotangent, step=1e-6):
    """`d (out . cotangent) / d base`, one central difference per element."""

    values = np.array(base, dtype=np.float64)
    out = np.empty_like(values)
    flat = values.reshape(-1)
    for index in range(flat.size):
        original = flat[index]

        flat[index] = original + step
        high = float((function(values) * cotangent).sum())

        flat[index] = original - step
        low = float((function(values) * cotangent).sum())

        flat[index] = original
        out.reshape(-1)[index] = (high - low) / (2.0 * step)
    return out


@pytest.mark.parametrize("temperature", [0.5, 1.0, 2.5])
def test_the_gradient_to_the_points_matches_finite_differences(data, temperature):
    points, centroids = data
    cotangent = np.random.default_rng(1).standard_normal((7, 4))

    tensor = _t(points, requires_grad=True)
    (
        soft_assignment(tensor, _t(centroids), temperature) * _t(cotangent)
    ).sum().backward()

    expected = _numeric_jacobian_vector(
        lambda p: _reference(p, centroids, temperature), points, cotangent
    )
    np.testing.assert_allclose(tensor.grad.numpy(), expected, rtol=1e-6, atol=1e-8)


@pytest.mark.parametrize("temperature", [0.5, 1.0, 2.5])
def test_the_gradient_to_the_centroids_matches_finite_differences(data, temperature):
    points, centroids = data
    cotangent = np.random.default_rng(2).standard_normal((7, 4))

    tensor = _t(centroids, requires_grad=True)
    (soft_assignment(_t(points), tensor, temperature) * _t(cotangent)).sum().backward()

    expected = _numeric_jacobian_vector(
        lambda c: _reference(points, c, temperature), centroids, cotangent
    )
    np.testing.assert_allclose(tensor.grad.numpy(), expected, rtol=1e-6, atol=1e-8)


def test_both_arguments_get_a_gradient_from_one_backward(data):
    points, centroids = data
    p, c = _t(points, requires_grad=True), _t(centroids, requires_grad=True)
    soft_assignment(p, c).sum().backward()
    assert p.grad is not None and c.grad is not None


def test_the_rows_summing_to_one_makes_their_total_gradient_vanish(data):
    """A structural check the finite differences cannot fake.

    Every row sums to one whatever the inputs are, so the derivative of that
    sum is exactly zero -- which is the softmax Jacobian-vector product in the
    backward being the real one and not merely close to it.
    """

    points, centroids = data
    p = _t(points, requires_grad=True)
    soft_assignment(p, _t(centroids)).sum().backward()
    np.testing.assert_allclose(p.grad.numpy(), np.zeros_like(points), atol=1e-12)


def test_the_argument_that_asked_for_no_gradient_does_not_get_one(data):
    points, centroids = data
    fixed = _t(points)
    learned = _t(centroids, requires_grad=True)

    out = soft_assignment(fixed, learned)
    assert out.requires_grad
    out.sum().backward()

    assert learned.grad is not None
    assert fixed.grad is None  # what a built-in operation would leave it as


def test_nothing_requiring_a_gradient_leaves_the_answer_off_the_graph(data):
    points, centroids = data
    assert not soft_assignment(_t(points), _t(centroids)).requires_grad


# --- how it is reached -------------------------------------------------------


def test_the_kernel_is_exported_from_the_package():
    assert mt.kernels.soft_assignment is soft_assignment
    assert mt.kernels.__all__ == ["soft_assignment"]


def test_the_kernel_registers_nothing():
    before = set(mt.list_custom_ops_py())
    soft_assignment(_t([[0.0]], requires_grad=True), _t([[1.0]])).sum().backward()
    assert set(mt.list_custom_ops_py()) == before
