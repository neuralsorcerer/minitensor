# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""Gradient clipping and inspection.

`engine::optim::GradientUtils` implemented all of this -- norm computation,
clipping by norm and by value, and the gradient-presence queries -- and none of
it had a binding, so clipping gradients from Python meant reaching into `.grad`
and rescaling by hand.
"""

from __future__ import annotations

import numpy as np
import pytest

import minitensor as mt
from minitensor import nn


def _with_grads(*gradients):
    """Parameters whose `.grad` is exactly the given array.

    `loss = sum(p * g)` has `d loss / d p == g`, so this sets a gradient without
    depending on any particular op's backward being right.
    """
    params = []
    for gradient in gradients:
        gradient = np.asarray(gradient, dtype=np.float64)
        parameter = mt.Tensor(np.ones_like(gradient), dtype="float64").requires_grad_(
            True
        )
        (parameter * mt.as_tensor(gradient)).sum().backward()
        params.append(parameter)
    return params


GRADS = [np.array([3.0, 4.0]), np.array([12.0])]
TOTAL = float(np.sqrt(9 + 16 + 144))  # 13.0


def test_grad_norm_is_the_combined_l2_norm():
    params = _with_grads(*GRADS)
    assert nn.grad_norm(params) == pytest.approx(TOTAL, rel=1e-12)
    # Reading the norm must not change the gradients.
    for parameter, gradient in zip(params, GRADS):
        np.testing.assert_array_equal(parameter.grad.numpy(), gradient)


@pytest.mark.parametrize("max_norm", [1.0, 5.0, 12.999, 13.0, 100.0])
def test_clip_grad_norm_scales_to_the_cap(max_norm):
    params = _with_grads(*GRADS)
    returned = nn.clip_grad_norm_(params, max_norm)

    # The return value is the norm *before* clipping, as in PyTorch.
    assert returned == pytest.approx(TOTAL, rel=1e-12)

    # PyTorch's coefficient, epsilon included.
    coefficient = max_norm / (TOTAL + 1e-6) if TOTAL > max_norm else 1.0
    for parameter, gradient in zip(params, GRADS):
        np.testing.assert_allclose(
            parameter.grad.numpy(), gradient * coefficient, rtol=1e-12
        )

    if TOTAL > max_norm:
        assert nn.grad_norm(params) <= max_norm + 1e-9


def test_clip_grad_norm_leaves_gradients_under_the_cap_alone():
    params = _with_grads(*GRADS)
    nn.clip_grad_norm_(params, 1000.0)
    for parameter, gradient in zip(params, GRADS):
        np.testing.assert_array_equal(parameter.grad.numpy(), gradient)


def test_clip_grad_value_clamps_elementwise():
    gradient = np.array([-5.0, -0.5, 0.0, 0.5, 5.0])

    params = _with_grads(gradient)
    nn.clip_grad_value_(params, 1.0)
    np.testing.assert_allclose(params[0].grad.numpy(), np.clip(gradient, -1.0, 1.0))

    params = _with_grads(gradient)
    nn.clip_grad_value_(params, min_value=-2.0, max_value=0.5)
    np.testing.assert_allclose(params[0].grad.numpy(), np.clip(gradient, -2.0, 0.5))


def test_clipping_works_on_float32_gradients():
    parameter = mt.Tensor(np.ones(3, dtype=np.float32)).requires_grad_(True)
    (
        parameter * mt.as_tensor(np.array([3.0, 4.0, 0.0], dtype=np.float32))
    ).sum().backward()

    assert nn.clip_grad_norm_([parameter], 1.0) == pytest.approx(5.0, rel=1e-6)
    np.testing.assert_allclose(parameter.grad.numpy(), [0.6, 0.8, 0.0], atol=1e-6)


def test_parameters_without_gradients_are_skipped_not_rejected():
    fresh = mt.zeros((3,), requires_grad=True)
    assert nn.grad_norm([fresh]) == 0.0
    assert nn.count_parameters_with_gradients([fresh]) == 0
    nn.clip_grad_norm_([fresh], 1.0)
    nn.clip_grad_value_([fresh], 1.0)


def test_counts_only_parameters_that_hold_a_gradient():
    with_grad = _with_grads(np.array([1.0]))[0]
    without = mt.zeros((3,), requires_grad=True)
    assert nn.count_parameters_with_gradients([with_grad, without]) == 1
    assert nn.count_parameters_with_gradients([]) == 0


@pytest.mark.parametrize(
    "call",
    [
        lambda p: nn.clip_grad_norm_(p, 0.0),
        lambda p: nn.clip_grad_norm_(p, -1.0),
        lambda p: nn.clip_grad_norm_(p, float("nan")),
        lambda p: nn.clip_grad_norm_(p, float("inf")),
        lambda p: nn.clip_grad_value_(p, 0.0),
        lambda p: nn.clip_grad_value_(p),
        lambda p: nn.clip_grad_value_(p, 1.0, min_value=-1.0),
        lambda p: nn.clip_grad_value_(p, min_value=1.0, max_value=-1.0),
        lambda p: nn.clip_grad_value_(p, min_value=-1.0),
    ],
)
def test_invalid_arguments_are_rejected(call):
    with pytest.raises(ValueError):
        call(_with_grads(np.array([1.0])))


def test_clipping_bounds_a_training_step():
    # End to end: with a huge gradient, clipping is what keeps the update
    # bounded by lr * max_norm instead of lr * |grad|.
    parameter = mt.Tensor(np.array([0.0]), dtype="float64").requires_grad_(True)
    optimizer = mt.optim.SGD([parameter], 1.0)

    optimizer.zero_grad(True)
    (parameter * mt.as_tensor(np.array([1000.0]))).sum().backward()
    nn.clip_grad_norm_([parameter], 2.0)
    optimizer.step()

    assert abs(float(parameter.numpy()[0])) == pytest.approx(2.0, rel=1e-5)
