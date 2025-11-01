# Copyright (c) 2025 Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import pytest

import minitensor as mt


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_asin_matches_numpy_and_grad(dtype):
    values = np.linspace(-0.9, 0.9, 7, dtype=getattr(np, dtype))
    tensor = mt.Tensor(values.tolist(), dtype=dtype, requires_grad=True)

    result = tensor.asin()
    np.testing.assert_allclose(
        result.numpy(),
        np.arcsin(values),
        rtol=1e-5 if dtype == "float32" else 1e-12,
        atol=1e-6 if dtype == "float32" else 1e-12,
    )

    loss = result.sum()
    loss.backward()

    expected_grad = (1.0 / np.sqrt(1.0 - values**2)).astype(values.dtype)
    np.testing.assert_allclose(
        tensor.grad.numpy(),
        expected_grad,
        rtol=1e-5 if dtype == "float32" else 1e-12,
        atol=1e-6 if dtype == "float32" else 1e-12,
    )


def test_acos_matches_numpy_and_grad():
    values = np.array([-0.85, -0.25, 0.25, 0.85], dtype=np.float64)
    tensor = mt.Tensor(values.tolist(), dtype="float64", requires_grad=True)

    result = tensor.acos()
    np.testing.assert_allclose(result.numpy(), np.arccos(values), rtol=1e-12, atol=1e-12)

    loss = result.sum()
    loss.backward()

    expected_grad = -(1.0 / np.sqrt(1.0 - values**2))
    np.testing.assert_allclose(
        tensor.grad.numpy(), expected_grad, rtol=1e-12, atol=1e-12
    )


def test_asin_and_acos_invalid_inputs_produce_nan():
    tensor = mt.Tensor([1.1, -1.25], dtype="float32")
    asin_result = tensor.asin()
    acos_result = tensor.acos()
    assert np.isnan(asin_result.numpy()).all()
    assert np.isnan(acos_result.numpy()).all()


def test_atan_forward_and_backward():
    values = np.array([-2.0, -0.5, 0.0, 0.75, 1.5], dtype=np.float32)
    tensor = mt.Tensor(values.tolist(), dtype="float32", requires_grad=True)

    result = tensor.atan()
    np.testing.assert_allclose(result.numpy(), np.arctan(values), rtol=1e-6, atol=1e-6)

    loss = result.sum()
    loss.backward()

    expected_grad = (1.0 / (1.0 + values**2)).astype(np.float32)
    np.testing.assert_allclose(
        tensor.grad.numpy(), expected_grad, rtol=1e-5, atol=1e-6
    )


def test_functional_and_top_level_forwarders():
    tensor = mt.Tensor([-0.5, 0.0, 0.5], dtype="float32")

    for name in ("asin", "acos", "atan"):
        method_result = getattr(tensor, name)()
        np.testing.assert_allclose(
            getattr(mt.functional, name)(tensor).numpy(), method_result.numpy()
        )
        np.testing.assert_allclose(getattr(mt, name)(tensor).numpy(), method_result.numpy())
