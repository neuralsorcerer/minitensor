# Copyright (c) 2025 Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np

import minitensor as mt


def test_sinh_matches_numpy_and_grad():
    values = np.linspace(-2.0, 2.0, 9, dtype=np.float32)
    tensor = mt.Tensor(values.tolist(), dtype="float32", requires_grad=True)

    result = tensor.sinh()
    np.testing.assert_allclose(result.numpy(), np.sinh(values), rtol=1e-6, atol=1e-6)

    loss = result.sum()
    loss.backward()

    expected_grad = np.cosh(values).astype(np.float32)
    np.testing.assert_allclose(
        tensor.grad.numpy(), expected_grad, rtol=1e-5, atol=1e-6
    )



def test_cosh_matches_numpy_and_grad():
    values = np.linspace(-1.5, 1.5, 7, dtype=np.float64)
    tensor = mt.Tensor(values.tolist(), dtype="float64", requires_grad=True)

    result = tensor.cosh()
    np.testing.assert_allclose(result.numpy(), np.cosh(values), rtol=1e-12, atol=1e-12)

    loss = result.sum()
    loss.backward()

    expected_grad = np.sinh(values)
    np.testing.assert_allclose(
        tensor.grad.numpy(), expected_grad, rtol=1e-12, atol=1e-12
    )



def test_asinh_forward_and_backward():
    values = np.array([-3.0, -0.5, 0.0, 0.75, 2.5], dtype=np.float32)
    tensor = mt.Tensor(values.tolist(), dtype="float32", requires_grad=True)

    result = tensor.asinh()
    np.testing.assert_allclose(result.numpy(), np.arcsinh(values), rtol=1e-6, atol=1e-6)

    loss = result.sum()
    loss.backward()

    expected_grad = (1.0 / np.sqrt(1.0 + values**2)).astype(np.float32)
    np.testing.assert_allclose(
        tensor.grad.numpy(), expected_grad, rtol=1e-5, atol=1e-6
    )



def test_acosh_domain_and_grad():
    values = np.array([1.25, 1.5, 3.0, 5.0], dtype=np.float64)
    tensor = mt.Tensor(values.tolist(), dtype="float64", requires_grad=True)

    result = tensor.acosh()
    np.testing.assert_allclose(result.numpy(), np.arccosh(values), rtol=1e-12, atol=1e-12)

    loss = result.sum()
    loss.backward()

    expected_grad = 1.0 / np.sqrt((values - 1.0) * (values + 1.0))
    np.testing.assert_allclose(
        tensor.grad.numpy(), expected_grad, rtol=1e-12, atol=1e-12
    )



def test_acosh_invalid_inputs_produce_nan():
    tensor = mt.Tensor([0.5, 0.75], dtype="float32")
    result = tensor.acosh()
    assert np.isnan(result.numpy()).all()



def test_atanh_forward_and_backward():
    values = np.array([-0.75, -0.25, 0.25, 0.5], dtype=np.float32)
    tensor = mt.Tensor(values.tolist(), dtype="float32", requires_grad=True)

    result = tensor.atanh()
    np.testing.assert_allclose(result.numpy(), np.arctanh(values), rtol=1e-6, atol=1e-6)

    loss = result.sum()
    loss.backward()

    expected_grad = (1.0 / (1.0 - values**2)).astype(np.float32)
    np.testing.assert_allclose(
        tensor.grad.numpy(), expected_grad, rtol=1e-5, atol=1e-6
    )



def test_functional_and_top_level_forwarders():
    tensors = {
        "sinh": mt.Tensor([-1.25, 0.0, 2.0], dtype="float32"),
        "cosh": mt.Tensor([-1.25, 0.0, 2.0], dtype="float32"),
        "asinh": mt.Tensor([-2.0, -0.5, 1.25], dtype="float32"),
        "acosh": mt.Tensor([1.0, 1.5, 3.0], dtype="float32"),
        "atanh": mt.Tensor([-0.75, 0.0, 0.5], dtype="float32"),
    }

    for name, tensor in tensors.items():
        method_result = getattr(tensor, name)()
        np.testing.assert_allclose(
            getattr(mt.functional, name)(tensor).numpy(), method_result.numpy()
        )
        np.testing.assert_allclose(
            getattr(mt, name)(tensor).numpy(), method_result.numpy()
        )
