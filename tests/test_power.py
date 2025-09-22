# Copyright (c) 2025 Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import pytest

from minitensor.tensor import Tensor


def test_tensor_pow_scalar():
    x = Tensor([1.0, 2.0, 3.0], dtype="float32")
    y = x**2
    assert np.allclose(y.numpy(), np.array([1.0, 4.0, 9.0], dtype=np.float32))


def test_tensor_pow_tensor():
    base = Tensor([2.0, 3.0, 4.0], dtype="float32")
    exp = Tensor([1.0, 2.0, 0.5], dtype="float32")
    y = base**exp
    expected = np.array([2.0, 9.0, np.sqrt(4.0)], dtype=np.float32)
    assert np.allclose(y.numpy(), expected)


def test_tensor_pow_shape_mismatch_error():
    base = Tensor([1.0, 2.0], dtype="float32")
    exp = Tensor([3.0, 4.0, 5.0], dtype="float32")
    with pytest.raises(ValueError):
        _ = base**exp


def test_tensor_pow_dtype_mismatch_error():
    base = Tensor([1.0, 2.0], dtype="float32")
    exp = Tensor([1.0, 2.0], dtype="float64")
    with pytest.raises(TypeError):
        _ = base**exp


def test_negative_base_fractional_power_nan():
    base = Tensor([-1.0], dtype="float32")
    exp = Tensor([0.5], dtype="float32")
    y = base**exp
    assert np.isnan(y.numpy()[0])


def test_scalar_rpow_tensor():
    exp = Tensor([1.0, 2.0, 3.0], dtype="float32")
    result = 2.0**exp
    expected = np.power(2.0, exp.numpy())
    assert np.allclose(result.numpy(), expected)


def test_scalar_rpow_tensor_grad():
    exp = Tensor([0.3, -1.2, 2.0], dtype="float32", requires_grad=True)
    (2.5**exp).sum().backward()
    expected = np.power(2.5, exp.numpy()) * np.log(2.5)
    assert np.allclose(exp.grad.numpy(), expected, rtol=1e-5, atol=1e-6)


def test_tensor_pow_scalar_base_requires_grad():
    base = Tensor(2.0, dtype="float32", requires_grad=True)
    exp = Tensor([1.0, 2.0, -0.5], dtype="float32")
    (base**exp).sum().backward()
    exp_vals = exp.numpy()
    expected = np.sum(exp_vals * np.power(base.item(), exp_vals - 1.0))
    assert np.allclose(base.grad.numpy(), np.array(expected, dtype=np.float32))


def test_tensor_pow_scalar_exponent_requires_grad():
    base = Tensor([2.0, 3.0], dtype="float32")
    exp = Tensor([1.5], dtype="float32", requires_grad=True)
    (base**exp).sum().backward()
    base_vals = base.numpy()
    expected = np.power(base_vals, exp.item()) * np.log(base_vals)
    assert np.allclose(exp.grad.numpy(), np.array(expected.sum(), dtype=np.float32))


def test_numpy_power_dispatches_to_rust():
    base = Tensor([1.0, 2.0, 3.0], dtype="float32")
    left = np.power(base, 2.0)
    right = np.power(2.0, base)
    assert isinstance(left, Tensor)
    assert isinstance(right, Tensor)
    assert np.allclose(left.numpy(), (base**2.0).numpy())
    assert np.allclose(right.numpy(), (2.0**base).numpy())
