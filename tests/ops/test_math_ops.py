# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import pytest

import minitensor as mt
import minitensor.functional as F
from minitensor.tensor import Tensor


def test_subtraction_broadcasting():
    a = mt.Tensor([[5.0, 6.0], [7.0, 8.0]])
    b = mt.Tensor([1.0, 2.0])
    c = a - b
    expected = np.array([[4.0, 4.0], [6.0, 6.0]])
    np.testing.assert_allclose(c.numpy(), expected)


def test_multiplication_broadcasting():
    a = mt.Tensor([[1.0, 2.0], [3.0, 4.0]])
    b = mt.Tensor(2.0)
    c = a * b
    expected = np.array([[2.0, 4.0], [6.0, 8.0]])
    np.testing.assert_allclose(c.numpy(), expected)


def test_division_broadcasting_and_zero():
    a = mt.Tensor([[1.0, 2.0], [3.0, 4.0]])
    b = mt.Tensor([0.0, 2.0])
    c = a / b
    result = c.numpy()
    assert np.isinf(result[0, 0])
    np.testing.assert_allclose(result[0, 1], 1.0)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_division_by_zero_follows_ieee_754(dtype):
    # -x/0 -> -inf, 0/0 -> nan, +x/0 -> inf. Use 20
    # elements so both the SIMD body and its remainder tail are exercised and
    # must agree.
    pattern = np.array([-1.0, 0.0, 1.0, 2.0] * 5, dtype=dtype)
    zeros = np.zeros(20, dtype=dtype)
    result = (
        mt.tensor(pattern.tolist(), dtype=dtype)
        / mt.tensor(zeros.tolist(), dtype=dtype)
    ).numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        expected = pattern / zeros
    np.testing.assert_array_equal(result, expected)

    # Broadcast (scalar divisor) path must agree with the same-shape path.
    broadcast = (
        mt.tensor(pattern.tolist(), dtype=dtype) / mt.tensor(0.0, dtype=dtype)
    ).numpy()
    np.testing.assert_array_equal(broadcast, expected)


def test_boolean_arithmetic_matches_pytorch():
    a = mt.Tensor([True, False], dtype="bool")
    b = mt.Tensor([False, True], dtype="bool")

    added = a + b
    assert added.dtype == "bool"
    np.testing.assert_array_equal(added.numpy(), np.array([True, True]))

    with pytest.raises(ValueError):
        _ = a - b

    multiplied = a * b
    assert multiplied.dtype == "bool"
    np.testing.assert_array_equal(multiplied.numpy(), np.array([False, False]))

    divided = a / b
    assert divided.dtype == "float32"
    np.testing.assert_allclose(
        divided.numpy(), np.array([np.inf, 0.0], dtype=np.float32)
    )


def test_shape_mismatch_error():
    a = mt.Tensor([1.0, 2.0, 3.0])
    b = mt.Tensor([[1.0, 2.0]])
    with pytest.raises(ValueError):
        _ = a * b


def test_tensor_tensor_dtype_promotion():
    a = mt.Tensor([1.0, 2.0], dtype="float32")
    b = mt.Tensor([1, 2], dtype="int32")
    result = a + b
    assert result.dtype == "float32"
    np.testing.assert_allclose(result.numpy(), np.array([2.0, 4.0], dtype=np.float32))

    c = mt.Tensor([1, 2], dtype="int32")
    d = mt.Tensor([1, 2], dtype="int64")
    promoted = c + d
    assert promoted.dtype == "int64"
    np.testing.assert_array_equal(promoted.numpy(), np.array([2, 4], dtype=np.int64))

    e = mt.Tensor([1, 2], dtype="int32")
    f = mt.Tensor([1, 2], dtype="int32")
    quotient = e / f
    assert quotient.dtype == "float32"
    np.testing.assert_allclose(quotient.numpy(), np.array([1.0, 1.0], dtype=np.float32))


def test_empty_tensor_arithmetic():
    a = mt.Tensor([]).reshape([0])
    b = mt.Tensor([]).reshape([0])
    c = a + b
    m = a * b
    assert c.tolist() == []
    assert m.tolist() == []


def test_nan_propagation():
    a = mt.Tensor([np.nan, 1.0])
    b = mt.Tensor([1.0, 2.0])
    c = a + b
    result = c.numpy()
    assert np.isnan(result[0])
    np.testing.assert_allclose(result[1], 3.0)


def test_inf_minus_inf_nan():
    a = mt.Tensor([np.inf])
    b = mt.Tensor([np.inf])
    c = a - b
    assert np.isnan(c.numpy()).all()


def test_python_float_promotes_int_tensor():
    t = mt.Tensor([1, 2, 3], dtype="int32")
    result = t + 1.5
    assert result.dtype == "float32"
    np.testing.assert_allclose(
        result.numpy(), np.array([2.5, 3.5, 4.5], dtype=np.float32)
    )


def test_python_float_promotes_reverse_add():
    t = mt.Tensor([1, 2, 3], dtype="int32")
    result = 1.5 + t
    assert result.dtype == "float32"
    np.testing.assert_allclose(
        result.numpy(), np.array([2.5, 3.5, 4.5], dtype=np.float32)
    )


def test_python_int_preserves_int_dtype():
    t = mt.Tensor([1, 2, 3], dtype="int32")
    result = t + 1
    assert result.dtype == "int32"
    np.testing.assert_array_equal(result.numpy(), np.array([2, 3, 4], dtype=np.int32))


def test_float64_tensor_with_python_float():
    t = mt.Tensor([1.0, 2.0, 3.0], dtype="float64")
    result = t + 1.5
    assert result.dtype == "float64"
    np.testing.assert_allclose(
        result.numpy(), np.array([2.5, 3.5, 4.5], dtype=np.float64)
    )


def test_boolean_numeric_interactions():
    a = mt.Tensor([True, False], dtype="bool")
    b = mt.Tensor([1, 2], dtype="int32")
    summed = a + b
    assert summed.dtype == "int32"
    np.testing.assert_array_equal(summed.numpy(), np.array([2, 2], dtype=np.int32))

    divided = a / b
    assert divided.dtype == "float32"
    np.testing.assert_allclose(divided.numpy(), np.array([1.0, 0.0], dtype=np.float32))


def test_int64_tensor_with_python_float_promotes_to_float32():
    t = mt.Tensor([1, 2, 3], dtype="int64")
    result = t + 1.5
    assert result.dtype == "float32"
    np.testing.assert_allclose(
        result.numpy(), np.array([2.5, 3.5, 4.5], dtype=np.float32)
    )


def test_reverse_int64_tensor_with_python_float():
    t = mt.Tensor([1, 2, 3], dtype="int64")
    result = 1.5 + t
    assert result.dtype == "float32"
    np.testing.assert_allclose(
        result.numpy(), np.array([2.5, 3.5, 4.5], dtype=np.float32)
    )


def test_maximum_dtype_promotion():
    bools = mt.Tensor([True, False], dtype="bool")
    ints = mt.Tensor([0, 1], dtype="int32")
    floats = mt.Tensor([0.5, -1.5], dtype="float32")

    promoted = bools.maximum(ints)
    assert promoted.dtype == "int32"
    assert np.array_equal(promoted.numpy(), np.array([1, 1], dtype=np.int32))

    promoted_float = bools.maximum(floats)
    assert promoted_float.dtype == "float32"
    assert np.allclose(promoted_float.numpy(), np.array([1.0, 0.0], dtype=np.float32))

    mixed = ints.maximum(mt.Tensor([0.25, 2.5], dtype="float64"))
    assert mixed.dtype == "float64"
    assert np.allclose(mixed.numpy(), np.array([0.25, 2.5], dtype=np.float64))


def test_minimum_dtype_promotion():
    bools = mt.Tensor([True, False], dtype="bool")
    ints = mt.Tensor([0, 1], dtype="int32")
    floats = mt.Tensor([0.5, -1.5], dtype="float32")

    promoted = bools.minimum(ints)
    assert promoted.dtype == "int32"
    assert np.array_equal(promoted.numpy(), np.array([0, 0], dtype=np.int32))

    promoted_float = bools.minimum(floats)
    assert promoted_float.dtype == "float32"
    assert np.allclose(promoted_float.numpy(), np.array([0.5, -1.5], dtype=np.float32))

    mixed = ints.minimum(mt.Tensor([0.25, 2.5], dtype="float64"))
    assert mixed.dtype == "float64"
    assert np.allclose(mixed.numpy(), np.array([0.0, 1.0], dtype=np.float64))


def test_maximum_minimum_nan_behavior():
    a = mt.Tensor([np.nan, 1.0], dtype="float32")
    b = mt.Tensor([0.0, np.nan], dtype="float32")

    max_res = a.maximum(b).numpy()
    min_res = a.minimum(b).numpy()

    assert np.isnan(max_res[0]) and np.isnan(max_res[1])
    assert np.isnan(min_res[0]) and np.isnan(min_res[1])


def test_maximum_backward_flow():
    a = mt.Tensor([-1.0, 2.0, 3.0], requires_grad=True)
    b = mt.Tensor([0.0, 1.5, 3.0], requires_grad=True)

    out = a.maximum(b)
    out.sum().backward()

    np.testing.assert_allclose(
        a.grad.numpy(), np.array([0.0, 1.0, 1.0], dtype=np.float32)
    )
    np.testing.assert_allclose(
        b.grad.numpy(), np.array([1.0, 0.0, 0.0], dtype=np.float32)
    )


def test_minimum_backward_flow():
    a = mt.Tensor([-1.0, 2.0, 3.0], requires_grad=True)
    b = mt.Tensor([0.0, 1.5, 3.0], requires_grad=True)

    out = a.minimum(b)
    out.sum().backward()

    np.testing.assert_allclose(
        a.grad.numpy(), np.array([1.0, 0.0, 1.0], dtype=np.float32)
    )
    np.testing.assert_allclose(
        b.grad.numpy(), np.array([0.0, 1.0, 0.0], dtype=np.float32)
    )


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


def test_sqrt_forward_backward():
    x = Tensor([4.0, 9.0], dtype="float32", requires_grad=True)
    y = x.sqrt()
    assert np.allclose(y.numpy(), np.array([2.0, 3.0], dtype=np.float32))


def test_sqrt_negative_nan():
    x = Tensor([-1.0], dtype="float32")
    y = x.sqrt()
    assert np.isnan(y.numpy()).all()


def test_trigonometric_functions():
    angles = [
        0.0,
        np.pi / 6,
        np.pi / 4,
        np.pi / 2,
        np.pi,
        -np.pi / 6,
        -np.pi / 4,
        -np.pi / 2,
        -np.pi,
    ]
    x = Tensor(angles)
    sin_result = x.sin().tolist()
    cos_result = x.cos().tolist()

    # Test sine and cosine across the full range
    np.testing.assert_allclose(sin_result, np.sin(angles), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(cos_result, np.cos(angles), rtol=1e-6, atol=1e-6)

    # Test tangent only where defined (avoid singularities at ±pi/2)
    tan_angles = [a for a in angles if abs(np.cos(a)) > 1e-6]
    tan_result = Tensor(tan_angles).tan().tolist()
    np.testing.assert_allclose(tan_result, np.tan(tan_angles), rtol=1e-6, atol=1e-6)


def test_trig_large_values():
    x = Tensor([1e10])
    sin_res = x.sin().numpy()[0]
    cos_res = x.cos().numpy()[0]
    np.testing.assert_allclose(sin_res, np.sin(1e10), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(cos_res, np.cos(1e10), rtol=1e-6, atol=1e-6)


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
    np.testing.assert_allclose(
        result.numpy(), np.arccos(values), rtol=1e-12, atol=1e-12
    )

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
    np.testing.assert_allclose(tensor.grad.numpy(), expected_grad, rtol=1e-5, atol=1e-6)


def test_functional_and_top_level_forwarders():
    tensor = mt.Tensor([-0.5, 0.0, 0.5], dtype="float32")

    for name in ("asin", "acos", "atan"):
        method_result = getattr(tensor, name)()
        np.testing.assert_allclose(
            getattr(mt.functional, name)(tensor).numpy(), method_result.numpy()
        )
        np.testing.assert_allclose(
            getattr(mt, name)(tensor).numpy(), method_result.numpy()
        )


def test_sinh_matches_numpy_and_grad():
    values = np.linspace(-2.0, 2.0, 9, dtype=np.float32)
    tensor = mt.Tensor(values.tolist(), dtype="float32", requires_grad=True)

    result = tensor.sinh()
    np.testing.assert_allclose(result.numpy(), np.sinh(values), rtol=1e-6, atol=1e-6)

    loss = result.sum()
    loss.backward()

    expected_grad = np.cosh(values).astype(np.float32)
    np.testing.assert_allclose(tensor.grad.numpy(), expected_grad, rtol=1e-5, atol=1e-6)


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
    np.testing.assert_allclose(tensor.grad.numpy(), expected_grad, rtol=1e-5, atol=1e-6)


def test_acosh_domain_and_grad():
    values = np.array([1.25, 1.5, 3.0, 5.0], dtype=np.float64)
    tensor = mt.Tensor(values.tolist(), dtype="float64", requires_grad=True)

    result = tensor.acosh()
    np.testing.assert_allclose(
        result.numpy(), np.arccosh(values), rtol=1e-12, atol=1e-12
    )

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
    np.testing.assert_allclose(tensor.grad.numpy(), expected_grad, rtol=1e-5, atol=1e-6)


def test_functional_and_top_level_forwarders_hyperbolic_ops():
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


def test_tensor_sign_float_dtype():
    values = np.array([-2.5, 0.0, 3.25, -0.0], dtype=np.float32)
    tensor = mt.Tensor(values.tolist(), dtype="float32")

    result = tensor.sign()

    np.testing.assert_allclose(result.numpy(), np.sign(values))
    assert result.dtype == tensor.dtype


def test_tensor_sign_integer_dtype():
    tensor = mt.Tensor([-3, 0, 4, -7], dtype="int32")

    result = tensor.sign()

    np.testing.assert_array_equal(
        result.numpy(), np.array([-1, 0, 1, -1], dtype=np.int32)
    )
    assert result.dtype == tensor.dtype


def test_tensor_sign_rejects_boolean():
    tensor = mt.Tensor([True, False], dtype="bool")

    with pytest.raises(ValueError):
        tensor.sign()


def test_tensor_reciprocal_matches_numpy():
    values = np.array([2.0, -4.0, 0.25], dtype=np.float32)
    tensor = mt.Tensor(values.tolist(), dtype="float32")

    result = tensor.reciprocal()

    np.testing.assert_allclose(result.numpy(), np.reciprocal(values))
    assert result.dtype == tensor.dtype


def test_reciprocal_backward_propagates_gradients():
    tensor = mt.Tensor([2.0, -4.0], dtype="float32", requires_grad=True)

    reciprocal = tensor.reciprocal()
    loss = reciprocal.sum()
    loss.backward()

    expected_grad = np.array([-0.25, -0.0625], dtype=np.float32)
    np.testing.assert_allclose(tensor.grad.numpy(), expected_grad, rtol=1e-6, atol=1e-7)


def test_reciprocal_rejects_integers():
    tensor = mt.Tensor.arange(1, 4, dtype="int32")

    with pytest.raises(ValueError):
        tensor.reciprocal()


def test_functional_and_top_level_forwarders_sign_and_reciprocal():
    tensor = mt.Tensor([-3.0, -1.0, 0.5], dtype="float32")

    np.testing.assert_allclose(
        mt.functional.sign(tensor).numpy(), tensor.sign().numpy()
    )
    np.testing.assert_allclose(
        mt.functional.reciprocal(tensor).numpy(), tensor.reciprocal().numpy()
    )
    np.testing.assert_allclose(mt.sign(tensor).numpy(), tensor.sign().numpy())
    np.testing.assert_allclose(
        mt.reciprocal(tensor).numpy(), tensor.reciprocal().numpy()
    )


def _make_tensor_values():
    return np.array([-1.234, -0.5, 0.0, 2.718, 3.1415], dtype=np.float32)


def _round_half_away_from_zero(values: np.ndarray, decimals: int = 0) -> np.ndarray:
    multiplier = np.power(10.0, decimals, dtype=values.dtype)
    scaled = values * multiplier
    rounded = np.sign(scaled) * np.floor(np.abs(scaled) + 0.5)
    return rounded / multiplier


def test_tensor_round_defaults_to_zero_decimals():
    values = _make_tensor_values()
    tensor = mt.Tensor(values.tolist(), dtype="float32")

    rounded = tensor.round()

    np.testing.assert_allclose(rounded.numpy(), _round_half_away_from_zero(values))
    assert rounded.dtype == tensor.dtype


def test_tensor_round_with_decimals():
    values = _make_tensor_values()
    tensor = mt.Tensor(values.tolist(), dtype="float32")

    rounded = tensor.round(decimals=2)

    np.testing.assert_allclose(rounded.numpy(), _round_half_away_from_zero(values, 2))
    assert rounded.dtype == tensor.dtype


def test_floor_and_ceil_match_numpy():
    values = _make_tensor_values()
    tensor = mt.Tensor(values.tolist(), dtype="float32")

    floored = tensor.floor()
    ceiled = tensor.ceil()

    np.testing.assert_allclose(floored.numpy(), np.floor(values))
    np.testing.assert_allclose(ceiled.numpy(), np.ceil(values))


def test_rounding_ops_raise_for_integer_tensors():
    tensor = mt.Tensor.arange(-3, 4, dtype="int32")

    with pytest.raises(ValueError):
        tensor.round()

    with pytest.raises(ValueError):
        tensor.floor()

    with pytest.raises(ValueError):
        tensor.ceil()


def test_functional_round_floor_ceil_forwarders():
    values = _make_tensor_values()
    tensor = mt.Tensor(values.tolist(), dtype="float32")

    rounded = mt.functional.round(tensor, decimals=1)
    floored = mt.functional.floor(tensor)
    ceiled = mt.functional.ceil(tensor)

    np.testing.assert_allclose(rounded.numpy(), np.round(values, 1))
    np.testing.assert_allclose(floored.numpy(), np.floor(values))
    np.testing.assert_allclose(ceiled.numpy(), np.ceil(values))


def test_clip_float_range():
    tensor = mt.Tensor([-2.0, -0.5, 0.25, 1.5], dtype="float32")
    clipped = tensor.clip(-1.0, 1.0)
    np.testing.assert_allclose(
        clipped.numpy(), np.array([-1.0, -0.5, 0.25, 1.0], dtype=np.float32)
    )
    assert clipped.dtype == tensor.dtype


def test_clip_with_single_bound_int():
    tensor = mt.Tensor.arange(-3, 3, dtype="int32")
    clipped = tensor.clip(min=0)
    np.testing.assert_array_equal(
        clipped.numpy(), np.clip(np.arange(-3, 3, dtype=np.int32), 0, None)
    )
    assert clipped.dtype == "int32"


def test_clamp_alias_matches_clip():
    tensor = mt.Tensor([-2.0, 0.0, 2.0], dtype="float64")
    clip_result = tensor.clip(-0.5, 0.5)
    clamp_result = tensor.clamp(-0.5, 0.5)
    np.testing.assert_allclose(clip_result.numpy(), clamp_result.numpy())


def test_clamp_min_max_helpers():
    tensor = mt.Tensor([-2.0, 0.0, 2.0], dtype="float64")
    min_only = tensor.clamp_min(-0.25)
    max_only = tensor.clamp_max(1.25)
    np.testing.assert_allclose(
        min_only.numpy(), np.array([-0.25, 0.0, 2.0], dtype=np.float64)
    )
    np.testing.assert_allclose(
        max_only.numpy(), np.array([-2.0, 0.0, 1.25], dtype=np.float64)
    )


def test_clip_raises_for_invalid_bounds():
    tensor = mt.Tensor([1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        tensor.clip(2.0, 1.0)


def test_functional_clip_uses_tensor_method():
    tensor = mt.Tensor([-1.0, 0.25, 1.5])
    clipped = mt.functional.clip(tensor, -0.5, 0.75)
    np.testing.assert_allclose(
        clipped.numpy(), np.array([-0.5, 0.25, 0.75], dtype=np.float32)
    )


def test_nan_to_num_defaults_match_dtype_limits_float32():
    x = mt.Tensor([float("nan"), float("inf"), -float("inf"), -2.5, 3.0])

    out = x.nan_to_num()

    expected = np.nan_to_num(
        np.array([np.nan, np.inf, -np.inf, -2.5, 3.0], dtype=np.float32)
    )
    np.testing.assert_allclose(out.numpy(), expected)
    assert out.dtype == "float32"


def test_nan_to_num_accepts_custom_replacements_float64_and_functional():
    x = mt.Tensor([float("nan"), float("inf"), -float("inf"), 7.0], dtype="float64")

    out = F.nan_to_num(x, nan=-1.5, posinf=9.0, neginf=-9.0)

    expected = np.array([-1.5, 9.0, -9.0, 7.0], dtype=np.float64)
    np.testing.assert_allclose(out.numpy(), expected)
    assert out.dtype == "float64"


def test_nan_to_num_preserves_exact_tensors():
    ints = mt.Tensor([1, -2, 3], dtype="int64")
    bools = mt.Tensor([True, False], dtype="bool")

    np.testing.assert_array_equal(ints.nan_to_num(nan=99.0).numpy(), ints.numpy())
    np.testing.assert_array_equal(
        F.nan_to_num(bools, posinf=5.0).numpy(), bools.numpy()
    )


def test_nan_to_num_empty_tensor_keeps_shape():
    x = mt.empty(0, 3)

    out = x.nan_to_num(nan=1.0, posinf=2.0, neginf=-2.0)

    assert out.shape == (0, 3)
    assert out.numpy().shape == (0, 3)


def test_nan_to_num_backward_masks_replaced_entries():
    x = mt.Tensor(
        [float("nan"), float("inf"), -float("inf"), -2.0, 3.0], requires_grad=True
    )

    y = x.nan_to_num(nan=0.0, posinf=10.0, neginf=-10.0).sum()
    y.backward()

    np.testing.assert_allclose(
        x.grad.numpy(), np.array([0.0, 0.0, 0.0, 1.0, 1.0], dtype=np.float32)
    )
    mt.clear_autograd_graph()


def test_functional_finite_predicates_match_tensor_methods():
    values = mt.Tensor([float("nan"), float("inf"), float("-inf"), -1.5, 0.0])
    expected = {
        "isnan": np.array([True, False, False, False, False], dtype=np.bool_),
        "isinf": np.array([False, True, True, False, False], dtype=np.bool_),
        "isfinite": np.array([False, False, False, True, True], dtype=np.bool_),
    }

    for name, expected_mask in expected.items():
        functional_result = getattr(F, name)(values)
        top_level_result = getattr(mt, name)(values)
        method_result = getattr(values, name)()

        assert functional_result.dtype == "bool"
        assert top_level_result.dtype == "bool"
        np.testing.assert_array_equal(functional_result.numpy(), method_result.numpy())
        np.testing.assert_array_equal(top_level_result.numpy(), expected_mask)


def test_functional_finite_predicates_non_float_and_empty_edges():
    int_values = mt.Tensor([-2, 0, 7], dtype="int32")
    bool_values = mt.Tensor([True, False], dtype="bool")
    empty_values = mt.Tensor([], dtype="float64")

    for values in (int_values, bool_values):
        np.testing.assert_array_equal(
            F.isnan(values).numpy(), np.zeros(values.shape, bool)
        )
        np.testing.assert_array_equal(
            F.isinf(values).numpy(), np.zeros(values.shape, bool)
        )
        np.testing.assert_array_equal(
            F.isfinite(values).numpy(), np.ones(values.shape, bool)
        )

    assert F.isnan(empty_values).shape == (0,)
    assert F.isinf(empty_values).shape == (0,)
    assert F.isfinite(empty_values).shape == (0,)
    np.testing.assert_array_equal(
        F.isnan(empty_values).numpy(), np.array([], dtype=bool)
    )
    np.testing.assert_array_equal(
        F.isinf(empty_values).numpy(), np.array([], dtype=bool)
    )
    np.testing.assert_array_equal(
        F.isfinite(empty_values).numpy(), np.array([], dtype=bool)
    )


def test_pow_broadcasts_shapes():
    rng = np.random.default_rng(11)
    base = np.abs(rng.standard_normal((3, 1, 5))) + 0.5
    exponent = rng.standard_normal((1, 4, 5))

    result = mt.Tensor(base.tolist(), dtype="float64") ** mt.Tensor(
        exponent.tolist(), dtype="float64"
    )
    np.testing.assert_allclose(result.numpy(), base**exponent, rtol=1e-6)

    # trailing-dim broadcast against a 1-D exponent
    base2 = np.abs(rng.standard_normal((3, 4))) + 0.5
    exp2 = rng.standard_normal((4,))
    result2 = mt.Tensor(base2.tolist(), dtype="float64") ** mt.Tensor(
        exp2.tolist(), dtype="float64"
    )
    np.testing.assert_allclose(result2.numpy(), base2**exp2, rtol=1e-6)


def test_pow_broadcast_gradients():
    rng = np.random.default_rng(12)
    base = np.abs(rng.standard_normal((3, 4))) + 0.5
    exponent = rng.standard_normal((4,)) * 0.5

    mb = mt.Tensor(base.tolist(), dtype="float64", requires_grad=True)
    me = mt.Tensor(exponent.tolist(), dtype="float64", requires_grad=True)
    (mb**me).sum().backward()

    expected_base_grad = exponent * base ** (exponent - 1.0)
    expected_exp_grad = (base**exponent * np.log(base)).sum(axis=0)
    np.testing.assert_allclose(mb.grad.numpy(), expected_base_grad, rtol=1e-6)
    np.testing.assert_allclose(me.grad.numpy(), expected_exp_grad, rtol=1e-6)


def test_pow_incompatible_shapes_error():
    a = mt.Tensor(np.ones((2, 3), dtype=np.float32))
    b = mt.Tensor(np.ones((2, 4), dtype=np.float32))
    with pytest.raises(ValueError):
        _ = a**b
