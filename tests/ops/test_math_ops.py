# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

import math

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


def test_sqrt_rsqrt_ieee_edge_cases():
    # Dedicated sqrt/rsqrt kernels must follow IEEE on the values powf() got
    # wrong: sqrt(-inf)/sqrt(x<0) = NaN (not +inf), sqrt(-0) = -0,
    # rsqrt(-inf) = NaN, rsqrt(-0) = -inf.
    values = np.array([-4.0, -np.inf, -0.0, 0.0, np.inf, 4.0], dtype=np.float64)
    for dtype in ("float32", "float64"):
        v = values.astype(dtype)
        sq = mt.from_numpy(v).sqrt().numpy()
        rs = mt.from_numpy(v).rsqrt().numpy()
        with np.errstate(all="ignore"):
            exp_sq = np.sqrt(v)
            exp_rs = 1.0 / np.sqrt(v)
        np.testing.assert_array_equal(np.isnan(sq), np.isnan(exp_sq))
        np.testing.assert_array_equal(sq[~np.isnan(sq)], exp_sq[~np.isnan(exp_sq)])
        np.testing.assert_array_equal(np.isnan(rs), np.isnan(exp_rs))
        np.testing.assert_array_equal(rs[~np.isnan(rs)], exp_rs[~np.isnan(exp_rs)])
        # signed zero preserved by sqrt(-0) = -0
        assert np.signbit(sq[2])


def test_sqrt_rsqrt_gradients_match_analytic():
    x = Tensor([0.25, 1.0, 4.0, 9.0], dtype="float64", requires_grad=True)
    x.sqrt().sum().backward()
    # d/dx sqrt(x) = 0.5 / sqrt(x)
    np.testing.assert_allclose(
        x.grad.numpy(), 0.5 / np.sqrt([0.25, 1.0, 4.0, 9.0]), rtol=1e-9
    )
    x = Tensor([0.25, 1.0, 4.0, 9.0], dtype="float64", requires_grad=True)
    x.rsqrt().sum().backward()
    # d/dx x**-0.5 = -0.5 * x**-1.5
    np.testing.assert_allclose(
        x.grad.numpy(), -0.5 * np.power([0.25, 1.0, 4.0, 9.0], -1.5), rtol=1e-9
    )


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


def test_tensor_sign_propagates_nan():
    # NaN must propagate through sign, not collapse
    # to 0 (the >0/<0/else fallthrough used to return 0 for NaN).
    for dtype in ("float32", "float64"):
        values = np.array([np.nan, -np.inf, np.inf, -3.0, 0.0], dtype=dtype)
        result = mt.Tensor(values.tolist(), dtype=dtype).sign()
        np.testing.assert_array_equal(result.numpy(), np.sign(values))
        assert np.isnan(result.numpy()[0])


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


def _round_half_to_even(values: np.ndarray, decimals: int = 0) -> np.ndarray:
    """Reference rounding: halves go to the even neighbour.

    This is the IEEE 754 default and what Python's built-in ``round`` does,
    so ``np.round`` is the reference. An earlier version of
    this helper rounded halves away from zero, matching Rust's ``f32::round``,
    which the implementation reached for by default; ``-0.5`` came back as
    ``-1.0`` where every reference library gives ``-0.0``.
    """
    return np.round(values, decimals)


def test_tensor_round_defaults_to_zero_decimals():
    values = _make_tensor_values()
    tensor = mt.Tensor(values.tolist(), dtype="float32")

    rounded = tensor.round()

    np.testing.assert_allclose(rounded.numpy(), _round_half_to_even(values))
    assert rounded.dtype == tensor.dtype


def test_tensor_round_with_decimals():
    values = _make_tensor_values()
    tensor = mt.Tensor(values.tolist(), dtype="float32")

    rounded = tensor.round(decimals=2)

    np.testing.assert_allclose(rounded.numpy(), _round_half_to_even(values, 2))
    assert rounded.dtype == tensor.dtype


# The parameterless rounding modes, each against the NumPy function it is
# named after.
_ROUNDING_MODES = [
    ("floor", np.floor),
    ("ceil", np.ceil),
    ("trunc", np.trunc),
    ("frac", lambda a: a - np.trunc(a)),
]


@pytest.mark.parametrize("name, reference", _ROUNDING_MODES)
def test_rounding_modes_match_numpy(name, reference):
    values = _make_tensor_values()
    tensor = mt.Tensor(values.tolist(), dtype="float32")
    expected = reference(values)

    np.testing.assert_allclose(getattr(tensor, name)().numpy(), expected)
    np.testing.assert_allclose(getattr(mt, name)(tensor).numpy(), expected)
    np.testing.assert_allclose(getattr(mt.functional, name)(tensor).numpy(), expected)


def test_trunc_rounds_towards_zero_where_floor_and_ceil_do_not():
    # The whole content of "towards zero": trunc agrees with floor above zero
    # and with ceil below it, so on a symmetric input it agrees with neither
    # everywhere.
    tensor = mt.Tensor([-2.7, -0.3, 0.3, 2.7], dtype="float64")
    np.testing.assert_allclose(tensor.trunc().numpy(), [-2.0, -0.0, 0.0, 2.0])
    np.testing.assert_allclose(tensor.floor().numpy(), [-3.0, -1.0, 0.0, 2.0])
    np.testing.assert_allclose(tensor.ceil().numpy(), [-2.0, -0.0, 1.0, 3.0])


def test_trunc_and_frac_reconstruct_the_input_exactly():
    values = np.array([-2.75, -0.5, 0.0, 0.5, 2.75, 1e7 + 0.5], dtype=np.float64)
    tensor = mt.Tensor(values.tolist(), dtype="float64")
    whole, fraction = tensor.trunc().numpy(), tensor.frac().numpy()

    # Exactly, not approximately: both terms share an exponent range, so the
    # subtraction that produced `frac` was lossless.
    np.testing.assert_array_equal(whole + fraction, values)
    # `frac` carries the input's sign and never reaches 1.
    assert np.all(np.abs(fraction) < 1.0)
    assert np.all(np.sign(fraction[fraction != 0]) == np.sign(values[fraction != 0]))


def test_frac_of_a_non_finite_input_is_nan():
    tensor = mt.Tensor([float("inf"), float("-inf"), float("nan")], dtype="float64")
    assert np.all(np.isnan(tensor.frac().numpy()))


@pytest.mark.parametrize("name", ["round", "floor", "ceil", "trunc", "frac"])
def test_rounding_ops_raise_for_integer_tensors(name):
    tensor = mt.Tensor.arange(-3, 4, dtype="int32")

    with pytest.raises(ValueError):
        getattr(tensor, name)()


def test_functional_round_forwarder_takes_decimals():
    values = _make_tensor_values()
    tensor = mt.Tensor(values.tolist(), dtype="float32")

    np.testing.assert_allclose(
        mt.functional.round(tensor, decimals=1).numpy(), np.round(values, 1)
    )


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


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_clamp_propagates_nan(dtype):
    # NaN must pass through clamp/clip unchanged, as minitensor's own
    # maximum/minimum do. A previous
    # implementation used Rust f64::max/min, which return the non-NaN operand,
    # so clamp(nan, -1, 1) silently returned the -1 bound.
    tensor = mt.Tensor([float("nan"), 5.0, -5.0, 0.3], dtype=dtype)
    for got in (tensor.clamp(-1.0, 1.0), tensor.clip(-1.0, 1.0)):
        out = got.numpy()
        assert np.isnan(out[0])
        np.testing.assert_allclose(out[1:], np.array([1.0, -1.0, 0.3], dtype=out.dtype))

    nan_vec = mt.Tensor([float("nan"), -2.0, 3.0], dtype=dtype)
    lo = nan_vec.clamp_min(0.0).numpy()
    hi = nan_vec.clamp_max(0.0).numpy()
    assert np.isnan(lo[0]) and np.isnan(hi[0])
    np.testing.assert_allclose(lo[1:], np.array([0.0, 3.0], dtype=lo.dtype))
    np.testing.assert_allclose(hi[1:], np.array([-2.0, 0.0], dtype=hi.dtype))


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


@pytest.mark.parametrize(
    "name, reference, domain",
    [
        ("log2", math.log2, (0.05, 8.0)),
        ("log10", math.log10, (0.05, 8.0)),
        ("erf", math.erf, (-4.0, 4.0)),
        ("erfc", math.erfc, (-4.0, 4.0)),
    ],
)
def test_elementwise_math_matches_stdlib(name, reference, domain):
    values = np.linspace(domain[0], domain[1], 41)
    got = getattr(mt.Tensor(values, dtype="float64"), name)().numpy()
    expected = np.array([reference(v) for v in values])
    np.testing.assert_allclose(got, expected, rtol=1e-12, atol=1e-300)


def test_erfc_keeps_the_tail_that_one_minus_erf_loses():
    """`erfc` exists precisely for the regime where `1 - erf(x)` cancels away.

    Once `erf(x)` rounds to 1 the subtraction returns exactly zero, so the
    dedicated routine is not an optimisation — it is the difference between an
    answer and no answer at all.
    """
    x = mt.Tensor([6.0, 10.0, 20.0], dtype="float64")
    direct = x.erfc().numpy()
    naive = 1.0 - x.erf().numpy()

    np.testing.assert_allclose(
        direct, [math.erfc(6.0), math.erfc(10.0), math.erfc(20.0)], rtol=1e-12
    )
    assert np.all(naive == 0.0), "1 - erf no longer cancels; revisit this test"
    assert np.all(direct > 0.0)


@pytest.mark.parametrize(
    "name, domain",
    [
        ("log2", (0.3, 4.0)),
        ("log10", (0.3, 4.0)),
        ("erf", (-3.0, 3.0)),
        ("erfc", (-3.0, 3.0)),
    ],
)
def test_elementwise_math_gradcheck(name, domain):
    rng = np.random.default_rng(7)
    a = rng.uniform(domain[0], domain[1], size=(3, 4))
    weights = rng.standard_normal((3, 4))

    t = mt.Tensor(a, dtype="float64", requires_grad=True)
    (getattr(t, name)() * mt.Tensor(weights, dtype="float64")).sum().backward()
    analytic = t.grad.numpy()
    mt.clear_autograd_graph()

    h = 1e-6
    for idx in np.ndindex(3, 4):
        plus, minus = a.copy(), a.copy()
        plus[idx] += h
        minus[idx] -= h
        central = float(
            (
                getattr(mt.Tensor(plus, dtype="float64"), name)().numpy()
                - getattr(mt.Tensor(minus, dtype="float64"), name)().numpy()
            )
            .__mul__(weights)
            .sum()
        ) / (2 * h)
        np.testing.assert_allclose(analytic[idx], central, atol=1e-7)


def test_erfc_gradient_is_the_negation_of_erf():
    # erfc = 1 - erf, so the derivatives differ only in sign. Checking the
    # relation catches a dropped sign that a lone gradcheck tolerance might not.
    for x in (-1.0, 0.0, 1.5):
        a = mt.Tensor([x], dtype="float64", requires_grad=True)
        a.erf().backward()
        b = mt.Tensor([x], dtype="float64", requires_grad=True)
        b.erfc().backward()
        assert a.grad.item() == pytest.approx(-b.grad.item(), rel=1e-12)
        mt.clear_autograd_graph()

    # d/dx erf(0) = 2/sqrt(pi)
    origin = mt.Tensor([0.0], dtype="float64", requires_grad=True)
    origin.erf().backward()
    assert origin.grad.item() == pytest.approx(2.0 / math.sqrt(math.pi))
    mt.clear_autograd_graph()


def test_erf_saturates_and_propagates_nan():
    t = mt.Tensor([float("-inf"), 0.0, float("inf"), float("nan")], dtype="float64")
    erf = t.erf().numpy()
    np.testing.assert_array_equal(erf[:3], [-1.0, 0.0, 1.0])
    assert math.isnan(erf[3])

    erfc = t.erfc().numpy()
    np.testing.assert_array_equal(erfc[:3], [2.0, 1.0, 0.0])
    assert math.isnan(erfc[3])


@pytest.mark.parametrize("name", ["log2", "log10"])
def test_log_bases_agree_with_log_on_edge_cases(name):
    # Whatever `log` does at 0, negatives, inf and NaN, the other bases must do
    # too -- they are the same function up to a positive constant factor.
    t = mt.Tensor([0.0, -1.0, float("inf"), float("nan")], dtype="float64")
    base = t.log().numpy()
    other = getattr(t, name)().numpy()
    for b, o in zip(base, other):
        if math.isnan(b):
            assert math.isnan(o)
        else:
            assert math.copysign(1.0, b) == math.copysign(1.0, o)
            assert math.isinf(b) == math.isinf(o)


@pytest.mark.parametrize("name", ["log2", "log10", "erf", "erfc"])
@pytest.mark.parametrize("dtype", ["int64", "bool"])
def test_elementwise_math_rejects_non_float_dtypes(name, dtype):
    with pytest.raises(Exception):
        getattr(mt.Tensor([1, 0], dtype=dtype), name)()


@pytest.mark.parametrize("name", ["log2", "log10", "erf", "erfc"])
def test_elementwise_math_available_as_free_functions(name):
    t = mt.Tensor([1.5, 2.5])
    np.testing.assert_allclose(
        getattr(F, name)(t).numpy(), getattr(t, name)().numpy(), rtol=1e-6
    )
    assert hasattr(mt, name)
