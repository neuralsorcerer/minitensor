# Copyright (c) 2025 Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

import math

import numpy as np
import pytest

import minitensor as mt
from minitensor.numpy_compat import empty_like
from minitensor.tensor import Tensor


def test_eye_int32():
    x = Tensor.eye(3, dtype="int32")
    assert x.dtype == "int32"
    assert np.array_equal(x.numpy(), np.eye(3, dtype=np.int32))


def test_full_bool():
    x = Tensor.full([2, 2], 1, dtype="bool")
    assert x.dtype == "bool"
    assert np.array_equal(x.numpy(), np.ones((2, 2), dtype=bool))


def test_arange_int64():
    x = Tensor.arange(0, 5, dtype="int64")
    assert x.dtype == "int64"
    assert np.array_equal(x.numpy(), np.arange(0, 5, dtype=np.int64))


def test_rand_bool_shape_and_dtype():
    x = Tensor.rand(2, 2, dtype="bool")
    arr = x.numpy()
    assert x.dtype == "bool"
    assert arr.dtype == np.bool_
    assert arr.shape == (2, 2)


def test_randn_int32_dtype():
    x = Tensor.randn(3, dtype="int32")
    arr = x.numpy()
    assert x.dtype == "int32"
    assert arr.dtype == np.int32
    assert arr.shape == (3,)


def test_uniform_respects_bounds_and_dtype():
    low, high = -2.5, 4.25
    x = Tensor.uniform(4, 5, low=low, high=high, dtype="float64")
    arr = x.numpy()

    assert x.dtype == "float64"
    assert arr.dtype == np.float64
    assert arr.shape == (4, 5)
    assert (arr >= low).all()
    assert (arr < high).all()


def _truncated_normal_moments(
    mean: float, std: float, lower: float, upper: float
) -> tuple[float, float]:
    alpha = (lower - mean) / std
    beta = (upper - mean) / std

    def _pdf(x: float) -> float:
        return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)

    def _cdf(x: float) -> float:
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

    z = _cdf(beta) - _cdf(alpha)
    mean_offset = (_pdf(alpha) - _pdf(beta)) / z
    variance_term = 1.0 + (alpha * _pdf(alpha) - beta * _pdf(beta)) / z - mean_offset**2
    expected_mean = mean + mean_offset * std
    expected_var = variance_term * std * std
    return expected_mean, expected_var


def test_truncated_normal_matches_bounds_and_moments():
    mean, std = 0.25, 1.5
    lower, upper = -1.0, 2.0
    tensor = Tensor.truncated_normal(4096, mean=mean, std=std, lower=lower, upper=upper)
    values = tensor.numpy()
    expected_mean, expected_var = _truncated_normal_moments(mean, std, lower, upper)

    assert tensor.dtype == "float32"
    assert values.dtype == np.float32
    assert values.shape == (4096,)
    assert np.all(values >= lower)
    assert np.all(values <= upper)
    assert abs(values.mean() - expected_mean) < 0.1
    assert abs(values.var(ddof=0) - expected_var) < 0.15


def test_truncated_normal_like_honours_overrides_and_bounds():
    base = Tensor.zeros((128, 64), dtype="float32", requires_grad=True)
    result = Tensor.truncated_normal_like(
        base,
        mean=1.0,
        std=0.5,
        lower=0.0,
        upper=2.0,
        dtype="float64",
        requires_grad=False,
    )
    values = result.numpy()
    expected_mean, _ = _truncated_normal_moments(1.0, 0.5, 0.0, 2.0)

    assert result.shape == base.shape
    assert result.dtype == "float64"
    assert result.requires_grad is False
    assert np.all(values >= 0.0)
    assert np.all(values <= 2.0)
    assert abs(values.mean() - expected_mean) < 0.05


def test_truncated_normal_like_defaults_float_dtype_for_integers():
    base = Tensor.ones((16,), dtype="int32")
    result = Tensor.truncated_normal_like(base, std=0.25)

    assert result.shape == base.shape
    assert result.dtype == mt.get_default_dtype()
    assert np.all(result.numpy() >= -0.5)
    assert np.all(result.numpy() <= 0.5)


def test_truncated_normal_validates_parameters():
    with pytest.raises(ValueError, match="positive finite"):
        Tensor.truncated_normal(4, std=0.0)

    with pytest.raises(ValueError, match="greater than lower"):
        Tensor.truncated_normal(4, lower=1.0, upper=0.5)

    with pytest.raises(ValueError, match="float32 or float64"):
        Tensor.truncated_normal(2, dtype="int32")


def test_xavier_uniform_matches_fan_bounds():
    fan_out, fan_in = 6, 4
    tensor = Tensor.xavier_uniform(fan_out, fan_in, dtype="float32")
    values = tensor.numpy()
    bound = np.sqrt(6.0 / (fan_in + fan_out))

    assert tensor.dtype == "float32"
    assert values.dtype == np.float32
    assert values.shape == (fan_out, fan_in)
    assert np.all(values >= -bound - 1e-6)
    assert np.all(values < bound + 1e-6)


def test_randint_defaults_to_int64_and_respects_bounds():
    x = Tensor.randint(0, 10, 2, 3)
    arr = x.numpy()
    assert x.dtype == "int64"
    assert arr.dtype == np.int64
    assert arr.shape == (2, 3)
    assert (arr >= 0).all()
    assert (arr < 10).all()


def test_randint_supports_int32_dtype():
    x = Tensor.randint(-5, 5, (6,), dtype="int32")
    arr = x.numpy()
    assert x.dtype == "int32"
    assert arr.dtype == np.int32
    assert arr.shape == (6,)
    assert (arr >= -5).all()
    assert (arr < 5).all()


def test_rand_like_matches_reference_metadata():
    base = Tensor.ones((2, 3), dtype="float64", requires_grad=True)
    result = Tensor.rand_like(base)

    assert result.shape == base.shape
    assert result.dtype == "float64"
    assert result.requires_grad is True


def test_rand_like_defaults_to_float_for_integer_inputs():
    base = Tensor.ones((2, 2), dtype="int32")
    result = Tensor.rand_like(base)

    assert result.shape == base.shape
    assert result.dtype == mt.get_default_dtype()


def test_uniform_like_inherits_reference_metadata():
    base = Tensor.ones((3, 2), dtype="int32", requires_grad=True)
    result = Tensor.uniform_like(base, low=-3.0, high=5.0)
    arr = result.numpy()

    assert result.shape == base.shape
    assert result.dtype == "int32"
    assert result.requires_grad is True
    assert (arr >= -3).all()
    assert (arr < 5).all()


def test_he_normal_like_matches_reference_metadata_and_variance():
    base = Tensor.zeros((256, 128), dtype="float32", requires_grad=True)
    result = Tensor.he_normal_like(base)
    values = result.numpy()
    expected_std = np.sqrt(2.0 / 128.0)

    assert result.shape == base.shape
    assert result.dtype == "float32"
    assert result.requires_grad is True
    assert np.isfinite(values).all()
    assert abs(values.mean()) < expected_std * 0.2
    np.testing.assert_allclose(values.std(ddof=0), expected_std, rtol=0.2)


def test_randn_like_can_override_dtype():
    base = Tensor.ones((4,), dtype="float32")
    result = Tensor.randn_like(base, dtype="float64")

    assert result.shape == base.shape
    assert result.dtype == "float64"


def test_randint_like_honours_reference_shape_and_bounds():
    base = Tensor.ones((5, 2), dtype="float32")
    result = Tensor.randint_like(base, 3, 8)
    values = result.numpy()

    assert result.shape == base.shape
    assert result.dtype == "int64"
    assert (values >= 3).all()
    assert (values < 8).all()


def test_randint_like_respects_integer_reference_dtype():
    base = Tensor.ones((6,), dtype="int32")
    result = Tensor.randint_like(base, -2, 2)

    assert result.dtype == "int32"


def test_uniform_raises_for_invalid_bounds():
    with pytest.raises(ValueError, match="high to be greater"):
        Tensor.uniform(2, low=1.0, high=0.0)


def test_fan_initializers_reject_non_float_dtypes():
    with pytest.raises(ValueError, match="float32 or float64"):
        Tensor.xavier_uniform(3, 3, dtype="int32")

    integer_base = Tensor.ones((3, 3), dtype="int32")
    with pytest.raises(ValueError, match="float32 or float64"):
        Tensor.he_uniform_like(integer_base)


def test_fan_initializers_require_positive_dimensions():
    with pytest.raises(ValueError, match="at least 1"):
        Tensor.lecun_normal(0, 4)


def test_lecun_uniform_like_allows_dtype_override():
    base = Tensor.zeros((12, 6), dtype="float32")
    result = Tensor.lecun_uniform_like(base, dtype="float64", requires_grad=True)
    values = result.numpy()
    bound = np.sqrt(3.0 / 6.0)

    assert result.shape == base.shape
    assert result.dtype == "float64"
    assert result.requires_grad is True
    assert values.dtype == np.float64
    assert np.all(values >= -bound - 1e-6)
    assert np.all(values < bound + 1e-6)


def test_randperm_returns_a_permutation():
    perm = Tensor.randperm(8)
    arr = perm.numpy()
    assert perm.dtype == "int64"
    assert arr.dtype == np.int64
    assert sorted(arr.tolist()) == list(range(8))


def test_randperm_allows_int32_dtype():
    perm = Tensor.randperm(5, dtype="int32")
    arr = perm.numpy()
    assert perm.dtype == "int32"
    assert arr.dtype == np.int32
    assert sorted(arr.tolist()) == list(range(5))


def test_empty_like_shape_and_dtype():
    a = Tensor.ones([2, 3], dtype="float32")
    b = empty_like(a)
    assert b.shape == a.shape
    assert b.dtype == a.dtype


def test_tensor_empty_like_matches_reference_metadata():
    base = Tensor.ones((1, 4), dtype="float64", requires_grad=True)
    result = Tensor.empty_like(base)

    assert result.shape == base.shape
    assert result.dtype == "float64"
    assert result.requires_grad is True


def test_tensor_zeros_like_preserves_shape_and_grad():
    base = Tensor.rand((3, 2), requires_grad=True)
    result = Tensor.zeros_like(base)

    assert result.shape == base.shape
    assert result.requires_grad is True
    assert np.all(result.numpy() == 0)


def test_tensor_ones_like_allows_dtype_override():
    base = Tensor.zeros((5,), dtype="float32")
    result = Tensor.ones_like(base, dtype="float64", requires_grad=True)

    assert result.shape == base.shape
    assert result.dtype == "float64"
    assert result.requires_grad is True
    np.testing.assert_allclose(result.numpy(), np.ones(5, dtype=np.float64))


def test_tensor_full_like_respects_fill_value():
    base = Tensor.ones((2, 2), dtype="float32")
    result = Tensor.full_like(base, 7.5, dtype="float64")

    assert result.shape == base.shape
    assert result.dtype == "float64"
    np.testing.assert_allclose(result.numpy(), np.full((2, 2), 7.5, dtype=np.float64))


def test_tensor_new_empty_inherits_reference_metadata():
    base = Tensor.ones((2, 3), dtype="float64", requires_grad=True)
    result = base.new_empty((4,))

    assert result.shape == (4,)
    assert result.dtype == "float64"
    assert result.requires_grad is True


def test_tensor_new_zeros_allows_overrides():
    base = Tensor.rand((2,), dtype="float32")
    result = base.new_zeros([3, 1], dtype="float64", requires_grad=True)

    assert result.shape == (3, 1)
    assert result.dtype == "float64"
    assert result.requires_grad is True
    np.testing.assert_array_equal(result.numpy(), np.zeros((3, 1), dtype=np.float64))


def test_tensor_new_ones_defaults_to_reference_metadata():
    base = Tensor.zeros((2,), dtype="float32", requires_grad=True)
    result = base.new_ones((2,))

    assert result.shape == (2,)
    assert result.dtype == "float32"
    assert result.requires_grad is True
    np.testing.assert_array_equal(result.numpy(), np.ones((2,), dtype=np.float32))


def test_tensor_new_full_respects_fill_and_defaults():
    base = Tensor.rand((1,), dtype="int32", requires_grad=True)
    result = base.new_full(5, 3)

    assert result.shape == (5,)
    assert result.dtype == "int32"
    assert result.requires_grad is True
    np.testing.assert_array_equal(result.numpy(), np.full((5,), 3, dtype=np.int32))


def test_tensor_new_tensor_defaults_to_reference_metadata():
    base = Tensor.ones((2, 2), dtype="float64", requires_grad=True)
    result = base.new_tensor([[1.5, 2.5], [3.5, 4.5]])

    assert result.dtype == "float64"
    assert result.device == base.device
    assert result.requires_grad is True
    np.testing.assert_allclose(
        result.numpy(), np.array([[1.5, 2.5], [3.5, 4.5]], dtype=np.float64)
    )


def test_tensor_new_tensor_overrides_dtype_and_requires_grad():
    base = Tensor.zeros((2,), dtype="float32", requires_grad=False)
    source = Tensor.arange(0, 4).reshape((2, 2))
    result = base.new_tensor(source, dtype="float64", requires_grad=True)

    assert result.dtype == "float64"
    assert result.requires_grad is True
    np.testing.assert_allclose(result.numpy(), source.numpy().astype(np.float64))


def test_tensor_new_tensor_accepts_tensor_wrappers():
    class Wrapper:
        def __init__(self, tensor: Tensor) -> None:
            self._tensor = tensor

    base = Tensor.ones((1,), dtype="float32", requires_grad=True)
    wrapped = Wrapper(Tensor.full((1,), 7, dtype="int32"))
    result = base.new_tensor(wrapped, requires_grad=False)

    assert result.dtype == "float32"
    assert result.requires_grad is False
    np.testing.assert_array_equal(result.numpy(), np.array([7.0], dtype=np.float32))


def test_tensor_as_tensor_defaults_to_reference_tensor_metadata():
    base = Tensor.arange(0, 4, dtype="float32", requires_grad=True).reshape((2, 2))
    alias = Tensor.as_tensor(base)

    assert alias.dtype == "float32"
    assert alias.device == base.device
    assert alias.requires_grad is True
    np.testing.assert_allclose(alias.numpy(), base.numpy())

    mt.clear_autograd_graph()
    alias.sum().backward()

    assert base.grad is not None
    np.testing.assert_allclose(base.grad.numpy(), np.ones((2, 2), dtype=np.float32))


def test_tensor_as_tensor_respects_copy_flag_for_tensors():
    base = Tensor.arange(0, 4, dtype="float32", requires_grad=True).reshape((2, 2))
    expected = np.arange(0, 4, dtype=np.float32).reshape((2, 2))

    clone = Tensor.as_tensor(base, copy=True)

    mt.clear_autograd_graph()
    clone.sum().backward()

    assert base.grad is None
    assert clone.grad is not None
    np.testing.assert_allclose(clone.numpy(), expected)


def test_tensor_as_tensor_infers_dtype_from_python_data():
    data = [[1, 2, 3], [4, 5, 6]]
    tensor = Tensor.as_tensor(data)

    assert tensor.dtype == "int64"
    np.testing.assert_array_equal(tensor.numpy(), np.array(data, dtype=np.int64))


def test_tensor_as_tensor_allows_overrides():
    array = np.arange(6, dtype=np.int32).reshape((2, 3))
    tensor = Tensor.as_tensor(array, dtype="float64", requires_grad=True)

    assert tensor.dtype == "float64"
    assert tensor.requires_grad is True
    np.testing.assert_allclose(tensor.numpy(), array.astype(np.float64))


def test_mixed_type_creation_error():
    with pytest.raises(TypeError):
        Tensor([1, "a"])


def test_linspace_matches_numpy_float32():
    tensor = Tensor.linspace(0.0, 1.0, 5)
    expected = np.linspace(0.0, 1.0, 5, dtype=np.float32)
    assert tensor.shape == (5,)
    np.testing.assert_allclose(tensor.numpy(), expected)
    assert tensor.numpy().dtype == np.float32


def test_linspace_single_step_returns_start():
    tensor = Tensor.linspace(-3.5, 7.5, 1)
    assert tensor.shape == (1,)
    np.testing.assert_allclose(tensor.numpy(), np.array([-3.5], dtype=np.float32))


def test_logspace_matches_numpy_float64():
    tensor = Tensor.logspace(0.0, 3.0, 4, base=10.0, dtype="float64")
    expected = np.logspace(0.0, 3.0, 4, base=10.0, dtype=np.float64)
    np.testing.assert_allclose(tensor.numpy(), expected)
    assert tensor.numpy().dtype == np.float64
