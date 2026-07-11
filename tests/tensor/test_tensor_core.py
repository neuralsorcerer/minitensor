# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

import decimal
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


class IndexLike:
    def __init__(self, value: int):
        self._value = value

    def __index__(self) -> int:
        return self._value


def test_reshape_and_transpose_roundtrip():
    t = mt.Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    reshaped = t.reshape([3, 2])
    np.testing.assert_allclose(
        reshaped.numpy(), np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    )
    transposed = reshaped.transpose(0, 1)
    np.testing.assert_allclose(
        transposed.numpy(), np.array([[1.0, 3.0, 5.0], [2.0, 4.0, 6.0]])
    )


def _numpy_strides_in_elements(array: np.ndarray) -> tuple[int, ...]:
    return tuple(s // array.itemsize for s in array.strides)


def test_strides_and_contiguity_follow_backend_layout():
    base = mt.arange(6).reshape(2, 3)
    base_np = base.numpy()
    assert base.is_contiguous()
    assert base.strides == (3, 1)
    assert base.strides == _numpy_strides_in_elements(base_np)

    transposed = base.transpose(0, 1)
    transposed_np = transposed.numpy()
    assert transposed.shape == (3, 2)
    assert transposed.strides == _numpy_strides_in_elements(transposed_np)
    assert isinstance(transposed.is_contiguous(), bool)

    materialized = transposed.contiguous()
    materialized_np = materialized.numpy()
    assert materialized.is_contiguous()
    assert materialized.strides == _numpy_strides_in_elements(materialized_np)
    np.testing.assert_allclose(materialized_np, transposed_np)


def test_expand_materializes_at_python_boundary():
    # The engine's kernels assume contiguous storage, so tensors handed back
    # to Python are materialized. expand() therefore returns a contiguous
    # tensor whose values match NumPy broadcasting, and every downstream
    # operation on it is safe.
    base = mt.arange(0.0, 3.0, dtype="float32").reshape(3, 1)
    expanded = base.expand(3, 4)
    assert expanded.is_contiguous()

    expected = np.broadcast_to(
        np.arange(0.0, 3.0, dtype=np.float32).reshape(3, 1),
        (3, 4),
    )
    np.testing.assert_allclose(expanded.numpy(), expected)

    materialized = expanded.contiguous()
    assert materialized.is_contiguous()
    np.testing.assert_allclose(materialized.numpy(), expected)


def test_expanded_tensor_computes_correctly():
    # Regression test: operations on expanded tensors used to read only the
    # small base buffer and silently return garbage (or panic).
    base = mt.Tensor([[1.0], [2.0], [3.0]])
    expanded = base.expand(3, 4)
    expected = np.broadcast_to(
        np.array([[1.0], [2.0], [3.0]], dtype=np.float32), (3, 4)
    )

    np.testing.assert_allclose(expanded.sum().numpy(), expected.sum())
    np.testing.assert_allclose(expanded.exp().numpy(), np.exp(expected), rtol=1e-6)

    ones = mt.Tensor(np.ones((3, 4), dtype=np.float32))
    np.testing.assert_allclose((expanded + ones).numpy(), expected + 1.0)
    np.testing.assert_allclose((expanded * expanded).numpy(), expected * expected)
    np.testing.assert_allclose(expanded.mean(1).numpy(), expected.mean(axis=1))
    np.testing.assert_allclose(
        expanded.matmul(mt.Tensor(np.ones((4, 2), dtype=np.float32))).numpy(),
        expected @ np.ones((4, 2), dtype=np.float32),
    )


def test_reshape_invalid_size():
    t = mt.Tensor([1.0, 2.0, 3.0, 4.0])
    with pytest.raises(ValueError):
        t.reshape([3, 2])


def test_reshape_infer_dim():
    t = mt.arange(6)
    r = t.reshape(2, -1)
    np.testing.assert_allclose(r.numpy(), np.arange(6, dtype=np.float32).reshape(2, 3))


def test_reshape_multiple_negative_one_error():
    t = mt.arange(6)
    with pytest.raises(ValueError):
        t.reshape(-1, -1)


def test_reshape_infer_mismatch_error():
    t = mt.arange(5)
    with pytest.raises(ValueError):
        t.reshape(4, -1)


def test_reshape_zero_dim_with_inference_error():
    t = mt.arange(0)
    with pytest.raises(ValueError):
        t.reshape(-1, 0)


def test_reshape_backward_preserves_gradients():
    base = mt.arange(0.0, 6.0, dtype="float32", requires_grad=True)
    reshaped = base.reshape((2, 3))
    reshaped.sum().backward()

    grad = base.grad
    assert grad is not None
    np.testing.assert_allclose(grad.numpy(), np.ones(6, dtype=np.float32))


def test_transpose_invalid_dim():
    t = mt.Tensor([[1.0, 2.0], [3.0, 4.0]])
    with pytest.raises(IndexError):
        t.transpose(0, 2)


def test_transpose_negative_dim():
    t = mt.Tensor([[1.0, 2.0], [3.0, 4.0]])
    tr = t.transpose(0, -1)
    np.testing.assert_allclose(tr.numpy(), np.array([[1.0, 3.0], [2.0, 4.0]]))


def test_transpose_negative_dim_out_of_range():
    t = mt.Tensor([[1.0, 2.0], [3.0, 4.0]])
    with pytest.raises(IndexError):
        t.transpose(0, -3)


def test_sum_keepdim():
    t = mt.Tensor([[1.0, 2.0], [3.0, 4.0]])
    s = t.sum(dim=[1], keepdim=True)
    np.testing.assert_allclose(s.numpy(), np.array([[3.0], [7.0]]))
    s_no = t.sum(dim=[1], keepdim=False)
    np.testing.assert_allclose(s_no.numpy(), np.array([3.0, 7.0]))


def test_prod_keepdim():
    t = mt.Tensor([[1.0, 2.0], [3.0, 4.0]])
    p = t.prod(dim=[1], keepdim=True)
    np.testing.assert_allclose(p.numpy(), np.array([[2.0], [12.0]]))
    p_no = t.prod(dim=[1], keepdim=False)
    np.testing.assert_allclose(p_no.numpy(), np.array([2.0, 12.0]))


def test_backward_non_scalar_error():
    t = mt.Tensor([1.0, 2.0], requires_grad=True)
    with pytest.raises(RuntimeError):
        t.backward()


def test_backward_requires_grad_flag():
    t = mt.Tensor([1.0])
    with pytest.raises(
        RuntimeError, match="does not require grad and does not have a grad_fn"
    ):
        t.backward()


def test_detach_inplace_clears_grad_and_stops_tracking():
    base = mt.Tensor([1.0, 2.0, 3.0], requires_grad=True)
    out = (base * base).sum()
    out.backward()
    assert base.grad is not None

    base.detach_()
    assert base.requires_grad is False
    assert base.grad is None

    new_out = (base * 3).sum()
    assert new_out.requires_grad is False
    with pytest.raises(
        RuntimeError, match="does not require grad and does not have a grad_fn"
    ):
        new_out.backward()


def test_detach_inplace_updates_future_views():
    base = mt.Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)

    base.detach_()

    assert base.requires_grad is False

    new_view = base[0]
    assert new_view.requires_grad is False

    result = (new_view * 2).sum()
    assert result.requires_grad is False


def test_backward_requires_retain_graph_for_second_call():
    x = mt.Tensor([2.0], requires_grad=True)
    y = (x * x).sum()

    y.backward()

    with pytest.raises(RuntimeError) as exc:
        y.backward()
    assert "retain_graph" in str(exc.value).lower()

    mt.clear_autograd_graph()


def test_backward_with_retain_graph_allows_multiple_calls():
    x = mt.Tensor([3.0], requires_grad=True)
    y = (x * x).sum()

    y.backward(retain_graph=True)
    first_grad = x.grad.numpy().copy()
    x.zero_grad()

    y.backward(retain_graph=True)
    np.testing.assert_allclose(x.grad.numpy(), first_grad)

    mt.clear_autograd_graph()


def test_backward_create_graph_not_supported():
    x = mt.Tensor([1.0], requires_grad=True)
    y = (x * x).sum()

    with pytest.raises(NotImplementedError):
        y.backward(create_graph=True)

    mt.clear_autograd_graph()


def test_sum_negative_dim():
    t = mt.Tensor([[1.0, 2.0], [3.0, 4.0]])
    r = t.sum(dim=[-1])
    np.testing.assert_allclose(r.numpy(), np.array([3.0, 7.0]))


def test_sum_multiple_dims():
    t = mt.Tensor([[1.0, 2.0], [3.0, 4.0]])
    r = t.sum(dim=[0, 1])
    np.testing.assert_allclose(r.numpy(), np.array(10.0, dtype=np.float32))
    r_keep = t.sum(dim=[0, 1], keepdim=True)
    np.testing.assert_allclose(r_keep.numpy(), np.array([[10.0]], dtype=np.float32))


def test_sum_invalid_dim():
    t = mt.Tensor([[1.0, 2.0], [3.0, 4.0]])
    with pytest.raises(IndexError):
        t.sum(dim=[2])
    with pytest.raises(IndexError):
        t.sum(dim=[-3])


def test_sum_rejects_boolean_dims():
    t = mt.Tensor([[1.0, 2.0], [3.0, 4.0]])
    with pytest.raises(TypeError):
        t.sum(dim=True)
    with pytest.raises(TypeError):
        t.sum(dim=[0, True])
    with pytest.raises(TypeError):
        t.sum(dim=np.bool_(True))
    with pytest.raises(TypeError):
        t.sum(dim=[np.bool_(False)])


def test_prod_all():
    t = mt.Tensor([1.0, 2.0, 3.0])
    p = t.prod()
    np.testing.assert_allclose(p.numpy(), np.array(6.0, dtype=np.float32))


def test_bool_prod():
    t = mt.Tensor([[True, True], [True, False]], dtype="bool")
    p_all = t.prod()
    np.testing.assert_array_equal(p_all.numpy(), np.array(False))
    p_dim = t.prod(dim=[1], keepdim=False)
    np.testing.assert_array_equal(p_dim.numpy(), np.array([True, False]))


def test_prod_negative_dim():
    t = mt.Tensor([[1.0, 2.0], [3.0, 4.0]])
    r = t.prod(dim=[-1])
    np.testing.assert_allclose(r.numpy(), np.array([2.0, 12.0]))


def test_mean_negative_dim():
    t = mt.Tensor([[1.0, 2.0], [3.0, 4.0]])
    r = t.mean(dim=[-1])
    np.testing.assert_allclose(r.numpy(), np.array([1.5, 3.5]))


def test_mean_int_tensor():
    t = mt.Tensor([1, 2, 3], dtype="int32")
    r = t.mean()
    assert r.dtype == "float32"
    np.testing.assert_allclose(r.numpy(), np.array(2.0, dtype=np.float32))


def test_any_all_keepdim():
    t = mt.Tensor([[1.0, 0.0], [0.0, 2.0]])
    any_res = t.any(dim=1, keepdim=True)
    all_res = t.all(dim=0)
    np.testing.assert_array_equal(any_res.numpy(), np.array([[True], [True]]))
    np.testing.assert_array_equal(all_res.numpy(), np.array([False, False]))


def test_any_all_negative_dim():
    t = mt.Tensor([[1.0, 0.0], [0.0, 2.0]])
    any_res = t.any(dim=-1)
    all_res = t.all(dim=-2)
    np.testing.assert_array_equal(any_res.numpy(), np.array([True, True]))
    np.testing.assert_array_equal(all_res.numpy(), np.array([False, False]))


def test_max_negative_dim():
    t = mt.arange(6).reshape(2, 3)
    vals, idx = t.max(dim=-1)
    np.testing.assert_allclose(vals.numpy(), np.max(t.numpy(), axis=-1))
    np.testing.assert_array_equal(idx.numpy(), np.argmax(t.numpy(), axis=-1))


def test_empty_tensor_reductions():
    t = mt.Tensor(np.array([], dtype=np.float32))
    s = t.sum()
    m = t.mean()
    np.testing.assert_allclose(s.numpy(), np.array([0.0], dtype=np.float32))
    # mean of an empty tensor is 0/0 = NaN, matching NumPy and PyTorch.
    assert np.isnan(m.numpy())


def test_cumsum_and_backward():
    t = mt.Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    c0 = t.cumsum(0)
    c1 = t.cumsum(1)
    np.testing.assert_allclose(c0.numpy(), np.cumsum(t.numpy(), axis=0))
    np.testing.assert_allclose(c1.numpy(), np.cumsum(t.numpy(), axis=1))
    c0.sum().backward()
    np.testing.assert_allclose(
        t.grad.numpy(), np.array([[2.0, 2.0], [1.0, 1.0]], dtype=np.float32)
    )


def test_masked_fill_with_scalar():
    base = mt.arange(0.0, 4.0, dtype="float32").reshape(2, 2)
    mask = mt.Tensor([[True, False], [False, True]], dtype="bool")

    filled = base.masked_fill(mask, 10.0)
    expected = np.array([[10.0, 1.0], [2.0, 10.0]], dtype=np.float32)
    np.testing.assert_allclose(filled.numpy(), expected)


def test_masked_fill_float_scalar_promotes_int_tensor():
    base = mt.arange(0, 4, dtype="int32").reshape(2, 2)
    mask = mt.Tensor([[True, False], [False, True]], dtype="bool")

    filled = base.masked_fill(mask, 0.5)
    expected = np.array([[0.5, 1.0], [2.0, 0.5]], dtype=np.float32)

    assert filled.dtype == "float32"
    np.testing.assert_allclose(filled.numpy(), expected)


def test_masked_fill_decimal_scalar_promotes_int_tensor():
    base = mt.arange(0, 4, dtype="int32").reshape(2, 2)
    mask = mt.Tensor([[True, False], [False, True]], dtype="bool")

    filled = base.masked_fill(mask, decimal.Decimal("0.5"))
    expected = np.array([[0.5, 1.0], [2.0, 0.5]], dtype=np.float32)

    assert filled.dtype == "float32"
    np.testing.assert_allclose(filled.numpy(), expected)


def test_masked_fill_index_like_scalar_retains_integer_dtype():
    base = mt.arange(0, 4, dtype="int32").reshape(2, 2)
    mask = mt.Tensor([[True, False], [False, True]], dtype="bool")

    filled = base.masked_fill(mask, IndexLike(7))

    assert filled.dtype == "int32"
    np.testing.assert_array_equal(
        filled.numpy(), np.array([[7, 1], [2, 7]], dtype=np.int32)
    )


def test_masked_fill_tensor_gradient():
    base = mt.arange(0.0, 4.0, dtype="float32", requires_grad=True).reshape(2, 2)
    mask = mt.Tensor([[True, False], [False, True]], dtype="bool")
    values = mt.full((1, 2), 5.0, dtype="float32", requires_grad=True)

    output = base.masked_fill(mask, values)
    loss = output.sum()
    loss.backward()

    base_grad = base.grad
    assert base_grad is not None
    np.testing.assert_allclose(
        base_grad.numpy(),
        np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32),
    )

    values_grad = values.grad
    assert values_grad is not None
    np.testing.assert_allclose(
        values_grad.numpy(), np.array([[1.0, 1.0]], dtype=np.float32)
    )


def test_where_float_scalar_promotes_int_tensor():
    condition = mt.Tensor([[True, False], [False, True]], dtype="bool")
    input_tensor = mt.arange(0, 4, dtype="int32").reshape(2, 2)

    result = mt.where(condition, input_tensor, 0.5)
    expected = np.array([[0.0, 0.5], [0.5, 3.0]], dtype=np.float32)

    assert result.dtype == "float32"
    np.testing.assert_allclose(result.numpy(), expected)


def test_where_index_like_scalar_retains_integer_dtype():
    condition = mt.Tensor([[True, False], [False, True]], dtype="bool")
    input_tensor = mt.arange(0, 4, dtype="int32").reshape(2, 2)

    result = input_tensor.where(condition, IndexLike(9))

    assert result.dtype == "int32"
    np.testing.assert_array_equal(
        result.numpy(), np.array([[0, 9], [9, 3]], dtype=np.int32)
    )


def test_where_decimal_scalar_promotes_int_tensor():
    condition = mt.Tensor([[True, False], [False, True]], dtype="bool")
    input_tensor = mt.arange(0, 4, dtype="int32").reshape(2, 2)

    result = mt.where(condition, decimal.Decimal("0.5"), input_tensor)
    expected = np.array([[0.5, 1.0], [2.0, 0.5]], dtype=np.float32)

    assert result.dtype == "float32"
    np.testing.assert_allclose(result.numpy(), expected)


def test_tensor_creation_int_list_preserves_large_values():
    values = [2**40 + 3, -(2**35) - 7]
    tensor = mt.Tensor(values, dtype="int64")
    np.testing.assert_array_equal(tensor.numpy(), np.array(values, dtype=np.int64))


def test_tensor_creation_decimal_nested_sequence_uses_float64():
    data = [[decimal.Decimal("0.1"), decimal.Decimal("0.2")]]
    tensor = mt.Tensor(data, dtype="float64")
    expected = np.array([[0.1, 0.2]], dtype=np.float64)
    np.testing.assert_allclose(tensor.numpy(), expected)


def test_tensor_creation_index_like_nested_sequence_uses_integer_dtype():
    data = [[IndexLike(1), IndexLike(2)], [IndexLike(3), IndexLike(4)]]
    tensor = mt.Tensor(data, dtype="int64")

    assert tensor.dtype == "int64"
    np.testing.assert_array_equal(
        tensor.numpy(), np.array([[1, 2], [3, 4]], dtype=np.int64)
    )


def test_tensor_creation_inconsistent_nested_lengths_errors():
    with pytest.raises(ValueError, match="Inconsistent nested sequence lengths"):
        mt.Tensor([[1, 2], [3]])


def test_cumprod_and_backward():
    t = mt.Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    p0 = t.cumprod(0)
    p1 = t.cumprod(1)
    np.testing.assert_allclose(p0.numpy(), np.cumprod(t.numpy(), axis=0))
    np.testing.assert_allclose(p1.numpy(), np.cumprod(t.numpy(), axis=1))
    p0.sum().backward()
    np.testing.assert_allclose(
        t.grad.numpy(), np.array([[4.0, 5.0], [1.0, 2.0]], dtype=np.float32)
    )


def test_cumprod_backward_with_zero():
    t = mt.Tensor([1.0, 0.0, 2.0], requires_grad=True)
    t.cumprod(0).sum().backward()
    np.testing.assert_allclose(
        t.grad.numpy(), np.array([1.0, 3.0, 0.0], dtype=np.float32)
    )


def test_sum_with_nan_propagation():
    t = mt.Tensor([1.0, np.nan])
    s = t.sum()
    assert np.isnan(s.numpy()).all()


def test_log_negative_returns_neg_infinity():
    t = mt.Tensor([-1.0])
    r = t.log()
    assert np.isneginf(r.numpy()).all()


def test_mul_overflow_results_infinity():
    a = mt.Tensor([3.4e38])
    b = mt.Tensor([10.0])
    res = a * b
    assert np.isinf(res.numpy()).all()


def test_exp_extreme_values():
    t = mt.Tensor([1000.0, -1000.0])
    r = t.exp()
    result = r.numpy()
    assert np.isinf(result[0])
    assert result[1] == 0.0


def test_tensor_copy_inplace_casts_dtype_and_preserves_requires_grad():
    target = mt.zeros((2, 2), dtype="float32", requires_grad=True)
    source = mt.Tensor([[1.5, -2.0], [3.25, 4.75]], dtype="float64")

    returned = target.copy_(source)

    assert returned is target
    assert target.dtype == "float32"
    assert target.requires_grad is True
    np.testing.assert_allclose(
        target.numpy(),
        np.array([[1.5, -2.0], [3.25, 4.75]], dtype=np.float32),
    )


def test_tensor_copy_raises_for_shape_mismatch():
    target = mt.zeros((2, 2), dtype="float32")
    source = mt.ones((3,), dtype="float32")

    with pytest.raises(ValueError, match="copy_ expected source"):
        target.copy_(source)


def test_tensor_fill_inplace_casts_scalar_to_tensor_dtype():
    tensor = mt.ones((2, 3), dtype="int32", requires_grad=True)

    returned = tensor.fill_(7.8)

    assert returned is tensor
    assert tensor.dtype == "int32"
    assert tensor.requires_grad is True
    np.testing.assert_array_equal(tensor.numpy(), np.full((2, 3), 7, dtype=np.int32))


def test_tensor_fill_accepts_boolean_input():
    tensor = mt.zeros((4,), dtype="bool")

    tensor.fill_(True)

    np.testing.assert_array_equal(tensor.numpy(), np.ones((4,), dtype=bool))


def test_set_default_dtype():
    mt.set_default_dtype("float64")
    try:
        t = mt.Tensor([1.0, 2.0])
        assert t.dtype == "float64"
    finally:
        mt.set_default_dtype("float32")


def test_large_ones_initialization():
    t = mt.Tensor.ones([5000])
    arr = t.numpy()
    assert arr.shape == (5000,)
    assert np.all(arr == 1.0)


def test_allclose_parallel():
    a = mt.Tensor.ones([5000])
    b = a + mt.Tensor.ones([5000]) * 1e-6
    assert a.allclose(b, rtol=1e-5, atol=1e-5)
    c = a + mt.Tensor.ones([5000]) * 1e-2
    assert not a.allclose(c, rtol=1e-5, atol=1e-5)


def test_array_equal_parallel():
    a = mt.Tensor.ones([5000])
    b = a.clone()
    assert a.array_equal(b)
    c = mt.Tensor.zeros([5000])
    assert not a.array_equal(c)


def test_clone_preserves_autograd_history():
    dev = mt.device("cpu")
    base = mt.Tensor(
        np.array([1.0, -2.0, 3.0], dtype=np.float32), requires_grad=True, device=dev
    )
    base.zero_grad(set_to_none=True)

    cloned = base.clone()
    assert cloned.requires_grad

    scale = mt.Tensor(np.array([0.5, 0.25, -1.5], dtype=np.float32), device=dev)
    out = (cloned * scale).sum()
    out.backward()

    np.testing.assert_allclose(
        base.grad.numpy(),
        np.array([0.5, 0.25, -1.5], dtype=np.float32),
    )


def test_empty_ones_tensor():
    t = mt.Tensor.ones([0])
    assert t.numpy().size == 0


def test_concatenate_negative_axis():
    a = mt.arange(6).reshape(2, 3)
    b = mt.arange(6, 12).reshape(2, 3)
    core_res = mt.numpy_compat.concatenate([a._tensor, b._tensor], axis=-1)
    res = mt.Tensor.__new__(mt.Tensor)
    res._tensor = core_res
    np.testing.assert_allclose(
        res.numpy(), np.concatenate([a.numpy(), b.numpy()], axis=-1)
    )


def test_stack_negative_axis():
    a = mt.arange(6).reshape(2, 3)
    b = mt.arange(6, 12).reshape(2, 3)
    core_res = mt.numpy_compat.stack([a._tensor, b._tensor], axis=-1)
    res = mt.Tensor.__new__(mt.Tensor)
    res._tensor = core_res
    np.testing.assert_allclose(res.numpy(), np.stack([a.numpy(), b.numpy()], axis=-1))


def test_split_negative_axis():
    t = mt.arange(12).reshape(3, 4)
    core_parts = mt.numpy_compat.split(t._tensor, 2, axis=-1)
    parts = []
    for cp in core_parts:
        wrapper = mt.Tensor.__new__(mt.Tensor)
        wrapper._tensor = cp
        parts.append(wrapper)
    np_parts = np.split(t.numpy(), 2, axis=-1)
    for p, n in zip(parts, np_parts):
        np.testing.assert_allclose(p.numpy(), n)


def test_functional_basic_reductions_match_tensor_methods_and_numpy():
    x_np = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    x = mt.Tensor(x_np.tolist())

    np.testing.assert_allclose(mt.functional.sum(x).numpy(), x.sum().numpy())
    np.testing.assert_allclose(mt.sum(x, dim=1).numpy(), x_np.sum(axis=1))
    np.testing.assert_allclose(mt.functional.prod(x, dim=0).numpy(), x_np.prod(axis=0))
    np.testing.assert_allclose(
        mt.mean(x, dim=1, keepdim=True).numpy(), x_np.mean(axis=1, keepdims=True)
    )


def test_functional_boolean_reductions_are_exported_and_correct():
    mask = mt.Tensor([[True, True, False], [True, True, True]], dtype="bool")

    assert mt.functional.all(mask, dim=1).numpy().tolist() == [False, True]
    assert mt.any(mask, dim=0).numpy().tolist() == [True, True, True]


def test_to_float64():
    x = Tensor([1.5, -2.3], dtype="float32")
    y = x.to("float64")
    assert y.dtype == "float64"
    assert np.allclose(y.numpy(), np.array([1.5, -2.3], dtype=np.float64))


def test_to_accepts_device_string():
    x = Tensor([1.0, 2.0, 3.0], dtype="float32")
    y = x.to("cpu")
    assert y.device == "cpu"
    np.testing.assert_allclose(y.numpy(), np.array([1.0, 2.0, 3.0], dtype=np.float32))


def test_to_accepts_device_object_and_dtype_keyword():
    x = Tensor([1.5, -2.3], dtype="float32")
    cpu = mt.device("cpu")
    y = x.to(cpu, dtype="float64")
    assert y.device == "cpu"
    assert y.dtype == "float64"
    np.testing.assert_allclose(y.numpy(), np.array([1.5, -2.3], dtype=np.float64))


def test_astype_int():
    x = Tensor([1.5, -2.3], dtype="float32")
    y = x.astype("int32")
    assert y.dtype == "int32"
    assert np.array_equal(y.numpy(), np.array([1, -2], dtype=np.int32))


def test_astype_nan_and_overflow():
    x = Tensor([float("nan"), 1e40, -1e40], dtype="float32")
    y = x.astype("int32")
    assert np.array_equal(
        y.numpy(),
        np.array([0, np.iinfo(np.int32).max, np.iinfo(np.int32).min], dtype=np.int32),
    )


def test_astype_bool_from_float():
    x = Tensor([-0.1, 0.0, 2.0], dtype="float32")
    y = x.astype("bool")
    assert y.dtype == "bool"
    assert np.array_equal(y.numpy(), np.array([True, False, True], dtype=bool))


def test_int64_to_float32():
    x = Tensor([1, -2, 3], dtype="int64")
    y = x.astype("float32")
    assert y.dtype == "float32"
    assert np.array_equal(y.numpy(), np.array([1.0, -2.0, 3.0], dtype=np.float32))


def test_bool_to_int64_and_float():
    x = Tensor([True, False, True], dtype="bool")
    y = x.astype("int64")
    z = x.astype("float64")
    assert y.dtype == "int64"
    assert z.dtype == "float64"
    assert np.array_equal(y.numpy(), np.array([1, 0, 1], dtype=np.int64))
    assert np.array_equal(z.numpy(), np.array([1.0, 0.0, 1.0], dtype=np.float64))


def test_empty_tensor_conversion():
    x = Tensor([], dtype="float32")
    y = x.astype("int32")
    assert y.dtype == "int32"
    assert y.numpy().size == 0


@pytest.mark.parametrize("src_dtype", ["float32", "float64", "int32", "int64", "bool"])
@pytest.mark.parametrize(
    "target_dtype", ["float32", "float64", "int32", "int64", "bool"]
)
def test_empty_tensor_all_dtype_conversions(src_dtype, target_dtype):
    if src_dtype == target_dtype:
        pytest.skip("identity conversion")
    x = Tensor([], dtype=src_dtype)
    y = x.astype(target_dtype)
    assert y.dtype == target_dtype
    assert y.numpy().size == 0


def test_large_astype_parallel():
    data = np.arange(2048, dtype=np.float32)
    x = Tensor(data, dtype="float32")
    y = x.astype("int32")
    assert y.dtype == "int32"
    assert np.array_equal(y.numpy(), data.astype(np.int32))


def test_tolist_returns_python_scalar_for_zero_dim():
    tensor = mt.Tensor(3.5)

    value = tensor.tolist()

    assert isinstance(value, float)
    assert value == pytest.approx(3.5)


def test_tolist_preserves_python_bool():
    tensor = mt.Tensor(True, dtype="bool")

    value = tensor.tolist()

    assert isinstance(value, bool)
    assert value is True


def test_item_returns_native_python_scalar():
    tensor = mt.Tensor(7, dtype="int64")

    value = tensor.item()

    assert isinstance(value, int)
    assert value == 7


def test_item_error_matches_pytorch_message():
    tensor = mt.Tensor([1.0, 2.0, 3.0])

    with pytest.raises(RuntimeError) as exc_info:
        tensor.item()

    assert "a Tensor with 3 elements cannot be converted to Scalar" in str(
        exc_info.value
    )


class _EquivalentToDefaultDtype:
    def __eq__(self, other: object) -> bool:
        return other == mt.get_default_dtype()


def test_default_dtype_context_restores():
    original = mt.get_default_dtype()
    with mt.default_dtype("float64"):
        assert mt.get_default_dtype() == "float64"
        t = mt.Tensor([1.0, 2.0])
        assert t.dtype == "float64"
    assert mt.get_default_dtype() == original


def test_default_dtype_context_restores_on_exception():
    original = mt.get_default_dtype()
    with pytest.raises(RuntimeError):
        with mt.default_dtype("float64"):
            raise RuntimeError("boom")
    assert mt.get_default_dtype() == original


def test_default_dtype_context_invalid_dtype():
    original = mt.get_default_dtype()
    with pytest.raises(ValueError):
        with mt.default_dtype("not-a-real-dtype"):
            pass
    assert mt.get_default_dtype() == original


def test_default_dtype_context_same_dtype_avoids_redundant_setter_calls(monkeypatch):
    original = mt.get_default_dtype()
    calls: list[str] = []
    original_set_default_dtype = mt.set_default_dtype

    def _tracking_set_default_dtype(dtype: str) -> None:
        calls.append(dtype)
        original_set_default_dtype(dtype)

    monkeypatch.setattr(mt, "set_default_dtype", _tracking_set_default_dtype)

    with mt.default_dtype(original):
        assert mt.get_default_dtype() == original

    assert calls == []
    assert mt.get_default_dtype() == original


def test_default_dtype_context_non_string_equal_to_default_still_validates_type():
    original = mt.get_default_dtype()

    with pytest.raises(TypeError):
        with mt.default_dtype(_EquivalentToDefaultDtype()):
            pass

    assert mt.get_default_dtype() == original


def test_manual_seed_makes_rand_deterministic():
    mt.manual_seed(123)
    first = mt.rand(2, 3).numpy()
    mt.manual_seed(123)
    second = mt.rand(2, 3).numpy()
    np.testing.assert_array_equal(first, second)


def test_manual_seed_makes_randn_deterministic():
    mt.manual_seed(321)
    first = mt.randn(2, 3).numpy()
    mt.manual_seed(321)
    second = mt.randn(2, 3).numpy()
    np.testing.assert_array_equal(first, second)


def test_manual_seed_makes_rand_like_deterministic():
    base = mt.ones((4, 2), dtype="float32")
    mt.manual_seed(111)
    first = mt.rand_like(base).numpy()
    mt.manual_seed(111)
    second = mt.rand_like(base).numpy()
    np.testing.assert_array_equal(first, second)


def test_manual_seed_makes_randn_like_deterministic():
    base = mt.ones((3, 5), dtype="float64")
    mt.manual_seed(222)
    first = mt.randn_like(base).numpy()
    mt.manual_seed(222)
    second = mt.randn_like(base).numpy()
    np.testing.assert_array_equal(first, second)


def test_manual_seed_affects_module_initialization():
    mt.manual_seed(999)
    layer_a = mt.nn.DenseLayer(4, 3)
    mt.manual_seed(999)
    layer_b = mt.nn.DenseLayer(4, 3)

    params_a = [param.numpy() for param in layer_a.parameters()]
    params_b = [param.numpy() for param in layer_b.parameters()]

    for left, right in zip(params_a, params_b):
        np.testing.assert_array_equal(left, right)


def test_manual_seed_controls_dropout_mask():
    x = mt.ones(10, dtype="float32")

    mt.manual_seed(42)
    dropout_a = mt.nn.Dropout(0.5)
    out_a = dropout_a(x).numpy()

    mt.manual_seed(42)
    dropout_b = mt.nn.Dropout(0.5)
    out_b = dropout_b(x).numpy()

    np.testing.assert_array_equal(out_a, out_b)


def test_manual_seed_makes_randint_deterministic():
    mt.manual_seed(2025)
    first = mt.randint(0, 10, 6).numpy()
    mt.manual_seed(2025)
    second = mt.randint(0, 10, 6).numpy()
    np.testing.assert_array_equal(first, second)


def test_manual_seed_makes_randperm_deterministic():
    mt.manual_seed(77)
    first = mt.randperm(9).numpy()
    mt.manual_seed(77)
    second = mt.randperm(9).numpy()
    np.testing.assert_array_equal(first, second)


def test_manual_seed_makes_randint_like_deterministic():
    base = mt.ones((2, 4), dtype="float32")
    mt.manual_seed(333)
    first = mt.randint_like(base, 0, 5).numpy()
    mt.manual_seed(333)
    second = mt.randint_like(base, 0, 5).numpy()
    np.testing.assert_array_equal(first, second)


def test_atleast_1d_shapes_scalars_vectors_and_multiple_inputs():
    scalar = mt.atleast_1d(3.5)
    vector = mt.Tensor([1, 2, 3])
    first, second = mt.atleast_1d(1, vector)

    assert scalar.shape == (1,)
    np.testing.assert_allclose(scalar.numpy(), np.array([3.5], dtype=np.float32))
    assert first.shape == (1,)
    assert second is vector
    assert second.shape == (3,)


def test_atleast_2d_matches_numpy_shape_conventions():
    scalar = mt.atleast_2d(2)
    vector = mt.Tensor([1.0, 2.0, 3.0])
    matrix = mt.Tensor([[1.0, 2.0], [3.0, 4.0]])
    row, unchanged = mt.atleast_2d(vector, matrix)

    assert scalar.shape == (1, 1)
    assert row.shape == (1, 3)
    assert unchanged is matrix
    np.testing.assert_allclose(row.numpy(), np.atleast_2d(vector.numpy()))


def test_atleast_3d_matches_numpy_shape_conventions_and_errors_without_inputs():
    scalar = mt.atleast_3d(2)
    vector = mt.Tensor([1.0, 2.0, 3.0])
    matrix = mt.Tensor([[1.0, 2.0], [3.0, 4.0]])
    cube = mt.ones(2, 3, 4)
    vector_3d, matrix_3d, unchanged = mt.atleast_3d(vector, matrix, cube)

    assert scalar.shape == (1, 1, 1)
    assert vector_3d.shape == (1, 3, 1)
    assert matrix_3d.shape == (2, 2, 1)
    assert unchanged is cube
    np.testing.assert_allclose(vector_3d.numpy(), np.atleast_3d(vector.numpy()))
    np.testing.assert_allclose(matrix_3d.numpy(), np.atleast_3d(matrix.numpy()))
    with pytest.raises(TypeError, match="requires at least one input"):
        mt.atleast_3d()


def test_atleast_helpers_handle_empty_and_higher_rank_inputs():
    empty_vector = mt.Tensor([])
    empty_matrix = mt.Tensor.zeros(0, 2)
    cube = mt.ones(2, 3, 4)

    assert mt.atleast_1d(empty_vector) is empty_vector
    assert mt.atleast_2d(empty_vector).shape == (1, 0)
    assert mt.atleast_3d(empty_vector).shape == (1, 0, 1)
    assert mt.atleast_3d(empty_matrix).shape == (0, 2, 1)
    assert mt.atleast_1d(cube) is cube
    assert mt.atleast_2d(cube) is cube
    assert mt.atleast_3d(cube) is cube


@pytest.mark.parametrize("helper", [mt.atleast_1d, mt.atleast_2d])
def test_atleast_helpers_reject_missing_inputs(helper):
    with pytest.raises(TypeError, match="requires at least one input"):
        helper()
