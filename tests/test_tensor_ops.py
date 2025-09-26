# Copyright (c) 2025 Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import pytest

import minitensor as mt


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


def test_contiguous_materializes_expand():
    base = mt.arange(0.0, 3.0, dtype="float32").reshape(3, 1)
    expanded = base.expand(3, 4)
    assert not expanded.is_contiguous()

    materialized = expanded.contiguous()
    assert materialized.is_contiguous()
    expected = np.broadcast_to(
        np.arange(0.0, 3.0, dtype=np.float32).reshape(3, 1),
        (3, 4),
    )
    np.testing.assert_allclose(materialized.numpy(), expected)


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
    assert np.isinf(m.numpy())


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
