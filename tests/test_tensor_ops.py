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


def test_reshape_invalid_size():
    t = mt.Tensor([1.0, 2.0, 3.0, 4.0])
    with pytest.raises(ValueError):
        t.reshape([3, 2])


def test_transpose_invalid_dim():
    t = mt.Tensor([[1.0, 2.0], [3.0, 4.0]])
    with pytest.raises(IndexError):
        t.transpose(0, 2)


def test_sum_keepdim():
    t = mt.Tensor([[1.0, 2.0], [3.0, 4.0]])
    s = t.sum(dim=[1], keepdim=True)
    np.testing.assert_allclose(s.numpy(), np.array([[3.0], [7.0]]))
    s_no = t.sum(dim=[1], keepdim=False)
    np.testing.assert_allclose(s_no.numpy(), np.array([3.0, 7.0]))


def test_mean_int_error():
    t = mt.Tensor([[1, 2], [3, 4]], dtype="int32")
    with pytest.raises(ValueError):
        t.mean(dim=[0])


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


def test_mean_negative_dim():
    t = mt.Tensor([[1.0, 2.0], [3.0, 4.0]])
    r = t.mean(dim=[-1])
    np.testing.assert_allclose(r.numpy(), np.array([1.5, 3.5]))


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


def test_empty_tensor_reductions():
    t = mt.Tensor(np.array([], dtype=np.float32))
    s = t.sum()
    m = t.mean()
    np.testing.assert_allclose(s.numpy(), np.array([0.0], dtype=np.float32))
    assert np.isinf(m.numpy())


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


def test_empty_ones_tensor():
    t = mt.Tensor.ones([0])
    assert t.numpy().size == 0
