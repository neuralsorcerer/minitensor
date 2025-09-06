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


def test_sum_negative_dim_error():
    t = mt.Tensor([[1.0, 2.0], [3.0, 4.0]])
    with pytest.raises(OverflowError):
        t.sum(dim=[-1])


def test_sum_multiple_dims_not_implemented():
    t = mt.Tensor([[1.0, 2.0], [3.0, 4.0]])
    with pytest.raises(NotImplementedError):
        t.sum(dim=[0, 1])


def test_any_all_keepdim():
    t = mt.Tensor([[1.0, 0.0], [0.0, 2.0]])
    any_res = t.any(dim=1, keepdim=True)
    all_res = t.all(dim=0)
    np.testing.assert_array_equal(any_res.numpy(), np.array([[True], [True]]))
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
