# Copyright (c) 2025 Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import pytest

import minitensor as mt
import minitensor.functional as F


def test_repeat_interleave_basic():
    t = mt.arange(0, 5)
    r = t.repeat_interleave(2)
    np_r = np.repeat(np.arange(0, 5), 2)
    assert np.array_equal(r.numpy(), np_r)


def test_repeat_interleave_dim():
    t = mt.arange(0, 6).reshape((3, 2))
    r = t.repeat_interleave((1, 2, 3), dim=0)
    np_r = np.repeat(np.arange(0, 6).reshape((3, 2)), (1, 2, 3), axis=0)
    assert np.array_equal(r.numpy(), np_r)


def test_repeat_interleave_negative_dim():
    t = mt.arange(0, 6).reshape((2, 3))
    r = t.repeat_interleave(2, dim=-1)
    np_r = np.repeat(np.arange(0, 6).reshape((2, 3)), 2, axis=-1)
    assert np.array_equal(r.numpy(), np_r)


def test_repeat_interleave_zero():
    t = mt.arange(0, 3)
    r = t.repeat_interleave([0, 1, 2])
    np_r = np.repeat(np.arange(0, 3), [0, 1, 2])
    assert np.array_equal(r.numpy(), np_r)
    assert r.shape == np_r.shape


def test_repeat_interleave_mismatch_raises():
    t = mt.arange(0, 3)
    with pytest.raises(ValueError):
        t.repeat_interleave([1, 2], dim=0)


def test_functional_and_top_level_repeat_interleave():
    t = mt.arange(0, 4)
    r_method = t.repeat_interleave(2)
    r_func = F.repeat_interleave(t, 2, output_size=8)
    r_top = mt.repeat_interleave(t, 2, output_size=8)
    expected = np.repeat(np.arange(0, 4), 2)
    assert np.array_equal(r_method.numpy(), expected)
    assert np.array_equal(r_func.numpy(), expected)
    assert np.array_equal(r_top.numpy(), expected)


def test_repeat_interleave_tensor_repeats():
    t = mt.arange(0, 6).reshape((3, 2))
    repeats = mt.Tensor([1, 2, 1], dtype="int64")
    r = t.repeat_interleave(repeats, dim=0)
    np_r = np.repeat(np.arange(0, 6).reshape((3, 2)), [1, 2, 1], axis=0)
    assert np.array_equal(r.numpy(), np_r)


def test_repeat_interleave_output_size_validation():
    t = mt.arange(0, 4)
    repeats = mt.Tensor([1, 0, 2, 1], dtype="int64")
    r = mt.repeat_interleave(t, repeats, output_size=4)
    expected = np.repeat(np.arange(0, 4), [1, 0, 2, 1])
    assert np.array_equal(r.numpy(), expected)
    with pytest.raises(ValueError):
        mt.repeat_interleave(t, repeats, output_size=3)


def test_repeat_interleave_negative_repeat_raises():
    t = mt.arange(0, 3)
    with pytest.raises(ValueError):
        t.repeat_interleave([-1, 1, 1])


def test_repeat_interleave_backward_gradients():
    x = mt.arange(0.0, 3.0, dtype="float32", requires_grad=True)
    y = x.repeat_interleave([1, 2, 1])
    y.sum().backward()
    grad = x.grad
    assert grad is not None
    np.testing.assert_allclose(
        grad.numpy(), np.array([1.0, 2.0, 1.0], dtype=np.float32)
    )


def test_repeat_interleave_flatten_backward():
    base = mt.arange(0.0, 6.0, dtype="float32", requires_grad=True)
    reshaped = base.reshape((2, 3))
    repeated = reshaped.repeat_interleave(2)
    repeated.sum().backward()

    grad = base.grad
    assert grad is not None
    np.testing.assert_allclose(grad.numpy(), np.full(6, 2.0, dtype=np.float32))


def test_repeat_interleave_tensor_invalid_dtype():
    t = mt.arange(0, 3)
    repeats = mt.Tensor([1.0, 2.0, 3.0], dtype="float32")
    with pytest.raises(TypeError):
        t.repeat_interleave(repeats)
