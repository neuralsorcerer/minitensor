# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import pytest

import minitensor as mt
from minitensor import Tensor
from minitensor import functional as F
from minitensor import repeat as top_repeat
from minitensor.functional import repeat as F_repeat


def test_cat_functional_and_top_level():
    a = mt.arange(0, 6).reshape((2, 3))
    b = mt.arange(6, 12).reshape((2, 3))
    func_res = F.cat([a, b], dim=1)
    top_res = mt.cat([a, b], dim=1)
    np.testing.assert_array_equal(
        func_res.numpy(), np.concatenate([a.numpy(), b.numpy()], axis=1)
    )
    np.testing.assert_array_equal(top_res.numpy(), func_res.numpy())


def test_cat_negative_dim():
    a = mt.arange(0, 6).reshape((2, 3))
    b = mt.arange(6, 12).reshape((2, 3))
    res = mt.cat([a, b], dim=-1)
    np.testing.assert_array_equal(
        res.numpy(), np.concatenate([a.numpy(), b.numpy()], axis=-1)
    )


def test_cat_zero_dim():
    a = mt.zeros((2, 0, 3))
    b = mt.zeros((2, 0, 3))
    res = mt.cat([a, b], dim=1)
    np.testing.assert_array_equal(
        res.numpy(), np.concatenate([a.numpy(), b.numpy()], axis=1)
    )


def test_stack_functional_and_top_level():
    a = mt.arange(0, 6).reshape((2, 3))
    b = mt.arange(6, 12).reshape((2, 3))
    func_res = F.stack([a, b], dim=0)
    top_res = mt.stack([a, b], dim=0)
    np.testing.assert_array_equal(
        func_res.numpy(), np.stack([a.numpy(), b.numpy()], axis=0)
    )
    np.testing.assert_array_equal(top_res.numpy(), func_res.numpy())


def test_stack_negative_dim():
    a = mt.arange(0, 6).reshape((2, 3))
    b = mt.arange(6, 12).reshape((2, 3))
    res = F.stack([a, b], dim=-1)
    np.testing.assert_array_equal(
        res.numpy(), np.stack([a.numpy(), b.numpy()], axis=-1)
    )


def test_chunk_method_functional_top_level():
    t = mt.arange(0, 8).reshape((2, 4))
    method_parts = t.chunk(2, dim=1)
    func_parts = F.chunk(t, 2, dim=1)
    top_parts = mt.chunk(t, 2, dim=1)
    np_parts = np.split(t.numpy(), 2, axis=1)

    assert len(method_parts) == len(func_parts) == len(top_parts) == 2
    for m, f, tp, n in zip(method_parts, func_parts, top_parts, np_parts):
        np.testing.assert_array_equal(m.numpy(), n)
        np.testing.assert_array_equal(f.numpy(), n)
        np.testing.assert_array_equal(tp.numpy(), n)


def test_chunk_negative_dim():
    t = mt.arange(0, 9).reshape((3, 3))
    parts = mt.chunk(t, 3, dim=-1)
    np_parts = np.split(t.numpy(), 3, axis=-1)
    assert len(parts) == 3
    for p, n in zip(parts, np_parts):
        np.testing.assert_array_equal(p.numpy(), n)


def test_chunk_invalid_sections():
    t = mt.arange(0, 6)
    with pytest.raises(ValueError):
        mt.chunk(t, 4, dim=0)


def test_split_method_functional_top_level_int():
    t = mt.arange(0, 10)
    method_parts = t.split(4)
    func_parts = F.split(t, 4)
    top_parts = mt.split(t, 4)
    expected = [np.arange(0, 4), np.arange(4, 8), np.arange(8, 10)]

    assert len(method_parts) == len(func_parts) == len(top_parts) == len(expected)
    for m, f, tp, n in zip(method_parts, func_parts, top_parts, expected):
        np.testing.assert_array_equal(m.numpy(), n)
        np.testing.assert_array_equal(f.numpy(), n)
        np.testing.assert_array_equal(tp.numpy(), n)


def test_split_negative_dim():
    t = mt.arange(0, 9).reshape((3, 3))
    parts = mt.split(t, 2, dim=-1)
    np_parts = np.array_split(t.numpy(), 2, axis=-1)

    assert len(parts) == len(np_parts)
    for p, n in zip(parts, np_parts):
        np.testing.assert_array_equal(p.numpy(), n)


def test_split_with_explicit_sections_and_dim():
    t = mt.arange(0, 10).reshape((2, 5))
    method_parts = t.split([2, 3], dim=1)
    func_parts = F.split(t, [2, 3], dim=1)
    top_parts = mt.split(t, [2, 3], dim=1)
    np_parts = np.array_split(t.numpy(), [2], axis=1)

    assert len(method_parts) == len(func_parts) == len(top_parts) == 2
    for m, f, tp, n in zip(method_parts, func_parts, top_parts, np_parts):
        np.testing.assert_array_equal(m.numpy(), n)
        np.testing.assert_array_equal(f.numpy(), n)
        np.testing.assert_array_equal(tp.numpy(), n)


def test_split_size_mismatch_raises():
    t = mt.arange(0, 6)
    with pytest.raises(ValueError):
        t.split([2, 5])


class IndexLike:
    def __init__(self, value: int):
        self._value = value

    def __index__(self) -> int:  # pragma: no cover - simple accessor
        return self._value


def test_tensor_repeat_basic():
    t = Tensor([1, 2])
    r = t.repeat(2)
    assert r.numpy().tolist() == [1, 2, 1, 2]


def test_tensor_repeat_multi_dim():
    t = Tensor([[1, 2], [3, 4]])
    r = t.repeat(2, 3)
    expected = np.tile(np.array([[1, 2], [3, 4]]), (2, 3))
    assert r.shape == expected.shape
    assert np.array_equal(r.numpy(), expected)


def test_repeat_functional_and_top_level():
    t = Tensor([1, 2])
    r_func = F_repeat(t, 2, 3, 1)
    r_top = top_repeat(t, 2, 3, 1)
    expected = np.tile(np.array([1, 2]), (2, 3, 1))
    assert np.array_equal(r_func.numpy(), expected)
    assert np.array_equal(r_top.numpy(), expected)


def test_repeat_errors():
    t = Tensor([[1, 2], [3, 4]])
    with pytest.raises(ValueError):
        t.repeat(2)
    with pytest.raises(ValueError):
        t.repeat(2, -1)


def test_repeat_zero():
    t = Tensor([[1, 2], [3, 4]])
    r = t.repeat(0, 2)
    assert r.shape == (0, 4)
    assert r.numel() == 0


def test_repeat_accepts_index_like_scalars():
    t = Tensor([[1, 2], [3, 4]])
    repeated = t.repeat(IndexLike(1), IndexLike(2))
    assert repeated.shape == (2, 4)

    via_sequence = t.repeat([IndexLike(1), IndexLike(2)])
    assert via_sequence.shape == (2, 4)

    numpy_repeats = (np.int64(2), np.int64(1))
    repeated_numpy = t.repeat(*numpy_repeats)
    assert repeated_numpy.shape == (4, 2)


def test_repeat_rejects_non_integer_values():
    t = Tensor([1, 2])
    with pytest.raises(TypeError):
        t.repeat(2.5)

    with pytest.raises(TypeError):
        t.repeat([1, 2.2])


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


def test_roll_1d():
    t = mt.arange(0, 5)
    r = t.roll(2)
    np_r = np.roll(np.arange(0, 5), 2)
    np.testing.assert_array_equal(r.numpy(), np_r)


def test_roll_multi_dim():
    t = mt.arange(0, 12).reshape((3, 4))
    r = t.roll((1, 2), dims=(0, 1))
    np_r = np.roll(t.numpy(), shift=(1, 2), axis=(0, 1))
    np.testing.assert_array_equal(r.numpy(), np_r)


def test_roll_negative_shift():
    t = mt.arange(0, 5)
    r = t.roll(-1)
    np_r = np.roll(np.arange(0, 5), -1)
    np.testing.assert_array_equal(r.numpy(), np_r)


def test_roll_mismatch_raises():
    t = mt.arange(0, 5)
    with pytest.raises(ValueError):
        t.roll((1, 2), dims=(0,))


def test_functional_and_top_level_roll():
    t = mt.arange(0, 5)
    r_func = F.roll(t, 1)
    r_top = mt.roll(t, 1)
    np.testing.assert_array_equal(r_func.numpy(), r_top.numpy())


def test_flip_1d():
    t = mt.arange(0, 5)
    r = t.flip(0)
    np_r = np.flip(np.arange(0, 5), 0)
    np.testing.assert_array_equal(r.numpy(), np_r)


def test_flip_multi_dim():
    t = mt.arange(0, 12).reshape((3, 4))
    r = t.flip((0, 1))
    np_r = np.flip(t.numpy(), axis=(0, 1))
    np.testing.assert_array_equal(r.numpy(), np_r)


def test_flip_negative_dims():
    t = mt.arange(0, 6).reshape((2, 3))
    r = t.flip(-1)
    np_r = np.flip(t.numpy(), axis=-1)
    np.testing.assert_array_equal(r.numpy(), np_r)


def test_flip_duplicate_dims_error():
    t = mt.arange(0, 5)
    with pytest.raises(ValueError):
        t.flip((0, 0))


def test_functional_and_top_level_flip():
    t = mt.arange(0, 5)
    r_func = F.flip(t, 0)
    r_top = mt.flip(t, 0)
    np.testing.assert_array_equal(r_func.numpy(), r_top.numpy())
