# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""Basic-subscript assignment lands every value where the subscript names.

The write walks the selection a row at a time -- a contiguous copy, a fill
for a broadcast value, a strided loop otherwise -- where it used to unravel
every element's coordinates, which made `t[...] = v` cost twelve times a
`copy_`. These check the walk against NumPy's assignment over random
subscripts, steps and broadcasts.
"""

from __future__ import annotations

import numpy as np
import pytest

import minitensor as mt


def _random_case(rng):
    shape = tuple(int(d) for d in rng.integers(1, 5, rng.integers(1, 4)))
    key = []
    for size in shape:
        if rng.integers(0, 3) == 0:
            key.append(int(rng.integers(0, size)))
        else:
            start = int(rng.integers(0, size))
            stop = int(rng.integers(start, size + 1))
            key.append(slice(start, stop, int(rng.integers(1, 3))))
    key = tuple(key[: rng.integers(1, len(key) + 1)])
    base = rng.standard_normal(shape)
    selected = base[key].shape
    if selected and rng.random() < 0.3:
        selected = selected[1:]
    value_shape = tuple(1 if rng.random() < 0.3 else d for d in selected)
    return base, key, rng.standard_normal(value_shape)


@pytest.mark.parametrize("dtype", ["float64", "float32", "int64"])
def test_assignment_agrees_with_numpy(dtype):
    rng = np.random.default_rng(11)
    for _ in range(600):
        base, key, value = _random_case(rng)
        base = base.astype(dtype)
        value = (value * 10).astype(dtype)
        want = base.copy()
        want[key] = value
        tensor = mt.Tensor(base, dtype=dtype)
        tensor[key] = mt.Tensor(value, dtype=dtype) if value.ndim else value.item()
        np.testing.assert_array_equal(tensor.numpy(), want)


def test_a_whole_tensor_and_a_long_row_are_written_exactly():
    values = np.arange(300_000, dtype=np.float32)
    tensor = mt.Tensor(np.zeros(300_000, dtype=np.float32))
    tensor[...] = mt.Tensor(values)
    np.testing.assert_array_equal(tensor.numpy(), values)
    matrix = mt.Tensor(np.zeros((3, 200_000), dtype=np.float32))
    matrix[1] = mt.Tensor(values[:200_000])
    np.testing.assert_array_equal(matrix.numpy()[1], values[:200_000])
    assert not matrix.numpy()[0].any() and not matrix.numpy()[2].any()
    # A strided write and a broadcast fill along the same long axis.
    matrix[2, ::2] = 7.0
    assert (matrix.numpy()[2, ::2] == 7.0).all() and not matrix.numpy()[2, 1::2].any()


@pytest.mark.parametrize("size", [5, 1000, 300_000])
@pytest.mark.parametrize("density", [0.0, 0.01, 0.5, 0.99, 1.0])
@pytest.mark.parametrize("mode", ["scalar", "values", "rows"])
def test_masked_assignment_agrees_with_numpy(size, density, mode):
    # The masked write is a branch-free select, split into bands that each
    # start from the values the bands before them consumed.
    rng = np.random.default_rng(size)
    if mode == "rows":
        base = rng.standard_normal((size // 5 + 1, 5))
        mask = rng.random(base.shape[0]) < density
        value = rng.standard_normal((int(mask.sum()), 5))
    else:
        base = rng.standard_normal(size)
        mask = rng.random(size) < density
        value = 7.0 if mode == "scalar" else rng.standard_normal(int(mask.sum()))
    want = base.copy()
    want[mask] = value
    tensor = mt.Tensor(base, dtype="float64")
    tensor[mt.Tensor(mask)] = value if mode == "scalar" else mt.Tensor(value)
    np.testing.assert_array_equal(tensor.numpy(), want)


@pytest.mark.parametrize("dtype", ["float64", "int32"])
def test_assignment_along_one_indexed_axis_agrees_with_numpy(dtype):
    # `t[:, idx] = v` with every other axis whole is written in one pass in
    # storage order; positions may repeat, wrap negatively and broadcast.
    rng = np.random.default_rng(23)
    for _ in range(500):
        shape = tuple(int(d) for d in rng.integers(1, 6, rng.integers(1, 4)))
        axis = int(rng.integers(0, len(shape)))
        count = int(rng.integers(1, 6))
        positions = [int(p) for p in rng.integers(-shape[axis], shape[axis], count)]
        key = tuple([slice(None)] * axis + [positions])
        base = (rng.standard_normal(shape) * 10).astype(dtype)
        selection = list(shape)
        selection[axis] = count
        choice = rng.integers(0, 3)
        if choice == 0:
            value = np.asarray(3, dtype=dtype)
        elif choice == 1:
            value = (rng.standard_normal(selection) * 10).astype(dtype)
        else:
            value = (rng.standard_normal(selection[axis:]) * 10).astype(dtype)
        want = base.copy()
        want[key] = value
        tensor = mt.Tensor(base, dtype=dtype)
        tensor[key] = mt.Tensor(value, dtype=dtype) if value.ndim else value.item()
        np.testing.assert_array_equal(tensor.numpy(), want)


def test_a_value_read_from_the_target_along_an_axis_is_read_first():
    grid = mt.Tensor(np.arange(12.0).reshape(3, 4))
    grid[:, [0, 1]] = grid[:, [1, 0]]
    assert grid.tolist()[0] == [1.0, 0.0, 2.0, 3.0]
    grid[:, [0, 1, 2, 3]] = grid
    assert grid.tolist()[0] == [1.0, 0.0, 2.0, 3.0]
