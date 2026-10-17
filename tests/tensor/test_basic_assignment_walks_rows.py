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
