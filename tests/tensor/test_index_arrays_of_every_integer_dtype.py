# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""Any integer array names positions, and an int too large names none here.

Only int32 and int64 arrays were read as positions, and the refusal of the
rest said an integer array could index a tensor. A Python int past the
platform's width was refused as a type rather than reported out of bounds.
"""

from __future__ import annotations

import numpy as np
import pytest

import minitensor as mt

DTYPES = ["int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64"]


@pytest.mark.parametrize("dtype", DTYPES)
def test_an_integer_array_of_any_width_reads_and_writes_positions(dtype):
    values = np.arange(12.0).reshape(3, 4)
    tensor = mt.Tensor(values)
    positions = np.array([2, 0], dtype=dtype)
    np.testing.assert_array_equal(tensor[positions].numpy(), values[positions])
    np.testing.assert_array_equal(tensor[:, positions].numpy(), values[:, positions])
    written = mt.Tensor(np.zeros((3, 4)))
    written[positions] = 1.0
    assert written.numpy()[:, 0].tolist() == [1.0, 0.0, 1.0]


@pytest.mark.parametrize(
    "key", [2**70, -(2**70), (0, 2**70), np.array([2**63 + 5], dtype=np.uint64)]
)
def test_an_index_past_every_axis_is_out_of_bounds(key):
    with pytest.raises(IndexError, match="out of bounds"):
        mt.Tensor(np.zeros((3, 4)))[key]


@pytest.mark.parametrize("dtype", ["int32", "uint8", "int16"])
def test_gather_and_scatter_take_narrower_positions(dtype):
    values = np.arange(12.0).reshape(3, 4)
    positions = np.array([[0, 3]] * 3, dtype=dtype)
    gathered = mt.gather(mt.Tensor(values), 1, mt.as_tensor(positions))
    np.testing.assert_array_equal(
        gathered.numpy(), np.take_along_axis(values, positions.astype(np.int64), 1)
    )
    added = mt.scatter_add(
        mt.zeros(3), 0, mt.as_tensor(np.array([2, 0, 2], dtype=dtype)), mt.ones(3)
    )
    assert added.tolist() == [1.0, 0.0, 2.0]
    source = mt.Tensor(values, requires_grad=True)
    mt.gather(source, 1, mt.as_tensor(positions)).sum().backward()
    want = np.zeros_like(values)
    want[:, [0, 3]] = 1.0
    np.testing.assert_array_equal(source.grad.numpy(), want)
