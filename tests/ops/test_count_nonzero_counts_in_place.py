# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""`count_nonzero` counts the elements where they lie.

It built a boolean mask of the input and summed it, which for a boolean input
copied the tensor into a mask identical to it -- seven times the cost of the
sum. A boolean tensor is now summed as it is, and a count over the whole of any
other tensor is folded straight from its elements.
"""

from __future__ import annotations

import numpy as np
import pytest

import minitensor as mt


@pytest.mark.parametrize("dtype", ["float32", "float64", "int32", "int64", "bool"])
def test_the_whole_tensor_count_of_every_dtype(dtype):
    rng = np.random.default_rng(71)
    values = rng.integers(-2, 3, 300_001)
    values = values != 0 if dtype == "bool" else values.astype(dtype)
    if dtype.startswith("float"):
        values[7] = np.nan  # NaN is not zero, so it counts.
        values[8] = -0.0  # And negative zero is zero.
    tensor = mt.Tensor(values)
    got = mt.count_nonzero(tensor)
    assert got.dtype == "int64" and got.shape == ()
    assert got.item() == np.count_nonzero(values)


@pytest.mark.parametrize("dtype", ["float32", "bool"])
def test_keepdim_dims_and_a_strided_view(dtype):
    values = np.random.default_rng(73).integers(0, 2, (40, 30))
    values = values != 0 if dtype == "bool" else values.astype(dtype)
    tensor = mt.Tensor(values)
    whole = mt.count_nonzero(tensor, keepdim=True)
    assert whole.shape == (1, 1) and whole.tolist() == [[np.count_nonzero(values)]]
    for axis in (0, 1):
        along = mt.count_nonzero(tensor, axis)
        assert along.dtype == "int64"
        assert along.tolist() == np.count_nonzero(values, axis).tolist()
    assert mt.count_nonzero(tensor.T).item() == np.count_nonzero(values)
    assert mt.count_nonzero(tensor[::3, 1::2]).item() == np.count_nonzero(
        values[::3, 1::2]
    )


def test_an_empty_tensor_counts_nothing():
    assert mt.count_nonzero(mt.zeros((0, 3))).item() == 0
    assert mt.count_nonzero(mt.Tensor(np.zeros((0,), dtype=bool))).item() == 0
