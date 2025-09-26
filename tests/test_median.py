# Copyright (c) 2025 Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import pytest

import minitensor as mt


def test_median_global_even_length():
    x = mt.Tensor([3.0, 1.0, 4.0, 2.0], dtype="float32")
    median = x.median()
    assert median.shape == ()
    assert median.numpy() == pytest.approx(2.0)


def test_median_with_dim_returns_indices():
    x = mt.Tensor([[1.0, 3.0, 2.0], [4.0, 6.0, 5.0]], dtype="float32")
    values, indices = x.median(dim=1)
    np.testing.assert_allclose(values.numpy(), np.array([2.0, 5.0], dtype=np.float32))
    np.testing.assert_array_equal(indices.numpy(), np.array([2, 2], dtype=np.int64))


def test_median_keepdim_matches_pytorch_shape():
    x = mt.Tensor([[1.0, 2.0], [3.0, 4.0]], dtype="float32")
    values, indices = x.median(dim=1, keepdim=True)
    assert values.shape == (2, 1)
    assert indices.shape == (2, 1)
    np.testing.assert_allclose(
        values.numpy(), np.array([[1.0], [3.0]], dtype=np.float32)
    )
    np.testing.assert_array_equal(indices.numpy(), np.zeros((2, 1), dtype=np.int64))


def test_median_empty_tensor_raises():
    x = mt.Tensor(np.empty((0, 3), dtype=np.float32))
    with pytest.raises(RuntimeError):
        x.median()
