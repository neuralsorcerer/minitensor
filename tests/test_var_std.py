# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import pytest

import minitensor as mt


def test_var_std_support_tuple_dims_and_keepdim():
    data = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    tensor = mt.Tensor(data)

    var = tensor.var(dim=(1, 2), unbiased=False, keepdim=True)
    std = tensor.std(dim=(1, 2), unbiased=False, keepdim=False)

    np.testing.assert_allclose(
        var.numpy(), data.var(axis=(1, 2), keepdims=True), rtol=1e-6
    )
    np.testing.assert_allclose(std.numpy(), data.std(axis=(1, 2)), rtol=1e-6)


def test_unbiased_var_single_sample_returns_nan_without_warning():
    tensor = mt.Tensor([[1.0], [2.0]], dtype="float32")
    result = tensor.var(dim=1, unbiased=True)

    assert result.shape == (2,)
    assert np.isnan(result.numpy()).all()


def test_var_rejects_duplicate_and_invalid_dims_like_other_reductions():
    tensor = mt.Tensor(np.arange(6, dtype=np.float32).reshape(2, 3))

    np.testing.assert_allclose(
        tensor.var(dim=(1, -1), unbiased=False).numpy(),
        tensor.var(dim=1, unbiased=False).numpy(),
        rtol=1e-6,
    )

    with pytest.raises(IndexError):
        tensor.var(dim=2)
