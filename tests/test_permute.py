# Copyright (c) 2025 Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import pytest

import minitensor as mt


def test_permute_reorders_dimensions():
    x = mt.arange(24).reshape(2, 3, 4)
    y = x.permute(2, 0, 1)
    expected = np.arange(24, dtype=np.float32).reshape(2, 3, 4).transpose(2, 0, 1)
    assert np.allclose(y.numpy(), expected)


def test_permute_supports_negative_dims():
    x = mt.arange(24).reshape(2, 3, 4)
    y = x.permute(2, -3, -2)
    expected = np.arange(24, dtype=np.float32).reshape(2, 3, 4).transpose(2, 0, 1)
    assert np.allclose(y.numpy(), expected)


def test_permute_accepts_sequence():
    x = mt.arange(24).reshape(2, 3, 4)
    y = x.permute([2, 0, 1])
    expected = np.arange(24, dtype=np.float32).reshape(2, 3, 4).transpose(2, 0, 1)
    assert np.allclose(y.numpy(), expected)


def test_permute_invalid_dims_raises():
    x = mt.arange(6).reshape(1, 2, 3)
    with pytest.raises(ValueError):
        x.permute(0, 0, 1)


def test_permute_dim_length_mismatch():
    x = mt.arange(6).reshape(1, 2, 3)
    with pytest.raises(ValueError):
        x.permute(0, 1)


def test_permute_out_of_range():
    x = mt.arange(6).reshape(1, 2, 3)
    with pytest.raises(IndexError):
        x.permute(0, 1, 3)
