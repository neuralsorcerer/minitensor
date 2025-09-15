# Copyright (c) 2025 Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import pytest

from minitensor import flatten
from minitensor import functional as F
from minitensor import ravel, view
from minitensor.tensor import Tensor


def test_flatten_and_ravel():
    t = Tensor.ones([2, 3, 4])
    f = t.flatten()
    r = t.ravel()
    assert f.shape == (24,)
    assert r.shape == (24,)
    assert np.array_equal(f.numpy(), r.numpy())


def test_flatten_range_and_error():
    t = Tensor.arange(0, 24).reshape([2, 3, 4])
    f = t.flatten(1, -1)
    assert f.shape == (2, 12)
    with pytest.raises(ValueError):
        t.flatten(2, 0)


def test_flatten_negative_start_dim():
    t = Tensor.arange(0, 24).reshape([2, 3, 4])
    f = t.flatten(-3, -2)
    assert f.shape == (6, 4)


def test_functional_flatten():
    t = Tensor.arange(0, 24).reshape([2, 3, 4])
    f = F.flatten(t, 1, -1)
    assert f.shape == (2, 12)
    assert np.array_equal(f.numpy(), t.flatten(1, -1).numpy())


def test_top_level_flatten():
    t = Tensor.arange(0, 24).reshape([2, 3, 4])
    f = flatten(t, 1, -1)
    assert f.shape == (2, 12)
    assert np.array_equal(f.numpy(), t.flatten(1, -1).numpy())


def test_functional_ravel():
    t = Tensor.arange(0, 24).reshape([2, 3, 4])
    r = F.ravel(t)
    assert r.shape == (24,)
    assert np.array_equal(r.numpy(), t.ravel().numpy())


def test_top_level_ravel():
    t = Tensor.arange(0, 24).reshape([2, 3, 4])
    r = ravel(t)
    assert r.shape == (24,)
    assert np.array_equal(r.numpy(), t.ravel().numpy())


def test_functional_view():
    t = Tensor.arange(0, 24)
    v = F.view(t, 2, 12)
    assert v.shape == (2, 12)
    assert np.array_equal(v.numpy(), t.view(2, 12).numpy())


def test_top_level_view():
    t = Tensor.arange(0, 24)
    v = view(t, 2, 12)
    assert v.shape == (2, 12)
    assert np.array_equal(v.numpy(), t.view(2, 12).numpy())


def test_view_invalid_shape():
    t = Tensor.arange(0, 10)
    with pytest.raises(ValueError):
        F.view(t, 3, 4)
