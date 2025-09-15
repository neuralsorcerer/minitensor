# Copyright (c) 2025 Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import pytest

from minitensor import Tensor
from minitensor import functional as F
from minitensor import squeeze, unsqueeze


def test_unsqueeze_negative_dim():
    t = Tensor([[1, 2, 3]])
    u = t.unsqueeze(-1)
    assert u.shape == (1, 3, 1)
    u2 = t.unsqueeze(-2)
    assert u2.shape == (1, 1, 3)
    with pytest.raises(IndexError):
        t.unsqueeze(3)
    with pytest.raises(IndexError):
        t.unsqueeze(-4)


def test_squeeze_behavior():
    t = Tensor([[[1.0], [2.0], [3.0]]])  # shape (1,3,1)
    s = t.squeeze()
    assert s.shape == (3,)
    s_neg = t.squeeze(-1)
    assert s_neg.shape == (1, 3)
    s_neg2 = t.squeeze(-3)
    assert s_neg2.shape == (3, 1)
    scalar = Tensor([[[1.0]]]).squeeze()
    assert scalar.shape == ()
    with pytest.raises(IndexError):
        t.squeeze(3)
    with pytest.raises(IndexError):
        t.squeeze(-4)


def test_functional_squeeze_unsqueeze():
    t = Tensor.arange(0, 6).reshape([1, 2, 3, 1])

    s = F.squeeze(t, 0)
    assert np.array_equal(s.numpy(), t.squeeze(0).numpy())

    u = F.unsqueeze(s, 0)
    assert np.array_equal(u.numpy(), s.unsqueeze(0).numpy())


def test_top_level_squeeze_unsqueeze():
    t = Tensor.arange(0, 6).reshape([2, 3])

    u = unsqueeze(t, 0)
    assert np.array_equal(u.numpy(), t.unsqueeze(0).numpy())

    s = squeeze(u, 0)
    assert np.array_equal(s.numpy(), t.numpy())
