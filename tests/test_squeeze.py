# Copyright (c) 2025 Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest

from minitensor import Tensor


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
