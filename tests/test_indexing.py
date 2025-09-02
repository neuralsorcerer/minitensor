# Copyright (c) 2025 Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import minitensor as mt


def test_all_any():
    t = mt.Tensor([[1.0, 0.0], [2.0, 3.0]])
    assert t.any().tolist() == [True]
    assert t.all().tolist() == [False]
    b = mt.Tensor([[True, False], [True, True]], dtype="bool")
    res = b.all(dim=1)
    assert res.tolist() == [False, True]


def test_indexing_and_assignment():
    t = mt.Tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]])
    assert t[0, 1].tolist() == [1.0]
    t[0, 1] = 10.0
    assert t[0, 1].tolist() == [10.0]
    col = t[:, 1]
    assert col.tolist() == [10.0, 4.0]
