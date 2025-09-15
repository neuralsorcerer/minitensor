# Copyright (c) 2025 Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np

import minitensor as mt
from minitensor import functional as F


def test_cat_functional_and_top_level():
    a = mt.arange(0, 6).reshape((2, 3))
    b = mt.arange(6, 12).reshape((2, 3))
    func_res = F.cat([a, b], dim=1)
    top_res = mt.cat([a, b], dim=1)
    np.testing.assert_array_equal(
        func_res.numpy(), np.concatenate([a.numpy(), b.numpy()], axis=1)
    )
    np.testing.assert_array_equal(top_res.numpy(), func_res.numpy())


def test_cat_negative_dim():
    a = mt.arange(0, 6).reshape((2, 3))
    b = mt.arange(6, 12).reshape((2, 3))
    res = mt.cat([a, b], dim=-1)
    np.testing.assert_array_equal(
        res.numpy(), np.concatenate([a.numpy(), b.numpy()], axis=-1)
    )


def test_stack_functional_and_top_level():
    a = mt.arange(0, 6).reshape((2, 3))
    b = mt.arange(6, 12).reshape((2, 3))
    func_res = F.stack([a, b], dim=0)
    top_res = mt.stack([a, b], dim=0)
    np.testing.assert_array_equal(
        func_res.numpy(), np.stack([a.numpy(), b.numpy()], axis=0)
    )
    np.testing.assert_array_equal(top_res.numpy(), func_res.numpy())


def test_stack_negative_dim():
    a = mt.arange(0, 6).reshape((2, 3))
    b = mt.arange(6, 12).reshape((2, 3))
    res = F.stack([a, b], dim=-1)
    np.testing.assert_array_equal(
        res.numpy(), np.stack([a.numpy(), b.numpy()], axis=-1)
    )
