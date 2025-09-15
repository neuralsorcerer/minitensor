# Copyright (c) 2025 Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import pytest

import minitensor as mt
from minitensor import functional as F


def test_chunk_method_functional_top_level():
    t = mt.arange(0, 8).reshape((2, 4))
    method_parts = t.chunk(2, dim=1)
    func_parts = F.chunk(t, 2, dim=1)
    top_parts = mt.chunk(t, 2, dim=1)
    np_parts = np.split(t.numpy(), 2, axis=1)

    assert len(method_parts) == len(func_parts) == len(top_parts) == 2
    for m, f, tp, n in zip(method_parts, func_parts, top_parts, np_parts):
        np.testing.assert_array_equal(m.numpy(), n)
        np.testing.assert_array_equal(f.numpy(), n)
        np.testing.assert_array_equal(tp.numpy(), n)


def test_chunk_negative_dim():
    t = mt.arange(0, 9).reshape((3, 3))
    parts = mt.chunk(t, 3, dim=-1)
    np_parts = np.split(t.numpy(), 3, axis=-1)
    assert len(parts) == 3
    for p, n in zip(parts, np_parts):
        np.testing.assert_array_equal(p.numpy(), n)


def test_chunk_invalid_sections():
    t = mt.arange(0, 6)
    with pytest.raises(ValueError):
        mt.chunk(t, 4, dim=0)
