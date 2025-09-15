# Copyright (c) 2025 Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest

import minitensor as mt
from minitensor.functional import moveaxis as F_moveaxis
from minitensor.functional import movedim as F_movedim


def test_tensor_movedim_single_and_negative():
    x = mt.arange(6).reshape(1, 2, 3)
    y = x.movedim(0, -1)
    expected = x.permute(1, 2, 0)
    assert y.tolist() == expected.tolist()
    assert y.shape == (2, 3, 1)


def test_tensor_movedim_sequence():
    x = mt.arange(24).reshape(2, 3, 4)
    y = x.movedim((0, 2), (1, 0))
    expected = x.permute(2, 0, 1)
    assert y.tolist() == expected.tolist()
    assert y.shape == (4, 2, 3)


def test_tensor_moveaxis_alias():
    x = mt.arange(6).reshape(1, 2, 3)
    y = x.moveaxis(0, 2)
    expected = x.movedim(0, 2)
    assert y.tolist() == expected.tolist()


def test_functional_and_top_level_movedim_moveaxis():
    x = mt.arange(6).reshape(1, 2, 3)
    expected = x.permute(1, 2, 0)
    assert F_movedim(x, 0, 2).tolist() == expected.tolist()
    assert F_moveaxis(x, 0, 2).tolist() == expected.tolist()
    assert mt.movedim(x, 0, 2).tolist() == expected.tolist()
    assert mt.moveaxis(x, 0, 2).tolist() == expected.tolist()


def test_movedim_invalid_length():
    x = mt.arange(6)
    with pytest.raises(ValueError):
        x.movedim((0, 1), (0,))
