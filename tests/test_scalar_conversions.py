# Copyright (c) 2025 Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest

import minitensor as mt


def test_tolist_returns_python_scalar_for_zero_dim():
    tensor = mt.Tensor(3.5)

    value = tensor.tolist()

    assert isinstance(value, float)
    assert value == pytest.approx(3.5)


def test_tolist_preserves_python_bool():
    tensor = mt.Tensor(True, dtype="bool")

    value = tensor.tolist()

    assert isinstance(value, bool)
    assert value is True


def test_item_returns_native_python_scalar():
    tensor = mt.Tensor(7, dtype="int64")

    value = tensor.item()

    assert isinstance(value, int)
    assert value == 7


def test_item_error_matches_pytorch_message():
    tensor = mt.Tensor([1.0, 2.0, 3.0])

    with pytest.raises(RuntimeError) as exc_info:
        tensor.item()

    assert "a Tensor with 3 elements cannot be converted to Scalar" in str(
        exc_info.value
    )
