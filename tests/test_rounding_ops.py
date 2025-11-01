# Copyright (c) 2025 Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import pytest

import minitensor as mt


def _make_tensor_values():
    return np.array([-1.234, -0.5, 0.0, 2.718, 3.1415], dtype=np.float32)


def _round_half_away_from_zero(values: np.ndarray, decimals: int = 0) -> np.ndarray:
    multiplier = np.power(10.0, decimals, dtype=values.dtype)
    scaled = values * multiplier
    rounded = np.sign(scaled) * np.floor(np.abs(scaled) + 0.5)
    return rounded / multiplier


def test_tensor_round_defaults_to_zero_decimals():
    values = _make_tensor_values()
    tensor = mt.Tensor(values.tolist(), dtype="float32")

    rounded = tensor.round()

    np.testing.assert_allclose(rounded.numpy(), _round_half_away_from_zero(values))
    assert rounded.dtype == tensor.dtype


def test_tensor_round_with_decimals():
    values = _make_tensor_values()
    tensor = mt.Tensor(values.tolist(), dtype="float32")

    rounded = tensor.round(decimals=2)

    np.testing.assert_allclose(rounded.numpy(), _round_half_away_from_zero(values, 2))
    assert rounded.dtype == tensor.dtype


def test_floor_and_ceil_match_numpy():
    values = _make_tensor_values()
    tensor = mt.Tensor(values.tolist(), dtype="float32")

    floored = tensor.floor()
    ceiled = tensor.ceil()

    np.testing.assert_allclose(floored.numpy(), np.floor(values))
    np.testing.assert_allclose(ceiled.numpy(), np.ceil(values))


def test_rounding_ops_raise_for_integer_tensors():
    tensor = mt.Tensor.arange(-3, 4, dtype="int32")

    with pytest.raises(ValueError):
        tensor.round()

    with pytest.raises(ValueError):
        tensor.floor()

    with pytest.raises(ValueError):
        tensor.ceil()


def test_functional_round_floor_ceil_forwarders():
    values = _make_tensor_values()
    tensor = mt.Tensor(values.tolist(), dtype="float32")

    rounded = mt.functional.round(tensor, decimals=1)
    floored = mt.functional.floor(tensor)
    ceiled = mt.functional.ceil(tensor)

    np.testing.assert_allclose(rounded.numpy(), np.round(values, 1))
    np.testing.assert_allclose(floored.numpy(), np.floor(values))
    np.testing.assert_allclose(ceiled.numpy(), np.ceil(values))
