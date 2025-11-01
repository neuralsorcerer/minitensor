# Copyright (c) 2025 Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import pytest

import minitensor as mt


def test_clip_float_range():
    tensor = mt.Tensor([-2.0, -0.5, 0.25, 1.5], dtype="float32")
    clipped = tensor.clip(-1.0, 1.0)
    np.testing.assert_allclose(clipped.numpy(), np.array([-1.0, -0.5, 0.25, 1.0], dtype=np.float32))
    assert clipped.dtype == tensor.dtype


def test_clip_with_single_bound_int():
    tensor = mt.Tensor.arange(-3, 3, dtype="int32")
    clipped = tensor.clip(min=0)
    np.testing.assert_array_equal(clipped.numpy(), np.clip(np.arange(-3, 3, dtype=np.int32), 0, None))
    assert clipped.dtype == "int32"


def test_clamp_alias_matches_clip():
    tensor = mt.Tensor([-2.0, 0.0, 2.0], dtype="float64")
    clip_result = tensor.clip(-0.5, 0.5)
    clamp_result = tensor.clamp(-0.5, 0.5)
    np.testing.assert_allclose(clip_result.numpy(), clamp_result.numpy())


def test_clamp_min_max_helpers():
    tensor = mt.Tensor([-2.0, 0.0, 2.0], dtype="float64")
    min_only = tensor.clamp_min(-0.25)
    max_only = tensor.clamp_max(1.25)
    np.testing.assert_allclose(min_only.numpy(), np.array([-0.25, 0.0, 2.0], dtype=np.float64))
    np.testing.assert_allclose(max_only.numpy(), np.array([-2.0, 0.0, 1.25], dtype=np.float64))


def test_clip_raises_for_invalid_bounds():
    tensor = mt.Tensor([1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        tensor.clip(2.0, 1.0)


def test_functional_clip_uses_tensor_method():
    tensor = mt.Tensor([-1.0, 0.25, 1.5])
    clipped = mt.functional.clip(tensor, -0.5, 0.75)
    np.testing.assert_allclose(clipped.numpy(), np.array([-0.5, 0.25, 0.75], dtype=np.float32))
