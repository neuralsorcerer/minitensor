# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import pytest

import minitensor as mt


def test_allclose_top_level_accepts_tensor_like_inputs_and_tolerances():
    assert mt.allclose([1.0, 2.0], np.array([1.0, 2.0 + 1e-6], dtype=np.float32))
    assert not mt.allclose([1.0, 2.0], [1.0, 2.1], rtol=1e-6, atol=1e-6)


def test_allclose_equal_nan_matches_numpy_style_option():
    left = mt.tensor([1.0, float("nan"), float("inf")])
    right = mt.tensor([1.0, float("nan"), float("inf")])

    assert not left.allclose(right)
    assert left.allclose(right, equal_nan=True)
    assert mt.allclose(left, right, equal_nan=True)


def test_allclose_handles_signed_zero_and_infinities():
    assert mt.allclose(
        [0.0, float("inf"), -float("inf")], [-0.0, float("inf"), -float("inf")]
    )
    assert not mt.allclose([float("inf")], [-float("inf")])


def test_equality_helpers_promote_compatible_numeric_dtypes():
    ints = mt.tensor([1, 2, 3], dtype="int64")
    floats = mt.tensor([1.0, 2.0, 3.0], dtype="float32")

    assert ints.allclose(floats)
    assert mt.allclose(ints, floats)
    assert mt.array_equal(ints, floats)


def test_allclose_rejects_invalid_tolerances():
    tensor = mt.tensor([1.0])

    with pytest.raises(ValueError, match="rtol and atol"):
        tensor.allclose(tensor, rtol=-1.0)

    with pytest.raises(ValueError, match="rtol and atol"):
        mt.allclose(tensor, tensor, atol=float("nan"))


def test_array_equal_top_level_accepts_tensor_like_inputs():
    assert mt.array_equal([1, 2, 3], np.array([1, 2, 3], dtype=np.int64))
    assert not mt.array_equal([1, 2, 3], [1, 2, 4])
