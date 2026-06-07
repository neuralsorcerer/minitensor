# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np

import minitensor as mt
import minitensor.functional as F


def test_nan_to_num_defaults_match_dtype_limits_float32():
    x = mt.Tensor([float("nan"), float("inf"), -float("inf"), -2.5, 3.0])

    out = x.nan_to_num()

    expected = np.nan_to_num(
        np.array([np.nan, np.inf, -np.inf, -2.5, 3.0], dtype=np.float32)
    )
    np.testing.assert_allclose(out.numpy(), expected)
    assert out.dtype == "float32"


def test_nan_to_num_accepts_custom_replacements_float64_and_functional():
    x = mt.Tensor([float("nan"), float("inf"), -float("inf"), 7.0], dtype="float64")

    out = F.nan_to_num(x, nan=-1.5, posinf=9.0, neginf=-9.0)

    expected = np.array([-1.5, 9.0, -9.0, 7.0], dtype=np.float64)
    np.testing.assert_allclose(out.numpy(), expected)
    assert out.dtype == "float64"


def test_nan_to_num_preserves_exact_tensors():
    ints = mt.Tensor([1, -2, 3], dtype="int64")
    bools = mt.Tensor([True, False], dtype="bool")

    np.testing.assert_array_equal(ints.nan_to_num(nan=99.0).numpy(), ints.numpy())
    np.testing.assert_array_equal(
        F.nan_to_num(bools, posinf=5.0).numpy(), bools.numpy()
    )


def test_nan_to_num_empty_tensor_keeps_shape():
    x = mt.empty(0, 3)

    out = x.nan_to_num(nan=1.0, posinf=2.0, neginf=-2.0)

    assert out.shape == (0, 3)
    assert out.numpy().shape == (0, 3)


def test_nan_to_num_backward_masks_replaced_entries():
    x = mt.Tensor(
        [float("nan"), float("inf"), -float("inf"), -2.0, 3.0], requires_grad=True
    )

    y = x.nan_to_num(nan=0.0, posinf=10.0, neginf=-10.0).sum()
    y.backward()

    np.testing.assert_allclose(
        x.grad.numpy(), np.array([0.0, 0.0, 0.0, 1.0, 1.0], dtype=np.float32)
    )
    mt.clear_autograd_graph()
