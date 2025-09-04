# Copyright (c) 2025 Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np

from minitensor.tensor import Tensor


def test_to_float64():
    x = Tensor([1.5, -2.3], dtype="float32")
    y = x.to("float64")
    assert y.dtype == "float64"
    assert np.allclose(y.numpy(), np.array([1.5, -2.3], dtype=np.float64))


def test_astype_int():
    x = Tensor([1.5, -2.3], dtype="float32")
    y = x.astype("int32")
    assert y.dtype == "int32"
    assert np.array_equal(y.numpy(), np.array([1, -2], dtype=np.int32))
