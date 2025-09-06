# Copyright (c) 2025 Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np

from minitensor import Tensor


def test_trigonometric_functions():
    angles = [
        0.0,
        np.pi / 6,
        np.pi / 4,
        np.pi / 2,
        np.pi,
        -np.pi / 6,
        -np.pi / 4,
        -np.pi / 2,
        -np.pi,
    ]
    x = Tensor(angles)
    sin_result = x.sin().tolist()
    cos_result = x.cos().tolist()

    # Test sine and cosine across the full range
    np.testing.assert_allclose(sin_result, np.sin(angles), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(cos_result, np.cos(angles), rtol=1e-6, atol=1e-6)

    # Test tangent only where defined (avoid singularities at ±pi/2)
    tan_angles = [a for a in angles if abs(np.cos(a)) > 1e-6]
    tan_result = Tensor(tan_angles).tan().tolist()
    np.testing.assert_allclose(tan_result, np.tan(tan_angles), rtol=1e-6, atol=1e-6)


def test_trig_large_values():
    x = Tensor([1e10])
    sin_res = x.sin().numpy()[0]
    cos_res = x.cos().numpy()[0]
    np.testing.assert_allclose(sin_res, np.sin(1e10), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(cos_res, np.cos(1e10), rtol=1e-6, atol=1e-6)
