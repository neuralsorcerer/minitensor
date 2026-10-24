# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""`frexp` takes a subnormal's exponent off exactly.

The mantissa was the product with `exp2(-exponent)`, and for a subnormal that
power is past the top of the dtype: `frexp(1e-310)` had an infinite mantissa.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import minitensor as mt


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_subnormals_and_extremes(dtype):
    info = np.finfo(dtype)
    tiny = float(info.smallest_subnormal)
    values = np.array(
        [
            tiny,
            -tiny,
            tiny * 3,
            float(info.tiny) / 2,
            float(info.tiny),
            float(info.max),
        ],
        dtype=dtype,
    )
    mantissa, exponent = mt.frexp(mt.Tensor(values, dtype=dtype))
    for value, m, e in zip(values.tolist(), mantissa.tolist(), exponent.tolist()):
        want_m, want_e = math.frexp(value)
        assert (m, e) == (want_m, want_e), value
        assert math.ldexp(m, e) == value


def test_its_gradient_is_the_power_it_took_off():
    x = mt.Tensor([1e-300, 3.0, 1e300], dtype="float64", requires_grad=True)
    mantissa, exponent = mt.frexp(x)
    mantissa.sum().backward()
    want = [math.ldexp(1.0, -e) for e in exponent.tolist()]
    assert x.grad.tolist() == want
