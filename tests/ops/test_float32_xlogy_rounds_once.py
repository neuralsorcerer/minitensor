# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""A float32 `xlogy` is the float32 nearest `x * log(y)`.

It took the logarithm in float32 and the product in float32, two roundings,
and a third of its answers were an ulp out. It is now taken in float64 and
narrowed once, as float32's other compositions of a logarithm are.
"""

from __future__ import annotations

import math

import numpy as np

import minitensor as mt


def test_float32_xlogy_is_the_float64_one_narrowed():
    rng = np.random.default_rng(1123)
    x = (rng.standard_normal(4000) * 10.0 ** rng.uniform(-3, 3, 4000)).astype(
        np.float32
    )
    y = np.abs(rng.standard_normal(4000) * 10.0 ** rng.uniform(-3, 3, 4000)).astype(
        np.float32
    )
    got = mt.xlogy(mt.Tensor(x, dtype="float32"), mt.Tensor(y, dtype="float32"))
    want = [float(np.float32(float(a) * math.log(float(b)))) for a, b in zip(x, y)]
    assert got.tolist() == want


def test_the_zero_limit_and_the_ends():
    x = mt.Tensor([0.0, 0.0, 2.0, -3.0, 1.0], dtype="float32")
    y = mt.Tensor([0.0, math.inf, 0.0, 0.0, math.nan], dtype="float32")
    got = mt.xlogy(x, y).tolist()
    assert got[:4] == [0.0, 0.0, -math.inf, math.inf]
    assert math.isnan(got[4])
