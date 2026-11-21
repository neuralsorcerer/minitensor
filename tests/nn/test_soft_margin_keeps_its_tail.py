# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""`soft_margin_loss`, `log(1 + e^z)` for `z = -y x`, and its slope past 20.

The loss was `softplus(z)` with softplus's threshold at 20, past which it is
`z` and its slope one: that leaves out `log1p(e^-z)`, 5.7e5 float64 ulps of
the loss just past 20 and 1.8e7 of the slope. It now takes softplus's stable
form everywhere.
"""

from __future__ import annotations

import decimal
import math
from decimal import Decimal

import numpy as np

import minitensor as mt
from minitensor import functional as F


def _loss_and_slope(x: float, y: float):
    with decimal.localcontext(prec=50, Emax=10**6, Emin=-(10**6)):
        z = -Decimal(y) * Decimal(x)
        e = z.exp()
        return float((1 + e).ln()), float(-Decimal(y) * e / (1 + e))


def test_float64_loss_and_slope_past_the_old_threshold():
    rng = np.random.default_rng(22)
    xs = np.concatenate([rng.uniform(-60.0, 60.0, 600), -rng.uniform(19.0, 40.0, 400)])
    ys = rng.choice([-1.0, 1.0], len(xs))
    t = mt.Tensor(xs, dtype="float64", requires_grad=True)
    loss = F.soft_margin_loss(t, mt.Tensor(ys, dtype="float64"), reduction="none")
    values = loss.numpy().tolist()
    loss.sum().backward()
    slopes = t.grad.numpy().tolist()
    mt.clear_autograd_graph()
    for x, y, v, g in zip(xs.tolist(), ys.tolist(), values, slopes):
        want_value, want_slope = _loss_and_slope(x, y)
        assert abs(v - want_value) <= 2 * math.ulp(want_value), (x, y, v, want_value)
        assert abs(g - want_slope) <= 2 * math.ulp(want_slope), (x, y, g, want_slope)
