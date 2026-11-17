# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""`binary_cross_entropy_with_logits` keeps its digits where the loss is small.

The loss of a logit `x` against a target `t` was formed as
`(1 - t) x + softplus(-x)`. Where the logit is negative and the target near
zero, the first term is near `x` and the second near `-x`, and their sum --
the small loss of a confident, correct prediction -- came out with the error
of a number the size of `x`: at a logit of -20 against a target of 0 the loss
is 2.06e-9 and was some 10^9 ulps out. It is now `k |x| + c log1p(exp(-|x|))`,
the same number written as two terms that do not cancel, and a float32 loss
is taken in float64 and narrowed once.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import minitensor as mt
from minitensor import functional as F


def _reference(x: float, t: float, w: float = 1.0) -> float:
    """The loss in float64 from the two terms that do not cancel, with a
    `log1p` of an `exp` that is never of a positive number."""

    magnitude = abs(x)
    tail = math.log1p(math.exp(-magnitude))
    k = (1.0 - t) if x >= 0 else w * t
    c = (1.0 - t) + w * t
    return k * magnitude + c * tail


def test_a_confident_correct_negative_keeps_its_digits():
    x = mt.Tensor([-20.0, -35.0, -700.0], dtype="float64")
    t = mt.Tensor([0.0, 0.0, 0.0], dtype="float64")
    got = F.binary_cross_entropy_with_logits(x, t, reduction="none").tolist()
    # `log1p(exp(x))`, the loss itself, to the last digit.
    want = [math.log1p(math.exp(v)) for v in (-20.0, -35.0, -700.0)]
    assert got == want
    narrow = F.binary_cross_entropy_with_logits(
        mt.Tensor([-20.0, -40.0], dtype="float32"),
        mt.Tensor([0.0, 0.0], dtype="float32"),
        reduction="none",
    ).tolist()
    assert narrow == [
        float(np.float32(math.log1p(math.exp(-20.0)))),
        float(np.float32(math.log1p(math.exp(-40.0)))),
    ]


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("weighted", [False, True])
def test_every_term_is_the_two_term_form(dtype, weighted):
    rng = np.random.default_rng(1129 + weighted)
    x = (rng.standard_normal(3000) * 20).astype(dtype)
    t = np.where(rng.uniform(size=3000) < 0.3, 0.0, rng.uniform(size=3000))
    t = t.astype(dtype)
    w = rng.uniform(0.2, 3.0, 3000).astype(dtype)
    args = [mt.Tensor(x, dtype=dtype), mt.Tensor(t, dtype=dtype)]
    if weighted:
        args.append(mt.Tensor(w, dtype=dtype))
    got = F.binary_cross_entropy_with_logits(*args, reduction="none").numpy()
    want = np.array(
        [
            _reference(float(a), float(b), float(c) if weighted else 1.0)
            for a, b, c in zip(x, t, w)
        ]
    )
    if dtype == "float32":
        # Taken in float64 and narrowed once.
        np.testing.assert_array_equal(got, want.astype(np.float32))
    else:
        np.testing.assert_allclose(got, want, rtol=4e-16, atol=0)


def test_the_ends():
    inf = math.inf
    for dtype in ("float32", "float64"):
        x = mt.Tensor([inf, inf, -inf, -inf, math.nan, 0.0], dtype=dtype)
        t = mt.Tensor([1.0, 0.25, 0.0, 0.5, 0.5, 0.5], dtype=dtype)
        got = F.binary_cross_entropy_with_logits(x, t, reduction="none").tolist()
        assert got[:4] == [0.0, inf, 0.0, inf]
        assert math.isnan(got[4])
        assert got[5] == float(np.array(math.log(2.0), dtype=dtype))


def test_the_gradient_is_unchanged():
    rng = np.random.default_rng(1151)
    x0 = rng.standard_normal(40) * 5
    t0 = rng.uniform(size=40)
    x = mt.Tensor(x0, requires_grad=True)
    F.binary_cross_entropy_with_logits(x, mt.Tensor(t0), reduction="sum").backward()
    want = 1.0 / (1.0 + np.exp(-x0)) - t0
    np.testing.assert_allclose(x.grad.numpy(), want, rtol=1e-13, atol=1e-16)
