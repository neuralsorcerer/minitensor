# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""The regression losses take an integer target as the integer it is.

An int32 or int64 target was cast to the predictions' float dtype before
the loss, which rounded every target past 2^24 for float32 predictions, and
past 2^53 for float64 ones: a prediction of `2^24` against a target of
`2^24 + 1` lost nothing. `mse_loss` and `l1_loss` are now of the exact
differences, rounded once, and the losses of the difference alone --
`huber_loss`, `smooth_l1_loss` and `log_cosh_loss` -- of the difference
rounded once from the exact one, as `predictions - targets` gives it. A
Python int or list the predictions' dtype does not hold is read exactly too,
as arithmetic reads it.
"""

from __future__ import annotations

import math
from fractions import Fraction

import numpy as np
import pytest

import minitensor as mt
from minitensor import nn


def _float32(value: Fraction) -> float:
    """The float32 nearest `value`, ties to even."""

    if value == 0:
        return 0.0
    sign = -1.0 if value < 0 else 1.0
    magnitude = abs(value)
    exponent = magnitude.numerator.bit_length() - magnitude.denominator.bit_length()
    if Fraction(2) ** exponent > magnitude:
        exponent -= 1
    unit = Fraction(2) ** max(exponent - 23, -149)
    rounded = round(magnitude / unit) * unit
    return sign * (math.inf if rounded >= Fraction(2) ** 128 else float(rounded))


def _rounded(value: Fraction, dtype: str) -> float:
    return _float32(value) if dtype == "float32" else float(value)


def _want(p, t, squared: bool, reduction: str, dtype: str):
    differences = [Fraction(float(a)) - Fraction(int(b)) for a, b in zip(p, t)]
    terms = [d * d if squared else abs(d) for d in differences]
    if reduction == "none":
        return [_rounded(term, dtype) for term in terms]
    total = sum(terms, Fraction(0))
    return _rounded(total / len(terms) if reduction == "mean" else total, dtype)


def _draw(rng, dtype: str, target_dtype: str, n: int = 40):
    """Predictions beside integer targets near them, where the targets are
    past what the predictions' dtype holds."""

    if dtype == "float32":
        scale = 2**26 if target_dtype == "int64" else 2**25
    else:
        scale = 2**58 if target_dtype == "int64" else 2**30
    targets = rng.integers(-scale, scale, n, dtype=np.int64)
    predictions = (targets + rng.integers(-8, 9, n)).astype(dtype)
    return predictions, targets.astype(target_dtype)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("target_dtype", ["int32", "int64"])
@pytest.mark.parametrize("squared", [True, False], ids=["mse", "l1"])
def test_mse_and_l1_of_exact_integers(dtype, target_dtype, squared):
    rng = np.random.default_rng(653)
    loss = nn.mse_loss if squared else nn.l1_loss
    for trial in range(10):
        p, t = _draw(rng, dtype, target_dtype)
        tensors = mt.Tensor(p, dtype=dtype), mt.Tensor(t, dtype=target_dtype)
        for reduction in ("mean", "sum", "none"):
            got = loss(*tensors, reduction)
            assert str(got.dtype) == dtype
            got = got.tolist() if reduction == "none" else got.item()
            assert got == _want(p, t, squared, reduction, dtype), (trial, reduction)


def test_targets_past_float64():
    # int64 targets past 2^53, which float64 rounds, beside float64
    # predictions: a whole number of units apart, and some at int64's ends.
    rng = np.random.default_rng(659)
    t = rng.integers(2**60, 2**62, 50, dtype=np.int64)
    t[:3] = [np.iinfo(np.int64).max, np.iinfo(np.int64).min, 2**53 + 1]
    p = t.astype(np.float64)
    for squared in (True, False):
        loss = nn.mse_loss if squared else nn.l1_loss
        tensors = mt.Tensor(p, dtype="float64"), mt.Tensor(t, dtype="int64")
        for reduction in ("mean", "sum", "none"):
            got = loss(*tensors, reduction)
            got = got.tolist() if reduction == "none" else got.item()
            assert got == _want(p, t, squared, reduction, "float64"), reduction


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_losses_of_the_difference(dtype):
    # Each is a function of the difference alone, so against an integer
    # target it is the loss of the difference arithmetic gives, rounded once
    # from the exact one, against zero.
    rng = np.random.default_rng(661)
    p, t = _draw(rng, dtype, "int64")
    predictions, targets = mt.Tensor(p, dtype=dtype), mt.Tensor(t, dtype="int64")
    difference = predictions - targets
    zeros = mt.Tensor(np.zeros(len(p), dtype=dtype), dtype=dtype)
    for loss in (
        lambda a, b, r: nn.huber_loss(a, b, r, delta=2.0),
        lambda a, b, r: nn.smooth_l1_loss(a, b, r, beta=3.0),
        nn.log_cosh_loss,
    ):
        for reduction in ("mean", "sum", "none"):
            got = loss(predictions, targets, reduction).tolist()
            assert got == loss(difference, zeros, reduction).tolist(), reduction
    # The difference is the exact one, so a target one past a prediction
    # loses one.
    got = nn.huber_loss(
        mt.tensor([2.0**24], dtype="float32"), mt.tensor([2**24 + 1], dtype="int64")
    )
    assert got.item() == 0.5


def test_python_integers():
    x = mt.tensor([2.0**24, 0.0], dtype="float32")
    target = [2**24 + 1, 3]
    assert nn.mse_loss(x, target).item() == 5.0
    assert nn.l1_loss(x, target, "none").tolist() == [1.0, 3.0]
    assert nn.huber_loss(x, target, "sum").item() == 0.5 + 2.5
    assert nn.log_cosh_loss(x, target).dtype == x.dtype
    # As arithmetic reads them.
    assert (x - target).tolist() == [-1.0, -3.0]
    x = mt.tensor([2.0**60], dtype="float64")
    assert nn.mse_loss(x, [2**60 + 3]).item() == 9.0


@pytest.mark.parametrize("squared", [True, False], ids=["mse", "l1"])
def test_gradients(squared):
    x = mt.tensor([2.0**24, 1.5], dtype="float32", requires_grad=True)
    t = mt.tensor([2**24 + 1, 2], dtype="int64")
    loss = nn.mse_loss if squared else nn.l1_loss
    loss(x, t).backward()
    want = [-1.0, -0.5] if squared else [-0.5, -0.5]
    assert x.grad.tolist() == want
    assert str(x.grad.dtype) == "float32"
