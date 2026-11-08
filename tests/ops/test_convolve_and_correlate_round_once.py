# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""`convolve` and `correlate` give each overlap as the float nearest its exact
dot.

Each overlap's products were added in floats, and of random signals about
one overlap in four came out other than the float nearest the exact sum.
Each is that float now, in every mode, for a signal longer or shorter than
its kernel, and the gradient is the exact products'.
"""

from __future__ import annotations

import math
from fractions import Fraction

import numpy as np
import pytest

import minitensor as mt


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


def _want(a, v, mode: str, flip: bool, dtype: str) -> list[float]:
    """The exact sliding products of the two sequences, each rounded once."""

    exact_a = np.array([Fraction(float(x)) for x in a], dtype=object)
    exact_v = np.array([Fraction(float(x)) for x in v], dtype=object)
    product = np.convolve if flip else np.correlate
    rounded = _float32 if dtype == "float32" else float
    return [rounded(value) for value in product(exact_a, exact_v, mode)]


def _draw(rng, n: int, dtype: str):
    return (rng.normal(size=n) * 10.0 ** rng.uniform(-3, 3, n)).astype(dtype)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("mode", ["full", "same", "valid"])
@pytest.mark.parametrize("name, flip", [("correlate", False), ("convolve", True)])
def test_random_sequences(dtype, mode, name, flip):
    rng = np.random.default_rng(821)
    function = getattr(mt, name)
    for trial in range(80):
        a = _draw(rng, int(rng.integers(1, 30)), dtype)
        v = _draw(rng, int(rng.integers(1, 12)), dtype)
        if trial % 4 == 0:
            # The kernel longer than the signal.
            a, v = v, a
        got = function(mt.Tensor(a, dtype=dtype), mt.Tensor(v, dtype=dtype), mode)
        assert str(got.dtype) == dtype
        assert got.tolist() == _want(a, v, mode, flip, dtype), (trial, len(a), len(v))


def test_cancellation_and_range():
    tiny = 2.0**-53
    # `1 + 2^-53 + 2^-53`: a plain sum drops both small products.
    got = mt.correlate(mt.tensor([1.0, tiny, tiny], dtype="float64"), [1.0, 1.0, 1.0])
    assert got.tolist() == [1.0 + 2.0**-52]
    # Products past the largest float, in an overlap that is not.
    big = float(np.finfo(np.float64).max) / 2
    got = mt.convolve(mt.tensor([big, big], dtype="float64"), [4.0, -4.0])
    assert got.tolist() == [math.inf, 0.0, -math.inf]
    got = mt.correlate(mt.tensor([big, 3.0], dtype="float64"), [4.0, -4.0], "valid")
    assert got.tolist() == [math.inf]
    got = mt.correlate(
        mt.tensor([big, big, 1.0], dtype="float64"), [2.0, -2.0, 1.0], "valid"
    )
    assert got.tolist() == [1.0]


def test_mixed_dtypes_integers_and_long_signals():
    rng = np.random.default_rng(827)
    a = _draw(rng, 40, "float32")
    v = _draw(rng, 7, "float64")
    got = mt.convolve(mt.Tensor(a, dtype="float32"), mt.Tensor(v, dtype="float64"))
    assert str(got.dtype) == "float64"
    assert got.tolist() == _want(a.astype(np.float64), v, "full", True, "float64")
    got = mt.correlate(mt.tensor([1, 2, 3]), mt.tensor([4, 5]), "full")
    assert str(got.dtype) == "float64" and got.tolist() == [5.0, 14.0, 23.0, 12.0]
    # Long enough to share the overlaps across the pool.
    for dtype in ("float32", "float64"):
        a, v = _draw(rng, 30_000, dtype), _draw(rng, 33, dtype)
        got = mt.convolve(mt.Tensor(a, dtype=dtype), mt.Tensor(v, dtype=dtype), "same")
        assert got.tolist() == _want(a, v, "same", True, dtype), dtype
        a, v = _draw(rng, 20_000, dtype), _draw(rng, 20_000, dtype)
        got = mt.correlate(mt.Tensor(a, dtype=dtype), mt.Tensor(v, dtype=dtype))
        assert got.tolist() == _want(a, v, "valid", False, dtype), dtype


@pytest.mark.parametrize("mode", ["full", "same", "valid"])
@pytest.mark.parametrize("name", ["correlate", "convolve"])
def test_gradients_match_central_differences(mode, name):
    rng = np.random.default_rng(839)
    function = getattr(mt, name)
    for lengths in ((7, 3), (3, 7), (5, 5)):
        base = [rng.normal(size=n) for n in lengths]
        count = len(function(*(mt.Tensor(b) for b in base), mode).tolist())
        weights = rng.normal(size=count)
        tensors = [mt.Tensor(b.copy(), requires_grad=True) for b in base]
        (function(*tensors, mode) * mt.Tensor(weights)).sum().backward()

        def total(a, v):
            got = function(mt.Tensor(a), mt.Tensor(v), mode).numpy()
            return float((got * weights).sum())

        eps = 1e-6
        for which, tensor in enumerate(tensors):
            for index in range(len(base[which])):
                up = [b.copy() for b in base]
                down = [b.copy() for b in base]
                up[which][index] += eps
                down[which][index] -= eps
                numeric = (total(*up) - total(*down)) / (2 * eps)
                assert tensor.grad.tolist()[index] == pytest.approx(
                    numeric, rel=1e-6, abs=1e-9
                ), (lengths, which, index)
