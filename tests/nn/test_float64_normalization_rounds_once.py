# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""Each output of a float64 normalization is the float nearest its exact value.

Each value less the mean, over the root of the variance plus `eps` -- or, for
`rms_norm`, over the root of the mean square plus `eps` -- then times the
weight and plus the bias. Taken step by step in float64, most outputs of a
float64 `layer_norm`, `rms_norm`, `group_norm` or `instance_norm` came out an
ulp or more off. The references here are exact: the statistics as fractions,
and each output placed against the halfway points either side of it by
comparing squares, so no root is ever rounded.
"""

from __future__ import annotations

import math
from fractions import Fraction

import numpy as np
import pytest

import minitensor as mt
from minitensor import functional as F


def _neighbours(value: float) -> tuple[float, float]:
    return math.nextafter(value, -math.inf), math.nextafter(value, math.inf)


def _standardized(row, eps: Fraction, centred: bool):
    """`(sign, U, D)` for each value of `row`: the standardized value is
    `sign * sqrt(U / D)`."""

    values = [Fraction(float(v)) for v in row]
    n = len(values)
    total = sum(values)
    squares = sum(v * v for v in values)
    if centred:
        divisor = n * squares - total * total + eps * n * n
        return (
            [((n * v - total) > 0) - ((n * v - total) < 0) for v in values],
            [(n * v - total) ** 2 for v in values],
            divisor,
        )
    divisor = squares + eps * n
    return [(v > 0) - (v < 0) for v in values], [n * v * v for v in values], divisor


def _nearest(x, sign: int, square: Fraction, divisor: Fraction, weight=None, bias=None):
    """The float64 nearest `weight * sign * sqrt(square / divisor) + bias`,
    found by walking from a float guess and comparing squares exactly.

    A missing weight or bias is the value that changes nothing, one or
    negative zero. A zero standardized value is the formula's: a zero `x`
    keeps its sign, a value equal to a mean that is not zero gives a positive
    zero; and a zero product makes `w * y + b` exact in float64."""

    weight = 1.0 if weight is None else float(weight)
    bias = -0.0 if bias is None else float(bias)
    if square == 0:
        return (x if x == 0 else 0.0) * weight + bias
    if weight == 0:
        return math.copysign(_nearest(x, 1, square, divisor), sign) * weight + bias
    gain = Fraction(weight) * sign
    b = Fraction(bias)

    def above(m: Fraction) -> int:
        # sign of (gain * sqrt(square / divisor) + b - m)
        v = m - b
        if gain > 0:
            if v <= 0:
                return 1
            difference = gain * gain * square - v * v * divisor
        else:
            if v >= 0:
                return -1
            difference = v * v * divisor - gain * gain * square
        return (difference > 0) - (difference < 0)

    guess = float(gain) * math.sqrt(float(square / divisor)) + float(b)
    candidate = guess
    while True:
        low, high = _neighbours(candidate)
        order = above((Fraction(candidate) + Fraction(high)) / 2)
        if order > 0:
            candidate = high
            continue
        if order == 0:
            found = (
                candidate if (np.float64(candidate).view(np.int64) & 1) == 0 else high
            )
            break
        order = above((Fraction(low) + Fraction(candidate)) / 2)
        if order < 0:
            candidate = low
            continue
        if order == 0:
            found = (
                candidate if (np.float64(candidate).view(np.int64) & 1) == 0 else low
            )
            break
        found = candidate
        break
    if found == 0:
        # A zero result takes the sign of the exact value, and an exact zero,
        # the product cancelling the bias, is the positive one.
        return -0.0 if above(Fraction(0)) < 0 else 0.0
    return found


def _rows(rng, count, width, kind):
    values = rng.standard_normal((count, width))
    if kind == "spread":
        values *= 2.0 ** rng.integers(-20, 20, (count, width))
    elif kind == "offset":
        values += 1e6
    elif kind == "repeated":
        values = np.round(values * 4) / 4
    return values


def _same(got, want, what):
    got = np.asarray(got, dtype=np.float64).ravel()
    want = np.asarray(want, dtype=np.float64).ravel()
    differ = int((got.view(np.int64) != want.view(np.int64)).sum())
    assert differ == 0, f"{what}: {differ} of {got.size} differ"


@pytest.mark.parametrize("kind", ["plain", "spread", "offset", "repeated"])
@pytest.mark.parametrize("eps", [1e-5, 0.0, 3.0])
def test_layer_norm(kind, eps):
    rng = np.random.default_rng(len(kind) * 31 + int(eps * 7))
    x = _rows(rng, 6, 9, kind)
    weight = rng.standard_normal(9)
    bias = rng.standard_normal(9)
    plain = F.layer_norm(mt.tensor(x), (9,), eps=eps).numpy()
    affine = F.layer_norm(
        mt.tensor(x), (9,), mt.tensor(weight), mt.tensor(bias), eps=eps
    ).numpy()
    want_plain, want_affine = [], []
    for row in x:
        signs, squares, divisor = _standardized(row, Fraction(eps), True)
        if divisor == 0:
            want_plain += [math.nan] * len(row)
            want_affine += [math.nan] * len(row)
            continue
        for i, (s, u) in enumerate(zip(signs, squares)):
            want_plain.append(_nearest(row[i], s, u, divisor))
            want_affine.append(_nearest(row[i], s, u, divisor, weight[i], bias[i]))
    if np.isnan(want_plain).any():
        assert np.isnan(plain).ravel()[np.isnan(want_plain)].all()
        return
    _same(plain, want_plain, "layer_norm")
    _same(affine, want_affine, "layer_norm with weight and bias")


@pytest.mark.parametrize("kind", ["plain", "spread", "repeated"])
def test_rms_norm(kind):
    rng = np.random.default_rng(len(kind) * 17)
    x = _rows(rng, 6, 11, kind)
    weight = rng.standard_normal(11)
    plain = F.rms_norm(mt.tensor(x), (11,), eps=1e-6).numpy()
    scaled = F.rms_norm(mt.tensor(x), (11,), mt.tensor(weight), eps=1e-6).numpy()
    want_plain, want_scaled = [], []
    for row in x:
        signs, squares, divisor = _standardized(row, Fraction(1e-6), False)
        for i, (s, u) in enumerate(zip(signs, squares)):
            want_plain.append(_nearest(row[i], s, u, divisor))
            want_scaled.append(_nearest(row[i], s, u, divisor, weight[i]))
    _same(plain, want_plain, "rms_norm")
    _same(scaled, want_scaled, "rms_norm with weight")


def test_group_and_instance_norm_standardize_exactly():
    # Without weight and bias each output is the standardized value itself.
    rng = np.random.default_rng(3)
    x = rng.standard_normal((2, 4, 5)) * 3.0 + 1.5
    grouped = F.group_norm(mt.tensor(x), 2).numpy()
    instance = F.instance_norm(mt.tensor(x)).numpy()
    for n in range(2):
        for g in range(2):
            row = x[n, 2 * g : 2 * g + 2].ravel()
            signs, squares, divisor = _standardized(row, Fraction(1e-5), True)
            want = [_nearest(v, s, u, divisor) for v, s, u in zip(row, signs, squares)]
            _same(grouped[n, 2 * g : 2 * g + 2].ravel(), want, "group_norm")
        for c in range(4):
            signs, squares, divisor = _standardized(x[n, c], Fraction(1e-5), True)
            want = [
                _nearest(v, s, u, divisor) for v, s, u in zip(x[n, c], signs, squares)
            ]
            _same(instance[n, c], want, "instance_norm")


def test_a_constant_row_and_specials():
    # A constant row standardizes to zeros, and NaN where nothing is added
    # under the root; a row holding a NaN or an infinity gives what the
    # formula gives.
    x = np.array([[2.5] * 4, [1.0, np.nan, 2.0, 3.0], [1.0, np.inf, 2.0, 3.0]])
    got = F.layer_norm(mt.tensor(x), (4,), eps=1e-5).numpy()
    assert (got[0] == 0).all() and not np.signbit(got[0]).any()
    assert np.isnan(got[1]).all() and np.isnan(got[2]).all()
    got = F.layer_norm(mt.tensor(x[:1]), (4,), eps=0.0).numpy()
    assert np.isnan(got).all()


def test_the_gradient_still_flows():
    rng = np.random.default_rng(5)
    x = mt.tensor(rng.standard_normal((3, 6)), requires_grad=True)
    w = mt.tensor(rng.standard_normal(6), requires_grad=True)
    b = mt.tensor(rng.standard_normal(6), requires_grad=True)
    assert mt.gradcheck(
        lambda x, w, b: (
            F.layer_norm(x, (6,), w, b) * mt.tensor(np.arange(18.0).reshape(3, 6))
        ).sum(),
        [x, w, b],
    )


@pytest.mark.parametrize("kind", ["plain", "spread", "offset"])
def test_batch_norm_with_batch_statistics(kind):
    rng = np.random.default_rng(len(kind) * 13 + 1)
    x = _rows(rng, 4 * 3, 5, kind).reshape(4, 3, 5)
    weight = rng.standard_normal(3)
    bias = rng.standard_normal(3)
    got = F.batch_norm(
        mt.tensor(x), None, None, mt.tensor(weight), mt.tensor(bias), training=True
    ).numpy()
    plain = F.batch_norm(mt.tensor(x), None, None, training=True).numpy()
    for c in range(3):
        values = x[:, c, :].ravel()
        signs, squares, divisor = _standardized(values, Fraction(1e-5), True)
        want = [
            _nearest(v, s, u, divisor, weight[c], bias[c])
            for v, s, u in zip(values, signs, squares)
        ]
        _same(got[:, c, :].ravel(), want, "batch_norm")
        want = [_nearest(v, s, u, divisor) for v, s, u in zip(values, signs, squares)]
        _same(plain[:, c, :].ravel(), want, "batch_norm without weight and bias")


def _signs(values):
    return [bool(np.signbit(v)) for v in np.asarray(values, dtype=np.float64).ravel()]


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_a_zero_output_is_signed_as_the_formula_signs_it(dtype):
    # A zero input keeps its sign, as `x - 0` and `x * r` do; a value equal
    # to a mean that is not zero gives a positive zero; and a weight and bias
    # sign the zero as `w * y + b` does, a missing one changing nothing.
    def tensor(values):
        return mt.tensor(np.array(values, dtype=dtype))

    x = tensor([[-0.0, 3.0, 4.0, 0.0]])
    assert _signs(F.rms_norm(x, (4,)).numpy()[:, [0, 3]]) == [True, False]
    scaled = F.rms_norm(x, (4,), tensor([-2.0, 1.0, 1.0, 2.0])).numpy()
    assert _signs(scaled[:, [0, 3]]) == [False, False]

    x = tensor([[-0.0, 1.0, -1.0, 0.0], [0.0, 1.0, -1.0, 0.0], [2.0, 1.0, 3.0, 2.0]])
    assert _signs(F.layer_norm(x, (4,)).numpy()[:, 0]) == [True, False, False]
    weight = tensor([-1.0, 1.0, 1.0, 1.0])
    assert _signs(F.layer_norm(x, (4,), weight).numpy()[:, 0]) == [False, True, True]
    shifted = F.layer_norm(x, (4,), weight, tensor([0.0] * 4)).numpy()
    assert _signs(shifted[:, 0]) == [False, False, False]
    shifted = F.layer_norm(x, (4,), weight, tensor([-0.0] * 4)).numpy()
    assert _signs(shifted[:, 0]) == [False, True, True]

    grouped = F.group_norm(x.reshape(1, 3, 4), 3).numpy().reshape(3, 4)
    assert _signs(grouped[:, 0]) == [True, False, False]

    x = tensor([-0.0, 1.0, -1.0, 0.0]).reshape(4, 1, 1)
    plain = F.batch_norm(x, None, None, training=True).numpy().ravel()
    assert _signs(plain[[0, 3]]) == [True, False]
    shifted = F.batch_norm(x, None, None, None, tensor([0.0]), training=True)
    assert _signs(shifted.numpy().ravel()[[0, 3]]) == [False, False]
    running = F.batch_norm(x, tensor([0.0]), tensor([1.0]), training=False)
    assert _signs(running.numpy().ravel()[[0, 3]]) == [True, False]


def test_a_value_too_small_for_any_float64_rounds_to_the_zero_on_its_side():
    # The first value standardizes to -1/3 exactly, and a third of the
    # smallest float64 rounds to zero -- the zero below, its sign the exact
    # value's, where step by step a positive bias turned it positive.
    x = mt.tensor(np.array([[0.0] * 9 + [1.0]]))
    smallest = mt.tensor(np.full(10, 5e-324))
    for bias in (None, mt.tensor(np.zeros(10))):
        got = F.layer_norm(x, (10,), smallest, bias, eps=0.0).numpy()
        assert got[0, 0] == 0 and np.signbit(got[0, 0])
