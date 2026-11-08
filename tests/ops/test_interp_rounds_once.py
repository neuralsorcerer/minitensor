# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""`interp` is the float64 nearest the exact interpolation.

`y_lo + slope * (x - x_lo)` rounded the slope, the offset, their product
and the sum, and of random samples and queries about two values in five came
out other than the float64 nearest the exact value. With a period, the
queries and the samples were wrapped by a rounded quotient and floor, which
for a query far from zero could land anywhere. Each value is that float64
now, the wrap exact; a query on a sample is the last value at its position,
and with one sample `left` and `right` still apply beside it.
"""

from __future__ import annotations

import bisect
import math
from fractions import Fraction

import numpy as np
import pytest

import minitensor as mt


def _want(x, xp, fp, left=None, right=None) -> list[float]:
    """The exact interpolation of finite samples at each query, rounded once."""

    places = [Fraction(float(p)) for p in xp]
    samples = [Fraction(float(v)) for v in fp]
    out = []
    for q in np.ravel(x).tolist():
        if math.isnan(q):
            out.append(math.nan)
        elif q < float(xp[0]):
            out.append(float(samples[0]) if left is None else float(left))
        elif q > float(xp[-1]):
            out.append(float(samples[-1]) if right is None else float(right))
        else:
            exact = Fraction(q)
            upper = bisect.bisect_right(places, exact)
            if upper == len(places):
                out.append(float(samples[-1]))
                continue
            lower = upper - 1
            out.append(float(_between(exact, places, samples, lower, upper)))
    return out


def _between(exact, places, samples, lower, upper) -> Fraction:
    if places[lower] == exact:
        return samples[lower]
    width = places[upper] - places[lower]
    return (
        samples[lower] * (places[upper] - exact)
        + samples[upper] * (exact - places[lower])
    ) / width


def _want_periodic(x, xp, fp, period: float) -> list[float]:
    """The exact interpolation on a circle, the wraps exact too."""

    span = Fraction(abs(period))
    wrapped = sorted(
        ((Fraction(float(p)) % span, i) for i, p in enumerate(xp)),
        key=lambda pair: pair[0],
    )
    places = [wrapped[-1][0] - span] + [w for w, _ in wrapped] + [wrapped[0][0] + span]
    owners = [wrapped[-1][1]] + [i for _, i in wrapped] + [wrapped[0][1]]
    samples = [Fraction(float(fp[i])) for i in owners]
    out = []
    for q in np.ravel(x).tolist():
        if not math.isfinite(q):
            out.append(math.nan)
            continue
        exact = Fraction(q) % span
        upper = bisect.bisect_right(places, exact)
        out.append(float(_between(exact, places, samples, upper - 1, upper)))
    return out


def _same(got, want) -> bool:
    got = np.ravel(got).tolist()
    return len(got) == len(want) and all(
        (math.isnan(a) and math.isnan(b)) or a == b for a, b in zip(got, want)
    )


def _samples(rng, m: int, spread: float = 2.0):
    xp = np.sort(rng.normal(size=m) * 10.0 ** rng.uniform(-spread, spread, m))
    fp = rng.normal(size=m) * 10.0 ** rng.uniform(-spread, spread, m)
    return xp, fp


def test_random_samples_and_queries():
    rng = np.random.default_rng(773)
    for trial in range(200):
        m = int(rng.integers(2, 30))
        xp, fp = _samples(rng, m)
        x = rng.uniform(xp[0] - 1.0, xp[-1] + 1.0, 25)
        got = mt.interp(mt.Tensor(x), mt.Tensor(xp), mt.Tensor(fp))
        assert str(got.dtype) == "float64"
        assert _same(got.tolist(), _want(x, xp, fp)), trial


def test_wide_and_close_positions():
    rng = np.random.default_rng(787)
    for trial in range(100):
        m = int(rng.integers(2, 12))
        # Positions far from zero and close together, and values of every size.
        base = 10.0 ** rng.uniform(0, 12)
        xp = base + np.cumsum(rng.uniform(1e-6, 1.0, m))
        fp = rng.normal(size=m) * 10.0 ** rng.integers(-150, 150, m)
        x = rng.uniform(xp[0], xp[-1], 20)
        got = mt.interp(mt.Tensor(x), mt.Tensor(xp), mt.Tensor(fp)).tolist()
        assert _same(got, _want(x, xp, fp)), trial
    # Products of the positions and the values past the largest float.
    big = float(np.finfo(np.float64).max) / 2
    got = mt.interp(
        mt.tensor([0.0, big / 3], dtype="float64"), [-big, big], [-big, big]
    ).tolist()
    assert got == _want([0.0, big / 3], [-big, big], [-big, big])


def test_on_the_samples_and_at_one_position():
    xp = [0.0, 1.0, 1.0, 2.0, 3.0, 3.0]
    fp = [5.0, 1.0, 7.0, -2.0, 4.0, 9.0]
    queries = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]
    got = mt.interp(mt.tensor(queries), xp, fp).tolist()
    # The last value at a position, from either side.
    assert got == [5.0, 3.0, 7.0, 2.5, -2.0, 9.0]
    assert got == _want(queries, xp, fp)


def test_outside_the_range_and_one_sample():
    xp, fp = [1.0, 2.0, 4.0], [3.0, -1.0, 5.0]
    queries = [-math.inf, 0.0, 1.0, 4.0, 5.0, math.inf, math.nan]
    got = mt.interp(mt.tensor(queries), xp, fp).tolist()
    assert _same(got, [3.0, 3.0, 3.0, 5.0, 5.0, 5.0, math.nan])
    got = mt.interp(mt.tensor(queries), xp, fp, left=-9.0, right=9.0).tolist()
    assert _same(got, [-9.0, -9.0, 3.0, 5.0, 9.0, 9.0, math.nan])
    # One sample is its value on it, and `left` and `right` beside it.
    got = mt.interp(mt.tensor([0.0, 2.0, 5.0, math.nan]), [2.0], [7.0]).tolist()
    assert _same(got, [7.0, 7.0, 7.0, math.nan])
    got = mt.interp(mt.tensor([0.0, 2.0, 5.0]), [2.0], [7.0], left=-1.0, right=1.0)
    assert got.tolist() == [-1.0, 7.0, 1.0]


def test_values_that_are_not_finite():
    inf = math.inf
    got = mt.interp(
        mt.tensor([0.5, 1.5, 2.5]), [0.0, 1.0, 2.0, 3.0], [inf, inf, 1.0, -inf]
    )
    assert got.tolist() == [inf, inf, -inf]
    got = mt.interp(mt.tensor([0.5, 1.0]), [0.0, 1.0, 2.0], [inf, -inf, 2.0]).tolist()
    assert math.isnan(got[0]) and got[1] == -inf


@pytest.mark.parametrize("period", [4.0, -4.0, 0.7, 360.0])
def test_a_period(period):
    rng = np.random.default_rng(797)
    for trial in range(60):
        m = int(rng.integers(1, 12))
        xp = rng.uniform(-3 * abs(period), 3 * abs(period), m)
        fp = rng.normal(size=m) * 10.0 ** rng.uniform(-2, 2, m)
        x = np.concatenate(
            [
                rng.uniform(-5 * abs(period), 5 * abs(period), 20),
                rng.normal(size=4) * 1e15,
                [-1e-20, 0.0, abs(period), -abs(period), float(xp[0])],
            ]
        )
        got = mt.interp(mt.Tensor(x), mt.Tensor(xp), mt.Tensor(fp), period=period)
        assert _same(got.tolist(), _want_periodic(x, xp, fp, period)), trial


def test_a_period_far_from_zero_and_at_one_position():
    # 2^60 + 2560 is 2 more than a multiple of 3.
    got = mt.interp(
        mt.tensor([2.0**60 + 2560], dtype="float64"), [0.0, 1.0], [0.0, 1.0], period=3.0
    )
    assert got.tolist() == [0.5]
    # Samples a whole period apart are at one position: on it the last of
    # them, and halfway round, halfway from the last of them a period back to
    # the first.
    got = mt.interp(
        mt.tensor([1.0, 2.0]), [1.0, 3.0, 5.0], [10.0, 20.0, 30.0], period=2
    )
    assert got.tolist() == [30.0, 20.0]
    assert math.isnan(mt.interp(mt.tensor([math.inf]), [0.0], [1.0], period=1).item())
    with pytest.raises(ValueError):
        mt.interp(mt.tensor([0.0]), [0.0], [1.0], period=math.inf)
    with pytest.raises(ValueError):
        mt.interp(mt.tensor([0.0]), [0.0], [1.0], period=0.0)


def test_python_numbers_are_read_at_float64():
    # 0.1 and 0.3 as float64s, not first as the default float32s.
    got = mt.interp(0.2, [0.1, 0.3], [1.0, 3.0]).item()
    assert got == _want([0.2], [0.1, 0.3], [1.0, 3.0])[0]
    got = mt.interp([0.1, 0.25], mt.tensor([0.0, 1.0], dtype="float64"), [0.0, 0.1])
    assert got.tolist() == _want([0.1, 0.25], [0.0, 1.0], [0.0, 0.1])


def test_shapes_and_dtypes():
    xp, fp = [0.0, 1.0, 3.0], [0.0, 1.0, -1.0]
    query = np.array([[0.1, 0.7], [1.3, 2.9]], dtype=np.float32)
    got = mt.interp(mt.Tensor(query, dtype="float32"), xp, fp)
    assert str(got.dtype) == "float64" and tuple(got.shape) == (2, 2)
    assert got.tolist() == np.array(_want(query, xp, fp)).reshape(2, 2).tolist()
    got = mt.interp(mt.tensor([1, 2], dtype="int64"), [0, 4], [0, 1])
    assert got.tolist() == [0.25, 0.5]


def test_gradients():
    xp = mt.tensor([0.0, 1.0, 3.0], dtype="float64", requires_grad=True)
    fp = mt.tensor([2.0, 4.0, -2.0], dtype="float64", requires_grad=True)
    x = mt.tensor([-1.0, 0.5, 2.0, 5.0], dtype="float64", requires_grad=True)
    weights = mt.tensor([1.0, 2.0, 3.0, 4.0], dtype="float64")
    (mt.interp(x, xp, fp) * weights).sum().backward()
    # Held at the ends beyond them; the slopes 2 and -3 inside.
    assert x.grad.tolist() == [0.0, 4.0, -9.0, 0.0]
    # 0.5 shares itself between the first two values, 2 between the last
    # two, and the ends take what lies beyond them.
    assert fp.grad.tolist() == [1.0 + 1.0, 1.0 + 1.5, 1.5 + 4.0]
    # Moving a position moves its segments: -slope times each share.
    assert xp.grad.tolist() == [-2.0 * 1.0, -2.0 * 1.0 + 3.0 * 1.5, 3.0 * 1.5]
    # With `left` and `right` given, the ends take nothing beyond them.
    fp = mt.tensor([2.0, 4.0, -2.0], dtype="float64", requires_grad=True)
    (
        mt.interp(x.detach(), xp.detach(), fp, left=0.0, right=0.0) * weights
    ).sum().backward()
    assert fp.grad.tolist() == [1.0, 2.5, 1.5]


def test_gradients_match_central_differences_with_a_period():
    rng = np.random.default_rng(809)
    base_x = rng.uniform(-10, 10, 7)
    base_xp = rng.uniform(0, 4, 5)
    base_fp = rng.normal(size=5)
    weights = rng.normal(size=7)

    def total(x, xp, fp):
        got = mt.interp(mt.Tensor(x), mt.Tensor(xp), mt.Tensor(fp), period=4.0)
        return float((got.numpy() * weights).sum())

    tensors = [
        mt.Tensor(a.copy(), requires_grad=True) for a in (base_x, base_xp, base_fp)
    ]
    (mt.interp(*tensors, period=4.0) * mt.Tensor(weights)).sum().backward()
    eps = 1e-6
    for which, tensor in enumerate(tensors):
        for index in range(tensor.shape[0]):
            up = [base_x.copy(), base_xp.copy(), base_fp.copy()]
            down = [base_x.copy(), base_xp.copy(), base_fp.copy()]
            up[which][index] += eps
            down[which][index] -= eps
            numeric = (total(*up) - total(*down)) / (2 * eps)
            assert tensor.grad.tolist()[index] == pytest.approx(
                numeric, rel=1e-5, abs=1e-7
            )
