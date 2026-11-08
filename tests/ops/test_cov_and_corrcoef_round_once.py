# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""`cov` and `corrcoef` are the floats nearest the exact matrices.

Each entry was centred by a rounded mean, multiplied out in a matrix product
and divided, a rounding at every step, and about two in three entries of the
covariance and the correlation matrices of random rows were an ulp or more
from the exact one -- in float32, rounded from float64, as in float64. With
weights `w` totalling `W`, each covariance is now the exact
`(W Σ w x y - Σ w x Σ w y) / D` rounded once, and each correlation the exact
`n_xy / sqrt(n_xx n_yy)` of those numerators.
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


def _rounded(value: Fraction, narrow: bool) -> float:
    if narrow:
        return _float32(value)
    try:
        return float(value)
    except OverflowError:
        return math.inf if value > 0 else -math.inf


def _root(square: Fraction, narrow: bool) -> float:
    """The float nearest the root of `square`, bracketed until it is
    known, or exact where `square` is the square of a fraction."""

    top, bottom = square.numerator, square.denominator
    if math.isqrt(top) ** 2 == top and math.isqrt(bottom) ** 2 == bottom:
        return _rounded(Fraction(math.isqrt(top), math.isqrt(bottom)), narrow)
    scale = (top.bit_length() - bottom.bit_length()) // 2
    extra = 64
    while True:
        bits = extra - scale
        root = math.isqrt(top * 4**bits // bottom)
        low = _rounded(Fraction(root, 2**bits), narrow)
        if low == _rounded(Fraction(root + 1, 2**bits), narrow):
            return low
        extra *= 2


def _exact(rows, frequencies=None, reliabilities=None, correction=1):
    """Every pair's exact numerator and the exact divisor."""

    def exact(value):
        if isinstance(value, (int, np.integer)):
            return Fraction(int(value))
        return Fraction(float(value))

    xs = [[exact(value) for value in row] for row in rows]
    weights = [Fraction(1)] * len(xs[0])
    if frequencies is not None:
        weights = [w * Fraction(float(f)) for w, f in zip(weights, frequencies)]
    if reliabilities is not None:
        weights = [w * Fraction(float(a)) for w, a in zip(weights, reliabilities)]
    total = sum(weights)
    sums = [sum(w * x for w, x in zip(weights, row)) for row in xs]
    numerators = [
        [
            total * sum(w * x * y for w, x, y in zip(weights, a, b)) - sa * sb
            for b, sb in zip(xs, sums)
        ]
        for a, sa in zip(xs, sums)
    ]
    if reliabilities is None:
        divisor = total * (total - correction)
    else:
        reliable = sum(w * Fraction(float(a)) for w, a in zip(weights, reliabilities))
        divisor = total * total - correction * reliable
    return numerators, divisor


def _want_cov(rows, narrow, **weights):
    numerators, divisor = _exact(rows, **weights)
    return [[_rounded(n / divisor, narrow) for n in row] for row in numerators]


def _want_corrcoef(rows, narrow):
    numerators, _ = _exact(rows)
    out = []
    for i, row in enumerate(numerators):
        out.append([])
        for j, n in enumerate(row):
            own = numerators[i][i] * numerators[j][j]
            if own == 0:
                out[-1].append(math.nan)
            elif n == 0:
                out[-1].append(0.0)
            else:
                root = _root(n * n / own, narrow)
                out[-1].append(-root if n < 0 else root)
    return out


def _same(got, want) -> bool:
    flat_got, flat_want = np.ravel(got).tolist(), np.ravel(want).tolist()
    return len(flat_got) == len(flat_want) and all(
        (math.isnan(x) and math.isnan(y)) or x == y for x, y in zip(flat_got, flat_want)
    )


def _draw(rng, shape, kind, dtype):
    rows = rng.normal(size=shape)
    if kind == "offset":
        # A mean far above the spread, where centring by a rounded mean
        # loses the most.
        rows = rows + 1e4
    elif kind == "scales":
        rows = rows * 2.0 ** rng.integers(-30, 31, size=(shape[0], 1))
    elif kind == "past the range":
        # Rows whose products, or sums of them, pass an end of the range.
        ends = [-560, 520] if dtype == "float64" else [-70, 60]
        rows = rows * 2.0 ** rng.choice(ends, size=(shape[0], 1))
    return rows.astype(dtype)


KINDS = ["normal", "offset", "scales", "past the range"]


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("kind", KINDS)
def test_covariance_and_correlation(dtype, kind):
    rng = np.random.default_rng(613)
    narrow = dtype == "float32"
    for trial in range(6):
        shape = (int(rng.integers(1, 5)), int(rng.integers(2, 30)))
        rows = _draw(rng, shape, kind, dtype)
        tensor = mt.Tensor(rows, dtype=dtype)
        for correction in (0, 1):
            got = mt.cov(tensor, correction).reshape([shape[0], shape[0]]).tolist()
            assert _same(got, _want_cov(rows, narrow, correction=correction)), trial
        got = mt.corrcoef(tensor).reshape([shape[0], shape[0]]).tolist()
        assert _same(got, _want_corrcoef(rows, narrow)), trial


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_weights(dtype):
    rng = np.random.default_rng(617)
    rows = (rng.normal(size=(3, 25)) + 50.0).astype(dtype)
    counts = rng.integers(1, 5, 25)
    reliability = rng.uniform(0.25, 2.0, 25)
    tensor = mt.Tensor(rows, dtype=dtype)
    narrow = dtype == "float32"
    counts_t = mt.Tensor(counts.astype(np.int64), dtype="int64")
    reliability_t = mt.Tensor(reliability, dtype="float64")
    for correction in (0, 1, 2):
        got = mt.cov(tensor, correction, counts_t).tolist()
        want = _want_cov(rows, narrow, frequencies=counts, correction=correction)
        assert _same(got, want)
        got = mt.cov(tensor, correction, None, reliability_t).tolist()
        want = _want_cov(rows, narrow, reliabilities=reliability, correction=correction)
        assert _same(got, want)
        got = mt.cov(tensor, correction, counts_t, reliability_t).tolist()
        want = _want_cov(
            rows,
            narrow,
            frequencies=counts,
            reliabilities=reliability,
            correction=correction,
        )
        assert _same(got, want)


def test_integers():
    rng = np.random.default_rng(619)
    small = rng.integers(-1000, 1000, size=(3, 17)).astype(np.int32)
    got = mt.cov(mt.Tensor(small, dtype="int32"))
    assert str(got.dtype) == "float32"
    assert _same(got.tolist(), _want_cov(small.tolist(), True))
    # int64s past 2^53, near each other and far apart.
    big = 2**60 + rng.integers(-(2**20), 2**20, size=(2, 9))
    wide = np.vstack([big, rng.integers(-(2**62), 2**62, size=(1, 9))]).astype(np.int64)
    got = mt.cov(mt.Tensor(wide, dtype="int64")).tolist()
    assert _same(got, _want_cov(wide.tolist(), False))
    got = mt.corrcoef(mt.Tensor(wide, dtype="int64")).tolist()
    assert _same(got, _want_corrcoef(wide.tolist(), False))


def test_specials_and_constant_rows():
    nan, inf = math.nan, math.inf
    rows = [[1.0, 2.0, 4.0], [3.0, nan, 1.0], [2.0, 2.0, 2.0], [0.5, -1.0, inf]]
    got = mt.cov(mt.tensor(rows, dtype="float64")).tolist()
    for i in range(4):
        for j in range(4):
            if 1 in (i, j) or 3 in (i, j):
                assert math.isnan(got[i][j])
            elif 2 in (i, j):
                assert got[i][j] == 0.0
    assert got[0][0] == _want_cov([rows[0]], False)[0][0]
    got = mt.corrcoef(mt.tensor(rows, dtype="float64")).tolist()
    assert got[0][0] == 1.0
    assert all(math.isnan(v) for v in got[2]) and all(math.isnan(v) for v in got[1])
    # A variance past the largest float, beside entries that are not.
    rows = [[1e20, 2e20, 4e20], [1e-20, 3e-20, 2e-20]]
    got = mt.cov(mt.Tensor(np.array(rows, dtype=np.float32))).tolist()
    assert got[0][0] == inf
    want = _want_cov(np.array(rows, dtype=np.float32).tolist(), True)
    assert got[0][1] == got[1][0] == want[0][1] and got[1][1] == want[1][1]


def test_errors():
    rows = mt.tensor([[1.0, 2.0, 4.0], [3.0, 1.0, 2.0]], dtype="float64")
    with pytest.raises(ValueError, match="non-positive divisor"):
        mt.cov(rows, 3)
    with pytest.raises(ValueError, match="positive value"):
        mt.cov(rows, 1, mt.tensor([0.0, 0.0, 0.0], dtype="float64"))
    with pytest.raises(ValueError, match="non-positive divisor"):
        mt.corrcoef(mt.tensor([[1.0], [2.0]], dtype="float64"))


@pytest.mark.parametrize("weighted", [False, True])
def test_gradients(weighted):
    rng = np.random.default_rng(631)
    rows = rng.normal(size=(3, 6))
    counts = np.array([1, 2, 1, 3, 1, 2]) if weighted else None
    reliability = rng.uniform(0.5, 1.5, 6) if weighted else None

    def cov(values):
        tensor = mt.Tensor(values, dtype="float64", requires_grad=True)
        weights = {}
        if weighted:
            weights = {
                "fweights": mt.Tensor(counts.astype(np.int64), dtype="int64"),
                "aweights": mt.Tensor(reliability, dtype="float64"),
            }
        return tensor, mt.cov(tensor, 1, **weights)

    probe = rng.normal(size=(3, 3))
    for function in ("cov", "corrcoef"):
        if function == "corrcoef" and weighted:
            continue

        def value(values):
            if function == "cov":
                return cov(values)
            tensor = mt.Tensor(values, dtype="float64", requires_grad=True)
            return tensor, mt.corrcoef(tensor)

        tensor, result = value(rows)
        (result * mt.Tensor(probe, dtype="float64")).sum().backward()
        analytic = np.array(tensor.grad.tolist())
        numeric = np.zeros_like(rows)
        step = 1e-6
        for index in np.ndindex(*rows.shape):
            up, down = rows.copy(), rows.copy()
            up[index] += step
            down[index] -= step
            plus = (np.array(value(up)[1].tolist()) * probe).sum()
            minus = (np.array(value(down)[1].tolist()) * probe).sum()
            numeric[index] = (plus - minus) / (2 * step)
        np.testing.assert_allclose(analytic, numeric, rtol=1e-6, atol=1e-8)
