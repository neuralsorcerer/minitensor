# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""The softmax family rounds once.

Each term `exp(x - max)` was taken of a rounded difference, so the
difference's error went into the exponent: `|x - max|` ulps of the answer --
24 of a float64 probability at a spread of 30, and 49 of a float32 one down a
non-last axis, where the difference was rounded to float32. A float32 term
was also rounded to float32 before its quotient, and a quarter of the
probabilities came out an ulp from the nearest float32.

Each term is now taken of the exact difference and kept in float64 until the
result rounds, and a float64 total carries its additions' errors: a float32
probability, log-probability, log-sum-exp or cross-entropy is the float32
nearest its exact value on every path -- the last axis, a column, a slice
long enough to be split into bands, under a mask -- and a float64 one is
within a few ulps.
"""

from __future__ import annotations

import math
from decimal import Decimal, localcontext
from fractions import Fraction

import numpy as np
import pytest

import minitensor as mt
from minitensor import functional as F


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


def _nearest(value: Decimal, dtype: str) -> float:
    exact = Fraction(value)
    return _float32(exact) if dtype == "float32" else float(exact)


def _exact(values, mask=None, counts=None):
    """The exact shares, log-probabilities and log-sum-exp of one slice.

    `counts`, where given, says how many times each value stands in the
    slice, so a long one built from a few values costs a few exponentials.
    """

    with localcontext() as context:
        context.prec = 60
        kept = [i for i in range(len(values)) if mask is None or not mask[i]]
        x = {i: Decimal(float(values[i])) for i in kept}
        terms = {i: v.exp() for i, v in x.items()}
        total = sum(terms[i] * (1 if counts is None else counts[i]) for i in kept)
        lse = total.ln()
        shares = [
            terms[i] / total if i in x else Decimal(0) for i in range(len(values))
        ]
        logs = [x[i] - lse if i in x else None for i in range(len(values))]
        return shares, logs, lse


def _ulps(got: float, want: float) -> float:
    return 0.0 if got == want else abs(got - want) / math.ulp(want)


def _check(got, want, dtype, ulps):
    """Float32 values are the nearest float32; float64 within `ulps`."""

    for g, w in zip(got, want):
        g = float(g)
        if w is None:
            assert g == -math.inf
            continue
        w = _nearest(w, dtype)
        if dtype == "float32":
            assert g == w, (g, w)
        else:
            assert _ulps(g, w) <= ulps, (g, w, _ulps(g, w))


def _slices(array, dim):
    return array if dim == array.ndim - 1 else np.moveaxis(array, dim, -1)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("dim", [0, 1])
def test_softmax_and_log_softmax_round_once(dtype, dim):
    rng = np.random.default_rng(1201 + dim)
    x = (rng.standard_normal((12, 40)) * 15).astype(dtype)
    t = mt.Tensor(x, dtype=dtype)
    probabilities = _slices(F.softmax(t, dim).numpy(), dim)
    logs = _slices(F.log_softmax(t, dim).numpy(), dim)
    totals = F.logsumexp(t, dim).numpy()
    for values, p, lp, lse in zip(_slices(x, dim), probabilities, logs, totals):
        shares, exact_logs, exact_lse = _exact(values)
        _check(p, shares, dtype, 4)
        _check(lp, exact_logs, dtype, 2)
        _check([lse], [exact_lse], dtype, 1)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_a_float64_softmax_keeps_its_small_probabilities(dtype):
    # A confident row: every probability but one is a term near exp(-30),
    # whose rounded difference put 24 ulps into it.
    rng = np.random.default_rng(1213)
    logits = rng.standard_normal((50, 10)) * 0.5
    target = rng.integers(0, 10, 50)
    logits[np.arange(50), target] += 30.0
    logits = logits.astype(dtype)
    t = mt.Tensor(logits, dtype=dtype)
    probabilities = F.softmax(t, 1).numpy()
    losses = F.cross_entropy(
        t, mt.Tensor(target, dtype="int64"), reduction="none"
    ).numpy()
    for row, p, k, loss in zip(logits, probabilities, target, losses):
        shares, logs, _ = _exact(row)
        _check(p, shares, dtype, 4)
        with localcontext() as context:
            context.prec = 60
            _check([loss], [-logs[k]], dtype, 2)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("shape", [(140000,), (50000, 3)])
def test_slices_split_into_bands(dtype, shape):
    # Past 131072 elements a first-axis softmax is cut into bands of rows,
    # each with its own partial totals. The values come from a palette of
    # sixteen, so the exact total is sixteen exponentials.
    rng = np.random.default_rng(1223)
    palette = (rng.standard_normal(16) * 6).astype(dtype)
    x = palette[rng.integers(0, 16, shape)]
    t = mt.Tensor(x, dtype=dtype)
    probabilities = _slices(F.softmax(t, 0).numpy(), 0).reshape(-1, shape[0])
    logs = _slices(F.log_softmax(t, 0).numpy(), 0).reshape(-1, shape[0])
    totals = np.atleast_1d(F.logsumexp(t, 0).numpy())
    columns = _slices(x, 0).reshape(-1, shape[0])
    for column, p, lp, lse in zip(columns, probabilities, logs, totals):
        values, counts = np.unique(column, return_counts=True)
        shares, exact_logs, exact_lse = _exact(values, counts=counts)
        at = np.searchsorted(values, column)
        sample = rng.choice(shape[0], 300, replace=False)
        _check(p[sample], [shares[i] for i in at[sample]], dtype, 4)
        _check(lp[sample], [exact_logs[i] for i in at[sample]], dtype, 2)
        _check([lse], [exact_lse], dtype, 1)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("dim", [0, 1])
def test_masked_softmax_rounds_once(dtype, dim):
    rng = np.random.default_rng(1229 + dim)
    x = (rng.standard_normal((10, 24)) * 8).astype(dtype)
    mask = rng.uniform(size=(10, 24)) < 0.3
    t, m = mt.Tensor(x, dtype=dtype), mt.Tensor(mask)
    probabilities = _slices(F.masked_softmax(t, m, dim).numpy(), dim)
    logs = _slices(F.masked_log_softmax(t, m, dim).numpy(), dim)
    for values, hidden, p, lp in zip(
        _slices(x, dim), _slices(mask, dim), probabilities, logs
    ):
        shares, exact_logs, _ = _exact(values, hidden)
        _check(p, shares, dtype, 4)
        _check(lp, exact_logs, dtype, 2)


@pytest.mark.parametrize("dim", [0, 1])
def test_float32_log_softmax_gradient_rounds_once(dim):
    rng = np.random.default_rng(1231 + dim)
    x = (rng.standard_normal((6, 30)) * 4).astype(np.float32)
    upstream = rng.standard_normal((6, 30)).astype(np.float32)
    a = mt.Tensor(x, dtype="float32", requires_grad=True)
    F.log_softmax(a, dim).backward(mt.Tensor(upstream, dtype="float32"))
    got = _slices(a.grad.numpy(), dim)
    logs = _slices(F.log_softmax(mt.Tensor(x, dtype="float32"), dim).numpy(), dim)
    with localcontext() as context:
        context.prec = 60
        for g_row, l_row, u_row in zip(got, logs, _slices(upstream, dim)):
            total = sum(Decimal(float(u)) for u in u_row)
            # The gradient of the log-probabilities the forward returned.
            want = [
                Decimal(float(u)) - Decimal(float(lp)).exp() * total
                for u, lp in zip(u_row, l_row)
            ]
            _check(g_row, want, "float32", 0)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_the_ends(dtype):
    inf = math.inf

    def signs(values):
        return [math.copysign(1.0, v) if v == 0 else v for v in values]

    # An entry that takes no part leaves its partner's log-probability an
    # exact zero, along a row and down a column.
    assert signs(F.log_softmax(mt.Tensor([0.0, -inf], dtype=dtype), 0).tolist()) == [
        1.0,
        -inf,
    ]
    column = F.log_softmax(mt.Tensor([[0.0], [-inf]], dtype=dtype), 0).tolist()
    assert signs([column[0][0]]) == [1.0] and column[1][0] == -inf
    assert F.softmax(mt.Tensor([-inf, -inf, 0.0], dtype=dtype), 0).tolist() == [
        0.0,
        0.0,
        1.0,
    ]
    assert F.softmax(mt.Tensor([-inf, -inf], dtype=dtype), 0).tolist() == [0.0, 0.0]
    assert F.log_softmax(mt.Tensor([-inf, -inf], dtype=dtype), 0).tolist() == [
        -inf,
        -inf,
    ]
    assert F.logsumexp(mt.Tensor([0.0, -inf], dtype=dtype), 0).item() == 0.0
    # Two thousand terms of exp(-120) are 1.5e-49 together: below float32,
    # where the maximum's log-probability is the negative zero nearest it,
    # and in float64 that number itself.
    row = mt.Tensor([0.0] + [-120.0] * 2000, dtype=dtype)
    top = F.log_softmax(row, 0).tolist()[0]
    with localcontext() as context:
        # Enough digits that one plus the small total keeps all of it.
        context.prec = 120
        want = -(1 + 2000 * Decimal(-120).exp()).ln()
    assert top == _nearest(want, dtype)
    assert math.copysign(1.0, top) == -1.0
    for value in (math.nan,):
        got = F.softmax(mt.Tensor([value, 1.0], dtype=dtype), 0).tolist()
        assert all(math.isnan(v) for v in got)
