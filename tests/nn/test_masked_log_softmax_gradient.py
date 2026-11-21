# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""`masked_log_softmax`'s gradient against exact values.

Its backward took its probabilities as `exp` of its rounded output and formed
`g - p S` in the tensor's dtype, as `log_softmax`'s did, so that a confident,
correct slice's gradient at its likeliest entry -- `-(1 - p)` under a negative
log-likelihood -- was all rounding. It is now formed from the input over each
slice's unmasked entries, as `log_softmax`'s is, and a masked entry's gradient
is zero.
"""

from __future__ import annotations

import decimal
import math
from decimal import Decimal
from fractions import Fraction

import numpy as np
import pytest

import minitensor as mt
from minitensor import functional as F

_DIGITS = {"float32": (24, -149, 128), "float64": (53, -1074, 1024)}


def _rounded(value: Decimal, dtype: str) -> float:
    """The float nearest a decimal value, ties to even, rounded once."""

    if value == 0:
        return 0.0
    exact = Fraction(abs(value))
    digits, least, top = _DIGITS[dtype]
    e = exact.numerator.bit_length() - exact.denominator.bit_length()
    if Fraction(2) ** e > exact:
        e -= 1
    unit = Fraction(2) ** max(e - (digits - 1), least)
    near = round(exact / unit) * unit
    out = math.inf if near >= Fraction(2) ** top else float(near)
    return -out if value < 0 else out


def _exact(row, hidden, upstream):
    """`g - p S` over a slice's unmasked entries, exactly, and the larger of
    its two terms; zero at the rest."""

    kept = [i for i in range(len(row)) if not hidden[i]]
    top = max(row[i] for i in kept)
    terms = {i: (Decimal(row[i]) - Decimal(top)).exp() for i in kept}
    total = sum(terms.values())
    s = sum(Decimal(upstream[i]) for i in kept)
    out = []
    for i in range(len(row)):
        if hidden[i]:
            out.append((Decimal(0), Decimal(0)))
            continue
        g, ps = Decimal(upstream[i]), terms[i] / total * s
        out.append((g - ps, max(abs(g), abs(ps))))
    return out


def _case(confident: bool, rng):
    rows, classes = 40, 8
    z = rng.standard_normal((rows, classes)) * 3.0
    hidden = rng.random((rows, classes)) < 0.3
    target = rng.integers(0, classes, rows)
    hidden[np.arange(rows), target] = False
    if confident:
        best = np.where(hidden, -np.inf, z).max(axis=1)
        z[np.arange(rows), target] = best + rng.uniform(5.0, 40.0, rows)
        # A negative log-likelihood's gradient: on the target alone.
        upstream = np.zeros((rows, classes))
        upstream[np.arange(rows), target] = -1.0
    else:
        upstream = rng.standard_normal((rows, classes))
    return z, hidden, upstream


def _check(got: float, want: Decimal, terms: Decimal, dtype: str, exact: bool):
    """Within an ulp of the exact value in float32, and in float64 within four
    of it where its terms do not cancel -- a negative log-likelihood's -- or
    else within two of the larger term: an arbitrary incoming gradient's
    `g - p S` can cancel, and its float64 terms are only so exact."""

    expected = _rounded(want, dtype)
    if dtype == "float32" or exact:
        unit = float(np.spacing(np.array(abs(expected), dtype=dtype)))
        bound = 1 if dtype == "float32" else 4
    else:
        unit = float(np.spacing(np.array(float(terms), dtype=dtype)))
        bound = 2
    assert abs(got - expected) <= bound * unit, (got, expected)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("confident", [True, False], ids=["confident", "ordinary"])
@pytest.mark.parametrize("dim", [0, 1])
def test_against_exact_values(dim, confident, dtype):
    rng = np.random.default_rng(53)
    z, hidden, upstream = _case(confident, rng)
    z, upstream = z.astype(dtype), upstream.astype(dtype)
    if dim == 0:
        z, hidden, upstream = z.T.copy(), hidden.T.copy(), upstream.T.copy()
    x = mt.Tensor(z, dtype=dtype, requires_grad=True)
    out = F.masked_log_softmax(x, mt.Tensor(hidden.tolist()), dim)
    out.backward(mt.Tensor(upstream, dtype=dtype))
    got = x.grad.numpy().astype(np.float64)
    if dim == 0:
        z, hidden, upstream, got = z.T, hidden.T, upstream.T, got.T
    with decimal.localcontext(prec=60, Emin=-(10**6), Emax=10**6):
        for r in range(z.shape[0]):
            row = z[r].astype(np.float64).tolist()
            want = _exact(row, hidden[r].tolist(), upstream[r].astype(np.float64))
            for c, (w, terms) in enumerate(want):
                if hidden[r, c]:
                    assert got[r, c] == 0.0, (r, c, got[r, c])
                    continue
                _check(got[r, c], w, terms, dtype, confident)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_a_broadcast_mask(dtype):
    # One mask row for every row of logits.
    rng = np.random.default_rng(59)
    z = rng.standard_normal((6, 5)) * 3.0
    hidden = np.array([[False, True, False, False, True]])
    upstream = rng.standard_normal((6, 5))
    x = mt.Tensor(z.astype(dtype), dtype=dtype, requires_grad=True)
    out = F.masked_log_softmax(x, mt.Tensor(hidden.tolist()), 1)
    out.backward(mt.Tensor(upstream.astype(dtype), dtype=dtype))
    got = x.grad.numpy().astype(np.float64)
    with decimal.localcontext(prec=60):
        for r in range(6):
            row = z[r].astype(dtype).astype(np.float64).tolist()
            g = upstream[r].astype(dtype).astype(np.float64)
            for c, (w, terms) in enumerate(_exact(row, hidden[0].tolist(), g)):
                _check(got[r, c], w, terms, dtype, False)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_the_ends(dtype):
    # Every entry hidden, or every unmasked one `-inf`: no mass, the gradient
    # the incoming one at the unmasked entries. A `+inf` takes all the mass;
    # a NaN makes the unmasked entries NaN. Masked entries are zero throughout.
    inf, nan = math.inf, math.nan
    z = [[1.0, 2.0, 3.0], [-inf, -inf, 5.0], [inf, 0.0, 1.0], [nan, 0.0, 1.0]]
    hidden = [[True, True, True], [False, False, True], [False, False, True]]
    hidden.append([False, False, True])
    g = [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]
    x = mt.Tensor(z, dtype=dtype, requires_grad=True)
    F.masked_log_softmax(x, mt.Tensor(hidden), 1).backward(mt.Tensor(g, dtype=dtype))
    got = x.grad.tolist()
    assert got[0] == [0.0, 0.0, 0.0]
    assert got[1] == [1.0, 2.0, 0.0]
    # `+inf` holds the mass: g - 1 * (1 + 2) at it, g at the finite entry.
    assert got[2] == [-2.0, 2.0, 0.0]
    assert math.isnan(got[3][0]) and math.isnan(got[3][1]) and got[3][2] == 0.0
