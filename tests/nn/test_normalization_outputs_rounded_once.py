# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""The normalization layers' outputs against exact values, in both dtypes.

Each output is `w y + b`, `y` the standardized value. A float32 one was the
float64 formula narrowed: where the bias all but cancels the product, the
float64 roundings outweighed what was left, and a bias of `-w y` rounded to
the dtype left outputs 7.7 ulps out. Each output is now the value nearest
its exact one.
"""

from __future__ import annotations

import decimal
from decimal import Decimal

import numpy as np
import pytest

import minitensor as mt
from minitensor import functional as F

_EPS = 1e-5


def _standardized(values, centred=True):
    """The exact standardized values of a row."""

    X = [Decimal(float(v)) for v in values]
    n = len(X)
    mean = sum(X) / n if centred else Decimal(0)
    r = 1 / (sum((v - mean) ** 2 for v in X) / n + Decimal(_EPS)).sqrt()
    return [(v - mean) * r for v in X]


def _ulps(got, want: Decimal, dtype: str) -> float:
    if want == 0:
        return 0.0 if got == 0 else float("inf")
    unit = Decimal(float(np.spacing(np.array(abs(float(want)), dtype=dtype))))
    return float(abs(Decimal(float(got)) - want) / unit)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_layer_norm_where_the_bias_cancels(dtype):
    rng = np.random.default_rng(31)
    x = rng.standard_normal((3, 48)).astype(dtype)
    w = rng.uniform(0.5, 2.0, 48).astype(dtype)
    with decimal.localcontext(prec=80):
        first = _standardized(x[0])
        # The bias that leaves the first row only the rounding of `w y`.
        b = np.array([-float(Decimal(float(c)) * y) for c, y in zip(w, first)])
        b = b.astype(dtype)
        out = F.layer_norm(
            mt.Tensor(x, dtype=dtype),
            [48],
            mt.Tensor(w, dtype=dtype),
            mt.Tensor(b, dtype=dtype),
            _EPS,
        ).numpy()
        for row in range(3):
            for k, y in enumerate(_standardized(x[row])):
                want = Decimal(float(w[k])) * y + Decimal(float(b[k]))
                assert _ulps(out[row, k], want, dtype) <= 0.5, (row, k)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_rms_norm_outputs(dtype):
    rng = np.random.default_rng(32)
    x = rng.standard_normal((4, 40)).astype(dtype)
    w = rng.uniform(0.5, 2.0, 40).astype(dtype)
    out = F.rms_norm(mt.Tensor(x, dtype=dtype), [40], mt.Tensor(w, dtype=dtype), _EPS)
    out = out.numpy()
    with decimal.localcontext(prec=80):
        for row in range(4):
            for k, y in enumerate(_standardized(x[row], centred=False)):
                want = Decimal(float(w[k])) * y
                assert _ulps(out[row, k], want, dtype) <= 0.5, (row, k)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_batch_norm_where_the_bias_cancels(dtype):
    rng = np.random.default_rng(33)
    x = rng.standard_normal((5, 3, 4)).astype(dtype)
    w = rng.uniform(0.5, 2.0, 3).astype(dtype)
    with decimal.localcontext(prec=80):
        channels = [_standardized(x[:, c, :].ravel()) for c in range(3)]
        b = np.array([-float(Decimal(float(w[c])) * channels[c][0]) for c in range(3)])
        b = b.astype(dtype)
        out = F.batch_norm(
            mt.Tensor(x, dtype=dtype),
            None,
            None,
            mt.Tensor(w, dtype=dtype),
            mt.Tensor(b, dtype=dtype),
            True,
            0.1,
            _EPS,
        ).numpy()
        for c in range(3):
            got = out[:, c, :].ravel()
            for k, y in enumerate(channels[c]):
                want = Decimal(float(w[c])) * y + Decimal(float(b[c]))
                assert _ulps(got[k], want, dtype) <= 0.5, (c, k)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_a_row_holding_a_nan_takes_the_formula(dtype):
    x = np.random.default_rng(34).standard_normal((3, 6)).astype(dtype)
    x[1, 2] = np.nan
    out = F.layer_norm(mt.Tensor(x, dtype=dtype), [6], None, None, _EPS).numpy()
    assert np.all(np.isnan(out[1]))
    assert np.all(np.isfinite(out[[0, 2]]))
