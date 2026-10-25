# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""A confident prediction's log-probability keeps its digits.

`log_softmax(x)_i` is `x_i - max - log(total)` for the total of
`exp(x - max)`, and a slice with one dominant entry has a total of
`1 + rest` for a small `rest`. The logarithm of that total kept only the
digits of `rest` that survived being added to one, and `x - (max + log)`
cancelled the rest away for the dominant entry: in float32 the log-probability
of the logits `[20, 0 x 999]` came out 7% off, and its cross-entropy with it;
in float64 5e-9. `logsumexp([0, -20])` was 0 in float32. The maximum's own term
now comes out of the total and back through `log1p`, and the maximum off each
value before the logarithm does.
"""

from __future__ import annotations

import math
from fractions import Fraction

import numpy as np
import pytest

import minitensor as mt

F = mt.functional


def _reference(values) -> np.ndarray:
    """`log_softmax` in float64 from the float64 exponentials, with the
    dominant term's `log1p` taken exactly."""

    values = np.asarray(values, dtype=np.float64)
    top = values.max()
    first = int(np.argmax(values))
    rest = math.fsum(math.exp(v - top) for i, v in enumerate(values) if i != first)
    return (values - top) - math.log1p(rest)


CONFIDENT = [
    [10.0] + [0.0] * 9,
    [20.0] + [0.0] * 999,
    [12.0, 1.0, -3.0, 2.0],
    [30.0, 3.0, -1.0],
    [3.0, 3.0, 1.0],
    [0.5, -40.0],
]


@pytest.mark.parametrize("logits", CONFIDENT, ids=lambda v: f"{len(v)}x{v[0]}")
@pytest.mark.parametrize("dtype,tolerance", [("float32", 1e-6), ("float64", 1e-14)])
@pytest.mark.parametrize("layout", ["last", "column", "middle", "masked"])
def test_log_softmax_of_confident_logits(logits, dtype, tolerance, layout):
    values = np.array(logits, dtype=dtype)
    want = _reference(values.astype(np.float64))
    if layout == "last":
        got = F.log_softmax(mt.Tensor(values, dtype=dtype), dim=0).numpy()
    elif layout == "column":
        got = F.log_softmax(mt.Tensor(values.reshape(-1, 1), dtype=dtype), dim=0)
        got = got.numpy().ravel()
    elif layout == "middle":
        tiled = np.stack([values, values], axis=1).reshape(1, -1, 2)
        got = F.log_softmax(mt.Tensor(tiled, dtype=dtype), dim=1).numpy()[0, :, 0]
    else:
        mask = mt.Tensor(np.zeros(len(values), dtype=bool), dtype="bool")
        got = F.masked_log_softmax(mt.Tensor(values, dtype=dtype), mask, dim=0).numpy()
    np.testing.assert_allclose(got, want, rtol=tolerance, atol=0)


@pytest.mark.parametrize("dtype,tolerance", [("float32", 1e-6), ("float64", 1e-14)])
def test_cross_entropy_of_a_confident_prediction(dtype, tolerance):
    for logits in CONFIDENT:
        values = np.array(logits, dtype=dtype).reshape(1, -1)
        target = mt.Tensor(np.array([0]), dtype="int64")
        loss = F.cross_entropy(mt.Tensor(values, dtype=dtype), target).item()
        want = -_reference(values.ravel().astype(np.float64))[0]
        assert math.isclose(loss, want, rel_tol=tolerance), (logits, loss, want)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_logsumexp_near_zero(dtype):
    want = math.log1p(math.exp(-20.0))
    tolerance = 1e-6 if dtype == "float32" else 1e-15
    for tensor, dim in (
        (mt.Tensor([0.0, -20.0], dtype=dtype), 0),
        (mt.Tensor([[0.0], [-20.0]], dtype=dtype), 0),
        (mt.Tensor([[0.0, -20.0]], dtype=dtype), 1),
    ):
        got = tensor.logsumexp(dim=dim).numpy().item()
        assert math.isclose(got, want, rel_tol=tolerance), (got, want)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_ordinary_logits_are_unchanged(dtype):
    rng = np.random.default_rng(313)
    values = rng.normal(0, 3, (64, 100)).astype(dtype)
    wide = values.astype(np.float64)
    top = wide.max(axis=1, keepdims=True)
    want = (wide - top) - np.log(np.exp(wide - top).sum(axis=1, keepdims=True))
    got = F.log_softmax(mt.Tensor(values, dtype=dtype), dim=1).numpy()
    tolerance = 2e-6 if dtype == "float32" else 1e-13
    np.testing.assert_allclose(got, want, rtol=tolerance, atol=tolerance)
    lse = mt.Tensor(values, dtype=dtype).logsumexp(dim=1).numpy()
    np.testing.assert_allclose(
        lse,
        (top + np.log(np.exp(wide - top).sum(1, keepdims=True))).ravel(),
        rtol=tolerance,
    )


def test_the_limits_stay_the_limits():
    inf = math.inf
    got = F.log_softmax(mt.Tensor([inf, 1.0, -inf]), dim=0).tolist()
    assert got[0] == 0.0 and got[1] == -inf and got[2] == -inf
    assert F.log_softmax(mt.Tensor([-inf, -inf]), dim=0).tolist() == [-inf, -inf]
    assert math.isnan(F.log_softmax(mt.Tensor([math.nan, 1.0]), dim=0).tolist()[1])
    assert mt.Tensor([-inf, -inf]).logsumexp(dim=0).item() == -inf
    assert F.log_softmax(mt.Tensor([7.0]), dim=0).tolist() == [0.0]


def test_the_gradient_is_softmax_less_the_one_hot():
    values = np.array([20.0, 0.0, 0.0], dtype=np.float64)
    x = mt.Tensor(values, dtype="float64", requires_grad=True)
    F.log_softmax(x, dim=0)[0].backward()
    probabilities = np.exp(_reference(values))
    want = -probabilities
    want[0] += 1.0
    np.testing.assert_allclose(x.grad.numpy(), want, rtol=1e-12, atol=1e-18)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_logsumexp_with_a_gradient_keeps_its_digits(dtype):
    tolerance = 1e-6 if dtype == "float32" else 1e-15
    x = mt.Tensor([0.0, -20.0], dtype=dtype, requires_grad=True)
    value = x.logsumexp(dim=0)
    assert math.isclose(value.item(), math.log1p(math.exp(-20.0)), rel_tol=tolerance)
    value.backward()
    share = math.exp(-20.0) / (1 + math.exp(-20.0))
    np.testing.assert_allclose(x.grad.numpy(), [1 - share, share], rtol=tolerance)
    grid = mt.Tensor([[0.0, -20.0], [-30.0, -25.0]], dtype=dtype, requires_grad=True)
    want = math.log1p(math.exp(-20.0) + math.exp(-30.0) + math.exp(-25.0))
    assert math.isclose(grid.logsumexp(dim=[0, 1]).item(), want, rel_tol=tolerance)
    ties = mt.Tensor([3.0, 3.0, 1.0], dtype=dtype, requires_grad=True)
    total = ties.logsumexp(dim=0)
    assert math.isclose(total.item(), 3 + math.log(2 + math.exp(-2)), rel_tol=tolerance)
    total.backward()
    weights = np.exp(np.array([3.0, 3.0, 1.0]) - (3 + math.log(2 + math.exp(-2))))
    np.testing.assert_allclose(ties.grad.numpy(), weights, rtol=tolerance)


def test_softmax_is_within_an_ulp_or_two():
    rng = np.random.default_rng(317)
    for _ in range(20):
        n = int(rng.integers(10, 3000))
        values = rng.normal(0, 3, n).astype(np.float32)
        wide = values.astype(np.float64)
        top = wide.max()
        lse = top + math.log(math.fsum(np.exp(wide - top)))
        want = np.exp(wide - lse)
        for got in (
            F.softmax(mt.Tensor(values), dim=0).numpy(),
            F.softmax(mt.Tensor(values.reshape(-1, 1)), dim=0).numpy().ravel(),
        ):
            ulps = np.abs(got - want) / np.spacing(want.astype(np.float32))
            assert ulps.max() < 2.0
