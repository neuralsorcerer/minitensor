# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""Softmax along the first axis has one block, and one block is one core.

The softmax kernels cut their work into blocks -- the slices along the reduced
axis -- and hand one to each task. Along the last axis that is thousands of
them; along the *first* axis it is exactly one, however large the tensor, so
the whole reduction ran on one core with the rest of the pool watching. A
4096x512 softmax over dim 0 took 14.9ms against NumPy's 8.9, while the same
tensor over dim 1 took 3.0 against 9.1.

The block's rows split instead: each band reads its own rows, keeps its own
column accumulators, and the bands merge at the end. The band boundaries come
from the block's shape alone and never from the thread count, because here the
partition decides how the column sums are grouped -- so the answer is the same
on any machine, which `test_reduction_determinism.py` checks by running it at
several thread counts.

These tests pin the arithmetic across the split: against a float64 reference,
at sizes where a band is a single row, and at sizes where the bands do not
divide the axis evenly.
"""

from __future__ import annotations

import numpy as np
import pytest

import minitensor as mt

# Shapes whose reduced axis is the first one and whose blocks are big enough to
# be cut up: a tall one, a wide one, one where a band is a single row, one that
# no band count divides evenly, and a 1-D one, where the "columns" are single
# values and the band is a contiguous run.
BANDED = [
    ((4096, 512), 0),
    ((512, 4096), 0),
    ((8, 100_000), 0),
    ((513, 4097), 0),
    ((37, 4096), 0),
    ((64, 64, 512), 0),
    ((200_003,), 0),
]


def _data(shape, seed=17):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(shape) * 6).astype(np.float32)


def _softmax_reference(x, axis):
    peak = x.max(axis=axis, keepdims=True)
    e = np.exp(x - peak)
    return e / e.sum(axis=axis, keepdims=True)


def _log_softmax_reference(x, axis):
    peak = x.max(axis=axis, keepdims=True)
    return x - peak - np.log(np.exp(x - peak).sum(axis=axis, keepdims=True))


@pytest.mark.parametrize("shape,dim", BANDED)
@pytest.mark.parametrize("op", ["softmax", "log_softmax"])
def test_the_split_answer_is_the_exact_one(shape, dim, op):
    values = _data(shape)
    exact = (_softmax_reference if op == "softmax" else _log_softmax_reference)(
        values.astype(np.float64), dim
    )
    got = getattr(mt.Tensor(values, dtype="float32"), op)(dim).numpy()
    relative = np.abs(got.astype(np.float64) - exact).max() / np.abs(exact).max()
    assert relative < 1e-5, f"{relative:.3e}"


@pytest.mark.parametrize("shape,dim", BANDED)
def test_every_column_still_sums_to_one(shape, dim):
    got = mt.Tensor(_data(shape, seed=19), dtype="float32").softmax(dim).numpy()
    totals = got.astype(np.float64).sum(axis=dim)
    assert float(np.abs(totals - 1.0).max()) < 1e-5


@pytest.mark.parametrize("rows", [63, 64, 65, 127, 128, 129, 8191, 8192, 8193])
def test_the_band_boundary_is_not_special(rows):
    """The kernel aims for 64 bands and blocks each of them by 8192 rows, so a
    row count either side of both counts has to give the same answer as one on
    it."""
    values = _data((rows, 512), seed=23)
    exact = _softmax_reference(values.astype(np.float64), 0)
    got = mt.Tensor(values, dtype="float32").softmax(0).numpy()
    relative = np.abs(got.astype(np.float64) - exact).max() / np.abs(exact).max()
    assert relative < 1e-5, f"{relative:.3e}"


def test_a_column_with_no_answer_says_so_rather_than_returning_one():
    """A column holding a NaN or a `+inf` has no softmax.

    The column-wise kernel used to divide only where the total was positive,
    which left a poisoned column holding its raw exponentials -- and those look
    like an answer. `softmax([[1, 1], [nan, 2]], dim=0)` came back with 1.0 in
    the first column: a column that sums to one and means nothing. NumPy and
    PyTorch both answer NaN there, and so does this library's own contiguous
    kernel, which is what the column-wise one now agrees with.
    """
    for poison in (np.nan, np.inf):
        values = np.array([[1.0, 1.0], [poison, 2.0]], dtype=np.float32)
        got = mt.from_numpy(values).softmax(0).numpy()
        assert np.isnan(got[:, 0]).all(), got
        np.testing.assert_allclose(
            got[:, 1], _softmax_reference(values[:, 1], 0), rtol=1e-6
        )


def test_an_all_negative_infinity_column_is_still_zeros():
    """The one total that must *not* divide: nothing to normalize, no NaN."""
    values = np.array([[-np.inf, 1.0], [-np.inf, 2.0]], dtype=np.float32)
    got = mt.from_numpy(values).softmax(0).numpy()
    assert (got[:, 0] == 0.0).all()
    np.testing.assert_allclose(got[:, 1], _softmax_reference(values[:, 1], 0), rtol=1e-6)

    logged = mt.from_numpy(values).log_softmax(0).numpy()
    assert np.isneginf(logged[:, 0]).all()


def test_the_gradient_still_matches_the_definition():
    """The forward pass changed shape; the backward pass reads its output."""
    values = _data((2048, 16), seed=29)
    t = mt.Tensor(values, dtype="float32", requires_grad=True)
    weights = _data((2048, 16), seed=31)
    (t.softmax(0) * mt.from_numpy(weights)).sum().backward()

    probabilities = _softmax_reference(values.astype(np.float64), 0)
    expected = probabilities * (
        weights - (probabilities * weights).sum(axis=0, keepdims=True)
    )
    np.testing.assert_allclose(
        mt.get_gradient(t).numpy(), expected, rtol=1e-4, atol=1e-6
    )
    mt.clear_autograd_graph()
