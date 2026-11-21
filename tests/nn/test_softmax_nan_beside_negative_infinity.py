# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""A NaN makes its softmax slice NaN, beside `-inf`s as beside anything else.

A slice's maximum passes over a NaN, and beside nothing but `-inf` the slice
was taken for one of `-inf`: `softmax([-inf, nan])` came back `[0, 0]` and
`log_softmax` `[-inf, -inf]` -- along a row, down a column, across the bands a
long first axis is cut into, and among a masked softmax's unmasked entries --
where `logsumexp` of the same slice was NaN. A masked softmax also divided its
masked entries' zeros by a NaN total, so that they came out NaN where they take
no part; they are zero, as they were already beside a `+inf`.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import minitensor as mt
from minitensor import functional as F

nan, inf = math.nan, math.inf


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize(
    "row", [[-inf, nan], [nan, -inf, -inf], [-inf, -inf, nan, -inf], [nan]]
)
def test_along_a_row(row, dtype):
    x = mt.Tensor(row, dtype=dtype)
    assert np.isnan(F.softmax(x, 0).numpy()).all()
    assert np.isnan(F.log_softmax(x, 0).numpy()).all()
    assert math.isnan(F.logsumexp(x, 0).item())


@pytest.mark.parametrize("dtype", ["float32", "float64"])
# 45,000 rows are enough for the first axis to be cut into bands.
@pytest.mark.parametrize("rows", [2, 45_000])
def test_down_a_column(rows, dtype):
    values = np.full((rows, 3), -inf)
    values[-1] = [nan, -inf, 1.0]
    x = mt.Tensor(values, dtype=dtype)
    p, lp = F.softmax(x, 0).numpy(), F.log_softmax(x, 0).numpy()
    for out in (p, lp):
        assert np.isnan(out[:, 0]).all()
        assert not np.isnan(out[:, 1:]).any()
    # The other columns keep their limits: no mass, and all of it.
    assert (p[:, 1] == 0).all() and (lp[:, 1] == -inf).all()
    assert p[-1, 2] == 1.0 and lp[-1, 2] == 0.0
    assert math.isnan(F.logsumexp(x, 0).tolist()[0])


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("dim", [0, 1])
def test_masked(dim, dtype):
    # Every slice has an unmasked NaN, beside an unmasked `-inf`, a finite
    # value and a `+inf`, and one masked entry.
    values = np.array([[-inf, nan, 2.0], [1.0, nan, 2.0], [inf, nan, 2.0]])
    mask = np.array([[False, False, True]] * 3)
    if dim == 0:
        values, mask = values.T.copy(), mask.T.copy()
    x, m = mt.Tensor(values, dtype=dtype), mt.Tensor(mask.tolist())
    p = F.masked_softmax(x, m, dim).numpy()
    lp = F.masked_log_softmax(x, m, dim).numpy()
    assert np.isnan(p[~mask]).all() and (p[mask] == 0).all()
    assert np.isnan(lp[~mask]).all() and (lp[mask] == -inf).all()


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_a_slice_of_negative_infinities_alone_keeps_its_limits(dtype):
    x = mt.Tensor([[-inf, -inf], [-inf, -inf]], dtype=dtype)
    assert F.softmax(x, 1).tolist() == [[0.0, 0.0], [0.0, 0.0]]
    assert F.log_softmax(x, 0).tolist() == [[-inf, -inf], [-inf, -inf]]
    m = mt.Tensor([[False, True], [False, False]])
    assert F.masked_softmax(x, m, 1).tolist() == [[0.0, 0.0], [0.0, 0.0]]
    assert F.masked_log_softmax(x, m, 1).tolist() == [[-inf, -inf], [-inf, -inf]]


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_the_gradients_of_a_nan_slice_are_nan(dtype):
    for op in (F.softmax, F.log_softmax):
        rows = [[-inf, nan, -inf], [0.0, 1.0, 2.0]]
        x = mt.Tensor(rows, dtype=dtype, requires_grad=True)
        out = op(x, 1)
        out.backward(mt.Tensor(np.ones((2, 3)), dtype=dtype))
        grad = x.grad.numpy()
        mt.clear_autograd_graph()
        assert np.isnan(grad[0]).all() and not np.isnan(grad[1]).any(), op
