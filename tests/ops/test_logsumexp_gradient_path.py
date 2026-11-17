# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""`logsumexp` answers the same whether or not a gradient is wanted.

With a gradient wanted, or over several axes, its value was composed from
`sub`, `exp`, `sum` and `log1p` in the input's dtype: a float32 input's terms
were rounded to float32, its exponents taken of rounded differences, and a sum
that cancelled kept nothing -- a float64 `logsumexp` of normalised
log-probabilities was some 1e16 ulps out when its gradient was tracked. Every
`logsumexp` but an empty one now takes the fused value, and its gradient is
the softmax over the reduced axes times the output's.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import minitensor as mt
from minitensor import functional as F

_DIMS = [0, 1, 2, (0, 2), (1, 2), (0, 1, 2)]


def _softmax_over(values, dims):
    peak = values.max(axis=dims, keepdims=True)
    weights = np.exp(values - peak)
    return weights / weights.sum(axis=dims, keepdims=True)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("dims", _DIMS)
@pytest.mark.parametrize("keepdim", [False, True])
def test_the_value_does_not_depend_on_the_gradient(dtype, dims, keepdim):
    rng = np.random.default_rng(1409)
    x = (rng.standard_normal((4, 5, 6)) * 6).astype(dtype)
    as_list = list(dims) if isinstance(dims, tuple) else dims
    plain = F.logsumexp(mt.Tensor(x, dtype=dtype), as_list, keepdim).numpy()
    tracked = F.logsumexp(
        mt.Tensor(x, dtype=dtype, requires_grad=True), as_list, keepdim
    )
    assert tracked.requires_grad
    np.testing.assert_array_equal(tracked.detach().numpy(), plain)


@pytest.mark.parametrize("dims", _DIMS)
@pytest.mark.parametrize("keepdim", [False, True])
def test_the_gradient_is_the_softmax_over_the_reduced_axes(dims, keepdim):
    rng = np.random.default_rng(1423)
    x = rng.standard_normal((4, 5, 6)) * 4
    a = mt.Tensor(x, dtype="float64", requires_grad=True)
    as_list = list(dims) if isinstance(dims, tuple) else dims
    out = F.logsumexp(a, as_list, keepdim)
    upstream = rng.standard_normal(out.shape)
    out.backward(mt.Tensor(upstream, dtype="float64"))
    axes = dims if isinstance(dims, tuple) else (dims,)
    shaped = upstream if keepdim else np.expand_dims(upstream, axes)
    want = _softmax_over(x, axes) * shaped
    np.testing.assert_allclose(a.grad.numpy(), want, rtol=1e-14, atol=1e-300)
    mt.clear_autograd_graph()


def test_log_probabilities_keep_their_digits_with_a_gradient():
    rng = np.random.default_rng(1427)
    # A total of one up to the rounding of the logarithms, nudged off it so
    # the answer is a number the float holds well.
    x = np.log(rng.dirichlet(np.ones(12), 30)) + rng.uniform(-1e-9, 1e-9, (30, 1))
    a = mt.Tensor(x, dtype="float64", requires_grad=True)
    tracked = F.logsumexp(a, 1).detach().numpy()
    plain = F.logsumexp(mt.Tensor(x, dtype="float64"), 1).numpy()
    np.testing.assert_array_equal(tracked, plain)
    from decimal import Decimal, localcontext

    for row, got in zip(x, tracked):
        with localcontext() as context:
            context.prec = 60
            want = float(sum(Decimal(float(v)).exp() for v in row).ln())
        assert got == want or abs(got - want) <= math.ulp(want), (got, want)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_rows_at_an_infinity(dtype):
    inf = math.inf
    x = mt.Tensor(
        [[-inf, -inf, -inf], [inf, 1.0, 2.0], [0.0, -inf, 1.0]],
        dtype=dtype,
        requires_grad=True,
    )
    out = F.logsumexp(x, 1)
    got = out.detach().tolist()
    assert got[0] == -inf and got[1] == inf
    out.backward(mt.Tensor([1.0, 1.0, 1.0], dtype=dtype))
    grad = x.grad.tolist()
    # Nothing moves a sum of nothing, and an infinity takes all of the slope.
    assert grad[0] == [0.0, 0.0, 0.0]
    assert grad[1] == [1.0, 0.0, 0.0]
    assert grad[2][1] == 0.0
    mt.clear_autograd_graph()


def test_an_empty_reduction_is_log_zero():
    x = mt.Tensor(np.zeros((3, 0)), dtype="float64", requires_grad=True)
    got = F.logsumexp(x, 1).detach().tolist()
    assert got == [-math.inf] * 3
