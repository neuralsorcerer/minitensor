# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""A product's gradients read its operands where they lie.

`d(a @ b)` is `g @ b^T` for `a` and `a^T @ g` for `b`, and the backward used to
transpose each operand into a copy and multiply that -- a full copy of a weight
matrix on every pass, half again the products' own cost. The products now read
the operands in their stored layout. These pin the gradients against the two
formulas for every arrangement the forward reduces a product to.
"""

from __future__ import annotations

import numpy as np
import pytest

import minitensor as mt


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize(
    "lhs,rhs",
    [
        ((5, 7), (7, 3)),
        ((4, 5, 7), (4, 7, 3)),
        ((2, 3, 5, 7), (2, 3, 7, 4)),
        ((2, 1, 5, 7), (1, 3, 7, 4)),
        ((4, 5, 7), (7, 3)),
        ((5, 7), (4, 7, 3)),
        ((0, 7), (7, 3)),
        ((5, 0), (0, 3)),
    ],
    ids=str,
)
def test_both_gradients_are_the_two_formulas(lhs, rhs, dtype):
    rng = np.random.default_rng(179)
    a = rng.normal(0, 1, lhs).astype(dtype)
    b = rng.normal(0, 1, rhs).astype(dtype)
    g = rng.normal(0, 1, np.broadcast_shapes(lhs[:-2], rhs[:-2]) + (lhs[-2], rhs[-1]))
    g = g.astype(dtype)
    left = mt.Tensor(a, dtype=dtype, requires_grad=True)
    right = mt.Tensor(b, dtype=dtype, requires_grad=True)
    (left @ right).backward(mt.Tensor(g, dtype=dtype))

    def reduce_to(grad, shape):
        # A broadcast operand's gradient is summed over the axes it was
        # broadcast along.
        while grad.ndim > len(shape):
            grad = grad.sum(0)
        axes = tuple(i for i, n in enumerate(shape) if n == 1 and grad.shape[i] != 1)
        return grad.sum(axes, keepdims=True) if axes else grad

    wide = np.float64
    want_left = reduce_to(g.astype(wide) @ np.swapaxes(b, -1, -2).astype(wide), lhs)
    want_right = reduce_to(np.swapaxes(a, -1, -2).astype(wide) @ g.astype(wide), rhs)
    tolerance = 1e-5 if dtype == "float32" else 1e-12
    np.testing.assert_allclose(
        left.grad.numpy(), want_left, rtol=tolerance, atol=tolerance
    )
    np.testing.assert_allclose(
        right.grad.numpy(), want_right, rtol=tolerance, atol=tolerance
    )


def test_only_the_operand_that_asks_gets_a_gradient():
    a = mt.Tensor(np.ones((3, 4)), dtype="float64", requires_grad=True)
    b = mt.Tensor(np.ones((4, 2)), dtype="float64")
    (a @ b).sum().backward()
    np.testing.assert_array_equal(a.grad.numpy(), np.full((3, 4), 2.0))
    assert b.grad is None


def test_a_transposed_weight_is_differentiated_through_its_transpose():
    rng = np.random.default_rng(181)
    x = rng.normal(0, 1, (6, 8))
    w = rng.normal(0, 1, (5, 8))
    g = rng.normal(0, 1, (6, 5))
    X = mt.Tensor(x, dtype="float64", requires_grad=True)
    W = mt.Tensor(w, dtype="float64", requires_grad=True)
    (X @ W.T).backward(mt.Tensor(g, dtype="float64"))
    np.testing.assert_allclose(X.grad.numpy(), g @ w, rtol=1e-12)
    np.testing.assert_allclose(W.grad.numpy(), g.T @ x, rtol=1e-12)
