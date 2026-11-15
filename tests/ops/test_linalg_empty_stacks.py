# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""A stack of no matrices, or of empty ones, answers with an empty stack.

`det`, `slogdet`, `logdet`, `inv`, `solve`, `lu` and `matrix_power` with a
negative power counted the matrices of a stack as at least one, so a stack with
a dimension of zero read a matrix past the end of its data and panicked; and
`matrix_exp` took the largest column norm of a stack that has none, and
refused. Each answers now with the shape the stack implies.
"""

from __future__ import annotations

import numpy as np
import pytest

import minitensor as mt

SHAPES = [(0, 2, 2), (3, 0, 3, 3), (0, 1, 1), (0, 5, 5), (2, 0, 0), (0, 0)]


def _identities(shape):
    return np.zeros(shape) + np.eye(shape[-1])


@pytest.mark.parametrize("shape", SHAPES)
def test_determinants(shape):
    x = mt.tensor(_identities(shape))
    batch = shape[:-2]
    det = mt.det(x)
    sign, log = mt.slogdet(x)
    assert tuple(det.shape) == batch
    assert tuple(sign.shape) == batch and tuple(log.shape) == batch
    assert tuple(mt.logdet(x).shape) == batch
    # The determinant of an empty matrix is the empty product.
    assert (det.numpy() == 1).all()
    assert (sign.numpy() == 1).all() and (log.numpy() == 0).all()


@pytest.mark.parametrize("shape", SHAPES)
def test_inverses_solves_and_factors(shape):
    x = mt.tensor(_identities(shape))
    assert tuple(mt.inv(x).shape) == shape
    assert tuple(mt.solve(x, mt.tensor(np.zeros(shape[:-1]))).shape) == shape[:-1]
    assert tuple(mt.solve(x, mt.tensor(np.zeros(shape))).shape) == shape
    assert tuple(mt.matrix_power(x, -2).shape) == shape
    assert tuple(mt.matrix_exp(x).shape) == shape
    assert all(tuple(part.shape) == shape for part in mt.lu(x))


@pytest.mark.parametrize("shape", SHAPES)
def test_gradients_reach_an_empty_stack(shape):
    for op in (
        lambda t: mt.det(t),
        lambda t: mt.slogdet(t)[1],
        lambda t: mt.inv(t),
        lambda t: mt.solve(t, mt.tensor(np.zeros(shape))),
    ):
        x = mt.tensor(_identities(shape), requires_grad=True)
        op(x).sum().backward()
        assert tuple(x.grad.shape) == shape
