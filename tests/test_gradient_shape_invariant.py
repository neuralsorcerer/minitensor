# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""A gradient must have the same shape and dtype as the tensor it belongs to.

This is the invariant a broadcasting backward breaks: `add`ing a `(3, 1)` to a
`(3, 4)` broadcasts the first operand, and the backward has to sum the incoming
gradient back down to `(3, 1)` rather than hand back the `(3, 4)` it received.
Forgetting that reduction produces a gradient an optimizer cannot apply, and it
only shows up for the shapes that actually broadcast -- which is why it is swept
across every combination here rather than spot-checked.

Nothing failed when this was written. It is kept because the failure mode is
shape-dependent and silent until the exact pairing occurs.
"""

import numpy as np
import pytest

import minitensor as mt


@pytest.fixture(autouse=True)
def _clean_graph():
    mt.clear_autograd_graph()
    yield
    mt.clear_autograd_graph()


def _positive(shape, seed):
    values = np.abs(np.random.default_rng(seed).standard_normal(shape)) + 1.0
    return mt.Tensor(values, dtype="float64", requires_grad=True)


# Pairs where at least one side broadcasts, plus an equal-shape control.
BROADCAST_PAIRS = [
    ((3, 1), (3, 4)),
    ((1, 4), (3, 4)),
    ((4,), (3, 4)),
    ((1,), (3, 4)),
    ((2, 1, 4), (2, 3, 4)),
    ((1, 3, 1), (2, 3, 4)),
    ((3, 4), (3, 4)),
]

# `add`, `sub`, `mul` and `div` exist only as operators -- there is no
# `mt.add`. Reaching for the named function silently skips them, which is how
# the first version of this sweep tested half of what it claimed to.
BINARY_OPS = [
    ("add", lambda a, b: a + b),
    ("sub", lambda a, b: a - b),
    ("mul", lambda a, b: a * b),
    ("div", lambda a, b: a / b),
    ("pow", lambda a, b: mt.pow(a, b)),
    ("maximum", lambda a, b: mt.maximum(a, b)),
    ("minimum", lambda a, b: mt.minimum(a, b)),
    ("logaddexp", lambda a, b: mt.logaddexp(a, b)),
]


@pytest.mark.parametrize("lhs_shape,rhs_shape", BROADCAST_PAIRS)
@pytest.mark.parametrize("name,op", BINARY_OPS, ids=[c[0] for c in BINARY_OPS])
def test_broadcast_gradients_match_their_operand(name, op, lhs_shape, rhs_shape):
    lhs, rhs = _positive(lhs_shape, 0), _positive(rhs_shape, 1)

    op(lhs, rhs).sum().backward()

    for label, tensor, shape in (("lhs", lhs, lhs_shape), ("rhs", rhs, rhs_shape)):
        assert tensor.grad is not None, f"{name}: no gradient for {label}"
        assert tuple(tensor.grad.shape) == shape, (
            f"{name}: {label} gradient has shape {tuple(tensor.grad.shape)}, "
            f"but the operand is {shape}"
        )


SHAPE_AND_REDUCTION_OPS = [
    ("sum-dim", lambda t: t.sum(0, False)),
    ("sum-keepdim", lambda t: t.sum(0, True)),
    ("mean-dim", lambda t: t.mean(1, False)),
    ("max-dim", lambda t: t.max(0, False)[0]),
    ("min-dim", lambda t: t.min(1, False)[0]),
    ("softmax", lambda t: t.softmax(dim=1)),
    ("logsumexp", lambda t: t.logsumexp([1])),
    ("norm", lambda t: t.norm()),
    ("transpose", lambda t: t.transpose(0, 1)),
    ("reshape", lambda t: t.reshape((12,))),
    ("repeat", lambda t: t.repeat([2, 1])),
    ("roll", lambda t: t.roll(1, 0)),
    ("flip", lambda t: t.flip([0])),
    ("cumsum", lambda t: t.cumsum(0)),
    ("matmul", lambda t: t.matmul(t.transpose(0, 1))),
]


@pytest.mark.parametrize(
    "name,op", SHAPE_AND_REDUCTION_OPS, ids=[c[0] for c in SHAPE_AND_REDUCTION_OPS]
)
def test_reduction_and_reshape_gradients_match_the_input(name, op):
    x = _positive((3, 4), 2)
    op(x).sum().backward()
    assert x.grad is not None, f"{name}: no gradient"
    assert tuple(x.grad.shape) == (3, 4)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize(
    "name,op",
    SHAPE_AND_REDUCTION_OPS,
    ids=[c[0] for c in SHAPE_AND_REDUCTION_OPS],
)
def test_gradients_keep_their_tensor_dtype(name, op, dtype):
    # A float32 parameter with a float64 gradient is a step an optimizer either
    # rejects or silently widens.
    values = np.abs(np.random.default_rng(3).standard_normal((3, 4))) + 1.0
    x = mt.Tensor(values, dtype=dtype, requires_grad=True)
    op(x).sum().backward()
    assert x.grad is not None
    assert (
        x.grad.dtype == dtype
    ), f"{name}: {dtype} tensor got a {x.grad.dtype} gradient"
