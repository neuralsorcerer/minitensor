# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""Float64 gradcheck for the operations that take arguments.

`test_gradcheck_op_surface` sweeps everything reachable as a no-argument unary
method by walking `dir(Tensor)`. What it cannot reach is the larger half: an op
needing a dimension, an index tensor, a second operand, a mode. Those need
fixtures written by hand, which is this file.

The selection is deliberate rather than exhaustive. It concentrates on the
places a backward is easy to get wrong and where the rest of the suite checks
only at `rtol=3e-2`:

* **Broadcast gradient reduction.** When an operand was broadcast, its gradient
  has to be summed back down to its own shape. Getting the axes wrong is the
  classic error, and it is invisible in any test whose operands already match.
* **Routing.** `max`, `topk`, `sort`, `median`, `clamp`, `maximum` and the
  scans send the gradient to some inputs and not others. An off-by-one in the
  routing is arithmetically silent -- the values are all plausible.
* **Gather and scatter.** The backward is the opposite permutation, and a
  wrong one still produces a gradient of the right shape and magnitude.
* **Recently added.** `cummax`, `cummin`, `logcumsumexp` and `scatter_reduce`
  are the newest arrivals here, and new code is where errors are.

Ops with their own dedicated finite-difference tests -- the linalg
factorisations, conv, pooling, interpolate, einsum, the losses -- are not
repeated.
"""

from __future__ import annotations

import numpy as np
import pytest

import minitensor as mt

_RNG = np.random.default_rng(7)

_VECTOR = [0.31, -1.27, 0.88, 2.04]
_POSITIVE = [0.3, 0.9, 1.7, 2.6]
_MATRIX = [[0.4, -1.1, 0.7], [1.9, 0.25, -0.6]]


def _t(values, requires_grad=True):
    return mt.Tensor(
        np.ascontiguousarray(np.asarray(values, np.float64)),
        dtype="float64",
        requires_grad=requires_grad,
    )


def _ints(values):
    return mt.Tensor(values, dtype="int64")


#: (id, function of the checked tensors, factory building fresh inputs).
#: Inputs come from a factory rather than being built once at collection: each
#: check clears and rewrites `.grad`, so a shared tensor would carry state
#: between cases.
CASES = [
    # --- scans -------------------------------------------------------------
    ("cumsum", lambda t: t.cumsum(0).sum(), lambda: (_t(_VECTOR),)),
    ("cumprod", lambda t: t.cumprod(0).sum(), lambda: (_t(_POSITIVE),)),
    ("cummax", lambda t: mt.cummax(t, 0)[0].sum(), lambda: (_t(_VECTOR),)),
    ("cummin", lambda t: mt.cummin(t, 0)[0].sum(), lambda: (_t(_VECTOR),)),
    (
        "logcumsumexp",
        lambda t: mt.logcumsumexp(t, 0).sum(),
        lambda: (_t(_VECTOR),),
    ),
    # --- selection and routing --------------------------------------------
    ("topk", lambda t: t.topk(2)[0].sum(), lambda: (_t(_VECTOR),)),
    ("sort", lambda t: t.sort()[0].sum(), lambda: (_t(_VECTOR),)),
    ("median", lambda t: t.median().sum(), lambda: (_t(_VECTOR),)),
    ("quantile", lambda t: t.quantile(0.4).sum(), lambda: (_t(_VECTOR),)),
    ("kthvalue", lambda t: mt.kthvalue(t, 2)[0].sum(), lambda: (_t(_VECTOR),)),
    ("max_dim", lambda t: t.max(0)[0].sum(), lambda: (_t(_MATRIX),)),
    ("min_dim", lambda t: t.min(1)[0].sum(), lambda: (_t(_MATRIX),)),
    ("amax", lambda t: t.amax(0).sum(), lambda: (_t(_MATRIX),)),
    ("clamp", lambda t: t.clamp(-0.5, 1.0).sum(), lambda: (_t(_VECTOR),)),
    (
        "maximum",
        lambda a, b: a.maximum(b).sum(),
        lambda: (_t(_VECTOR), _t([0.0, -0.5, 1.2, 1.0])),
    ),
    (
        "minimum",
        lambda a, b: a.minimum(b).sum(),
        lambda: (_t(_VECTOR), _t([0.0, -0.5, 1.2, 1.0])),
    ),
    # --- broadcast gradient reduction --------------------------------------
    (
        "add_broadcast",
        lambda a, b: (a + b).sum(),
        lambda: (_t(_MATRIX), _t([[0.5], [1.5]])),
    ),
    (
        "sub_broadcast",
        lambda a, b: (a - b).sum(),
        lambda: (_t(_MATRIX), _t([0.7, -1.3, 2.1])),
    ),
    (
        "mul_broadcast",
        lambda a, b: (a * b).sum(),
        lambda: (_t(_MATRIX), _t([0.7, -1.3, 2.1])),
    ),
    (
        "div_broadcast",
        lambda a, b: (a / b).sum(),
        lambda: (_t(_MATRIX), _t([[2.0], [3.0]])),
    ),
    (
        "pow_broadcast",
        lambda a, b: a.pow(b).sum(),
        lambda: (_t([[1.4, 2.1, 0.8]]), _t([[2.0], [3.0]])),
    ),
    # --- matmul family -----------------------------------------------------
    (
        "matmul_2d",
        lambda a, b: a.matmul(b).sum(),
        lambda: (_t(_MATRIX), _t(_RNG.standard_normal((3, 2)))),
    ),
    (
        "matmul_batched",
        lambda a, b: a.matmul(b).sum(),
        lambda: (
            _t(_RNG.standard_normal((2, 3, 4))),
            _t(_RNG.standard_normal((2, 4, 2))),
        ),
    ),
    (
        "dot",
        lambda a, b: a.dot(b).sum(),
        lambda: (_t(_VECTOR), _t([1.1, -0.4, 0.9, 2.2])),
    ),
    # --- gather and scatter ------------------------------------------------
    (
        "index_select",
        lambda t: t.index_select(0, _ints([2, 0, 2])).sum(),
        lambda: (_t(_VECTOR),),
    ),
    (
        "gather",
        lambda t: t.gather(0, _ints([3, 1, 1, 0])).sum(),
        lambda: (_t(_VECTOR),),
    ),
    (
        "masked_select",
        lambda t: t.masked_select(
            mt.Tensor([True, False, True, True], dtype="bool")
        ).sum(),
        lambda: (_t(_VECTOR),),
    ),
    (
        "scatter_reduce",
        lambda t: mt.scatter_reduce(
            mt.Tensor(np.zeros(4), dtype="float64"),
            0,
            _ints([0, 1, 1, 3]),
            t,
            "sum",
        ).sum(),
        lambda: (_t(_VECTOR),),
    ),
    ("getitem_slice", lambda t: t[1:3].sum(), lambda: (_t(_VECTOR),)),
    (
        "getitem_fancy",
        lambda t: t[_ints([0, 2, 2])].sum(),
        lambda: (_t(_VECTOR),),
    ),
    # --- shape --------------------------------------------------------------
    (
        "cat",
        lambda a, b: mt.cat([a, b], 0).sum(),
        lambda: (_t(_VECTOR), _t([1.0, 2.0])),
    ),
    (
        "stack",
        lambda a, b: mt.stack([a, b], 0).sum(),
        lambda: (_t(_VECTOR), _t([1.0, 2.0, 3.0, 4.0])),
    ),
    ("split", lambda t: mt.split(t, 2, 0)[1].sum(), lambda: (_t(_VECTOR),)),
    ("repeat", lambda t: t.repeat([2]).sum(), lambda: (_t(_VECTOR),)),
    ("flip", lambda t: t.flip([0]).sum(), lambda: (_t(_VECTOR),)),
    ("roll", lambda t: t.roll(1, 0).sum(), lambda: (_t(_VECTOR),)),
    ("narrow", lambda t: t.narrow(0, 1, 2).sum(), lambda: (_t(_VECTOR),)),
    ("transpose", lambda t: t.transpose(0, 1).sum(), lambda: (_t(_MATRIX),)),
    # --- reductions with a dimension ---------------------------------------
    ("sum_keepdim", lambda t: t.sum(1, True).sum(), lambda: (_t(_MATRIX),)),
    ("mean_dim", lambda t: t.mean(0).sum(), lambda: (_t(_MATRIX),)),
    (
        "prod_dim",
        lambda t: t.prod(1).sum(),
        lambda: (_t([[1.3, 0.7, 2.1], [0.9, 1.8, 0.5]]),),
    ),
    ("var_dim", lambda t: t.var(1).sum(), lambda: (_t(_MATRIX),)),
    ("std_dim", lambda t: t.std(1).sum(), lambda: (_t(_MATRIX),)),
    ("norm", lambda t: t.norm(2.0).sum(), lambda: (_t(_VECTOR),)),
    ("logsumexp_dim", lambda t: t.logsumexp(1).sum(), lambda: (_t(_MATRIX),)),
    ("softmax_dim", lambda t: t.softmax(1).sum(), lambda: (_t(_MATRIX),)),
]


@pytest.fixture(autouse=True)
def _clean_graph():
    yield
    mt.clear_autograd_graph()


#: A floor on the case count, so a refactor that empties the table cannot leave
#: a file that passes by checking nothing.
MINIMUM_CASES = 40


def test_the_table_is_populated():
    assert (
        len(CASES) >= MINIMUM_CASES
    ), f"only {len(CASES)} cases, expected at least {MINIMUM_CASES}"


@pytest.mark.parametrize(
    "func,build", [(f, b) for _, f, b in CASES], ids=[name for name, _, _ in CASES]
)
def test_gradient_matches_finite_differences(func, build):
    assert mt.gradcheck(func, build(), atol=1e-6, rtol=1e-5)
