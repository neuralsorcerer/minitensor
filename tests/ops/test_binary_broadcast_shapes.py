# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""Every binary operator produces the broadcast shape, including `pow`.

Broadcasting says the result has rank `max(rank(a), rank(b))` and each of its
dimensions is the larger of the two right-aligned inputs. A one-element operand
is not exempt from the rank half of that: `(1, 1) ** (3,)` is `(1, 3)`, because
the base contributes a leading axis that the exponent does not have.

`pow` had two fast paths keyed on `numel() == 1` -- a scalar base mapped over
the exponent, and a scalar exponent mapped over the base -- and each took the
*other* operand's shape verbatim as the output shape. That is right whenever
the one-element operand has the lower rank, which is the common case
(`x ** 2.0`), and wrong otherwise, silently dropping a dimension:

    Tensor(ones((1, 1))) ** Tensor(full((3,), 2.0))   ->  shape (3,)   not (1, 3)
    Tensor(ones((3,)))   ** Tensor(full((1, 1), 2.0)) ->  shape (3,)   not (1, 3)

Nine of the shape pairs below hit it. Nothing else did: `+`, `-`, `*`, `/`,
`//`, `%`, the six comparisons, `maximum`, `minimum` and `logaddexp` all agreed
with NumPy on all 225 pairs already, which is why the sweep is written against
every operator rather than against `pow` alone -- the point is that `pow` is no
longer the exception.

Only the shape metadata was wrong. A one-element operand broadcasts to the
other's element count either way, so the fast paths themselves still produce
the right elements in the right order and remain in place.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

import minitensor as mt

SHAPES = [
    (),
    (1,),
    (3,),
    (4,),
    (1, 1),
    (3, 1),
    (1, 4),
    (3, 4),
    (1, 1, 1),
    (2, 1, 1),
    (1, 1, 4),
    (1, 3, 4),
    (2, 1, 4),
    (2, 3, 1),
    (2, 3, 4),
]


# Values are kept small, positive and away from zero so that `**`, `//` and `%`
# are all defined on them and none of the results depend on a special case.
def _values(shape):
    if not shape:
        return np.float64(2.0)
    n = int(np.prod(shape))
    return (np.arange(n, dtype=np.float64).reshape(shape) % 4) + 1.0


OPS = {
    "add": (lambda a, b: a + b, lambda a, b: a + b),
    "sub": (lambda a, b: a - b, lambda a, b: a - b),
    "mul": (lambda a, b: a * b, lambda a, b: a * b),
    "div": (lambda a, b: a / b, lambda a, b: a / b),
    "pow": (lambda a, b: a**b, lambda a, b: a**b),
    "floor_divide": (lambda a, b: a // b, lambda a, b: a.floor_divide(b)),
    "remainder": (np.remainder, lambda a, b: a.remainder(b)),
    "lt": (lambda a, b: a < b, lambda a, b: a.lt(b)),
    "le": (lambda a, b: a <= b, lambda a, b: a.le(b)),
    "gt": (lambda a, b: a > b, lambda a, b: a.gt(b)),
    "ge": (lambda a, b: a >= b, lambda a, b: a.ge(b)),
    "eq": (lambda a, b: a == b, lambda a, b: a.eq(b)),
    "ne": (lambda a, b: a != b, lambda a, b: a.ne(b)),
    "maximum": (np.maximum, lambda a, b: a.maximum(b)),
    "minimum": (np.minimum, lambda a, b: a.minimum(b)),
    "logaddexp": (np.logaddexp, lambda a, b: a.logaddexp(b)),
}

PAIRS = list(itertools.product(SHAPES, repeat=2))


@pytest.mark.parametrize("op", list(OPS), ids=list(OPS))
def test_every_shape_pair_agrees_with_numpy(op):
    """Accepted or rejected the same way, and where both accept, the same shape
    and the same values. `(3,)` against `(4,)` is in the sweep on purpose: an
    incompatible pair has to raise rather than truncate to the shorter side."""
    npf, mtf = OPS[op]
    wrong = []

    for left, right in PAIRS:
        a, b = _values(left), _values(right)
        try:
            expected = np.asarray(npf(a, b))
            numpy_accepted = True
        except Exception:
            expected = None
            numpy_accepted = False

        try:
            got = mtf(mt.Tensor(a, dtype="float64"), mt.Tensor(b, dtype="float64"))
            accepted = True
        except Exception:
            got = None
            accepted = False

        where = f"{left} {op} {right}"
        if accepted != numpy_accepted:
            verb = "accepts" if numpy_accepted else "rejects"
            wrong.append(f"{where}: NumPy {verb} and this does not")
        elif numpy_accepted:
            if tuple(got.shape_vec()) != expected.shape:
                wrong.append(
                    f"{where}: shape {tuple(got.shape_vec())}, want {expected.shape}"
                )
            elif not np.allclose(
                got.numpy().astype(np.float64),
                expected.astype(np.float64),
                rtol=1e-12,
            ):
                wrong.append(f"{where}: values differ")

    assert not wrong, "\n".join(wrong)


# --- the cases `pow` used to get wrong --------------------------------------


@pytest.mark.parametrize(
    "base,exponent,expected",
    [
        ((1, 1), (3,), (1, 3)),
        ((3,), (1, 1), (1, 3)),
        ((1,), (), (1,)),
        ((1, 1), (), (1, 1)),
        ((1, 1), (1,), (1, 1)),
        ((1, 1), (5,), (1, 5)),
        ((5,), (1, 1), (1, 5)),
        ((1, 1, 1), (3, 4), (1, 3, 4)),
        ((3, 4), (1, 1, 1), (1, 3, 4)),
    ],
)
def test_a_one_element_operand_still_contributes_its_rank(base, exponent, expected):
    a = np.full(base, 2.0)
    b = np.full(exponent, 3.0)
    assert np.asarray(a**b).shape == expected, "premise: this is NumPy's shape"

    result = mt.Tensor(a, dtype="float64") ** mt.Tensor(b, dtype="float64")
    assert tuple(result.shape_vec()) == expected
    np.testing.assert_allclose(result.numpy(), a**b, rtol=1e-12)


def test_the_scalar_fast_paths_are_still_taken():
    """The fix is to the output shape only. A one-element operand of equal or
    lower rank keeps producing exactly what it did, which is what `x ** 2.0`
    and every other ordinary use goes through."""
    x = np.arange(1, 25, dtype=np.float64).reshape(2, 3, 4)
    tensor = mt.Tensor(x, dtype="float64")

    for exponent in [2.0, 1.0, 3.0, 0.5, -1.0]:
        scalar = mt.Tensor(np.array([exponent]), dtype="float64")
        np.testing.assert_allclose(
            (tensor**scalar).numpy(), x**exponent, rtol=1e-12, atol=0
        )
        assert tuple((tensor**scalar).shape_vec()) == x.shape

    base = mt.Tensor(np.array([2.0]), dtype="float64")
    np.testing.assert_allclose((base**tensor).numpy(), 2.0**x, rtol=1e-12)
    assert tuple((base**tensor).shape_vec()) == x.shape


# --- gradients through the corrected shape ----------------------------------


@pytest.mark.parametrize(
    "base,exponent",
    [
        ((1, 1), (3,)),
        ((3,), (1, 1)),
        ((1,), ()),
        ((1, 1), (2, 3, 4)),
        ((2, 3, 4), (1, 1)),
    ],
)
def test_gradients_survive_the_reshaped_output(base, exponent):
    """The backward pass works on flat slices and hands each gradient its own
    operand's shape, so the wider output shape must not disturb it."""

    def total(base_value, exponent_value):
        b = mt.Tensor(np.full(base, base_value), dtype="float64")
        e = mt.Tensor(np.full(exponent, exponent_value), dtype="float64")
        return float((b**e).sum().item())

    b = mt.Tensor(np.full(base, 2.0), dtype="float64", requires_grad=True)
    e = mt.Tensor(np.full(exponent, 3.0), dtype="float64", requires_grad=True)
    (b**e).sum().backward()

    assert tuple(b.grad.shape_vec()) == base
    assert tuple(e.grad.shape_vec()) == exponent

    h = 1e-6
    assert float(b.grad.sum().item()) == pytest.approx(
        (total(2.0 + h, 3.0) - total(2.0 - h, 3.0)) / (2 * h), rel=1e-5
    )
    assert float(e.grad.sum().item()) == pytest.approx(
        (total(2.0, 3.0 + h) - total(2.0, 3.0 - h)) / (2 * h), rel=1e-5
    )
