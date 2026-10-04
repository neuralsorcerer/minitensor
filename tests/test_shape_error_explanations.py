# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""Shape errors say which rule the shapes broke.

They used to read "Shape mismatch: expected [2, 3], got [4]" with advice to
reshape, whatever the operation: for two shapes that fail to broadcast, an
element count that will not divide, a tensor that will not join another, or
an input with the wrong channel count, that names neither the rule nor the
dimension at fault. The first line is unchanged; the suggestion now explains.
"""

import pytest

import minitensor as mt
import minitensor.nn as nn

CASES = {
    "broadcast": (
        lambda: mt.randn(2, 3) + mt.randn(4),
        r"\[2, 3\] and \[4\] do not broadcast: aligned from the right, dimension -1 is 3",
    ),
    "reshape": (
        lambda: mt.randn(6).reshape(4, 2),
        r"6 elements cannot be reshaped to \[4, 2\], which holds 8",
    ),
    "reshape with -1": (
        lambda: mt.randn(6).reshape(4, -1),
        r"6 elements are not a whole number of the 4",
    ),
    "cat dimensions": (
        lambda: mt.cat([mt.randn(2, 3), mt.randn(2, 4)]),
        r"along dimension 0 needs every other dimension to match, but dimension 1 is 3",
    ),
    "cat rank": (
        lambda: mt.cat([mt.randn(2, 3), mt.randn(3)]),
        r"tensors of one rank.*stack adds a new dimension",
    ),
    "conv channels": (
        lambda: nn.Conv2d(3, 4, 3)(mt.randn(1, 2, 8, 8)),
        r"input has 2 channels \(shape \[1, 2, 8, 8\]\).*reads 3",
    ),
}


@pytest.mark.parametrize("name", list(CASES))
def test_the_error_names_the_rule(name):
    call, explanation = CASES[name]
    with pytest.raises(ValueError, match=explanation):
        call()


def test_a_shape_error_is_still_one():
    with pytest.raises(ValueError, match="Shape mismatch"):
        mt.zeros(2) + mt.zeros(3)


def test_matmul_promotes_mixed_dtypes_like_the_elementwise_ops():
    a = mt.randn(2, 3, requires_grad=True)
    b = mt.randn(3, 2).astype("float64").requires_grad_(True)
    product = a @ b
    assert product.dtype == "float64"
    reference = a.detach().astype("float64") @ b.detach()
    assert product.tolist() == reference.tolist()
    product.sum().backward()
    # The gradient returns to each operand in its own dtype.
    assert a.grad.dtype == "float32" and b.grad.dtype == "float64"
    ints = mt.tensor([[1, 2], [3, 4]], dtype="int64")
    assert (mt.ones(2, 2) @ ints).dtype == "float32"
    assert (
        mt.bmm(mt.ones(1, 2, 2), mt.ones(1, 2, 2).astype("float64")).dtype == "float64"
    )
