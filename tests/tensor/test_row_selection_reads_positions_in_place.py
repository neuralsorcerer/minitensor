# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""`t[positions]` selects rows without copying positions that need nothing.

The positions were copied twice on the way in -- once out of the index, and
again while negative ones were wrapped -- before the selection that reads
them once: a million of them cost two fresh eight-megabyte buffers and ten
times the selection itself. Positions already in range are now the
selection's input as they stand. A strided NumPy index was walked with an
iterator that paid 18 ns a position, and is now made contiguous first. The
answers, and the errors, are what they were; these pin them for every form an
index comes in.
"""

from __future__ import annotations

import numpy as np
import pytest

import minitensor as mt

ROWS = np.arange(40.0).reshape(10, 4)


def _forms(positions):
    positions = np.asarray(positions)
    yield "int64 tensor", mt.Tensor(positions.astype(np.int64))
    yield "int32 tensor", mt.Tensor(positions.astype(np.int32))
    yield "int64 array", positions.astype(np.int64)
    yield "uint16 array", (
        positions.astype(np.uint16) if positions.min(initial=0) >= 0 else None
    )
    yield "strided array", np.repeat(positions.astype(np.int64), 2)[::2]
    yield "reversed array", positions.astype(np.int64)[::-1].copy()[::-1]
    yield "list", positions.tolist()


@pytest.mark.parametrize(
    "positions",
    [[0, 3, 3, 9], [9, 8, 7], [-1, -10, 2], list(range(10)), [5]],
    ids=["repeats", "descending", "negatives", "all", "one"],
)
def test_every_form_selects_the_same_rows(positions):
    tensor = mt.Tensor(ROWS, dtype="float64")
    for name, index in _forms(positions):
        if index is None:
            continue
        np.testing.assert_array_equal(
            tensor[index].numpy(), ROWS[positions], err_msg=name
        )


@pytest.mark.parametrize("bad", [10, -11])
def test_a_position_out_of_range_is_reported_whatever_its_form(bad):
    tensor = mt.Tensor(ROWS, dtype="float64")
    for name, index in _forms([0, bad, 1]):
        if index is None:
            continue
        with pytest.raises(IndexError, match=f"index {bad} is out of bounds"):
            tensor[index]


def test_the_selection_is_still_differentiable():
    weights = mt.Tensor(ROWS, dtype="float64", requires_grad=True)
    weights[mt.Tensor([1, 1, -1])].sum().backward()
    want = np.zeros_like(ROWS)
    want[1] = 2
    want[-1] = 1
    np.testing.assert_array_equal(weights.grad.numpy(), want)


def test_an_empty_index_selects_no_rows():
    tensor = mt.Tensor(ROWS, dtype="float64")
    assert tuple(tensor[mt.Tensor(np.array([], dtype=np.int64))].shape) == (0, 4)
    assert tuple(tensor[np.array([], dtype=np.int64)].shape) == (0, 4)
