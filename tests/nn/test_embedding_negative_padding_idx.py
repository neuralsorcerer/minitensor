# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""`Embedding(..., padding_idx=-1)` counts from the end, as `embedding` does.

The functional form took a negative `padding_idx` as a position from the end
of the table; the layer refused it outright, so the same table could be built
one way and not the other.
"""

import pytest

import minitensor as mt
import minitensor.nn as nn


@pytest.mark.parametrize("index,row", [(-1, 4), (-5, 0), (2, 2)])
def test_padding_idx_names_the_row_counted_from_either_end(index, row):
    layer = nn.Embedding(5, 3, padding_idx=index)
    assert layer.padding_idx == row
    weight = dict(layer.named_parameters())["weight"]
    assert weight.numpy()[row].tolist() == [0.0, 0.0, 0.0]
    layer(mt.tensor([[row, (row + 1) % 5]], dtype="int64")).sum().backward()
    assert weight.grad.numpy()[row].tolist() == [0.0, 0.0, 0.0]
    assert weight.grad.numpy()[(row + 1) % 5].tolist() == [1.0, 1.0, 1.0]


@pytest.mark.parametrize("index", [5, -6])
def test_a_padding_idx_off_the_table_is_an_index_error(index):
    with pytest.raises(IndexError, match="out of range for a table of 5 rows"):
        nn.Embedding(5, 3, padding_idx=index)
