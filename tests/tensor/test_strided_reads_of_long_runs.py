# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""A long strided selection reads every element it names, split or not.

A selection is copied as runs, and the parallel split hands out whole runs, so
a single strided run -- `x[::2]` of a vector -- went to one core however long
it was. It is now split within the run, which must not move any element.
"""

from __future__ import annotations

import numpy as np
import pytest

import minitensor as mt


@pytest.mark.parametrize(
    "selection",
    [
        slice(None, None, 2),
        slice(1, None, 3),
        slice(5, 900_001, 7),
        slice(None, None, 1 << 19),
        slice(3, 1_000_000, 64),
    ],
)
def test_a_long_strided_slice_reads_what_it_names(selection):
    values = np.random.default_rng(31).standard_normal(1 << 20)
    tensor = mt.Tensor(values, dtype="float64")
    np.testing.assert_array_equal(tensor[selection].numpy(), values[selection])


def test_a_strided_column_of_a_tall_matrix_and_its_gradient():
    values = np.random.default_rng(37).standard_normal((300_000, 3))
    tensor = mt.Tensor(values, dtype="float64", requires_grad=True)
    column = tensor[::2, 1]
    np.testing.assert_array_equal(column.detach().numpy(), values[::2, 1])
    column.sum().backward()
    want = np.zeros_like(values)
    want[::2, 1] = 1.0
    np.testing.assert_array_equal(tensor.grad.numpy(), want)
