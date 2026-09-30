# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""An out-of-range position is reported against the dimension it indexes.

The index checks behind `gather`, `index_select`, `scatter`, `narrow` and
basic indexing all reported "dimension 0", whichever axis the bad position
was on.
"""

from __future__ import annotations

import pytest

import minitensor as mt


def _x():
    return mt.arange(24, dtype="float64").reshape(2, 3, 4)


def _index(value, shape):
    return mt.full(shape, value, dtype="int64")


@pytest.mark.parametrize(
    ("call", "message"),
    [
        (
            lambda x: x.gather(2, _index(5, (2, 3, 1))),
            "index 5 .* dimension 2 with size 4",
        ),
        (
            lambda x: x.index_select(1, _index(3, (1,))),
            "index 3 .* dimension 1 with size 3",
        ),
        (
            lambda x: x.scatter(
                1, _index(3, (2, 1, 4)), mt.zeros(2, 1, 4, dtype="float64")
            ),
            "index 3 .* dimension 1 with size 3",
        ),
        (lambda x: x.narrow(2, 5, 0), "index 5 .* dimension 2 with size 4"),
        (lambda x: x[0, 5], "index 5 .* dimension 1 with size 3"),
    ],
    ids=["gather", "index_select", "scatter", "narrow-start", "getitem"],
)
def test_the_error_names_the_indexed_dimension(call, message):
    with pytest.raises(IndexError, match=message):
        call(_x())


def test_narrow_past_the_end_names_start_length_and_dimension():
    with pytest.raises(ValueError, match="start 3 plus length 2 .* dimension 2"):
        _x().narrow(2, 3, 2)
