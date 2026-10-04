# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""A negative count or size is a `ValueError` that says what it got.

Parameters that hold a count, a length or a size used to be extracted as
unsigned, so `-1` surfaced as the conversion's `OverflowError: can't convert
negative int to unsigned`, which names neither the argument nor the value.
They now take a signed int and refuse a negative one with a `ValueError`;
the argument's name is attached to the error as a note.
"""

import pytest

import minitensor as mt
import minitensor.functional as F
import minitensor.nn as nn
import minitensor.optim as optim

X = mt.randn(2, 3, 8)


def _layer():
    return nn.DenseLayer(4, 2)


CASES = {
    "eye": lambda: mt.eye(-1),
    "chunk": lambda: X.chunk(-2),
    "narrow length": lambda: X.narrow(0, 0, -1),
    "adaptive_avg_pool1d": lambda: F.adaptive_avg_pool1d(X, -1),
    "adaptive_max_pool1d": lambda: F.adaptive_max_pool1d(X, -1),
    "DenseLayer": lambda: nn.DenseLayer(-1, 2),
    "Embedding": lambda: nn.Embedding(-3, 2),
    "layer_norm int": lambda: F.layer_norm(X, -8),
    "layer_norm sequence": lambda: F.layer_norm(X, [3, -8]),
    "LayerNorm": lambda: nn.LayerNorm(-8),
    "Tensor.layer_norm": lambda: X.layer_norm([-8]),
    "rms_norm": lambda: F.rms_norm(X, -8),
    "MultiStepLR milestones": lambda: optim.MultiStepLR(
        optim.SGD(_layer().parameters(), lr=0.1), milestones=[2, -1]
    ),
}


@pytest.mark.parametrize("name", list(CASES))
def test_negative_size_is_a_value_error(name):
    with pytest.raises(ValueError, match="non-negative integer, got -"):
        CASES[name]()


def test_the_argument_is_named():
    with pytest.raises(ValueError) as info:
        F.adaptive_avg_pool1d(X, -1)
    assert "output_size" in " ".join(getattr(info.value, "__notes__", [])) + str(
        info.value
    )


def test_the_last_parameter_is_converted_too():
    # A size in the final position was the one an earlier pass missed.
    with pytest.raises(ValueError, match="got -4"):
        F.adaptive_max_pool1d(X, -4)


def test_a_non_integer_shape_is_still_a_type_error():
    with pytest.raises(TypeError):
        F.layer_norm(X, "eight")


def test_valid_sizes_keep_working():
    assert F.adaptive_avg_pool1d(X, 4).shape == (2, 3, 4)
    assert F.layer_norm(X, 8).shape == (2, 3, 8)
    assert F.layer_norm(X, [3, 8]).shape == (2, 3, 8)
    assert X.narrow(2, 1, 0).shape == (2, 3, 0)
    assert mt.eye(0).shape == (0, 0)
