# Copyright (c) 2026 Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest

from minitensor.nn import DenseLayer, Sequential

ERROR_NESTED_SEQUENTIAL = "Nested Sequential modules are not supported"


def test_sequential_rejects_nested_sequential_modules_cleanly():
    inner = Sequential([DenseLayer(3, 4)])

    with pytest.raises(TypeError, match=ERROR_NESTED_SEQUENTIAL):
        Sequential([DenseLayer(3, 4), inner])


def test_sequential_add_module_rejects_nested_sequential_modules_cleanly():
    outer = Sequential()
    inner = Sequential([DenseLayer(3, 4)])

    with pytest.raises(TypeError, match=ERROR_NESTED_SEQUENTIAL):
        outer.add_module("nested", inner)


def test_sequential_add_module_failure_does_not_mutate_existing_modules():
    outer = Sequential([DenseLayer(2, 3)])
    base_params = outer.num_parameters()

    with pytest.raises(TypeError, match=ERROR_NESTED_SEQUENTIAL):
        outer.add_module("nested", Sequential([DenseLayer(3, 4)]))

    assert outer.num_parameters() == base_params


def test_sequential_add_module_still_accepts_valid_layer_after_failed_insert():
    outer = Sequential([DenseLayer(2, 3)])

    with pytest.raises(TypeError, match=ERROR_NESTED_SEQUENTIAL):
        outer.add_module("nested", Sequential([DenseLayer(3, 4)]))

    outer.add_module("next", DenseLayer(3, 5))
    assert outer.num_parameters() == (2 * 3 + 3) + (3 * 5 + 5)
