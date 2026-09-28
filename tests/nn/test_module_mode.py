# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""A module's mode can be read back, and `train()`/`eval()` return the module.

The mode could be set but not read: only the layers that behave differently in
the two modes kept a flag, and none was reachable from Python, so code that
had to restore a model's mode after an evaluation could not tell what to
restore it to.
"""

from __future__ import annotations

import copy

import numpy as np

import minitensor as mt

nn = mt.nn


def _model():
    inner = nn.Sequential([nn.Dropout(0.5)])
    return nn.Sequential([nn.DenseLayer(4, 4), inner]), inner


def test_a_new_module_is_in_training_mode():
    assert nn.DenseLayer(4, 3).training
    assert nn.Dropout(0.5).training
    assert nn.Sequential().training


def test_train_and_eval_return_the_module_and_set_the_mode():
    model, _ = _model()
    assert model.eval() is model
    assert not model.training
    assert model.train() is model
    assert model.training
    assert model.train(False) is model
    assert not model.training


def test_a_sequential_sets_the_mode_of_everything_inside():
    model, inner = _model()
    model.eval()
    assert [part.training for part in model] == [False, False]
    assert not inner[0].training

    inner.train()
    assert inner.training and inner[0].training
    assert not model.training


def test_the_mode_read_back_is_the_one_the_layers_run_in():
    model, inner = _model()
    x = mt.ones(64, 4)
    model.eval()
    assert not inner[0].training
    np.testing.assert_array_equal(model(x).numpy(), model(x).numpy())

    inner[0].train()
    assert inner[0].training
    assert not np.array_equal(model(x).numpy(), model(x).numpy())


def test_a_copy_keeps_the_mode_of_each_part():
    model, inner = _model()
    model.eval()
    inner.train()
    copied = copy.deepcopy(model)
    assert (copied.training, copied[1].training, copied[1][0].training) == (
        False,
        True,
        True,
    )
