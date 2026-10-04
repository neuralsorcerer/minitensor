# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""Devices, shapes and losses pickle and copy; devices and shapes compare.

None of the three could be pickled, copied or deep-copied, so neither could a
training configuration holding one. Every compiled class also reported its
module as `builtins`, which is where pickle went looking for it. Two
`Device("cpu")` compared unequal, and a device never equalled the string
`Tensor.device` gives; a shape equalled its tuple but could not be hashed.
"""

import copy
import pickle

import numpy as np
import pytest

import minitensor as mt
import minitensor.nn as nn
import minitensor.optim as optim

ROUND_TRIPS = [
    pytest.param(lambda o: pickle.loads(pickle.dumps(o)), id="pickle"),
    pytest.param(copy.copy, id="copy"),
    pytest.param(copy.deepcopy, id="deepcopy"),
]


@pytest.mark.parametrize("trip", ROUND_TRIPS)
@pytest.mark.parametrize(
    "device", [mt.Device.cpu(), mt.Device.cuda(1)], ids=["cpu", "cuda1"]
)
def test_a_device_round_trips(trip, device):
    assert trip(device) == device


@pytest.mark.parametrize("trip", ROUND_TRIPS)
def test_a_shape_round_trips(trip):
    shape = mt.zeros(2, 3, 4).shape
    assert trip(shape) == (2, 3, 4)


def test_devices_compare_by_what_they_name():
    cpu = mt.Device.cpu()
    assert cpu == mt.Device("cpu")
    assert cpu == "cpu" and mt.zeros(1).device == cpu
    assert cpu != mt.Device.cuda(0) and cpu != "cuda" and cpu != 3
    assert hash(cpu) == hash(mt.Device("cpu")) == hash("cpu")
    assert {cpu: 1}["cpu"] == 1


def test_a_shape_hashes_like_its_tuple():
    shape = mt.zeros(2, 3).shape
    assert hash(shape) == hash((2, 3))
    assert {shape: "x"}[(2, 3)] == "x"
    assert (2, 3) in {shape}


LOSSES = [
    lambda: nn.MSELoss(reduction="sum"),
    lambda: nn.MAELoss(),
    lambda: nn.HuberLoss(delta=0.3, reduction="none"),
    lambda: nn.SmoothL1Loss(beta=0.2),
    lambda: nn.LogCoshLoss(),
    lambda: nn.FocalLoss(alpha=0.5, gamma=1.0),
]


@pytest.mark.parametrize("trip", ROUND_TRIPS)
@pytest.mark.parametrize("build", LOSSES)
def test_a_regression_loss_round_trips_with_its_settings(trip, build):
    loss = build()
    clone = trip(loss)
    assert type(clone) is type(loss) and repr(clone) == repr(loss)
    prediction = mt.tensor([[0.2, 0.7], [0.9, 0.1]])
    target = mt.tensor([[0.0, 1.0], [1.0, 0.0]])
    assert np.allclose(
        clone(prediction, target).numpy(), loss(prediction, target).numpy()
    )


@pytest.mark.parametrize("trip", ROUND_TRIPS)
def test_a_weighted_cross_entropy_round_trips(trip):
    loss = nn.CrossEntropyLoss(
        weight=mt.tensor([1.0, 2.0, 3.0]), ignore_index=2, label_smoothing=0.1
    )
    clone = trip(loss)
    assert clone.ignore_index == 2 and clone.label_smoothing == pytest.approx(0.1)
    assert np.array_equal(clone.weight.numpy(), loss.weight.numpy())
    logits = mt.randn(4, 3)
    labels = mt.tensor([0, 1, 2, 1], dtype="int64")
    assert np.allclose(clone(logits, labels).numpy(), loss(logits, labels).numpy())


@pytest.mark.parametrize("trip", ROUND_TRIPS)
def test_a_logits_loss_keeps_its_positive_weight(trip):
    clone = trip(nn.BCEWithLogitsLoss(pos_weight=mt.tensor([2.0])))
    assert clone.pos_weight.tolist() == [2.0]


@pytest.mark.parametrize(
    "cls,module",
    [
        (nn.DenseLayer, "minitensor.nn"),
        (nn.MSELoss, "minitensor.nn"),
        (nn.LSTM, "minitensor.nn"),
        (optim.Adam, "minitensor.optim"),
        (optim.StepLR, "minitensor.optim"),
        (optim.OptimizerState, "minitensor.optim"),
        (mt.Device, "minitensor._core"),
    ],
)
def test_classes_name_the_module_they_live_in(cls, module):
    assert cls.__module__ == module
