# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""An optimizer's saved state pickles, so a checkpoint can be one file.

A model's `state_dict()` is a dict of tensors and a scheduler's is a dict of
numbers, and both pickled; the optimizer's `OptimizerState` did not, so
`pickle.dump({"model": ..., "optim": opt.state_dict()}, f)` failed. It now
travels as the bytes `save` writes, checked on the way back in as a file is.
"""

import copy
import pickle

import numpy as np
import pytest

import minitensor as mt
import minitensor.optim as optim

TARGET = np.array([0.5, -1.0, 2.0, 0.25])


def _loss(w):
    return ((w - mt.tensor(TARGET, dtype="float64")) ** 2).sum() - w.cos().sum()


def _train(opt, w, steps):
    for _ in range(steps):
        opt.zero_grad()
        _loss(w).backward()
        opt.step()


def _param():
    return mt.tensor(np.zeros(4), dtype="float64", requires_grad=True)


@pytest.mark.parametrize(
    "cls,kwargs",
    [
        (optim.Adam, dict(lr=0.1, amsgrad=True)),
        (optim.SGD, dict(lr=0.05, momentum=0.9)),
        (optim.NAdam, dict(lr=0.1)),
        (optim.Rprop, dict(lr=0.05)),
    ],
    ids=["Adam", "SGD", "NAdam", "Rprop"],
)
def test_a_pickled_checkpoint_resumes_the_same_run(cls, kwargs):
    straight = _param()
    _train(cls([straight], **kwargs), straight, 10)

    first = _param()
    opt = cls([first], **kwargs)
    _train(opt, first, 5)
    blob = pickle.dumps({"params": [first], "optim": opt.state_dict()})

    checkpoint = pickle.loads(blob)
    resumed = checkpoint["params"][0]
    again = cls([resumed], **kwargs)
    again.load_state_dict(checkpoint["optim"])
    _train(again, resumed, 5)
    assert np.allclose(resumed.numpy(), straight.numpy(), rtol=1e-12, atol=1e-14)


def test_the_state_copies_with_its_contents():
    w = _param()
    opt = optim.Adam([w], lr=0.1)
    _train(opt, w, 3)
    state = opt.state_dict()
    for clone in (copy.copy(state), copy.deepcopy(state)):
        assert clone.algorithm == "Adam"
        assert clone.step_count == 3
        assert clone.buffer_names() == state.buffer_names()


@pytest.mark.parametrize("data", [b"", b"\xff" * 40, b"\x05abc"])
def test_corrupt_bytes_are_refused_as_a_corrupt_file_is(data):
    with pytest.raises(OSError, match="deserialization failed"):
        optim.OptimizerState._from_bytes(data)
