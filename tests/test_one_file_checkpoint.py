# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""A model's state, an optimizer's and a scheduler's pickle together.

`StateDict` could not be pickled or copied, so the one-file checkpoint --
`pickle.dump({"model": model.state_dict(), "optim": ..., "sched": ...})` --
failed on the model's part even once the optimizer's state pickled. It also
lacked `get`, and did not count as a mapping, though it has the rest of a
mapping's read side.
"""

import collections.abc
import copy
import pickle

import numpy as np
import pytest

import minitensor as mt
import minitensor.nn as nn
import minitensor.optim as optim

X = mt.randn(16, 3)
Y = mt.randn(16, 1)


def _model():
    return nn.Sequential([nn.DenseLayer(3, 4), nn.BatchNorm1d(4), nn.DenseLayer(4, 1)])


def _step(model, opt, sched):
    opt.zero_grad()
    nn.mse_loss(model(X), Y).backward()
    opt.step()
    sched.step()


def test_a_pickled_checkpoint_of_all_three_resumes_the_run():
    model = _model()
    opt = optim.Adam(model.parameters(), lr=0.01)
    sched = optim.StepLR(opt, step_size=2)
    for _ in range(4):
        _step(model, opt, sched)

    blob = pickle.dumps(
        {
            "model": model.state_dict(),
            "optim": opt.state_dict(),
            "sched": sched.state_dict(),
        }
    )
    checkpoint = pickle.loads(blob)
    resumed = _model()
    resumed.load_state_dict(checkpoint["model"])
    resumed_opt = optim.Adam(resumed.parameters(), lr=0.01)
    resumed_opt.load_state_dict(checkpoint["optim"])
    resumed_sched = optim.StepLR(resumed_opt, step_size=2)
    resumed_sched.load_state_dict(checkpoint["sched"])

    for _ in range(3):
        _step(model, opt, sched)
        _step(resumed, resumed_opt, resumed_sched)
    for ours, theirs in zip(model.parameters(), resumed.parameters()):
        assert np.array_equal(ours.numpy(), theirs.numpy())
    assert resumed_opt.lr == opt.lr


@pytest.mark.parametrize(
    "trip", [copy.copy, copy.deepcopy, lambda s: pickle.loads(pickle.dumps(s))]
)
def test_a_state_dict_copies_with_every_entry(trip):
    model = _model()
    model(X)
    state = model.state_dict()
    clone = trip(state)
    assert list(clone) == list(state)
    for name in state:
        assert clone[name].dtype == state[name].dtype
        assert np.array_equal(clone[name].numpy(), state[name].numpy())


def test_a_state_dict_reads_as_a_mapping():
    state = nn.DenseLayer(3, 2).state_dict()
    assert isinstance(state, collections.abc.Mapping)
    assert state.get("weight").shape == (2, 3)
    assert state.get("missing") is None
    assert state.get("missing", 7) == 7


@pytest.mark.parametrize("data", [b"", b"\x01\x02\x03"])
def test_corrupt_state_bytes_are_refused(data):
    state_type = type(nn.DenseLayer(1, 1).state_dict())
    with pytest.raises(OSError, match="deserialization failed"):
        state_type._from_bytes(data)
