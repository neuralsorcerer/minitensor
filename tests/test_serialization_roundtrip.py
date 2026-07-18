# Copyright (c) 2026 Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""Model serialization must round-trip parameters and forward outputs exactly.

Saving then reloading a model into a differently-initialized instance must
reproduce both the stored parameters and the forward output bit-for-bit,
across every supported on-disk format.
"""

import numpy as np
import pytest

import minitensor as mt
from minitensor import nn


def _params(model):
    return [p.numpy().copy() for p in model.parameters()]


def _fresh_mlp(seed):
    mt.manual_seed(seed)
    return nn.Sequential([nn.DenseLayer(5, 8), nn.ReLU(), nn.DenseLayer(8, 3)])


@pytest.mark.parametrize("fmt", ["json", "bin", "msgpack"])
def test_mlp_roundtrip_preserves_params_and_output(tmp_path, fmt):
    model = _fresh_mlp(123)
    x = mt.from_numpy(np.random.RandomState(1).randn(4, 5).astype(np.float32))
    out_before = model(x).numpy().copy()
    params_before = _params(model)

    path = str(tmp_path / f"model.{fmt}")
    model.save(path, fmt)
    loaded = type(model).load_state_from(path, fmt)

    reloaded = _fresh_mlp(999)  # different initial params
    reloaded.load_state_dict(loaded)

    for a, b in zip(params_before, _params(reloaded)):
        np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(out_before, reloaded(x).numpy())


def test_auto_format_roundtrip(tmp_path):
    model = _fresh_mlp(123)
    params_before = _params(model)
    path = str(tmp_path / "model.bin")
    model.save(path)  # format inferred
    loaded = type(model).load_state_from(path)
    reloaded = _fresh_mlp(999)
    reloaded.load_state_dict(loaded)
    for a, b in zip(params_before, _params(reloaded)):
        np.testing.assert_array_equal(a, b)


def test_conv_batchnorm_roundtrip(tmp_path):
    mt.manual_seed(5)
    cnn = nn.Sequential([nn.Conv2d(2, 3, 3), nn.BatchNorm2d(3), nn.ReLU()])
    x = mt.from_numpy(np.random.RandomState(2).randn(1, 2, 6, 6).astype(np.float32))
    out_before = cnn(x).numpy().copy()
    params_before = _params(cnn)

    path = str(tmp_path / "cnn.bin")
    cnn.save(path)
    loaded = type(cnn).load_state_from(path)

    mt.manual_seed(77)
    cnn2 = nn.Sequential([nn.Conv2d(2, 3, 3), nn.BatchNorm2d(3), nn.ReLU()])
    cnn2.load_state_dict(loaded)

    for a, b in zip(params_before, _params(cnn2)):
        np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(out_before, cnn2(x).numpy())
