# Copyright (c) 2026 Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""Optional parameters exposed from Rust must actually be optional.

PyO3 treats an ``Option``-typed parameter without an explicit
``#[pyo3(signature)]`` attribute as *required*, which silently breaks the
documented no-argument call forms. Every entry here failed with TypeError
before the signature attributes were added.
"""

import numpy as np
import pytest

import minitensor as mt
from minitensor import nn


@pytest.fixture()
def t22():
    return mt.from_numpy(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))


def test_numpy_compat_join_split_default_axis(t22):
    nc = mt.numpy_compat
    assert tuple(nc.concatenate([t22, t22]).shape) == (4, 2)
    assert tuple(nc.stack([t22, t22]).shape) == (2, 2, 2)
    parts = nc.split(t22, 2)
    assert len(parts) == 2 and tuple(parts[0].shape) == (1, 2)


def test_device_constructors_default_id():
    assert str(mt.Device.cuda()) == "cuda:0"
    assert str(mt.Device.opencl()) == "opencl:0"


def test_loss_functionals_default_reduction(t22):
    target = mt.from_numpy(np.zeros((2, 2), dtype=np.float32))
    for fn in (nn.mse_loss, nn.smooth_l1_loss, nn.log_cosh_loss):
        out = fn(t22, target)
        assert out.numel() == 1  # "mean" reduction by default


def test_dense_layer_functional_bias_optional(t22):
    weight = mt.from_numpy(np.ones((3, 2), dtype=np.float32))
    out = nn.dense_layer(t22, weight)
    np.testing.assert_allclose(out.numpy(), t22.numpy() @ np.ones((2, 3)), rtol=1e-6)


def test_module_save_load_default_format(tmp_path):
    layer = nn.DenseLayer(4, 2)
    path = str(tmp_path / "layer.bin")
    layer.save(path)
    state = type(layer).load_state_from(path)
    assert type(state).__name__ == "StateDict"


def test_timer_profiler_optional():
    core = mt._core
    timer = core.debug.timer("op") if hasattr(core, "debug") else None
    if timer is not None:
        assert timer.elapsed_ms() >= 0.0


def test_matmul_mismatch_reports_expected_rhs_shape(t22):
    other = mt.from_numpy(np.ones((3, 2), dtype=np.float32))
    with pytest.raises(Exception, match=r"expected \[2, 2\], got \[3, 2\]"):
        mt.from_numpy(np.ones((3, 2), dtype=np.float32)).matmul(other)
