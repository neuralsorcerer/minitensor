# Copyright (c) Soumyadip Sarkar.
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


def test_public_namespace_has_no_import_artifacts():
    # `minitensor/__init__.py` aliases its imports privately (`import sys as
    # _sys`, ...) so they stay out of the public namespace. `contextmanager`
    # was the one exception and showed up on `dir(minitensor)` as if it were
    # part of the API. `annotations` is the unavoidable `__future__` binding
    # every module using it has.
    public = {name for name in dir(mt) if not name.startswith("_")}
    unadvertised = public - set(mt.__all__)
    assert unadvertised == {"annotations"}, sorted(unadvertised)

    # Everything advertised must actually resolve, with no duplicates.
    assert [name for name in mt.__all__ if not hasattr(mt, name)] == []
    assert len(mt.__all__) == len(set(mt.__all__))


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


def test_matmul_mismatch_names_both_operands(t22):
    """Both shapes as passed, and neither invented.

    This used to read "expected [2, 2], got [3, 2]" for `(3,2) @ (3,2)`: the
    `[2, 2]` labelled "expected" was synthesised from one dimension of each
    operand and was never a shape the caller had.
    """
    other = mt.from_numpy(np.ones((3, 2), dtype=np.float32))
    with pytest.raises(Exception) as excinfo:
        mt.from_numpy(np.ones((3, 2), dtype=np.float32)).matmul(other)

    message = str(excinfo.value)
    assert "[3, 2] and [3, 2]" in message
    assert "[2, 2]" not in message
    # Both operands ending in 2 is the signature of a missing transpose.
    assert "transpose" in message


def test_numeric_protocol_dunders(t22):
    np.testing.assert_allclose(abs(t22).numpy(), np.abs(t22.numpy()), rtol=1e-6)
    np.testing.assert_allclose((+t22).numpy(), t22.numpy(), rtol=1e-6)

    scalar = mt.from_numpy(np.array([2.5], dtype=np.float32))
    assert float(scalar) == 2.5
    assert int(scalar) == 2
    assert int(mt.from_numpy(np.array([True]))) == 1

    with pytest.raises(TypeError):
        float(t22)
    with pytest.raises(TypeError):
        int(t22)


def test_adam_adamw_default_learning_rate():
    # Adam/AdamW default every hyperparameter except lr in their signature;
    # lr was required, rather than defaulting to 1e-3.
    from minitensor import optim

    for cls in (optim.Adam, optim.AdamW):
        opt = cls(nn.DenseLayer(4, 2).parameters())
        assert opt is not None


def test_module_load_state_dict_default_device(tmp_path):
    # load_state_dict's device argument was required, breaking the common
    # in-memory / same-device reload path.
    mt.manual_seed(1)
    model = nn.DenseLayer(4, 3)
    sd = model.state_dict()
    mt.manual_seed(2)
    other = nn.DenseLayer(4, 3)
    other.load_state_dict(sd)  # no device argument
    x = mt.from_numpy(np.random.RandomState(0).randn(2, 4).astype(np.float32))
    np.testing.assert_allclose(model(x).numpy(), other(x).numpy(), rtol=1e-6)


# The two constructor families disagree on dtype, which is easy to trip over and
# was previously undocumented. Pin the rule so it cannot drift silently.
@pytest.mark.parametrize(
    "numpy_dtype", ["float64", "float32", "int64", "int32", "bool"]
)
def test_tensor_constructors_default_to_float32_regardless_of_source_dtype(numpy_dtype):
    array = np.ones(3, dtype=numpy_dtype)
    assert mt.Tensor(array).dtype == "float32"
    assert mt.tensor(array).dtype == "float32"
    # ...but an explicit dtype is always honoured.
    assert mt.Tensor(array, dtype="float64").dtype == "float64"


@pytest.mark.parametrize(
    "numpy_dtype", ["float64", "float32", "int64", "int32", "bool"]
)
def test_from_numpy_and_as_tensor_keep_the_source_dtype(numpy_dtype):
    array = np.ones(3, dtype=numpy_dtype)
    assert mt.from_numpy(array).dtype == numpy_dtype
    assert mt.as_tensor(array).dtype == numpy_dtype


def test_float64_survives_from_numpy_but_not_tensor():
    # The practical consequence: full precision only reaches the engine through
    # the preserving constructors. `float()` forces the comparison to happen in
    # double -- comparing a float32 scalar against a Python float directly would
    # round the literal down to float32 first and call them equal.
    value = 0.1234567890123456789
    array = np.array([value], dtype=np.float64)
    assert float(mt.from_numpy(array).numpy()[0]) == value
    assert float(mt.Tensor(array).numpy()[0]) != value  # rounded to float32
    assert float(mt.Tensor(array, dtype="float64").numpy()[0]) == value
