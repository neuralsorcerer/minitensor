# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""End-to-end exercise of the dynamic plugin path.

Skipped unless the extension was built with `--features dynamic-loading` and
the bundled Rust example has been compiled:

    cargo build --release --manifest-path examples/rust_plugin_example/Cargo.toml
    maturin develop --release --features dynamic-loading

The example's operations used to be stubs -- `rust_gelu` returned its input
unchanged, and `rust_abs` and `rust_clamp` passed the incoming gradient straight
through, which is wrong wherever `|x|` bends or `clamp` saturates. Since this is
the file someone copies to write their own plugin, the tests check the maths,
not just that the library loads.
"""

import math
import os
import sysconfig

import numpy as np
import pytest

import minitensor as mt

_SUFFIX = {"linux": ".so", "darwin": ".dylib", "win32": ".dll"}.get(
    sysconfig.get_platform().split("-")[0], ".so"
)
_PLUGIN = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "examples",
    "rust_plugin_example",
    "target",
    "release",
    f"librust_plugin_example{_SUFFIX}",
)


def _dynamic_loading_available():
    if not os.path.exists(_PLUGIN):
        return False
    try:
        mt.plugins.load_plugin(_PLUGIN)
    except NotImplementedError:
        return False
    except Exception:
        return False
    mt.plugins.unload_plugin("rust_example_plugin")
    return True


pytestmark = pytest.mark.skipif(
    not _dynamic_loading_available(),
    reason="built without dynamic-loading, or the example plugin is not compiled",
)


@pytest.fixture
def plugin():
    mt.plugins.load_plugin(_PLUGIN)
    yield
    mt.plugins.unload_plugin("rust_example_plugin")


def _t(array):
    return mt.Tensor(np.ascontiguousarray(array, dtype=np.float64), dtype="float64")


_X = np.array([1.0, -2.0, 3.0, 0.5, -0.25], dtype=np.float64)
_LO, _HI = np.array([-1.0]), np.array([2.0])


def test_loading_registers_the_plugin_and_its_operations(plugin):
    names = [info.name for info in mt.plugins.list_plugins()]
    assert "rust_example_plugin" in names
    assert set(mt.list_custom_ops_py()) >= {"rust_gelu", "rust_abs", "rust_clamp"}

    info = mt.plugins.get_plugin_info("rust_example_plugin")
    assert info.name == "rust_example_plugin"
    assert str(info.version) == "1.0.0"


def test_unloading_removes_the_plugin_and_its_operations():
    mt.plugins.load_plugin(_PLUGIN)
    assert mt.is_custom_op_registered_py("rust_gelu")
    mt.plugins.unload_plugin("rust_example_plugin")
    assert not mt.is_custom_op_registered_py("rust_gelu")
    assert "rust_example_plugin" not in [i.name for i in mt.plugins.list_plugins()]


FORWARD_CASES = [
    (
        "rust_gelu",
        [_X],
        _X * 0.5 * (1 + np.vectorize(math.erf)(_X / np.sqrt(2))),
    ),
    ("rust_abs", [_X], np.abs(_X)),
    ("rust_clamp", [_X, _LO, _HI], np.clip(_X, -1.0, 2.0)),
]


@pytest.mark.parametrize(
    "name,args,expected", FORWARD_CASES, ids=[c[0] for c in FORWARD_CASES]
)
def test_plugin_operations_compute_their_named_function(plugin, name, args, expected):
    got = np.asarray(mt.execute_custom_op_py(name, [_t(a) for a in args]).numpy())
    np.testing.assert_allclose(got, expected, rtol=1e-12, atol=1e-13)


def _analytic(name, sources):
    mt.clear_autograd_graph()
    tensors = [
        mt.Tensor(
            np.ascontiguousarray(s, dtype=np.float64),
            dtype="float64",
            requires_grad=(i == 0),
        )
        for i, s in enumerate(sources)
    ]
    mt.execute_custom_op_py(name, tensors).sum().backward()
    grad = tensors[0].grad
    result = None if grad is None else grad.numpy().copy()
    mt.clear_autograd_graph()
    return result


def _numeric(name, sources, eps=1e-6):
    base = [np.asarray(s, dtype=np.float64) for s in sources]
    flat = base[0].reshape(-1).copy()
    out = np.zeros_like(flat)
    for i in range(flat.size):
        values = {}
        for sign in (1, -1):
            shifted = flat.copy()
            shifted[i] += sign * eps
            args = list(base)
            args[0] = shifted.reshape(base[0].shape)
            values[sign] = (
                mt.execute_custom_op_py(name, [_t(a) for a in args]).sum().item()
            )
        out[i] = (values[1] - values[-1]) / (2 * eps)
    return out.reshape(base[0].shape)


@pytest.mark.parametrize(
    "name,sources",
    [("rust_gelu", [_X]), ("rust_abs", [_X]), ("rust_clamp", [_X, _LO, _HI])],
    ids=["rust_gelu", "rust_abs", "rust_clamp"],
)
def test_plugin_gradients_match_finite_differences(plugin, name, sources):
    analytic = _analytic(name, sources)
    assert analytic is not None
    np.testing.assert_allclose(analytic, _numeric(name, sources), rtol=1e-5, atol=1e-7)


def test_abs_gradient_is_the_sign_not_the_incoming_gradient(plugin):
    # The specific failure the identity backward produced: every negative input
    # got a +1 gradient instead of -1, so the model moved them the wrong way.
    np.testing.assert_array_equal(_analytic("rust_abs", [_X]), np.sign(_X))


def test_clamp_gradient_is_zero_outside_the_bounds(plugin):
    grad = _analytic("rust_clamp", [_X, _LO, _HI])
    inside = (_X >= -1.0) & (_X <= 2.0)
    np.testing.assert_array_equal(grad, inside.astype(np.float64))


def test_clamp_accepts_float32_bounds_too(plugin):
    x32 = mt.Tensor(_X.astype(np.float32))
    lo = mt.Tensor(np.array([-1.0], dtype=np.float32))
    hi = mt.Tensor(np.array([2.0], dtype=np.float32))
    got = np.asarray(mt.execute_custom_op_py("rust_clamp", [x32, lo, hi]).numpy())
    np.testing.assert_allclose(got, np.clip(_X, -1.0, 2.0), rtol=1e-6)
