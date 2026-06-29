# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest

import minitensor as mt
from minitensor.tensor import Tensor


def test_zeros_like_preserves_metadata():
    base = Tensor.ones((2, 3), dtype="float32", requires_grad=True)

    result = mt.numpy_compat.zeros_like(base)

    assert isinstance(result, Tensor)
    assert result.device == base.device
    assert result.dtype == base.dtype
    assert result.requires_grad is True
    np.testing.assert_array_equal(result.numpy(), np.zeros((2, 3), dtype=np.float32))


def test_ones_like_accepts_array_like_and_dtype_override():
    base = np.arange(4, dtype=np.int32).reshape(2, 2)

    result = mt.numpy_compat.ones_like(base, dtype="float64")

    assert isinstance(result, Tensor)
    assert result.device == "cpu"
    assert result.dtype == "float64"
    assert result.requires_grad is False
    np.testing.assert_array_equal(result.numpy(), np.ones((2, 2), dtype=np.float64))


def test_empty_like_matches_shape_and_requires_grad():
    base = Tensor.arange(0, 6, dtype="float32", requires_grad=True).reshape(2, 3)

    result = mt.numpy_compat.empty_like(base)

    assert isinstance(result, Tensor)
    assert result.shape == base.shape
    assert result.dtype == base.dtype
    assert result.device == base.device
    assert result.requires_grad is True


def test_full_like_uses_source_metadata():
    base = Tensor.ones((3,), dtype="float32", requires_grad=True)

    result = mt.numpy_compat.full_like(base, 7.5)

    assert isinstance(result, Tensor)
    assert result.device == base.device
    assert result.dtype == base.dtype
    assert result.requires_grad is True
    np.testing.assert_allclose(result.numpy(), np.full((3,), 7.5, dtype=np.float32))


def test_np_asarray_returns_numpy_array():
    t = Tensor([[1, 2], [3, 4]], dtype="float32")
    arr = np.asarray(t)
    assert isinstance(arr, np.ndarray)
    assert arr.dtype == np.float32
    assert np.array_equal(arr, np.array([[1, 2], [3, 4]], dtype=np.float32))


def test_np_asarray_with_dtype():
    t = Tensor([1, 2, 3], dtype="float32")
    arr = np.asarray(t, dtype=np.float64)
    assert arr.dtype == np.float64
    assert np.array_equal(arr, np.array([1.0, 2.0, 3.0], dtype=np.float64))


def test_np_add_dispatches_to_tensor():
    a = Tensor([1, 2, 3])
    b = Tensor([4, 5, 6])
    result = np.add(a, b)
    assert isinstance(result, Tensor)
    assert np.array_equal(result.numpy(), np.array([5, 7, 9]))


def test_np_multiply_with_numpy_array():
    t = Tensor([1, 2, 3])
    arr = np.array([2, 2, 2])
    result = np.multiply(t, arr)
    assert isinstance(result, Tensor)
    assert np.array_equal(result.numpy(), np.array([2, 4, 6]))


def test_np_multiply_int32_array():
    t = Tensor([1, 2, 3], dtype="int32")
    arr = np.array([2, 2, 2], dtype=np.int32)
    result = np.multiply(t, arr)
    assert isinstance(result, Tensor)
    assert result.dtype == "int32"
    assert np.array_equal(result.numpy(), np.array([2, 4, 6], dtype=np.int32))


def test_np_negative_returns_tensor():
    t = Tensor([1, -2, 3])
    result = np.negative(t)
    assert isinstance(result, Tensor)
    assert np.array_equal(result.numpy(), np.array([-1, 2, -3]))


def test_unary_minus_tensor():
    t = Tensor([1, -2, 3])
    result = -t
    assert isinstance(result, Tensor)
    assert np.array_equal(result.numpy(), np.array([-1, 2, -3]))


def test_np_trig_dispatches_to_tensor():
    t = Tensor([0.0, np.pi / 2, np.pi])
    sin_result = np.sin(t)
    cos_result = np.cos(t)
    assert isinstance(sin_result, Tensor)
    assert isinstance(cos_result, Tensor)
    np.testing.assert_allclose(
        sin_result.numpy(), np.sin([0.0, np.pi / 2, np.pi]), rtol=1e-6, atol=1e-6
    )
    np.testing.assert_allclose(
        cos_result.numpy(), np.cos([0.0, np.pi / 2, np.pi]), rtol=1e-6, atol=1e-6
    )


def test_np_power_dispatches_to_tensor():
    a = Tensor([2.0, 3.0], dtype="float32")
    b = Tensor([3.0, 2.0], dtype="float32")
    result = np.power(a, b)
    assert isinstance(result, Tensor)
    np.testing.assert_allclose(result.numpy(), np.array([8.0, 9.0], dtype=np.float32))


def test_np_add_dtype_promotion():
    t = Tensor([1, 2, 3]).astype("float64")
    arr = np.array([1, 2, 3], dtype=np.int32)
    result = np.add(t, arr)
    assert isinstance(result, Tensor)
    assert result.dtype == "float64"
    np.testing.assert_allclose(result.numpy(), np.array([2, 4, 6], dtype=np.float64))


def test_operator_add_with_numpy_array():
    t = Tensor([1, 2, 3], dtype="int32")
    arr = np.array([0.5, 1.5, 2.5], dtype=np.float64)
    result = t + arr
    assert isinstance(result, Tensor)
    assert result.dtype == "float64"
    np.testing.assert_allclose(
        result.numpy(), np.array([1.5, 3.5, 5.5], dtype=np.float64)
    )


def test_operator_comparison_with_numpy_array():
    t = Tensor([1.0, 2.0, 3.0], dtype="float32")
    arr = np.array([1.0, 1.5, 3.5], dtype=np.float32)
    result = t.gt(arr)
    assert isinstance(result, Tensor)
    np.testing.assert_array_equal(result.numpy(), np.array([False, True, False]))


def test_from_numpy_int_and_bool():
    int_arr = np.array([1, 2, 3], dtype=np.int32)
    t_int = Tensor.from_numpy(int_arr)
    assert t_int.dtype == "int32"
    assert np.array_equal(t_int.numpy(), int_arr)

    bool_arr = np.array([True, False], dtype=np.bool_)
    t_bool = Tensor.from_numpy(bool_arr)
    assert t_bool.dtype == "bool"
    assert np.array_equal(t_bool.numpy(), bool_arr)


def test_np_maximum_minimum_dispatch():
    a = Tensor([1, 3, 2], dtype="float32")
    b = np.array([2, 1, 4], dtype=np.float32)
    max_res = np.maximum(a, b)
    min_res = np.minimum(a, b)
    assert isinstance(max_res, Tensor)
    assert isinstance(min_res, Tensor)
    assert np.array_equal(max_res.numpy(), np.array([2, 3, 4], dtype=np.float32))
    assert np.array_equal(min_res.numpy(), np.array([1, 1, 2], dtype=np.float32))


def test_np_maximum_minimum_bool():
    a = Tensor([True, False], dtype="bool")
    b = np.array([False, True], dtype=np.bool_)
    max_res = np.maximum(a, b)
    min_res = np.minimum(a, b)
    assert max_res.dtype == "bool"
    assert min_res.dtype == "bool"
    assert np.array_equal(max_res.numpy(), np.array([True, True], dtype=np.bool_))
    assert np.array_equal(min_res.numpy(), np.array([False, False], dtype=np.bool_))


def test_from_numpy_nan_inf_preserved():
    arr = np.array([np.nan, np.inf, -np.inf], dtype=np.float32)
    t = Tensor.from_numpy(arr)
    out = t.numpy()
    assert t.dtype == "float32"
    assert np.isnan(out[0]) and np.isposinf(out[1]) and np.isneginf(out[2])


class _DummyModule(types.ModuleType):
    def __getattr__(self, name: str):
        def _noop(*_args, **_kwargs):
            return None

        return _noop


class _TensorMeta(type):
    def __getattr__(cls, name: str):
        def _noop(*_args, **_kwargs):
            return None

        return _noop


class _DummyTensor(metaclass=_TensorMeta):
    pass


class _DummyDevice:
    cpu = "cpu"
    cuda = "cuda"


def _load_stubbed_module(monkeypatch: pytest.MonkeyPatch):
    module_name = "minitensor_stubbed"
    module_path = Path(__file__).resolve().parents[2] / "minitensor" / "__init__.py"

    core = types.ModuleType(f"{module_name}._core")
    core.Tensor = _DummyTensor
    core.Device = _DummyDevice
    core.get_default_dtype = lambda: "float32"
    core.set_default_dtype = lambda _dtype: None
    core.manual_seed = lambda _seed: None
    core.get_gradient = lambda: None
    core.clear_autograd_graph = lambda: None
    core.is_autograd_graph_consumed = lambda: False
    core.mark_autograd_graph_consumed = lambda: None
    core.functional = _DummyModule(f"{module_name}._core.functional")
    core.nn = _DummyModule(f"{module_name}._core.nn")
    core.optim = _DummyModule(f"{module_name}._core.optim")
    core.numpy_compat = None
    core.plugins = None
    core.serialization = None

    version = types.ModuleType(f"{module_name}._version")
    version.__version__ = "0.0.0"
    version.__version_tuple__ = (0, 0, 0)

    monkeypatch.setitem(sys.modules, module_name, None)
    monkeypatch.setitem(sys.modules, f"{module_name}._core", core)
    monkeypatch.setitem(sys.modules, f"{module_name}._version", version)

    spec = importlib.util.spec_from_file_location(
        module_name,
        module_path,
        submodule_search_locations=[str(module_path.parent)],
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_available_submodules_contains_expected_keys():
    flags = mt.available_submodules()
    assert set(flags.keys()) == {
        "functional",
        "nn",
        "optim",
        "numpy_compat",
        "plugins",
        "serialization",
    }


def test_list_public_api_has_top_level_tensor():
    api = mt.list_public_api()
    assert "top_level" in api
    assert "Tensor" in api["top_level"]


def test_top_level_public_api_matches_exported_globals():
    expected_names = {
        name
        for name in (
            *mt._FUNCTIONAL_FORWARDERS,
            "Tensor",
            "manual_seed",
            "default_dtype",
            "available_submodules",
            "list_public_api",
            "api_summary",
            "search_api",
            "describe_api",
            "help",
        )
    }
    top_level_names = set(mt.list_public_api()["top_level"])
    missing = expected_names - top_level_names
    assert not missing


def test_all_functional_forwarders_are_bound_to_functional_module():
    for name in mt._FUNCTIONAL_FORWARDERS:
        assert getattr(mt, name) is getattr(mt.functional, name)


def test_functional_forwarder_binding_rejects_duplicates_and_missing_names():
    with pytest.raises(RuntimeError, match="Duplicate functional forwarders: relu"):
        mt._bind_functional_forwarders(("relu", "relu"))

    with pytest.raises(RuntimeError, match="Missing functional forwarders: missing"):
        mt._bind_functional_forwarders(("missing",))


@pytest.mark.parametrize(
    "name",
    [
        "relu",
        "hardshrink",
        "sigmoid",
        "softplus",
        "gelu",
        "elu",
        "selu",
        "silu",
        "tanh",
        "log1p",
        "expm1",
        "logaddexp",
        "maximum",
        "minimum",
        "layer_norm",
    ],
)
def test_common_math_and_activation_helpers_are_top_level_exports(name):
    assert hasattr(mt, name)
    assert name in mt.__all__
    assert name in mt.list_public_api()["top_level"]
    assert mt.describe_api(name).startswith(f"- {name}:")


def test_all_only_contains_existing_attributes():
    missing = [name for name in mt.__all__ if not hasattr(mt, name)]
    assert missing == []


def test_custom_op_helpers_are_exported_when_available():
    for name in mt._OPTIONAL_TOP_LEVEL_EXPORTS:
        assert hasattr(mt, name)
        assert name in mt.__all__
        assert name in mt.list_public_api()["top_level"]


def test_api_summary_counts_match_list_public_api():
    api = mt.list_public_api()
    summary = mt.api_summary()
    assert summary["version"] == mt.__version__
    assert summary["available_submodules"] == mt.available_submodules()
    assert summary["counts"] == {module: len(names) for module, names in api.items()}


def test_search_api_filters_and_scopes():
    results = mt.search_api("tensor")
    assert any(item.endswith("Tensor") for item in results)

    top_level_hits = mt.search_api("tensor", module="top_level")
    assert "Tensor" in top_level_hits


def test_search_api_rejects_unknown_module():
    try:
        mt.search_api("tensor", module="unknown")
    except ValueError as exc:
        assert "Unknown module" in str(exc)
    else:
        raise AssertionError("Expected ValueError for unknown module")


def test_search_api_rejects_whitespace_module():
    with pytest.raises(ValueError, match="Unknown module"):
        mt.search_api("tensor", module="   ")


def test_help_prints_and_returns(capsys):
    output = mt.help()
    captured = capsys.readouterr()
    assert "MiniTensor" in output
    assert "MiniTensor" in captured.out


def test_search_api_returns_empty_for_blank_query():
    assert mt.search_api("") == []
    assert mt.search_api("   ") == []


def test_search_api_normalizes_module_name_and_query():
    top_level_hits = mt.search_api("  tensor  ", module=" TOP_LEVEL ")
    assert "Tensor" in top_level_hits


def test_search_api_rejects_non_string_inputs():
    with pytest.raises(TypeError, match="query must be a string"):
        mt.search_api(None)  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="module must be a string"):
        mt.search_api("tensor", module=1)  # type: ignore[arg-type]


def test_iter_public_names_handles_none():
    assert mt._iter_public_names(None) == []


@pytest.mark.parametrize(
    "symbol",
    [
        "Tensor..shape",
        ".Tensor",
        "Tensor.",
        "Tensor .shape",
        "Tensor\t.shape",
        "Tensor\n.shape",
    ],
)
def test_resolve_symbol_invalid_paths(symbol):
    with pytest.raises(ValueError, match="Invalid symbol path"):
        mt._resolve_symbol(symbol)


def test_resolve_symbol_errors_and_describe_api():
    with pytest.raises(ValueError, match="symbol must be a non-empty string"):
        mt._resolve_symbol("")

    with pytest.raises(ValueError, match="symbol must be a non-empty string"):
        mt._resolve_symbol("   ")

    with pytest.raises(TypeError, match="symbol must be a string"):
        mt._resolve_symbol(None)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="Unknown symbol root"):
        mt._resolve_symbol("does_not_exist")

    with pytest.raises(ValueError, match="Unknown symbol"):
        mt._resolve_symbol("Tensor.missing")

    assert mt._resolve_symbol(" Tensor ") is mt.Tensor

    description = mt.describe_api("Tensor")
    assert description.startswith("- Tensor:")


def test_resolve_symbol_reports_missing_optional_modules(monkeypatch):
    original_numpy_compat = mt.numpy_compat
    original_plugins = mt.plugins
    original_serialization = mt.serialization
    try:
        mt.numpy_compat = None
        with pytest.raises(ValueError, match="numpy_compat"):
            mt._resolve_symbol("numpy_compat")

        mt.plugins = None
        with pytest.raises(ValueError, match="plugins"):
            mt._resolve_symbol("plugins")

        mt.serialization = None
        with pytest.raises(ValueError, match="serialization"):
            mt._resolve_symbol("serialization")
    finally:
        mt.numpy_compat = original_numpy_compat
        mt.plugins = original_plugins
        mt.serialization = original_serialization


def test_resolve_symbol_handles_core_modules():
    assert mt._resolve_symbol("functional") is mt.functional
    assert mt._resolve_symbol("nn") is mt.nn
    assert mt._resolve_symbol("optim") is mt.optim


def test_resolve_symbol_accepts_available_optional_modules():
    original_numpy_compat = mt.numpy_compat
    original_plugins = mt.plugins
    original_serialization = mt.serialization
    dummy_numpy = object()
    dummy_plugins = object()
    dummy_serialization = object()
    try:
        mt.numpy_compat = dummy_numpy
        mt.plugins = dummy_plugins
        mt.serialization = dummy_serialization
        assert mt._resolve_symbol("numpy_compat") is dummy_numpy
        assert mt._resolve_symbol("plugins") is dummy_plugins
        assert mt._resolve_symbol("serialization") is dummy_serialization
    finally:
        mt.numpy_compat = original_numpy_compat
        mt.plugins = original_plugins
        mt.serialization = original_serialization


def test_search_api_normalizes_core_module_names():
    hits = mt.search_api("relu", module=" FuNcTiOnAl ")
    assert "relu" in hits


def test_bound_api_helpers_preserve_callable_metadata():
    assert mt.search_api.__name__ == "search_api"
    assert mt.search_api.__doc__
    assert mt._resolve_symbol.__name__ == "_resolve_symbol"


def test_module_public_names_covers_optional_and_optim_branches():
    def _public_module(name: str):
        module = types.ModuleType(name)
        module.alpha = object()
        module.beta = object()
        return module

    original_nn = mt.nn
    original_optim = mt.optim
    original_numpy_compat = mt.numpy_compat
    original_plugins = mt.plugins
    original_serialization = mt.serialization
    try:
        mt.nn = _public_module("nn")
        mt.optim = _public_module("optim")
        mt.numpy_compat = _public_module("numpy_compat")
        mt.plugins = _public_module("plugins")
        mt.serialization = _public_module("serialization")

        assert mt._module_public_names("nn") == ["alpha", "beta"]
        assert mt._module_public_names("optim") == ["alpha", "beta"]
        assert mt._module_public_names("numpy_compat") == ["alpha", "beta"]
        assert mt._module_public_names("plugins") == ["alpha", "beta"]
        assert mt._module_public_names("serialization") == ["alpha", "beta"]
    finally:
        mt.nn = original_nn
        mt.optim = original_optim
        mt.numpy_compat = original_numpy_compat
        mt.plugins = original_plugins
        mt.serialization = original_serialization


def test_stubbed_search_api_rejects_missing_optional_module(monkeypatch):
    stubbed = _load_stubbed_module(monkeypatch)
    with pytest.raises(ValueError, match="Unknown module"):
        stubbed.search_api("cross", module="numpy_compat")


def test_stubbed_import_sets_cross_none(monkeypatch):
    stubbed = _load_stubbed_module(monkeypatch)
    assert stubbed.numpy_compat is None
    assert stubbed.cross is None


def test_stubbed_import_omits_missing_custom_op_helpers(monkeypatch):
    stubbed = _load_stubbed_module(monkeypatch)
    for name in stubbed._OPTIONAL_TOP_LEVEL_EXPORTS:
        assert not hasattr(stubbed, name)
        assert name not in stubbed.__all__
        assert name not in stubbed.list_public_api()["top_level"]


def test_help_skips_modules_with_none_namespace(monkeypatch):
    original_plugins = mt.plugins
    try:
        mt.plugins = None
        output = mt.help()
        assert "[plugins]" not in output
    finally:
        mt.plugins = original_plugins


def test_api_module_helper_edge_cases():
    assert mt._api_module_names(include_optional=False) == (
        "top_level",
        "functional",
        "nn",
        "optim",
    )
    assert mt._api_module_namespace("top_level") is None
    assert mt._api_module_title("unknown_module") == "unknown_module"


def test_module_public_names_rejects_non_string_input():
    with pytest.raises(TypeError, match="module must be a string"):
        mt._module_public_names(None)  # type: ignore[arg-type]
