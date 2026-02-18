# Copyright (c) 2026 Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

import importlib.util
import sys
import types
from pathlib import Path

import pytest

import minitensor as mt


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
    module_path = Path(__file__).resolve().parents[1] / "minitensor" / "__init__.py"

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

    with pytest.raises(TypeError, match="module must be a string or None"):
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


def test_stubbed_import_sets_cross_none(monkeypatch):
    stubbed = _load_stubbed_module(monkeypatch)
    assert stubbed.numpy_compat is None
    assert stubbed.cross is None
