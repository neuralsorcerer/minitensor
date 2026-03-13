# Copyright (c) 2025 Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""Public Python API surface that directly re-exports the Rust backend."""

from __future__ import annotations

import sys as _sys
import types as _types
from contextlib import contextmanager
from typing import NamedTuple

from . import _core as _C

try:  # pragma: no cover - fallback when version metadata missing
    from ._version import __version__, __version_tuple__
except ImportError:  # pragma: no cover
    __version__ = "0.1.0"
    __version_tuple__ = (0, 1, 0)

_rust_version = getattr(_C, "__version__", None)
if _rust_version:
    __version__ = _rust_version

Tensor = _C.Tensor
tensor = Tensor

Device = _C.Device
device = Device
cpu = Device.cpu
cuda = Device.cuda

zeros = Tensor.zeros
ones = Tensor.ones
empty = Tensor.empty
rand = Tensor.rand
randn = Tensor.randn
truncated_normal = Tensor.truncated_normal
rand_like = Tensor.rand_like
randn_like = Tensor.randn_like
truncated_normal_like = Tensor.truncated_normal_like
randint = Tensor.randint
randint_like = Tensor.randint_like
randperm = Tensor.randperm
eye = Tensor.eye
full = Tensor.full
full_like = Tensor.full_like
uniform = Tensor.uniform
uniform_like = Tensor.uniform_like
xavier_uniform = Tensor.xavier_uniform
xavier_uniform_like = Tensor.xavier_uniform_like
xavier_normal = Tensor.xavier_normal
xavier_normal_like = Tensor.xavier_normal_like
he_uniform = Tensor.he_uniform
he_uniform_like = Tensor.he_uniform_like
he_normal = Tensor.he_normal
he_normal_like = Tensor.he_normal_like
lecun_uniform = Tensor.lecun_uniform
lecun_uniform_like = Tensor.lecun_uniform_like
lecun_normal = Tensor.lecun_normal
lecun_normal_like = Tensor.lecun_normal_like
empty_like = Tensor.empty_like
zeros_like = Tensor.zeros_like
ones_like = Tensor.ones_like
linspace = Tensor.linspace
logspace = Tensor.logspace
arange = Tensor.arange
from_numpy = Tensor.from_numpy
from_numpy_shared = Tensor.from_numpy_shared
as_tensor = Tensor.as_tensor

get_default_dtype = _C.get_default_dtype
set_default_dtype = _C.set_default_dtype
manual_seed = _C.manual_seed
get_gradient = _C.get_gradient
clear_autograd_graph = _C.clear_autograd_graph
is_autograd_graph_consumed = _C.is_autograd_graph_consumed
mark_autograd_graph_consumed = _C.mark_autograd_graph_consumed

functional = _C.functional
_sys.modules[__name__ + ".functional"] = functional

nn = _C.nn
_sys.modules[__name__ + ".nn"] = nn

optim = _C.optim
_sys.modules[__name__ + ".optim"] = optim

numpy_compat = getattr(_C, "numpy_compat", None)
if numpy_compat is not None:
    _sys.modules[__name__ + ".numpy_compat"] = numpy_compat
    cross = getattr(numpy_compat, "cross", None)
else:
    cross = None

plugins = getattr(_C, "plugins", None)
if plugins is not None:
    _sys.modules[__name__ + ".plugins"] = plugins

serialization = getattr(_C, "serialization", None)
if serialization is not None:
    _sys.modules[__name__ + ".serialization"] = serialization


@contextmanager
def default_dtype(dtype: str):
    """Temporarily switch the global default dtype within a ``with`` block.

    This helper restores the previous default dtype even if an exception is
    raised inside the managed block. It relies on the Rust backend for
    validation so any invalid ``dtype`` values will propagate the backend
    ``ValueError`` after ensuring the prior dtype is reinstated.

    Parameters
    ----------
    dtype:
        The name of the dtype to activate (for example ``"float64"``).
    """

    previous = get_default_dtype()
    if isinstance(dtype, str) and dtype == previous:
        yield
        return

    try:
        set_default_dtype(dtype)
        yield
    finally:
        set_default_dtype(previous)


def available_submodules() -> dict[str, bool]:
    """Return availability flags for optional MiniTensor submodules."""

    return {
        module_name: _api_module_namespace(module_name) is not None
        for module_name in _NON_TOP_LEVEL_API_MODULES
    }


def list_public_api() -> dict[str, list[str]]:
    """List public API symbols by module."""

    return {
        module_name: _module_public_names(module_name)
        for module_name in _api_module_names(include_optional=True)
    }


def api_summary() -> dict[str, object]:
    """Return a summary of available MiniTensor APIs."""

    api = list_public_api()
    return {
        "version": __version__,
        "available_submodules": available_submodules(),
        "counts": {module: len(items) for module, items in api.items()},
    }


def search_api(query: str, module: str | None = None) -> list[str]:
    """Search for public API symbols matching a query string."""

    if not isinstance(query, str):
        raise TypeError("query must be a string")

    query_normalized = query.strip()
    if not query_normalized:
        return []

    query_folded = query_normalized.casefold()

    if module is not None:
        if not isinstance(module, str):
            raise TypeError("module must be a string or None")

        module_names = _module_public_names(module)

        return sorted(name for name in module_names if query_folded in name.casefold())

    api = list_public_api()
    matches: list[str] = []
    for module_name, names in api.items():
        for name in names:
            if query_folded in name.casefold():
                matches.append(f"{module_name}.{name}")
    return sorted(matches)


def _module_public_names(module: str) -> list[str]:
    module_normalized = module.strip()
    if not module_normalized:
        raise ValueError(f"Unknown module: {module}")

    module_folded = module_normalized.casefold()
    if module_folded == "top_level":
        return sorted(__all__)

    module_namespace = _api_module_namespace(module_folded)
    if module_namespace is not None:
        return sorted(_iter_public_names(module_namespace))

    raise ValueError(f"Unknown module: {module}")


def describe_api(symbol: str) -> str:
    """Return a one-line description for an API symbol."""

    target = _resolve_symbol(symbol)
    return _describe_symbol(symbol, target)


def help() -> str:
    """Return a formatted help string for all public MiniTensor APIs."""

    sections = [(_api_module_title("top_level"), _describe_symbols(__all__, globals()))]
    for module_name in _api_module_names(include_optional=True):
        if module_name == "top_level":
            continue

        module_namespace = _api_module_namespace(module_name)

        sections.append(
            (
                _api_module_title(module_name),
                _describe_symbols(
                    _iter_public_names(module_namespace), module_namespace
                ),
            )
        )

    lines = [f"MiniTensor {__version__} API Reference"]
    for title, items in sections:
        lines.append("")
        lines.append(f"[{title}]")
        lines.extend(items)
    output = "\n".join(lines)
    print(output)
    return output


def _iter_public_names(module: object) -> list[str]:
    if module is None:
        return []
    return [name for name in dir(module) if name and not name.startswith("_")]


class _ApiModuleSpec(NamedTuple):
    attr: str | None
    optional_error: str | None
    title: str


_API_MODULE_SPECS: dict[str, _ApiModuleSpec] = {
    "top_level": _ApiModuleSpec(attr=None, optional_error=None, title="Top-level"),
    "functional": _ApiModuleSpec(
        attr="functional", optional_error=None, title="functional"
    ),
    "nn": _ApiModuleSpec(attr="nn", optional_error=None, title="nn"),
    "optim": _ApiModuleSpec(attr="optim", optional_error=None, title="optim"),
    "numpy_compat": _ApiModuleSpec(
        attr="numpy_compat",
        optional_error="numpy_compat is not available in this build",
        title="numpy_compat",
    ),
    "plugins": _ApiModuleSpec(
        attr="plugins",
        optional_error="plugins are not available in this build",
        title="plugins",
    ),
    "serialization": _ApiModuleSpec(
        attr="serialization",
        optional_error="serialization is not available in this build",
        title="serialization",
    ),
}
_CORE_API_MODULES = ("top_level", "functional", "nn", "optim")
_OPTIONAL_API_MODULES = ("numpy_compat", "plugins", "serialization")
_NON_TOP_LEVEL_API_MODULES = tuple(
    module_name
    for module_name in _CORE_API_MODULES + _OPTIONAL_API_MODULES
    if module_name != "top_level"
)


def _api_module_names(*, include_optional: bool) -> tuple[str, ...]:
    if not include_optional:
        return _CORE_API_MODULES

    return _CORE_API_MODULES + tuple(
        module_name
        for module_name in _OPTIONAL_API_MODULES
        if _api_module_namespace(module_name) is not None
    )


def _api_module_namespace(module_name: str) -> object | None:
    spec = _API_MODULE_SPECS.get(module_name)
    if spec is None:
        return None

    attr_name = spec.attr
    if attr_name is None:
        return None
    return globals().get(attr_name)


def _api_module_unavailable_error(module_name: str) -> str | None:
    spec = _API_MODULE_SPECS.get(module_name)
    if spec is None:
        return None
    return spec.optional_error


def _api_module_title(module_name: str) -> str:
    spec = _API_MODULE_SPECS.get(module_name)
    if spec is None:
        return module_name
    return spec.title


def _resolve_symbol(symbol: str) -> object:
    if not isinstance(symbol, str):
        raise TypeError("symbol must be a string")

    normalized_symbol = symbol.strip()
    if not normalized_symbol:
        raise ValueError("symbol must be a non-empty string")

    if (
        normalized_symbol.startswith(".")
        or normalized_symbol.endswith(".")
        or ".." in normalized_symbol
        or any(ch.isspace() for ch in normalized_symbol)
    ):
        raise ValueError(f"Invalid symbol path: {symbol}")

    parts = normalized_symbol.split(".")
    obj = _resolve_symbol_root(parts[0])

    for part in parts[1:]:
        try:
            obj = getattr(obj, part)
        except AttributeError as exc:
            raise ValueError(f"Unknown symbol: {symbol}") from exc
    return obj


def _resolve_symbol_root(root: str) -> object:
    module_namespace = _api_module_namespace(root)
    if module_namespace is not None:
        return module_namespace

    unavailable_message = _api_module_unavailable_error(root)
    if unavailable_message is not None:
        raise ValueError(unavailable_message)

    obj = globals().get(root)
    if obj is None:
        raise ValueError(f"Unknown symbol root: {root}")
    return obj


def _describe_symbols(names: list[str], namespace: object) -> list[str]:
    return [_describe_symbol(name, getattr(namespace, name, None)) for name in names]


def _describe_symbol(name: str, obj: object) -> str:
    if obj is None:
        return f"- {name}: <missing>"
    doc = getattr(obj, "__doc__", "") or ""
    summary = doc.strip().splitlines()[0].strip() if doc.strip() else ""
    if not summary:
        summary = f"{type(obj).__name__}"
    return f"- {name}: {summary}"


_FUNCTIONAL_FORWARDERS = (
    "cat",
    "stack",
    "split",
    "chunk",
    "index_select",
    "gather",
    "narrow",
    "topk",
    "sort",
    "argsort",
    "median",
    "quantile",
    "nanquantile",
    "nansum",
    "nanmean",
    "nanmax",
    "nanmin",
    "logsumexp",
    "softmax",
    "log_softmax",
    "masked_softmax",
    "masked_log_softmax",
    "softsign",
    "rsqrt",
    "reciprocal",
    "sign",
    "reshape",
    "view",
    "triu",
    "tril",
    "diagonal",
    "trace",
    "solve",
    "flatten",
    "ravel",
    "transpose",
    "permute",
    "movedim",
    "moveaxis",
    "swapaxes",
    "swapdims",
    "squeeze",
    "unsqueeze",
    "expand",
    "repeat",
    "repeat_interleave",
    "flip",
    "roll",
    "clip",
    "clamp",
    "clamp_min",
    "clamp_max",
    "round",
    "floor",
    "ceil",
    "sin",
    "cos",
    "tan",
    "asin",
    "acos",
    "atan",
    "sinh",
    "cosh",
    "asinh",
    "acosh",
    "atanh",
    "where",
    "masked_fill",
)

for _name in _FUNCTIONAL_FORWARDERS:
    globals()[_name] = getattr(functional, _name)

for _name in dir(nn):
    if _name.startswith("_") or not _name:
        continue
    _member = getattr(nn, _name)
    if callable(_member) and _name[0].islower():
        setattr(functional, _name, _member)

dot = getattr(functional, "dot")
bmm = getattr(functional, "bmm")

_tensor_module = _types.ModuleType(__name__ + ".tensor")
for _name in (
    "Tensor",
    "tensor",
    "zeros",
    "ones",
    "empty",
    "rand",
    "randn",
    "rand_like",
    "randn_like",
    "truncated_normal",
    "truncated_normal_like",
    "uniform",
    "uniform_like",
    "xavier_uniform",
    "xavier_uniform_like",
    "xavier_normal",
    "xavier_normal_like",
    "he_uniform",
    "he_uniform_like",
    "he_normal",
    "he_normal_like",
    "lecun_uniform",
    "lecun_uniform_like",
    "lecun_normal",
    "lecun_normal_like",
    "randint",
    "randint_like",
    "randperm",
    "eye",
    "full",
    "full_like",
    "empty_like",
    "zeros_like",
    "ones_like",
    "linspace",
    "logspace",
    "arange",
    "from_numpy",
    "from_numpy_shared",
    "as_tensor",
    "get_default_dtype",
    "set_default_dtype",
    "manual_seed",
    "default_dtype",
):
    setattr(_tensor_module, _name, globals()[_name])

_sys.modules[_tensor_module.__name__] = _tensor_module


__all__ = [
    "Tensor",
    "tensor",
    "Device",
    "device",
    "cpu",
    "cuda",
    "zeros",
    "ones",
    "empty",
    "rand",
    "randn",
    "rand_like",
    "randn_like",
    "truncated_normal",
    "truncated_normal_like",
    "uniform",
    "uniform_like",
    "xavier_uniform",
    "xavier_uniform_like",
    "xavier_normal",
    "xavier_normal_like",
    "he_uniform",
    "he_uniform_like",
    "he_normal",
    "he_normal_like",
    "lecun_uniform",
    "lecun_uniform_like",
    "lecun_normal",
    "lecun_normal_like",
    "randint",
    "randint_like",
    "randperm",
    "eye",
    "full",
    "full_like",
    "empty_like",
    "zeros_like",
    "ones_like",
    "linspace",
    "logspace",
    "arange",
    "from_numpy",
    "from_numpy_shared",
    "as_tensor",
    "get_default_dtype",
    "set_default_dtype",
    "default_dtype",
    "available_submodules",
    "list_public_api",
    "api_summary",
    "search_api",
    "describe_api",
    "help",
    "get_gradient",
    "clear_autograd_graph",
    "is_autograd_graph_consumed",
    "mark_autograd_graph_consumed",
    "functional",
    "nn",
    "optim",
    "numpy_compat",
    "cross",
    "plugins",
    "serialization",
    "execute_custom_op_py",
    "is_custom_op_registered_py",
    "list_custom_ops_py",
    "register_example_custom_ops",
    "unregister_custom_op_py",
    "dot",
    "bmm",
    "cat",
    "stack",
    "split",
    "chunk",
    "index_select",
    "gather",
    "narrow",
    "topk",
    "sort",
    "argsort",
    "median",
    "quantile",
    "nanquantile",
    "logsumexp",
    "softmax",
    "log_softmax",
    "softsign",
    "rsqrt",
    "reshape",
    "view",
    "triu",
    "tril",
    "diagonal",
    "trace",
    "solve",
    "flatten",
    "ravel",
    "transpose",
    "permute",
    "movedim",
    "moveaxis",
    "swapaxes",
    "swapdims",
    "squeeze",
    "unsqueeze",
    "expand",
    "repeat",
    "repeat_interleave",
    "flip",
    "roll",
    "where",
    "masked_fill",
    "masked_softmax",
    "masked_log_softmax",
]
