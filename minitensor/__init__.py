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
    try:
        set_default_dtype(dtype)
        yield
    finally:
        set_default_dtype(previous)


def available_submodules() -> dict[str, bool]:
    """Return availability flags for optional MiniTensor submodules."""

    return {
        "functional": functional is not None,
        "nn": nn is not None,
        "optim": optim is not None,
        "numpy_compat": numpy_compat is not None,
        "plugins": plugins is not None,
        "serialization": serialization is not None,
    }


def list_public_api() -> dict[str, list[str]]:
    """List public API symbols by module."""

    api = {
        "top_level": sorted(__all__),
        "functional": sorted(_iter_public_names(functional)),
        "nn": sorted(_iter_public_names(nn)),
        "optim": sorted(_iter_public_names(optim)),
    }

    if numpy_compat is not None:
        api["numpy_compat"] = sorted(_iter_public_names(numpy_compat))
    if plugins is not None:
        api["plugins"] = sorted(_iter_public_names(plugins))
    if serialization is not None:
        api["serialization"] = sorted(_iter_public_names(serialization))

    return api


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

    if not query:
        return []

    api = list_public_api()
    query_lower = query.lower()

    if module is not None:
        if module not in api:
            raise ValueError(f"Unknown module: {module}")
        return sorted(name for name in api[module] if query_lower in name.lower())

    matches: list[str] = []
    for module_name, names in api.items():
        for name in names:
            if query_lower in name.lower():
                matches.append(f"{module_name}.{name}")
    return sorted(matches)


def describe_api(symbol: str) -> str:
    """Return a one-line description for an API symbol."""

    target = _resolve_symbol(symbol)
    return _describe_symbol(symbol, target)


def help() -> str:
    """Return a formatted help string for all public MiniTensor APIs."""

    sections = [
        ("Top-level", _describe_symbols(__all__, globals())),
        ("functional", _describe_symbols(_iter_public_names(functional), functional)),
        ("nn", _describe_symbols(_iter_public_names(nn), nn)),
        ("optim", _describe_symbols(_iter_public_names(optim), optim)),
    ]

    if numpy_compat is not None:
        sections.append(
            (
                "numpy_compat",
                _describe_symbols(_iter_public_names(numpy_compat), numpy_compat),
            )
        )
    if plugins is not None:
        sections.append(
            ("plugins", _describe_symbols(_iter_public_names(plugins), plugins))
        )
    if serialization is not None:
        sections.append(
            (
                "serialization",
                _describe_symbols(_iter_public_names(serialization), serialization),
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


def _resolve_symbol(symbol: str) -> object:
    parts = symbol.split(".")
    if not parts:
        raise ValueError("symbol must be a non-empty string")

    root = parts[0]
    if root == "functional":
        obj: object = functional
    elif root == "nn":
        obj = nn
    elif root == "optim":
        obj = optim
    elif root == "numpy_compat":
        if numpy_compat is None:
            raise ValueError("numpy_compat is not available in this build")
        obj = numpy_compat
    elif root == "plugins":
        if plugins is None:
            raise ValueError("plugins are not available in this build")
        obj = plugins
    elif root == "serialization":
        if serialization is None:
            raise ValueError("serialization is not available in this build")
        obj = serialization
    else:
        obj = globals().get(root)
        if obj is None:
            raise ValueError(f"Unknown symbol root: {root}")

    for part in parts[1:]:
        try:
            obj = getattr(obj, part)
        except AttributeError as exc:
            raise ValueError(f"Unknown symbol: {symbol}") from exc
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
