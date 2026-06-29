# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""Public API introspection helpers for MiniTensor."""

from __future__ import annotations

import builtins as _builtins
import sys as _sys
from collections.abc import MutableMapping
from functools import wraps as _wraps
from typing import Any, Callable, NamedTuple


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


def _public_namespace() -> MutableMapping[str, object]:
    return _sys.modules["minitensor"].__dict__


def _bind_namespace(
    function: Callable[..., object], namespace: MutableMapping[str, object]
) -> Callable[..., object]:
    """Bind a namespace-aware helper to a concrete package namespace."""

    @_wraps(function)
    def wrapper(*args: Any, **kwargs: Any) -> object:
        kwargs["namespace"] = namespace
        return function(*args, **kwargs)

    return wrapper


def available_submodules(
    *, namespace: MutableMapping[str, object] | None = None
) -> dict[str, bool]:
    """Return availability flags for optional MiniTensor submodules."""

    return {
        module_name: _api_module_namespace(module_name, namespace=namespace) is not None
        for module_name in _NON_TOP_LEVEL_API_MODULES
    }


def list_public_api(
    *, namespace: MutableMapping[str, object] | None = None
) -> dict[str, list[str]]:
    """List public API symbols by module."""

    return {
        module_name: _module_public_names(module_name, namespace=namespace)
        for module_name in _api_module_names(include_optional=True, namespace=namespace)
    }


def api_summary(
    *, namespace: MutableMapping[str, object] | None = None
) -> dict[str, object]:
    """Return a summary of available MiniTensor APIs."""

    namespace = _public_namespace() if namespace is None else namespace
    api = list_public_api(namespace=namespace)
    return {
        "version": namespace["__version__"],
        "available_submodules": available_submodules(namespace=namespace),
        "counts": {module: len(items) for module, items in api.items()},
    }


def search_api(
    query: str,
    module: str | None = None,
    *,
    namespace: MutableMapping[str, object] | None = None,
) -> list[str]:
    """Search for public API symbols matching a query string."""

    if not isinstance(query, str):
        raise TypeError("query must be a string")

    query_normalized = query.strip()
    if not query_normalized:
        return []

    query_folded = query_normalized.casefold()

    if module is not None:
        module_names = _module_public_names(module, namespace=namespace)
        return [name for name in module_names if query_folded in name.casefold()]

    api = list_public_api(namespace=namespace)
    matches: list[str] = []
    for module_name, names in api.items():
        for name in names:
            if query_folded in name.casefold():
                matches.append(f"{module_name}.{name}")
    return sorted(matches)


def _module_public_names(
    module: str, *, namespace: MutableMapping[str, object] | None = None
) -> list[str]:
    if not isinstance(module, str):
        raise TypeError("module must be a string")

    module_normalized = module.strip()
    if not module_normalized:
        raise ValueError(f"Unknown module: {module}")

    module_folded = module_normalized.casefold()
    if module_folded == "top_level":
        return sorted(
            (_public_namespace() if namespace is None else namespace)["__all__"]
        )

    module_namespace = _api_module_namespace(module_folded, namespace=namespace)
    if module_namespace is not None:
        return sorted(_iter_public_names(module_namespace))

    raise ValueError(f"Unknown module: {module}")


def describe_api(
    symbol: str, *, namespace: MutableMapping[str, object] | None = None
) -> str:
    """Return a one-line description for an API symbol."""

    target = _resolve_symbol(symbol, namespace=namespace)
    return _describe_symbol(symbol, target)


def help(*, namespace: MutableMapping[str, object] | None = None) -> str:
    """Return a formatted help string for all public MiniTensor APIs."""

    namespace = _public_namespace() if namespace is None else namespace
    sections = [
        (
            _api_module_title("top_level"),
            _describe_symbols(namespace["__all__"], namespace),
        )
    ]
    for module_name in _api_module_names(include_optional=True, namespace=namespace):
        if module_name == "top_level":
            continue

        module_namespace = _api_module_namespace(module_name, namespace=namespace)

        sections.append(
            (
                _api_module_title(module_name),
                _describe_symbols(
                    _iter_public_names(module_namespace), module_namespace
                ),
            )
        )

    lines = [f"MiniTensor {namespace['__version__']} API Reference"]
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


def _api_module_names(
    *, include_optional: bool, namespace: MutableMapping[str, object] | None = None
) -> tuple[str, ...]:
    if not include_optional:
        return _CORE_API_MODULES

    return _CORE_API_MODULES + tuple(
        module_name
        for module_name in _OPTIONAL_API_MODULES
        if _api_module_namespace(module_name, namespace=namespace) is not None
    )


def _api_module_namespace(
    module_name: str, *, namespace: MutableMapping[str, object] | None = None
) -> object | None:
    spec = _API_MODULE_SPECS.get(module_name)
    if spec is None:
        return None

    attr_name = spec.attr
    if attr_name is None:
        return None
    return (_public_namespace() if namespace is None else namespace).get(attr_name)


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


def _resolve_symbol(
    symbol: str, *, namespace: MutableMapping[str, object] | None = None
) -> object:
    if not isinstance(symbol, str):
        raise TypeError("symbol must be a string")

    normalized_symbol = symbol.strip()
    if not normalized_symbol:
        raise ValueError("symbol must be a non-empty string")

    if (
        normalized_symbol.startswith(".")
        or normalized_symbol.endswith(".")
        or ".." in normalized_symbol
        or _builtins.any(ch.isspace() for ch in normalized_symbol)
    ):
        raise ValueError(f"Invalid symbol path: {symbol}")

    parts = normalized_symbol.split(".")
    obj = _resolve_symbol_root(parts[0], namespace=namespace)

    for part in parts[1:]:
        try:
            obj = getattr(obj, part)
        except AttributeError as exc:
            raise ValueError(f"Unknown symbol: {symbol}") from exc
    return obj


def _resolve_symbol_root(
    root: str, *, namespace: MutableMapping[str, object] | None = None
) -> object:
    module_namespace = _api_module_namespace(root, namespace=namespace)
    if module_namespace is not None:
        return module_namespace

    unavailable_message = _api_module_unavailable_error(root)
    if unavailable_message is not None:
        raise ValueError(unavailable_message)

    obj = (_public_namespace() if namespace is None else namespace).get(root)
    if obj is None:
        raise ValueError(f"Unknown symbol root: {root}")
    return obj


def _describe_symbols(names: list[str], namespace: object) -> list[str]:
    if isinstance(namespace, MutableMapping):
        return [_describe_symbol(name, namespace.get(name)) for name in names]
    return [_describe_symbol(name, getattr(namespace, name, None)) for name in names]


def _describe_symbol(name: str, obj: object) -> str:
    if obj is None:
        return f"- {name}: <missing>"
    doc = getattr(obj, "__doc__", "") or ""
    summary = doc.strip().splitlines()[0].strip() if doc.strip() else ""
    if not summary:
        summary = f"{type(obj).__name__}"
    return f"- {name}: {summary}"
