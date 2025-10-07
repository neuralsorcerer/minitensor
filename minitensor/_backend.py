# Copyright (c) 2025 Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from threading import RLock
from typing import Callable, Dict, Iterable, Optional

try:
    from . import _core as core
except ImportError as exc:  # pragma: no cover - surfaced during import
    raise ImportError(
        "The minitensor core extension is not built. "
        "Run `maturin develop` or install the package."
    ) from exc

# Autograd helpers exposed by the Rust backend that the Python APIs depend on.
_AUTOGRAD_REQUIRED_HELPERS: Iterable[str] = (
    "get_gradient",
    "clear_autograd_graph",
    "is_autograd_graph_consumed",
    "mark_autograd_graph_consumed",
)

# Cache resolved helper callables so we only look them up on the extension once.
_AUTOGRAD_HELPERS: Dict[str, Optional[Callable[..., object]]] = {}
_AUTOGRAD_LOCK = RLock()
_MISSING_REQUIRED: Optional[tuple[str, ...]] = None
_SENTINEL = object()


def _load_autograd_helper(name: str) -> Optional[Callable[..., object]]:
    """Resolve ``name`` from the Rust backend and cache the result."""

    with _AUTOGRAD_LOCK:
        cached = _AUTOGRAD_HELPERS.get(name, _SENTINEL)
        if cached is not _SENTINEL:
            return cached  # type: ignore[return-value]

        helper = getattr(core, name, None)
        _AUTOGRAD_HELPERS[name] = helper
        return helper


def _ensure_required_helpers() -> None:
    """Validate that all required autograd helpers exist in the backend."""

    global _MISSING_REQUIRED

    with _AUTOGRAD_LOCK:
        if _MISSING_REQUIRED == ():
            return

        missing = []
        for helper_name in _AUTOGRAD_REQUIRED_HELPERS:
            helper = _load_autograd_helper(helper_name)
            if helper is None:
                missing.append(helper_name)

        if missing:
            _MISSING_REQUIRED = tuple(missing)
            missing_list = ", ".join(missing)
            raise RuntimeError(
                "The loaded Rust backend is missing required autograd helpers: "
                f"{missing_list}. Rebuild minitensor (for example with `pip install -e .`) "
                "so that the Python package and compiled extension stay in sync."
            )

        _MISSING_REQUIRED = ()


def call_autograd_function(name: str, *args, **kwargs):
    """Invoke the Rust autograd helper ``name`` or raise a rebuild hint."""

    _ensure_required_helpers()
    helper = _load_autograd_helper(name)
    if helper is None:
        raise RuntimeError(
            "The requested autograd helper is unavailable in the loaded Rust backend. "
            "Rebuild minitensor (for example with `pip install -e .`) so that the "
            "Python package and compiled extension stay in sync."
        )
    return helper(*args, **kwargs)


def optional_autograd_function(name: str) -> Optional[Callable[..., object]]:
    """Return the helper ``name`` if available without raising when it is missing."""

    return _load_autograd_helper(name)


def autograd_get_gradient(tensor_core) -> Optional[object]:
    """Fetch the last computed gradient tensor for ``tensor_core``."""

    return call_autograd_function("get_gradient", tensor_core)


def autograd_clear_graph() -> None:
    """Clear the global autograd graph maintained by the backend."""

    call_autograd_function("clear_autograd_graph")


def autograd_is_graph_consumed() -> bool:
    """Return ``True`` if the global autograd graph has already been freed."""

    helper = optional_autograd_function("is_autograd_graph_consumed")
    if helper is None:
        return False
    return bool(helper())


def autograd_mark_graph_consumed() -> None:
    """Mark the global autograd graph as consumed if the helper exists."""

    helper = optional_autograd_function("mark_autograd_graph_consumed")
    if helper is not None:
        helper()


__all__ = [
    "core",
    "call_autograd_function",
    "optional_autograd_function",
    "autograd_get_gradient",
    "autograd_clear_graph",
    "autograd_is_graph_consumed",
    "autograd_mark_graph_consumed",
]
