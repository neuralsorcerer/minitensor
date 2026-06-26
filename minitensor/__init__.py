# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""Public Python API surface that directly re-exports the Rust backend."""

from __future__ import annotations

import builtins as _builtins
import operator as _operator
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

_OPTIONAL_TOP_LEVEL_EXPORTS = (
    "execute_custom_op_py",
    "is_custom_op_registered_py",
    "list_custom_ops_py",
    "register_example_custom_ops",
    "unregister_custom_op_py",
)

for _name in _OPTIONAL_TOP_LEVEL_EXPORTS:
    _member = getattr(_C, _name, None)
    if _member is not None:
        globals()[_name] = _member


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


def _normalize_dimension(dim: object, name: str) -> int:
    if isinstance(dim, bool):
        raise TypeError(f"{name} dimensions must be integers, not bool")

    try:
        normalized = _operator.index(dim)
    except TypeError as exc:
        raise TypeError(f"{name} dimensions must be integers") from exc

    if normalized < 0:
        raise ValueError(f"{name} dimensions must be non-negative")
    return normalized


def _normalize_shape_argument(shape: object, name: str) -> tuple[int, ...]:
    if isinstance(shape, bool):
        raise TypeError(f"{name} dimensions must be integers, not bool")

    try:
        return (_normalize_dimension(shape, name),)
    except TypeError:
        pass

    try:
        dims = tuple(shape)  # type: ignore[arg-type]
    except TypeError as exc:
        raise TypeError(f"{name} must be an int or an iterable of ints") from exc

    return tuple(_normalize_dimension(dim, name) for dim in dims)


def broadcast_shapes(*shapes: object) -> tuple[int, ...]:
    """Return the shape produced by NumPy/PyTorch-style broadcasting.

    Each argument may be a single non-negative integer dimension or an iterable
    of non-negative integer dimensions. Scalar shapes are represented by an
    empty iterable, e.g. ``broadcast_shapes((), (2, 3)) == (2, 3)``.
    """

    if not shapes:
        return ()

    normalized_shapes = [
        _normalize_shape_argument(shape, f"shapes[{index}]")
        for index, shape in enumerate(shapes)
    ]
    result_reversed: list[int] = []
    max_rank = _builtins.max(len(shape) for shape in normalized_shapes)

    for axis_from_end in range(max_rank):
        resolved = 1
        for shape in normalized_shapes:
            if axis_from_end >= len(shape):
                continue
            dim = shape[-1 - axis_from_end]
            if dim == 1 or dim == resolved:
                continue
            if resolved == 1:
                resolved = dim
                continue
            raise ValueError(
                "shapes cannot be broadcast together: "
                + ", ".join(str(shape) for shape in normalized_shapes)
            )
        result_reversed.append(resolved)

    return tuple(reversed(result_reversed))


def can_broadcast(*shapes: object) -> bool:
    """Return ``True`` when shapes can broadcast without creating tensors."""

    try:
        broadcast_shapes(*shapes)
    except (TypeError, ValueError):
        return False
    return True


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
        module_names = _module_public_names(module)
        return [name for name in module_names if query_folded in name.casefold()]

    api = list_public_api()
    matches: list[str] = []
    for module_name, names in api.items():
        for name in names:
            if query_folded in name.casefold():
                matches.append(f"{module_name}.{name}")
    return sorted(matches)


def _module_public_names(module: str) -> list[str]:
    if not isinstance(module, str):
        raise TypeError("module must be a string")

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
        or _builtins.any(ch.isspace() for ch in normalized_symbol)
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
    "nan_to_num",
    "logsumexp",
    "softmax",
    "log_softmax",
    "masked_softmax",
    "masked_log_softmax",
    "sum",
    "prod",
    "mean",
    "all",
    "any",
    "max",
    "min",
    "argmax",
    "argmin",
    "cumsum",
    "cumprod",
    "std",
    "var",
    "relu",
    "hardshrink",
    "sigmoid",
    "softplus",
    "gelu",
    "elu",
    "selu",
    "silu",
    "softsign",
    "tanh",
    "layer_norm",
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
    "log1p",
    "expm1",
    "logaddexp",
    "array_equal",
    "allclose",
    "where",
    "one_hot",
    "masked_fill",
)


def _bind_functional_forwarders(names: tuple[str, ...]) -> None:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for name in names:
        if name in seen:
            duplicates.add(name)
        else:
            seen.add(name)

    if duplicates:
        raise RuntimeError(
            "Duplicate functional forwarders: " + ", ".join(sorted(duplicates))
        )

    missing = [name for name in names if not hasattr(functional, name)]
    if missing:
        raise RuntimeError("Missing functional forwarders: " + ", ".join(missing))

    for name in names:
        globals()[name] = getattr(functional, name)


_bind_functional_forwarders(_FUNCTIONAL_FORWARDERS)

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


_BASE_EXPORTS = (
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
    "manual_seed",
    "default_dtype",
    "available_submodules",
    "list_public_api",
    "api_summary",
    "broadcast_shapes",
    "can_broadcast",
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
    "dot",
    "bmm",
)

__all__ = [
    name
    for name in (
        *_BASE_EXPORTS,
        *_OPTIONAL_TOP_LEVEL_EXPORTS,
        *_FUNCTIONAL_FORWARDERS,
    )
    if name in globals()
]
