# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""Public Python API surface that directly re-exports the Rust backend."""

from __future__ import annotations

import sys as _sys
import types as _types
from contextlib import contextmanager

from . import _api as _api_helpers
from . import _core as _C
from ._api import _CORE_API_MODULES, _OPTIONAL_API_MODULES
from ._exports import (
    _FUNCTIONAL_FORWARDERS,
    _bind_functional_forwarders,
    _ensure_unique_names,
)
from ._shape import (
    atleast_1d,
    atleast_2d,
    atleast_3d,
    broadcast_shapes,
    broadcast_tensors,
    broadcast_to,
    can_broadcast,
)

_api_namespace = globals()
available_submodules = _api_helpers._bind_namespace(
    _api_helpers.available_submodules, namespace=_api_namespace
)
list_public_api = _api_helpers._bind_namespace(
    _api_helpers.list_public_api, namespace=_api_namespace
)
api_summary = _api_helpers._bind_namespace(
    _api_helpers.api_summary, namespace=_api_namespace
)
search_api = _api_helpers._bind_namespace(
    _api_helpers.search_api, namespace=_api_namespace
)
_module_public_names = _api_helpers._bind_namespace(
    _api_helpers._module_public_names, namespace=_api_namespace
)
describe_api = _api_helpers._bind_namespace(
    _api_helpers.describe_api, namespace=_api_namespace
)
help = _api_helpers._bind_namespace(_api_helpers.help, namespace=_api_namespace)
_iter_public_names = _api_helpers._iter_public_names
_api_module_names = _api_helpers._bind_namespace(
    _api_helpers._api_module_names, namespace=_api_namespace
)
_api_module_namespace = _api_helpers._bind_namespace(
    _api_helpers._api_module_namespace, namespace=_api_namespace
)
_api_module_unavailable_error = _api_helpers._api_module_unavailable_error
_api_module_title = _api_helpers._api_module_title
_resolve_symbol = _api_helpers._bind_namespace(
    _api_helpers._resolve_symbol, namespace=_api_namespace
)

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


_bind_functional_forwarders(_FUNCTIONAL_FORWARDERS, globals())

for _name in dir(nn):
    if _name.startswith("_") or not _name:
        continue
    _member = getattr(nn, _name)
    if callable(_member) and _name[0].islower():
        setattr(functional, _name, _member)

dot = getattr(functional, "dot")
bmm = getattr(functional, "bmm")

_TENSOR_EXPORTS = (
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
)
_ensure_unique_names(_TENSOR_EXPORTS, "tensor exports")

_tensor_module = _types.ModuleType(__name__ + ".tensor")
for _name in _TENSOR_EXPORTS:
    setattr(_tensor_module, _name, globals()[_name])

_sys.modules[_tensor_module.__name__] = _tensor_module


_BASE_EXPORTS = (
    *_TENSOR_EXPORTS,
    "Device",
    "device",
    "cpu",
    "cuda",
    "available_submodules",
    "list_public_api",
    "api_summary",
    "broadcast_to",
    "broadcast_shapes",
    "broadcast_tensors",
    "can_broadcast",
    "atleast_1d",
    "atleast_2d",
    "atleast_3d",
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
_ensure_unique_names(_BASE_EXPORTS, "base exports")

_ALL_EXPORT_CANDIDATES = (
    *_BASE_EXPORTS,
    *_OPTIONAL_TOP_LEVEL_EXPORTS,
    *_FUNCTIONAL_FORWARDERS,
)
_ensure_unique_names(_ALL_EXPORT_CANDIDATES, "top-level public exports")

__all__ = [name for name in _ALL_EXPORT_CANDIDATES if name in globals()]
