# Copyright (c) 2025 Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""Public Python API surface that directly re-exports the Rust backend."""

from __future__ import annotations

from contextlib import contextmanager
import sys as _sys
import types as _types

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
rand_like = Tensor.rand_like
randn_like = Tensor.randn_like
randint = Tensor.randint
randint_like = Tensor.randint_like
randperm = Tensor.randperm
eye = Tensor.eye
full = Tensor.full
full_like = Tensor.full_like
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
    "logsumexp",
    "softmax",
    "log_softmax",
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
    "reciprocal",
    "sign",
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
]
