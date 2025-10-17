# Copyright (c) 2025 Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import sys
from typing import Any, Iterable

# Import the compiled Rust extension and backend helpers
from . import _core as _C
from . import functional, nn, optim
from .tensor import Tensor, get_default_dtype, set_default_dtype

try:
    from ._version import __version__, __version_tuple__
except ImportError:  # pragma: no cover - fallback for editable installs
    __version__ = "0.1.0"
    __version_tuple__ = (0, 1, 0)

try:
    _rust_version = _C.__version__
except AttributeError:
    _rust_version = None
else:
    if _rust_version and _rust_version != __version__:
        __version__ = _rust_version


functional = functional
nn = nn
optim = optim
numpy_compat = getattr(_C, "numpy_compat", None)

# Tensor factories map directly to backend implementations.
tensor = Tensor
zeros = Tensor.zeros
ones = Tensor.ones
rand = Tensor.rand
randn = Tensor.randn
eye = Tensor.eye
full = Tensor.full
arange = Tensor.arange
from_numpy = Tensor.from_numpy
from_numpy_shared = Tensor.from_numpy_shared

_DEVICE_CLASS = _C.Device
device = _DEVICE_CLASS
cpu = _DEVICE_CLASS.cpu
cuda = _DEVICE_CLASS.cuda

get_gradient = _C.get_gradient
clear_autograd_graph = _C.clear_autograd_graph
is_autograd_graph_consumed = _C.is_autograd_graph_consumed
mark_autograd_graph_consumed = _C.mark_autograd_graph_consumed

_plugins = getattr(_C, "plugins", None)
if _plugins is not None:
    plugins = _plugins
    sys.modules[__name__ + ".plugins"] = plugins
else:  # pragma: no cover - plugins optional in lightweight builds
    plugins = None

serialization = getattr(_C, "serialization", None)

_execute_custom = getattr(_C, "execute_custom_op_py", None)
_is_registered = getattr(_C, "is_custom_op_registered_py", None)
_list_ops = getattr(_C, "list_custom_ops_py", None)
_register_examples = getattr(_C, "register_example_custom_ops", None)
_unregister = getattr(_C, "unregister_custom_op_py", None)

if _execute_custom is not None:
    execute_custom_op_py = _execute_custom
    is_custom_op_registered_py = _is_registered
    list_custom_ops_py = _list_ops
    register_example_custom_ops = _register_examples
    unregister_custom_op_py = _unregister
else:  # pragma: no cover - custom ops optional
    execute_custom_op_py = None
    is_custom_op_registered_py = None
    list_custom_ops_py = None
    register_example_custom_ops = None
    unregister_custom_op_py = None

_FUNCTIONAL_FORWARDERS: Iterable[str] = (
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
)

for _name in _FUNCTIONAL_FORWARDERS:
    globals()[_name] = getattr(functional, _name)

dot = getattr(functional, "dot")


def _require_numpy(feature: str) -> None:
    if numpy_compat is None:
        raise ModuleNotFoundError(
            "NumPy support is not available; install the 'numpy' package to "
            f"use minitensor.{feature}.",
        )


def asarray(data: Any, dtype: str | None = None, requires_grad: bool = False):
    _require_numpy("asarray")
    return numpy_compat.asarray(data, dtype=dtype, requires_grad=requires_grad)


if numpy_compat is not None:
    cross = numpy_compat.cross
else:  # pragma: no cover - executed only when NumPy missing

    def cross(*_args, **_kwargs):  # type: ignore[override]
        _require_numpy("cross")


__all__ = [
    "Tensor",
    "tensor",
    "functional",
    "nn",
    "optim",
    "numpy_compat",
    "zeros",
    "ones",
    "rand",
    "randn",
    "eye",
    "full",
    "arange",
    "from_numpy",
    "from_numpy_shared",
    "asarray",
    "cat",
    "stack",
    "chunk",
    "dot",
    "index_select",
    "gather",
    "narrow",
    "reshape",
    "view",
    "triu",
    "tril",
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
    "topk",
    "sort",
    "argsort",
    "median",
    "softsign",
    "softmax",
    "log_softmax",
    "rsqrt",
    "logsumexp",
    "split",
    "cross",
    "device",
    "cpu",
    "cuda",
    "get_gradient",
    "clear_autograd_graph",
    "is_autograd_graph_consumed",
    "mark_autograd_graph_consumed",
    "set_default_dtype",
    "get_default_dtype",
]

if plugins is not None:
    __all__.append("plugins")
    if execute_custom_op_py is not None:
        __all__.extend(
            [
                "execute_custom_op_py",
                "list_custom_ops_py",
                "is_custom_op_registered_py",
                "register_example_custom_ops",
                "unregister_custom_op_py",
            ]
        )

if serialization is not None:
    __all__.append("serialization")
