# Copyright (c) 2025 Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

import importlib
import os
import subprocess
import sys
from typing import Optional, Sequence, Union

# Import the compiled Rust extension and backend helpers
from . import _core as _minitensor_core
from . import functional, nn, optim

# Re-export core classes and functions
from .tensor import Tensor, get_default_dtype, set_default_dtype

_NUMPY_MODULE_NAME = "numpy"
_NUMPY_ERROR: ModuleNotFoundError | None = None
_NUMPY_READY = False


def _ensure_reference_dependency() -> None:
    """Ensure NumPy is importable for reference implementations in tests."""

    global _NUMPY_ERROR, _NUMPY_READY

    if _NUMPY_READY:
        return

    try:
        importlib.import_module(_NUMPY_MODULE_NAME)
    except ModuleNotFoundError as exc:
        python = sys.executable or "python3"
        cmd = [python, "-m", "pip", "install", _NUMPY_MODULE_NAME]
        env = os.environ.copy()
        env.setdefault("PIP_DISABLE_PIP_VERSION_CHECK", "1")
        try:
            subprocess.check_call(cmd, env=env)
        except Exception as install_exc:
            _NUMPY_ERROR = ModuleNotFoundError(
                "NumPy is required for MiniTensor's test suite. "
                "Install it manually with 'pip install numpy'."
            )
            raise _NUMPY_ERROR from install_exc
        else:
            try:
                importlib.import_module(_NUMPY_MODULE_NAME)
            except ModuleNotFoundError as post_install_exc:  # pragma: no cover
                _NUMPY_ERROR = ModuleNotFoundError(
                    "NumPy could not be imported even after installation."
                )
                raise _NUMPY_ERROR from post_install_exc
    _NUMPY_READY = True


try:
    _ensure_reference_dependency()
except ModuleNotFoundError:
    numpy_compat = None
else:
    try:
        from . import numpy_compat  # type: ignore  # noqa: F401
    except ImportError:
        numpy_compat = None


def _ensure_numpy_compat(feature: str) -> None:
    """Raise a helpful error when NumPy interoperability is unavailable."""

    if numpy_compat is None:
        if _NUMPY_ERROR is not None:
            raise _NUMPY_ERROR
        raise ModuleNotFoundError(
            "NumPy support is not available; install the 'numpy' package to "
            f"use minitensor.{feature}."
        )


# Custom operations and plugin system (if available)
try:
    from ._core import (
        execute_custom_op_py,
        is_custom_op_registered_py,
        list_custom_ops_py,
    )
    from ._core import plugins as _plugins
    from ._core import (
        register_example_custom_ops,
        unregister_custom_op_py,
    )

    plugins = _plugins
    sys.modules[__name__ + ".plugins"] = plugins
except Exception:
    execute_custom_op_py = None
    is_custom_op_registered_py = None
    list_custom_ops_py = None
    register_example_custom_ops = None
    unregister_custom_op_py = None
    plugins = None

# Serialization (if available)
try:
    serialization = _minitensor_core.serialization
except Exception:
    serialization = None

# Version information
try:
    from ._version import __version__, __version_tuple__
except ImportError:
    # Fallback version if _version.py is not available
    __version__ = "0.1.0"
    __version_tuple__ = (0, 1, 0)

# Also try to get version from Rust extension if available
try:
    _rust_version = _minitensor_core.__version__
    # Use Rust version if it's different (for development builds)
    if _rust_version != __version__:
        __version__ = _rust_version
except (AttributeError, NameError):
    pass


# Core tensor creation helpers map directly to the backend-backed Tensor APIs.
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


def dot(input: Tensor, other: Tensor) -> Tensor:
    """Compute the dot product of two 1D tensors."""
    if not isinstance(input, Tensor) or not isinstance(other, Tensor):
        raise TypeError("dot expects Tensor inputs")

    return input.dot(other)


def _require_tensor(value, *, argument: str) -> Tensor:
    if not isinstance(value, Tensor):
        raise TypeError(f"{argument} must be a Tensor instance")
    return value


def get_gradient(tensor: Tensor) -> Optional[Tensor]:
    """Return the last computed gradient for ``tensor`` if it exists."""

    tensor = _require_tensor(tensor, argument="tensor")
    rust_grad = _minitensor_core.get_gradient(tensor._tensor)
    if rust_grad is None:
        return None

    return Tensor._wrap_gradient_tensor(rust_grad)


def clear_autograd_graph() -> None:
    """Explicitly clear the global autograd graph maintained by the Rust backend."""

    _minitensor_core.clear_autograd_graph()
    Tensor._reset_graph_consumed_flags()


def is_autograd_graph_consumed() -> bool:
    """Check whether the global autograd graph has already been consumed."""

    return _minitensor_core.is_autograd_graph_consumed()


def mark_autograd_graph_consumed() -> None:
    """Mark the global autograd graph as consumed after a backward call."""

    _minitensor_core.mark_autograd_graph_consumed()


# NumPy compatibility functions (commonly used ones at top level)
def asarray(data, dtype=None, requires_grad=False):
    """Convert input to tensor (NumPy compatibility)."""
    if numpy_compat is None:
        return Tensor(data, dtype=dtype, requires_grad=requires_grad)
    return numpy_compat.asarray(data, dtype, requires_grad)


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


def cross(a, b, axis=-1):
    """Compute the 3D cross product (NumPy compatibility)."""
    _ensure_numpy_compat("cross")
    return numpy_compat.cross(a, b, axis=axis)


# Device management
def device(device_str):
    """Create a device object."""
    return _minitensor_core.Device(device_str)


def cpu():
    """Get CPU device."""
    return device("cpu")


def cuda(device_id=0):
    """Get CUDA device."""
    return device(f"cuda:{device_id}")


__all__ = [
    "Tensor",
    "nn",
    "optim",
    "functional",
    "zeros",
    "ones",
    "rand",
    "randn",
    "eye",
    "full",
    "tensor",
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

if numpy_compat is not None:
    __all__.append("numpy_compat")
# Add plugins to __all__ if available
if plugins is not None:
    __all__.append("plugins")
if register_example_custom_ops is not None:
    __all__.extend(
        [
            "register_example_custom_ops",
            "unregister_custom_op_py",
            "execute_custom_op_py",
            "list_custom_ops_py",
            "is_custom_op_registered_py",
        ]
    )
if serialization is not None:
    __all__.append("serialization")
