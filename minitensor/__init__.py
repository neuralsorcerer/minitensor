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

from . import functional, nn, optim

# Import the compiled Rust extension and backend helpers
from ._backend import autograd_clear_graph as _backend_clear_autograd_graph
from ._backend import autograd_get_gradient as _backend_get_gradient
from ._backend import autograd_is_graph_consumed as _backend_is_autograd_graph_consumed
from ._backend import (
    autograd_mark_graph_consumed as _backend_mark_autograd_graph_consumed,
)
from ._backend import core as _minitensor_core

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


# Core tensor creation functions
def zeros(*args, **kwargs):
    """Create a tensor filled with zeros."""
    return Tensor.zeros(*args, **kwargs)


def ones(*args, **kwargs):
    """Create a tensor filled with ones."""
    return Tensor.ones(*args, **kwargs)


def rand(*args, **kwargs):
    """Create a tensor with random values from uniform distribution."""
    return Tensor.rand(*args, **kwargs)


def randn(*args, **kwargs):
    """Create a tensor with random values from normal distribution."""
    return Tensor.randn(*args, **kwargs)


def eye(*args, **kwargs):
    """Create an identity matrix."""
    return Tensor.eye(*args, **kwargs)


def full(*args, **kwargs):
    """Create a tensor filled with a specific value."""
    return Tensor.full(*args, **kwargs)


def tensor(data, *args, **kwargs):
    """Create a tensor from data."""
    return Tensor(data, *args, **kwargs)


def dot(input: Tensor, other: Tensor) -> Tensor:
    """Compute the dot product of two 1D tensors."""
    if not isinstance(input, Tensor) or not isinstance(other, Tensor):
        raise TypeError("dot expects Tensor inputs")

    return input.dot(other)


def arange(*args, **kwargs):
    """Create a tensor with values from a range."""
    return Tensor.arange(*args, **kwargs)


def from_numpy(array, requires_grad=False):
    """Create a tensor from a NumPy array."""
    try:
        return Tensor.from_numpy(array, requires_grad)
    except AttributeError:
        # Fallback if Rust extension is not available
        raise NotImplementedError(
            "from_numpy requires the Rust extension to be built. Please build the project with 'maturin develop' or 'pip install -e .'"
        )


def from_numpy_shared(array, requires_grad=False):
    """Create a tensor from a NumPy array with zero-copy when possible."""
    try:
        return Tensor.from_numpy_shared(array, requires_grad)
    except AttributeError:
        # Fallback to regular from_numpy
        return from_numpy(array, requires_grad)


def _require_tensor(value, *, argument: str) -> Tensor:
    if not isinstance(value, Tensor):
        raise TypeError(f"{argument} must be a Tensor instance")
    return value


def get_gradient(tensor: Tensor) -> Optional[Tensor]:
    """Return the last computed gradient for ``tensor`` if it exists."""

    tensor = _require_tensor(tensor, argument="tensor")
    rust_grad = _backend_get_gradient(tensor._tensor)
    if rust_grad is None:
        return None

    return Tensor._wrap_gradient_tensor(rust_grad)


def clear_autograd_graph() -> None:
    """Explicitly clear the global autograd graph maintained by the Rust backend."""

    _backend_clear_autograd_graph()
    Tensor._reset_graph_consumed_flags()


def is_autograd_graph_consumed() -> bool:
    """Check whether the global autograd graph has already been consumed."""

    return _backend_is_autograd_graph_consumed()


def mark_autograd_graph_consumed() -> None:
    """Mark the global autograd graph as consumed after a backward call."""

    _backend_mark_autograd_graph_consumed()


# NumPy compatibility functions (commonly used ones at top level)
def asarray(data, dtype=None, requires_grad=False):
    """Convert input to tensor (NumPy compatibility)."""
    if numpy_compat is None:
        return Tensor(data, dtype=dtype, requires_grad=requires_grad)
    return numpy_compat.asarray(data, dtype, requires_grad)


def cat(tensors, dim: int = 0):
    """Concatenate tensors along an existing dimension."""
    return functional.cat(tensors, dim)


def stack(tensors, dim: int = 0):
    """Stack tensors along a new dimension."""
    return functional.stack(tensors, dim)


def split(input, split_size_or_sections, dim: int = 0):
    """Split ``input`` into chunks along ``dim``."""
    return functional.split(input, split_size_or_sections, dim)


def chunk(input, chunks: int, dim: int = 0):
    """Split ``input`` into ``chunks`` along ``dim``."""
    return functional.chunk(input, chunks, dim)


def index_select(input, dim: int, indices):
    """Select elements along ``dim`` using integer ``indices``."""
    return functional.index_select(input, dim, indices)


def gather(input, dim: int, index):
    """Gather elements from ``input`` along ``dim`` using ``index`` tensor."""
    return functional.gather(input, dim, index)


def narrow(input, dim: int, start: int, length: int):
    """Narrow ``input`` along ``dim`` starting at ``start`` for ``length`` elements."""
    return functional.narrow(input, dim, start, length)


def topk(input, k, dim=None, largest=True, sorted=True):
    """Return the top-``k`` values and indices along ``dim``."""

    return functional.topk(input, k, dim=dim, largest=largest, sorted=sorted)


def sort(input, dim=-1, descending=False, stable=False):
    """Return sorted values and indices of ``input`` along ``dim``."""

    return functional.sort(input, dim=dim, descending=descending, stable=stable)


def argsort(input, dim=-1, descending=False, stable=False):
    """Return indices that would sort ``input`` along ``dim``."""

    return functional.argsort(input, dim=dim, descending=descending, stable=stable)


def median(input, dim=None, keepdim=False):
    """Return the median of ``input`` optionally along ``dim``."""

    return functional.median(input, dim=dim, keepdim=keepdim)


def logsumexp(input, dim=None, keepdim: bool = False):
    """Compute a numerically stable log-sum-exp reduction."""

    return functional.logsumexp(input, dim=dim, keepdim=keepdim)


def softmax(input, dim=None):
    """Compute softmax along ``dim``."""

    return functional.softmax(input, dim=dim)


def log_softmax(input, dim=None):
    """Compute log-softmax along ``dim``."""

    return functional.log_softmax(input, dim=dim)


def softsign(input):
    """Softsign activation computed in Rust."""

    return functional.softsign(input)


def rsqrt(input):
    """Reciprocal square root evaluated by the Rust backend."""

    return functional.rsqrt(input)


def cross(a, b, axis=-1):
    """Compute the 3D cross product (NumPy compatibility)."""
    _ensure_numpy_compat("cross")
    return numpy_compat.cross(a, b, axis=axis)


# Functional operation wrappers
def reshape(input, shape):
    """Return ``input`` with a new shape."""
    return functional.reshape(input, shape)


def view(input, *shape):
    """Return a view of ``input`` with a new shape."""
    return functional.view(input, *shape)


def triu(input, diagonal: int = 0):
    """Return the upper triangular part of ``input``."""
    return functional.triu(input, diagonal)


def tril(input, diagonal: int = 0):
    """Return the lower triangular part of ``input``."""
    return functional.tril(input, diagonal)


def flatten(input, start_dim: int = 0, end_dim: int = -1):
    """Return a view of ``input`` with dimensions flattened."""
    return functional.flatten(input, start_dim, end_dim)


def ravel(input):
    """Flatten all dimensions of ``input``."""
    return functional.ravel(input)


def transpose(input, dim0: int, dim1: int):
    """Swap two dimensions of ``input``."""
    return functional.transpose(input, dim0, dim1)


def permute(input, dims: Sequence[int]):
    """Reorder dimensions of ``input`` according to ``dims``."""
    return functional.permute(input, dims)


def movedim(input, source, destination):
    """Move tensor dimensions to new positions."""
    return functional.movedim(input, source, destination)


moveaxis = movedim


def swapaxes(input, axis0: int, axis1: int):
    """Swap two axes of ``input``."""
    return functional.swapaxes(input, axis0, axis1)


swapdims = swapaxes


def squeeze(input, dim: Optional[int] = None):
    """Remove dimensions of size 1 from ``input``."""
    return functional.squeeze(input, dim)


def unsqueeze(input, dim: int):
    """Insert a dimension of size 1 at position ``dim`` in ``input``."""
    return functional.unsqueeze(input, dim)


def expand(input, *shape: int):
    """Return ``input`` expanded to a larger ``shape``."""
    return functional.expand(input, *shape)


def repeat(input, *repeats: int):
    """Repeat ``input`` along each dimension."""
    return functional.repeat(input, *repeats)


def repeat_interleave(
    input,
    repeats,
    dim: Optional[Union[int, Sequence[int]]] = None,
    output_size: Optional[int] = None,
):
    """Repeat elements of ``input`` along a dimension."""

    return functional.repeat_interleave(input, repeats, dim, output_size)


def flip(input, dims: Union[int, Sequence[int]]):
    """Flip ``input`` along specified dimensions."""
    return functional.flip(input, dims)


def roll(input, shifts, dims: Optional[Union[int, Sequence[int]]] = None):
    """Roll ``input`` along specified dimensions."""
    return functional.roll(input, shifts, dims)


def where(condition, input, other):
    """Select elements from ``input`` or ``other`` based on ``condition``."""
    return functional.where(condition, input, other)


def masked_fill(input, mask, value):
    """Fill elements of ``input`` where ``mask`` is ``True``."""
    return functional.masked_fill(input, mask, value)


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
